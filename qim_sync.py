"""QIM/base-4 in-band synchronization transport.

This is the alternative sync physical layer proposed for experiments:

1. Synchronization metadata is packed into the same logical fields as the
   tone transport (header, sequence, next_frame_index, optional session nonce,
   CRC-16).
2. Each byte is represented by four base-4 digits (two bits per digit).
3. Each base-4 digit d in {0,1,2,3} is written into one or more consecutive
   PCM samples by quantizing the sample down to a multiple of Q and selecting
   one of four residues: 0, Q/4, Q/2, 3Q/4.

For the teacher-style defaults Q=64, the four residue levels are exactly
0, 16, 32 and 48. With repetition=1 this is the literal one-digit-per-sample
scheme. With repetition>1 each digit is written into several consecutive
samples and RX chooses the digit with minimum aggregate circular distance,
which provides simple redundancy against sample perturbations.

Unlike the tone transport, QIM does not add extra samples and does not add an
independent audible waveform. It slightly changes the values of existing PCM
samples. Lossy codecs such as Opus may destroy sample-level residues, so this
transport is intentionally experimental and should be evaluated, not assumed
reliable.
"""

from __future__ import annotations

import binascii
import numpy as np

from audio_sync import AudioData
from sync import SyncInfo


class QIMSyncCodec:
    FULL_HEADER = 0xA1
    PERIODIC_HEADER = 0xA0

    # A fixed base-4 sync word. It carries no metadata; it only marks the
    # beginning of a QIM packet. Its irregular pattern reduces accidental
    # matches compared with a trivial 0,1,2,3 repetition.
    PREAMBLE_DIGITS = np.array(
        [0, 3, 1, 2, 3, 0, 2, 1, 1, 3, 2, 0, 3, 1, 0, 2],
        dtype=np.int8,
    )

    def __init__(
        self,
        sample_rate: int = 44100,
        base: int = 64,
        repetition: int = 1,
        detection_threshold: float = 0.80,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if base <= 0 or base % 4 != 0:
            raise ValueError("QIM base must be > 0 and divisible by 4")
        if repetition <= 0:
            raise ValueError("QIM repetition must be >= 1")
        if not (0.0 < detection_threshold <= 1.0):
            raise ValueError("QIM detection_threshold must be in (0, 1]")

        self.sample_rate = int(sample_rate)
        self.base = int(base)
        self.step = self.base // 4
        self.repetition = int(repetition)
        self.detection_threshold = float(detection_threshold)

        self._centers = np.arange(4, dtype=np.int64) * self.step

        # Precompute circular distances for every possible residue.
        #
        # This is important for real-time performance. The preamble detector
        # scans many possible starting positions, so recalculating these
        # distances in Python for every position is far too expensive.
        residue_values = np.arange(
            self.base,
            dtype=np.int64,
        ).reshape(-1, 1)

        direct = np.abs(
            residue_values
            - self._centers.reshape(1, -1)
        )

        self._distance_lut = np.minimum(
            direct,
            self.base - direct,
        ).astype(np.int16)

        self._preamble_symbols = np.repeat(
            self.PREAMBLE_DIGITS,
            self.repetition,
        ).astype(np.int8)

    @staticmethod
    def _crc16(data: bytes) -> int:
        return binascii.crc_hqx(data, 0xFFFF)

    @staticmethod
    def _nonce_to_bytes(nonce: str) -> bytes:
        try:
            raw = bytes.fromhex(nonce)
        except ValueError as exc:
            raise ValueError(
                "session_nonce for QIM sync must be a hex string"
            ) from exc

        if len(raw) != 16:
            raise ValueError(
                "session_nonce for QIM sync must contain exactly 16 bytes "
                "(32 hex characters)"
            )

        return raw

    def _pack(self, info: SyncInfo) -> bytes:
        if info.next_frame_index < 0 or info.next_frame_index >= 2**32:
            raise ValueError(
                "next_frame_index must fit uint32"
            )

        body = bytearray()

        body.append(
            self.FULL_HEADER
            if info.is_full
            else self.PERIODIC_HEADER
        )

        body.append(
            info.sequence & 0xFF
        )

        body.extend(
            info.next_frame_index.to_bytes(
                4,
                "big",
                signed=False,
            )
        )

        if info.is_full:
            if info.session_nonce is None:
                raise ValueError(
                    "full QIM sync requires session_nonce"
                )

            body.extend(
                self._nonce_to_bytes(
                    info.session_nonce
                )
            )

        crc = self._crc16(
            bytes(body)
        )

        body.extend(
            crc.to_bytes(
                2,
                "big",
            )
        )

        return bytes(body)

    def _unpack(self, packet: bytes) -> SyncInfo:
        if len(packet) < self.periodic_packet_bytes:
            raise ValueError(
                "QIM sync packet is too short"
            )

        data = packet[:-2]

        expected_crc = int.from_bytes(
            packet[-2:],
            "big",
        )

        actual_crc = self._crc16(
            data
        )

        if actual_crc != expected_crc:
            raise ValueError(
                "QIM sync CRC mismatch"
            )

        header = data[0]
        sequence = data[1]

        next_frame_index = int.from_bytes(
            data[2:6],
            "big",
        )

        if header == self.PERIODIC_HEADER:
            if len(packet) != self.periodic_packet_bytes:
                raise ValueError(
                    "invalid periodic QIM sync length"
                )

            return SyncInfo(
                next_frame_index=next_frame_index,
                sequence=sequence,
                session_nonce=None,
            )

        if header == self.FULL_HEADER:
            if len(packet) != self.full_packet_bytes:
                raise ValueError(
                    "invalid full QIM sync length"
                )

            return SyncInfo(
                next_frame_index=next_frame_index,
                sequence=sequence,
                session_nonce=data[6:22].hex(),
            )

        raise ValueError(
            "unknown QIM sync header"
        )

    @property
    def periodic_packet_bytes(self) -> int:
        return 1 + 1 + 4 + 2

    @property
    def full_packet_bytes(self) -> int:
        return 1 + 1 + 4 + 16 + 2

    @property
    def preamble_samples(self) -> int:
        return int(
            self._preamble_symbols.size
        )

    def marker_samples_for(
        self,
        info: SyncInfo,
    ) -> int:

        packet_bytes = (
            self.full_packet_bytes
            if info.is_full
            else self.periodic_packet_bytes
        )

        return (
            self.preamble_samples
            + packet_bytes
            * 4
            * self.repetition
        )

    @staticmethod
    def _bytes_to_base4(
        packet: bytes,
    ) -> np.ndarray:

        digits = np.empty(
            len(packet) * 4,
            dtype=np.int8,
        )

        pos = 0

        for byte in packet:
            digits[pos] = (
                byte >> 6
            ) & 0x03

            digits[pos + 1] = (
                byte >> 4
            ) & 0x03

            digits[pos + 2] = (
                byte >> 2
            ) & 0x03

            digits[pos + 3] = (
                byte
            ) & 0x03

            pos += 4

        return digits

    @staticmethod
    def _base4_to_bytes(
        digits: np.ndarray,
    ) -> bytes:

        digits = np.asarray(
            digits,
            dtype=np.int64,
        ).reshape(-1)

        if digits.size % 4:
            raise ValueError(
                "base-4 digit count must be divisible by 4"
            )

        if np.any(
            (digits < 0)
            | (digits > 3)
        ):
            raise ValueError(
                "invalid base-4 digit"
            )

        out = bytearray()

        for pos in range(
            0,
            digits.size,
            4,
        ):
            d0, d1, d2, d3 = (
                int(v)
                for v
                in digits[pos:pos + 4]
            )

            out.append(
                (d0 << 6)
                | (d1 << 4)
                | (d2 << 2)
                | d3
            )

        return bytes(out)

    def encode(
        self,
        info: SyncInfo,
    ) -> np.ndarray:

        packet = self._pack(
            info
        )

        payload_digits = self._bytes_to_base4(
            packet
        )

        payload_symbols = np.repeat(
            payload_digits,
            self.repetition,
        ).astype(
            np.int8
        )

        return np.concatenate(
            (
                self._preamble_symbols,
                payload_symbols,
            )
        )

    def embed(
        self,
        carrier: np.ndarray,
        symbols: np.ndarray,
    ) -> np.ndarray:

        carrier = np.asarray(
            carrier,
            dtype=np.float64,
        ).reshape(-1)

        symbols = np.asarray(
            symbols,
            dtype=np.int64,
        ).reshape(-1)

        if carrier.size != symbols.size:
            raise ValueError(
                "carrier and QIM symbol lengths must match"
            )

        if np.any(
            (symbols < 0)
            | (symbols > 3)
        ):
            raise ValueError(
                "QIM symbols must be in range 0..3"
            )

        if carrier.size == 0:
            return carrier.copy()

        pcm = np.rint(
            np.clip(
                carrier,
                -0.999,
                0.999,
            )
            * 32767.0
        ).astype(
            np.int64
        )

        base = (
            np.floor_divide(
                pcm,
                self.base,
            )
            * self.base
        )

        encoded = (
            base
            + symbols * self.step
        )

        safe_lo = -32734
        safe_hi = 32734

        encoded = np.where(
            encoded < safe_lo,
            encoded + self.base,
            encoded,
        )

        encoded = np.where(
            encoded > safe_hi,
            encoded - self.base,
            encoded,
        )

        encoded = np.clip(
            encoded,
            safe_lo,
            safe_hi,
        )

        return (
            encoded.astype(
                np.float64
            )
            / 32767.0
        )

    def _sample_residues(
        self,
        samples: np.ndarray,
    ) -> np.ndarray:

        samples = np.asarray(
            samples,
            dtype=np.float64,
        ).reshape(-1)

        pcm = np.rint(
            np.clip(
                samples,
                -1.0,
                0.999969482421875,
            )
            * 32768.0
        ).astype(
            np.int64
        )

        return np.mod(
            pcm,
            self.base,
        )

    def _distance_matrix(
        self,
        residues: np.ndarray,
    ) -> np.ndarray:

        residues = np.asarray(
            residues,
            dtype=np.int64,
        ).reshape(-1, 1)

        direct = np.abs(
            residues
            - self._centers.reshape(
                1,
                -1,
            )
        )

        return np.minimum(
            direct,
            self.base - direct,
        )

    def _decode_sample_digits(
        self,
        samples: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        residues = self._sample_residues(
            samples
        )

        distances = self._distance_matrix(
            residues
        )

        digits = np.argmin(
            distances,
            axis=1,
        ).astype(
            np.int8
        )

        confidence = (
            1.0
            - (
                np.min(
                    distances,
                    axis=1,
                ).astype(
                    np.float64
                )
                / (
                    self.step / 2.0
                )
            )
        )

        confidence = np.clip(
            confidence,
            0.0,
            1.0,
        )

        return (
            digits,
            confidence,
        )

    def _decode_digit_groups_with_confidence(
        self,
        samples: np.ndarray,
        digit_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:

        samples = np.asarray(
            samples,
            dtype=np.float64,
        ).reshape(-1)

        needed = (
            digit_count
            * self.repetition
        )

        if samples.size < needed:
            raise ValueError(
                "not enough samples for QIM digits"
            )

        residues = self._sample_residues(
            samples[:needed]
        )

        out = np.empty(
            digit_count,
            dtype=np.int8,
        )

        confidence = np.empty(
            digit_count,
            dtype=np.float64,
        )

        for idx in range(
            digit_count
        ):
            group = residues[
                idx * self.repetition:
                (idx + 1) * self.repetition
            ]

            distances = self._distance_matrix(
                group
            )

            totals = np.sum(
                distances,
                axis=0,
            ).astype(
                np.float64
            )

            order = np.argsort(
                totals
            )

            best_idx = int(
                order[0]
            )

            best = float(
                totals[best_idx]
            )

            second = float(
                totals[int(order[1])]
            )

            out[idx] = best_idx

            confidence[idx] = np.clip(
                (
                    second - best
                )
                / max(
                    second,
                    1e-12,
                ),
                0.0,
                1.0,
            )

        return (
            out,
            confidence,
        )

    def _decode_digit_groups(
        self,
        samples: np.ndarray,
        digit_count: int,
    ) -> np.ndarray:

        digits, _confidence = (
            self._decode_digit_groups_with_confidence(
                samples,
                digit_count,
            )
        )

        return digits

    def _decode_bytes(
        self,
        samples: np.ndarray,
        byte_count: int,
    ) -> bytes:

        digit_count = (
            byte_count * 4
        )

        digits = self._decode_digit_groups(
            samples,
            digit_count,
        )

        return self._base4_to_bytes(
            digits
        )

    def _candidate_packet_size(
        self,
        samples_after_preamble: np.ndarray,
    ) -> int | None:

        needed = (
            4
            * self.repetition
        )

        if (
            samples_after_preamble.size
            < needed
        ):
            return None

        header = self._decode_bytes(
            samples_after_preamble,
            1,
        )[0]

        if (
            header
            == self.FULL_HEADER
        ):
            return (
                self.full_packet_bytes
            )

        if (
            header
            == self.PERIODIC_HEADER
        ):
            return (
                self.periodic_packet_bytes
            )

        return -1

    def find_preamble(
        self,
        samples: np.ndarray,
    ) -> tuple[int | None, float]:

        """
        Find the QIM preamble using repetition-aware
        vectorized decoding.

        Example for repetition=4:

            4 samples
                ↓
            one logical QIM digit

        This is done for all 16 preamble digits and
        for all possible sliding start positions at once.
        """

        samples = np.asarray(
            samples,
            dtype=np.float64,
        ).reshape(-1)

        n = self.preamble_samples

        if samples.size < n:
            return (
                None,
                0.0,
            )

        residues = self._sample_residues(
            samples
        )

        digit_count = int(
            self.PREAMBLE_DIGITS.size
        )

        limit = int(
            residues.size
            - n
            + 1
        )

        # All possible sliding windows.
        #
        # Shape:
        # [candidate_start, preamble_sample]
        starts = np.arange(
            limit,
            dtype=np.int64,
        ).reshape(
            -1,
            1,
        )

        offsets = np.arange(
            n,
            dtype=np.int64,
        ).reshape(
            1,
            -1,
        )

        windows = residues[
            starts + offsets
        ]

        # Fast LUT lookup.
        #
        # Output shape:
        # [candidate, sample, 4 possible symbols]
        distances = self._distance_lut[
            windows
        ]

        # Convert:
        #
        # [candidate, 64 samples, 4]
        #
        # into:
        #
        # [candidate, 16 logical digits, repeat=4, 4]
        distances = distances.reshape(
            limit,
            digit_count,
            self.repetition,
            4,
        )

        # Combine all repeated samples belonging
        # to the same logical digit.
        totals = np.sum(
            distances,
            axis=2,
        )

        # Winner for every logical digit.
        got = np.argmin(
            totals,
            axis=2,
        ).astype(
            np.int8
        )

        # Fraction of the 16 preamble digits
        # that were reconstructed correctly.
        scores = np.mean(
            got
            == self.PREAMBLE_DIGITS.reshape(
                1,
                -1,
            ),
            axis=1,
        ).astype(
            np.float64
        )

        best_score = (
            float(
                np.max(
                    scores
                )
            )
            if scores.size
            else 0.0
        )

        hits = np.flatnonzero(
            scores
            >= self.detection_threshold
        )

        if hits.size:
            # Earliest valid preamble, not simply
            # the strongest one later in the buffer.
            pos = int(
                hits[0]
            )

            return (
                pos,
                float(
                    scores[pos]
                ),
            )

        return (
            None,
            best_score,
        )

    def decode_exact_marker(
        self,
        marker: np.ndarray,
    ) -> SyncInfo:

        marker = np.asarray(
            marker,
            dtype=np.float64,
        ).reshape(-1)

        if marker.size < (
            self.preamble_samples
            + 4 * self.repetition
        ):
            raise ValueError(
                "QIM marker is too short"
            )

        payload = marker[
            self.preamble_samples:
        ]

        packet_size = (
            self._candidate_packet_size(
                payload
            )
        )

        if (
            packet_size is None
            or packet_size < 0
        ):
            raise ValueError(
                "unknown QIM sync marker header"
            )

        needed = (
            packet_size
            * 4
            * self.repetition
        )

        packet = self._decode_bytes(
            payload[:needed],
            packet_size,
        )

        return self._unpack(
            packet
        )


class QIMSyncStreamDecoder:
    """
    Streaming QIM detector with diagnostics.
    """

    def __init__(
        self,
        codec: QIMSyncCodec,
    ):
        self.codec = codec

        self.buffer = np.empty(
            0,
            dtype=np.float64,
        )

        self.total_preamble_hits = 0
        self.total_bad_header = 0
        self.total_crc_fail = 0
        self.total_decode_fail = 0
        self.total_periodic_ok = 0
        self.total_full_ok = 0

        self._diag_samples = 0
        self._diag_sum_squares = 0.0
        self._diag_peak = 0.0
        self._diag_best_preamble_score = 0.0
        self._diag_preamble_hits = 0
        self._diag_bad_header = 0
        self._diag_crc_fail = 0
        self._diag_decode_fail = 0
        self._diag_periodic_ok = 0
        self._diag_full_ok = 0
        self._diag_last_bad_header = None

    def _record_audio(
        self,
        samples: np.ndarray,
    ) -> None:

        if samples.size == 0:
            return

        values = np.asarray(
            samples,
            dtype=np.float64,
        ).reshape(-1)

        self._diag_samples += int(
            values.size
        )

        self._diag_sum_squares += float(
            np.dot(
                values,
                values,
            )
        )

        self._diag_peak = max(
            self._diag_peak,
            float(
                np.max(
                    np.abs(
                        values
                    )
                )
            ),
        )

    def _record_preamble_score(
        self,
        score: float,
    ) -> None:

        self._diag_best_preamble_score = max(
            self._diag_best_preamble_score,
            float(score),
        )

    def _record_preamble_hit(
        self,
    ) -> None:

        self.total_preamble_hits += 1
        self._diag_preamble_hits += 1

    def _record_bad_header(
        self,
        header: int | None,
    ) -> None:

        self.total_bad_header += 1
        self._diag_bad_header += 1
        self._diag_last_bad_header = header

    def _record_crc_fail(
        self,
    ) -> None:

        self.total_crc_fail += 1
        self._diag_crc_fail += 1

    def _record_decode_fail(
        self,
    ) -> None:

        self.total_decode_fail += 1
        self._diag_decode_fail += 1

    def _record_ok(
        self,
        info: SyncInfo,
    ) -> None:

        if info.is_full:
            self.total_full_ok += 1
            self._diag_full_ok += 1

        else:
            self.total_periodic_ok += 1
            self._diag_periodic_ok += 1

    @staticmethod
    def _dbfs(
        value: float,
    ) -> float:

        if value <= 1e-12:
            return -240.0

        return (
            20.0
            * float(
                np.log10(
                    value
                )
            )
        )

    def diagnostics_snapshot(
        self,
        reset_interval: bool = True,
    ) -> dict:

        if self._diag_samples > 0:
            rms = float(
                np.sqrt(
                    self._diag_sum_squares
                    / self._diag_samples
                )
            )

        else:
            rms = 0.0

        result = {
            "samples":
                self._diag_samples,

            "rms_dbfs":
                self._dbfs(rms),

            "peak_dbfs":
                self._dbfs(
                    self._diag_peak
                ),

            "best_preamble_score":
                self._diag_best_preamble_score,

            "threshold":
                self.codec.detection_threshold,

            "preamble_hits":
                self._diag_preamble_hits,

            "bad_header":
                self._diag_bad_header,

            "crc_fail":
                self._diag_crc_fail,

            "decode_fail":
                self._diag_decode_fail,

            "periodic_ok":
                self._diag_periodic_ok,

            "full_ok":
                self._diag_full_ok,

            "last_bad_header":
                self._diag_last_bad_header,

            "buffered_samples":
                int(
                    self.buffer.size
                ),

            "total_preamble_hits":
                self.total_preamble_hits,

            "total_bad_header":
                self.total_bad_header,

            "total_crc_fail":
                self.total_crc_fail,

            "total_decode_fail":
                self.total_decode_fail,

            "total_periodic_ok":
                self.total_periodic_ok,

            "total_full_ok":
                self.total_full_ok,
        }

        if reset_interval:
            self._diag_samples = 0
            self._diag_sum_squares = 0.0
            self._diag_peak = 0.0
            self._diag_best_preamble_score = 0.0
            self._diag_preamble_hits = 0
            self._diag_bad_header = 0
            self._diag_crc_fail = 0
            self._diag_decode_fail = 0
            self._diag_periodic_ok = 0
            self._diag_full_ok = 0
            self._diag_last_bad_header = None

        return result

    def feed(
        self,
        samples: np.ndarray,
    ) -> list[AudioData | SyncInfo]:

        samples = np.asarray(
            samples,
            dtype=np.float64,
        ).reshape(-1)

        self._record_audio(
            samples
        )

        if samples.size:
            self.buffer = np.concatenate(
                (
                    self.buffer,
                    samples,
                )
            )

        events: list[
            AudioData | SyncInfo
        ] = []

        keep = (
            self.codec.preamble_samples
            - 1
        )

        while True:
            if (
                self.buffer.size
                < self.codec.preamble_samples
            ):
                break

            pos, score = (
                self.codec.find_preamble(
                    self.buffer
                )
            )

            self._record_preamble_score(
                score
            )

            if pos is None:
                safe = (
                    self.buffer.size
                    - keep
                )

                if safe > 0:
                    events.append(
                        AudioData(
                            self.buffer[
                                :safe
                            ].copy()
                        )
                    )

                    self.buffer = (
                        self.buffer[
                            safe:
                        ]
                    )

                break

            if pos > 0:
                events.append(
                    AudioData(
                        self.buffer[
                            :pos
                        ].copy()
                    )
                )

                self.buffer = (
                    self.buffer[
                        pos:
                    ]
                )

                continue

            min_header = (
                self.codec.preamble_samples
                + 4
                * self.codec.repetition
            )

            if (
                self.buffer.size
                < min_header
            ):
                break

            payload = self.buffer[
                self.codec.preamble_samples:
            ]

            header = int(
                self.codec._decode_bytes(
                    payload,
                    1,
                )[0]
            )

            packet_bytes = (
                self.codec._candidate_packet_size(
                    payload
                )
            )

            if packet_bytes is None:
                break

            if packet_bytes < 0:
                self._record_preamble_hit()

                self._record_bad_header(
                    header
                )

                events.append(
                    AudioData(
                        self.buffer[
                            :1
                        ].copy()
                    )
                )

                self.buffer = (
                    self.buffer[
                        1:
                    ]
                )

                continue

            marker_len = (
                self.codec.preamble_samples
                + packet_bytes
                * 4
                * self.codec.repetition
            )

            if (
                self.buffer.size
                < marker_len
            ):
                break

            self._record_preamble_hit()

            marker_rx = self.buffer[
                :marker_len
            ].copy()

            try:
                info = (
                    self.codec.decode_exact_marker(
                        marker_rx
                    )
                )

            except ValueError as exc:
                if (
                    "CRC mismatch"
                    in str(exc)
                ):
                    self._record_crc_fail()

                else:
                    self._record_decode_fail()

                events.append(
                    AudioData(
                        self.buffer[
                            :1
                        ].copy()
                    )
                )

                self.buffer = (
                    self.buffer[
                        1:
                    ]
                )

                continue

            self._record_ok(
                info
            )

            # SyncInfo is emitted before the QIM-modified audio
            # so app.py can correct frame_index / nonce first.
            events.append(
                info
            )

            events.append(
                AudioData(
                    marker_rx
                )
            )

            self.buffer = (
                self.buffer[
                    marker_len:
                ]
            )

        return events

    def flush(
        self,
    ) -> list[AudioData]:

        if self.buffer.size == 0:
            return []

        out = [
            AudioData(
                self.buffer.copy()
            )
        ]

        self.buffer = np.empty(
            0,
            dtype=np.float64,
        )

        return out