"""QIM/base-4 in-band synchronization transport.

Protocol used by this file:

* fixed balanced QIM preamble (32 logical base-4 digits);
* payload bytes are converted to base-4 (2 bits per logical digit);
* repetition is INTERLEAVED: the complete payload is written R times instead
  of writing one logical digit R adjacent times;
* CRC32 protects the packet;
* RX does soft/list decoding: it keeps distances to all four QIM residues and
  searches several likely payload variants instead of requiring every logical
  digit to be hard-decoded correctly on the first try;
* packet type is constrained to A0 (PERIODIC) or A1 (FULL);
* a recovered packet is accepted only when its CRC32 is valid.

Both peers must use this same file because CRC width, preamble and repetition
layout are part of the wire format.
"""

from __future__ import annotations

import binascii
import heapq
from dataclasses import dataclass

import numpy as np

from audio_sync import AudioData
from sync import SyncInfo


@dataclass
class _SoftHypothesis:
    shift: int
    header: int
    packet_bytes: int
    header_cost: int
    raw_header: int | None


class QIMSyncCodec:
    FULL_HEADER = 0xA1
    PERIODIC_HEADER = 0xA0

    # 32 logical digits, balanced:
    # each value 0/1/2/3 occurs exactly 8 times.
    PREAMBLE_DIGITS = np.array(
        [
            0, 3, 1, 2, 3, 0, 2, 1,
            1, 3, 2, 0, 3, 1, 0, 2,
            2, 0, 3, 1, 0, 2, 1, 3,
            3, 1, 0, 2, 1, 3, 2, 0,
        ],
        dtype=np.int8,
    )

    def __init__(
        self,
        sample_rate: int = 44100,
        base: int = 64,
        repetition: int = 1,
        detection_threshold: float = 0.80,
        quantum: int | None = None,
    ):
        # Current app.py uses base=...
        # quantum=... is accepted too for compatibility.
        if quantum is not None:
            base = quantum

        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")

        if base <= 0 or base % 4 != 0:
            raise ValueError(
                "QIM base/quantum must be > 0 and divisible by 4"
            )

        if repetition <= 0:
            raise ValueError(
                "QIM repetition must be >= 1"
            )

        if not (0.0 < detection_threshold <= 1.0):
            raise ValueError(
                "QIM detection_threshold must be in (0, 1]"
            )

        self.sample_rate = int(sample_rate)

        self.base = int(base)
        self.quantum = self.base

        self.step = self.base // 4

        self.repetition = int(repetition)

        self.detection_threshold = float(
            detection_threshold
        )

        self._centers = (
            np.arange(
                4,
                dtype=np.int64,
            )
            * self.step
        )

        residue_values = np.arange(
            self.base,
            dtype=np.int64,
        ).reshape(
            -1,
            1,
        )

        direct = np.abs(
            residue_values
            - self._centers.reshape(
                1,
                -1,
            )
        )

        self._distance_lut = np.minimum(
            direct,
            self.base - direct,
        ).astype(
            np.int16
        )

        # Preamble still uses local repetition so detector can
        # aggregate R observations of every preamble digit.
        #
        # PAYLOAD repetition is different:
        # whole copies of payload are transmitted.
        self._preamble_symbols = np.repeat(
            self.PREAMBLE_DIGITS,
            self.repetition,
        ).astype(
            np.int8
        )

    @staticmethod
    def _crc32(
        data: bytes,
    ) -> int:
        return (
            binascii.crc32(data)
            & 0xFFFFFFFF
        )

    def _pack(
        self,
        info: SyncInfo,
    ) -> bytes:
        if (
            info.next_frame_index < 0
            or info.next_frame_index >= 2**32
        ):
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
            info.sequence
            & 0xFF
        )

        body.extend(
            info.next_frame_index.to_bytes(
                4,
                "big",
                signed=False,
            )
        )


        crc = self._crc32(
            bytes(body)
        )

        body.extend(
            crc.to_bytes(
                4,
                "big",
            )
        )

        return bytes(body)

    def _unpack(
        self,
        packet: bytes,
    ) -> SyncInfo:
        if (
            len(packet)
            < self.periodic_packet_bytes
        ):
            raise ValueError(
                "QIM sync packet is too short"
            )

        data = packet[:-4]

        expected_crc = int.from_bytes(
            packet[-4:],
            "big",
        )

        actual_crc = self._crc32(
            data
        )

        if (
            actual_crc
            != expected_crc
        ):
            raise ValueError(
                "QIM sync CRC mismatch"
            )

        header = data[0]
        sequence = data[1]

        next_frame_index = int.from_bytes(
            data[2:6],
            "big",
        )

        if (
            header
            == self.PERIODIC_HEADER
        ):
            if (
                len(packet)
                != self.periodic_packet_bytes
            ):
                raise ValueError(
                    "invalid periodic QIM sync length"
                )

            return SyncInfo(
                next_frame_index=next_frame_index,
                sequence=sequence,
                is_full=False,
            )

        if (
            header
            == self.FULL_HEADER
        ):
            if (
                len(packet)
                != self.full_packet_bytes
            ):
                raise ValueError(
                    "invalid full QIM sync length"
                )

            return SyncInfo(
                next_frame_index=next_frame_index,
                sequence=sequence,
                is_full=True,
            )

        raise ValueError(
            "unknown QIM sync header"
        )

    @property
    def periodic_packet_bytes(
        self,
    ) -> int:
        # header
        # sequence
        # uint32 frame
        # CRC32
        return (
            1
            + 1
            + 4
            + 4
        )

    @property
    def full_packet_bytes(
        self,
    ) -> int:
        return self.periodic_packet_bytes

    @property
    def preamble_samples(
        self,
    ) -> int:
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
                "base-4 digit count "
                "must be divisible by 4"
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
                in digits[
                    pos:
                    pos + 4
                ]
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

        payload_digits = (
            self._bytes_to_base4(
                packet
            )
        )

        # OLD R=4:
        #
        # d0 d0 d0 d0
        # d1 d1 d1 d1
        # d2 d2 d2 d2
        #
        # NEW R=4:
        #
        # full payload copy #1
        # full payload copy #2
        # full payload copy #3
        # full payload copy #4
        #
        # Same logical symbol is now separated in time.
        payload_symbols = np.tile(
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

        if (
            carrier.size
            != symbols.size
        ):
            raise ValueError(
                "carrier and QIM symbol "
                "lengths must match"
            )

        if np.any(
            (symbols < 0)
            | (symbols > 3)
        ):
            raise ValueError(
                "QIM symbols must be 0..3"
            )

        if carrier.size == 0:
            return carrier.copy()

        # Match app.py TX int16 conversion.
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

        lattice = (
            np.floor_divide(
                pcm,
                self.base,
            )
            * self.base
        )

        encoded = (
            lattice
            + symbols
            * self.step
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

        # app.py RX:
        #
        # int16 -> float / 32768
        #
        # so 32768 reconstructs received PCM integers.
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
        ).reshape(
            -1,
            1,
        )

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
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        residues = (
            self._sample_residues(
                samples
            )
        )

        distances = (
            self._distance_matrix(
                residues
            )
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

    def _decode_digit_groups(
        self,
        samples: np.ndarray,
        digit_count: int,
    ) -> np.ndarray:
        """
        Decode interleaved/copy-major repeated payload.
        """

        needed = (
            digit_count
            * self.repetition
        )

        if (
            samples.size
            < needed
        ):
            raise ValueError(
                "not enough samples "
                "for QIM digits"
            )

        residues = (
            self._sample_residues(
                samples[
                    :needed
                ]
            )
        )

        distances = (
            self._distance_matrix(
                residues
            )
        )

        distances = distances.reshape(
            self.repetition,
            digit_count,
            4,
        )

        totals = np.sum(
            distances,
            axis=0,
        )

        return np.argmin(
            totals,
            axis=1,
        ).astype(
            np.int8
        )

    def _decode_bytes(
        self,
        samples: np.ndarray,
        byte_count: int,
    ) -> bytes:
        digit_count = (
            byte_count
            * 4
        )

        digits = (
            self._decode_digit_groups(
                samples,
                digit_count,
            )
        )

        return (
            self._base4_to_bytes(
                digits
            )
        )

    def _candidate_packet_size(
        self,
        samples_after_preamble: np.ndarray,
    ) -> int | None:
        """
        Compatibility helper only.

        Streaming decoder below DOES NOT require exact hard header.
        """

        if (
            samples_after_preamble.size
            < 4
        ):
            return None

        digits, _ = (
            self._decode_sample_digits(
                samples_after_preamble[
                    :4
                ]
            )
        )

        header = (
            self._base4_to_bytes(
                digits
            )[0]
        )

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
    ) -> tuple[
        int | None,
        float,
    ]:
        samples = np.asarray(
            samples,
            dtype=np.float64,
        ).reshape(-1)

        n = (
            self.preamble_samples
        )

        if (
            samples.size
            < n
        ):
            return (
                None,
                0.0,
            )

        residues = (
            self._sample_residues(
                samples
            )
        )

        digit_count = int(
            self.PREAMBLE_DIGITS.size
        )

        limit = int(
            residues.size
            - n
            + 1
        )

        if limit <= 0:
            return (
                None,
                0.0,
            )

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

        distances = (
            self._distance_lut[
                windows
            ]
        )

        distances = distances.reshape(
            limit,
            digit_count,
            self.repetition,
            4,
        )

        totals = np.sum(
            distances,
            axis=2,
        )

        got = np.argmin(
            totals,
            axis=2,
        ).astype(
            np.int8
        )

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
        """
        Local/exact helper.

        Real network RX uses soft decoder below.
        """

        marker = np.asarray(
            marker,
            dtype=np.float64,
        ).reshape(-1)

        if (
            marker.size
            < self.preamble_samples + 4
        ):
            raise ValueError(
                "QIM marker is too short"
            )

        payload = marker[
            self.preamble_samples:
        ]

        for packet_bytes in (
            self.periodic_packet_bytes,
            self.full_packet_bytes,
        ):
            needed = (
                packet_bytes
                * 4
                * self.repetition
            )

            if (
                payload.size
                < needed
            ):
                continue

            try:
                packet = (
                    self._decode_bytes(
                        payload[
                            :needed
                        ],
                        packet_bytes,
                    )
                )

                return (
                    self._unpack(
                        packet
                    )
                )

            except ValueError:
                pass

        raise ValueError(
            "could not decode QIM marker"
        )


class QIMSyncStreamDecoder:
    """
    Streaming QIM decoder.

    Main differences from old decoder:

    1. Header does not have to hard-decode exactly.
       Only A0 and A1 are legal hypotheses.

    2. Every logical digit retains costs for ALL
       four possible QIM symbols.

    3. Beam search keeps several likely packet
       interpretations.

    4. Candidate is accepted ONLY if CRC32 passes.
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

        self.phase_search_radius = max(
            8,
            2 * self.codec.repetition,
        )

        # Number of complete packet candidates
        # retained during soft decoding.
        self.beam_width = 1024

        # Avoid running a full beam over every
        # shift/type combination.
        self.max_hypotheses = 10

        self.total_preamble_hits = 0
        self.total_bad_header = 0
        self.total_crc_fail = 0
        self.total_decode_fail = 0

        self.total_periodic_ok = 0
        self.total_full_ok = 0

        self.total_soft_repaired = 0

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

        self._diag_soft_repaired = 0

        self._diag_last_bad_header = None

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

    @staticmethod
    def _byte_digits(
        value: int,
    ) -> np.ndarray:
        return np.array(
            [
                (value >> 6) & 0x03,
                (value >> 4) & 0x03,
                (value >> 2) & 0x03,
                value & 0x03,
            ],
            dtype=np.int8,
        )

    def _record_audio(
        self,
        samples: np.ndarray,
    ) -> None:
        if samples.size == 0:
            return

        x = np.asarray(
            samples,
            dtype=np.float64,
        ).reshape(-1)

        self._diag_samples += int(
            x.size
        )

        self._diag_sum_squares += float(
            np.dot(
                x,
                x,
            )
        )

        self._diag_peak = max(
            self._diag_peak,
            float(
                np.max(
                    np.abs(
                        x
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
        repaired: bool,
    ) -> None:
        if info.is_full:
            self.total_full_ok += 1
            self._diag_full_ok += 1

        else:
            self.total_periodic_ok += 1
            self._diag_periodic_ok += 1

        if repaired:
            self.total_soft_repaired += 1
            self._diag_soft_repaired += 1

    def _logical_preamble_debug(
        self,
    ) -> tuple[
        str,
        int,
    ]:
        try:
            samples = self.buffer[
                :self.codec.preamble_samples
            ]

            residues = (
                self.codec._sample_residues(
                    samples
                )
            )

            distances = (
                self.codec._distance_matrix(
                    residues
                )
            )

            distances = distances.reshape(
                int(
                    self.codec.PREAMBLE_DIGITS.size
                ),
                self.codec.repetition,
                4,
            )

            totals = np.sum(
                distances,
                axis=1,
            )

            got = np.argmin(
                totals,
                axis=1,
            ).astype(
                np.int8
            )

        except Exception:
            return (
                "short",
                0,
            )

        matches = int(
            np.sum(
                got
                == self.codec.PREAMBLE_DIGITS
            )
        )

        text = "".join(
            str(
                int(v)
            )
            for v
            in got
        )

        return (
            text,
            matches,
        )

    def _raw_header_first_copy(
        self,
        payload_start: int,
    ) -> int | None:
        if (
            payload_start < 0
            or self.buffer.size
            < payload_start + 4
        ):
            return None

        digits, _ = (
            self.codec._decode_sample_digits(
                self.buffer[
                    payload_start:
                    payload_start + 4
                ]
            )
        )

        return int(
            self.codec._base4_to_bytes(
                digits
            )[0]
        )

    def _header_cost_first_copy(
        self,
        payload_start: int,
        header: int,
    ) -> int | None:
        if (
            payload_start < 0
            or self.buffer.size
            < payload_start + 4
        ):
            return None

        residues = (
            self.codec._sample_residues(
                self.buffer[
                    payload_start:
                    payload_start + 4
                ]
            )
        )

        distances = (
            self.codec._distance_matrix(
                residues
            )
        )

        expected = (
            self._byte_digits(
                header
            )
        )

        return int(
            sum(
                distances[
                    i,
                    int(
                        expected[i]
                    ),
                ]
                for i
                in range(4)
            )
        )

    def _build_hypotheses(
        self,
    ) -> tuple[
        list[_SoftHypothesis],
        bool,
    ]:
        hypotheses: list[
            _SoftHypothesis
        ] = []

        waiting_for_full = False

        for shift in range(
            -self.phase_search_radius,
            self.phase_search_radius + 1,
        ):
            start = (
                self.codec.preamble_samples
                + shift
            )

            if start < 0:
                continue

            raw_header = (
                self._raw_header_first_copy(
                    start
                )
            )

            for (
                header,
                packet_bytes,
            ) in (
                (
                    self.codec.PERIODIC_HEADER,
                    self.codec.periodic_packet_bytes,
                ),
                (
                    self.codec.FULL_HEADER,
                    self.codec.full_packet_bytes,
                ),
            ):
                hcost = (
                    self._header_cost_first_copy(
                        start,
                        header,
                    )
                )

                if hcost is None:
                    continue

                needed = (
                    packet_bytes
                    * 4
                    * self.codec.repetition
                )

                if (
                    self.buffer.size
                    < start + needed
                ):
                    if (
                        header
                        == self.codec.FULL_HEADER
                    ):
                        waiting_for_full = True

                    continue

                hypotheses.append(
                    _SoftHypothesis(
                        shift=shift,
                        header=header,
                        packet_bytes=packet_bytes,
                        header_cost=hcost,
                        raw_header=raw_header,
                    )
                )

        hypotheses.sort(
            key=lambda h: (
                h.header_cost,
                abs(
                    h.shift
                ),
                0
                if h.header
                == self.codec.FULL_HEADER
                else 1,
            )
        )

        return (
            hypotheses,
            waiting_for_full,
        )

    def _payload_costs(
        self,
        start: int,
        packet_bytes: int,
    ) -> np.ndarray:
        digit_count = (
            packet_bytes
            * 4
        )

        needed = (
            digit_count
            * self.codec.repetition
        )

        samples = self.buffer[
            start:
            start + needed
        ]

        if (
            samples.size
            < needed
        ):
            raise ValueError(
                "not enough samples "
                "for soft QIM payload"
            )

        residues = (
            self.codec._sample_residues(
                samples
            )
        )

        distances = (
            self.codec._distance_matrix(
                residues
            )
        )

        # encode() writes:
        #
        # payload copy 1
        # payload copy 2
        # ...
        #
        # so reshape is:
        #
        # [R][logical digit][possible symbol]
        distances = distances.reshape(
            self.codec.repetition,
            digit_count,
            4,
        )

        return np.sum(
            distances,
            axis=0,
        ).astype(
            np.int64
        )

    def _forced_hard_digits(
        self,
        costs: np.ndarray,
        header: int,
    ) -> np.ndarray:
        hard = np.argmin(
            costs,
            axis=1,
        ).astype(
            np.int8
        )

        # Header is not arbitrary.
        # It MUST be exactly A0 or A1.
        hard[:4] = (
            self._byte_digits(
                header
            )
        )

        return hard

    def _crc_valid_packet(
        self,
        packet: bytes,
    ) -> bool:
        if len(packet) not in (
            self.codec.periodic_packet_bytes,
            self.codec.full_packet_bytes,
        ):
            return False

        rx_crc = int.from_bytes(
            packet[-4:],
            "big",
        )

        calc_crc = (
            self.codec._crc32(
                packet[:-4]
            )
        )

        return (
            rx_crc
            == calc_crc
        )

    def _soft_decode_hypothesis(
        self,
        hypothesis: _SoftHypothesis,
    ) -> tuple[
        SyncInfo,
        int,
        int,
        bytes,
    ] | None:
        start = (
            self.codec.preamble_samples
            + hypothesis.shift
        )

        costs = (
            self._payload_costs(
                start,
                hypothesis.packet_bytes,
            )
        )

        digit_count = (
            costs.shape[0]
        )

        expected_header_digits = (
            self._byte_digits(
                hypothesis.header
            )
        )

        hard_digits = (
            self._forced_hard_digits(
                costs,
                hypothesis.header,
            )
        )

        hard_packet = (
            self.codec._base4_to_bytes(
                hard_digits
            )
        )

        # Fast path:
        #
        # after aggregating all R payload copies,
        # hard decision may already be correct.
        if (
            self._crc_valid_packet(
                hard_packet
            )
        ):
            try:
                info = (
                    self.codec._unpack(
                        hard_packet
                    )
                )

                return (
                    info,
                    0,
                    0,
                    hard_packet,
                )

            except ValueError:
                pass

        # Soft/list decoding.
        #
        # Beam element:
        #
        # (
        #     total QIM distance,
        #     tuple of chosen base4 digits
        # )
        #
        # First four digits are constrained to
        # legal packet header A0 or A1.
        header_cost = int(
            sum(
                costs[
                    i,
                    int(
                        expected_header_digits[i]
                    ),
                ]
                for i
                in range(4)
            )
        )

        beam: list[
            tuple[
                int,
                tuple[int, ...],
            ]
        ] = [
            (
                header_cost,
                tuple(
                    int(v)
                    for v
                    in expected_header_digits
                ),
            )
        ]

        for pos in range(
            4,
            digit_count,
        ):
            row = costs[
                pos
            ]

            # Try all 4 symbols, but in likely order.
            order = np.argsort(
                row
            )

            expanded: list[
                tuple[
                    int,
                    tuple[int, ...],
                ]
            ] = []

            for (
                current_cost,
                digits,
            ) in beam:
                for choice in order:
                    d = int(
                        choice
                    )

                    expanded.append(
                        (
                            current_cost
                            + int(
                                row[d]
                            ),
                            digits
                            + (d,),
                        )
                    )

            if (
                len(expanded)
                > self.beam_width
            ):
                beam = heapq.nsmallest(
                    self.beam_width,
                    expanded,
                    key=lambda item: item[0],
                )

            else:
                expanded.sort(
                    key=lambda item: item[0]
                )

                beam = expanded

        beam.sort(
            key=lambda item: item[0]
        )

        hard_arr = np.asarray(
            hard_digits,
            dtype=np.int8,
        )

        # We do NOT accept the lowest-distance packet blindly.
        #
        # We walk likely packets until one satisfies CRC32.
        for (
            rank,
            (
                _score,
                digits_tuple,
            ),
        ) in enumerate(
            beam,
            start=1,
        ):
            candidate_digits = np.asarray(
                digits_tuple,
                dtype=np.int8,
            )

            packet = (
                self.codec._base4_to_bytes(
                    candidate_digits
                )
            )

            if not self._crc_valid_packet(
                packet
            ):
                continue

            try:
                info = (
                    self.codec._unpack(
                        packet
                    )
                )

            except ValueError:
                continue

            if (
                hypothesis.header
                == self.codec.FULL_HEADER
                and not info.is_full
            ):
                continue

            if (
                hypothesis.header
                == self.codec.PERIODIC_HEADER
                and info.is_full
            ):
                continue

            changed = int(
                np.sum(
                    candidate_digits
                    != hard_arr
                )
            )

            return (
                info,
                rank,
                changed,
                packet,
            )

        return None

    def _hard_failure_details(
        self,
        hypothesis: _SoftHypothesis,
    ) -> tuple[
        bytes,
        int,
        int,
    ]:
        start = (
            self.codec.preamble_samples
            + hypothesis.shift
        )

        costs = (
            self._payload_costs(
                start,
                hypothesis.packet_bytes,
            )
        )

        hard_digits = (
            self._forced_hard_digits(
                costs,
                hypothesis.header,
            )
        )

        packet = (
            self.codec._base4_to_bytes(
                hard_digits
            )
        )

        rx_crc = int.from_bytes(
            packet[-4:],
            "big",
        )

        calc_crc = (
            self.codec._crc32(
                packet[:-4]
            )
        )

        return (
            packet,
            rx_crc,
            calc_crc,
        )

    def _try_candidate_soft(
        self,
        preamble_score: float,
    ) -> tuple[
        str,
        object | None,
    ]:
        (
            hypotheses,
            waiting_for_full,
        ) = self._build_hypotheses()

        if not hypotheses:
            if waiting_for_full:
                return (
                    "need_more",
                    None,
                )

            return (
                "decode_fail",
                None,
            )

        (
            preamble_text,
            preamble_matches,
        ) = (
            self._logical_preamble_debug()
        )

        tried = 0

        for hypothesis in hypotheses[
            :self.max_hypotheses
        ]:
            tried += 1

            try:
                decoded = (
                    self._soft_decode_hypothesis(
                        hypothesis
                    )
                )

            except Exception:
                decoded = None

            if decoded is None:
                continue

            (
                info,
                rank,
                changed,
                packet,
            ) = decoded

            repaired = (
                rank != 0
                or changed != 0
                or hypothesis.shift != 0
            )

            consume_len = (
                self.codec.preamble_samples
                + hypothesis.shift
                + hypothesis.packet_bytes
                * 4
                * self.codec.repetition
            )

            raw_header_text = (
                "none"
                if hypothesis.raw_header is None
                else (
                    f"0x"
                    f"{hypothesis.raw_header:02X}"
                )
            )

            print(
                "QIM_SOFT_OK "
                f"score={preamble_score:.3f} "
                f"pre={preamble_matches}/"
                f"{self.codec.PREAMBLE_DIGITS.size} "
                f"shift={hypothesis.shift:+d} "
                f"raw_header={raw_header_text} "
                f"header=0x{hypothesis.header:02X} "
                f"rank={rank} "
                f"changed_digits={changed} "
                f"seq={info.sequence} "
                f"next={info.next_frame_index} "
                f"full={info.is_full} "
                f"repaired={repaired} "
                f"packet={packet.hex()}",
                flush=True,
            )

            return (
                "ok",
                (
                    info,
                    consume_len,
                    repaired,
                ),
            )

        best = hypotheses[0]

        try:
            (
                hard_packet,
                rx_crc,
                calc_crc,
            ) = (
                self._hard_failure_details(
                    best
                )
            )

            hard_text = (
                hard_packet.hex()
            )

            crc_text = (
                f"crc_rx=0x{rx_crc:08X} "
                f"crc_calc=0x{calc_crc:08X}"
            )

        except Exception:
            hard_text = (
                "unavailable"
            )

            crc_text = (
                "crc_rx=none "
                "crc_calc=none"
            )

        raw_header_text = (
            "none"
            if best.raw_header is None
            else (
                f"0x"
                f"{best.raw_header:02X}"
            )
        )

        print(
            "QIM_SOFT_FAIL "
            f"score={preamble_score:.3f} "
            f"pre={preamble_matches}/"
            f"{self.codec.PREAMBLE_DIGITS.size} "
            f"got={preamble_text} "
            f"tried={tried} "
            f"best_shift={best.shift:+d} "
            f"raw_header={raw_header_text} "
            f"forced_header=0x{best.header:02X} "
            f"header_cost={best.header_cost} "
            f"hard_packet={hard_text} "
            f"{crc_text}",
            flush=True,
        )

        return (
            "crc_fail",
            best.raw_header,
        )

    def diagnostics_snapshot(
        self,
        reset_interval: bool = True,
    ) -> dict:
        if (
            self._diag_samples > 0
        ):
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
                self._dbfs(
                    rms
                ),

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

            "soft_repaired":
                self._diag_soft_repaired,

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

            "total_soft_repaired":
                self.total_soft_repaired,
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

            self._diag_soft_repaired = 0

            self._diag_last_bad_header = None

        return result

    def feed(
        self,
        samples: np.ndarray,
    ) -> list[
        AudioData | SyncInfo
    ]:
        samples = np.asarray(
            samples,
            dtype=np.float64,
        ).reshape(-1)

        self._record_audio(
            samples
        )

        if samples.size:
            self.buffer = (
                np.concatenate(
                    (
                        self.buffer,
                        samples,
                    )
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

            (
                pos,
                score,
            ) = (
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

            # Candidate begins at buffer[0].
            #
            # Need at least four first-copy payload samples
            # before header hypotheses can be ranked.
            if (
                self.buffer.size
                < self.codec.preamble_samples + 4
            ):
                break

            self._record_preamble_hit()

            (
                status,
                data,
            ) = (
                self._try_candidate_soft(
                    score
                )
            )

            if (
                status
                == "need_more"
            ):
                # We counted it before finding out
                # that complete FULL packet has not
                # arrived yet.
                self.total_preamble_hits -= 1
                self._diag_preamble_hits -= 1

                break

            if (
                status
                == "ok"
            ):
                (
                    info,
                    consume_len,
                    repaired,
                ) = data

                self._record_ok(
                    info,
                    repaired,
                )

                marker_rx = (
                    self.buffer[
                        :consume_len
                    ].copy()
                )

                # SyncInfo MUST be delivered first.
                # app.py will set frame index
                # before the QIM-modified audio region
                # enters decrypt().
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
                        consume_len:
                    ]
                )

                continue

            if (
                status
                == "crc_fail"
            ):
                self._record_crc_fail()

                if data is not None:
                    self._diag_last_bad_header = int(
                        data
                    )

            elif (
                status
                == "bad_header"
            ):
                self._record_bad_header(
                    data
                )

            else:
                self._record_decode_fail()

            # Reject only ONE sample.
            #
            # If this was a false/offset preamble,
            # a real nearby preamble must stay searchable.
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

        return events

    def flush(
        self,
    ) -> list[
        AudioData
    ]:
        if (
            self.buffer.size == 0
        ):
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