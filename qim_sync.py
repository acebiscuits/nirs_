"""QIM/base-4 in-band synchronization transport.

1. Sync metadata -> bytes + CRC16
2. Byte -> 4 base-4 digits
3. Each digit -> QIM residue
4. repetition > 1 means one logical digit is encoded into several samples
"""

from __future__ import annotations

import binascii
import numpy as np

from audio_sync import AudioData
from sync import SyncInfo


class QIMSyncCodec:
    FULL_HEADER = 0xA1
    PERIODIC_HEADER = 0xA0

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
            raise ValueError(
                "QIM detection_threshold must be in (0, 1]"
            )

        self.sample_rate = int(sample_rate)
        self.base = int(base)
        self.step = self.base // 4
        self.repetition = int(repetition)
        self.detection_threshold = float(detection_threshold)

        self._centers = (
            np.arange(4, dtype=np.int64)
            * self.step
        )

        # residue -> distance to 0/1/2/3 QIM centers
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
        return binascii.crc_hqx(
            data,
            0xFFFF,
        )

    @staticmethod
    def _nonce_to_bytes(
        nonce: str,
    ) -> bytes:

        try:
            raw = bytes.fromhex(
                nonce
            )

        except ValueError as exc:
            raise ValueError(
                "session_nonce for QIM sync must be hex"
            ) from exc

        if len(raw) != 16:
            raise ValueError(
                "session_nonce must contain exactly "
                "16 bytes / 32 hex chars"
            )

        return raw

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

    def _unpack(
        self,
        packet: bytes,
    ) -> SyncInfo:

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
    def periodic_packet_bytes(
        self,
    ) -> int:
        return 1 + 1 + 4 + 2

    @property
    def full_packet_bytes(
        self,
    ) -> int:
        return (
            1
            + 1
            + 4
            + 16
            + 2
        )

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
                in digits[
                    pos:pos + 4
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
                "QIM symbols must be 0..3"
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

        needed = (
            digit_count
            * self.repetition
        )

        if samples.size < needed:
            raise ValueError(
                "not enough samples for QIM digits"
            )

        residues = (
            self._sample_residues(
                samples[:needed]
            )
        )

        out = np.empty(
            digit_count,
            dtype=np.int8,
        )

        for idx in range(
            digit_count
        ):
            group = residues[
                idx * self.repetition:
                (idx + 1)
                * self.repetition
            ]

            distances = (
                self._distance_matrix(
                    group
                )
            )

            totals = np.sum(
                distances,
                axis=0,
            )

            out[idx] = int(
                np.argmin(
                    totals
                )
            )

        return out

    def _decode_bytes(
        self,
        samples: np.ndarray,
        byte_count: int,
    ) -> bytes:

        digit_count = (
            byte_count * 4
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

        if samples.size < n:
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
    Streaming QIM detector.

    Дополнительно:
    - детальная диагностика header;
    - поиск небольшого sample shift;
    - попытка recovery только если весь packet проходит CRC.
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

        # repeat=4:
        # проверяем +/-8 samples
        # относительно границы preamble/header.
        self.phase_search_radius = max(
            8,
            2 * self.codec.repetition,
        )

        self.total_preamble_hits = 0
        self.total_bad_header = 0
        self.total_crc_fail = 0
        self.total_decode_fail = 0
        self.total_periodic_ok = 0
        self.total_full_ok = 0
        self.total_phase_recovered = 0

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
        self._diag_phase_recovered = 0

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
        phase_shift: int = 0,
    ) -> None:

        if info.is_full:
            self.total_full_ok += 1
            self._diag_full_ok += 1

        else:
            self.total_periodic_ok += 1
            self._diag_periodic_ok += 1

        if phase_shift != 0:
            self.total_phase_recovered += 1
            self._diag_phase_recovered += 1

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

    @staticmethod
    def _digits_text(
        digits: np.ndarray,
    ) -> str:

        return "".join(
            str(
                int(v)
            )
            for v
            in np.asarray(
                digits
            ).reshape(-1)
        )

    def _decode_groups_debug(
        self,
        samples: np.ndarray,
        digit_count: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:

        needed = (
            digit_count
            * self.codec.repetition
        )

        if samples.size < needed:
            raise ValueError(
                "not enough samples for QIM debug groups"
            )

        residues = (
            self.codec._sample_residues(
                samples[:needed]
            )
        )

        residues = residues.reshape(
            digit_count,
            self.codec.repetition,
        )

        digits = np.empty(
            digit_count,
            dtype=np.int8,
        )

        costs = np.empty(
            (
                digit_count,
                4,
            ),
            dtype=np.int64,
        )

        margins = np.empty(
            digit_count,
            dtype=np.int64,
        )

        for idx in range(
            digit_count
        ):
            dist = (
                self.codec._distance_matrix(
                    residues[idx]
                )
            )

            summed = np.sum(
                dist,
                axis=0,
            ).astype(
                np.int64
            )

            costs[idx] = summed

            order = np.argsort(
                summed
            )

            digits[idx] = int(
                order[0]
            )

            margins[idx] = int(
                summed[order[1]]
                - summed[order[0]]
            )

        return (
            digits,
            residues,
            costs,
            margins,
        )

    def _logical_preamble_debug(
        self,
    ) -> tuple[
        str,
        int,
    ]:

        try:
            got = (
                self.codec._decode_digit_groups(
                    self.buffer[
                        :self.codec.preamble_samples
                    ],
                    int(
                        self.codec.PREAMBLE_DIGITS.size
                    ),
                )
            )

        except ValueError:
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

        return (
            self._digits_text(
                got
            ),
            matches,
        )

    def _header_probe(
        self,
        shift: int,
    ) -> dict | None:

        start = (
            self.codec.preamble_samples
            + int(shift)
        )

        needed = (
            4
            * self.codec.repetition
        )

        if start < 0:
            return None

        if (
            self.buffer.size
            < start + needed
        ):
            return None

        payload = self.buffer[
            start:
        ]

        (
            digits,
            residues,
            costs,
            margins,
        ) = self._decode_groups_debug(
            payload,
            4,
        )

        header = int(
            self.codec._base4_to_bytes(
                digits
            )[0]
        )

        periodic_digits = (
            self._byte_digits(
                self.codec.PERIODIC_HEADER
            )
        )

        full_digits = (
            self._byte_digits(
                self.codec.FULL_HEADER
            )
        )

        periodic_hamming = int(
            np.sum(
                digits
                != periodic_digits
            )
        )

        full_hamming = int(
            np.sum(
                digits
                != full_digits
            )
        )

        hamming = min(
            periodic_hamming,
            full_hamming,
        )

        periodic_cost = int(
            sum(
                costs[
                    i,
                    int(
                        periodic_digits[i]
                    ),
                ]
                for i
                in range(4)
            )
        )

        full_cost = int(
            sum(
                costs[
                    i,
                    int(
                        full_digits[i]
                    ),
                ]
                for i
                in range(4)
            )
        )

        expected_cost = min(
            periodic_cost,
            full_cost,
        )

        near = (
            "A0"
            if periodic_cost
            <= full_cost
            else "A1"
        )

        return {
            "shift": int(
                shift
            ),
            "start": int(
                start
            ),
            "header": header,
            "digits": digits,
            "residues": residues,
            "costs": costs,
            "margins": margins,
            "hamming": hamming,
            "expected_cost": expected_cost,
            "near": near,
        }

    def _ordered_phase_shifts(
        self,
    ) -> list[int]:

        out = [
            0
        ]

        for delta in range(
            1,
            self.phase_search_radius + 1,
        ):
            out.extend(
                (
                    -delta,
                    delta,
                )
            )

        return out

    def _candidate_diagnostic_line(
        self,
        score: float,
        probes: list[dict],
    ) -> None:

        (
            preamble_text,
            preamble_matches,
        ) = self._logical_preamble_debug()

        nominal = next(
            (
                p
                for p
                in probes
                if p["shift"] == 0
            ),
            None,
        )

        best = (
            min(
                probes,
                key=lambda p: (
                    p["hamming"],
                    p["expected_cost"],
                    abs(
                        p["shift"]
                    ),
                ),
            )
            if probes
            else None
        )

        if nominal is None:
            nominal_text = "none"

        else:
            groups = [
                ",".join(
                    str(
                        int(v)
                    )
                    for v
                    in row
                )
                for row
                in nominal["residues"]
            ]

            nominal_text = (
                f"0x{nominal['header']:02X} "
                f"digits={self._digits_text(nominal['digits'])} "
                f"near={nominal['near']} "
                f"hd={nominal['hamming']} "
                f"margin="
                f"{','.join(str(int(v)) for v in nominal['margins'])} "
                f"res="
                f"{'|'.join(groups)}"
            )

        if best is None:
            best_text = "none"

        else:
            best_text = (
                f"shift={best['shift']:+d} "
                f"hdr=0x{best['header']:02X} "
                f"digits="
                f"{self._digits_text(best['digits'])} "
                f"near={best['near']} "
                f"hd={best['hamming']} "
                f"cost={best['expected_cost']}"
            )

        exact_headers = [
            (
                f"{p['shift']:+d}:"
                f"0x{p['header']:02X}"
            )
            for p
            in probes
            if p["header"]
            in (
                self.codec.PERIODIC_HEADER,
                self.codec.FULL_HEADER,
            )
        ]

        exact_text = (
            ",".join(
                exact_headers
            )
            if exact_headers
            else "none"
        )

        print(
            "QIM_PHY_CANDIDATE "
            f"score={score:.3f} "
            f"pre={preamble_matches}/16 "
            f"got={preamble_text} "
            f"nominal=[{nominal_text}] "
            f"best=[{best_text}] "
            f"exact_headers={exact_text}",
            flush=True,
        )

    def _try_candidate_with_phase(
        self,
        score: float,
    ) -> tuple[
        str,
        object | None,
    ]:

        probes = []

        any_valid_header = False
        any_crc_fail = False
        any_decode_fail = False

        need_more_for_valid_header = False

        for shift in (
            self._ordered_phase_shifts()
        ):
            probe = (
                self._header_probe(
                    shift
                )
            )

            if probe is None:
                continue

            probes.append(
                probe
            )

            header = probe[
                "header"
            ]

            if header not in (
                self.codec.PERIODIC_HEADER,
                self.codec.FULL_HEADER,
            ):
                continue

            any_valid_header = True

            packet_bytes = (
                self.codec.full_packet_bytes
                if header
                == self.codec.FULL_HEADER
                else self.codec.periodic_packet_bytes
            )

            payload_start = probe[
                "start"
            ]

            needed = (
                packet_bytes
                * 4
                * self.codec.repetition
            )

            consume_len = (
                payload_start
                + needed
            )

            if (
                self.buffer.size
                < consume_len
            ):
                need_more_for_valid_header = True
                continue

            packet = (
                self.codec._decode_bytes(
                    self.buffer[
                        payload_start:
                        payload_start + needed
                    ],
                    packet_bytes,
                )
            )

            try:
                info = (
                    self.codec._unpack(
                        packet
                    )
                )

            except ValueError as exc:
                if (
                    "CRC mismatch"
                    in str(exc)
                ):
                    any_crc_fail = True

                    rx_crc = (
                        int.from_bytes(
                            packet[-2:],
                            "big",
                        )
                    )

                    calc_crc = (
                        self.codec._crc16(
                            packet[:-2]
                        )
                    )

                    print(
                        "QIM_PHY_CRC_FAIL "
                        f"score={score:.3f} "
                        f"shift={shift:+d} "
                        f"header=0x{header:02X} "
                        f"packet={packet.hex()} "
                        f"crc_rx=0x{rx_crc:04X} "
                        f"crc_calc=0x{calc_crc:04X}",
                        flush=True,
                    )

                else:
                    any_decode_fail = True

                continue

            self._candidate_diagnostic_line(
                score,
                probes,
            )

            if shift != 0:
                print(
                    "QIM_PHY_PHASE_RECOVER "
                    f"score={score:.3f} "
                    f"shift={shift:+d} "
                    f"header=0x{header:02X} "
                    f"seq={info.sequence} "
                    f"next={info.next_frame_index} "
                    f"full={info.is_full}",
                    flush=True,
                )

            return (
                "ok",
                (
                    info,
                    consume_len,
                    shift,
                ),
            )

        if need_more_for_valid_header:
            return (
                "need_more",
                None,
            )

        self._candidate_diagnostic_line(
            score,
            probes,
        )

        if any_crc_fail:
            return (
                "crc_fail",
                None,
            )

        if (
            any_valid_header
            or any_decode_fail
        ):
            return (
                "decode_fail",
                None,
            )

        nominal = next(
            (
                p
                for p
                in probes
                if p["shift"] == 0
            ),
            None,
        )

        nominal_header = (
            None
            if nominal is None
            else int(
                nominal["header"]
            )
        )

        return (
            "bad_header",
            nominal_header,
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

            "phase_recovered":
                self._diag_phase_recovered,

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

            "total_phase_recovered":
                self.total_phase_recovered,
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
            self._diag_phase_recovered = 0

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

            self._record_preamble_hit()

            status, data = (
                self._try_candidate_with_phase(
                    score
                )
            )

            if status == "need_more":
                self.total_preamble_hits -= 1
                self._diag_preamble_hits -= 1

                break

            if status == "ok":
                (
                    info,
                    consume_len,
                    shift,
                ) = data

                self._record_ok(
                    info,
                    phase_shift=shift,
                )

                marker_rx = (
                    self.buffer[
                        :consume_len
                    ].copy()
                )

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

            if status == "crc_fail":
                self._record_crc_fail()

            elif status == "decode_fail":
                self._record_decode_fail()

            else:
                self._record_bad_header(
                    data
                )

            # Reject only one sample:
            # nearby real preamble must remain searchable.
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