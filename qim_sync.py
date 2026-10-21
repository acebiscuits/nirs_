"""QIM/base-4 in-band synchronization using PCM residues.

This is the residue-based synchronization transport. It does NOT add a chirp,
tone, or separate waveform. During a short marker interval it slightly moves
existing int16 PCM samples onto one of four QIM residue classes:

    y = Q * floor(x / Q) + (Q / 4) * d,   d in {0,1,2,3}

For Q=64 the four target residues are 0, 16, 32 and 48.

Protocol:
- balanced 32-digit base-4 preamble;
- header A0/A1 distinguishes PERIODIC/FULL (there is no chirp to do that);
- sequence byte for duplicate/stale diagnostics;
- uint32 absolute next_frame_index;
- CRC32;
- repetition sends complete payload copies separated in time;
- RX combines all copies by summed circular distance to the four residue levels.

No beam search/list search is used here: work is bounded and predictable.
"""

from __future__ import annotations

import binascii
from dataclasses import dataclass

import numpy as np

from audio_sync import AudioData
from sync import SyncInfo


class QIMSyncCodec:
    FULL_HEADER = 0xA1
    PERIODIC_HEADER = 0xA0

    # Balanced: 8 occurrences of each digit 0..3.
    PREAMBLE_DIGITS = np.array(
        [
            0, 3, 1, 2, 3, 0, 2, 1,
            1, 3, 2, 0, 3, 1, 0, 2,
            2, 0, 3, 1, 0, 2, 1, 3,
            3, 2, 0, 1, 2, 1, 3, 0,
        ],
        dtype=np.int8,
    )

    def __init__(
        self,
        sample_rate: int = 48000,
        base: int = 64,
        repetition: int = 4,
        detection_threshold: float = 0.75,
        quantum: int | None = None,
    ):
        if quantum is not None:
            base = quantum
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if base <= 0 or base % 4 != 0:
            raise ValueError("QIM base/quantum must be > 0 and divisible by 4")
        if repetition <= 0:
            raise ValueError("QIM repetition must be >= 1")
        if not (0.0 < detection_threshold <= 1.0):
            raise ValueError("QIM detection_threshold must be in (0, 1]")

        self.sample_rate = int(sample_rate)
        self.base = int(base)
        self.quantum = self.base
        self.step = self.base // 4
        self.repetition = int(repetition)
        self.detection_threshold = float(detection_threshold)

        self._centers = np.arange(4, dtype=np.int64) * self.step

        residues = np.arange(self.base, dtype=np.int64).reshape(-1, 1)
        direct = np.abs(residues - self._centers.reshape(1, -1))
        self._distance_lut = np.minimum(direct, self.base - direct).astype(np.int16)

        # Preamble repetition is adjacent so one logical preamble digit can be
        # decided from R neighbouring samples.
        self._preamble_symbols = np.repeat(
            self.PREAMBLE_DIGITS,
            self.repetition,
        ).astype(np.int8)

    @staticmethod
    def _crc32(data: bytes) -> int:
        return binascii.crc32(data) & 0xFFFFFFFF

    @property
    def packet_bytes(self) -> int:
        # header + sequence + uint32 frame + CRC32
        return 1 + 1 + 4 + 4

    @property
    def periodic_packet_bytes(self) -> int:
        return self.packet_bytes

    @property
    def full_packet_bytes(self) -> int:
        return self.packet_bytes

    @property
    def preamble_samples(self) -> int:
        return int(self._preamble_symbols.size)

    @property
    def payload_digit_count(self) -> int:
        return self.packet_bytes * 4

    def marker_samples_for(self, info: SyncInfo | None = None) -> int:
        return self.preamble_samples + self.payload_digit_count * self.repetition

    def _pack(self, info: SyncInfo) -> bytes:
        if not (0 <= int(info.next_frame_index) < 2**32):
            raise ValueError("next_frame_index must fit uint32")

        body = bytearray()
        body.append(self.FULL_HEADER if info.is_full else self.PERIODIC_HEADER)
        body.append(int(info.sequence) & 0xFF)
        body.extend(int(info.next_frame_index).to_bytes(4, "big", signed=False))
        crc = self._crc32(bytes(body))
        body.extend(crc.to_bytes(4, "big", signed=False))
        return bytes(body)

    def _unpack(self, packet: bytes) -> SyncInfo:
        if len(packet) != self.packet_bytes:
            raise ValueError("wrong QIM packet length")

        data = packet[:-4]
        crc_rx = int.from_bytes(packet[-4:], "big", signed=False)
        if self._crc32(data) != crc_rx:
            raise ValueError("QIM CRC32 mismatch")

        header = data[0]
        sequence = data[1]
        next_frame_index = int.from_bytes(data[2:6], "big", signed=False)

        if header == self.FULL_HEADER:
            is_full = True
        elif header == self.PERIODIC_HEADER:
            is_full = False
        else:
            raise ValueError("unknown QIM sync header")

        return SyncInfo(
            next_frame_index=next_frame_index,
            sequence=sequence,
            is_full=is_full,
        )

    @staticmethod
    def _bytes_to_base4(packet: bytes) -> np.ndarray:
        out = np.empty(len(packet) * 4, dtype=np.int8)
        pos = 0
        for byte in packet:
            out[pos] = (byte >> 6) & 0x03
            out[pos + 1] = (byte >> 4) & 0x03
            out[pos + 2] = (byte >> 2) & 0x03
            out[pos + 3] = byte & 0x03
            pos += 4
        return out

    @staticmethod
    def _base4_to_bytes(digits: np.ndarray) -> bytes:
        digits = np.asarray(digits, dtype=np.int64).reshape(-1)
        if digits.size % 4:
            raise ValueError("base-4 digit count must be divisible by 4")
        if np.any((digits < 0) | (digits > 3)):
            raise ValueError("invalid base-4 digit")

        out = bytearray()
        for pos in range(0, digits.size, 4):
            d0, d1, d2, d3 = (int(v) for v in digits[pos:pos + 4])
            out.append((d0 << 6) | (d1 << 4) | (d2 << 2) | d3)
        return bytes(out)

    def encode(self, info: SyncInfo) -> np.ndarray:
        """Return QIM symbols 0..3, not an audible waveform."""
        payload_digits = self._bytes_to_base4(self._pack(info))

        # Complete packet copies are separated in time. This is more useful
        # against short codec bursts than R adjacent copies of every payload digit.
        payload_symbols = np.tile(payload_digits, self.repetition).astype(np.int8)
        return np.concatenate((self._preamble_symbols, payload_symbols))

    def embed(self, carrier: np.ndarray, symbols: np.ndarray) -> np.ndarray:
        """Write residue symbols into existing float PCM; no signal is added."""
        carrier = np.asarray(carrier, dtype=np.float64).reshape(-1)
        symbols = np.asarray(symbols, dtype=np.int64).reshape(-1)

        if carrier.size != symbols.size:
            raise ValueError("carrier and QIM symbol lengths must match")
        if np.any((symbols < 0) | (symbols > 3)):
            raise ValueError("QIM symbols must be in range 0..3")
        if carrier.size == 0:
            return carrier.copy()

        pcm = np.rint(np.clip(carrier, -0.999, 0.999) * 32767.0).astype(np.int64)
        lattice = np.floor_divide(pcm, self.base) * self.base
        encoded = lattice + symbols * self.step

        # Keep chosen residue near int16 rails by moving one equivalent lattice step.
        safe_lo = -32734
        safe_hi = 32734
        encoded = np.where(encoded < safe_lo, encoded + self.base, encoded)
        encoded = np.where(encoded > safe_hi, encoded - self.base, encoded)
        encoded = np.clip(encoded, safe_lo, safe_hi)
        return encoded.astype(np.float64) / 32767.0

    def _sample_residues(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float64).reshape(-1)
        pcm = np.rint(
            np.clip(samples, -1.0, 0.999969482421875) * 32768.0
        ).astype(np.int64)
        return np.mod(pcm, self.base)

    def _distance_matrix(self, residues: np.ndarray) -> np.ndarray:
        residues = np.asarray(residues, dtype=np.int64).reshape(-1)
        return self._distance_lut[residues]

    def _decode_adjacent_groups(
        self,
        samples: np.ndarray,
        logical_digits: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        needed = logical_digits * self.repetition
        samples = np.asarray(samples, dtype=np.float64).reshape(-1)
        if samples.size < needed:
            raise ValueError("not enough samples for QIM digit groups")

        residues = self._sample_residues(samples[:needed])
        distances = self._distance_matrix(residues).reshape(
            logical_digits,
            self.repetition,
            4,
        )
        totals = np.sum(distances, axis=1, dtype=np.int64)
        order = np.argsort(totals, axis=1)
        digits = order[:, 0].astype(np.int8)
        best = np.take_along_axis(totals, order[:, :1], axis=1)[:, 0].astype(np.float64)
        second = np.take_along_axis(totals, order[:, 1:2], axis=1)[:, 0].astype(np.float64)
        confidence = np.clip((second - best) / np.maximum(second, 1.0), 0.0, 1.0)
        return digits, confidence

    def _decode_interleaved_payload(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        digit_count = self.payload_digit_count
        needed = digit_count * self.repetition
        samples = np.asarray(samples, dtype=np.float64).reshape(-1)
        if samples.size < needed:
            raise ValueError("not enough samples for QIM payload")

        residues = self._sample_residues(samples[:needed]).reshape(
            self.repetition,
            digit_count,
        )
        distances = self._distance_lut[residues]  # [R, digits, 4]
        totals = np.sum(distances, axis=0, dtype=np.int64)
        order = np.argsort(totals, axis=1)
        digits = order[:, 0].astype(np.int8)
        best = np.take_along_axis(totals, order[:, :1], axis=1)[:, 0].astype(np.float64)
        second = np.take_along_axis(totals, order[:, 1:2], axis=1)[:, 0].astype(np.float64)
        confidence = np.clip((second - best) / np.maximum(second, 1.0), 0.0, 1.0)
        return digits, confidence

    def preamble_score_at(self, samples: np.ndarray) -> float:
        got, confidence = self._decode_adjacent_groups(
            samples,
            int(self.PREAMBLE_DIGITS.size),
        )
        matches = got == self.PREAMBLE_DIGITS
        # Main term is exact logical-digit match. Confidence only breaks ties.
        return float(np.mean(matches) * 0.90 + np.mean(confidence) * 0.10)

    def find_preamble(self, samples: np.ndarray) -> tuple[int | None, float]:
        samples = np.asarray(samples, dtype=np.float64).reshape(-1)
        n = self.preamble_samples
        if samples.size < n:
            return None, 0.0

        best_pos: int | None = None
        best_score = 0.0
        limit = samples.size - n + 1
        for pos in range(limit):
            score = self.preamble_score_at(samples[pos:pos + n])
            if score > best_score:
                best_score = score
                best_pos = pos
            if score >= self.detection_threshold:
                return pos, score
        return None, best_score

    def decode_exact_marker(self, marker: np.ndarray) -> SyncInfo:
        marker = np.asarray(marker, dtype=np.float64).reshape(-1)
        needed = self.marker_samples_for()
        if marker.size < needed:
            raise ValueError("short QIM marker")

        pre = marker[:self.preamble_samples]
        if self.preamble_score_at(pre) < self.detection_threshold:
            raise ValueError("QIM preamble below threshold")

        payload = marker[self.preamble_samples:needed]
        digits, _confidence = self._decode_interleaved_payload(payload)
        packet = self._base4_to_bytes(digits)
        return self._unpack(packet)


@dataclass
class _Diag:
    samples: int = 0
    sum_sq: float = 0.0
    peak: float = 0.0
    best_preamble: float = 0.0
    preamble_hits: int = 0
    crc_fail: int = 0
    decode_fail: int = 0
    periodic_ok: int = 0
    full_ok: int = 0


class QIMSyncStreamDecoder:
    """Streaming residue detector with bounded hard decoding (no beam search)."""

    def __init__(self, codec: QIMSyncCodec):
        self.codec = codec
        self.buffer = np.empty(0, dtype=np.float64)
        self.diag = _Diag()
        self.total_preamble_hits = 0
        self.total_crc_fail = 0
        self.total_decode_fail = 0
        self.total_periodic_ok = 0
        self.total_full_ok = 0

    @staticmethod
    def _dbfs(value: float) -> float:
        if value <= 0.0:
            return -240.0
        return 20.0 * float(np.log10(max(value, 1e-12)))

    def clear_buffer(self) -> None:
        self.buffer = np.empty(0, dtype=np.float64)

    def diagnostics_snapshot(self, reset_interval: bool = True) -> dict:
        rms = (
            float(np.sqrt(self.diag.sum_sq / self.diag.samples))
            if self.diag.samples > 0
            else 0.0
        )
        result = {
            "samples": self.diag.samples,
            "rms_dbfs": self._dbfs(rms),
            "peak_dbfs": self._dbfs(self.diag.peak),
            "best_preamble_score": self.diag.best_preamble,
            "threshold": self.codec.detection_threshold,
            "preamble_hits": self.diag.preamble_hits,
            "crc_fail": self.diag.crc_fail,
            "decode_fail": self.diag.decode_fail,
            "periodic_ok": self.diag.periodic_ok,
            "full_ok": self.diag.full_ok,
            "buffered_samples": int(self.buffer.size),
            "total_preamble_hits": self.total_preamble_hits,
            "total_crc_fail": self.total_crc_fail,
            "total_decode_fail": self.total_decode_fail,
            "total_periodic_ok": self.total_periodic_ok,
            "total_full_ok": self.total_full_ok,
        }
        if reset_interval:
            self.diag = _Diag()
        return result

    def _record_audio(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        x = np.asarray(samples, dtype=np.float64).reshape(-1)
        self.diag.samples += int(x.size)
        self.diag.sum_sq += float(np.dot(x, x))
        self.diag.peak = max(self.diag.peak, float(np.max(np.abs(x))))

    def feed(self, samples: np.ndarray) -> list[AudioData | SyncInfo]:
        samples = np.asarray(samples, dtype=np.float64).reshape(-1)
        self._record_audio(samples)
        if samples.size:
            self.buffer = np.concatenate((self.buffer, samples))

        events: list[AudioData | SyncInfo] = []
        keep = self.codec.preamble_samples - 1
        marker_len = self.codec.marker_samples_for()

        while True:
            if self.buffer.size < self.codec.preamble_samples:
                break

            pos, score = self.codec.find_preamble(self.buffer)
            self.diag.best_preamble = max(self.diag.best_preamble, float(score))

            if pos is None:
                safe = self.buffer.size - keep
                if safe > 0:
                    events.append(AudioData(self.buffer[:safe].copy()))
                    self.buffer = self.buffer[safe:]
                break

            if pos > 0:
                events.append(AudioData(self.buffer[:pos].copy()))
                self.buffer = self.buffer[pos:]
                continue

            if self.buffer.size < marker_len:
                break

            self.diag.preamble_hits += 1
            self.total_preamble_hits += 1
            marker = self.buffer[:marker_len].copy()

            try:
                info = self.codec.decode_exact_marker(marker)
            except ValueError as exc:
                if "CRC32" in str(exc):
                    self.diag.crc_fail += 1
                    self.total_crc_fail += 1
                else:
                    self.diag.decode_fail += 1
                    self.total_decode_fail += 1

                # Shift one sample only so a nearby correctly aligned preamble
                # is still searchable.
                events.append(AudioData(self.buffer[:1].copy()))
                self.buffer = self.buffer[1:]
                continue

            if info.is_full:
                self.diag.full_ok += 1
                self.total_full_ok += 1
            else:
                self.diag.periodic_ok += 1
                self.total_periodic_ok += 1

            # Apply sync before this QIM-modified region enters decrypt().
            events.append(info)
            events.append(AudioData(marker))
            self.buffer = self.buffer[marker_len:]

        return events

    def flush(self) -> list[AudioData]:
        if self.buffer.size == 0:
            return []
        out = [AudioData(self.buffer.copy())]
        self.buffer = np.empty(0, dtype=np.float64)
        return out
