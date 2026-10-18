"""Robust in-band synchronization for lossy WebRTC/Opus audio.

Design goals:
- FULL acquisition uses a long chirp + DTMF-like 16-ary symbols + CRC32.
- Periodic tracking carries only frame_index modulo 64.
- Periodic checkpoints have no CRC requirement. They are repeated three times
  and decoded with confidence; app.py performs plausibility checks and requires
  repeated agreement before applying a correction.
- Decoder work is bounded. There is no beam search and no expensive candidate
  explosion in the real-time audio path.
- Detector is a TAP: it does not hold audio. Events carry an offset inside the
  current feed block so app.py can apply sync at the detected marker boundary.

Both peers must run the same version of this file.
"""

from __future__ import annotations

import binascii
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from sync import SyncInfo


@dataclass(frozen=True)
class CheckpointInfo:
    """Small periodic tracking observation.

    frame_mod is the sender frame index modulo ``modulus`` at the END of the
    checkpoint marker. confidence is a 0..1 physical-layer quality estimate.
    """

    frame_mod: int
    confidence: float
    modulus: int = 64


@dataclass(frozen=True)
class DetectedSyncEvent:
    """A sync event whose marker ends inside the latest feed() block."""

    end_offset: int
    payload: SyncInfo | CheckpointInfo
    preamble_score: float


@dataclass
class _Candidate:
    kind: str
    start_abs: int
    preamble_score: float


class RobustSyncCodec:
    """Audio modem optimized for Opus/WebRTC instead of sample-exact QIM."""

    FULL_MAGIC = b"RS"
    FULL_VERSION = 1
    CHECKPOINT_MAGIC = 0b10  # top two bits of one checkpoint byte

    def __init__(
        self,
        sample_rate: int = 48000,
        marker_amplitude: float = 0.30,
        full_preamble_ms: float = 60.0,
        full_symbol_ms: float = 8.0,
        checkpoint_preamble_ms: float = 16.0,
        checkpoint_symbol_ms: float = 5.0,
        full_detection_threshold: float = 0.50,
        checkpoint_detection_threshold: float = 0.48,
        checkpoint_min_confidence: float = 0.20,
        checkpoint_modulus: int = 64,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if not (0.0 < marker_amplitude <= 0.8):
            raise ValueError("marker_amplitude must be in (0, 0.8]")
        if checkpoint_modulus != 64:
            raise ValueError("current checkpoint wire format requires modulus=64")

        self.sample_rate = int(sample_rate)
        self.marker_amplitude = float(marker_amplitude)
        self.full_detection_threshold = float(full_detection_threshold)
        self.checkpoint_detection_threshold = float(checkpoint_detection_threshold)
        self.checkpoint_min_confidence = float(checkpoint_min_confidence)
        self.checkpoint_modulus = int(checkpoint_modulus)

        self.full_preamble_samples = max(
            256,
            round(self.sample_rate * float(full_preamble_ms) / 1000.0),
        )
        self.full_symbol_samples = max(
            128,
            round(self.sample_rate * float(full_symbol_ms) / 1000.0),
        )
        self.checkpoint_preamble_samples = max(
            128,
            round(self.sample_rate * float(checkpoint_preamble_ms) / 1000.0),
        )
        self.checkpoint_symbol_samples = max(
            96,
            round(self.sample_rate * float(checkpoint_symbol_ms) / 1000.0),
        )

        # DTMF-like 4x4 frequency grid. Two simultaneous tones represent one
        # nibble. These are deliberately inside the ordinary speech/Opus band.
        self.low_freqs = np.array([900.0, 1350.0, 1800.0, 2250.0])
        self.high_freqs = np.array([3000.0, 3600.0, 4200.0, 4800.0])

        self._full_preamble = self._build_chirp(
            self.full_preamble_samples,
            700.0,
            5600.0,
            amplitude=self.marker_amplitude,
        )
        # Opposite sweep direction distinguishes short checkpoints from FULL.
        self._checkpoint_preamble = self._build_chirp(
            self.checkpoint_preamble_samples,
            5200.0,
            1200.0,
            amplitude=self.marker_amplitude * 0.80,
        )

        self._full_symbols = [
            self._build_symbol(i, self.full_symbol_samples, self.marker_amplitude)
            for i in range(16)
        ]
        self._checkpoint_symbols = [
            self._build_symbol(
                i,
                self.checkpoint_symbol_samples,
                self.marker_amplitude * 0.72,
            )
            for i in range(16)
        ]

        self._full_preamble_norm = float(np.linalg.norm(self._full_preamble)) + 1e-12
        self._checkpoint_preamble_norm = (
            float(np.linalg.norm(self._checkpoint_preamble)) + 1e-12
        )
        self._full_symbol_norms = np.array(
            [float(np.linalg.norm(x)) + 1e-12 for x in self._full_symbols]
        )
        self._checkpoint_symbol_norms = np.array(
            [float(np.linalg.norm(x)) + 1e-12 for x in self._checkpoint_symbols]
        )

    @staticmethod
    def _crc32(data: bytes) -> int:
        return binascii.crc32(data) & 0xFFFFFFFF

    @staticmethod
    def _nonce_to_bytes(nonce: str) -> bytes:
        try:
            raw = bytes.fromhex(nonce)
        except ValueError as exc:
            raise ValueError("session_nonce must be hexadecimal") from exc
        if len(raw) != 16:
            raise ValueError("session_nonce must be exactly 16 bytes / 32 hex chars")
        return raw

    def _build_chirp(
        self,
        count: int,
        f0: float,
        f1: float,
        amplitude: float,
    ) -> np.ndarray:
        t = np.arange(count, dtype=np.float64) / self.sample_rate
        duration = count / self.sample_rate
        k = (f1 - f0) / max(duration, 1e-12)
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
        return amplitude * np.sin(phase) * np.hanning(count)

    def _build_symbol(
        self,
        nibble: int,
        count: int,
        amplitude: float,
    ) -> np.ndarray:
        row = (int(nibble) >> 2) & 0x03
        col = int(nibble) & 0x03
        t = np.arange(count, dtype=np.float64) / self.sample_rate
        tone = (
            np.sin(2.0 * np.pi * self.low_freqs[row] * t)
            + np.sin(2.0 * np.pi * self.high_freqs[col] * t)
        )
        tone *= amplitude * 0.5

        ramp = min(max(8, count // 10), count // 2)
        env = np.ones(count, dtype=np.float64)
        if ramp > 1:
            edge = np.sin(np.linspace(0.0, np.pi / 2.0, ramp)) ** 2
            env[:ramp] = edge
            env[-ramp:] = edge[::-1]
        return tone * env

    @property
    def full_packet_bytes(self) -> int:
        # 2-byte magic + version + uint32 end-frame + 16-byte nonce + CRC32
        return 2 + 1 + 4 + 16 + 4

    @property
    def full_raw_samples(self) -> int:
        return self.full_preamble_samples + self.full_packet_bytes * 2 * self.full_symbol_samples

    @property
    def checkpoint_raw_samples(self) -> int:
        # One byte is repeated three times; two nibble-symbols per copy.
        return self.checkpoint_preamble_samples + 3 * 2 * self.checkpoint_symbol_samples

    @staticmethod
    def _pad_to_hops(samples: np.ndarray, hop_size: int) -> np.ndarray:
        if hop_size <= 0:
            raise ValueError("hop_size must be > 0")
        count = int(samples.size)
        padded = ((count + hop_size - 1) // hop_size) * hop_size
        if padded == count:
            return samples.astype(np.float64, copy=False)
        return np.pad(samples, (0, padded - count)).astype(np.float64, copy=False)

    def full_span_frames(self, hop_size: int) -> int:
        return (self.full_raw_samples + hop_size - 1) // hop_size

    def checkpoint_span_frames(self, hop_size: int) -> int:
        return (self.checkpoint_raw_samples + hop_size - 1) // hop_size

    def full_marker_samples_for(self, hop_size: int) -> int:
        return self.full_span_frames(hop_size) * hop_size

    def checkpoint_marker_samples_for(self, hop_size: int) -> int:
        return self.checkpoint_span_frames(hop_size) * hop_size

    def _pack_full(self, info: SyncInfo) -> bytes:
        if not info.is_full or info.session_nonce is None:
            raise ValueError("FULL marker requires session_nonce")
        if not (0 <= info.next_frame_index < 2**32):
            raise ValueError("next_frame_index must fit uint32")

        body = bytearray(self.FULL_MAGIC)
        body.append(self.FULL_VERSION)
        body.extend(int(info.next_frame_index).to_bytes(4, "big"))
        body.extend(self._nonce_to_bytes(info.session_nonce))
        crc = self._crc32(bytes(body))
        body.extend(crc.to_bytes(4, "big"))
        return bytes(body)

    def _unpack_full(self, packet: bytes) -> SyncInfo:
        if len(packet) != self.full_packet_bytes:
            raise ValueError("wrong FULL packet length")
        data = packet[:-4]
        crc_rx = int.from_bytes(packet[-4:], "big")
        if self._crc32(data) != crc_rx:
            raise ValueError("FULL CRC32 mismatch")
        if data[:2] != self.FULL_MAGIC:
            raise ValueError("FULL magic mismatch")
        if data[2] != self.FULL_VERSION:
            raise ValueError("FULL version mismatch")

        next_frame = int.from_bytes(data[3:7], "big")
        nonce = data[7:23].hex()
        return SyncInfo(
            next_frame_index=next_frame,
            sequence=0,
            session_nonce=nonce,
        )

    @staticmethod
    def _bytes_to_nibbles(data: bytes) -> list[int]:
        out: list[int] = []
        for b in data:
            out.append((b >> 4) & 0x0F)
            out.append(b & 0x0F)
        return out

    @staticmethod
    def _nibbles_to_bytes(nibbles: list[int]) -> bytes:
        if len(nibbles) % 2:
            raise ValueError("nibble count must be even")
        out = bytearray()
        for i in range(0, len(nibbles), 2):
            out.append(((nibbles[i] & 0x0F) << 4) | (nibbles[i + 1] & 0x0F))
        return bytes(out)

    def encode_full(self, info: SyncInfo, hop_size: int) -> np.ndarray:
        packet = self._pack_full(info)
        parts = [self._full_preamble]
        for nibble in self._bytes_to_nibbles(packet):
            parts.append(self._full_symbols[nibble])
        raw = np.concatenate(parts)
        return self._pad_to_hops(raw, hop_size)

    def encode_checkpoint(self, frame_index: int, hop_size: int) -> np.ndarray:
        mod = int(frame_index) % self.checkpoint_modulus
        byte = (self.CHECKPOINT_MAGIC << 6) | mod
        hi = (byte >> 4) & 0x0F
        lo = byte & 0x0F
        parts = [self._checkpoint_preamble]
        # Three complete copies. The decoder votes on all three.
        for _ in range(3):
            parts.append(self._checkpoint_symbols[hi])
            parts.append(self._checkpoint_symbols[lo])
        raw = np.concatenate(parts)
        return self._pad_to_hops(raw, hop_size)

    @staticmethod
    def _corr_score(x: np.ndarray, template: np.ndarray, template_norm: float) -> float:
        if x.size != template.size:
            return 0.0
        x_norm = float(np.linalg.norm(x)) + 1e-12
        return abs(float(np.dot(x, template))) / (x_norm * template_norm)

    def preamble_score(self, kind: str, x: np.ndarray) -> float:
        if kind == "full":
            return self._corr_score(x, self._full_preamble, self._full_preamble_norm)
        if kind == "checkpoint":
            return self._corr_score(
                x,
                self._checkpoint_preamble,
                self._checkpoint_preamble_norm,
            )
        raise ValueError(f"unknown preamble kind: {kind}")

    @staticmethod
    def _decode_from_bank_ranked(
        symbol: np.ndarray,
        bank: list[np.ndarray],
        norms: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return symbol candidates ordered from most to least likely.

        Scores are normalized correlations. Keeping the runner-up candidates
        lets FULL recovery repair a *small* number of uncertain symbols while
        CRC32 remains mandatory. The search is deliberately hard-bounded.
        """
        x = np.asarray(symbol, dtype=np.float64).reshape(-1)
        x_norm = float(np.linalg.norm(x)) + 1e-12
        scores = np.empty(16, dtype=np.float64)
        for idx, template in enumerate(bank):
            scores[idx] = abs(float(np.dot(x, template))) / (x_norm * norms[idx])
        order = np.argsort(scores)[::-1]
        return order.astype(np.int8), scores

    @classmethod
    def _decode_from_bank(
        cls,
        symbol: np.ndarray,
        bank: list[np.ndarray],
        norms: np.ndarray,
    ) -> tuple[int, float, float]:
        order, scores = cls._decode_from_bank_ranked(symbol, bank, norms)
        best_idx = int(order[0])
        best = float(scores[best_idx])
        second = float(scores[int(order[1])])
        confidence = max(0.0, min(1.0, (best - second) / max(best, 1e-12)))
        return best_idx, confidence, best

    def decode_full_from(self, marker: np.ndarray) -> tuple[SyncInfo, float]:
        """Decode authoritative FULL with strict CRC32.

        The hard decision is tried first. If CRC32 fails, only the five most
        ambiguous nibble positions are considered and at most two of them may
        switch to their second/third correlation choice. This is intentionally
        tiny (<= 51 repaired packets), so it cannot recreate the old multi-second
        beam-search stalls. A repaired FULL is still accepted only if CRC32,
        magic and version are all valid.
        """
        marker = np.asarray(marker, dtype=np.float64).reshape(-1)
        needed = self.full_raw_samples
        if marker.size < needed:
            raise ValueError("short FULL marker")

        pos = self.full_preamble_samples
        nibbles: list[int] = []
        confidences: list[float] = []
        absolute_scores: list[float] = []
        ranked_choices: list[list[int]] = []

        for _ in range(self.full_packet_bytes * 2):
            symbol = marker[pos:pos + self.full_symbol_samples]
            order, scores = self._decode_from_bank_ranked(
                symbol,
                self._full_symbols,
                self._full_symbol_norms,
            )
            best = int(order[0])
            second = int(order[1])
            third = int(order[2])
            best_score = float(scores[best])
            second_score = float(scores[second])
            confidence = max(
                0.0,
                min(1.0, (best_score - second_score) / max(best_score, 1e-12)),
            )

            nibbles.append(best)
            confidences.append(confidence)
            absolute_scores.append(best_score)
            ranked_choices.append([best, second, third])
            pos += self.full_symbol_samples

        packet = self._nibbles_to_bytes(nibbles)
        quality = float(np.mean(confidences)) if confidences else 0.0
        if absolute_scores:
            quality = 0.75 * quality + 0.25 * float(np.mean(absolute_scores))

        try:
            return self._unpack_full(packet), quality
        except ValueError:
            pass

        # Bounded CRC-aided repair: only low-confidence symbols are touched.
        # We do NOT relax CRC. It remains the acceptance criterion.
        ambiguous = np.argsort(np.asarray(confidences, dtype=np.float64))[:5]
        base = list(nibbles)

        # Try one or two altered positions. Each altered nibble may use its
        # second or third correlation-ranked candidate.
        for changed_count in (1, 2):
            for positions in combinations([int(x) for x in ambiguous], changed_count):
                alternatives = [ranked_choices[p][1:3] for p in positions]
                for choices in product(*alternatives):
                    repaired = base.copy()
                    for p, value in zip(positions, choices):
                        repaired[p] = int(value)
                    candidate = self._nibbles_to_bytes(repaired)
                    try:
                        return self._unpack_full(candidate), quality
                    except ValueError:
                        continue

        raise ValueError("FULL CRC32/magic/version recovery failed")

    def decode_checkpoint_from(self, marker: np.ndarray) -> CheckpointInfo:
        marker = np.asarray(marker, dtype=np.float64).reshape(-1)
        if marker.size < self.checkpoint_raw_samples:
            raise ValueError("short checkpoint marker")

        pos = self.checkpoint_preamble_samples
        decoded: list[tuple[int, float]] = []

        for _copy in range(3):
            hi, c_hi, s_hi = self._decode_from_bank(
                marker[pos:pos + self.checkpoint_symbol_samples],
                self._checkpoint_symbols,
                self._checkpoint_symbol_norms,
            )
            pos += self.checkpoint_symbol_samples
            lo, c_lo, s_lo = self._decode_from_bank(
                marker[pos:pos + self.checkpoint_symbol_samples],
                self._checkpoint_symbols,
                self._checkpoint_symbol_norms,
            )
            pos += self.checkpoint_symbol_samples

            value = ((hi & 0x0F) << 4) | (lo & 0x0F)
            if (value >> 6) != self.CHECKPOINT_MAGIC:
                continue

            # Blend margin confidence with absolute template fit.
            conf = 0.7 * min(c_hi, c_lo) + 0.3 * min(s_hi, s_lo)
            decoded.append((value & 0x3F, float(conf)))

        if not decoded:
            raise ValueError("checkpoint magic not recovered")

        weight_by_value: dict[int, float] = {}
        count_by_value: dict[int, int] = {}
        for value, conf in decoded:
            weight_by_value[value] = weight_by_value.get(value, 0.0) + max(conf, 0.01)
            count_by_value[value] = count_by_value.get(value, 0) + 1

        best_value = max(
            weight_by_value,
            key=lambda value: (count_by_value[value], weight_by_value[value]),
        )
        agrees = count_by_value[best_value]
        total_weight = sum(weight_by_value.values()) + 1e-12
        vote_weight = weight_by_value[best_value] / total_weight
        member_conf = np.mean([conf for value, conf in decoded if value == best_value])
        confidence = float(
            np.clip(
                0.55 * (agrees / 3.0) + 0.25 * vote_weight + 0.20 * member_conf,
                0.0,
                1.0,
            )
        )

        if confidence < self.checkpoint_min_confidence:
            raise ValueError("checkpoint confidence too low")

        return CheckpointInfo(
            frame_mod=int(best_value),
            confidence=confidence,
            modulus=self.checkpoint_modulus,
        )


class RobustSyncStreamDecoder:
    """Bounded streaming detector that never performs combinatorial search."""

    def __init__(self, codec: RobustSyncCodec, hop_size: int):
        if hop_size <= 0:
            raise ValueError("hop_size must be > 0")
        self.codec = codec
        self.hop_size = int(hop_size)
        self.full_marker_samples = codec.full_marker_samples_for(hop_size)
        self.checkpoint_marker_samples = codec.checkpoint_marker_samples_for(hop_size)

        self._buffer = np.empty(0, dtype=np.float64)
        self._buffer_start_abs = 0
        self._total_abs = 0
        self._next_search_abs = 0
        self._candidate: _Candidate | None = None

        # Coarse correlation stride. At 48 kHz, 8 samples ~= 0.17 ms.
        self._search_step = 2
        self._refine_radius = 32

        self._diag_samples = 0
        self._diag_sum_sq = 0.0
        self._diag_peak = 0.0
        self._diag_best_full = 0.0
        self._diag_best_checkpoint = 0.0
        self._diag_full_candidates = 0
        self._diag_checkpoint_candidates = 0
        self._diag_full_ok = 0
        self._diag_checkpoint_ok = 0
        self._diag_full_fail = 0
        self._diag_checkpoint_fail = 0

        self.total_full_ok = 0
        self.total_checkpoint_ok = 0
        self.total_full_fail = 0
        self.total_checkpoint_fail = 0

    @staticmethod
    def _dbfs(value: float) -> float:
        if value <= 1e-12:
            return -240.0
        return 20.0 * float(np.log10(value))

    def _record_audio(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        self._diag_samples += int(x.size)
        self._diag_sum_sq += float(np.dot(x, x))
        self._diag_peak = max(self._diag_peak, float(np.max(np.abs(x))))

    def _ensure_buffer_limit(self) -> None:
        # Keep enough history for one complete FULL candidate plus two blocks.
        keep = self.full_marker_samples + 2 * self.hop_size
        if self._buffer.size <= keep:
            return

        drop = int(self._buffer.size - keep)
        # Never drop into an active candidate.
        if self._candidate is not None:
            candidate_rel = self._candidate.start_abs - self._buffer_start_abs
            drop = min(drop, max(0, int(candidate_rel)))
        if drop <= 0:
            return
        self._buffer = self._buffer[drop:]
        self._buffer_start_abs += drop

    def _slice_abs(self, start_abs: int, count: int) -> np.ndarray | None:
        rel = int(start_abs - self._buffer_start_abs)
        if rel < 0 or rel + count > self._buffer.size:
            return None
        return self._buffer[rel:rel + count]

    def _score_at(self, kind: str, start_abs: int) -> float:
        n = (
            self.codec.full_preamble_samples
            if kind == "full"
            else self.codec.checkpoint_preamble_samples
        )
        window = self._slice_abs(start_abs, n)
        if window is None:
            return 0.0
        return self.codec.preamble_score(kind, window)

    def _refine(self, kind: str, coarse_abs: int) -> tuple[int, float]:
        best_abs = int(coarse_abs)
        best_score = self._score_at(kind, best_abs)
        for delta in range(-self._refine_radius, self._refine_radius + 1):
            pos = coarse_abs + delta
            if pos < self._buffer_start_abs:
                continue
            score = self._score_at(kind, pos)
            if score > best_score:
                best_score = score
                best_abs = pos
        return best_abs, float(best_score)

    def _find_candidate(self) -> _Candidate | None:
        if self._candidate is not None:
            return self._candidate

        max_preamble = max(
            self.codec.full_preamble_samples,
            self.codec.checkpoint_preamble_samples,
        )
        latest_start = self._total_abs - max_preamble - self._refine_radius
        if latest_start < self._next_search_abs:
            return None

        start = max(self._next_search_abs, self._buffer_start_abs)
        # Align coarse scan positions to a stable absolute grid.
        if start % self._search_step:
            start += self._search_step - (start % self._search_step)

        for pos in range(start, latest_start + 1, self._search_step):
            full_score = self._score_at("full", pos)
            self._diag_best_full = max(self._diag_best_full, full_score)
            if full_score >= self.codec.full_detection_threshold:
                refined_abs, refined_score = self._refine("full", pos)
                self._diag_best_full = max(self._diag_best_full, refined_score)
                if refined_score >= self.codec.full_detection_threshold:
                    self._diag_full_candidates += 1
                    self._candidate = _Candidate(
                        kind="full",
                        start_abs=refined_abs,
                        preamble_score=refined_score,
                    )
                    return self._candidate

            cp_score = self._score_at("checkpoint", pos)
            self._diag_best_checkpoint = max(self._diag_best_checkpoint, cp_score)
            if cp_score >= self.codec.checkpoint_detection_threshold:
                refined_abs, refined_score = self._refine("checkpoint", pos)
                self._diag_best_checkpoint = max(
                    self._diag_best_checkpoint,
                    refined_score,
                )
                if refined_score >= self.codec.checkpoint_detection_threshold:
                    self._diag_checkpoint_candidates += 1
                    self._candidate = _Candidate(
                        kind="checkpoint",
                        start_abs=refined_abs,
                        preamble_score=refined_score,
                    )
                    return self._candidate

        self._next_search_abs = latest_start + 1
        return None

    def _decode_candidate(self) -> tuple[int, SyncInfo | CheckpointInfo, float] | None:
        candidate = self._candidate
        if candidate is None:
            return None

        marker_samples = (
            self.full_marker_samples
            if candidate.kind == "full"
            else self.checkpoint_marker_samples
        )
        marker = self._slice_abs(candidate.start_abs, marker_samples)
        if marker is None:
            return None

        end_abs = candidate.start_abs + marker_samples
        try:
            if candidate.kind == "full":
                payload, _quality = self.codec.decode_full_from(marker)
                self.total_full_ok += 1
                self._diag_full_ok += 1
            else:
                payload = self.codec.decode_checkpoint_from(marker)
                self.total_checkpoint_ok += 1
                self._diag_checkpoint_ok += 1

            score = candidate.preamble_score
            self._candidate = None
            self._next_search_abs = end_abs
            return end_abs, payload, score

        except ValueError:
            if candidate.kind == "full":
                self.total_full_fail += 1
                self._diag_full_fail += 1
            else:
                self.total_checkpoint_fail += 1
                self._diag_checkpoint_fail += 1

            # Resume shortly after the false candidate instead of rescanning the
            # exact same preamble forever.
            self._next_search_abs = candidate.start_abs + self._search_step
            self._candidate = None
            return (-1, CheckpointInfo(0, 0.0), candidate.preamble_score)

    def skip_samples(self, count: int) -> None:
        """Tell detector that audio samples were deliberately dropped upstream.

        The real-time bridge may discard stale queued PCM to prevent seconds of
        latency. A preamble may not be matched across such a discontinuity, so
        buffered detector state is cleared and the absolute cursor advances.
        Lifetime diagnostic counters are intentionally preserved.
        """
        count = max(0, int(count))
        if count == 0:
            return
        self._total_abs += count
        self._buffer = np.empty(0, dtype=np.float64)
        self._buffer_start_abs = self._total_abs
        self._next_search_abs = self._total_abs
        self._candidate = None

    def feed(self, samples: np.ndarray) -> list[DetectedSyncEvent]:
        x = np.asarray(samples, dtype=np.float64).reshape(-1)
        feed_start_abs = self._total_abs
        feed_end_abs = feed_start_abs + int(x.size)
        self._record_audio(x)

        if x.size:
            self._buffer = np.concatenate((self._buffer, x))
            self._total_abs = feed_end_abs

        events: list[DetectedSyncEvent] = []

        # At most a small fixed number of decode attempts per audio block.
        for _ in range(4):
            candidate = self._find_candidate()
            if candidate is None:
                break

            marker_samples = (
                self.full_marker_samples
                if candidate.kind == "full"
                else self.checkpoint_marker_samples
            )
            if self._total_abs < candidate.start_abs + marker_samples:
                break

            decoded = self._decode_candidate()
            if decoded is None:
                break
            end_abs, payload, preamble_score = decoded
            if end_abs < 0:
                continue

            # Event should normally end inside this feed. If detection completed
            # late, clamp to block start; app.py will apply it before processing
            # the current block rather than replaying old buffered audio.
            end_offset = int(np.clip(end_abs - feed_start_abs, 0, x.size))
            events.append(
                DetectedSyncEvent(
                    end_offset=end_offset,
                    payload=payload,
                    preamble_score=float(preamble_score),
                )
            )

        self._ensure_buffer_limit()
        return events

    def diagnostics_snapshot(self, reset_interval: bool = True) -> dict:
        if self._diag_samples:
            rms = float(np.sqrt(self._diag_sum_sq / self._diag_samples))
        else:
            rms = 0.0

        result = {
            "samples": self._diag_samples,
            "rms_dbfs": self._dbfs(rms),
            "peak_dbfs": self._dbfs(self._diag_peak),
            "best_full_preamble": self._diag_best_full,
            "best_checkpoint_preamble": self._diag_best_checkpoint,
            "full_candidates": self._diag_full_candidates,
            "checkpoint_candidates": self._diag_checkpoint_candidates,
            "full_ok": self._diag_full_ok,
            "checkpoint_ok": self._diag_checkpoint_ok,
            "full_fail": self._diag_full_fail,
            "checkpoint_fail": self._diag_checkpoint_fail,
            "total_full_ok": self.total_full_ok,
            "total_checkpoint_ok": self.total_checkpoint_ok,
            "total_full_fail": self.total_full_fail,
            "total_checkpoint_fail": self.total_checkpoint_fail,
            "buffered_samples": int(self._buffer.size),
        }

        if reset_interval:
            self._diag_samples = 0
            self._diag_sum_sq = 0.0
            self._diag_peak = 0.0
            self._diag_best_full = 0.0
            self._diag_best_checkpoint = 0.0
            self._diag_full_candidates = 0
            self._diag_checkpoint_candidates = 0
            self._diag_full_ok = 0
            self._diag_checkpoint_ok = 0
            self._diag_full_fail = 0
            self._diag_checkpoint_fail = 0

        return result
