#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import os
import secrets
import socket
import sys
import threading
import time

import numpy as np

from VirtualDevices import (
    AudioBridge,
    PulseVirtualDevices,
    RATE,
    get_default_audio_devices,
    stop_event,
)
from crypto.mdct import MDCTCipher
from crypto.stft import STFTCipher
from robust_sync import (
    CheckpointInfo,
    DetectedSyncEvent,
    RobustSyncCodec,
    RobustSyncStreamDecoder,
)
from sync import FrameSynchronizer, SyncInfo


@dataclass
class _PendingMarker:
    kind: str  # FULL | CHECKPOINT
    payload: SyncInfo | int


class LiveProcessor:
    """Real-time cipher + robust in-band sync.

    Architecture:
      * one strong CRC32 FULL establishes nonce + absolute frame index;
      * after lock both sides count frames locally;
      * short periodic checkpoints carry only frame_index mod 64;
      * checkpoints are confidence/plausibility checked and a non-zero
        correction must repeat before it is applied;
      * sync detection has bounded work and never runs a beam search.
    """

    def __init__(
        self,
        tx_cipher,
        rx_cipher,
        sample_rate: int,
        sync_period_seconds: float,
        full_sync_period_seconds: float,
        marker_amplitude: float,
        ciphertext_gain: float,
        node_name: str,
        sync_diag_interval: float,
        tx_diag_interval: float,
        initial_full_repeats: int,
        checkpoint_confirmations: int,
        checkpoint_max_drift: int,
        checkpoint_min_confidence: float,
    ):
        if not (0.0 < ciphertext_gain <= 1.0):
            raise ValueError("ciphertext_gain must be in (0, 1]")
        if initial_full_repeats < 1:
            raise ValueError("initial_full_repeats must be >= 1")
        if checkpoint_confirmations < 1:
            raise ValueError("checkpoint_confirmations must be >= 1")
        if checkpoint_max_drift < 1:
            raise ValueError("checkpoint_max_drift must be >= 1")
        if not (0.0 <= checkpoint_min_confidence <= 1.0):
            raise ValueError("checkpoint_min_confidence must be in [0, 1]")
        if tx_cipher.hop_size != rx_cipher.hop_size:
            raise ValueError("TX and RX hop sizes must match")

        self.tx_cipher = tx_cipher
        self.rx_cipher = rx_cipher
        self.sample_rate = int(sample_rate)
        self.node_name = node_name
        self.ciphertext_gain = float(ciphertext_gain)
        self.hop_size = int(tx_cipher.hop_size)

        self.tx_sync = FrameSynchronizer(
            sample_rate=sample_rate,
            sync_period_seconds=sync_period_seconds,
            full_sync_period_seconds=full_sync_period_seconds,
        )
        self.rx_sync = FrameSynchronizer(
            sample_rate=sample_rate,
            sync_period_seconds=sync_period_seconds,
            full_sync_period_seconds=full_sync_period_seconds,
        )

        self.sync_codec = RobustSyncCodec(
            sample_rate=sample_rate,
            marker_amplitude=marker_amplitude,
        )
        self.sync_decoder = RobustSyncStreamDecoder(
            self.sync_codec,
            hop_size=self.hop_size,
        )

        self.periodic_interval_frames = self.tx_sync.periodic_interval_frames(tx_cipher)
        self.full_interval_frames = self.tx_sync.full_interval_frames(tx_cipher)

        self.initial_full_repeats = int(initial_full_repeats)
        self.initial_full_remaining = self.initial_full_repeats - 1
        self.checkpoint_confirmations = int(checkpoint_confirmations)
        self.checkpoint_max_drift = int(checkpoint_max_drift)
        self.checkpoint_min_confidence = float(checkpoint_min_confidence)

        self.rx_locked = False
        self._pending_checkpoint_drift: int | None = None
        self._pending_checkpoint_count = 0

        self.tx_warmup_done = False
        self.pending_marker: _PendingMarker | None = None
        self.active_marker = np.empty(0, dtype=np.float64)
        self.active_marker_pos = 0
        self.active_marker_kind: str | None = None

        self._last_checkpoint_scheduled_frame = 0
        self._last_full_scheduled_frame = 0

        if sync_diag_interval <= 0:
            raise ValueError("sync_diag_interval must be > 0")
        self.sync_diag_interval = float(sync_diag_interval)
        self._next_sync_diag_at = time.monotonic() + self.sync_diag_interval

        if tx_diag_interval <= 0:
            raise ValueError("tx_diag_interval must be > 0")
        self.tx_diag_interval = float(tx_diag_interval)
        self._next_tx_diag_at = time.monotonic() + self.tx_diag_interval
        self._tx_raw_sum_sq = 0.0
        self._tx_raw_peak = 0.0
        self._tx_raw_samples = 0
        self._tx_tunnel_sum_sq = 0.0
        self._tx_tunnel_peak = 0.0
        self._tx_tunnel_samples = 0

        self._rx_process_max_ms = 0.0
        self._rx_process_calls = 0

    def _log(self, direction: str, event: str, **fields) -> None:
        ts = datetime.now().astimezone().strftime("%H:%M:%S.%f")[:-3]
        tail = " ".join(f"{key}={value}" for key, value in fields.items())
        print(
            f"{ts} [{self.node_name}] {direction} {event}"
            + (f" {tail}" if tail else ""),
            flush=True,
        )

    @staticmethod
    def _bytes_to_float(data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.int16).astype(np.float64) / 32768.0

    @staticmethod
    def _float_to_bytes(samples: np.ndarray) -> bytes:
        samples = np.asarray(samples, dtype=np.float64).reshape(-1)
        if samples.size == 0:
            return b""
        samples = np.clip(samples, -0.999, 0.999)
        return np.rint(samples * 32767.0).astype(np.int16).tobytes()

    def _queue_full(self, start_frame: int) -> None:
        span = self.sync_codec.full_span_frames(self.hop_size)
        target_end_frame = int(start_frame) + span
        info = self.tx_sync.make_full_sync(
            self.tx_cipher,
            next_frame_index=target_end_frame,
        )
        self.pending_marker = _PendingMarker("FULL", info)
        self._last_full_scheduled_frame = int(start_frame)
        self._last_checkpoint_scheduled_frame = int(start_frame)

    def _queue_checkpoint(self, start_frame: int) -> None:
        span = self.sync_codec.checkpoint_span_frames(self.hop_size)
        target_end_frame = int(start_frame) + span
        self.pending_marker = _PendingMarker("CHECKPOINT", target_end_frame)
        self._last_checkpoint_scheduled_frame = int(start_frame)

    def _start_pending_marker_at_boundary(self) -> None:
        if self.pending_marker is None:
            return
        if self.active_marker_pos < self.active_marker.size:
            return

        pending = self.pending_marker
        self.pending_marker = None
        self.active_marker_kind = pending.kind

        if pending.kind == "FULL":
            assert isinstance(pending.payload, SyncInfo)
            self.active_marker = self.sync_codec.encode_full(
                pending.payload,
                hop_size=self.hop_size,
            )
            self._log(
                "TX->REMOTE",
                "FULL EMIT",
                next=pending.payload.next_frame_index,
                local_tx=self.tx_cipher.frame_index,
                marker_ms=f"{1000.0 * self.active_marker.size / self.sample_rate:.1f}",
                crc="CRC32",
            )
        else:
            target = int(pending.payload)
            self.active_marker = self.sync_codec.encode_checkpoint(
                target,
                hop_size=self.hop_size,
            )
            self._log(
                "TX->REMOTE",
                "CHECKPOINT EMIT",
                target_mod=target % 64,
                target=target,
                local_tx=self.tx_cipher.frame_index,
                marker_ms=f"{1000.0 * self.active_marker.size / self.sample_rate:.1f}",
            )

        self.active_marker_pos = 0

    def _apply_active_marker(self, encrypted: np.ndarray) -> np.ndarray:
        if encrypted.size == 0:
            return encrypted
        if self.active_marker_pos >= self.active_marker.size:
            return encrypted

        out = encrypted.copy()
        pos = 0

        while pos < out.size and self.active_marker_pos < self.active_marker.size:
            count = min(
                out.size - pos,
                self.active_marker.size - self.active_marker_pos,
            )
            marker_piece = self.active_marker[
                self.active_marker_pos:self.active_marker_pos + count
            ]

            # Overlay rather than insert samples. This cannot accumulate stream
            # latency. FULL is intentionally stronger, checkpoint is lighter.
            if self.active_marker_kind == "FULL":
                out[pos:pos + count] = (
                    0.58 * out[pos:pos + count]
                    + marker_piece
                )
            else:
                out[pos:pos + count] = (
                    0.82 * out[pos:pos + count]
                    + marker_piece
                )

            pos += count
            self.active_marker_pos += count

        np.clip(out, -0.98, 0.98, out=out)
        return out

    def _schedule_next_marker(self) -> None:
        if self.pending_marker is not None:
            return
        if self.active_marker_pos < self.active_marker.size:
            return

        next_index = int(self.tx_cipher.frame_index)

        # Acquisition first: send the configured number of strong FULL markers
        # consecutively. They are expensive but only happen at startup.
        if self.initial_full_remaining > 0:
            self._queue_full(next_index)
            self.initial_full_remaining -= 1
            return

        # Rare absolute re-anchor. CRC32 makes this authoritative.
        if next_index - self._last_full_scheduled_frame >= self.full_interval_frames:
            self._queue_full(next_index)
            return

        # Normal tracking is tiny modulo-64 metadata, not an absolute packet.
        if next_index - self._last_checkpoint_scheduled_frame >= self.periodic_interval_frames:
            self._queue_checkpoint(next_index)

    @staticmethod
    def _dbfs_from_power(sum_sq: float, count: int) -> float:
        if count <= 0 or sum_sq <= 0.0:
            return -240.0
        rms = float(np.sqrt(sum_sq / count))
        return 20.0 * float(np.log10(max(rms, 1e-12)))

    @staticmethod
    def _dbfs_from_peak(peak: float) -> float:
        if peak <= 0.0:
            return -240.0
        return 20.0 * float(np.log10(max(float(peak), 1e-12)))

    def _accumulate_tx_raw(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        x = np.asarray(samples, dtype=np.float64).reshape(-1)
        self._tx_raw_sum_sq += float(np.dot(x, x))
        self._tx_raw_peak = max(self._tx_raw_peak, float(np.max(np.abs(x))))
        self._tx_raw_samples += int(x.size)

    def _accumulate_tx_tunnel(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        x = np.asarray(samples, dtype=np.float64).reshape(-1)
        self._tx_tunnel_sum_sq += float(np.dot(x, x))
        self._tx_tunnel_peak = max(self._tx_tunnel_peak, float(np.max(np.abs(x))))
        self._tx_tunnel_samples += int(x.size)

    def _maybe_log_tx_diagnostics(self) -> None:
        now = time.monotonic()
        if now < self._next_tx_diag_at:
            return
        self._next_tx_diag_at = now + self.tx_diag_interval

        self._log(
            "TX->REMOTE",
            "AUDIO_DIAG",
            raw_samples=self._tx_raw_samples,
            raw_rms_dbfs=f"{self._dbfs_from_power(self._tx_raw_sum_sq, self._tx_raw_samples):.1f}",
            raw_peak_dbfs=f"{self._dbfs_from_peak(self._tx_raw_peak):.1f}",
            tunnel_samples=self._tx_tunnel_samples,
            tunnel_rms_dbfs=f"{self._dbfs_from_power(self._tx_tunnel_sum_sq, self._tx_tunnel_samples):.1f}",
            tunnel_peak_dbfs=f"{self._dbfs_from_peak(self._tx_tunnel_peak):.1f}",
        )

        self._tx_raw_sum_sq = 0.0
        self._tx_raw_peak = 0.0
        self._tx_raw_samples = 0
        self._tx_tunnel_sum_sq = 0.0
        self._tx_tunnel_peak = 0.0
        self._tx_tunnel_samples = 0

    def on_tx_gap(self, dropped_blocks: int, block_samples: int) -> None:
        """Advance TX crypto over real-time underflow without replaying old audio.

        Pulse will effectively transmit silence while Python is late. For the
        current linear spectral transforms, encrypting zero PCM also produces
        zero PCM, so we advance the cipher with zeros and discard the result.
        Any in-progress marker is invalid after a gap; queue a fresh FULL anchor.
        """
        dropped_blocks = max(0, int(dropped_blocks))
        block_samples = max(1, int(block_samples))
        if dropped_blocks == 0:
            return

        before = int(self.tx_cipher.frame_index)
        zero = np.zeros(block_samples, dtype=np.float64)
        for _ in range(dropped_blocks):
            self.tx_cipher.encrypt(zero)
        after = int(self.tx_cipher.frame_index)

        self.pending_marker = None
        self.active_marker = np.empty(0, dtype=np.float64)
        self.active_marker_pos = 0
        self.active_marker_kind = None
        self._queue_full(after)

        self._log(
            "TX->REMOTE",
            "REALTIME GAP",
            dropped_blocks=dropped_blocks,
            frame_before=before,
            frame_after=after,
            action="ZERO_ADVANCE_AND_FULL",
        )

    def on_rx_gap(self, dropped_blocks: int, block_samples: int) -> None:
        """Advance RX over dropped stale PCM instead of building latency.

        Missing ciphertext is represented by zero PCM solely to advance streaming
        state/frame counters. Output for the gap is discarded. Detector history is
        reset so no preamble can straddle the discontinuity.
        """
        dropped_blocks = max(0, int(dropped_blocks))
        block_samples = max(1, int(block_samples))
        if dropped_blocks == 0:
            return

        skipped_samples = dropped_blocks * block_samples
        self.sync_decoder.skip_samples(skipped_samples)

        before = int(self.rx_cipher.frame_index)
        if self.rx_locked:
            zero = np.zeros(block_samples, dtype=np.float64)
            for _ in range(dropped_blocks):
                self.rx_cipher.decrypt(zero)
        after = int(self.rx_cipher.frame_index)

        self._pending_checkpoint_drift = None
        self._pending_checkpoint_count = 0
        self._log(
            "RX<-REMOTE",
            "REALTIME GAP",
            dropped_blocks=dropped_blocks,
            frame_before=before,
            frame_after=after,
            locked=self.rx_locked,
            action=("ZERO_ADVANCE" if self.rx_locked else "WAIT_FULL"),
        )

    def process_tx(self, data: bytes) -> bytes:
        pcm = self._bytes_to_float(data)
        self._accumulate_tx_raw(pcm)

        before = int(self.tx_cipher.frame_index)
        encrypted = self.tx_cipher.encrypt(pcm)
        after = int(self.tx_cipher.frame_index)

        if encrypted.size == 0:
            self._maybe_log_tx_diagnostics()
            return b""

        if not self.tx_warmup_done:
            self.tx_warmup_done = True
            # Current implementation's first returned transform hop is warmup.
            # The next returned hop begins at frame `after`.
            self._queue_full(after)
            return b""

        if after - before > 1:
            raise RuntimeError(
                "Audio block produced more than one spectral frame; "
                "BLOCK must not exceed cipher hop size"
            )

        encrypted = encrypted * self.ciphertext_gain
        self._start_pending_marker_at_boundary()
        encrypted = self._apply_active_marker(encrypted)

        self._accumulate_tx_tunnel(encrypted)
        self._maybe_log_tx_diagnostics()

        if after > before:
            self._schedule_next_marker()

        return self._float_to_bytes(encrypted)

    @staticmethod
    def _nearest_modulo_frame(local_frame: int, frame_mod: int, modulus: int) -> int:
        """Resolve modulo value to the absolute frame nearest local prediction."""
        local_frame = int(local_frame)
        frame_mod = int(frame_mod) % int(modulus)
        modulus = int(modulus)

        k = round((local_frame - frame_mod) / modulus)
        candidates = [
            frame_mod + (k - 1) * modulus,
            frame_mod + k * modulus,
            frame_mod + (k + 1) * modulus,
        ]
        candidates = [x for x in candidates if x >= 0]
        return min(candidates, key=lambda x: abs(x - local_frame))

    def _handle_full(self, info: SyncInfo, preamble_score: float) -> None:
        local_before = int(self.rx_cipher.frame_index)
        is_new_session = (
            self.rx_locked
            and info.session_nonce != self.rx_cipher.session_nonce
        )

        if not self.rx_locked or is_new_session:
            self.rx_sync.apply_sync(self.rx_cipher, info, hard=True)
            self.rx_locked = True
            action = "HARD_LOCK" if not is_new_session else "HARD_NEW_SESSION"
        else:
            # Same nonce + CRC32-valid absolute checkpoint: preserve overlap
            # state and only re-anchor the frame counter at marker END.
            self.rx_sync.apply_sync(self.rx_cipher, info, hard=False)
            action = "ABS_REANCHOR"

        self._pending_checkpoint_drift = None
        self._pending_checkpoint_count = 0

        self._log(
            "RX<-REMOTE",
            "FULL APPLY",
            next=info.next_frame_index,
            local_before=local_before,
            drift=info.next_frame_index - local_before,
            local_after=self.rx_cipher.frame_index,
            action=action,
            preamble=f"{preamble_score:.3f}",
            crc="CRC32_OK",
        )

    def _handle_checkpoint(self, cp: CheckpointInfo, preamble_score: float) -> None:
        if not self.rx_locked:
            self._log(
                "RX<-REMOTE",
                "CHECKPOINT IGNORE",
                reason="unlocked",
                mod=cp.frame_mod,
                confidence=f"{cp.confidence:.3f}",
            )
            return

        local = int(self.rx_cipher.frame_index)
        resolved = self._nearest_modulo_frame(local, cp.frame_mod, cp.modulus)
        drift = int(resolved - local)

        if cp.confidence < self.checkpoint_min_confidence:
            self._pending_checkpoint_drift = None
            self._pending_checkpoint_count = 0
            self._log(
                "RX<-REMOTE",
                "CHECKPOINT REJECT",
                reason="low_confidence",
                mod=cp.frame_mod,
                confidence=f"{cp.confidence:.3f}",
                local=local,
            )
            return

        if abs(drift) > self.checkpoint_max_drift:
            self._pending_checkpoint_drift = None
            self._pending_checkpoint_count = 0
            self._log(
                "RX<-REMOTE",
                "CHECKPOINT REJECT",
                reason="implausible_drift",
                mod=cp.frame_mod,
                confidence=f"{cp.confidence:.3f}",
                local=local,
                resolved=resolved,
                drift=drift,
                max_drift=self.checkpoint_max_drift,
            )
            return

        if drift == 0:
            self._pending_checkpoint_drift = None
            self._pending_checkpoint_count = 0
            self._log(
                "RX<-REMOTE",
                "CHECKPOINT TRACK",
                mod=cp.frame_mod,
                confidence=f"{cp.confidence:.3f}",
                local=local,
                drift=0,
                preamble=f"{preamble_score:.3f}",
            )
            return

        if self._pending_checkpoint_drift == drift:
            self._pending_checkpoint_count += 1
        else:
            self._pending_checkpoint_drift = drift
            self._pending_checkpoint_count = 1

        if self._pending_checkpoint_count < self.checkpoint_confirmations:
            self._log(
                "RX<-REMOTE",
                "CHECKPOINT SUSPECT",
                mod=cp.frame_mod,
                confidence=f"{cp.confidence:.3f}",
                local=local,
                resolved=resolved,
                drift=drift,
                confirmations=f"{self._pending_checkpoint_count}/{self.checkpoint_confirmations}",
            )
            return

        self.rx_cipher.set_frame_index(resolved)
        self._log(
            "RX<-REMOTE",
            "CHECKPOINT APPLY",
            mod=cp.frame_mod,
            confidence=f"{cp.confidence:.3f}",
            local_before=local,
            resolved=resolved,
            drift=drift,
            confirmations=self._pending_checkpoint_count,
            preamble=f"{preamble_score:.3f}",
        )
        self._pending_checkpoint_drift = None
        self._pending_checkpoint_count = 0

    def _handle_detected_event(self, event: DetectedSyncEvent) -> None:
        if isinstance(event.payload, SyncInfo):
            self._handle_full(event.payload, event.preamble_score)
        else:
            self._handle_checkpoint(event.payload, event.preamble_score)

    def _decrypt_segment(self, segment: np.ndarray, output_parts: list[np.ndarray]) -> None:
        if segment.size == 0:
            return

    # Пока FULL не пойман — не глушим RX.
    # Просто пропускаем входящий звук напрямую.
        if not self.rx_locked:
            output_parts.append(segment.copy())
            return

        ciphertext = segment / self.ciphertext_gain
        restored = self.rx_cipher.decrypt(ciphertext)

        if restored.size:
            output_parts.append(restored)

    def _maybe_log_sync_diagnostics(self) -> None:
        now = time.monotonic()
        if now < self._next_sync_diag_at:
            return
        self._next_sync_diag_at = now + self.sync_diag_interval

        stats = self.sync_decoder.diagnostics_snapshot(reset_interval=True)
        self._log(
            "RX<-REMOTE",
            "SYNC_DIAG",
            samples=stats["samples"],
            rms_dbfs=f'{stats["rms_dbfs"]:.1f}',
            peak_dbfs=f'{stats["peak_dbfs"]:.1f}',
            full_pre=f'{stats["best_full_preamble"]:.3f}',
            cp_pre=f'{stats["best_checkpoint_preamble"]:.3f}',
            full_candidates=stats["full_candidates"],
            cp_candidates=stats["checkpoint_candidates"],
            full_ok=stats["full_ok"],
            cp_ok=stats["checkpoint_ok"],
            full_fail=stats["full_fail"],
            cp_fail=stats["checkpoint_fail"],
            total_full_ok=stats["total_full_ok"],
            total_cp_ok=stats["total_checkpoint_ok"],
            buffered=stats["buffered_samples"],
            locked=self.rx_locked,
            frame=self.rx_cipher.frame_index,
            pending_drift=(
                "none"
                if self._pending_checkpoint_drift is None
                else self._pending_checkpoint_drift
            ),
            rx_proc_max_ms=f"{self._rx_process_max_ms:.2f}",
        )
        self._rx_process_max_ms = 0.0
        self._rx_process_calls = 0

    def process_rx(self, data: bytes) -> bytes:
        started = time.perf_counter()
        incoming = self._bytes_to_float(data)
        events = self.sync_decoder.feed(incoming)
        events.sort(key=lambda event: event.end_offset)

        output_parts: list[np.ndarray] = []
        cursor = 0

        # Process audio up to each marker END with the old state, then apply the
        # event at that exact stream position. For initial FULL, pre-lock audio is
        # simply discarded; data after the FULL boundary uses the new state.
        for event in events:
            end = max(cursor, min(int(event.end_offset), int(incoming.size)))
            self._decrypt_segment(incoming[cursor:end], output_parts)
            self._handle_detected_event(event)
            cursor = end

        self._decrypt_segment(incoming[cursor:], output_parts)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._rx_process_calls += 1
        self._rx_process_max_ms = max(self._rx_process_max_ms, elapsed_ms)
        self._maybe_log_sync_diagnostics()

        if not output_parts:
            return b""
        return self._float_to_bytes(np.concatenate(output_parts))


def build_cipher(
    transform: str,
    operation: str,
    key: bytes,
    session_nonce: str,
    key_schedule: str,
):
    if transform == "stft":
        return STFTCipher(
            operation=operation,
            key=key,
            session_nonce=session_nonce,
            key_schedule=key_schedule,
        )
    if transform == "mdct":
        return MDCTCipher(
            operation=operation,
            key=key,
            session_nonce=session_nonce,
            key_schedule=key_schedule,
        )
    raise ValueError(f"unknown transform: {transform}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transform", choices=["stft", "mdct"], default="stft")
    parser.add_argument(
        "--operation",
        choices=["invert", "shuffle", "signflip"],
        default="signflip",
    )
    parser.add_argument(
        "--key-schedule",
        choices=["fixed", "per_frame"],
        default="per_frame",
    )
    parser.add_argument("--key", default="key.bin")
    parser.add_argument("--sync-period", type=float, default=0.5)
    parser.add_argument(
        "--full-sync-period",
        type=float,
        default=30.0,
        help="rare CRC32 absolute re-anchor interval",
    )
    parser.add_argument("--marker-amplitude", type=float, default=0.30)
    parser.add_argument("--ciphertext-gain", type=float, default=0.68)
    parser.add_argument(
        "--initial-full-repeats",
        type=int,
        default=2,
        help="strong FULL markers sent at startup",
    )
    parser.add_argument(
        "--checkpoint-confirmations",
        type=int,
        default=2,
        help="same non-zero modulo correction must repeat before apply",
    )
    parser.add_argument(
        "--checkpoint-max-drift",
        type=int,
        default=4,
        help="reject modulo checkpoints farther than this many frames from prediction",
    )
    parser.add_argument(
        "--checkpoint-min-confidence",
        type=float,
        default=0.30,
        help="reject weak periodic checkpoint decodes",
    )
    parser.add_argument(
        "--sync-diag-interval",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--tx-diag-interval",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--node",
        default=socket.gethostname(),
        help="short node label, e.g. PC-A or PC-B",
    )
    parser.add_argument("--log-dir", default="logs/robust_sync")
    return parser.parse_args()


class _TeeStream:
    def __init__(self, terminal, log_file, lock: threading.Lock):
        self.terminal = terminal
        self.log_file = log_file
        self.lock = lock

    def write(self, text: str) -> int:
        with self.lock:
            self.terminal.write(text)
            self.log_file.write(text)
            self.log_file.flush()
        return len(text)

    def flush(self) -> None:
        with self.lock:
            self.terminal.flush()
            self.log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())


class RunLog:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        self.file = None
        self.path: Path | None = None
        self.lock = threading.Lock()

    @staticmethod
    def _safe(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)

    def __enter__(self):
        log_dir = Path(self.args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{stamp}_{self._safe(self.args.node)}_"
            f"{self.args.transform}_{self.args.operation}_robust.log"
        )
        self.path = log_dir / filename
        self.file = self.path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = _TeeStream(self.old_stdout, self.file, self.lock)
        sys.stderr = _TeeStream(self.old_stderr, self.file, self.lock)

        print("=" * 80)
        print("RUN START          :", datetime.now().astimezone().isoformat(timespec="milliseconds"))
        print("LOG FILE           :", self.path.resolve())
        print("COMMAND            :", " ".join(sys.argv))
        print("PYTHON             :", sys.version.replace("\n", " "))
        print("PID                :", os.getpid())
        print("CWD                :", Path.cwd())
        print("PARAMETERS:")
        for key, value in sorted(vars(self.args).items()):
            print(f"  {key:28s} = {value}")
        print("=" * 80)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            print("=" * 80)
            print("RUN END            :", datetime.now().astimezone().isoformat(timespec="milliseconds"))
            print("RESULT             :", "ERROR" if exc_type else "OK")
            print("=" * 80)
        finally:
            sys.stdout = self.old_stdout
            sys.stderr = self.old_stderr
            if self.file is not None:
                self.file.close()
        if self.path is not None:
            self.old_stdout.write(f"Run log: {self.path.resolve()}\n")
            self.old_stdout.flush()
        return False


def run(args: argparse.Namespace) -> None:
    key_path = Path(args.key)
    key = key_path.read_bytes()
    if not key:
        raise RuntimeError(f"empty key file: {key_path}")

    print("Key path          :", key_path.resolve())
    print("Key SHA256 prefix :", hashlib.sha256(key).hexdigest()[:16])

    tx_nonce = secrets.token_hex(16)
    rx_placeholder_nonce = "00" * 16

    tx_cipher = build_cipher(
        args.transform,
        args.operation,
        key,
        tx_nonce,
        args.key_schedule,
    )
    rx_cipher = build_cipher(
        args.transform,
        args.operation,
        key,
        rx_placeholder_nonce,
        args.key_schedule,
    )

    processor = LiveProcessor(
        tx_cipher=tx_cipher,
        rx_cipher=rx_cipher,
        sample_rate=RATE,
        sync_period_seconds=args.sync_period,
        full_sync_period_seconds=args.full_sync_period,
        marker_amplitude=args.marker_amplitude,
        ciphertext_gain=args.ciphertext_gain,
        node_name=args.node,
        sync_diag_interval=args.sync_diag_interval,
        tx_diag_interval=args.tx_diag_interval,
        initial_full_repeats=args.initial_full_repeats,
        checkpoint_confirmations=args.checkpoint_confirmations,
        checkpoint_max_drift=args.checkpoint_max_drift,
        checkpoint_min_confidence=args.checkpoint_min_confidence,
    )

    virtual_devices = PulseVirtualDevices()
    virtual_devices.cleanup_stale()
    mic, speakers = get_default_audio_devices()

    print("Physical mic        :", mic)
    print("Physical speakers   :", speakers)
    print("Sample rate         :", RATE)
    print("Transform           :", args.transform)
    print("Operation           :", args.operation)
    print("Key schedule        :", args.key_schedule)
    print("Sync                : robust chirp + DTMF-like modem")
    print("Sync period         :", args.sync_period, "s")
    print("Full refresh period :", args.full_sync_period, "s")
    print("Initial FULL x      :", args.initial_full_repeats)
    print("Checkpoint modulo   : 64")
    print("Checkpoint confirms :", args.checkpoint_confirmations)
    print("Checkpoint max drift:", args.checkpoint_max_drift, "frames")
    print("FULL marker         :", f"{1000.0 * processor.sync_codec.full_marker_samples_for(processor.hop_size) / RATE:.1f} ms")
    print("Checkpoint marker   :", f"{1000.0 * processor.sync_codec.checkpoint_marker_samples_for(processor.hop_size) / RATE:.1f} ms")

    bridge = None
    try:
        # Only create virtual endpoints. Device selection remains manual.
        virtual_devices.start(cleanup=False)
        bridge = AudioBridge(
            mic=mic,
            speakers=speakers,
            process_tx=processor.process_tx,
            process_rx=processor.process_rx,
            on_tx_gap=processor.on_tx_gap,
            on_rx_gap=processor.on_rx_gap,
        )
        bridge.start()

        print()
        print("Messenger settings:")
        print("  Microphone = TX_Microphone")
        print("  Speakers   = RX_sink")
        print("Ctrl+C = stop")

        while not stop_event.wait(0.5):
            pass
    finally:
        if bridge is not None:
            bridge.stop()
        virtual_devices.stop()


def main() -> None:
    args = parse_args()
    with RunLog(args):
        run(args)


if __name__ == "__main__":
    main()
