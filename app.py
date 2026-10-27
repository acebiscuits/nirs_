#!/usr/bin/env python3

from __future__ import annotations

import argparse
import select
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import zlib
import wave
import queue
import csv
import subprocess
import os
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
    DetectedMarkerWindow,
    DetectedSyncEvent,
    RobustSyncCodec,
    RobustSyncStreamDecoder,
)
from sync import FrameSynchronizer, SyncInfo


class AudioWavRecorder:
    """Non-blocking, bounded 5-tap PCM16 recorder; files are not part of the live data path.

    All WAVs use one rate/channel layout.  RX masked capture is recorded at its
    actual pre-decrypt position; RX speaker capture is exactly the PCM bytes
    returned to the pacat playback worker.  The streams may differ in length
    due to transform warmup and fixed RX look-ahead; align with RX CP log
    coordinates, not by assuming all stems have the same start time.
    """

    TAP_NAMES = ("tx_mic", "tx_tunnel", "rx_input", "rx_masked", "rx_speaker")

    def __init__(self, directory: Path, sample_rate: int, duration_sec: float):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sample_rate = int(sample_rate)
        self.max_frames = int(round(duration_sec * sample_rate))
        self.queue = queue.Queue(maxsize=4096)
        self.frames = {name: 0 for name in self.TAP_NAMES}
        self.drops = {name: 0 for name in self.TAP_NAMES}
        self.paths = {name: self.directory / (name + ".wav") for name in self.TAP_NAMES}
        self.files = {}
        for name, path in self.paths.items():
            file = wave.open(str(path), "wb")
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(sample_rate)
            self.files[name] = file
        self.event_path = self.directory / "events.csv"
        self.event_file = self.event_path.open("w", encoding="utf-8", newline="")
        self.event_writer = csv.writer(self.event_file)
        self.event_writer.writerow(["wallclock", "direction", "event", "fields"])
        self.worker = threading.Thread(target=self._worker, name="wav-recorder", daemon=True)
        self.worker.start()

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                break
            kind, name, data = item
            try:
                if kind == "wav":
                    remaining = self.max_frames - self.frames[name]
                    if remaining > 0:
                        payload = data[:remaining * 2]
                        self.files[name].writeframesraw(payload)
                        self.frames[name] += len(payload) // 2
                elif kind == "event":
                    self.event_writer.writerow(data)
            finally:
                self.queue.task_done()

    def tap(self, name: str, pcm_bytes: bytes):
        if name not in self.frames or not pcm_bytes or self.frames[name] >= self.max_frames:
            return
        try:
            self.queue.put_nowait(("wav", name, pcm_bytes))
        except queue.Full:
            self.drops[name] += 1

    def event(self, direction: str, event: str, fields: dict):
        if event not in {"CP PREAMBLE DETECT", "CP CUT APPLIED", "CP FULL CUT APPLIED", "CP CUT TOO LATE", "FULL APPLY", "CHECKPOINT TRACK", "CHECKPOINT APPLY", "MODE CHANGE", "REALTIME GAP"}:
            return
        line = [datetime.now().astimezone().isoformat(timespec="milliseconds"), direction, event,
                " ".join("%s=%s" % (k, v) for k,v in fields.items())]
        try:
            self.queue.put_nowait(("event", "", line))
        except queue.Full:
            pass

    def close(self):
        self.queue.put(None)
        self.queue.join()
        self.worker.join(timeout=10.0)
        for f in self.files.values():
            f.close()
        self.event_file.close()
        print("AUDIO CAPTURE DIR  :", self.directory.resolve())
        for name in self.TAP_NAMES:
            print("AUDIO CAPTURE      : %s frames=%d dropped_chunks=%d path=%s" %
                  (name, self.frames[name], self.drops[name], self.paths[name].resolve()), flush=True)


@dataclass
class _PendingMarker:
    kind: str  # FULL | CHECKPOINT
    payload: SyncInfo | int


class LiveProcessor:
    """Real-time cipher + robust in-band sync.

    Architecture:
      * one strong CRC32 FULL establishes absolute frame index;
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
        full_marker_amplitude: float,
        checkpoint_marker_amplitude: float,
        full_preamble_amplitude: float,
        full_symbol_amplitude: float,
        checkpoint_preamble_amplitude: float,
        checkpoint_symbol_amplitude: float,
        full_audio_gain: float,
        checkpoint_audio_gain: float,
        full_preamble_ms: float,
        full_symbol_ms: float,
        checkpoint_preamble_ms: float,
        checkpoint_symbol_ms: float,
        full_detection_threshold: float,
        checkpoint_detection_threshold: float,
        ciphertext_gain: float,
        node_name: str,
        sync_diag_interval: float,
        tx_diag_interval: float,
        initial_full_repeats: int,
        checkpoint_confirmations: int,
        checkpoint_max_drift: int,
        checkpoint_min_confidence: float,
        checkpoint_large_drift_confirmations: int,
        checkpoint_large_drift_tolerance: int,
        checkpoint_large_drift_min_confidence: float,
        rx_sync_mute: str,
        rx_sync_lookahead_ms: float,
        rx_full_mute_threshold: float,
        rx_checkpoint_mute_threshold: float,
        rx_sync_mute_fade_ms: float,
        rx_checkpoint_suppress: str = "preamble",
        rx_mode: str = "correct",
        wrong_key: bytes | None = None,
        recorder: AudioWavRecorder | None = None,
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
        if checkpoint_large_drift_confirmations < 2:
            raise ValueError("checkpoint_large_drift_confirmations must be >= 2")
        if checkpoint_large_drift_tolerance < 0:
            raise ValueError("checkpoint_large_drift_tolerance must be >= 0")
        if not (0.0 <= checkpoint_large_drift_min_confidence <= 1.0):
            raise ValueError("checkpoint_large_drift_min_confidence must be in [0, 1]")
        if tx_cipher.hop_size != rx_cipher.hop_size:
            raise ValueError("TX and RX hop sizes must match")

        self.recorder = recorder
        self.tx_cipher = tx_cipher
        self.rx_cipher = rx_cipher
        # One decryptor, two test keys. Switching keys at the spectral frame
        # boundary preserves its stream buffers and counter without running a
        # second expensive MDCT/STFT decryptor in parallel.
        self._rx_correct_key = rx_cipher.key
        self._rx_wrong_key = wrong_key or hashlib.sha256(b"WRONG-RX-TEST:" + self._rx_correct_key).digest()
        if rx_mode not in {"mute", "raw", "masked_raw", "correct", "wrong"}:
            raise ValueError("rx_mode must be mute/raw/masked_raw/correct/wrong")
        self.rx_mode = rx_mode
        self._rx_mode_lock = threading.Lock()
        self._last_unlocked_log = 0.0
        self.sample_rate = int(sample_rate)
        self.node_name = node_name
        self.ciphertext_gain = float(ciphertext_gain)
        self.hop_size = int(tx_cipher.hop_size)

        if not (0.0 <= full_audio_gain <= 1.0):
            raise ValueError("full_audio_gain must be in [0, 1]")
        if not (0.0 <= checkpoint_audio_gain <= 1.0):
            raise ValueError("checkpoint_audio_gain must be in [0, 1]")
        if rx_sync_mute not in {"off", "checkpoint", "all"}:
            raise ValueError("rx_sync_mute must be off, checkpoint or all")
        if rx_checkpoint_suppress not in {"preamble", "all"}:
            raise ValueError("rx_checkpoint_suppress must be preamble or all")
        if rx_sync_lookahead_ms < 0.0:
            raise ValueError("rx_sync_lookahead_ms must be >= 0")
        if not (0.0 <= rx_full_mute_threshold <= 1.0):
            raise ValueError("rx_full_mute_threshold must be in [0, 1]")
        if not (0.0 <= rx_checkpoint_mute_threshold <= 1.0):
            raise ValueError("rx_checkpoint_mute_threshold must be in [0, 1]")
        if rx_sync_mute_fade_ms < 0.0:
            raise ValueError("rx_sync_mute_fade_ms must be >= 0")

        self.full_audio_gain = float(full_audio_gain)
        self.checkpoint_audio_gain = float(checkpoint_audio_gain)
        self.rx_sync_mute = rx_sync_mute
        self.rx_checkpoint_suppress = rx_checkpoint_suppress
        self.rx_sync_lookahead_samples = int(
            round(self.sample_rate * float(rx_sync_lookahead_ms) / 1000.0)
        )
        self.rx_full_mute_threshold = float(rx_full_mute_threshold)
        self.rx_checkpoint_mute_threshold = float(rx_checkpoint_mute_threshold)
        self.rx_sync_mute_fade_samples = int(
            round(self.sample_rate * float(rx_sync_mute_fade_ms) / 1000.0)
        )

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
            full_marker_amplitude=full_marker_amplitude,
            checkpoint_marker_amplitude=checkpoint_marker_amplitude,
            full_preamble_amplitude=full_preamble_amplitude,
            full_symbol_amplitude=full_symbol_amplitude,
            checkpoint_preamble_amplitude=checkpoint_preamble_amplitude,
            checkpoint_symbol_amplitude=checkpoint_symbol_amplitude,
            full_preamble_ms=full_preamble_ms,
            full_symbol_ms=full_symbol_ms,
            checkpoint_preamble_ms=checkpoint_preamble_ms,
            checkpoint_symbol_ms=checkpoint_symbol_ms,
            full_detection_threshold=full_detection_threshold,
            checkpoint_detection_threshold=checkpoint_detection_threshold,
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
        self.checkpoint_large_drift_confirmations = int(checkpoint_large_drift_confirmations)
        self.checkpoint_large_drift_tolerance = int(checkpoint_large_drift_tolerance)
        self.checkpoint_large_drift_min_confidence = float(checkpoint_large_drift_min_confidence)

        self.rx_locked = False
        self._pending_checkpoint_drift: int | None = None
        self._pending_checkpoint_count = 0

        # Large-drift reacquisition is intentionally separate from the normal
        # +/- few-frame correction path.  A single modulo checkpoint is never
        # allowed to jump the decryptor by a large amount.  Several consecutive
        # high-confidence checkpoints must independently point to approximately
        # the same offset first.
        self._large_drift_estimate: float | None = None
        self._large_drift_count = 0

        self.tx_warmup_done = False
        self.pending_marker: _PendingMarker | None = None
        self.active_marker = np.empty(0, dtype=np.float64)
        self.active_marker_pos = 0
        self.active_marker_kind: str | None = None

        self._last_checkpoint_scheduled_frame = 0
        self._last_full_scheduled_frame = 0
        # Don't flood the voice channel with 200ms FULL packets on every
        # tiny capture gap. This happened dozens of times in the MDCT logs.
        self._gap_full_pending = False
        self._last_gap_full_frame = -(2 * self.sample_rate // self.hop_size)
        self._gap_full_cooldown_frames = max(
            1, (2 * self.sample_rate + self.hop_size - 1) // self.hop_size
        )

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

        # Independent proof of the audio path: counters reflect bytes delivered
        # to the physical-speaker pacat process, not what a browser UI claims.
        self._rx_meter_mode = self.rx_mode
        self._rx_meter_next_at = time.monotonic() + 2.0
        self._rx_meter_in_sq = 0.0
        self._rx_meter_out_sq = 0.0
        self._rx_meter_in_count = 0
        self._rx_meter_out_count = 0
        self._rx_meter_out_nonzero = 0
        self._rx_meter_in_crc = 0
        self._rx_meter_out_crc = 0
        self._rx_meter_in_peak = 0.0
        self._rx_meter_out_peak = 0.0

        # RX speaker suppression.  Detector sees the original audio immediately,
        # while playback/decrypt waits a short fixed look-ahead.  As soon as a
        # sync preamble is recognized we know the complete marker interval and
        # can silence that interval before it reaches local speakers.
        self._rx_input_total_abs = 0
        self._rx_hold_start_abs = 0
        self._rx_hold = np.empty(0, dtype=np.float64)
        self._rx_pending_events: list[DetectedSyncEvent] = []
        self._rx_mute_intervals: list[tuple[int, int, str, float]] = []
        self._rx_cp_preamble_zeroed = 0
        self._rx_cp_preamble_zeroed_total = 0
        self._rx_cp_full_zeroed_total = 0
        self._rx_cp_detected_total = 0
        self._rx_cp_detected_late_total = 0
        self._rx_cp_masked_segments_total = 0
        self._rx_cp_masked_input_power = 0.0
        self._rx_current_feed_end_abs = 0

    def get_rx_mode(self) -> str:
        with self._rx_mode_lock:
            return self.rx_mode

    def set_rx_mode(self, mode: str) -> None:
        if mode not in {"mute", "raw", "masked_raw", "correct", "wrong"}:
            raise ValueError("mode must be mute/raw/masked_raw/correct/wrong")
        with self._rx_mode_lock:
            old = self.rx_mode
            self.rx_mode = mode
        self._log(
            "RX<-REMOTE", "MODE CHANGE",
            old=old, new=mode.upper(), locked=self.rx_locked,
            frame=self.rx_cipher.frame_index,
            reason=("MUTE=RX output forced to zero" if mode == "mute" else
                    "RAW=received ciphertext; sync-mute bypassed" if mode == "raw" else
                    "MASKED_RAW=received PCM after sync-mute; no decrypt" if mode == "masked_raw" else
                    "CORRECT=real key; silence until FULL" if mode == "correct" else
                    "WRONG=deliberately wrong key; silence until FULL"),
        )

    def _log(self, direction: str, event: str, **fields) -> None:
        ts = datetime.now().astimezone().strftime("%H:%M:%S.%f")[:-3]
        tail = " ".join(f"{key}={value}" for key, value in fields.items())
        print(
            f"{ts} [{self.node_name}] {direction} {event}"
            + (f" {tail}" if tail else ""),
            flush=True,
        )
        if self.recorder is not None:
            self.recorder.event(direction, event, fields)

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
                    self.full_audio_gain * out[pos:pos + count]
                    + marker_piece
                )
            else:
                out[pos:pos + count] = (
                    self.checkpoint_audio_gain * out[pos:pos + count]
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

        if self._gap_full_pending:
            if next_index - self._last_gap_full_frame >= self._gap_full_cooldown_frames:
                self._queue_full(next_index)
                self._last_gap_full_frame = next_index
                self._gap_full_pending = False
            # Do not emit checkpoints while an emergency FULL is waiting.
            return

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
        # Schedule at most one recovery FULL per two seconds, instead of one
        # for every dropped BLOCK (which rendered speech unusable in MDCT).
        self._gap_full_pending = True

        self._log(
            "TX->REMOTE",
            "REALTIME GAP",
            dropped_blocks=dropped_blocks,
            frame_before=before,
            frame_after=after,
            action="ZERO_ADVANCE_FULL_PENDING_RATE_LIMITED",
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
        # The detector already received the held 100ms, but the decryptor has
        # not: its audio was waiting in the RX look-ahead queue. On a capture
        # discontinuity we discard that stale held audio and advance the cipher
        # over BOTH the held portion and the newly missing part. Otherwise the
        # decoder's absolute counter diverges from the playback cursor.
        held_samples = int(self._rx_hold.size)
        self.sync_decoder.skip_samples(skipped_samples)
        self._rx_input_total_abs += skipped_samples
        self._rx_hold = np.empty(0, dtype=np.float64)
        self._rx_hold_start_abs = self._rx_input_total_abs
        self._rx_pending_events.clear()
        self._rx_mute_intervals.clear()

        before = int(self.rx_cipher.frame_index)
        if self.rx_locked:
            missing_for_cipher = held_samples + skipped_samples
            block = np.zeros(min(block_samples, self.hop_size), dtype=np.float64)
            while missing_for_cipher > 0:
                count = min(missing_for_cipher, block.size)
                self.rx_cipher.decrypt(block[:count])
                missing_for_cipher -= count
        after = int(self.rx_cipher.frame_index)

        self._pending_checkpoint_drift = None
        self._pending_checkpoint_count = 0
        self._log(
            "RX<-REMOTE",
            "REALTIME GAP",
            dropped_blocks=dropped_blocks,
            held_discarded_samples=held_samples,
            gap_skipped_samples=skipped_samples,
            frame_before=before,
            frame_after=after,
            locked=self.rx_locked,
            action=("ZERO_ADVANCE_CLEAR_LOOKAHEAD" if self.rx_locked else "WAIT_FULL_CLEAR_LOOKAHEAD"),
            decoder_app_offset_samples=int(self.sync_decoder._total_abs - self._rx_input_total_abs),
        )

    def process_tx(self, data: bytes) -> bytes:
        if self.recorder is not None:
            self.recorder.tap("tx_mic", data)
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

        encoded_bytes = self._float_to_bytes(encrypted)
        if self.recorder is not None:
            self.recorder.tap("tx_tunnel", encoded_bytes)
        return encoded_bytes

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
        if not self.rx_locked:
            self.rx_sync.apply_sync(self.rx_cipher, info, hard=True)
            self.rx_locked = True
            action = "HARD_LOCK"
        else:
            # Same nonce + CRC32-valid absolute checkpoint: preserve overlap
            # state and only re-anchor the frame counter at marker END.
            self.rx_sync.apply_sync(self.rx_cipher, info, hard=False)
            action = "ABS_REANCHOR"

        self._pending_checkpoint_drift = None
        self._pending_checkpoint_count = 0
        self._large_drift_estimate = None
        self._large_drift_count = 0

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
            self._large_drift_estimate = None
            self._large_drift_count = 0
            self._log(
                "RX<-REMOTE",
                "CHECKPOINT REJECT",
                reason="low_confidence",
                mod=cp.frame_mod,
                confidence=f"{cp.confidence:.3f}",
                local=local,
            )
            return

        # Normal tracking path: small corrections are cheap and can use the
        # existing confirmation rule.  A zero-drift checkpoint also clears any
        # stale large-drift hypothesis.
        if abs(drift) <= self.checkpoint_max_drift:
            self._large_drift_estimate = None
            self._large_drift_count = 0

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
        else:
            # modulo-N can only identify an offset inside roughly +/- N/2.
            # Exactly N/2 is ambiguous (e.g. +32 and -32 for modulo 64), so it
            # is never auto-applied.
            ambiguous_limit = cp.modulus // 2
            if abs(drift) >= ambiguous_limit:
                self._pending_checkpoint_drift = None
                self._pending_checkpoint_count = 0
                self._large_drift_estimate = None
                self._large_drift_count = 0
                self._log(
                    "RX<-REMOTE",
                    "CHECKPOINT REJECT",
                    reason="ambiguous_modulo_drift",
                    mod=cp.frame_mod,
                    confidence=f"{cp.confidence:.3f}",
                    local=local,
                    resolved=resolved,
                    drift=drift,
                    modulus=cp.modulus,
                )
                return

            if cp.confidence < self.checkpoint_large_drift_min_confidence:
                self._pending_checkpoint_drift = None
                self._pending_checkpoint_count = 0
                self._large_drift_estimate = None
                self._large_drift_count = 0
                self._log(
                    "RX<-REMOTE",
                    "CHECKPOINT REJECT",
                    reason="large_drift_low_confidence",
                    mod=cp.frame_mod,
                    confidence=f"{cp.confidence:.3f}",
                    local=local,
                    resolved=resolved,
                    drift=drift,
                    min_confidence=f"{self.checkpoint_large_drift_min_confidence:.3f}",
                )
                return

            # Do not let the normal small-drift confirmation state leak into a
            # large reacquisition decision.
            self._pending_checkpoint_drift = None
            self._pending_checkpoint_count = 0

            if (
                self._large_drift_estimate is not None
                and abs(drift - self._large_drift_estimate)
                <= self.checkpoint_large_drift_tolerance
            ):
                self._large_drift_count += 1
                # Running mean makes +25,+26,+26 count as one stable hypothesis
                # instead of three unrelated exact integers.
                self._large_drift_estimate += (
                    float(drift) - self._large_drift_estimate
                ) / float(self._large_drift_count)
            else:
                self._large_drift_estimate = float(drift)
                self._large_drift_count = 1

            if self._large_drift_count < self.checkpoint_large_drift_confirmations:
                self._log(
                    "RX<-REMOTE",
                    "CHECKPOINT LARGE_SUSPECT",
                    mod=cp.frame_mod,
                    confidence=f"{cp.confidence:.3f}",
                    local=local,
                    resolved=resolved,
                    drift=drift,
                    estimate=f"{self._large_drift_estimate:.2f}",
                    confirmations=(
                        f"{self._large_drift_count}/"
                        f"{self.checkpoint_large_drift_confirmations}"
                    ),
                    tolerance=self.checkpoint_large_drift_tolerance,
                )
                return

            # The current checkpoint is itself modulo-consistent and belongs to
            # the stable hypothesis, so re-anchor to its exact resolved frame.
            estimate = self._large_drift_estimate
            confirmations = self._large_drift_count
            self.rx_cipher.set_frame_index(resolved)
            self._log(
                "RX<-REMOTE",
                "CHECKPOINT LARGE_APPLY",
                mod=cp.frame_mod,
                confidence=f"{cp.confidence:.3f}",
                local_before=local,
                resolved=resolved,
                drift=drift,
                estimate=f"{estimate:.2f}",
                confirmations=confirmations,
                preamble=f"{preamble_score:.3f}",
            )
            self._large_drift_estimate = None
            self._large_drift_count = 0
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

    def _decrypt_segment(
        self,
        segment: np.ndarray,
        output_parts: list[np.ndarray],
        raw_unmodified: np.ndarray | None = None,
    ) -> None:
        if segment.size == 0:
            return

        # RAW is the actual incoming PCM before any marker suppression.
        raw_listen = segment if raw_unmodified is None else raw_unmodified
        mode = self.get_rx_mode()
        if mode == "mute" and not self.rx_locked:
            output_parts.append(np.zeros_like(segment))
            return
        if not self.rx_locked:
            # CRITICAL: previous build silently passed ciphertext unchanged in
            # CORRECT mode. That made an unencrypted/bypassed browser route look
            # like successful decryption. Fail closed instead.
            if mode == "raw":
                output_parts.append(raw_listen.copy())
            elif mode == "masked_raw":
                output_parts.append(segment.copy())
            else:
                output_parts.append(np.zeros_like(segment))
            return

        # Even in RAW we must advance the same live cipher so switching back to
        # CORRECT does not leave frame_index and overlap state behind.
        ciphertext = segment / self.ciphertext_gain
        # Use a different key ONLY while calculating one selected WRONG
        # segment. Restore the key in all cases; no second expensive cipher.
        self.rx_cipher.key = self._rx_wrong_key if mode == "wrong" else self._rx_correct_key
        try:
            restored = self.rx_cipher.decrypt(ciphertext)
        finally:
            self.rx_cipher.key = self._rx_correct_key

        if mode == "mute":
            # Advance the cipher, but output the same sample count as a normal
            # decrypted segment would have generated: digital silence.
            if restored.size:
                output_parts.append(np.zeros_like(restored))
        elif mode == "raw":
            # Raw received PCM BEFORE optional sync-mute, no decrypt or gain
            # compensation. Decoder state was still advanced above.
            output_parts.append(raw_listen.copy())
        elif mode == "masked_raw":
            # Same incoming PCM, but WITH preamble suppression. Makes it
            # possible to prove muting without STFT overlap/decryption.
            output_parts.append(segment.copy())
        elif restored.size:
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
            rx_mode=self.get_rx_mode().upper(),
            frame=self.rx_cipher.frame_index,
            pending_drift=(
                "none"
                if self._pending_checkpoint_drift is None
                else self._pending_checkpoint_drift
            ),
            large_drift=(
                "none"
                if self._large_drift_estimate is None
                else f"{self._large_drift_estimate:.2f}:{self._large_drift_count}"
            ),
            rx_proc_max_ms=f"{self._rx_process_max_ms:.2f}",
            cp_preamble_zeroed_samples=self._rx_cp_preamble_zeroed,
            cp_preamble_zeroed_total=self._rx_cp_preamble_zeroed_total,
            cp_full_zeroed_total=self._rx_cp_full_zeroed_total,
            cp_cut_mode=self.rx_checkpoint_suppress,
            cp_detected_total=self._rx_cp_detected_total,
            cp_late_total=self._rx_cp_detected_late_total,
            cp_mask_segments_total=self._rx_cp_masked_segments_total,
            cp_muted_input_energy=f"{self._rx_cp_masked_input_power:.5f}",
            decoder_app_offset_samples=int(self.sync_decoder._total_abs - self._rx_input_total_abs),
        )
        self._rx_process_max_ms = 0.0
        self._rx_process_calls = 0
        self._rx_cp_preamble_zeroed = 0

        # IMPORTANT: do not reset RX look-ahead/mute absolute state here.
        # Those cursors must remain monotonic for the lifetime of the stream.
        # diagnostics_snapshot() resets only interval counters inside the detector.

    def _meter_rx_path(self, incoming: np.ndarray, outgoing: bytes) -> None:
        """Every two seconds prove whether THIS application emitted audio."""
        mode = self.get_rx_mode()
        if mode != self._rx_meter_mode:
            self._rx_meter_mode = mode
            self._rx_meter_next_at = time.monotonic() + 2.0
            self._rx_meter_in_sq = self._rx_meter_out_sq = 0.0
            self._rx_meter_in_count = self._rx_meter_out_count = 0
            self._rx_meter_out_nonzero = 0
            self._rx_meter_in_crc = self._rx_meter_out_crc = 0
            self._rx_meter_in_peak = self._rx_meter_out_peak = 0.0

        incoming = np.asarray(incoming, dtype=np.float64)
        if incoming.size:
            self._rx_meter_in_sq += float(np.dot(incoming, incoming))
            self._rx_meter_in_count += int(incoming.size)
            self._rx_meter_in_peak = max(self._rx_meter_in_peak, float(np.max(np.abs(incoming))))
            self._rx_meter_in_crc = zlib.crc32(
                np.rint(np.clip(incoming, -1.0, 0.9999695) * 32768.0).astype('<i2').tobytes(),
                self._rx_meter_in_crc,
            )

        if outgoing:
            arr = np.frombuffer(outgoing, dtype='<i2')
            norm = arr.astype(np.float64) / 32768.0
            self._rx_meter_out_sq += float(np.dot(norm, norm))
            self._rx_meter_out_count += int(norm.size)
            self._rx_meter_out_nonzero += int(np.count_nonzero(arr))
            self._rx_meter_out_peak = max(self._rx_meter_out_peak, float(np.max(np.abs(norm))))
            self._rx_meter_out_crc = zlib.crc32(outgoing, self._rx_meter_out_crc)

        if time.monotonic() < self._rx_meter_next_at:
            return
        self._rx_meter_next_at = time.monotonic() + 2.0
        def dbfs(sumsq: float, count: int) -> str:
            if not count or sumsq <= 0:
                return '-240.0'
            return f'{20.0 * np.log10(max(np.sqrt(sumsq / count), 1e-12)):.1f}'
        self._log(
            'RX<-REMOTE', 'AUDIO_PATH',
            mode=mode.upper(), locked=self.rx_locked,
            input_samples=self._rx_meter_in_count,
            input_rms_dbfs=dbfs(self._rx_meter_in_sq, self._rx_meter_in_count),
            input_peak_dbfs=dbfs(self._rx_meter_in_peak ** 2, 1),
            input_crc32=f'{self._rx_meter_in_crc:08x}',
            output_samples=self._rx_meter_out_count,
            output_rms_dbfs=dbfs(self._rx_meter_out_sq, self._rx_meter_out_count),
            output_peak_dbfs=dbfs(self._rx_meter_out_peak ** 2, 1),
            output_nonzero=self._rx_meter_out_nonzero,
            output_crc32=f'{self._rx_meter_out_crc:08x}',
        )
        self._rx_meter_in_sq = self._rx_meter_out_sq = 0.0
        self._rx_meter_in_count = self._rx_meter_out_count = 0
        self._rx_meter_out_nonzero = 0
        self._rx_meter_in_crc = self._rx_meter_out_crc = 0
        self._rx_meter_in_peak = self._rx_meter_out_peak = 0.0

    def _should_mute_window(self, window: DetectedMarkerWindow) -> bool:
        if self.rx_sync_mute == "off":
            return False
        if window.kind == "checkpoint":
            if self.rx_sync_mute not in {"checkpoint", "all"}:
                return False
            # The physical decoder has ALREADY accepted this preamble.
            # In preamble-only mode suppression must not introduce a second,
            # stricter threshold that lets a detected chirp reach speakers.
            return True  # valid decoded preamble -> mute, independent of symbol decoding
        if window.kind == "full":
            if self.rx_sync_mute != "all":
                return False
            return window.preamble_score >= self.rx_full_mute_threshold
        return False

    def _add_mute_window(self, window: DetectedMarkerWindow) -> None:
        start_abs = int(window.start_abs)
        if window.kind == "checkpoint":
            self._rx_cp_detected_total += 1
            late_samples = max(0, int(self._rx_hold_start_abs) - start_abs)
            if late_samples:
                self._rx_cp_detected_late_total += 1
            will_cut = self._should_mute_window(window)
            self._log(
                "RX<-REMOTE", "CP PREAMBLE DETECT",
                score=f"{window.preamble_score:.3f}",
                start_abs=start_abs,
                detector_delay_ms=f"{1000.0 * (self._rx_current_feed_end_abs - start_abs) / self.sample_rate:.2f}",
                playback_cursor=self._rx_hold_start_abs,
                late_samples=late_samples,
                will_cut=will_cut,
                rx_mode=self.get_rx_mode().upper(),
                raw_bypasses_cut=(self.get_rx_mode() == "raw"),
            )
            if late_samples:
                self._log(
                    "RX<-REMOTE", "CP CUT TOO LATE",
                    already_played_samples=late_samples,
                    advice="increase_--rx-sync-lookahead-ms",
                )
        if not self._should_mute_window(window):
            return

        end_abs = int(window.end_abs)
        if window.kind == "checkpoint" and self.rx_checkpoint_suppress == "preamble":
            # The preamble is the only part silenced.  The DTMF-like data
            # remains in the received PCM and is decrypted as usual; the
            # synchronization detector always receives unmodified input.
            end_abs = start_abs + self.sync_codec.checkpoint_preamble_samples
            window = DetectedMarkerWindow(
                kind="checkpoint_preamble",
                start_abs=start_abs,
                end_abs=end_abs,
                preamble_score=window.preamble_score,
            )
        if end_abs <= start_abs:
            return

        # Merge overlapping/adjacent windows.  False duplicate candidates around
        # one real preamble therefore cannot create a forest of tiny intervals.
        merged: list[tuple[int, int, str, float]] = []
        inserted = False
        for old_start, old_end, old_kind, old_score in self._rx_mute_intervals:
            if old_end < start_abs - 1 or end_abs < old_start - 1:
                merged.append((old_start, old_end, old_kind, old_score))
                continue
            start_abs = min(start_abs, old_start)
            end_abs = max(end_abs, old_end)
            window_kind = window.kind if window.kind == old_kind else "mixed"
            window_score = max(float(window.preamble_score), float(old_score))
            window = DetectedMarkerWindow(
                kind=window_kind,
                start_abs=start_abs,
                end_abs=end_abs,
                preamble_score=window_score,
            )
            inserted = True

        merged.append(
            (
                start_abs,
                end_abs,
                window.kind,
                float(window.preamble_score),
            )
        )
        merged.sort(key=lambda item: item[0])
        self._rx_mute_intervals = merged

        self._log(
            "RX<-REMOTE",
            "SYNC MUTE",
            kind=window.kind,
            preamble=f"{window.preamble_score:.3f}",
            muted_ms=f"{1000.0 * (end_abs - start_abs) / self.sample_rate:.1f}",
            checkpoint_mode=self.rx_checkpoint_suppress,
            note="scheduled_for_delayed_playback",
        )

    def _mask_sync_audio(self, samples: np.ndarray, start_abs: int) -> np.ndarray:
        if samples.size == 0 or not self._rx_mute_intervals:
            return samples

        out = samples.copy()
        seg_start = int(start_abs)
        seg_end = seg_start + int(out.size)
        fade = max(0, int(self.rx_sync_mute_fade_samples))

        for mute_start, mute_end, kind, _score in self._rx_mute_intervals:
            if kind in {"checkpoint_preamble", "checkpoint", "mixed"}:
                # DIAGNOSTIC: strictly zero the entire detected CP interval in
                # 'all' mode (including symbols AND alignment padding).  The
                # decoder receives unmodified RX audio, and can still decode.
                # Use only a 1-ms taper OUTSIDE the marked interval.  Never
                # put a nonzero fade INSIDE the marker, as the old all mode did.
                edge = min(fade, self.sample_rate // 1000)
                left = max(seg_start, mute_start - edge)
                right = min(seg_end, mute_end + edge)
                if right <= left:
                    continue
                indices = np.arange(left, right, dtype=np.int64)
                gains = np.zeros(indices.size, dtype=np.float64)
                if edge > 0:
                    pre = indices < mute_start
                    post = indices >= mute_end
                    if np.any(pre):
                        u = (mute_start - indices[pre]) / float(edge)
                        gains[pre] = np.sin(0.5 * np.pi * np.clip(u, 0, 1)) ** 2
                    if np.any(post):
                        u = (indices[post] - mute_end) / float(edge)
                        gains[post] = np.sin(0.5 * np.pi * np.clip(u, 0, 1)) ** 2
                in_piece = out[left - seg_start:right - seg_start].copy()
                out[left - seg_start:right - seg_start] *= gains
                zero_left = max(left, mute_start)
                zero_right = min(right, mute_end)
                if zero_right > zero_left:
                    zeroed = zero_right - zero_left
                    event = "CP CUT APPLIED" if kind == "checkpoint_preamble" else "CP FULL CUT APPLIED"
                    if kind == "checkpoint_preamble":
                        self._rx_cp_preamble_zeroed += zeroed
                        self._rx_cp_preamble_zeroed_total += zeroed
                        total = self._rx_cp_preamble_zeroed_total
                    else:
                        self._rx_cp_full_zeroed_total += zeroed
                        total = self._rx_cp_full_zeroed_total
                    self._rx_cp_masked_segments_total += 1
                    self._rx_cp_masked_input_power += float(np.dot(in_piece, in_piece))
                    self._log(
                        "RX<-REMOTE", event,
                        start_abs=zero_left,
                        end_abs=zero_right,
                        zeroed_samples=zeroed,
                        zeroed_total=total,
                        input_peak=f"{float(np.max(np.abs(in_piece))):.5f}",
                        cut_pcm_peak=f"{float(np.max(np.abs(out[zero_left-seg_start:zero_right-seg_start]))):.8f}",
                        rx_mode=self.get_rx_mode().upper(),
                        output_uses_mask=(self.get_rx_mode() != "raw"),
                    )
                continue

            # Full marker and legacy full-checkpoint muting are unchanged.
            left = max(seg_start, mute_start)
            right = min(seg_end, mute_end)
            if right <= left:
                continue

            i0 = left - seg_start
            i1 = right - seg_start
            gain = np.zeros(i1 - i0, dtype=np.float64)
            if fade > 0:
                for idx in range(i1 - i0):
                    abs_pos = left + idx
                    dist_from_start = abs_pos - mute_start
                    dist_to_end = mute_end - abs_pos - 1
                    edge = min(dist_from_start, dist_to_end)
                    if edge < fade:
                        gain[idx] = np.cos(
                            0.5 * np.pi * max(0.0, float(edge)) / float(fade)
                        ) ** 2
            out[i0:i1] *= gain

        return out

    def _process_released_audio(
        self,
        raw: np.ndarray,
        raw_start_abs: int,
    ) -> bytes:
        if raw.size == 0:
            return b""

        raw_start_abs = int(raw_start_abs)
        raw_end_abs = raw_start_abs + int(raw.size)
        output_parts: list[np.ndarray] = []
        cursor_abs = raw_start_abs

        self._rx_pending_events.sort(key=lambda event: event.end_abs)

        while self._rx_pending_events and self._rx_pending_events[0].end_abs <= raw_end_abs:
            event = self._rx_pending_events.pop(0)
            event_end = max(cursor_abs, min(int(event.end_abs), raw_end_abs))

            if event_end > cursor_abs:
                i0 = cursor_abs - raw_start_abs
                i1 = event_end - raw_start_abs
                unmodified = raw[i0:i1]
                piece = self._mask_sync_audio(unmodified, cursor_abs)
                if self.recorder is not None:
                    self.recorder.tap("rx_masked", self._float_to_bytes(piece))
                self._decrypt_segment(piece, output_parts, raw_unmodified=unmodified)

            self._handle_detected_event(event)
            cursor_abs = event_end

        if cursor_abs < raw_end_abs:
            i0 = cursor_abs - raw_start_abs
            unmodified = raw[i0:]
            piece = self._mask_sync_audio(unmodified, cursor_abs)
            if self.recorder is not None:
                self.recorder.tap("rx_masked", self._float_to_bytes(piece))
            self._decrypt_segment(piece, output_parts, raw_unmodified=unmodified)

        if not output_parts:
            return b""
        return self._float_to_bytes(np.concatenate(output_parts))

    def process_rx(self, data: bytes) -> bytes:
        if self.recorder is not None:
            self.recorder.tap("rx_input", data)
        started = time.perf_counter()
        incoming = self._bytes_to_float(data)
        block_start_abs = int(self._rx_input_total_abs)
        block_end_abs = block_start_abs + int(incoming.size)
        self._rx_current_feed_end_abs = block_end_abs

        # Detector always receives untouched WebRTC audio.  Speaker muting is a
        # separate branch and therefore cannot make sync detection worse.
        events = self.sync_decoder.feed(incoming)
        windows = self.sync_decoder.pop_marker_windows()

        for window in windows:
            self._add_mute_window(window)

        self._rx_pending_events.extend(events)
        self._rx_input_total_abs = block_end_abs

        if incoming.size:
            if self._rx_hold.size == 0:
                self._rx_hold_start_abs = block_start_abs
            self._rx_hold = np.concatenate((self._rx_hold, incoming))

        release_until_abs = max(
            self._rx_hold_start_abs,
            block_end_abs - self.rx_sync_lookahead_samples,
        )
        release_count = min(
            int(self._rx_hold.size),
            max(0, int(release_until_abs - self._rx_hold_start_abs)),
        )

        if release_count > 0:
            release_start_abs = int(self._rx_hold_start_abs)
            released = self._rx_hold[:release_count].copy()
            self._rx_hold = self._rx_hold[release_count:]
            self._rx_hold_start_abs += release_count
            out_bytes = self._process_released_audio(released, release_start_abs)
        else:
            out_bytes = b""

        # Remove mute windows that are fully behind the playback cursor.
        self._rx_mute_intervals = [
            interval
            for interval in self._rx_mute_intervals
            if interval[1] > self._rx_hold_start_abs
        ]

        # A decoded event should normally still be ahead of the delayed playback
        # cursor.  If the user intentionally sets too little look-ahead, apply a
        # late event immediately rather than letting it sit forever.
        late_events = [
            event
            for event in self._rx_pending_events
            if event.end_abs <= self._rx_hold_start_abs
        ]
        if late_events:
            for event in sorted(late_events, key=lambda item: item.end_abs):
                self._handle_detected_event(event)
            late_ids = {id(event) for event in late_events}
            self._rx_pending_events = [
                event for event in self._rx_pending_events if id(event) not in late_ids
            ]

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._rx_process_calls += 1
        self._rx_process_max_ms = max(self._rx_process_max_ms, elapsed_ms)
        self._maybe_log_sync_diagnostics()
        self._meter_rx_path(incoming, out_bytes)
        if self.recorder is not None:
            self.recorder.tap("rx_speaker", out_bytes)
        return out_bytes


def build_cipher(
    transform: str,
    operation: str,
    key: bytes,
    key_schedule: str,
):
    """Create cipher without legacy session_nonce, which is no longer on wire.

    Compatible with both current and older STFTCipher/MDCTCipher signatures.
    """
    if transform == "stft":
        return STFTCipher(
            operation=operation,
            key=key,
            key_schedule=key_schedule,
        )
    if transform == "mdct":
        return MDCTCipher(
            operation=operation,
            key=key,
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
    parser.add_argument("--record-audio", action="store_true", help="record 5 PCM16 WAV taps and sync events for debugging")
    parser.add_argument("--record-seconds", type=float, default=35.0, help="maximum duration PER TAP in seconds (default: 35)")
    parser.add_argument("--record-dir", default="", help="output directory (default: log-dir/audio/<timestamp>)")
    parser.add_argument("--keep-loopbacks", action="store_true", help="do NOT unload PulseAudio module-loopback on exit (default: remove all)")
    parser.add_argument(
        "--rx-mode", choices=["mute", "raw", "masked_raw", "correct", "wrong"], default="correct",
        help="initial RX listening mode; change live with 0/1/2/3/4 then ENTER",
    )
    parser.add_argument("--sync-period", type=float, default=0.5)
    parser.add_argument(
        "--full-sync-period",
        type=float,
        default=30.0,
        help="rare CRC32 absolute re-anchor interval",
    )
    parser.add_argument(
        "--full-marker-amplitude",
        type=float,
        default=0.30,
        help="FULL modem base amplitude",
    )
    parser.add_argument(
        "--checkpoint-marker-amplitude",
        type=float,
        default=0.30,
        help="checkpoint modem base amplitude; historical 0.80/0.72 internal ratios remain",
    )
    parser.add_argument("--full-preamble-amplitude", type=float, default=0.30)
    parser.add_argument("--full-symbol-amplitude", type=float, default=0.16)
    parser.add_argument("--checkpoint-preamble-amplitude", type=float, default=0.55)
    parser.add_argument("--checkpoint-symbol-amplitude", type=float, default=0.01)#0.14
    parser.add_argument("--full-audio-gain", type=float, default=1)#0.58
    parser.add_argument("--checkpoint-audio-gain", type=float, default=0.67)
    parser.add_argument("--full-preamble-ms", type=float, default=60.0)
    parser.add_argument("--full-symbol-ms", type=float, default=8.0)
    parser.add_argument("--checkpoint-preamble-ms", type=float, default=12.0)
    parser.add_argument("--checkpoint-symbol-ms", type=float, default=4.0)
    parser.add_argument("--full-detection-threshold", type=float, default=0.50)
    parser.add_argument("--checkpoint-detection-threshold", type=float, default=0.48)
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
        "--checkpoint-large-drift-confirmations",
        type=int,
        default=3,
        help="high-confidence large drift must persist this many checkpoints before re-anchor",
    )
    parser.add_argument(
        "--checkpoint-large-drift-tolerance",
        type=int,
        default=2,
        help="large-drift observations within +/- this many frames belong to one hypothesis",
    )
    parser.add_argument(
        "--checkpoint-large-drift-min-confidence",
        type=float,
        default=0.60,
        help="minimum confidence for large modulo re-acquisition",
    )
    parser.add_argument(
        "--rx-sync-mute",
        choices=["off", "checkpoint", "all"],
        default="all",
        help="suppress detected modem markers from local speakers",
    )
    parser.add_argument(
        "--rx-checkpoint-suppress",
        choices=["preamble", "all"],
        default="preamble",
        help="preamble: mute first 12ms only; all: strictly zero ENTIRE checkpoint incl. data (diagnostic)",
    )
    parser.add_argument(
        "--rx-sync-lookahead-ms",
        type=float,
        default=100.0,
        help="fixed RX look-ahead used to mute a marker from its beginning",
    )
    parser.add_argument(
        "--rx-full-mute-threshold",
        type=float,
        default=0.50,
        help="minimum FULL preamble score before local speaker muting",
    )
    parser.add_argument(
        "--rx-checkpoint-mute-threshold",
        type=float,
        default=0.50,
        help="minimum checkpoint preamble score before local speaker muting",
    )
    parser.add_argument(
        "--rx-sync-mute-fade-ms",
        type=float,
        default=3.0,
        help="fade at mute window edges to avoid clicks",
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

    # Full sync now carries only next_frame_index and CRC32; no session_nonce.
    tx_cipher = build_cipher(
        args.transform, args.operation, key, args.key_schedule,
    )
    rx_cipher = build_cipher(
        args.transform, args.operation, key, args.key_schedule,
    )

    recorder = None
    if args.record_audio:
        if not (1.0 <= args.record_seconds <= 600.0):
            raise ValueError("--record-seconds must be between 1 and 600")
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        dirname = args.record_dir or str(Path(args.log_dir) / "audio" / (
            "%s_%s_%s_%s" % (stamp, args.node, args.transform, args.operation)))
        recorder = AudioWavRecorder(Path(dirname), RATE, args.record_seconds)
        print("AUDIO CAPTURE ON   :", Path(dirname).resolve())
        print("AUDIO CAPTURE FILES:", ", ".join(AudioWavRecorder.TAP_NAMES))
        print("NOTE: WAVs may contain your microphone/other people's speech; review before sharing.")

    processor = LiveProcessor(
        tx_cipher=tx_cipher,
        rx_cipher=rx_cipher,
        sample_rate=RATE,
        sync_period_seconds=args.sync_period,
        full_sync_period_seconds=args.full_sync_period,
        full_marker_amplitude=args.full_marker_amplitude,
        checkpoint_marker_amplitude=args.checkpoint_marker_amplitude,
        full_preamble_amplitude=args.full_preamble_amplitude,
        full_symbol_amplitude=args.full_symbol_amplitude,
        checkpoint_preamble_amplitude=args.checkpoint_preamble_amplitude,
        checkpoint_symbol_amplitude=args.checkpoint_symbol_amplitude,
        full_audio_gain=args.full_audio_gain,
        checkpoint_audio_gain=args.checkpoint_audio_gain,
        full_preamble_ms=args.full_preamble_ms,
        full_symbol_ms=args.full_symbol_ms,
        checkpoint_preamble_ms=args.checkpoint_preamble_ms,
        checkpoint_symbol_ms=args.checkpoint_symbol_ms,
        full_detection_threshold=args.full_detection_threshold,
        checkpoint_detection_threshold=args.checkpoint_detection_threshold,
        ciphertext_gain=args.ciphertext_gain,
        node_name=args.node,
        sync_diag_interval=args.sync_diag_interval,
        tx_diag_interval=args.tx_diag_interval,
        initial_full_repeats=args.initial_full_repeats,
        checkpoint_confirmations=args.checkpoint_confirmations,
        checkpoint_max_drift=args.checkpoint_max_drift,
        checkpoint_min_confidence=args.checkpoint_min_confidence,
        checkpoint_large_drift_confirmations=args.checkpoint_large_drift_confirmations,
        checkpoint_large_drift_tolerance=args.checkpoint_large_drift_tolerance,
        checkpoint_large_drift_min_confidence=args.checkpoint_large_drift_min_confidence,
        rx_sync_mute=args.rx_sync_mute,
        rx_sync_lookahead_ms=args.rx_sync_lookahead_ms,
        rx_full_mute_threshold=args.rx_full_mute_threshold,
        rx_checkpoint_mute_threshold=args.rx_checkpoint_mute_threshold,
        rx_sync_mute_fade_ms=args.rx_sync_mute_fade_ms,
        rx_checkpoint_suppress=args.rx_checkpoint_suppress,
        rx_mode=args.rx_mode,
        wrong_key=hashlib.sha256(b"WRONG-RX-TEST:" + key).digest(),
        recorder=recorder,
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
    print("RX MODE             :", processor.get_rx_mode().upper())
    if processor.get_rx_mode() == "raw":
        print("WARNING: RAW passes chirps unchanged! Type 4+ENTER to hear muted RAW, 2+ENTER for CORRECT.")
    print("RX fail-closed      : yes (CORRECT/WRONG silent until FULL)")
    print("WRONG test          : uses deliberately different key")
    if args.operation == "invert":
        print("WARNING: --operation invert is not keyed; WRONG cannot differ from CORRECT.")
    print("Sync period         :", args.sync_period, "s")
    print("Full refresh period :", args.full_sync_period, "s")
    print("FULL preamble amp   :", args.full_preamble_amplitude)
    print("FULL symbol amp     :", args.full_symbol_amplitude)
    print("CP preamble amp     :", args.checkpoint_preamble_amplitude)
    print("CP symbol amp       :", args.checkpoint_symbol_amplitude)
    print("FULL audio gain     :", args.full_audio_gain)
    print("CP audio gain       :", args.checkpoint_audio_gain)
    print("RX sync mute        :", args.rx_sync_mute)
    print("RX CHECKPOINT CUT   :", args.rx_checkpoint_suppress.upper(),
          "(PREAMBLE=chirp only; ALL=entire CP preamble+data+padding, strict digital zero)")
    print("CP preamble duration:", args.checkpoint_preamble_ms, "ms")
    print("CP data duration    :", 6 * args.checkpoint_symbol_ms,
          "ms (muted with ALL; kept with PREAMBLE)")
    print("RX mute lookahead   :", args.rx_sync_lookahead_ms, "ms")
    print("Initial FULL x      :", args.initial_full_repeats)
    print("Checkpoint modulo   : 64")
    print("Checkpoint confirms :", args.checkpoint_confirmations)
    print("Checkpoint max drift:", args.checkpoint_max_drift, "frames")
    print("Large drift confirms:", args.checkpoint_large_drift_confirmations)
    print("Large drift tolerance:", args.checkpoint_large_drift_tolerance, "frames")
    print("Large drift min conf :", args.checkpoint_large_drift_min_confidence)
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
        print("RX LISTENING MODES (enter digit and press ENTER):", flush=True)
        print("  0 = MUTE     : zero RX output; proves whether sound bypasses app", flush=True)
        print("  1 = RAW      : received ciphertext, NO sync-mute (control)", flush=True)
        print("  4 = MASKED_RAW: received ciphertext WITH sync-mute, NO decryption", flush=True)
        print("  2 = CORRECT  : correct key, only AFTER CRC32-verified FULL", flush=True)
        print("  3 = WRONG    : deliberately wrong key, only AFTER FULL", flush=True)
        print("  ENTER = show state   |   q = stop   |   Ctrl+C = stop", flush=True)
        print("ACTIVE RX MODE:", processor.get_rx_mode().upper(), flush=True)

        # Keep UI input out of the real-time TX/RX worker threads. A changed
        # mode takes effect at the next processed RX segment.
        while not stop_event.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.25)
            except (OSError, ValueError):
                # Non-interactive invocation without attached stdin.
                stop_event.wait(0.25)
                continue
            if not ready:
                continue
            line = sys.stdin.readline()
            if line == "":
                # EOF: keep running until Ctrl+C rather than spinning.
                stop_event.wait(0.25)
                continue
            cmd = line.strip().lower()
            if cmd in {"0", "1", "2", "3", "4"}:
                processor.set_rx_mode({"0": "mute", "1": "raw", "2": "correct", "3": "wrong", "4": "masked_raw"}[cmd])
            elif cmd in {"q", "quit", "exit"}:
                print("Stopping by console command...", flush=True)
                stop_event.set()
            elif cmd == "":
                print(
                    "RX STATUS: mode=%s locked=%s frame=%d full_ok=%d cp_ok=%d"
                    % (
                        processor.get_rx_mode().upper(),
                        processor.rx_locked,
                        processor.rx_cipher.frame_index,
                        getattr(processor.sync_decoder, "total_full_ok", -1),
                        getattr(processor.sync_decoder, "total_checkpoint_ok", -1),
                    ), flush=True,
                )
            else:
                print("Unknown command. Use 0 / 1 / 2 / 3 / 4 / ENTER / q", flush=True)
    finally:
        if bridge is not None:
            bridge.stop()
        if recorder is not None:
            recorder.close()
        # User-requested cleanup: all PulseAudio module-loopback instances.
        # Run BEFORE removing our null sinks. Can be disabled if the host uses
        # unrelated loopbacks (e.g. audio monitoring/routing).
        if not args.keep_loopbacks:
            try:
                out = virtual_devices.pactl("list", "short", "modules")
                ids = [line.split()[0] for line in out.splitlines()
                       if len(line.split()) >= 2 and line.split()[1] == "module-loopback"]
                for module_id in ids:
                    try:
                        virtual_devices.pactl("unload-module", module_id)
                        print("LOOPBACK CLEANUP: unloaded module", module_id, flush=True)
                    except (subprocess.CalledProcessError, OSError) as err:
                        print("LOOPBACK CLEANUP: failed", module_id, str(err), flush=True)
                print("LOOPBACK CLEANUP: found=%d" % len(ids), flush=True)
            except (subprocess.CalledProcessError, OSError) as err:
                print("LOOPBACK CLEANUP: unavailable:", str(err), flush=True)
        virtual_devices.stop()


def main() -> None:
    args = parse_args()
    with RunLog(args):
        run(args)


if __name__ == "__main__":
    main()
