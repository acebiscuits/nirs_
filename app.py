#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import hashlib
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
from audio_sync import AudioData
from qim_sync import QIMSyncCodec, QIMSyncStreamDecoder
from crypto.mdct import MDCTCipher
from crypto.stft import STFTCipher
from sync import FrameSynchronizer, SyncInfo


class LiveProcessor:
    """Real-time spectral cipher + teacher-style residue/QIM synchronization."""

    def __init__(
        self,
        tx_cipher,
        rx_cipher,
        sample_rate: int,
        sync_period_seconds: float,
        full_sync_period_seconds: float,
        qim_quantum: int,
        qim_repeat: int,
        qim_detection_threshold: float,
        qim_diag_interval: float,
        tx_diag_interval: float,
        initial_full_repeats: int,
        node_name: str,
    ):
        if initial_full_repeats < 1:
            raise ValueError("initial_full_repeats must be >= 1")
        if tx_cipher.hop_size != rx_cipher.hop_size:
            raise ValueError("TX and RX hop sizes must match")

        self.tx_cipher = tx_cipher
        self.rx_cipher = rx_cipher
        self.sample_rate = int(sample_rate)
        self.hop_size = int(tx_cipher.hop_size)
        self.node_name = node_name

        # IMPORTANT: QIM does not add an independent waveform, so there is no
        # reason to attenuate the whole encrypted speech for headroom.
        self.audio_gain = 1.0

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

        self.sync_codec = QIMSyncCodec(
            sample_rate=sample_rate,
            base=qim_quantum,
            repetition=qim_repeat,
            detection_threshold=qim_detection_threshold,
        )
        self.sync_decoder = QIMSyncStreamDecoder(self.sync_codec)

        self.initial_full_repeats = int(initial_full_repeats)
        self.initial_full_remaining = self.initial_full_repeats - 1

        self.rx_locked = False
        self.last_rx_sequence: int | None = None

        self.tx_warmup_done = False
        self.pending_tx_sync: SyncInfo | None = None
        self.active_marker = np.empty(0, dtype=np.int8)
        self.active_marker_pos = 0

        if qim_diag_interval <= 0:
            raise ValueError("qim_diag_interval must be > 0")
        self.qim_diag_interval = float(qim_diag_interval)
        self._next_qim_diag_at = time.monotonic() + self.qim_diag_interval

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

    def _sequence_diagnostics(self, sequence: int) -> tuple[str, int]:
        if self.last_rx_sequence is None:
            return "FIRST", 0
        delta = (int(sequence) - int(self.last_rx_sequence)) & 0xFF
        if delta == 0:
            return "DUPLICATE", 0
        if delta >= 128:
            return "STALE", 0
        return "OK", max(0, delta - 1)

    def _start_pending_marker_at_boundary(self) -> None:
        if self.pending_tx_sync is None:
            return
        if self.active_marker_pos < self.active_marker.size:
            return

        info = self.pending_tx_sync
        self.pending_tx_sync = None
        self.active_marker = self.sync_codec.encode(info)
        self.active_marker_pos = 0

        kind = "FULL" if info.is_full else "PERIODIC"
        marker_ms = 1000.0 * self.active_marker.size / self.sample_rate
        self._log(
            "TX->REMOTE",
            f"{kind} EMIT",
            seq=info.sequence,
            next=info.next_frame_index,
            local_tx=self.tx_cipher.frame_index,
            transport="qim_residue",
            marker_ms=f"{marker_ms:.3f}",
        )

    def _apply_active_marker(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0 or self.active_marker_pos >= self.active_marker.size:
            return samples

        out = samples.copy()
        pos = 0
        while pos < out.size and self.active_marker_pos < self.active_marker.size:
            count = min(
                out.size - pos,
                self.active_marker.size - self.active_marker_pos,
            )
            symbols = self.active_marker[
                self.active_marker_pos:self.active_marker_pos + count
            ]
            out[pos:pos + count] = self.sync_codec.embed(
                out[pos:pos + count],
                symbols,
            )
            pos += count
            self.active_marker_pos += count
        return out

    def _schedule_next_marker(self) -> None:
        if self.pending_tx_sync is not None:
            return
        if self.active_marker_pos < self.active_marker.size:
            return

        next_index = int(self.tx_cipher.frame_index)

        if (
            self.initial_full_remaining > 0
            and self.tx_sync.should_schedule_periodic(self.tx_cipher, next_index)
        ):
            self.pending_tx_sync = self.tx_sync.make_full_sync(
                self.tx_cipher,
                next_frame_index=next_index,
            )
            self.initial_full_remaining -= 1
            return

        if self.tx_sync.should_schedule_full(self.tx_cipher, next_index):
            self.pending_tx_sync = self.tx_sync.make_full_sync(
                self.tx_cipher,
                next_frame_index=next_index,
            )
        elif self.tx_sync.should_schedule_periodic(self.tx_cipher, next_index):
            self.pending_tx_sync = self.tx_sync.make_periodic_sync(next_index)

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
            self.pending_tx_sync = self.tx_sync.make_full_sync(
                self.tx_cipher,
                next_frame_index=after,
            )
            return b""

        if after - before > 1:
            raise RuntimeError(
                "Audio block produced more than one spectral frame; "
                "BLOCK must not exceed cipher hop size"
            )

        # No 0.2/0.6/0.8 gain here: QIM changes residue positions of the
        # existing samples instead of adding a second waveform.
        self._start_pending_marker_at_boundary()
        encrypted = self._apply_active_marker(encrypted)

        self._accumulate_tx_tunnel(encrypted)
        self._maybe_log_tx_diagnostics()

        if after > before:
            self._schedule_next_marker()

        return self._float_to_bytes(encrypted)

    def on_tx_gap(self, dropped_blocks: int, block_samples: int) -> None:
        dropped_blocks = max(0, int(dropped_blocks))
        block_samples = max(1, int(block_samples))
        if dropped_blocks == 0:
            return

        before = int(self.tx_cipher.frame_index)
        zero = np.zeros(block_samples, dtype=np.float64)
        for _ in range(dropped_blocks):
            self.tx_cipher.encrypt(zero)
        after = int(self.tx_cipher.frame_index)

        self.pending_tx_sync = self.tx_sync.make_full_sync(
            self.tx_cipher,
            next_frame_index=after,
        )
        self.active_marker = np.empty(0, dtype=np.int8)
        self.active_marker_pos = 0

        self._log(
            "TX->REMOTE",
            "REALTIME GAP",
            dropped_blocks=dropped_blocks,
            frame_before=before,
            frame_after=after,
            action="ZERO_ADVANCE_AND_FULL",
        )

    def _handle_sync(self, info: SyncInfo) -> None:
        local_before = int(self.rx_cipher.frame_index)
        kind = "FULL" if info.is_full else "PERIODIC"
        seq_status, missing = self._sequence_diagnostics(info.sequence)

        if self.rx_locked and seq_status in {"DUPLICATE", "STALE"}:
            self._log(
                "RX<-REMOTE",
                f"{kind} IGNORE",
                seq=info.sequence,
                next=info.next_frame_index,
                local_before=local_before,
                seq_status=seq_status,
            )
            return

        if not self.rx_locked:
            if not info.is_full:
                self._log(
                    "RX<-REMOTE",
                    "PERIODIC DETECTED-BUT-UNLOCKED",
                    seq=info.sequence,
                    next=info.next_frame_index,
                    crc="CRC32_OK",
                )
                return
            self.rx_sync.apply_sync(self.rx_cipher, info, hard=True)
            self.rx_locked = True
            action = "HARD_LOCK"
        else:
            self.rx_sync.apply_sync(self.rx_cipher, info, hard=False)
            action = "SOFT_SET"

        local_after = int(self.rx_cipher.frame_index)
        self._log(
            "RX<-REMOTE",
            f"{kind} APPLY",
            seq=info.sequence,
            next=info.next_frame_index,
            crc="CRC32_OK",
            local_before=local_before,
            drift=info.next_frame_index - local_before,
            action=action,
            local_after=local_after,
            missed_seq=missing,
        )
        self.last_rx_sequence = int(info.sequence)

    def _maybe_log_qim_diagnostics(self) -> None:
        now = time.monotonic()
        if now < self._next_qim_diag_at:
            return
        self._next_qim_diag_at = now + self.qim_diag_interval

        stats = self.sync_decoder.diagnostics_snapshot(reset_interval=True)
        self._log(
            "RX<-REMOTE",
            "QIM_DIAG",
            samples=stats["samples"],
            rms_dbfs=f'{stats["rms_dbfs"]:.1f}',
            peak_dbfs=f'{stats["peak_dbfs"]:.1f}',
            best_preamble=f'{stats["best_preamble_score"]:.3f}',
            threshold=f'{stats["threshold"]:.3f}',
            preamble_hits=stats["preamble_hits"],
            crc_fail=stats["crc_fail"],
            decode_fail=stats["decode_fail"],
            periodic_ok=stats["periodic_ok"],
            full_ok=stats["full_ok"],
            total_periodic_ok=stats["total_periodic_ok"],
            total_full_ok=stats["total_full_ok"],
            buffered=stats["buffered_samples"],
            locked=self.rx_locked,
            frame=self.rx_cipher.frame_index,
        )

    def process_rx(self, data: bytes) -> bytes:
        incoming = self._bytes_to_float(data)
        events = self.sync_decoder.feed(incoming)
        self._maybe_log_qim_diagnostics()
        output_parts: list[np.ndarray] = []

        for event in events:
            if isinstance(event, SyncInfo):
                self._handle_sync(event)
                continue

            assert isinstance(event, AudioData)
            if not self.rx_locked:
                # Preserve the previous diagnostic behaviour: before initial
                # lock, remote audio is still audible rather than hard-muted.
                output_parts.append(event.samples.copy())
                continue

            restored = self.rx_cipher.decrypt(event.samples)
            if restored.size:
                output_parts.append(restored)

        if not output_parts:
            return b""
        return self._float_to_bytes(np.concatenate(output_parts))

    def on_rx_gap(self, dropped_blocks: int, block_samples: int) -> None:
        dropped_blocks = max(0, int(dropped_blocks))
        block_samples = max(1, int(block_samples))
        if dropped_blocks == 0:
            return

        # A QIM packet can no longer be decoded if a chunk in its middle was
        # discarded. Throw away only the short parser tail; do not build backlog.
        self.sync_decoder.clear_buffer()

        before = int(self.rx_cipher.frame_index)
        if self.rx_locked:
            zero = np.zeros(block_samples, dtype=np.float64)
            for _ in range(dropped_blocks):
                self.rx_cipher.decrypt(zero)
        after = int(self.rx_cipher.frame_index)

        self._log(
            "RX<-REMOTE",
            "REALTIME GAP",
            dropped_blocks=dropped_blocks,
            frame_before=before,
            frame_after=after,
            locked=self.rx_locked,
            action=("ZERO_ADVANCE" if self.rx_locked else "WAIT_FULL"),
        )


def build_cipher(
    transform: str,
    operation: str,
    key: bytes,
    key_schedule: str,
):
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
    parser.add_argument("--sync-period", type=float, default=0.5)
    parser.add_argument("--full-sync-period", type=float, default=5.0)
    parser.add_argument(
        "--qim-quantum",
        type=int,
        default=64,
        help="QIM period Q; 64 -> residues 0/16/32/48",
    )
    parser.add_argument(
        "--qim-repeat",
        type=int,
        default=4,
        help="QIM repetition factor; complete payload copies are interleaved",
    )
    parser.add_argument(
        "--qim-detection-threshold",
        type=float,
        default=0.75,
        help="minimum QIM preamble score",
    )
    parser.add_argument("--qim-diag-interval", type=float, default=2.0)
    parser.add_argument("--tx-diag-interval", type=float, default=2.0)
    parser.add_argument(
        "--initial-full-repeats",
        type=int,
        default=3,
        help="FULL attempts during acquisition, spaced by --sync-period",
    )
    parser.add_argument(
        "--node",
        default=socket.gethostname(),
        help="short node label, e.g. PC-A or PC-B",
    )
    parser.add_argument("--log-dir", default="logs/qim_residue")
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
            f"{self.args.transform}_{self.args.operation}_qim_residue.log"
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

    tx_cipher = build_cipher(
        args.transform,
        args.operation,
        key,
        args.key_schedule,
    )
    rx_cipher = build_cipher(
        args.transform,
        args.operation,
        key,
        args.key_schedule,
    )

    processor = LiveProcessor(
        tx_cipher=tx_cipher,
        rx_cipher=rx_cipher,
        sample_rate=RATE,
        sync_period_seconds=args.sync_period,
        full_sync_period_seconds=args.full_sync_period,
        qim_quantum=args.qim_quantum,
        qim_repeat=args.qim_repeat,
        qim_detection_threshold=args.qim_detection_threshold,
        qim_diag_interval=args.qim_diag_interval,
        tx_diag_interval=args.tx_diag_interval,
        initial_full_repeats=args.initial_full_repeats,
        node_name=args.node,
    )

    physical_mic, physical_speakers = get_default_audio_devices()

    print("Physical mic        :", physical_mic)
    print("Physical speakers   :", physical_speakers)
    print("Sample rate         :", RATE)
    print("Transform           :", args.transform)
    print("Operation           :", args.operation)
    print("Key schedule        :", args.key_schedule)
    print("Sync                : QIM residues (teacher method)")
    print("QIM formula         : Q*floor(x/Q) + (Q/4)*d")
    print("QIM quantum         :", args.qim_quantum)
    print("QIM residue levels  :", [
        0,
        args.qim_quantum // 4,
        args.qim_quantum // 2,
        3 * args.qim_quantum // 4,
    ])
    print("QIM repetition      :", args.qim_repeat)
    print("QIM detect threshold:", args.qim_detection_threshold)
    print("Ciphertext gain     : 1.0 (NO global attenuation for QIM)")
    print("Sync period         :", args.sync_period, "s")
    print("Full refresh period :", args.full_sync_period, "s")
    print("Initial FULL x      :", args.initial_full_repeats)
    marker_ms = 1000.0 * processor.sync_codec.marker_samples_for() / RATE
    print("QIM marker length   :", f"{marker_ms:.3f} ms")

    devices = PulseVirtualDevices()
    bridge = AudioBridge(
        mic=physical_mic,
        speakers=physical_speakers,
        process_tx=processor.process_tx,
        process_rx=processor.process_rx,
        on_tx_gap=processor.on_tx_gap,
        on_rx_gap=processor.on_rx_gap,
        rate=RATE,
    )

    devices.start()
    print("\nMessenger settings:")
    print("  Microphone = TX_Microphone")
    print("  Speakers   = RX_sink")
    print("Ctrl+C = stop")

    try:
        bridge.start()
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        bridge.stop()
        devices.stop()


def main() -> None:
    args = parse_args()
    with RunLog(args):
        run(args)


if __name__ == "__main__":
    main()
