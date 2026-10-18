#!/usr/bin/env python3

from __future__ import annotations

import argparse
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
from audio_sync import AudioData, AudioSyncCodec, AudioSyncStreamDecoder
from qim_sync import QIMSyncCodec, QIMSyncStreamDecoder
from crypto.mdct import MDCTCipher
from crypto.stft import STFTCipher
from sync import FrameSynchronizer, SyncInfo


class LiveProcessor:
    """Owns TX/RX ciphers and in-band synchronization for one PC."""

    def __init__(
        self,
        tx_cipher,
        rx_cipher,
        sample_rate: int,
        sync_period_seconds: float,
        full_sync_period_seconds: float,
        marker_amplitude: float,
        ciphertext_gain: float,
        sync_transport: str,
        sync_mode: str,
        node_name: str,
        symbol_ms: float,
        preamble_ms: float,
        detection_threshold: float,
        qim_quantum: int,
        qim_repeat: int,
        qim_detection_threshold: float,
        qim_diag_interval: float,
        tx_diag_interval: float,
        initial_full_repeats: int,
    ):
        if not (0.0 < ciphertext_gain <= 1.0):
            raise ValueError("ciphertext_gain must be in (0, 1]")

        self.tx_cipher = tx_cipher
        self.rx_cipher = rx_cipher
        if sync_transport not in {"tone", "qim"}:
            raise ValueError("sync_transport must be 'tone' or 'qim'")
        self.sync_transport = sync_transport

        if sync_mode not in {"overlay", "replace"}:
            raise ValueError("sync_mode must be 'overlay' or 'replace'")
        self.sync_mode = sync_mode
        self.node_name = node_name

        # The tone overlay needs headroom before a waveform is added. QIM does
        # not add an independent waveform, so scaling the whole ciphertext would
        # only add needless quantization loss.
        self.ciphertext_gain = (
            float(ciphertext_gain) if self.sync_transport == "tone" else 1.0
        )

        if initial_full_repeats < 1:
            raise ValueError("initial_full_repeats must be >= 1")
        self.initial_full_repeats = int(initial_full_repeats)
        # One FULL is emitted immediately after warm-up. The remaining FULLs
        # are repeated at normal periodic-sync boundaries, so a lost first
        # FULL does not mute RX until the 5-second refresh.
        self.initial_full_remaining = self.initial_full_repeats - 1

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

        if self.sync_transport == "tone":
            self.sync_codec = AudioSyncCodec(
                sample_rate=sample_rate,
                marker_amplitude=marker_amplitude,
                symbol_ms=symbol_ms,
                preamble_ms=preamble_ms,
                detection_threshold=detection_threshold,
            )
            # Compatibility with both audio_sync.py variants:
            # newer decoder accepts mode=overlay|replace; older one is overlay-only.
            try:
                self.sync_decoder = AudioSyncStreamDecoder(
                    self.sync_codec,
                    mode=self.sync_mode,
                )
            except TypeError:
                if self.sync_mode != "overlay":
                    raise RuntimeError(
                        "This audio_sync.py supports only overlay mode; "
                        "use --sync-mode overlay or update audio_sync.py"
                    )
                self.sync_decoder = AudioSyncStreamDecoder(self.sync_codec)
        else:
            self.sync_codec = QIMSyncCodec(
                sample_rate=sample_rate,
                base=qim_quantum,
                repetition=qim_repeat,
                detection_threshold=qim_detection_threshold,
            )
            self.sync_decoder = QIMSyncStreamDecoder(self.sync_codec)

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

        self.rx_locked = False
        self.last_rx_sequence: int | None = None

        # The first transform output hop is a boundary/warm-up hop.
        # For STFT it can be badly amplified by edge normalization after a
        # spectral modification; for MDCT it lacks the previous TDAC neighbour.
        # We drop exactly that one hop, then start FULL_SYNC on the next hop.
        self.tx_warmup_done = False
        self.pending_tx_sync: SyncInfo | None = None
        self.active_marker = np.empty(0, dtype=np.float64)
        self.active_marker_pos = 0


    def _log(self, direction: str, event: str, **fields) -> None:
        ts = datetime.now().astimezone().strftime("%H:%M:%S.%f")[:-3]
        tail = " ".join(f"{key}={value}" for key, value in fields.items())
        print(f"{ts} [{self.node_name}] {direction} {event}" + (f" {tail}" if tail else ""), flush=True)

    def _sequence_diagnostics(self, sequence: int) -> tuple[str, int]:
        if self.last_rx_sequence is None:
            return "FIRST", 0

        delta = (sequence - self.last_rx_sequence) & 0xFF
        if delta == 0:
            return "DUPLICATE", 0
        if delta >= 128:
            return "STALE", 0
        return "OK", max(0, delta - 1)

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

    def _start_pending_marker_at_boundary(self) -> None:
        if self.pending_tx_sync is None or self.active_marker_pos < self.active_marker.size:
            return

        self.active_marker = self.sync_codec.encode(self.pending_tx_sync)
        self.active_marker_pos = 0

        kind = "FULL" if self.pending_tx_sync.is_full else "PERIODIC"
        marker_ms = 1000.0 * self.active_marker.size / self.sync_codec.sample_rate
        self._log(
            "TX->REMOTE",
            f"{kind} EMIT",
            seq=self.pending_tx_sync.sequence,
            next=self.pending_tx_sync.next_frame_index,
            local_tx=self.tx_cipher.frame_index,
            transport=self.sync_transport,
            mode=(self.sync_mode if self.sync_transport == "tone" else "qim"),
            marker_ms=f"{marker_ms:.3f}",
        )
        self.pending_tx_sync = None

    def _apply_active_marker(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return samples

        out = samples.copy()
        pos = 0

        while pos < out.size and self.active_marker_pos < self.active_marker.size:
            count = min(
                out.size - pos,
                self.active_marker.size - self.active_marker_pos,
            )
            marker_piece = self.active_marker[
                self.active_marker_pos:self.active_marker_pos + count
            ]

            if self.sync_transport == "qim":
                out[pos:pos + count] = self.sync_codec.embed(
                    out[pos:pos + count],
                    marker_piece,
                )
            elif self.sync_mode == "overlay":
                out[pos:pos + count] += marker_piece
            else:
                out[pos:pos + count] = marker_piece

            pos += count
            self.active_marker_pos += count

        return out

    def _schedule_next_marker(self) -> None:
        # Do not queue another marker while one is still active/pending. If one
        # periodic packet is skipped, the following one will correct the state.
        if self.pending_tx_sync is not None:
            return
        if self.active_marker_pos < self.active_marker.size:
            return

        next_index = self.tx_cipher.frame_index

        # During acquisition, repeat FULL_SYNC on the short periodic cadence.
        # This fixes the old behaviour where losing the first FULL meant that
        # RX stayed muted until the next 5-second FULL refresh.
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
        before = self.tx_cipher.frame_index
        encrypted = self.tx_cipher.encrypt(pcm)
        after = self.tx_cipher.frame_index

        if encrypted.size == 0:
            self._maybe_log_tx_diagnostics()
            return b""

        if not self.tx_warmup_done:
            self.tx_warmup_done = True
            # The next returned hop starts spectral frame `after`.
            self.pending_tx_sync = self.tx_sync.make_full_sync(
                self.tx_cipher,
                next_frame_index=after,
            )
            return b""

        # BLOCK=1024 and the current hop sizes guarantee at most one new
        # spectral frame per call. Keep this explicit so a future config change
        # does not silently break marker-boundary scheduling.
        if after - before > 1:
            raise RuntimeError(
                "Audio block produced more than one spectral frame; "
                "reduce BLOCK or add per-frame TX callbacks"
            )

        # Tone overlay uses headroom. QIM keeps gain=1.0 and changes only the
        # selected low-level PCM residue positions.
        encrypted = encrypted * self.ciphertext_gain

        # A pending marker always starts at the beginning of this output hop.
        self._start_pending_marker_at_boundary()
        encrypted = self._apply_active_marker(encrypted)
        self._accumulate_tx_tunnel(encrypted)
        self._maybe_log_tx_diagnostics()

        # If this call completed frame N-1, any newly scheduled marker starts
        # on the NEXT output hop and therefore carries next_frame_index=N.
        if after > before:
            self._schedule_next_marker()

        return self._float_to_bytes(encrypted)

    def _handle_sync(self, info: SyncInfo) -> None:
        local_before = self.rx_cipher.frame_index
        kind = "FULL" if info.is_full else "PERIODIC"
        seq_status, missing = self._sequence_diagnostics(info.sequence)

        # Audio order cannot legitimately go backwards. Do not let a duplicate or
        # stale packet rewind the frame counter. A new-session FULL is handled
        # below regardless of sequence because its nonce changes the epoch.
        is_new_session = (
            info.is_full
            and self.rx_locked
            and info.session_nonce != self.rx_cipher.session_nonce
        )

        if self.rx_locked and not is_new_session and seq_status in {"DUPLICATE", "STALE"}:
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
                    crc="OK",
                )
                return

            self.rx_sync.apply_sync(self.rx_cipher, info, hard=True)
            self.rx_locked = True
            action = "HARD_LOCK"

        elif is_new_session:
            self.rx_sync.apply_sync(self.rx_cipher, info, hard=True)
            action = "HARD_NEW_SESSION"

        else:
            self.rx_sync.apply_sync(self.rx_cipher, info, hard=False)
            action = "SOFT_SET"

        local_after = self.rx_cipher.frame_index
        drift = info.next_frame_index - local_before

        self._log(
            "RX<-REMOTE",
            f"{kind} APPLY",
            seq=info.sequence,
            next=info.next_frame_index,
            crc="OK",
            local_before=local_before,
            drift=drift,
            action=action,
            local_after=local_after,
            missed_seq=missing,
        )
        self.last_rx_sequence = info.sequence

    def _maybe_log_qim_diagnostics(self) -> None:
        if self.sync_transport != "qim":
            return

        now = time.monotonic()
        if now < self._next_qim_diag_at:
            return
        self._next_qim_diag_at = now + self.qim_diag_interval

        stats = self.sync_decoder.diagnostics_snapshot(reset_interval=True)
        bad_header = stats["last_bad_header"]
        bad_header_text = "none" if bad_header is None else f"0x{bad_header:02X}"

        self._log(
            "RX<-REMOTE",
            "QIM_DIAG",
            samples=stats["samples"],
            rms_dbfs=f'{stats["rms_dbfs"]:.1f}',
            peak_dbfs=f'{stats["peak_dbfs"]:.1f}',
            best_preamble=f'{stats["best_preamble_score"]:.3f}',
            threshold=f'{stats["threshold"]:.3f}',
            preamble_hits=stats["preamble_hits"],
            bad_header=stats["bad_header"],
            last_bad_header=bad_header_text,
            crc_fail=stats["crc_fail"],
            decode_fail=stats["decode_fail"],
            periodic_ok=stats["periodic_ok"],
            full_ok=stats["full_ok"],
            total_ok=stats["total_periodic_ok"] + stats["total_full_ok"],
            buffered=stats["buffered_samples"],
            locked=self.rx_locked,
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
                # Do not attempt per-frame decryption before the remote nonce and
                # frame-0 boundary have been established.
                continue

            ciphertext = event.samples / self.ciphertext_gain
            restored = self.rx_cipher.decrypt(ciphertext)
            if restored.size:
                output_parts.append(restored)

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
    parser.add_argument("--full-sync-period", type=float, default=5.0)
    parser.add_argument(
        "--sync-transport",
        choices=["tone", "qim"],
        default="tone",
        help="tone=current chirp/two-tone modem; qim=teacher-style modulo quantization",
    )
    parser.add_argument("--marker-amplitude", type=float, default=0.30)
    parser.add_argument("--symbol-ms", type=float, default=4.0)
    parser.add_argument("--preamble-ms", type=float, default=24.0)
    parser.add_argument("--detection-threshold", type=float, default=0.52)
    parser.add_argument(
        "--qim-quantum",
        type=int,
        default=64,
        help="QIM quantization period; 64 gives residues 0/16/32/48",
    )
    parser.add_argument(
        "--qim-repeat",
        type=int,
        default=1,
        help="repeat each base-4 QIM digit across N consecutive samples; 1=literal scheme",
    )
    parser.add_argument(
        "--qim-detection-threshold",
        type=float,
        default=0.80,
        help="minimum 0..1 preamble residue-pattern score for QIM",
    )
    parser.add_argument(
        "--qim-diag-interval",
        type=float,
        default=2.0,
        help="seconds between QIM RX diagnostic summary lines",
    )
    parser.add_argument(
        "--tx-diag-interval",
        type=float,
        default=2.0,
        help="seconds between physical-mic / encrypted-tunnel TX level diagnostics",
    )
    parser.add_argument(
        "--initial-full-repeats",
        type=int,
        default=3,
        help="FULL_SYNC attempts during initial acquisition, spaced by --sync-period",
    )
    parser.add_argument("--ciphertext-gain", type=float, default=0.60)
    parser.add_argument(
        "--sync-mode",
        choices=["overlay", "replace"],
        default="overlay",
        help="overlay adds the marker to ciphertext; replace overwrites that interval",
    )
    parser.add_argument(
        "--node",
        default=socket.gethostname(),
        help="short node label used in logs, e.g. PC-A or PC-B",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="directory for per-run log files",
    )
    return parser.parse_args()


class _TeeStream:
    """Mirror stdout/stderr to terminal and one line-buffered run log."""

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
            f"{self.args.transform}_{self.args.operation}_"
            f"{self.args.sync_transport}_{self.args.sync_mode}.log"
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
            print(f"  {key:20s} = {value}")
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

    # Each direction has its own streaming state. TX gets a fresh random nonce;
    # RX starts with a placeholder and learns the remote TX nonce from FULL_SYNC.
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
        sync_transport=args.sync_transport,
        sync_mode=args.sync_mode,
        node_name=args.node,
        symbol_ms=args.symbol_ms,
        preamble_ms=args.preamble_ms,
        detection_threshold=args.detection_threshold,
        qim_quantum=args.qim_quantum,
        qim_repeat=args.qim_repeat,
        qim_detection_threshold=args.qim_detection_threshold,
        qim_diag_interval=args.qim_diag_interval,
        tx_diag_interval=args.tx_diag_interval,
        initial_full_repeats=args.initial_full_repeats,
    )

    # IMPORTANT: remove stale tunnel endpoints before reading physical defaults.
    # If a previous run was restarted while the browser call stayed alive, those
    # stale endpoints may still own Pulse defaults / browser streams.
    virtual_devices = PulseVirtualDevices()
    virtual_devices.cleanup_stale()
    mic, speakers = get_default_audio_devices()

    print("Physical mic     :", mic)
    print("Physical speakers:", speakers)
    print("Transform        :", args.transform)
    print("Operation        :", args.operation)
    print("Key schedule     :", args.key_schedule)
    print("Sync transport   :", args.sync_transport)
    print("Sync mode        :", args.sync_mode if args.sync_transport == "tone" else "qim")
    print("Node label       :", args.node)
    print("Sync period      :", args.sync_period, "s")
    print("Full sync period :", args.full_sync_period, "s")
    print("Initial FULL x   :", args.initial_full_repeats)
    print("Marker amplitude :", args.marker_amplitude)
    print("Preamble         :", args.preamble_ms, "ms")
    print("Symbol           :", args.symbol_ms, "ms")
    print("Detect threshold :", args.detection_threshold)
    print("QIM quantum      :", args.qim_quantum)
    print("QIM repetition   :", args.qim_repeat)
    print("QIM detect thr   :", args.qim_detection_threshold)
    print("QIM diag interval:", args.qim_diag_interval, "s")
    print("TX diag interval :", args.tx_diag_interval, "s")
    print("Periodic marker  :", f"{1000.0 * processor.sync_codec.marker_samples_for(SyncInfo(0, 0, None)) / RATE:.1f} ms")
    print("FULL marker      :", f"{1000.0 * processor.sync_codec.marker_samples_for(SyncInfo(0, 0, tx_nonce)) / RATE:.1f} ms")

    bridge = None

    try:
        # Only create the virtual endpoints. We deliberately do NOT change
        # PulseAudio defaults and do NOT move/search browser streams.
        # Select TX_Microphone / RX_sink manually in the call UI.
        virtual_devices.start(cleanup=False)
        bridge = AudioBridge(
            mic=mic,
            speakers=speakers,
            process_tx=processor.process_tx,
            process_rx=processor.process_rx,
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
