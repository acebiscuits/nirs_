#!/usr/bin/env python3
from __future__ import annotations

import signal
import subprocess
import threading
import queue
import time
from collections.abc import Callable


RATE = 48000
CHANNELS = 1
BLOCK = 1024

TX_SINK = "tx_enc_sink"
TX_SOURCE = "tx_mic"
RX_SINK = "rx_sink"
RX_SOURCE = f"{RX_SINK}.monitor"

stop_event = threading.Event()


def signal_handler(sig, frame):
    stop_event.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def _run_pactl(*args: str) -> str:
    result = subprocess.run(
        ["pactl", *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _short_name_map(kind: str) -> dict[int, str]:
    """Return PulseAudio object index -> object name for short list output."""
    out = _run_pactl("list", "short", kind)
    result: dict[int, str] = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            result[int(parts[0])] = parts[1]
        except ValueError:
            continue
    return result



class PulseVirtualDevices:
    """Create/remove only the virtual PulseAudio endpoints.

    This class intentionally does not change system defaults and does not
    inspect or move application/browser streams. Device selection stays manual.
    """

    def __init__(self):
        self.module_ids: list[int] = []

    @staticmethod
    def pactl(*args: str) -> str:
        return _run_pactl(*args)

    def load_module(self, module_name: str, *args: str) -> int:
        module_id = int(self.pactl("load-module", module_name, *args))
        self.module_ids.append(module_id)
        return module_id

    def cleanup_stale(self) -> None:
        output = self.pactl("list", "short", "modules")
        modules: list[tuple[int, int]] = []

        for line in output.splitlines():
            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue

            module_id = int(parts[0])
            args = parts[2]

            # remap-source must go before its master sink
            if f"source_name={TX_SOURCE}" in args:
                modules.append((0, module_id))
            elif f"sink_name={TX_SINK}" in args:
                modules.append((1, module_id))
            elif f"sink_name={RX_SINK}" in args:
                modules.append((1, module_id))

        for _, module_id in sorted(modules):
            try:
                self.pactl("unload-module", str(module_id))
            except subprocess.CalledProcessError:
                pass

    def start(self, cleanup: bool = True) -> None:
        if cleanup:
            self.cleanup_stale()

        try:
            self.load_module(
                "module-null-sink",
                f"sink_name={TX_SINK}",
                f"rate={RATE}",
                "sink_properties=device.description=TX_sink",
            )
            self.load_module(
                "module-null-sink",
                f"sink_name={RX_SINK}",
                f"rate={RATE}",
                "sink_properties=device.description=RX_sink",
            )
            self.load_module(
                "module-remap-source",
                f"master={TX_SINK}.monitor",
                f"source_name={TX_SOURCE}",
                "source_properties=device.description=TX_Microphone",
            )

            print("Virtual devices created:")
            print("  input : TX_Microphone")
            print("  output: RX_sink")
            print("Pulse defaults unchanged; select these devices manually in the call UI")
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        for module_id in reversed(self.module_ids):
            try:
                self.pactl("unload-module", str(module_id))
            except subprocess.CalledProcessError:
                pass

        self.module_ids.clear()
        print("Virtual devices removed")


def get_default_audio_devices() -> tuple[str, str]:
    """Return physical source/sink, explicitly excluding our tunnel endpoints."""
    result = subprocess.run(
        ["pactl", "info"],
        check=True,
        text=True,
        capture_output=True,
    )

    default_source = None
    default_sink = None

    for line in result.stdout.splitlines():
        if line.startswith("Default Source:"):
            default_source = line.split(":", 1)[1].strip()
        elif line.startswith("Default Sink:"):
            default_sink = line.split(":", 1)[1].strip()

    virtual_sources = {TX_SOURCE, RX_SOURCE, f"{TX_SINK}.monitor"}
    virtual_sinks = {TX_SINK, RX_SINK}

    if not default_source or default_source in virtual_sources:
        sources = _short_name_map("sources")
        candidates = [
            name
            for name in sources.values()
            if name not in virtual_sources
            and not name.endswith(".monitor")
            and name.startswith("alsa_input")
        ]
        if not candidates:
            candidates = [
                name
                for name in sources.values()
                if name not in virtual_sources and not name.endswith(".monitor")
            ]
        default_source = candidates[0] if candidates else None

    if not default_sink or default_sink in virtual_sinks:
        sinks = _short_name_map("sinks")
        candidates = [
            name
            for name in sinks.values()
            if name not in virtual_sinks and name.startswith("alsa_output")
        ]
        if not candidates:
            candidates = [name for name in sinks.values() if name not in virtual_sinks]
        default_sink = candidates[0] if candidates else None

    if not default_source:
        raise RuntimeError("Could not determine physical PulseAudio source")
    if not default_sink:
        raise RuntimeError("Could not determine physical PulseAudio sink")

    return default_source, default_sink


class AudioBridge:
    """Moves PCM between PulseAudio and user-provided TX/RX processors."""

    def __init__(
        self,
        mic: str,
        speakers: str,
        process_tx: Callable[[bytes], bytes],
        process_rx: Callable[[bytes], bytes],
        on_tx_gap: Callable[[int, int], None] | None = None,
        on_rx_gap: Callable[[int, int], None] | None = None,
        rate: int = RATE,
        channels: int = CHANNELS,
        block: int = BLOCK,
        queue_blocks: int = 4,
    ):
        self.mic = mic
        self.speakers = speakers
        self.process_tx = process_tx
        self.process_rx = process_rx
        self.on_tx_gap = on_tx_gap
        self.on_rx_gap = on_rx_gap
        self.rate = int(rate)
        self.channels = int(channels)
        self.block = int(block)
        self.block_bytes = self.block * self.channels * 2
        self.queue_blocks = max(2, int(queue_blocks))

        self.processes: list[subprocess.Popen] = []
        self.processes_lock = threading.Lock()
        self.threads: list[threading.Thread] = []

    def _register_process(self, process: subprocess.Popen) -> None:
        with self.processes_lock:
            self.processes.append(process)

    def _unregister_process(self, process: subprocess.Popen) -> None:
        with self.processes_lock:
            if process in self.processes:
                self.processes.remove(process)

    @staticmethod
    def _read_exact(stream, size: int) -> bytes:
        data = bytearray()
        while not stop_event.is_set() and len(data) < size:
            chunk = stream.read(size - len(data))
            if not chunk:
                return b""
            data.extend(chunk)

        if len(data) != size:
            return b""
        return bytes(data)

    def _pipe(
        self,
        source: str,
        sink: str,
        name: str,
        processor: Callable[[bytes], bytes],
        on_gap: Callable[[int, int], None] | None,
    ) -> None:
        reader = subprocess.Popen(
            [
                "parec",
                "--device=" + source,
                "--format=s16le",
                "--rate=" + str(self.rate),
                "--channels=" + str(self.channels),
                "--latency-msec=20",
                "--process-time-msec=10",
                "--raw",
            ],
            stdout=subprocess.PIPE,
            bufsize=0,
        )

        writer = subprocess.Popen(
            [
                "pacat",
                "--playback",
                "--device=" + sink,
                "--format=s16le",
                "--rate=" + str(self.rate),
                "--channels=" + str(self.channels),
                "--latency-msec=20",
                "--process-time-msec=10",
                "--raw",
            ],
            stdin=subprocess.PIPE,
            bufsize=0,
        )

        self._register_process(reader)
        self._register_process(writer)
        print(name, "started")

        # Capture is deliberately separated from processing. If a processor ever
        # stalls, parec is still drained continuously and the OLDEST queued audio
        # is dropped. This prevents the old 3-10 second backlog failure mode.
        audio_queue: queue.Queue[tuple[int, bytes]] = queue.Queue(maxsize=self.queue_blocks)
        capture_done = threading.Event()
        dropped_blocks = 0
        dropped_lock = threading.Lock()

        def capture_loop() -> None:
            nonlocal dropped_blocks
            seq = 0
            try:
                assert reader.stdout is not None
                while not stop_event.is_set():
                    data = self._read_exact(reader.stdout, self.block_bytes)
                    if not data:
                        break

                    item = (seq, data)
                    seq += 1
                    try:
                        audio_queue.put_nowait(item)
                    except queue.Full:
                        # Real-time policy: throw away the OLDEST pending block,
                        # never keep seconds of stale speech. Sequence numbers let
                        # the processor notice the exact gap later and advance its
                        # cipher state over synthetic zero audio.
                        try:
                            audio_queue.get_nowait()
                        except queue.Empty:
                            pass
                        with dropped_lock:
                            dropped_blocks += 1
                        try:
                            audio_queue.put_nowait(item)
                        except queue.Full:
                            pass
            finally:
                capture_done.set()

        capture_thread = threading.Thread(
            target=capture_loop,
            name=f"{name}-capture",
            daemon=True,
        )
        capture_thread.start()

        last_drop_report = time.monotonic()
        last_reported_drops = 0

        try:
            assert writer.stdin is not None
            expected_seq = 0

            while not stop_event.is_set():
                try:
                    seq, data = audio_queue.get(timeout=0.10)
                except queue.Empty:
                    if capture_done.is_set():
                        break
                    continue

                if seq > expected_seq and on_gap is not None:
                    gap_blocks = int(seq - expected_seq)
                    on_gap(gap_blocks, self.block)
                expected_seq = int(seq + 1)

                processed = processor(data)
                if processed:
                    writer.stdin.write(processed)

                now = time.monotonic()
                if now - last_drop_report >= 2.0:
                    with dropped_lock:
                        current_drops = dropped_blocks
                    if current_drops != last_reported_drops:
                        print(
                            f"{name} REALTIME_DROP "
                            f"dropped_blocks={current_drops} "
                            f"queue_max={self.queue_blocks}",
                            flush=True,
                        )
                        last_reported_drops = current_drops
                    last_drop_report = now

        except BrokenPipeError:
            pass
        except Exception as exc:
            print(name, "error:", exc)
            stop_event.set()
        finally:
            capture_done.set()
            capture_thread.join(timeout=0.5)

            with dropped_lock:
                final_drops = dropped_blocks
            if final_drops:
                print(
                    f"{name} REALTIME_DROP_TOTAL dropped_blocks={final_drops}",
                    flush=True,
                )

            for process in (reader, writer):
                if process.poll() is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                self._unregister_process(process)
            print(name, "stopped")

    def start(self) -> None:
        tx_thread = threading.Thread(
            target=self._pipe,
            args=(self.mic, TX_SINK, "TX", self.process_tx, self.on_tx_gap),
            daemon=True,
        )
        rx_thread = threading.Thread(
            target=self._pipe,
            args=(RX_SOURCE, self.speakers, "RX", self.process_rx, self.on_rx_gap),
            daemon=True,
        )

        self.threads = [tx_thread, rx_thread]
        for thread in self.threads:
            thread.start()

    def stop(self) -> None:
        stop_event.set()

        with self.processes_lock:
            processes = list(self.processes)

        # terminate parec/pacat to unblock read()
        for process in processes:
            if process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass

        for thread in self.threads:
            thread.join(timeout=2.0)
