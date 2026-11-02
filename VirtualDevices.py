#!/usr/bin/env python3
from __future__ import annotations

import signal
import subprocess
import threading


RATE = 48000

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

