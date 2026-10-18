#!/usr/bin/env python3
"""Analyze synchronization experiment logs and optional WAV captures.

Useful diploma metrics:
- TX sync markers emitted (FULL/PERIODIC)
- RX valid sync packets applied
- inferred lost sync packets from sequence gaps
- approximate reception ratio
- frame-index drift before correction
- initial FULL lock time
- two-log end-to-end matching: A TX -> B RX and B TX -> A RX
- optional WAV RMS/peak/crest factor/spectral centroid/band-energy/spectrum

Examples:
  python3 sync_stats.py logs/PC-A.log
  python3 sync_stats.py logs/PC-A.log logs/PC-B.log --out stats_run1
  python3 sync_stats.py logs/PC-A.log logs/PC-B.log --wav capture.wav --out stats_run1
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import re
import wave

import numpy as np


EVENT_RE = re.compile(
    r"^(?P<time>\d{2}:\d{2}:\d{2}\.\d{3}) "
    r"\[(?P<node>[^\]]+)\] "
    r"(?P<direction>TX->REMOTE|RX<-REMOTE) "
    r"(?P<kind>FULL|PERIODIC) "
    r"(?P<action>EMIT|APPLY|IGNORE|DETECTED-BUT-UNLOCKED)"
    r"(?: (?P<fields>.*))?$"
)
FIELD_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^ ]+)")


def _number(value: str):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    params = {}
    run_start = None
    in_params = False
    events = []

    for line_no, line in enumerate(lines, 1):
        if line.startswith("RUN START") and ":" in line:
            raw = line.split(":", 1)[1].strip()
            try:
                run_start = datetime.fromisoformat(raw)
            except ValueError:
                pass

        if line.strip() == "PARAMETERS:":
            in_params = True
            continue
        if in_params:
            if line.startswith("="):
                in_params = False
            else:
                m = re.match(r"\s{2}([^=]+?)\s*=\s*(.*)$", line)
                if m:
                    params[m.group(1).strip()] = m.group(2).strip()
            continue

        m = EVENT_RE.match(line)
        if not m:
            continue
        fields = {fm.group("key"): _number(fm.group("value")) for fm in FIELD_RE.finditer(m.group("fields") or "")}
        if "action" in fields:
            fields["sync_action"] = fields.pop("action")
        event = {
            "source_log": str(path),
            "line": line_no,
            "time": m.group("time"),
            "node": m.group("node"),
            "direction": m.group("direction"),
            "kind": m.group("kind"),
            "action": m.group("action"),
            **fields,
        }
        events.append(event)

    # Convert time-of-day to elapsed seconds from RUN START when possible.
    if run_start is not None:
        prev_dt = run_start
        for event in events:
            t = datetime.strptime(event["time"], "%H:%M:%S.%f").time()
            event_dt = datetime.combine(prev_dt.date(), t, tzinfo=run_start.tzinfo)
            if event_dt < prev_dt - timedelta(hours=12):
                event_dt += timedelta(days=1)
            event["elapsed_s"] = (event_dt - run_start).total_seconds()
            prev_dt = event_dt

    return {
        "path": str(path),
        "params": params,
        "run_start": run_start.isoformat() if run_start else None,
        "events": events,
    }


def _percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summarize_log(parsed: dict) -> dict:
    events = parsed["events"]
    tx = [e for e in events if e["direction"] == "TX->REMOTE" and e["action"] == "EMIT"]
    rx_apply = [e for e in events if e["direction"] == "RX<-REMOTE" and e["action"] == "APPLY"]
    rx_ignore = [e for e in events if e["direction"] == "RX<-REMOTE" and e["action"] == "IGNORE"]
    rx_unlocked = [e for e in events if e["direction"] == "RX<-REMOTE" and e["action"] == "DETECTED-BUT-UNLOCKED"]

    inferred_missing = sum(int(e.get("missed_seq", 0)) for e in rx_apply)
    received_unique = len(rx_apply)
    denominator = received_unique + inferred_missing
    receive_ratio = received_unique / denominator if denominator else None

    drift = [int(e["drift"]) for e in rx_apply if isinstance(e.get("drift"), int)]
    abs_drift = [abs(x) for x in drift]
    first_lock = next((e for e in rx_apply if e.get("sync_action") in {"HARD_LOCK", "HARD_NEW_SESSION"}), None)

    return {
        "log": parsed["path"],
        "node": next((e["node"] for e in events), parsed["params"].get("node")),
        "parameters": parsed["params"],
        "tx_emitted_total": len(tx),
        "tx_emitted_full": sum(e["kind"] == "FULL" for e in tx),
        "tx_emitted_periodic": sum(e["kind"] == "PERIODIC" for e in tx),
        "rx_applied_total": len(rx_apply),
        "rx_applied_full": sum(e["kind"] == "FULL" for e in rx_apply),
        "rx_applied_periodic": sum(e["kind"] == "PERIODIC" for e in rx_apply),
        "rx_ignored": len(rx_ignore),
        "rx_periodic_while_unlocked": len(rx_unlocked),
        "rx_inferred_missing_from_seq_gaps": inferred_missing,
        "rx_approx_receive_ratio": receive_ratio,
        "first_rx_full_lock_elapsed_s": first_lock.get("elapsed_s") if first_lock else None,
        "drift_frames": {
            "count": len(drift),
            "mean": float(np.mean(drift)) if drift else None,
            "median": float(np.median(drift)) if drift else None,
            "mean_abs": float(np.mean(abs_drift)) if drift else None,
            "p95_abs": _percentile(abs_drift, 95),
            "max_abs": max(abs_drift) if abs_drift else None,
            "zero_fraction": (sum(x == 0 for x in drift) / len(drift)) if drift else None,
        },
    }


def _packet_key(event: dict):
    seq = event.get("seq")
    nxt = event.get("next")
    if not isinstance(seq, int) or not isinstance(nxt, int):
        return None
    return event["kind"], seq, nxt


def cross_node_summary(a: dict, b: dict) -> list[dict]:
    out = []
    for src, dst in ((a, b), (b, a)):
        src_tx = [e for e in src["events"] if e["direction"] == "TX->REMOTE" and e["action"] == "EMIT"]
        dst_rx = [e for e in dst["events"] if e["direction"] == "RX<-REMOTE" and e["action"] == "APPLY"]
        rx_keys = {_packet_key(e) for e in dst_rx}
        rx_keys.discard(None)
        matched = [e for e in src_tx if _packet_key(e) in rx_keys]
        src_node = next((e["node"] for e in src["events"]), Path(src["path"]).stem)
        dst_node = next((e["node"] for e in dst["events"]), Path(dst["path"]).stem)
        out.append({
            "direction": f"{src_node} TX -> {dst_node} RX",
            "tx_emitted": len(src_tx),
            "rx_matched": len(matched),
            "unmatched": len(src_tx) - len(matched),
            "delivery_ratio": len(matched) / len(src_tx) if src_tx else None,
            "matched_full": sum(e["kind"] == "FULL" for e in matched),
            "matched_periodic": sum(e["kind"] == "PERIODIC" for e in matched),
        })
    return out


def write_events_csv(parsed_logs: list[dict], path: Path) -> None:
    events = [e for parsed in parsed_logs for e in parsed["events"]]
    keys = [
        "source_log", "line", "time", "elapsed_s", "node", "direction", "kind", "action",
        "seq", "next", "local_tx", "local_before", "local_after", "drift", "missed_seq",
        "seq_status", "crc", "sync_action", "transport", "mode", "marker_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for e in events:
            w.writerow(e)


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
    x = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return rate, x


def analyze_wav(path: Path, out_dir: Path) -> dict:
    rate, x = read_wav(path)
    if x.size == 0:
        raise ValueError(f"{path}: empty WAV")

    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
    peak_dbfs = 20.0 * math.log10(max(peak, 1e-12))
    crest_db = 20.0 * math.log10(max(peak / max(rms, 1e-12), 1e-12))

    nfft = 8192
    hop = nfft // 2
    if x.size < nfft:
        padded = np.zeros(nfft)
        padded[:x.size] = x
        chunks = [padded]
    else:
        chunks = [x[i:i+nfft] for i in range(0, x.size - nfft + 1, hop)]
    win = np.hanning(nfft)
    power = np.zeros(nfft // 2 + 1)
    for chunk in chunks:
        spec = np.fft.rfft(chunk * win)
        power += np.abs(spec) ** 2
    power /= max(1, len(chunks))
    freq = np.fft.rfftfreq(nfft, 1.0 / rate)
    power_sum = float(np.sum(power)) + 1e-30
    centroid = float(np.sum(freq * power) / power_sum)
    db = 10.0 * np.log10(power / np.max(power) + 1e-20)

    bands = [(0,1000),(1000,3000),(3000,6500),(6500,rate/2)]
    band_energy = {}
    for lo, hi in bands:
        mask = (freq >= lo) & (freq < hi)
        band_energy[f"{lo:g}-{hi:g}_Hz_pct"] = float(100.0 * np.sum(power[mask]) / power_sum)

    stem = path.stem
    spectrum_csv = out_dir / f"{stem}_spectrum.csv"
    with spectrum_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frequency_hz", "relative_power_db"])
        w.writerows(zip(freq, db))

    png = None
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(freq, db)
        ax.set_xlabel("Frequency, Hz")
        ax.set_ylabel("Relative power, dB")
        ax.set_title(f"Spectrum: {path.name}")
        ax.set_xlim(0, rate / 2)
        ax.set_ylim(-100, 5)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        png = out_dir / f"{stem}_spectrum.png"
        fig.savefig(png, dpi=160)
        plt.close(fig)
    except Exception:
        png = None

    return {
        "wav": str(path),
        "sample_rate": rate,
        "samples": int(x.size),
        "duration_s": float(x.size / rate),
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": peak_dbfs,
        "crest_factor_db": crest_db,
        "spectral_centroid_hz": centroid,
        "band_energy_percent": band_energy,
        "spectrum_csv": str(spectrum_csv),
        "spectrum_png": str(png) if png else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="one or two app.py run logs")
    ap.add_argument("--wav", action="append", default=[], help="optional 16-bit PCM WAV capture; may be repeated")
    ap.add_argument("--out", default="sync_stats_out", help="output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed = [parse_log(Path(p)) for p in args.logs]
    summaries = [summarize_log(p) for p in parsed]
    cross = cross_node_summary(parsed[0], parsed[1]) if len(parsed) >= 2 else []
    wav_stats = [analyze_wav(Path(w), out_dir) for w in args.wav]

    result = {
        "logs": summaries,
        "cross_node": cross,
        "wav": wav_stats,
    }

    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_events_csv(parsed, out_dir / "events.csv")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nWritten: {out_dir / 'summary.json'}")
    print(f"Written: {out_dir / 'events.csv'}")
    if args.wav:
        print("Spectrum CSV/PNG files are in:", out_dir)


if __name__ == "__main__":
    main()
