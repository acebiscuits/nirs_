
#!/usr/bin/env python3
"""Inspect five captured WAV stems and verify CP suppression against events.csv.

Usage: python3 inspect_audio.py logs/local_audio/audio/CAPTURE_NAME
Python 3.8+, no external dependencies.
"""
import argparse
import array
import csv
import math
import re
import sys
import wave
from pathlib import Path


def read_wav(path):
    if not path.exists():
        return None
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, 48000):
            raise ValueError("expected 48kHz mono PCM16: " + str(path))
        a = array.array("h")
        a.frombytes(w.readframes(w.getnframes()))
        if sys.byteorder != "little":
            a.byteswap()
        return a


def rms_db(x):
    if not len(x):
        return float("-inf")
    power = sum(int(v) * int(v) for v in x) / len(x)
    return 20 * math.log10(max(1e-12, math.sqrt(power) / 32768))


def parse_fields(text):
    return dict(re.findall(r"(\w+)=([^ ]+)", text))


def merge_adjacent(spans):
    out = []
    for start, end in sorted(spans):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(end, out[-1][1]))
        else:
            out.append((start, end))
    return out


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("directory")
    args = argp.parse_args()
    directory = Path(args.directory)
    audio = {name: read_wav(directory / (name + ".wav")) for name in
             ("tx_mic", "tx_tunnel", "rx_input", "rx_masked", "rx_speaker")}
    for name, x in audio.items():
        if x is None:
            print("%-12s: MISSING" % name)
        else:
            print("%-12s: %.2fs RMS=%.1f dBFS" % (name, len(x) / 48000, rms_db(x)))
    with (directory / "events.csv").open(encoding="utf-8", newline="") as f:
        events = list(csv.DictReader(f))
    gap = sum(e["event"] == "REALTIME GAP" and e["direction"] == "RX<-REMOTE" for e in events)
    if gap:
        print("WARNING: RX gaps=%d; absolute sample positions in compact WAV need gap mapping." % gap)
    masked = audio.get("rx_masked")
    incoming = audio.get("rx_input")
    for event_name, expected in (("CP CUT APPLIED", 576), ("CP FULL CUT APPLIED", 2048)):
        pieces = []
        for e in events:
            if e["event"] == event_name:
                fld = parse_fields(e["fields"])
                pieces.append((int(fld["start_abs"]), int(fld["end_abs"])))
        groups = merge_adjacent(pieces)
        tested, bad, skipped = 0, 0, 0
        original_db = []
        for a, b in groups:
            if gap or masked is None or b > len(masked) or a < 0:
                skipped += 1
                continue
            tested += 1
            if any(masked[a:b]):
                bad += 1
            if incoming is not None and b <= len(incoming):
                original_db.append(rms_db(incoming[a:b]))
        print("%s: %d blocks, %d logged spans, verified=%d, nonzero=%d, skipped=%d" %
              (event_name, len(groups), len(pieces), tested, bad, skipped))
        if groups:
            sizes = sorted(set(b - a for a, b in groups))
            print("  sizes(samples)=%s, expected=%d" % (sizes[:8], expected))
        if original_db:
            print("  mean input block RMS %.1f dBFS" % (sum(original_db) / len(original_db)))
    print("Note: rx_speaker.wav follows LIVE listening-mode switches; compare by mode, not entire-file RMS only.")


if __name__ == "__main__":
    main()
