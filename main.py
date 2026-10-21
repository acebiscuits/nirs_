#!/usr/bin/env python3
"""Experimental speech scrambling over an audio call. Not authenticated encryption."""
import argparse
import csv
import json
import hashlib
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from VirtualDevices import PulseVirtualDevices, TX_SINK, RX_SOURCE, get_default_audio_devices
from crypto.stft import STFTCipher
from crypto.mdct import MDCTCipher
from v2_timepack import FixedGuardSyncV1, SyncV1Config

VOICE_RATE, WIRE_RATE, VOICE_FRAME, WIRE_FRAME, GUARD = 16000, 20000, 320, 400, 80


def pcm(x):
    return (np.clip(np.asarray(x), -1, 1) * 32767).astype('<i2').tobytes()


def frames(stream, count):
    while True:
        data = stream.read(count * 2)
        if len(data) != count * 2: return
        yield np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768


def wav(path, rate):
    f = wave.open(str(path), 'wb')
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate)
    return f


def make_cipher(args, key, offset=0):
    common = dict(operation=args.operation, key=key, session_nonce='thesis-static', key_schedule='per_frame')
    if args.cipher == 'stft': cipher = STFTCipher(**common, frame_size=640, hop_size=320)
    else: cipher = MDCTCipher(**common, half_size=320)
    cipher.reset(offset)
    # Initial overlap/context: spectral windows are twice the transport payload.
    return cipher


class FramedCipher:
    def __init__(self, args, key, decrypt=False, offset=0):
        self.obj = make_cipher(args, key, offset)
        self.decrypt = decrypt
        self.next_index = 0
        self.offset = offset
        self.obj._process(np.zeros(VOICE_FRAME), decrypt=decrypt)  # prime overlap, emits nothing

    def process(self, index, samples):
        if index != self.next_index:
            self.obj.reset(index + self.offset)
            self.obj._process(np.zeros(VOICE_FRAME), decrypt=self.decrypt)
        self.next_index = index + 1
        out = self.obj.decrypt(samples) if self.decrypt else self.obj.encrypt(samples)
        if len(out) != VOICE_FRAME: raise RuntimeError(f'Cipher emitted {len(out)} instead of {VOICE_FRAME} samples')
        return np.clip(out, -1, 1).astype(np.float32)


def pilot_frame(index):
    # Deterministic, band-limited reference signal; no secret transmitted here.
    t = (index * VOICE_FRAME + np.arange(VOICE_FRAME)) / VOICE_RATE
    # Distinct incommensurate tones make correct time/phase alignment measurable.
    x = sum(np.sin(2 * np.pi * f * t + phase) for f, phase in
            ((353, .2), (617, 1.1), (947, 2.0), (1307, .7), (1831, 2.7)))
    return (0.075 * x).astype(np.float32)


def pilot_score(reference, received):
    # Compare a reference against possible transport/algorithm latency.
    # Discard first 0.3 s due to STFT/MDCT startup and capture boundaries.
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(received, dtype=np.float64)
    best = 0.0
    for lag in range(-960, 961, 32):
        a, b = max(0, -lag), max(0, lag)
        length = min(len(x) - a, len(y) - b)
        if length < 8000: continue
        u = x[a:a+length]; v = y[b:b+length]
        denom = np.linalg.norm(u) * np.linalg.norm(v)
        if denom > 1e-8: best = max(best, float(np.dot(u, v) / denom))
    return round(best, 4)


def guard_bank(cfg):
    # Same PN generator as receiver; exposes exactly the templates used in detection.
    from v2_timepack import makeGuardBank
    return makeGuardBank(WIRE_RATE, GUARD, cfg.spinnerSize * cfg.superSpinnerSize)


def log_json(fp, event, **kwargs):
    record = {'time': time.time(), 'event': event, **kwargs}
    fp.write(json.dumps(record, ensure_ascii=False, default=str) + '\n'); fp.flush()
    if event not in ('frame',): print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


def run_tx(args, key, folder, events):
    cfg = SyncV1Config(rate=WIRE_RATE, frameSamples=WIRE_FRAME, coreSamples=VOICE_FRAME, guardSamples=GUARD)
    bank = guard_bank(cfg)
    encoder = FramedCipher(args, key)
    source = None; player = None; devices = None
    try:
        if args.input:
            source = wave.open(args.input, 'rb')
            if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, VOICE_RATE):
                raise ValueError('Input WAV must be mono PCM16 at 16000 Hz')
            read_frame = lambda: source.readframes(VOICE_FRAME)
        else:
            mic, _ = get_default_audio_devices()
            if args.mic: mic = args.mic
            devices = PulseVirtualDevices(); devices.start()
            print('Telemost microphone: TX_Microphone. Configure your call, then press Enter.', flush=True)
            input('TX: ENTER to begin transmission > ')
            source = subprocess.Popen(['parecord','--raw',f'--device={mic}','--rate=16000','--format=s16le','--channels=1'], stdout=subprocess.PIPE, bufsize=1)
            player = subprocess.Popen(['paplay','--raw',f'--device={TX_SINK}','--rate=20000','--format=s16le','--channels=1'], stdin=subprocess.PIPE, bufsize=0)
            read_frame = lambda: source.stdout.read(VOICE_FRAME * 2)
        with wav(folder/'tx_plain.wav', VOICE_RATE) as plain, wav(folder/'tx_wire.wav', WIRE_RATE) as wire:
            idx = 0; start = time.monotonic()
            while not args.duration or time.monotonic() - start < args.duration:
                data = read_frame()
                if len(data) != 2 * VOICE_FRAME: break
                voice = np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768
                if idx < args.pilot_frames: voice = pilot_frame(idx)
                encrypted = encoder.process(idx, voice)
                marker = bank[idx % len(bank)] * args.guard_amp
                packet = np.concatenate((encrypted * args.tx_gain, marker)).astype(np.float32)
                packed = pcm(packet)
                plain.writeframes(pcm(voice)); wire.writeframes(packed)
                if player: player.stdin.write(packed)
                if idx % 250 == 0: log_json(events, 'tx_progress', frame=idx, seconds=idx/50)
                idx += 1
            log_json(events, 'tx_done', frames=idx)
    finally:
        if isinstance(source, subprocess.Popen):
            source.terminate(); source.wait(timeout=3)
        elif source: source.close()
        if player:
            if player.stdin: player.stdin.close()
            try: player.wait(timeout=3)
            except subprocess.TimeoutExpired: player.terminate()
        if devices: devices.stop()


def run_rx(args, key, folder, events):
    # The raw 20k stream is NEVER connected to the speaker; only extracted 16k cores are.
    correct = FramedCipher(args, key, decrypt=True)
    wrong_key = bytes((b ^ 0xA5) for b in key)
    wrong = FramedCipher(args, wrong_key, decrypt=True)
    outputs = {name: wav(folder/f'rx_{name}.wav', VOICE_RATE) for name in ('cipher', 'correct', 'wrong')}
    raw = wav(folder/'rx_wire.wav', WIRE_RATE)
    rows = open(folder/'frames.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(rows)
    writer.writerow(['frame','status','corr','guard_pos','guard_source','reason','monitor','rms_cipher','rms_correct'])
    stats = {'frames': 0, 'lost': 0, 'reacquired': 0, 'acquired': 0}
    pilot_capture = {k: [] for k in ('cipher', 'correct', 'wrong')}
    pilot_index = []
    player = None; source = None; devices = None
    selection = args.monitor
    try:
        if args.input:
            source = wave.open(args.input, 'rb')
            if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, WIRE_RATE):
                raise ValueError('RX WAV must be mono PCM16 at 20000 Hz')
            blocks = iter(lambda: source.readframes(args.io_samples), b'')
        else:
            _, headphones = get_default_audio_devices()
            if args.speakers: headphones = args.speakers
            devices = PulseVirtualDevices(); devices.start()
            print('Telemost speakers: RX_sink. Set them in the call, then press Enter.', flush=True)
            input('RX: ENTER to start searching for PN sync > ')
            source = subprocess.Popen(['parecord','--raw',f'--device={RX_SOURCE}','--rate=20000','--format=s16le','--channels=1'], stdout=subprocess.PIPE, bufsize=0)
            if args.listen:
                player = subprocess.Popen(['paplay','--raw',f'--device={headphones}','--rate=16000','--format=s16le','--channels=1'], stdin=subprocess.PIPE, bufsize=0)
            blocks = iter(lambda: source.stdout.read(args.io_samples * 2), b'')

        def on_frame(index, wire_frame, status, meta):
            valid = status == 'ok'
            payload = np.asarray(wire_frame[:VOICE_FRAME], dtype=np.float32) / args.tx_gain if valid else np.zeros(VOICE_FRAME, dtype=np.float32)
            payload = np.clip(payload, -1, 1)
            decoded = correct.process(index, payload)
            incorrect = wrong.process(index, payload)
            branches = {'cipher': payload, 'correct': decoded, 'wrong': incorrect}
            for name, samples in branches.items():
                outputs[name].writeframes(pcm(samples))
                if index < args.pilot_frames and index >= 0: pilot_capture[name].append((index, samples.copy()))
            if player:
                active = ('cipher', 'correct', 'wrong')[(index // 250) % 3] if selection == 'rotate' else selection
                player.stdin.write(pcm(branches[active]))
            stats['frames'] += 1; stats['lost'] += (not valid)
            rms = lambda arr: round(float(np.sqrt(np.mean(arr.astype(np.float64)**2))), 6)
            writer.writerow([index, status, meta.get('corr',''), meta.get('guard_pos',''), meta.get('guard_source',''), meta.get('reason',''), (('cipher', 'correct', 'wrong')[(index // 250) % 3] if selection == 'rotate' else selection), rms(payload), rms(decoded)])
            if not valid or index % 50 == 0:
                log_json(events, 'frame', frame=index, status=status, corr=meta.get('corr'), reason=meta.get('reason'), guard_source=meta.get('guard_source'))

        def on_event(event):
            stats['acquired'] += event.get('event') == 'acquired'
            stats['reacquired'] += event.get('event') == 'reacquired'
            log_json(events, 'sync_'+event.get('event','unknown'), **{k:v for k,v in event.items() if k != 'event'})

        cfg = SyncV1Config(rate=WIRE_RATE, frameSamples=WIRE_FRAME, coreSamples=VOICE_FRAME, guardSamples=GUARD)
        sync = FixedGuardSyncV1(cfg, on_frame, on_event)
        start = time.monotonic()
        for block in blocks:
            if args.duration and time.monotonic()-start > args.duration: break
            if len(block) % 2: block = block[:-1]
            raw.writeframes(block)
            if block: sync.push(np.frombuffer(block, dtype='<i2').astype(np.float32)/32768, time.monotonic())
        if args.input: sync.finish()
        stats.update(sync.stats); stats['state_final'] = sync.state
        if args.pilot_frames >= 50 and all(len(pilot_capture[k]) >= 45 for k in pilot_capture):
            scores = {}
            for name, entries in pilot_capture.items():
                # Only consecutive portions are comparable with a known reference.
                reference = np.concatenate([pilot_frame(i) for i, _ in entries])
                received = np.concatenate([data for _, data in entries])
                scores[name] = pilot_score(reference, received)
            stats['pilot_correlations'] = scores
            c, other = scores['correct'], max(scores['cipher'], scores['wrong'])
            verdict = 'LIKELY_OK' if c >= .20 and c - other >= .10 else 'UNVERIFIED'
            stats['pilot_verdict'] = verdict
            log_json(events, 'pilot_check', verdict=verdict, scores=scores,
                     note='Correlation with known calibration phrase, NOT cryptographic authentication')
        else:
            stats['pilot_verdict'] = 'INSUFFICIENT_DATA'
            log_json(events, 'pilot_check', verdict='INSUFFICIENT_DATA')
        (folder/'summary.json').write_text(json.dumps(stats, indent=2))
        log_json(events, 'rx_done', **stats)
    finally:
        rows.close(); raw.close()
        for f in outputs.values(): f.close()
        if isinstance(source, subprocess.Popen):
            source.terminate(); source.wait(timeout=3)
        elif source: source.close()
        if player:
            if player.stdin: player.stdin.close()
            try: player.wait(timeout=3)
            except subprocess.TimeoutExpired: player.terminate()
        if devices: devices.stop()


def main():
    p = argparse.ArgumentParser(description='Experimental STFT/MDCT audio + V2 64-PN synchronization over 20k TimePack')
    p.add_argument('--role', choices=['tx', 'rx'], required=True)
    p.add_argument('--cipher', choices=['stft','mdct'], default='stft')
    p.add_argument('--operation', choices=['signflip','shuffle','invert'], default='signflip')
    p.add_argument('--key-file', required=True)
    p.add_argument('--input', help='TX: 16k PCM16 mono WAV; RX: 20k PCM16 mono WAV')
    p.add_argument('--out', default='runs/test')
    p.add_argument('--monitor', choices=['cipher','correct','wrong','rotate'], default='correct')
    p.add_argument('--listen', action='store_true')
    p.add_argument('--mic'); p.add_argument('--speakers')
    p.add_argument('--duration', type=float, default=0)
    p.add_argument('--pilot-seconds', type=float, default=2.0, help='Calibration tone before speech; 0 disables')
    p.add_argument('--tx-gain', type=float, default=.7)
    p.add_argument('--guard-amp', type=float, default=.15)
    p.add_argument('--io-samples', type=int, default=400)
    args = p.parse_args()
    args.pilot_frames = int(round(args.pilot_seconds * 50))
    if args.pilot_frames < 0: p.error("pilot-seconds must be >= 0")
    if not (0 < args.tx_gain <= 1 and 0 < args.guard_amp <= 1): p.error('invalid gain')
    if args.operation == 'invert': print('WARNING: invert is not key-dependent; do not treat this as keyed encryption.', flush=True)
    key = Path(args.key_file).read_bytes()
    if len(key) < 16: p.error('key-file must contain at least 16 bytes')
    folder = Path(args.out); folder.mkdir(parents=True, exist_ok=True)
    with open(folder/'events.jsonl', 'w', encoding='utf-8') as events:
        log_json(events, 'start', role=args.role, cipher=args.cipher, operation=args.operation, monitor=args.monitor, nonce='fixed internal constant', rates=[VOICE_RATE,WIRE_RATE])
        try:
            if args.role == 'tx': run_tx(args, key, folder, events)
            else: run_rx(args, key, folder, events)
        except KeyboardInterrupt: log_json(events, 'interrupted')
        except Exception as exc:
            log_json(events, 'fatal', error=repr(exc)); raise


if __name__ == '__main__': main()
