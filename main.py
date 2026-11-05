#!/usr/bin/env python3
"""Experimental speech scrambling over an audio call. Not authenticated encryption."""
import argparse
import csv
import json
import threading
import hashlib
import os
import subprocess
import sys
import time
import wave
from math import gcd
from pathlib import Path
from dataclasses import replace


os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from scipy.signal import resample_poly
from VirtualDevices import PulseVirtualDevices, TX_SINK, RX_SOURCE, get_default_audio_devices, stop_event
from crypto.adapter import make_frame_cipher
from v2_timepack import FixedGuardSyncV1, SyncV1Config

# VOICE_RATE, WIRE_RATE, VOICE_FRAME, WIRE_FRAME, GUARD = 16000, 20000, 320, 400, 80


def pcm(x):
    return (np.clip(np.asarray(x), -1, 1) * 32767).astype('<i2').tobytes()


def wav(path, rate):
    f = wave.open(str(path), 'wb')
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate)
    return f


def pcm_read_exact(stream, length):
    data = bytearray()
    while len(data) < length and not stop_event.is_set():
        chunk = stream.read(length - len(data))
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


def monitor_input(args):
    """Keyboard commands: 0 raw, 1 correct, 2 wrong, 3 mute, q stop."""
    print('RX MONITOR: 0=raw(cipher)  1=correct  2=wrong  3=mute  r=rotate  q=quit  [Enter]', flush=True)
    while not stop_event.is_set():
        try:
            cmd = sys.stdin.readline()
        except (OSError, ValueError):
            return
        if not cmd:
            return
        cmd = cmd.strip().lower()
        names = {'0': 'cipher', '1': 'correct', '2': 'wrong', '3': 'mute', 'r': 'rotate'}
        if cmd == 'q':
            stop_event.set()
            return
        if cmd in names:
            args.monitor_state['mode'] = names[cmd]
            print('MONITOR CHANGED:', names[cmd], flush=True)

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

def make_sync_config():
    base = SyncV1Config()
    k = WIRE_RATE / 20000.0          # коэффициент от эталонной калибровки
    return replace(
        base,
        rate=WIRE_RATE,
        frameSamples=WIRE_FRAME,
        coreSamples=VOICE_FRAME,
        guardSamples=GUARD,
        # ⬇ только сэмпловые поля масштабируются
        trackRadius           = int(round(base.trackRadius           * k)),
        trackRecoveryRadius   = int(round(base.trackRecoveryRadius   * k)),
        trainRadius           = int(round(base.trainRadius           * k)),
        trackContinuityRadius = int(round(base.trackContinuityRadius * k)),
        trackChainOffsetSpread= int(round(base.trackChainOffsetSpread* k)),
        maxTimingJump         = int(round(base.maxTimingJump         * k)),
        maxCoreAdjust         = int(round(base.maxCoreAdjust         * k)),
        gapRecoveryCoherence  = int(round(base.gapRecoveryCoherence  * k)),
        gapRecoveryRadius     = int(round(base.gapRecoveryRadius     * k)),
        acquireScanStrideSamples = int(round(base.acquireScanStrideSamples * k)),
        # пороги 0..1, счётчики кадров, *_Frames, *_Alpha — не трогаем
    )

def guard_bank(cfg):
    # Same PN generator as receiver; exposes exactly the templates used in detection.
    from v2_timepack import makeGuardBank
    return makeGuardBank(WIRE_RATE, GUARD, cfg.spinnerSize * cfg.superSpinnerSize)


_log_lock = threading.Lock()

def log_json(fp, event, **kwargs):
    record = {'time': time.time(), 'event': event, **kwargs}
    with _log_lock:
        fp.write(json.dumps(record, ensure_ascii=False, default=str) + '\n'); fp.flush()
        if event not in ('frame',): print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


def run_tx(args, key, folder, events):
    # cfg = SyncV1Config(rate=WIRE_RATE, frameSamples=WIRE_FRAME, coreSamples=VOICE_FRAME, guardSamples=GUARD)
    cfg = make_sync_config()
    bank = guard_bank(cfg)
    encoder = make_frame_cipher(args, key, decrypt=False)
    source = None; player = None; devices = None
    try:
        if args.input:
            source = wave.open(args.input, 'rb')
            if (source.getnchannels(), source.getsampwidth()) != (1, 2):
                raise ValueError('TX WAV must be mono PCM16')
            input_rate = source.getframerate()
            if not 8000 <= input_rate <= 192000:
                raise ValueError('Unsupported input sample rate')
            data = np.frombuffer(source.readframes(source.getnframes()), dtype='<i2').astype(np.float64) / 32768.0
            if input_rate != VOICE_RATE:
                factor = gcd(input_rate, VOICE_RATE)
                data = resample_poly(data, VOICE_RATE // factor, input_rate // factor)
            # Duration for WAV input is AUDIO duration, not time taken by CPU to encode it.
            if args.duration:
                data = data[:int(round(args.duration * VOICE_RATE))]
            byte_buffer = pcm(data)
            cursor = 0
            def read_frame():
                nonlocal cursor
                if cursor >= len(byte_buffer):
                    return b''
                block = byte_buffer[cursor:cursor + VOICE_FRAME * 2]
                cursor += VOICE_FRAME * 2
                return block.ljust(VOICE_FRAME * 2, b'\x00')
        else:
            mic, _ = get_default_audio_devices()
            if args.mic: mic = args.mic
            if not args.shared_devices:
                devices = PulseVirtualDevices(); devices.start()
                print('Call microphone: TX_Microphone. Configure your call, then press Enter.', flush=True)
                input('TX: ENTER to begin transmission > ')
            source = subprocess.Popen(['parec','--raw',f'--device={mic}','--rate=16000','--format=s16le','--channels=1','--latency-msec=30'], stdout=subprocess.PIPE, bufsize=0)
            player = subprocess.Popen(['pacat','--playback','--raw',f'--device={TX_SINK}','--rate=20000','--format=s16le','--channels=1','--latency-msec=30'], stdin=subprocess.PIPE, bufsize=0)
            read_frame = lambda: pcm_read_exact(source.stdout, VOICE_FRAME * 2)
        with wav(folder/'tx_plain.wav', VOICE_RATE) as plain, wav(folder/'tx_wire.wav', WIRE_RATE) as wire:
            idx = 0; start = time.monotonic()
            while not stop_event.is_set() and (args.input or not args.duration or time.monotonic() - start < args.duration):
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
    correct = make_frame_cipher(args, key, decrypt=True)
    wrong_key = bytes((b ^ 0xA5) for b in key)
    wrong = make_frame_cipher(args, wrong_key, decrypt=True)
    outputs = {name: wav(folder/f'rx_{name}.wav', VOICE_RATE) for name in ('cipher', 'correct', 'wrong')}
    raw = wav(folder/'rx_wire.wav', WIRE_RATE)
    rows = open(folder/'frames.csv', 'w', newline='', encoding='utf-8')
    writer = csv.writer(rows)
    writer.writerow(['frame','status','corr','guard_pos','guard_source','reason','monitor','rms_cipher','rms_correct'])
    stats = {'frames': 0, 'lost': 0, 'reacquired': 0, 'acquired': 0}
    pilot_capture = {k: [] for k in ('cipher', 'correct', 'wrong')}
    player = None; source = None; devices = None
    selection = args.monitor_state
    try:
        if args.input:
            source = wave.open(args.input, 'rb')
            if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, WIRE_RATE):
                raise ValueError('RX WAV must be mono PCM16 at 20000 Hz')
            blocks = iter(lambda: source.readframes(args.io_samples), b'')
        else:
            _, headphones = get_default_audio_devices()
            if args.speakers: headphones = args.speakers
            if not args.shared_devices:
                devices = PulseVirtualDevices(); devices.start()
                print('Call speakers: RX_sink. Configure your call, then press Enter.', flush=True)
                input('RX: ENTER to start PN synchronization > ')
            threading.Thread(target=monitor_input, args=(args,), daemon=True, name='monitor-control').start()
            source = subprocess.Popen(['parec','--raw',f'--device={RX_SOURCE}','--rate=20000','--format=s16le','--channels=1','--latency-msec=30'], stdout=subprocess.PIPE, bufsize=0)
            if args.listen:
                player = subprocess.Popen(['pacat','--playback','--raw',f'--device={headphones}','--rate=16000','--format=s16le','--channels=1','--latency-msec=30'], stdin=subprocess.PIPE, bufsize=0)
            blocks = iter(lambda: pcm_read_exact(source.stdout, args.io_samples * 2), b'')

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
            mode = selection['mode']
            active = ('cipher', 'correct', 'wrong')[(index // 250) % 3] if mode == 'rotate' else mode
            if player:
                player.stdin.write(pcm(branches[active]) if active != 'mute' else bytes(VOICE_FRAME * 2))
            stats['frames'] += 1; stats['lost'] += (not valid)
            rms = lambda arr: round(float(np.sqrt(np.mean(arr.astype(np.float64)**2))), 6)
            writer.writerow([index, status, meta.get('corr',''), meta.get('guard_pos',''), meta.get('guard_source',''), meta.get('reason',''), active, rms(payload), rms(decoded)])
            if not valid or index % 50 == 0:
                log_json(events, 'frame', frame=index, status=status, corr=meta.get('corr'), reason=meta.get('reason'), guard_source=meta.get('guard_source'))

        def on_event(event):
            stats['acquired'] += event.get('event') == 'acquired'
            stats['reacquired'] += event.get('event') == 'reacquired'
            log_json(events, 'sync_'+event.get('event','unknown'), **{k:v for k,v in event.items() if k != 'event'})

        # cfg = SyncV1Config(rate=WIRE_RATE, frameSamples=WIRE_FRAME, coreSamples=VOICE_FRAME, guardSamples=GUARD)
        cfg = make_sync_config()
        sync = FixedGuardSyncV1(cfg, on_frame, on_event)
        start = time.monotonic()
        rx_cpu = 0.0
        rx_wall = 0.0
        for block in blocks:
            if stop_event.is_set(): break
            if args.duration and time.monotonic()-start > args.duration: break
            if len(block) % 2: block = block[:-1]
            raw.writeframes(block)
            if block:
                t_wall=time.perf_counter(); t_cpu=time.process_time()
                sync.push(np.frombuffer(block, dtype='<i2').astype(np.float32)/32768, time.monotonic())
                rx_cpu += time.process_time()-t_cpu
                rx_wall += time.perf_counter()-t_wall
        if args.input: sync.finish()
        stats.update(sync.stats); stats['state_final'] = sync.state
        stats['processing_cpu_sec'] = round(rx_cpu, 3)
        stats['processing_wall_sec'] = round(rx_wall, 3)
        stats['input_audio_sec'] = round(raw.getnframes()/WIRE_RATE, 3)
        stats['cpu_per_audio_sec'] = round(rx_cpu/max(0.001, stats['input_audio_sec']), 3)
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


def cleanup_loopbacks():
    """User-requested cleanup; affects ALL local module-loopback instances."""
    try:
        out = subprocess.run(['pactl', 'list', 'short', 'modules'], capture_output=True, text=True, timeout=5)
        if out.returncode:
            return
        for line in out.stdout.splitlines():
            fields = line.split()
            if len(fields) > 1 and fields[1] == 'module-loopback':
                subprocess.run(['pactl', 'unload-module', fields[0]], capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


def run_duplex(args, key, folder, events):
    """Single process owns the shared virtual devices; separate TX/RX workers."""
    devices = PulseVirtualDevices()
    devices.start()
    errors = []
    def worker(fn, subdir):
        try:
            dest = folder / subdir
            dest.mkdir(exist_ok=True, parents=True)
            fn(args, key, dest, events)
        except Exception as exc:
            errors.append((subdir, repr(exc)))
            stop_event.set()
    try:
        print('In the call select microphone TX_Microphone and speaker RX_sink.', flush=True)
        input('Configure the call on this PC; ENTER to start TX+RX > ')
        rx = threading.Thread(target=worker, args=(run_rx, 'rx'), daemon=True)
        tx = threading.Thread(target=worker, args=(run_tx, 'tx'), daemon=True)
        rx.start()
        tx.start()
        while not stop_event.is_set() and (rx.is_alive() or tx.is_alive()):
            rx.join(timeout=0.2)
            tx.join(timeout=0.2)
        stop_event.set()
        if errors:
            raise RuntimeError('Duplex worker failures: %s' % errors)
    finally:
        stop_event.set()
        # Give TX/RX time to close WAV headers and their PulseAudio pipes before
        # unloading the shared virtual devices. Do not abandon daemon workers.
        for thread in (locals().get('rx'), locals().get('tx')):
            if thread is not None and thread.is_alive():
                thread.join(timeout=3.0)
        devices.stop()


def main():
    stop_event.clear()
    if len(sys.argv) > 1 and sys.argv[1] == 'selftest':
        tests = [('test_sync_faults.py', []), ('test_smoke.py', [])]
        if '--full' in sys.argv[2:] or '--codec' in sys.argv[2:]:
            tests.append(('test_integration.py', ['--codec'] if '--codec' in sys.argv[2:] else []))
        for test, opts in tests:
            print('RUNNING:', test, opts, flush=True)
            subprocess.run([sys.executable, str(Path(__file__).parent / test)] + opts, check=True)
        return
    if len(sys.argv) > 1 and sys.argv[1] in ('tx', 'rx', 'duplex'):
        sys.argv[1:2] = ['--role', sys.argv[1]]
    elif len(sys.argv) > 1 and sys.argv[1] == 'run':
        del sys.argv[1]
    p = argparse.ArgumentParser(description='Independent STFT/DCT4 frame transforms + V2 64-PN synchronization over 20k TimePack')
    p.add_argument('--role', choices=['tx', 'rx', 'duplex'], required=True)
    p.add_argument('--cipher', choices=['stft','stft-overlap','dct4','mdct-legacy'], default='stft')
    p.add_argument('--plugin', help='Optional module:Class using the two-method frame adapter')
    p.add_argument('--operation', default='signflip', help='signflip/shuffle/invert for built-ins; plug-ins own their operation')
    p.add_argument('--key-file', required=True)
    p.add_argument('--input', help='TX: PCM16 mono WAV at 16k/44.1k/48k/etc; RX: 20k PCM16 mono WAV')
    p.add_argument('--out', default='runs/test')
    p.add_argument('--monitor', choices=['mute','cipher','correct','wrong','rotate'], default='correct')
    p.add_argument('--listen', action='store_true')
    p.add_argument('--mic'); p.add_argument('--speakers')
    p.add_argument('--duration', type=float, default=0)
    p.add_argument('--pilot-seconds', type=float, default=0.0, help='Calibration tone before speech; 0 disables')
    p.add_argument('--tx-gain', type=float, default=.7)
    p.add_argument('--guard-amp', type=float, default=.15)
    p.add_argument('--io-samples', type=int, default=400)
    p.add_argument('--keep-loopbacks', action='store_true', help='Do not unload PulseAudio module-loopback when exiting')
    p.add_argument('--wire-rate', type=int, default=20000, help='Wire sample rate (default: 20000). Must be a multiple of 4000 for clean 4ms guard.')
    args = p.parse_args()
    args.monitor_state = {'mode': args.monitor}
    args.shared_devices = args.role == 'duplex'


    global VOICE_RATE, WIRE_RATE, VOICE_FRAME, WIRE_FRAME, GUARD
    WIRE_RATE = args.wire_rate
    # Сохраняем пропорцию 0.8 (голос : провод) и длительности 20ms / 4ms
    VOICE_RATE  = int(round(WIRE_RATE * 0.8))
    WIRE_FRAME  = WIRE_RATE // 50          # 20 мс
    VOICE_FRAME = VOICE_RATE // 50         # 20 мс
    GUARD       = WIRE_RATE // 250         # 4 мс


    if args.role == 'duplex' and args.input: p.error('duplex uses live audio; file mode requires separate TX/RX')
    if args.role == 'duplex': args.listen = True
    if args.io_samples <= 0: p.error('io-samples must be positive')
    args.pilot_frames = int(round(args.pilot_seconds * 50))
    if args.pilot_frames < 0: p.error("pilot-seconds must be >= 0")
    if not (0 < args.tx_gain <= 1 and 0 < args.guard_amp <= 1): p.error('invalid gain')
    if args.operation == 'invert' and not args.plugin: print('WARNING: invert is not key-dependent.', flush=True)
    if args.cipher == 'mdct-legacy' and not args.plugin: print('WARNING: MDCT legacy shuffle/signflip are NOT generally reversible after PCM synthesis; use stft/dct4 for live tests.', flush=True)
    if args.cipher == 'stft-overlap' and not args.plugin: print('WARNING: Overlapped STFT shuffle/signflip are NOT generally reversible after PCM synthesis; compare against --cipher stft.', flush=True)
    key = Path(args.key_file).read_bytes()
    if len(key) < 16: p.error('key-file must contain at least 16 bytes')
    folder = Path(args.out); folder.mkdir(parents=True, exist_ok=True)
    with open(folder/'events.jsonl', 'w', encoding='utf-8') as events:
        log_json(events, 'start', role=args.role, cipher=args.cipher, operation=args.operation, monitor=args.monitor, guard_cycle=64, key_frame_period=64, rates=[VOICE_RATE,WIRE_RATE])
        try:
            if args.role == 'tx': run_tx(args, key, folder, events)
            elif args.role == 'rx': run_rx(args, key, folder, events)
            else: run_duplex(args, key, folder, events)
        except KeyboardInterrupt: log_json(events, 'interrupted')
        except Exception as exc:
            log_json(events, 'fatal', error=repr(exc)); raise
        finally:
            stop_event.set()
            if not args.input and not args.keep_loopbacks:
                cleanup_loopbacks()


if __name__ == '__main__': main()
