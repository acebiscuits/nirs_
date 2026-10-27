#!/usr/bin/env python3
"""Offline single-PC sanity check of spectral signflip, no PulseAudio/WebRTC.

Tests that the same simulated transmitted PCM produces different results
under matching and mismatching decryption keys. It does NOT check sync or
browser routing; use an app loopback for those.
"""
import argparse
import hashlib
import zlib
import numpy as np
from crypto.stft import STFTCipher
from crypto.mdct import MDCTCipher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--transform', choices=['stft', 'mdct'], default='stft')
    ap.add_argument('--operation', choices=['signflip', 'shuffle', 'invert'], default='signflip')
    ap.add_argument('--key', default='key.bin')
    args = ap.parse_args()
    from pathlib import Path
    key = Path(args.key).read_bytes()
    if not key:
        raise SystemExit('Key file is empty')
    cls = {'stft': STFTCipher, 'mdct': MDCTCipher}[args.transform]
    enc = cls(args.operation, key=key, key_schedule='per_frame')
    good = cls(args.operation, key=key, key_schedule='per_frame')
    wrongkey = hashlib.sha256(b'WRONG-RX-TEST:' + key).digest()
    bad = cls(args.operation, key=wrongkey, key_schedule='per_frame')
    fs = 48000
    samples = 1024
    duration = 1.7 if args.transform == 'mdct' else 3.0
    N = int(np.ceil(fs * duration / samples)) * samples
    t = np.arange(N) / fs
    # Voice-like AM and harmonic mixture. Local synthetic stimulus, not recorded speech.
    sig = (0.065*np.sin(2*np.pi*170*t)*(0.5+0.5*np.sin(2*np.pi*3.5*t)**2) +
           0.020*np.sin(2*np.pi*430*t) +
           0.012*np.sin(2*np.pi*780*t))
    outs = {k: [] for k in ('wire', 'correct', 'wrong')}
    for offset in range(0, N, samples):
        encrypted = enc.encrypt(sig[offset:offset+samples])
        if encrypted.size == 0:
            continue
        # Model project physical transfer: ciphertext_gain, 16-bit quantization.
        pcm = np.rint(np.clip(encrypted*0.68, -0.999, 0.999)*32767).astype('<i2')
        incoming = pcm.astype(np.float64)/32768.0
        unscaled = incoming/0.68
        outs['wire'].append(incoming)
        outs['correct'].append(good.decrypt(unscaled))
        outs['wrong'].append(bad.decrypt(unscaled))
    for name in outs:
        outs[name] = np.concatenate(outs[name])
    cut = int(.25 * fs)  # startup overlap-edge artifacts are measured separately
    def rms(x):
        return np.sqrt(np.mean(np.square(x))) if x.size else 0.0
    goodx = outs['correct'][cut:]
    badx = outs['wrong'][cut:]
    rawx = outs['wire'][cut:]
    diff = rms(goodx-badx)
    corr = np.corrcoef(goodx, badx)[0,1]
    print('OFFLINE CIPHER TEST:', args.transform, args.operation)
    print('Note: no WebRTC, no FULL/CP synchronization, no audio routing.')
    print('raw RMS      : %.5f' % rms(rawx))
    print('correct RMS  : %.5f' % rms(goodx))
    print('wrong RMS    : %.5f' % rms(badx))
    print('correct-wrong difference RMS: %.5f' % diff)
    print('correct-wrong correlation   : %.4f' % corr)
    for mode, data in (('RAW',rawx),('CORRECT',goodx),('WRONG',badx)):
        b=np.rint(np.clip(data,-.999,.999)*32767).astype('<i2').tobytes()
        print('%7s output CRC32           : %08x' %(mode,zlib.crc32(b)))
    if diff <= 0.005 or abs(corr) >= .95:
        print('WARNING: wrong and correct produce unexpectedly similar output.')
        raise SystemExit(1)
    print('PASS: on the same PCM, correct and wrong decryptions are different.')


if __name__=='__main__':
    main()
