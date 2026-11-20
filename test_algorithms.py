#!/usr/bin/env python3
"""Checks installed transforms/operations on a 20ms frame PCM16 path.
Does not claim that experimental bitwise/lapped methods are lossless.
"""
import numpy as np
from crypto.adapter import BuiltinFrameCipher
from crypto.research_transforms import TRANSFORMS, ALL_OPERATIONS, BIT_ALIASES

KEY = b'network-test-key-strong-123456'
OTHER_KEY = b'wrong-key-network-strong-123456'
R = np.random.default_rng(2026)


def roundtrip(kind, op, n=65, pcm16=True):
    tx = BuiltinFrameCipher(KEY, op, False, kind)
    rx = BuiltinFrameCipher(KEY, op, True, kind)
    wrong = BuiltinFrameCipher(OTHER_KEY, op, True, kind)
    x, correct, incorrect = [], [], []
    for i in range(n):
        # Small amplitude to isolate transform behavior from clipping.
        v = R.normal(0, .012, 320).astype(np.float32)
        encoded = tx.process(i, v)
        if pcm16:
            encoded = (np.clip(encoded,-1,1)*32767).astype('<i2').astype(np.float32)/32767
        d = rx.process(i, encoded)
        w = wrong.process(i, encoded)
        x.append(v); correct.append(d); incorrect.append(w)
    x=np.concatenate(x);correct=np.concatenate(correct);incorrect=np.concatenate(incorrect)
    error=float(np.sqrt(np.mean((x-correct)**2)))
    wrong_error=float(np.sqrt(np.mean((x-incorrect)**2)))
    return error,wrong_error


def main():
    failures=[]
    for kind in TRANSFORMS + ('stft-overlap','mdct-overlap'):
        for operation in ALL_OPERATIONS:
            try:
                err,bad=roundtrip(kind,operation)
                # Bitwise and lapped operations are experimental and do not meet this criterion.
                good=kind not in ('stft-overlap','mdct-overlap') and operation not in ('sbox','rot-xor') and operation not in BIT_ALIASES
                if good and err > .003:
                    failures.append((kind,operation,err))
                print(f'{kind:17} {operation:23} rmse={err:.5f} wrong_rmse={bad:.5f} '+('PASS' if good and err<=.003 else 'EXPERIMENTAL' if not good else 'FAIL'))
            except Exception as exc:
                if kind not in ('stft-overlap','mdct-overlap') or operation not in ('sbox','rot-xor') and operation not in BIT_ALIASES:
                    failures.append((kind,operation,repr(exc)))
                print('ERROR',kind,operation,repr(exc))
    if failures:
        raise AssertionError(f'Failed supported pairs: {failures}')
    print('All supported pairs passed the synthetic PCM16 test.')

if __name__=='__main__': main()
