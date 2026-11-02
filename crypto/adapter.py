"""Public plug-in contract: __init__(key, operation, decrypt); process(index, pcm320).

Input and output: exactly 320 real PCM samples at 16 kHz per frame.
A plug-in receives the absolute frame index (modulo 64 on reacquisition may be
used by algorithms). Key generation/state management belong to the plug-in.
"""
import importlib
import hashlib
import numpy as np
from scipy.fft import dct
from .mdct import MDCTCipher

FRAME = 320

class BuiltinFrameCipher:
    """Critically sampled *independent* real-valued transforms, exactly invertible
    before clipping/quantization. No STFT-overlap reanalysis ambiguity.

    `stft` here is a 20ms rectangular STFT with hop=frame (no overlap).
    `dct4` is orthonormal type-IV DCT, not a lapped MDCT.
    `mdct-legacy` keeps the original lapped MDCT as a research baseline.
    """
    def __init__(self, key, operation='signflip', decrypt=False, kind='stft'):
        self.decrypt = decrypt
        self.kind = kind
        self.operation = operation
        self.key = key
        self.next_index = 0
        if operation not in ('signflip','shuffle','invert'):
            raise ValueError('Unsupported built-in operation: '+str(operation))
        if kind == 'mdct-legacy':
            self.obj = MDCTCipher(operation, key=key, key_schedule='per_frame', half_size=320)
            self.obj._process(np.zeros(FRAME), decrypt=decrypt)
        elif kind not in ('stft','dct4'):
            raise ValueError('Unknown transform: '+str(kind))

    def _rng(self, index):
        h = hashlib.sha512(self.key + (int(index) % 64).to_bytes(8,'big')).digest()
        return np.random.default_rng(np.random.SeedSequence(np.frombuffer(h, dtype=np.uint32), pool_size=8))

    def process(self, index, samples):
        index = int(index)
        samples = np.asarray(samples, dtype=np.float64)
        if index < 0 or samples.shape != (FRAME,) or not np.all(np.isfinite(samples)):
            raise ValueError('Expected nonnegative index and 320 finite PCM samples')
        if self.kind == 'mdct-legacy':
            if index != self.next_index:
                self.obj.reset(index)
                self.obj._process(np.zeros(FRAME), decrypt=self.decrypt)
            self.next_index = index+1
            result = self.obj.decrypt(samples) if self.decrypt else self.obj.encrypt(samples)
        else:
            if self.kind == 'stft':
                coeff = np.fft.rfft(samples, n=FRAME)
                # DC and Nyquist always remain real and at their original positions.
                inner = coeff[1:-1].copy()
            else:
                coeff = dct(samples, type=4, norm='ortho')
                inner = coeff.copy()
            if self.operation == 'invert':
                inner = inner[::-1]
            elif self.operation == 'shuffle':
                perm = self._rng(index).permutation(len(inner))
                inner = inner[np.argsort(perm)] if self.decrypt else inner[perm]
            else:
                inner *= self._rng(index).choice(np.array([-1.,1.]),len(inner))
            if self.kind == 'stft':
                coeff[1:-1] = inner
                result = np.fft.irfft(coeff, n=FRAME)
            else:
                result = dct(inner, type=4, norm='ortho')
            self.next_index = index+1
        if result.shape != (FRAME,) or not np.all(np.isfinite(result)):
            raise RuntimeError('Transformation did not produce 320 finite samples')
        return np.clip(result, -1., 1.).astype(np.float32)


def make_frame_cipher(args, key, decrypt=False):
    if not getattr(args, 'plugin', None):
        return BuiltinFrameCipher(key, args.operation, decrypt, args.cipher)
    if ':' not in args.plugin:
        raise ValueError('--plugin must be module:Class')
    module, cls_name = args.plugin.rsplit(':', 1)
    cls = getattr(importlib.import_module(module), cls_name)
    plugin = cls(key=key, operation=args.operation, decrypt=decrypt)
    if not callable(getattr(plugin, 'process', None)):
        raise TypeError('Plugin must implement process(index, pcm320)')
    return _CheckedPlugin(plugin)

class _CheckedPlugin:
    def __init__(self, obj):
        self.obj = obj
    def process(self, index, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.shape != (FRAME,) or not np.all(np.isfinite(samples)):
            raise ValueError('Plugin input must be 320 finite samples')
        out = np.asarray(self.obj.process(int(index), samples), dtype=np.float32)
        if out.shape != (FRAME,) or not np.all(np.isfinite(out)):
            raise ValueError('Plugin output must be 320 finite samples')
        return np.clip(out, -1., 1.)
