"""Public plug-in contract: __init__(key, operation, decrypt); process(index, pcm320).

Input/output: 320 samples at 16 kHz (default 20 kHz wire rate).
A plug-in receives the absolute frame index (modulo 64 on reacquisition may be
used by algorithms). Key generation/state management belong to the plug-in.
"""
import importlib
import hashlib
import numpy as np
from scipy.fft import dct
from .mdct import MDCTCipher
from .stft_overlap import STFTOverlap


from .research_transforms import (TRANSFORMS, ALL_OPERATIONS, forward, inverse,
                                  apply_operation, configure_frame_size)

FRAME = 320  # Must match the 20ms voice payload in main.py at 20k wire rate.


class BuiltinFrameCipher:
    """Frame cipher using original standalone research algorithms.

    Lapped transforms are experimental: PCM synthesis/reanalysis is not
    generally an exact inverse for arbitrary coefficient modifications.
    """
    def __init__(self, key, operation='signflip', decrypt=False, kind='stft', frame_size=320):
        if operation not in ALL_OPERATIONS:
            raise ValueError('Unknown operation: ' + str(operation))
        if kind not in TRANSFORMS + ('stft-overlap', 'mdct-legacy', 'mdct-overlap'):
            raise ValueError('Unknown transform: ' + str(kind))
        if operation != 'invert' and not key:
            raise ValueError('Key is required for this operation')
        self.frame_size = int(frame_size)
        configure_frame_size(self.frame_size)
        self.key = bytes(key)
        self.operation = operation
        self.decrypt = bool(decrypt)
        self.kind = kind
        self.next_index = None
        if kind in ('mdct-overlap', 'mdct-legacy'):
            self.obj = MDCTCipher(operation, key=key, key_schedule='per_frame', half_size=self.frame_size)
            # Pre-roll: one 320-sample hop of startup latency.
            self.obj._process(np.zeros(self.frame_size), decrypt=decrypt)
        elif kind == 'stft-overlap':
            self.obj = STFTOverlap(key=key, operation=operation, decrypt=decrypt, hop=self.frame_size)

    def process(self, index, samples):
        index = int(index)
        samples = np.asarray(samples, dtype=np.float64)
        if index < 0 or samples.shape != (self.frame_size,) or not np.all(np.isfinite(samples)):
            raise ValueError(f'Expected nonnegative index and {self.frame_size} finite PCM samples')
        if self.kind == 'stft-overlap':
            result = self.obj.process(index, samples)
        elif self.kind in ('mdct-legacy', 'mdct-overlap'):
            if self.next_index is not None and index != self.next_index:
                self.obj.reset(index)
                self.obj._process(np.zeros(self.frame_size), decrypt=self.decrypt)
            result = self.obj.decrypt(samples) if self.decrypt else self.obj.encrypt(samples)
        else:
            coeff = forward(self.kind, samples)
            changed = apply_operation(self.kind, coeff, self.operation,
                                      self.key, index, self.decrypt)
            result = inverse(self.kind, changed)
        self.next_index = index + 1
        result = np.asarray(result, dtype=np.float64)
        if result.shape != (self.frame_size,) or not np.all(np.isfinite(result)):
            raise RuntimeError(f'Transform returned invalid frame shape {result.shape}')
        return np.clip(result, -1., 1.).astype(np.float32)


def make_frame_cipher(args, key, decrypt=False):
    frame_size = int(getattr(args, 'voice_frame', 320))
    if not getattr(args, 'plugin', None):
        return BuiltinFrameCipher(key, args.operation, decrypt, args.cipher, frame_size=frame_size)
    if ':' not in args.plugin:
        raise ValueError('--plugin must be module:Class')
    module, cls_name = args.plugin.rsplit(':', 1)
    cls = getattr(importlib.import_module(module), cls_name)
    plugin = cls(key=key, operation=args.operation, decrypt=decrypt)
    if not callable(getattr(plugin, 'process', None)):
        raise TypeError('Plugin must implement process(index, pcm320)')
    return _CheckedPlugin(plugin, frame_size)

class _CheckedPlugin:
    def __init__(self, obj, frame_size):
        self.obj = obj
        self.frame_size = frame_size
    def process(self, index, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.shape != (self.frame_size,) or not np.all(np.isfinite(samples)):
            raise ValueError(f'Plugin input must be {self.frame_size} finite samples')
        out = np.asarray(self.obj.process(int(index), samples), dtype=np.float32)
        if out.shape != (FRAME,) or not np.all(np.isfinite(out)):
            raise ValueError(f'Plugin output must be {self.frame_size} finite samples')
        return np.clip(out, -1., 1.)
