"""Legacy-style streaming STFT with 50% overlap for NIRS TimePack V2.

External plugin API: __init__(key, operation, decrypt), process(index, pcm320).
Select with --plugin crypto.stft_overlap:STFTOverlap. No modifications to main.py.

16 kHz, 640-sample Hann window, 320-sample hop, 640-point rFFT;
analysis Hann, overlap-add synthesis and sum-of-Hann normalization (p=1),
matching the original streaming STFT design scaled to TimePack's frame size.
One hop of start-up latency on each side. RX mask index is offset by
one hop to account for the delayed TX overlap-add output. Gaps reset OLA.

CAUTION: Coefficient scrambling in an overlapping STFT is generally NOT
perfectly invertible when the intermediate result has been synthesized into
PCM and reanalysed. This preserves the original experiment, not a claim of
lossless recovery or Opus resistance.
"""

import hashlib

import numpy as np


class STFTOverlap:
    """One 320-sample PCM hop per call; outputs one 320-sample hop."""

    HOP = 320
    WINDOW_SIZE = 640
    OPERATIONS = ('invert', 'shuffle', 'signflip')

    def __init__(self, key: bytes, operation: str = 'signflip', decrypt: bool = False):
        if operation not in self.OPERATIONS:
            raise ValueError('STFTOverlap operation must be invert, shuffle or signflip')
        if operation != 'invert' and not key:
            raise ValueError('Key required for shuffle/signflip')
        self.key = bytes(key)
        self.operation = operation
        self.decrypt = bool(decrypt)
        # Nonzero-endpoint Hann is the original window_nonzero(640).
        self.window = np.hanning(self.WINDOW_SIZE + 2)[1:-1].astype(np.float64)
        self._clear()

    def _clear(self):
        self._previous = np.zeros(self.HOP, dtype=np.float64)
        self._ola = np.zeros(self.WINDOW_SIZE, dtype=np.float64)
        self._norm = np.zeros(self.WINDOW_SIZE, dtype=np.float64)
        self._expected_index = None

    def _rng(self, index):
        # Only 64 distinguishable PN indices are carried by TimePack V2.
        # Therefore the mask depends on index % 64 (also on late join).
        h = hashlib.sha512(self.key + (int(index) % 64).to_bytes(8, 'big')).digest()
        return np.random.default_rng(np.random.SeedSequence(
            np.frombuffer(h, dtype=np.uint32), pool_size=8))

    def _transform(self, spectrum, index):
        if self.operation == 'invert':
            # Same full-bin reversal as the original STFTCipher.
            return spectrum[::-1].copy()
        if self.operation == 'signflip':
            spectrum = spectrum.copy()
            mask = self._rng(index).choice(np.array([-1.0, 1.0]),
                                            size=spectrum.size - 2)
            spectrum[1:-1] *= mask
            return spectrum
        perm = self._rng(index).permutation(spectrum.size)
        # As in original STFTCipher, shuffle includes DC and Nyquist.
        return spectrum[np.argsort(perm)] if self.decrypt else spectrum[perm]

    def process(self, index: int, pcm320):
        index = int(index)
        frame = np.asarray(pcm320, dtype=np.float64)
        if index < 0 or frame.shape != (self.HOP,) or not np.all(np.isfinite(frame)):
            raise ValueError('Expected nonnegative index and 320 finite PCM samples')

        # Gap, backward jump or PN modulo-64 reacquisition: avoid mixing
        # unrelated chunks in the lapped overlap-add buffers.
        if self._expected_index is not None and index != self._expected_index:
            self._clear()
        self._expected_index = index + 1

        windowed = np.concatenate((self._previous, frame)) * self.window
        self._previous = frame.copy()
        spectrum = np.fft.rfft(windowed, n=self.WINDOW_SIZE)
        # TX emits its first encrypted content one hop later because of
        # 50% overlap. RX analyses that content one hop later as well;
        # match the corresponding TX spectral mask on the RX side.
        spectrum = self._transform(spectrum, index - (1 if self.decrypt else 0))
        reconstructed = np.fft.irfft(spectrum, n=self.WINDOW_SIZE)

        self._ola += reconstructed
        self._norm += self.window
        out = np.divide(self._ola[:self.HOP], self._norm[:self.HOP],
                        out=np.zeros(self.HOP, dtype=np.float64),
                        where=self._norm[:self.HOP] > 1e-12).copy()

        self._ola[:-self.HOP] = self._ola[self.HOP:]
        self._ola[-self.HOP:] = 0.0
        self._norm[:-self.HOP] = self._norm[self.HOP:]
        self._norm[-self.HOP:] = 0.0
        if not np.all(np.isfinite(out)):
            raise RuntimeError('Nonfinite STFT output')
        return np.clip(out, -1.0, 1.0).astype(np.float32)
