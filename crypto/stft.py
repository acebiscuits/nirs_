#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from .base import SpectralCipher


class STFTCipher(SpectralCipher):
    """Потоковый STFT + invert/shuffle/signflip + overlap-add.

    Параметры по умолчанию соответствуют исследовательскому варианту:
    frame_size=2048, hop_size=1024, fft_size=2048.
    """

    SUPPORTED_OPERATIONS = {"invert", "shuffle", "signflip"}

    def __init__(
        self,
        operation: str,
        key: bytes | None = None,
        session_nonce: str = "stft-live-v1",
        key_schedule: str = "per_frame",#опции: fixed, per_frame
        frame_size: int = 2048,
        hop_size: int = 1024,
        fft_size: int | None = None,
    ):
        super().__init__(
            operation=operation,
            key=key,
            session_nonce=session_nonce,
            key_schedule=key_schedule,
        )

        if self.operation not in self.SUPPORTED_OPERATIONS:
            raise ValueError(
                f"STFT operation must be one of {sorted(self.SUPPORTED_OPERATIONS)}"
            )

        if self.operation in {"shuffle", "signflip"}:
            self._require_key()

        if frame_size <= 0:
            raise ValueError("frame_size must be > 0")
        if hop_size <= 0 or hop_size > frame_size:
            raise ValueError("hop_size must be in range 1..frame_size")

        self.frame_size = int(frame_size)
        self.hop_size = int(hop_size)
        self.fft_size = int(fft_size if fft_size is not None else frame_size)

        if self.fft_size < self.frame_size:
            raise ValueError("fft_size must be >= frame_size")

        self.window = self._window_nonzero(self.frame_size)

        self.input_buffer = np.empty(0, dtype=np.float64)
        self.ola_buffer = np.zeros(self.frame_size, dtype=np.float64)
        self.norm_buffer = np.zeros(self.frame_size, dtype=np.float64)

    @staticmethod
    def _window_nonzero(frame_size: int) -> np.ndarray:
        """Аналог window_nonzero() из исследовательского stft.py."""
        extra = 0

        while True:
            window = np.hanning(frame_size + extra)

            if extra:
                start = extra // 2
                window = window[start:start + frame_size]

            zero_count = window.size - np.count_nonzero(window)
            if zero_count == 0:
                return window.astype(np.float64)

            extra += zero_count

    def _reset_stream_state(self) -> None:
        self.input_buffer = np.empty(0, dtype=np.float64)
        self.ola_buffer.fill(0.0)
        self.norm_buffer.fill(0.0)

    def _modify_spectrum(self, spectrum: np.ndarray, decrypt: bool) -> np.ndarray:
        spectrum = np.asarray(spectrum).copy()

        if self.operation == "invert":
            return spectrum[::-1]

        if self.operation == "signflip":
            # Как в исследовательской STFT-версии: DC и Nyquist не трогаем.
            if spectrum.shape[0] <= 2:
                return spectrum

            rng = self._rng_for_frame("stft-signflip")
            mask = rng.choice(
                np.array([-1.0, 1.0]),
                size=spectrum.shape[0] - 2,
            ).astype(np.float64)

            spectrum[1:-1] *= mask
            return spectrum

        if self.operation == "shuffle":
            rng = self._rng_for_frame("stft-shuffle")
            perm = rng.permutation(spectrum.shape[0])

            if decrypt:
                return spectrum[np.argsort(perm)]

            return spectrum[perm]
        
        # if self.operation == "shuffle": - тут аналог того что выше, но с сохранением dc и Nyquist
        #     if spectrum.shape[0] <= 2:
        #         return spectrum

        #     rng = self._rng_for_frame(
        #         "stft-shuffle"
        #     )

        #     inner_perm = rng.permutation(
        #         spectrum.shape[0] - 2
        #     )

        #     perm = np.arange(
        #         spectrum.shape[0]
        #     )

        #     perm[1:-1] = (
        #         inner_perm + 1
        #     )
        #     if decrypt:
        #         return spectrum[
        #             np.argsort(perm)
        #         ]

        #     return spectrum[perm]

        raise RuntimeError(f"unknown STFT operation: {self.operation}")

    def _process(self, samples: np.ndarray, decrypt: bool) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float64).reshape(-1)

        if samples.size == 0:
            return np.empty(0, dtype=np.float64)

        self.input_buffer = np.concatenate((self.input_buffer, samples))
        output_parts: list[np.ndarray] = []

        while self.input_buffer.size >= self.frame_size:
            frame = self.input_buffer[:self.frame_size]
            self.input_buffer = self.input_buffer[self.hop_size:]

            # Analysis window -> FFT.
            windowed = frame * self.window
            spectrum = np.fft.rfft(windowed, n=self.fft_size)

            # Собственно спектральное преобразование.
            spectrum = self._modify_spectrum(spectrum, decrypt=decrypt)

            # IFFT. При p=1 в твоем исходном STFT синтезирующее окно
            # фактически window**0, т.е. повторно окно здесь не умножаем.
            reconstructed = np.fft.irfft(spectrum, n=self.fft_size)[:self.frame_size]

            # Streaming overlap-add + та же нормализация sum(window),
            # которая использовалась в исследовательском istft().
            self.ola_buffer += reconstructed
            self.norm_buffer += self.window

            denom = self.norm_buffer[:self.hop_size]
            ready = np.divide(
                self.ola_buffer[:self.hop_size],
                denom,
                out=np.zeros(self.hop_size, dtype=np.float64),
                where=np.abs(denom) > 1e-12,
            )
            output_parts.append(ready.copy())

            self.ola_buffer[:-self.hop_size] = self.ola_buffer[self.hop_size:]
            self.ola_buffer[-self.hop_size:] = 0.0

            self.norm_buffer[:-self.hop_size] = self.norm_buffer[self.hop_size:]
            self.norm_buffer[-self.hop_size:] = 0.0

            self.frame_index += 1

        if not output_parts:
            return np.empty(0, dtype=np.float64)

        return np.concatenate(output_parts)

    def encrypt(self, samples: np.ndarray) -> np.ndarray:
        return self._process(samples, decrypt=False)

    def decrypt(self, samples: np.ndarray) -> np.ndarray:
        return self._process(samples, decrypt=True)
