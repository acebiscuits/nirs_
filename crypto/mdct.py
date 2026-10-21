#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from .base import SpectralCipher


class MDCTCipher(SpectralCipher):
    """Потоковый MDCT + invert/shuffle/signflip + TDAC overlap-add.

    По умолчанию half_size=2500, т.е. frame_size=5000 и hop_size=2500,
    как в присланном исследовательском mdct.py.
    """

    SUPPORTED_OPERATIONS = {"invert", "shuffle", "signflip"}

    def __init__(
        self,
        operation: str,
        key: bytes | None = None,
        key_schedule: str = "per_frame",
        half_size: int = 2500,
    ):
        super().__init__(
            operation=operation,
            key=key,
            key_schedule=key_schedule,
        )

        if self.operation not in self.SUPPORTED_OPERATIONS:
            raise ValueError(
                f"MDCT operation must be one of {sorted(self.SUPPORTED_OPERATIONS)}"
            )

        if self.operation in {"shuffle", "signflip"}:
            self._require_key()

        if half_size <= 0:
            raise ValueError("half_size must be > 0")

        self.half_size = int(half_size)
        self.frame_size = 2 * self.half_size
        self.hop_size = self.half_size

        if self.frame_size % 4 != 0:
            raise ValueError("2 * half_size must be divisible by 4")

        self.window = np.sin(
            np.pi / self.frame_size * (np.arange(self.frame_size) + 0.5)
        ).astype(np.float64)

        self.input_buffer = np.empty(0, dtype=np.float64)
        self.ola_buffer = np.zeros(self.frame_size, dtype=np.float64)

    def _reset_stream_state(self) -> None:
        self.input_buffer = np.empty(0, dtype=np.float64)
        self.ola_buffer.fill(0.0)

    @staticmethod
    def _mdct4(x: np.ndarray) -> np.ndarray:
        N = x.shape[0]
        M = N // 2
        N4 = N // 4

        rot = np.roll(x, N4)
        rot[:N4] = -rot[:N4]

        t = np.arange(0, N4)
        w = np.exp(-1j * 2 * np.pi * (t + 1.0 / 8.0) / N)

        c = (
            np.take(rot, 2 * t)
            - np.take(rot, N - 2 * t - 1)
            - 1j * (
                np.take(rot, M + 2 * t)
                - np.take(rot, M - 2 * t - 1)
            )
        )

        c = (2.0 / np.sqrt(N)) * w * np.fft.fft(0.5 * c * w, N4)

        y = np.zeros(M, dtype=np.float64)
        y[2 * t] = np.real(c[t])
        y[M - 2 * t - 1] = -np.imag(c[t])
        return y

    @staticmethod
    def _imdct4(x: np.ndarray) -> np.ndarray:
        N = x.shape[0]
        M = N // 2
        N2 = N * 2

        t = np.arange(0, M)
        w = np.exp(-1j * 2 * np.pi * (t + 1.0 / 8.0) / N2)

        c = np.take(x, 2 * t) + 1j * np.take(x, N - 2 * t - 1)
        c = 0.5 * w * c
        c = np.fft.fft(c, M)
        c = ((8.0 / np.sqrt(N2)) * w) * c

        rot = np.zeros(N2, dtype=np.float64)
        rot[2 * t] = np.real(c[t])
        rot[N + 2 * t] = np.imag(c[t])

        odd = np.arange(1, N2, 2)
        rot[odd] = -rot[N2 - odd - 1]

        y = np.zeros(N2, dtype=np.float64)

        t1 = np.arange(0, 3 * M)
        y[t1] = rot[t1 + M]

        t2 = np.arange(3 * M, N2)
        y[t2] = -rot[t2 - 3 * M]

        return y

    def _modify_coefficients(self, coeffs: np.ndarray, decrypt: bool) -> np.ndarray:
        coeffs = np.asarray(coeffs).copy()

        if self.operation == "invert":
            return coeffs[::-1]

        if self.operation == "signflip":
            rng = self._rng_for_frame()
            mask = rng.choice(
                np.array([-1.0, 1.0]),
                size=coeffs.shape[0],
            ).astype(np.float64)
            return coeffs * mask

        if self.operation == "shuffle":
            rng = self._rng_for_frame()
            perm = rng.permutation(coeffs.shape[0])

            if decrypt:
                return coeffs[np.argsort(perm)]

            return coeffs[perm]

        raise RuntimeError(f"unknown MDCT operation: {self.operation}")

    def _process(self, samples: np.ndarray, decrypt: bool) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float64).reshape(-1)

        if samples.size == 0:
            return np.empty(0, dtype=np.float64)

        self.input_buffer = np.concatenate((self.input_buffer, samples))
        output_parts: list[np.ndarray] = []

        while self.input_buffer.size >= self.frame_size:
            frame = self.input_buffer[:self.frame_size]
            self.input_buffer = self.input_buffer[self.hop_size:]

            windowed = frame * self.window
            coeffs = self._mdct4(windowed)
            coeffs = self._modify_coefficients(coeffs, decrypt=decrypt)

            reconstructed = self._imdct4(coeffs)
            reconstructed *= self.window

            # TDAC overlap-add: готовая часть равна одному hop.
            self.ola_buffer += reconstructed
            ready = self.ola_buffer[:self.hop_size].copy()
            output_parts.append(ready)

            self.ola_buffer[:-self.hop_size] = self.ola_buffer[self.hop_size:]
            self.ola_buffer[-self.hop_size:] = 0.0

            self.frame_index += 1

        if not output_parts:
            return np.empty(0, dtype=np.float64)

        return np.concatenate(output_parts)

    def encrypt(self, samples: np.ndarray) -> np.ndarray:
        return self._process(samples, decrypt=False)

    def decrypt(self, samples: np.ndarray) -> np.ndarray:
        return self._process(samples, decrypt=True)
