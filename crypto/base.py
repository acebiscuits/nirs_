#!/usr/bin/env python3
from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib

import numpy as np


class SpectralCipher(ABC):
    """Общий контракт для потоковых спектральных преобразований.

    Один экземпляр используется только в одном направлении потока:
    TX -> encrypt() или RX -> decrypt(). Это важно, потому что внутри
    конкретных реализаций есть streaming-буферы и frame_index.
    """

    SUPPORTED_KEY_SCHEDULES = {"fixed", "per_frame"}

    def __init__(
        self,
        operation: str,
        key: bytes | None = None,
        session_nonce: str = "live-v1",
        key_schedule: str = "per_frame",
    ):
        if key_schedule not in self.SUPPORTED_KEY_SCHEDULES:
            raise ValueError(
                f"key_schedule must be one of {sorted(self.SUPPORTED_KEY_SCHEDULES)}"
            )

        self.operation = operation.lower()
        self.key = key
        self.session_nonce = session_nonce
        self.key_schedule = key_schedule
        self.frame_index = 0

    @abstractmethod
    def encrypt(self, samples: np.ndarray) -> np.ndarray:
        """Принять очередной PCM chunk и вернуть 0..N обработанных samples."""
        raise NotImplementedError

    @abstractmethod
    def decrypt(self, samples: np.ndarray) -> np.ndarray:
        """Принять очередной PCM chunk и вернуть 0..N восстановленных samples."""
        raise NotImplementedError

    @abstractmethod
    def _reset_stream_state(self) -> None:
        """Сброс внутренних streaming-буферов конкретного преобразования."""
        raise NotImplementedError

    def reset(self, frame_index: int = 0, session_nonce: str | None = None) -> None:
        """Полностью сбросить потоковое состояние и установить номер кадра."""
        

        if frame_index < 0:
            raise ValueError(
                "frame_index must be >= 0"
            )

        if session_nonce is not None:
            self.session_nonce = session_nonce

        self.frame_index = frame_index
        self._reset_stream_state()

    def set_frame_index(self, frame_index: int) -> None:
        """Изменить только счетчик кадра.

        Использовать это можно только на известной границе спектрального кадра.
        Метод оставлен как hook для будущей initial/periodic resync.
        """
        if frame_index < 0:
            raise ValueError("frame_index must be >= 0")

        self.frame_index = frame_index

    def _require_key(self) -> None:
        if not self.key:
            raise ValueError(f"operation '{self.operation}' requires key bytes")

    def _rng_for_frame(self, domain: str) -> np.random.Generator:
        """Детерминированный RNG для fixed или конкретного frame_index.

        per_frame специально сделан random-access: зная абсолютный frame_index,
        RX может сразу восстановить нужную mask/permutation после resync,
        не прокручивая RNG от начала звонка.
        """
        self._require_key()

        effective_frame = self.frame_index if self.key_schedule == "per_frame" else 0

        h = hashlib.sha512()
        h.update(self.key)
        h.update(self.session_nonce.encode("utf-8"))
        h.update(domain.encode("utf-8"))
        # h.update(str(effective_frame).encode("ascii"))
        h.update(effective_frame.to_bytes(8, byteorder="big", signed=False)) # - это якобы однозначное кодирование с помощью 8 байт

        seed_words = np.frombuffer(h.digest(), dtype=np.uint32)
        seed_seq = np.random.SeedSequence(seed_words, pool_size=8)
        return np.random.default_rng(seed_seq)
