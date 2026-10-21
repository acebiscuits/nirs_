from abc import ABC, abstractmethod
import hashlib

import numpy as np


class SpectralCipher(ABC):
    SUPPORTED_KEY_SCHEDULES = {"fixed", "per_frame"}

    def __init__(
        self,
        operation: str,
        key: bytes,
        key_schedule: str = "fixed",
    ):
        if key_schedule not in self.SUPPORTED_KEY_SCHEDULES:
            raise ValueError(
                f"key_schedule must be one of {sorted(self.SUPPORTED_KEY_SCHEDULES)}"
            )

        self.operation = operation.lower()
        self.key = key
        self.key_schedule = key_schedule
        self.frame_index = 0

    @abstractmethod
    def encrypt(self, samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def decrypt(self, samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _reset_stream_state(self) -> None:
        raise NotImplementedError

    def reset(self, frame_index: int = 0) -> None:
        if frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        self.frame_index = frame_index
        self._reset_stream_state()

    def set_frame_index(self, frame_index: int) -> None:
        if frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        self.frame_index = frame_index

    def _require_key(self) -> None:
        if not self.key:
            raise ValueError(f"operation '{self.operation}' requires key bytes")

    def _rng_for_frame(self) -> np.random.Generator:
        self._require_key()
        effective_frame = self.frame_index if self.key_schedule == "per_frame" else 0

        h = hashlib.sha512()
        h.update(self.key)
        h.update(effective_frame.to_bytes(8, byteorder="big", signed=False))

        seed_words = np.frombuffer(h.digest(), dtype=np.uint32)
        seed_seq = np.random.SeedSequence(seed_words, pool_size=8)
        return np.random.default_rng(seed_seq)
