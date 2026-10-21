#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class SyncInfo:
    next_frame_index: int
    sequence: int = 0
    is_full: bool = True


class FrameSynchronizer:
    def __init__(
        self,
        sample_rate: int,
        sync_period_seconds: float = 0.5,
        full_sync_period_seconds: float = 5.0,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if sync_period_seconds <= 0:
            raise ValueError("sync_period_seconds must be > 0")
        if full_sync_period_seconds <= 0:
            raise ValueError("full_sync_period_seconds must be > 0")
        if full_sync_period_seconds < sync_period_seconds:
            raise ValueError("full_sync_period_seconds must be >= sync_period_seconds")

        self.sample_rate = int(sample_rate)
        self.sync_period_seconds = float(sync_period_seconds)
        self.full_sync_period_seconds = float(full_sync_period_seconds)
        self._sequence = 0

    @staticmethod
    def _frames_for_period(cipher, sample_rate: int, seconds: float) -> int:
        return max(1, round(seconds * sample_rate / cipher.hop_size))

    def periodic_interval_frames(self, cipher) -> int:
        return self._frames_for_period(cipher, self.sample_rate, self.sync_period_seconds)

    def full_interval_frames(self, cipher) -> int:
        return self._frames_for_period(cipher, self.sample_rate, self.full_sync_period_seconds)

    def _next_sequence(self) -> int:
        value = self._sequence
        self._sequence = (self._sequence + 1) & 0xFF
        return value

    def should_schedule_periodic(self, cipher, next_frame_index: int) -> bool:
        if next_frame_index <= 0:
            return False
        return next_frame_index % self.periodic_interval_frames(cipher) == 0

    def should_schedule_full(self, cipher, next_frame_index: int) -> bool:
        if next_frame_index <= 0:
            return False
        return next_frame_index % self.full_interval_frames(cipher) == 0

    def make_full_sync(self, cipher, next_frame_index: int | None = None) -> SyncInfo:
        if next_frame_index is None:
            next_frame_index = cipher.frame_index
        return SyncInfo(
            next_frame_index=next_frame_index,
            sequence=self._next_sequence(),
            is_full=True,
        )

    def make_periodic_sync(self, next_frame_index: int) -> SyncInfo:
        return SyncInfo(
            next_frame_index=next_frame_index,
            sequence=self._next_sequence(),
            is_full=False,
        )

    @staticmethod
    def apply_sync(cipher, sync: SyncInfo, hard: bool = False) -> None:
        if sync.next_frame_index < 0:
            raise ValueError("sync next_frame_index must be >= 0")

        if hard:
            cipher.reset(frame_index=sync.next_frame_index)
        else:
            cipher.set_frame_index(sync.next_frame_index)
