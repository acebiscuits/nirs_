#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class SyncInfo:
    """Absolute frame marker. Type comes from chirp (not a nonce field)."""
    next_frame_index: int
    sequence: int = 0
    is_full: bool = True


class FrameSynchronizer:
    """Pure synchronization logic.

    This class decides WHEN a marker is needed and WHAT metadata it contains.
    It does not generate audio. audio_sync.py is responsible for turning
    SyncInfo into PCM and detecting it on RX.
    """

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
        return self._frames_for_period(
            cipher,
            self.sample_rate,
            self.sync_period_seconds,
        )

    def full_interval_frames(self, cipher) -> int:
        return self._frames_for_period(
            cipher,
            self.sample_rate,
            self.full_sync_period_seconds,
        )

    def _next_sequence(self) -> int:
        value = self._sequence
        self._sequence = (self._sequence + 1) & 0xFF
        return value

    def should_schedule_periodic(self, cipher, next_frame_index: int) -> bool:
        """Called after a frame has finished.

        If True, the marker must start on the NEXT output hop. Therefore the
        packet carries next_frame_index, not the frame that just finished.
        """
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
        """Apply a received sync packet.

        Normal periodic resync is SOFT: only the frame number is corrected and
        STFT/MDCT overlap buffers are preserved.

        hard=True is for initial lock or a real stream restart. It clears the
        transform streaming state and therefore may create a short transient.
        """
        if sync.next_frame_index < 0:
            raise ValueError("sync next_frame_index must be >= 0")

        if hard:
            cipher.reset(frame_index=sync.next_frame_index)
        else:
            cipher.set_frame_index(sync.next_frame_index)
