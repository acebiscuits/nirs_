"""Experimental in-band synchronization marker for the audio path.

The marker is OVERLAID on encrypted PCM instead of inserted as extra samples.
That is important: inserting extra samples into a real-time stream would make
latency grow forever because TX would output audio faster than real time.

Packet protection:
- a correlation preamble tells RX where a marker starts;
- CRC-16 rejects corrupted metadata;
- absolute next_frame_index allows direct resync;
- sequence helps diagnostics/duplicate handling;

This is a prototype physical layer. It must be tested through the actual Opus
path before being treated as reliable. No lossy audio channel can guarantee
zero packet loss; CRC + periodic retransmission make failures detectable and
self-correcting rather than silently applying a wrong frame number.
"""

from __future__ import annotations

import binascii
from dataclasses import dataclass

import numpy as np

from sync import SyncInfo


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray


class AudioSyncCodec:
    FULL_HEADER = 0xA1
    PERIODIC_HEADER = 0xA0

    def __init__(
        self,
        sample_rate: int = 44100,
        marker_amplitude: float = 0.30,
        symbol_ms: float = 4.0,
        preamble_ms: float = 24.0,
        detection_threshold: float = 0.52,
    ):
        self.sample_rate = int(sample_rate)
        self.marker_amplitude = float(marker_amplitude)
        self.symbol_samples = max(32, round(sample_rate * symbol_ms / 1000.0))
        self.preamble_samples = max(128, round(sample_rate * preamble_ms / 1000.0))
        self.detection_threshold = float(detection_threshold)

        # Two simultaneous tones encode one hexadecimal digit (4 bits):
        # 2 bits choose one low tone, 2 bits choose one high tone.
        self.low_freqs = np.array([1200.0, 1600.0, 2000.0, 2400.0])
        self.high_freqs = np.array([3200.0, 3800.0, 4400.0, 5000.0])

        self._preamble = self._build_preamble()
        self._symbol_bank = [self._build_symbol(i) for i in range(16)]

    def _build_preamble(self) -> np.ndarray:
        n = self.preamble_samples
        t = np.arange(n, dtype=np.float64) / self.sample_rate
        duration = n / self.sample_rate
        f0 = 900.0
        f1 = 6200.0
        k = (f1 - f0) / duration#скорость изменения чатсоты
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)#тут считается просто фаза синусоиды
        window = np.hanning(n)
        return self.marker_amplitude * np.sin(phase) * window

    def _build_symbol(self, nibble: int) -> np.ndarray:
        row = (nibble >> 2) & 0x03
        col = nibble & 0x03
        n = self.symbol_samples
        t = np.arange(n, dtype=np.float64) / self.sample_rate

        # Keep the two-tone sum bounded. A short Hann ramp reduces clicks.
        tone = (
            np.sin(2.0 * np.pi * self.low_freqs[row] * t)
            + np.sin(2.0 * np.pi * self.high_freqs[col] * t)
        )
        tone *= self.marker_amplitude * 0.5

        ramp = min(max(4, n // 12), n // 2)
        env = np.ones(n, dtype=np.float64)
        if ramp > 1:
            edge = np.sin(np.linspace(0.0, np.pi / 2.0, ramp)) ** 2
            env[:ramp] = edge
            env[-ramp:] = edge[::-1]

        return tone * env

    @staticmethod
    def _crc16(data: bytes) -> int:
        return binascii.crc_hqx(data, 0xFFFF)

    def _pack(self, info: SyncInfo) -> bytes:
        if info.next_frame_index < 0 or info.next_frame_index >= 2**32:
            raise ValueError("next_frame_index must fit uint32")

        header = self.FULL_HEADER if info.is_full else self.PERIODIC_HEADER
        body = bytearray()
        body.append(header)
        body.append(info.sequence & 0xFF)
        body.extend(info.next_frame_index.to_bytes(4, "big", signed=False))


        crc = self._crc16(bytes(body))
        body.extend(crc.to_bytes(2, "big"))
        return bytes(body)

    def _unpack(self, packet: bytes) -> SyncInfo:
        if len(packet) < 8:
            raise ValueError("sync packet is too short")

        data = packet[:-2]
        expected_crc = int.from_bytes(packet[-2:], "big")
        actual_crc = self._crc16(data)
        if actual_crc != expected_crc:
            raise ValueError("sync CRC mismatch")

        header = data[0]
        sequence = data[1]
        next_frame_index = int.from_bytes(data[2:6], "big")

        if header == self.PERIODIC_HEADER:
            if len(packet) != self.periodic_packet_bytes:
                raise ValueError("invalid periodic sync length")
            return SyncInfo(
                next_frame_index=next_frame_index,
                sequence=sequence,
                is_full=False,
            )

        if header == self.FULL_HEADER:
            if len(packet) != self.full_packet_bytes:
                raise ValueError("invalid full sync length")
            return SyncInfo(
                next_frame_index=next_frame_index,
                sequence=sequence,
                is_full=True,
            )

        raise ValueError("unknown sync header")

    @property
    def periodic_packet_bytes(self) -> int:
        # header + seq + uint32 frame + uint16 crc
        return 1 + 1 + 4 + 2

    @property
    def full_packet_bytes(self) -> int:
        return self.periodic_packet_bytes

    def marker_samples_for(self, info: SyncInfo) -> int:
        packet_bytes = self.full_packet_bytes if info.is_full else self.periodic_packet_bytes
        return self.preamble_samples + packet_bytes * 2 * self.symbol_samples

    def encode(self, info: SyncInfo) -> np.ndarray:
        packet = self._pack(info)
        parts = [self._preamble]

        for byte in packet:
            parts.append(self._symbol_bank[(byte >> 4) & 0x0F])
            parts.append(self._symbol_bank[byte & 0x0F])

        return np.concatenate(parts).astype(np.float64, copy=False)

    def _decode_symbol(self, symbol: np.ndarray) -> int:
        if symbol.size != self.symbol_samples:
            raise ValueError("bad symbol length")

        # Normalized correlation against all 16 known symbols. Because the
        # marker is overlaid on encrypted speech, the wanted symbol acts as a
        # deterministic component in additional broadband noise.
        x = symbol.astype(np.float64, copy=False)
        x_norm = np.linalg.norm(x) + 1e-12

        best_idx = 0
        best_score = -np.inf
        for idx, template in enumerate(self._symbol_bank):
            score = abs(float(np.dot(x, template))) / (
                x_norm * (np.linalg.norm(template) + 1e-12)
            )
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _decode_bytes(self, samples: np.ndarray, byte_count: int) -> bytes:
        needed = byte_count * 2 * self.symbol_samples
        if samples.size < needed:
            raise ValueError("not enough samples for sync payload")

        out = bytearray()
        pos = 0
        for _ in range(byte_count):
            hi = self._decode_symbol(samples[pos:pos + self.symbol_samples])
            pos += self.symbol_samples
            lo = self._decode_symbol(samples[pos:pos + self.symbol_samples])
            pos += self.symbol_samples
            out.append((hi << 4) | lo)

        return bytes(out)

    def _candidate_packet_size(self, samples_after_preamble: np.ndarray) -> int | None:
        if samples_after_preamble.size < 2 * self.symbol_samples:
            return None

        hi = self._decode_symbol(samples_after_preamble[:self.symbol_samples])
        lo = self._decode_symbol(
            samples_after_preamble[self.symbol_samples:2 * self.symbol_samples]
        )
        header = (hi << 4) | lo

        if header == self.FULL_HEADER:
            return self.full_packet_bytes
        if header == self.PERIODIC_HEADER:
            return self.periodic_packet_bytes
        return -1

    def decode_exact_marker(self, marker: np.ndarray) -> SyncInfo:
        marker = np.asarray(marker, dtype=np.float64).reshape(-1)
        if marker.size < self.preamble_samples + 2 * self.symbol_samples:
            raise ValueError("marker is too short")

        payload = marker[self.preamble_samples:]
        packet_size = self._candidate_packet_size(payload)
        if packet_size is None or packet_size < 0:
            raise ValueError("unknown sync marker header")

        packet = self._decode_bytes(payload, packet_size)
        return self._unpack(packet)

    def find_preamble(self, samples: np.ndarray) -> tuple[int | None, float]:
        samples = np.asarray(samples, dtype=np.float64).reshape(-1)
        n = self.preamble_samples
        if samples.size < n:
            return None, 0.0

        template = self._preamble
        template_energy = float(np.dot(template, template)) + 1e-12
        corr = np.correlate(samples, template, mode="valid")

        squared = samples * samples
        energy = np.convolve(squared, np.ones(n, dtype=np.float64), mode="valid")
        denom = np.sqrt(energy * template_energy) + 1e-12
        score = corr / denom

        idx = int(np.argmax(score))
        best = float(score[idx])
        if best < self.detection_threshold:
            return None, best
        return idx, best


class AudioSyncStreamDecoder:
    """Streaming marker detector that preserves audio sample count.

    It returns an ordered list containing AudioData chunks and SyncInfo events.
    Marker samples are not deleted. After a packet is decoded, the locally
    generated marker is subtracted from that region, so a perfect channel gives
    the original encrypted PCM back exactly (apart from floating error).
    """

    def __init__(self, codec: AudioSyncCodec):
        self.codec = codec
        self.buffer = np.empty(0, dtype=np.float64)

    def feed(self, samples: np.ndarray) -> list[AudioData | SyncInfo]:
        samples = np.asarray(samples, dtype=np.float64).reshape(-1)
        if samples.size:
            self.buffer = np.concatenate((self.buffer, samples))

        events: list[AudioData | SyncInfo] = []
        keep = self.codec.preamble_samples - 1

        while True:
            if self.buffer.size < self.codec.preamble_samples:
                break

            pos, _ = self.codec.find_preamble(self.buffer)

            if pos is None:
                safe = self.buffer.size - keep
                if safe > 0:
                    events.append(AudioData(self.buffer[:safe].copy()))
                    self.buffer = self.buffer[safe:]
                break

            if pos > 0:
                events.append(AudioData(self.buffer[:pos].copy()))
                self.buffer = self.buffer[pos:]
                continue

            # Candidate starts at buffer[0]. Determine packet type from header.
            min_header = self.codec.preamble_samples + 2 * self.codec.symbol_samples
            if self.buffer.size < min_header:
                break

            payload = self.buffer[self.codec.preamble_samples:]
            packet_bytes = self.codec._candidate_packet_size(payload)
            if packet_bytes is None:
                break
            if packet_bytes < 0:
                # False preamble. Release one sample and keep searching.
                events.append(AudioData(self.buffer[:1].copy()))
                self.buffer = self.buffer[1:]
                continue

            marker_len = (
                self.codec.preamble_samples
                + packet_bytes * 2 * self.codec.symbol_samples
            )
            if self.buffer.size < marker_len:
                break

            marker_rx = self.buffer[:marker_len].copy()
            try:
                info = self.codec.decode_exact_marker(marker_rx)
            except ValueError:
                # Corrupted/false packet: never apply a guessed frame number.
                events.append(AudioData(self.buffer[:1].copy()))
                self.buffer = self.buffer[1:]
                continue

            marker_local = self.codec.encode(info)
            cleaned = marker_rx - marker_local

            # Order matters: apply sync BEFORE cleaned marker-region audio is
            # passed to the spectral decryptor.
            events.append(info)
            events.append(AudioData(cleaned))
            self.buffer = self.buffer[marker_len:]

        return events

    def flush(self) -> list[AudioData]:
        if self.buffer.size == 0:
            return []
        out = [AudioData(self.buffer.copy())]
        self.buffer = np.empty(0, dtype=np.float64)
        return out