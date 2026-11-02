"""Offline SuperSpinner 8x8 regression suite. No Telemost/PulseAudio required."""
import time
import numpy as np
from v2_timepack import FixedGuardSyncV1, SyncV1Config, makeGuardBank

RATE, CORE, FRAME = 20000, 320, 400


def generate(frames=600):
    rng = np.random.default_rng(20261018)
    bank = makeGuardBank(20000, 80, 64)
    original = np.empty((frames, FRAME), np.float32)
    for i in range(frames):
        core = rng.standard_normal(CORE).astype(np.float32) * 0.05
        original[i, :CORE] = core
        original[i, CORE:] = bank[i % 64] * 0.12
    return original


def run(x, clockJump=False):
    found, events = {}, []
    config = SyncV1Config(rate=RATE, frameSamples=FRAME, coreSamples=CORE, guardSamples=80, superSpinnerSize=8)
    sync = FixedGuardSyncV1(config, lambda i, data, status, meta: found.__setitem__(i, (data.copy(), status)), lambda e: events.append(e))
    x = np.ascontiguousarray(x.reshape(-1), dtype=np.float32)
    clock = 0.0
    start = time.perf_counter()
    for pos in range(0, len(x), 200):
        chunk = x[pos:pos + 200]
        clock += len(chunk) / RATE
        # Simulate the old wall-clock bug: after 5 s the clock jumps by 2 s.
        stamp = clock + (2.0 if clockJump and clock >= 5 else 0.0)
        sync.push(chunk, stamp)
    sync.finish()
    elapsed = time.perf_counter() - start
    return sync, found, events, elapsed


def check(label, tx, wire, expectedLoss, maxMisses=0, clockJump=False):
    sync, decoded, events, elapsed = run(wire, clockJump)
    got = sync.stats
    assert sync.lastFrame == len(tx) - 1, f"{label}: frameIndex drift {sync.lastFrame}!={len(tx)-1}"
    assert got['frames_lost'] == expectedLoss, f"{label}: expected {expectedLoss} lost, got {got['frames_lost']}"
    assert got['track_misses'] <= maxMisses, f"{label}: unexpected track_misses {got['track_misses']}"
    # Check recovered payload frame identity at every intact frame, not only modulo 64.
    bad = []
    for i, (y, status) in decoded.items():
        if status != 'ok':
            continue
        if np.corrcoef(y[:CORE], tx[i, :CORE])[0, 1] < 0.985:
            bad.append(i)
    assert not bad, f"{label}: wrong frame identity in {bad[:10]}"
    print(f"PASS {label:<18} frames={len(decoded):4d}, lost={got['frames_lost']:3d}, misses={got['track_misses']:2d}, elapsed={elapsed:.3f}s")
    return elapsed


def main():
    tx = generate()
    zero8 = np.zeros((8, FRAME), np.float32)
    zero16 = np.zeros((16, FRAME), np.float32)
    times = []
    times.append(check('clean', tx, tx, 0))
    times.append(check('wall_clock_jump', tx, tx, 0, clockJump=True))
    times.append(check('drop8', tx, np.concatenate((tx[:200], tx[208:])), 8))
    times.append(check('drop16', tx, np.concatenate((tx[:200], tx[216:])), 16))
    times.append(check('insert8', tx, np.concatenate((tx[:200], zero8, tx[200:])), 0))
    times.append(check('insert16', tx, np.concatenate((tx[:200], zero16, tx[200:])), 0))
    damaged = tx.copy()
    damaged[100:105, CORE:] = 0
    damaged[310:314, :CORE] = 0
    # Ignore payload corruption in the intentionally damaged frames.
    sync, decoded, _, elapsed = run(damaged)
    assert sync.lastFrame == 599 and sync.stats['track_misses'] == 0
    print(f"PASS {'damaged_guards':<18} frames={len(decoded):4d}, lost={sync.stats['frames_lost']:3d}, misses={sync.stats['track_misses']:2d}, elapsed={elapsed:.3f}s")
    print(f"ALL TESTS PASSED. Clean streaming speed: {12/times[0]:.1f}x realtime on this machine.")


if __name__ == '__main__':
    main()