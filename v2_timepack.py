from dataclasses import dataclass
import time
import numpy as np
from scipy.signal import find_peaks, firwin
try:
    from scipy.fft import rfft, irfft, next_fast_len
except ImportError:
    from numpy.fft import rfft, irfft
    try:
        from scipy.fftpack import next_fast_len
    except ImportError:
        def next_fast_len(n):
            return 1 << (int(n) - 1).bit_length()
# API class names stay V1-compatible; this file is the V2 implementation.
_GUARD_SEEDS = [30015, 34260, 34763, 30724, 33147, 32761, 31392, 34460]
_SUPER_GUARD_SEEDS = [50419, 50861, 50480, 50640, 50845, 50066, 50891, 50624, 50064, 50770, 50653, 50481, 50785, 50305, 50253, 50154, 50670, 50698, 50097, 50393, 50085, 50656, 50853, 50897, 50472, 50896, 50712, 50833, 50408, 50939, 50635, 50098, 50921, 50241, 50963, 50375, 50747, 50639, 50682, 50959, 50035, 50637, 50722, 50884, 50675, 50623, 50037, 50730, 50311, 50353, 50791, 50511, 50029, 50892, 50379, 50317]
# 64 distinct matched PN patterns: 8 fast positions x 8 slower super-phases.
# All 64 templates are deterministic and were greedily selected for low cross-correlation.

@dataclass
class SyncV1Config:
    rate: int = 16000
    frameSamples: int = 320
    coreSamples: int = 240
    guardSamples: int = 80
    spinnerSize: int = 8
    superSpinnerSize: int = 8  # 1 = original 8-phase protocol; 8 = 64-state super spinner
    acquireThreshold: float = 0.28
    trackThreshold: float = 0.2
    acquireConfirm: int = 8
    reacquireConfirm: int = 3
    trackRadius: int = 72
    trackRecoveryRadius: int = 128
    trainRadius: int = 8
    trackConfirm: int = 3
    trackContinuityRadius: int = 16
    trackChainOffsetSpread: int = 12
    trackLookaheadFrames: int = 20
    trackStrongThreshold: float = 0.70
    maxTimingJump: int = 24
    maxCoreAdjust: int = 64
    periodAlpha: float = 0.02
    gapRecoveryInterpolationOnlyFrames: int = 4
    gapRecoveryConfirm: int = 3
    gapRecoveryCoherence: int = 8
    gapRecoveryMargin: float = 0.04
    gapRecoveryThreshold: float = 0.28
    gapRecoveryRadius: int = 16
    gapRecoveryMaxFrames: int = 64
    gapRecoveryAllowEstimated: bool = True
    gapRecoveryMaxAnchorPredictionError: float = 1.0
    wallAlpha: float = 0.08
    searchWindowFrames: int = 32
    acquireScanStrideSamples: int = 640  # search new PCM candidate positions every 40 ms
    maxPredictionErrorFrames: float = 3.5
    codeMargin: float = 0.04
    phaseSwitchMargin: float = 0.08
    # Discontinuities <=24 physical frames are in scope; 64-frame global slips are not.
    maxAnchorDiscontinuityFrames: int = 24
    # A weak chain must agree with physical sample timing before it can update frameIndex.
    minWideAnchorCorr: float = 0.38

def makeGuardBank(rate=16000, samples=80, count=8):
    seeds = _GUARD_SEEDS + _SUPER_GUARD_SEEDS
    if count < 2 or count > len(seeds):
        raise ValueError(f'guard count must be from 2 to {len(seeds)}')
    taps = firwin(31, [300, 3400], pass_zero=False, fs=rate)
    guards = []
    for seed in seeds[:count]:
        rng = np.random.default_rng(seed)
        raw = rng.choice((-1.0, 1.0), size=samples + 30)
        x = np.convolve(raw, taps, mode='same')[15:15 + samples]
        ramp = min(6, samples // 4)
        if ramp:
            r = np.sin(np.linspace(0, np.pi / 2, ramp)) ** 2
            w = np.ones(samples)
            w[:ramp] = r
            w[-ramp:] = r[::-1]
            x *= w
        x /= np.sqrt(np.mean(x * x) + 1e-12)
        guards.append(x.astype(np.float32))
    return np.stack(guards)

def makeGuard(rate=16000, samples=80):
    return makeGuardBank(rate, samples, 2)[0]

class _SampleBuffer:

    def __init__(self):
        self.base = 0
        self.data = np.empty(0, dtype=np.float32)

    @property
    def end(self):
        return self.base + len(self.data)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        if len(x):
            self.data = np.concatenate((self.data, x))

    def slice(self, start, end):
        if start < self.base or end > self.end or end < start:
            return None
        return self.data[start - self.base:end - self.base]

    def discardBefore(self, index):
        index = min(max(index, self.base), self.end)
        n = index - self.base
        if n:
            self.data = self.data[n:]
            self.base = index

class FixedGuardSyncV1:

    def __init__(self, cfg=None, onFrame=None, onEvent=None):
        self.cfg = cfg or SyncV1Config()
        if self.cfg.rate != 20000:
            raise ValueError('Timepack experimental sync expects 20000 Hz')
        if self.cfg.coreSamples + self.cfg.guardSamples != self.cfg.frameSamples:
            raise ValueError('coreSamples + guardSamples must equal frameSamples')
        if self.cfg.spinnerSize != 8 or self.cfg.superSpinnerSize not in (1, 8):
            raise ValueError("super-spinner requires spinnerSize=8 and superSpinnerSize=1 or 8")
        self.guardCycle = self.cfg.spinnerSize * self.cfg.superSpinnerSize
        self.guards = makeGuardBank(self.cfg.rate, self.cfg.guardSamples, self.guardCycle)
        self.detectorTaps = firwin(31, [300, 3400], pass_zero=False, fs=self.cfg.rate).astype(np.float32)
        self.detectorHalf = (len(self.detectorTaps) - 1) // 2
        self.detectorGuards = np.stack([np.convolve(guard, self.detectorTaps, mode='same') for guard in self.guards]).astype(np.float32)
        self.detectorGuardNorms = np.linalg.norm(self.detectorGuards, axis=1)
        self._fftBankCache = {}
        self.onFrame = onFrame or (lambda *args: None)
        self.onEvent = onEvent or (lambda *args: None)
        self.buf = _SampleBuffer()
        self.state = 'SEARCH'
        self.lastGuard = None
        self.lastFrame = -1
        self.periodEstimate = float(self.cfg.frameSamples)
        self.epochWall = None
        self.tailIndex = 0
        self.tailWall = None
        self.lastReacquireSearchEnd = -1
        self.lastAcquireCheckedHi = None
        self.lastReacquireCheckedHi = None
        self.stats = {'frames_ok': 0, 'frames_lost': 0, 'acquisitions': 0, 'reacquisitions': 0, 'track_misses': 0, 'spinner_mismatches': 0}

    def push(self, samples, tailWall=None):
        samples = np.asarray(samples, dtype=np.float32)
        if not len(samples):
            return
        self.buf.append(samples)
        self.tailIndex = self.buf.end
        self.tailWall = time.monotonic() if tailWall is None else tailWall
        while True:
            if self.state == 'SEARCH':
                progressed = self._acquire()
            elif self.state == 'LOCKED':
                progressed = self._track()
            else:
                progressed = self._reacquire()
            if not progressed:
                break
        self._prune()

    def finish(self):
        half = self.detectorHalf
        if half > 0:
            self.buf.append(np.zeros(half, dtype=np.float32))
            if self.tailWall is not None:
                self.tailWall += half / self.cfg.rate
            self.tailIndex = self.buf.end
        while self.state == 'LOCKED':
            if not self._track(final=True):
                break

    def _wallAt(self, sampleIndex):
        if self.tailWall is None:
            return None
        return self.tailWall - (self.tailIndex - sampleIndex) / self.cfg.rate

    def _predictFrame(self, wall):
        if wall is None or self.epochWall is None:
            return None
        frameSec = self.cfg.frameSamples / self.cfg.rate
        return (wall - self.epochWall) / frameSec

    def _alignPredictor(self, frameIndex, guardPos, hard=False):
        wall = self._wallAt(guardPos)
        if wall is None:
            return
        frameSec = self.cfg.frameSamples / self.cfg.rate
        observedEpoch = wall - frameIndex * frameSec
        if self.epochWall is None or hard:
            self.epochWall = observedEpoch
        else:
            a = self.cfg.wallAlpha
            self.epochWall = (1.0 - a) * self.epochWall + a * observedEpoch

    def _bankCorrRange(self, lo, hi, codes=None):
        L = self.cfg.guardSamples
        half = self.detectorHalf
        lo = max(int(lo), self.buf.base + half)
        hi = min(int(hi), self.buf.end - L - half)
        if hi < lo:
            return (None, None)
        raw = self.buf.slice(lo - half, hi + half + L)
        if raw is None:
            return (None, None)
        x = np.ascontiguousarray(np.convolve(raw, self.detectorTaps, mode='valid'), dtype=np.float32)
        if len(x) < L:
            return (None, None)
        # A cumulative sum is much cheaper than convolving an 80-tap rectangle.
        power = np.concatenate(([0.0], np.cumsum(x.astype(np.float64) ** 2)))
        energy = np.sqrt(np.maximum(0.0, power[L:] - power[:-L])).astype(np.float32)
        ix = np.arange(self.guardCycle) if codes is None else np.asarray(codes, dtype=int)
        templates = self.detectorGuards[ix]
        norms = self.detectorGuardNorms[ix]
        npositions = len(x) - L + 1

        if npositions >= 320 and len(ix) >= 16:
            # One vectorized FFT bank pass replaces 64 x sliding-window dot products.
            nfft = next_fast_len(len(x) + L - 1)
            cacheKey = (nfft, tuple(int(i) for i in ix))
            spectrumBank = self._fftBankCache.get(cacheKey)
            if spectrumBank is None:
                spectrumBank = rfft(templates[:, ::-1].copy(), n=nfft, axis=1)
                if len(self._fftBankCache) >= 6:
                    self._fftBankCache.clear()
                self._fftBankCache[cacheKey] = spectrumBank
            fx = rfft(x, n=nfft)
            corrNumerator = irfft(spectrumBank * fx[np.newaxis, :], n=nfft, axis=1)[:, L - 1:L - 1 + npositions]
        else:
            strides = (x.strides[0], x.strides[0])
            windows = np.lib.stride_tricks.as_strided(x, shape=(npositions, L), strides=strides, writeable=False)
            corrNumerator = (windows @ templates.T).T

        # FFT roundoff over true silence must NEVER create fake correlations > 1.
        # A mathematical normalized correlation lives in [-1, 1].
        denominator = energy[np.newaxis, :] * norms[:, np.newaxis]
        corr = np.zeros_like(corrNumerator, dtype=np.float32)
        np.divide(corrNumerator, denominator, out=corr, where=denominator > 1e-5)
        np.clip(corr, -1.0, 1.0, out=corr)
        return np.arange(lo, lo + npositions), corr

    def _bestAnyGuard(self, lo, hi):
        positions, corr = self._bankCorrRange(lo, hi)
        if positions is None:
            return (None, None, 0.0)
        flat = int(np.argmax(corr))
        code, posIndex = np.unravel_index(flat, corr.shape)
        return (int(positions[posIndex]), int(code), float(corr[code, posIndex]))

    def _bestExpectedGuard(self, code, lo, hi):
        positions, corr = self._bankCorrRange(lo, hi, codes=[code])
        if positions is None:
            return (None, 0.0, 0.0)
        i = int(np.argmax(corr[0]))
        value = float(corr[0, i])
        pos = int(positions[i])
        _, allAtPeak = self._bankCorrRange(pos, pos)
        if allAtPeak is None:
            return (None, 0.0, 0.0)
        self._lastExpectedPeak = (int(pos), allAtPeak[:, 0].copy())
        others = np.delete(allAtPeak[:, 0], code)
        margin = value - float(np.max(others)) if len(others) else 1.0
        return (pos, value, margin)

    def _findTrains(self, lo, hi, threshold, confirmCount):
        # Compute all positions, including confirmation frames, in ONE correlation pass.
        # The old implementation recomputed a full 64-template bank for every candidate.
        lo, hi = int(lo), int(hi)
        extra = (confirmCount - 1) * self.cfg.frameSamples + self.cfg.trainRadius
        positions, corr = self._bankCorrRange(lo, hi + extra)
        if positions is None:
            return []
        allowed = int(np.searchsorted(positions, hi, side='right'))
        if allowed < 3:
            return []
        bestCorr = np.max(corr[:, :allowed], axis=0)
        bestCode = np.argmax(corr[:, :allowed], axis=0)
        peaks, _ = find_peaks(bestCorr, height=threshold, distance=max(20, self.cfg.guardSamples // 2))
        trains = []
        firstPosition = int(positions[0])
        radius = self.cfg.trainRadius
        for peak in peaks:
            firstPos = int(positions[peak])
            firstCode = int(bestCode[peak])
            firstValue = float(bestCorr[peak])
            firstColumn = corr[:, peak]
            firstMargin = firstValue - float(max(np.max(firstColumn[:firstCode], initial=-1.0), np.max(firstColumn[firstCode + 1:], initial=-1.0)))
            if firstMargin < self.cfg.codeMargin:
                continue
            values, margins, positionsFound = [firstValue], [firstMargin], [firstPos]
            for j in range(1, confirmCount):
                expectedPos = firstPos + j * self.cfg.frameSamples
                expectedCode = (firstCode + j) % self.guardCycle
                left = max(0, expectedPos - radius - firstPosition)
                right = min(corr.shape[1], expectedPos + radius + 1 - firstPosition)
                if left >= right:
                    break
                column = corr[expectedCode, left:right]
                local = int(np.argmax(column))
                ix = left + local
                value = float(column[local])
                scores = corr[:, ix]
                margin = value - float(max(np.max(scores[:expectedCode], initial=-1.0), np.max(scores[expectedCode + 1:], initial=-1.0)))
                if value < threshold or margin < self.cfg.codeMargin:
                    break
                positionsFound.append(int(positions[ix]))
                values.append(value)
                margins.append(margin)
            if len(values) == confirmCount:
                trains.append({'pos': firstPos, 'phase': firstCode, 'score': float(np.mean(values)), 'min_margin': float(min(margins)), 'positions': positionsFound, 'corrs': values, 'min_corr': float(min(values))})
        return trains

    def _resolveFrameIndex(self, phase, guardPos):
        # FRAME ID MUST BE DERIVED FROM PHYSICAL PCM SAMPLES, NOT time.monotonic().
        # Wall-clock prediction can drift by seconds when the Python receiver lags.
        # 64-state phase then disambiguates +/-8 or +/-16 jumps around that estimate.
        if self.lastGuard is None or self.lastFrame < 0:
            return int(phase), float(phase)
        advance = (guardPos - self.lastGuard) / self.periodEstimate
        predicted = self.lastFrame + advance
        cycle = self.guardCycle
        k = int(round((predicted - phase) / cycle))
        candidates = [phase + (k + dk) * cycle for dk in (-1, 0, 1)]
        valid = [n for n in candidates if n > self.lastFrame]
        if not valid:
            return None, float(predicted)
        best = min(valid, key=lambda n: (abs(n - predicted), n))
        # Never mistake another 64-frame revolution for a justified timeline jump.
        if abs(best - predicted) > self.cfg.maxAnchorDiscontinuityFrames:
            return None, float(predicted)
        return int(best), float(predicted)

    def _coreAt(self, guardPos):
        return self.buf.slice(guardPos - self.cfg.coreSamples, guardPos)

    def _coreBetweenPositions(self, previousGuardPos, guardPos):
        start = previousGuardPos + self.cfg.guardSamples
        end = guardPos
        core = self.buf.slice(start, end)
        if core is None or len(core) < 2:
            return (None, None)
        correction = len(core) - self.cfg.coreSamples
        if abs(correction) > self.cfg.maxCoreAdjust:
            return (None, correction)
        if len(core) == self.cfg.coreSamples:
            return (core.copy(), 0)
        src = np.arange(len(core), dtype=np.float64)
        dst = np.linspace(0.0, len(core) - 1.0, self.cfg.coreSamples)
        normalized = np.interp(dst, src, core).astype(np.float32)
        return (normalized, correction)

    def _coreBetweenGuards(self, guardPos):
        if self.lastGuard is None:
            return (self._coreAt(guardPos), 0)
        return self._coreBetweenPositions(self.lastGuard, guardPos)

    def _recoverGap(self, oldFrame, oldGuard, newFrame, newGuard, newGuardCorr):
        # Recover buffered frames between two trusted anchors; short gaps use interpolation only.
        logicalGap = newFrame - oldFrame
        if logicalGap <= 0:
            return {'emitted': 0, 'recovered': 0, 'lost': 0, 'detected_guards': 0, 'estimated_guards': 0}
        if logicalGap > self.cfg.gapRecoveryMaxFrames:
            for frameIdx in range(oldFrame + 1, newFrame):
                self._emit(frameIdx, None, 'lost', reason='gap_too_huge_for_recovery')
            core = self._coreAt(newGuard)
            status = 'ok' if core is not None else 'lost'
            self._emit(newFrame, core.copy() if core is not None else None, status, guard_pos=newGuard, corr=newGuardCorr, recovered_gap=False, guard_source='anchor', reason='' if status == 'ok' else 'anchor_core_missing')
            recovered = 1 if status == 'ok' else 0
            return {'emitted': logicalGap, 'recovered': recovered, 'lost': logicalGap - recovered, 'detected_guards': 0, 'estimated_guards': 0}
        positions = {oldFrame: int(oldGuard), newFrame: int(newGuard)}
        sources = {oldFrame: 'anchor', newFrame: 'anchor'}
        correlations = {oldFrame: None, newFrame: float(newGuardCorr)}
        physicalSpan = float(newGuard - oldGuard)
        timelineDelta = physicalSpan - logicalGap * self.cfg.frameSamples

        if abs(timelineDelta) >= self.cfg.frameSamples - self.cfg.maxTimingJump:
            recovered = 0
            lost = 0

            for frameIdx in range(oldFrame + 1, newFrame):
                self._emit(frameIdx, None, "lost", reason="whole_frame_timeline_gap")
                lost += 1

            core = self._coreAt(newGuard)
            status = "ok" if core is not None else "lost"

            self._emit(
                newFrame, core.copy() if core is not None else None, status,
                guard_pos=newGuard, corr=newGuardCorr, recovered_gap=True,
                guard_source="anchor", timeline_delta_samples=float(timelineDelta),
                reason="" if status == "ok" else "anchor_core_missing"
            )

            recovered += int(status == "ok")
            lost += int(status != "ok")

            return {
                "emitted": logicalGap, "recovered": recovered, "lost": lost,
                "detected_guards": 0, "estimated_guards": 0
            }
        candidates = {}
        allowPnRefinement = logicalGap > self.cfg.gapRecoveryInterpolationOnlyFrames
        if allowPnRefinement:
            for frameIdx in range(oldFrame + 1, newFrame):
                alpha = (frameIdx - oldFrame) / logicalGap
                predictedPos = oldGuard + alpha * physicalSpan
                phase = frameIdx % self.guardCycle
                guardPos, corr, margin = self._bestExpectedGuard(phase, predictedPos - self.cfg.gapRecoveryRadius, predictedPos + self.cfg.gapRecoveryRadius)
                if guardPos is None or corr < self.cfg.gapRecoveryThreshold or margin < self.cfg.gapRecoveryMargin:
                    continue
                candidates[frameIdx] = {'pos': int(guardPos), 'corr': float(corr), 'margin': float(margin), 'offset': float(guardPos - predictedPos)}
        accepted = set()
        if candidates:
            run = []
            runOffsets = []

            def flushRun():
                if len(run) >= self.cfg.gapRecoveryConfirm:
                    accepted.update(run)
            previousFrame = None
            for frameIdx in range(oldFrame + 1, newFrame):
                candidate = candidates.get(frameIdx)
                if candidate is None:
                    flushRun()
                    run = []
                    runOffsets = []
                    previousFrame = None
                    continue
                offset = candidate['offset']
                if previousFrame is not None and frameIdx == previousFrame + 1:
                    prospectiveOffsets = runOffsets + [offset]
                    spread = max(prospectiveOffsets) - min(prospectiveOffsets)
                    if spread <= self.cfg.gapRecoveryCoherence:
                        run.append(frameIdx)
                        runOffsets.append(offset)
                    else:
                        flushRun()
                        run = [frameIdx]
                        runOffsets = [offset]
                else:
                    flushRun()
                    run = [frameIdx]
                    runOffsets = [offset]
                previousFrame = frameIdx
            flushRun()
        for frameIdx in sorted(accepted):
            candidate = candidates[frameIdx]
            positions[frameIdx] = candidate['pos']
            sources[frameIdx] = 'detected'
            correlations[frameIdx] = candidate['corr']
        anchorFrames = sorted(positions)
        for leftIndex in range(len(anchorFrames) - 1):
            leftFrame = anchorFrames[leftIndex]
            rightFrame = anchorFrames[leftIndex + 1]
            leftPos = positions[leftFrame]
            rightPos = positions[rightFrame]
            spanFrames = rightFrame - leftFrame
            for frameIdx in range(leftFrame + 1, rightFrame):
                if frameIdx in positions:
                    continue
                alpha = (frameIdx - leftFrame) / spanFrames
                estimatedPos = int(round(leftPos + alpha * (rightPos - leftPos)))
                positions[frameIdx] = estimatedPos
                sources[frameIdx] = 'estimated'
                correlations[frameIdx] = None
        recovered = 0
        lost = 0
        for frameIdx in range(oldFrame + 1, newFrame + 1):
            prevGuardPos = positions[frameIdx - 1]
            guardPos = positions[frameIdx]
            core, correction = self._coreBetweenPositions(prevGuardPos, guardPos)
            usesEstimate = sources[frameIdx - 1] == 'estimated' or sources[frameIdx] == 'estimated'
            canUse = core is not None and (self.cfg.gapRecoveryAllowEstimated or not usesEstimate)
            if canUse:
                status = 'ok'
                payload = core
                recovered += 1
            else:
                status = 'lost'
                payload = None
                lost += 1
            self._emit(frameIdx, payload, status, guard_pos=guardPos, corr='' if correlations[frameIdx] is None else correlations[frameIdx], recovered_gap=True, guard_source=sources[frameIdx], previous_guard_source=sources[frameIdx - 1], core_input_samples='' if correction is None else self.cfg.coreSamples + correction, core_correction_samples='' if correction is None else correction, reason='' if canUse else 'estimated_boundary_not_allowed' if core is not None else 'gap_core_adjust_too_large')
        return {'emitted': logicalGap, 'recovered': recovered, 'lost': lost, 'detected_guards': sum((1 for frameIndex in range(oldFrame + 1, newFrame) if sources[frameIndex] == 'detected')), 'estimated_guards': sum((1 for frameIndex in range(oldFrame + 1, newFrame) if sources[frameIndex] == 'estimated'))}

    def _emit(self, frameIndex, core, status, **meta):
        out = np.zeros(self.cfg.frameSamples, dtype=np.float32)
        if status == 'ok' and core is not None:
            out[:self.cfg.coreSamples] = core
        if status == 'ok':
            self.stats['frames_ok'] += 1
        else:
            self.stats['frames_lost'] += 1
        meta['spinner_phase'] = frameIndex % self.cfg.spinnerSize
        meta['super_phase'] = (frameIndex // self.cfg.spinnerSize) % self.cfg.superSpinnerSize
        meta['guard_code'] = frameIndex % self.guardCycle
        self.onFrame(frameIndex, out, status, meta)

    def _event(self, name, **fields):
        fields['event'] = name
        fields['state'] = self.state
        self.onEvent(fields)

    def _acquire(self):
        confirmSpan = self.cfg.guardSamples + (self.cfg.acquireConfirm - 1) * (self.cfg.frameSamples + self.cfg.trainRadius)
        if len(self.buf.data) < confirmSpan:
            return False
        window = self.cfg.searchWindowFrames * self.cfg.frameSamples
        hi = self.buf.end - confirmSpan
        prev = self.lastAcquireCheckedHi
        if prev is not None and hi - prev < self.cfg.acquireScanStrideSamples:
            return False
        lo = max(self.buf.base, self.buf.end - window, (prev + 1) if prev is not None else self.buf.base)
        if lo > hi:
            return False
        # Only NEW candidate start positions. _findTrains itself includes enough
        # future samples to confirm the full PN sequence for every candidate.
        self.lastAcquireCheckedHi = hi
        trains = self._findTrains(lo, hi, self.cfg.acquireThreshold, self.cfg.acquireConfirm)
        if not trains:
            return False
        train = min(trains, key=lambda t: t['pos'])
        guardPos = train['pos']
        phase = train['phase']
        frameIndex = phase
        for n in range(frameIndex):
            self._emit(n, None, 'lost', reason='before_initial_lock')
        core = self._coreAt(guardPos)
        if core is None:
            return False
        self._emit(frameIndex, core.copy(), 'ok', guard_pos=guardPos, corr=train['score'], timing_error_samples=0.0, observed_period='', period_estimate=self.periodEstimate, predicted_frame=frameIndex)
        self.lastGuard = guardPos
        self.lastFrame = frameIndex
        self._alignPredictor(frameIndex, guardPos, hard=True)
        self.state = 'LOCKED'
        self.stats['acquisitions'] += 1
        self._event('acquired', guard_pos=guardPos, frame=frameIndex, spinner_phase=phase % self.cfg.spinnerSize, super_phase=phase // self.cfg.spinnerSize, guard_code=phase, corr=train['score'])
        return True

    def _findConfirmedTrackAnchor(self):
        if self.lastGuard is None or self.lastFrame < 0:
            return None, False

        confirm = max(2, int(self.cfg.trackConfirm))
        confirmSpan = self.cfg.guardSamples + (confirm - 1) * (self.cfg.frameSamples + self.cfg.trainRadius) + self.detectorHalf
        lo = max(self.buf.base + self.detectorHalf, int(self.lastGuard + self.cfg.frameSamples // 2))
        hiAvailable = self.buf.end - confirmSpan
        hiLimit = int(self.lastGuard + (self.cfg.trackLookaheadFrames + 2) * self.cfg.frameSamples)

        if hiAvailable < lo:
            return None, True

        trains = self._findTrains(lo, min(hiAvailable, hiLimit), self.cfg.trackThreshold, confirm)
        candidates = []

        for train in trains:
            frameIndex, predicted = self._resolveFrameIndex(train["phase"], train["pos"])
            if frameIndex is None or frameIndex <= self.lastFrame:
                continue

            logicalAdvance = frameIndex - self.lastFrame
            if logicalAdvance > self.cfg.trackLookaheadFrames + 2:
                continue

            predictionError = abs(frameIndex - predicted)
            limit = self.cfg.maxAnchorDiscontinuityFrames if train["min_corr"] >= self.cfg.minWideAnchorCorr else self.cfg.maxPredictionErrorFrames
            if predictionError > limit:
                continue

            physicalGap = train["pos"] - self.lastGuard
            timelineDelta = physicalGap - logicalAdvance * self.cfg.frameSamples

            candidates.append({
                "frame": frameIndex, "guard_pos": train["pos"], "corr": train["score"],
                "min_corr": train["min_corr"], "margin": train["min_margin"],
                "predicted": predicted, "prediction_error": predictionError,
                "timing_error": float(timelineDelta), "timeline_delta": float(timelineDelta),
                "chain_positions": train["positions"], "chain_corrs": train["corrs"],
            })

        searchComplete = hiAvailable >= hiLimit
        strong = [c for c in candidates if c["min_corr"] >= self.cfg.trackStrongThreshold]
        if strong:
            # Prioritize the physically plausible anchor; corr is a tie-breaker.
            best = min(strong, key=lambda c: (c["prediction_error"], -c["min_corr"], -c["corr"]))
            return best, False

        if not searchComplete:
            return None, True
        if candidates:
            # With 64 patterns, a weak chain must not override the absolute super phase.
            conservative = [c for c in candidates if c["prediction_error"] <= self.cfg.maxAnchorDiscontinuityFrames and c["min_corr"] >= self.cfg.minWideAnchorCorr or c["prediction_error"] <= self.cfg.maxPredictionErrorFrames]
            if conservative:
                best = min(conservative, key=lambda c: (c["prediction_error"], -c["corr"], -c["margin"]))
                return best, False

        return None, False

    def _track(self, final=False):
        # Fast-path small timing errors; otherwise buffer and confirm a later anchor before recovery.
        frameIndex = self.lastFrame + 1
        expectedPos = self.lastGuard + self.periodEstimate
        if final:
            if self.buf.end < expectedPos + self.cfg.guardSamples:
                return False
            lo = max(self.buf.base, expectedPos - self.cfg.trackRadius)
            hi = min(expectedPos + self.cfg.trackRadius, self.buf.end - self.cfg.guardSamples)
            if hi < lo:
                return False
        else:
            needEnd = expectedPos + self.cfg.trackRadius + self.cfg.guardSamples
            if self.buf.end < needEnd:
                return False
            lo = expectedPos - self.cfg.trackRadius
            hi = expectedPos + self.cfg.trackRadius
        expectedPhase = frameIndex % self.guardCycle
        expectedGuardPos, expectedCorr, expectedMargin = self._bestExpectedGuard(expectedPhase, lo, hi)
        samePosPhase = None
        samePosCorr = 0.0
        if expectedGuardPos is not None:
            peak = getattr(self, '_lastExpectedPeak', None)
            if peak is not None and peak[0] == int(expectedGuardPos):
                sameScores = peak[1]
            else:
                _, samePosBank = self._bankCorrRange(expectedGuardPos, expectedGuardPos)
                sameScores = samePosBank[:, 0] if samePosBank is not None else None
            if sameScores is not None:
                samePosPhase = int(np.argmax(sameScores))
                samePosCorr = float(sameScores[samePosPhase])
        conflict = expectedGuardPos is not None and samePosPhase is not None and (samePosPhase != expectedPhase) and (samePosCorr >= self.cfg.trackThreshold) and (samePosCorr - expectedCorr >= self.cfg.phaseSwitchMargin)
        shiftedConfirmed = False
        nextExpectedCorr = None
        nextObservedCorr = None
        nextExpectedGuardPos = None
        if conflict and (not final):
            nextExpectedPos = expectedPos + self.periodEstimate
            nextNeedEnd = nextExpectedPos + self.cfg.trackRadius + self.cfg.guardSamples
            if self.buf.end < nextNeedEnd:
                return False
            nextLo = nextExpectedPos - self.cfg.trackRadius
            nextHi = nextExpectedPos + self.cfg.trackRadius
            nextExpectedPhase = (expectedPhase + 1) % self.guardCycle
            nextObservedPhase = (samePosPhase + 1) % self.guardCycle
            nextExpectedGuardPos, nextExpectedCorr, _ = self._bestExpectedGuard(nextExpectedPhase, nextLo, nextHi)
            if nextExpectedGuardPos is not None:
                _, nextSameBank = self._bankCorrRange(nextExpectedGuardPos, nextExpectedGuardPos)
                if nextSameBank is not None:
                    nextScores = nextSameBank[:, 0]
                    nextObservedCorr = float(nextScores[nextObservedPhase])
            if nextObservedCorr is not None:
                shiftedConfirmed = nextObservedCorr >= self.cfg.trackThreshold and nextObservedCorr - nextExpectedCorr >= self.cfg.phaseSwitchMargin
        if conflict and shiftedConfirmed:
            self.lastReacquireCheckedHi = None
            self.state = 'REACQUIRE'
            self.stats['track_misses'] += 1
            self.stats['spinner_mismatches'] += 1
            self._event('spinner_mismatch', expected_frame=frameIndex, expected_phase=expectedPhase, expected_corr=expectedCorr, expected_margin=expectedMargin, observed_phase=samePosPhase, guard_pos=expectedGuardPos, corr=samePosCorr, expected_guard_pos=expectedGuardPos, expected_guard_error=float(expectedGuardPos - expectedPos) if expectedGuardPos is not None else None, observed_guard_error=float(expectedGuardPos - expectedPos) if expectedGuardPos is not None else None, next_expected_guard_pos=nextExpectedGuardPos, next_expected_corr=nextExpectedCorr, next_observed_corr=nextObservedCorr, phase_shift_confirmed=True)
            return True
        candidateUsable = expectedGuardPos is not None and expectedCorr >= self.cfg.trackThreshold
        candidateTimingError = float(expectedGuardPos - expectedPos) if expectedGuardPos is not None else None
        candidateTrusted = candidateUsable and abs(candidateTimingError) <= self.cfg.trackContinuityRadius
        if candidateTrusted:
            guardPos = expectedGuardPos
            corr = expectedCorr
        else:
            anchor, needMore = self._findConfirmedTrackAnchor()
            if needMore:
                return False
            if anchor is not None:
                oldFrame = self.lastFrame
                oldGuard = self.lastGuard
                anchorFrame = anchor['frame']
                anchorGuard = anchor['guard_pos']
                recovery = self._recoverGap(oldFrame=oldFrame, oldGuard=oldGuard, newFrame=anchorFrame, newGuard=anchorGuard, newGuardCorr=anchor['corr'])
                physicalGap = anchorGuard - oldGuard
                logicalGap = anchorFrame - oldFrame
                timelineDelta = physicalGap - logicalGap * self.cfg.frameSamples
                self.lastFrame = anchorFrame
                self.lastGuard = anchorGuard
                self._alignPredictor(anchorFrame, anchorGuard, hard=abs(timelineDelta) > self.cfg.maxTimingJump)
                self._event('track_gap_recovered', suspect_frame=frameIndex, suspect_expected_phase=expectedPhase, suspect_corr=expectedCorr, suspect_guard_pos=expectedGuardPos, suspect_timing_error=candidateTimingError, old_frame=oldFrame, old_guard_pos=oldGuard, anchor_frame=anchorFrame, anchor_guard_pos=anchorGuard, anchor_corr=anchor['corr'], anchor_timing_error=anchor['timing_error'], confirm_positions=anchor['chain_positions'], confirm_corrs=anchor['chain_corrs'], gap_recovered=recovery['recovered'], gap_lost=recovery['lost'], detected_gap_guards=recovery['detected_guards'], estimated_gap_guards=recovery['estimated_guards'], timeline_delta_samples=timelineDelta)
                return True
            anyPos, anyPhase, anyCorr = self._bestAnyGuard(lo, hi)
            self.lastReacquireCheckedHi = None
            self.state = 'REACQUIRE'
            self.stats['track_misses'] += 1
            self._event('track_miss', expected_frame=frameIndex, expected_guard=expectedPos, expected_phase=expectedPhase, expected_corr=expectedCorr, expected_margin=expectedMargin, expected_guard_pos=expectedGuardPos, expected_guard_error=candidateTimingError, best_phase=anyPhase, best_guard_pos=anyPos, best_corr=anyCorr, reason='no_confirmed_track_chain')
            return True
        observedPeriod = guardPos - self.lastGuard
        timingError = guardPos - expectedPos
        wall = self._wallAt(guardPos)
        predictedFrame = self._predictFrame(wall)
        core, coreCorrection = self._coreBetweenGuards(guardPos)
        damaged = core is None
        self._emit(frameIndex, None if damaged else core, 'lost' if damaged else 'ok', guard_pos=guardPos, corr=corr, timing_error_samples=float(timingError), observed_period=float(observedPeriod), period_estimate=float(self.periodEstimate), predicted_frame=predictedFrame, core_input_samples='' if coreCorrection is None else self.cfg.coreSamples + coreCorrection, core_correction_samples='' if coreCorrection is None else coreCorrection, reason='core_adjust_too_large' if damaged else '')
        if abs(observedPeriod - self.cfg.frameSamples) <= 4:
            a = self.cfg.periodAlpha
            self.periodEstimate = (1.0 - a) * self.periodEstimate + a * observedPeriod
        self.lastGuard = guardPos
        self.lastFrame = frameIndex
        self._alignPredictor(frameIndex, guardPos, hard=False)
        return True

    def _reacquire(self):
        # Reacquire with a short confirmed train, then recover the buffered gap when frame identity is reliable.
        if self.buf.end - self.lastReacquireSearchEnd < self.cfg.frameSamples:
            return False
        self.lastReacquireSearchEnd = self.buf.end
        confirmSpan = self.cfg.guardSamples + (self.cfg.reacquireConfirm - 1) * (self.cfg.frameSamples + self.cfg.trainRadius)
        if len(self.buf.data) < confirmSpan:
            return False
        window = self.cfg.searchWindowFrames * self.cfg.frameSamples
        hi = self.buf.end - confirmSpan
        prev = self.lastReacquireCheckedHi
        lo = max(self.buf.base, self.lastGuard + self.cfg.frameSamples // 2 if self.lastGuard is not None else self.buf.base, self.buf.end - window, (prev + 1) if prev is not None else self.buf.base)
        if lo > hi:
            return False
        self.lastReacquireCheckedHi = hi
        trains = self._findTrains(lo, hi, self.cfg.acquireThreshold, self.cfg.reacquireConfirm)
        if not trains:
            return False
        candidates = []
        for train in trains:
            frameIndex, predicted = self._resolveFrameIndex(train['phase'], train['pos'])
            if frameIndex is None:
                continue
            predictionError = abs(frameIndex - predicted)
            limit = self.cfg.maxAnchorDiscontinuityFrames if train["min_corr"] >= self.cfg.minWideAnchorCorr else self.cfg.maxPredictionErrorFrames
            if predictionError > limit:
                continue
            candidates.append({'train': train, 'frame': frameIndex, 'predicted': predicted, 'prediction_error': predictionError})
        if not candidates:
            return False
        strong = [c for c in candidates if c["train"].get("min_corr", 0.0) >= self.cfg.trackStrongThreshold]
        if strong:
            best = min(strong, key=lambda c: (c["prediction_error"], -c["train"]["min_corr"], -c["train"]["score"]))
        else:
            best = min(candidates, key=lambda c: (c["prediction_error"], -c["train"]["score"], c["train"]["pos"]))
        train = best['train']
        frameIndex = best['frame']
        guardPos = train['pos']
        oldFrame = self.lastFrame
        oldGuard = self.lastGuard
        physicalGap = guardPos - oldGuard
        logicalGap = frameIndex - oldFrame
        timingDiscontinuity = abs(physicalGap - logicalGap * self.cfg.frameSamples) > self.cfg.maxTimingJump
        if best['prediction_error'] <= self.cfg.gapRecoveryMaxAnchorPredictionError:
            recovery = self._recoverGap(oldFrame=oldFrame, oldGuard=oldGuard, newFrame=frameIndex, newGuard=guardPos, newGuardCorr=train['score'])
            gapRecoveryAttempted = True
        else:
            for n in range(oldFrame + 1, frameIndex):
                self._emit(n, None, 'lost', reason='reacquire_anchor_uncertain')
            core = self._coreAt(guardPos)
            anchorStatus = 'ok' if core is not None else 'lost'
            self._emit(frameIndex, core.copy() if core is not None else None, anchorStatus, guard_pos=guardPos, corr=train['score'], recovered_gap=False, reason='' if anchorStatus == 'ok' else 'anchor_core_missing')
            recoveredCount = 1 if anchorStatus == 'ok' else 0
            recovery = {'recovered': recoveredCount, 'lost': logicalGap - recoveredCount, 'detected_guards': 0, 'estimated_guards': 0}
            gapRecoveryAttempted = False
        self.periodEstimate = float(self.cfg.frameSamples)
        self.lastGuard = guardPos
        self.lastFrame = frameIndex
        self._alignPredictor(frameIndex, guardPos, hard=True)
        self.state = 'LOCKED'
        self.lastReacquireCheckedHi = None
        self.stats['reacquisitions'] += 1
        self._event('reacquired', guard_pos=guardPos, recovered_frame=frameIndex, spinner_phase=train['phase'] % self.cfg.spinnerSize, super_phase=train['phase'] // self.cfg.spinnerSize, guard_code=train['phase'], predicted_frame=best['predicted'], prediction_error_frames=best['prediction_error'], gap_frames=max(0, frameIndex - oldFrame - 1), gap_recovery_attempted=gapRecoveryAttempted, gap_recovered=recovery['recovered'], gap_lost=recovery['lost'], detected_gap_guards=recovery['detected_guards'], estimated_gap_guards=recovery['estimated_guards'], timing_discontinuity=timingDiscontinuity)
        return True

    def _prune(self):
        normalKeep = self.cfg.searchWindowFrames * self.cfg.frameSamples + self.cfg.coreSamples + self.cfg.trackRadius
        normalStart = self.buf.end - normalKeep
        discardBefore = normalStart
        if self.state == 'REACQUIRE' and self.lastGuard is not None:
            recoveryKeep = self.cfg.gapRecoveryMaxFrames * self.cfg.frameSamples + self.cfg.guardSamples + self.cfg.gapRecoveryRadius
            capStart = self.buf.end - recoveryKeep
            recoveryStart = self.lastGuard
            discardBefore = max(capStart, min(normalStart, recoveryStart))
        self.buf.discardBefore(max(self.buf.base, int(discardBefore)))