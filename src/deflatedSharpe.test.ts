import { describe, it, expect } from 'vitest'
import {
  normalPpf,
  psrFromStats,
  expectedMaxSharpe,
  deflatedSharpeFromStats,
  deflatedCorpusVerdict,
} from './deflatedSharpe.js'
import type { DeflatedCorpusTrial } from './modelTrainerTypes.js'

// The TS Deflated-Sharpe port MUST reproduce BlackSwan/trainer/sharpe.py's psr_from_stats / expected_max_sharpe
// / deflated_sharpe_ratio. Golden values below were generated from that Python reference (scipy); the TS
// normal CDF is Abramowitz-Stegun (~1.5e-7), so the PSR pins are to 6 decimals — plenty for a probability gate.

describe('normalPpf (Acklam inverse normal)', () => {
  it('matches known standard-normal quantiles', () => {
    expect(normalPpf(0.5)).toBeCloseTo(0, 12)
    expect(normalPpf(0.975)).toBeCloseTo(1.959963984540054, 8) // the 1.96 of a 95% CI
    expect(normalPpf(0.99)).toBeCloseTo(2.3263478740408408, 8)
    expect(normalPpf(0.001)).toBeCloseTo(-3.090232306167813, 8)
    expect(normalPpf(0.025)).toBeCloseTo(-1.959963984540054, 8) // symmetry
  })

  it('is monotone and handles the degenerate ends', () => {
    expect(normalPpf(0)).toBe(-Infinity)
    expect(normalPpf(1)).toBe(Infinity)
    expect(normalPpf(0.3)).toBeLessThan(normalPpf(0.7))
  })
})

describe('psrFromStats (TS twin of sharpe.py psr_from_stats)', () => {
  it('matches the Python golden vectors', () => {
    expect(psrFromStats(0.0, 0.0, 3.0, 100)).toBeCloseTo(0.5, 6) // SR = benchmark ⇒ Phi(0)
    expect(psrFromStats(0.1, 0.0, 3.0, 101)).toBeCloseTo(0.8407413278013518, 6) // normal moments
    expect(psrFromStats(0.15, -0.5, 4.0, 200, 0.05)).toBeCloseTo(0.911495153669269, 6)
  })

  it('is 0 when undefined (n < 2 or non-positive denominator)', () => {
    expect(psrFromStats(0.5, 0.0, 3.0, 1)).toBe(0)
    // A large negative skew with a big Sharpe can drive the denominator <= 0 ⇒ undefined ⇒ 0.
    expect(psrFromStats(5.0, 5.0, 3.0, 100)).toBe(0)
  })
})

describe('expectedMaxSharpe (TS twin of sharpe.py expected_max_sharpe)', () => {
  it('matches the Python golden vectors', () => {
    expect(expectedMaxSharpe(100, 0.5)).toBeCloseTo(1.2653014466008423, 7)
    expect(expectedMaxSharpe(1000, 1.0)).toBeCloseTo(3.255121513652723, 7)
  })

  it('is 0 for a single trial (no multiple testing) or no cross-trial spread', () => {
    expect(expectedMaxSharpe(1, 0.5)).toBe(0)
    expect(expectedMaxSharpe(0, 0.5)).toBe(0)
    expect(expectedMaxSharpe(100, 0)).toBe(0)
  })

  it('rises with more trials (a higher bar to clear)', () => {
    expect(expectedMaxSharpe(10, 1)).toBeLessThan(expectedMaxSharpe(100, 1))
  })
})

describe('deflatedSharpeFromStats (TS twin of sharpe.py deflated_sharpe_ratio)', () => {
  it('deflates a modest Sharpe toward 0 under heavy multiple testing', () => {
    // SR 0.2 against a deflation level ~1.27 (100 trials, std 0.5) ⇒ DSR ≈ 0 (Python golden 4.67e-123).
    expect(deflatedSharpeFromStats(0.2, 0.0, 3.0, 500, 100, 0.5)).toBeCloseTo(0, 9)
  })

  it('equals PSR(benchmark 0) when there is a single trial (no deflation)', () => {
    expect(deflatedSharpeFromStats(0.1, 0.0, 3.0, 101, 1, 0.5)).toBeCloseTo(psrFromStats(0.1, 0.0, 3.0, 101), 12)
  })
})

// A null corpus: K trials whose TRUE Sharpe is 0, each observed over `nObs` samples. The sample Sharpe of a
// zero-edge strategy has std ~ 1/sqrt(nObs - 1), so drawing at that spread makes the corpus a faithful
// "nothing here" screen — exactly the eight-null campaign the deflation is meant to read. Deterministic LCG +
// Box-Muller: no unseeded randomness, so the property is a pin, not a flake.
const NULL_N_OBS = 501
const NULL_SR_STD = 1 / Math.sqrt(NULL_N_OBS - 1)

function nullCorpus(seed: number, k: number): DeflatedCorpusTrial[] {
  let s = seed >>> 0
  const next = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return (s + 1) / 4294967297
  }
  const trials: DeflatedCorpusTrial[] = []
  for (let i = 0; i < k; i++) {
    const z = Math.sqrt(-2 * Math.log(next())) * Math.cos(2 * Math.PI * next())
    trials.push(nullTrial(`null-${seed}-${i}`, z * NULL_SR_STD))
  }
  return trials
}

function nullTrial(key: string, sharpe: number): DeflatedCorpusTrial {
  return {
    key,
    metrics: { oos_sharpe: sharpe, oos_ret_skew: 0, oos_ret_kurt: 3, oos_n_obs: NULL_N_OBS },
  }
}

function bestOf(trials: DeflatedCorpusTrial[]): DeflatedCorpusTrial {
  return trials.reduce((a, b) =>
    (b.metrics?.oos_sharpe ?? 0) > (a.metrics?.oos_sharpe ?? 0) ? b : a,
  )
}

describe('deflatedCorpusVerdict (the screen path’s deflated verdict)', () => {
  const SEEDS = [1, 2, 3, 4, 5]

  it('REJECTS the best of a null corpus at 0.95, with its DSR near 0.5', () => {
    const dsrs: number[] = []
    for (const seed of SEEDS) {
      const trials = nullCorpus(seed, 200)
      const v = deflatedCorpusVerdict(trials, bestOf(trials))
      expect(v.applicable).toBe(true)
      expect(v.pass).toBe(false)
      expect(v.nCorpus).toBe(200)
      expect(v.nTrials).toBe(200)
      expect(v.threshold).toBe(0.95)
      // SR* is the bar the best of 200 null trials has to clear — near the realised max, hence DSR near 0.5.
      expect(v.deflationLevel).toBeGreaterThan(0)
      expect(v.dsr).toBeDefined()
      expect(v.dsr!).toBeGreaterThan(0.1)
      expect(v.dsr!).toBeLessThan(0.9)
      dsrs.push(v.dsr!)
    }
    const median = [...dsrs].sort((a, b) => a - b)[Math.floor(dsrs.length / 2)]
    expect(median).toBeGreaterThan(0.35)
    expect(median).toBeLessThan(0.65)
  })

  it('PASSES the very same candidate at nTrials = 1 — the rejection IS the deflation', () => {
    for (const seed of SEEDS) {
      const trials = nullCorpus(seed, 200)
      const candidate = bestOf(trials)
      expect(deflatedCorpusVerdict(trials, candidate).pass).toBe(false)
      const undeflated = deflatedCorpusVerdict(trials, candidate, { nTrials: 1 })
      expect(undeflated.applicable).toBe(true)
      expect(undeflated.pass).toBe(true)
      expect(undeflated.deflationLevel).toBe(0)
      expect(undeflated.dsr!).toBeGreaterThan(0.95)
    }
  })

  it('still clears 0.95 for a genuine edge at the same corpus size', () => {
    const trials = nullCorpus(1, 200)
    const candidate = nullTrial('real-edge', 0.3)
    const v = deflatedCorpusVerdict([...trials, candidate], candidate)
    expect(v.applicable).toBe(true)
    expect(v.pass).toBe(true)
    expect(v.nCorpus).toBe(201)
    expect(v.candidateSharpe).toBe(0.3)
    expect(v.candidateKey).toBe('real-edge')
    expect(v.dsr!).toBeGreaterThan(0.95)
    expect(v.candidateSharpe!).toBeGreaterThan(v.deflationLevel)
  })

  it('strictly LOWERS the DSR as the honest trial count is raised', () => {
    const trials = nullCorpus(1, 200)
    const candidate = nullTrial('lead', 0.3)
    const corpus = [...trials, candidate]
    const persisted = deflatedCorpusVerdict(corpus, candidate)
    const searched = deflatedCorpusVerdict(corpus, candidate, { nTrials: 1100 })
    const honest = deflatedCorpusVerdict(corpus, candidate, { nTrials: 20888 })
    expect(searched.dsr!).toBeLessThan(persisted.dsr!)
    expect(honest.dsr!).toBeLessThan(searched.dsr!)
    expect(honest.deflationLevel).toBeGreaterThan(searched.deflationLevel)
    expect(searched.deflationLevel).toBeGreaterThan(persisted.deflationLevel)
    // The override's effect is legible: the count used is NOT the persisted corpus size.
    expect(honest.nTrials).toBe(20888)
    expect(honest.nCorpus).toBe(201)
    expect(honest.detail).toContain('20888')
    expect(honest.detail).toContain('201')
  })

  it('ignores a non-usable nTrials override and deflates by the persisted corpus', () => {
    const trials = nullCorpus(1, 200)
    const candidate = bestOf(trials)
    const base = deflatedCorpusVerdict(trials, candidate)
    for (const nTrials of [0, -5, Number.NaN, Number.POSITIVE_INFINITY]) {
      const v = deflatedCorpusVerdict(trials, candidate, { nTrials })
      expect(v.nTrials).toBe(200)
      expect(v.dsr!).toBe(base.dsr!)
    }
  })

  it('reads the moment bundle off caller-named metric keys and a caller-named threshold', () => {
    const renamed = nullCorpus(1, 200).map((t) => ({
      key: t.key,
      metrics: {
        sr: t.metrics!.oos_sharpe,
        sk: t.metrics!.oos_ret_skew,
        ku: t.metrics!.oos_ret_kurt,
        n: t.metrics!.oos_n_obs,
      },
    }))
    const candidate = renamed.reduce((a, b) => (b.metrics.sr > a.metrics.sr ? b : a))
    const opts = { metric: 'sr', skewMetric: 'sk', kurtosisMetric: 'ku', nObsMetric: 'n' }
    const v = deflatedCorpusVerdict(renamed, candidate, opts)
    expect(v.applicable).toBe(true)
    expect(v.candidateSharpe).toBe(candidate.metrics.sr)
    // The SAME corpus under the default keys is unreadable — proof the caller-named keys did the work.
    expect(deflatedCorpusVerdict(renamed, candidate).notApplicable).toBe('insufficient-trials')
    // A lenient caller-named threshold turns the same near-0.5 DSR into a pass.
    const lenient = deflatedCorpusVerdict(renamed, candidate, { ...opts, threshold: 0.2 })
    expect(lenient.threshold).toBe(0.2)
    expect(lenient.dsr).toBe(v.dsr)
    expect(lenient.pass).toBe(true)
    expect(v.pass).toBe(false)
  })

  describe('not-applicable (never a fabricated rejection)', () => {
    it('refuses a corpus of fewer than 2 trials', () => {
      const one = nullCorpus(1, 1)
      for (const corpus of [[] as DeflatedCorpusTrial[], one]) {
        const v = deflatedCorpusVerdict(corpus, nullTrial('c', 0.3))
        expect(v.applicable).toBe(false)
        expect(v.pass).toBe(false)
        expect(v.notApplicable).toBe('insufficient-trials')
        expect(v.dsr).toBeUndefined()
        expect(v.detail).toContain('2 trials')
      }
    })

    it('refuses a corpus with zero cross-trial spread', () => {
      const flat = [1, 2, 3, 4, 5].map((i) => nullTrial(`flat-${i}`, 0.12))
      const v = deflatedCorpusVerdict(flat, flat[0])
      expect(v.applicable).toBe(false)
      expect(v.pass).toBe(false)
      expect(v.notApplicable).toBe('no-spread')
      expect(v.trialSrStd).toBe(0)
      expect(v.dsr).toBeUndefined()
    })

    it('refuses a candidate missing a moment, naming what is missing', () => {
      const trials = nullCorpus(1, 200)
      const missing = { key: 'partial', metrics: { oos_sharpe: 0.3, oos_ret_skew: 0 } }
      const v = deflatedCorpusVerdict([...trials, missing], missing)
      expect(v.applicable).toBe(false)
      expect(v.pass).toBe(false)
      expect(v.notApplicable).toBe('candidate-moments-missing')
      expect(v.dsr).toBeUndefined()
      expect(v.detail).toContain('oos_ret_kurt')
      expect(v.detail).toContain('oos_n_obs')
      // The corpus itself WAS usable — the deflation level is still reported, only the verdict is withheld.
      expect(v.deflationLevel).toBeGreaterThan(0)
    })

    it('treats a non-finite moment as missing', () => {
      const trials = nullCorpus(1, 200)
      const nan = nullTrial('nan', Number.NaN)
      expect(deflatedCorpusVerdict([...trials, nan], nan).notApplicable).toBe(
        'candidate-moments-missing',
      )
      const inf = {
        key: 'inf',
        metrics: { ...nullTrial('inf', 0.3).metrics!, oos_n_obs: Number.POSITIVE_INFINITY },
      }
      expect(deflatedCorpusVerdict([...trials, inf], inf).notApplicable).toBe(
        'candidate-moments-missing',
      )
    })

    it('refuses a candidate whose track record is too short to read', () => {
      const trials = nullCorpus(1, 200)
      // PSR is UNDEFINED below 2 observations — psrFromStats returns 0 there, and a DSR of 0 is the most
      // damning number this gate can print. Reporting it as a verdict certifies "no edge" from no evidence.
      for (const nObs of [0, 1, 1.5, -5]) {
        const short = {
          key: 'short',
          metrics: { ...nullTrial('short', 0.3).metrics!, oos_n_obs: nObs },
        }
        const v = deflatedCorpusVerdict([...trials, short], short)
        expect(v.applicable).toBe(false)
        expect(v.pass).toBe(false)
        expect(v.notApplicable).toBe('candidate-psr-undefined')
        expect(v.dsr).toBeUndefined()
        expect(v.detail).toContain('oos_n_obs')
      }
      // Two observations is the shortest READABLE track record: a verdict, not a withholding.
      const two = { key: 'two', metrics: { ...nullTrial('two', 0.3).metrics!, oos_n_obs: 2 } }
      const readable = deflatedCorpusVerdict([...trials, two], two)
      expect(readable.applicable).toBe(true)
      expect(readable.notApplicable).toBeUndefined()
      expect(readable.dsr).toBeGreaterThan(0)
    })

    it('refuses a candidate whose moments leave the PSR denominator non-positive', () => {
      const trials = nullCorpus(1, 200)
      // A large skew against a large Sharpe drives 1 - g3*SR + (g4-1)/4*SR^2 <= 0 — the closed form has no
      // value there, and psrFromStats says so by returning 0.
      const degenerate = {
        key: 'degenerate',
        metrics: { oos_sharpe: 5, oos_ret_skew: 5, oos_ret_kurt: 3, oos_n_obs: 501 },
      }
      const v = deflatedCorpusVerdict([...trials, degenerate], degenerate)
      expect(v.applicable).toBe(false)
      expect(v.pass).toBe(false)
      expect(v.notApplicable).toBe('candidate-psr-undefined')
      expect(v.dsr).toBeUndefined()
      expect(v.detail).toContain('oos_ret_skew')
    })

    it('refuses an honest trial count too large for the deflation level to be represented', () => {
      const trials = nullCorpus(1, 200)
      const candidate = nullTrial('lead', 0.3)
      const corpus = [...trials, candidate]
      const readable = deflatedCorpusVerdict(corpus, candidate, { nTrials: 1e15 })
      expect(readable.applicable).toBe(true)
      expect(Number.isFinite(readable.deflationLevel)).toBe(true)
      expect(readable.pass).toBe(false)
      // At 1e17, `1 - 1/N` rounds to exactly 1 and SR* saturates to Infinity — the correction is no longer a
      // number, so the DSR of 0 that falls out of it is arithmetic, not evidence.
      const overflowed = deflatedCorpusVerdict(corpus, candidate, { nTrials: 1e17 })
      expect(overflowed.applicable).toBe(false)
      expect(overflowed.pass).toBe(false)
      expect(overflowed.notApplicable).toBe('deflation-overflowed')
      expect(overflowed.dsr).toBeUndefined()
      expect(overflowed.deflationLevel).toBe(Number.POSITIVE_INFINITY)
      expect(overflowed.detail).toContain('100000000000000000')
    })

    it('is distinguishable from a real rejection', () => {
      const trials = nullCorpus(1, 200)
      const rejected = deflatedCorpusVerdict(trials, bestOf(trials))
      const na = deflatedCorpusVerdict([], bestOf(trials))
      expect([rejected.applicable, rejected.pass]).toEqual([true, false])
      expect([na.applicable, na.pass]).toEqual([false, false])
      expect(rejected.notApplicable).toBeUndefined()
      expect(typeof rejected.dsr).toBe('number')
      expect(na.dsr).toBeUndefined()
      // A DSR of exactly 0 is itself an EARNED verdict (a badly negative Sharpe gets one), so the number can
      // never be the tell — only `applicable` / `notApplicable` separate a withheld read from a real one.
      const earnedZero = deflatedCorpusVerdict(trials, nullTrial('flat-out-bad', -0.9))
      expect(earnedZero.applicable).toBe(true)
      expect(earnedZero.pass).toBe(false)
      expect(earnedZero.dsr).toBe(0)
      expect(earnedZero.notApplicable).toBeUndefined()
    })
  })
})
