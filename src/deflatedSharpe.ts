// A4.3 — the engine-side (TS) Deflated-Sharpe-Ratio port. diagnoseSearch is a Python-free read path, so the
// DSR the honesty gate needs is reimplemented here from the per-run moment bundle BlackSwan already emits
// (oos_sharpe / oos_ret_skew / oos_ret_kurt / oos_n_obs) and pinned to Python golden vectors from
// BlackSwan/trainer/sharpe.py (`psr_from_stats` / `deflated_sharpe_ratio`). Kurtosis is NON-excess (normal == 3),
// matching the emitted `oos_ret_kurt`. References: Bailey & López de Prado, "The Deflated Sharpe Ratio".

import type {
  DeflatedCorpusNotApplicable,
  DeflatedCorpusOptions,
  DeflatedCorpusTrial,
  DeflatedCorpusVerdict,
} from './modelTrainerTypes.js'
import { DEFAULT_DSR_THRESHOLD, DSR_MOMENT_METRIC_KEYS } from './modelTrainerConstants.js'
import { normalCdf, stdOf } from './xaiUtils.js'

const EULER_MASCHERONI = 0.5772156649015329

// Acklam's rational approximation of the inverse standard-normal CDF (|relative error| < 1.15e-9 across (0,1)).
// Only the DSR deflation level needs it (the engine's normalCdf has no inverse).
const A = [
  -3.969683028665376e1, 2.20946098424521e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1,
  2.506628277459239e0,
]
const B = [
  -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1,
]
const C = [
  -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0, -2.549732539343734e0,
  4.374664141464968e0, 2.938163982698783e0,
]
const D = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0, 3.754408661907416e0]

export function normalPpf(p: number): number {
  if (p <= 0) return -Infinity
  if (p >= 1) return Infinity
  const pLow = 0.02425
  const pHigh = 1 - pLow
  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p))
    return (
      (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) /
      ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1)
    )
  }
  if (p <= pHigh) {
    const q = p - 0.5
    const r = q * q
    return (
      ((((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q) /
      (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1)
    )
  }
  const q = Math.sqrt(-2 * Math.log(1 - p))
  return -(
    (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) /
    ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1)
  )
}

// `1 - g3*SR + (g4-1)/4 * SR^2` — the PSR's variance term. The single source of the closed form: the verdict
// layer needs it to tell "the PSR is undefined here" from "the PSR is 0 here".
function psrDenominator(sharpe: number, skewness: number, kurtosis: number): number {
  return 1 - skewness * sharpe + ((kurtosis - 1) / 4) * sharpe * sharpe
}

/**
 * PSR from a precomputed moment bundle — the TS twin of `psr_from_stats`. Probability the TRUE Sharpe exceeds
 * `srBenchmark` given the observed Sharpe + the sample's length/skew/kurtosis. 0 when undefined (n < 2 or a
 * non-positive denominator). `kurtosis` is NON-excess.
 */
export function psrFromStats(
  sharpe: number,
  skewness: number,
  kurtosis: number,
  nObs: number,
  srBenchmark = 0,
): number {
  const n = Math.trunc(nObs)
  if (n < 2) return 0
  const denom = psrDenominator(sharpe, skewness, kurtosis)
  if (!(denom > 0) || !Number.isFinite(denom)) return 0
  const z = ((sharpe - srBenchmark) * Math.sqrt(n - 1)) / Math.sqrt(denom)
  return normalCdf(z)
}

/**
 * The deflation level SR* — the expected MAXIMUM Sharpe across `nTrials` independent trials under the null
 * (true SR = 0), scaled by the cross-trial Sharpe std. 0 for < 2 trials (no multiple testing) or no spread.
 * TS twin of `expected_max_sharpe`.
 */
export function expectedMaxSharpe(nTrials: number, trialSrStd: number): number {
  if (nTrials < 2 || trialSrStd <= 0) return 0
  const g = EULER_MASCHERONI
  const bracket = (1 - g) * normalPpf(1 - 1 / nTrials) + g * normalPpf(1 - 1 / (nTrials * Math.E))
  return trialSrStd * bracket
}

/**
 * DSR from a precomputed moment bundle: PSR measured against the expected-max-Sharpe deflation level for
 * `nTrials` configs tried. DSR > ~0.95 means the observed Sharpe is unlikely to be multiple-testing luck.
 * TS twin of `deflated_sharpe_ratio` — the honesty gate's aggregate over the search.
 */
export function deflatedSharpeFromStats(
  sharpe: number,
  skewness: number,
  kurtosis: number,
  nObs: number,
  nTrials: number,
  trialSrStd: number,
): number {
  return psrFromStats(sharpe, skewness, kurtosis, nObs, expectedMaxSharpe(nTrials, trialSrStd))
}

function momentOf(trial: DeflatedCorpusTrial, key: string): number | undefined {
  const v = trial.metrics?.[key]
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined
}

/**
 * The corpus-level deflated verdict the SCREEN path never had: read ONE candidate against the
 * multiple-testing bar its own corpus implies. `nTrials` and `trialSrStd` — the two quantities a screen
 * computes nowhere, which is why nothing it judges is deflated — are derived here from the trials handed in
 * (their count, and the standard deviation of their Sharpes), then fed to {@link deflatedSharpeFromStats}.
 *
 * The honest trial count usually EXCEEDS the persisted corpus (configs searched but never written to disk),
 * so {@link DeflatedCorpusOptions.nTrials} overrides it; raising it strictly lowers the DSR, and the result
 * reports `nTrials` alongside `nCorpus` so the override is never invisible.
 *
 * An unusable corpus (< 2 trials, no cross-trial spread, a trial count too large for SR* to be represented)
 * or an unreadable candidate (a missing moment, or moments the PSR closed form has no value at) returns
 * `applicable: false` with NO `dsr` — a screen that reports `pass: false` when it could not evaluate is
 * indistinguishable from a real rejection, and a DSR of 0 is an earned verdict elsewhere.
 */
export function deflatedCorpusVerdict(
  trials: DeflatedCorpusTrial[],
  candidate: DeflatedCorpusTrial,
  options: DeflatedCorpusOptions = {},
): DeflatedCorpusVerdict {
  const sharpeKey = options.metric ?? DSR_MOMENT_METRIC_KEYS.sharpe
  const skewKey = options.skewMetric ?? DSR_MOMENT_METRIC_KEYS.skew
  const kurtKey = options.kurtosisMetric ?? DSR_MOMENT_METRIC_KEYS.kurtosis
  const nObsKey = options.nObsMetric ?? DSR_MOMENT_METRIC_KEYS.nObs
  const threshold = options.threshold ?? DEFAULT_DSR_THRESHOLD
  const corpusSharpes = trials
    .map((t) => momentOf(t, sharpeKey))
    .filter((v): v is number => v !== undefined)
  const nCorpus = corpusSharpes.length
  const override = Number(options.nTrials)
  const nTrials = Number.isFinite(override) && override >= 1 ? Math.floor(override) : nCorpus
  const trialSrStd = stdOf(corpusSharpes)
  const deflationLevel = expectedMaxSharpe(nTrials, trialSrStd)
  const searched =
    nTrials === nCorpus ? `${nTrials} trial(s)` : `${nTrials} trial(s) (${nCorpus} persisted)`
  const base = {
    nCorpus,
    nTrials,
    trialSrStd,
    deflationLevel,
    threshold,
    candidateKey: candidate.key,
  }
  const withhold = (
    notApplicable: DeflatedCorpusNotApplicable,
    detail: string,
  ): DeflatedCorpusVerdict => ({
    ...base,
    applicable: false,
    pass: false,
    notApplicable,
    detail,
  })
  if (nCorpus < 2) {
    return withhold('insufficient-trials', `need ≥ 2 trials reporting ${sharpeKey}, got ${nCorpus}`)
  }
  if (!(trialSrStd > 0)) {
    return withhold('no-spread', `${sharpeKey} has no cross-trial spread over ${searched}`)
  }
  if (!Number.isFinite(deflationLevel)) {
    return withhold(
      'deflation-overflowed',
      `deflation level is not representable over ${searched} — trial count too large`,
    )
  }
  const sharpe = momentOf(candidate, sharpeKey)
  const skew = momentOf(candidate, skewKey)
  const kurt = momentOf(candidate, kurtKey)
  const nObs = momentOf(candidate, nObsKey)
  const absent = [
    [sharpeKey, sharpe],
    [skewKey, skew],
    [kurtKey, kurt],
    [nObsKey, nObs],
  ].flatMap(([key, v]) => (v === undefined ? [key as string] : []))
  if (absent.length) {
    return withhold('candidate-moments-missing', `candidate missing ${absent.join(', ')}`)
  }
  if (Math.trunc(nObs!) < 2) {
    return withhold(
      'candidate-psr-undefined',
      `candidate ${nObsKey} ${nObs} is too short to read (need ≥ 2)`,
    )
  }
  const denom = psrDenominator(sharpe!, skew!, kurt!)
  if (!(denom > 0) || !Number.isFinite(denom)) {
    return withhold(
      'candidate-psr-undefined',
      `candidate ${skewKey}/${kurtKey} leave the PSR undefined at ${sharpeKey} ${sharpe}`,
    )
  }
  const dsr = deflatedSharpeFromStats(sharpe!, skew!, kurt!, nObs!, nTrials, trialSrStd)
  return {
    ...base,
    applicable: true,
    pass: dsr >= threshold,
    candidateSharpe: sharpe,
    dsr,
    detail: `DSR ${dsr.toFixed(3)} vs ${threshold} over ${searched} (SR* ${deflationLevel.toFixed(3)})`,
  }
}

// --- powered-null primitives (TS twins of trainer/sharpe.py; pinned to its golden vectors) ------------------
// A null is only informative once it is POWERED: "nothing cleared t>3" is indistinguishable from "the sample
// had no power" until you report the smallest effect it could have detected and a confidence bound on the true
// Sharpe. These reuse the SAME non-normality-adjusted variance term as the PSR (the Lo / Mertens SE), so the
// Sharpe SE, the minimum detectable effect, and the DSR all speak one geometry. Per-observation Sharpe units.

export type PoweredNullClass = 'survivor' | 'powered-null' | 'inconclusive'

export interface PoweredNullVerdict {
  verdict: PoweredNullClass
  se: number
  lowerBound: number
  upperBound: number
  mde: number
  powerAtEcon: number
}

/** Lo (2002) / Mertens non-normality-adjusted SE of the per-observation Sharpe: sqrt(denom/(n-1)) with the PSR
 * denominator (`kurtosis` NON-excess). Infinity when undefined (n < 2 or a non-positive denominator). */
export function sharpeStandardError(
  sharpe: number,
  skewness: number,
  kurtosis: number,
  nObs: number,
): number {
  const n = Math.trunc(nObs)
  if (n < 2) return Infinity
  const denom = psrDenominator(sharpe, skewness, kurtosis)
  if (!(denom > 0) || !Number.isFinite(denom)) return Infinity
  return Math.sqrt(denom / (n - 1))
}

/** Two-sided (1-alpha) confidence interval for the true per-observation Sharpe. (-Inf, Inf) when SE undefined. */
export function sharpeConfidenceInterval(
  sharpe: number,
  skewness: number,
  kurtosis: number,
  nObs: number,
  alpha = 0.05,
): [number, number] {
  const se = sharpeStandardError(sharpe, skewness, kurtosis, nObs)
  if (!Number.isFinite(se)) return [-Infinity, Infinity]
  const z = normalPpf(1 - alpha / 2)
  return [sharpe - z * se, sharpe + z * se]
}

/** Power of the one-sided level-alpha test H0: SR<=0 to detect a true per-obs Sharpe `srAlt` at `nObs`. */
export function sharpePower(
  srAlt: number,
  nObs: number,
  alpha = 0.05,
  skewness = 0,
  kurtosis = 3,
): number {
  const n = Math.trunc(nObs)
  if (n < 2) return 0
  const denom = psrDenominator(srAlt, skewness, kurtosis)
  if (!(denom > 0) || !Number.isFinite(denom)) return 0
  return normalCdf((srAlt * Math.sqrt(n)) / Math.sqrt(denom) - normalPpf(1 - alpha))
}

/** Smallest true per-obs Sharpe a one-sided level-alpha test detects at `power` given `nObs`; iterated because
 * the moment-adjusted denominator depends on the effect (seeded at the normal denom=1 solution). Inf for n<2. */
export function minimumDetectableSharpe(
  nObs: number,
  alpha = 0.05,
  power = 0.8,
  skewness = 0,
  kurtosis = 3,
): number {
  const n = Math.trunc(nObs)
  if (n < 2) return Infinity
  const za = normalPpf(1 - alpha)
  const zb = normalPpf(power)
  let sr = (za + zb) / Math.sqrt(n)
  for (let i = 0; i < 64; i++) {
    const denom = psrDenominator(sr, skewness, kurtosis)
    if (!(denom > 0) || !Number.isFinite(denom)) break
    const nxt = (za + zb) * Math.sqrt(denom / n)
    if (Math.abs(nxt - sr) < 1e-13) {
      sr = nxt
      break
    }
    sr = nxt
  }
  return sr
}

/** Benjamini-Hochberg FDR at level q. Returns a boolean array aligned to `pvalues` (true = reject). Step-up:
 * reject the largest-rank ordered p with p_(k) <= (k/m)*q, and everything ranked below it. */
export function benjaminiHochberg(pvalues: number[], q = 0.05): boolean[] {
  const m = pvalues.length
  if (m === 0) return []
  const order = pvalues.map((_, i) => i).sort((a, b) => pvalues[a] - pvalues[b])
  let kmax = -1
  for (let k = 0; k < m; k++) {
    if (pvalues[order[k]] <= (q * (k + 1)) / m) kmax = k
  }
  const reject = new Array<boolean>(m).fill(false)
  for (let k = 0; k <= kmax; k++) reject[order[k]] = true
  return reject
}

/** Lo (2002) / Newey-West variance-inflation factor for the Sharpe SE under serial correlation:
 * eta = 1 + 2*sum_{k=1..q} (1 - k/(q+1)) * rho_k (`autocorrs` = [rho_1, rho_2, ...], first q used, Bartlett-
 * weighted). Floored at a small positive so strong negative autocorrelation shrinks but never inverts the
 * variance. eta = 1 for q < 1. TS twin of `newey_west_inflation`. */
export function neweyWestInflation(autocorrs: number[], q: number): number {
  const qq = Math.trunc(q)
  if (qq < 1) return 1
  let s = 0
  for (let k = 1; k <= qq && k <= autocorrs.length; k++) {
    s += (1 - k / (qq + 1)) * autocorrs[k - 1]
  }
  return Math.max(1 + 2 * s, 1e-6)
}

/** Benjamini-Yekutieli FDR under ARBITRARY dependence: BH with q scaled by 1/H_m, H_m = sum_{i=1..m} 1/i.
 * Strictly no less conservative than BH. TS twin of `benjamini_yekutieli`. */
export function benjaminiYekutieli(pvalues: number[], q = 0.05): boolean[] {
  const m = pvalues.length
  if (m === 0) return []
  let hm = 0
  for (let i = 1; i <= m; i++) hm += 1 / i
  return benjaminiHochberg(pvalues, q / hm)
}

/** Per-cell POWERED verdict (per-obs Sharpe units) via one-sided (1-alpha) bounds on the true Sharpe:
 * survivor = lower bound > 0; powered-null = upper bound < `srEcon` (we REJECT a true Sharpe >= the
 * economically meaningful level — an earned null); inconclusive = neither (the sample cannot rule out
 * `srEcon`, so absence of evidence is not evidence). Also returns SE, both bounds, the MDE, and power at
 * `srEcon`. TS twin of `powered_null_verdict`. */
export function poweredNullVerdict(
  sharpe: number,
  skewness: number,
  kurtosis: number,
  nObs: number,
  srEcon: number,
  alpha = 0.05,
  power = 0.8,
): PoweredNullVerdict {
  const se = sharpeStandardError(sharpe, skewness, kurtosis, nObs)
  const mde = minimumDetectableSharpe(nObs, alpha, power, skewness, kurtosis)
  const powerAtEcon = sharpePower(srEcon, nObs, alpha, skewness, kurtosis)
  if (!Number.isFinite(se)) {
    return { verdict: 'inconclusive', se, lowerBound: -Infinity, upperBound: Infinity, mde, powerAtEcon }
  }
  const za = normalPpf(1 - alpha)
  const lowerBound = sharpe - za * se
  const upperBound = sharpe + za * se
  const verdict: PoweredNullClass =
    lowerBound > 0 ? 'survivor' : upperBound < srEcon ? 'powered-null' : 'inconclusive'
  return { verdict, se, lowerBound, upperBound, mde, powerAtEcon }
}
