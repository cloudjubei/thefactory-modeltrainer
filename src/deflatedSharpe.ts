// A4.3 — the engine-side (TS) Deflated-Sharpe-Ratio port. diagnoseSearch is a Python-free read path, so the
// DSR the honesty gate needs is reimplemented here from the per-run moment bundle BlackSwan already emits
// (oos_sharpe / oos_ret_skew / oos_ret_kurt / oos_n_obs) and pinned to Python golden vectors from
// BlackSwan/trainer/sharpe.py (`psr_from_stats` / `deflated_sharpe_ratio`). Kurtosis is NON-excess (normal == 3),
// matching the emitted `oos_ret_kurt`. References: Bailey & López de Prado, "The Deflated Sharpe Ratio".

import { normalCdf } from './xaiUtils.js'

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
  const denom = 1 - skewness * sharpe + ((kurtosis - 1) / 4) * sharpe * sharpe
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
