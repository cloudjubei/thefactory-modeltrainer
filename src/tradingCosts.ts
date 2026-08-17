// Engine-side (TS) port of BlackSwan/trainer/trading_costs.py — the canonical net-of-cost primitives the
// ML-trading honesty gauntlet is built on. Turnover is the summed absolute weight change (both legs) with the
// first period charged the entry from a zero (cash) book; the net per-period return is gross minus fee times
// turnover. Weights[t] are the positions held into period t and must already be point-in-time (lagged by the
// caller); no look-ahead is introduced here. Golden-pinned to the Python reference in tradingCosts.test.ts.

function assertMatrix(m: number[][], name: string): void {
  if (!Array.isArray(m)) throw new Error(`${name} must be a 2-D array`)
}

export function turnoverSeries(weights: number[][]): number[] {
  assertMatrix(weights, 'weights')
  const t = weights.length
  if (t === 0) return []
  const n = weights[0].length
  const out = new Array<number>(t)
  for (let i = 0; i < t; i++) {
    let s = 0
    for (let j = 0; j < n; j++) {
      const prev = i === 0 ? 0 : weights[i - 1][j]
      s += Math.abs(weights[i][j] - prev)
    }
    out[i] = s
  }
  return out
}

export function portfolioGrossReturns(weights: number[][], assetReturns: number[][]): number[] {
  assertMatrix(weights, 'weights')
  assertMatrix(assetReturns, 'assetReturns')
  const t = weights.length
  if (t === 0) return []
  if (assetReturns.length !== t || weights[0].length !== assetReturns[0].length) {
    throw new Error('weights and assetReturns must share shape')
  }
  const n = weights[0].length
  const out = new Array<number>(t)
  for (let i = 0; i < t; i++) {
    let s = 0
    for (let j = 0; j < n; j++) s += weights[i][j] * assetReturns[i][j]
    out[i] = s
  }
  return out
}

export function netReturnSeries(weights: number[][], assetReturns: number[][], fee: number): number[] {
  const gross = portfolioGrossReturns(weights, assetReturns)
  if (gross.length === 0) return gross
  const turn = turnoverSeries(weights)
  return gross.map((g, i) => g - fee * turn[i])
}
