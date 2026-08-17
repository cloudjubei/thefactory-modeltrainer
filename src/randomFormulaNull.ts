// Engine-side (TS) port of BlackSwan/trainer/random_formula_null.py — the honesty gauntlet's centrepiece: a mined
// alpha only counts if it beats RANDOM formulas of the same structural complexity, evaluated the same way, net of
// cost. The deterministic scorers (signalToWeights, formulaSharpe, nullPvalue) are golden-pinned to Python in
// randomFormulaNull.test.ts. The random-formula GENERATION cannot be cross-language golden-pinned (Python's PCG64
// and a JS PRNG differ), so randomFormula/randomFormulaNull take an injected uniform rng () => [0,1) and are
// behaviour-tested (determinism, shape, a null centred near zero) rather than value-pinned.

import { netReturnSeries } from './tradingCosts.js'

const EPS = 1e-12

function sanitize(x: number): number {
  return Number.isFinite(x) ? x : 0
}

export function signalToWeights(signal: number[][]): number[][] {
  if (signal.length === 0) return []
  const n = signal[0].length
  return signal.map((row) => {
    const clean = row.map(sanitize)
    const mean = clean.reduce((a, b) => a + b, 0) / n
    const dem = clean.map((v) => v - mean)
    const gross = dem.reduce((a, b) => a + Math.abs(b), 0)
    return gross > EPS ? dem.map((v) => v / gross) : dem.map(() => 0)
  })
}

function stdSample(v: number[]): number {
  const n = v.length
  if (n < 2) return 0
  const mean = v.reduce((a, b) => a + b, 0) / n
  const ss = v.reduce((a, b) => a + (b - mean) * (b - mean), 0)
  return Math.sqrt(ss / (n - 1))
}

export function formulaSharpe(
  signal: number[][],
  assetReturns: number[][],
  fee = 0,
  periodsPerYear = 252,
): number {
  const weights = signalToWeights(signal)
  if (weights.length === 0) return 0
  const n = weights[0].length
  const lagged = [new Array<number>(n).fill(0), ...weights.slice(0, -1)]
  const net = netReturnSeries(lagged, assetReturns, fee)
  if (net.length < 2) return 0
  const sd = stdSample(net)
  if (!(sd > EPS)) return 0
  const mean = net.reduce((a, b) => a + b, 0) / net.length
  return (mean / sd) * Math.sqrt(periodsPerYear)
}

export function nullPvalue(observed: number, nullSharpes: number[]): number {
  let ge = 0
  for (const s of nullSharpes) if (s >= observed) ge++
  return (1 + ge) / (nullSharpes.length + 1)
}

type Panel = number[][]

function mapPanel(x: Panel, f: (v: number) => number): Panel {
  return x.map((row) => row.map(f))
}

function csRank(x: Panel): Panel {
  return x.map((row) => {
    const n = row.length
    if (n < 2) return row.map(() => 0)
    const idx = row.map((_, i) => i).sort((a, b) => row[a] - row[b])
    const rank = new Array<number>(n)
    idx.forEach((orig, r) => (rank[orig] = r / (n - 1) - 0.5))
    return rank
  })
}

function delay(x: Panel): Panel {
  const n = x[0].length
  return [new Array<number>(n).fill(0), ...x.slice(0, -1)]
}

function tsMean(x: Panel, k = 3): Panel {
  return x.map((_, i) => {
    const lo = Math.max(0, i - k + 1)
    const win = x.slice(lo, i + 1)
    return x[i].map((_, j) => win.reduce((a, row) => a + row[j], 0) / win.length)
  })
}

function csZscore(x: Panel): Panel {
  return x.map((row) => {
    const n = row.length
    const mean = row.reduce((a, b) => a + b, 0) / n
    const sd = Math.sqrt(row.reduce((a, b) => a + (b - mean) * (b - mean), 0) / n)
    return sd > EPS ? row.map((v) => (v - mean) / sd) : row.map(() => 0)
  })
}

const UNARY: Array<(x: Panel) => Panel> = [
  (x) => mapPanel(x, (v) => -v),
  (x) => mapPanel(x, Math.sign),
  (x) => mapPanel(x, Math.abs),
  csRank,
  delay,
  (x) => tsMean(x, 3),
  csZscore,
]

const BINARY: Array<(a: Panel, b: Panel) => Panel> = [
  (a, b) => a.map((row, i) => row.map((v, j) => v + b[i][j])),
  (a, b) => a.map((row, i) => row.map((v, j) => v - b[i][j])),
  (a, b) => a.map((row, i) => row.map((v, j) => v * b[i][j])),
  (a, b) => a.map((row, i) => row.map((v, j) => (Math.abs(b[i][j]) > EPS ? v / b[i][j] : 0))),
]

export function randomFormula(rng: () => number, features: Panel[], depth: number): Panel {
  if (features.length === 0) throw new Error('features must be non-empty')
  const pick = (n: number) => Math.min(n - 1, Math.floor(rng() * n))
  const build = (d: number): Panel => {
    if (d <= 0 || rng() < 0.15) return features[pick(features.length)].map((row) => row.slice())
    if (rng() < 0.5) return mapPanel(UNARY[pick(UNARY.length)](build(d - 1)), sanitize)
    return mapPanel(BINARY[pick(BINARY.length)](build(d - 1), build(d - 1)), sanitize)
  }
  return build(depth)
}

export function randomFormulaNull(
  rng: () => number,
  features: Panel[],
  assetReturns: number[][],
  depth: number,
  k: number,
  fee = 0,
  periodsPerYear = 252,
): number[] {
  const out = new Array<number>(k)
  for (let i = 0; i < k; i++) {
    out[i] = formulaSharpe(randomFormula(rng, features, depth), assetReturns, fee, periodsPerYear)
  }
  return out
}
