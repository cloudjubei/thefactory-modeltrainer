import type {
  AnalysisCriterion,
  AnalysisRun,
  ExperimentRecommendation,
  LeverImportance,
  OfatAnalysis,
  OfatEffect,
  OfatLevel,
  RunValueAggregate,
} from './modelTrainerTypes.js'
import {
  XAI_BOOTSTRAP_ITERATIONS,
  XAI_CI_LEVEL,
  XAI_FDR_ALPHA,
  XAI_MIN_SEEDS,
} from './modelTrainerConstants.js'
import { canonicalConfigString } from './modelTrainerUtils.js'

// The xAI config-effect engine: deterministic, non-LLM analysis over stored run records. Bootstrap
// interval/difference estimates are SEEDED from the data (a fixed PRNG) so the same runs always give
// the same numbers — a hard requirement so the user can re-run analysis and trust it didn't drift.

// How many top setups (by the criterion) the recommender will suggest more seeds for.
const TOP_SETUPS_FOR_SEED_REC = 3
// Safety cap on missing-cell recommendations so a wide sweep can't flood the panel.
const MAX_MISSING_CELL_RECS = 12

/** Deterministic PRNG (mulberry32) — fixed seed in, reproducible stream out. */
function makeRng(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s = (s + 0x6d2b79f5) >>> 0
    let t = Math.imul(s ^ (s >>> 15), 1 | s)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** A stable integer seed derived from a sample, so a sample's bootstrap is reproducible. */
function seedFrom(values: number[]): number {
  let h = 2166136261
  const text = values.join(',')
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619)
  return h >>> 0
}

function meanOf(values: number[]): number {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0
}

function medianOf(values: number[]): number {
  if (!values.length) return 0
  const s = [...values].sort((a, b) => a - b)
  const m = Math.floor(s.length / 2)
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

function stdOf(values: number[]): number {
  if (values.length < 2) return 0
  const m = meanOf(values)
  return Math.sqrt(values.reduce((a, b) => a + (b - m) ** 2, 0) / (values.length - 1))
}

/** Interquartile mean — trim the top/bottom 25%, mean the middle. The rliable-recommended robust aggregate. */
export function iqm(values: number[]): number {
  if (!values.length) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const trim = Math.floor(sorted.length * 0.25)
  const mid = sorted.slice(trim, sorted.length - trim)
  return meanOf(mid)
}

function percentileOf(sorted: number[], p: number): number {
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))))
  return sorted[idx]
}

function resample(values: number[], rng: () => number): number[] {
  return values.map(() => values[Math.floor(rng() * values.length)])
}

/** Robust aggregate of a sample with a deterministic bootstrap CI of the IQM. */
export function aggregateRunValues(values: number[]): RunValueAggregate {
  const n = values.length
  if (!n) return { n: 0, mean: 0, iqm: 0, median: 0, std: 0, min: 0, max: 0, ci: [0, 0] }
  const point = iqm(values)
  let ci: [number, number] = [point, point]
  if (n > 1) {
    const rng = makeRng(seedFrom(values))
    const boot: number[] = []
    for (let i = 0; i < XAI_BOOTSTRAP_ITERATIONS; i++) boot.push(iqm(resample(values, rng)))
    boot.sort((a, b) => a - b)
    const lo = (1 - XAI_CI_LEVEL) / 2
    ci = [percentileOf(boot, lo), percentileOf(boot, 1 - lo)]
  }
  return {
    n,
    mean: meanOf(values),
    iqm: point,
    median: medianOf(values),
    std: stdOf(values),
    min: Math.min(...values),
    max: Math.max(...values),
    ci,
  }
}

/** Read the criterion's numeric value off a run; `undefined` when absent/non-finite. */
export function criterionValueOf(run: AnalysisRun, criterion: AnalysisCriterion): number | undefined {
  let v: unknown
  if (criterion.key === 'objective') v = run.objective
  else if (criterion.key === 'durationMs') v = run.durationMs
  else v = run.metrics?.[criterion.key]
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined
}

function datasetSigOf(run: AnalysisRun): string {
  const d = run.dataset as Record<string, unknown> | undefined
  if (!d || typeof d !== 'object') return ''
  return ['asset', 'timeframe', 'candles', 'from', 'to']
    .filter((k) => d[k] !== undefined && d[k] !== null)
    .map((k) => `${k}=${d[k]}`)
    .join('|')
}

function configWithout(config: Record<string, unknown>, ...omit: string[]): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(config)) if (!omit.includes(k)) out[k] = v
  return out
}

/** The held-fixed context for a one-factor contrast: every lever EXCEPT `lever` (and the seed) + the dataset. */
function controlSignatureOf(run: AnalysisRun, lever: string): string {
  return `${canonicalConfigString(configWithout(run.config, lever, 'seed'))}||${datasetSigOf(run)}`
}

function orientedBetterFirst(direction: 'max' | 'min'): (a: number, b: number) => number {
  return direction === 'max' ? (a, b) => b - a : (a, b) => a - b
}

/** Bootstrap CI + two-sided p of the IQM DIFFERENCE, oriented so positive = better per the criterion. */
function bootstrapDiff(
  toValues: number[],
  fromValues: number[],
  direction: 'max' | 'min',
): { ci: [number, number]; pValue: number; delta: number } {
  const orient = (a: number, b: number) => (direction === 'max' ? a - b : b - a)
  const delta = orient(iqm(toValues), iqm(fromValues))
  if (toValues.length < 2 || fromValues.length < 2) {
    // A single seed on either side carries NO variance information, so the difference can't be called
    // significant (the N<5 pitfall) — report the point delta with pValue=1 (never significant).
    return { ci: [delta, delta], pValue: 1, delta }
  }
  const rng = makeRng(seedFrom([...toValues, NaN, ...fromValues].map((v) => (Number.isNaN(v) ? 0 : v))))
  const diffs: number[] = []
  for (let i = 0; i < XAI_BOOTSTRAP_ITERATIONS; i++) {
    diffs.push(orient(iqm(resample(toValues, rng)), iqm(resample(fromValues, rng))))
  }
  diffs.sort((a, b) => a - b)
  const lo = (1 - XAI_CI_LEVEL) / 2
  const ci: [number, number] = [percentileOf(diffs, lo), percentileOf(diffs, 1 - lo)]
  const below = diffs.filter((d) => d <= 0).length / diffs.length
  const pValue = Math.min(1, 2 * Math.min(below, 1 - below))
  return { ci, pValue, delta }
}

/** Benjamini-Hochberg FDR: returns a rejected[] mask controlling the false-discovery rate at `alpha`. */
function benjaminiHochberg(pValues: number[], alpha: number): boolean[] {
  const m = pValues.length
  const rejected = new Array<boolean>(m).fill(false)
  if (!m) return rejected
  const order = pValues.map((p, i) => ({ p, i })).sort((a, b) => a.p - b.p)
  let maxK = -1
  for (let k = 0; k < m; k++) if (order[k].p <= ((k + 1) / m) * alpha) maxK = k
  for (let k = 0; k <= maxK; k++) rejected[order[k].i] = true
  return rejected
}

function distinctValues(runs: AnalysisRun[], lever: string): Map<string, unknown> {
  const out = new Map<string, unknown>()
  for (const r of runs) if (lever in r.config) out.set(String(r.config[lever]), r.config[lever])
  return out
}

function validRunsFor(runs: AnalysisRun[], criterion: AnalysisCriterion): AnalysisRun[] {
  return runs.filter((r) => r.status === 'completed' && criterionValueOf(r, criterion) !== undefined)
}

function leversOf(runs: AnalysisRun[]): string[] {
  const keys = new Set<string>()
  for (const r of runs) for (const k of Object.keys(r.config)) if (k !== 'seed') keys.add(k)
  return [...keys]
}

/**
 * Every clean one-factor-at-a-time contrast for `lever`: groups of runs identical on every OTHER lever
 * (and dataset), so the only thing varying is `lever`. Each contrast carries the levels' robust aggregates
 * and the pairwise effects vs the worst level, with Benjamini-Hochberg-corrected significance.
 */
export function ofatContrasts(
  runs: AnalysisRun[],
  lever: string,
  criterion: AnalysisCriterion,
): OfatAnalysis[] {
  const valid = validRunsFor(runs, criterion).filter((r) => lever in r.config)
  const groups = new Map<string, AnalysisRun[]>()
  for (const r of valid) {
    const sig = controlSignatureOf(r, lever)
    const g = groups.get(sig)
    if (g) g.push(r)
    else groups.set(sig, [r])
  }
  const analyses: OfatAnalysis[] = []
  for (const [controlSignature, group] of groups) {
    const byValue = new Map<string, AnalysisRun[]>()
    for (const r of group) {
      const v = String(r.config[lever])
      const b = byValue.get(v)
      if (b) b.push(r)
      else byValue.set(v, [r])
    }
    if (byValue.size < 2) continue
    const levelValues = new Map<string, number[]>()
    const levels: OfatLevel[] = []
    for (const [value, vruns] of byValue) {
      const values = vruns.map((r) => criterionValueOf(r, criterion)!).filter((x) => x !== undefined)
      levelValues.set(value, values)
      levels.push({
        value,
        runKeys: vruns.map((r) => r.key),
        seeds: new Set(vruns.map((r) => r.seed ?? 0)).size,
        aggregate: aggregateRunValues(values),
      })
    }
    levels.sort((a, b) => orientedBetterFirst(criterion.direction)(a.aggregate.iqm, b.aggregate.iqm))
    const baseline = levels[levels.length - 1]
    const baselineValues = levelValues.get(baseline.value)!
    const effects: OfatEffect[] = []
    for (const level of levels.slice(0, -1)) {
      const { ci, pValue, delta } = bootstrapDiff(
        levelValues.get(level.value)!,
        baselineValues,
        criterion.direction,
      )
      effects.push({
        from: baseline.value,
        to: level.value,
        delta,
        diffCi: ci,
        significant: false,
        pValue,
      })
    }
    const rejected = benjaminiHochberg(effects.map((e) => e.pValue), XAI_FDR_ALPHA)
    effects.forEach((e, i) => {
      e.significant = rejected[i] && (e.diffCi[0] > 0 || e.diffCi[1] < 0)
    })
    analyses.push({ lever, criterion, controlSignature, levels, effects })
  }
  return analyses
}

/**
 * Cheap, surrogate-free global importance: rank levers by the spread (variance) of their per-value
 * marginal IQMs, as a fraction of the total. A screening view — confounded across other levers (use the
 * OFAT contrasts for the controlled read), but a fast "which knobs move the needle".
 */
export function leverImportances(
  runs: AnalysisRun[],
  criterion: AnalysisCriterion,
): LeverImportance[] {
  const valid = validRunsFor(runs, criterion)
  const raw: { lever: string; variance: number; values: number; best: string; worst: string }[] = []
  for (const lever of leversOf(valid)) {
    const byValue = new Map<string, number[]>()
    for (const r of valid) {
      if (!(lever in r.config)) continue
      const v = String(r.config[lever])
      const cv = criterionValueOf(r, criterion)!
      const b = byValue.get(v)
      if (b) b.push(cv)
      else byValue.set(v, [cv])
    }
    if (byValue.size < 2) continue
    const marginals = [...byValue.entries()].map(([value, vals]) => ({ value, iqm: iqm(vals) }))
    const m = meanOf(marginals.map((x) => x.iqm))
    const variance = meanOf(marginals.map((x) => (x.iqm - m) ** 2))
    const sorted = [...marginals].sort((a, b) =>
      orientedBetterFirst(criterion.direction)(a.iqm, b.iqm),
    )
    raw.push({
      lever,
      variance,
      values: byValue.size,
      best: sorted[0].value,
      worst: sorted[sorted.length - 1].value,
    })
  }
  const total = raw.reduce((a, b) => a + b.variance, 0) || 1
  return raw
    .map((r) => ({
      lever: r.lever,
      importance: r.variance / total,
      values: r.values,
      bestValue: r.best,
      worstValue: r.worst,
    }))
    .sort((a, b) => b.importance - a.importance)
}

function setupSignatureOf(run: AnalysisRun): string {
  return `${canonicalConfigString(configWithout(run.config, 'seed'))}||${datasetSigOf(run)}`
}

/** Smallest non-negative seed integers not already used, count `need`. */
function freshSeeds(used: Set<number>, need: number): number[] {
  const out: number[] = []
  let s = 0
  while (out.length < need) {
    if (!used.has(s)) out.push(s)
    s++
  }
  return out
}

function thinSeedRecommendations(
  valid: AnalysisRun[],
  criterion: AnalysisCriterion,
): ExperimentRecommendation[] {
  const setups = new Map<string, AnalysisRun[]>()
  for (const r of valid) {
    const sig = setupSignatureOf(r)
    const g = setups.get(sig)
    if (g) g.push(r)
    else setups.set(sig, [r])
  }
  const ranked = [...setups.values()]
    .map((rs) => ({ rs, score: iqm(rs.map((r) => criterionValueOf(r, criterion)!)) }))
    .sort((a, b) => orientedBetterFirst(criterion.direction)(a.score, b.score))
  const recs: ExperimentRecommendation[] = []
  for (const { rs } of ranked.slice(0, TOP_SETUPS_FOR_SEED_REC)) {
    const used = new Set(rs.map((r) => r.seed ?? 0))
    if (used.size >= XAI_MIN_SEEDS) continue
    const need = XAI_MIN_SEEDS - used.size
    recs.push({
      kind: 'thin-seeds',
      reason: `Top setup has ${used.size} seed(s) — run ${need} more for a trustworthy interval (≥${XAI_MIN_SEEDS}).`,
      runCount: need,
      spec: { fixed: configWithout(rs[0].config, 'seed'), seeds: freshSeeds(used, need) },
      priority: 100 - used.size,
    })
  }
  return recs
}

function missingCellRecommendations(
  valid: AnalysisRun[],
  criterion: AnalysisCriterion,
): ExperimentRecommendation[] {
  const swept = leversOf(valid).filter((l) => distinctValues(valid, l).size >= 2)
  const recs: ExperimentRecommendation[] = []
  for (let i = 0; i < swept.length && recs.length < MAX_MISSING_CELL_RECS; i++) {
    for (let j = i + 1; j < swept.length && recs.length < MAX_MISSING_CELL_RECS; j++) {
      const lA = swept[i]
      const lB = swept[j]
      const contexts = new Map<string, AnalysisRun[]>()
      for (const r of valid) {
        if (!(lA in r.config) || !(lB in r.config)) continue
        const sig = `${canonicalConfigString(configWithout(r.config, lA, lB, 'seed'))}||${datasetSigOf(r)}`
        const g = contexts.get(sig)
        if (g) g.push(r)
        else contexts.set(sig, [r])
      }
      for (const group of contexts.values()) {
        const valsA = distinctValues(group, lA)
        const valsB = distinctValues(group, lB)
        if (valsA.size < 2 || valsB.size < 2) continue
        const observed = new Set(group.map((r) => `${r.config[lA]} ${r.config[lB]}`))
        const context = configWithout(group[0].config, lA, lB, 'seed')
        for (const [, origA] of valsA) {
          for (const [, origB] of valsB) {
            if (recs.length >= MAX_MISSING_CELL_RECS) break
            if (observed.has(`${origA} ${origB}`)) continue
            recs.push({
              kind: 'missing-cell',
              reason: `Untested cell: ${lA}=${origA} × ${lB}=${origB} (the rest of this grid was run).`,
              runCount: 1,
              spec: { fixed: { ...context, [lA]: origA, [lB]: origB }, seeds: [0] },
              priority: 40,
            })
          }
        }
      }
    }
  }
  return recs
}

/**
 * Deterministically recommend the next experiments to run — variance-thin top setups that need more
 * seeds, and missing cells in the factorial grids the user has already started. Each carries a launchable
 * {@link ExperimentSpec}, so the viewer can fire them as a batched campaign (closing the analyse→run loop).
 */
export function recommendExperiments(
  runs: AnalysisRun[],
  criterion: AnalysisCriterion,
): ExperimentRecommendation[] {
  const valid = validRunsFor(runs, criterion)
  if (!valid.length) return []
  return [...thinSeedRecommendations(valid, criterion), ...missingCellRecommendations(valid, criterion)].sort(
    (a, b) => b.priority - a.priority,
  )
}
