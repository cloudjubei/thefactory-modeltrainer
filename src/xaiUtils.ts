import type {
  AblationPath,
  AblationStep,
  AnalysisCriterion,
  AnalysisRun,
  ConfigSpaceAnalysis,
  EnvironmentSummary,
  ConfigSurrogate,
  ExperimentRecommendation,
  FanovaImportance,
  InteractionGrid,
  LeverCoupling,
  LeverImportance,
  OfatAnalysis,
  OfatEffect,
  OfatLevel,
  PcaPoint,
  PcaProjection,
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
  const raw: {
    lever: string
    variance: number
    values: number
    best: string
    worst: string
    minRuns: number
  }[] = []
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
      minRuns: Math.min(...[...byValue.values()].map((vals) => vals.length)),
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
      minRuns: r.minRuns,
      confident: r.minRuns >= XAI_MIN_SEEDS,
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
  opts?: { surrogate?: ConfigSurrogate; setups?: AnalysisRun[] },
): ExperimentRecommendation[] {
  const valid = validRunsFor(runs, criterion)
  if (!valid.length) return []
  // Acquisition scores the surrogate (passed in pre-fit by the whole-space bundle, else fit here); the
  // setup-level source keeps the incumbent + candidate scale consistent with that surrogate. Thin-seeds /
  // missing-cell always use the raw runs — they reason about per-config seed counts + observed cells.
  const acqSource = opts?.setups ?? valid
  const all = [
    ...acquisitionRecommendations(acqSource, criterion, opts?.surrogate),
    ...thinSeedRecommendations(valid, criterion),
    ...missingCellRecommendations(valid, criterion),
  ].sort((a, b) => b.priority - a.priority)
  // Dedup by the launched config — an acquisition pick can coincide with a missing-cell gap; keep the
  // higher-priority one (first after the sort) for each distinct `fixed` config.
  const seen = new Set<string>()
  return all.filter((rec) => {
    const key = canonicalConfigString(rec.spec.fixed ?? {})
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

// --- Phase 3: a seeded random-forest config→criterion surrogate + ablation/fANOVA/interaction reads ---
// The surrogate is the validated retraining-free model (Hutter et al. / Biedenkapp et al.): it predicts
// the criterion for unobserved configs so the ablation path can step through intermediate nodes and
// fANOVA can marginalise. Deterministic — the forest is seeded from the data.

const SURROGATE_TREES = 64
const SURROGATE_MAX_DEPTH = 8
const SURROGATE_MIN_LEAF = 2

// Acquisition (Phase 2): how many candidate configs the EI search scores, and how many top suggestions
// it returns. The candidate grid is the cartesian product of OBSERVED lever values (capped + sampled).
const MAX_ACQUISITION_CANDIDATES = 2000
const TOP_ACQUISITION_RECS = 5

type SurrogateNode =
  | { leaf: number }
  | { lever: string; kind: 'num'; threshold: number; left: SurrogateNode; right: SurrogateNode }
  | { lever: string; kind: 'cat'; value: string; left: SurrogateNode; right: SurrogateNode }

interface SurrogateRow {
  config: Record<string, unknown>
  y: number
}

function varianceOf(values: number[]): number {
  if (values.length < 2) return 0
  const m = meanOf(values)
  return values.reduce((a, b) => a + (b - m) ** 2, 0) / values.length
}

function leverKindsOf(runs: AnalysisRun[]): { name: string; kind: 'num' | 'cat' }[] {
  return leversOf(runs).map((name) => {
    const allNumeric = runs.every(
      (r) => !(name in r.config) || typeof r.config[name] === 'number',
    )
    return { name, kind: allNumeric ? 'num' : 'cat' }
  })
}

function sampleSubset<T>(items: T[], k: number, rng: () => number): T[] {
  const pool = [...items]
  const out: T[] = []
  for (let i = 0; i < k && pool.length; i++) out.push(pool.splice(Math.floor(rng() * pool.length), 1)[0])
  return out
}

function buildSurrogateTree(
  rows: SurrogateRow[],
  levers: { name: string; kind: 'num' | 'cat' }[],
  rng: () => number,
  depth: number,
): SurrogateNode {
  const ys = rows.map((r) => r.y)
  const parentVar = varianceOf(ys)
  if (depth >= SURROGATE_MAX_DEPTH || rows.length <= SURROGATE_MIN_LEAF || parentVar === 0) {
    return { leaf: meanOf(ys) }
  }
  const tried = sampleSubset(levers, Math.max(1, Math.round(Math.sqrt(levers.length))), rng)
  let best:
    | { lever: string; kind: 'num'; threshold: number; left: SurrogateRow[]; right: SurrogateRow[]; score: number }
    | { lever: string; kind: 'cat'; value: string; left: SurrogateRow[]; right: SurrogateRow[]; score: number }
    | undefined
  const consider = (
    lever: { name: string; kind: 'num' | 'cat' },
    pick: (r: SurrogateRow) => boolean,
    extra: { threshold?: number; value?: string },
  ) => {
    const left = rows.filter(pick)
    const right = rows.filter((r) => !pick(r))
    if (!left.length || !right.length) return
    const score =
      parentVar -
      (left.length * varianceOf(left.map((r) => r.y)) + right.length * varianceOf(right.map((r) => r.y))) /
        rows.length
    if (!best || score > best.score) {
      best = { lever: lever.name, kind: lever.kind, ...extra, left, right, score } as typeof best
    }
  }
  for (const lever of tried) {
    if (lever.kind === 'num') {
      const nums = [...new Set(rows.map((r) => Number(r.config[lever.name])).filter(Number.isFinite))].sort(
        (a, b) => a - b,
      )
      for (let i = 0; i + 1 < nums.length; i++) {
        const threshold = (nums[i] + nums[i + 1]) / 2
        consider(lever, (r) => Number(r.config[lever.name]) <= threshold, { threshold })
      }
    } else {
      for (const value of new Set(rows.map((r) => String(r.config[lever.name])))) {
        consider(lever, (r) => String(r.config[lever.name]) === value, { value })
      }
    }
  }
  if (!best || best.score <= 0) return { leaf: meanOf(ys) }
  const left = buildSurrogateTree(best.left, levers, rng, depth + 1)
  const right = buildSurrogateTree(best.right, levers, rng, depth + 1)
  return best.kind === 'num'
    ? { lever: best.lever, kind: 'num', threshold: best.threshold, left, right }
    : { lever: best.lever, kind: 'cat', value: best.value, left, right }
}

/** Fit the seeded random-forest surrogate over the completed runs' (config → criterion) pairs. */
export function fitConfigSurrogate(
  runs: AnalysisRun[],
  criterion: AnalysisCriterion,
): ConfigSurrogate {
  const valid = validRunsFor(runs, criterion)
  const rows: SurrogateRow[] = valid.map((r) => ({ config: r.config, y: criterionValueOf(r, criterion)! }))
  const levers = leverKindsOf(valid)
  const mean = rows.length ? meanOf(rows.map((r) => r.y)) : 0
  if (rows.length < 2 || !levers.length) return { trees: [], levers, mean }
  const rng = makeRng(seedFrom(rows.map((r) => r.y)))
  const trees: SurrogateNode[] = []
  for (let t = 0; t < SURROGATE_TREES; t++) {
    const sample = rows.map(() => rows[Math.floor(rng() * rows.length)])
    trees.push(buildSurrogateTree(sample, levers, rng, 0))
  }
  return { trees, levers, mean }
}

function predictSurrogateTree(node: SurrogateNode, config: Record<string, unknown>): number {
  let cur = node
  while (!('leaf' in cur)) {
    if (cur.kind === 'num') {
      const v = Number(config[cur.lever])
      cur = (Number.isFinite(v) ? v : cur.threshold) <= cur.threshold ? cur.left : cur.right
    } else {
      cur = String(config[cur.lever]) === cur.value ? cur.left : cur.right
    }
  }
  return cur.leaf
}

/** Predict the criterion for any config from a fitted surrogate (the forest mean). */
export function predictConfig(surrogate: ConfigSurrogate, config: Record<string, unknown>): number {
  const trees = surrogate.trees as SurrogateNode[]
  if (!trees.length) return surrogate.mean
  return meanOf(trees.map((t) => predictSurrogateTree(t, config)))
}

/**
 * Predict the criterion AND the surrogate's epistemic uncertainty at a config: the forest mean plus the
 * std of the per-tree predictions (tree disagreement). Disagreement is high where the data is sparse, so
 * `std` is the explore signal Expected Improvement balances against the predicted mean.
 */
export function predictConfigStats(
  surrogate: ConfigSurrogate,
  config: Record<string, unknown>,
): { mean: number; std: number } {
  const trees = surrogate.trees as SurrogateNode[]
  if (!trees.length) return { mean: surrogate.mean, std: 0 }
  const preds = trees.map((t) => predictSurrogateTree(t, config))
  return { mean: meanOf(preds), std: Math.sqrt(varianceOf(preds)) }
}

// Standard-normal helpers for the closed-form EI. erf via Abramowitz & Stegun 7.1.26.
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1
  const ax = Math.abs(x)
  const t = 1 / (1 + 0.3275911 * ax)
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-ax * ax)
  return sign * y
}
function normalCdf(z: number): number {
  return 0.5 * (1 + erf(z / Math.SQRT2))
}
function normalPdf(z: number): number {
  return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI)
}

/**
 * Expected Improvement of a candidate over the incumbent `best`, given the surrogate's predicted `mean`
 * and uncertainty `std`, direction-aware. Closed form under a Gaussian: `improve·Φ(z) + std·φ(z)` with
 * `z = improve/std` and `improve` the (oriented) gain over `best`. Zero std collapses to the raw gain.
 */
export function expectedImprovement(
  mean: number,
  std: number,
  best: number,
  direction: 'max' | 'min',
): number {
  const improve = direction === 'min' ? best - mean : mean - best
  if (std <= 1e-9) return Math.max(0, improve)
  const z = improve / std
  return improve * normalCdf(z) + std * normalPdf(z)
}

/** Cartesian product of per-lever value lists, capped: enumerate all when small, else sample deterministically. */
function cappedCartesian(lists: unknown[][], cap: number, rng: () => number): unknown[][] {
  const total = lists.reduce((a, l) => a * Math.max(1, l.length), 1)
  if (total <= cap) {
    let acc: unknown[][] = [[]]
    for (const list of lists) {
      const next: unknown[][] = []
      for (const combo of acc) for (const v of list) next.push([...combo, v])
      acc = next
    }
    return acc
  }
  const out: unknown[][] = []
  const seen = new Set<string>()
  for (let attempts = 0; out.length < cap && attempts < cap * 4; attempts++) {
    const combo = lists.map((l) => l[Math.floor(rng() * l.length)])
    const key = combo.map(String).join('')
    if (seen.has(key)) continue
    seen.add(key)
    out.push(combo)
  }
  return out
}

const fmtAcq = (x: number): string =>
  !Number.isFinite(x) ? 'n/a' : Math.abs(x) >= 100 ? x.toFixed(1) : Number(x.toPrecision(3)).toString()

/**
 * Acquisition recommendations (Phase 2): score every UNRUN config in the explored grid (cartesian product
 * of observed lever values) by Expected Improvement against the surrogate, and surface the top few — the
 * configs that most likely beat the current best. This is the "which way to explore" climb: it actively
 * moves toward the optimum, not just fills factorial gaps. Deterministic (seeded surrogate + sampler).
 */
function acquisitionRecommendations(
  valid: AnalysisRun[],
  criterion: AnalysisCriterion,
  prefit?: ConfigSurrogate,
): ExperimentRecommendation[] {
  const surrogate = prefit ?? fitConfigSurrogate(valid, criterion)
  if (!surrogate.trees.length || surrogate.levers.length === 0) return []
  const ys = valid.map((r) => criterionValueOf(r, criterion)!)
  const best = criterion.direction === 'min' ? Math.min(...ys) : Math.max(...ys)
  const observed = new Set(valid.map((r) => canonicalConfigString(configWithout(r.config, 'seed'))))
  const rng = makeRng(seedFrom(ys))
  const valueLists = surrogate.levers.map((l) => [...distinctValues(valid, l.name).values()])
  const scored = cappedCartesian(valueLists, MAX_ACQUISITION_CANDIDATES, rng)
    .map((combo) => {
      const config: Record<string, unknown> = {}
      surrogate.levers.forEach((l, i) => (config[l.name] = combo[i]))
      return config
    })
    .filter((config) => !observed.has(canonicalConfigString(config)))
    .map((config) => {
      const { mean, std } = predictConfigStats(surrogate, config)
      return { config, mean, std, ei: expectedImprovement(mean, std, best, criterion.direction) }
    })
    .filter((s) => s.ei > 0)
    .sort((a, b) => b.ei - a.ei)
  return scored.slice(0, TOP_ACQUISITION_RECS).map((s, i) => ({
    kind: 'acquisition',
    reason: `Surrogate predicts ${fmtAcq(s.mean)} ± ${fmtAcq(s.std)} (best so far ${fmtAcq(best)}) — expected improvement ${fmtAcq(s.ei)}; the strongest unrun config toward the optimum.`,
    runCount: 1,
    spec: { fixed: s.config, seeds: [0] },
    priority: 90 - i,
  }))
}

function observedValues(runs: AnalysisRun[], lever: string): Map<string, unknown> {
  return distinctValues(runs, lever)
}

/**
 * fANOVA MAIN-effect importance from the surrogate: each lever's marginal (averaging the surrogate over
 * the observed configs with that lever pinned to each value) and the variance it explains as a fraction
 * of the surrogate's total prediction variance. Captures interactions implicitly via the forest.
 */
export function fanovaImportances(
  surrogate: ConfigSurrogate,
  runs: AnalysisRun[],
  criterion: AnalysisCriterion,
): FanovaImportance[] {
  const valid = validRunsFor(runs, criterion)
  const configs = valid.map((r) => r.config)
  if (configs.length < 2) return []
  const totalVar = varianceOf(configs.map((c) => predictConfig(surrogate, c))) || 1
  const out: FanovaImportance[] = []
  for (const { name } of surrogate.levers) {
    const values = [...observedValues(valid, name).values()]
    if (values.length < 2) continue
    // MAIN effect: variance of the marginal (lever pinned, averaged over the other levers).
    const marginals = values.map((v) =>
      meanOf(configs.map((c) => predictConfig(surrogate, { ...c, [name]: v }))),
    )
    // TOTAL effect: at each observed config, the variance from sweeping THIS lever over its values
    // (so interactions count), averaged across configs. main ≤ total; total − main = interaction share.
    const perConfigVar = configs.map((c) =>
      varianceOf(values.map((v) => predictConfig(surrogate, { ...c, [name]: v }))),
    )
    out.push({
      lever: name,
      importance: varianceOf(marginals) / totalVar,
      total: meanOf(perConfigVar) / totalVar,
      values: values.length,
      // The distinct observed values themselves, so the viewer can link each to its runs (mirror parity).
      valueList: values,
    })
  }
  return out.sort((a, b) => b.importance - a.importance)
}

/**
 * Pairwise COUPLING strength for every swept lever pair: the 2-way ANOVA interaction term — the variance
 * of `joint(a,b) − mainA(a) − mainB(b) + grand` over the pair's value grid, as a fraction of total
 * variance. High strength ⇒ the two levers must be tuned together (one's best value depends on the
 * other); ~0 ⇒ they act independently (the heatmap is additive). Sorted strongest-first.
 */
export function leverCouplings(
  surrogate: ConfigSurrogate,
  runs: AnalysisRun[],
  criterion: AnalysisCriterion,
  onlyLevers?: string[],
): LeverCoupling[] {
  const valid = validRunsFor(runs, criterion)
  const configs = valid.map((r) => r.config)
  if (configs.length < 2) return []
  const totalVar = varianceOf(configs.map((c) => predictConfig(surrogate, c))) || 1
  const grand = meanOf(configs.map((c) => predictConfig(surrogate, c)))
  // O(levers²): when the bundle restricts to its high-effect levers, skip the inert rest entirely.
  let swept = surrogate.levers.map((l) => l.name).filter((n) => observedValues(valid, n).size >= 2)
  if (onlyLevers) swept = swept.filter((n) => onlyLevers.includes(n))
  const mainEffect = (lever: string): Map<string, number> => {
    const m = new Map<string, number>()
    for (const [k, v] of observedValues(valid, lever)) {
      m.set(k, meanOf(configs.map((c) => predictConfig(surrogate, { ...c, [lever]: v }))))
    }
    return m
  }
  const out: LeverCoupling[] = []
  for (let i = 0; i < swept.length; i++) {
    for (let j = i + 1; j < swept.length; j++) {
      const lA = swept[i]
      const lB = swept[j]
      const valsA = observedValues(valid, lA)
      const valsB = observedValues(valid, lB)
      const mainA = mainEffect(lA)
      const mainB = mainEffect(lB)
      const residuals: number[] = []
      for (const [ka, va] of valsA) {
        for (const [kb, vb] of valsB) {
          const joint = meanOf(configs.map((c) => predictConfig(surrogate, { ...c, [lA]: va, [lB]: vb })))
          residuals.push(joint - mainA.get(ka)! - mainB.get(kb)! + grand)
        }
      }
      out.push({ leverA: lA, leverB: lB, strength: varianceOf(residuals) / totalVar })
    }
  }
  return out.sort((a, b) => b.strength - a.strength)
}

/**
 * The greedy ablation path from the worst observed config (baseline) to the best (incumbent): at each
 * step apply the single differing-lever change that most improves the surrogate prediction. Returns
 * `undefined` when there aren't ≥2 runs or the two configs don't differ.
 */
export function ablationPath(
  surrogate: ConfigSurrogate,
  runs: AnalysisRun[],
  criterion: AnalysisCriterion,
): AblationPath | undefined {
  const valid = validRunsFor(runs, criterion)
  if (valid.length < 2) return undefined
  const sorted = [...valid].sort((a, b) =>
    orientedBetterFirst(criterion.direction)(criterionValueOf(a, criterion)!, criterionValueOf(b, criterion)!),
  )
  const incumbent = sorted[0].config
  const baseline = sorted[sorted.length - 1].config
  const orient = (a: number, b: number) => (criterion.direction === 'max' ? a - b : b - a)
  const diffLevers = surrogate.levers
    .map((l) => l.name)
    .filter((l) => String(baseline[l]) !== String(incumbent[l]))
  if (!diffLevers.length) return undefined
  const baselinePredicted = predictConfig(surrogate, baseline)
  const steps: AblationStep[] = []
  let current: Record<string, unknown> = { ...baseline }
  let prev = baselinePredicted
  const remaining = new Set(diffLevers)
  while (remaining.size) {
    let pick: { lever: string; predicted: number; gain: number } | undefined
    for (const lever of remaining) {
      const predicted = predictConfig(surrogate, { ...current, [lever]: incumbent[lever] })
      const gain = orient(predicted, prev)
      if (!pick || gain > pick.gain) pick = { lever, predicted, gain }
    }
    if (!pick) break
    current = { ...current, [pick.lever]: incumbent[pick.lever] }
    steps.push({
      lever: pick.lever,
      from: String(baseline[pick.lever]),
      to: String(incumbent[pick.lever]),
      predicted: pick.predicted,
      gain: pick.gain,
    })
    prev = pick.predicted
    remaining.delete(pick.lever)
  }
  return {
    baseline,
    incumbent,
    baselinePredicted,
    incumbentPredicted: predictConfig(surrogate, incumbent),
    steps,
  }
}

/**
 * The surrogate-predicted criterion across two levers' observed value grid — the interaction surface that
 * answers "does A help universally or only at some B?". `undefined` when either lever has <2 values.
 */
export function interactionGrid(
  surrogate: ConfigSurrogate,
  runs: AnalysisRun[],
  criterion: AnalysisCriterion,
  leverA: string,
  leverB: string,
): InteractionGrid | undefined {
  const valid = validRunsFor(runs, criterion)
  const configs = valid.map((r) => r.config)
  const valsA = observedValues(valid, leverA)
  const valsB = observedValues(valid, leverB)
  if (valsA.size < 2 || valsB.size < 2 || !configs.length) return undefined
  const valuesA = [...valsA.keys()]
  const valuesB = [...valsB.keys()]
  const cells: number[] = []
  for (const a of valuesA) {
    for (const b of valuesB) {
      cells.push(
        meanOf(configs.map((c) => predictConfig(surrogate, { ...c, [leverA]: valsA.get(a), [leverB]: valsB.get(b) }))),
      )
    }
  }
  return { leverA, leverB, valuesA, valuesB, cells }
}

// --- Phase 4: deterministic PCA projection of the explored configs, coloured by performance ----------
// A 2-D intuition map (clusters/outliers), NOT a navigable space — the PC axes are not levers. One point
// per SETUP; numeric levers standardised, categorical one-hot; top-2 PCs via power iteration + deflation.

function dotVec(a: number[], b: number[]): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}
function normalizeVec(v: number[]): number[] {
  const norm = Math.sqrt(dotVec(v, v)) || 1
  return v.map((x) => x / norm)
}
/** Largest eigenpair of a symmetric matrix via power iteration from a FIXED (deterministic) start. */
function topEigenpair(m: number[][], d: number): { vector: number[]; value: number } {
  let v = normalizeVec(Array.from({ length: d }, (_, i) => Math.sin(i + 1) + 1e-3))
  let value = 0
  for (let iter = 0; iter < 200; iter++) {
    const mv = m.map((row) => dotVec(row, v))
    value = dotVec(v, mv)
    v = normalizeVec(mv)
  }
  return { vector: v, value }
}

/**
 * Project the explored SETUPS (config minus seed) onto the top-2 principal components, coloured by the
 * setup's criterion IQM. Numeric levers are z-scored, categorical levers one-hot encoded, and the PCs are
 * found by deterministic power iteration. Returns `null` with fewer than 3 setups or no encodable levers.
 */
export function pcaProjection(
  runs: AnalysisRun[],
  criterion: AnalysisCriterion,
): PcaProjection | null {
  const valid = validRunsFor(runs, criterion)
  if (valid.length < 3) return null
  const bySetup = new Map<string, AnalysisRun[]>()
  for (const r of valid) {
    const sig = setupSignatureOf(r)
    const g = bySetup.get(sig)
    if (g) g.push(r)
    else bySetup.set(sig, [r])
  }
  const rows = [...bySetup.values()].map((rs) => ({
    config: configWithout(rs[0].config, 'seed'),
    value: iqm(rs.map((r) => criterionValueOf(r, criterion)!)),
    runKeys: rs.map((r) => r.key),
  }))
  if (rows.length < 3) return null
  // Encode: numeric lever → one standardised column; categorical → one-hot per observed value.
  const columns: ((c: Record<string, unknown>) => number)[] = []
  for (const l of leverKindsOf(valid)) {
    if (l.kind === 'num') {
      columns.push((c) => {
        const v = Number(c[l.name])
        return Number.isFinite(v) ? v : 0
      })
    } else {
      for (const val of new Set(rows.map((r) => String(r.config[l.name])))) {
        columns.push((c) => (String(c[l.name]) === val ? 1 : 0))
      }
    }
  }
  const d = columns.length
  if (d === 0) return null
  const raw = rows.map((r) => columns.map((col) => col(r.config)))
  const n = raw.length
  const means: number[] = []
  const stds: number[] = []
  for (let j = 0; j < d; j++) {
    const colv = raw.map((row) => row[j])
    means.push(meanOf(colv))
    stds.push(Math.sqrt(varianceOf(colv)) || 1)
  }
  const X = raw.map((row) => row.map((v, j) => (v - means[j]) / stds[j]))
  const C: number[][] = Array.from({ length: d }, () => new Array(d).fill(0))
  for (let a = 0; a < d; a++) {
    for (let b = a; b < d; b++) {
      let s = 0
      for (let i = 0; i < n; i++) s += X[i][a] * X[i][b]
      C[a][b] = C[b][a] = s / n
    }
  }
  const trace = C.reduce((acc, row, i) => acc + row[i], 0) || 1
  const pc1 = topEigenpair(C, d)
  const C2 = C.map((row, a) => row.map((val, b) => val - pc1.value * pc1.vector[a] * pc1.vector[b]))
  const pc2 = d >= 2 ? topEigenpair(C2, d) : { vector: new Array(d).fill(0), value: 0 }
  const points: PcaPoint[] = rows.map((r, i) => ({
    x: dotVec(X[i], pc1.vector),
    y: dotVec(X[i], pc2.vector),
    value: r.value,
    key: r.runKeys[0],
    runKeys: r.runKeys,
  }))
  return {
    points,
    explainedVariance: [Math.max(0, pc1.value) / trace, Math.max(0, pc2.value) / trace],
    features: d,
  }
}

// --- Phase 5: the whole-space bundle ---
// One pass over EVERY run (seeds folded into setups so the surrogate trains on the denoised, far smaller
// distinct-config set) producing surrogate + fANOVA + coupling + ablation + PCA + recommendations. Run
// server-side and cached so the browser renders instead of fits. Coupling is O(levers²), so it is searched
// only among the high-effect levers — the inert rest can't interact meaningfully anyway.
const CONFIG_SPACE_TOP_LEVERS = 8
const CONFIG_SPACE_MIN_TOTAL = 0.03

/**
 * Fold every seed of a config into ONE synthetic run whose criterion value is the setup's IQM — the
 * denoised dataset the whole-space surrogate trains on (turns 5000 noisy runs into a few hundred setups).
 */
export function aggregateToSetupRuns(
  runs: AnalysisRun[],
  criterion: AnalysisCriterion,
): AnalysisRun[] {
  const valid = validRunsFor(runs, criterion)
  const groups = new Map<string, AnalysisRun[]>()
  for (const r of valid) {
    const sig = setupSignatureOf(r)
    const g = groups.get(sig)
    if (g) g.push(r)
    else groups.set(sig, [r])
  }
  const out: AnalysisRun[] = []
  for (const group of groups.values()) {
    const value = iqm(group.map((r) => criterionValueOf(r, criterion)!))
    const rep = group[0]
    const setup: AnalysisRun = { ...rep, config: configWithout(rep.config, 'seed'), status: 'completed' }
    // Store the aggregate where this criterion reads it, so the original criterion still works on setups.
    if (criterion.key === 'objective') setup.objective = value
    else if (criterion.key === 'durationMs') setup.durationMs = value
    else setup.metrics = { ...(rep.metrics ?? {}), [criterion.key]: value }
    out.push(setup)
  }
  return out
}

function pickKeys(config: Record<string, unknown>, keys: string[]): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const k of keys) if (k in config) out[k] = config[k]
  return out
}

// The value a conditional lever takes where it doesn't apply — a single sentinel so the surrogate sees it
// as constant there (no spurious effect) and the UI can show "doesn't apply here".
const CONDITIONAL_NA = 'n/a'

/** A conditional lever applies to a config only when every one of its `appliesWhen` conditions is met. */
function leverApplies(config: Record<string, unknown>, conds: Record<string, unknown[]>): boolean {
  return Object.entries(conds).every(([k, vals]) => vals.map(String).includes(String(config[k])))
}

/**
 * Normalise a config so any CONDITIONAL lever that doesn't apply (its `appliesWhen` isn't satisfied — e.g.
 * `forward_horizon` on a non-supervised model) is pinned to {@link CONDITIONAL_NA}. The lever then has no
 * variance where it's irrelevant, so the surrogate/fANOVA/interaction grid stop attributing noise to it.
 */
function normalizeConditionalLevers(
  config: Record<string, unknown>,
  appliesWhen: Record<string, Record<string, unknown[]>>,
): Record<string, unknown> {
  let out = config
  for (const [lever, conds] of Object.entries(appliesWhen)) {
    if (lever in out && !leverApplies(out, conds)) {
      if (out === config) out = { ...config }
      out[lever] = CONDITIONAL_NA
    }
  }
  return out
}

function dropNaValues(config: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(config)) if (v !== CONDITIONAL_NA) out[k] = v
  return out
}

/** A run is in an environment when every one of that environment's context-lever values matches its config. */
function matchesContext(config: Record<string, unknown>, env: Record<string, unknown>): boolean {
  return Object.entries(env).every(([k, v]) => String(config[k]) === String(v))
}

/** The distinct environments (combinations of context-lever values) across the runs, most-run first. */
function summarizeEnvironments(
  runs: AnalysisRun[],
  criterion: AnalysisCriterion,
  contextLevers: string[],
): EnvironmentSummary[] {
  if (!contextLevers.length) return []
  const groups = new Map<string, AnalysisRun[]>()
  for (const r of runs) {
    const sig = canonicalConfigString(pickKeys(r.config, contextLevers))
    const g = groups.get(sig)
    if (g) g.push(r)
    else groups.set(sig, [r])
  }
  const out: EnvironmentSummary[] = []
  for (const [signature, group] of groups) {
    const setupValues = aggregateToSetupRuns(group, criterion)
      .map((s) => criterionValueOf(s, criterion))
      .filter((v): v is number => v !== undefined)
    const best = setupValues.length
      ? criterion.direction === 'min'
        ? Math.min(...setupValues)
        : Math.max(...setupValues)
      : 0
    out.push({ signature, values: pickKeys(group[0].config, contextLevers), runCount: group.length, best })
  }
  return out.sort((a, b) => b.runCount - a.runCount)
}

/**
 * The whole-space bundle, scoped to ONE environment over the MODEL levers. `opts.contextLevers` are the
 * environment + dataset levers (market mechanics + which data): the analysis filters to a single
 * environment, strips those levers, and never recommends changing them — so configs from different
 * environments aren't blended. The cross-environment comparison (`environments`, `contextImportances`)
 * spans all runs. With no context levers it analyses the whole space together (backward compatible).
 */
export function computeConfigSpaceAnalysis(
  runs: AnalysisRun[],
  criterion: AnalysisCriterion,
  opts?: {
    contextLevers?: string[]
    environment?: Record<string, unknown>
    appliesWhen?: Record<string, Record<string, unknown[]>>
  },
): ConfigSpaceAnalysis | null {
  const valid0base = validRunsFor(runs, criterion)
  if (!valid0base.length) return null
  // Pin each conditional lever to `n/a` wherever it doesn't apply, so it can't pollute the analysis (e.g.
  // forward_horizon only varies among supervised models; it's inert for the rest).
  const appliesWhen = opts?.appliesWhen
  const valid0 = appliesWhen
    ? valid0base.map((r) => ({ ...r, config: normalizeConditionalLevers(r.config, appliesWhen) }))
    : valid0base
  const present = leversOf(valid0)
  const contextLevers = (opts?.contextLevers ?? []).filter((l) => present.includes(l))
  const environments = summarizeEnvironments(valid0, criterion, contextLevers)
  const contextImportances = contextLevers.length
    ? leverImportances(valid0, criterion).filter((s) => contextLevers.includes(s.lever))
    : []
  // Pick the environment to analyse within: the requested one, else the most-run one.
  const environment = !contextLevers.length
    ? null
    : opts?.environment
      ? pickKeys(opts.environment, contextLevers)
      : (environments[0]?.values ?? null)
  const envRuns = environment ? valid0.filter((r) => matchesContext(r.config, environment)) : valid0
  // Strip the context levers so the surrogate/importances/recommender see ONLY the model levers.
  const valid = contextLevers.length
    ? envRuns.map((r) => ({ ...r, config: configWithout(r.config, ...contextLevers) }))
    : envRuns
  if (!valid.length) return null

  const screening = leverImportances(valid, criterion)
  const ofat: Record<string, OfatAnalysis[]> = {}
  for (const s of screening) ofat[s.lever] = ofatContrasts(valid, s.lever, criterion)
  const setups = aggregateToSetupRuns(valid, criterion)
  const surrogate = fitConfigSurrogate(setups, criterion)
  const importances = fanovaImportances(surrogate, setups, criterion)
  const coupledLevers = [...importances]
    .filter((f) => f.total > CONFIG_SPACE_MIN_TOTAL)
    .sort((a, b) => b.total - a.total)
    .slice(0, CONFIG_SPACE_TOP_LEVERS)
    .map((f) => f.lever)
  const couplings = leverCouplings(surrogate, setups, criterion, coupledLevers)
  const ablation = ablationPath(surrogate, setups, criterion) ?? null
  const pca = pcaProjection(valid, criterion)
  const rawRecs = recommendExperiments(valid, criterion, { surrogate, setups })
  // Make each recommendation launchable + honest: re-normalise its conditional levers (so it never proposes
  // e.g. forward_horizon for an RL model), drop the inapplicable n/a placeholders, stamp the environment's
  // context values back on, and dedupe configs that collapse to the same thing.
  const recSeen = new Set<string>()
  const recommendations: ExperimentRecommendation[] = []
  for (const r of rawRecs) {
    let fixed = r.spec.fixed ?? {}
    if (appliesWhen) fixed = normalizeConditionalLevers(fixed, appliesWhen)
    fixed = dropNaValues(fixed)
    if (environment) fixed = { ...environment, ...fixed }
    const key = canonicalConfigString(fixed)
    if (recSeen.has(key)) continue
    recSeen.add(key)
    recommendations.push({ ...r, spec: { ...r.spec, fixed } })
  }

  return {
    criterion: { key: criterion.key, direction: criterion.direction },
    runCount: envRuns.length,
    setupCount: setups.length,
    levers: surrogate.levers.map((l) => l.name),
    setups,
    screening,
    ofat,
    surrogate,
    importances,
    couplings,
    coupledLevers,
    ablation,
    pca,
    recommendations,
    environment,
    environments,
    contextImportances,
    contextLevers,
  }
}
