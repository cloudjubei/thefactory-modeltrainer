import {
  EXPLORATION_ACTIVE_IMPORTANCE_FLOOR,
  EXPLORATION_BASIN_MIN_SPAN_FRACTION,
  EXPLORATION_BASIN_NOISE_MARGIN,
  EXPLORATION_BATCH_MAX,
  EXPLORATION_COVERAGE_PER_LEVER,
  EXPLORATION_DRY_ROUNDS,
  EXPLORATION_MAX_ACTIVE_LEVERS,
  EXPLORATION_LADDER_MAX_REGION_AXES,
  EXPLORATION_MAX_REFINE_DEPTH,
  EXPLORATION_MAX_REGION_AXES,
  EXPLORATION_REFINE_MAX_STEP_FRACTION,
  EXPLORATION_REFINE_MIN_STEP_FRACTION,
  EXPLORATION_SCREEN_SAMPLES,
  XAI_MIN_SEEDS,
} from './modelTrainerConstants.js'
import type {
  AnalysisCriterion,
  AnalysisRun,
  Basin,
  ExperimentRecommendation,
  ExplorationBudget,
  ExplorationState,
  ExplorationStep,
  FitnessObjective,
  TrainerManifest,
} from './modelTrainerTypes.js'
import {
  assembleChampionVerdict,
  convergenceGatedBySplits,
  incumbentSplitHoldout,
  splitLeversOf,
  type SplitHoldout,
} from './diagnosticsUtils.js'
import {
  aggregateToSetupRuns,
  criterionValueOf,
  iqm,
  leverImportances,
  paretoFrontier,
  recommendExperiments,
} from './xaiUtils.js'

// The exploration autopilot's pure core: `nextExplorationStep` is a staged reducer over the run archive.
// It composes the xAI primitives (importances → basins → acquisition) into the S0→S4 search and emits the
// next batch + advanced state. No I/O, no launching — the activity layer drives the loop and persists state.

// ---------------------------------------------------------------------------------------------------
// small pure helpers

const isBetter = (a: number, b: number, dir: 'max' | 'min'): boolean => (dir === 'max' ? a > b : a < b)
/** Oriented gain of `v` over `ref` (always ≥0 means "better by that much"). */
const gainOver = (v: number, ref: number, dir: 'max' | 'min'): number => (dir === 'max' ? v - ref : ref - v)

function std(nums: number[]): number {
  if (nums.length < 2) return 0
  const m = nums.reduce((a, b) => a + b, 0) / nums.length
  return Math.sqrt(nums.reduce((a, b) => a + (b - m) ** 2, 0) / (nums.length - 1))
}
const mean = (nums: number[]): number => (nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : 0)
const uniq = <T,>(xs: T[]): T[] => [...new Set(xs)]
const seedRange = (n: number): number[] => Array.from({ length: n }, (_, i) => i)

/**
 * A robust objective noise-floor estimate from a POPULATED archive: the median within-setup (config-minus-seed)
 * seed std across every setup that has ≥2 seeds. Returns undefined when no setup is seed-replicated. Lets a
 * project with lots of pre-existing runs calibrate WITHOUT re-running the exact manifest-default config × N seeds
 * (which it may never hold — the default combo was never run, or conditional levers get stamped n/a at runtime).
 */
function archiveNoiseFloor(runs: AnalysisRun[], criterion: AnalysisCriterion): number | undefined {
  const groups = new Map<string, number[]>()
  for (const r of runs) {
    const v = criterionValueOf(r, criterion)
    if (v == null) continue
    const key = JSON.stringify(
      Object.keys(r.config || {})
        .filter((k) => k !== 'seed')
        .sort()
        .map((k) => [k, (r.config as Record<string, unknown>)[k]]),
    )
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key)!.push(v)
  }
  const stds: number[] = []
  for (const vals of groups.values()) if (vals.length >= 2) stds.push(std(vals))
  if (!stds.length) return undefined
  stds.sort((a, b) => a - b)
  return stds[Math.floor(stds.length / 2)]
}

function without(obj: Record<string, unknown>, ...keys: string[]): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(obj)) if (!keys.includes(k)) out[k] = v
  return out
}

/** True when every key in `keys` matches (by stringified value) between `config` and `target`. */
function matchesConfig(config: Record<string, unknown>, target: Record<string, unknown>, keys: string[]): boolean {
  return keys.every((k) => String(config[k]) === String(target[k]))
}

/** Levers the search may vary: model-scope (or unscoped), excluding `seed` and infrastructure. */
function searchableLevers(manifest: TrainerManifest): string[] {
  return Object.entries(manifest.levers)
    .filter(([name, spec]) => name !== 'seed' && (spec.scope ?? 'model') === 'model')
    .map(([name]) => name)
}

const isNumericManifestLever = (manifest: TrainerManifest, lever: string): boolean =>
  manifest.levers[lever]?.type === 'number'

function defaultsOf(manifest: TrainerManifest): Record<string, unknown> {
  const d: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(manifest.levers)) if (v.default !== undefined) d[k] = v.default
  return d
}

const criterionOf = (state: ExplorationState): AnalysisCriterion => ({
  key: 'objective',
  direction: state.objective.direction,
})

/** The trivial-baseline reference: the mean of the standard-battery baseline metric, if reported. */
function baselineOf(runs: AnalysisRun[]): number | undefined {
  const keys = ['baseline', 'baseline_return', 'baseline_score']
  const vals: number[] = []
  for (const r of runs) {
    const m = r.metrics
    if (!m) continue
    for (const k of keys) {
      if (typeof m[k] === 'number') {
        vals.push(m[k])
        break
      }
    }
  }
  return vals.length ? mean(vals) : undefined
}

function bestObjectiveOf(runs: AnalysisRun[], criterion: AnalysisCriterion): number | undefined {
  let best: number | undefined
  for (const r of runs) {
    const v = criterionValueOf(r, criterion)
    if (v == null) continue
    if (best === undefined || isBetter(v, best, criterion.direction)) best = v
  }
  return best
}

// ---------------------------------------------------------------------------------------------------
// basin clustering — the enumerated maxima

/** A value is categorical (a region axis) when it's a boolean, an object/array, or a non-numeric string. */
function isCategoricalValue(v: unknown): boolean {
  if (typeof v === 'number') return false
  if (typeof v === 'boolean') return true
  if (typeof v === 'string') return !Number.isFinite(Number(v))
  return true // arrays / objects (e.g. net_arch [64,64])
}

/**
 * A region axis is a categorical active lever; numeric active levers climb WITHIN one. The manifest's DECLARED
 * type is authoritative when available — a `choice`/`boolean` lever is always a region axis (even one whose
 * values look numeric, e.g. n_layers ∈ {1,2,3}), a `number` lever never is. Only when the type is unknown do we
 * fall back to inspecting the observed values.
 */
function regionLeversOf(setups: AnalysisRun[], activeLevers: string[], manifest?: TrainerManifest): string[] {
  return activeLevers.filter((lever) => {
    const type = manifest?.levers[lever]?.type
    if (type === 'choice' || type === 'boolean') return true
    if (type === 'number') return false
    const vals = setups.map((s) => s.config[lever]).filter((v) => v !== undefined)
    return vals.length > 0 && vals.some(isCategoricalValue)
  })
}

const regionKey = (config: Record<string, unknown>, regionLevers: string[]): string =>
  JSON.stringify(regionLevers.map((l) => [l, config[l]]))

/**
 * Cluster the run archive into basins — one per good discrete region (a candidate local maximum). Numeric
 * active levers form the local surface climbed within a basin; when there are no categorical active levers
 * the whole space is a single basin. A region qualifies only when its peak beats the trivial baseline by
 * more than the noise floor AND captures a real fraction of the baseline→best span.
 */
export function clusterBasins(
  runs: AnalysisRun[],
  criterion: AnalysisCriterion,
  activeLevers: string[],
  noiseFloor: number,
  baseline?: number,
  manifest?: TrainerManifest,
): Basin[] {
  const setups = aggregateToSetupRuns(runs, criterion)
  if (!setups.length) return []
  const dir = criterion.direction
  const regionLevers = regionLeversOf(setups, activeLevers, manifest)

  const byRegion = new Map<string, AnalysisRun[]>()
  for (const s of setups) {
    const key = regionKey(s.config, regionLevers)
    const g = byRegion.get(key)
    if (g) g.push(s)
    else byRegion.set(key, [s])
  }

  const peaks = [...byRegion.entries()].map(([, group]) => {
    let peak = group[0]
    for (const s of group) if (isBetter(criterionValueOf(s, criterion)!, criterionValueOf(peak, criterion)!, dir)) peak = s
    return peak
  })
  const peakVals = peaks.map((p) => criterionValueOf(p, criterion)!)
  const best = peakVals.reduce((a, b) => (isBetter(a, b, dir) ? a : b))
  const worst = peakVals.reduce((a, b) => (isBetter(a, b, dir) ? b : a))
  const ref = baseline ?? baselineOf(runs) ?? worst
  const margin = Math.max(EXPLORATION_BASIN_NOISE_MARGIN * noiseFloor, EXPLORATION_BASIN_MIN_SPAN_FRACTION * Math.abs(best - ref))

  const basins: Basin[] = []
  for (const peak of peaks) {
    const value = criterionValueOf(peak, criterion)!
    if (gainOver(value, ref, dir) < margin) continue // no better than random → not a maximum
    const region: Record<string, unknown> = {}
    for (const l of regionLevers) region[l] = peak.config[l]
    const memberRunKeys = runs
      .filter((r) => matchesConfig(r.config, peak.config, regionLevers))
      .map((r) => r.key)
    basins.push({
      id: regionKey(peak.config, regionLevers),
      region,
      centerConfig: peak.config,
      peakObjective: value,
      peakCI: peak.ci,
      peakSeeds: peak.seeds ?? 1,
      plateaued: false,
      memberRunKeys,
    })
  }
  const sorted = basins.sort((a, b) => (isBetter(a.peakObjective, b.peakObjective, dir) ? -1 : 1))
  return qualifyParetoBasins(sorted, runs, manifest?.fitness)
}

/**
 * A6 — Pareto-qualify basins on the scorecard's MULTI-objective `fitness`. With a single objective the scalar
 * `peakObjective` ranking already suffices, so this only annotates when ≥2 objectives are declared: each
 * basin's fitness vector is its member runs' robust (IQM) value per objective (a missing metric is mapped to
 * the WORST value for its direction, so it's dominated), and the non-dominated basins (`paretoFrontier`) are
 * flagged `onParetoFront` — the trade-off candidates worth keeping even when they aren't the single-scalar best.
 */
export function qualifyParetoBasins(
  basins: Basin[],
  runs: AnalysisRun[],
  fitness: FitnessObjective[] | undefined,
): Basin[] {
  if (!fitness || fitness.length < 2 || basins.length < 2) return basins
  const runByKey = new Map(runs.map((r) => [r.key, r]))
  const points = basins.map((b) =>
    fitness.map((f) => {
      const vals = b.memberRunKeys
        .map((k) => runByKey.get(k))
        .filter((r): r is AnalysisRun => !!r)
        .map((r) => criterionValueOf(r, { key: f.metric, direction: f.direction }))
        .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
      return vals.length ? iqm(vals) : f.direction === 'min' ? Infinity : -Infinity
    }),
  )
  const front = new Set(paretoFrontier(points, fitness.map((f) => f.direction)))
  return basins.map((b, i) => ({ ...b, onParetoFront: front.has(i) }))
}

const bestBasin = (basins: Basin[], criterion: AnalysisCriterion): Basin | undefined =>
  basins.reduce<Basin | undefined>(
    (best, b) => (!best || isBetter(b.peakObjective, best.peakObjective, criterion.direction) ? b : best),
    undefined,
  )

// ---------------------------------------------------------------------------------------------------
// batch builders

/** A deterministic space-filling sample across the full space — the screen stage's probe. */
function screenSampleRec(manifest: TrainerManifest, levers: string[], n: number): ExperimentRecommendation {
  const configs: Array<{ config: Record<string, unknown> }> = []
  for (let i = 0; i < n; i++) {
    const config: Record<string, unknown> = {}
    for (let j = 0; j < levers.length; j++) {
      const lever = levers[j]
      const spec = manifest.levers[lever]
      // decorrelate levers by permuting each one's stratum index independently
      const idx = (i + j * 5 + 1) % n
      if (spec.type === 'choice' && spec.choices?.length) {
        config[lever] = spec.choices[idx % spec.choices.length]
      } else if (spec.type === 'boolean') {
        config[lever] = idx % 2 === 0
      } else {
        const [lo, hi] = spec.range ?? [0, 1]
        config[lever] = lo + ((idx + 0.5) / n) * (hi - lo)
      }
    }
    configs.push({ config })
  }
  return {
    kind: 'missing-cell',
    reason: `screen: ${n} space-filling samples across ${levers.length} levers`,
    runCount: configs.length,
    spec: { configs },
    priority: 90,
  }
}

/** Ensure every value of a categorical active lever has been tried at least once (basin discovery). */
function discreteCoverageRecs(
  runs: AnalysisRun[],
  manifest: TrainerManifest,
  activeLevers: string[],
  frozenLevers: Record<string, unknown>,
): ExperimentRecommendation[] {
  const recs: ExperimentRecommendation[] = []
  const defaults = defaultsOf(manifest)
  for (const lever of activeLevers) {
    const spec = manifest.levers[lever]
    if (spec.type !== 'choice' || !spec.choices?.length) continue
    const seen = new Set(runs.map((r) => String(r.config[lever])))
    const missing = spec.choices.filter((c) => !seen.has(String(c)))
    if (!missing.length) continue
    recs.push({
      kind: 'missing-cell',
      reason: `cover untried ${lever}: ${missing.map(String).join(', ')}`,
      runCount: missing.length,
      spec: { fixed: { ...defaults, ...frozenLevers }, sweep: { [lever]: missing } },
      priority: 80,
    })
  }
  return recs
}

/** Coordinate-ascent refinement + seed-stabilization inside one basin. Exported for direct unit testing. */
export function localRefineRecs(
  basin: Basin,
  runs: AnalysisRun[],
  manifest: TrainerManifest,
  state: ExplorationState,
): ExperimentRecommendation[] {
  const recs: ExperimentRecommendation[] = []
  const region = basin.region
  const regionKeys = Object.keys(region)
  const center = basin.centerConfig
  const numericActive = state.activeLevers.filter((l) => isNumericManifestLever(manifest, l) && !(l in region))
  const regionRuns = runs.filter((r) => matchesConfig(r.config, region, regionKeys))

  // 1. stabilize the peak with more seeds
  if (basin.peakSeeds < XAI_MIN_SEEDS) {
    const peakRuns = regionRuns.filter((r) => matchesConfig(r.config, center, numericActive))
    const have = new Set(peakRuns.map((r) => r.seed ?? 0))
    const need = seedRange(XAI_MIN_SEEDS).filter((s) => !have.has(s))
    if (need.length) {
      recs.push({
        kind: 'thin-seeds',
        reason: `stabilize basin ${basin.id} peak (+${need.length} seeds)`,
        runCount: need.length,
        spec: { fixed: without(center, 'seed'), seeds: need },
        priority: 60,
      })
    }
  }

  // 2. ADAPTIVE coordinate-ascent sweep of each numeric lever around the center. The step is HALF the
  //    distance to the nearest tried point on each side (falling back to the range edge) — so it bisects
  //    toward the peak, TIGHTENING every round as neighbours close in. Capped at range/8 (never jumps
  //    coarser than the old fixed step) and floored at range/64 (below that the lever is resolved, so the
  //    basin can plateau). A fixed step overshoots a narrow optimum; this seats it.
  for (const lever of numericActive) {
    const spec = manifest.levers[lever]
    const range = spec.range
    if (!range) continue
    const span = range[1] - range[0]
    if (span <= 0) continue
    const c = Number(center[lever])
    const maxStep = span * EXPLORATION_REFINE_MAX_STEP_FRACTION
    // DEEPENING: each refine level halves the resolution floor, so a basin that plateaued at the coarse floor
    // can be climbed further once the ladder deepens (the "never stop while the space is uncovered" contract).
    const minStep = (span * EXPLORATION_REFINE_MIN_STEP_FRACTION) / Math.pow(2, state.refineDepth ?? 0)
    const triedVals = regionRuns.map((r) => Number(r.config[lever])).filter((v) => isFinite(v))
    const eps = minStep / 4
    // nearest tried point strictly below / above the center, else the range edge
    const below = Math.max(range[0], ...triedVals.filter((v) => v < c - eps))
    const above = Math.min(range[1], ...triedVals.filter((v) => v > c + eps))
    const loStep = Math.min((c - below) / 2, maxStep)
    const hiStep = Math.min((above - c) / 2, maxStep)
    const tol = minStep / 2
    const fresh: number[] = []
    for (const v of [c - loStep, c - loStep / 2, c + hiStep / 2, c + hiStep]) {
      if (v < range[0] || v > range[1]) continue
      if (Math.abs(v - c) < minStep) continue // finer than the resolution floor → resolved here
      if (triedVals.some((t) => Math.abs(t - v) <= tol)) continue
      if (fresh.some((f) => Math.abs(f - v) <= tol)) continue
      fresh.push(Number(v.toFixed(6)))
    }
    if (fresh.length) {
      recs.push({
        kind: 'acquisition',
        reason: `climb ${lever} around ${c.toFixed(3)} in basin ${basin.id}`,
        runCount: fresh.length * XAI_MIN_SEEDS,
        spec: { fixed: without(center, lever, 'seed'), sweep: { [lever]: fresh }, seeds: seedRange(XAI_MIN_SEEDS) },
        priority: 50,
      })
    }
  }
  return recs
}

// ---------------------------------------------------------------------------------------------------
// the reducer

export function initExplorationState(
  manifest: TrainerManifest,
  budget?: Partial<ExplorationBudget>,
): ExplorationState {
  return {
    recordType: manifest.recordType,
    objective: manifest.objective,
    stage: 'calibrate',
    activeLevers: [],
    frozenLevers: {},
    basins: [],
    budget: { spentRuns: 0, maxRuns: budget?.maxRuns, maxConcurrent: budget?.maxConcurrent },
    regret: [],
    dryRounds: 0,
    done: false,
  }
}

/**
 * Trim a batch to fit the remaining run budget — pack whole recs (smallest first) up to `remaining`, so the
 * autopilot can never launch past its ceiling. If nothing fits, keep the single smallest rec to make progress
 * (the next round then hits the ceiling and converges); overshoot is bounded by one rec.
 */
function clampBatch(batch: ExperimentRecommendation[], remaining: number): ExperimentRecommendation[] {
  if (!Number.isFinite(remaining)) return batch
  if (remaining <= 0) return []
  const out: ExperimentRecommendation[] = []
  let used = 0
  for (const rec of [...batch].sort((a, b) => a.runCount - b.runCount)) {
    if (used + rec.runCount <= remaining) {
      out.push(rec)
      used += rec.runCount
    }
  }
  if (!out.length && batch.length) return [batch.reduce((m, r) => (r.runCount < m.runCount ? r : m))]
  return out
}

function appendRegret(
  regret: ExplorationState['regret'],
  runsSpent: number,
  best: number,
): ExplorationState['regret'] {
  const last = regret[regret.length - 1]
  if (last && last.runsSpent === runsSpent) {
    if (best === last.bestObjective) return regret
    return [...regret.slice(0, -1), { runsSpent, bestObjective: best }]
  }
  return [...regret, { runsSpent, bestObjective: best }]
}

export function nextExplorationStep(
  state: ExplorationState,
  runs: AnalysisRun[],
  manifest: TrainerManifest,
  opts?: { targetObjective?: number; exhausted?: boolean },
): ExplorationStep {
  const criterion = criterionOf(state)
  const archiveRuns = runs.length
  // Budget + progress count runs THIS exploration produced, not the whole pre-existing archive: capture the
  // archive size at map start (first step), then spentRuns = archiveNow - baseline. On a project with 20k runs
  // this keeps a run budget from tripping "budget exhausted" on round 0. Reset to 0 when the archive is emptied.
  const baselineRuns = archiveRuns === 0 ? 0 : state.baselineRuns ?? archiveRuns
  const spentRuns = Math.max(0, archiveRuns - baselineRuns)
  const best = bestObjectiveOf(runs, criterion)

  const withMeta = (s: ExplorationState): ExplorationState => ({
    ...s,
    baselineRuns,
    budget: { ...s.budget, spentRuns },
    regret: best === undefined ? s.regret : appendRegret(s.regret, spentRuns, best),
    updatedAt: state.updatedAt,
  })
  const remaining = state.budget.maxRuns != null ? Math.max(0, state.budget.maxRuns - spentRuns) : Infinity
  const mk = (
    stage: ExplorationState['stage'],
    batch: ExperimentRecommendation[],
    rationale: string,
    stateNext: ExplorationState,
    done: boolean,
  ): ExplorationStep => ({
    stage,
    batch: done ? [] : clampBatch(batch, remaining),
    rationale,
    stateNext: withMeta(stateNext),
    done,
  })

  if (state.paused) return mk(state.stage, [], 'paused', state, state.done)
  if (state.budget.maxRuns !== undefined && spentRuns >= state.budget.maxRuns) {
    return converge(state, runs, criterion, mk, 'budget exhausted')
  }
  // An EMPTY run archive (e.g. the user deleted every run) makes any advanced/converged state stale — there is
  // nothing to have converged on, so restart from calibrate rather than re-declaring a convergence (or trying
  // to refine/cover) over runs that no longer exist. Resets the derived fields, keeps budget/steer. Keys off the
  // ARCHIVE being empty, not spentRuns (which is 0 on round 0 of every populated project too).
  if (archiveRuns === 0) {
    const fresh: ExplorationState = {
      ...state,
      stage: 'calibrate',
      done: false,
      basins: [],
      activeLevers: [],
      frozenLevers: {},
      refineDepth: 0,
      dryRounds: 0,
      declaredBasinId: undefined,
      noiseFloor: undefined,
    }
    return stepCalibrate(fresh, runs, manifest, criterion, mk)
  }
  if (state.done || state.stage === 'converged') {
    return mk('converged', [], 'converged', { ...state, stage: 'converged', done: true }, true)
  }

  let step: ExplorationStep
  switch (state.stage) {
    case 'calibrate':
      step = stepCalibrate(state, runs, manifest, criterion, mk)
      break
    case 'screen':
      step = stepScreen(state, runs, manifest, criterion, mk)
      break
    case 'global':
      step = stepGlobal(state, runs, manifest, criterion, mk, opts)
      break
    case 'local':
      step = stepLocal(state, runs, manifest, criterion, mk, opts)
      break
    default:
      step = converge(state, runs, criterion, mk, 'done', manifest)
      break
  }
  // Split-consistency GATE: never crown an incumbent that isn't robust across the declared split axis while
  // there are unrun splits + budget to try them — the deterministic fix for single-window false-convergence.
  const budgetLeft = state.budget.maxRuns == null || spentRuns < state.budget.maxRuns
  return gateConvergenceOnSplits(step, state, runs, manifest, criterion, budgetLeft, mk)
}

type Mk = (
  stage: ExplorationState['stage'],
  batch: ExperimentRecommendation[],
  rationale: string,
  stateNext: ExplorationState,
  done: boolean,
) => ExplorationStep

function stepCalibrate(
  state: ExplorationState,
  runs: AnalysisRun[],
  manifest: TrainerManifest,
  criterion: AnalysisCriterion,
  mk: Mk,
): ExplorationStep {
  const defaults = defaultsOf(manifest)
  const nonSeedKeys = Object.keys(defaults).filter((k) => k !== 'seed')
  const calibRuns = runs.filter((r) => matchesConfig(r.config, defaults, nonSeedKeys))
  const seedsPresent = new Set(calibRuns.map((r) => r.seed ?? 0)).size

  if (seedsPresent >= XAI_MIN_SEEDS) {
    const vals = calibRuns.map((r) => criterionValueOf(r, criterion)).filter((v): v is number => v != null)
    return stepScreen({ ...state, stage: 'screen', noiseFloor: std(vals) }, runs, manifest, criterion, mk)
  }

  // Populated project whose archive doesn't hold the manifest-default config × N seeds (the default combo was
  // never run, or conditional levers stamp differently at runtime so produced runs don't match it): estimate the
  // noise floor from ANY seed-replicated setups on record and advance — otherwise the search stalls at calibrate
  // re-proposing a default-config batch it can't recognize (the BlackSwan stuck-at-calibrate bug).
  const archiveNoise = archiveNoiseFloor(runs, criterion)
  if (archiveNoise !== undefined) {
    return stepScreen({ ...state, stage: 'screen', noiseFloor: archiveNoise }, runs, manifest, criterion, mk)
  }

  const rec: ExperimentRecommendation = {
    kind: 'thin-seeds',
    reason: `calibrate seed-noise on the default config (×${XAI_MIN_SEEDS} seeds)`,
    runCount: XAI_MIN_SEEDS,
    spec: { fixed: without(defaults, 'seed'), seeds: seedRange(XAI_MIN_SEEDS) },
    priority: 100,
  }
  return mk('calibrate', [rec], 'measure the objective noise floor', state, false)
}

function stepScreen(
  state: ExplorationState,
  runs: AnalysisRun[],
  manifest: TrainerManifest,
  criterion: AnalysisCriterion,
  mk: Mk,
): ExplorationStep {
  const searchable = searchableLevers(manifest)
  const setups = aggregateToSetupRuns(runs, criterion)
  if (setups.length < EXPLORATION_SCREEN_SAMPLES) {
    const rec = screenSampleRec(manifest, searchable, EXPLORATION_SCREEN_SAMPLES - setups.length)
    return mk('screen', [rec], 'space-filling screen of the full space', state, false)
  }

  const imps = leverImportances(runs, criterion)
  const impOf = (lever: string): number => imps.find((i) => i.lever === lever)?.importance ?? 0
  const discrete = searchable.filter((l) => !isNumericManifestLever(manifest, l))
  const numeric = searchable.filter((l) => isNumericManifestLever(manifest, l))

  // Basin AXES = at most EXPLORATION_MAX_REGION_AXES categorical levers, `model_name` (the model-identity lever)
  // preferred, then by importance. More categoricals than that would explode the region cross-product (BlackSwan
  // has 14 discretes), so the rest are frozen and tuned WITHIN a basin — this is what keeps `model_name` the
  // TOP-level basin axis rather than one of many co-equal ones.
  const regionAxes = discrete
    .slice()
    .sort((a, b) => (b === 'model_name' ? 1 : 0) - (a === 'model_name' ? 1 : 0) || impOf(b) - impOf(a) || (a < b ? -1 : 1))
    .slice(0, EXPLORATION_MAX_REGION_AXES)

  // Numeric climb dims: important numerics filling the remaining active budget (always at least one when numeric
  // levers exist — a many-categorical manifest must NOT be left with zero numeric slots to climb).
  let activeNumeric = numeric.filter((l) => impOf(l) >= EXPLORATION_ACTIVE_IMPORTANCE_FLOOR)
  if (!activeNumeric.length && numeric.length) {
    activeNumeric = [numeric.slice().sort((a, b) => impOf(b) - impOf(a))[0]] // keep at least one climb dimension
  }
  const numericSlots = Math.max(numeric.length ? 1 : 0, EXPLORATION_MAX_ACTIVE_LEVERS - regionAxes.length)
  activeNumeric = activeNumeric.sort((a, b) => impOf(b) - impOf(a)).slice(0, numericSlots)

  const steerActive = (state.steer?.pinActive ?? []).filter((l) => searchable.includes(l))
  const active = uniq([...regionAxes, ...activeNumeric, ...steerActive])

  // Freeze every searchable lever NOT active at its best-so-far value. A numeric freezes to a number; a
  // categorical/boolean keeps its category value (coercing it to a number would set NaN).
  const frozenLevers: Record<string, unknown> = {}
  for (const l of searchable) {
    if (active.includes(l)) continue
    const best = imps.find((i) => i.lever === l)?.bestValue
    if (best === undefined) frozenLevers[l] = manifest.levers[l].default
    else frozenLevers[l] = isNumericManifestLever(manifest, l) ? Number(best) : best
  }
  if (state.steer?.pinFrozen) Object.assign(frozenLevers, state.steer.pinFrozen)

  return stepGlobal(
    { ...state, stage: 'global', activeLevers: active, frozenLevers, dryRounds: 0 },
    runs,
    manifest,
    criterion,
    mk,
    undefined,
  )
}

function stepGlobal(
  state: ExplorationState,
  runs: AnalysisRun[],
  manifest: TrainerManifest,
  criterion: AnalysisCriterion,
  mk: Mk,
  opts?: { targetObjective?: number; exhausted?: boolean },
): ExplorationStep {
  const basins = clusterBasins(runs, criterion, state.activeLevers, state.noiseFloor ?? 0, baselineOf(runs), manifest)
  const known = new Set(state.basins.map((b) => b.id))
  const foundNew = basins.some((b) => !known.has(b.id))
  const dryRounds = foundNew ? 0 : state.dryRounds + 1
  const state2 = { ...state, basins, dryRounds }

  // The controller says every global proposal is already run — global search has nothing new to add, so
  // advance to climbing the basins (drop the flag so `local` refines rather than immediately converging).
  if (opts?.exhausted) {
    return stepLocal({ ...state2, stage: 'local' }, runs, manifest, criterion, mk, {
      targetObjective: opts.targetObjective,
    })
  }

  if (dryRounds >= EXPLORATION_DRY_ROUNDS) {
    return stepLocal({ ...state2, stage: 'local' }, runs, manifest, criterion, mk, opts)
  }

  const coverage = discreteCoverageRecs(runs, manifest, state.activeLevers, state.frozenLevers)
  const acquisition = coverage.length ? [] : globalAcquisitionRecs(state, runs, manifest, criterion)
  const batch = [...coverage, ...acquisition].slice(0, EXPLORATION_BATCH_MAX)
  if (!batch.length) {
    // nothing new to probe — fall through to climbing the basins we have
    return stepLocal({ ...state2, stage: 'local' }, runs, manifest, criterion, mk, opts)
  }
  return mk('global', batch, `explore for new basins (round, dry=${dryRounds})`, state2, false)
}

/** Acquisition over the reduced (active-lever-only) space, with the frozen levers merged back into each spec. */
function globalAcquisitionRecs(
  state: ExplorationState,
  runs: AnalysisRun[],
  manifest: TrainerManifest,
  criterion: AnalysisCriterion,
): ExperimentRecommendation[] {
  const frozenKeys = Object.keys(state.frozenLevers)
  const reduced = runs.map((r) => ({ ...r, config: without(r.config, ...frozenKeys) }))
  const recs = recommendExperiments(reduced, criterion)
  return recs.slice(0, 4).map((rec) => ({
    ...rec,
    spec: { ...rec.spec, fixed: { ...state.frozenLevers, ...(rec.spec.fixed ?? {}) } },
  }))
}

function stepLocal(
  state: ExplorationState,
  runs: AnalysisRun[],
  manifest: TrainerManifest,
  criterion: AnalysisCriterion,
  mk: Mk,
  opts?: { targetObjective?: number; exhausted?: boolean },
): ExplorationStep {
  const clustered = clusterBasins(runs, criterion, state.activeLevers, state.noiseFloor ?? 0, baselineOf(runs), manifest)
  const target = opts?.targetObjective
  const withPlateau = clustered.map((b) => ({
    ...b,
    plateaued: isPlateaued(b, runs, manifest, state, criterion, target),
  }))
  const pending = withPlateau.filter((b) => !b.plateaued)
  const state2 = { ...state, basins: withPlateau }

  // When the current active subspace is climbed out — every basin plateaued, no fresh refinement, or the
  // controller reports the batch fully redundant — DON'T converge. A plateau in this subspace is not the whole
  // space: ESCALATE (unfreeze a fixed lever, then deepen the numeric resolution). Convergence is reserved for
  // when that ladder is fully dry — the only honest "the search space is covered".
  const batch = pending.flatMap((b) => localRefineRecs(b, runs, manifest, state)).slice(0, EXPLORATION_BATCH_MAX)
  if (opts?.exhausted || !pending.length || !batch.length) {
    return expand(state2, withPlateau, runs, manifest, criterion, mk, target)
  }
  return mk('local', batch, `climb ${pending.length} basin(s)`, state2, false)
}

/**
 * The escalation ladder — what the search does INSTEAD of converging when the current subspace is climbed out.
 * Rung 1 (widen): unfreeze the most-important still-fixed numeric lever and sweep it, re-opening the search
 * along a previously-ignored dimension. Rung 2 (cover): keep SPACE-FILLING the active numeric space until it
 * holds the density target of distinct setups — WITHOUT this a pure-numeric problem converges the instant its
 * single best neighbourhood is locally resolved, leaving the rest of the space (other good regions) untried.
 * Only when every lever is unfrozen AND the space is covered at the current resolution does it converge —
 * "Explore more" then raises `refineDepth` for a finer, denser sweep. Resolution is NEVER auto-deepened.
 */
function expand(
  state: ExplorationState,
  basins: Basin[],
  runs: AnalysisRun[],
  manifest: TrainerManifest,
  criterion: AnalysisCriterion,
  mk: Mk,
  _target?: number,
): ExplorationStep {
  // Rung 1 — unfreeze the highest-importance numeric lever that is still pinned, and sweep it across its range.
  const frozenNumeric = Object.keys(state.frozenLevers).filter(
    (l) => isNumericManifestLever(manifest, l) && !!manifest.levers[l]?.range,
  )
  if (frozenNumeric.length) {
    const imps = leverImportances(runs, criterion)
    const impOf = (l: string): number => imps.find((i) => i.lever === l)?.importance ?? 0
    const lever = frozenNumeric.slice().sort((a, b) => impOf(b) - impOf(a) || (a < b ? -1 : 1))[0]
    const frozenLevers = without(state.frozenLevers, lever)
    const activeLevers = uniq([...state.activeLevers, lever])
    const widened = { ...state, stage: 'global' as const, activeLevers, frozenLevers, dryRounds: 0, basins }
    const sweep = unfreezeSweepRec(lever, frozenLevers, runs, manifest, criterion)
    if (sweep) {
      return mk('global', [sweep], `unfreeze ${lever} — widen the search into a previously-fixed lever`, widened, false)
    }
    // The lever was already swept across its range — just re-enter global search on the widened space.
    return stepGlobal(widened, runs, manifest, criterion, mk, undefined)
  }

  // Rung 2 — space-fill the active numeric space until it reaches the coverage density target.
  const coverage = coverageGridRecs(state, runs, manifest, criterion)
  if (coverage.length) {
    return mk('local', coverage, `cover the space — ${coverage[0].runCount} space-filling sample(s)`, { ...state, stage: 'local', basins }, false)
  }

  // Rung 3 — the numeric space is unfrozen AND covered: grow into a still-FROZEN CATEGORICAL dimension (open a
  // new basin axis) so a lever the screen capped out of the region cross-product finally gets its other values
  // explored. One per escalation, importance-ranked, and only up to EXPLORATION_LADDER_MAX_REGION_AXES active
  // categorical axes so the region product can't explode on a many-categorical manifest. Deliberately AFTER
  // numerics+coverage — exhaust the current space before multiplying it.
  const setups = aggregateToSetupRuns(runs, criterion)
  const regionAxes = regionLeversOf(setups, state.activeLevers, manifest)
  const frozenCategorical = Object.keys(state.frozenLevers).filter(
    (l) => !isNumericManifestLever(manifest, l) && (categoricalChoices(manifest, l)?.length ?? 0) > 1,
  )
  if (frozenCategorical.length && regionAxes.length < EXPLORATION_LADDER_MAX_REGION_AXES) {
    const imps = leverImportances(runs, criterion)
    const impOf = (l: string): number => imps.find((i) => i.lever === l)?.importance ?? 0
    const lever = frozenCategorical.slice().sort((a, b) => impOf(b) - impOf(a) || (a < b ? -1 : 1))[0]
    const frozenLevers = without(state.frozenLevers, lever)
    const activeLevers = uniq([...state.activeLevers, lever])
    const widened = { ...state, stage: 'global' as const, activeLevers, frozenLevers, dryRounds: 0, basins }
    const sweep = unfreezeCategoricalRec(lever, frozenLevers, runs, manifest, criterion)
    if (sweep) {
      return mk('global', [sweep], `unfreeze ${lever} — open a new basin axis into a fixed categorical`, widened, false)
    }
    // Every value already tried — just widen (re-cluster with the new axis so its tried values become basins).
    return stepGlobal(widened, runs, manifest, criterion, mk, undefined)
  }

  // Every lever unfrozen, every peak resolved, and the space sampled to the density target — genuinely covered.
  const finest = (state.refineDepth ?? 0) >= EXPLORATION_MAX_REFINE_DEPTH
  const reason = finest
    ? 'search space fully covered at the finest resolution'
    : 'search space covered — Explore more to sample finer'
  return converge({ ...state, basins }, runs, criterion, mk, reason)
}

/**
 * A deterministic low-discrepancy (Halton) coordinate in [0,1) — the space-filling sequence the coverage rung
 * draws from. Deterministic so a run is reproducible and unit-testable (no RNG).
 */
function halton(index: number, base: number): number {
  let result = 0
  let f = 1 / base
  let i = index
  while (i > 0) {
    result += f * (i % base)
    i = Math.floor(i / base)
    f /= base
  }
  return result
}

const HALTON_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

/**
 * Space-filling COVERAGE — PER categorical region. Each discrete region (each `model_name`/algo value combo) is
 * sampled toward `EXPLORATION_COVERAGE_PER_LEVER × #active-numeric × (1 + refineDepth)` distinct setups with a
 * deterministic Halton maximin fill, holding that region's categoricals fixed and the other levers at the best
 * config. This is what stops the search abandoning an under-sampled value (dqn getting 5 runs while ppo gets
 * hundreds): every region's own space is explored to the SAME density. The round's batch is round-robined across
 * under-covered regions (so each progresses) and capped at {@link EXPLORATION_BATCH_MAX}. Returns [] once every
 * region meets its target (or is saturated) — the convergence gate. Exported for direct unit testing.
 */
export function coverageGridRecs(
  state: ExplorationState,
  runs: AnalysisRun[],
  manifest: TrainerManifest,
  criterion: AnalysisCriterion,
): ExperimentRecommendation[] {
  // Only levers with a NON-DEGENERATE range can be space-filled — a [x,x] lever yields one value forever, so
  // counting it toward coverage would make the target unreachable and the search never converge (mirrors the
  // `span > 0` guard in localRefineRecs).
  const numericActive = state.activeLevers.filter((l) => {
    const range = manifest.levers[l]?.range
    return isNumericManifestLever(manifest, l) && !!range && range[1] > range[0]
  })
  if (!numericActive.length) return []
  const setups = aggregateToSetupRuns(runs, criterion)
  const regionLevers = regionLeversOf(setups, state.activeLevers, manifest)
  const perRegionTarget = EXPLORATION_COVERAGE_PER_LEVER * numericActive.length * (1 + (state.refineDepth ?? 0))
  const ranges = numericActive.map((l) => manifest.levers[l].range as [number, number])
  // Coverage samples probe BREADTH at a single seed — seeds are added later when a region is climbed — so the
  // champion's seed is NOT carried onto them (it would pin every sample to one noise draw).
  const best = without(bestRunConfig(runs, criterion), 'seed')

  // Group setups into their discrete regions (a pure-numeric problem is one region under key '').
  const regions = new Map<string, { region: Record<string, unknown>; setups: AnalysisRun[] }>()
  for (const s of setups) {
    const rk = regionKey(s.config, regionLevers)
    let entry = regions.get(rk)
    if (!entry) {
      const region: Record<string, unknown> = {}
      for (const l of regionLevers) region[l] = s.config[l]
      entry = { region, setups: [] }
      regions.set(rk, entry)
    }
    entry.setups.push(s)
  }

  // A queue of fresh maximin candidates per UNDER-covered region.
  const queues: Array<Array<{ config: Record<string, unknown> }>> = []
  for (const { region, setups: rSetups } of regions.values()) {
    if (rSetups.length >= perRegionTarget) continue
    const need = Math.min(perRegionTarget - rSetups.length, EXPLORATION_BATCH_MAX)
    const fresh = maximinCoverageConfigs(numericActive, ranges, rSetups, { ...best, ...region }, need)
    if (fresh.length) queues.push(fresh)
  }
  const regionCount = queues.length
  if (!regionCount) return []

  // Round-robin across regions so EVERY under-covered value makes progress, up to the batch cap. When there are
  // MORE under-covered regions than the cap, one round can't serve them all — so ROTATE the start each round (by
  // the monotonically-growing setup count) so the served window slides and no region is permanently starved.
  const configs: Array<{ config: Record<string, unknown> }> = []
  let idx = setups.length % regionCount
  while (configs.length < EXPLORATION_BATCH_MAX && queues.some((q) => q.length)) {
    const q = queues[idx % regionCount]
    if (q.length) configs.push(q.shift() as { config: Record<string, unknown> })
    idx++
  }
  if (!configs.length) return []
  return [
    {
      kind: 'missing-cell',
      reason: `cover ${regionCount} region(s): ${configs.length} space-filling sample(s)`,
      runCount: configs.length,
      spec: { configs },
      priority: 75,
    },
  ]
}

/**
 * Halton maximin fill for ONE region's numeric space: up to `need` fresh configs (base = the region's fixed
 * levers), each as far as possible (normalised L∞) from every tried setup and from the others already chosen,
 * never duplicating a quantized signature. Returns fewer (or none) when the representable space is saturated.
 */
function maximinCoverageConfigs(
  numericActive: string[],
  ranges: Array<[number, number]>,
  rSetups: AnalysisRun[],
  base: Record<string, unknown>,
  need: number,
): Array<{ config: Record<string, unknown> }> {
  const sigOf = (config: Record<string, unknown>): string => numericActive.map((l) => Number(config[l]).toFixed(6)).join('|')
  const normOf = (config: Record<string, unknown>): number[] =>
    numericActive.map((l, j) => (Number(config[l]) - ranges[j][0]) / (ranges[j][1] - ranges[j][0] || 1))
  const tried = rSetups.map((s) => normOf(s.config))
  const triedSigs = new Set(rSetups.map((s) => sigOf(s.config)))
  const minDistToTried = (pt: number[]): number => {
    let best = Infinity
    for (const t of tried) {
      let d = 0
      for (let j = 0; j < pt.length; j++) d = Math.max(d, Math.abs(pt[j] - t[j]))
      if (d < best) best = d
    }
    return tried.length ? best : Infinity
  }
  const pool: Array<{ norm: number[]; config: Record<string, unknown> }> = []
  for (let i = 1; i <= need * 12 + 48; i++) {
    const norm = numericActive.map((_, j) => halton(i, HALTON_BASES[j % HALTON_BASES.length]))
    const config: Record<string, unknown> = { ...base }
    for (let j = 0; j < numericActive.length; j++) {
      const [lo, hi] = ranges[j]
      config[numericActive[j]] = Number((lo + norm[j] * (hi - lo)).toFixed(6))
    }
    pool.push({ norm, config })
  }
  const chosenNorms: number[][] = []
  const chosenSigs = new Set<string>()
  const out: Array<{ config: Record<string, unknown> }> = []
  while (out.length < need) {
    let pick = -1
    let pickDist = -1
    for (let k = 0; k < pool.length; k++) {
      const c = pool[k]
      const sig = sigOf(c.config)
      if (triedSigs.has(sig) || chosenSigs.has(sig)) continue // already run / already chosen — never dup
      let d = minDistToTried(c.norm)
      for (const cn of chosenNorms) {
        let dd = 0
        for (let j = 0; j < cn.length; j++) dd = Math.max(dd, Math.abs(c.norm[j] - cn[j]))
        d = Math.min(d, dd)
      }
      if (d > pickDist) {
        pickDist = d
        pick = k
      }
    }
    if (pick < 0) break // representable space saturated
    chosenNorms.push(pool[pick].norm)
    chosenSigs.add(sigOf(pool[pick].config))
    out.push({ config: pool[pick].config })
    pool.splice(pick, 1)
  }
  return out
}

/**
 * A space-filling sweep of a freshly-unfrozen numeric lever across its full range (excluding already-tried
 * values), holding the other levers at the best-known config. Guarantees NEW configs so the widening probes
 * the lever rather than re-proposing runs already in the archive. Null when the range is already covered.
 */
function unfreezeSweepRec(
  lever: string,
  frozenLevers: Record<string, unknown>,
  runs: AnalysisRun[],
  manifest: TrainerManifest,
  criterion: AnalysisCriterion,
): ExperimentRecommendation | undefined {
  const range = manifest.levers[lever]?.range
  if (!range) return undefined
  const [lo, hi] = range
  const n = 5
  const tried = new Set(runs.map((r) => Number(r.config[lever])).filter(isFinite).map((v) => v.toFixed(6)))
  const fresh: number[] = []
  for (let i = 0; i < n; i++) {
    const v = Number((lo + ((i + 0.5) / n) * (hi - lo)).toFixed(6))
    if (!tried.has(v.toFixed(6))) fresh.push(v)
  }
  if (!fresh.length) return undefined
  const best = bestRunConfig(runs, criterion)
  const fixed = { ...without(best, lever, 'seed'), ...frozenLevers }
  return {
    kind: 'acquisition',
    reason: `unfreeze ${lever}: sweep ${fresh.length} values across its range`,
    runCount: fresh.length * XAI_MIN_SEEDS,
    spec: { fixed, sweep: { [lever]: fresh }, seeds: seedRange(XAI_MIN_SEEDS) },
    priority: 70,
  }
}

/**
 * The discrete value set of a categorical basin-axis lever: a `choice` lever's `choices`, or `[false, true]`
 * for a `boolean` (which carries no `choices` array). Undefined for anything else — so numerics + degenerate
 * single-choice levers are never treated as a categorical axis to open.
 */
function categoricalChoices(manifest: TrainerManifest, lever: string): unknown[] | undefined {
  const spec = manifest.levers[lever]
  if (!spec) return undefined
  if (Array.isArray(spec.choices) && spec.choices.length > 1) return spec.choices
  if (spec.type === 'boolean') return [false, true]
  return undefined
}

/**
 * Try the UNTRIED values of a freshly-promoted categorical basin axis (holding the other levers at the best
 * config), so opening the axis actually produces new regions to cluster. Handles `boolean` levers (values
 * `false`/`true`) as well as `choice`. Null when every value is already in the archive (the caller then just
 * widens + re-clusters over the existing runs).
 */
function unfreezeCategoricalRec(
  lever: string,
  frozenLevers: Record<string, unknown>,
  runs: AnalysisRun[],
  manifest: TrainerManifest,
  criterion: AnalysisCriterion,
): ExperimentRecommendation | undefined {
  const choices = categoricalChoices(manifest, lever)
  if (!choices || choices.length < 2) return undefined
  const tried = new Set(runs.map((r) => JSON.stringify(r.config[lever])))
  const fresh = choices.filter((c) => !tried.has(JSON.stringify(c)))
  if (!fresh.length) return undefined
  const best = bestRunConfig(runs, criterion)
  const fixed = { ...without(best, lever, 'seed'), ...frozenLevers }
  return {
    kind: 'acquisition',
    reason: `unfreeze ${lever}: try ${fresh.length} untried value(s)`,
    runCount: fresh.length * XAI_MIN_SEEDS,
    spec: { fixed, sweep: { [lever]: fresh }, seeds: seedRange(XAI_MIN_SEEDS) },
    priority: 65,
  }
}

/** The config of the single best run by the criterion (the current champion to hold while probing a lever). */
function bestRunConfig(runs: AnalysisRun[], criterion: AnalysisCriterion): Record<string, unknown> {
  let best: AnalysisRun | undefined
  for (const r of runs) {
    const v = criterionValueOf(r, criterion)
    if (v == null) continue
    const bv = best ? criterionValueOf(best, criterion) : undefined
    if (best === undefined || bv == null || isBetter(v, bv, criterion.direction)) best = r
  }
  return best ? { ...best.config } : {}
}

/**
 * When no region cleared the basin margin (a hard problem, or too few runs yet) the run archive still has a
 * best — declare it as a (weak) basin so the search never reports "no maximum found" while good runs exist.
 */
function syntheticBestBasin(
  runs: AnalysisRun[],
  criterion: AnalysisCriterion,
  activeLevers: string[],
  manifest?: TrainerManifest,
): Basin | undefined {
  const setups = aggregateToSetupRuns(runs, criterion)
  if (!setups.length) return undefined
  const dir = criterion.direction
  let best = setups[0]
  for (const s of setups) if (isBetter(criterionValueOf(s, criterion)!, criterionValueOf(best, criterion)!, dir)) best = s
  const regionLevers = regionLeversOf(setups, activeLevers, manifest)
  const region: Record<string, unknown> = {}
  for (const l of regionLevers) region[l] = best.config[l]
  const memberRunKeys = runs.filter((r) => matchesConfig(r.config, best.config, regionLevers)).map((r) => r.key)
  return {
    id: regionKey(best.config, regionLevers),
    region,
    centerConfig: best.config,
    peakObjective: criterionValueOf(best, criterion)!,
    peakCI: best.ci,
    peakSeeds: best.seeds ?? 1,
    plateaued: true,
    memberRunKeys,
  }
}

function isPlateaued(
  basin: Basin,
  runs: AnalysisRun[],
  manifest: TrainerManifest,
  state: ExplorationState,
  criterion: AnalysisCriterion,
  target?: number,
): boolean {
  const enoughSeeds = basin.peakSeeds >= XAI_MIN_SEEDS
  if (!enoughSeeds) return false
  if (target != null) {
    const gap = gainOver(target, basin.peakObjective, criterion.direction)
    if (gap <= (state.noiseFloor ?? 0)) return true // within noise of the known ceiling — can't do better
  }
  return localRefineRecs(basin, runs, manifest, state).length === 0
}

function converge(
  state: ExplorationState,
  runs: AnalysisRun[],
  criterion: AnalysisCriterion,
  mk: Mk,
  reason: string,
  manifest?: TrainerManifest,
): ExplorationStep {
  let basins = state.basins.length
    ? state.basins
    : clusterBasins(runs, criterion, state.activeLevers, state.noiseFloor ?? 0, baselineOf(runs), manifest)
  if (!basins.length) {
    // No region cleared the basin margin — still declare the best run so the maxima view is never empty.
    const synth = syntheticBestBasin(runs, criterion, state.activeLevers, manifest)
    if (synth) basins = [synth]
  }
  const declared = bestBasin(basins, criterion)
  return mk(
    'converged',
    [],
    `converged: ${reason}`,
    { ...state, stage: 'converged', done: true, basins, declaredBasinId: declared?.id },
    true,
  )
}

/**
 * The split-consistency convergence GATE (A5). When a step would crown an incumbent (`converged`) but a
 * splitAxis is declared, budget remains, and the incumbent is not-replicated / single-split-luck WITH unrun
 * splits, replace the convergence with `missing-cell` split-fill recs — replicate the incumbent across its
 * missing splits before trusting it. The search stays at its current stage (done:false) and re-checks after
 * the fills run; loop-safe because each pass shrinks the missing splits (a budget-exhausted convergence
 * bypasses the gate — nothing more can be run). A no-op when no splitAxis is declared. Exported for testing.
 */
export function gateConvergenceOnSplits(
  step: ExplorationStep,
  state: ExplorationState,
  runs: AnalysisRun[],
  manifest: TrainerManifest | undefined,
  criterion: AnalysisCriterion,
  budgetLeft: boolean,
  mk: Mk,
): ExplorationStep {
  if (!step.done || step.stage !== 'converged' || !budgetLeft) return step
  const splitLevers = splitLeversOf(manifest)
  if (!splitLevers.length) return step
  const holdout = incumbentSplitHoldout(runs, splitLevers, criterion, { baseline: baselineOf(runs) })
  // 1) Split-consistency gate (existing): the incumbent isn't robust across the split axis AND there are unrun
  //    splits + budget to try them → replicate it across them before crowning (fixes single-window luck).
  if (convergenceGatedBySplits(holdout) && holdout.missingSplitConfigs.length && holdout.incumbentConfig) {
    const base = without(holdout.incumbentConfig, 'seed')
    const batch: ExperimentRecommendation[] = holdout.missingSplitConfigs.map((sc) => ({
      kind: 'missing-cell',
      reason: `split-consistency gate — replicate the incumbent across ${Object.entries(sc)
        .map(([k, v]) => `${k}=${v}`)
        .join(', ')} before crowning it`,
      runCount: XAI_MIN_SEEDS,
      spec: { fixed: { ...base, ...sc }, seeds: seedRange(XAI_MIN_SEEDS) },
      priority: 80,
    }))
    return mk(
      state.stage,
      batch,
      `gated: incumbent not yet robust across ${splitLevers.join(', ')} — replicating it across ${batch.length} unrun split(s)`,
      state,
      false,
    )
  }
  // 2) Champion "declare steady" gate (A4.3): the incumbent is split-robust (or has no unrun splits) but the
  //    honest composite verdict isn't STEADY, and some evaluated split is thinly seeded → top it up to
  //    XAI_MIN_SEEDS before accepting the verdict. Bounded (a genuinely-failing, fully-seeded champion has
  //    nothing to fill → converges as not-steady, which the human then pivots on), and budget-capped above —
  //    so this can only make convergence STRICTER, never stall.
  const championFill = championSeedFill(holdout, runs, splitLevers, criterion, manifest)
  if (championFill.length) {
    return mk(
      state.stage,
      championFill,
      `gated: incumbent not yet a STEADY champion — seeding ${championFill.length} thin split(s) to ${XAI_MIN_SEEDS} before the verdict`,
      state,
      false,
    )
  }
  return step
}

/**
 * The A4.3 champion seed-fill: when the manifest declares champion gates and the composite verdict is NOT
 * steady, top up any EVALUATED split the incumbent has fewer than {@link XAI_MIN_SEEDS} seeds on — so a
 * not-steady verdict is well-estimated before it's accepted (a thin split can flip a borderline median /
 * seed-stability). Returns [] when there are no champion gates, the verdict is steady or not-applicable, or
 * every evaluated split is already well-seeded (nothing productive to do → let it converge honestly).
 */
function championSeedFill(
  holdout: SplitHoldout,
  runs: AnalysisRun[],
  splitLevers: string[],
  criterion: AnalysisCriterion,
  manifest: TrainerManifest | undefined,
): ExperimentRecommendation[] {
  const d = manifest?.diagnostics
  if (!d || !((d.championGates && d.championGates.length) || d.dsr || d.stability)) return []
  if (!holdout.incumbentConfig) return []
  const verdict = assembleChampionVerdict(d, { runs, splitLevers, criterion })
  if (verdict.steady !== false) return [] // steady, or not applicable → nothing to gate on
  const base = without(holdout.incumbentConfig, 'seed')
  const batch: ExperimentRecommendation[] = []
  for (const es of holdout.evaluatedSplits) {
    if (es.seeds >= XAI_MIN_SEEDS) continue
    batch.push({
      kind: 'missing-cell',
      reason: `champion gate — seed the incumbent to ${XAI_MIN_SEEDS} on ${Object.entries(es.splitConfig)
        .map(([k, v]) => `${k}=${v}`)
        .join(', ')} for a sound steady verdict`,
      runCount: XAI_MIN_SEEDS - es.seeds,
      spec: { fixed: { ...base, ...es.splitConfig }, seeds: seedRange(XAI_MIN_SEEDS) },
      priority: 78,
    })
  }
  return batch
}
