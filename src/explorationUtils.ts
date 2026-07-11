import {
  EXPLORATION_ACTIVE_IMPORTANCE_FLOOR,
  EXPLORATION_BASIN_MIN_SPAN_FRACTION,
  EXPLORATION_BASIN_NOISE_MARGIN,
  EXPLORATION_DRY_ROUNDS,
  EXPLORATION_MAX_ACTIVE_LEVERS,
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
  TrainerManifest,
} from './modelTrainerTypes.js'
import { aggregateToSetupRuns, criterionValueOf, leverImportances, recommendExperiments } from './xaiUtils.js'

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

/** A region axis is a categorical active lever; numeric active levers climb WITHIN one. */
function regionLeversOf(setups: AnalysisRun[], activeLevers: string[]): string[] {
  return activeLevers.filter((lever) => {
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
): Basin[] {
  const setups = aggregateToSetupRuns(runs, criterion)
  if (!setups.length) return []
  const dir = criterion.direction
  const regionLevers = regionLeversOf(setups, activeLevers)

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
  return basins.sort((a, b) => (isBetter(a.peakObjective, b.peakObjective, dir) ? -1 : 1))
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

/** Coordinate-ascent refinement + seed-stabilization inside one basin. */
function localRefineRecs(
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

  // 2. finer coordinate-ascent sweep of each numeric lever around the center
  for (const lever of numericActive) {
    const spec = manifest.levers[lever]
    const range = spec.range
    if (!range) continue
    const c = Number(center[lever])
    const step = (range[1] - range[0]) / 8
    const cands = [c - step, c - step / 2, c + step / 2, c + step].filter((v) => v >= range[0] && v <= range[1])
    const tried = new Set(regionRuns.map((r) => Number(r.config[lever]).toFixed(4)))
    const fresh = uniq(cands.filter((v) => !tried.has(v.toFixed(4))).map((v) => Number(v.toFixed(4))))
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
  opts?: { targetObjective?: number },
): ExplorationStep {
  const criterion = criterionOf(state)
  const spentRuns = runs.length
  const best = bestObjectiveOf(runs, criterion)

  const withMeta = (s: ExplorationState): ExplorationState => ({
    ...s,
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
  if (state.done || state.stage === 'converged') {
    return mk('converged', [], 'converged', { ...state, stage: 'converged', done: true }, true)
  }
  if (state.budget.maxRuns !== undefined && spentRuns >= state.budget.maxRuns) {
    return converge(state, runs, criterion, mk, 'budget exhausted')
  }

  switch (state.stage) {
    case 'calibrate':
      return stepCalibrate(state, runs, manifest, criterion, mk)
    case 'screen':
      return stepScreen(state, runs, manifest, criterion, mk)
    case 'global':
      return stepGlobal(state, runs, manifest, criterion, mk, opts)
    case 'local':
      return stepLocal(state, runs, manifest, criterion, mk, opts)
    default:
      return converge(state, runs, criterion, mk, 'done')
  }
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

  if (seedsPresent < XAI_MIN_SEEDS) {
    const rec: ExperimentRecommendation = {
      kind: 'thin-seeds',
      reason: `calibrate seed-noise on the default config (×${XAI_MIN_SEEDS} seeds)`,
      runCount: XAI_MIN_SEEDS,
      spec: { fixed: without(defaults, 'seed'), seeds: seedRange(XAI_MIN_SEEDS) },
      priority: 100,
    }
    return mk('calibrate', [rec], 'measure the objective noise floor', state, false)
  }

  const vals = calibRuns.map((r) => criterionValueOf(r, criterion)).filter((v): v is number => v != null)
  const noiseFloor = std(vals)
  return stepScreen({ ...state, stage: 'screen', noiseFloor }, runs, manifest, criterion, mk)
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

  // discrete levers are basin AXES — never frozen. numeric levers are kept only if they move the objective.
  let activeNumeric = numeric.filter((l) => impOf(l) >= EXPLORATION_ACTIVE_IMPORTANCE_FLOOR)
  if (!activeNumeric.length && numeric.length) {
    activeNumeric = [numeric.slice().sort((a, b) => impOf(b) - impOf(a))[0]] // keep at least one climb dimension
  }
  activeNumeric = activeNumeric.sort((a, b) => impOf(b) - impOf(a)).slice(0, Math.max(0, EXPLORATION_MAX_ACTIVE_LEVERS - discrete.length))

  const steerActive = (state.steer?.pinActive ?? []).filter((l) => searchable.includes(l))
  const active = uniq([...discrete, ...activeNumeric, ...steerActive])

  // only numeric levers are ever frozen (categoricals are always-active basin axes), so freeze at the
  // best observed value coerced back to a number.
  const frozenLevers: Record<string, unknown> = {}
  for (const l of searchable) {
    if (active.includes(l)) continue
    const best = imps.find((i) => i.lever === l)?.bestValue
    frozenLevers[l] = best !== undefined ? Number(best) : manifest.levers[l].default
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
  opts?: { targetObjective?: number },
): ExplorationStep {
  const basins = clusterBasins(runs, criterion, state.activeLevers, state.noiseFloor ?? 0, baselineOf(runs))
  const known = new Set(state.basins.map((b) => b.id))
  const foundNew = basins.some((b) => !known.has(b.id))
  const dryRounds = foundNew ? 0 : state.dryRounds + 1
  const state2 = { ...state, basins, dryRounds }

  if (dryRounds >= EXPLORATION_DRY_ROUNDS) {
    return stepLocal({ ...state2, stage: 'local' }, runs, manifest, criterion, mk, opts)
  }

  const coverage = discreteCoverageRecs(runs, manifest, state.activeLevers, state.frozenLevers)
  const acquisition = coverage.length ? [] : globalAcquisitionRecs(state, runs, manifest, criterion)
  const batch = [...coverage, ...acquisition].slice(0, 8)
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
  opts?: { targetObjective?: number },
): ExplorationStep {
  const clustered = clusterBasins(runs, criterion, state.activeLevers, state.noiseFloor ?? 0, baselineOf(runs))
  const target = opts?.targetObjective
  const withPlateau = clustered.map((b) => ({
    ...b,
    plateaued: isPlateaued(b, runs, manifest, state, criterion, target),
  }))
  const pending = withPlateau.filter((b) => !b.plateaued)
  const state2 = { ...state, basins: withPlateau }

  if (!pending.length) return converge(state2, runs, criterion, mk, 'all basins plateaued')

  const batch = pending.flatMap((b) => localRefineRecs(b, runs, manifest, state)).slice(0, 8)
  if (!batch.length) return converge(state2, runs, criterion, mk, 'no refinements remain')
  return mk('local', batch, `climb ${pending.length} basin(s)`, state2, false)
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
): ExplorationStep {
  const basins = state.basins.length
    ? state.basins
    : clusterBasins(runs, criterion, state.activeLevers, state.noiseFloor ?? 0, baselineOf(runs))
  const declared = bestBasin(basins, criterion)
  return mk(
    'converged',
    [],
    `converged: ${reason}`,
    { ...state, stage: 'converged', done: true, basins, declaredBasinId: declared?.id },
    true,
  )
}
