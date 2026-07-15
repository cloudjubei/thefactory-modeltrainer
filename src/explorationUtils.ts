import {
  EXPLORATION_ACTIVE_IMPORTANCE_FLOOR,
  EXPLORATION_BASIN_MIN_SPAN_FRACTION,
  EXPLORATION_BASIN_NOISE_MARGIN,
  EXPLORATION_COVERAGE_PER_LEVER,
  EXPLORATION_DRY_ROUNDS,
  EXPLORATION_MAX_ACTIVE_LEVERS,
  EXPLORATION_MAX_REFINE_DEPTH,
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
  if (state.budget.maxRuns !== undefined && spentRuns >= state.budget.maxRuns) {
    return converge(state, runs, criterion, mk, 'budget exhausted')
  }
  // An EMPTY run archive (e.g. the user deleted every run) makes any advanced/converged state stale — there is
  // nothing to have converged on, so restart from calibrate rather than re-declaring a convergence (or trying
  // to refine/cover) over runs that no longer exist. Resets the derived fields, keeps budget/steer.
  if (spentRuns === 0) {
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
  opts?: { targetObjective?: number; exhausted?: boolean },
): ExplorationStep {
  const basins = clusterBasins(runs, criterion, state.activeLevers, state.noiseFloor ?? 0, baselineOf(runs))
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
  opts?: { targetObjective?: number; exhausted?: boolean },
): ExplorationStep {
  const clustered = clusterBasins(runs, criterion, state.activeLevers, state.noiseFloor ?? 0, baselineOf(runs))
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
  const batch = pending.flatMap((b) => localRefineRecs(b, runs, manifest, state)).slice(0, 8)
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
 * Space-filling COVERAGE of the active numeric space: keeps proposing Halton-spread configs (other levers held
 * at the best-known config) until the archive holds `EXPLORATION_COVERAGE_PER_LEVER × #active-numeric ×
 * (1 + refineDepth)` distinct setups. Candidates are chosen maximin — farthest (normalised L∞) from every
 * tried setup — so each round meaningfully fills the emptiest regions. Returns [] once the target is met (the
 * convergence gate) or nothing fresh remains. Exported for direct unit testing.
 */
export function coverageGridRecs(
  state: ExplorationState,
  runs: AnalysisRun[],
  manifest: TrainerManifest,
  criterion: AnalysisCriterion,
): ExperimentRecommendation[] {
  const numericActive = state.activeLevers.filter(
    (l) => isNumericManifestLever(manifest, l) && !!manifest.levers[l]?.range,
  )
  if (!numericActive.length) return []
  const setups = aggregateToSetupRuns(runs, criterion)
  const target = EXPLORATION_COVERAGE_PER_LEVER * numericActive.length * (1 + (state.refineDepth ?? 0))
  if (setups.length >= target) return []

  const ranges = numericActive.map((l) => manifest.levers[l].range as [number, number])
  const tried = setups.map((s) => numericActive.map((l, j) => (Number(s.config[l]) - ranges[j][0]) / (ranges[j][1] - ranges[j][0] || 1)))
  const minDistToTried = (pt: number[]): number => {
    let best = Infinity
    for (const t of tried) {
      let d = 0
      for (let j = 0; j < pt.length; j++) d = Math.max(d, Math.abs(pt[j] - t[j]))
      if (d < best) best = d
    }
    return tried.length ? best : Infinity
  }

  const best = bestRunConfig(runs, criterion)
  const need = Math.min(8, target - setups.length)
  // Draw a pool of Halton candidates, then greedily take the ones farthest from BOTH the tried set and each
  // other — a maximin fill that avoids clustering with existing runs or within the batch.
  const pool: Array<{ norm: number[]; config: Record<string, unknown> }> = []
  for (let i = 1; i <= need * 12 + 48; i++) {
    const norm = numericActive.map((_, j) => halton(i, HALTON_BASES[j % HALTON_BASES.length]))
    const config: Record<string, unknown> = { ...best }
    for (let j = 0; j < numericActive.length; j++) {
      const [lo, hi] = ranges[j]
      config[numericActive[j]] = Number((lo + norm[j] * (hi - lo)).toFixed(6))
    }
    pool.push({ norm, config })
  }
  const chosenNorms: number[][] = []
  const configs: Array<{ config: Record<string, unknown> }> = []
  while (configs.length < need) {
    let pick = -1
    let pickDist = -1
    for (let k = 0; k < pool.length; k++) {
      const c = pool[k]
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
    if (pick < 0 || pickDist <= 0) break // nothing fresh left to add
    chosenNorms.push(pool[pick].norm)
    configs.push({ config: pool[pick].config })
    pool.splice(pick, 1)
  }
  if (!configs.length) return []
  return [
    {
      kind: 'missing-cell',
      reason: `cover the space: ${configs.length} space-filling sample(s) across ${numericActive.length} lever(s)`,
      runCount: configs.length,
      spec: { configs },
      priority: 75,
    },
  ]
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
): Basin | undefined {
  const setups = aggregateToSetupRuns(runs, criterion)
  if (!setups.length) return undefined
  const dir = criterion.direction
  let best = setups[0]
  for (const s of setups) if (isBetter(criterionValueOf(s, criterion)!, criterionValueOf(best, criterion)!, dir)) best = s
  const regionLevers = regionLeversOf(setups, activeLevers)
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
): ExplorationStep {
  let basins = state.basins.length
    ? state.basins
    : clusterBasins(runs, criterion, state.activeLevers, state.noiseFloor ?? 0, baselineOf(runs))
  if (!basins.length) {
    // No region cleared the basin margin — still declare the best run so the maxima view is never empty.
    const synth = syntheticBestBasin(runs, criterion, state.activeLevers)
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
