// Server-side diagnostics primitives — the split-consistency HOLDOUT that gates the exploration
// autopilot's convergence, mirroring the viewer's `split-consistency` check (viewer/diagnostics.js).
// Deterministic (no LLM): the incumbent (best SETUP, folding seeds + the split levers) must be evaluated
// on enough distinct split values AND beat the baseline on each, or a `converged` declaration is
// single-split luck and must be blocked. Pure — the controller (explorationUtils) and a future
// `diagnoseSearch` chat tool both consume it.

import type {
  AnalysisRun,
  AnalysisCriterion,
  MetricAlignment,
  TrainerDiagnostics,
  TrainerGate,
  ChampionGate,
} from './modelTrainerTypes'
import { applyGateOp } from './modelTrainerUtils.js'
import { medianOf, stdOf, aggregateRunValues } from './xaiUtils.js'
import { deflatedSharpeFromStats } from './deflatedSharpe.js'

export type SplitVerdict = 'unverifiable' | 'not-replicated' | 'single-split-luck' | 'robust'

export interface SplitHoldout {
  verdict: SplitVerdict
  /** Distinct split values the incumbent has runs on. */
  evaluated: number
  /** Of those, how many the incumbent beats the baseline on. */
  held: number
  /** Every distinct split value present in the run set. */
  splitValues: string[]
  /** Split values the incumbent has NOT been run on (the split-fill target). */
  missingSplits: string[]
  /** The missing splits as `{splitLever: value}` configs — what the gate replicates the incumbent onto. */
  missingSplitConfigs: Record<string, unknown>[]
  /** The EVALUATED splits with the incumbent's `{splitLever: value}` config + its seed count there — the
   * champion-verdict seed-fill target (top up a thinly-seeded split before accepting a not-steady verdict). */
  evaluatedSplits: Array<{ splitConfig: Record<string, unknown>; seeds: number }>

  /** The incumbent setup's config (seed + split levers included as-run), or null when there are no runs. */
  incumbentConfig: Record<string, unknown> | null
}

function valueOf(run: AnalysisRun, criterion: AnalysisCriterion): number | undefined {
  if (criterion.key === 'objective') return typeof run.objective === 'number' ? run.objective : undefined
  if (criterion.key === 'durationMs') return typeof run.durationMs === 'number' ? run.durationMs : undefined
  const v = run.metrics?.[criterion.key]
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length
}

function keyOf(config: Record<string, unknown>, keys: string[]): string {
  return keys.map((k) => `${k}=${String(config[k])}`).join('·')
}

function setupKey(config: Record<string, unknown>, exclude: Set<string>): string {
  return keyOf(
    config,
    Object.keys(config)
      .filter((k) => !exclude.has(k))
      .sort(),
  )
}

/** Read a run's value for a raw metric key (`objective` ⇒ summary.objective, else summary.metrics[key]); undefined when absent/non-finite. */
function metricOf(run: AnalysisRun, key: string): number | undefined {
  if (key === 'objective') return typeof run.objective === 'number' ? run.objective : undefined
  const v = run.metrics?.[key]
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined
}

/**
 * The completed runs (with a finite criterion value) and the INCUMBENT setup's runs — the best setup by
 * mean-of-criterion across its seeds+splits, chosen from gate-ACCEPTED runs (falling back to all completed
 * when none pass). Shared by {@link incumbentSplitHoldout} (the split verdict) and {@link assembleChampionVerdict}
 * (the champion gates) so both reason about the SAME incumbent setup.
 */
function selectIncumbent(
  runs: AnalysisRun[],
  splitLevers: string[],
  criterion: AnalysisCriterion,
): { completed: AnalysisRun[]; incRuns: AnalysisRun[] } {
  const completed = runs.filter(
    (r) => (r.status ?? 'completed') === 'completed' && valueOf(r, criterion) !== undefined,
  )
  if (!completed.length) return { completed, incRuns: [] }
  const acceptedRuns = completed.filter((r) => r.accepted !== false)
  const incumbentPool = acceptedRuns.length ? acceptedRuns : completed
  const exclude = new Set<string>(['seed', ...splitLevers])
  const bySetup = new Map<string, AnalysisRun[]>()
  for (const r of incumbentPool) {
    const k = setupKey(r.config, exclude)
    const arr = bySetup.get(k)
    if (arr) arr.push(r)
    else bySetup.set(k, [r])
  }
  let incRuns: AnalysisRun[] = []
  let incVal = criterion.direction === 'max' ? -Infinity : Infinity
  for (const rs of bySetup.values()) {
    const v = mean(rs.map((r) => valueOf(r, criterion)!))
    if (criterion.direction === 'max' ? v > incVal : v < incVal) {
      incVal = v
      incRuns = rs
    }
  }
  return { completed, incRuns }
}

/**
 * The split-consistency holdout for the current run set. The incumbent is the best SETUP by
 * `criterion.direction`, folding over seed AND the `splitLevers` (so it's a config, not one lucky split).
 * It "holds" on a split when its folded value there beats `baseline` (default 0, direction-aware). The
 * verdict gates convergence: `not-replicated` (< minSplits splits evaluated → fill more), `single-split-luck`
 * (fails some evaluated split → not a robust edge), `robust`, or `unverifiable` (no split axis).
 */
export function incumbentSplitHoldout(
  runs: AnalysisRun[],
  splitLevers: string[],
  criterion: AnalysisCriterion,
  opts?: { minSplits?: number; baseline?: number },
): SplitHoldout {
  const minSplits = opts?.minSplits ?? 2
  const baseline = opts?.baseline ?? 0
  const base: SplitHoldout = {
    verdict: 'unverifiable',
    evaluated: 0,
    held: 0,
    splitValues: [],
    missingSplits: [],
    missingSplitConfigs: [],
    evaluatedSplits: [],
    incumbentConfig: null,
  }
  if (!splitLevers.length) return base

  // Incumbent = the best SETUP by mean-of-criterion among gate-ACCEPTED runs (see selectIncumbent). The split
  // universe below stays over ALL completed runs (a rejected setup still counts as a split we've covered).
  const { completed, incRuns } = selectIncumbent(runs, splitLevers, criterion)
  if (!completed.length) return { ...base, verdict: 'not-replicated' }

  // Map each split-key string back to its {lever: value} config (from any run carrying it) so the gate can
  // build split-fill recs without parsing the key string.
  const splitConfigByKey = new Map<string, Record<string, unknown>>()
  for (const r of completed) {
    const k = keyOf(r.config, splitLevers)
    if (!splitConfigByKey.has(k)) {
      splitConfigByKey.set(k, Object.fromEntries(splitLevers.map((l) => [l, r.config[l]])))
    }
  }

  const splitValues = [...splitConfigByKey.keys()].sort()
  let evaluated = 0
  let held = 0
  const evaluatedSet = new Set<string>()
  const evaluatedSplits: SplitHoldout['evaluatedSplits'] = []
  for (const sv of splitValues) {
    const rs = incRuns.filter((r) => keyOf(r.config, splitLevers) === sv)
    if (!rs.length) continue
    evaluated++
    evaluatedSet.add(sv)
    evaluatedSplits.push({ splitConfig: splitConfigByKey.get(sv)!, seeds: rs.length })
    const v = mean(rs.map((r) => valueOf(r, criterion)!))
    if (criterion.direction === 'max' ? v > baseline : v < baseline) held++
  }
  const missingSplits = splitValues.filter((sv) => !evaluatedSet.has(sv))

  let verdict: SplitVerdict
  if (evaluated < minSplits) verdict = 'not-replicated'
  else if (held < evaluated) verdict = 'single-split-luck'
  else verdict = 'robust'

  return {
    verdict,
    evaluated,
    held,
    splitValues,
    missingSplits,
    missingSplitConfigs: missingSplits.map((sv) => splitConfigByKey.get(sv)!),
    evaluatedSplits,
    incumbentConfig: incRuns.length ? incRuns[0].config : null,
  }
}

/**
 * The convergence GATE: true when the split-consistency verdict must BLOCK a `converged` declaration —
 * the incumbent is under-replicated across the split axis, or wins on some splits and fails others.
 */
export function convergenceGatedBySplits(holdout: SplitHoldout): boolean {
  return holdout.verdict === 'not-replicated' || holdout.verdict === 'single-split-luck'
}

/**
 * Evaluate ONE champion gate at COHORT level: the metric's MEDIAN across the champion setup's seeds must
 * satisfy `gate` on EVERY split value the champion was run on (the multi-window AND-of-medians family; also
 * how the capture / drawdown / trades gates are evaluated). A split contributes only when ≥1 of the
 * champion's runs there CARRIES the metric; when the metric is absent on every split the gate is NOT
 * applicable (skipped — e.g. the capture metrics' do-nothing sentinel), so it never rejects the champion.
 */
function evalCohortMedianGate(gate: TrainerGate, incRuns: AnalysisRun[], splitLevers: string[]): ChampionGate {
  const rendered = typeof gate.value === 'number' ? String(gate.value) : gate.value.metric
  const label = gate.label ?? `${gate.metric} ${gate.op} ${rendered}`
  const bySplit = new Map<string, AnalysisRun[]>()
  for (const r of incRuns) {
    const sv = splitLevers.length ? keyOf(r.config, splitLevers) : '·'
    const arr = bySplit.get(sv)
    if (arr) arr.push(r)
    else bySplit.set(sv, [r])
  }
  let windows = 0
  let held = 0
  let firstFail: string | undefined
  for (const [sv, rs] of bySplit) {
    const vals = rs.map((r) => metricOf(r, gate.metric)).filter((v): v is number => v !== undefined)
    if (!vals.length) continue // no data for this metric on this split → not part of the AND
    windows++
    const actual = medianOf(vals)
    let bound: number
    if (typeof gate.value === 'number') bound = gate.value
    else {
      const bv = rs
        .map((r) => metricOf(r, (gate.value as { metric: string }).metric))
        .filter((v): v is number => v !== undefined)
      bound = bv.length ? medianOf(bv) : NaN
    }
    if (applyGateOp(actual, gate.op, bound)) held++
    else if (firstFail === undefined) firstFail = `${sv} (median ${actual.toFixed(3)})`
  }
  if (!windows) return { label, kind: 'cohort-median', applicable: false, pass: false, detail: `${gate.metric} not emitted` }
  const pass = held === windows
  return {
    label,
    kind: 'cohort-median',
    applicable: true,
    pass,
    detail: pass ? `holds on all ${windows} window(s)` : `fails on ${firstFail} of ${windows}`,
  }
}

/**
 * The A4.3 Deflated-Sharpe cohort gate: the champion's Sharpe (median across its seeds), deflated for the
 * multiple testing of every distinct SETUP tried, must clear `threshold`. `n_trials` = the count of distinct
 * setups (the multiple-testing framing — NOT raw runs or windows); `trial_sr_std` = the std of each setup's
 * representative (median) Sharpe. Skipped (not applicable) when the champion emits no Sharpe metric.
 */
function evalDsrGate(
  dsr: NonNullable<TrainerDiagnostics['dsr']>,
  completed: AnalysisRun[],
  incRuns: AnalysisRun[],
  splitLevers: string[],
): ChampionGate {
  const sharpeKey = dsr.metric ?? 'oos_sharpe'
  const skewKey = dsr.skewMetric ?? 'oos_ret_skew'
  const kurtKey = dsr.kurtosisMetric ?? 'oos_ret_kurt'
  const nObsKey = dsr.nObsMetric ?? 'oos_n_obs'
  const threshold = dsr.threshold ?? 0.95
  const label = `deflated Sharpe ≥ ${threshold}`
  const finite = (v: number | undefined): v is number => v !== undefined
  const incSharpes = incRuns.map((r) => metricOf(r, sharpeKey)).filter(finite)
  if (!incSharpes.length) {
    return { label, kind: 'dsr', applicable: false, pass: false, detail: `${sharpeKey} not emitted` }
  }
  // n_trials = distinct SETUPS; trial_sr_std = std of each setup's median Sharpe (the cross-trial spread).
  const exclude = new Set<string>(['seed', ...splitLevers])
  const bySetup = new Map<string, number[]>()
  for (const r of completed) {
    const v = metricOf(r, sharpeKey)
    if (v === undefined) continue
    const k = setupKey(r.config, exclude)
    const arr = bySetup.get(k)
    if (arr) arr.push(v)
    else bySetup.set(k, [v])
  }
  const setupSharpes = [...bySetup.values()].map((vs) => medianOf(vs))
  // The honest trial count is a FLOOR on the observed setups, never a replacement — an override may only ever
  // deflate harder (see TrainerDiagnostics.dsr.nTrials).
  const declaredTrials = Number.isFinite(dsr.nTrials) ? Math.floor(dsr.nTrials as number) : 0
  const nTrials = Math.max(setupSharpes.length, declaredTrials)
  const trialSrStd = stdOf(setupSharpes)
  const incSharpe = medianOf(incSharpes)
  const skewVals = incRuns.map((r) => metricOf(r, skewKey)).filter(finite)
  const kurtVals = incRuns.map((r) => metricOf(r, kurtKey)).filter(finite)
  const nObsVals = incRuns.map((r) => metricOf(r, nObsKey)).filter(finite)
  const incSkew = skewVals.length ? medianOf(skewVals) : 0
  const incKurt = kurtVals.length ? medianOf(kurtVals) : 3
  const incNObs = nObsVals.length ? medianOf(nObsVals) : 0
  const dsrValue = deflatedSharpeFromStats(incSharpe, incSkew, incKurt, incNObs, nTrials, trialSrStd)
  const obsOk = dsr.minObs === undefined || incNObs >= dsr.minObs
  const pass = dsrValue >= threshold && obsOk
  const detail =
    `DSR ${dsrValue.toFixed(3)} vs ${threshold} over ${nTrials} setup(s)` +
    (obsOk ? '' : `; n_obs ${incNObs} < ${dsr.minObs}`)
  return { label, kind: 'dsr', applicable: true, pass, detail }
}

/**
 * The A4.3 seed-stability gate: the champion `metric` (default the ranking criterion) must have a bootstrap CI
 * width across its seeds no wider than `maxCiWidth` — a point estimate that swings with the seed is not a
 * steady edge. Skipped (not applicable) with fewer than 2 evaluable seeds.
 */
function evalStabilityGate(
  stability: NonNullable<TrainerDiagnostics['stability']>,
  incRuns: AnalysisRun[],
  criterion: AnalysisCriterion,
): ChampionGate {
  const key = stability.metric ?? criterion.key
  const label = `seed-stable (CI width ≤ ${stability.maxCiWidth})`
  const vals = incRuns.map((r) => metricOf(r, key)).filter((v): v is number => v !== undefined)
  if (vals.length < 2) {
    return { label, kind: 'seed-stability', applicable: false, pass: false, detail: 'need ≥ 2 seeds' }
  }
  const agg = aggregateRunValues(vals)
  const width = agg.ci[1] - agg.ci[0]
  const pass = width <= stability.maxCiWidth
  return {
    label,
    kind: 'seed-stability',
    applicable: true,
    pass,
    detail: `CI width ${width.toFixed(3)} vs ${stability.maxCiWidth}`,
  }
}

/** The cohort + split context the champion gates evaluate over (diagnoseSearch's projected runs). */
export interface ChampionVerdictContext {
  runs: AnalysisRun[]
  splitLevers: string[]
  criterion: AnalysisCriterion
}

/**
 * A4.3 composite champion "declare steady" verdict — the loop's stop condition the human approves against.
 * Folds the champion-gate families over the INCUMBENT setup: each declared {@link TrainerDiagnostics.championGates}
 * as a cohort median-per-window gate (multi-window-AND-of-medians / capture / drawdown / trades); DSR
 * deflation and seed-stability land in follow-up units. `steady` is UNDEFINED when the manifest declares no
 * champion config (not applicable — the verdict stays the split check alone); else true iff ≥1 gate is
 * APPLICABLE and every applicable gate holds (fail-closed — a champion with only skipped gates is never
 * declared steady; never a vacuous `true`). AUGMENTS the split verdict — it can only make the bar stricter.
 */
export function assembleChampionVerdict(
  diagnostics: TrainerDiagnostics | undefined,
  ctx: ChampionVerdictContext,
): { steady?: boolean; championGates: ChampionGate[] } {
  const gates = diagnostics?.championGates ?? []
  const hasConfig = gates.length > 0 || !!diagnostics?.dsr || !!diagnostics?.stability
  if (!hasConfig) return { championGates: [] }
  const { completed, incRuns } = selectIncumbent(ctx.runs, ctx.splitLevers, ctx.criterion)
  const championGates: ChampionGate[] = gates.map((g) => evalCohortMedianGate(g, incRuns, ctx.splitLevers))
  if (diagnostics?.dsr) championGates.push(evalDsrGate(diagnostics.dsr, completed, incRuns, ctx.splitLevers))
  if (diagnostics?.stability) championGates.push(evalStabilityGate(diagnostics.stability, incRuns, ctx.criterion))
  const applicable = championGates.filter((g) => g.applicable)
  const steady = applicable.length > 0 && applicable.every((g) => g.pass)
  return { steady, championGates }
}

/** The split levers a manifest declares for split-consistency (`diagnostics.splitAxis.levers`), or []. */
export function splitLeversOf(
  manifest: { diagnostics?: { splitAxis?: { levers?: unknown } } } | undefined,
): string[] {
  const levers = manifest?.diagnostics?.splitAxis?.levers
  return Array.isArray(levers) ? levers.filter((l): l is string => typeof l === 'string') : []
}

/** Pearson correlation of two equal-length series, or null when undefined (n < 3 or a flat series). */
function pearson(xs: number[], ys: number[]): number | null {
  if (xs.length < 3) return null
  const mx = mean(xs)
  const my = mean(ys)
  let sxy = 0
  let sxx = 0
  let syy = 0
  for (let i = 0; i < xs.length; i++) {
    const dx = xs[i] - mx
    const dy = ys[i] - my
    sxy += dx * dy
    sxx += dx * dx
    syy += dy * dy
  }
  if (sxx === 0 || syy === 0) return null
  return sxy / Math.sqrt(sxx * syy)
}

/**
 * The reward–success alignment diagnostic: for each fitness/gate metric, correlate the run OBJECTIVE
 * (the reward/steering proxy) against that metric across the cohort. A strong correlation means the
 * reward is a good proxy for success on that metric (CartPole: reward = success); a near-zero
 * correlation means the reward is a MISALIGNED proxy — a high reward does not imply a good model
 * (BlackSwan) — and selection must read the scorecard, not the reward.
 */
export function rewardFitnessAlignment(runs: AnalysisRun[], metrics: string[]): MetricAlignment[] {
  const completed = runs.filter((r) => (r.status ?? 'completed') === 'completed')
  return metrics.map((metric) => {
    const xs: number[] = []
    const ys: number[] = []
    for (const r of completed) {
      const x = valueOf(r, { key: 'objective', direction: 'max' })
      const y = valueOf(r, { key: metric, direction: 'max' })
      if (x === undefined || y === undefined) continue
      xs.push(x)
      ys.push(y)
    }
    return { metric, r: pearson(xs, ys), n: xs.length }
  })
}

/** A one-line read of the PRIMARY metric's {@link MetricAlignment} — the "is the reward a good proxy?" answer. */
export function narrateAlignment(
  alignments: MetricAlignment[],
  primaryMetric: string,
  weakThreshold = 0.3,
): string {
  const a = alignments.find((x) => x.metric === primaryMetric) ?? alignments[0]
  if (!a || a.r === null) {
    return `Reward–success alignment can't be measured yet (need ≥3 runs with a finite ${primaryMetric}). Once enough land, this reports whether a high reward actually predicts a good ${primaryMetric}.`
  }
  const r = a.r
  if (Math.abs(r) < weakThreshold) {
    return `The reward is a MISALIGNED proxy: it barely tracks ${a.metric} across ${a.n} runs (r=${r.toFixed(2)}) — a high reward does NOT imply success, so rank by the scorecard, not the reward.`
  }
  if (r > 0) {
    return `The reward is a GOOD proxy for ${a.metric} (r=${r.toFixed(2)} over ${a.n} runs) — it tracks success, so the objective is aligned.`
  }
  return `The reward is INVERTED against ${a.metric} (r=${r.toFixed(2)} over ${a.n} runs) — climbing the reward moves AWAY from success; the objective is actively misaligned.`
}

/** A one-paragraph human read of a {@link SplitHoldout} — the `diagnoseSearch` narrative + do-next. */
export function narrateSplitHoldout(
  holdout: SplitHoldout,
  splitLevers: string[],
  objectiveName: string,
  totalRuns: number,
): string {
  const axis = splitLevers.join(', ') || 'the split axis'
  switch (holdout.verdict) {
    case 'unverifiable':
      return `No split axis is declared, so generalization across markets/folds/windows can't be assessed (${totalRuns} runs). Declare diagnostics.splitAxis to enable the split-consistency check.`
    case 'not-replicated':
      return `The incumbent was evaluated on only ${holdout.evaluated} of ${holdout.splitValues.length} ${axis} split(s) — replicate it across the remaining ${holdout.missingSplits.length} before trusting it as a ${objectiveName} winner.`
    case 'single-split-luck':
      return `The incumbent beats the baseline on only ${holdout.held}/${holdout.evaluated} ${axis} split(s) — single-split luck, not a robust ${objectiveName} edge. Re-rank by worst-split objective and require consistency across splits before crowning a winner.`
    case 'robust':
      return `The incumbent holds across all ${holdout.evaluated} evaluated ${axis} split(s) — a robust ${objectiveName} edge.`
  }
}
