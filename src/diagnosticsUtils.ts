// Server-side diagnostics primitives — the split-consistency HOLDOUT that gates the exploration
// autopilot's convergence, mirroring the viewer's `split-consistency` check (viewer/diagnostics.js).
// Deterministic (no LLM): the incumbent (best SETUP, folding seeds + the split levers) must be evaluated
// on enough distinct split values AND beat the baseline on each, or a `converged` declaration is
// single-split luck and must be blocked. Pure — the controller (explorationUtils) and a future
// `diagnoseSearch` chat tool both consume it.

import type {
  AnalysisRun,
  AnalysisCriterion,
  GateOp,
  MetricAlignment,
  TrainerDiagnostics,
  TrainerGate,
  ChampionGate,
  VerifyCheck,
  VerifyVerdict,
} from './modelTrainerTypes'
import { applyGateOp } from './modelTrainerUtils.js'
import {
  medianOf,
  stdOf,
  iqm,
  aggregateRunValues,
  bootstrapDiff,
  pairedBootstrapDiff,
  normalCdf,
} from './xaiUtils.js'
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

  /** §C.7 per-split effect + CI vs the baseline — populated only when a CI-based held-test (`alpha`) is asked
   * for; [] under the mean test. Surfaces the interval the boolean `held` collapses. */
  splitEffects: Array<{
    split: string
    delta: number
    ci: [number, number]
    pValue: number
    held: boolean
  }>

  /** §C.3 the declared locked-TEST split values present in the run set — EXCLUDED from incumbent selection. */
  testSplits: string[]
  /** §C.3 how many locked-TEST splits the incumbent was evaluated on (the test consumed for the final verdict). */
  testConsumed: number
  /** §C.3 of the consumed test splits, how many the incumbent held on. */
  testHeld: number
}

function valueOf(run: AnalysisRun, criterion: AnalysisCriterion): number | undefined {
  if (criterion.key === 'objective')
    return typeof run.objective === 'number' ? run.objective : undefined
  if (criterion.key === 'durationMs')
    return typeof run.durationMs === 'number' ? run.durationMs : undefined
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
): { completed: AnalysisRun[]; incRuns: AnalysisRun[]; runnerUpRuns: AnalysisRun[] } {
  const completed = runs.filter(
    (r) => (r.status ?? 'completed') === 'completed' && valueOf(r, criterion) !== undefined,
  )
  if (!completed.length) return { completed, incRuns: [], runnerUpRuns: [] }
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
  // Rank setups by mean-of-criterion (stable, so ties keep insertion order — the first-wins incumbent the
  // strict-inequality scan used to give); the second is the runner-up the seed-significance gate tests against.
  const setups = [...bySetup.values()].map((rs) => ({
    rs,
    val: mean(rs.map((r) => valueOf(r, criterion)!)),
  }))
  setups.sort((a, b) => (criterion.direction === 'max' ? b.val - a.val : a.val - b.val))
  return {
    completed,
    incRuns: setups.length ? setups[0].rs : [],
    runnerUpRuns: setups.length > 1 ? setups[1].rs : [],
  }
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
  opts?: { minSplits?: number; baseline?: number; alpha?: number; testValues?: unknown[] },
): SplitHoldout {
  const minSplits = opts?.minSplits ?? 2
  const baseline = opts?.baseline ?? 0
  const alpha = opts?.alpha
  const testSet = new Set((opts?.testValues ?? []).map((v) => String(v)))
  const base: SplitHoldout = {
    verdict: 'unverifiable',
    evaluated: 0,
    held: 0,
    splitValues: [],
    missingSplits: [],
    missingSplitConfigs: [],
    evaluatedSplits: [],
    incumbentConfig: null,
    splitEffects: [],
    testSplits: [],
    testConsumed: 0,
    testHeld: 0,
  }
  if (!splitLevers.length) return base

  // §C.3 a run is a locked-TEST run when any of its split-lever values is declared a test value. Selection
  // must NOT read the test, so the incumbent is chosen from the non-test runs only; the incumbent is then
  // evaluated on ALL splits (including the test) so the test is consumed ONCE, post-selection.
  const isTestRun = (r: AnalysisRun) => splitLevers.some((l) => testSet.has(String(r.config[l])))
  const selectionRuns = testSet.size ? runs.filter((r) => !isTestRun(r)) : runs
  const { incRuns: selIncRuns } = selectIncumbent(selectionRuns, splitLevers, criterion)

  const exclude = new Set<string>(['seed', ...splitLevers])
  const allCompleted = runs.filter(
    (r) => (r.status ?? 'completed') === 'completed' && valueOf(r, criterion) !== undefined,
  )
  if (!allCompleted.length || !selIncRuns.length) return { ...base, verdict: 'not-replicated' }
  // The incumbent's runs across ALL splits (incl. the test), accepted-preferred, mirroring selectIncumbent's
  // own pool choice — so with no test declared this is exactly the runs selectIncumbent already returned.
  const incKey = setupKey(selIncRuns[0].config, exclude)
  const acceptedAll = allCompleted.filter((r) => r.accepted !== false)
  const incPool = acceptedAll.length ? acceptedAll : allCompleted
  const incRuns = incPool.filter((r) => setupKey(r.config, exclude) === incKey)

  // Map each split-key string back to its {lever: value} config (from any run carrying it) so the gate can
  // build split-fill recs without parsing the key string.
  const splitConfigByKey = new Map<string, Record<string, unknown>>()
  for (const r of allCompleted) {
    const k = keyOf(r.config, splitLevers)
    if (!splitConfigByKey.has(k)) {
      splitConfigByKey.set(k, Object.fromEntries(splitLevers.map((l) => [l, r.config[l]])))
    }
  }
  const isTestSplit = (sv: string) =>
    Object.values(splitConfigByKey.get(sv) ?? {}).some((v) => testSet.has(String(v)))

  const splitValues = [...splitConfigByKey.keys()].sort()
  let evaluated = 0
  let held = 0
  let testConsumed = 0
  let testHeld = 0
  const evaluatedSet = new Set<string>()
  const evaluatedSplits: SplitHoldout['evaluatedSplits'] = []
  const splitEffects: SplitHoldout['splitEffects'] = []
  for (const sv of splitValues) {
    const rs = incRuns.filter((r) => keyOf(r.config, splitLevers) === sv)
    if (!rs.length) continue
    evaluated++
    evaluatedSet.add(sv)
    evaluatedSplits.push({ splitConfig: splitConfigByKey.get(sv)!, seeds: rs.length })
    const vals = rs.map((r) => valueOf(r, criterion)!)
    let heldHere: boolean
    if (alpha !== undefined) {
      // §C.7 CI-based held-test: the split holds only when the incumbent's seed CI clears the baseline (a
      // high-variance split whose mean merely edges it no longer reads as robust).
      const d = bootstrapDiff(
        vals,
        vals.map(() => baseline),
        criterion.direction,
      )
      heldHere = d.ci[0] > 0 && d.pValue < alpha
      splitEffects.push({ split: sv, delta: d.delta, ci: d.ci, pValue: d.pValue, held: heldHere })
    } else {
      const v = mean(vals)
      heldHere = criterion.direction === 'max' ? v > baseline : v < baseline
    }
    if (heldHere) held++
    if (isTestSplit(sv)) {
      testConsumed++
      if (heldHere) testHeld++
    }
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
    splitEffects,
    testSplits: splitValues.filter(isTestSplit),
    testConsumed,
    testHeld,
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
function evalCohortMedianGate(
  gate: TrainerGate,
  incRuns: AnalysisRun[],
  splitLevers: string[],
): ChampionGate {
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
  if (!windows)
    return {
      label,
      kind: 'cohort-median',
      applicable: false,
      pass: false,
      detail: `${gate.metric} not emitted`,
    }
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
  floorTrials = 0,
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
    return {
      label,
      kind: 'dsr',
      applicable: false,
      pass: false,
      detail: `${sharpeKey} not emitted`,
    }
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
  // deflate harder (see TrainerDiagnostics.dsr.nTrials). The generic diagnostics.searchSpace floor is honoured
  // too, so the Sharpe path and the metric-agnostic best-of-N gate deflate against the SAME search size.
  const declaredTrials = Number.isFinite(dsr.nTrials) ? Math.floor(dsr.nTrials as number) : 0
  const genericTrials = Number.isFinite(floorTrials) ? Math.floor(floorTrials) : 0
  const nTrials = Math.max(setupSharpes.length, declaredTrials, genericTrials)
  const trialSrStd = stdOf(setupSharpes)
  const incSharpe = medianOf(incSharpes)
  const skewVals = incRuns.map((r) => metricOf(r, skewKey)).filter(finite)
  const kurtVals = incRuns.map((r) => metricOf(r, kurtKey)).filter(finite)
  const nObsVals = incRuns.map((r) => metricOf(r, nObsKey)).filter(finite)
  const incSkew = skewVals.length ? medianOf(skewVals) : 0
  const incKurt = kurtVals.length ? medianOf(kurtVals) : 3
  const incNObs = nObsVals.length ? medianOf(nObsVals) : 0
  const dsrValue = deflatedSharpeFromStats(
    incSharpe,
    incSkew,
    incKurt,
    incNObs,
    nTrials,
    trialSrStd,
  )
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
    return {
      label,
      kind: 'seed-stability',
      applicable: false,
      pass: false,
      detail: 'need ≥ 2 seeds',
    }
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

/**
 * §C.2 metric-agnostic best-of-N multiplicity gate — the generic sibling of {@link evalDsrGate} (DSR is its
 * Sharpe specialisation). Under the null that every distinct SETUP shares one true mean, the best of `nTrials`
 * noisy setups sits ~E[max]·(cross-setup std) above the mean by luck alone; so the champion (the argmax setup,
 * `metric` median across its seeds) must be an outlier even against THAT best-of-N null — the order-statistic
 * family p-value `1 − Φ(z)^nTrials` (z = the champion's cross-setup z-score) must be < `alpha`. `nTrials` is the
 * distinct-setup count, floored by the honest declared search size. Skipped when < 2 setups, no cross-setup
 * spread, or the metric is absent.
 */
function evalBestOfNGate(
  multiplicity: NonNullable<TrainerDiagnostics['multiplicity']>,
  completed: AnalysisRun[],
  incRuns: AnalysisRun[],
  splitLevers: string[],
  criterion: AnalysisCriterion,
  declaredTrials: number,
): ChampionGate {
  const key = multiplicity.metric ?? criterion.key
  const alpha = multiplicity.alpha ?? 0.05
  const label = `best-of-N deflated (p < ${alpha})`
  const finite = (v: number | undefined): v is number => v !== undefined
  const incVals = incRuns.map((r) => metricOf(r, key)).filter(finite)
  if (!incVals.length || !incRuns.length) {
    return {
      label,
      kind: 'best-of-n',
      applicable: false,
      pass: false,
      detail: `${key} not emitted`,
    }
  }
  const exclude = new Set<string>(['seed', ...splitLevers])
  const incKey = setupKey(incRuns[0].config, exclude)
  const bySetup = new Map<string, number[]>()
  for (const r of completed) {
    const v = metricOf(r, key)
    if (v === undefined) continue
    const k = setupKey(r.config, exclude)
    const arr = bySetup.get(k)
    if (arr) arr.push(v)
    else bySetup.set(k, [v])
  }
  // The null is the REST of the field — the champion's own setup is excluded so a lone real winner can't
  // inflate the noise scale it's judged against (the DSR path avoids this via each run's own sample length).
  const otherVals = [...bySetup.entries()]
    .filter(([k]) => k !== incKey)
    .map(([, vs]) => medianOf(vs))
  const trialStd = stdOf(otherVals)
  if (otherVals.length < 2 || !(trialStd > 0)) {
    return {
      label,
      kind: 'best-of-n',
      applicable: false,
      pass: false,
      detail: `${key}: need ≥ 2 other setups with spread (have ${otherVals.length})`,
    }
  }
  const nTrials = Math.max(
    bySetup.size,
    Number.isFinite(declaredTrials) ? Math.floor(declaredTrials) : 0,
  )
  const nullMean = mean(otherVals)
  const incVal = medianOf(incVals)
  const z =
    criterion.direction === 'max' ? (incVal - nullMean) / trialStd : (nullMean - incVal) / trialStd
  const familyP = 1 - Math.pow(normalCdf(z), nTrials)
  const pass = familyP < alpha
  return {
    label,
    kind: 'best-of-n',
    applicable: true,
    pass,
    detail: `champion z ${z.toFixed(2)} over ${nTrials} setup(s) ⇒ family p ${familyP.toFixed(3)} vs ${alpha}`,
  }
}

/**
 * §C.5 degeneracy gate: the champion's cohort must NOT satisfy any declared {@link TrainerDiagnostics.degenerateWhen}
 * rule (e.g. `draw_rate == 1`, `n_trades == 0`). The rule's metric is taken as the incumbent's cohort MEDIAN;
 * a match means the "win" is a degenerate artifact and the champion is not steady. These screens were declared
 * but engine-inert (viewer-only) before — a leaked/degenerate champion sailed through. Skipped (not applicable)
 * when no rule's metric is emitted.
 */
function evalDegeneracyGate(
  degenerateWhen: NonNullable<TrainerDiagnostics['degenerateWhen']>,
  incRuns: AnalysisRun[],
): ChampionGate {
  const label = 'not degenerate'
  let anyApplicable = false
  let firstDegenerate: string | undefined
  for (const rule of degenerateWhen) {
    const vals = incRuns
      .map((r) => metricOf(r, rule.metric))
      .filter((v): v is number => v !== undefined)
    if (!vals.length) continue
    anyApplicable = true
    const med = medianOf(vals)
    if (applyGateOp(med, rule.op as GateOp, rule.value)) {
      firstDegenerate = `${rule.metric} ${rule.op} ${rule.value} (median ${med.toFixed(3)})`
      break
    }
  }
  if (!anyApplicable) {
    return {
      label,
      kind: 'not-degenerate',
      applicable: false,
      pass: false,
      detail: 'no degeneracy metric emitted',
    }
  }
  const pass = firstDegenerate === undefined
  return {
    label,
    kind: 'not-degenerate',
    applicable: true,
    pass,
    detail: pass ? 'no degeneracy rule matched' : `degenerate: ${firstDegenerate}`,
  }
}

/**
 * §C.4 seed-significance gate: the champion's advantage over the RUNNER-UP setup on `metric` (default the
 * ranking criterion) must exceed seed noise — the delta's bootstrap CI (paired on common seeds when available,
 * else independent) must exclude 0 AND p < `alpha`. This is the delta-vs-seed-variance test the point-estimate
 * crowning never had: a champion that only edges the runner-up by a lucky seed is not steady. Not applicable
 * with no runner-up (a single setup) or fewer than 2 evaluable seeds a side.
 */
function evalSeedSignificanceGate(
  significance: NonNullable<TrainerDiagnostics['significance']>,
  incRuns: AnalysisRun[],
  runnerUpRuns: AnalysisRun[],
  criterion: AnalysisCriterion,
): ChampionGate {
  const key = significance.metric ?? criterion.key
  const alpha = significance.alpha ?? 0.05
  const label = `beats runner-up (p < ${alpha})`
  const finite = (v: number | undefined): v is number => v !== undefined
  const incVals = incRuns.map((r) => metricOf(r, key)).filter(finite)
  const runVals = runnerUpRuns.map((r) => metricOf(r, key)).filter(finite)
  if (incVals.length < 2 || runVals.length < 2) {
    const detail = runnerUpRuns.length ? 'need ≥ 2 seeds a side' : 'no runner-up setup'
    return { label, kind: 'seed-significance', applicable: false, pass: false, detail }
  }
  const seedVal = (r: AnalysisRun): [number, number] | undefined => {
    const v = metricOf(r, key)
    return typeof r.seed === 'number' && v !== undefined ? [r.seed, v] : undefined
  }
  const incBySeed = new Map(incRuns.map(seedVal).filter((p): p is [number, number] => !!p))
  const runBySeed = new Map(runnerUpRuns.map(seedVal).filter((p): p is [number, number] => !!p))
  const common = [...incBySeed.keys()].filter((s) => runBySeed.has(s)).sort((a, b) => a - b)
  const { ci, pValue, delta } =
    common.length >= 2
      ? pairedBootstrapDiff(
          common.map((s) => incBySeed.get(s)!),
          common.map((s) => runBySeed.get(s)!),
          criterion.direction,
        )
      : bootstrapDiff(incVals, runVals, criterion.direction)
  const pass = ci[0] > 0 && pValue < alpha
  return {
    label,
    kind: 'seed-significance',
    applicable: true,
    pass,
    detail: `Δ ${delta.toFixed(3)} [${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}] p=${pValue.toFixed(3)}`,
    effect: { delta, ci, pValue },
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
  const hasConfig =
    gates.length > 0 ||
    !!diagnostics?.dsr ||
    !!diagnostics?.stability ||
    !!diagnostics?.significance ||
    !!diagnostics?.multiplicity ||
    !!diagnostics?.degenerateWhen?.length
  if (!hasConfig) return { championGates: [] }
  const { completed, incRuns, runnerUpRuns } = selectIncumbent(
    ctx.runs,
    ctx.splitLevers,
    ctx.criterion,
  )
  const championGates: ChampionGate[] = gates.map((g) =>
    evalCohortMedianGate(g, incRuns, ctx.splitLevers),
  )
  if (diagnostics?.degenerateWhen?.length)
    championGates.push(evalDegeneracyGate(diagnostics.degenerateWhen, incRuns))
  if (diagnostics?.dsr)
    championGates.push(
      evalDsrGate(
        diagnostics.dsr,
        completed,
        incRuns,
        ctx.splitLevers,
        diagnostics.searchSpace?.nTrials ?? 0,
      ),
    )
  if (diagnostics?.stability)
    championGates.push(evalStabilityGate(diagnostics.stability, incRuns, ctx.criterion))
  if (diagnostics?.significance)
    championGates.push(
      evalSeedSignificanceGate(diagnostics.significance, incRuns, runnerUpRuns, ctx.criterion),
    )
  if (diagnostics?.multiplicity)
    championGates.push(
      evalBestOfNGate(
        diagnostics.multiplicity,
        completed,
        incRuns,
        ctx.splitLevers,
        ctx.criterion,
        diagnostics.searchSpace?.nTrials ?? 0,
      ),
    )
  const applicable = championGates.filter((g) => g.applicable)
  const steady = applicable.length > 0 && applicable.every((g) => g.pass)
  return { steady, championGates }
}

/**
 * §C.6 the first-class adversarial-verify pass for a CLAIMED improvement — the "try to refute this winner"
 * battery that was previously a manual workflow. It re-derives the incumbent and runs up to four INDEPENDENT
 * lenses, each of which can only make the bar stricter: (1) seed-stability — the incumbent's edge over the
 * baseline survives re-seeding (CI excludes the baseline); (2) split-robust — it holds across the declared
 * split axis; (3) nuisance-robust — sibling setups that differ ONLY on a declared-irrelevant lever still clear
 * the baseline (the win isn't an artifact of a nuisance knob); (4) aggregator-agreement — the incumbent is the
 * argmax setup whether seeds are folded by mean, median, or IQM (an independent re-analysis). `verified` iff ≥1
 * lens was applicable and all applicable lenses passed; `unverifiable` when NONE could run — the naive
 * single-seed, no-split, no-nuisance sweep that today's point-estimate crowning would pass silently.
 */
export function verifyImprovement(
  runs: AnalysisRun[],
  criterion: AnalysisCriterion,
  opts?: { baseline?: number; alpha?: number; splitLevers?: string[]; nuisanceLevers?: string[] },
): VerifyVerdict {
  const baseline = opts?.baseline ?? 0
  const alpha = opts?.alpha ?? 0.05
  const splitLevers = opts?.splitLevers ?? []
  const nuisance = opts?.nuisanceLevers ?? []
  const finite = (v: number | undefined): v is number => v !== undefined
  const { incRuns } = selectIncumbent(runs, splitLevers, criterion)
  const incumbentConfig = incRuns.length ? incRuns[0].config : null
  const incVals = incRuns.map((r) => valueOf(r, criterion)).filter(finite)
  const clears = (v: number) => (criterion.direction === 'max' ? v > baseline : v < baseline)
  const checks: VerifyCheck[] = []

  // 1) seed-stability — the incumbent's advantage over the baseline survives re-seeding.
  if (incVals.length >= 2) {
    const d = bootstrapDiff(
      incVals,
      incVals.map(() => baseline),
      criterion.direction,
    )
    checks.push({
      name: 'seed-stability',
      applicable: true,
      pass: d.ci[0] > 0 && d.pValue < alpha,
      detail: `Δ vs baseline [${d.ci[0].toFixed(3)}, ${d.ci[1].toFixed(3)}] p=${d.pValue.toFixed(3)}`,
    })
  } else {
    checks.push({
      name: 'seed-stability',
      applicable: false,
      pass: false,
      detail: `only ${incVals.length} seed`,
    })
  }

  // 2) split-robust — the win holds across the declared split axis.
  if (splitLevers.length) {
    const h = incumbentSplitHoldout(runs, splitLevers, criterion, { baseline })
    checks.push({
      name: 'split-robust',
      applicable: h.evaluated >= 2,
      pass: h.verdict === 'robust',
      detail: `${h.held}/${h.evaluated} split(s) held`,
    })
  } else {
    checks.push({
      name: 'split-robust',
      applicable: false,
      pass: false,
      detail: 'no split axis declared',
    })
  }

  // 3) nuisance-robust — sibling setups differing ONLY on a declared nuisance lever still clear the baseline.
  const completed = runs.filter(
    (r) => (r.status ?? 'completed') === 'completed' && valueOf(r, criterion) !== undefined,
  )
  const nuKeys = incumbentConfig ? nuisance.filter((l) => l in incumbentConfig) : []
  if (incumbentConfig && nuKeys.length) {
    const identity = new Set(['seed', ...splitLevers, ...nuisance])
    const incIdentity = setupKey(incumbentConfig, identity)
    const siblings = completed.filter(
      (r) =>
        setupKey(r.config, identity) === incIdentity &&
        nuKeys.some((l) => String(r.config[l]) !== String(incumbentConfig[l])),
    )
    const bySib = new Map<string, number[]>()
    for (const r of siblings) {
      const k = setupKey(r.config, new Set(['seed', ...splitLevers]))
      const arr = bySib.get(k)
      if (arr) arr.push(valueOf(r, criterion)!)
      else bySib.set(k, [valueOf(r, criterion)!])
    }
    const allClear = [...bySib.values()].every((vs) => clears(mean(vs)))
    checks.push({
      name: 'nuisance-robust',
      applicable: bySib.size > 0,
      pass: bySib.size > 0 && allClear,
      detail: bySib.size ? `${bySib.size} nuisance neighbour(s)` : 'no nuisance neighbours run',
    })
  } else {
    checks.push({
      name: 'nuisance-robust',
      applicable: false,
      pass: false,
      detail: nuisance.length
        ? 'incumbent carries no nuisance lever'
        : 'no nuisance levers declared',
    })
  }

  // 4) aggregator-agreement — the incumbent is the argmax setup under mean, median AND IQM (independent re-read).
  if (incVals.length >= 2 && incumbentConfig) {
    const exclude = new Set<string>(['seed', ...splitLevers])
    const bySetup = new Map<string, number[]>()
    for (const r of completed) {
      const k = setupKey(r.config, exclude)
      const arr = bySetup.get(k)
      if (arr) arr.push(valueOf(r, criterion)!)
      else bySetup.set(k, [valueOf(r, criterion)!])
    }
    const argmaxBy = (agg: (vs: number[]) => number): string => {
      let best = ''
      let bestV = criterion.direction === 'max' ? -Infinity : Infinity
      for (const [k, vs] of bySetup) {
        const v = agg(vs)
        if (criterion.direction === 'max' ? v > bestV : v < bestV) {
          bestV = v
          best = k
        }
      }
      return best
    }
    const incKey = setupKey(incumbentConfig, exclude)
    const agree =
      argmaxBy(mean) === incKey && argmaxBy(medianOf) === incKey && argmaxBy(iqm) === incKey
    checks.push({
      name: 'aggregator-agreement',
      applicable: bySetup.size >= 2,
      pass: agree,
      detail: agree
        ? 'mean/median/IQM agree on the incumbent'
        : 'aggregators disagree on the incumbent',
    })
  } else {
    checks.push({
      name: 'aggregator-agreement',
      applicable: false,
      pass: false,
      detail: 'incumbent has < 2 seeds',
    })
  }

  const applicable = checks.filter((c) => c.applicable)
  return {
    verified: applicable.length > 0 && applicable.every((c) => c.pass),
    unverifiable: applicable.length === 0,
    incumbentConfig,
    checks,
  }
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

/** Average-rank transform (ties share the mean rank) — the basis for a Spearman rank correlation. */
function rankOf(xs: number[]): number[] {
  const order = xs.map((v, i) => [v, i] as [number, number]).sort((a, b) => a[0] - b[0])
  const ranks = new Array<number>(xs.length).fill(0)
  for (let i = 0; i < order.length; ) {
    let j = i
    while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++
    const avg = (i + j) / 2
    for (let k = i; k <= j; k++) ranks[order[k][1]] = avg
    i = j + 1
  }
  return ranks
}

/** Spearman rank correlation — Pearson on the rank-transformed series; null when < 3 pairs or no spread. */
function spearman(xs: number[], ys: number[]): number | null {
  if (xs.length < 3) return null
  return pearson(rankOf(xs), rankOf(ys))
}

/**
 * §C.8 the proxy-vs-true SELECTION-REGRET diagnostic — what a bare correlation cannot see. Folds each setup
 * (over seeds; the caller's `splitLevers` are folded too so a setup is one config, not one split) to its mean
 * OBJECTIVE (the reward proxy) and its mean `metric` (the truth), then reports: `pearson`/`spearman` (do they
 * co-move) AND `regret` — the true-metric gap between the setup that WINS on the objective and the setup that
 * wins on the metric. A large regret with a high correlation is exactly the tail divergence that makes crowning
 * by reward pick a worse model. NaN regret when no setup carries both.
 */
export function proxyAlignment(
  runs: AnalysisRun[],
  metric: string,
  direction: 'max' | 'min' = 'max',
  splitLevers: string[] = [],
): import('./modelTrainerTypes').ProxyAlignment {
  const completed = runs.filter((r) => (r.status ?? 'completed') === 'completed')
  const exclude = new Set<string>(['seed', ...splitLevers])
  const bySetup = new Map<string, { obj: number[]; met: number[] }>()
  for (const r of completed) {
    const o = valueOf(r, { key: 'objective', direction: 'max' })
    const m = valueOf(r, { key: metric, direction })
    if (o === undefined || m === undefined) continue
    const k = setupKey(r.config, exclude)
    const e = bySetup.get(k) ?? { obj: [], met: [] }
    e.obj.push(o)
    e.met.push(m)
    bySetup.set(k, e)
  }
  const setups = [...bySetup.values()].map((e) => ({ obj: mean(e.obj), met: mean(e.met) }))
  const pear = pearson(
    setups.map((s) => s.obj),
    setups.map((s) => s.met),
  )
  const spear = spearman(
    setups.map((s) => s.obj),
    setups.map((s) => s.met),
  )
  let regret = NaN
  if (setups.length) {
    const byObj = [...setups].sort((a, b) => b.obj - a.obj)[0]
    const byMet = [...setups].sort((a, b) =>
      direction === 'max' ? b.met - a.met : a.met - b.met,
    )[0]
    regret = direction === 'max' ? byMet.met - byObj.met : byObj.met - byMet.met
  }
  return { metric, pearson: pear, spearman: spear, regret, n: setups.length }
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
