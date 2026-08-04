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
  ChampionGate,
} from './modelTrainerTypes'

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
    incumbentConfig: null,
  }
  if (!splitLevers.length) return base

  const completed = runs.filter(
    (r) => (r.status ?? 'completed') === 'completed' && valueOf(r, criterion) !== undefined,
  )
  if (!completed.length) return { ...base, verdict: 'not-replicated' }

  // Choose the incumbent from gate-ACCEPTED runs only (a gate-failing run is never crowned when an
  // accepted one exists); fall back to all completed when none pass so the verdict still has an incumbent.
  // `accepted === undefined` ⇒ gates not evaluated ⇒ treated as accepted. The split universe below stays
  // over ALL completed runs (a rejected setup still counts as a split we've covered).
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
  const evaluatedSplits = new Set<string>()
  for (const sv of splitValues) {
    const rs = incRuns.filter((r) => keyOf(r.config, splitLevers) === sv)
    if (!rs.length) continue
    evaluated++
    evaluatedSplits.add(sv)
    const v = mean(rs.map((r) => valueOf(r, criterion)!))
    if (criterion.direction === 'max' ? v > baseline : v < baseline) held++
  }
  const missingSplits = splitValues.filter((sv) => !evaluatedSplits.has(sv))

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
 * A4.3 composite champion "declare steady" verdict (SCAFFOLD). Folds the champion-gate families
 * (multi-window-AND-of-medians / DSR deflation / beta-capture / seed-stability) into ONE steady/not-steady
 * envelope with per-gate detail — the loop's stop condition the human approves against. The families are
 * wired in follow-up units; today it returns the EMPTY envelope, honestly: `steady` is UNDEFINED when the
 * manifest declares no champion config (not applicable — the verdict stays the split check alone), and
 * `false` when it declares any (fail-closed — nothing is proven steady until a family evaluates it; never a
 * vacuous `true`). An empty `championGates: []` array is not "declared config": there is nothing to prove.
 */
export function assembleChampionVerdict(diagnostics?: TrainerDiagnostics): {
  steady?: boolean
  championGates: ChampionGate[]
} {
  const hasConfig = !!(
    diagnostics &&
    ((diagnostics.championGates && diagnostics.championGates.length > 0) ||
      diagnostics.dsr ||
      diagnostics.stability)
  )
  const championGates: ChampionGate[] = []
  return hasConfig ? { steady: false, championGates } : { championGates }
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
