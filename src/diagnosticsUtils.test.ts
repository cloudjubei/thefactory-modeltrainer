import { describe, it, expect } from 'vitest'

import {
  incumbentSplitHoldout,
  convergenceGatedBySplits,
  splitLeversOf,
  narrateSplitHoldout,
  rewardFitnessAlignment,
  narrateAlignment,
  assembleChampionVerdict,
} from './diagnosticsUtils'
import type { AnalysisRun, AnalysisCriterion } from './modelTrainerTypes'

const crit: AnalysisCriterion = { key: 'objective', direction: 'max' }
const run = (config: Record<string, unknown>, objective: number, seed = 0): AnalysisRun => ({
  key: `${JSON.stringify(config)}-${seed}`,
  config,
  objective,
  seed,
  status: 'completed',
})

describe('incumbentSplitHoldout', () => {
  it('is unverifiable when no split axis is declared', () => {
    expect(incumbentSplitHoldout([run({ lr: 1 }, 5)], [], crit).verdict).toBe('unverifiable')
  })

  it('is not-replicated when the incumbent ran on fewer than minSplits splits', () => {
    const runs = [
      run({ lr: 1, window: '2024' }, 10), // incumbent (best), only on 2024
      run({ lr: 2, window: '2024' }, 3),
      run({ lr: 2, window: '2022' }, 3),
    ]
    const h = incumbentSplitHoldout(runs, ['window'], crit)
    expect(h.verdict).toBe('not-replicated')
    expect(h.missingSplits).toContain('window=2022')
    // The missing splits are exposed as configs (the split-fill target the gate replicates the incumbent on).
    expect(h.missingSplitConfigs).toContainEqual({ window: '2022' })
  })

  it('is single-split-luck when the incumbent wins one split but fails another', () => {
    const runs = [
      run({ lr: 1, window: '2024' }, 20), // great in 2024
      run({ lr: 1, window: '2022' }, -5), // negative in 2022
    ]
    const h = incumbentSplitHoldout(runs, ['window'], crit)
    expect(h.evaluated).toBe(2)
    expect(h.held).toBe(1)
    expect(h.verdict).toBe('single-split-luck')
    expect(convergenceGatedBySplits(h)).toBe(true)
  })

  it('is robust when the incumbent beats the baseline on every evaluated split', () => {
    const runs = [run({ lr: 1, window: '2024' }, 8), run({ lr: 1, window: '2022' }, 3)]
    const h = incumbentSplitHoldout(runs, ['window'], crit)
    expect(h.verdict).toBe('robust')
    expect(h.held).toBe(2)
    expect(convergenceGatedBySplits(h)).toBe(false)
  })

  it('respects a custom baseline', () => {
    const runs = [run({ lr: 1, window: '2024' }, 8), run({ lr: 1, window: '2022' }, 3)]
    // baseline 5 -> the 2022 value (3) no longer holds -> single-split-luck
    expect(incumbentSplitHoldout(runs, ['window'], crit, { baseline: 5 }).verdict).toBe('single-split-luck')
  })

  it('folds seeds — the incumbent is the best SETUP, not the best single run', () => {
    const runs = [
      run({ lr: 1, window: '2024' }, 9, 0),
      run({ lr: 1, window: '2024' }, 7, 1), // same setup, second seed -> mean 8
      run({ lr: 1, window: '2022' }, 4, 0),
    ]
    const h = incumbentSplitHoldout(runs, ['window'], crit)
    expect(h.incumbentConfig).toMatchObject({ lr: 1 })
    expect(h.verdict).toBe('robust')
  })

  it('prefers the best gate-ACCEPTED setup over a higher-ranked rejected one', () => {
    const runs = [
      { ...run({ lr: 9, window: '2024' }, 100), accepted: false }, // best objective but REJECTED
      { ...run({ lr: 9, window: '2022' }, 90), accepted: false },
      { ...run({ lr: 1, window: '2024' }, 30), accepted: true }, // lower objective, ACCEPTED
      { ...run({ lr: 1, window: '2022' }, 20), accepted: true },
    ]
    const h = incumbentSplitHoldout(runs, ['window'], crit)
    expect(h.incumbentConfig).toMatchObject({ lr: 1 })
    expect(h.verdict).toBe('robust')
  })

  it('falls back to all runs when NO run passes gates (still yields an incumbent)', () => {
    const runs = [
      { ...run({ lr: 1, window: '2024' }, 8), accepted: false },
      { ...run({ lr: 1, window: '2022' }, 3), accepted: false },
    ]
    const h = incumbentSplitHoldout(runs, ['window'], crit)
    expect(h.incumbentConfig).toMatchObject({ lr: 1 })
    expect(h.verdict).toBe('robust')
  })

  it('treats runs with no acceptance flag as accepted (gates not evaluated)', () => {
    const runs = [run({ lr: 1, window: '2024' }, 8), run({ lr: 1, window: '2022' }, 3)]
    expect(incumbentSplitHoldout(runs, ['window'], crit).verdict).toBe('robust')
  })

  it('uses a metrics.* criterion when asked', () => {
    const r = (w: string, v: number): AnalysisRun => ({
      key: `${w}`,
      config: { window: w },
      metrics: { return_vs_hold_pct: v },
      status: 'completed',
    })
    const h = incumbentSplitHoldout([r('2024', 6), r('2022', 2)], ['window'], {
      key: 'return_vs_hold_pct',
      direction: 'max',
    })
    expect(h.verdict).toBe('robust')
  })

  it('ranks by a min-direction durationMs criterion (lower is better) and drops runs missing the metric', () => {
    const durCrit: AnalysisCriterion = { key: 'durationMs', direction: 'min' }
    const runs: AnalysisRun[] = [
      { key: 'f1', config: { lr: 1, window: '2024' }, durationMs: 100, status: 'completed' },
      { key: 'f2', config: { lr: 1, window: '2022' }, durationMs: 200, status: 'completed' },
      { key: 's1', config: { lr: 9, window: '2024' }, durationMs: 1000, status: 'completed' },
      { key: 's2', config: { lr: 9, window: '2022' }, durationMs: 1100, status: 'completed' },
      { key: 'nodur', config: { lr: 1, window: '2019' }, status: 'completed' }, // no durationMs -> dropped
    ]
    const h = incumbentSplitHoldout(runs, ['window'], durCrit, { baseline: 500 })
    expect(h.incumbentConfig).toMatchObject({ lr: 1 }) // faster setup (mean 150) beats the slow one (mean 1050)
    expect(h.splitValues).toEqual(['window=2022', 'window=2024']) // the durationMs-less 2019 run is out of the universe
    expect(h.held).toBe(2) // min-direction: both incumbent splits (100, 200) sit under the 500 baseline
    expect(h.verdict).toBe('robust')
  })

  it('is not-replicated with a split axis but no completed runs (empty-completed guard)', () => {
    const runs: AnalysisRun[] = [
      { key: 'a', config: { lr: 1, window: '2024' }, objective: 5, status: 'failed' },
      { key: 'b', config: { lr: 1, window: '2022' }, objective: 5, status: 'running' },
    ]
    const h = incumbentSplitHoldout(runs, ['window'], crit)
    expect(h.verdict).toBe('not-replicated')
    expect(h.evaluated).toBe(0)
    expect(h.splitValues).toEqual([])
    expect(h.incumbentConfig).toBeNull()
  })

  it('defaults a missing status to completed and drops explicitly non-completed runs', () => {
    const runs: AnalysisRun[] = [
      { key: 'a', config: { lr: 1, window: '2024' }, objective: 8 }, // no status -> defaulted to completed
      { key: 'b', config: { lr: 1, window: '2022' }, objective: 3 }, // no status -> defaulted to completed
      { key: 'c', config: { lr: 1, window: '2020' }, objective: 99, status: 'failed' }, // dropped from the universe
    ]
    const h = incumbentSplitHoldout(runs, ['window'], crit)
    expect(h.splitValues).toEqual(['window=2022', 'window=2024']) // the failed 2020 split is not covered
    expect(h.held).toBe(2)
    expect(h.verdict).toBe('robust')
  })
})

describe('rewardFitnessAlignment', () => {
  // A run carrying the objective (reward proxy) plus arbitrary summary metrics.
  const r = (objective: number, metrics: Record<string, number>): AnalysisRun => ({
    key: `${objective}-${JSON.stringify(metrics)}`,
    config: {},
    objective,
    metrics,
    status: 'completed',
  })

  it('reports r≈+1 when a metric moves exactly with the objective', () => {
    const runs = [r(1, { m: 1 }), r(2, { m: 2 }), r(3, { m: 3 }), r(4, { m: 4 })]
    const [a] = rewardFitnessAlignment(runs, ['m'])
    expect(a.metric).toBe('m')
    expect(a.r).toBeCloseTo(1, 6)
    expect(a.n).toBe(4)
  })

  it('reports r≈-1 when a metric moves opposite the objective', () => {
    const runs = [r(1, { m: 4 }), r(2, { m: 3 }), r(3, { m: 2 }), r(4, { m: 1 })]
    expect(rewardFitnessAlignment(runs, ['m'])[0].r).toBeCloseTo(-1, 6)
  })

  it('reports r≈0 when the reward is DECORRELATED from the metric (the misaligned-proxy case)', () => {
    const runs = [r(-1, { m: 1 }), r(0, { m: 0 }), r(1, { m: 1 })]
    expect(rewardFitnessAlignment(runs, ['m'])[0].r).toBeCloseTo(0, 6)
  })

  it('returns r=null with fewer than 3 finite pairs', () => {
    const a = rewardFitnessAlignment([r(1, { m: 1 }), r(2, { m: 2 })], ['m'])[0]
    expect(a.r).toBeNull()
    expect(a.n).toBe(2)
  })

  it('returns r=null when the metric has zero variance (undefined correlation)', () => {
    const runs = [r(1, { m: 5 }), r(2, { m: 5 }), r(3, { m: 5 })]
    expect(rewardFitnessAlignment(runs, ['m'])[0].r).toBeNull()
  })

  it('counts only runs where BOTH the objective and the metric are finite', () => {
    const runs = [r(1, { m: 1 }), r(2, {}), r(3, { m: 3 }), r(4, { m: 4 })]
    const a = rewardFitnessAlignment(runs, ['m'])[0]
    expect(a.n).toBe(3) // the metric-less run is dropped
  })

  it('handles multiple metrics independently', () => {
    const runs = [r(1, { a: 1, b: 3 }), r(2, { a: 2, b: 2 }), r(3, { a: 3, b: 1 })]
    const out = rewardFitnessAlignment(runs, ['a', 'b'])
    expect(out.map((x) => x.metric)).toEqual(['a', 'b'])
    expect(out[0].r).toBeCloseTo(1, 6)
    expect(out[1].r).toBeCloseTo(-1, 6)
  })

  it('drops runs missing the objective (reward proxy) from the correlation', () => {
    const runs: AnalysisRun[] = [
      r(1, { m: 1 }),
      r(2, { m: 2 }),
      r(3, { m: 3 }),
      { key: 'no-obj', config: {}, metrics: { m: 4 }, status: 'completed' }, // no objective -> pair dropped
    ]
    const a = rewardFitnessAlignment(runs, ['m'])[0]
    expect(a.n).toBe(3)
  })

  it('excludes non-completed runs but treats a missing status as completed', () => {
    const runs: AnalysisRun[] = [
      r(1, { m: 1 }),
      r(2, { m: 2 }),
      { key: 'no-status', config: {}, objective: 3, metrics: { m: 3 } }, // no status -> defaulted to completed, kept
      { key: 'running', config: {}, objective: 99, metrics: { m: 99 }, status: 'running' }, // not completed, dropped
    ]
    const a = rewardFitnessAlignment(runs, ['m'])[0]
    expect(a.n).toBe(3) // the two r() runs + the status-less run; the running run is excluded
  })
})

describe('narrateAlignment', () => {
  it('flags a decorrelated primary metric as a misaligned reward proxy', () => {
    const msg = narrateAlignment([{ metric: 'return_vs_hold_pct', r: 0.05, n: 40 }], 'return_vs_hold_pct')
    expect(msg).toMatch(/misaligned|does not (predict|imply)/i)
    expect(msg).toMatch(/return_vs_hold_pct/)
  })

  it('confirms a well-aligned reward when the primary metric tracks it', () => {
    const msg = narrateAlignment([{ metric: 'eval_return', r: 0.92, n: 40 }], 'eval_return')
    expect(msg).toMatch(/aligned|good proxy|tracks/i)
  })

  it('is explicit when alignment cannot be measured yet', () => {
    const msg = narrateAlignment([{ metric: 'x', r: null, n: 2 }], 'x')
    expect(msg).toMatch(/can'?t|cannot|not enough|insufficient/i)
  })

  it('falls back to the first alignment when the primary metric is absent', () => {
    const msg = narrateAlignment([{ metric: 'other', r: 0.9, n: 40 }], 'missing')
    expect(msg).toMatch(/good proxy|aligned|tracks/i)
    expect(msg).toMatch(/other/)
  })

  it("reports can't-measure when there are no alignments at all", () => {
    const msg = narrateAlignment([], 'return_vs_hold_pct')
    expect(msg).toMatch(/can'?t|cannot/i)
    expect(msg).toMatch(/return_vs_hold_pct/)
  })

  it('flags an INVERTED reward when the primary metric strongly anti-correlates', () => {
    const msg = narrateAlignment([{ metric: 'return_vs_hold_pct', r: -0.8, n: 40 }], 'return_vs_hold_pct')
    expect(msg).toMatch(/INVERTED/)
    expect(msg).toMatch(/-0\.80/)
  })
})

describe('splitLeversOf', () => {
  it('reads diagnostics.splitAxis.levers, else []', () => {
    expect(splitLeversOf({ diagnostics: { splitAxis: { levers: ['walk_forward_window'] } } })).toEqual([
      'walk_forward_window',
    ])
    expect(splitLeversOf({})).toEqual([])
    expect(splitLeversOf(undefined)).toEqual([])
  })
})

describe('narrateSplitHoldout', () => {
  const h = (over: Partial<Parameters<typeof narrateSplitHoldout>[0]>) =>
    narrateSplitHoldout(
      { verdict: 'robust', evaluated: 3, held: 3, splitValues: ['a', 'b', 'c'], missingSplits: [], missingSplitConfigs: [], incumbentConfig: {}, ...over } as any,
      ['walk_forward_window'],
      'return_vs_hold_pct',
      100,
    )
  it('narrates each verdict with a do-next', () => {
    expect(h({ verdict: 'unverifiable' })).toMatch(/Declare diagnostics.splitAxis/)
    expect(h({ verdict: 'not-replicated', evaluated: 1, missingSplits: ['b', 'c'] })).toMatch(/only 1 of 3/)
    expect(h({ verdict: 'single-split-luck', evaluated: 3, held: 1 })).toMatch(/single-split luck/)
    expect(h({ verdict: 'robust' })).toMatch(/holds across all 3/)
  })

  it("uses 'the split axis' as the axis name when no split levers are named", () => {
    const msg = narrateSplitHoldout(
      { verdict: 'not-replicated', evaluated: 1, held: 0, splitValues: ['a', 'b'], missingSplits: ['b'], missingSplitConfigs: [], evaluatedSplits: [], incumbentConfig: {} },
      [],
      'return_vs_hold_pct',
      50,
    )
    expect(msg).toMatch(/the split axis/)
  })
})

describe('assembleChampionVerdict (A4.3 champion "declare steady" verdict)', () => {
  const ctxOf = (runs: AnalysisRun[], splitLevers = ['window']) => ({ runs, splitLevers, criterion: crit })
  // A run of the incumbent setup {lr:1} on a walk-forward window, carrying the given summary metrics.
  const cr = (window: string, obj: number, metrics: Record<string, number>, seed = 0): AnalysisRun => ({
    key: `${window}-${seed}`,
    config: { lr: 1, window },
    objective: obj,
    metrics,
    seed,
    status: 'completed',
  })

  it('no champion config ⇒ steady undefined (not applicable), no gates', () => {
    const c = ctxOf([cr('2024', 5, {})])
    expect(assembleChampionVerdict(undefined, c)).toEqual({ championGates: [] })
    expect(assembleChampionVerdict({}, c)).toEqual({ championGates: [] })
    expect(assembleChampionVerdict({ splitAxis: { levers: ['window'] } }, c)).toEqual({ championGates: [] })
  })

  it('steady when the metric MEDIAN across seeds beats the bound on EVERY window', () => {
    const runs = [
      cr('2024', 5, { return_vs_hold_pct: 6 }, 0),
      cr('2024', 5, { return_vs_hold_pct: 4 }, 1), // median 5 > 0
      cr('2022', 3, { return_vs_hold_pct: 2 }, 0),
      cr('2022', 3, { return_vs_hold_pct: 8 }, 1), // median 5 > 0
    ]
    const v = assembleChampionVerdict({ championGates: [{ metric: 'return_vs_hold_pct', op: '>', value: 0 }] }, ctxOf(runs))
    expect(v.steady).toBe(true)
    expect(v.championGates).toHaveLength(1)
    expect(v.championGates[0]).toMatchObject({ applicable: true, pass: true, kind: 'cohort-median' })
  })

  it('NOT steady when the median fails on even ONE window (the multi-window AND)', () => {
    const runs = [
      cr('2024', 5, { return_vs_hold_pct: 6 }, 0),
      cr('2024', 5, { return_vs_hold_pct: 4 }, 1), // median 5 > 0 ✓
      cr('2022', -3, { return_vs_hold_pct: -5 }, 0),
      cr('2022', -3, { return_vs_hold_pct: -1 }, 1), // median -3 < 0 ✗
    ]
    const v = assembleChampionVerdict({ championGates: [{ metric: 'return_vs_hold_pct', op: '>', value: 0 }] }, ctxOf(runs))
    expect(v.steady).toBe(false)
    expect(v.championGates[0].pass).toBe(false)
  })

  it('uses the MEDIAN not the mean — one outlier seed cannot carry a window', () => {
    // mean([-1,-1,9]) = 2.33 > 0 but median = -1 < 0 ⇒ the window fails on medians.
    const runs = [
      cr('2024', 1, { return_vs_hold_pct: -1 }, 0),
      cr('2024', 1, { return_vs_hold_pct: -1 }, 1),
      cr('2024', 1, { return_vs_hold_pct: 9 }, 2),
    ]
    const v = assembleChampionVerdict({ championGates: [{ metric: 'return_vs_hold_pct', op: '>', value: 0 }] }, ctxOf(runs))
    expect(v.steady).toBe(false)
  })

  it('SKIPS a gate whose metric is absent (sentinel) — it never rejects, but cannot alone prove steady', () => {
    const runs = [cr('2024', 5, { return_vs_hold_pct: 6 }), cr('2022', 3, { return_vs_hold_pct: 4 })]
    // down_capture is not emitted here (the do-nothing sentinel) ⇒ that gate is skipped, not failed.
    const v = assembleChampionVerdict(
      {
        championGates: [
          { metric: 'return_vs_hold_pct', op: '>', value: 0 },
          { metric: 'down_capture', op: '<', value: 0.8 },
        ],
      },
      ctxOf(runs),
    )
    expect(v.championGates.map((g) => g.applicable)).toEqual([true, false])
    expect(v.steady).toBe(true) // the applicable gate passes; the skipped capture gate does not block
  })

  it('a champion with ONLY skipped gates is never declared steady (fail-closed)', () => {
    const runs = [cr('2024', 5, {}), cr('2022', 3, {})]
    const v = assembleChampionVerdict({ championGates: [{ metric: 'down_capture', op: '<', value: 0.8 }] }, ctxOf(runs))
    expect(v.championGates[0].applicable).toBe(false)
    expect(v.steady).toBe(false)
  })

  // --- DSR gate (family b): deflated Sharpe across the distinct setups tried ---
  const m = (sharpe: number, nObs = 500) => ({
    oos_sharpe: sharpe,
    oos_ret_skew: 0,
    oos_ret_kurt: 3,
    oos_n_obs: nObs,
  })
  const dr = (lr: number, window: string, obj: number, sharpe: number, nObs = 500): AnalysisRun => ({
    key: `${lr}-${window}`,
    config: { lr, window },
    objective: obj,
    metrics: m(sharpe, nObs),
    seed: 0,
    status: 'completed',
  })

  it('DSR gate passes when the champion Sharpe clears the multiple-testing deflation', () => {
    // Incumbent {lr:1} (best objective) has a strong Sharpe 0.5; the only other setup sits at 0.1 ⇒ the
    // deflation level (2 trials) is ~0.15, well below 0.5 ⇒ DSR ≈ 1.
    const runs = [dr(1, '2024', 10, 0.5), dr(1, '2022', 10, 0.5), dr(2, '2024', 1, 0.1), dr(2, '2022', 1, 0.1)]
    const v = assembleChampionVerdict({ dsr: { threshold: 0.95 } }, { runs, splitLevers: ['window'], criterion: crit })
    const dsr = v.championGates.find((g) => g.kind === 'dsr')!
    expect(dsr.applicable).toBe(true)
    expect(dsr.pass).toBe(true)
    expect(v.steady).toBe(true)
  })

  it('DSR gate fails when the champion Sharpe sits below the deflation level (a multiple-testing artifact)', () => {
    // Incumbent {lr:1} wins on OBJECTIVE but its Sharpe (0.1) is under the deflation level set by the sweep.
    const runs = [dr(1, '2024', 10, 0.1, 30), dr(1, '2022', 10, 0.1, 30), dr(2, '2024', 1, 0.6), dr(2, '2022', 1, 0.6)]
    const v = assembleChampionVerdict({ dsr: { threshold: 0.95 } }, { runs, splitLevers: ['window'], criterion: crit })
    expect(v.championGates.find((g) => g.kind === 'dsr')!.pass).toBe(false)
    expect(v.steady).toBe(false)
  })

  it('DSR gate deflates by the HONEST trial count, not just the setups in front of it', () => {
    // The setups in the current comparison are NOT the search that produced the champion. A campaign that
    // searched 20,888 configs and then compares two of them would otherwise be deflated as though it had
    // tried two — the lenient direction, and exactly how a multiple-testing artifact gets certified. The
    // override raises nTrials, which raises the deflation level SR*, which can only lower DSR.
    const runs = [dr(1, '2024', 10, 0.5), dr(1, '2022', 10, 0.5), dr(2, '2024', 1, 0.1), dr(2, '2022', 1, 0.1)]
    const ctx = { runs, splitLevers: ['window'], criterion: crit }
    const bare = assembleChampionVerdict({ dsr: { threshold: 0.95 } }, ctx)
    expect(bare.championGates.find((g) => g.kind === 'dsr')!.pass).toBe(true)

    const honest = assembleChampionVerdict({ dsr: { threshold: 0.95, nTrials: 20888 } }, ctx)
    const gate = honest.championGates.find((g) => g.kind === 'dsr')!
    expect(gate.applicable).toBe(true)
    expect(gate.pass).toBe(false)
    expect(gate.detail).toContain('20888')
  })

  it('the DSR trial-count override is a FLOOR — it can never deflate by fewer trials than were compared', () => {
    // A gate override must only ever make the gate stricter. Honouring a value below the observed setup
    // count would let a misconfigured manifest weaken the correction, which is the one direction a gate
    // must never move.
    const runs = [dr(1, '2024', 10, 0.5), dr(1, '2022', 10, 0.5), dr(2, '2024', 1, 0.1), dr(2, '2022', 1, 0.1)]
    const ctx = { runs, splitLevers: ['window'], criterion: crit }
    for (const nTrials of [0, 1, -5, Number.NaN]) {
      const g = assembleChampionVerdict({ dsr: { threshold: 0.95, nTrials } }, ctx).championGates.find(
        (x) => x.kind === 'dsr',
      )!
      expect(g.detail).toContain('2 setup(s)')
    }
  })

  it('DSR gate is skipped when the champion emits no Sharpe metric', () => {
    const v = assembleChampionVerdict({ dsr: { threshold: 0.95 } }, ctxOf([cr('2024', 5, {}), cr('2022', 3, {})]))
    const dsr = v.championGates.find((g) => g.kind === 'dsr')!
    expect(dsr.applicable).toBe(false)
    expect(v.steady).toBe(false)
  })

  // --- seed-stability gate (family d) ---
  const sr = (seed: number, obj: number): AnalysisRun => ({
    key: `s${seed}`,
    config: { lr: 1, window: '2024' },
    objective: obj,
    seed,
    status: 'completed',
  })

  it('seed-stability passes for a tight champion and fails for a seed-swinging one', () => {
    const tight = assembleChampionVerdict(
      { stability: { maxCiWidth: 2 } },
      { runs: [sr(0, 10), sr(1, 10), sr(2, 10), sr(3, 10)], splitLevers: ['window'], criterion: crit },
    )
    expect(tight.championGates.find((g) => g.kind === 'seed-stability')!.pass).toBe(true)
    expect(tight.steady).toBe(true)

    const wide = assembleChampionVerdict(
      { stability: { maxCiWidth: 2 } },
      { runs: [sr(0, 0), sr(1, 20), sr(2, 0), sr(3, 20)], splitLevers: ['window'], criterion: crit },
    )
    expect(wide.championGates.find((g) => g.kind === 'seed-stability')!.pass).toBe(false)
    expect(wide.steady).toBe(false)
  })

  it('every declared family is ANDed — one failing family sinks steady even if the others pass', () => {
    // champion median beats hold on both windows AND a tight seed CI, but the DSR is a lucky-draw artifact.
    const runs = [
      dr(1, '2024', 10, 0.1, 30),
      dr(1, '2022', 10, 0.1, 30),
      dr(2, '2024', 1, 0.6),
      dr(2, '2022', 1, 0.6),
    ].map((r) => ({ ...r, metrics: { ...r.metrics, return_vs_hold_pct: 5 } }))
    const v = assembleChampionVerdict(
      {
        championGates: [{ metric: 'return_vs_hold_pct', op: '>', value: 0 }],
        dsr: { threshold: 0.95 },
        stability: { maxCiWidth: 5 },
      },
      { runs, splitLevers: ['window'], criterion: crit },
    )
    const kinds = v.championGates.map((g) => g.kind)
    expect(kinds).toEqual(['cohort-median', 'dsr', 'seed-stability'])
    expect(v.championGates.find((g) => g.kind === 'cohort-median')!.pass).toBe(true)
    expect(v.championGates.find((g) => g.kind === 'dsr')!.pass).toBe(false)
    expect(v.steady).toBe(false) // the AND of all applicable families
  })

  it('groups all runs into ONE window when no split lever is declared', () => {
    const runs: AnalysisRun[] = [
      { key: 'a', config: { lr: 1 }, objective: 5, metrics: { return_vs_hold_pct: 6 }, seed: 0, status: 'completed' },
      { key: 'b', config: { lr: 1 }, objective: 5, metrics: { return_vs_hold_pct: 4 }, seed: 1, status: 'completed' },
    ]
    const v = assembleChampionVerdict(
      { championGates: [{ metric: 'return_vs_hold_pct', op: '>', value: 0 }] },
      { runs, splitLevers: [], criterion: crit },
    )
    // No split axis -> the whole cohort is one window ('·'); median([6,4]) = 5 > 0.
    expect(v.championGates[0]).toMatchObject({ applicable: true, pass: true, detail: 'holds on all 1 window(s)' })
    expect(v.steady).toBe(true)
  })

  it('supports a RATIO gate — the metric is compared against another metric median, not a literal', () => {
    const runs: AnalysisRun[] = [
      { key: 'a', config: { lr: 1, window: '2024' }, objective: 5, metrics: { up_capture: 0.9, down_capture: 0.5 }, seed: 0, status: 'completed' },
      { key: 'b', config: { lr: 1, window: '2024' }, objective: 5, metrics: { up_capture: 0.8, down_capture: 0.4 }, seed: 1, status: 'completed' },
    ]
    const v = assembleChampionVerdict(
      { championGates: [{ metric: 'up_capture', op: '>', value: { metric: 'down_capture' } }] },
      { runs, splitLevers: ['window'], criterion: crit },
    )
    const g = v.championGates[0]
    expect(g.label).toBe('up_capture > down_capture') // rendered from the referenced-metric name
    expect(g.applicable).toBe(true)
    // up_capture median 0.85 vs the resolved down_capture-median bound 0.45.
    expect(g.pass).toBe(true)
  })

  it('metricOf returns undefined for a run missing the objective (non-number branch)', () => {
    const scoreCrit: AnalysisCriterion = { key: 'score', direction: 'max' }
    const runs: AnalysisRun[] = [
      { key: 'a', config: { lr: 1, window: '2024' }, objective: 5, metrics: { score: 1 }, seed: 0, status: 'completed' },
      { key: 'b', config: { lr: 1, window: '2024' }, metrics: { score: 1 }, seed: 1, status: 'completed' }, // no objective
    ]
    const v = assembleChampionVerdict(
      { championGates: [{ metric: 'objective', op: '>', value: 0 }] },
      { runs, splitLevers: ['window'], criterion: scoreCrit },
    )
    const g = v.championGates[0]
    // run 'b' drops out via metricOf's undefined branch; median([5]) = 5 > 0.
    expect(g.applicable).toBe(true)
    expect(g.pass).toBe(true)
    expect(g.detail).toMatch(/holds on all 1 window\(s\)/)
  })

  it('defaults the DSR threshold to 0.95 when unset', () => {
    const runs = [dr(1, '2024', 10, 0.5), dr(1, '2022', 10, 0.5), dr(2, '2024', 1, 0.1), dr(2, '2022', 1, 0.1)]
    const v = assembleChampionVerdict({ dsr: {} }, { runs, splitLevers: ['window'], criterion: crit })
    const dsr = v.championGates.find((g) => g.kind === 'dsr')!
    expect(dsr.label).toBe('deflated Sharpe ≥ 0.95')
    expect(dsr.applicable).toBe(true)
  })

  it('skips completed runs that lack the Sharpe metric when counting trials', () => {
    const runs: AnalysisRun[] = [
      dr(1, '2024', 10, 0.5),
      dr(1, '2022', 10, 0.5),
      dr(2, '2024', 1, 0.1),
      dr(2, '2022', 1, 0.1),
      { key: 'no-sharpe', config: { lr: 3, window: '2024' }, objective: 0, metrics: { return_vs_hold_pct: 1 }, seed: 0, status: 'completed' },
    ]
    const v = assembleChampionVerdict({ dsr: { threshold: 0.95 } }, { runs, splitLevers: ['window'], criterion: crit })
    const dsr = v.championGates.find((g) => g.kind === 'dsr')!
    expect(dsr.applicable).toBe(true)
    // the Sharpe-less setup {lr:3} is not counted as a trial: n_trials stays 2 (lr1, lr2).
    expect(dsr.detail).toMatch(/over 2 setup\(s\)/)
  })

  it('defaults skew=0 / kurt=3 / n_obs=0 when the champion emits only Sharpe (DSR ⇒ 0 on n_obs<2)', () => {
    const bare = (lr: number, window: string, obj: number, sharpe: number): AnalysisRun => ({
      key: `${lr}-${window}`,
      config: { lr, window },
      objective: obj,
      metrics: { oos_sharpe: sharpe },
      seed: 0,
      status: 'completed',
    })
    const runs = [bare(1, '2024', 10, 5), bare(1, '2022', 10, 5), bare(2, '2024', 1, 0.1), bare(2, '2022', 1, 0.1)]
    const v = assembleChampionVerdict({ dsr: { threshold: 0.95 } }, { runs, splitLevers: ['window'], criterion: crit })
    const dsr = v.championGates.find((g) => g.kind === 'dsr')!
    expect(dsr.applicable).toBe(true)
    // n_obs defaults to 0 -> psr has n<2 -> DSR 0.000 -> under the 0.95 threshold.
    expect(dsr.detail).toMatch(/DSR 0\.000/)
    expect(dsr.pass).toBe(false)
  })

  it('fails the DSR gate when n_obs is below dsr.minObs (obsOk false ⇒ detail cites n_obs)', () => {
    const runs = [dr(1, '2024', 10, 0.5, 300), dr(1, '2022', 10, 0.5, 300), dr(2, '2024', 1, 0.1, 300), dr(2, '2022', 1, 0.1, 300)]
    const v = assembleChampionVerdict({ dsr: { threshold: 0.95, minObs: 1000 } }, { runs, splitLevers: ['window'], criterion: crit })
    const dsr = v.championGates.find((g) => g.kind === 'dsr')!
    expect(dsr.pass).toBe(false)
    expect(dsr.detail).toMatch(/n_obs 300 < 1000/)
  })

  it('honors dsr.minObs when satisfied (no n_obs suffix on the detail)', () => {
    const runs = [dr(1, '2024', 10, 0.5, 2000), dr(1, '2022', 10, 0.5, 2000), dr(2, '2024', 1, 0.1, 2000), dr(2, '2022', 1, 0.1, 2000)]
    const v = assembleChampionVerdict({ dsr: { threshold: 0.95, minObs: 1000 } }, { runs, splitLevers: ['window'], criterion: crit })
    const dsr = v.championGates.find((g) => g.kind === 'dsr')!
    expect(dsr.detail).not.toMatch(/n_obs/)
  })

  it('SKIPS the seed-stability gate with fewer than 2 evaluable seeds (fail-closed)', () => {
    const v = assembleChampionVerdict(
      { stability: { maxCiWidth: 2 } },
      { runs: [sr(0, 10)], splitLevers: ['window'], criterion: crit },
    )
    const g = v.championGates.find((x) => x.kind === 'seed-stability')!
    expect(g.applicable).toBe(false)
    expect(g.detail).toBe('need ≥ 2 seeds')
    expect(v.steady).toBe(false)
  })

  it('a RATIO gate FAILS a split where the referenced bound metric is absent (bound ⇒ NaN)', () => {
    const runs: AnalysisRun[] = [
      { key: 'a', config: { lr: 1, window: '2024' }, objective: 5, metrics: { up_capture: 0.9 }, seed: 0, status: 'completed' },
      { key: 'b', config: { lr: 1, window: '2024' }, objective: 5, metrics: { up_capture: 0.8 }, seed: 1, status: 'completed' },
    ]
    const v = assembleChampionVerdict(
      { championGates: [{ metric: 'up_capture', op: '>', value: { metric: 'down_capture' } }] },
      { runs, splitLevers: ['window'], criterion: crit },
    )
    const g = v.championGates[0]
    // up_capture is present (gate is applicable) but down_capture is absent ⇒ bound NaN ⇒ 0.85 > NaN is false.
    expect(g.applicable).toBe(true)
    expect(g.pass).toBe(false)
  })
})
