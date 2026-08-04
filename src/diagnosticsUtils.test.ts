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
})
