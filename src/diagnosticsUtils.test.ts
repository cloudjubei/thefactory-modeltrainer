import { describe, it, expect } from 'vitest'

import {
  incumbentSplitHoldout,
  convergenceGatedBySplits,
  splitLeversOf,
  narrateSplitHoldout,
  rewardFitnessAlignment,
  narrateAlignment,
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
