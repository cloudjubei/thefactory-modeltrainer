import { describe, it, expect } from 'vitest'

import {
  incumbentSplitHoldout,
  convergenceGatedBySplits,
  splitLeversOf,
  narrateSplitHoldout,
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
