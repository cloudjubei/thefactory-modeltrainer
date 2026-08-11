import { describe, expect, it } from 'vitest'
import type { AnalysisCriterion, AnalysisRun, ExperimentCell } from './modelTrainerTypes.js'
import {
  ablationPath,
  aggregateExperimentCells,
  aggregateRunValues,
  aggregateToSetupRuns,
  benjaminiHochberg,
  bootstrapDiff,
  pairedBootstrapDiff,
  computeConfigSpaceAnalysis,
  paretoFrontier,
  criterionValueOf,
  expectedImprovement,
  fanovaImportances,
  fitConfigSurrogate,
  interactionGrid,
  iqm,
  leverCouplings,
  leverImportances,
  medianOf,
  normalizeByEnvironment,
  normalizeConditionalLevers,
  pcaProjection,
  ofatContrasts,
  predictConfig,
  predictConfigStats,
  recommendExperiments,
} from './xaiUtils.js'

const MAX: AnalysisCriterion = { key: 'objective', direction: 'max' }
const DS = { asset: 'BTC', timeframe: '1h', candles: 100, from: 'a', to: 'b' }

function run(
  key: string,
  config: Record<string, unknown>,
  objective: number,
  opts: Partial<AnalysisRun> = {},
): AnalysisRun {
  return {
    key,
    config,
    objective,
    status: 'completed',
    dataset: DS,
    seed: 0,
    ...opts,
  }
}

describe('iqm', () => {
  it('trims the top and bottom 25% and means the middle (robust to outliers)', () => {
    expect(iqm([1, 2, 3, 4, 100])).toBe(3)
  })

  it('is the plain mean when there are too few values to trim', () => {
    expect(iqm([10, 20])).toBe(15)
  })

  it('returns 0 for an empty sample', () => {
    expect(iqm([])).toBe(0)
  })
})

describe('aggregateRunValues', () => {
  it('reports n/mean/iqm/median/min/max and a bracketing CI', () => {
    const agg = aggregateRunValues([10, 12, 14, 16, 18])
    expect(agg.n).toBe(5)
    expect(agg.mean).toBe(14)
    expect(agg.median).toBe(14)
    expect(agg.min).toBe(10)
    expect(agg.max).toBe(18)
    expect(agg.ci[0]).toBeLessThanOrEqual(agg.iqm)
    expect(agg.ci[1]).toBeGreaterThanOrEqual(agg.iqm)
  })

  it('is deterministic — same input gives the identical bootstrap CI', () => {
    const a = aggregateRunValues([5, 7, 9, 11, 13, 15])
    const b = aggregateRunValues([5, 7, 9, 11, 13, 15])
    expect(a.ci).toEqual(b.ci)
  })

  it('handles a single value (degenerate CI)', () => {
    const agg = aggregateRunValues([42])
    expect(agg.iqm).toBe(42)
    expect(agg.ci).toEqual([42, 42])
  })
})

describe('aggregateExperimentCells', () => {
  const cell = (
    key: string,
    opts: { objective?: number; vh?: number; status?: 'completed' | 'failed' } = {},
  ): ExperimentCell => ({
    key,
    config: {},
    status: opts.status || 'completed',
    ...(opts.objective === undefined ? {} : { objective: opts.objective }),
    ...(opts.vh === undefined ? {} : { metrics: { return_vs_hold_pct: opts.vh } }),
  })
  const ctx = { direction: 'max' as const, assessedAt: '2026-08-06T00:00:00Z' }

  it('returns unverifiable with a null aggregate when there are no cells', () => {
    const { aggregate, verdict } = aggregateExperimentCells([], ctx)
    expect(aggregate).toBeNull()
    expect(verdict.kind).toBe('unverifiable')
    expect(verdict.passed).toBe(false)
    expect(verdict.source).toBe('auto')
    expect(verdict.assessedAt).toBe('2026-08-06T00:00:00Z')
  })

  it('excludes failed cells from the aggregate and the benchmark read', () => {
    const cells = [cell('a', { objective: 10, vh: 5 }), cell('b', { status: 'failed', vh: 99 })]
    const { aggregate, verdict } = aggregateExperimentCells(cells, ctx)
    expect(aggregate?.n).toBe(1)
    // one completed cell that clears → looked good on a single split, not yet robust
    expect(verdict.kind).toBe('single-split-luck')
    expect(verdict.passed).toBe(false)
  })

  it('is unverifiable when completed cells report no benchmark metric (even with objectives)', () => {
    const cells = [cell('a', { objective: 10 }), cell('b', { objective: 12 })]
    const { aggregate, verdict } = aggregateExperimentCells(cells, ctx)
    expect(aggregate?.n).toBe(2)
    expect(aggregate?.mean).toBe(11)
    expect(verdict.kind).toBe('unverifiable')
    expect(verdict.passed).toBe(false)
  })

  it('is single-split-luck when the ONLY benchmark cell clears the bar', () => {
    const { verdict } = aggregateExperimentCells([cell('a', { objective: 1, vh: 3 })], ctx)
    expect(verdict.kind).toBe('single-split-luck')
    expect(verdict.passed).toBe(false)
  })

  it('is not-replicated when the only benchmark cell misses the bar', () => {
    const { verdict } = aggregateExperimentCells([cell('a', { objective: 1, vh: -3 })], ctx)
    expect(verdict.kind).toBe('not-replicated')
    expect(verdict.passed).toBe(false)
  })

  it('is ROBUST (passed) when EVERY benchmark cell clears across ≥2 splits', () => {
    const cells = [cell('a', { objective: 2, vh: 1 }), cell('b', { objective: 3, vh: 4 })]
    const { verdict } = aggregateExperimentCells(cells, ctx)
    expect(verdict.kind).toBe('robust')
    expect(verdict.passed).toBe(true)
    expect(verdict.rationale).toContain('2/2')
  })

  it('is not-replicated when a single split fails (robustness demands ALL clear)', () => {
    const cells = [cell('a', { vh: 1 }), cell('b', { vh: 2 }), cell('c', { vh: -1 })]
    const { verdict } = aggregateExperimentCells(cells, ctx)
    expect(verdict.kind).toBe('not-replicated')
    expect(verdict.passed).toBe(false)
  })

  it('is not-replicated when no cell clears across multiple splits', () => {
    const cells = [cell('a', { vh: -1 }), cell('b', { vh: -2 })]
    const { verdict } = aggregateExperimentCells(cells, ctx)
    expect(verdict.kind).toBe('not-replicated')
    expect(verdict.passed).toBe(false)
  })

  it('honours a min-direction benchmark (clears when the metric is BELOW the threshold)', () => {
    const benchmark = { metric: 'max_drawdown_pct', threshold: -5, direction: 'min' as const }
    const cells: ExperimentCell[] = [
      { key: 'a', config: {}, status: 'completed', objective: 1, metrics: { max_drawdown_pct: -8 } },
      { key: 'b', config: {}, status: 'completed', objective: 1, metrics: { max_drawdown_pct: -6 } },
    ]
    const { verdict } = aggregateExperimentCells(cells, { ...ctx, benchmark })
    expect(verdict.kind).toBe('robust')
    expect(verdict.passed).toBe(true)
  })

  it('respects a tunable minSplits and never labels a multi-split cohort single-split-luck', () => {
    // Under minSplits=3, two all-clearing cells are NOT enough to be robust, but they are TWO splits — so
    // the sub-threshold verdict must be not-replicated, never the (single-split) luck label.
    const cells = [cell('a', { objective: 1, vh: 3 }), cell('b', { objective: 2, vh: 4 })]
    const v = aggregateExperimentCells(cells, { ...ctx, minSplits: 3 }).verdict
    expect(v.kind).toBe('not-replicated')
    expect(v.passed).toBe(false)
    // A genuine single split still reads as single-split-luck.
    expect(aggregateExperimentCells([cell('a', { vh: 1 })], { ...ctx, minSplits: 3 }).verdict.kind).toBe(
      'single-split-luck',
    )
  })

  it('aggregates objectives per the objective DIRECTION independent of the benchmark metric', () => {
    const cells = [
      cell('a', { objective: 10, vh: 1 }),
      cell('b', { objective: 20, vh: 1 }),
      cell('c', { objective: 30, vh: 1 }),
    ]
    const { aggregate } = aggregateExperimentCells(cells, ctx)
    expect(aggregate?.n).toBe(3)
    expect(aggregate?.min).toBe(10)
    expect(aggregate?.max).toBe(30)
    expect(aggregate?.median).toBe(20)
  })
})

describe('criterionValueOf', () => {
  it('reads the objective', () => {
    expect(criterionValueOf(run('a', {}, 3.5), MAX)).toBe(3.5)
  })

  it('reads a metrics key', () => {
    const r = run('a', {}, 0, { metrics: { win_pct: 55 } })
    expect(criterionValueOf(r, { key: 'win_pct', direction: 'max' })).toBe(55)
  })

  it('reads durationMs for the runtime criterion', () => {
    const r = run('a', {}, 0, { durationMs: 1234 })
    expect(criterionValueOf(r, { key: 'durationMs', direction: 'min' })).toBe(1234)
  })

  it('returns undefined for a missing/non-numeric metric', () => {
    expect(
      criterionValueOf(run('a', {}, 0, { metrics: {} }), { key: 'ghost', direction: 'max' }),
    ).toBeUndefined()
  })
})

describe('ofatContrasts', () => {
  const fourRuns = [
    run('a', { lr: 0.1, batch_size: 64 }, 10, { seed: 0 }),
    run('b', { lr: 0.1, batch_size: 64 }, 12, { seed: 1 }),
    run('c', { lr: 0.1, batch_size: 128 }, 20, { seed: 0 }),
    run('d', { lr: 0.1, batch_size: 128 }, 22, { seed: 1 }),
  ]

  it('contrasts runs that differ ONLY by the chosen lever, holding all else fixed', () => {
    const contrasts = ofatContrasts(fourRuns, 'batch_size', MAX)
    expect(contrasts).toHaveLength(1)
    const c = contrasts[0]
    expect(c.lever).toBe('batch_size')
    expect(c.levels.map((l) => l.value)).toEqual(['128', '64']) // best-first for max
    expect(c.levels[0].seeds).toBe(2)
    expect(c.levels[0].aggregate.iqm).toBeCloseTo(21)
    expect(c.levels[1].aggregate.iqm).toBeCloseTo(11)
  })

  it('orients the effect delta so positive is always BETTER (max criterion)', () => {
    const c = ofatContrasts(fourRuns, 'batch_size', MAX)[0]
    // baseline is the worst level (64); the 128 effect is +10
    expect(c.effects[0].from).toBe('64')
    expect(c.effects[0].to).toBe('128')
    expect(c.effects[0].delta).toBeCloseTo(10)
  })

  it('orients delta for a MIN criterion (lower is better)', () => {
    const runtimeRuns = [
      run('a', { lr: 0.1, batch_size: 64 }, 0, { seed: 0, durationMs: 100 }),
      run('b', { lr: 0.1, batch_size: 64 }, 0, { seed: 1, durationMs: 110 }),
      run('c', { lr: 0.1, batch_size: 128 }, 0, { seed: 0, durationMs: 50 }),
      run('d', { lr: 0.1, batch_size: 128 }, 0, { seed: 1, durationMs: 60 }),
    ]
    const c = ofatContrasts(runtimeRuns, 'batch_size', { key: 'durationMs', direction: 'min' })[0]
    // 128 (≈55ms) is better than 64 (≈105ms) → positive improvement of ≈50
    expect(c.levels[0].value).toBe('128')
    expect(c.effects[0].delta).toBeCloseTo(50)
  })

  it('excludes runs that differ in more than the chosen lever (no clean contrast)', () => {
    const confounded = [
      run('a', { lr: 0.1, batch_size: 64 }, 10),
      run('b', { lr: 0.2, batch_size: 128 }, 20),
    ]
    expect(ofatContrasts(confounded, 'batch_size', MAX)).toEqual([])
  })

  it('separates contrasts by their held-fixed context', () => {
    const twoContexts = [
      run('a', { lr: 0.1, batch_size: 64 }, 10),
      run('b', { lr: 0.1, batch_size: 128 }, 20),
      run('c', { lr: 0.2, batch_size: 64 }, 5),
      run('d', { lr: 0.2, batch_size: 128 }, 8),
    ]
    const contrasts = ofatContrasts(twoContexts, 'batch_size', MAX)
    expect(contrasts).toHaveLength(2)
    expect(new Set(contrasts.map((c) => c.controlSignature)).size).toBe(2)
  })

  it('ignores non-completed runs and runs missing the criterion value', () => {
    const mixed = [
      ...fourRuns,
      run('e', { lr: 0.1, batch_size: 256 }, 0, { status: 'failed' }),
      run('f', { lr: 0.1, batch_size: 512 }, Number.NaN as unknown as number),
    ]
    const c = ofatContrasts(mixed, 'batch_size', MAX)[0]
    expect(c.levels.map((l) => l.value).sort()).toEqual(['128', '64'])
  })

  it('reports a degenerate interval and never "significant" when each level has one seed', () => {
    const single = [
      run('a', { lr: 0.1, batch_size: 64 }, 10, { seed: 0 }),
      run('b', { lr: 0.1, batch_size: 128 }, 20, { seed: 0 }),
    ]
    const c = ofatContrasts(single, 'batch_size', MAX)[0]
    expect(c.levels[0].aggregate.n).toBe(1)
    expect(c.effects[0].delta).toBeCloseTo(10)
    expect(c.effects[0].diffCi).toEqual([10, 10])
    expect(c.effects[0].significant).toBe(false) // one seed each → can't assess variance
  })

  it('flags a significant effect when the difference CI excludes zero', () => {
    const wide = [
      run('a', { lr: 0.1, batch_size: 64 }, 1, { seed: 0 }),
      run('b', { lr: 0.1, batch_size: 64 }, 1.1, { seed: 1 }),
      run('c', { lr: 0.1, batch_size: 64 }, 0.9, { seed: 2 }),
      run('d', { lr: 0.1, batch_size: 64 }, 1, { seed: 3 }),
      run('e', { lr: 0.1, batch_size: 64 }, 1, { seed: 4 }),
      run('f', { lr: 0.1, batch_size: 128 }, 50, { seed: 0 }),
      run('g', { lr: 0.1, batch_size: 128 }, 51, { seed: 1 }),
      run('h', { lr: 0.1, batch_size: 128 }, 49, { seed: 2 }),
      run('i', { lr: 0.1, batch_size: 128 }, 50, { seed: 3 }),
      run('j', { lr: 0.1, batch_size: 128 }, 50, { seed: 4 }),
    ]
    const c = ofatContrasts(wide, 'batch_size', MAX)[0]
    expect(c.effects[0].significant).toBe(true)
    expect(c.effects[0].diffCi[0]).toBeGreaterThan(0)
  })
})

describe('leverImportances', () => {
  it('ranks a lever that swings the outcome above one that does not', () => {
    const runs = [
      run('a', { lr: 0.1, gamma: 0.9 }, 10),
      run('b', { lr: 0.2, gamma: 0.9 }, 90),
      run('c', { lr: 0.1, gamma: 0.99 }, 11),
      run('d', { lr: 0.2, gamma: 0.99 }, 89),
    ]
    const imp = leverImportances(runs, MAX)
    expect(imp[0].lever).toBe('lr') // lr swings 10→90, gamma barely moves it
    expect(imp[0].importance).toBeGreaterThan(imp[1].importance)
    expect(imp[0].importance).toBeGreaterThanOrEqual(0)
    expect(imp[0].importance).toBeLessThanOrEqual(1)
  })

  it('skips levers with a single observed value', () => {
    const runs = [run('a', { lr: 0.1, fixed: 'x' }, 10), run('b', { lr: 0.2, fixed: 'x' }, 20)]
    expect(leverImportances(runs, MAX).map((i) => i.lever)).toEqual(['lr'])
  })

  it('flags low confidence + the weakest leg when a value has too few runs', () => {
    const runs = [run('a', { lr: 0.1 }, 10), run('b', { lr: 0.2 }, 20)]
    const imp = leverImportances(runs, MAX)[0]
    expect(imp.minRuns).toBe(1)
    expect(imp.confident).toBe(false)
  })

  it('is confident when every value has at least the min-seeds bar of runs', () => {
    const runs = []
    for (const lr of [0.1, 0.2])
      for (let s = 0; s < 5; s++)
        runs.push(run(`${lr}_${s}`, { lr }, 10 + lr * 100 + s, { seed: s }))
    const imp = leverImportances(runs, MAX)[0]
    expect(imp.minRuns).toBe(5)
    expect(imp.confident).toBe(true)
  })
})

describe('conditional levers — the "doesn\'t-apply" sentinel is excluded from importance', () => {
  // A forward_horizon-style lever: 'n/a' on the models it doesn't apply to, a real value on the ones it does.
  // The applicable (supervised) models are performance outliers but the lever has NO effect among them, so
  // it must NOT inherit their between-model-class variance — i.e. it is scored only where it applies.
  it('leverImportances scores a conditional lever only where it applies (not across the n/a boundary)', () => {
    const runs = [
      run('r1', { model_name: 'rl', forward_horizon: 'n/a' }, 100),
      run('r2', { model_name: 'rl', forward_horizon: 'n/a' }, 100),
      run('a1', { model_name: 'ars', forward_horizon: 'n/a' }, 100),
      run('a2', { model_name: 'ars', forward_horizon: 'n/a' }, 100),
      run('s1', { model_name: 'sup', forward_horizon: 1 }, 0),
      run('s2', { model_name: 'sup', forward_horizon: 5 }, 0),
    ]
    const imp = leverImportances(runs, MAX)
    const fh = imp.find((i) => i.lever === 'forward_horizon')!
    expect(fh).toBeDefined()
    expect(fh.importance).toBe(0) // no effect among supervised runs; NOT inflated by the n/a bucket
    expect(fh.minRuns).toBe(1) // counted over the applicable (supervised) runs only
    expect(imp[0].lever).toBe('model_name') // the real driver wins
  })

  it('ofatContrasts never forms a contrast level from the doesn\'t-apply sentinel', () => {
    const runs = [
      run('s1', { model_name: 'sup', forward_horizon: 1 }, 10),
      run('s2', { model_name: 'sup', forward_horizon: 5 }, 20),
      run('r1', { model_name: 'rl', forward_horizon: 'n/a' }, 99),
    ]
    const values = ofatContrasts(runs, 'forward_horizon', MAX).flatMap((c) =>
      c.levels.map((l) => l.value),
    )
    expect(values).not.toContain('n/a')
  })

  it('computeConfigSpaceAnalysis: a conditional lever that only applies to outlier models is not a top driver', () => {
    const runs: AnalysisRun[] = []
    for (let i = 0; i < 5; i++) {
      runs.push(run(`rl${i}`, { model_name: 'rl', lr: 0.1 }, 100, { seed: i }))
      runs.push(run(`ars${i}`, { model_name: 'ars', lr: 0.1 }, 100, { seed: i }))
    }
    for (let i = 0; i < 3; i++) {
      runs.push(run(`sa${i}`, { model_name: 'sup', lr: 0.1, forward_horizon: 1 }, 0, { seed: i }))
      runs.push(run(`sb${i}`, { model_name: 'sup', lr: 0.1, forward_horizon: 5 }, 0, { seed: i }))
    }
    const a = computeConfigSpaceAnalysis(runs, MAX, {
      appliesWhen: { forward_horizon: { model_name: ['sup'] } },
    })!
    const fh = a.screening.find((s) => s.lever === 'forward_horizon')!
    expect(fh).toBeDefined()
    expect(fh.importance).toBeLessThan(0.05) // not the top driver — model_name carries the real variance
    const fhFanova = a.importances.find((f) => f.lever === 'forward_horizon')
    if (fhFanova) expect(fhFanova.importance).toBeLessThan(0.05)
  })
})

describe('ignoreLevers (device is never a lever)', () => {
  it('strips ignored levers from the whole-space analysis', () => {
    const runs = [
      run('a', { lr: 0.1, device: 'cpu' }, 10),
      run('b', { lr: 0.2, device: 'mps' }, 90),
      run('c', { lr: 0.1, device: 'mps' }, 11),
      run('d', { lr: 0.2, device: 'cpu' }, 89),
    ]
    const a = computeConfigSpaceAnalysis(runs, MAX, { ignoreLevers: ['device'] })!
    expect(a.screening.some((s) => s.lever === 'device')).toBe(false)
    expect(a.levers).not.toContain('device')
    expect(a.setups.every((s) => !('device' in s.config))).toBe(true)
    expect(a.screening.some((s) => s.lever === 'lr')).toBe(true) // real levers survive
  })
})

describe('convergence (best-so-far)', () => {
  it('emits a time-ordered best-so-far series over the runs', () => {
    const runs = [
      run('c', { lr: 0.3 }, 20, { seed: 0, ranAt: '2026-01-03' }),
      run('a', { lr: 0.1 }, 10, { seed: 0, ranAt: '2026-01-01' }),
      run('b', { lr: 0.2 }, 30, { seed: 0, ranAt: '2026-01-02' }),
    ]
    const a = computeConfigSpaceAnalysis(runs, MAX)!
    expect(a.convergence.map((p) => p.best)).toEqual([10, 30, 30]) // sorted by time, running max
    expect(a.convergence.map((p) => p.index)).toEqual([1, 2, 3])
  })

  it('excludes untimestamped runs (they have no valid temporal position)', () => {
    const runs = [
      run('a', { lr: 0.1 }, 10, { seed: 0, ranAt: '2026-01-01' }),
      run('b', { lr: 0.2 }, 99, { seed: 0 }), // no ranAt → not part of the time-ordered series
      run('c', { lr: 0.3 }, 20, { seed: 0, ranAt: '2026-01-02' }),
    ]
    const a = computeConfigSpaceAnalysis(runs, MAX)!
    expect(a.convergence.map((p) => p.best)).toEqual([10, 20])
  })
})

describe('aggregateToSetupRuns', () => {
  it('records the distinct used seed numbers (sorted) on each setup', () => {
    const runs = [
      run('a', { lr: 0.1 }, 10, { seed: 2 }),
      run('b', { lr: 0.1 }, 11, { seed: 0 }),
      run('c', { lr: 0.1 }, 12, { seed: 0 }),
    ]
    expect(aggregateToSetupRuns(runs, MAX)[0].seedList).toEqual([0, 2])
  })

  it('folds seeds to one setup and retains the bootstrap CI + seed count', () => {
    const runs = []
    for (let s = 0; s < 5; s++) runs.push(run(`a${s}`, { lr: 0.1 }, 10 + s, { seed: s }))
    const setups = aggregateToSetupRuns(runs, MAX)
    expect(setups).toHaveLength(1)
    expect(setups[0].seeds).toBe(5)
    expect(Array.isArray(setups[0].ci)).toBe(true)
    // the IQM the criterion reads must sit within its own CI
    expect(setups[0].ci![0]).toBeLessThanOrEqual(setups[0].objective!)
    expect(setups[0].ci![1]).toBeGreaterThanOrEqual(setups[0].objective!)
  })
})

describe('paretoFrontier', () => {
  it('keeps only the point that dominates the rest (max/max)', () => {
    // [2,2] is >= every other on both axes and strictly better — it dominates them all.
    expect(
      paretoFrontier(
        [
          [1, 1],
          [2, 2],
          [2, 1],
          [1, 2],
        ],
        ['max', 'max'],
      ),
    ).toEqual([1])
  })

  it('handles mixed directions (return up, drawdown down) and drops dominated points', () => {
    // [return, drawdown]; idx3 [9,6] is dominated by idx0 [10,5] (>= return AND <= drawdown).
    expect(
      paretoFrontier(
        [
          [10, 5],
          [8, 2],
          [12, 8],
          [9, 6],
        ],
        ['max', 'min'],
      ),
    ).toEqual([0, 1, 2])
  })

  it('keeps tied (equal) points — neither dominates the other', () => {
    expect(
      paretoFrontier(
        [
          [5, 5],
          [5, 5],
        ],
        ['max', 'max'],
      ),
    ).toEqual([0, 1])
  })
})

describe('recommendExperiments', () => {
  it('recommends more seeds for a variance-thin top setup', () => {
    const runs = [run('a', { lr: 0.1, batch_size: 64 }, 100, { seed: 0 })]
    const recs = recommendExperiments(runs, MAX)
    const thin = recs.find((r) => r.kind === 'thin-seeds')
    expect(thin).toBeDefined()
    expect(thin!.spec.fixed).toMatchObject({ lr: 0.1, batch_size: 64 })
    expect((thin!.spec.seeds || []).length).toBeGreaterThan(0)
    expect(thin!.runCount).toBe((thin!.spec.seeds || []).length)
  })

  it('does not recommend more seeds for a setup that already has enough', () => {
    const runs = [0, 1, 2, 3, 4].map((s) =>
      run(`s${s}`, { lr: 0.1, batch_size: 64 }, 10 + s, { seed: s }),
    )
    expect(recommendExperiments(runs, MAX).find((r) => r.kind === 'thin-seeds')).toBeUndefined()
  })

  it('recommends filling a missing factorial cell', () => {
    const runs = [
      run('a', { lr: 0.1, batch_size: 64 }, 10, { seed: 0 }),
      run('b', { lr: 0.1, batch_size: 128 }, 20, { seed: 0 }),
      run('c', { lr: 0.2, batch_size: 64 }, 5, { seed: 0 }),
      // missing: lr=0.2, batch_size=128
    ]
    const recs = recommendExperiments(runs, MAX)
    // The untested cell is recommended — as `acquisition` when the surrogate rates it promising (it
    // supersedes the bare `missing-cell` via dedup), else as `missing-cell` for coverage.
    const rec = recs.find(
      (r) =>
        (r.spec.fixed as Record<string, unknown>)?.lr === 0.2 &&
        (r.spec.fixed as Record<string, unknown>)?.batch_size === 128,
    )
    expect(rec).toBeDefined()
    expect(['acquisition', 'missing-cell']).toContain(rec!.kind)
  })

  it('does not recommend a missing cell when the grid is complete', () => {
    const runs = [
      run('a', { lr: 0.1, batch_size: 64 }, 10, { seed: 0 }),
      run('b', { lr: 0.1, batch_size: 128 }, 20, { seed: 0 }),
      run('c', { lr: 0.2, batch_size: 64 }, 5, { seed: 0 }),
      run('d', { lr: 0.2, batch_size: 128 }, 8, { seed: 0 }),
    ]
    const recs = recommendExperiments(runs, MAX)
    expect(recs.find((r) => r.kind === 'missing-cell')).toBeUndefined()
  })

  it('returns recommendations ordered by descending priority', () => {
    const runs = [
      run('a', { lr: 0.1, batch_size: 64 }, 100, { seed: 0 }),
      run('b', { lr: 0.1, batch_size: 128 }, 20, { seed: 0 }),
      run('c', { lr: 0.2, batch_size: 64 }, 5, { seed: 0 }),
    ]
    const recs = recommendExperiments(runs, MAX)
    for (let i = 1; i < recs.length; i++) {
      expect(recs[i - 1].priority).toBeGreaterThanOrEqual(recs[i].priority)
    }
  })

  it('skips pair-contexts where a lever does not vary and runs missing a lever', () => {
    const runs = [
      run('a', { lr: 0.1, gamma: 0.9, batch_size: 64 }, 10, { seed: 0 }),
      run('b', { lr: 0.2, gamma: 0.9, batch_size: 64 }, 11, { seed: 0 }),
      run('c', { lr: 0.1, gamma: 0.99, batch_size: 128 }, 12, { seed: 0 }),
      run('d', { lr: 0.1, gamma: 0.9, batch_size: 128 }, 13, { seed: 0 }),
      run('e', { lr: 0.3, batch_size: 64 }, 9, { seed: 0 }), // no gamma → skipped for (gamma,*) pairs
    ]
    // Exercises the "context where a lever doesn't vary" and "run missing a pair lever" guards.
    expect(Array.isArray(recommendExperiments(runs, MAX))).toBe(true)
  })

  it('returns nothing for an empty run set', () => {
    expect(recommendExperiments([], MAX)).toEqual([])
  })
})

describe('acquisition (Phase 2)', () => {
  describe('expectedImprovement', () => {
    it('collapses to the raw oriented gain when std is 0', () => {
      expect(expectedImprovement(15, 0, 10, 'max')).toBe(5)
      expect(expectedImprovement(5, 0, 10, 'max')).toBe(0) // no improvement
      expect(expectedImprovement(5, 0, 10, 'min')).toBe(5) // lower is better
      expect(expectedImprovement(15, 0, 10, 'min')).toBe(0)
    })

    it('is positive at the incumbent due to uncertainty (φ(0)·std), and grows with std', () => {
      const atBest = expectedImprovement(10, 2, 10, 'max')
      expect(atBest).toBeCloseTo(2 * 0.39894, 3) // std · φ(0)
      expect(expectedImprovement(8, 5, 10, 'max')).toBeGreaterThan(0) // below best but uncertain
      expect(expectedImprovement(10, 4, 10, 'max')).toBeGreaterThan(atBest) // more uncertainty → more EI
    })
  })

  describe('predictConfigStats', () => {
    const grid = [0.1, 0.2, 0.3].flatMap((lr) =>
      [0, 1].map((s) => run(`${lr}-${s}`, { lr }, lr * 100, { seed: s })),
    )

    it('returns the forest mean (matching predictConfig) plus a non-negative std', () => {
      const s = fitConfigSurrogate(grid, MAX)
      const stats = predictConfigStats(s, { lr: 0.2 })
      expect(stats.mean).toBeCloseTo(predictConfig(s, { lr: 0.2 }), 10)
      expect(stats.std).toBeGreaterThanOrEqual(0)
    })

    it('reports zero uncertainty for an empty (untrained) surrogate', () => {
      const s = fitConfigSurrogate([run('a', { lr: 0.1 }, 10)], MAX) // <2 rows → no trees
      expect(predictConfigStats(s, { lr: 0.1 })).toEqual({ mean: s.mean, std: 0 })
    })
  })

  describe('acquisitionRecommendations (via recommendExperiments)', () => {
    // lr=0.2 clearly best; the (0.2,128) corner is untested and should look promising to the surrogate.
    const runs = [
      run('a', { lr: 0.1, batch_size: 64 }, 10, { seed: 0 }),
      run('b', { lr: 0.1, batch_size: 128 }, 12, { seed: 0 }),
      run('c', { lr: 0.2, batch_size: 64 }, 30, { seed: 0 }),
    ]

    it('surfaces an acquisition rec for the strongest unrun config, ranked above missing-cell coverage', () => {
      const recs = recommendExperiments(runs, MAX)
      const acq = recs.find((r) => r.kind === 'acquisition')
      expect(acq).toBeDefined()
      expect(acq!.spec.fixed).toMatchObject({ lr: 0.2, batch_size: 128 }) // the untested corner
      expect(acq!.reason).toMatch(/expected improvement/i)
      const cell = recs.find((r) => r.kind === 'missing-cell')
      if (cell) expect(acq!.priority).toBeGreaterThan(cell.priority)
    })

    it('measures expected improvement against the BEST observed value, not the worst', () => {
      // runs span 10..30; the incumbent for a MAX criterion is 30. The reason must cite that, not the min.
      const acq = recommendExperiments(runs, MAX).find((r) => r.kind === 'acquisition')!
      expect(acq.reason).toMatch(/best so far 30\b/)
    })

    it('never recommends an already-run config', () => {
      const acqs = recommendExperiments(runs, MAX).filter((r) => r.kind === 'acquisition')
      const ran = new Set(runs.map((r) => `${r.config.lr}|${r.config.batch_size}`))
      for (const a of acqs) {
        const f = a.spec.fixed as Record<string, unknown>
        expect(ran.has(`${f.lr}|${f.batch_size}`)).toBe(false)
      }
    })

    it('is deterministic — identical runs give identical recommendations', () => {
      expect(recommendExperiments(runs, MAX)).toEqual(recommendExperiments([...runs], MAX))
    })

    it('returns no acquisition recs when the grid is fully explored', () => {
      const full = [
        run('a', { lr: 0.1, batch_size: 64 }, 10, { seed: 0 }),
        run('b', { lr: 0.1, batch_size: 128 }, 12, { seed: 0 }),
        run('c', { lr: 0.2, batch_size: 64 }, 30, { seed: 0 }),
        run('d', { lr: 0.2, batch_size: 128 }, 28, { seed: 0 }),
      ]
      expect(recommendExperiments(full, MAX).find((r) => r.kind === 'acquisition')).toBeUndefined()
    })
  })
})

describe('config surrogate (Phase 3)', () => {
  // A grid where lr drives the objective strongly and gamma barely moves it, replicated across seeds.
  const grid: AnalysisRun[] = []
  for (const lr of [0.1, 0.2, 0.3]) {
    for (const gamma of [0.9, 0.99]) {
      for (let s = 0; s < 3; s++) {
        grid.push(
          run(`${lr}_${gamma}_${s}`, { lr, gamma }, lr * 100 + gamma + s * 0.5, { seed: s }),
        )
      }
    }
  }

  it('fits deterministically — same runs give the same prediction', () => {
    const a = fitConfigSurrogate(grid, MAX)
    const b = fitConfigSurrogate(grid, MAX)
    const cfg = { lr: 0.2, gamma: 0.9 }
    expect(predictConfig(a, cfg)).toBe(predictConfig(b, cfg))
  })

  it('learns the monotone lr relationship (higher lr ⇒ higher prediction)', () => {
    const s = fitConfigSurrogate(grid, MAX)
    expect(predictConfig(s, { lr: 0.3, gamma: 0.9 })).toBeGreaterThan(
      predictConfig(s, { lr: 0.1, gamma: 0.9 }),
    )
  })

  it('handles categorical (string-valued) levers', () => {
    const runs: AnalysisRun[] = []
    for (const algo of ['ppo', 'dqn']) {
      for (let s = 0; s < 4; s++) {
        runs.push(
          run(`${algo}_${s}`, { algo, lr: 0.1 }, algo === 'ppo' ? 90 + s : 10 + s, { seed: s }),
        )
      }
    }
    const sur = fitConfigSurrogate(runs, MAX)
    expect(predictConfig(sur, { algo: 'ppo', lr: 0.1 })).toBeGreaterThan(
      predictConfig(sur, { algo: 'dqn', lr: 0.1 }),
    )
    expect(fanovaImportances(sur, runs, MAX)[0].lever).toBe('algo')
  })

  it('predicts the mean from an unfittable (too few runs) surrogate', () => {
    const s = fitConfigSurrogate([run('a', { lr: 0.1 }, 42)], MAX)
    expect(s.trees).toHaveLength(0)
    expect(predictConfig(s, { lr: 0.9 })).toBe(42)
  })

  it('fanovaImportances ranks the driving lever first, in [0,1]', () => {
    const s = fitConfigSurrogate(grid, MAX)
    const imp = fanovaImportances(s, grid, MAX)
    expect(imp[0].lever).toBe('lr')
    expect(imp[0].importance).toBeGreaterThanOrEqual(0)
    expect(imp[0].importance).toBeLessThanOrEqual(1)
    expect(imp[0].importance).toBeGreaterThan(imp[1].importance)
  })

  it('ablationPath steps from worst to best over the differing levers', () => {
    const s = fitConfigSurrogate(grid, MAX)
    const path = ablationPath(s, grid, MAX)!
    expect(path.steps.length).toBeGreaterThanOrEqual(1)
    // every step changes a lever that actually differs between baseline and incumbent
    for (const step of path.steps) expect(String(path.baseline[step.lever])).not.toBe(step.to)
    // the incumbent is predicted at least as good as the baseline (max criterion)
    expect(path.incumbentPredicted).toBeGreaterThanOrEqual(path.baselinePredicted)
  })

  it('ablationPath is undefined with too few runs', () => {
    expect(
      ablationPath(
        fitConfigSurrogate([run('a', { lr: 0.1 }, 1)], MAX),
        [run('a', { lr: 0.1 }, 1)],
        MAX,
      ),
    ).toBeUndefined()
  })

  it('interactionGrid spans both levers and is deterministic', () => {
    const s = fitConfigSurrogate(grid, MAX)
    const g1 = interactionGrid(s, grid, MAX, 'lr', 'gamma')!
    const g2 = interactionGrid(s, grid, MAX, 'lr', 'gamma')!
    expect(g1.valuesA).toHaveLength(3)
    expect(g1.valuesB).toHaveLength(2)
    expect(g1.cells).toHaveLength(6)
    expect(g1.cells).toEqual(g2.cells)
  })

  it('interactionGrid is undefined when a lever has a single value', () => {
    expect(
      interactionGrid(fitConfigSurrogate(grid, MAX), grid, MAX, 'lr', 'missing'),
    ).toBeUndefined()
  })

  it('interactionGrid suppresses cells for an inapplicable conditional lever (n/a, not extrapolated)', () => {
    // forward_horizon applies only to the supervised model; rl runs carry the 'n/a' sentinel.
    const condRuns: AnalysisRun[] = []
    for (const fh of [1, 5]) {
      for (let s = 0; s < 3; s++)
        condRuns.push(run(`sup-${fh}-${s}`, { model_name: 'sup', forward_horizon: fh }, 10 + fh, { seed: s }))
    }
    for (let s = 0; s < 3; s++)
      condRuns.push(run(`rl-${s}`, { model_name: 'rl', forward_horizon: 'n/a' }, 50, { seed: s }))
    const s = fitConfigSurrogate(condRuns, MAX)
    const applies = { forward_horizon: { model_name: ['sup'] } }
    const g = interactionGrid(s, condRuns, MAX, 'forward_horizon', 'model_name', applies)!
    expect(g.valuesA).toEqual(['1', '5']) // observedValues strips the 'n/a' sentinel
    const rlCol = g.valuesB.indexOf('rl')
    const supCol = g.valuesB.indexOf('sup')
    expect(rlCol).toBeGreaterThanOrEqual(0)
    // Every forward_horizon value crossed with the rl model is invalid → null (not a surrogate guess).
    for (let ai = 0; ai < g.valuesA.length; ai++) {
      expect(g.cells[ai * g.valuesB.length + rlCol]).toBeNull()
      expect(typeof g.cells[ai * g.valuesB.length + supCol]).toBe('number')
    }
    // The reverse axis order is suppressed symmetrically.
    const g2 = interactionGrid(s, condRuns, MAX, 'model_name', 'forward_horizon', applies)!
    const rlRow = g2.valuesA.indexOf('rl')
    for (let bj = 0; bj < g2.valuesB.length; bj++)
      expect(g2.cells[rlRow * g2.valuesB.length + bj]).toBeNull()
  })

  it('interactionGrid without appliesWhen keeps every cell numeric (back-compat)', () => {
    const s = fitConfigSurrogate(grid, MAX)
    const g = interactionGrid(s, grid, MAX, 'lr', 'gamma')!
    expect(g.cells.every((c) => typeof c === 'number')).toBe(true)
  })

  it('fanovaImportances total-effect is ≥ the main effect for every lever', () => {
    const imp = fanovaImportances(fitConfigSurrogate(grid, MAX), grid, MAX)
    for (const f of imp) expect(f.total).toBeGreaterThanOrEqual(f.importance - 1e-9)
  })
})

describe('coupling / total-effect (Phase 3)', () => {
  // ADDITIVE: a drives the objective, b adds a little, NO interaction.
  const additive: AnalysisRun[] = []
  for (const a of [0, 1, 2]) {
    for (const b of [0, 1]) {
      for (let s = 0; s < 3; s++)
        additive.push(run(`add-${a}-${b}-${s}`, { a, b }, 10 * a + b, { seed: s }))
    }
  }
  // PURE INTERACTION (XOR): neither a nor b matters alone; only their combination does.
  const xor: AnalysisRun[] = []
  for (const a of [0, 1]) {
    for (const b of [0, 1]) {
      for (let s = 0; s < 4; s++)
        xor.push(run(`xor-${a}-${b}-${s}`, { a, b }, a === b ? 0 : 10, { seed: s }))
    }
  }

  it('additive design: total ≈ main (little interaction) and coupling is near zero', () => {
    const s = fitConfigSurrogate(additive, MAX)
    const aImp = fanovaImportances(s, additive, MAX).find((f) => f.lever === 'a')!
    expect(aImp.total - aImp.importance).toBeLessThan(0.15) // mostly main effect
    const coupling = leverCouplings(s, additive, MAX)[0]
    expect(coupling.strength).toBeLessThan(0.15)
  })

  it('XOR design: main effects vanish but total-effect + coupling are large', () => {
    const s = fitConfigSurrogate(xor, MAX)
    const imp = fanovaImportances(s, xor, MAX)
    const a = imp.find((f) => f.lever === 'a')!
    expect(a.importance).toBeLessThan(0.2) // marginal of a is flat
    expect(a.total).toBeGreaterThan(0.5) // but a matters a lot at each fixed b
    const coupling = leverCouplings(s, xor, MAX)
    expect(coupling[0].strength).toBeGreaterThan(0.5) // a×b are strongly coupled
    expect([coupling[0].leverA, coupling[0].leverB].sort()).toEqual(['a', 'b'])
  })

  it('is deterministic and empty for too few runs', () => {
    expect(leverCouplings(fitConfigSurrogate(xor, MAX), xor, MAX)).toEqual(
      leverCouplings(fitConfigSurrogate(xor, MAX), xor, MAX),
    )
    expect(
      leverCouplings(
        fitConfigSurrogate([run('a', { a: 1 }, 1)], MAX),
        [run('a', { a: 1 }, 1)],
        MAX,
      ),
    ).toEqual([])
  })
})

describe('pcaProjection (Phase 4)', () => {
  // lr carries most of the variance (0/5/10), gamma barely moves (0.90/0.91).
  const runs: AnalysisRun[] = []
  for (const lr of [0, 5, 10]) {
    for (const gamma of [0.9, 0.91]) {
      for (let s = 0; s < 2; s++)
        runs.push(run(`${lr}-${gamma}-${s}`, { lr, gamma }, lr + s * 0.1, { seed: s }))
    }
  }

  it('returns null below 3 setups', () => {
    expect(pcaProjection([run('a', { lr: 1 }, 1), run('b', { lr: 2 }, 2)], MAX)).toBeNull()
  })

  it('projects one point per setup with values = the setup IQM, and explained variance in [0,1]', () => {
    const p = pcaProjection(runs, MAX)!
    expect(p.points).toHaveLength(6) // 3 lr × 2 gamma setups (seeds collapsed)
    for (const ev of p.explainedVariance) {
      expect(ev).toBeGreaterThanOrEqual(0)
      expect(ev).toBeLessThanOrEqual(1)
    }
    expect(p.explainedVariance[0]).toBeGreaterThanOrEqual(p.explainedVariance[1])
    expect(p.features).toBe(2) // two numeric levers → two columns
  })

  it('PC1 captures the dominant lever — setups separate along x by lr', () => {
    const p = pcaProjection(runs, MAX)!
    // Mean PC1 coordinate per lr group should be monotone (the dominant axis spreads them out).
    const byLr = new Map<number, number[]>()
    runs.map((r) => r.config.lr as number).forEach((lr) => byLr.set(lr, []))
    p.points.forEach((pt) => {
      // recover lr from the first run key "lr-gamma-seed"
      const lr = Number(pt.key.split('-')[0])
      byLr.get(lr)!.push(pt.x)
    })
    const meanX = (lr: number) => byLr.get(lr)!.reduce((a, b) => a + b, 0) / byLr.get(lr)!.length
    const spread = Math.abs(meanX(10) - meanX(0))
    expect(spread).toBeGreaterThan(0.5) // the lr extremes are well separated on PC1
    expect(p.explainedVariance[0]).toBeGreaterThan(0.4)
  })

  it('is deterministic — identical runs give an identical projection', () => {
    expect(pcaProjection(runs, MAX)).toEqual(pcaProjection([...runs], MAX))
  })

  it('handles categorical (one-hot) levers and separates the algorithms on the plane', () => {
    const cat: AnalysisRun[] = []
    for (const algo of ['ppo', 'dqn', 'sac']) {
      for (let s = 0; s < 2; s++)
        cat.push(run(`${algo}-${s}`, { algo, lr: 0.1 }, algo === 'ppo' ? 90 : 10, { seed: s }))
    }
    const p = pcaProjection(cat, MAX)!
    expect(p).not.toBeNull()
    expect(p.points).toHaveLength(3) // one per algo setup
    expect(p.features).toBe(4) // 3 one-hot (algo) + 1 numeric (lr, constant → std 0 guarded)
    // distinct algos land at distinct points (the encoding actually varied them)
    const xs = p.points.map((pt) => Math.round(pt.x * 1e6))
    expect(new Set(xs).size).toBeGreaterThan(1)
  })
})

describe('computeConfigSpaceAnalysis (whole-space bundle)', () => {
  function spaceRuns(): AnalysisRun[] {
    const runs: AnalysisRun[] = []
    let k = 0
    for (const lr of [0.1, 0.2, 0.5])
      for (const bs of [32, 64])
        for (const seed of [0, 1, 2])
          runs.push(
            run(`r${k++}`, { lr, batch_size: bs }, lr * 100 + bs * 0.1 + seed * 0.01, { seed }),
          )
    return runs
  }

  it('folds seeds into setups and bundles every read off ONE surrogate', () => {
    const a = computeConfigSpaceAnalysis(spaceRuns(), MAX)!
    expect(a).not.toBeNull()
    expect(a.runCount).toBe(18)
    expect(a.setupCount).toBe(6) // 3 lr × 2 batch_size — the 3 seeds folded into each
    expect(a.surrogate.trees.length).toBeGreaterThan(0)
    expect(a.levers.sort()).toEqual(['batch_size', 'lr'])
    expect(a.importances).toHaveLength(2)
    expect(a.pca).not.toBeNull()
    expect(a.pca!.points).toHaveLength(6) // one point per setup
    expect(Array.isArray(a.recommendations)).toBe(true)
    expect(a.criterion).toEqual({ key: 'objective', direction: 'max' })
    // Config-effects folded in: screening importances + per-lever OFAT, computed over the raw runs.
    expect(a.screening.map((s) => s.lever).sort()).toEqual(['batch_size', 'lr'])
    expect(a.ofat).toBeTypeOf('object')
    expect(Object.keys(a.ofat).sort()).toEqual(['batch_size', 'lr'])
    expect(a.ofat.lr).toEqual(ofatContrasts(spaceRuns(), 'lr', MAX))
    // Setups (distinct configs) are shipped so the viewer can marginalise the surrogate for interactions.
    expect(a.setups).toHaveLength(6)
    expect(a.setups.every((s) => 'lr' in s.config && 'batch_size' in s.config)).toBe(true)
    // The interaction grid the viewer will draw is reproducible from the embedded surrogate + setups.
    const grid = interactionGrid(a.surrogate, a.setups, MAX, 'lr', 'batch_size')
    expect(grid).not.toBeNull()
  })

  it('searches coupling only among the high-effect levers, skipping inert ones', () => {
    const runs: AnalysisRun[] = []
    let k = 0
    // lr and batch_size drive the objective; `noise` has two values but no effect → inert.
    for (const lr of [0.1, 0.9])
      for (const bs of [32, 64])
        for (const noise of ['x', 'y'])
          for (const seed of [0, 1])
            runs.push(run(`r${k++}`, { lr, batch_size: bs, noise }, lr * 100 + bs, { seed }))
    const a = computeConfigSpaceAnalysis(runs, MAX)!
    expect(a.coupledLevers).not.toContain('noise')
    for (const c of a.couplings) {
      expect(a.coupledLevers).toContain(c.leverA)
      expect(a.coupledLevers).toContain(c.leverB)
    }
  })

  it('is deterministic — identical runs give an identical bundle', () => {
    expect(computeConfigSpaceAnalysis(spaceRuns(), MAX)).toEqual(
      computeConfigSpaceAnalysis(spaceRuns(), MAX),
    )
  })

  it('returns null when there are no valid runs', () => {
    expect(computeConfigSpaceAnalysis([], MAX)).toBeNull()
  })

  function envRuns(): AnalysisRun[] {
    // Two environments (transaction_fee 0.001 vs 0.01) over one model lever (lr). Fee is CONTEXT.
    const out: AnalysisRun[] = []
    let k = 0
    for (const transaction_fee of [0.001, 0.01])
      for (const lr of [0.1, 0.5])
        for (const seed of [0, 1])
          out.push(
            run(
              `r${k++}`,
              { transaction_fee, lr },
              (transaction_fee === 0.001 ? 100 : 50) + lr * 10,
              { seed },
            ),
          )
    return out
  }

  it('scopes the analysis to one environment over MODEL levers only, never tuning context', () => {
    const a = computeConfigSpaceAnalysis(envRuns(), MAX, { contextLevers: ['transaction_fee'] })!
    expect(a.environments).toHaveLength(2)
    expect(a.environments[0].runCount).toBe(4)
    expect(a.environment).toEqual({ transaction_fee: 0.001 }) // most-run (tie → insertion order)
    expect(a.runCount).toBe(4) // only that environment's runs
    expect(a.levers).toEqual(['lr']) // context lever stripped from the model space
    expect(a.importances.every((f) => f.lever !== 'transaction_fee')).toBe(true)
    expect(a.contextImportances.map((s) => s.lever)).toEqual(['transaction_fee'])
    // every recommendation stays IN this environment (carries its fee) and varies only model levers
    expect(a.recommendations.length).toBeGreaterThan(0)
    for (const rec of a.recommendations) {
      expect(rec.spec.fixed?.transaction_fee).toBe(0.001)
      expect('transaction_fee' in (rec.spec.sweep ?? {})).toBe(false)
    }
  })

  it('targets a requested environment', () => {
    const a = computeConfigSpaceAnalysis(envRuns(), MAX, {
      contextLevers: ['transaction_fee'],
      environment: { transaction_fee: 0.01 },
    })!
    expect(a.environment).toEqual({ transaction_fee: 0.01 })
    for (const rec of a.recommendations) expect(rec.spec.fixed?.transaction_fee).toBe(0.01)
  })

  it('analyses the whole space together when there are no context levers', () => {
    const a = computeConfigSpaceAnalysis(spaceRuns(), MAX)!
    expect(a.environment).toBeNull()
    expect(a.environments).toEqual([])
    expect(a.contextImportances).toEqual([])
  })

  it('honours appliesWhen — a conditional lever is pinned n/a where it does not apply', () => {
    // forward_horizon applies only to model 'sup'; for 'rl' it's swept but inert. Objective ignores it for rl.
    const runs: AnalysisRun[] = []
    let k = 0
    for (const model_name of ['rl', 'sup'])
      for (const forward_horizon of [1, 3])
        for (const seed of [0, 1])
          runs.push(
            run(
              `r${k++}`,
              { model_name, forward_horizon },
              model_name === 'sup' ? forward_horizon * 10 : 5,
              {
                seed,
              },
            ),
          )
    const a = computeConfigSpaceAnalysis(runs, MAX, {
      appliesWhen: { forward_horizon: { model_name: ['sup'] } },
    })!
    // rl runs all collapse to forward_horizon='n/a' (one setup); sup keeps its real horizons
    const rlSetups = a.setups.filter((s) => s.config.model_name === 'rl')
    expect(rlSetups.length).toBe(1)
    expect(rlSetups[0].config.forward_horizon).toBe('n/a')
    const supHorizons = a.setups
      .filter((s) => s.config.model_name === 'sup')
      .map((s) => s.config.forward_horizon)
    expect(supHorizons.sort()).toEqual([1, 3])
    // a recommendation for an rl model never carries a real forward_horizon (the n/a placeholder is dropped)
    for (const rec of a.recommendations) {
      if (rec.spec.fixed?.model_name === 'rl') {
        expect('forward_horizon' in (rec.spec.fixed ?? {})).toBe(false)
      }
    }
  })
})

describe('normalizeConditionalLevers', () => {
  it('pins an inapplicable conditional lever to n/a and leaves applicable ones', () => {
    const aw = { forward_horizon: { model_name: ['sup'] } }
    expect(normalizeConditionalLevers({ model_name: 'rl', forward_horizon: 5 }, aw)).toEqual({
      model_name: 'rl',
      forward_horizon: 'n/a',
    })
    expect(normalizeConditionalLevers({ model_name: 'sup', forward_horizon: 5 }, aw)).toEqual({
      model_name: 'sup',
      forward_horizon: 5,
    })
  })

  it('does not add an ABSENT conditional lever (only pins present ones)', () => {
    const aw = { forward_horizon: { model_name: ['sup'] } }
    expect(normalizeConditionalLevers({ model_name: 'rl' }, aw)).toEqual({ model_name: 'rl' })
  })

  it('cascades CHAINED conditionals to n/a regardless of key order (fixpoint)', () => {
    // C applies only when B='on'; B applies only when A='x'. With A='other', B→n/a, so C must also →n/a —
    // even though C is listed BEFORE its controller B (the order that exposed the old single-pass bug).
    const cfg = { A: 'other', B: 'on', C: 5 }
    const aw = { C: { B: ['on'] }, B: { A: ['x'] } }
    expect(normalizeConditionalLevers(cfg, aw)).toEqual({ A: 'other', B: 'n/a', C: 'n/a' })
    expect(cfg).toEqual({ A: 'other', B: 'on', C: 5 }) // input not mutated
  })

  it('tolerates a SINGLE (non-array) appliesWhen value instead of crashing (vals.map bug)', () => {
    // A manifest may declare `appliesWhen: { model_name: 'mcts' }` (a bare value). leverApplies must coerce
    // it to an array; the old `vals.map` threw "vals.map is not a function" and dropped every run silently.
    const aw = { mcts_sims: { model_name: 'mcts' } } as unknown as Record<string, Record<string, unknown[]>>
    expect(normalizeConditionalLevers({ model_name: 'mcts', mcts_sims: 80 }, aw)).toEqual({
      model_name: 'mcts',
      mcts_sims: 80,
    })
    expect(normalizeConditionalLevers({ model_name: 'random', mcts_sims: 80 }, aw)).toEqual({
      model_name: 'random',
      mcts_sims: 'n/a',
    })
  })
})

describe('normalizeByEnvironment', () => {
  const CTX = ['env']

  it('re-expresses each run relative to its OWN environment so cross-environment comparison is scale-fair', () => {
    // env A raw [10,20,30], env B raw [100,200,300]: B dwarfs A on RAW scale, but the relative structure is
    // identical. The top config of each env must standardise to the SAME value; the median config to ~0.
    const runs = [
      run('a-lo', { env: 'A', lr: 1 }, 10),
      run('a-mid', { env: 'A', lr: 2 }, 20),
      run('a-hi', { env: 'A', lr: 3 }, 30),
      run('b-lo', { env: 'B', lr: 1 }, 100),
      run('b-mid', { env: 'B', lr: 2 }, 200),
      run('b-hi', { env: 'B', lr: 3 }, 300),
    ]
    const z = normalizeByEnvironment(runs, MAX, CTX)
    expect(z['a-mid']).toBeCloseTo(0)
    expect(z['b-mid']).toBeCloseTo(0)
    expect(z['a-hi']).toBeCloseTo(z['b-hi']) // scale-fair across the two very-different-magnitude environments
    expect(z['a-hi']).toBeGreaterThan(0)
    expect(z['a-lo']).toBeCloseTo(z['b-lo'])
    expect(z['a-lo']).toBeLessThan(0)
  })

  it('orients so higher is always better — for a min criterion a LOWER raw value scores HIGHER', () => {
    const MIN: AnalysisCriterion = { key: 'durationMs', direction: 'min' }
    const runs = [
      run('fast', { env: 'A', lr: 1 }, 0, { durationMs: 10 }),
      run('mid', { env: 'A', lr: 2 }, 0, { durationMs: 20 }),
      run('slow', { env: 'A', lr: 3 }, 0, { durationMs: 30 }),
    ]
    const z = normalizeByEnvironment(runs, MIN, CTX)
    expect(z['fast']).toBeGreaterThan(0)
    expect(z['slow']).toBeLessThan(0)
  })

  it('maps a zero-spread environment (all setups equal) to neutral 0', () => {
    const runs = [run('x', { env: 'A', lr: 1 }, 5), run('y', { env: 'A', lr: 2 }, 5)]
    const z = normalizeByEnvironment(runs, MAX, CTX)
    expect(z['x']).toBe(0)
    expect(z['y']).toBe(0)
  })

  it('omits runs with no criterion value', () => {
    const runs = [
      run('a', { env: 'A', lr: 1 }, 10),
      run('b', { env: 'A', lr: 2 }, 20),
      run('c', { env: 'A', lr: 3 }, NaN),
    ]
    const z = normalizeByEnvironment(runs, MAX, CTX)
    expect('c' in z).toBe(false)
    expect('a' in z).toBe(true)
  })

  it('sets the environment scale from seed-folded SETUPS, so a many-seeded config cannot skew it', () => {
    // Two configs (setups): lr=1 at 10 (3 seeds), lr=2 at 30 (1 seed). Scale is set by the 2 setups, not 4
    // runs; median 20, so the two setups land symmetric around 0 and every seed of a config shares its z.
    const runs = [
      run('a1', { env: 'A', lr: 1 }, 10, { seed: 0 }),
      run('a2', { env: 'A', lr: 1 }, 10, { seed: 1 }),
      run('a3', { env: 'A', lr: 1 }, 10, { seed: 2 }),
      run('b1', { env: 'A', lr: 2 }, 30, { seed: 0 }),
    ]
    const z = normalizeByEnvironment(runs, MAX, CTX)
    expect(z['a1']).toBe(z['a2'])
    expect(z['a2']).toBe(z['a3'])
    expect(z['b1']).toBeCloseTo(-z['a1'])
    expect(z['a1']).toBeLessThan(0)
  })

  it('with no context levers, standardises over the whole set as one implicit environment', () => {
    const runs = [run('a', { lr: 1 }, 10), run('b', { lr: 2 }, 20), run('c', { lr: 3 }, 30)]
    const z = normalizeByEnvironment(runs, MAX, [])
    expect(z['b']).toBeCloseTo(0)
    expect(z['c']).toBeGreaterThan(0)
    expect(z['a']).toBeLessThan(0)
  })
})

// A4.3 scaffold: these robust-stats helpers were module-private and are now EXPORTED for the champion-verdict
// families (multi-window medians, DSR deflation, seed-stability). Pin their numeric output so the export — and
// any later lift into a shared stats module — cannot silently change them (they are deterministic + seeded).
describe('exported robust-stats helpers (champion-verdict families)', () => {
  it('medianOf — odd, even, empty', () => {
    expect(medianOf([3, 1, 2])).toBe(2)
    expect(medianOf([4, 1, 3, 2])).toBe(2.5)
    expect(medianOf([])).toBe(0)
  })

  it('benjaminiHochberg — the BH step-up mask at alpha', () => {
    // m=3: 0.01<=1/3·0.05, 0.02<=2/3·0.05 reject; 0.5>3/3·0.05 does not ⇒ [T,T,F] (by original index).
    expect(benjaminiHochberg([0.01, 0.02, 0.5], 0.05)).toEqual([true, true, false])
    expect(benjaminiHochberg([0.9], 0.05)).toEqual([false])
    expect(benjaminiHochberg([], 0.05)).toEqual([])
  })

  it('bootstrapDiff — N<2 point-delta branch (no variance ⇒ never significant)', () => {
    expect(bootstrapDiff([5], [1], 'max')).toEqual({ ci: [4, 4], pValue: 1, delta: 4 })
    // direction flips the orientation: min prefers the SMALLER, so [1] beats [5] by 4.
    expect(bootstrapDiff([1], [5], 'min')).toEqual({ ci: [4, 4], pValue: 1, delta: 4 })
  })

  it('bootstrapDiff — bootstrap path over constant samples is exactly determined', () => {
    // Every resample of a constant sample is the same, so every bootstrap diff = 1, ci=[1,1], and NO diff is
    // <=0 ⇒ pValue 0. This exercises the seeded loop without a fragile snapshot.
    expect(bootstrapDiff([2, 2, 2], [1, 1, 1], 'max')).toEqual({ ci: [1, 1], pValue: 0, delta: 1 })
  })

  it('pairedBootstrapDiff — N<2 point-delta branch (no variance ⇒ never significant)', () => {
    expect(pairedBootstrapDiff([5], [1], 'max')).toEqual({ ci: [4, 4], pValue: 1, delta: 4 })
  })

  it('pairedBootstrapDiff — constant paired samples are exactly determined', () => {
    // Constant samples ⇒ every paired resample diff = 1, ci=[1,1], no diff <=0 ⇒ pValue 0.
    expect(pairedBootstrapDiff([2, 2, 2], [1, 1, 1], 'max')).toEqual({ ci: [1, 1], pValue: 0, delta: 1 })
  })

  it('pairedBootstrapDiff — a tight, clear separation excludes 0 and is significant', () => {
    const d = pairedBootstrapDiff([0.86, 0.84, 0.85, 0.85, 0.84], [0.55, 0.56, 0.54, 0.55, 0.56], 'max')
    expect(d.ci[0]).toBeGreaterThan(0)
    expect(d.pValue).toBeLessThan(0.05)
  })

  it('pairedBootstrapDiff — genuinely-overlapping seeds (A wins some, B wins others) straddle 0 and are not significant', () => {
    const d = pairedBootstrapDiff([0.7, 0.52, 0.68, 0.54, 0.61], [0.55, 0.66, 0.57, 0.64, 0.6], 'max')
    expect(d.ci[0]).toBeLessThan(0)
    expect(d.ci[1]).toBeGreaterThan(0)
    expect(d.pValue).toBeGreaterThan(0.05)
  })

  it('pairedBootstrapDiff — truncates to the common length when the two sides differ (degrades, not throws)', () => {
    const d = pairedBootstrapDiff([0.9, 0.9, 0.9], [0.1, 0.1], 'max')
    expect(d.delta).toBeCloseTo(0.8, 10)
    expect(Number.isFinite(d.pValue)).toBe(true)
  })
})

// Degenerate / edge branches of the engine — empty samples, missing seeds, min-direction orientation, the
// sampled (vs enumerated) acquisition grid, and the whole-space bundle's null exits. Each pins a concrete
// output so the branch can't be mutated away silently.
describe('edge branches', () => {
  const MINdur: AnalysisCriterion = { key: 'durationMs', direction: 'min' }

  it('aggregateRunValues on an empty sample is the all-zero aggregate with a degenerate CI', () => {
    expect(aggregateRunValues([])).toEqual({
      n: 0,
      mean: 0,
      iqm: 0,
      median: 0,
      std: 0,
      min: 0,
      max: 0,
      ci: [0, 0],
    })
  })

  it('fitConfigSurrogate on no runs is an empty, zero-mean surrogate', () => {
    const s = fitConfigSurrogate([], MAX)
    expect(s.trees).toEqual([])
    expect(s.levers).toEqual([])
    expect(s.mean).toBe(0)
  })

  it('fanovaImportances returns [] with fewer than two configs', () => {
    const one = [run('a', { lr: 0.1 }, 10)]
    expect(fanovaImportances(fitConfigSurrogate(one, MAX), one, MAX)).toEqual([])
  })

  it('ofatContrasts folds a missing seed to 0 when counting distinct seeds', () => {
    const runs = [
      run('a', { lr: 0.1, batch_size: 64 }, 10, { seed: undefined }),
      run('b', { lr: 0.1, batch_size: 128 }, 20, { seed: undefined }),
    ]
    const c = ofatContrasts(runs, 'batch_size', MAX)[0]
    expect(c.levels.map((l) => l.value).sort()).toEqual(['128', '64'])
    expect(c.levels.every((l) => l.seeds === 1)).toBe(true) // seed undefined → 0 → one distinct seed
  })

  it('recommends more seeds even when the sole run carries no seed (nullish → 0)', () => {
    const runs = [run('a', { lr: 0.1, batch_size: 64 }, 100, { seed: undefined })]
    const thin = recommendExperiments(runs, MAX).find((r) => r.kind === 'thin-seeds')
    expect(thin).toBeDefined()
    expect((thin!.spec.seeds || []).length).toBeGreaterThan(0)
  })

  it('acquisition SAMPLES (not enumerates) a candidate grid over the cap, deterministically', () => {
    // 13 × 13 × 13 = 2197 candidate configs exceeds MAX_ACQUISITION_CANDIDATES (2000), forcing the seeded
    // sampling branch of cappedCartesian rather than the full-enumeration branch.
    const many = Array.from({ length: 13 }, (_, i) => run(`m${i}`, { a: i, b: i, c: i }, i))
    const r1 = recommendExperiments(many, MAX)
    const r2 = recommendExperiments([...many], MAX)
    expect(Array.isArray(r1)).toBe(true)
    expect(r1).toEqual(r2) // a seeded sampler ⇒ identical output for identical input
  })

  describe('ablationPath orientation & no-op', () => {
    it('orients gains for a MIN criterion (the fastest incumbent predicts at least as low as the baseline)', () => {
      const runs: AnalysisRun[] = []
      for (const lr of [0.1, 0.2])
        for (const gamma of [0.9, 0.99])
          for (let s = 0; s < 3; s++)
            runs.push(
              run(`${lr}-${gamma}-${s}`, { lr, gamma }, 0, {
                seed: s,
                durationMs: lr * 1000 + gamma * 10 + s,
              }),
            )
      const path = ablationPath(fitConfigSurrogate(runs, MINdur), runs, MINdur)!
      expect(path.steps.length).toBeGreaterThanOrEqual(1)
      expect(path.incumbentPredicted).toBeLessThanOrEqual(path.baselinePredicted)
    })

    it('is undefined when baseline and incumbent share every lever value (nothing to ablate)', () => {
      const runs = [run('a', { lr: 0.1 }, 10), run('b', { lr: 0.1 }, 20)]
      expect(ablationPath(fitConfigSurrogate(runs, MAX), runs, MAX)).toBeUndefined()
    })
  })

  describe('pcaProjection degenerate encodings', () => {
    it('puts all variance on PC1 when two numeric levers are perfectly correlated', () => {
      const runs = [
        run('a', { lr: 1, bs: 10 }, 1),
        run('b', { lr: 2, bs: 20 }, 2),
        run('c', { lr: 3, bs: 30 }, 3),
      ]
      const p = pcaProjection(runs, MAX)!
      expect(p.features).toBe(2)
      expect(p.points).toHaveLength(3)
      expect(p.explainedVariance[0]).toBeCloseTo(1) // one real dimension carries all the variance
      expect(p.explainedVariance[1]).toBeCloseTo(0) // the deflated 2nd component explains ~zero variance
      expect(p.explainedVariance[0]).toBeGreaterThan(p.explainedVariance[1])
    })

    it('returns null with ≥3 distinct setups but no encodable levers', () => {
      const ds = (candles: number) => ({ asset: 'BTC', timeframe: '1h', candles, from: 'a', to: 'b' })
      const runs = [
        run('a', {}, 10, { dataset: ds(100) }),
        run('b', {}, 20, { dataset: ds(200) }),
        run('c', {}, 30, { dataset: ds(300) }),
      ]
      expect(pcaProjection(runs, MAX)).toBeNull()
    })
  })

  describe('aggregateToSetupRuns — seed default & criterion field placement', () => {
    it('folds a missing seed to 0 in the recorded seed set/list', () => {
      const runs = [
        run('a', { lr: 0.1 }, 10, { seed: undefined }),
        run('b', { lr: 0.1 }, 12, { seed: 1 }),
      ]
      const setup = aggregateToSetupRuns(runs, MAX)[0]
      expect(setup.seedList).toEqual([0, 1]) // undefined → 0, sorted
      expect(setup.seeds).toBe(2)
    })

    it('stores the aggregate on durationMs for the runtime criterion', () => {
      const runs = [
        run('a', { lr: 0.1 }, 0, { seed: 0, durationMs: 100 }),
        run('b', { lr: 0.1 }, 0, { seed: 1, durationMs: 200 }),
      ]
      const setup = aggregateToSetupRuns(runs, MINdur)[0]
      expect(setup.durationMs).toBeCloseTo(150) // iqm([100, 200])
    })

    it('stores the aggregate under a metrics key for a metric criterion', () => {
      const WIN: AnalysisCriterion = { key: 'win_pct', direction: 'max' }
      const runs = [
        run('a', { lr: 0.1 }, 0, { seed: 0, metrics: { win_pct: 40 } }),
        run('b', { lr: 0.1 }, 0, { seed: 1, metrics: { win_pct: 60 } }),
      ]
      const setup = aggregateToSetupRuns(runs, WIN)[0]
      expect(setup.metrics!.win_pct).toBeCloseTo(50)
    })
  })

  describe('computeConfigSpaceAnalysis — direction, missing environment, single setup', () => {
    function envTimedRuns(): AnalysisRun[] {
      const out: AnalysisRun[] = []
      let k = 0
      for (const transaction_fee of [0.001, 0.01])
        for (const lr of [0.1, 0.5])
          for (const seed of [0, 1]) {
            k++
            out.push(
              run(`e${k}`, { transaction_fee, lr }, 0, {
                seed,
                durationMs: (transaction_fee === 0.001 ? 100 : 50) + lr * 10,
                ranAt: `2026-01-${String(k).padStart(2, '0')}`,
              }),
            )
          }
      return out
    }

    it('honours a MIN criterion in both the environment best and the convergence curve', () => {
      const a = computeConfigSpaceAnalysis(envTimedRuns(), MINdur, {
        contextLevers: ['transaction_fee'],
      })!
      expect(a.criterion.direction).toBe('min')
      expect(a.environments).toHaveLength(2)
      expect(a.environments[0].best).toBe(101) // MIN over the env's setups (101 < 105)
      const bests = a.convergence.map((p) => p.best)
      expect(bests).toHaveLength(4)
      expect(bests[0]).toBe(101)
      for (let i = 1; i < bests.length; i++) expect(bests[i]).toBeLessThanOrEqual(bests[i - 1]) // running min
    })

    it('returns null when the requested environment matches no runs', () => {
      const a = computeConfigSpaceAnalysis(envTimedRuns(), MAX, {
        contextLevers: ['transaction_fee'],
        environment: { transaction_fee: 999 },
      })
      expect(a).toBeNull()
    })

    it('sets ablation to null when there is only a single setup (nothing to ablate)', () => {
      const runs = [
        run('a', { lr: 0.1 }, 10, { seed: 0 }),
        run('b', { lr: 0.1 }, 11, { seed: 1 }),
        run('c', { lr: 0.1 }, 12, { seed: 2 }),
      ]
      const a = computeConfigSpaceAnalysis(runs, MAX)!
      expect(a).not.toBeNull()
      expect(a.ablation).toBeNull()
    })
  })
})
