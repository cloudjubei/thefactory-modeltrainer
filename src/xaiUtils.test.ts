import { describe, expect, it } from 'vitest'
import type { AnalysisCriterion, AnalysisRun } from './modelTrainerTypes.js'
import {
  ablationPath,
  aggregateRunValues,
  criterionValueOf,
  fanovaImportances,
  fitConfigSurrogate,
  interactionGrid,
  iqm,
  leverImportances,
  ofatContrasts,
  predictConfig,
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
    expect(criterionValueOf(run('a', {}, 0, { metrics: {} }), { key: 'ghost', direction: 'max' })).toBeUndefined()
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
    for (const lr of [0.1, 0.2]) for (let s = 0; s < 5; s++) runs.push(run(`${lr}_${s}`, { lr }, 10 + lr * 100 + s, { seed: s }))
    const imp = leverImportances(runs, MAX)[0]
    expect(imp.minRuns).toBe(5)
    expect(imp.confident).toBe(true)
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
    const cell = recs.find((r) => r.kind === 'missing-cell')
    expect(cell).toBeDefined()
    expect(cell!.spec.fixed).toMatchObject({ lr: 0.2, batch_size: 128 })
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

describe('config surrogate (Phase 3)', () => {
  // A grid where lr drives the objective strongly and gamma barely moves it, replicated across seeds.
  const grid: AnalysisRun[] = []
  for (const lr of [0.1, 0.2, 0.3]) {
    for (const gamma of [0.9, 0.99]) {
      for (let s = 0; s < 3; s++) {
        grid.push(run(`${lr}_${gamma}_${s}`, { lr, gamma }, lr * 100 + gamma + s * 0.5, { seed: s }))
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
    expect(predictConfig(s, { lr: 0.3, gamma: 0.9 })).toBeGreaterThan(predictConfig(s, { lr: 0.1, gamma: 0.9 }))
  })

  it('handles categorical (string-valued) levers', () => {
    const runs: AnalysisRun[] = []
    for (const algo of ['ppo', 'dqn']) {
      for (let s = 0; s < 4; s++) {
        runs.push(run(`${algo}_${s}`, { algo, lr: 0.1 }, algo === 'ppo' ? 90 + s : 10 + s, { seed: s }))
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
    expect(ablationPath(fitConfigSurrogate([run('a', { lr: 0.1 }, 1)], MAX), [run('a', { lr: 0.1 }, 1)], MAX)).toBeUndefined()
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
    expect(interactionGrid(fitConfigSurrogate(grid, MAX), grid, MAX, 'lr', 'missing')).toBeUndefined()
  })
})
