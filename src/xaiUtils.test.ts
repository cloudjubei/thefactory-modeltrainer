import { describe, expect, it } from 'vitest'
import type { AnalysisCriterion, AnalysisRun } from './modelTrainerTypes.js'
import {
  ablationPath,
  aggregateRunValues,
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
