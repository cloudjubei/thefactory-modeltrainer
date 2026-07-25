import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/diagnostics.js is the no-build browser "Research Diagnostician" view; load it as CommonJS the
// same way explorationViewer.test.ts loads viewer/exploration.js, so the ACTUAL viewer logic (the 7-check
// diagnosis + campaign generators) is unit-tested here (vitest only scans src/**).
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'diagnostics.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const Diagnostics: any = mod.exports

// --- fixtures -----------------------------------------------------------------------------------------

// CartPole-like: objective max, ceiling declared via hypothesisBenchmark on the objective metric itself.
const CARTPOLE_MANIFEST = {
  recordType: 'cartpole-run',
  objective: { name: 'eval_return_mean', direction: 'max' as const },
  hypothesisBenchmark: { metric: 'eval_return_mean', threshold: 475 },
  levers: {
    model_name: { type: 'choice', choices: ['ppo', 'a2c'], scope: 'model' },
    learning_rate: { type: 'number', range: [1e-4, 1e-2], scope: 'model' },
    seed: { type: 'number', scope: 'model' },
  },
}

// BlackSwan-like: objective max, a per-run benchmark metric that is the NULL (beats buy&hold), and a
// declared split axis. Best setup wins in window A, loses in window B => single-split luck.
const BLACKSWAN_MANIFEST = {
  recordType: 'bs-run',
  objective: { name: 'traded_return', direction: 'max' as const },
  hypothesisBenchmark: { metric: 'return_vs_hold_pct', threshold: 0 },
  diagnostics: { splitAxis: { levers: ['window'], kind: 'walk_forward' } },
  levers: {
    model_name: { type: 'choice', choices: ['x', 'y'], scope: 'model' },
    window: { type: 'choice', choices: ['A', 'B'], scope: 'dataset' },
    lr: { type: 'number', range: [0, 1], scope: 'model' },
    seed: { type: 'number', scope: 'model' },
  },
}

// Wine-like: objective MIN, no benchmark (no null declared).
const WINE_MANIFEST = {
  recordType: 'wine-run',
  objective: { name: 'val_rmse', direction: 'min' as const },
  levers: {
    model_name: { type: 'choice', choices: ['gbm', 'rf'], scope: 'model' },
    max_depth: { type: 'number', range: [1, 8], scope: 'model' },
    seed: { type: 'number', scope: 'model' },
  },
}

function rec(config: any, objective: number, metrics: any = {}, extra: any = {}) {
  return { key: JSON.stringify(config) + ':' + (extra.status || 'c'), config, objective, metrics, seed: config.seed, status: 'completed', ...extra }
}

// a converged CartPole cohort: a clear winner cluster at the 500 ceiling, seeded; plus weaker setups.
function cartpoleConverged() {
  const runs: any[] = []
  for (const s of [0, 1, 2, 3, 4]) runs.push(rec({ model_name: 'ppo', learning_rate: 0.003, seed: s }, 499 + (s % 2), { eval_return_mean: 499 + (s % 2) }))
  for (const s of [0, 1, 2]) runs.push(rec({ model_name: 'a2c', learning_rate: 0.001, seed: s }, 120 + s, { eval_return_mean: 120 + s }))
  for (const s of [0, 1, 2]) runs.push(rec({ model_name: 'ppo', learning_rate: 0.0001, seed: s }, 60 + s, { eval_return_mean: 60 + s }))
  return runs
}

// a single-split-luck BlackSwan cohort: setup {x,lr:0.5} beats hold in window A, loses in window B.
function blackswanOverfit() {
  const runs: any[] = []
  for (const s of [0, 1, 2]) runs.push(rec({ model_name: 'x', window: 'A', lr: 0.5, seed: s }, 80 + s, { return_vs_hold_pct: 60 + s, traded_return: 80 + s }))
  for (const s of [0, 1, 2]) runs.push(rec({ model_name: 'x', window: 'B', lr: 0.5, seed: s }, -40 + s, { return_vs_hold_pct: -50 + s, traded_return: -40 + s }))
  for (const s of [0, 1, 2]) runs.push(rec({ model_name: 'y', window: 'A', lr: 0.2, seed: s }, 10 + s, { return_vs_hold_pct: -5 + s, traded_return: 10 + s }))
  return runs
}

function wineConverged() {
  const runs: any[] = []
  for (const s of [0, 1, 2, 3, 4]) runs.push(rec({ model_name: 'gbm', max_depth: 3, seed: s }, 0.56 + s * 0.001, { val_rmse: 0.56 + s * 0.001 }))
  for (const s of [0, 1, 2]) runs.push(rec({ model_name: 'rf', max_depth: 6, seed: s }, 0.85 + s * 0.01, { val_rmse: 0.85 + s * 0.01 }))
  return runs
}

// --- resolveSpec --------------------------------------------------------------------------------------

describe('resolveSpec', () => {
  it('derives a CEILING target when hypothesisBenchmark.metric === objective', () => {
    const s = Diagnostics.resolveSpec(CARTPOLE_MANIFEST)
    expect(s.target).toBe(475)
    expect(s.nullBaseline).toBeFalsy()
  })
  it('derives a NULL baseline (per-run metric) when the benchmark metric differs from the objective', () => {
    const s = Diagnostics.resolveSpec(BLACKSWAN_MANIFEST)
    expect(s.nullBaseline).toBeTruthy()
    expect(s.nullBaseline.perRunMetric).toBe('return_vs_hold_pct')
    expect(s.target).toBeUndefined()
  })
  it('reads a declared split axis, and leaves it empty when none is declared/inferable', () => {
    expect(Diagnostics.resolveSpec(BLACKSWAN_MANIFEST).splitLevers).toEqual(['window'])
    expect(Diagnostics.resolveSpec(CARTPOLE_MANIFEST).splitLevers).toEqual([])
  })
})

// --- partitionCohort ----------------------------------------------------------------------------------

describe('partitionCohort', () => {
  it('counts decision-grade setups (>= min seeds) and flags degenerate/non-completed', () => {
    const spec = Diagnostics.resolveSpec(CARTPOLE_MANIFEST)
    const runs = cartpoleConverged().concat([
      rec({ model_name: 'ppo', learning_rate: 0.5, seed: 0 }, 0, {}, { status: 'failed' }),
      rec({ model_name: 'ppo', learning_rate: 0.9, seed: 0 }, 9, {}, { health: { status: 'degenerate', flags: ['x'] } }),
    ])
    const c = Diagnostics.partitionCohort(runs, spec)
    expect(c.total).toBe(13)
    expect(c.completed).toBe(12)
    expect(c.failed).toBe(1)
    expect(c.degenerate).toBe(1)
    // the ppo/0.003 setup has 5 seeds => at least one decision-grade setup
    expect(c.decisionGradeN).toBeGreaterThanOrEqual(1)
  })
})

// --- individual checks --------------------------------------------------------------------------------

describe('checks', () => {
  it('nullCeiling: reports AT-CEILING when the incumbent CI reaches the declared target', () => {
    const spec = Diagnostics.resolveSpec(CARTPOLE_MANIFEST)
    const f = Diagnostics.checkNullCeiling(cartpoleConverged(), spec)
    expect(f.category).toBe('null-ceiling')
    expect(['at-ceiling', 'beats-null']).toContain(f.verdict)
    expect(f.severity === 'ok').toBe(true)
  })

  it('nullCeiling: single-window winners still beat the null on the per-run metric', () => {
    const spec = Diagnostics.resolveSpec(BLACKSWAN_MANIFEST)
    const f = Diagnostics.checkNullCeiling(blackswanOverfit(), spec)
    expect(f.category).toBe('null-ceiling')
  })

  it('splitConsistency: flags single-split-luck when the incumbent wins one split and loses another', () => {
    const spec = Diagnostics.resolveSpec(BLACKSWAN_MANIFEST)
    const f = Diagnostics.checkSplitConsistency(blackswanOverfit(), spec)
    expect(f.category).toBe('split-consistency')
    expect(f.verdict).toBe('single-split-luck')
    expect(f.severity).toBe('blocker')
  })

  it('splitConsistency: unverifiable (info) when no split axis is declared', () => {
    const spec = Diagnostics.resolveSpec(CARTPOLE_MANIFEST)
    const f = Diagnostics.checkSplitConsistency(cartpoleConverged(), spec)
    expect(f.verdict).toBe('unverifiable')
    expect(f.severity).toBe('info')
  })

  it('discriminability: flags under-seeding when most setups have a single seed', () => {
    const spec = Diagnostics.resolveSpec(WINE_MANIFEST)
    const singleSeed = [0, 1, 2, 3, 4, 5].map((i) => rec({ model_name: 'gbm', max_depth: i, seed: 0 }, 0.6 + i * 0.05, { val_rmse: 0.6 + i * 0.05 }))
    const f = Diagnostics.checkDiscriminability(singleSeed, spec)
    expect(f.category).toBe('objective-discriminability')
    expect(f.verdict).toBe('under-seeded')
  })

  it('incumbentSeparation: a clear winner (min objective) is distinguishable', () => {
    const spec = Diagnostics.resolveSpec(WINE_MANIFEST)
    const f = Diagnostics.checkIncumbentSeparation(wineConverged(), spec)
    expect(f.category).toBe('incumbent-separation')
    expect(f.verdict).toBe('distinguishable')
    expect(f.severity).toBe('ok')
  })
})

// --- diagnose (composer) ------------------------------------------------------------------------------

describe('diagnose', () => {
  it('CartPole converged => no blockers, verdict converged/winner', () => {
    const d = Diagnostics.diagnose({ runs: cartpoleConverged(), manifest: CARTPOLE_MANIFEST })
    expect(d.findings.some((f: any) => f.severity === 'blocker')).toBe(false)
    expect(['converged', 'winner-emerging', 'at-ceiling']).toContain(d.verdict)
    expect(d.headline.doNext).toBeTruthy()
  })

  it('Wine converged (min objective) => distinguishable winner, no blockers', () => {
    const d = Diagnostics.diagnose({ runs: wineConverged(), manifest: WINE_MANIFEST })
    expect(d.findings.some((f: any) => f.severity === 'blocker')).toBe(false)
    expect(['converged', 'winner-emerging']).toContain(d.verdict)
  })

  it('BlackSwan overfit => a split-consistency blocker drives the verdict', () => {
    const d = Diagnostics.diagnose({ runs: blackswanOverfit(), manifest: BLACKSWAN_MANIFEST })
    expect(d.findings.some((f: any) => f.category === 'split-consistency' && f.severity === 'blocker')).toBe(true)
    expect(d.verdict).toBe('single-split-luck')
  })

  it('empty corpus => a single cohort blocker, never a crash', () => {
    const d = Diagnostics.diagnose({ runs: [], manifest: CARTPOLE_MANIFEST })
    expect(d.findings.length).toBeGreaterThanOrEqual(1)
    expect(d.findings[0].category).toBe('cohort-integrity')
  })
})

// --- campaign generators ------------------------------------------------------------------------------

describe('reseedSpecs', () => {
  it('adds the MISSING seeds for the top-N promising setups (never re-runs everything)', () => {
    const spec = Diagnostics.resolveSpec(CARTPOLE_MANIFEST)
    const runs = cartpoleConverged() // ppo/0.003 has seeds 0..4 already; a2c/0.001 has 0..2
    const out = Diagnostics.reseedSpecs({ runs, manifest: CARTPOLE_MANIFEST, spec, topN: 3, targetSeeds: 5 })
    expect(out.length).toBe(1)
    const configs = out[0].configs.map((c: any) => c.config)
    // a2c/0.001 (3 seeds) should get seeds 3 and 4 added; ppo/0.003 (5 seeds) needs none
    expect(configs.some((c: any) => c.model_name === 'a2c' && c.seed === 3)).toBe(true)
    expect(configs.every((c: any) => !(c.model_name === 'ppo' && c.learning_rate === 0.003))).toBe(true)
  })
})

describe('replicateSpecs', () => {
  it('produces ONE campaign per split value, each replicating the shortlist across seeds', () => {
    const spec = Diagnostics.resolveSpec(BLACKSWAN_MANIFEST)
    const runs = blackswanOverfit()
    const out = Diagnostics.replicateSpecs({ runs, manifest: BLACKSWAN_MANIFEST, spec, topN: 2, seeds: [0, 1, 2, 3, 4] })
    // one spec per distinct window value (A, B)
    const windows = new Set(out.flatMap((s: any) => s.configs.map((c: any) => c.config.window)))
    expect(windows).toEqual(new Set(['A', 'B']))
    expect(out.length).toBe(2)
    // every config in a spec pins that spec's single window
    for (const s of out) {
      const ws = new Set(s.configs.map((c: any) => c.config.window))
      expect(ws.size).toBe(1)
    }
  })
})

describe('analyze + paint (worker split)', () => {
  it('analyze returns a slim, clone-safe result (cohort as counts, no run arrays) matching diagnose', () => {
    const data = { runs: blackswanOverfit(), manifest: BLACKSWAN_MANIFEST }
    const a = Diagnostics.analyze(data)
    const d = Diagnostics.diagnose(data)
    expect(a.d.verdict).toBe(d.verdict)
    expect(a.d.cohort.validCount).toBe(d.cohort.valid.length)
    expect(a.d.cohort.valid).toBeUndefined() // the 20k run-object arrays must NOT cross the worker boundary
    expect(Array.isArray(a.d.findings)).toBe(true)
    expect(a.campaigns).toHaveProperty('reseed')
    expect(a.campaigns).toHaveProperty('replicate')
    expect(Array.isArray(a.steps)).toBe(true)
    expect(() => JSON.stringify(a)).not.toThrow() // structured-clone-safe
  })

  it('paint builds the plan DOM from a slim analysis without touching removed run arrays', () => {
    const data = { runs: blackswanOverfit(), manifest: BLACKSWAN_MANIFEST }
    const a = Diagnostics.analyze(data)
    const container: any = { innerHTML: '', querySelectorAll: () => [] }
    expect(() => Diagnostics.paint(container, a, data, { canDiscuss: false })).not.toThrow()
    expect(container.innerHTML).toContain('Your plan')
  })
})

describe('planSteps', () => {
  it('maps an actionable finding to a launchable step carrying its NEW-run count', () => {
    const d = Diagnostics.diagnose({ runs: blackswanOverfit(), manifest: BLACKSWAN_MANIFEST })
    const steps = Diagnostics.planSteps(d, { reseed: [], replicate: [{ configs: [{ config: { a: 1 } }, { config: { a: 2 } }] }] })
    const split = steps.find((s: any) => s.category === 'split-consistency')
    expect(split).toBeTruthy()
    expect(split.action.kind).toBe('replicate')
    expect(split.newRuns).toBe(2)
    expect(split.done).toBe(false)
  })

  it('marks a launchable step DONE when its action adds 0 new runs (already on record — no re-recommending)', () => {
    const d = Diagnostics.diagnose({ runs: blackswanOverfit(), manifest: BLACKSWAN_MANIFEST })
    const steps = Diagnostics.planSteps(d, { reseed: [], replicate: [] })
    const split = steps.find((s: any) => s.category === 'split-consistency')
    expect(split.done).toBe(true)
    expect(split.newRuns).toBe(0)
  })

  it('excludes ok/info findings from the plan (they are not action items)', () => {
    const d = Diagnostics.diagnose({ runs: cartpoleConverged(), manifest: CARTPOLE_MANIFEST })
    const steps = Diagnostics.planSteps(d, { reseed: [], replicate: [] })
    expect(steps.every((s: any) => s.severity === 'blocker' || s.severity === 'caution')).toBe(true)
  })
})

describe('checkCrossAssetRobustness', () => {
  const settestFor = (runKey: string, cells: Record<string, number | null>) => ({
    runKey,
    trainedValues: { asset: 'BTCUSDT' },
    levers: {
      asset: Object.fromEntries(
        Object.entries(cells).map(([value, vsHold]) => [
          value,
          vsHold === null
            ? { value, status: 'failed', error: 'no data', evaluatedAt: 'T' }
            : { value, status: 'completed', returnVsHold: vsHold, objective: 1, evaluatedAt: 'T' },
        ]),
      ),
    },
    updatedAt: 'T',
  })
  const spec = () => (Diagnostics as any).resolveSpec(BLACKSWAN_MANIFEST)
  const runs = () => blackswanOverfit()
  const incumbentKeys = () =>
    runs()
      .filter((r: any) => r.config.model_name === 'x' && r.config.window === 'A')
      .map((r: any) => r.key)

  it('is unverifiable-info with no settest matrices at all', () => {
    const f = Diagnostics.checkCrossAssetRobustness(runs(), spec(), [])
    expect(f.severity).toBe('info')
    expect(f.verdict).toBe('unverifiable')
  })

  it('cautions when matrices exist but none cover the incumbent', () => {
    const f = Diagnostics.checkCrossAssetRobustness(runs(), spec(), [settestFor('someone-else', { ETHUSDT: 2 })])
    expect(f.severity).toBe('caution')
    expect(f.verdict).toBe('not-cross-tested')
  })

  it('is ok/robust when every completed incumbent cell beats hold', () => {
    const keys = incumbentKeys()
    const f = Diagnostics.checkCrossAssetRobustness(runs(), spec(), [
      settestFor(keys[0], { ETHUSDT: 3, SOLUSDT: 1 }),
      settestFor(keys[1], { ETHUSDT: 0.5 }),
    ])
    expect(f.severity).toBe('ok')
    expect(f.verdict).toBe('robust')
  })

  it('is a blocker when NO incumbent cell beats hold — the edge does not travel', () => {
    const f = Diagnostics.checkCrossAssetRobustness(runs(), spec(), [
      settestFor(incumbentKeys()[0], { ETHUSDT: -4, SOLUSDT: -2 }),
    ])
    expect(f.severity).toBe('blocker')
    expect(f.verdict).toBe('asset-bound')
  })

  it('cautions on a mixed result (failed cells ignored for the ratio)', () => {
    const f = Diagnostics.checkCrossAssetRobustness(runs(), spec(), [
      settestFor(incumbentKeys()[0], { ETHUSDT: 3, SOLUSDT: -2, XRPUSDT: null }),
    ])
    expect(f.severity).toBe('caution')
    expect(f.verdict).toBe('partial')
  })

  it('diagnose folds the check in when settests are supplied', () => {
    const out = Diagnostics.diagnose({ manifest: BLACKSWAN_MANIFEST, runs: runs(), settests: [] })
    expect(out.findings.some((f: any) => f.category === 'cross-asset')).toBe(true)
  })
})
