// §C pre-registration probe battery — the cheap, offline demonstration that each rigorous-ML-evaluation
// gap (docs/implementation-plan.md §C.2) BITES on the real engine BEFORE any build, mirroring §B's
// discipline: a pre-registered demonstration, then build only what survives. Each probe encodes a
// DIVERGENCE — the verdict crowns/holds/proves an incumbent while the engine's OWN significance or truth
// primitive (bootstrapDiff / rewardFitnessAlignment / held-out selection) disagrees on identical inputs.
// The fix's ingredients already exist and are simply not wired into the layer that crowns a champion.
// When the corresponding step (S1..S9) lands, the crowning assertion flips — that is the red-green for the
// build. Grounded entirely in exported engine surface (no ModelTrainerTools instance, no Python, no game).
import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

import {
  assembleChampionVerdict,
  incumbentSplitHoldout,
  proxyAlignment,
  verifyImprovement,
} from './diagnosticsUtils'
import { bootstrapDiff } from './xaiUtils'
import { validateTrainingRunSummary, validateRunProvenance } from './modelTrainerUtils'
import type {
  AnalysisRun,
  AnalysisCriterion,
  TrainerDiagnostics,
  TrainerGate,
  GateOp,
} from './modelTrainerTypes'

// viewer/hypothesis.js is the no-build browser module for the single-context verdict; load it as CommonJS
// the same way src/hypothesisViewer.test.ts does, so the ACTUAL viewer logic is probed here.
const require = createRequire(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))
const hpath = join(here, '..', 'viewer', 'hypothesis.js')
const hmod = new Module(hpath)
hmod.filename = hpath
hmod.paths = []
hmod._compile(readFileSync(hpath, 'utf8'), hpath)
const H: any = hmod.exports

const WIN: AnalysisCriterion = { key: 'win_rate', direction: 'max' }

const run = (
  config: Record<string, unknown>,
  metrics: Record<string, number>,
  opts: { objective?: number; seed?: number; status?: string; accepted?: boolean } = {},
): AnalysisRun => ({
  key: `${JSON.stringify(config)}-${opts.seed ?? 0}`,
  config: { ...config, seed: opts.seed ?? 0 },
  metrics,
  objective: opts.objective ?? metrics.win_rate,
  seed: opts.seed ?? 0,
  status: opts.status ?? 'completed',
  accepted: opts.accepted,
})

/** One run per seed for a setup, metrics.win_rate = the seed's value (objective mirrors it). */
const cohort = (
  config: Record<string, unknown>,
  values: number[],
  extraMetrics: Record<string, number> = {},
): AnalysisRun[] =>
  values.map((v, i) => run(config, { win_rate: v, ...extraMetrics }, { objective: v, seed: i }))

const gate = (metric: string, op: GateOp, value: number): TrainerGate => ({ metric, op, value })

describe('§C.0 — a board-game cohort runs & collapses to the objective with ZERO engine config', () => {
  it('no diagnostics ⇒ champion verdict inapplicable (steady undefined) and split holdout unverifiable', () => {
    const runs = [
      ...cohort({ setup: 'A' }, [0.6, 0.58, 0.62]),
      ...cohort({ setup: 'B' }, [0.5, 0.52, 0.49]),
    ]
    const v = assembleChampionVerdict(undefined, { runs, splitLevers: [], criterion: WIN })
    expect(v.steady).toBeUndefined()
    expect(v.championGates).toEqual([])
    expect(incumbentSplitHoldout(runs, [], WIN).verdict).toBe('unverifiable')
  })
})

describe('§C.4/S1 — seed-significance gate: the champion must beat the runner-up beyond seed noise', () => {
  const diagnostics: TrainerDiagnostics = {
    championGates: [gate('win_rate', '>', 0.5)],
    stability: { metric: 'win_rate', maxCiWidth: 0.3 },
    significance: { alpha: 0.05 },
  }

  it('FIXED: two setups that genuinely overlap across seeds (A wins some, B wins others) are NOT crowned steady', () => {
    const A = [0.7, 0.52, 0.68, 0.54, 0.61] // mean 0.610
    const B = [0.55, 0.66, 0.57, 0.64, 0.6] // mean 0.604 — interleaved, not a consistent edge
    const runs = [...cohort({ setup: 'A' }, A), ...cohort({ setup: 'B' }, B)]
    const v = assembleChampionVerdict(diagnostics, { runs, splitLevers: [], criterion: WIN })
    expect(v.steady).toBe(false)
    expect(v.championGates.find((g) => g.kind === 'seed-significance')?.pass).toBe(false)

    const d = bootstrapDiff(A, B, 'max')
    expect(d.pValue).toBeGreaterThan(0.05)
    expect(d.ci[0]).toBeLessThan(0)
  })

  it('FIXED: a clearly-separated champion (0.85 vs 0.55, tight) IS still crowned steady', () => {
    const A = [0.86, 0.84, 0.85, 0.85, 0.84]
    const B = [0.55, 0.56, 0.54, 0.55, 0.56]
    const runs = [...cohort({ setup: 'A' }, A), ...cohort({ setup: 'B' }, B)]
    const v = assembleChampionVerdict(diagnostics, { runs, splitLevers: [], criterion: WIN })
    expect(v.steady).toBe(true)
    expect(v.championGates.find((g) => g.kind === 'seed-significance')?.pass).toBe(true)
  })
})

describe('§C.2/S2 — best-of-N gate: a pure-noise best-of-30 champion is deflated to not-steady', () => {
  const noiseCohort = (): AnalysisRun[] => {
    const runs: AnalysisRun[] = []
    for (let i = 0; i < 30; i++) {
      const m = 0.44 + i * (0.12 / 29) // 30 setups spread 0.44..0.56 (mean 0.50) — pure best-of-N noise
      runs.push(...cohort({ setup: `s${i}` }, [m - 0.005, m, m + 0.005]))
    }
    return runs
  }
  const diagnostics: TrainerDiagnostics = {
    championGates: [gate('win_rate', '>', 0.5)],
    dsr: { threshold: 0.95 },
    stability: { metric: 'win_rate', maxCiWidth: 0.3 },
    multiplicity: { metric: 'win_rate', alpha: 0.05 },
  }

  it('FIXED: DSR stays inapplicable to win_rate, but the generic best-of-N gate deflates the noise champion', () => {
    const v = assembleChampionVerdict(diagnostics, {
      runs: noiseCohort(),
      splitLevers: [],
      criterion: WIN,
    })
    expect(v.championGates.find((g) => g.kind === 'dsr')?.applicable).toBe(false)
    expect(v.championGates.find((g) => g.kind === 'best-of-n')?.pass).toBe(false)
    expect(v.steady).toBe(false)
  })

  it('FIXED: a genuine outlier (0.80 among 0.44..0.56 noise) survives the best-of-N deflation', () => {
    const runs = [
      ...noiseCohort().filter((r) => r.config.setup !== 's29'),
      ...cohort({ setup: 'real' }, [0.79, 0.8, 0.81]),
    ]
    const v = assembleChampionVerdict(diagnostics, { runs, splitLevers: [], criterion: WIN })
    expect(v.championGates.find((g) => g.kind === 'best-of-n')?.pass).toBe(true)
    expect(v.steady).toBe(true)
  })
})

describe('§C.9/S3 — a provenance detector + a generic search-space floor (was: never enforced)', () => {
  it('FIXED: validateRunProvenance flags a run missing the reproducibility tuple (soft — never hard-fails)', () => {
    const bare = validateTrainingRunSummary({ objective: 0.62, metrics: { win_rate: 0.62 } })
    expect(bare.objective).toBe(0.62)
    const flag = validateRunProvenance(bare)
    expect(flag.complete).toBe(false)
    expect(flag.missing).toEqual(
      expect.arrayContaining(['gitCommit', 'configHash', 'seed', 'dataVersion']),
    )

    const full = validateRunProvenance({
      seed: 1,
      provenance: { gitCommit: 'a', configHash: 'b', dataVersion: 'v1' },
    })
    expect(full).toEqual({ complete: true, missing: [] })
  })

  it('FIXED: the generic diagnostics.searchSpace floor is honoured by the best-of-N gate (any consumer can declare it)', () => {
    const runs = [
      ...Array.from({ length: 5 }, (_, i) => cohort({ setup: `s${i}` }, [0.44 + i * 0.03])).flat(),
      ...cohort({ setup: 'real' }, [0.62]),
    ]
    const base: TrainerDiagnostics = { multiplicity: { metric: 'win_rate', alpha: 0.05 } }
    const ctx = { runs, splitLevers: [] as string[], criterion: WIN }
    const lax = assembleChampionVerdict(base, ctx)
    const strict = assembleChampionVerdict({ ...base, searchSpace: { nTrials: 5000 } }, ctx)
    expect(lax.championGates.find((g) => g.kind === 'best-of-n')?.pass).toBe(true)
    expect(strict.championGates.find((g) => g.kind === 'best-of-n')?.pass).toBe(false)
  })
})

describe('§C.1/S4 — pre-registration: a declarable seed-quorum + a fail-closed-with-reason benchmark', () => {
  const bgRun = (winRate: number, seed: number, extra: Record<string, number> = {}) => ({
    key: `r${seed}`,
    summary: {
      config: { setup: 'A', seed },
      objective: winRate,
      status: 'completed',
      metrics: { win_rate: winRate, ...extra },
    },
  })
  const bench = { metric: 'win_rate', threshold: 0.5, direction: 'max', quorum: 0.5 }

  it('FIXED: with a declared quorum, one lucky seed (1/5) does NOT prove the hypothesis', () => {
    const lucky = [bgRun(0.62, 0), bgRun(0.48, 1), bgRun(0.48, 2), bgRun(0.48, 3), bgRun(0.48, 4)]
    const measured = H.measuredFromRuns(lucky, 'max', bench)
    expect(measured.beatsHold).toBe(false)
    expect(H.autoVerdictFor(measured, 5)).toBe('disproved')

    const majority = [bgRun(0.62, 0), bgRun(0.61, 1), bgRun(0.63, 2), bgRun(0.6, 3), bgRun(0.48, 4)]
    expect(H.autoVerdictFor(H.measuredFromRuns(majority, 'max', bench), 5)).toBe('proven')
  })

  it('FIXED: omitting hypothesisBenchmark yields untested WITH AN EXPLICIT REASON (not a silent trading-default miss)', () => {
    const runs = [bgRun(0.9, 0), bgRun(0.9, 1), bgRun(0.9, 2)]
    const measured = H.measuredFromRuns(runs, 'max', undefined)
    expect(measured.beatsHold).toBeNull()
    expect(measured.reason).toMatch(/no hypothesisBenchmark declared/i)
    expect(H.autoVerdictFor(measured, 1)).toBe('untested')
  })
})

describe('§C.7/S5 — CI-based held-test: a high-variance split whose mean merely edges the baseline is not robust', () => {
  const splitA = [0.9, 0.3, 0.36] // mean 0.520 — but the CI straddles 0.5
  const splitB = [0.85, 0.32, 0.35] // mean 0.507
  const runs = [
    ...splitA.map((v, i) => run({ setup: 'X', board_size: 'A' }, { win_rate: v }, { seed: i })),
    ...splitB.map((v, i) => run({ setup: 'X', board_size: 'B' }, { win_rate: v }, { seed: i })),
  ]

  it('FIXED: with splitAxis.alpha the split does NOT hold and the effect+CI is surfaced', () => {
    const h = incumbentSplitHoldout(runs, ['board_size'], WIN, { baseline: 0.5, alpha: 0.05 })
    expect(h.verdict).not.toBe('robust')
    expect(h.held).toBe(0)
    const effA = h.splitEffects.find((e) => e.split === 'board_size=A')!
    expect(effA.ci[0]).toBeLessThan(0) // CI straddles the baseline — the interval the boolean used to drop
    expect(effA.held).toBe(false)
  })

  it('mean-mode (no alpha) is unchanged — BlackSwan walk-forward semantics preserved', () => {
    const h = incumbentSplitHoldout(runs, ['board_size'], WIN, { baseline: 0.5 })
    expect(h.verdict).toBe('robust') // mean 0.52 / 0.507 both > 0.5
    expect(h.splitEffects).toEqual([])
  })
})

describe('§C.3/S6 — a locked held-out TEST role: selection excludes the test, then consumes it once', () => {
  const mk = (setup: string, opp: string, v: number, seed = 0) =>
    run({ setup, opponent: opp }, { win_rate: v }, { seed })
  const runs = [
    mk('X', 'rungA', 0.5),
    mk('X', 'rungB', 0.5),
    mk('X', 'LOCKED', 0.9), // X wins ONLY on the locked test
    mk('Y', 'rungA', 0.55),
    mk('Y', 'rungB', 0.55),
    mk('Y', 'LOCKED', 0.5),
  ]

  it('BITE (no test declared): the incumbent is still crowned USING the LOCKED split', () => {
    expect(
      incumbentSplitHoldout(runs, ['opponent'], WIN, { baseline: 0.4 }).incumbentConfig?.setup,
    ).toBe('X')
  })

  it('FIXED: declaring LOCKED as the test excludes it from selection ⇒ Y is crowned, and the test is accounted', () => {
    const h = incumbentSplitHoldout(runs, ['opponent'], WIN, {
      baseline: 0.4,
      testValues: ['LOCKED'],
    })
    expect(h.incumbentConfig?.setup).toBe('Y') // selection no longer reads the locked test
    expect(h.testSplits).toEqual(['opponent=LOCKED'])
    expect(h.testConsumed).toBe(1) // the test is evaluated ONCE, post-selection
    expect(h.testHeld).toBe(1) // Y holds 0.5 > baseline 0.4 on the locked test
  })
})

describe('§C.8/S7 — proxy selection-regret: a high correlation can still hide crowning the wrong winner', () => {
  const mk = (setup: string, reward: number, winRate: number, seed: number) =>
    run({ setup, split: 's' }, { win_rate: winRate }, { objective: reward, seed })

  it('FIXED: proxyAlignment reports a POSITIVE selection regret even though the reward correlation is high', () => {
    const runs = [
      mk('a', 1.0, 0.5, 0),
      mk('a', 1.0, 0.5, 1),
      mk('b', 2.0, 0.6, 0),
      mk('b', 2.0, 0.6, 1),
      mk('c', 3.0, 0.7, 0),
      mk('c', 3.0, 0.7, 1), // truth-best win_rate 0.70
      mk('d', 5.0, 0.62, 0),
      mk('d', 5.0, 0.62, 1), // reward-best, but win_rate 0.62 < 0.70
    ]
    const pa = proxyAlignment(runs, 'win_rate')
    expect(pa.pearson).not.toBeNull()
    expect(pa.pearson!).toBeGreaterThan(0.3) // correlation still reads "GOOD"
    expect(pa.regret).toBeCloseTo(0.08, 5) // crowning by reward (d, 0.62) leaves 0.08 win_rate vs truth-best (c, 0.70)
  })
})

describe('§C.6/S8 — first-class adversarial verify: a naive best-of-12 single-seed fluke is flagged UNVERIFIABLE', () => {
  it('FIXED: verifyImprovement flags the naive fluke unverifiable — no lens can run', () => {
    const sweep: AnalysisRun[] = []
    for (let i = 0; i < 12; i++)
      sweep.push(...cohort({ setup: `c${i}` }, [0.5 + (i === 7 ? 0.2 : 0)])) // c7 flukes, 1 seed
    const v = verifyImprovement(sweep, WIN, { baseline: 0.5 })
    expect(v.unverifiable).toBe(true)
    expect(v.verified).toBe(false)
    expect(v.checks.every((c) => !c.applicable)).toBe(true)
  })

  it('FIXED: a well-seeded, split-robust champion verifies across the applicable lenses', () => {
    const good: AnalysisRun[] = []
    for (const board of ['A', 'B']) {
      for (let s = 0; s < 5; s++)
        good.push(
          run({ setup: 'win', board }, { win_rate: 0.8 + (s % 2 ? 0.01 : -0.01) }, { seed: s }),
        )
      for (let s = 0; s < 5; s++)
        good.push(run({ setup: 'weak', board }, { win_rate: 0.55 }, { seed: s }))
    }
    const v = verifyImprovement(good, WIN, { baseline: 0.5, splitLevers: ['board'] })
    expect(v.verified).toBe(true)
    expect(v.unverifiable).toBe(false)
    expect(v.checks.find((c) => c.name === 'seed-stability')).toMatchObject({
      applicable: true,
      pass: true,
    })
    expect(v.checks.find((c) => c.name === 'split-robust')).toMatchObject({
      applicable: true,
      pass: true,
    })
  })
})

describe('§C.5/S9 — declared degeneracy screens now bite the champion verdict (were viewer-only)', () => {
  const base = {
    degenerateWhen: [{ metric: 'draw_rate', op: '==', value: 1 }],
    championGates: [gate('win_rate', '>', 0.5)],
    stability: { metric: 'win_rate', maxCiWidth: 0.3 },
  } as TrainerDiagnostics

  it('FIXED: a fully-degenerate cohort (all draws) is NOT crowned steady', () => {
    const runs = cohort({ setup: 'A' }, [0.6, 0.58, 0.62], { draw_rate: 1 })
    const v = assembleChampionVerdict(base, { runs, splitLevers: [], criterion: WIN })
    expect(v.championGates.find((g) => g.kind === 'not-degenerate')?.pass).toBe(false)
    expect(v.steady).toBe(false)
  })

  it('FIXED: a healthy cohort (no draws) passes the degeneracy gate and is steady', () => {
    const runs = cohort({ setup: 'A' }, [0.6, 0.58, 0.62], { draw_rate: 0 })
    const v = assembleChampionVerdict(base, { runs, splitLevers: [], criterion: WIN })
    expect(v.championGates.find((g) => g.kind === 'not-degenerate')?.pass).toBe(true)
    expect(v.steady).toBe(true)
  })
})
