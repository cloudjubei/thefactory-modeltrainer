import { describe, it, expect } from 'vitest'
import type {
  AnalysisRun,
  ExplorationState,
  TrainerManifest,
} from './modelTrainerTypes.js'
import {
  initExplorationState,
  nextExplorationStep,
  clusterBasins,
  localRefineRecs,
} from './explorationUtils.js'
import type { Basin } from './modelTrainerTypes.js'
import { XAI_MIN_SEEDS } from './modelTrainerConstants.js'

// A synthetic project: one discrete lever `algo` (the basin axis), one important continuous lever
// `lr`, one INERT continuous lever `noise_knob` (screening must freeze it), and `seed` (the noise dim).
const MANIFEST: TrainerManifest = {
  name: 'synthetic',
  recordType: 'synthetic-run',
  run: 'noop',
  objective: { name: 'score', direction: 'max' },
  levers: {
    // default lr=0.1 is deliberately OFF-peak (A peaks at 0.5) so the climb is a real improvement to observe
    algo: { type: 'choice', choices: ['A', 'B', 'C'], default: 'A' },
    lr: { type: 'number', range: [0, 1], default: 0.1 },
    noise_knob: { type: 'number', range: [0, 1], default: 0.5 },
    seed: { type: 'number', default: 0 },
  },
}

// A random-policy baseline every run reports (the standard-battery `baseline` metric). Regions no better
// than this are not "maxima worth pursuing".
const BASELINE = 20

// The known surface: A is the GLOBAL max (peak 500 at lr=0.5), B a LOCAL max (peak 470 at lr=0.3),
// C sits AT the baseline (~20) and must NOT count as a basin. `noise_knob` has zero effect (screening
// must freeze it).
function trueScore(config: Record<string, unknown>): number {
  const algo = String(config.algo)
  const lr = Number(config.lr ?? 0.5)
  const seed = Number(config.seed ?? 0)
  // deterministic seed jitter, tiny vs the A/B gap — sets a small noise floor
  const jitter = (((seed * 37) % 7) - 3) * 0.4 // in [-1.2, 1.2]
  let base: number
  if (algo === 'A') base = 500 - 1600 * (lr - 0.5) ** 2
  else if (algo === 'B') base = 470 - 1600 * (lr - 0.3) ** 2
  else base = BASELINE // C: no better than random
  return base + jitter
}

let runSeq = 0
function evaluate(config: Record<string, unknown>): AnalysisRun {
  const score = trueScore(config)
  return {
    key: `run-${runSeq++}`,
    config: { ...config },
    objective: score,
    metrics: { score, baseline: BASELINE },
    seed: Number(config.seed ?? 0),
    status: 'completed',
  }
}

// Minimal faithful expander for the specs the strategist emits (configs | fixed + sweep, × seeds). Enough
// to drive the loop; the real matrix planner is exercised by the activity integration, not this unit.
function expandSpec(
  spec: {
    fixed?: Record<string, unknown>
    sweep?: Record<string, unknown[]>
    seeds?: number[]
    configs?: Array<{ config: Record<string, unknown> }>
  },
  manifest: TrainerManifest,
): Record<string, unknown>[] {
  const defaults: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(manifest.levers)) defaults[k] = v.default
  let combos: Record<string, unknown>[]
  if (spec.configs && spec.configs.length) {
    combos = spec.configs.map((c) => ({ ...defaults, ...c.config }))
  } else {
    const base = { ...defaults, ...(spec.fixed ?? {}) }
    combos = [base]
    for (const [lever, values] of Object.entries(spec.sweep ?? {})) {
      const next: Record<string, unknown>[] = []
      for (const c of combos) for (const val of values) next.push({ ...c, [lever]: val })
      combos = next
    }
  }
  const seeds = spec.seeds && spec.seeds.length ? spec.seeds : undefined
  return combos.flatMap((c) => (seeds ? seeds.map((s) => ({ ...c, seed: s })) : [{ ...c }]))
}

function drive(
  manifest: TrainerManifest,
  opts?: { maxRuns?: number; targetObjective?: number; maxRounds?: number },
): { state: ExplorationState; runs: AnalysisRun[]; rounds: number } {
  runSeq = 0
  let state = initExplorationState(manifest, { maxRuns: opts?.maxRuns })
  const runs: AnalysisRun[] = []
  let rounds = 0
  const maxRounds = opts?.maxRounds ?? 200
  while (!state.done && rounds < maxRounds) {
    const step = nextExplorationStep(state, runs, manifest, { targetObjective: opts?.targetObjective })
    state = step.stateNext
    for (const rec of step.batch) for (const cfg of expandSpec(rec.spec, manifest)) runs.push(evaluate(cfg))
    rounds++
  }
  return { state, runs, rounds }
}

describe('initExplorationState', () => {
  it('starts in calibrate with the manifest objective and empty maps', () => {
    const s = initExplorationState(MANIFEST, { maxRuns: 400 })
    expect(s.stage).toBe('calibrate')
    expect(s.objective).toEqual({ name: 'score', direction: 'max' })
    expect(s.recordType).toBe('synthetic-run')
    expect(s.basins).toEqual([])
    expect(s.done).toBe(false)
    expect(s.budget.maxRuns).toBe(400)
    expect(s.budget.spentRuns).toBe(0)
  })
})

describe('S0 calibrate', () => {
  it('emits a default-config batch across seeds when the archive is empty', () => {
    const s = initExplorationState(MANIFEST)
    const step = nextExplorationStep(s, [], MANIFEST)
    expect(step.stage).toBe('calibrate')
    expect(step.batch.length).toBeGreaterThan(0)
    const spec = step.batch[0].spec
    // default config, multiple seeds
    expect(spec.fixed?.algo).toBe('A')
    expect((spec.seeds ?? []).length).toBeGreaterThanOrEqual(5)
    expect(step.done).toBe(false)
  })

  it('measures a noise floor and advances to screen once the calibration seeds are in', () => {
    const s = initExplorationState(MANIFEST)
    const seeds = [0, 1, 2, 3, 4]
    // the default config (matches manifest defaults) is what S0 calibrates on
    const runs = seeds.map((seed) => evaluate({ algo: 'A', lr: 0.1, noise_knob: 0.5, seed }))
    const step = nextExplorationStep(s, runs, MANIFEST)
    expect(step.stateNext.noiseFloor).toBeGreaterThanOrEqual(0)
    expect(step.stateNext.noiseFloor).toBeLessThan(30) // tiny vs the A/B gap
    expect(step.stage).toBe('screen')
    expect(step.batch.length).toBeGreaterThan(0)
  })
})

describe('S1 screen', () => {
  it('keeps important levers active and freezes the inert one', () => {
    // an INDEPENDENT grid over (algo × lr × noise_knob) so noise_knob is decorrelated from the objective —
    // its marginal means are then flat and its importance ~0, the signal screening must act on.
    const runs: AnalysisRun[] = []
    let seed = 0
    for (const algo of ['A', 'B', 'C']) {
      for (const lr of [0, 0.2, 0.4, 0.6, 0.8, 1.0]) {
        for (const nk of [0, 0.5, 1.0]) runs.push(evaluate({ algo, lr, noise_knob: nk, seed: seed++ % 5 }))
      }
    }
    const state: ExplorationState = {
      ...initExplorationState(MANIFEST),
      stage: 'screen',
      noiseFloor: 1,
    }
    const step = nextExplorationStep(state, runs, MANIFEST)
    // once enough samples exist it should have partitioned + moved to global
    expect(step.stateNext.activeLevers).toContain('algo')
    expect(step.stateNext.activeLevers).toContain('lr')
    expect(step.stateNext.activeLevers).not.toContain('noise_knob')
    expect(step.stateNext.activeLevers).not.toContain('seed')
    expect(Object.keys(step.stateNext.frozenLevers)).toContain('noise_knob')
    expect(step.stage).toBe('global')
  })
})

describe('clusterBasins', () => {
  it('finds one basin per good discrete region and picks the right peak', () => {
    const runs: AnalysisRun[] = []
    for (const algo of ['A', 'B', 'C']) {
      for (const lr of [0.1, 0.3, 0.5, 0.7, 0.9]) {
        for (const seed of [0, 1, 2, 3, 4]) runs.push(evaluate({ algo, lr, noise_knob: 0.5, seed }))
      }
    }
    const basins = clusterBasins(
      runs,
      { key: 'objective', direction: 'max' },
      ['algo', 'lr'],
      1, // noiseFloor
    )
    const regions = basins.map((b) => String(b.region.algo)).sort()
    expect(regions).toEqual(['A', 'B']) // C is flat at baseline -> not a basin
    const a = basins.find((b) => b.region.algo === 'A')!
    expect(a.peakObjective).toBeGreaterThan(480)
    expect(Number(a.centerConfig.lr)).toBeCloseTo(0.5, 1)
    expect(a.peakSeeds).toBe(5)
  })
})

describe('adaptive coordinate-ascent step', () => {
  // A basin on `lr` (range [0,1]) whose peak we climb. peakSeeds high so the seed-stabilization rec is
  // skipped and only the coordinate sweep is emitted. `state.activeLevers` includes lr so it's swept.
  const basinAt = (lr: number): Basin => ({
    id: 'algo=A',
    region: { algo: 'A' },
    centerConfig: { algo: 'A', lr, seed: 0 },
    peakObjective: 500,
    peakSeeds: XAI_MIN_SEEDS,
    plateaued: false,
    memberRunKeys: [],
  })
  const stateWith = (): ExplorationState => ({
    ...initExplorationState(MANIFEST),
    stage: 'local',
    activeLevers: ['algo', 'lr'],
    noiseFloor: 1,
  })
  const runAt = (lr: number): AnalysisRun => ({
    key: `r-${lr}`,
    config: { algo: 'A', lr, seed: 0 },
    objective: 400,
    metrics: { score: 400, baseline: BASELINE },
    seed: 0,
    status: 'completed',
  })
  const sweptValues = (basin: Basin, runs: AnalysisRun[]): number[] => {
    const recs = localRefineRecs(basin, runs, MANIFEST, stateWith())
    const sweep = recs.find((r) => r.spec.sweep && 'lr' in r.spec.sweep)
    return sweep ? (sweep.spec.sweep!.lr as number[]).slice().sort((a, b) => a - b) : []
  }

  it('SHRINKS the step toward the peak as tried neighbours tighten (adaptive bisection, not fixed range/8)', () => {
    // center 0.5, nearest tried neighbours at 0.4 and 0.6 → half-gap 0.05, FINER than the old range/8 (0.125)
    const cands = sweptValues(basinAt(0.5), [runAt(0.4), runAt(0.5), runAt(0.6)])
    expect(cands.length).toBeGreaterThan(0)
    // every proposed point sits strictly inside the (0.4, 0.6) bracket — the search is honing in, not
    // re-probing at the coarse fixed ±0.125 (which would propose 0.375 / 0.625, OUTSIDE the bracket)
    for (const v of cands) {
      expect(v).toBeGreaterThan(0.4)
      expect(v).toBeLessThan(0.6)
    }
    const maxOffset = Math.max(...cands.map((v) => Math.abs(v - 0.5)))
    expect(maxOffset).toBeLessThanOrEqual(0.05 + 1e-9)
  })

  it('never proposes a FIRST step coarser than range/8 (bounded when neighbours are the range edges)', () => {
    // only the center tried → neighbours default to the range edges [0,1]; step capped at range/8 = 0.125
    const cands = sweptValues(basinAt(0.5), [runAt(0.5)])
    expect(cands.length).toBeGreaterThan(0)
    const maxOffset = Math.max(...cands.map((v) => Math.abs(v - 0.5)))
    expect(maxOffset).toBeLessThanOrEqual(0.125 + 1e-9)
  })

  it('PLATEAUS (emits no sweep) once the bracket is tighter than the min-step floor', () => {
    // neighbours 0.49 and 0.51 → half-gap 0.005 < range/64 (0.0156): resolved, nothing left to try
    const cands = sweptValues(basinAt(0.5), [runAt(0.49), runAt(0.5), runAt(0.51)])
    expect(cands).toEqual([])
  })

  it('drives a NARROW off-grid peak much closer than the old fixed step could', () => {
    // A sharp Gaussian-ish peak at lr=0.53 (off the 1/16 grid the old fixed step lands on). K large ⇒ narrow.
    const NARROW: TrainerManifest = { ...MANIFEST, recordType: 'narrow-run' }
    const trueNarrow = (c: Record<string, unknown>): number => {
      const lr = Number(c.lr ?? 0.1)
      const seed = Number(c.seed ?? 0)
      const jitter = (((seed * 37) % 7) - 3) * 0.2
      const base = String(c.algo) === 'A' ? 500 - 9000 * (lr - 0.53) ** 2 : BASELINE
      return base + jitter
    }
    let seq = 0
    let state = initExplorationState(NARROW, { maxRuns: 800 })
    const runs: AnalysisRun[] = []
    let rounds = 0
    while (!state.done && rounds < 300) {
      const step = nextExplorationStep(state, runs, NARROW, { targetObjective: 500 })
      state = step.stateNext
      for (const rec of step.batch)
        for (const cfg of expandSpec(rec.spec, NARROW)) {
          const score = trueNarrow(cfg)
          runs.push({ key: `n-${seq++}`, config: { ...cfg }, objective: score, metrics: { score, baseline: BASELINE }, seed: Number(cfg.seed ?? 0), status: 'completed' })
        }
      rounds++
    }
    const declared = state.basins.find((b) => b.id === state.declaredBasinId)
    expect(declared).toBeTruthy()
    // the adaptive step seats the peak within ~490 of 500; the OLD fixed range/8 stalled ~1 step short (~430)
    expect(declared!.peakObjective).toBeGreaterThan(490)
    expect(Number(declared!.centerConfig.lr)).toBeCloseTo(0.53, 1)
  })
})

describe('exhausted stage advance (no dead-end when the space is already covered)', () => {
  it('advances GLOBAL → local when told the proposed batch is fully redundant', () => {
    const runs = [
      ...[0.2, 0.5, 0.8].flatMap((lr) => [0, 1, 2, 3, 4].map((s) => evaluate({ algo: 'A', lr, seed: s }))),
    ]
    const state: ExplorationState = { ...initExplorationState(MANIFEST), stage: 'global', activeLevers: ['algo', 'lr'], frozenLevers: {}, noiseFloor: 1 }
    const normal = nextExplorationStep(state, runs, MANIFEST, {})
    expect(normal.stage).toBe('global') // absent the flag it keeps probing globally
    const advanced = nextExplorationStep(state, runs, MANIFEST, { exhausted: true })
    expect(advanced.stage).not.toBe('global') // told it's exhausted → it moves on (local/converged), never dead-ends
  })

  it('does NOT converge from LOCAL on exhausted while fresh refinement remains — it escalates instead', () => {
    // Only lr=0.5 tried, no frozen levers: the coordinate sweep still has fresh points (0.375…0.625), so a
    // controller "exhausted" signal must NOT dead-end into converged — the space is nowhere near covered.
    const runs = [0.5].flatMap((lr) => [0, 1, 2, 3, 4].map((s) => evaluate({ algo: 'A', lr, seed: s })))
    const state: ExplorationState = { ...initExplorationState(MANIFEST), stage: 'local', activeLevers: ['algo', 'lr'], frozenLevers: {}, noiseFloor: 1 }
    const advanced = nextExplorationStep(state, runs, MANIFEST, { exhausted: true })
    expect(advanced.done).toBe(false)
    expect(advanced.stage).toBe('local')
    expect(advanced.batch.length).toBeGreaterThan(0)
  })
})

// The core contract the user demanded: "it should never converge unless the whole search space was covered."
// A plateau in the CURRENT active subspace must ESCALATE — unfreeze a previously-fixed lever, then deepen the
// numeric resolution — and only declare `converged` once that whole ladder is dry (space genuinely covered).
describe('escalation ladder — never converge until the space is covered', () => {
  // MANIFEST-only helper: a manifest with NO inert lever, so `frozenLevers: {}` is a realistic full state
  // (every searchable lever active) and rung-1 (unfreeze) is legitimately empty — isolating rung-2 (deepen).
  const M2: TrainerManifest = {
    name: 'no-inert',
    recordType: 'no-inert-run',
    run: 'noop',
    objective: { name: 'score', direction: 'max' },
    levers: {
      algo: { type: 'choice', choices: ['A', 'B', 'C'], default: 'A' },
      lr: { type: 'number', range: [0, 1], default: 0.1 },
      seed: { type: 'number', default: 0 },
    },
  }
  const aRun = (manifest: TrainerManifest, lr: number, seed: number): AnalysisRun => {
    const cfg = manifest === M2 ? { algo: 'A', lr, seed } : { algo: 'A', lr, noise_knob: 0.5, seed }
    return evaluate(cfg)
  }

  it('UNFREEZES a fixed numeric lever instead of converging when all basins plateaued (rung 1: widen)', () => {
    // A is tightly resolved on lr (bracket 0.49–0.51 ⇒ plateaued), and noise_knob is frozen — so instead of
    // "all basins plateaued → converged", the search must unfreeze noise_knob and probe it.
    runSeq = 0
    const runs: AnalysisRun[] = []
    for (const lr of [0.48, 0.49, 0.5, 0.51, 0.52]) for (const s of [0, 1, 2, 3, 4]) runs.push(aRun(MANIFEST, lr, s))
    const state: ExplorationState = {
      ...initExplorationState(MANIFEST),
      stage: 'local',
      activeLevers: ['algo', 'lr'],
      frozenLevers: { noise_knob: 0.5 },
      noiseFloor: 1,
    }
    const step = nextExplorationStep(state, runs, MANIFEST, {})
    expect(step.done).toBe(false)
    expect(step.stateNext.activeLevers).toContain('noise_knob') // widened into the fixed lever
    expect(Object.keys(step.stateNext.frozenLevers)).not.toContain('noise_knob') // no longer pinned
    expect(step.stage).toBe('global')
    expect(step.batch.length).toBeGreaterThan(0)
    expect(JSON.stringify(step.batch.map((b) => b.spec.sweep))).toContain('noise_knob')
  })

  it('DEEPENS the numeric resolution when a basin is resolved at the current floor but not a finer one (rung 2)', () => {
    // lr tried at 0.48/0.5/0.52 → half-gap 0.01: resolved at range/64 (0.0156) but NOT at range/128 (0.0078).
    // With no frozen levers, rung 1 is empty, so the search must deepen rather than converge.
    runSeq = 0
    const runs: AnalysisRun[] = []
    for (const lr of [0.48, 0.5, 0.52]) for (const s of [0, 1, 2, 3, 4]) runs.push(aRun(M2, lr, s))
    const state: ExplorationState = {
      ...initExplorationState(M2),
      stage: 'local',
      activeLevers: ['algo', 'lr'],
      frozenLevers: {},
      noiseFloor: 1,
    }
    const step = nextExplorationStep(state, runs, M2, {})
    expect(step.done).toBe(false)
    expect(step.stage).toBe('local')
    expect(step.stateNext.refineDepth).toBe(1) // deepened one level
    expect(step.batch.length).toBeGreaterThan(0)
  })

  it('does NOT converge while the numeric range is still UNCOVERED — even with points clustered at the peak', () => {
    // A pathological "looks converged" case: three runs bunched at 0.4995–0.5005. The old logic would call the
    // basin plateaued and converge, but 99% of the lr range is untried — so the search must keep probing it.
    runSeq = 0
    const runs: AnalysisRun[] = []
    for (const lr of [0.4995, 0.5, 0.5005]) for (const s of [0, 1, 2, 3, 4]) runs.push(aRun(M2, lr, s))
    const state: ExplorationState = {
      ...initExplorationState(M2),
      stage: 'local',
      activeLevers: ['algo', 'lr'],
      frozenLevers: {},
      noiseFloor: 1,
    }
    const step = nextExplorationStep(state, runs, M2, {})
    expect(step.done).toBe(false) // the space is nowhere near covered
    const swept = step.batch.flatMap((b) => (b.spec.sweep?.lr as number[]) ?? [])
    expect(swept.some((v) => v < 0.4 || v > 0.6)).toBe(true) // reaching out into the untried range
  })

  it('deepening the resolution proposes STRICTLY finer coordinate points (localRefineRecs honours refineDepth)', () => {
    const basin: Basin = {
      id: 'algo=A',
      region: { algo: 'A' },
      centerConfig: { algo: 'A', lr: 0.5, seed: 0 },
      peakObjective: 500,
      peakSeeds: XAI_MIN_SEEDS,
      plateaued: false,
      memberRunKeys: [],
    }
    const runs: AnalysisRun[] = [
      { key: 'a', config: { algo: 'A', lr: 0.48, seed: 0 }, objective: 490, metrics: { score: 490, baseline: BASELINE }, seed: 0, status: 'completed' },
      { key: 'b', config: { algo: 'A', lr: 0.52, seed: 0 }, objective: 490, metrics: { score: 490, baseline: BASELINE }, seed: 0, status: 'completed' },
    ]
    const base = { ...initExplorationState(M2), stage: 'local' as const, activeLevers: ['algo', 'lr'], noiseFloor: 1 }
    const shallow = localRefineRecs(basin, runs, M2, { ...base, refineDepth: 0 })
    const deep = localRefineRecs(basin, runs, M2, { ...base, refineDepth: 2 })
    // at the coarse floor (range/64 = 0.0156) the 0.48–0.52 bracket is resolved → no points; the finer floor
    // (range/256) admits the sub-0.0156 offsets, so deepening un-plateaus the basin.
    expect(shallow.length).toBe(0)
    expect(deep.length).toBeGreaterThan(0)
  })

  it('declares the best region as the maximum at convergence even when NO region cleared the basin margin', () => {
    // Objective barely above baseline (gain 2 < the min-span margin 3) ⇒ clusterBasins finds nothing, yet the
    // run history has a clear best — the UI must never report "no maximum found".
    const runs: AnalysisRun[] = []
    for (const lr of [0.4995, 0.5, 0.5005]) for (const s of [0, 1, 2, 3, 4]) {
      runs.push({ key: `w-${lr}-${s}`, config: { algo: 'A', lr, seed: s }, objective: 22, metrics: { score: 22, baseline: 20 }, seed: s, status: 'completed' })
    }
    expect(clusterBasins(runs, { key: 'objective', direction: 'max' }, ['algo', 'lr'], 1, 20)).toEqual([])
    const state: ExplorationState = { ...initExplorationState(M2), stage: 'local', activeLevers: ['algo', 'lr'], frozenLevers: {}, noiseFloor: 1 }
    const step = nextExplorationStep(state, runs, M2, {})
    expect(step.done).toBe(true)
    expect(step.stateNext.basins.length).toBeGreaterThanOrEqual(1) // a fallback basin synthesised from the best run
    expect(step.stateNext.declaredBasinId).toBeTruthy()
  })
})

describe('end-to-end synthetic drive', () => {
  it('finds ALL maxima, declares the global one, and converges with improving regret', () => {
    const { state, rounds } = drive(MANIFEST, { maxRuns: 600, targetObjective: 500, maxRounds: 200 })

    expect(state.done).toBe(true)
    expect(state.stage).toBe('converged')
    expect(rounds).toBeLessThan(200)

    // recall: both real maxima found, C rejected
    const regions = state.basins.map((b) => String(b.region.algo)).sort()
    expect(regions).toEqual(['A', 'B'])

    // the declared global max is A, near 500
    const declared = state.basins.find((b) => b.id === state.declaredBasinId)
    expect(declared).toBeTruthy()
    expect(declared!.region.algo).toBe('A')
    expect(declared!.peakObjective).toBeGreaterThan(485)

    // regret improves from first measurement to last
    expect(state.regret.length).toBeGreaterThan(1)
    expect(state.regret[state.regret.length - 1].bestObjective).toBeGreaterThan(state.regret[0].bestObjective)
  })

  it('respects the run budget as a hard ceiling', () => {
    const { state, runs } = drive(MANIFEST, { maxRuns: 60, maxRounds: 200 })
    expect(runs.length).toBeLessThanOrEqual(60 + 30) // may overshoot by at most one batch
    expect(state.done).toBe(true)
  })

  it('holds the stage and emits nothing while paused', () => {
    const paused: ExplorationState = { ...initExplorationState(MANIFEST), stage: 'global', paused: true }
    const step = nextExplorationStep(paused, [], MANIFEST)
    expect(step.batch).toEqual([])
    expect(step.stage).toBe('global')
    expect(step.done).toBe(false)
  })

  it('converges immediately and emits nothing once the budget is already spent', () => {
    const spent: ExplorationState = { ...initExplorationState(MANIFEST, { maxRuns: 0 }), stage: 'global' }
    const step = nextExplorationStep(spent, [], MANIFEST)
    expect(step.done).toBe(true)
    expect(step.stage).toBe('converged')
    expect(step.batch).toEqual([])
  })
})

// A Wine-like problem: MINIMISE, only NUMERIC levers → no categorical axis → a single basin (the whole
// space). Proves the same engine works with the objective flipped and just the manifest changed.
const WINE_BASELINE = 0.8
const WINE: TrainerManifest = {
  name: 'wine-like',
  recordType: 'wine-run',
  run: 'noop',
  objective: { name: 'rmse', direction: 'min' },
  levers: {
    lr: { type: 'number', range: [0.01, 0.3], default: 0.2 },
    depth: { type: 'number', range: [1, 10], default: 5 },
    seed: { type: 'number', default: 0 },
  },
}
function wineRmse(config: Record<string, unknown>): number {
  const lr = Number(config.lr ?? 0.2)
  const depth = Number(config.depth ?? 5)
  const seed = Number(config.seed ?? 0)
  const jitter = (((seed * 37) % 7) - 3) * 0.002
  return 0.2 + 4 * (lr - 0.05) ** 2 + 0.001 * (depth - 3) ** 2 + jitter
}
let wineSeq = 0
function wineEval(config: Record<string, unknown>): AnalysisRun {
  const rmse = wineRmse(config)
  return {
    key: `w-${wineSeq++}`,
    config: { ...config },
    objective: rmse,
    metrics: { rmse, baseline: WINE_BASELINE },
    seed: Number(config.seed ?? 0),
    status: 'completed',
  }
}

describe('min-direction (Wine-like) drive', () => {
  it('minimises to a single basin and declares it, with regret decreasing', () => {
    wineSeq = 0
    let state = initExplorationState(WINE, { maxRuns: 600 })
    const runs: AnalysisRun[] = []
    let rounds = 0
    while (!state.done && rounds < 200) {
      const step = nextExplorationStep(state, runs, WINE, { targetObjective: 0.2 })
      state = step.stateNext
      for (const rec of step.batch) for (const cfg of expandSpec(rec.spec, WINE)) runs.push(wineEval(cfg))
      rounds++
    }
    expect(state.done).toBe(true)
    expect(state.basins.length).toBe(1) // no categorical lever -> the whole space is one basin
    const declared = state.basins.find((b) => b.id === state.declaredBasinId)
    expect(declared).toBeTruthy()
    expect(declared!.peakObjective).toBeLessThan(0.3) // near the 0.2 minimum
    // regret DEcreases for a minimisation objective
    expect(state.regret[state.regret.length - 1].bestObjective).toBeLessThan(state.regret[0].bestObjective)
  })
})

describe('steer overrides', () => {
  it('forces a lever active (pinActive) and freezes another at a chosen value (pinFrozen)', () => {
    const runs: AnalysisRun[] = []
    let seed = 0
    for (const algo of ['A', 'B', 'C']) {
      for (const lr of [0, 0.2, 0.4, 0.6, 0.8, 1.0]) {
        for (const nk of [0, 0.5, 1.0]) runs.push(evaluate({ algo, lr, noise_knob: nk, seed: seed++ % 5 }))
      }
    }
    const state: ExplorationState = {
      ...initExplorationState(MANIFEST),
      stage: 'screen',
      noiseFloor: 1,
      steer: { pinActive: ['noise_knob'], pinFrozen: { lr: 0.5 } },
    }
    const step = nextExplorationStep(state, runs, MANIFEST)
    expect(step.stateNext.activeLevers).toContain('noise_knob') // forced active despite ~0 importance
    expect(step.stateNext.frozenLevers.lr).toBe(0.5) // forced frozen at the chosen value
  })
})

describe('boolean axis + range-less lever (screen sampling)', () => {
  it('drives a boolean-axis problem from scratch and converges to the flag=true basin', () => {
    const M: TrainerManifest = {
      name: 'boolproj',
      recordType: 'bool-run',
      run: 'noop',
      objective: { name: 'score', direction: 'max' },
      levers: {
        flag: { type: 'boolean', default: false }, // basin axis; true is the global region
        lr: { type: 'number', range: [0, 1], default: 0.1 },
        blip: { type: 'number', default: 0.5 }, // NO range -> exercises the [0,1] fallback + refine skip
        seed: { type: 'number', default: 0 },
      },
    }
    let n = 0
    const evalB = (config: Record<string, unknown>): AnalysisRun => {
      const flag = config.flag === true
      const lr = Number(config.lr ?? 0.1)
      const seed = Number(config.seed ?? 0)
      const jitter = (((seed * 37) % 7) - 3) * 0.4
      const base = (flag ? 400 : 300) + (100 - 400 * (lr - 0.5) ** 2)
      return {
        key: `b-${n++}`,
        config: { ...config },
        objective: base + jitter,
        metrics: { score: base + jitter, baseline: 20 },
        seed,
        status: 'completed',
      }
    }
    let state = initExplorationState(M, { maxRuns: 800 })
    const runs: AnalysisRun[] = []
    let rounds = 0
    while (!state.done && rounds < 200) {
      const step = nextExplorationStep(state, runs, M, { targetObjective: 500 })
      state = step.stateNext
      for (const rec of step.batch) for (const cfg of expandSpec(rec.spec, M)) runs.push(evalB(cfg))
      rounds++
    }
    expect(state.done).toBe(true)
    const regions = state.basins.map((b) => String(b.region.flag)).sort()
    expect(regions).toEqual(['false', 'true'])
    const declared = state.basins.find((b) => b.id === state.declaredBasinId)
    expect(declared!.region.flag).toBe(true)
    expect(declared!.peakObjective).toBeGreaterThan(485)
  })
})

describe('screen fallback when every numeric lever is inert', () => {
  it('keeps at least one numeric climb dimension active', () => {
    const M: TrainerManifest = {
      name: 'catonly',
      recordType: 'cat-run',
      run: 'noop',
      objective: { name: 'score', direction: 'max' },
      levers: {
        algo: { type: 'choice', choices: ['A', 'B'], default: 'A' },
        lr: { type: 'number', range: [0, 1], default: 0.5 }, // inert: objective depends only on algo
        seed: { type: 'number', default: 0 },
      },
    }
    const runs: AnalysisRun[] = []
    let seed = 0
    for (const algo of ['A', 'B']) {
      for (const lr of [0, 0.2, 0.4, 0.6, 0.8, 1.0]) {
        // 2 algos × 6 lr = 12 setups, at/above the screen-sample threshold so partitioning fires
        for (const s of [0, 1, 2]) {
          const score = algo === 'A' ? 500 : 100 // lr has NO effect
          runs.push({
            key: `k-${seed}`,
            config: { algo, lr, seed: s },
            objective: score,
            metrics: { score, baseline: 20 },
            seed: s,
            status: 'completed',
          })
          seed++
        }
      }
    }
    const state: ExplorationState = { ...initExplorationState(M), stage: 'screen', noiseFloor: 1 }
    const step = nextExplorationStep(state, runs, M)
    expect(step.stateNext.activeLevers).toContain('algo')
    expect(step.stateNext.activeLevers).toContain('lr') // kept as the sole climb dimension despite ~0 importance
  })
})

describe('global-stage discrete coverage', () => {
  it('proposes untried categorical values before climbing', () => {
    // archive has only algo A & B sampled; C is untried -> coverage must propose it
    const runs: AnalysisRun[] = []
    let seed = 0
    for (const algo of ['A', 'B']) {
      for (const lr of [0.3, 0.5, 0.7]) runs.push(evaluate({ algo, lr, noise_knob: 0.5, seed: seed++ % 5 }))
    }
    const state: ExplorationState = {
      ...initExplorationState(MANIFEST),
      stage: 'global',
      noiseFloor: 1,
      activeLevers: ['algo', 'lr'],
      frozenLevers: { noise_knob: 0.5 },
      basins: [],
    }
    const step = nextExplorationStep(state, runs, MANIFEST)
    const coverage = step.batch.find((r) => r.reason.includes('cover untried algo'))
    expect(coverage).toBeTruthy()
    expect(JSON.stringify(coverage!.spec.sweep)).toContain('C')
  })
})

describe('categorical levers are always-active basin axes', () => {
  it('keeps inert categorical levers active (never frozen), unlike an inert numeric lever', () => {
    const M: TrainerManifest = {
      name: 'axes',
      recordType: 'axes-run',
      run: 'noop',
      objective: { name: 'score', direction: 'max' },
      levers: {
        lr: { type: 'number', range: [0, 1], default: 0.5 },
        dead_num: { type: 'number', range: [0, 1], default: 0.5 }, // inert numeric -> frozen
        use_x: { type: 'boolean', default: false }, // inert categorical -> stays a basin axis
        variant: { type: 'choice', choices: ['p', 'q'], default: 'p' }, // inert categorical -> stays an axis
        seed: { type: 'number', default: 0 },
      },
    }
    const runs: AnalysisRun[] = []
    let seed = 0
    for (const lr of [0, 0.25, 0.5, 0.75, 1]) {
      for (const dn of [0, 0.5, 1]) {
        for (const ux of [false, true]) {
          for (const v of ['p', 'q']) {
            const score = 500 - 1600 * (lr - 0.5) ** 2 // depends ONLY on lr
            runs.push({
              key: `c-${seed}`,
              config: { lr, dead_num: dn, use_x: ux, variant: v, seed: seed % 5 },
              objective: score,
              metrics: { score, baseline: 20 },
              seed: seed % 5,
              status: 'completed',
            })
            seed++
          }
        }
      }
    }
    const state: ExplorationState = { ...initExplorationState(M), stage: 'screen', noiseFloor: 1 }
    const step = nextExplorationStep(state, runs, M)
    expect(step.stateNext.activeLevers).toEqual(expect.arrayContaining(['lr', 'use_x', 'variant']))
    expect(step.stateNext.activeLevers).not.toContain('dead_num') // inert numeric IS frozen
    expect(Object.keys(step.stateNext.frozenLevers)).toContain('dead_num')
    expect(typeof step.stateNext.frozenLevers.dead_num).toBe('number') // coerced to a number
  })
})
