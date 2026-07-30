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
  coverageGridRecs,
  gateConvergenceOnSplits,
} from './explorationUtils.js'
import type { Basin } from './modelTrainerTypes.js'
import { XAI_MIN_SEEDS, EXPLORATION_MAX_REFINE_DEPTH, EXPLORATION_BATCH_MAX, EXPLORATION_MAX_REGION_AXES } from './modelTrainerConstants.js'

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

  it('advances to screen from an existing seed-replicated archive when the default config is NOT seeded (populated project)', () => {
    runSeq = 0
    // A large pre-existing archive that never ran the manifest-default config (lr=0.1) but IS seed-replicated
    // elsewhere — calibrate must estimate the noise floor from it and advance, not stall re-proposing a default
    // batch (the BlackSwan stuck-at-calibrate bug).
    const archive = [0.3, 0.7].flatMap((lr) => [0, 1, 2, 3, 4].map((s) => evaluate({ algo: 'A', lr, noise_knob: 0.5, seed: s })))
    const step = nextExplorationStep(initExplorationState(MANIFEST), archive, MANIFEST)
    expect(step.stage).toBe('screen')
    expect(step.stateNext.noiseFloor).toBeGreaterThanOrEqual(0)
  })
})

describe('budget accounting on a populated archive', () => {
  it('counts runs THIS exploration produced, not the whole pre-existing archive (no instant converge)', () => {
    runSeq = 0
    const archive = [0.2, 0.5, 0.8].flatMap((lr) => [0, 1, 2, 3, 4].map((s) => evaluate({ algo: 'A', lr, noise_knob: 0.5, seed: s }))) // 15 runs
    // maxRuns:10 over a 15-run archive would OLD-converge instantly ("budget exhausted"); now producedRuns=0.
    const state: ExplorationState = { ...initExplorationState(MANIFEST, { maxRuns: 10 }), stage: 'global', activeLevers: ['algo', 'lr'], frozenLevers: {}, noiseFloor: 1 }
    const step = nextExplorationStep(state, archive, MANIFEST)
    expect(step.done).toBe(false)
    expect(step.stateNext.baselineRuns).toBe(15)
    expect(step.stateNext.budget.spentRuns).toBe(0) // this exploration has produced nothing yet
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

  it('CAPS categorical basin axes (model_name preferred) and keeps numeric climb slots on a high-cardinality manifest', () => {
    // BlackSwan-shaped: MANY categorical levers. Without a cap all 4 discrete would become basin axes (a 4-D
    // region cross-product) and `MAX_ACTIVE_LEVERS - 4` would leave numerics un-climbable. The cap must keep only
    // EXPLORATION_MAX_REGION_AXES categoricals (model_name always in), freeze the rest at their CATEGORICAL value,
    // and still activate numeric climb dims.
    const BS: TrainerManifest = {
      name: 'bs', recordType: 'bs-run', run: 'noop',
      objective: { name: 'ret', direction: 'max' },
      levers: {
        model_name: { type: 'choice', choices: ['m0', 'm1', 'm2'], default: 'm0' },
        net_arch: { type: 'choice', choices: ['a', 'b'], default: 'a' },
        optimizer: { type: 'choice', choices: ['adam', 'sgd'], default: 'adam' },
        use_x: { type: 'boolean', default: false },
        lr: { type: 'number', range: [0, 1], default: 0.5 },
        gamma: { type: 'number', range: [0.9, 1], default: 0.99 },
        buf: { type: 'number', range: [1, 100], default: 50 },
        seed: { type: 'number', default: 0 },
      },
    }
    const runs: AnalysisRun[] = []
    let s = 0
    for (const model of ['m0', 'm1', 'm2']) for (const na of ['a', 'b']) for (const lr of [0.2, 0.5, 0.8]) {
      const ret = (model === 'm1' ? 500 : 200) + (na === 'b' ? 30 : 0) - 200 * (lr - 0.5) ** 2 // model_name + lr matter most
      runs.push({ key: `bs-${s}`, config: { model_name: model, net_arch: na, optimizer: 'adam', use_x: false, lr, gamma: 0.99, buf: 50, seed: s % 5 }, objective: ret, metrics: { ret, baseline: 20 }, seed: s % 5, status: 'completed' })
      s++
    }
    const state: ExplorationState = { ...initExplorationState(BS), stage: 'screen', noiseFloor: 1 }
    const step = nextExplorationStep(state, runs, BS)
    const active = step.stateNext.activeLevers
    const activeCats = active.filter((l) => BS.levers[l].type !== 'number')
    expect(activeCats.length).toBeLessThanOrEqual(EXPLORATION_MAX_REGION_AXES) // NOT all 4 discrete
    expect(active).toContain('model_name') // the model-identity lever is always a basin axis
    expect(active.filter((l) => BS.levers[l].type === 'number').length).toBeGreaterThanOrEqual(1) // numerics DO climb
    // a frozen categorical keeps its category value — never coerced to NaN
    for (const [l, v] of Object.entries(step.stateNext.frozenLevers)) {
      if (BS.levers[l] && BS.levers[l].type !== 'number') expect(Number.isNaN(v as number)).toBe(false)
    }
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

  it('treats a `choice` lever with NUMERIC-looking values as a categorical basin axis (via the manifest)', () => {
    // n_layers is a discrete CHOICE whose values happen to be numbers — each is its own region/maximum, NOT a
    // continuous climb dimension. Without the manifest, value-based detection would wrongly treat it as numeric.
    const M: TrainerManifest = {
      name: 'layers', recordType: 'layers-run', run: 'noop',
      objective: { name: 'score', direction: 'max' },
      levers: {
        n_layers: { type: 'choice', choices: [1, 2, 3], default: 1 },
        lr: { type: 'number', range: [0, 1], default: 0.5 },
        seed: { type: 'number', default: 0 },
      },
    }
    const runs: AnalysisRun[] = []
    for (const n of [1, 2, 3]) for (const lr of [0.3, 0.5, 0.7]) for (const s of [0, 1, 2, 3, 4]) {
      const score = (n === 2 ? 500 : n === 1 ? 470 : 20) - 400 * (lr - 0.5) ** 2 // n_layers=2 best, =3 at baseline
      runs.push({ key: `l-${n}-${lr}-${s}`, config: { n_layers: n, lr, seed: s }, objective: score, metrics: { score, baseline: 20 }, seed: s, status: 'completed' })
    }
    const withManifest = clusterBasins(runs, { key: 'objective', direction: 'max' }, ['n_layers', 'lr'], 1, 20, M)
    // one basin per good n_layers value — enumerated as distinct maxima, not blurred into one climb dimension
    expect(withManifest.map((b) => Number(b.region.n_layers)).sort()).toEqual([1, 2])
    // WITHOUT the manifest, numeric-valued choices are mis-detected as a climb dim → collapses to one basin
    const withoutManifest = clusterBasins(runs, { key: 'objective', direction: 'max' }, ['n_layers', 'lr'], 1, 20)
    expect(withoutManifest.length).toBe(1)
  })

  it('treats an ARRAY-valued lever (e.g. net_arch [64,64]) as a categorical basin axis', () => {
    // Object/array-valued levers are inherently categorical — each distinct architecture is its own region.
    const runs: AnalysisRun[] = []
    const arches = [[64, 64], [256, 256]]
    let seed = 0
    for (const net of arches) for (const lr of [0.3, 0.5, 0.7]) for (const s of [0, 1, 2, 3, 4]) {
      const score = (net[0] === 256 ? 500 : 470) - 400 * (lr - 0.5) ** 2
      runs.push({ key: `n-${net[0]}-${lr}-${seed++}`, config: { net_arch: net, lr, seed: s }, objective: score, metrics: { score, baseline: 20 }, seed: s, status: 'completed' })
    }
    const basins = clusterBasins(runs, { key: 'objective', direction: 'max' }, ['net_arch', 'lr'], 1, 20)
    // one basin per distinct architecture (both clear the baseline), keyed by the array value
    expect(basins.length).toBe(2)
    const declared = basins[0] // sorted best-first
    expect(JSON.stringify(declared.region.net_arch)).toBe('[256,256]')
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
// A plateau in the current subspace must ESCALATE — unfreeze a fixed lever, then keep SPACE-FILLING the active
// numeric space to a density target — and only converge once every lever is unfrozen AND the space is covered.
describe('escalation ladder — never converge until the space is covered', () => {
  // A manifest with NO inert lever, so `frozenLevers: {}` is a realistic full state (every searchable lever
  // active) and rung-1 (unfreeze) is legitimately empty — isolating the coverage rung.
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
  // configs emitted by a coverage rec (space-filling samples ride in spec.configs, not a per-lever sweep)
  const coverageLrs = (step: ReturnType<typeof nextExplorationStep>): number[] =>
    step.batch.flatMap((b) => (b.spec.configs ?? []).map((c) => Number(c.config.lr)))
  // every lr value the step proposes, whether via a coverage config or a coordinate sweep
  const proposedLrs = (step: ReturnType<typeof nextExplorationStep>): number[] =>
    step.batch.flatMap((b) => [
      ...(b.spec.configs ?? []).map((c) => Number(c.config.lr)),
      ...(((b.spec.sweep?.lr as number[]) ?? []).map(Number)),
    ])

  it('UNFREEZES a fixed numeric lever instead of converging when all basins plateaued (rung 1: widen)', () => {
    // A is tightly resolved on lr (bracket 0.48–0.52 ⇒ plateaued), and noise_knob is frozen — so instead of
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

  it('SPACE-FILLS the under-sampled range instead of converging, without auto-deepening the resolution', () => {
    // The single basin on lr is locally resolved (bunched 0.48–0.52), but far fewer than the coverage target
    // of distinct setups exist — so the search must sample the REST of the range, and NOT bump refineDepth
    // (deepening is user-driven via "Explore more", not automatic).
    runSeq = 0
    const runs: AnalysisRun[] = []
    for (const lr of [0.48, 0.49, 0.5, 0.51, 0.52]) for (const s of [0, 1, 2, 3, 4]) runs.push(aRun(M2, lr, s))
    const state: ExplorationState = {
      ...initExplorationState(M2),
      stage: 'local',
      activeLevers: ['algo', 'lr'],
      frozenLevers: {},
      noiseFloor: 1,
    }
    const step = nextExplorationStep(state, runs, M2, {})
    expect(step.done).toBe(false)
    expect(step.stateNext.refineDepth ?? 0).toBe(0) // NOT auto-deepened
    const lrs = coverageLrs(step)
    expect(lrs.length).toBeGreaterThan(0)
    expect(lrs.some((v) => v < 0.4 || v > 0.6)).toBe(true) // reaching out into the untried range
  })

  // objective is irrelevant to coverage (it samples space, not maxima), so a helper for arbitrary algo/lr runs.
  const run2 = (algo: string, lr: number, s: number): AnalysisRun => ({
    key: `rr-${algo}-${lr}-${s}`,
    config: { algo, lr, seed: s },
    objective: 100,
    metrics: { score: 100, baseline: 20 },
    seed: s,
    status: 'completed',
  })

  it('covers EACH categorical region to the density target — never abandons an under-sampled value (dqn vs ppo)', () => {
    const runs: AnalysisRun[] = []
    for (let i = 0; i < 20; i++) for (const s of [0, 1, 2, 3, 4]) runs.push(run2('A', i / 20, s)) // A: densely covered
    for (const lr of [0.5, 0.6]) for (const s of [0, 1, 2, 3, 4]) runs.push(run2('B', lr, s)) // B: only 2 setups
    const state: ExplorationState = { ...initExplorationState(M2), stage: 'local', activeLevers: ['algo', 'lr'], noiseFloor: 1 }
    const recs = coverageGridRecs(state, runs, M2, { key: 'objective', direction: 'max' })
    const configs = recs.flatMap((r) => r.spec.configs ?? [])
    expect(configs.length).toBeGreaterThan(0)
    // it samples the UNDER-covered region B (not the already-saturated A)
    expect(configs.every((c) => c.config.algo === 'B')).toBe(true)
    expect(new Set(configs.map((c) => Number(c.config.lr))).size).toBeGreaterThan(1) // real spread within B
  })

  it('round-robins coverage across multiple under-covered regions so EVERY value progresses each round', () => {
    const runs: AnalysisRun[] = []
    for (const algo of ['A', 'B', 'C']) for (const lr of [0.5, 0.6]) for (const s of [0, 1, 2, 3, 4]) runs.push(run2(algo, lr, s))
    const state: ExplorationState = { ...initExplorationState(M2), stage: 'local', activeLevers: ['algo', 'lr'], noiseFloor: 1 }
    const recs = coverageGridRecs(state, runs, M2, { key: 'objective', direction: 'max' })
    const configs = recs.flatMap((r) => r.spec.configs ?? [])
    expect(configs.length).toBeLessThanOrEqual(EXPLORATION_BATCH_MAX) // one round is bounded by the batch cap
    const byRegion = new Set(configs.map((c) => String(c.config.algo)))
    expect(byRegion).toEqual(new Set(['A', 'B', 'C'])) // all three regions represented in the same round
  })

  it('rotates the round-robin start so MORE regions than the batch cap are NOT permanently starved', () => {
    // 26 categorical regions > EXPLORATION_BATCH_MAX (24): a single round can only serve 24, so the start must
    // ROTATE across rounds (by the growing setup count) or the last regions would never get sampled.
    const algos = Array.from({ length: 26 }, (_, i) => 'a' + i)
    const M26: TrainerManifest = {
      name: 'many', recordType: 'many-run', run: 'noop',
      objective: { name: 'score', direction: 'max' },
      levers: { algo: { type: 'choice', choices: algos, default: 'a0' }, lr: { type: 'number', range: [0, 1], default: 0.5 }, seed: { type: 'number', default: 0 } },
    }
    const mk = (extra: string[]): AnalysisRun[] => {
      const rs: AnalysisRun[] = []
      for (const algo of algos) for (const lr of [0.4, 0.6]) rs.push({ key: `m-${algo}-${lr}`, config: { algo, lr, seed: 0 }, objective: 100, metrics: { score: 100, baseline: 20 }, seed: 0, status: 'completed' })
      for (const k of extra) rs.push({ key: `x-${k}`, config: { algo: 'a0', lr: Number(k), seed: 0 }, objective: 100, metrics: { score: 100, baseline: 20 }, seed: 0, status: 'completed' })
      return rs
    }
    const state: ExplorationState = { ...initExplorationState(M26), stage: 'local', activeLevers: ['algo', 'lr'], noiseFloor: 1 }
    const served = (runs: AnalysisRun[]) => new Set(coverageGridRecs(state, runs, M26, { key: 'objective', direction: 'max' }).flatMap((r) => r.spec.configs ?? []).map((c) => String(c.config.algo)))
    const roundA = served(mk([])) // 52 setups → start 52%26 = 0
    const roundB = served(mk(['0.1'])) // 53 setups → start 53%26 = 1 → serves a different window
    expect(roundA.size).toBe(EXPLORATION_BATCH_MAX) // one round is capped
    expect([...roundA].sort()).not.toEqual([...roundB].sort()) // the served window ROTATED
    expect(roundB.has('a24')).toBe(true) // a region starved in round A is served in round B (no permanent starvation)
  })

  it('does NOT converge while the numeric range is still UNCOVERED — even with points clustered at the peak', () => {
    // The old logic called the basin plateaued and converged with 3 bunched runs; but 99% of the lr range is
    // untried, so the search must keep space-filling it.
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
    expect(step.done).toBe(false)
    expect(proposedLrs(step).some((v) => v < 0.4 || v > 0.6)).toBe(true) // reaching into the untried range
  })

  it('CONVERGES once the space is covered (density target met) AND the peak is resolved', () => {
    // A dense-near-peak + spread-across-range history: >= the coverage target of distinct lr setups, with the
    // peak bracketed tighter than the resolution floor — nothing left to sample, so it may finally converge.
    runSeq = 0
    const lrs = [0, 0.1, 0.2, 0.3, 0.4, 0.49, 0.495, 0.5, 0.505, 0.51, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    const runs: AnalysisRun[] = []
    for (const lr of lrs) for (const s of [0, 1, 2, 3, 4]) runs.push(aRun(M2, lr, s))
    const state: ExplorationState = {
      ...initExplorationState(M2),
      stage: 'local',
      activeLevers: ['algo', 'lr'],
      frozenLevers: {},
      noiseFloor: 1,
    }
    const step = nextExplorationStep(state, runs, M2, {})
    expect(step.done).toBe(true)
    expect(step.stage).toBe('converged')
    expect(step.rationale).toMatch(/covered/i)
    expect(step.stateNext.declaredBasinId).toBeTruthy()
  })

  it('coverageGridRecs space-fills up to the density target, then goes empty (the convergence gate)', () => {
    // < target distinct setups ⇒ it proposes space-filling configs; >= target ⇒ nothing (space covered).
    runSeq = 0
    const few: AnalysisRun[] = [0.5].flatMap((lr) => [0, 1, 2, 3, 4].map((s) => aRun(M2, lr, s)))
    const many: AnalysisRun[] = []
    for (let i = 0; i < 20; i++) for (const s of [0, 1, 2, 3, 4]) many.push(aRun(M2, i / 19, s))
    const state: ExplorationState = { ...initExplorationState(M2), stage: 'local', activeLevers: ['algo', 'lr'], noiseFloor: 1 }
    const sparse = coverageGridRecs(state, few, M2, { key: 'objective', direction: 'max' })
    const covered = coverageGridRecs(state, many, M2, { key: 'objective', direction: 'max' })
    expect(sparse.flatMap((r) => r.spec.configs ?? []).length).toBeGreaterThan(0)
    expect(covered).toEqual([]) // 20 spread setups >= the 16 target for one lever at depth 0
  })

  it('coverageGridRecs IGNORES a degenerate-range lever so the convergence gate can still close', () => {
    // A `number` lever whose range is a single point [0.5,0.5] can never yield distinct setups — if it counted
    // toward coverage, the target could never be met and the search would never converge. It must be excluded.
    const M: TrainerManifest = {
      name: 'degen', recordType: 'degen-run', run: 'noop',
      objective: { name: 'score', direction: 'max' },
      levers: { algo: { type: 'choice', choices: ['A', 'B'], default: 'A' }, pinned: { type: 'number', range: [0.5, 0.5], default: 0.5 }, seed: { type: 'number', default: 0 } },
    }
    const runs: AnalysisRun[] = [0, 1, 2, 3, 4].map((s) => ({ key: `d-${s}`, config: { algo: 'A', pinned: 0.5, seed: s }, objective: 500, metrics: { score: 500, baseline: 20 }, seed: s, status: 'completed' }))
    const state: ExplorationState = { ...initExplorationState(M), stage: 'local', activeLevers: ['algo', 'pinned'], noiseFloor: 1 }
    expect(coverageGridRecs(state, runs, M, { key: 'objective', direction: 'max' })).toEqual([]) // no coverable numeric axis
  })

  it('coverageGridRecs never emits DUPLICATE configs and closes once the representable space is saturated', () => {
    // A tiny range [0.5, 0.5000006]: at 1e-6 quantization only ~2 distinct points exist, far fewer than the
    // coverage target — the gate must still close (return []) rather than re-propose the same configs forever.
    const M: TrainerManifest = {
      name: 'tiny', recordType: 'tiny-run', run: 'noop',
      objective: { name: 'score', direction: 'max' },
      levers: { lr: { type: 'number', range: [0.5, 0.5000006], default: 0.5 }, seed: { type: 'number', default: 0 } },
    }
    const runs: AnalysisRun[] = [0.5, 0.5000003, 0.5000006].flatMap((lr) => [0, 1, 2, 3, 4].map((s) => ({ key: `t-${lr}-${s}`, config: { lr, seed: s }, objective: 500, metrics: { score: 500, baseline: 20 }, seed: s, status: 'completed' })))
    const state: ExplorationState = { ...initExplorationState(M), stage: 'local', activeLevers: ['lr'], noiseFloor: 1 }
    const recs = coverageGridRecs(state, runs, M, { key: 'objective', direction: 'max' })
    const configs = recs.flatMap((r) => r.spec.configs ?? [])
    const sigs = configs.map((c) => Number(c.config.lr).toFixed(6))
    expect(new Set(sigs).size).toBe(sigs.length) // no duplicate configs proposed
    for (const s of sigs) expect(['0.500000', '0.500001']).toContain(s) // only representable points, none re-tried at 0.500000
  })

  it('coverage samples do NOT pin the champion run seed (coverage is for breadth; seeds are added when climbing)', () => {
    runSeq = 0
    const runs: AnalysisRun[] = [0.5].flatMap((lr) => [7].map((s) => aRun(M2, lr, s))) // champion ran at seed 7
    const state: ExplorationState = { ...initExplorationState(M2), stage: 'local', activeLevers: ['algo', 'lr'], noiseFloor: 1 }
    const recs = coverageGridRecs(state, runs, M2, { key: 'objective', direction: 'max' })
    const configs = recs.flatMap((r) => r.spec.configs ?? [])
    expect(configs.length).toBeGreaterThan(0)
    for (const c of configs) expect(c.config.seed).toBeUndefined() // seed not carried from the champion
  })

  it('unfreezing a lever whose range is ALREADY fully sampled re-enters global search rather than re-proposing it', () => {
    // noise_knob is frozen but its 5 space-filling sweep points (0.1/0.3/0.5/0.7/0.9) are all already tried, so the
    // unfreeze rung has nothing fresh to sweep — it must still widen (add the lever to active) and re-enter global.
    runSeq = 0
    const runs: AnalysisRun[] = []
    const nk = [0.1, 0.3, 0.5, 0.7, 0.9]
    for (const lr of [0.48, 0.49, 0.5, 0.51, 0.52]) for (const s of [0, 1, 2, 3, 4]) runs.push(evaluate({ algo: 'A', lr, noise_knob: nk[s], seed: s }))
    const state: ExplorationState = { ...initExplorationState(MANIFEST), stage: 'local', activeLevers: ['algo', 'lr'], frozenLevers: { noise_knob: 0.5 }, noiseFloor: 1 }
    const step = nextExplorationStep(state, runs, MANIFEST, {})
    expect(step.done).toBe(false)
    expect(step.stateNext.activeLevers).toContain('noise_knob') // widened despite nothing fresh to sweep
    expect(Object.keys(step.stateNext.frozenLevers)).not.toContain('noise_knob')
  })

  it('declares "fully covered at the finest resolution" once refineDepth is at the cap and nothing remains', () => {
    // No numeric active lever (only the categorical basin axis) ⇒ coverage is trivially satisfied; at the max
    // refineDepth the convergence message reflects the finest resolution (the terminal of "Explore more").
    runSeq = 0
    const runs: AnalysisRun[] = [0, 1, 2, 3, 4].map((s) => evaluate({ algo: 'A', lr: 0.5, seed: s }))
    const state: ExplorationState = {
      ...initExplorationState(MANIFEST),
      stage: 'local',
      activeLevers: ['algo'],
      frozenLevers: {},
      refineDepth: EXPLORATION_MAX_REFINE_DEPTH,
      noiseFloor: 1,
    }
    const step = nextExplorationStep(state, runs, MANIFEST, {})
    expect(step.done).toBe(true)
    expect(step.stage).toBe('converged')
    expect(step.rationale).toMatch(/finest resolution/i)
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
    // Objective barely above baseline (gain 2 < the min-span margin) ⇒ clusterBasins finds nothing; yet with the
    // space covered (>= target spread setups) the run history has a clear best — never report "no maximum found".
    const runs: AnalysisRun[] = []
    for (let i = 0; i < 20; i++) for (const s of [0, 1, 2, 3, 4]) {
      runs.push({ key: `w-${i}-${s}`, config: { algo: 'A', lr: i / 19, seed: s }, objective: 22, metrics: { score: 22, baseline: 20 }, seed: s, status: 'completed' })
    }
    expect(clusterBasins(runs, { key: 'objective', direction: 'max' }, ['algo', 'lr'], 1, 20)).toEqual([])
    const state: ExplorationState = { ...initExplorationState(M2), stage: 'local', activeLevers: ['algo', 'lr'], frozenLevers: {}, noiseFloor: 1 }
    const step = nextExplorationStep(state, runs, M2, {})
    expect(step.done).toBe(true)
    expect(step.stateNext.basins.length).toBeGreaterThanOrEqual(1) // a fallback basin synthesised from the best run
    expect(step.stateNext.declaredBasinId).toBeTruthy()
  })
})

describe('stale state heals when the run archive is emptied', () => {
  it('restarts from calibrate when a CONVERGED state has zero runs (the user deleted every run)', () => {
    const converged: ExplorationState = {
      ...initExplorationState(MANIFEST),
      stage: 'converged',
      done: true,
      activeLevers: ['algo', 'lr'],
      frozenLevers: { noise_knob: 0.5 },
      basins: [{ id: 'algo=A', region: { algo: 'A' }, centerConfig: { algo: 'A', lr: 0.5 }, peakObjective: 500, peakSeeds: 5, plateaued: true, memberRunKeys: [] }],
      declaredBasinId: 'algo=A',
      refineDepth: 3,
    }
    const step = nextExplorationStep(converged, [], MANIFEST, {})
    expect(step.done).toBe(false)
    expect(step.stage).toBe('calibrate') // NOT a re-declared convergence over runs that no longer exist
    expect(step.batch.length).toBeGreaterThan(0) // re-measures the noise floor on the default config
    expect(step.stateNext.basins).toEqual([])
    expect(step.stateNext.declaredBasinId).toBeUndefined()
    expect(step.stateNext.refineDepth).toBe(0)
  })

  it('a zero-run state with a spent budget still converges (budget takes precedence over the reset)', () => {
    const spent: ExplorationState = { ...initExplorationState(MANIFEST, { maxRuns: 0 }), stage: 'global' }
    const step = nextExplorationStep(spent, [], MANIFEST)
    expect(step.done).toBe(true)
    expect(step.stage).toBe('converged')
  })

  it('a CONVERGED state with a populated archive stays converged (idempotent re-entry, declaration preserved)', () => {
    runSeq = 0
    const runs = [0.5].flatMap((lr) => [0, 1, 2, 3, 4].map((s) => evaluate({ algo: 'A', lr, noise_knob: 0.5, seed: s })))
    const converged: ExplorationState = {
      ...initExplorationState(MANIFEST),
      stage: 'converged',
      done: true,
      activeLevers: ['algo', 'lr'],
      basins: [{ id: 'algo=A', region: { algo: 'A' }, centerConfig: { algo: 'A', lr: 0.5 }, peakObjective: 500, peakSeeds: 5, plateaued: true, memberRunKeys: [] }],
      declaredBasinId: 'algo=A',
    }
    const step = nextExplorationStep(converged, runs, MANIFEST, {})
    expect(step.done).toBe(true)
    expect(step.stage).toBe('converged')
    expect(step.batch).toEqual([]) // emits nothing — re-entry is idempotent, never re-spawns work
    expect(step.stateNext.declaredBasinId).toBe('algo=A') // the declared maximum is preserved
  })
})

describe('escalation ladder end-to-end', () => {
  it('drives unfreeze → coverage → converge: a FROZEN lever is unfrozen, the space is covered, then it converges', () => {
    runSeq = 0
    // Seed a resolved lr peak for algo A so `local` plateaus immediately and the escalation ladder engages; the
    // inert noise_knob starts FROZEN and must be unfrozen (rung 1) before the search can honestly converge.
    const runs: AnalysisRun[] = []
    for (const lr of [0.4, 0.45, 0.5, 0.55, 0.6]) for (const s of [0, 1, 2, 3, 4]) runs.push(evaluate({ algo: 'A', lr, noise_knob: 0.5, seed: s }))
    let state: ExplorationState = {
      ...initExplorationState(MANIFEST, { maxRuns: 1200 }),
      stage: 'local',
      activeLevers: ['algo', 'lr'],
      frozenLevers: { noise_knob: 0.5 },
      noiseFloor: 1,
    }
    let unfroze = false
    let climbedNoiseKnob = false
    let rounds = 0
    while (!state.done && rounds < 300) {
      const step = nextExplorationStep(state, runs, MANIFEST, { targetObjective: 500 })
      state = step.stateNext
      if (state.activeLevers.includes('noise_knob')) unfroze = true
      // the freed lever is actually PROBED (swept or space-filled), not merely marked active
      if (step.batch.some((r) => (r.spec.sweep && 'noise_knob' in r.spec.sweep) || (r.spec.configs ?? []).length)) climbedNoiseKnob = true
      for (const rec of step.batch) for (const cfg of expandSpec(rec.spec, MANIFEST)) runs.push(evaluate(cfg))
      rounds++
    }
    expect(unfroze).toBe(true) // rung 1 widened into the previously-fixed lever
    expect(climbedNoiseKnob).toBe(true) // and the search actually probed the widened/covered space
    expect(state.done).toBe(true) // and only THEN converged
    expect(Object.keys(state.frozenLevers)).not.toContain('noise_knob')
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

// Plan §7 step 6 — the Wine reproducibility pass, done as a pure test: a MODEL-choice tabular surface (the shipped
// `model_name` basin axis) is driven end-to-end MANY times under DIFFERENT run-to-run noise realisations, and the
// autopilot must declare the SAME best model every time (zero declaration variance) — the property live Wine reruns
// are meant to confirm. `model_name` is the categorical basin axis; `learning_rate` is the numeric climb dimension.
describe('reproducibility — repeated autopilot runs declare the same model basin despite noise', () => {
  const TAB: TrainerManifest = {
    name: 'tabular-model-choice',
    recordType: 'tabular-run',
    run: 'noop',
    objective: { name: 'val_rmse', direction: 'min' },
    levers: {
      model_name: { type: 'choice', choices: ['gradient_boosting', 'random_forest', 'hist_gradient_boosting'], default: 'gradient_boosting' },
      learning_rate: { type: 'number', range: [0.005, 0.5], default: 0.1 },
      seed: { type: 'number', default: 0 },
    },
  }
  const TAB_BASELINE = 0.8
  // hist_gradient_boosting is the true best (lowest rmse), with a learning_rate optimum near 0.05. `phase` shifts
  // the deterministic seed jitter so each repeat sees a DIFFERENT noise realisation (never a seedless RNG).
  const rmseFor = (config: Record<string, unknown>, phase: number): number => {
    const model = String(config.model_name)
    const lr = Number(config.learning_rate ?? 0.1)
    const seed = Number(config.seed ?? 0)
    const floor = model === 'hist_gradient_boosting' ? 0.30 : model === 'gradient_boosting' ? 0.40 : 0.50
    const jitter = ((((seed + phase) * 37) % 11) - 5) * 0.004 // in [-0.02, 0.024]
    return floor + 1.2 * (lr - 0.05) ** 2 + jitter
  }
  const declaredModel = (phase: number): { model: string; runs: number } => {
    let n = 0
    let state = initExplorationState(TAB, { maxRuns: 400 })
    const runs: AnalysisRun[] = []
    let rounds = 0
    while (!state.done && rounds < 200) {
      const step = nextExplorationStep(state, runs, TAB, { targetObjective: 0.3 })
      state = step.stateNext
      for (const rec of step.batch)
        for (const cfg of expandSpec(rec.spec, TAB)) {
          const v = rmseFor(cfg, phase)
          runs.push({ key: `t-${phase}-${n++}`, config: { ...cfg }, objective: v, metrics: { val_rmse: v, baseline: TAB_BASELINE }, seed: Number(cfg.seed ?? 0), status: 'completed' })
        }
      rounds++
    }
    const declared = state.basins.find((b) => b.id === state.declaredBasinId)
    return { model: declared ? String(declared.region.model_name) : '(none)', runs: runs.length }
  }

  it('declares hist_gradient_boosting every time across 5 independent noise realisations', () => {
    const declarations = [0, 1, 2, 3, 4].map((phase) => declaredModel(phase))
    const models = declarations.map((d) => d.model)
    // ZERO declaration variance — the same winning model every run
    expect(new Set(models).size).toBe(1)
    expect(models[0]).toBe('hist_gradient_boosting')
    // and every run genuinely searched the space (not an early dead-end): comfortably more than calibrate+screen
    for (const d of declarations) expect(d.runs).toBeGreaterThan(40)
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

describe('gateConvergenceOnSplits (A5 split-consistency convergence gate)', () => {
  const criterion = { key: 'objective', direction: 'max' as const }
  const mk = (stage: any, batch: any, rationale: any, stateNext: any, done: any) => ({
    stage,
    batch,
    rationale,
    stateNext,
    done,
  })
  const state = { stage: 'local', budget: {} } as any
  const converged = {
    stage: 'converged',
    batch: [],
    rationale: 'converged: done',
    stateNext: state,
    done: true,
  } as any
  const manifest = { diagnostics: { splitAxis: { levers: ['window'] } }, levers: {} } as any
  const run = (config: any, objective: number, seed = 0) => ({
    key: JSON.stringify(config) + seed,
    config,
    objective,
    seed,
    status: 'completed',
  })

  it('replaces a not-yet-robust convergence with split-fill recs when unrun splits remain', () => {
    const runs = [
      run({ lr: 1, window: '2024' }, 20), // incumbent (best), only evaluated on 2024
      run({ lr: 2, window: '2022' }, 1),
      run({ lr: 2, window: '2023' }, 1),
    ]
    const out = gateConvergenceOnSplits(converged, state, runs, manifest, criterion, true, mk)
    expect(out.done).toBe(false)
    expect(out.stage).toBe('local')
    expect(out.batch.map((b: any) => b.kind)).toEqual(['missing-cell', 'missing-cell'])
    // Replicates the INCUMBENT (lr:1), not the field, across the missing windows.
    const fixeds = out.batch.map((b: any) => b.spec.fixed)
    expect(fixeds).toContainEqual({ lr: 1, window: '2022' })
    expect(fixeds).toContainEqual({ lr: 1, window: '2023' })
  })

  it('lets a fully-replicated incumbent converge (no unrun splits to fill)', () => {
    const runs = [run({ lr: 1, window: '2024' }, 8), run({ lr: 1, window: '2022' }, 3)]
    expect(gateConvergenceOnSplits(converged, state, runs, manifest, criterion, true, mk)).toBe(converged)
  })

  it('bypasses the gate when budget is exhausted', () => {
    const runs = [run({ lr: 1, window: '2024' }, 20), run({ lr: 2, window: '2022' }, 1)]
    expect(gateConvergenceOnSplits(converged, state, runs, manifest, criterion, false, mk)).toBe(converged)
  })

  it('is a no-op when the manifest declares no split axis', () => {
    const runs = [run({ lr: 1, window: '2024' }, 20), run({ lr: 2, window: '2022' }, 1)]
    const noSplit = { levers: {} } as any
    expect(gateConvergenceOnSplits(converged, state, runs, noSplit, criterion, true, mk)).toBe(converged)
  })

  it('never touches a non-converged step', () => {
    const inProgress = { stage: 'local', batch: [], rationale: 'x', stateNext: state, done: false } as any
    const runs = [run({ lr: 1, window: '2024' }, 20)]
    expect(gateConvergenceOnSplits(inProgress, state, runs, manifest, criterion, true, mk)).toBe(inProgress)
  })
})
