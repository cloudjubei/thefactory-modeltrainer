import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/scorecard.js is the no-build browser twin of the engine's computeScorecard; load it as CommonJS
// (same mechanism as hypothesisViewer.test.ts) and assert it matches the engine on the same cases, so the
// viewer and the server agree on which runs are "good".
const require = createRequire(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'scorecard.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const S: any = mod.exports

const objMax = { objective: { name: 'ret', direction: 'max' } }

describe('viewer computeScorecard', () => {
  it('collapses to the objective when no gates/fitness (accepted, fitness = objective)', () => {
    const card = S.computeScorecard(objMax, { objective: 42 })
    expect(card.gates).toEqual([])
    expect(card.accepted).toBe(true)
    expect(card.fitness).toEqual([{ metric: 'objective', direction: 'max', value: 42 }])
  })

  it('passes a literal gate and accepts', () => {
    const card = S.computeScorecard(
      { ...objMax, gates: [{ metric: 'oos_return_pct', op: '>', value: 0 }] },
      { objective: 1, metrics: { oos_return_pct: 5 } },
    )
    expect(card.gates[0]).toMatchObject({ bound: 0, actual: 5, pass: true })
    expect(card.accepted).toBe(true)
  })

  it('SKIPS a gate on an absent metric (not applicable) but does NOT vacuously accept when it is the only gate', () => {
    const card = S.computeScorecard({ ...objMax, gates: [{ metric: 'gone', op: '>', value: 0 }] }, { objective: 1, metrics: { other: 1 } })
    expect(card.gates[0].applicable).toBe(false)
    expect(card.gates[0].pass).toBe(false)
    expect(card.accepted).toBe(false) // no applicable gate ⇒ can't verify ⇒ not accepted
  })

  it('accepts when a skipped gate sits alongside a PASSING applicable gate', () => {
    const gates = [{ metric: 'gone', op: '>', value: 0 }, { metric: 'ret', op: '>', value: 0 }]
    const card = S.computeScorecard({ ...objMax, gates }, { objective: 1, metrics: { ret: 5 } })
    expect(card.accepted).toBe(true)
  })

  it('never accepts a failed / invalid / degenerate run even when gates pass', () => {
    const gates = [{ metric: 'ret', op: '>', value: 0 }]
    const metrics = { ret: 5 }
    expect(S.computeScorecard({ ...objMax, gates }, { objective: 1, metrics }).accepted).toBe(true)
    expect(S.computeScorecard({ ...objMax, gates }, { objective: 1, metrics, status: 'failed' }).accepted).toBe(false)
    expect(S.computeScorecard({ ...objMax, gates }, { objective: 1, metrics, health: { status: 'degenerate' } }).accepted).toBe(false)
  })

  it('FAILS a gate on a present but non-finite metric (measured garbage is not skipped)', () => {
    const card = S.computeScorecard({ ...objMax, gates: [{ metric: 'm', op: '>', value: 0 }] }, { objective: 1, metrics: { m: NaN } })
    expect(card.gates[0]).toMatchObject({ applicable: true, pass: false })
    expect(card.accepted).toBe(false)
  })

  it('resolves a metric-vs-metric gate bound (beat hold)', () => {
    const gate = { metric: 'oos_return_pct', op: '>', value: { metric: 'hold_return_pct' } }
    const card = S.computeScorecard({ ...objMax, gates: [gate] }, { objective: 1, metrics: { oos_return_pct: 10, hold_return_pct: 4 } })
    expect(card.gates[0]).toMatchObject({ bound: 4, actual: 10, pass: true, label: 'oos_return_pct > hold_return_pct' })
  })

  it('renders a default label and preserves a custom one', () => {
    const def = S.computeScorecard({ ...objMax, gates: [{ metric: 'm', op: '>=', value: 2 }] }, { objective: 1, metrics: { m: 3 } })
    expect(def.gates[0].label).toBe('m >= 2')
    const custom = S.computeScorecard({ ...objMax, gates: [{ metric: 'm', op: '>', value: 0, label: 'profitable' }] }, { objective: 1, metrics: { m: 1 } })
    expect(custom.gates[0].label).toBe('profitable')
  })

  it('builds a multi-objective fitness vector in declared order', () => {
    const card = S.computeScorecard(
      { ...objMax, fitness: [{ metric: 'a', direction: 'max' }, { metric: 'b', direction: 'min' }] },
      { objective: 1, metrics: { a: 2, b: 3 } },
    )
    expect(card.fitness).toEqual([
      { metric: 'a', direction: 'max', value: 2 },
      { metric: 'b', direction: 'min', value: 3 },
    ])
  })
})

describe('viewer primaryFitnessCriterion', () => {
  it('falls back to the objective', () => {
    expect(S.primaryFitnessCriterion(objMax)).toEqual({ key: 'objective', direction: 'max' })
  })
  it('uses the first fitness objective', () => {
    expect(S.primaryFitnessCriterion({ ...objMax, fitness: [{ metric: 'sharpe', direction: 'max' }] })).toEqual({
      key: 'sharpe',
      direction: 'max',
    })
  })
})

describe('viewer scorecardRankValue + compareScorecards', () => {
  const acc = (value: number) => ({ gates: [], accepted: true, fitness: [{ metric: 'x', direction: 'max', value }] })
  const rej = (value: number) => ({ gates: [], accepted: false, fitness: [{ metric: 'x', direction: 'max', value }] })

  it('orients min objectives so higher-is-better', () => {
    expect(S.scorecardRankValue({ gates: [], accepted: true, fitness: [{ metric: 'x', direction: 'min', value: 7 }] })).toBe(-7)
  })
  it('is -Infinity when unrankable', () => {
    expect(S.scorecardRankValue({ gates: [], accepted: true, fitness: [] })).toBe(-Infinity)
  })
  it('sorts accepted-first then by primary fitness', () => {
    const sorted = [rej(50), acc(2), acc(8), rej(90)].sort(S.compareScorecards)
    expect(sorted.map((c: any) => c.fitness[0].value)).toEqual([8, 2, 90, 50])
  })
})

describe('viewer scorecardSortValue (numeric column sort)', () => {
  const acc = (value: number) => ({ gates: [], accepted: true, fitness: [{ metric: 'x', direction: 'max', value }] })
  const rej = (value: number) => ({ gates: [], accepted: false, fitness: [{ metric: 'x', direction: 'max', value }] })

  it('ranks EVERY accepted run above EVERY rejected run, however large the rejected fitness', () => {
    expect(S.scorecardSortValue(acc(-999))).toBeGreaterThan(S.scorecardSortValue(rej(1e9)))
  })

  it('orders by primary fitness within the same acceptance tier', () => {
    expect(S.scorecardSortValue(acc(8))).toBeGreaterThan(S.scorecardSortValue(acc(2)))
    expect(S.scorecardSortValue(rej(8))).toBeGreaterThan(S.scorecardSortValue(rej(2)))
  })

  it('is finite even for an unrankable (non-finite fitness) run', () => {
    expect(Number.isFinite(S.scorecardSortValue(acc(NaN)))).toBe(true)
    expect(Number.isFinite(S.scorecardSortValue({ gates: [], accepted: true, fitness: [] }))).toBe(true)
  })

  it('a descending numeric sort by this value groups accepted-first then best fitness', () => {
    const cards = [rej(50), acc(2), acc(8), rej(90)]
    const sorted = [...cards].sort((a, b) => S.scorecardSortValue(b) - S.scorecardSortValue(a))
    expect(sorted.map((c: any) => c.fitness[0].value)).toEqual([8, 2, 90, 50])
  })
})

describe('viewer selectActiveScorecard', () => {
  const manifest = { objective: { name: 'ret', direction: 'max' }, gates: [{ metric: 'a', op: '>', value: 0 }], fitness: [{ metric: 'a', direction: 'max' }] }
  const cardX = { id: 'x', gates: [{ metric: 'b', op: '>', value: 1 }], fitness: [{ metric: 'b', direction: 'min' }] }
  const cardY = { id: 'y', gates: [{ metric: 'c', op: '<', value: 2 }], fitness: [{ metric: 'c', direction: 'max' }] }

  it('falls back to the manifest when there are no cards', () => {
    expect(S.selectActiveScorecard(manifest, [], null)).toEqual({ objective: manifest.objective, gates: manifest.gates, fitness: manifest.fitness })
  })
  it('returns the active card by id, grafting the manifest objective', () => {
    expect(S.selectActiveScorecard(manifest, [cardX, cardY], 'y')).toEqual({ objective: manifest.objective, gates: cardY.gates, fitness: cardY.fitness })
  })
  it('falls back to the first card when the active id is unknown', () => {
    expect(S.selectActiveScorecard(manifest, [cardX, cardY], 'gone').gates).toEqual(cardX.gates)
  })
  it('matches the engine twin: an active card overrides the manifest gates in computeScorecard', () => {
    const active = S.selectActiveScorecard(manifest, [cardX], 'x')
    // card x gates on metric b>1; a run with b=5 passes it, regardless of the manifest's a-gate.
    expect(S.computeScorecard(active, { objective: 1, metrics: { b: 5 } }).accepted).toBe(true)
    expect(S.computeScorecard(active, { objective: 1, metrics: { b: 0 } }).accepted).toBe(false)
  })
})

describe('viewer hasScorecard', () => {
  it('is false with no gates/fitness, true when either is declared', () => {
    expect(S.hasScorecard(objMax)).toBe(false)
    expect(S.hasScorecard({ ...objMax, gates: [{ metric: 'm', op: '>', value: 0 }] })).toBe(true)
    expect(S.hasScorecard({ ...objMax, fitness: [{ metric: 'm', direction: 'max' }] })).toBe(true)
  })
})
