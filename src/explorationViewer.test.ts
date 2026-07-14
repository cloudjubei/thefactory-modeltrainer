import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/exploration.js is the no-build browser Exploration view; load it as CommonJS the same way
// xaiViewer.test.ts loads viewer/xai.js, so the ACTUAL viewer logic (analyze/magma) is tested here.
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'exploration.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const Exploration: any = mod.exports

const MANIFEST = {
  recordType: 'demo-run',
  objective: { name: 'score', direction: 'max' as const },
  levers: {
    algo: { type: 'choice', choices: ['A', 'B'] },
    x: { type: 'number', range: [0, 1] },
    y: { type: 'number', range: [0, 1] },
    noise: { type: 'number', range: [0, 1] },
    seed: { type: 'number' },
  },
}
// x varies a lot, y moderately, noise barely — so the axes should be [x, y].
function runsFixture() {
  const runs: any[] = []
  let s = 0
  for (const algo of ['A', 'B']) {
    for (const x of [0, 0.25, 0.5, 0.75, 1]) {
      for (const y of [0.2, 0.5, 0.8]) {
        runs.push({ config: { algo, x, y, noise: 0.5, seed: s % 3 }, objective: 100 + x * 300 + y * 40 })
        s++
      }
    }
  }
  return runs
}

describe('Exploration.analyze', () => {
  it('chooses the two highest-ranked levers as the default X/Y axes (variance fallback, no Xai)', () => {
    const state = { activeLevers: ['algo', 'x', 'y'], basins: [], stage: 'global', budget: { spentRuns: 30 } }
    const a = Exploration.analyze({ manifest: { ...MANIFEST, recordType: 'axes-a' }, state, runs: runsFixture() })
    // x varies most, y next; noise is constant (filtered), seed excluded
    expect(a.vs.axisX).toBe('x')
    expect(a.vs.axisY).toBe('y')
    expect(a.rankedKeys).not.toContain('noise') // one observed value -> not an axis candidate
    expect(a.rankedKeys).not.toContain('seed')
    expect(a.dir).toBe('max')
  })

  it('falls back to model levers and sets two distinct axes when no state is present', () => {
    const a = Exploration.analyze({ manifest: { ...MANIFEST, recordType: 'axes-b' }, state: null, runs: runsFixture() })
    expect(a.vs.axisX).toBeTruthy()
    expect(a.vs.axisY).toBeTruthy()
    expect(a.vs.axisX).not.toBe(a.vs.axisY)
  })

  it('orients the color scale so a MIN objective maps its lowest value to the hot end', () => {
    const minManifest = { ...MANIFEST, recordType: 'min-c', objective: { name: 'rmse', direction: 'min' as const } }
    const runs = [
      { config: { algo: 'A', x: 0.1, y: 0.1 }, objective: 0.2 }, // best (lowest)
      { config: { algo: 'A', x: 0.9, y: 0.9 }, objective: 0.8 }, // worst (highest)
    ]
    const a = Exploration.analyze({ manifest: minManifest, state: null, runs })
    expect(a.nrm(0.2)).toBeCloseTo(1, 5) // lowest rmse → hottest
    expect(a.nrm(0.8)).toBeCloseTo(0, 5)
  })

  it('passes basins through and preserves the objective range', () => {
    const state = { basins: [{ id: 'b1', region: { algo: 'A' }, peakObjective: 420, centerConfig: { algo: 'A', x: 0.5, y: 0.5 } }], declaredBasinId: 'b1', stage: 'converged', budget: { spentRuns: 40 } }
    const a = Exploration.analyze({ manifest: { ...MANIFEST, recordType: 'basins-d' }, state, runs: runsFixture() })
    expect(a.basins).toHaveLength(1)
    expect(a.oMax).toBeGreaterThan(a.oMin)
  })
})

describe('Exploration.rankLevers', () => {
  it('ranks candidate axis levers (>1 observed value) by variance when no Xai engine is present', () => {
    const ranked = Exploration.rankLevers({ ...MANIFEST, recordType: 'rank-e' }, runsFixture())
    const keys = ranked.map((r: any) => r.lever)
    expect(keys[0]).toBe('x') // highest-variance numeric
    expect(keys).toContain('y')
    expect(keys).not.toContain('noise') // constant -> excluded
    // algo is categorical with >1 value → a candidate (kind 'cat')
    const algo = ranked.find((r: any) => r.lever === 'algo')
    expect(algo && algo.kind).toBe('cat')
  })
})

describe('Exploration.heatmapCells', () => {
  const cellsFor = (runs: any[], rt: string) => {
    const state = { activeLevers: ['algo', 'x', 'y'], basins: [], stage: 'global', budget: { spentRuns: runs.length } }
    const a = Exploration.analyze({ manifest: { ...MANIFEST, recordType: rt }, state, runs })
    a.vs.axisX = 'x'
    a.vs.axisY = 'y'
    return { a, ...Exploration.heatmapCells(a) }
  }

  it('bins on CONCRETE tried values (numeric axis = distinct sorted values, not range bins)', () => {
    const { xA, yA } = cellsFor(runsFixture(), 'cells-a')
    expect(xA.numeric).toBe(true)
    expect(xA.distinct).toEqual([0, 0.25, 0.5, 0.75, 1]) // the actual tried values, in order
    expect(yA.distinct).toEqual([0.2, 0.5, 0.8])
  })

  it('keeps EVERY run per X/Y cell (for subdivisions), sorted hottest-first', () => {
    const { xA, yA, cells } = cellsFor(runsFixture(), 'cells-b')
    const cell = cells[yA.index(0.5) * xA.n + xA.index(0.5)]
    expect(cell.runs.length).toBe(2) // algo A and B share coordinate (0.5,0.5)
    expect(cell.runs[0].t).toBeGreaterThanOrEqual(cell.runs[1].t)
    expect(cell.best).toBe(cell.runs[0].t)
    expect(cells.reduce((n: number, c: any) => n + c.runs.length, 0)).toBe(30)
  })

  it('leaves an untried (x,y) coordinate as an EMPTY cell even though both values exist on their axes', () => {
    const runs = runsFixture().filter((r) => !(r.config.x === 0.5 && r.config.y === 0.5))
    const { xA, yA, cells } = cellsFor(runs, 'cells-gap')
    expect(xA.distinct).toContain(0.5) // 0.5 still tried at other y's
    expect(yA.distinct).toContain(0.5)
    expect(cells[yA.index(0.5) * xA.n + xA.index(0.5)].runs.length).toBe(0) // the gap
  })

  it('pegging a lever filters which runs populate the grid', () => {
    const { a, cells } = cellsFor(runsFixture(), 'cells-peg')
    a.vs.pegs = { algo: 'A' }
    const { cells: pegged } = Exploration.heatmapCells(a)
    expect(pegged.reduce((n: number, c: any) => n + c.runs.length, 0)).toBe(15) // half (algo=A)
    expect(cells.reduce((n: number, c: any) => n + c.runs.length, 0)).toBe(30) // unpegged = all
  })

  it('makeAxis cellLabel is a CONCRETE value on both axis kinds (not a range)', () => {
    const numA = Exploration.makeAxis('x', runsFixture(), MANIFEST)
    expect(numA.kind).toBe('num')
    expect(String(numA.cellLabel(0))).not.toContain('–')
    expect(Number(numA.cellLabel(0))).toBe(0) // smallest tried x value
    const catA = Exploration.makeAxis('algo', runsFixture(), MANIFEST)
    expect(catA.kind).toBe('cat')
    expect(['A', 'B']).toContain(catA.cellLabel(0))
  })
})

describe('Exploration.magma', () => {
  it('returns an rgb() string and ramps from dark to light across [0,1]', () => {
    expect(Exploration.magma(0)).toMatch(/^rgb\(/)
    const lum = (s: string) => s.match(/\d+/g)!.map(Number).reduce((p: number, c: number) => p + c, 0)
    expect(lum(Exploration.magma(1))).toBeGreaterThan(lum(Exploration.magma(0)))
  })

  it('clamps out-of-range inputs', () => {
    expect(Exploration.magma(-5)).toBe(Exploration.magma(0))
    expect(Exploration.magma(5)).toBe(Exploration.magma(1))
  })
})
