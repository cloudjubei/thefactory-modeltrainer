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
  it('keeps EVERY run per X/Y cell (for subdivisions), sorted hottest-first, with empty cells present', () => {
    const state = { activeLevers: ['algo', 'x', 'y'], basins: [], stage: 'global', budget: { spentRuns: 30 } }
    const a = Exploration.analyze({ manifest: { ...MANIFEST, recordType: 'cells-a' }, state, runs: runsFixture() })
    expect(a.vs.axisX).toBe('x')
    expect(a.vs.axisY).toBe('y')
    const { xA, yA, cells } = Exploration.heatmapCells(a)
    // algo A and B share each (x,y) coordinate → that cell holds BOTH runs (they differ only on algo)
    const gi = xA.index(0.5)
    const gj = yA.index(0.5)
    const cell = cells[gj * xA.n + gi]
    expect(cell.runs.length).toBe(2)
    expect(cell.runs[0].t).toBeGreaterThanOrEqual(cell.runs[1].t) // hottest-first
    expect(cell.best).toBe(cell.runs[0].t)
    // the grid is mostly empty (24×24 numeric bins, only 15 distinct coords populated)
    expect(cells.some((c: any) => c.runs.length === 0)).toBe(true)
    // total runs across cells equals the in-grid run count
    const totalInCells = cells.reduce((n: number, c: any) => n + c.runs.length, 0)
    expect(totalInCells).toBe(30)
  })

  it('pegging a lever filters which runs populate the grid', () => {
    const state = { activeLevers: ['algo', 'x', 'y'], basins: [], stage: 'global', budget: { spentRuns: 30 } }
    const a = Exploration.analyze({ manifest: { ...MANIFEST, recordType: 'cells-peg' }, state, runs: runsFixture() })
    a.vs.pegs = { algo: 'A' }
    const { cells } = Exploration.heatmapCells(a)
    const total = cells.reduce((n: number, c: any) => n + c.runs.length, 0)
    expect(total).toBe(15) // half the runs (algo=A only)
  })

  it('makeAxis exposes a cellLabel: numeric → a range, categorical → the value', () => {
    const numA = Exploration.makeAxis('x', runsFixture(), MANIFEST, 1)
    expect(numA.kind).toBe('num')
    expect(String(numA.cellLabel(0))).toContain('–')
    const catA = Exploration.makeAxis('algo', runsFixture(), MANIFEST, 1)
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
