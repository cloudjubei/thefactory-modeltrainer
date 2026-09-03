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
        runs.push({
          config: { algo, x, y, noise: 0.5, seed: s % 3 },
          objective: 100 + x * 300 + y * 40,
        })
        s++
      }
    }
  }
  return runs
}

describe('Exploration.analyze', () => {
  it('chooses the two highest-ranked levers as the default X/Y axes (variance fallback, no Xai)', () => {
    const state = {
      activeLevers: ['algo', 'x', 'y'],
      basins: [],
      stage: 'global',
      budget: { spentRuns: 30 },
    }
    const a = Exploration.analyze({
      manifest: { ...MANIFEST, recordType: 'axes-a' },
      state,
      runs: runsFixture(),
    })
    // x varies most, y next; noise is constant (filtered), seed excluded
    expect(a.vs.axisX).toBe('x')
    expect(a.vs.axisY).toBe('y')
    expect(a.rankedKeys).not.toContain('noise') // one observed value -> not an axis candidate
    expect(a.rankedKeys).not.toContain('seed')
    expect(a.dir).toBe('max')
  })

  it('falls back to model levers and sets two distinct axes when no state is present', () => {
    const a = Exploration.analyze({
      manifest: { ...MANIFEST, recordType: 'axes-b' },
      state: null,
      runs: runsFixture(),
    })
    expect(a.vs.axisX).toBeTruthy()
    expect(a.vs.axisY).toBeTruthy()
    expect(a.vs.axisX).not.toBe(a.vs.axisY)
  })

  it('orients the color scale so a MIN objective maps its lowest value to the hot end', () => {
    const minManifest = {
      ...MANIFEST,
      recordType: 'min-c',
      objective: { name: 'rmse', direction: 'min' as const },
    }
    const runs = [
      { config: { algo: 'A', x: 0.1, y: 0.1 }, objective: 0.2 }, // best (lowest)
      { config: { algo: 'A', x: 0.9, y: 0.9 }, objective: 0.8 }, // worst (highest)
    ]
    const a = Exploration.analyze({ manifest: minManifest, state: null, runs })
    expect(a.nrm(0.2)).toBeCloseTo(1, 5) // lowest rmse → hottest
    expect(a.nrm(0.8)).toBeCloseTo(0, 5)
  })

  it('passes basins through and preserves the objective range', () => {
    const state = {
      basins: [
        {
          id: 'b1',
          region: { algo: 'A' },
          peakObjective: 420,
          centerConfig: { algo: 'A', x: 0.5, y: 0.5 },
        },
      ],
      declaredBasinId: 'b1',
      stage: 'converged',
      budget: { spentRuns: 40 },
    }
    const a = Exploration.analyze({
      manifest: { ...MANIFEST, recordType: 'basins-d' },
      state,
      runs: runsFixture(),
    })
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

// A conditional lever (n/a for most runs, e.g. forward_horizon only on the supervised models) must not be
// treated as a broad axis — its applicable fraction is tracked and it stays a peggable but non-default lever.
const COND_MANIFEST = {
  recordType: 'cond-run',
  objective: { name: 'score', direction: 'max' as const },
  levers: {
    lr: { type: 'number', scope: 'model' },
    gamma: { type: 'number', scope: 'model' },
    cond: { type: 'number', scope: 'model' },
    seed: { type: 'number', scope: 'model' },
  },
}
function condRuns() {
  const runs: any[] = []
  for (let i = 0; i < 20; i++) {
    const applies = i < 2 // cond applies to only 2 of 20 runs, with extreme values (5, 500)
    runs.push({
      config: {
        lr: (i % 5) * 0.25,
        gamma: 0.9 + (i % 3) * 0.03,
        cond: applies ? (i === 0 ? 5 : 500) : 'n/a',
        seed: i,
      },
      objective: 100 + (i % 5) * 10,
    })
  }
  return runs
}

describe('Exploration conditional-lever handling', () => {
  it('rankLevers records the applicable fraction of each lever (n/a values excluded)', () => {
    const ranked = Exploration.rankLevers(COND_MANIFEST, condRuns())
    const cond = ranked.find((r: any) => r.lever === 'cond')
    const lr = ranked.find((r: any) => r.lever === 'lr')
    expect(cond.applicable).toBeCloseTo(0.1, 5) // 2 / 20
    expect(lr.applicable).toBe(1)
  })

  it('analyze keeps a mostly-n/a conditional lever OUT of the default axes but still peggable', () => {
    const a = Exploration.analyze({ manifest: COND_MANIFEST, state: null, runs: condRuns() })
    expect(a.vs.axisX).not.toBe('cond')
    expect(a.vs.axisY).not.toBe('cond')
    expect(a.rankedKeys).toContain('cond') // >1 real value → still a peggable lever
  })
})

describe('Exploration.heatmapCells', () => {
  const cellsFor = (runs: any[], rt: string) => {
    const state = {
      activeLevers: ['algo', 'x', 'y'],
      basins: [],
      stage: 'global',
      budget: { spentRuns: runs.length },
    }
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

  it('zoom = grid density: zoom<1 GROUPS values; zoom=1 ≈ one cell per value; zoom>1 SUBDIVIDES past the values', () => {
    // 30 distinct x values in [0, 29/30]. Bin count = round(distinct × zoom): coarse groups them, fine keeps
    // one-per-value, and — the zoom-IN fix — zoom>1 keeps ADDING bins beyond the value count (untried gaps).
    const runs = Array.from({ length: 30 }, (_, i) => ({
      config: { algo: 'A', x: i / 30, y: 0.5 },
      objective: 100 + i,
    }))
    const coarse = Exploration.makeAxis('x', runs, MANIFEST, 0.2)
    const finest = Exploration.makeAxis('x', runs, MANIFEST, 1)
    const finer = Exploration.makeAxis('x', runs, MANIFEST, 3)
    expect(coarse.n).toBe(6) // round(30 × 0.2)
    expect(finest.n).toBe(30) // round(30 × 1)
    expect(finer.n).toBe(90) // round(30 × 3) — zoom-in KEEPS growing the grid (the bug fix)
    expect(coarse.n).toBeLessThan(finest.n)
    expect(finer.n).toBeGreaterThan(finest.n)
    // every x lands in some coarse cell, and multi-value cells read as a real range (never "x–x")
    expect(coarse.index(0.0)).toBe(0)
    expect(coarse.index(29 / 30)).toBe(5)
    expect(coarse.labels.some((l: string) => l.includes('–'))).toBe(true)
    expect(coarse.labels.every((l: string) => !/^(\S+)–\1$/.test(l))).toBe(true)
  })
})

describe('Exploration.manualCellConfigs', () => {
  const M = {
    recordType: 'demo-run',
    objective: { name: 'score', direction: 'max' as const },
    levers: {
      algo: { type: 'choice', choices: ['A', 'B'], default: 'A' },
      x: { type: 'number', range: [0, 1], default: 0.5 },
      y: { type: 'number', range: [0, 1], default: 0.5 },
      noise: { type: 'number', range: [0, 1], default: 0.5 },
      seed: { type: 'number', default: 0 },
    },
  }
  const runs = [
    { config: { algo: 'A', x: 0.5, y: 0.5, noise: 0.9 }, objective: 400 },
    { config: { algo: 'B', x: 0.25, y: 0.75, noise: 0.1 }, objective: 200 },
  ]
  const axes = (aX: string, aY: string) => ({
    xA: Exploration.makeAxis(aX, runs, M, 1),
    yA: Exploration.makeAxis(aY, runs, M, 1),
  })

  it('builds one config per cell: axis levers = the cell values, other levers = the BEST run, seed omitted', () => {
    const { xA, yA } = axes('x', 'y')
    const best = { algo: 'A', x: 0.5, y: 0.5, noise: 0.9, seed: 3 } // the top run's config
    const cx = xA.index(0.5)
    const cy = yA.index(0.5)
    const cfgs = Exploration.manualCellConfigs([{ x: cx, y: cy }], {
      xA,
      yA,
      pegs: {},
      best,
      manifest: M,
    })
    expect(cfgs).toHaveLength(1)
    expect(cfgs[0].x).toBe(xA.cellValue(cx)) // the cell's concrete run value (bin midpoint)
    expect(cfgs[0].y).toBe(yA.cellValue(cy))
    expect(cfgs[0].noise).toBe(0.9) // a non-axis lever takes the best run's value
    expect(cfgs[0].algo).toBe('A')
    expect('seed' in cfgs[0]).toBe(false) // seed left for the planner
  })

  it('a PEGGED lever overrides the best value (coerced to the lever type)', () => {
    const { xA, yA } = axes('x', 'y')
    const best = { algo: 'A', x: 0.5, y: 0.5, noise: 0.9 }
    const cfgs = Exploration.manualCellConfigs([{ x: xA.index(0.5), y: yA.index(0.5) }], {
      xA,
      yA,
      pegs: { noise: '0.2', algo: 'B' }, // peg values arrive as strings from the <select>
      best,
      manifest: M,
    })
    expect(cfgs[0].noise).toBe(0.2) // coerced to a number
    expect(cfgs[0].algo).toBe('B')
  })

  it('falls back to the manifest default when a lever is neither pegged nor in the best run', () => {
    const { xA, yA } = axes('x', 'y')
    const cfgs = Exploration.manualCellConfigs([{ x: xA.index(0.25), y: yA.index(0.75) }], {
      xA,
      yA,
      pegs: {},
      best: {},
      manifest: M,
    })
    expect(cfgs[0].noise).toBe(0.5) // manifest default
  })

  it('cellValue reproduces the TRIED value in an OCCUPIED cell (not the bin geometric midpoint)', () => {
    // Tried lr values [0.001, 0.003, 0.01]; the extreme 0.01 lands at the top edge of its bin, so a naive
    // midpoint would run 0.0085 — a config never tried. cellValue must return the real value the cell shows.
    const lrRuns = [0.001, 0.003, 0.01].map((lr) => ({
      config: { algo: 'A', x: lr, y: 0.5 },
      objective: 100,
    }))
    const ax = Exploration.makeAxis('x', lrRuns, M, 1)
    const i = ax.index(0.01)
    expect(ax.cellValue(i)).toBe(0.01) // the concrete tried value, matching cellLabel
    expect(Number(ax.cellLabel(i))).toBe(0.01)
  })

  it('cellValue on an EMPTY cell returns a fresh gap-fill value inside the cell', () => {
    const lrRuns = [0.0, 1.0].map((lr) => ({
      config: { algo: 'A', x: lr, y: 0.5 },
      objective: 100,
    }))
    const ax = Exploration.makeAxis('x', lrRuns, M, 3) // 3 bins over [0,1]; the middle bin is empty
    const mid = ax.cellValue(1)
    expect(mid).toBeGreaterThan(0) // a value strictly inside the empty middle bin
    expect(mid).toBeLessThan(1)
  })

  it('SKIPS an out-of-range (stale) cell index instead of launching an invalid config', () => {
    const { xA, yA } = axes('x', 'y')
    const cfgs = Exploration.manualCellConfigs(
      [
        { x: xA.index(0.5), y: yA.index(0.5) },
        { x: 999, y: 0 },
        { x: 0, y: -1 },
      ],
      { xA, yA, pegs: {}, best: {}, manifest: M },
    )
    expect(cfgs).toHaveLength(1) // only the in-range cell survives; the two stale ones are dropped
    expect(cfgs[0].algo).toBeDefined()
  })
})

describe('Exploration.magma', () => {
  it('returns an rgb() string and ramps from dark to light across [0,1]', () => {
    expect(Exploration.magma(0)).toMatch(/^rgb\(/)
    const lum = (s: string) =>
      s
        .match(/\d+/g)!
        .map(Number)
        .reduce((p: number, c: number) => p + c, 0)
    expect(lum(Exploration.magma(1))).toBeGreaterThan(lum(Exploration.magma(0)))
  })

  it('clamps out-of-range inputs', () => {
    expect(Exploration.magma(-5)).toBe(Exploration.magma(0))
    expect(Exploration.magma(5)).toBe(Exploration.magma(1))
  })
})
