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
  it('picks the two most-varied numeric active levers as heatmap axes and the categorical as the region axis', () => {
    const state = { activeLevers: ['algo', 'x', 'y'], basins: [], stage: 'global', budget: { spentRuns: 30 } }
    const a = Exploration.analyze({ manifest: MANIFEST, state, runs: runsFixture() })
    expect(a.axes).toEqual(['x', 'y'])
    expect(a.axes).not.toContain('noise')
    expect(a.axes).not.toContain('seed')
    expect(a.regionAxis).toBe('algo')
    expect(a.dir).toBe('max')
  })

  it('falls back to the model levers when no state is present', () => {
    const a = Exploration.analyze({ manifest: MANIFEST, state: null, runs: runsFixture() })
    expect(a.axes.length).toBe(2)
    expect(a.regionAxis).toBe('algo')
  })

  it('orients the color scale so a MIN objective maps its lowest value to the hot end', () => {
    const minManifest = { ...MANIFEST, objective: { name: 'rmse', direction: 'min' as const } }
    const runs = [
      { config: { algo: 'A', x: 0.1, y: 0.1 }, objective: 0.2 }, // best (lowest)
      { config: { algo: 'A', x: 0.9, y: 0.9 }, objective: 0.8 }, // worst (highest)
    ]
    const a = Exploration.analyze({ manifest: minManifest, state: null, runs })
    expect(a.nrm(0.2)).toBeCloseTo(1, 5) // lowest rmse → hottest
    expect(a.nrm(0.8)).toBeCloseTo(0, 5)
  })

  it('passes basins through and preserves the objective range', () => {
    const state = { activeLevers: ['algo', 'x', 'y'], basins: [{ id: 'b1', region: { algo: 'A' }, peakObjective: 420 }], declaredBasinId: 'b1', stage: 'converged', budget: { spentRuns: 40 } }
    const a = Exploration.analyze({ manifest: MANIFEST, state, runs: runsFixture() })
    expect(a.basins).toHaveLength(1)
    expect(a.oMax).toBeGreaterThan(a.oMin)
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
