import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/heatmap.js is the no-build browser module for the 2-D matrix/heatmap attribution renderer's PURE
// logic (value→intensity normalization, sequential-vs-diverging colour ramp, cell geometry, and cell
// bucketing under a cap). Load it as CommonJS the same way crossTestViewer.test.ts loads viewer/crossTest.js
// so the ACTUAL viewer logic is unit-tested directly (the DOM/SVG assembly stays untested in app.js).
const require = createRequire(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'heatmap.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const H: any = mod.exports

const hex = /^#[0-9a-f]{6}$/

describe('buildHeatmapModel — geometry', () => {
  it('emits one cell per grid entry with correct x/y/w/h from the opts geometry', () => {
    const m = H.buildHeatmapModel(
      [
        [0, 1],
        [1, 0],
      ],
      { rowLabels: ['r0', 'r1'], colLabels: ['c0', 'c1'], cell: 10, gutterLeft: 20, gutterTop: 5 },
    )
    expect(m.cells).toHaveLength(4)
    const at = (r: number, c: number) => m.cells.find((k: any) => k.r === r && k.c === c)
    expect(at(0, 0)).toMatchObject({ x: 20, y: 5, w: 10, h: 10 })
    expect(at(0, 1)).toMatchObject({ x: 30, y: 5 })
    expect(at(1, 0)).toMatchObject({ x: 20, y: 15 })
    expect(m.width).toBe(20 + 2 * 10)
    expect(m.height).toBe(5 + 2 * 10)
    for (const cell of m.cells) {
      expect(cell.t).toBeGreaterThanOrEqual(0)
      expect(cell.t).toBeLessThanOrEqual(1)
      expect(cell.fill).toMatch(hex)
    }
    expect(m.shown).toBe(4)
    expect(m.total).toBe(4)
  })
})

describe('buildHeatmapModel — colour scale', () => {
  it('uses a SEQUENTIAL ramp for an all-nonnegative matrix (min→t0, max→t1)', () => {
    const m = H.buildHeatmapModel([[0.2, 0.8]], { rowLabels: ['r0'], colLabels: ['c0', 'c1'] })
    expect(m.legend.diverging).toBe(false)
    const lo = m.cells.find((k: any) => k.value === 0.2)
    const hi = m.cells.find((k: any) => k.value === 0.8)
    expect(lo.t).toBe(0)
    expect(hi.t).toBe(1)
    expect(lo.fill).not.toBe(hi.fill)
  })

  it('uses a DIVERGING ramp centered at 0 when any value is negative (0 → neutral midpoint)', () => {
    const m = H.buildHeatmapModel(
      [
        [-1, 0],
        [1, 0.5],
      ],
      { rowLabels: ['r0', 'r1'], colLabels: ['c0', 'c1'] },
    )
    expect(m.legend.diverging).toBe(true)
    const zeroCells = m.cells.filter((k: any) => k.value === 0)
    for (const z of zeroCells) expect(z.t).toBeCloseTo(0.5, 10)
    // The neutral midpoint colour is a single well-formed hex, identical for every 0 cell.
    const mid = zeroCells[0].fill
    expect(mid).toMatch(hex)
    for (const z of zeroCells) expect(z.fill).toBe(mid)
    // Symmetric extremes: -1 and +1 map to the ramp ends.
    expect(m.cells.find((k: any) => k.value === -1).t).toBeCloseTo(0, 10)
    expect(m.cells.find((k: any) => k.value === 1).t).toBeCloseTo(1, 10)
  })

  it('is NaN-free when min === max (neutral fill, t = 0.5)', () => {
    const m = H.buildHeatmapModel([[1, 1]], { rowLabels: ['r0'], colLabels: ['c0', 'c1'] })
    for (const cell of m.cells) {
      expect(cell.t).toBe(0.5)
      expect(cell.fill).toMatch(hex)
      expect(cell.fill).not.toContain('NaN')
    }
  })
})

describe('buildHeatmapModel — degenerate + labels', () => {
  it('returns no cells for an empty matrix (no rows or no cols)', () => {
    expect(H.buildHeatmapModel([], {}).cells).toEqual([])
    expect(H.buildHeatmapModel([[]], {}).cells).toEqual([])
  })

  it('carries RAW label text (escaping is the renderer’s job) and one label per row/col', () => {
    const m = H.buildHeatmapModel([[1, 2]], { rowLabels: ['<r>'], colLabels: ['a&b', 'c'] })
    expect(m.rowLabels.map((l: any) => l.text)).toEqual(['<r>'])
    expect(m.colLabels.map((l: any) => l.text)).toEqual(['a&b', 'c'])
  })

  it('falls back to index labels when labels are missing or the wrong length', () => {
    const m = H.buildHeatmapModel([
      [1, 2],
      [3, 4],
    ])
    expect(m.rowLabels.map((l: any) => l.text)).toEqual(['0', '1'])
    expect(m.colLabels.map((l: any) => l.text)).toEqual(['0', '1'])
  })
})

describe('buildHeatmapModel — bucketing under a cell cap', () => {
  it('buckets an oversized grid to ≤ maxDim per axis, reporting shown < total', () => {
    const N = 10
    const big = Array.from({ length: N }, (_, r) => Array.from({ length: N }, (_, c) => r + c))
    const m = H.buildHeatmapModel(big, { maxDim: 5 })
    expect(m.total).toBe(N * N)
    expect(m.shown).toBeLessThan(m.total)
    expect(m.cells).toHaveLength(m.shown)
    // Every bucketed cell stays finite + in range.
    for (const cell of m.cells) {
      expect(Number.isFinite(cell.value)).toBe(true)
      expect(cell.t).toBeGreaterThanOrEqual(0)
      expect(cell.t).toBeLessThanOrEqual(1)
    }
  })

  it('does NOT bucket a grid within the cap (shown === total)', () => {
    const m = H.buildHeatmapModel(
      [
        [1, 2],
        [3, 4],
      ],
      { maxDim: 8 },
    )
    expect(m.shown).toBe(4)
    expect(m.total).toBe(4)
  })
})
