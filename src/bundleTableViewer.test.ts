import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/bundleTable.js is the no-build browser module that sorts the Datasets / Environments management
// tables (named lever bundles) the same numeric-aware way the Runs table sorts; load it as CommonJS the
// same way modelsViewer.test.ts loads viewer/models.js so the ACTUAL viewer logic is tested here.
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'bundleTable.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const B: any = mod.exports

const row = (name: string, values: Record<string, unknown>, def = false) => ({
  id: name,
  name,
  default: def,
  values,
})

describe('compareCells', () => {
  it('compares two numbers numerically (asc + desc)', () => {
    expect(B.compareCells(2, 10, 'asc')).toBeLessThan(0)
    expect(B.compareCells(2, 10, 'desc')).toBeGreaterThan(0)
  })
  it('compares numeric strings numerically, not lexically', () => {
    // '100' vs '20': lexical would put '100' first; numeric must put 20 first ascending.
    expect(B.compareCells('100', '20', 'asc')).toBeGreaterThan(0)
  })
  it('falls back to string order when not both numeric', () => {
    expect(B.compareCells('1d', '1h', 'asc')).toBeLessThan(0)
    expect(B.compareCells('1d', '1h', 'desc')).toBeGreaterThan(0)
  })
  it('sorts missing (undefined / null / empty) values last in BOTH directions', () => {
    for (const dir of ['asc', 'desc']) {
      expect(B.compareCells(undefined, 5, dir)).toBeGreaterThan(0)
      expect(B.compareCells(5, undefined, dir)).toBeLessThan(0)
      expect(B.compareCells(null, 5, dir)).toBeGreaterThan(0)
      expect(B.compareCells('', 5, dir)).toBeGreaterThan(0)
    }
  })
  it('treats 0 as a real value, not missing', () => {
    expect(B.compareCells(0, 5, 'asc')).toBeLessThan(0)
  })
  it('returns 0 for two missing values and for equal values', () => {
    expect(B.compareCells(undefined, null, 'asc')).toBe(0)
    expect(B.compareCells(7, 7, 'asc')).toBe(0)
  })
})

describe('sortRows', () => {
  const rows = [
    row('Beta', { window: 2024, asset: 'BTCUSDT' }),
    row('alpha', { window: 100, asset: 'ETHUSDT' }, true),
    row('Gamma', { window: 20, asset: 'BTCUSDT' }),
  ]
  it('does not mutate the input array', () => {
    const copy = rows.slice()
    B.sortRows(rows, 'window', 'asc')
    expect(rows).toEqual(copy)
  })
  it('sorts by a lever value numerically', () => {
    expect(B.sortRows(rows, 'window', 'asc').map((r: any) => r.values.window)).toEqual([20, 100, 2024])
    expect(B.sortRows(rows, 'window', 'desc').map((r: any) => r.values.window)).toEqual([2024, 100, 20])
  })
  it('sorts by name case-insensitively', () => {
    expect(B.sortRows(rows, 'name', 'asc').map((r: any) => r.name)).toEqual(['alpha', 'Beta', 'Gamma'])
  })
  it('sorts the default row first when sorting by "default" descending', () => {
    expect(B.sortRows(rows, 'default', 'desc')[0].name).toBe('alpha')
  })
  it('keeps the original order for rows that tie on the sort column (stable)', () => {
    // Beta and Gamma both have asset BTCUSDT; their input order (Beta before Gamma) must be preserved.
    const byAsset = B.sortRows(rows, 'asset', 'asc')
    const btc = byAsset.filter((r: any) => r.values.asset === 'BTCUSDT').map((r: any) => r.name)
    expect(btc).toEqual(['Beta', 'Gamma'])
  })
})

describe('nextSort', () => {
  it('flips direction when the same column is clicked again', () => {
    expect(B.nextSort('window', 'asc', 'window')).toEqual({ key: 'window', dir: 'desc' })
    expect(B.nextSort('window', 'desc', 'window')).toEqual({ key: 'window', dir: 'asc' })
  })
  it('starts a newly clicked column at the default direction (asc unless told otherwise)', () => {
    expect(B.nextSort('window', 'asc', 'asset')).toEqual({ key: 'asset', dir: 'asc' })
    expect(B.nextSort('window', 'asc', 'asset', 'desc')).toEqual({ key: 'asset', dir: 'desc' })
  })
})
