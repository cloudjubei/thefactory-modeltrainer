import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/crossTest.js is the no-build browser module for cross-test presentation (robustness verdict,
// matrix rows, missing values, the extend-window id); load it as CommonJS the same way
// dataViewer.test.ts loads viewer/data.js, so the ACTUAL viewer logic is unit-tested directly.
const require = createRequire(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'crossTest.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const CT: any = mod.exports

const cell = (value: string, returnVsHold: number | undefined, status = 'completed') => ({
  value,
  status,
  ...(returnVsHold !== undefined
    ? { returnVsHold, metrics: { return_vs_hold_pct: returnVsHold } }
    : {}),
  objective: 1,
  evaluatedAt: 'T',
})

const record = (assetCells: any[], windowCells: any[] = []) => ({
  runKey: 'r1',
  trainedValues: {
    asset: 'BTCUSDT',
    ...(windowCells.length ? { walk_forward_window: '2024' } : {}),
  },
  levers: {
    asset: Object.fromEntries(assetCells.map((c) => [c.value, c])),
    ...(windowCells.length
      ? { walk_forward_window: Object.fromEntries(windowCells.map((c) => [c.value, c])) }
      : {}),
  },
  updatedAt: 'T',
})

describe('crossTestVerdict', () => {
  it('robust when every completed cell beats hold', () => {
    const v = CT.crossTestVerdict(record([cell('ETHUSDT', 3), cell('SOLUSDT', 0.5)]))
    expect(v).toEqual({ state: 'robust', positive: 2, tested: 2, failed: 0 })
  })

  it('weak when none beat hold, mixed otherwise', () => {
    expect(CT.crossTestVerdict(record([cell('ETHUSDT', -3), cell('SOLUSDT', -1)])).state).toBe(
      'weak',
    )
    expect(CT.crossTestVerdict(record([cell('ETHUSDT', 3), cell('SOLUSDT', -1)])).state).toBe(
      'mixed',
    )
  })

  it('none when there are no completed cells (failed cells count separately)', () => {
    const v = CT.crossTestVerdict(record([cell('ETHUSDT', undefined, 'failed')]))
    expect(v.state).toBe('none')
    expect(v.failed).toBe(1)
    expect(CT.crossTestVerdict(undefined).state).toBe('none')
    expect(CT.crossTestVerdict({}).state).toBe('none')
  })

  it('pools cells across levers (asset + extended window)', () => {
    const v = CT.crossTestVerdict(record([cell('ETHUSDT', 2)], [cell('oos-2024', 1)]))
    expect(v.tested).toBe(2)
    expect(v.state).toBe('robust')
  })
})

describe('crossTestRows', () => {
  it('flattens the per-lever maps into ordered display rows', () => {
    const rows = CT.crossTestRows(
      record([cell('ETHUSDT', 3), cell('SOLUSDT', -1)], [cell('oos-2024', 1)]),
    )
    expect(rows.map((r: any) => [r.lever, r.value])).toEqual([
      ['asset', 'ETHUSDT'],
      ['asset', 'SOLUSDT'],
      ['walk_forward_window', 'oos-2024'],
    ])
    expect(rows[0].returnVsHold).toBe(3)
  })

  it('is empty for a missing record', () => {
    expect(CT.crossTestRows(undefined)).toEqual([])
  })
})

describe('missingAssets', () => {
  const manifest = {
    levers: {
      asset: {
        type: 'choice',
        choices: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT'],
        scope: 'dataset',
      },
    },
  }

  it('universe minus trained minus tested', () => {
    const run = { summary: { config: { asset: 'BTCUSDT' } } }
    expect(CT.missingAssets(manifest, run, record([cell('ETHUSDT', 1)]))).toEqual([
      'SOLUSDT',
      'XRPUSDT',
    ])
  })

  it('everything but the trained asset when nothing is tested yet', () => {
    const run = { summary: { config: { asset: 'ETHUSDT' } } }
    expect(CT.missingAssets(manifest, run, undefined)).toEqual(['BTCUSDT', 'SOLUSDT', 'XRPUSDT'])
  })

  it('empty when the manifest has no asset lever', () => {
    expect(CT.missingAssets({ levers: {} }, { summary: { config: {} } }, undefined)).toEqual([])
  })
})

describe('extendedWindowFor', () => {
  it('maps a fixed window to its open-ended oos variant', () => {
    expect(CT.extendedWindowFor('2024')).toBe('oos-2024')
    expect(CT.extendedWindowFor('alt-2025')).toBe('alt-oos-2025')
  })

  it('is null for an already-open window or an unknown shape', () => {
    expect(CT.extendedWindowFor('oos-2024')).toBeNull()
    expect(CT.extendedWindowFor('alt-oos-2024')).toBeNull()
    expect(CT.extendedWindowFor(undefined)).toBeNull()
    expect(CT.extendedWindowFor('whatever')).toBeNull()
  })
})

describe('verdictChip', () => {
  it('labels each state', () => {
    expect(CT.verdictChip({ state: 'robust', positive: 3, tested: 3, failed: 0 })).toEqual({
      label: '3/3 beat hold',
      cls: 'is-robust',
    })
    expect(CT.verdictChip({ state: 'mixed', positive: 1, tested: 3, failed: 0 }).cls).toBe(
      'is-mixed',
    )
    expect(CT.verdictChip({ state: 'weak', positive: 0, tested: 2, failed: 0 }).cls).toBe('is-weak')
    expect(CT.verdictChip({ state: 'none', positive: 0, tested: 0, failed: 2 })).toEqual({
      label: '2 failed',
      cls: 'is-none',
    })
    expect(CT.verdictChip({ state: 'none', positive: 0, tested: 0, failed: 0 }).label).toBe('—')
  })
})
