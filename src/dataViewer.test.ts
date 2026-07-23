import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/data.js is the no-build browser module for DATA-CATALOG presentation (coverage summary + status +
// mine request); load it as CommonJS the same way datasetsViewer.test.ts loads viewer/datasets.js, so the
// ACTUAL viewer logic that decides an instrument's status is unit-tested directly.
const require = createRequire(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'data.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const D: any = mod.exports

const cov = (start: string, end: string, months: number, gaps: string[] = []) => ({
  start,
  end,
  months,
  gaps,
})

describe('instrumentState', () => {
  it('is available when nothing is on disk', () => {
    expect(D.instrumentState({ symbol: 'GOLD', onDisk: {} })).toBe('available')
    expect(D.instrumentState({ symbol: 'GOLD' })).toBe('available')
  })

  it('is ready when on disk with no gaps', () => {
    expect(D.instrumentState({ onDisk: { '1d': cov('2018-01', '2026-06', 102) } })).toBe('ready')
  })

  it('is gaps when any interval has interior gaps', () => {
    expect(
      D.instrumentState({ onDisk: { '1d': cov('2018-01', '2026-06', 100, ['2019-05', '2019-06']) } }),
    ).toBe('gaps')
  })
})

describe('primaryInterval', () => {
  it('prefers 1d', () => {
    expect(D.primaryInterval({ onDisk: { '1m': cov('a', 'b', 1), '1d': cov('a', 'b', 1) } })).toBe('1d')
  })

  it('falls back to the finest interval on disk', () => {
    expect(D.primaryInterval({ onDisk: { '1h': cov('a', 'b', 1), '1m': cov('a', 'b', 1) } })).toBe('1h')
  })

  it('falls back to the first declared interval when nothing is on disk', () => {
    expect(D.primaryInterval({ intervals: ['1m', '1h', '1d'], onDisk: {} })).toBe('1m')
  })
})

describe('coverageLine', () => {
  it('formats an on-disk span', () => {
    expect(D.coverageLine({ onDisk: { '1d': cov('2018-01', '2026-06', 102) } })).toBe(
      '1d · 2018-01 → 2026-06 · 102 mo',
    )
  })

  it('appends a gap count (singular/plural)', () => {
    expect(D.coverageLine({ onDisk: { '1d': cov('2018-01', '2026-06', 101, ['2019-05']) } })).toBe(
      '1d · 2018-01 → 2026-06 · 101 mo · 1 gap',
    )
    expect(
      D.coverageLine({ onDisk: { '1d': cov('2018-01', '2026-06', 100, ['2019-05', '2019-06']) } }),
    ).toBe('1d · 2018-01 → 2026-06 · 100 mo · 2 gaps')
  })

  it('says not downloaded when off disk', () => {
    expect(D.coverageLine({ symbol: 'GOLD', onDisk: {} })).toBe('not downloaded')
  })
})

describe('classCounts', () => {
  it('counts how many instruments have any coverage', () => {
    const cls = {
      instruments: [
        { onDisk: { '1d': cov('a', 'b', 1) } },
        { onDisk: {} },
        { onDisk: { '1d': cov('a', 'b', 1, ['x']) } },
      ],
    }
    expect(D.classCounts(cls)).toEqual({ onDisk: 2, total: 3 })
  })

  it('handles an empty class', () => {
    expect(D.classCounts({ instruments: [] })).toEqual({ onDisk: 0, total: 0 })
    expect(D.classCounts({})).toEqual({ onDisk: 0, total: 0 })
  })
})

describe('mine request builders', () => {
  it('builds a per-instrument request scoped to its class', () => {
    expect(D.mineRequestForInstrument({ symbol: 'GOLD', assetClass: 'commodities' })).toEqual({
      symbols: ['GOLD'],
      class: 'commodities',
    })
    // Fundamentals reuse a stock ticker — the class must scope it.
    expect(D.mineRequestForInstrument({ symbol: 'AAPL', assetClass: 'fundamentals' })).toEqual({
      symbols: ['AAPL'],
      class: 'fundamentals',
    })
  })

  it('builds a per-class request from the class id', () => {
    expect(D.mineRequestForClass({ id: 'fx' })).toEqual({ class: 'fx' })
  })
})

describe('linkageForAsset', () => {
  const edges = [
    { asset: 'JPM', assetClass: 'stocks', proxy: 'T10Y2Y', edgeType: 'curve-slope' },
    { asset: 'USDCAD', assetClass: 'fx', proxy: 'WTI', edgeType: 'terms-of-trade' },
    { asset: 'AAPL', assetClass: 'fundamentals', proxy: 'X', edgeType: 't' },
  ]

  it('returns the edges from an asset', () => {
    expect(D.linkageForAsset(edges, 'JPM')).toEqual([edges[0]])
  })

  it('scopes by asset class', () => {
    expect(D.linkageForAsset(edges, 'AAPL', 'stocks')).toEqual([])
    expect(D.linkageForAsset(edges, 'AAPL', 'fundamentals')).toEqual([edges[2]])
  })

  it('handles missing edges', () => {
    expect(D.linkageForAsset(undefined, 'JPM')).toEqual([])
    expect(D.linkageForAsset(edges, 'NONE')).toEqual([])
  })
})
