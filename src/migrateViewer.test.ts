import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/migrate.js is the no-build browser module that runs in the sandboxed hub viewer; it can't be
// `import`ed (the repo is type:module, the file is an IIFE that attaches to `window`/`module.exports`).
// Load it as CommonJS the same way scripts/xaiParityCheck.mjs loads viewer/xai.js, so the ACTUAL code
// that runs in the viewer is what's tested here — no parallel TS copy to drift from.
const require = createRequire(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'migrate.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const Migrate: any = mod.exports

// The six DISTINCT historical_data strings actually present in BlackSwan's ledger-import runs (verified
// against .factory/data). Separators vary by era: `|`, `]`, `~` (filesystem-safe `.`). Only the 1h-step /
// 1h+1d observers map to a current fidelity_set; the sub-hourly stacks are retired (fail-fast today) and
// get a truthful `legacy:` identity, never force-merged into 1h+1d.
const HD_1H_1D = 'data_2017_to_2023vs2024_1h_1d_only_price_percent_day_of_week_none_1h_1h|1d_32_1_1_0.004_20'
const HD_1H_1D_LB10 = 'data_2017_to_2023vs2024_1h_1d_only_price_percent_day_of_week_none_1h_1h|1d_32_1_1_0.004_10'
const HD_5M = 'data_2017_to_2023vs2024_only_price_percent_only_price_percent_day_of_week_none_1m]5m_5m]1h]1d_1m]1m_5m]1h]1d_32_0~004_20'
const HD_15M = 'data_2017_to_2023vs2024_only_price_percent_only_price_percent_day_of_week_none_1m|15m_15m|1h|1d_1m|1m_15m|1h|1d_32_0.004_20'
const HD_1M_A = '2017vs2024_1m_1h_1d_only_price_percent_only_price_percent_day_of_week_none_1m|1h_1h|1d_1m|1h_1h|1d_1_0.004_20'
const HD_1M_B = 'data_2017_to_2023vs2024_only_price_percent_only_price_percent_day_of_week_none_1m|1h_1h|1d_1m|1m_1h|1d_32_0.004_20'

describe('fidelitySetFromLayers', () => {
  it('maps a clean 1h+1d layer stack to the canonical fidelity_set', () => {
    expect(Migrate.fidelitySetFromLayers(['1h', '1d'])).toEqual({
      fidelitySet: '1h+1d',
      layers: ['1h', '1d'],
      legacy: false,
    })
  })
  it('maps single + multi clean stacks to their canonical labels (order-independent)', () => {
    expect(Migrate.fidelitySetFromLayers(['1d']).fidelitySet).toBe('1d')
    expect(Migrate.fidelitySetFromLayers(['1h']).fidelitySet).toBe('1h')
    expect(Migrate.fidelitySetFromLayers(['1d', '1h']).fidelitySet).toBe('1h+1d')
    expect(Migrate.fidelitySetFromLayers(['1w', '1h', '1d']).fidelitySet).toBe('1h+1d+1w')
    expect(Migrate.fidelitySetFromLayers(['1w', '1d']).fidelitySet).toBe('1d+1w')
  })
  it('tags any sub-hourly stack as legacy, finest-first, never as a runnable label', () => {
    expect(Migrate.fidelitySetFromLayers(['1d', '1h', '5m', '1m'])).toEqual({
      fidelitySet: 'legacy:1m+5m+1h+1d',
      layers: ['1m', '5m', '1h', '1d'],
      legacy: true,
    })
    expect(Migrate.fidelitySetFromLayers(['15m', '1h', '1d', '1m']).fidelitySet).toBe(
      'legacy:1m+15m+1h+1d',
    )
  })
  it('returns null for empty/invalid input', () => {
    expect(Migrate.fidelitySetFromLayers([])).toBeNull()
    expect(Migrate.fidelitySetFromLayers(null)).toBeNull()
  })
})

describe('legacyLayersFromHistoricalData', () => {
  it('parses the observed layer union across pipe/bracket/tilde encodings', () => {
    expect(Migrate.legacyLayersFromHistoricalData(HD_1H_1D)).toEqual(['1h', '1d'])
    expect(Migrate.legacyLayersFromHistoricalData(HD_5M)).toEqual(['1m', '5m', '1h', '1d'])
    expect(Migrate.legacyLayersFromHistoricalData(HD_15M)).toEqual(['1m', '15m', '1h', '1d'])
    expect(Migrate.legacyLayersFromHistoricalData(HD_1M_A)).toEqual(['1m', '1h', '1d'])
  })
  it('returns null when there is no multi-layer group or no string', () => {
    expect(Migrate.legacyLayersFromHistoricalData('')).toBeNull()
    expect(Migrate.legacyLayersFromHistoricalData(undefined)).toBeNull()
    expect(Migrate.legacyLayersFromHistoricalData('data_2017vs2024_1h_only_price_32_20')).toBeNull()
  })
})

describe('walkForwardWindowFromHistoricalData', () => {
  it('derives the test year from the vsYYYY span', () => {
    expect(Migrate.walkForwardWindowFromHistoricalData(HD_1H_1D)).toBe('2024')
    expect(Migrate.walkForwardWindowFromHistoricalData(HD_1M_A)).toBe('2024')
  })
  it('returns null when no span is present', () => {
    expect(Migrate.walkForwardWindowFromHistoricalData('rl_ppo_combo')).toBeNull()
  })
})

describe('migrationPatchFor', () => {
  it('normalizes a live `auto` run from its own recorded layers', () => {
    const patch = Migrate.migrationPatchFor({
      config: { fidelity_set: 'auto', timeframe: '1h' },
      dataset: { layers: ['1h', '1d'] },
    })
    expect(patch).toEqual({ config: { fidelity_set: '1h+1d' } })
  })
  it('skips an `auto` run with no recorded layers (cannot safely resolve)', () => {
    expect(Migrate.migrationPatchFor({ config: { fidelity_set: 'auto' }, dataset: {} })).toBeNull()
  })
  it('backfills a clean 1h+1d import (fidelity_set + walk_forward_window + dataset.layers)', () => {
    const patch = Migrate.migrationPatchFor({
      config: { asset: 'BTCUSDT', timeframe: '1h', historical_data: HD_1H_1D },
      dataset: { asset: 'BTCUSDT', timeframe: '1h' },
    })
    expect(patch).toEqual({
      config: { fidelity_set: '1h+1d', walk_forward_window: '2024' },
      dataset: { layers: ['1h', '1d'] },
    })
  })
  it('tags a retired sub-hourly import with a legacy identity, not 1h+1d', () => {
    const patch = Migrate.migrationPatchFor({
      config: { timeframe: '1h', historical_data: HD_5M },
      dataset: { timeframe: '1h' },
    })
    expect(patch.config.fidelity_set).toBe('legacy:1m+5m+1h+1d')
    expect(patch.dataset.layers).toEqual(['1m', '5m', '1h', '1d'])
  })
  it('does not override an existing walk_forward_window on an import', () => {
    const patch = Migrate.migrationPatchFor({
      config: { timeframe: '1h', historical_data: HD_1H_1D_LB10, walk_forward_window: '2023' },
      dataset: {},
    })
    expect(patch.config.walk_forward_window).toBeUndefined()
    expect(patch.config.fidelity_set).toBe('1h+1d')
  })
  it('is a no-op for already-concrete runs (idempotent)', () => {
    expect(
      Migrate.migrationPatchFor({ config: { fidelity_set: '1h+1d', timeframe: '1h' } }),
    ).toBeNull()
    expect(
      Migrate.migrationPatchFor({ config: { fidelity_set: 'legacy:1m+5m+1h+1d' } }),
    ).toBeNull()
  })
  it('skips a run with neither fidelity_set nor historical_data', () => {
    expect(Migrate.migrationPatchFor({ config: { timeframe: '1h' }, dataset: {} })).toBeNull()
    expect(Migrate.migrationPatchFor({})).toBeNull()
  })
})
