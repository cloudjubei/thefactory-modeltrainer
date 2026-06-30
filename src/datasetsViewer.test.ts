import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/datasets.js is the no-build browser module for dataset IDENTITY (signature + dedup + run-naming);
// load it as CommonJS the same way modelsViewer.test.ts loads viewer/models.js, so the ACTUAL viewer logic
// that decides whether two datasets are "the same" is unit-tested directly.
const require = createRequire(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'datasets.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const D: any = mod.exports

// A manifest where walk_forward_window's scope is configurable — the WHOLE bug is that a manifest snapshot
// predating scope:'dataset' on walk_forward_window silently demotes it to a (default) model lever, so it
// drops out of dataset identity. `walkScope === undefined` reproduces that stale snapshot.
const mkManifest = (walkScope?: string) => ({
  name: 'BlackSwan',
  recordType: 'blackswan-run',
  run: 'x',
  objective: { name: 'return', direction: 'max' },
  levers: {
    model_name: { type: 'choice', choices: ['ppo-custom'], scope: 'model' },
    asset: { type: 'choice', choices: ['BTCUSDT'], scope: 'dataset' },
    timeframe: { type: 'choice', choices: ['1d', '1h'], scope: 'dataset' },
    fidelity_set: { type: 'choice', choices: ['1d'], scope: 'dataset' },
    walk_forward_window: walkScope
      ? { type: 'choice', choices: ['2022', '2023', '2024'], scope: walkScope }
      : { type: 'choice', choices: ['2022', '2023', '2024'] },
    stop_loss: { type: 'number', scope: 'environment' },
    seed: { type: 'number', scope: 'ignore' },
  },
})
const base = { asset: 'BTCUSDT', timeframe: '1d', fidelity_set: '1d', walk_forward_window: '2024' }

describe('datasetLeverKeys', () => {
  it('returns ONLY scope:"dataset" lever keys, in manifest order', () => {
    expect(D.datasetLeverKeys(mkManifest('dataset'))).toEqual([
      'asset',
      'timeframe',
      'fidelity_set',
      'walk_forward_window',
    ])
  })
  it('EXCLUDES walk_forward_window when the manifest snapshot lacks scope:"dataset" (the stale-manifest bug)', () => {
    expect(D.datasetLeverKeys(mkManifest()).includes('walk_forward_window')).toBe(false)
  })
  it('is empty for a manifest with no dataset levers', () => {
    expect(D.datasetLeverKeys({ levers: { lr: { type: 'number' } } })).toEqual([])
  })
})

describe('datasetSettingsSignature', () => {
  it('two datasets differing ONLY by walk_forward_window get DISTINCT signatures (the core regression)', () => {
    const m = mkManifest('dataset')
    const a = D.datasetSettingsSignature(m, { ...base, walk_forward_window: '2022' })
    const b = D.datasetSettingsSignature(m, { ...base, walk_forward_window: '2024' })
    expect(a).not.toBe(b)
  })
  it('COLLAPSES those two when the manifest is stale (walk_forward_window unscoped) — documents the bug', () => {
    const m = mkManifest() // no scope:'dataset' -> walk_forward_window excluded from identity
    const a = D.datasetSettingsSignature(m, { ...base, walk_forward_window: '2022' })
    const b = D.datasetSettingsSignature(m, { ...base, walk_forward_window: '2024' })
    expect(a).toBe(b)
  })
  it('String-coerces values so a choice string "2024" equals a numeric 2024 from a run config', () => {
    const m = mkManifest('dataset')
    expect(D.datasetSettingsSignature(m, { ...base, walk_forward_window: '2024' })).toBe(
      D.datasetSettingsSignature(m, { ...base, walk_forward_window: 2024 }),
    )
  })
  it('treats a missing dataset lever as empty (so a partial old dataset != a complete new one)', () => {
    const m = mkManifest('dataset')
    const partial = { asset: 'BTCUSDT', timeframe: '1d', fidelity_set: '1d' } // walk_forward_window missing
    expect(D.datasetSettingsSignature(m, partial)).not.toBe(D.datasetSettingsSignature(m, base))
  })
})

describe('findDuplicateDataset', () => {
  const m = mkManifest('dataset')
  const dsA = { id: 'a', name: '1d · 2022', settings: { ...base, walk_forward_window: '2022' } }
  it('does NOT flag a dataset that differs only by walk_forward_window as a duplicate', () => {
    const dup = D.findDuplicateDataset(m, [dsA], '1d · 2024', { ...base, walk_forward_window: '2024' }, 'new')
    expect(dup).toBeUndefined()
  })
  it('flags a same-settings dataset as a duplicate', () => {
    const dup = D.findDuplicateDataset(m, [dsA], 'Another name', { ...base, walk_forward_window: '2022' }, 'new')
    expect(dup && dup.id).toBe('a')
  })
  it('flags a same-NAME (case-insensitive) dataset as a duplicate', () => {
    const dup = D.findDuplicateDataset(m, [dsA], '  1D · 2022 ', { ...base, walk_forward_window: '2024' }, 'new')
    expect(dup && dup.id).toBe('a')
  })
  it('excludes the dataset being edited (exceptId)', () => {
    const dup = D.findDuplicateDataset(m, [dsA], '1d · 2022', { ...base, walk_forward_window: '2022' }, 'a')
    expect(dup).toBeUndefined()
  })
})

describe('runDatasetSignature + runDatasetName', () => {
  const m = mkManifest('dataset')
  const datasets = [
    { id: 'a', name: '1d · 2022', settings: { ...base, walk_forward_window: '2022' } },
    { id: 'b', name: '1d · 2024', settings: { ...base, walk_forward_window: '2024' } },
  ]
  const run = (wfw: unknown) => ({ summary: { config: { ...base, walk_forward_window: wfw } } })
  it('names a run by the dataset whose signature matches (number window matches string setting)', () => {
    expect(D.runDatasetName(m, datasets, run(2022))).toBe('1d · 2022')
    expect(D.runDatasetName(m, datasets, run(2024))).toBe('1d · 2024')
  })
  it('returns "Custom" when no named dataset matches', () => {
    expect(D.runDatasetName(m, datasets, run('2023'))).toBe('Custom')
  })
  it('groups runs of different windows under DIFFERENT signatures', () => {
    expect(D.runDatasetSignature(m, run(2022))).not.toBe(D.runDatasetSignature(m, run(2024)))
  })
})
