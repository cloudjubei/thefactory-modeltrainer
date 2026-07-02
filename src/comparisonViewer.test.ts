import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/comparison.js is the no-build browser module behind the Runs tab's "By dataset" / "By environment"
// COMPARISON views: hold every non-varying lever LOCKED and vary one axis (the dataset levers, or the
// environment levers), so the same setup is compared apples-to-apples across datasets / environments. Load
// it as CommonJS the way datasetsViewer.test.ts loads viewer/datasets.js so the ACTUAL logic is unit-tested.
const require = createRequire(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'comparison.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const C: any = mod.exports

const manifest = {
  name: 'BlackSwan',
  recordType: 'blackswan-run',
  objective: { name: 'return', direction: 'max' },
  levers: {
    model_name: { type: 'choice', choices: ['ppo', 'dqn'], scope: 'model' },
    net_arch: { type: 'choice', choices: ['[128]', '[256,256]'] },
    asset: { type: 'choice', choices: ['BTC', 'ETH'], scope: 'dataset' },
    timeframe: { type: 'choice', choices: ['1d', '1h'], scope: 'dataset' },
    stop_loss: { type: 'number', scope: 'environment' },
    seed: { type: 'number', scope: 'ignore' },
  },
}

const mkRun = (cfg: any, objective: number, metrics: any = {}) => ({
  key:
    cfg.model_name +
    '-' +
    cfg.asset +
    '-' +
    cfg.timeframe +
    '-' +
    cfg.stop_loss +
    '-' +
    (cfg.seed ?? 0),
  summary: { config: cfg, objective, metrics: { ...metrics } },
})

describe('axisLeverKeys / lockedLeverKeys', () => {
  it('By dataset: axis = the dataset levers', () => {
    expect(C.axisLeverKeys(manifest, 'dataset')).toEqual(['asset', 'timeframe'])
  })
  it('By dataset: locked = model + environment levers, never the ignored seed', () => {
    expect(C.lockedLeverKeys(manifest, 'dataset')).toEqual(['model_name', 'net_arch', 'stop_loss'])
  })
  it('By environment: axis = the environment levers', () => {
    expect(C.axisLeverKeys(manifest, 'environment')).toEqual(['stop_loss'])
  })
  it('By environment: locked = model + dataset levers, never the ignored seed', () => {
    expect(C.lockedLeverKeys(manifest, 'environment')).toEqual([
      'model_name',
      'net_arch',
      'asset',
      'timeframe',
    ])
  })
})

describe('runAxisSignature', () => {
  it('is the axis levers pinned off the run config', () => {
    const r = mkRun(
      { model_name: 'ppo', asset: 'BTC', timeframe: '1d', stop_loss: 0.02, seed: 1 },
      10,
    )
    expect(C.runAxisSignature(manifest, 'dataset', r)).toBe('asset=BTC · timeframe=1d')
    expect(C.runAxisSignature(manifest, 'environment', r)).toBe('stop_loss=0.02')
  })
})

describe('matchesLock', () => {
  const r = mkRun(
    { model_name: 'ppo', net_arch: '[256,256]', asset: 'BTC', timeframe: '1d', stop_loss: 0.02 },
    10,
  )
  it('matches when every locked value equals the run config (string-coerced)', () => {
    expect(C.matchesLock({ model_name: 'ppo', net_arch: '[256,256]', stop_loss: '0.02' }, r)).toBe(
      true,
    )
  })
  it('fails when any locked value differs', () => {
    expect(C.matchesLock({ model_name: 'dqn' }, r)).toBe(false)
  })
  it('an empty lock matches everything', () => {
    expect(C.matchesLock({}, r)).toBe(true)
  })
})

describe('distinctLockValues', () => {
  it('lists the present values per locked lever, numeric-aware sorted', () => {
    const runs = [
      mkRun(
        { model_name: 'ppo', net_arch: '[128]', asset: 'BTC', timeframe: '1d', stop_loss: 0.1 },
        5,
      ),
      mkRun(
        {
          model_name: 'dqn',
          net_arch: '[256,256]',
          asset: 'ETH',
          timeframe: '1h',
          stop_loss: 0.02,
        },
        8,
      ),
      mkRun(
        { model_name: 'ppo', net_arch: '[128]', asset: 'BTC', timeframe: '1d', stop_loss: 0.05 },
        6,
      ),
    ]
    const vals = C.distinctLockValues(manifest, 'dataset', runs)
    expect(vals.model_name).toEqual(['dqn', 'ppo'])
    expect(vals.net_arch).toEqual(['[128]', '[256,256]'])
    expect(vals.stop_loss).toEqual(['0.02', '0.05', '0.1'])
    expect(vals.asset).toBeUndefined()
  })
})

describe('bestRunLock', () => {
  const runs = [
    mkRun(
      {
        model_name: 'ppo',
        net_arch: '[128]',
        asset: 'BTC',
        timeframe: '1d',
        stop_loss: 0.1,
        seed: 1,
      },
      5,
    ),
    mkRun(
      {
        model_name: 'dqn',
        net_arch: '[256,256]',
        asset: 'ETH',
        timeframe: '1h',
        stop_loss: 0.02,
        seed: 2,
      },
      20,
    ),
  ]
  it('locks to the best run by objective (max direction), locked levers only', () => {
    expect(C.bestRunLock(manifest, 'dataset', runs, 'max')).toEqual({
      model_name: 'dqn',
      net_arch: '[256,256]',
      stop_loss: '0.02',
    })
  })
  it('honours a min-direction objective', () => {
    expect(C.bestRunLock(manifest, 'dataset', runs, 'min').model_name).toBe('ppo')
  })
  it('returns {} for no runs', () => {
    expect(C.bestRunLock(manifest, 'dataset', [], 'max')).toEqual({})
  })
})

describe('groupComparison', () => {
  const items = [
    {
      key: 'a1',
      axisSig: 'asset=BTC · timeframe=1d',
      axisLabel: 'BTC 1d',
      values: { objective: 10 },
    },
    {
      key: 'a2',
      axisSig: 'asset=BTC · timeframe=1d',
      axisLabel: 'BTC 1d',
      values: { objective: 20 },
    },
    {
      key: 'b1',
      axisSig: 'asset=BTC · timeframe=1h',
      axisLabel: 'BTC 1h',
      values: { objective: 5 },
    },
  ]
  it('groups by axis signature with min/avg/max per column + a run count', () => {
    const groups = C.groupComparison(items)
    const a = groups.find((g: any) => g.axisSig === 'asset=BTC · timeframe=1d')
    expect(a.count).toBe(2)
    expect(a.axisLabel).toBe('BTC 1d')
    expect(a.keys).toEqual(['a1', 'a2'])
    expect(a.stats.objective).toEqual({ min: 10, avg: 15, max: 20 })
    const b = groups.find((g: any) => g.axisSig === 'asset=BTC · timeframe=1h')
    expect(b.stats.objective).toEqual({ min: 5, avg: 5, max: 5 })
  })
  it('ignores non-finite values in the stats (NaN stats when none finite)', () => {
    const groups = C.groupComparison([
      { key: 'x', axisSig: 's', axisLabel: 'S', values: { objective: NaN } },
      { key: 'y', axisSig: 's', axisLabel: 'S', values: { objective: 8 } },
    ])
    expect(groups[0].count).toBe(2)
    expect(groups[0].stats.objective).toEqual({ min: 8, avg: 8, max: 8 })
    const empty = C.groupComparison([
      { key: 'z', axisSig: 's', axisLabel: 'S', values: { objective: NaN } },
    ])
    expect(Number.isNaN(empty[0].stats.objective.avg)).toBe(true)
  })
})

describe('sortComparisonGroups', () => {
  const groups = [
    {
      axisSig: 's1',
      axisLabel: 'Beta',
      count: 1,
      stats: { objective: { min: 5, avg: 5, max: 5 } },
    },
    {
      axisSig: 's2',
      axisLabel: 'Alpha',
      count: 3,
      stats: { objective: { min: 1, avg: 15, max: 30 } },
    },
  ]
  it('sorts by a column stat (avg) descending', () => {
    const out = C.sortComparisonGroups(groups, 'objective', 'desc')
    expect(out.map((g: any) => g.axisSig)).toEqual(['s2', 's1'])
  })
  it('sorts by the run count', () => {
    const out = C.sortComparisonGroups(groups, '#runs', 'desc')
    expect(out.map((g: any) => g.axisSig)).toEqual(['s2', 's1'])
  })
  it('sorts by the axis label (case-insensitive)', () => {
    const out = C.sortComparisonGroups(groups, 'axis', 'asc')
    expect(out.map((g: any) => g.axisLabel)).toEqual(['Alpha', 'Beta'])
  })
  it('does not mutate the input array', () => {
    const copy = groups.slice()
    C.sortComparisonGroups(groups, 'objective', 'asc')
    expect(groups).toEqual(copy)
  })
})
