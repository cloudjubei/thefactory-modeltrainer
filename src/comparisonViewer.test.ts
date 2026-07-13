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

// The real BlackSwan manifest leaves `seed` unscoped (→ 'model') — it must STILL never be locked (it's a
// nuisance param pooled over, matching setupKeyOfRun). And any tunable lever can be a single-lever "axis"
// (the By value view): axis = that one lever, locked = every other non-ignore lever except seed.
const manifestSeedModel = {
  ...manifest,
  levers: {
    ...manifest.levers,
    learning_rate: { type: 'number', range: [0.00005, 0.001], scope: 'model' },
    seed: { type: 'number', scope: 'model' },
  },
}

describe('seed is always a nuisance param (never locked), even when the manifest scopes it model', () => {
  it('By dataset lock never includes seed', () => {
    expect(C.lockedLeverKeys(manifestSeedModel, 'dataset')).not.toContain('seed')
  })
  it('By environment lock never includes seed', () => {
    expect(C.lockedLeverKeys(manifestSeedModel, 'environment')).not.toContain('seed')
  })
})

// A manifest with conditional (appliesWhen) levers gated on the model-identity lever — like BlackSwan, where
// prob_threshold only applies to supervised models and momentum_lookback only to momentum. Sweeping model_name
// (a By value axis on the CONTROL lever) must treat those gated levers as part of each model's identity: never
// LOCK them (else only siblings sharing the focus's gated values match) and never PIN them across models.
const manifestConditional = {
  name: 'BlackSwan',
  recordType: 'blackswan-run',
  objective: { name: 'return', direction: 'max' },
  levers: {
    model_name: { type: 'choice', choices: ['supervised-logreg', 'momentum', 'ppo'], scope: 'model' },
    learning_rate: { type: 'number', scope: 'model' },
    asset: { type: 'choice', choices: ['BTC'], scope: 'dataset' },
    stop_loss: { type: 'number', scope: 'environment' },
    prob_threshold: {
      type: 'number',
      scope: 'model',
      appliesWhen: { model_name: ['supervised-logreg'] },
    },
    momentum_lookback: { type: 'number', scope: 'model', appliesWhen: { model_name: ['momentum'] } },
    seed: { type: 'number', scope: 'model' },
  },
}

describe('axisGatedLevers (levers a control axis gates via appliesWhen)', () => {
  it('lists the levers whose appliesWhen names the axis lever', () => {
    expect(C.axisGatedLevers(manifestConditional, 'model_name')).toEqual({
      prob_threshold: true,
      momentum_lookback: true,
    })
  })
  it('is empty for a scope axis (appliesWhen never names dataset/environment)', () => {
    expect(C.axisGatedLevers(manifestConditional, 'dataset')).toEqual({})
    expect(C.axisGatedLevers(manifestConditional, 'environment')).toEqual({})
  })
  it('is empty for a lever that gates nothing', () => {
    expect(C.axisGatedLevers(manifestConditional, 'learning_rate')).toEqual({})
  })
})

describe('locking / matching across the model-identity axis excludes gated levers', () => {
  it('By model_name lock drops the axis lever AND every lever it gates', () => {
    expect(C.lockedLeverKeys(manifestConditional, 'model_name')).toEqual([
      'learning_rate',
      'asset',
      'stop_loss',
    ])
  })
  it('a model that differs only in model_name + its own gated levers matches the focus', () => {
    const focus = {
      model_name: 'supervised-logreg',
      learning_rate: 0.01,
      asset: 'BTC',
      stop_loss: 0.02,
      prob_threshold: 0.6,
    }
    // A momentum run: shared levers identical, its own gated lever set, the focus's gated lever normalised n/a.
    const momentum = {
      model_name: 'momentum',
      learning_rate: 0.01,
      asset: 'BTC',
      stop_loss: 0.02,
      momentum_lookback: 10,
      prob_threshold: 'n/a',
    }
    expect(C.sameSetupExceptAxis(manifestConditional, 'model_name', focus, momentum)).toBe(true)
  })
  it('a run that differs on a SHARED locked lever still does not match', () => {
    const focus = { model_name: 'supervised-logreg', learning_rate: 0.01, asset: 'BTC', stop_loss: 0.02 }
    const other = { model_name: 'ppo', learning_rate: 0.5, asset: 'BTC', stop_loss: 0.02 }
    expect(C.sameSetupExceptAxis(manifestConditional, 'model_name', focus, other)).toBe(false)
  })
  it('a scope axis is unaffected — its lock still holds every non-axis lever', () => {
    expect(C.lockedLeverKeys(manifestConditional, 'environment')).toEqual([
      'model_name',
      'learning_rate',
      'asset',
      'prob_threshold',
      'momentum_lookback',
    ])
  })
})

describe('single-lever axis (By value)', () => {
  it('axis = just that lever', () => {
    expect(C.axisLeverKeys(manifestSeedModel, 'learning_rate')).toEqual(['learning_rate'])
  })
  it('axis = [] for a lever that is not in the manifest', () => {
    expect(C.axisLeverKeys(manifestSeedModel, 'nope')).toEqual([])
  })
  it('locked = every non-ignore lever except the axis lever and seed', () => {
    expect(C.lockedLeverKeys(manifestSeedModel, 'learning_rate')).toEqual([
      'model_name',
      'net_arch',
      'asset',
      'timeframe',
      'stop_loss',
    ])
  })
  it('runAxisSignature pins just that lever', () => {
    const r = mkRun(
      {
        model_name: 'ppo',
        asset: 'BTC',
        timeframe: '1d',
        stop_loss: 0.02,
        learning_rate: 0.0003,
        seed: 1,
      },
      10,
    )
    expect(C.runAxisSignature(manifestSeedModel, 'learning_rate', r)).toBe('learning_rate=0.0003')
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

describe('sameSetupExceptAxis (strict across-axis grouping)', () => {
  const focus = {
    model_name: 'ppo',
    net_arch: '[128]',
    asset: 'BTC',
    timeframe: '1d',
    stop_loss: 0.05,
    learning_rate: 0.0003,
    seed: 0,
  }
  it('matches a run that differs ONLY in the axis lever (and seed, always pooled)', () => {
    expect(
      C.sameSetupExceptAxis(manifestSeedModel, 'learning_rate', focus, {
        ...focus,
        learning_rate: 0.001,
        seed: 7,
      }),
    ).toBe(true)
  })
  it('rejects a run that differs in a NON-axis lever', () => {
    expect(
      C.sameSetupExceptAxis(manifestSeedModel, 'learning_rate', focus, {
        ...focus,
        net_arch: '[256,256]',
      }),
    ).toBe(false)
  })
  it('treats unset ≡ null ≡ n/a, so a lever the focus never set cannot leak in runs that DO set it', () => {
    const f = { model_name: 'ppo', net_arch: '[128]', asset: 'BTC', timeframe: '1d', learning_rate: 0.0003 }
    expect(
      C.sameSetupExceptAxis(manifestSeedModel, 'learning_rate', f, { ...f, learning_rate: 0.001 }),
    ).toBe(true) // stop_loss unset on both
    expect(
      C.sameSetupExceptAxis(manifestSeedModel, 'learning_rate', f, {
        ...f,
        learning_rate: 0.001,
        stop_loss: 0.02,
      }),
    ).toBe(false) // run sets a lever the focus left unset
    expect(
      C.sameSetupExceptAxis(manifestSeedModel, 'learning_rate', { ...f, stop_loss: 'n/a' }, {
        ...f,
        learning_rate: 0.001,
      }),
    ).toBe(true) // focus 'n/a' ≡ run unset
  })
  it('By dataset: same model/env config across datasets (the axis levers are free to differ)', () => {
    const f = { model_name: 'ppo', net_arch: '[128]', asset: 'BTC', timeframe: '1d', stop_loss: 0.05, seed: 1 }
    expect(C.sameSetupExceptAxis(manifest, 'dataset', f, { ...f, asset: 'ETH', timeframe: '1h', seed: 3 })).toBe(true)
    expect(C.sameSetupExceptAxis(manifest, 'dataset', f, { ...f, asset: 'ETH', stop_loss: 0.02 })).toBe(false)
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
  it('sorts a NUMERIC axis (By value lever values) numerically, not lexically', () => {
    const numeric = [
      { axisSig: 'a', axisLabel: '128', count: 1, stats: {} },
      { axisSig: 'b', axisLabel: '16', count: 1, stats: {} },
      { axisSig: 'c', axisLabel: '4096', count: 1, stats: {} },
      { axisSig: 'd', axisLabel: '512', count: 1, stats: {} },
    ]
    expect(C.sortComparisonGroups(numeric, 'axis', 'asc').map((g: any) => g.axisLabel)).toEqual([
      '16',
      '128',
      '512',
      '4096',
    ])
    expect(C.sortComparisonGroups(numeric, 'axis', 'desc').map((g: any) => g.axisLabel)).toEqual([
      '4096',
      '512',
      '128',
      '16',
    ])
  })
  it('keeps a non-numeric axis (dataset/env names, comma-list arches) as a text sort', () => {
    const mixed = [
      { axisSig: 'a', axisLabel: '512,64', count: 1, stats: {} },
      { axisSig: 'b', axisLabel: '128,32', count: 1, stats: {} },
    ]
    expect(C.sortComparisonGroups(mixed, 'axis', 'asc').map((g: any) => g.axisLabel)).toEqual([
      '128,32',
      '512,64',
    ])
  })
  it('sorts by the attached per-group standing (robust z), NaN last', () => {
    const withStanding = [
      { axisSig: 'a', axisLabel: 'A', count: 1, stats: {}, standing: -0.3 },
      { axisSig: 'b', axisLabel: 'B', count: 1, stats: {}, standing: 1.2 },
      { axisSig: 'c', axisLabel: 'C', count: 1, stats: {}, standing: NaN },
    ]
    expect(C.sortComparisonGroups(withStanding, 'standing', 'desc').map((g: any) => g.axisSig)).toEqual([
      'b',
      'a',
      'c',
    ])
    expect(C.sortComparisonGroups(withStanding, 'standing', 'asc').map((g: any) => g.axisSig)).toEqual([
      'a',
      'b',
      'c',
    ])
  })
  it('does not mutate the input array', () => {
    const copy = groups.slice()
    C.sortComparisonGroups(groups, 'objective', 'asc')
    expect(groups).toEqual(copy)
  })
})

describe('leverOffValue / isLeverActive', () => {
  it('off value is the declared default, else false for a boolean and 0 for a number', () => {
    expect(C.leverOffValue({ type: 'boolean', default: false })).toBe(false)
    expect(C.leverOffValue({ type: 'boolean' })).toBe(false)
    expect(C.leverOffValue({ type: 'number', default: 0 })).toBe(0)
    expect(C.leverOffValue({ type: 'number' })).toBe(0)
    expect(C.leverOffValue({ type: 'choice', default: 'fixed' })).toBe('fixed')
  })
  it('a lever is active unless it declares active:false', () => {
    expect(C.isLeverActive({ type: 'number' })).toBe(true)
    expect(C.isLeverActive({ type: 'number', active: true })).toBe(true)
    expect(C.isLeverActive({ type: 'number', active: false })).toBe(false)
    expect(C.isLeverActive(undefined)).toBe(true)
  })
})

// A manifest shaped like BlackSwan's environment axis: 5 wired levers (2 booleans + 3 numeric exit mechanics),
// 3 unused levers (active:false) that must NEVER be swept, and two runtime dependencies the sim enforces —
// no_sell_action is inert while shorting; trailing_take_profit is inert unless take_profit is on.
const envManifest = {
  name: 'BlackSwan',
  recordType: 'blackswan-run',
  objective: { name: 'return', direction: 'max' },
  levers: {
    model_name: { type: 'choice', choices: ['ppo'], scope: 'model' },
    allow_shorting: { type: 'boolean', default: false, scope: 'environment' },
    no_sell_action: {
      type: 'boolean',
      default: false,
      scope: 'environment',
      dependsOn: { lever: 'allow_shorting', active: false },
    },
    stop_loss: { type: 'number', default: 0.02, range: [0, 0.1], scope: 'environment' },
    take_profit: { type: 'number', default: 0, range: [0, 0.05], scope: 'environment' },
    trailing_take_profit: {
      type: 'number',
      default: 0,
      range: [0, 0.01],
      scope: 'environment',
      dependsOn: { lever: 'take_profit', active: true },
    },
    transaction_fee: { type: 'number', default: 0.001, scope: 'environment', active: false },
    position_sizing: { type: 'choice', choices: ['fixed'], default: 'fixed', scope: 'environment', active: false },
    vol_target: { type: 'number', default: 0.02, scope: 'environment', active: false },
    seed: { type: 'number', scope: 'model' },
  },
}
const envFocus = {
  model_name: 'ppo',
  allow_shorting: false,
  no_sell_action: false,
  stop_loss: 0.02,
  take_profit: 0,
  trailing_take_profit: 0,
  transaction_fee: 0.001,
  position_sizing: 'fixed',
  vol_target: 0.02,
  seed: 0,
}

describe('axisSweepBundleSpec (environment)', () => {
  it('excludes active:false levers from the swept bundles and pins them (+ model levers) in fixed', () => {
    const spec = C.axisSweepBundleSpec(envManifest, 'environment', envFocus, {
      allow_shorting: [false],
      no_sell_action: [false],
      stop_loss: [0.02],
      take_profit: [0],
      trailing_take_profit: [0],
    })
    // model lever + the 3 unused environment levers land in fixed, never in a bundle.
    expect(spec.fixed).toEqual({
      model_name: 'ppo',
      transaction_fee: 0.001,
      position_sizing: 'fixed',
      vol_target: 0.02,
    })
    for (const b of spec.bundles) {
      expect(Object.keys(b).sort()).toEqual(
        ['allow_shorting', 'no_sell_action', 'stop_loss', 'take_profit', 'trailing_take_profit'].sort(),
      )
      expect('transaction_fee' in b).toBe(false)
    }
  })

  it('collapses a dependent lever to its off value where its control makes it inert, then dedupes', () => {
    const spec = C.axisSweepBundleSpec(envManifest, 'environment', envFocus, {
      allow_shorting: [false, true],
      no_sell_action: [false, true],
      stop_loss: [0.02],
      take_profit: [0],
      trailing_take_profit: [0],
    })
    // allow_shorting=false → no_sell_action free (2); allow_shorting=true → no_sell_action pinned false (1).
    // So 3 distinct bundles, NOT the naive 2×2=4.
    expect(spec.bundles.length).toBe(3)
    const shorting = spec.bundles.filter((b: any) => b.allow_shorting === true)
    expect(shorting.length).toBe(1)
    expect(shorting[0].no_sell_action).toBe(false)
  })

  it('trailing_take_profit only survives when take_profit is on', () => {
    const spec = C.axisSweepBundleSpec(envManifest, 'environment', envFocus, {
      allow_shorting: [false],
      no_sell_action: [false],
      stop_loss: [0.02],
      take_profit: [0, 0.05],
      trailing_take_profit: [0, 0.01],
    })
    // take_profit=0 → trailing forced 0 (1 bundle); take_profit=0.05 → trailing ∈ {0,0.01} (2). Total 3.
    expect(spec.bundles.length).toBe(3)
    for (const b of spec.bundles) {
      if (b.take_profit === 0) expect(b.trailing_take_profit).toBe(0)
    }
  })

  it('the full 5-lever grid prunes well below the naive cartesian product', () => {
    const spec = C.axisSweepBundleSpec(envManifest, 'environment', envFocus, {
      allow_shorting: [false, true],
      no_sell_action: [false, true],
      stop_loss: [0, 0.02, 0.05],
      take_profit: [0, 0.02, 0.05],
      trailing_take_profit: [0, 0.005, 0.01],
    })
    // Naive 2×2×3×3×3 = 108; conditional collapse + dedupe brings it to 63.
    expect(spec.bundles.length).toBe(63)
    expect(spec.bundles.length).toBeLessThan(108)
  })

  it('returns null when no active axis lever has values to sweep', () => {
    expect(C.axisSweepBundleSpec(envManifest, 'environment', envFocus, {})).toBeNull()
    expect(
      C.axisSweepBundleSpec(envManifest, 'environment', envFocus, { transaction_fee: [0.001, 0.002] }),
    ).toBeNull()
  })
})

describe('axisSweepBundleSpec (dataset, no dependencies)', () => {
  const focus = {
    model_name: 'ppo',
    net_arch: '[128]',
    asset: 'BTC',
    timeframe: '1d',
    stop_loss: 0.05,
    seed: 0,
  }
  it('pins locked (model+env) levers to the focus and returns the full cartesian of dataset values', () => {
    const spec = C.axisSweepBundleSpec(manifest, 'dataset', focus, {
      asset: ['BTC', 'ETH'],
      timeframe: ['1d', '1h'],
    })
    expect(spec.fixed).toEqual({ model_name: 'ppo', net_arch: '[128]', stop_loss: 0.05 })
    expect(spec.bundles.length).toBe(4)
    expect(spec.bundles).toContainEqual({ asset: 'BTC', timeframe: '1d' })
    expect(spec.bundles).toContainEqual({ asset: 'ETH', timeframe: '1h' })
  })
  it('omits an inapplicable (n/a) locked lever from fixed', () => {
    const spec = C.axisSweepBundleSpec(
      manifest,
      'environment',
      { ...focus, net_arch: 'n/a' },
      { stop_loss: [0.02, 0.05, 0.1] },
    )
    expect(spec.fixed).toEqual({ model_name: 'ppo', asset: 'BTC', timeframe: '1d' })
    expect(spec.bundles.length).toBe(3)
  })
})

describe('robustnessVerdict', () => {
  it('is robust when the config is at/above its environments’ typical config everywhere (min standing >= 0)', () => {
    const v = C.robustnessVerdict([1.2, 0.4, 0.1])
    expect(v.label).toBe('robust')
    expect(v.min).toBeCloseTo(0.1)
    expect(v.max).toBeCloseTo(1.2)
  })

  it('is mixed when strong in some environments but below typical in others', () => {
    const v = C.robustnessVerdict([1.5, -0.8])
    expect(v.label).toBe('mixed')
  })

  it('is weak when below its environments’ typical config everywhere (max standing <= 0)', () => {
    const v = C.robustnessVerdict([-0.3, -1.1])
    expect(v.label).toBe('weak')
  })

  it('is n/a with fewer than two environments (nothing to compare across)', () => {
    expect(C.robustnessVerdict([0.9]).label).toBe('n/a')
    expect(C.robustnessVerdict([]).label).toBe('n/a')
  })

  it('ignores non-finite standings', () => {
    const v = C.robustnessVerdict([NaN, 0.5, 0.2, undefined as any])
    expect(v.label).toBe('robust')
    expect(v.n).toBe(2)
  })
})

describe('assessRunReliability (probabilistic-edge / luck detection)', () => {
  it('flags DUBIOUS when a probabilistic threshold dominates AND the setup is unstable across datasets', () => {
    const weak = C.assessRunReliability({
      topLever: 'prob_threshold',
      topImportance: 0.72,
      topProbabilistic: true,
      robustness: 'weak',
      confident: true,
    })
    expect(weak.level).toBe('dubious')
    expect(weak.reasons.join(' ')).toContain('prob_threshold')
    const mixed = C.assessRunReliability({
      topLever: 'prob_threshold',
      topImportance: 0.6,
      topProbabilistic: true,
      robustness: 'mixed',
      confident: true,
    })
    expect(mixed.level).toBe('dubious')
  })

  it('softens to CAUTION when the edge is threshold-driven but robustness is unverified or low-confidence', () => {
    expect(
      C.assessRunReliability({
        topLever: 'prob_threshold',
        topImportance: 0.8,
        topProbabilistic: true,
        robustness: 'n/a',
        confident: true,
      }).level,
    ).toBe('threshold-driven')
    expect(
      C.assessRunReliability({
        topLever: 'prob_threshold',
        topImportance: 0.8,
        topProbabilistic: true,
        robustness: 'weak',
        confident: false,
      }).level,
    ).toBe('threshold-driven')
  })

  it('does NOT flag when the threshold-driven edge holds up across datasets (robust)', () => {
    expect(
      C.assessRunReliability({
        topLever: 'prob_threshold',
        topImportance: 0.8,
        topProbabilistic: true,
        robustness: 'robust',
        confident: true,
      }).level,
    ).toBe('ok')
  })

  it('does NOT flag when the dominant lever is not a probabilistic threshold', () => {
    expect(
      C.assessRunReliability({
        topLever: 'learning_rate',
        topImportance: 0.9,
        topProbabilistic: false,
        robustness: 'weak',
        confident: true,
      }).level,
    ).toBe('ok')
  })

  it('does NOT flag when no single lever dominates (share below half)', () => {
    expect(
      C.assessRunReliability({
        topLever: 'prob_threshold',
        topImportance: 0.3,
        topProbabilistic: true,
        robustness: 'weak',
        confident: true,
      }).level,
    ).toBe('ok')
  })

  it('is null-safe on empty input', () => {
    expect(C.assessRunReliability().level).toBe('ok')
    expect(C.assessRunReliability({}).level).toBe('ok')
  })
})
