import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
const require = createRequire(import.meta.url)
// Load the browser mirror as CommonJS (the repo is type:module, so a plain require treats it as ESM).
const mpath = '/Users/cloud/Documents/Work/thefactory-modeltrainer/viewer/xai.js'
const m = new Module(mpath)
m.filename = mpath
m.paths = []
m._compile(readFileSync(mpath, 'utf8'), mpath)
const mirror = m.exports
const ts = await import('/Users/cloud/Documents/Work/thefactory-modeltrainer/dist/xaiUtils.js')

const DS = { asset: 'BTC', timeframe: '1h', candles: 100, from: 'a', to: 'b' }
const run = (key, config, objective, opts = {}) => ({
  key,
  config,
  objective,
  status: 'completed',
  dataset: DS,
  seed: 0,
  ...opts,
})
const MAX = { key: 'objective', direction: 'max' }
const MIN = { key: 'durationMs', direction: 'min' }

const runs = []
let k = 0
for (const lr of [0.1, 0.2])
  for (const bs of [64, 128]) {
    if (lr === 0.2 && bs === 128) continue
    for (const seed of [0, 1, 2])
      runs.push(
        run('r' + k++, { lr, batch_size: bs }, 10 + lr * 100 + bs * 0.1 + seed, {
          seed,
          durationMs: 1000 + bs + seed * 10,
        }),
      )
  }
runs.push(run('thin', { lr: 0.5, batch_size: 64 }, 200, { seed: 0, durationMs: 900 }))

let fails = 0
const eq = (a, b, label) => {
  const A = JSON.stringify(a),
    B = JSON.stringify(b)
  if (A !== B) {
    fails++
    console.log('MISMATCH', label, '\n TS:', A.slice(0, 200), '\n JS:', B.slice(0, 200))
  }
}
eq(ts.iqm([1, 2, 3, 4, 100]), mirror.iqm([1, 2, 3, 4, 100]), 'iqm')
eq(
  ts.aggregateRunValues([5, 7, 9, 11, 13, 15]),
  mirror.aggregateRunValues([5, 7, 9, 11, 13, 15]),
  'aggregate',
)
for (const crit of [MAX, MIN]) {
  eq(
    ts.ofatContrasts(runs, 'batch_size', crit),
    mirror.ofatContrasts(runs, 'batch_size', crit),
    'ofat batch_size ' + crit.key,
  )
  eq(
    ts.ofatContrasts(runs, 'lr', crit),
    mirror.ofatContrasts(runs, 'lr', crit),
    'ofat lr ' + crit.key,
  )
  eq(ts.leverImportances(runs, crit), mirror.leverImportances(runs, crit), 'importance ' + crit.key)
  eq(
    ts.recommendExperiments(runs, crit),
    mirror.recommendExperiments(runs, crit),
    'recommend ' + crit.key,
  )
  // Phase 3: surrogate + ablation/fANOVA/interaction (rng-order sensitive — the strongest parity test)
  const sTs = ts.fitConfigSurrogate(runs, crit),
    sJs = mirror.fitConfigSurrogate(runs, crit)
  eq(sTs, sJs, 'surrogate trees ' + crit.key)
  for (const cfg of [
    { lr: 0.1, batch_size: 64 },
    { lr: 0.2, batch_size: 128 },
    { lr: 0.5, batch_size: 256 },
  ])
    eq(
      ts.predictConfig(sTs, cfg),
      mirror.predictConfig(sJs, cfg),
      'predict ' + crit.key + ' ' + JSON.stringify(cfg),
    )
  eq(
    ts.fanovaImportances(sTs, runs, crit),
    mirror.fanovaImportances(sJs, runs, crit),
    'fanova ' + crit.key,
  )
  eq(ts.ablationPath(sTs, runs, crit), mirror.ablationPath(sJs, runs, crit), 'ablation ' + crit.key)
  eq(
    ts.interactionGrid(sTs, runs, crit, 'lr', 'batch_size'),
    mirror.interactionGrid(sJs, runs, crit, 'lr', 'batch_size'),
    'interaction ' + crit.key,
  )
  // Phase 2/3/4: acquisition stats, coupling, and the PCA projection (rng/order sensitive).
  for (const cfg of [
    { lr: 0.2, batch_size: 128 },
    { lr: 0.5, batch_size: 128 },
  ])
    eq(
      ts.predictConfigStats(sTs, cfg),
      mirror.predictConfigStats(sJs, cfg),
      'predictStats ' + crit.key + ' ' + JSON.stringify(cfg),
    )
  eq(
    ts.expectedImprovement(94.8, 78.9, 26.4, crit.direction),
    mirror.expectedImprovement(94.8, 78.9, 26.4, crit.direction),
    'EI ' + crit.key,
  )
  eq(
    ts.leverCouplings(sTs, runs, crit),
    mirror.leverCouplings(sJs, runs, crit),
    'coupling ' + crit.key,
  )
  eq(ts.pcaProjection(runs, crit), mirror.pcaProjection(runs, crit), 'pca ' + crit.key)
}
console.log(
  fails === 0
    ? 'PARITY OK — viewer mirror == TS engine across all functions + criteria'
    : fails + ' MISMATCHES',
)
