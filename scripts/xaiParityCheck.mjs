import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
const require = createRequire(import.meta.url)
// Load the browser mirror as CommonJS (the repo is type:module, so a plain require treats it as ESM).
const mpath = '/Users/cloud/Documents/Work/thefactory-modeltrainer/viewer/xai.js'
const m = new Module(mpath); m.filename = mpath; m.paths = []
m._compile(readFileSync(mpath, 'utf8'), mpath)
const mirror = m.exports
const ts = await import('/Users/cloud/Documents/Work/thefactory-modeltrainer/dist/xaiUtils.js')

const DS = { asset: 'BTC', timeframe: '1h', candles: 100, from: 'a', to: 'b' }
const run = (key, config, objective, opts={}) => ({ key, config, objective, status:'completed', dataset:DS, seed:0, ...opts })
const MAX = { key:'objective', direction:'max' }
const MIN = { key:'durationMs', direction:'min' }

const runs = []
let k=0
for (const lr of [0.1,0.2]) for (const bs of [64,128]) {
  if (lr===0.2 && bs===128) continue
  for (const seed of [0,1,2]) runs.push(run('r'+(k++), {lr, batch_size:bs}, 10+lr*100+bs*0.1+seed, {seed, durationMs: 1000+bs+seed*10}))
}
runs.push(run('thin', {lr:0.5, batch_size:64}, 200, {seed:0, durationMs:900}))

let fails=0
const eq=(a,b,label)=>{ const A=JSON.stringify(a),B=JSON.stringify(b); if(A!==B){fails++; console.log('MISMATCH',label,'\n TS:',A.slice(0,200),'\n JS:',B.slice(0,200))} }
eq(ts.iqm([1,2,3,4,100]), mirror.iqm([1,2,3,4,100]), 'iqm')
eq(ts.aggregateRunValues([5,7,9,11,13,15]), mirror.aggregateRunValues([5,7,9,11,13,15]), 'aggregate')
for (const crit of [MAX, MIN]) {
  eq(ts.ofatContrasts(runs,'batch_size',crit), mirror.ofatContrasts(runs,'batch_size',crit), 'ofat batch_size '+crit.key)
  eq(ts.ofatContrasts(runs,'lr',crit), mirror.ofatContrasts(runs,'lr',crit), 'ofat lr '+crit.key)
  eq(ts.leverImportances(runs,crit), mirror.leverImportances(runs,crit), 'importance '+crit.key)
  eq(ts.recommendExperiments(runs,crit), mirror.recommendExperiments(runs,crit), 'recommend '+crit.key)
}
console.log(fails===0 ? 'PARITY OK — viewer mirror == TS engine across all functions + criteria' : (fails+' MISMATCHES'))
