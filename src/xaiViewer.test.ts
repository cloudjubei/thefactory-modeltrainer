import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/xai.js is the no-build browser xAI engine; load it as CommonJS the same way
// modelsViewer.test.ts loads viewer/models.js, so the ACTUAL viewer logic is tested here.
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'xai.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const Xai: any = mod.exports

// A manifest-levers map: forward_horizon applies only to supervised models, day_sell only to weekday,
// learning_rate always applies.
const levers = {
  model_name: { type: 'choice' },
  learning_rate: { type: 'number' },
  forward_horizon: {
    type: 'number',
    appliesWhen: { model_name: ['supervised-logreg', 'supervised-gbm'] },
  },
  day_sell: { type: 'number', appliesWhen: { model_name: ['weekday'] } },
}

describe('normalizeConditionalConfig', () => {
  it('pins an inapplicable conditional lever to the n/a sentinel (forward_horizon on a ppo run)', () => {
    const out = Xai.normalizeConditionalConfig(
      { model_name: 'ppo', forward_horizon: 1, day_sell: 4, learning_rate: 0.001 },
      levers,
    )
    expect(out.forward_horizon).toBe('n/a')
    expect(out.day_sell).toBe('n/a')
    expect(out.learning_rate).toBe(0.001)
    expect(out.model_name).toBe('ppo')
  })

  it('keeps a conditional lever where it DOES apply', () => {
    const out = Xai.normalizeConditionalConfig(
      { model_name: 'supervised-logreg', forward_horizon: 5, day_sell: 4 },
      levers,
    )
    expect(out.forward_horizon).toBe(5)
    expect(out.day_sell).toBe('n/a')
  })

  it('applies the weekday lever to a weekday run only', () => {
    const out = Xai.normalizeConditionalConfig(
      { model_name: 'weekday', forward_horizon: 1, day_sell: 3 },
      levers,
    )
    expect(out.day_sell).toBe(3)
    expect(out.forward_horizon).toBe('n/a')
  })

  it('leaves a non-conditional lever untouched and does not add absent keys', () => {
    const out = Xai.normalizeConditionalConfig({ model_name: 'ppo', learning_rate: 0.01 }, levers)
    expect(out.learning_rate).toBe(0.01)
    expect('forward_horizon' in out).toBe(false)
    expect('day_sell' in out).toBe(false)
  })

  it('does not mutate the input config', () => {
    const cfg = { model_name: 'ppo', forward_horizon: 1 }
    Xai.normalizeConditionalConfig(cfg, levers)
    expect(cfg.forward_horizon).toBe(1)
  })

  it('is safe on missing inputs', () => {
    expect(Xai.normalizeConditionalConfig(null, levers)).toEqual({})
    expect(Xai.normalizeConditionalConfig({ model_name: 'ppo' }, null)).toEqual({
      model_name: 'ppo',
    })
  })
})
