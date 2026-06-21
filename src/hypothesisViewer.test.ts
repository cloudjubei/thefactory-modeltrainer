import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/hypothesis.js is the no-build browser module for the Hypotheses registry; load it as CommonJS
// the same way migrateViewer.test.ts loads viewer/migrate.js, so the ACTUAL viewer logic is tested here.
const require = createRequire(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'hypothesis.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const H: any = mod.exports

const run = (
  key: string,
  config: Record<string, unknown>,
  opts: { objective?: number; vh?: number; status?: string } = {},
) => ({
  key,
  summary: {
    config,
    objective: opts.objective,
    status: opts.status || 'completed',
    metrics: opts.vh === undefined ? {} : { return_vs_hold_pct: opts.vh },
  },
})

describe('specMatchesConfig', () => {
  it('matches when every fixed lever equals the config (string-compared)', () => {
    expect(
      H.specMatchesConfig({ fixed: { model_name: 'a', lr: 0.5 } }, { model_name: 'a', lr: '0.5' }),
    ).toBe(true)
  })
  it('rejects when a fixed lever differs', () => {
    expect(H.specMatchesConfig({ fixed: { model_name: 'a' } }, { model_name: 'b' })).toBe(false)
  })
  it('matches when a swept lever value is one of the options', () => {
    expect(H.specMatchesConfig({ sweep: { steps: [100, 200] } }, { steps: 200 })).toBe(true)
  })
  it('rejects when a swept lever value is not an option', () => {
    expect(H.specMatchesConfig({ sweep: { steps: [100, 200] } }, { steps: 300 })).toBe(false)
  })
  it('tolerates a non-array sweep value', () => {
    expect(H.specMatchesConfig({ sweep: { steps: 100 } }, { steps: 100 })).toBe(true)
  })
  it('combines fixed AND sweep constraints', () => {
    const spec = { fixed: { model_name: 'a' }, sweep: { lr: [0.1, 0.2] } }
    expect(H.specMatchesConfig(spec, { model_name: 'a', lr: 0.2 })).toBe(true)
    expect(H.specMatchesConfig(spec, { model_name: 'a', lr: 0.3 })).toBe(false)
  })
  it('an empty spec matches NOTHING', () => {
    expect(H.specMatchesConfig({}, { model_name: 'a' })).toBe(false)
    expect(H.specMatchesConfig({ fixed: {}, sweep: {} }, { model_name: 'a' })).toBe(false)
  })
})

describe('hypothesisMatchingRuns', () => {
  const runs = [
    run('r1', { model_name: 'a' }),
    run('r2', { model_name: 'b' }),
    run('r3', { model_name: 'a' }),
  ]
  it('returns the consistent runs', () => {
    expect(
      H.hypothesisMatchingRuns({ fixed: { model_name: 'a' } }, runs).map((r: any) => r.key),
    ).toEqual(['r1', 'r3'])
  })
  it('returns [] for an empty spec', () => {
    expect(H.hypothesisMatchingRuns({}, runs)).toEqual([])
  })
})

describe('measuredFromRuns', () => {
  it('returns null when there are no runs', () => {
    expect(H.measuredFromRuns([], 'max')).toBeNull()
  })
  it('returns null when every run failed', () => {
    expect(H.measuredFromRuns([run('r', {}, { status: 'failed', objective: 1 })], 'max')).toBeNull()
  })
  it('beatsHold is null when no run carries return_vs_hold_pct', () => {
    const m = H.measuredFromRuns([run('r', {}, { objective: 5 })], 'max')
    expect(m).toMatchObject({ runs: 1, objective: 5, beatsHold: null })
  })
  it('beatsHold is true when a run beats hold OOS', () => {
    const m = H.measuredFromRuns(
      [run('a', {}, { objective: 5, vh: -1 }), run('b', {}, { objective: 6, vh: 3 })],
      'max',
    )
    expect(m.beatsHold).toBe(true)
  })
  it('beatsHold is false when present but no run beats hold', () => {
    const m = H.measuredFromRuns([run('a', {}, { objective: 5, vh: -1 })], 'max')
    expect(m.beatsHold).toBe(false)
  })
  it('objective honours the direction', () => {
    const rs = [run('a', {}, { objective: 5 }), run('b', {}, { objective: 9 })]
    expect(H.measuredFromRuns(rs, 'max').objective).toBe(9)
    expect(H.measuredFromRuns(rs, 'min').objective).toBe(5)
  })
  it('objective is NaN when no run carries a finite objective', () => {
    expect(Number.isNaN(H.measuredFromRuns([run('a', {})], 'max').objective)).toBe(true)
  })
})

describe('autoVerdictFor', () => {
  it('untested for null / no signal', () => {
    expect(H.autoVerdictFor(null)).toBe('untested')
    expect(H.autoVerdictFor({ runs: 1, objective: 1, beatsHold: null })).toBe('untested')
  })
  it('proven / disproved on the beats-hold signal', () => {
    expect(H.autoVerdictFor({ runs: 1, objective: 1, beatsHold: true })).toBe('proven')
    expect(H.autoVerdictFor({ runs: 1, objective: 1, beatsHold: false })).toBe('disproved')
  })
})

describe('effectiveVerdict', () => {
  const spec = { fixed: { model_name: 'a' } }
  it('a manual override wins over the runs', () => {
    const h = { spec, verdictSource: 'manual', status: 'proven' }
    expect(H.effectiveVerdict(h, [run('r', { model_name: 'a' }, { vh: -5 })], 'max')).toBe('proven')
  })
  it('falls through to auto when a manual status is invalid', () => {
    const h = { spec, verdictSource: 'manual', status: 'bogus' }
    expect(H.effectiveVerdict(h, [run('r', { model_name: 'a' }, { vh: 5 })], 'max')).toBe('proven')
  })
  it('auto derives from matching runs', () => {
    const h = { spec, verdictSource: 'auto', status: 'untested' }
    expect(H.effectiveVerdict(h, [run('r', { model_name: 'a' }, { vh: 5 })], 'max')).toBe('proven')
  })
})

describe('evaluateHypothesis', () => {
  const spec = { fixed: { model_name: 'a' } }
  it('never auto-flips a manual verdict (no write)', () => {
    const h = { spec, verdictSource: 'manual', status: 'disproved' }
    const out = H.evaluateHypothesis(h, [run('r', { model_name: 'a' }, { vh: 5 })], {
      direction: 'max',
      at: 'T',
    })
    expect(out.changed).toBe(false)
    expect(out.transition).toBeNull()
    expect(out.next).toBe(h)
  })
  it('records a transition naming the runs that flipped it', () => {
    const h = { spec, verdictSource: 'auto', status: 'untested' }
    const out = H.evaluateHypothesis(
      h,
      [run('r1', { model_name: 'a' }, { vh: 5 }), run('r2', { model_name: 'b' }, { vh: 5 })],
      { direction: 'max', at: 'T1' },
    )
    expect(out.changed).toBe(true)
    expect(out.next.status).toBe('proven')
    expect(out.next.verdictSource).toBe('auto')
    expect(out.next.evidence).toMatchObject({ at: 'T1', status: 'proven', matchedKeys: ['r1'] })
    expect(out.transition).toMatchObject({ from: 'untested', to: 'proven', byRunKeys: ['r1'] })
    expect(out.next.transitions).toHaveLength(1)
  })
  it('updates evidence (no transition) when runs grow but the verdict holds', () => {
    const h = {
      spec,
      verdictSource: 'auto',
      status: 'proven',
      evidence: { at: 'T0', status: 'proven', matchedKeys: ['r1'], measured: null },
    }
    const out = H.evaluateHypothesis(
      h,
      [run('r1', { model_name: 'a' }, { vh: 5 }), run('r2', { model_name: 'a' }, { vh: 5 })],
      { direction: 'max', at: 'T1' },
    )
    expect(out.changed).toBe(true)
    expect(out.transition).toBeNull()
    expect(out.next.evidence.matchedKeys).toEqual(['r1', 'r2'])
    expect(out.next.transitions).toBeUndefined()
  })
  it('is a no-op when neither the matched set nor the status changed', () => {
    const h = {
      spec,
      verdictSource: 'auto',
      status: 'proven',
      evidence: { at: 'T0', status: 'proven', matchedKeys: ['r1'], measured: null },
    }
    const out = H.evaluateHypothesis(h, [run('r1', { model_name: 'a' }, { vh: 5 })], {
      direction: 'max',
      at: 'T1',
    })
    expect(out.changed).toBe(false)
    expect(out.transition).toBeNull()
    expect(out.next).toBe(h)
  })
  it('byRunKeys are only the runs new since the last snapshot', () => {
    const h = {
      spec,
      verdictSource: 'auto',
      status: 'untested',
      evidence: { at: 'T0', status: 'untested', matchedKeys: ['r1'], measured: null },
    }
    const out = H.evaluateHypothesis(
      h,
      [run('r1', { model_name: 'a' }, { vh: -5 }), run('r2', { model_name: 'a' }, { vh: 5 })],
      { direction: 'max', at: 'T1' },
    )
    expect(out.next.status).toBe('proven')
    expect(out.transition.byRunKeys).toEqual(['r2'])
  })
})

describe('rollupPaperVerdict', () => {
  const manual = (id: string, status: string) => ({ id, spec: {}, verdictSource: 'manual', status })
  it('untested when the paper links no hypotheses', () => {
    expect(H.rollupPaperVerdict({ hypothesisIds: [] }, [], [], 'max')).toBe('untested')
  })
  it('holds-up when any linked hypothesis is proven', () => {
    const hyps = [manual('a', 'proven'), manual('b', 'disproved')]
    expect(H.rollupPaperVerdict({ hypothesisIds: ['a', 'b'] }, hyps, [], 'max')).toBe('holds-up')
  })
  it('fluff when every linked hypothesis is disproved', () => {
    const hyps = [manual('a', 'disproved'), manual('b', 'disproved')]
    expect(H.rollupPaperVerdict({ hypothesisIds: ['a', 'b'] }, hyps, [], 'max')).toBe('fluff')
  })
  it('untested for a mix of untested and disproved', () => {
    const hyps = [manual('a', 'untested'), manual('b', 'disproved')]
    expect(H.rollupPaperVerdict({ hypothesisIds: ['a', 'b'] }, hyps, [], 'max')).toBe('untested')
  })
  it('rolls up from a live auto verdict', () => {
    const hyps = [
      { id: 'a', spec: { fixed: { model_name: 'a' } }, verdictSource: 'auto', status: 'untested' },
    ]
    const runs = [run('r', { model_name: 'a' }, { vh: 5 })]
    expect(H.rollupPaperVerdict({ hypothesisIds: ['a'] }, hyps, runs, 'max')).toBe('holds-up')
  })
})
