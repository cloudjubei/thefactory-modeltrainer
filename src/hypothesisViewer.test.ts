import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/hypothesis.js is the no-build browser module for the Hypotheses registry; load it as CommonJS
// the same way scripts/xaiParityCheck.mjs loads viewer/xai.js, so the ACTUAL viewer logic is tested here.
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
  it('restricts the compare lever to its values (so the partition is well-defined)', () => {
    const spec = {
      fixed: { timeframe: '1h' },
      compare: { lever: 'model_name', values: ['ppo-custom', 'reppo-custom'] },
    }
    expect(H.specMatchesConfig(spec, { timeframe: '1h', model_name: 'reppo-custom' })).toBe(true)
    expect(H.specMatchesConfig(spec, { timeframe: '1h', model_name: 'dqn' })).toBe(false) // not a compared value
    expect(H.specMatchesConfig(spec, { timeframe: '1d', model_name: 'ppo-custom' })).toBe(false) // fixed differs
  })
  it('a compare-only spec is a real constraint (not empty)', () => {
    expect(
      H.specMatchesConfig({ compare: { lever: 'model_name', values: ['a'] } }, { model_name: 'a' }),
    ).toBe(true)
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
  it('stays untested when there are fewer than minRuns (not enough data to trust)', () => {
    expect(H.autoVerdictFor({ runs: 1, objective: 1, beatsHold: true }, 3)).toBe('untested')
    expect(H.autoVerdictFor({ runs: 3, objective: 1, beatsHold: true }, 3)).toBe('proven')
    expect(H.autoVerdictFor({ runs: 5, objective: 1, beatsHold: false }, 3)).toBe('disproved')
  })
})

describe('the minRuns gate threads through the verdict', () => {
  const spec = { fixed: { model_name: 'a' } }
  const oneWinner = [run('r1', { model_name: 'a' }, { vh: 5 })]
  it('effectiveVerdict honours minRuns', () => {
    const h = { spec, verdictSource: 'auto', status: 'untested' }
    expect(H.effectiveVerdict(h, oneWinner, 'max', 2)).toBe('untested')
    expect(H.effectiveVerdict(h, oneWinner, 'max', 1)).toBe('proven')
  })
  it('evaluateHypothesis honours opts.minRuns', () => {
    const h = { spec, verdictSource: 'auto', status: 'untested' }
    const gated = H.evaluateHypothesis(h, oneWinner, { direction: 'max', at: 'T', minRuns: 2 })
    expect(gated.next.status).toBe('untested')
    const allowed = H.evaluateHypothesis(h, oneWinner, { direction: 'max', at: 'T', minRuns: 1 })
    expect(allowed.next.status).toBe('proven')
  })
  it('rollupPaperVerdict honours minRuns (a one-run paper stays untested)', () => {
    const hyps = [{ id: 'a', spec, verdictSource: 'auto', status: 'untested' }]
    expect(H.rollupPaperVerdict({ hypothesisIds: ['a'] }, hyps, oneWinner, 'max', 2)).toBe(
      'untested',
    )
    expect(H.rollupPaperVerdict({ hypothesisIds: ['a'] }, hyps, oneWinner, 'max', 1)).toBe(
      'holds-up',
    )
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

describe('contextCells', () => {
  it('is empty for a single-context spec (no environments/datasets)', () => {
    expect(H.contextCells({ fixed: { model_name: 'a' } })).toEqual([])
  })
  it('returns the environment bundles when only environments span', () => {
    expect(
      H.contextCells({ environments: [{ allow_shorting: false }, { allow_shorting: true }] }),
    ).toEqual([{ allow_shorting: false }, { allow_shorting: true }])
  })
  it('returns the dataset bundles when only datasets span', () => {
    expect(H.contextCells({ datasets: [{ asset: 'BTC' }, { asset: 'ETH' }] })).toEqual([
      { asset: 'BTC' },
      { asset: 'ETH' },
    ])
  })
  it('crosses environments × datasets (env-major) when both span', () => {
    expect(
      H.contextCells({
        environments: [{ allow_shorting: false }, { allow_shorting: true }],
        datasets: [{ asset: 'BTC' }],
      }),
    ).toEqual([
      { allow_shorting: false, asset: 'BTC' },
      { allow_shorting: true, asset: 'BTC' },
    ])
  })
  it('turns a `compare` lever into one cell per value', () => {
    expect(
      H.contextCells({
        fixed: { timeframe: '1h' },
        compare: { lever: 'model_name', values: ['ppo-custom', 'reppo-custom'] },
      }),
    ).toEqual([{ model_name: 'ppo-custom' }, { model_name: 'reppo-custom' }])
  })
  it('crosses compare × datasets (compare-major)', () => {
    expect(
      H.contextCells({
        compare: { lever: 'model_name', values: ['ppo-custom', 'reppo-custom'] },
        datasets: [{ asset: 'BTC' }, { asset: 'ETH' }],
      }),
    ).toEqual([
      { model_name: 'ppo-custom', asset: 'BTC' },
      { model_name: 'ppo-custom', asset: 'ETH' },
      { model_name: 'reppo-custom', asset: 'BTC' },
      { model_name: 'reppo-custom', asset: 'ETH' },
    ])
  })
  it('ignores an empty/malformed compare', () => {
    expect(
      H.contextCells({ fixed: { a: 1 }, compare: { lever: 'model_name', values: [] } }),
    ).toEqual([])
    expect(H.contextCells({ compare: { values: ['x'] } })).toEqual([])
  })
})

describe('groupRunsByContext', () => {
  const spec = {
    fixed: { model_name: 'a' },
    environments: [{ allow_shorting: false }, { allow_shorting: true }],
  }
  it('returns null for a single-context spec', () => {
    expect(H.groupRunsByContext({ fixed: { model_name: 'a' } }, [])).toBeNull()
  })
  it('splits matching runs into disjoint context cells (never pooled)', () => {
    const runs = [
      run('lo1', { model_name: 'a', allow_shorting: false }),
      run('ls1', { model_name: 'a', allow_shorting: true }),
      run('lo2', { model_name: 'a', allow_shorting: false }),
    ]
    const groups = H.groupRunsByContext(spec, runs)
    expect(groups.map((g: any) => g.runs.map((r: any) => r.key))).toEqual([['lo1', 'lo2'], ['ls1']])
  })
  it('drops a run whose context matches no declared cell', () => {
    const groups = H.groupRunsByContext(spec, [
      run('x', { model_name: 'a', allow_shorting: 'maybe' }),
    ])
    expect(groups.every((g: any) => g.runs.length === 0)).toBe(true)
  })
  it('only considers runs consistent with the spec fixed/sweep', () => {
    const groups = H.groupRunsByContext(spec, [
      run('other', { model_name: 'b', allow_shorting: false }),
    ])
    expect(groups[0].runs).toHaveLength(0)
  })
})

describe('measuredByContext', () => {
  const spec = {
    fixed: { model_name: 'a' },
    environments: [{ allow_shorting: false }, { allow_shorting: true }],
  }
  it('returns null for a single-context spec', () => {
    expect(H.measuredByContext({ fixed: { model_name: 'a' } }, [], 'max')).toBeNull()
  })
  it('measures each cell on its own runs, never bundling across cells', () => {
    const runs = [
      run('lo1', { model_name: 'a', allow_shorting: false }, { objective: 5, vh: -1 }),
      run('ls1', { model_name: 'a', allow_shorting: true }, { objective: 9, vh: 3 }),
      run('ls2', { model_name: 'a', allow_shorting: true }, { objective: 7, vh: 2 }),
    ]
    const m = H.measuredByContext(spec, runs, 'max')
    expect(m[0]).toMatchObject({ context: { allow_shorting: false }, runKeys: ['lo1'] })
    expect(m[0].measured).toMatchObject({ runs: 1, objective: 5, beatsHold: false })
    expect(m[1]).toMatchObject({ context: { allow_shorting: true }, runKeys: ['ls1', 'ls2'] })
    expect(m[1].measured).toMatchObject({ runs: 2, objective: 9, beatsHold: true })
  })
})

describe('compareContexts', () => {
  const cells = (a: number, b: number) => [
    {
      context: { allow_shorting: false },
      runKeys: ['lo'],
      measured: { runs: 1, objective: a, beatsHold: true },
    },
    {
      context: { allow_shorting: true },
      runKeys: ['ls'],
      measured: { runs: 1, objective: b, beatsHold: true },
    },
  ]
  it('untested with fewer than two cells', () => {
    expect(H.compareContexts([cells(5, 9)[0]], { kind: 'beats-baseline' }, 'max')).toBe('untested')
  })
  it('untested when a cell has no measured read', () => {
    const c = cells(5, 9)
    c[1].measured = null as any
    expect(H.compareContexts(c, { kind: 'beats-baseline' }, 'max')).toBe('untested')
  })
  it('untested when a cell has fewer than minRuns', () => {
    expect(H.compareContexts(cells(5, 9), { kind: 'beats-baseline' }, 'max', 3)).toBe('untested')
  })
  it('beats-baseline: proven when the thesis cell beats the baseline (max)', () => {
    expect(H.compareContexts(cells(5, 9), { kind: 'beats-baseline' }, 'max')).toBe('proven')
  })
  it('beats-baseline: disproved when the thesis cell does not beat the baseline', () => {
    expect(H.compareContexts(cells(5, 3), { kind: 'beats-baseline' }, 'max')).toBe('disproved')
  })
  it('beats-baseline honours the min direction', () => {
    expect(H.compareContexts(cells(5, 3), { kind: 'beats-baseline' }, 'min')).toBe('proven')
  })
  it('beats-baseline honours an explicit baselineIndex', () => {
    expect(
      H.compareContexts(cells(5, 9), { kind: 'beats-baseline', baselineIndex: 1 }, 'max'),
    ).toBe('disproved')
  })
  it('invariant: proven when the objective spread is within tolerance', () => {
    expect(H.compareContexts(cells(5, 5.2), { kind: 'invariant', tolerance: 0.1 }, 'max')).toBe(
      'proven',
    )
  })
  it('invariant: disproved when the spread exceeds tolerance', () => {
    expect(H.compareContexts(cells(5, 9), { kind: 'invariant', tolerance: 0.1 }, 'max')).toBe(
      'disproved',
    )
  })
  it('differs: proven when the spread exceeds tolerance', () => {
    expect(H.compareContexts(cells(5, 9), { kind: 'differs', tolerance: 0.1 }, 'max')).toBe(
      'proven',
    )
  })
  it('differs: disproved when the spread is within tolerance', () => {
    expect(H.compareContexts(cells(5, 5.2), { kind: 'differs', tolerance: 0.1 }, 'max')).toBe(
      'disproved',
    )
  })
})

describe('the verdict branches on a context-spanning spec', () => {
  const spec = {
    fixed: { model_name: 'a' },
    environments: [{ allow_shorting: false }, { allow_shorting: true }],
  }
  const runs = [
    run('lo', { model_name: 'a', allow_shorting: false }, { objective: 5, vh: -1 }),
    run('ls', { model_name: 'a', allow_shorting: true }, { objective: 9, vh: 3 }),
  ]
  it('effectiveVerdict derives from the cross-context comparison (default beats-baseline)', () => {
    const h = { spec, verdictSource: 'auto', status: 'untested' }
    expect(H.effectiveVerdict(h, runs, 'max')).toBe('proven')
  })
  it('effectiveVerdict uses the declared comparison kind (invariant)', () => {
    const h = {
      spec,
      verdictSource: 'auto',
      status: 'untested',
      comparison: { kind: 'invariant', tolerance: 0.1 },
    }
    expect(H.effectiveVerdict(h, runs, 'max')).toBe('disproved')
  })
  it('a manual override still wins on a context-spanning hypothesis', () => {
    const h = {
      spec,
      verdictSource: 'manual',
      status: 'untested',
      comparison: { kind: 'beats-baseline' },
    }
    expect(H.effectiveVerdict(h, runs, 'max')).toBe('untested')
  })
  it('evaluateHypothesis sets the status from the comparison and matches runs across all cells', () => {
    const h = { spec, verdictSource: 'auto', status: 'untested' }
    const out = H.evaluateHypothesis(h, runs, { direction: 'max', at: 'T' })
    expect(out.next.status).toBe('proven')
    expect(out.next.evidence.matchedKeys).toEqual(['lo', 'ls'])
  })
})

describe('rollupPaperVerdict', () => {
  const manual = (id: string, status: string, weight?: number) => ({
    id,
    spec: {},
    verdictSource: 'manual',
    status,
    ...(weight ? { weight } : {}),
  })
  it('untested when the paper links no hypotheses', () => {
    expect(H.rollupPaperVerdict({ hypothesisIds: [] }, [], [], 'max')).toBe('untested')
  })
  it('holds-up when every DECIDED hypothesis is proven (untested ignored)', () => {
    const hyps = [manual('a', 'proven'), manual('b', 'untested')]
    expect(H.rollupPaperVerdict({ hypothesisIds: ['a', 'b'] }, hyps, [], 'max')).toBe('holds-up')
  })
  it('shaky — NOT holds-up — when proven and disproved are mixed', () => {
    const hyps = [manual('a', 'proven'), manual('b', 'disproved')]
    expect(H.rollupPaperVerdict({ hypothesisIds: ['a', 'b'] }, hyps, [], 'max')).toBe('shaky')
  })
  it('shaky for the 2-proven / 4-disproved case at equal weight (was wrongly holds-up)', () => {
    const hyps = ['p1', 'p2']
      .map((i) => manual(i, 'proven'))
      .concat(['d1', 'd2', 'd3', 'd4'].map((i) => manual(i, 'disproved')))
      .concat(['u1', 'u2', 'u3', 'u4', 'u5'].map((i) => manual(i, 'untested')))
    expect(H.rollupPaperVerdict({ hypothesisIds: hyps.map((h) => h.id) }, hyps, [], 'max')).toBe(
      'shaky',
    )
  })
  it('fluff when every decided hypothesis is disproved', () => {
    const hyps = [manual('a', 'disproved'), manual('b', 'disproved')]
    expect(H.rollupPaperVerdict({ hypothesisIds: ['a', 'b'] }, hyps, [], 'max')).toBe('fluff')
  })
  it('fluff when the only decided hypothesis is disproved (rest untested)', () => {
    const hyps = [manual('a', 'untested'), manual('b', 'disproved')]
    expect(H.rollupPaperVerdict({ hypothesisIds: ['a', 'b'] }, hyps, [], 'max')).toBe('fluff')
  })
  it('untested when nothing is decided yet', () => {
    const hyps = [manual('a', 'untested'), manual('b', 'untested')]
    expect(H.rollupPaperVerdict({ hypothesisIds: ['a', 'b'] }, hyps, [], 'max')).toBe('untested')
  })
  it('weighting (paper-assigned): a proven CENTRAL hypothesis outweighs a minor disproved one -> holds-up', () => {
    const hyps = [manual('central', 'proven'), manual('minor', 'disproved')]
    const paper = { hypothesisIds: ['central', 'minor'], hypothesisWeights: { central: 5 } }
    expect(H.rollupPaperVerdict(paper, hyps, [], 'max')).toBe('holds-up')
  })
  it('weighting (paper-assigned): a disproved CENTRAL hypothesis outweighs a minor proven one -> fluff', () => {
    const hyps = [manual('central', 'disproved'), manual('minor', 'proven')]
    const paper = { hypothesisIds: ['central', 'minor'], hypothesisWeights: { central: 5 } }
    expect(H.rollupPaperVerdict(paper, hyps, [], 'max')).toBe('fluff')
  })
})

describe('paperVerdictDetail', () => {
  const manual = (id: string, status: string, weight?: number) => ({
    id,
    spec: {},
    verdictSource: 'manual',
    status,
    ...(weight ? { weight } : {}),
  })
  it('reports counts, weighted score and a human explanation', () => {
    const hyps = [
      manual('a', 'proven'),
      manual('b', 'disproved'),
      manual('c', 'disproved'),
      manual('d', 'untested'),
    ]
    const d = H.paperVerdictDetail({ hypothesisIds: ['a', 'b', 'c', 'd'] }, hyps, [], 'max')
    expect(d.status).toBe('shaky')
    expect(d.counts).toEqual({ proven: 1, disproved: 2, untested: 1, proposed: 0, total: 4 })
    expect(d.score).toBeCloseTo(1 / 3, 5)
    expect(d.hasWeights).toBe(false)
    expect(typeof d.why).toBe('string')
    expect(d.why.length).toBeGreaterThan(0)
  })
  it('flags weighting in the detail when the PAPER assigns non-default weights', () => {
    const hyps = [manual('a', 'proven'), manual('b', 'disproved')]
    const d = H.paperVerdictDetail(
      { hypothesisIds: ['a', 'b'], hypothesisWeights: { a: 5 } },
      hyps,
      [],
      'max',
    )
    expect(d.hasWeights).toBe(true)
    expect(d.status).toBe('holds-up')
  })
  it('rolls up from a live auto verdict', () => {
    const hyps = [
      { id: 'a', spec: { fixed: { model_name: 'a' } }, verdictSource: 'auto', status: 'untested' },
    ]
    const runs = [run('r', { model_name: 'a' }, { vh: 5 })]
    expect(H.rollupPaperVerdict({ hypothesisIds: ['a'] }, hyps, runs, 'max')).toBe('holds-up')
  })
})

describe('comparisonCriterion (the success/failure rule for a context-spanning hypothesis)', () => {
  const spec = { fixed: { timeframe: '1d' }, compare: { lever: 'day_buy', values: [0, 2] } }
  it('beats-baseline: names the baseline value, what must beat it, the objective + the readiness gate', () => {
    const txt = H.comparisonCriterion(
      spec,
      { kind: 'beats-baseline', baselineIndex: 0 },
      { objectiveName: 'traded_return', direction: 'max', minRuns: 3 },
    )
    expect(txt).toContain('day_buy=0') // the baseline cell
    expect(txt).toContain('traded_return') // the objective
    expect(txt).toMatch(/at least 3 runs/) // the readiness gate matches compareContexts' minRuns check
    expect(txt).toMatch(/PROVEN/)
    expect(txt).toMatch(/DISPROVED/)
    expect(txt).toMatch(/highest/) // max direction
  })
  it('beats-baseline: respects a non-zero baselineIndex and min direction', () => {
    const txt = H.comparisonCriterion(
      spec,
      { kind: 'beats-baseline', baselineIndex: 1 },
      { objectiveName: 'loss', direction: 'min', minRuns: 5 },
    )
    expect(txt).toContain('day_buy=2') // baseline is index 1 now
    expect(txt).toMatch(/lowest/)
    expect(txt).toMatch(/at least 5 runs/)
  })
  it('invariant: states the relative-spread tolerance', () => {
    const txt = H.comparisonCriterion(
      spec,
      { kind: 'invariant', tolerance: 0.2 },
      { objectiveName: 'sharpe', minRuns: 3 },
    )
    expect(txt).toMatch(/INVARIANT/)
    expect(txt).toContain('0.2')
  })
  it('differs: defaults the tolerance to 0.1 when unspecified', () => {
    const txt = H.comparisonCriterion(
      spec,
      { kind: 'differs' },
      { objectiveName: 'sharpe', minRuns: 3 },
    )
    expect(txt).toMatch(/DIFFERS/)
    expect(txt).toContain('0.1')
  })
})

describe('paper scoring: proposed counter + explain + theses (additive lens)', () => {
  const row = (verdict: string, weight?: number, claim?: string) => ({ verdict, weight, claim })

  it('exports the holds-up / fluff thresholds (single source for the explainer)', () => {
    expect(H.PAPER_HOLDS_UP_AT).toBe(0.75)
    expect(H.PAPER_FLUFF_AT).toBe(0.25)
  })

  it('counts proposed hypotheses but EXCLUDES them from the decided score', () => {
    const d = H.scorePaperVerdict([row('proven'), row('proposed'), row('proposed')])
    expect(d.counts.proposed).toBe(2)
    expect(d.counts.proven).toBe(1)
    expect(d.status).toBe('holds-up') // proposed don't dilute: 1/1 decided proven
    expect(d.weighted.decided).toBe(1)
    expect(d.why).toMatch(/awaiting model implementation|to implement|implementation/i)
  })

  it('a paper with only proposed hypotheses stays untested (nothing decided)', () => {
    const d = H.scorePaperVerdict([row('proposed'), row('proposed')])
    expect(d.status).toBe('untested')
    expect(d.counts.proposed).toBe(2)
  })

  describe('paperVerdictExplain', () => {
    it('untested: says it needs a decided hypothesis (no NaN/pct math)', () => {
      const d = H.scorePaperVerdict([row('untested'), row('proposed')])
      const ex = H.paperVerdictExplain(d)
      expect(typeof ex.formula).toBe('string')
      expect(typeof ex.ladder).toBe('string')
      expect(ex.formula).toMatch(/no decided|proven or disproved/i)
      expect(ex.ladder).toMatch(/75%|holds up/i)
    })
    it('decided: shows the EXPLICIT weighted division (proven weight ÷ decided weight = %)', () => {
      const d = H.scorePaperVerdict([row('proven', 3), row('disproved', 1)])
      const ex = H.paperVerdictExplain(d)
      // 3 proven weight ÷ 4 decided weight = 75%
      expect(ex.formula).toMatch(/3/)
      expect(ex.formula).toMatch(/4/)
      expect(ex.formula).toMatch(/75%/)
      expect(ex.formula).toMatch(/÷|\/|divided/)
      expect(ex.ladder).toMatch(/75%/)
      expect(ex.ladder).toMatch(/25%/)
    })
    it('no weights: phrases the division in plain hypothesis counts', () => {
      const d = H.scorePaperVerdict([row('proven'), row('proven'), row('disproved')])
      const ex = H.paperVerdictExplain(d)
      expect(ex.formula).toMatch(/2/) // 2 proven
      expect(ex.formula).toMatch(/3/) // of 3 decided
      expect(ex.formula).toMatch(/67%|66%/)
    })
    it('flags undecided + proposed as not-counted', () => {
      const d = H.scorePaperVerdict([row('proven', 2), row('untested'), row('proposed')])
      const ex = H.paperVerdictExplain(d)
      expect(ex.formula).toMatch(/not counted|undecided|awaiting/i)
      expect(ex.formula).toMatch(/awaiting model implementation/i)
    })
  })

  describe('claims lens', () => {
    it('groupHypothesesByClaim groups by trimmed label, untagged into one bucket, first-seen order', () => {
      const groups = H.groupHypothesesByClaim([
        row('proven', 1, 'A'),
        row('disproved', 1, 'B'),
        row('proven', 1, ' A '),
        row('untested', 1),
      ])
      const byLabel = Object.fromEntries(groups.map((g: any) => [g.claim ?? '_', g.items.length]))
      expect(byLabel.A).toBe(2)
      expect(byLabel.B).toBe(1)
      expect(byLabel._).toBe(1) // untagged
    })
    it('scorePaperVerdict attaches multiClaim + per-claim details when >1 distinct label', () => {
      const d = H.scorePaperVerdict([
        row('proven', 1, 'Momentum holds'),
        row('proven', 1, 'Momentum holds'),
        row('disproved', 1, 'Vol-sizing helps'),
      ])
      expect(d.multiClaim).toBe(true)
      const claims = d.claims.map((t: any) => [t.claim, t.detail.status])
      expect(claims).toContainEqual(['Momentum holds', 'holds-up'])
      expect(claims).toContainEqual(['Vol-sizing helps', 'fluff'])
    })
    it('single-claim / untagged papers are NOT multi-claim (unchanged behaviour)', () => {
      expect(H.scorePaperVerdict([row('proven'), row('disproved')]).multiClaim).toBe(false)
      expect(
        H.scorePaperVerdict([row('proven', 1, 'X'), row('disproved', 1, 'X')]).multiClaim,
      ).toBe(false)
    })
    it('exposes passedClaims / totalClaims counts for the multi-claim status chip', () => {
      const d = H.scorePaperVerdict([
        row('proven', 1, 'Momentum holds'),
        row('disproved', 1, 'Vol-sizing helps'),
        row('proven', 1, 'Carry works'),
      ])
      expect(d.totalClaims).toBe(3)
      expect(d.passedClaims).toBe(2)
    })
  })
})

describe('proposed status derivation (blocked on an unimplemented model)', () => {
  const impl = (map: Record<string, boolean | null>) => (name: string) =>
    name in map ? map[name] : null
  it('requiresUnimplementedModel: a fixed model_name that is known-but-unimplemented', () => {
    expect(
      H.requiresUnimplementedModel({ fixed: { model_name: 'eiie' } }, impl({ eiie: false })),
    ).toBe(true)
    expect(
      H.requiresUnimplementedModel({ fixed: { model_name: 'ppo' } }, impl({ ppo: true })),
    ).toBe(false)
    expect(H.requiresUnimplementedModel({ fixed: { model_name: 'who' } }, impl({}))).toBe(false) // unknown
  })
  it('requiresUnimplementedModel: no model_name -> false', () => {
    expect(H.requiresUnimplementedModel({ fixed: { timeframe: '1d' } }, impl({}))).toBe(false)
  })
  it('requiresUnimplementedModel: compare over model_name -> proposed only when NO arm is implemented', () => {
    const spec = { compare: { lever: 'model_name', values: ['ppo', 'eiie'] } }
    expect(H.requiresUnimplementedModel(spec, impl({ ppo: true, eiie: false }))).toBe(false) // ppo runnable
    expect(H.requiresUnimplementedModel(spec, impl({ ppo: false, eiie: false }))).toBe(true)
    expect(H.requiresUnimplementedModel(spec, impl({}))).toBe(false) // all unknown
  })
  it('autoVerdictForHypothesis: untested + unimplemented model -> proposed; a decided run wins', () => {
    const h = { spec: { fixed: { model_name: 'eiie' } } }
    expect(H.autoVerdictForHypothesis(h, [], 'max', 0, impl({ eiie: false }))).toBe('proposed')
    const winner = [run('r', { model_name: 'eiie' }, { vh: 5 })]
    expect(H.autoVerdictForHypothesis(h, winner, 'max', 1, impl({ eiie: false }))).toBe('proven')
  })
  it('effectiveVerdict threads the resolver; a manual override still wins', () => {
    const auto = {
      spec: { fixed: { model_name: 'eiie' } },
      verdictSource: 'auto',
      status: 'untested',
    }
    expect(H.effectiveVerdict(auto, [], 'max', 0, impl({ eiie: false }))).toBe('proposed')
    const manual = {
      spec: { fixed: { model_name: 'eiie' } },
      verdictSource: 'manual',
      status: 'proven',
    }
    expect(H.effectiveVerdict(manual, [], 'max', 0, impl({ eiie: false }))).toBe('proven')
  })
  it('without a resolver, behaviour is unchanged (no proposed)', () => {
    const h = { spec: { fixed: { model_name: 'eiie' } }, verdictSource: 'auto', status: 'untested' }
    expect(H.effectiveVerdict(h, [], 'max', 0)).toBe('untested')
  })
})

// ---------------------------------------------------------------------------------------------------
// hypothesisHygiene — the per-hypothesis diagnosis of WHY a hypothesis is undecided (dead pins with
// cause, per-cell starvation, underplanned specs, metric absence), + the aggregate census.
// ---------------------------------------------------------------------------------------------------

const hygieneManifest = {
  name: 'demo',
  recordType: 'demo-run',
  objective: { name: 'score', direction: 'max' },
  levers: {
    model_name: { type: 'choice', choices: ['ppo', 'dqn', 'supervised-logreg'] },
    lr: { type: 'number', range: [0.0001, 0.1] },
    reward_model: { type: 'choice', choices: ['combo_unified'] },
    prob_threshold: {
      type: 'number',
      range: [0.5, 0.9],
      appliesWhen: { model_name: ['supervised-logreg'] },
    },
    timeframe: { type: 'choice', choices: ['1d', '1h'], scope: 'dataset' },
    seed: { type: 'number' },
  },
  migrations: [
    { match: { reward_model: 'combo_all' }, set: { reward_model: 'combo_unified' } },
  ],
}

describe('hypothesisHygiene — dead-pin causes', () => {
  const runs = [
    run('r1', { model_name: 'ppo', lr: 0.001, timeframe: '1d' }, { vh: 5, objective: 5 }),
    run('r2', { model_name: 'ppo', lr: 0.01, timeframe: '1d' }, { vh: -2, objective: -2 }),
    run('r3', { model_name: 'dqn', lr: 0.001, timeframe: '1h' }, { vh: 1, objective: 1 }),
  ]
  it('never-run: pinned value exists as a key in runs but never with that value — launchable, so starved', () => {
    const d = H.hypothesisHygiene(
      { spec: { fixed: { model_name: 'ppo', lr: 0.05 } } },
      runs,
      hygieneManifest,
      3,
    )
    const pin = d.pins.find((p: any) => p.lever === 'lr')
    expect(pin.matches).toBe(0)
    expect(pin.cause).toBe('never-run')
    expect(d.status).toBe('starved')
    expect(d.issues.some((i: any) => i.kind === 'dead-pin' && i.lever === 'lr')).toBe(true)
  })
  it('missing-key: pinned lever appears in NO run config (appliesWhen satisfied)', () => {
    const d = H.hypothesisHygiene(
      { spec: { fixed: { model_name: 'supervised-logreg', prob_threshold: 0.6 } } },
      runs,
      hygieneManifest,
      3,
    )
    const pin = d.pins.find((p: any) => p.lever === 'prob_threshold')
    expect(pin.cause).toBe('missing-key')
  })
  it('na-pinned: a conditional lever pinned alongside a control value it does not apply to', () => {
    const d = H.hypothesisHygiene(
      { spec: { fixed: { model_name: 'ppo', prob_threshold: 0.6 } } },
      runs,
      hygieneManifest,
      3,
    )
    const pin = d.pins.find((p: any) => p.lever === 'prob_threshold')
    expect(pin.cause).toBe('na-pinned')
    expect(d.status).toBe('blocked')
  })
  it('sweep-unrun: sweep options no run has tried are listed (launchable, so starved)', () => {
    const d = H.hypothesisHygiene(
      { spec: { fixed: { model_name: 'ppo' }, sweep: { lr: [0.001, 0.05] } } },
      runs,
      hygieneManifest,
      3,
    )
    const issue = d.issues.find((i: any) => i.kind === 'sweep-unrun')
    expect(issue.lever).toBe('lr')
    expect(issue.values).toEqual(['0.05'])
    expect(d.status).toBe('starved')
  })
  it('migrated: the fixed config would be rewritten by a manifest migration', () => {
    const d = H.hypothesisHygiene(
      { spec: { fixed: { model_name: 'ppo', reward_model: 'combo_all' } } },
      runs,
      hygieneManifest,
      3,
    )
    const pin = d.pins.find((p: any) => p.lever === 'reward_model')
    expect(pin.cause).toBe('migrated')
  })
  it('off-manifest: pinned value outside the declared choices / range', () => {
    const badChoice = H.hypothesisHygiene(
      { spec: { fixed: { model_name: 'sac' } } },
      runs,
      hygieneManifest,
      3,
    )
    expect(badChoice.pins.find((p: any) => p.lever === 'model_name').cause).toBe('off-manifest')
    const badRange = H.hypothesisHygiene(
      { spec: { fixed: { model_name: 'ppo', lr: 5 } } },
      runs,
      hygieneManifest,
      3,
    )
    expect(badRange.pins.find((p: any) => p.lever === 'lr').cause).toBe('off-manifest')
  })
  it('a live pin (matches > 0) carries no cause', () => {
    const d = H.hypothesisHygiene(
      { spec: { fixed: { model_name: 'ppo' } } },
      runs,
      hygieneManifest,
      3,
    )
    const pin = d.pins.find((p: any) => p.lever === 'model_name')
    expect(pin.matches).toBe(2)
    expect(pin.cause).toBeUndefined()
  })
})

describe('hypothesisHygiene — statuses + structural issues', () => {
  it('judged: a decided hypothesis reports status judged with no issues', () => {
    const runs3 = [
      run('a', { model_name: 'ppo' }, { vh: 5, objective: 5 }),
      run('b', { model_name: 'ppo', seed: 1 }, { vh: 4, objective: 4 }),
      run('c', { model_name: 'ppo', seed: 2 }, { vh: 3, objective: 3 }),
    ]
    const d = H.hypothesisHygiene({ spec: { fixed: { model_name: 'ppo' } } }, runs3, hygieneManifest, 3)
    expect(d.verdict).toBe('proven')
    expect(d.status).toBe('judged')
    expect(d.issues).toEqual([])
  })
  it('underplanned: a fully-pinned single-seed spec can never reach minRuns', () => {
    const d = H.hypothesisHygiene(
      { spec: { fixed: { model_name: 'ppo', lr: 0.001 }, seeds: [0] } },
      [run('a', { model_name: 'ppo', lr: 0.001 }, { vh: 5 })],
      hygieneManifest,
      3,
    )
    expect(d.plannedItems).toBe(1)
    expect(d.issues.some((i: any) => i.kind === 'underplanned')).toBe(true)
    expect(d.status).toBe('blocked')
  })
  it('a sweep or seed list large enough to reach minRuns is not underplanned', () => {
    const d = H.hypothesisHygiene(
      { spec: { fixed: { model_name: 'ppo' }, sweep: { lr: [0.001, 0.01] }, seeds: [0, 1] } },
      [],
      hygieneManifest,
      3,
    )
    expect(d.plannedItems).toBe(4)
    expect(d.issues.some((i: any) => i.kind === 'underplanned')).toBe(false)
    expect(d.status).toBe('starved')
  })
  it('starved: live pins, enough planned, just needs runs — reports runsNeeded', () => {
    const d = H.hypothesisHygiene(
      { spec: { fixed: { model_name: 'ppo' }, seeds: [0, 1, 2] } },
      [run('a', { model_name: 'ppo' }, { vh: 5 })],
      hygieneManifest,
      3,
    )
    expect(d.status).toBe('starved')
    expect(d.runsNeeded).toBe(2)
  })
  it('no-metric: matched live runs exist but none report return_vs_hold_pct', () => {
    const d = H.hypothesisHygiene(
      { spec: { fixed: { model_name: 'ppo' }, seeds: [0, 1, 2] } },
      [
        run('a', { model_name: 'ppo' }, { objective: 5 }),
        run('b', { model_name: 'ppo', seed: 1 }, { objective: 4 }),
        run('c', { model_name: 'ppo', seed: 2 }, { objective: 3 }),
      ],
      hygieneManifest,
      3,
    )
    expect(d.metricRuns).toBe(0)
    expect(d.issues.some((i: any) => i.kind === 'no-metric')).toBe(true)
    expect(d.status).toBe('blocked')
  })
  it('invalid-evidence: every matched run is failed/invalid', () => {
    const d = H.hypothesisHygiene(
      { spec: { fixed: { model_name: 'ppo' } } },
      [run('a', { model_name: 'ppo' }, { vh: 5, status: 'invalid' })],
      hygieneManifest,
      3,
    )
    expect(d.issues.some((i: any) => i.kind === 'invalid-evidence')).toBe(true)
    expect(d.status).toBe('starved')
  })
  it('cell-starved: a compare arm below minRuns blocks judgment and is reported per cell', () => {
    const h = {
      spec: { compare: { lever: 'model_name', values: ['ppo', 'dqn'] } },
      comparison: { kind: 'beats-baseline', baselineIndex: 0 },
    }
    const runsOneArm = [
      run('a', { model_name: 'ppo' }, { vh: 5, objective: 5 }),
      run('b', { model_name: 'ppo', seed: 1 }, { vh: 4, objective: 4 }),
    ]
    const d = H.hypothesisHygiene(h, runsOneArm, hygieneManifest, 2)
    expect(d.cells.length).toBe(2)
    const starved = d.issues.find((i: any) => i.kind === 'cell-starved')
    expect(starved).toBeTruthy()
    expect(d.status).toBe('starved')
    const dqnCell = d.cells.find((c: any) => c.context.model_name === 'dqn')
    expect(dqnCell.runs).toBe(0)
    expect(dqnCell.needed).toBe(2)
  })
  it('single-cell: an env-bundle spec that collapses to one cell can never compare', () => {
    const h = { spec: { fixed: { model_name: 'ppo' }, environments: [{ timeframe: '1d' }] } }
    const d = H.hypothesisHygiene(h, [], hygieneManifest, 2)
    expect(d.issues.some((i: any) => i.kind === 'single-cell')).toBe(true)
    expect(d.status).toBe('blocked')
  })
  it('baseline-out-of-range: comparison.baselineIndex beyond the cells is structural', () => {
    const h = {
      spec: { compare: { lever: 'model_name', values: ['ppo', 'dqn'] } },
      comparison: { kind: 'beats-baseline', baselineIndex: 5 },
    }
    const d = H.hypothesisHygiene(h, [], hygieneManifest, 2)
    expect(d.issues.some((i: any) => i.kind === 'baseline-out-of-range')).toBe(true)
    expect(d.status).toBe('blocked')
  })
})

describe('hypothesisHygieneCensus', () => {
  it('aggregates statuses and issue kinds across hypotheses', () => {
    const runs3 = [
      run('a', { model_name: 'ppo' }, { vh: 5, objective: 5 }),
      run('b', { model_name: 'ppo', seed: 1 }, { vh: 4, objective: 4 }),
      run('c', { model_name: 'ppo', seed: 2 }, { vh: 3, objective: 3 }),
    ]
    const hyps = [
      { id: 'h1', spec: { fixed: { model_name: 'ppo' } } }, // judged (proven)
      { id: 'h2', spec: { fixed: { model_name: 'ppo', reward_model: 'combo_all' } } }, // blocked (migrated pin)
      { id: 'h3', spec: { fixed: { model_name: 'dqn' }, seeds: [0, 1, 2] } }, // starved (never-run, launchable)
    ]
    const c = H.hypothesisHygieneCensus(hyps, runs3, hygieneManifest, 3)
    expect(c.total).toBe(3)
    expect(c.judged).toBe(1)
    expect(c.blocked).toBe(1)
    expect(c.starved).toBe(1)
    expect(c.byIssue['dead-pin']).toBe(2)
    expect(c.blockedIds).toEqual(['h2'])
  })
  it('a manual verdict counts as judged regardless of runs', () => {
    const hyps = [
      { id: 'h1', spec: { fixed: { model_name: 'ppo' } }, verdictSource: 'manual', status: 'proven' },
    ]
    const c = H.hypothesisHygieneCensus(hyps, [], hygieneManifest, 3)
    expect(c.judged).toBe(1)
  })
  // foldHygieneCensus is the shared classification path the viewer's CHUNKED warm folds a precomputed
  // diagnosis map through — it must classify identically to the one-shot census (which now delegates to it).
  it('foldHygieneCensus folds a precomputed diagnosis map identically to the one-shot census', () => {
    const runs3 = [
      run('a', { model_name: 'ppo' }, { vh: 5, objective: 5 }),
      run('b', { model_name: 'ppo', seed: 1 }, { vh: 4, objective: 4 }),
      run('c', { model_name: 'ppo', seed: 2 }, { vh: 3, objective: 3 }),
    ]
    const hyps = [
      { id: 'h1', spec: { fixed: { model_name: 'ppo' } } },
      { id: 'h2', spec: { fixed: { model_name: 'ppo', reward_model: 'combo_all' } } },
      { id: 'h3', spec: { fixed: { model_name: 'dqn' }, seeds: [0, 1, 2] } },
      { id: 'h4', spec: { fixed: { model_name: 'ppo' } }, verdictSource: 'manual', status: 'disproved' },
    ]
    const byId = new Map(
      hyps.map((h) => [h.id, H.hypothesisHygiene(h, runs3, hygieneManifest, 3)]),
    )
    const folded = H.foldHygieneCensus(hyps, byId)
    expect(folded.total).toBe(4)
    expect(folded.judged).toBe(2) // h1 proven + h4 manual
    expect(folded.blocked).toBe(1) // h2
    expect(folded.starved).toBe(1) // h3
    expect(folded.blockedIds).toEqual(['h2'])
    // A plain object map is accepted too.
    const asObj: Record<string, unknown> = {}
    byId.forEach((v, k) => (asObj[k] = v))
    expect(H.foldHygieneCensus(hyps, asObj).judged).toBe(2)
  })
})

describe('compareContexts — baselineIndex out of range must not silently mis-judge', () => {
  it('returns untested (not disproved) when baselineIndex exceeds the cells', () => {
    const perContext = [
      { context: { m: 'a' }, measured: { runs: 3, objective: 5, beatsHold: true } },
      { context: { m: 'b' }, measured: { runs: 3, objective: 7, beatsHold: true } },
    ]
    expect(
      H.compareContexts(perContext, { kind: 'beats-baseline', baselineIndex: 9 }, 'max', 3),
    ).toBe('untested')
  })
})

// ---------------------------------------------------------------------------------------------------
// hypothesisBenchmark — the manifest-declared SINGLE-CONTEXT judging rule. Absent, the trading line's
// historical default applies (return_vs_hold_pct > 0); a project like CartPole declares its own metric
// + threshold so hypotheses are judged by ITS definition of success, not BlackSwan's.
// ---------------------------------------------------------------------------------------------------

describe('measuredFromRuns with a manifest benchmark', () => {
  const cartRun = (key: string, evalReturn: number, cfg: Record<string, unknown> = {}) => ({
    key,
    summary: {
      config: { model_name: 'ppo', ...cfg },
      objective: evalReturn,
      status: 'completed',
      metrics: { eval_return_mean: evalReturn },
    },
  })
  const benchmark = { metric: 'eval_return_mean', threshold: 475 }

  it('judges by the declared metric + threshold instead of return_vs_hold_pct', () => {
    const m = H.measuredFromRuns([cartRun('a', 490)], 'max', benchmark)
    expect(m.beatsHold).toBe(true)
    const low = H.measuredFromRuns([cartRun('a', 100)], 'max', benchmark)
    expect(low.beatsHold).toBe(false)
  })
  it('stays null when no run reports the benchmark metric', () => {
    const m = H.measuredFromRuns(
      [run('a', { model_name: 'ppo' }, { objective: 490 })],
      'max',
      benchmark,
    )
    expect(m.beatsHold).toBe(null)
  })
  it('a min-direction benchmark proves when the best value is BELOW the threshold', () => {
    const wine = { metric: 'val_rmse', threshold: 0.6, direction: 'min' }
    const mk = (v: number) => ({
      key: 'k' + v,
      summary: { config: {}, objective: v, status: 'completed', metrics: { val_rmse: v } },
    })
    expect(H.measuredFromRuns([mk(0.5), mk(0.9)], 'min', wine).beatsHold).toBe(true)
    expect(H.measuredFromRuns([mk(0.7)], 'min', wine).beatsHold).toBe(false)
  })
  it('without a benchmark the historical default (return_vs_hold_pct > 0) still applies', () => {
    expect(H.measuredFromRuns([run('a', {}, { vh: 5 })], 'max').beatsHold).toBe(true)
    expect(H.measuredFromRuns([run('a', {}, { vh: -1 })], 'max').beatsHold).toBe(false)
  })

  it('threads through autoVerdictForHypothesis / effectiveVerdict / evaluateHypothesis', () => {
    const h = { spec: { fixed: { model_name: 'ppo' } } }
    const runs3 = [cartRun('a', 490), cartRun('b', 480, { seed: 1 }), cartRun('c', 495, { seed: 2 })]
    expect(H.autoVerdictForHypothesis(h, runs3, 'max', 3, undefined, benchmark)).toBe('proven')
    expect(H.effectiveVerdict(h, runs3, 'max', 3, undefined, benchmark)).toBe('proven')
    const out = H.evaluateHypothesis(h, runs3, { direction: 'max', at: 't', minRuns: 3, benchmark })
    expect(out.next.status).toBe('proven')
  })
})

describe('hypothesisHygiene with a manifest benchmark', () => {
  const cartManifest = {
    name: 'cartpole',
    recordType: 'cartpole-run',
    objective: { name: 'eval_return_mean', direction: 'max' },
    hypothesisBenchmark: { metric: 'eval_return_mean', threshold: 475 },
    levers: {
      model_name: { type: 'choice', choices: ['ppo', 'dqn'] },
      lr: { type: 'number' },
      seed: { type: 'number' },
    },
  }
  const cartRun = (key: string, evalReturn: number, seed: number) => ({
    key,
    summary: {
      config: { model_name: 'ppo', seed },
      objective: evalReturn,
      status: 'completed',
      metrics: { eval_return_mean: evalReturn },
    },
  })

  it('judges a CartPole hypothesis by ITS benchmark (no return_vs_hold_pct anywhere)', () => {
    const d = H.hypothesisHygiene(
      { spec: { fixed: { model_name: 'ppo' } } },
      [cartRun('a', 490, 0), cartRun('b', 480, 1), cartRun('c', 495, 2)],
      cartManifest,
      3,
    )
    expect(d.verdict).toBe('proven')
    expect(d.status).toBe('judged')
  })
  it('no-metric names the DECLARED benchmark metric, not return_vs_hold_pct', () => {
    const noMetricRuns = [
      run('a', { model_name: 'ppo' }, { objective: 1 }),
      run('b', { model_name: 'ppo', seed: 1 }, { objective: 2 }),
      run('c', { model_name: 'ppo', seed: 2 }, { objective: 3 }),
    ]
    const d = H.hypothesisHygiene(
      { spec: { fixed: { model_name: 'ppo' } } },
      noMetricRuns,
      cartManifest,
      3,
    )
    const issue = d.issues.find((i: any) => i.kind === 'no-metric')
    expect(issue.detail).toContain('eval_return_mean')
    expect(issue.detail).not.toContain('return_vs_hold_pct')
  })
  it('with NO declared benchmark and no default metric, no-metric says to declare one', () => {
    const noBench = { ...cartManifest, hypothesisBenchmark: undefined }
    const noMetricRuns = [
      run('a', { model_name: 'ppo' }, { objective: 1 }),
      run('b', { model_name: 'ppo', seed: 1 }, { objective: 2 }),
      run('c', { model_name: 'ppo', seed: 2 }, { objective: 3 }),
    ]
    const d = H.hypothesisHygiene(
      { spec: { fixed: { model_name: 'ppo' } } },
      noMetricRuns,
      noBench,
      3,
    )
    const issue = d.issues.find((i: any) => i.kind === 'no-metric')
    expect(issue.detail).toContain('hypothesisBenchmark')
  })
})

// The per-snapshot inverted run index — a perf accelerator whose ONLY contract is: matching over the
// candidate superset it yields must be BYTE-IDENTICAL to matching over the full run set. specMatchesConfig
// stays the verifier; the index just narrows the array it runs over. These tests pin that equivalence
// (superset + verifier === full scan) across every spec shape + value-type edge case, since a missed
// constraint would silently drop true matches and persist a WRONG verdict.
describe('buildRunIndex + candidateRunsFor', () => {
  const corpus = [
    run('r1', { model_name: 'ppo', lr: 0.5, gamma: 0.99 }, { vh: 1 }),
    run('r2', { model_name: 'ppo', lr: 0.1, gamma: 0.99 }, { vh: -1 }),
    run('r3', { model_name: 'dqn', lr: 0.5, gamma: 0.9 }, { vh: 2 }),
    run('r4', { model_name: 'a2c', lr: '0.5', gamma: 0.99 }, { vh: 3 }), // lr as STRING
    run('r5', { model_name: 'ppo', gamma: 0.99 }, { vh: 4 }), // MISSING lr
    run('r6', { model_name: 'ppo', lr: 0.5, gamma: 0.99, shorting: true }, { vh: 5 }), // boolean lever
    run('r7', { model_name: 'sac', lr: 0.3 }, { vh: 6 }),
  ]
  const sameMatch = (spec: any, runs: any[]) => {
    const idx = H.buildRunIndex(runs)
    const viaIndex = H.hypothesisMatchingRuns(spec, H.candidateRunsFor(spec, idx))
    const viaScan = H.hypothesisMatchingRuns(spec, runs)
    // Order within the candidate superset may differ from the full scan; compare as sorted key sets.
    const keys = (rs: any[]) => rs.map((r) => r.key).sort()
    expect(keys(viaIndex)).toEqual(keys(viaScan))
  }

  it('fixed-only spec: candidates match the full scan', () => {
    sameMatch({ fixed: { model_name: 'ppo' } }, corpus)
  })
  it('sweep-only spec (union of option buckets)', () => {
    sameMatch({ sweep: { lr: [0.5, 0.1] } }, corpus)
  })
  it('mixed fixed + sweep (AND across constraints, picks the tighter superset)', () => {
    sameMatch({ fixed: { model_name: 'ppo' }, sweep: { lr: [0.5, 0.1] } }, corpus)
  })
  it('compare spec (union over compare values)', () => {
    sameMatch({ compare: { lever: 'model_name', values: ['ppo', 'dqn'] } }, corpus)
  })
  it('context-spanning spec with compare + environments still matches identically', () => {
    sameMatch(
      { compare: { lever: 'model_name', values: ['ppo', 'a2c'] }, environments: [{ asset: 'btc' }] },
      corpus,
    )
  })
  it('number-vs-string parity: fixed lr=0.5 also catches the run storing "0.5"', () => {
    const spec = { fixed: { lr: 0.5 } }
    const idx = H.buildRunIndex(corpus)
    const got = H.hypothesisMatchingRuns(spec, H.candidateRunsFor(spec, idx))
      .map((r: any) => r.key)
      .sort()
    // r1, r3, r4 ("0.5"), r6 all have lr 0.5/"0.5"; r5 has no lr, r2/r7 differ.
    expect(got).toEqual(['r1', 'r3', 'r4', 'r6'])
  })
  it('boolean lever value indexes + matches', () => {
    sameMatch({ fixed: { shorting: true } }, corpus)
  })
  it('fixed lever value carried by NO run yields no candidates and no matches', () => {
    const spec = { fixed: { model_name: 'nonexistent' } }
    expect(H.candidateRunsFor(spec, H.buildRunIndex(corpus))).toEqual([])
    sameMatch(spec, corpus)
  })
  it('empty spec (no fixed/sweep/compare) yields [] — mirrors specMatchesConfig matching nothing', () => {
    expect(H.candidateRunsFor({}, H.buildRunIndex(corpus))).toEqual([])
    expect(H.candidateRunsFor({ environments: [{ asset: 'btc' }] }, H.buildRunIndex(corpus))).toEqual(
      [],
    )
    sameMatch({}, corpus)
    sameMatch({ environments: [{ asset: 'btc' }] }, corpus)
  })
  it('candidateRunsFor returns the SMALLEST superset among constraints (tighter than a dominant lever)', () => {
    // gamma=0.99 is on 5 runs; model_name=sac on 1. The spec ANDs both; the seed must be the sac bucket.
    const spec = { fixed: { gamma: 0.99, model_name: 'sac' } }
    const cands = H.candidateRunsFor(spec, H.buildRunIndex(corpus))
    // sac has gamma undefined, so the AND matches nothing — and the smallest seed (model_name=sac, 1 run)
    // keeps the candidate set to that single run, not all five gamma=0.99 runs.
    expect(cands.length).toBeLessThanOrEqual(1)
    sameMatch(spec, corpus)
  })
  it('handles an empty run set', () => {
    expect(H.candidateRunsFor({ fixed: { model_name: 'ppo' } }, H.buildRunIndex([]))).toEqual([])
  })
})
