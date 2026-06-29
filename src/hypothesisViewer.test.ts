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
    expect(H.contextCells({ fixed: { a: 1 }, compare: { lever: 'model_name', values: [] } })).toEqual([])
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
    expect(H.rollupPaperVerdict({ hypothesisIds: hyps.map((h) => h.id) }, hyps, [], 'max')).toBe('shaky')
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
    expect(d.counts).toEqual({ proven: 1, disproved: 2, untested: 1, total: 4 })
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
