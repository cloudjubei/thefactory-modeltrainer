import { describe, expect, it } from 'vitest'
import type { TrainerManifest } from './modelTrainerTypes.js'
import {
  blendJudgeScore,
  parseProgressMarker,
  buildJudgeSystemPrompt,
  buildJudgeUserContent,
  buildProposeSystemPrompt,
  buildProposeUserContent,
  canonicalConfigString,
  coerceHypothesisItems,
  coerceVerdictRows,
  expandExperimentMatrix,
  normalizeObjectiveScores,
  pickBestRun,
  totalCampaignUnits,
  validateTrainerManifest,
  validateTrainingRunSummary,
} from './modelTrainerUtils.js'

const hashByJson = (config: Record<string, unknown>): string => canonicalConfigString(config)

function manifest(overrides: Partial<TrainerManifest> = {}): TrainerManifest {
  return {
    name: 'demo',
    recordType: 'demo-run',
    run: 'python -m trainer.run --config-json {configPath} --summary-out {summaryOut}',
    calibrate: 'python -m trainer.run --calibrate --summary-out {summaryOut}',
    objective: { name: 'score', direction: 'max' },
    levers: {
      lr: { type: 'number', default: 0.01 },
      algo: { type: 'choice', choices: ['a', 'b'], default: 'a' },
      steps: { type: 'number', default: 100 },
    },
    eta: { unitsLever: 'steps' },
    ...overrides,
  }
}

describe('validateTrainerManifest', () => {
  it('accepts a valid manifest and returns it typed', () => {
    const raw = manifest() as unknown
    expect(validateTrainerManifest(raw)).toEqual(manifest())
  })

  it('rejects a non-object', () => {
    expect(() => validateTrainerManifest('nope')).toThrow(/manifest/i)
  })

  it('preserves an optional description (used to brief the in-app chat agent)', () => {
    const m = validateTrainerManifest({ ...manifest(), description: 'Trades BTC under real fees.' })
    expect(m.description).toBe('Trades BTC under real fees.')
  })

  it('rejects a missing name', () => {
    expect(() => validateTrainerManifest({ ...manifest(), name: '' })).toThrow(/name/)
  })

  it('rejects a missing recordType', () => {
    expect(() => validateTrainerManifest({ ...manifest(), recordType: undefined })).toThrow(
      /recordType/,
    )
  })

  it('rejects a run template without {configPath}', () => {
    expect(() =>
      validateTrainerManifest({ ...manifest(), run: 'python x --summary-out {summaryOut}' }),
    ).toThrow(/configPath/)
  })

  it('rejects a run template without {summaryOut}', () => {
    expect(() =>
      validateTrainerManifest({ ...manifest(), run: 'python x --config-json {configPath}' }),
    ).toThrow(/summaryOut/)
  })

  it('rejects a calibrate template without {summaryOut}', () => {
    expect(() => validateTrainerManifest({ ...manifest(), calibrate: 'python calibrate' })).toThrow(
      /summaryOut/,
    )
  })

  it('accepts an absent calibrate command', () => {
    const m = { ...manifest() }
    delete m.calibrate
    expect(validateTrainerManifest(m).calibrate).toBeUndefined()
  })

  it('accepts a valid evaluate template', () => {
    const m = {
      ...manifest(),
      evaluate:
        'python -m trainer.run --evaluate --config-json {configPath} --summary-out {summaryOut}',
    }
    expect(validateTrainerManifest(m).evaluate).toContain('--evaluate')
  })

  it('rejects an evaluate template without {configPath}', () => {
    expect(() =>
      validateTrainerManifest({ ...manifest(), evaluate: 'python e --summary-out {summaryOut}' }),
    ).toThrow(/configPath/)
  })

  it('rejects an evaluate template without {summaryOut}', () => {
    expect(() =>
      validateTrainerManifest({ ...manifest(), evaluate: 'python e --config-json {configPath}' }),
    ).toThrow(/summaryOut/)
  })

  it('rejects an invalid objective direction', () => {
    expect(() =>
      validateTrainerManifest({ ...manifest(), objective: { name: 'score', direction: 'up' } }),
    ).toThrow(/direction/)
  })

  it('rejects a missing objective name', () => {
    expect(() =>
      validateTrainerManifest({ ...manifest(), objective: { name: '', direction: 'max' } }),
    ).toThrow(/objective/)
  })

  it('rejects missing levers', () => {
    expect(() => validateTrainerManifest({ ...manifest(), levers: undefined })).toThrow(/levers/)
  })

  it('rejects an invalid lever type', () => {
    expect(() =>
      validateTrainerManifest({ ...manifest(), levers: { lr: { type: 'slider' } } }),
    ).toThrow(/lever/)
  })

  it('accepts a valid data declaration', () => {
    const m = {
      ...manifest(),
      data: [{ id: 'wine', files: [{ relPath: 'data/x.csv', url: 'https://e.x/x.csv' }] }],
    }
    expect(validateTrainerManifest(m).data?.[0].files[0].relPath).toBe('data/x.csv')
  })

  it('rejects a data entry without an id', () => {
    expect(() =>
      validateTrainerManifest({
        ...manifest(),
        data: [{ files: [{ relPath: 'a', url: 'u' }] }],
      }),
    ).toThrow(/data/)
  })

  it('rejects a data entry with no files', () => {
    expect(() =>
      validateTrainerManifest({ ...manifest(), data: [{ id: 'x', files: [] }] }),
    ).toThrow(/files/)
  })

  it('rejects a data file missing relPath or url', () => {
    expect(() =>
      validateTrainerManifest({
        ...manifest(),
        data: [{ id: 'x', files: [{ url: 'u' }] }],
      }),
    ).toThrow(/relPath/)
    expect(() =>
      validateTrainerManifest({
        ...manifest(),
        data: [{ id: 'x', files: [{ relPath: 'a' }] }],
      }),
    ).toThrow(/url/)
  })

  it('rejects an eta.unitsLever that names no lever', () => {
    expect(() => validateTrainerManifest({ ...manifest(), eta: { unitsLever: 'ghost' } })).toThrow(
      /unitsLever/,
    )
  })
})

describe('expandExperimentMatrix', () => {
  it('produces one item from defaults when the spec is empty', () => {
    const items = expandExperimentMatrix(manifest(), {}, hashByJson)
    expect(items).toHaveLength(1)
    expect(items[0].config).toEqual({ lr: 0.01, algo: 'a', steps: 100 })
  })

  it('applies fixed values over defaults', () => {
    const items = expandExperimentMatrix(manifest(), { fixed: { lr: 0.5 } }, hashByJson)
    expect(items[0].config.lr).toBe(0.5)
  })

  it('takes the cartesian product of sweep arrays', () => {
    const items = expandExperimentMatrix(
      manifest(),
      { sweep: { lr: [0.1, 0.2], algo: ['a', 'b', 'b2'] } },
      hashByJson,
    )
    expect(items).toHaveLength(6)
    const pairs = items.map((i) => `${i.config.lr}:${i.config.algo}`)
    expect(new Set(pairs).size).toBe(6)
  })

  it('multiplies configurations by seeds, setting config.seed', () => {
    const items = expandExperimentMatrix(
      manifest(),
      { sweep: { lr: [0.1, 0.2] }, seeds: [0, 1, 2] },
      hashByJson,
    )
    expect(items).toHaveLength(6)
    expect(items.map((i) => i.config.seed).filter((s) => s === 2)).toHaveLength(2)
  })

  it('crosses each configuration with environment bundles (applied together, not cartesian)', () => {
    const items = expandExperimentMatrix(
      manifest(),
      {
        sweep: { algo: ['a', 'b'] },
        environments: [
          { lr: 0.001, steps: 10 },
          { lr: 0.002, steps: 20 },
        ],
      },
      hashByJson,
    )
    // 2 algos × 2 environment bundles = 4 (NOT 2 algos × 2 lr × 2 steps = 8)
    expect(items).toHaveLength(4)
    const env0 = items.filter((i) => i.config.lr === 0.001)
    expect(env0).toHaveLength(2)
    // a bundle's keys apply TOGETHER: lr 0.001 always pairs with steps 10
    expect(env0.every((i) => i.config.steps === 10)).toBe(true)
  })

  it('multiplies environment bundles by seeds', () => {
    const items = expandExperimentMatrix(
      manifest(),
      { environments: [{ lr: 0.001 }, { lr: 0.002 }], seeds: [0, 1] },
      hashByJson,
    )
    expect(items).toHaveLength(4)
  })

  it('rejects an environment value that names no lever', () => {
    expect(() =>
      expandExperimentMatrix(manifest(), { environments: [{ ghost: 1 }] }, hashByJson),
    ).toThrow(/ghost/)
  })

  it('keys every item with the injected hash of its config', () => {
    const items = expandExperimentMatrix(manifest(), {}, hashByJson)
    expect(items[0].key).toBe(canonicalConfigString(items[0].config))
  })

  it('extracts units from the eta.unitsLever', () => {
    const items = expandExperimentMatrix(manifest(), { fixed: { steps: 250 } }, hashByJson)
    expect(items[0].units).toBe(250)
  })

  it('omits units when the manifest declares no eta', () => {
    const m = manifest()
    delete m.eta
    const items = expandExperimentMatrix(m, {}, hashByJson)
    expect(items[0].units).toBeUndefined()
  })

  it('rejects a sweep key that names no lever', () => {
    expect(() => expandExperimentMatrix(manifest(), { sweep: { ghost: [1] } }, hashByJson)).toThrow(
      /ghost/,
    )
  })

  it('rejects a fixed key that names no lever', () => {
    expect(() => expandExperimentMatrix(manifest(), { fixed: { ghost: 1 } }, hashByJson)).toThrow(
      /ghost/,
    )
  })

  it('rejects an empty sweep array', () => {
    expect(() => expandExperimentMatrix(manifest(), { sweep: { lr: [] } }, hashByJson)).toThrow(
      /empty/i,
    )
  })

  it('rejects a plan larger than the item cap', () => {
    const values = Array.from({ length: 30 }, (_, i) => i)
    expect(() =>
      expandExperimentMatrix(
        manifest(),
        { sweep: { lr: values, steps: values }, maxItems: 100 },
        hashByJson,
      ),
    ).toThrow(/100/)
  })
})

describe('canonicalConfigString', () => {
  it('is stable across key insertion order', () => {
    expect(canonicalConfigString({ a: 1, b: 2 })).toBe(canonicalConfigString({ b: 2, a: 1 }))
  })

  it('canonicalises nested objects', () => {
    expect(canonicalConfigString({ x: { a: 1, b: 2 } })).toBe(
      canonicalConfigString({ x: { b: 2, a: 1 } }),
    )
  })

  it('keeps arrays ordered', () => {
    expect(canonicalConfigString({ x: [1, 2] })).not.toBe(canonicalConfigString({ x: [2, 1] }))
  })
})

describe('pickBestRun', () => {
  it('picks the highest objective for max', () => {
    expect(
      pickBestRun(
        [
          { key: 'a', objective: 1 },
          { key: 'b', objective: 3 },
        ],
        'max',
      ),
    ).toEqual({ key: 'b', objective: 3 })
  })

  it('picks the lowest objective for min', () => {
    expect(
      pickBestRun(
        [
          { key: 'a', objective: 1 },
          { key: 'b', objective: 3 },
        ],
        'min',
      ),
    ).toEqual({ key: 'a', objective: 1 })
  })

  it('returns undefined for no entries', () => {
    expect(pickBestRun([], 'max')).toBeUndefined()
  })

  it('keeps the first entry on ties', () => {
    expect(
      pickBestRun(
        [
          { key: 'first', objective: 2 },
          { key: 'second', objective: 2 },
        ],
        'max',
      ),
    ).toEqual({ key: 'first', objective: 2 })
  })
})

describe('totalCampaignUnits', () => {
  it('sums units across items', () => {
    expect(
      totalCampaignUnits([
        { key: 'a', config: {}, units: 10 },
        { key: 'b', config: {}, units: 5 },
      ]),
    ).toBe(15)
  })

  it('returns undefined when any item lacks units', () => {
    expect(
      totalCampaignUnits([
        { key: 'a', config: {}, units: 10 },
        { key: 'b', config: {} },
      ]),
    ).toBeUndefined()
  })

  it('returns undefined for an empty plan', () => {
    expect(totalCampaignUnits([])).toBeUndefined()
  })
})

describe('normalizeObjectiveScores', () => {
  it('min-max normalises to 0-100 for max direction', () => {
    const scores = normalizeObjectiveScores(
      [
        { key: 'low', objective: 10 },
        { key: 'mid', objective: 55 },
        { key: 'high', objective: 100 },
      ],
      'max',
    )
    expect(scores.get('low')).toBe(0)
    expect(scores.get('mid')).toBe(50)
    expect(scores.get('high')).toBe(100)
  })

  it('inverts for min direction', () => {
    const scores = normalizeObjectiveScores(
      [
        { key: 'low', objective: 10 },
        { key: 'high', objective: 100 },
      ],
      'min',
    )
    expect(scores.get('low')).toBe(100)
    expect(scores.get('high')).toBe(0)
  })

  it('gives 50 to every run when all objectives are equal', () => {
    const scores = normalizeObjectiveScores(
      [
        { key: 'a', objective: 7 },
        { key: 'b', objective: 7 },
      ],
      'max',
    )
    expect(scores.get('a')).toBe(50)
    expect(scores.get('b')).toBe(50)
  })

  it('returns an empty map for no entries', () => {
    expect(normalizeObjectiveScores([], 'max').size).toBe(0)
  })
})

describe('blendJudgeScore', () => {
  it('blends objective and LLM scores by the weight', () => {
    expect(blendJudgeScore(100, 0, 0.5)).toBe(50)
    expect(blendJudgeScore(40, 80, 0.25)).toBe(50)
  })

  it('clamps out-of-range inputs', () => {
    expect(blendJudgeScore(150, -10, 0.5)).toBe(50)
  })

  it('clamps the weight to [0,1]', () => {
    expect(blendJudgeScore(100, 0, 2)).toBe(0)
  })
})

describe('coerceVerdictRows', () => {
  it('keeps valid rows, clamping and rounding scores', () => {
    const rows = coerceVerdictRows([
      { key: 'a', score: 88.6, why: 'solid' },
      { key: 'b', score: 250, why: 'too high' },
    ])
    expect(rows).toEqual([
      { key: 'a', score: 89, why: 'solid' },
      { key: 'b', score: 100, why: 'too high' },
    ])
  })

  it('drops rows without a key and tolerates junk', () => {
    const rows = coerceVerdictRows([null, 'x', { score: 5 }, { key: 'ok', score: 5 }])
    expect(rows).toEqual([{ key: 'ok', score: 5, why: '' }])
  })

  it('returns empty for a non-array', () => {
    expect(coerceVerdictRows('nope' as never)).toEqual([])
  })
})

describe('coerceHypothesisItems', () => {
  const m = manifest()

  it('keeps a valid proposal', () => {
    const items = coerceHypothesisItems(
      [
        {
          title: 'Higher lr',
          rationale: 'best run used the top lr',
          spec: { sweep: { lr: [0.2, 0.4] } },
        },
      ],
      m,
    )
    expect(items).toHaveLength(1)
    expect(items[0].spec.sweep).toEqual({ lr: [0.2, 0.4] })
  })

  it('drops a proposal naming an unknown lever', () => {
    expect(
      coerceHypothesisItems([{ title: 't', rationale: 'r', spec: { sweep: { ghost: [1] } } }], m),
    ).toEqual([])
  })

  it('drops a proposal with an empty sweep array', () => {
    expect(
      coerceHypothesisItems([{ title: 't', rationale: 'r', spec: { sweep: { lr: [] } } }], m),
    ).toEqual([])
  })

  it('drops a proposal with no sweep and no fixed', () => {
    expect(coerceHypothesisItems([{ title: 't', rationale: 'r', spec: {} }], m)).toEqual([])
  })

  it('accepts fixed-only specs and coerces integer seeds', () => {
    const items = coerceHypothesisItems(
      [{ title: 't', rationale: 'r', spec: { fixed: { lr: 0.9 }, seeds: [0, 1.7, 'x'] } }],
      m,
    )
    expect(items[0].spec.fixed).toEqual({ lr: 0.9 })
    expect(items[0].spec.seeds).toEqual([0, 1])
  })

  it('drops items missing a title or rationale', () => {
    expect(
      coerceHypothesisItems(
        [
          { rationale: 'r', spec: { fixed: { lr: 1 } } },
          { title: 't', spec: { fixed: { lr: 1 } } },
        ],
        m,
      ),
    ).toEqual([])
  })

  it('drops a fixed value naming an unknown lever', () => {
    expect(
      coerceHypothesisItems([{ title: 't', rationale: 'r', spec: { fixed: { ghost: 1 } } }], m),
    ).toEqual([])
  })

  it('returns empty for a non-array', () => {
    expect(coerceHypothesisItems('nope' as never, m)).toEqual([])
  })
})

describe('prompt builders', () => {
  const m = manifest()

  it('judge system prompt carries the objective, direction and output shape', () => {
    const prompt = buildJudgeSystemPrompt(m, 'prefer fewer trades')
    expect(prompt).toContain('score')
    expect(prompt).toContain(m.objective.name)
    expect(prompt).toContain('max')
    expect(prompt).toContain('prefer fewer trades')
    expect(prompt).toContain('"key"')
  })

  it('judge user content is the JSON of the runs', () => {
    const content = buildJudgeUserContent([
      { key: 'a', objective: 1, config: { lr: 0.1 }, metrics: { m1: 2 }, seed: 0 },
    ])
    const parsed = JSON.parse(content)
    expect(parsed[0]).toMatchObject({ key: 'a', objective: 1 })
  })

  it('propose system prompt names the levers, count and output shape', () => {
    const prompt = buildProposeSystemPrompt(m, 3, 'explore gamma')
    expect(prompt).toContain('3')
    expect(prompt).toContain('lr')
    expect(prompt).toContain('explore gamma')
    expect(prompt).toContain('"sweep"')
  })

  it('propose user content carries runs, verdicts and the best objective', () => {
    const content = buildProposeUserContent({
      manifest: m,
      runs: [{ key: 'a', objective: 9, config: { lr: 0.1 } }],
      verdicts: [{ key: 'a', score: 80, why: 'good' }],
      bestObjective: 9,
    })
    const parsed = JSON.parse(content)
    expect(parsed.bestObjective).toBe(9)
    expect(parsed.runs[0].key).toBe('a')
    expect(parsed.verdicts[0].score).toBe(80)
  })
})

describe('parseProgressMarker', () => {
  it('parses a @@PROGRESS line into its object', () => {
    expect(parseProgressMarker('@@PROGRESS {"phase":"train","done":1200,"total":1460}')).toEqual({
      phase: 'train',
      done: 1200,
      total: 1460,
    })
  })

  it('tolerates surrounding text and whitespace', () => {
    expect(parseProgressMarker('  noise @@PROGRESS {"phase":"test"} trailing')).toEqual({
      phase: 'test',
    })
  })

  it('returns undefined for non-marker lines', () => {
    expect(parseProgressMarker('just a normal log line')).toBeUndefined()
    expect(parseProgressMarker('')).toBeUndefined()
  })

  it('returns undefined for a marker with malformed JSON', () => {
    expect(parseProgressMarker('@@PROGRESS {not json}')).toBeUndefined()
  })

  it('returns undefined when the payload is not an object', () => {
    expect(parseProgressMarker('@@PROGRESS 42')).toBeUndefined()
  })
})

describe('validateTrainingRunSummary', () => {
  it('accepts a minimal valid summary', () => {
    expect(validateTrainingRunSummary({ objective: 1.5 })).toEqual({ objective: 1.5 })
  })

  it('rejects a non-object', () => {
    expect(() => validateTrainingRunSummary(undefined)).toThrow(/summary/i)
  })

  it('rejects a missing numeric objective', () => {
    expect(() => validateTrainingRunSummary({ metrics: {} })).toThrow(/objective/)
  })

  it('rejects a NaN objective', () => {
    expect(() => validateTrainingRunSummary({ objective: Number.NaN })).toThrow(/objective/)
  })
})
