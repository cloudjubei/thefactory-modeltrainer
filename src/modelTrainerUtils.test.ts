import { describe, expect, it } from 'vitest'
import type { TrainerManifest } from './modelTrainerTypes.js'
import {
  blendJudgeScore,
  parseProgressMarker,
  buildAnalyzePaperSystemPrompt,
  buildJudgeSystemPrompt,
  buildJudgeUserContent,
  buildProposeSystemPrompt,
  buildProposeUserContent,
  buildSuggestHypothesesSystemPrompt,
  buildSuggestHypothesesUserContent,
  buildXaiNarrateSystemPrompt,
  buildXaiNarrateUserContent,
  applyMigrationRules,
  findMigrationRule,
  migrateExperimentSpec,
  canonicalConfigString,
  coerceHypothesisItems,
  coercePaperDraft,
  coerceSuggestedHypotheses,
  coerceVerdictRows,
  looksLikeDataGathering,
  extractPaperText,
  expandExperimentMatrix,
  normalizeObjectiveScores,
  pickBestRun,
  totalCampaignUnits,
  datasetAlignmentSignature,
  diffDecisionTraces,
  validateDecisionTrace,
  validateTrainerManifest,
  validateTrainingRunSummary,
} from './modelTrainerUtils.js'
import type { TrainingRunSummary } from './modelTrainerTypes.js'

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

describe('applyMigrationRules', () => {
  const rules = [
    {
      match: { reward_model: 'combo_all' },
      set: {
        reward_model: 'combo_unified',
        combo_sell: 1000,
        combo_wrongaction: 0,
        combo_fee_penalty: 0,
      },
    },
    {
      match: { reward_model: 'combo_all_fee' },
      set: { reward_model: 'combo_unified', combo_sell: 1000, combo_wrongaction: 0 },
      keepOrDefault: { combo_fee_penalty: 1.0 },
    },
    {
      match: { reward_model: 'combo_all2', combo_noaction: 0 },
      set: { reward_model: 'combo_unified', combo_sell: 1000 },
      keepOrDefault: { combo_wrongaction: -0.01 },
    },
  ]

  it('returns null when no rule matches (idempotent for already-migrated configs)', () => {
    expect(
      applyMigrationRules({ reward_model: 'combo_unified', combo_sell: 1000 }, rules),
    ).toBeNull()
    expect(applyMigrationRules({ reward_model: 'profit_all2' }, rules)).toBeNull()
    expect(applyMigrationRules({ lr: 0.1 }, rules)).toBeNull()
  })

  it("applies the matching rule's set fields over the source config", () => {
    const out = applyMigrationRules(
      { reward_model: 'combo_all', combo_noaction: -1, lr: 0.1 },
      rules,
    )
    expect(out).toEqual({
      reward_model: 'combo_unified',
      combo_noaction: -1,
      lr: 0.1,
      combo_sell: 1000,
      combo_wrongaction: 0,
      combo_fee_penalty: 0,
    })
  })

  it('keeps an existing keepOrDefault value, else writes the default', () => {
    const kept = applyMigrationRules(
      { reward_model: 'combo_all_fee', combo_fee_penalty: 0.5 },
      rules,
    )
    expect(kept?.combo_fee_penalty).toBe(0.5)
    const defaulted = applyMigrationRules({ reward_model: 'combo_all_fee' }, rules)
    expect(defaulted?.combo_fee_penalty).toBe(1.0)
  })

  it('requires every match key (multi-key match) and compares loosely', () => {
    // combo_all2 only unifies under combo_noaction == 0
    expect(
      applyMigrationRules({ reward_model: 'combo_all2', combo_noaction: -1 }, rules),
    ).toBeNull()
    const out = applyMigrationRules({ reward_model: 'combo_all2', combo_noaction: 0 }, rules)
    expect(out?.reward_model).toBe('combo_unified')
    expect(out?.combo_wrongaction).toBe(-0.01)
    // a stored numeric 0 matched by a JSON 0 even across number/string boundaries
    expect(
      applyMigrationRules({ reward_model: 'combo_all2', combo_noaction: '0' }, rules)?.reward_model,
    ).toBe('combo_unified')
  })

  it('uses the FIRST matching rule and does not mutate the input', () => {
    const input = { reward_model: 'combo_all', combo_sell: 7 }
    const out = applyMigrationRules(input, rules)
    expect(out?.combo_sell).toBe(1000)
    expect(input.combo_sell).toBe(7)
  })

  it('returns null for an empty rule set', () => {
    expect(applyMigrationRules({ reward_model: 'combo_all' }, [])).toBeNull()
  })

  it('returns null (no rewrite) when the matched rule is a delete rule', () => {
    const rules = [{ matchNot: { reward_model: 'combo_unified' }, delete: true }]
    expect(applyMigrationRules({ reward_model: 'profit_percentage3' }, rules)).toBeNull()
  })
})

describe('findMigrationRule', () => {
  const rewrite = { match: { reward_model: 'combo_all' }, set: { reward_model: 'combo_unified' } }
  const del = { matchNot: { reward_model: 'combo_unified' }, delete: true }
  const rules = [rewrite, del]

  it('picks the rewrite rule for a matched old name (rewrite wins over the trailing delete rule)', () => {
    expect(findMigrationRule({ reward_model: 'combo_all' }, rules)).toBe(rewrite)
  })

  it('picks the delete rule (matchNot) for any other present reward_model', () => {
    expect(findMigrationRule({ reward_model: 'profit_percentage3' }, rules)).toBe(del)
    expect(findMigrationRule({ reward_model: 'profit_all2' }, rules)).toBe(del)
    expect(findMigrationRule({ reward_model: 'buy_sell_signal' }, rules)).toBe(del)
  })

  it('does NOT match the delete rule for the kept value (combo_unified)', () => {
    expect(findMigrationRule({ reward_model: 'combo_unified' }, rules)).toBeNull()
  })

  it('does NOT match a matchNot rule when the field is absent (e.g. hodl/supervised runs)', () => {
    expect(findMigrationRule({ model_name: 'hodl' }, rules)).toBeNull()
  })

  it('requires match fields to be PRESENT (a missing match key never fires)', () => {
    const rule = {
      match: { reward_model: 'combo_all', combo_noaction: 0 },
      set: { reward_model: 'combo_unified' },
    }
    expect(findMigrationRule({ reward_model: 'combo_all' }, [rule])).toBeNull()
    expect(findMigrationRule({ reward_model: 'combo_all', combo_noaction: 0 }, [rule])).toBe(rule)
  })

  it('ignores a rule with neither match nor matchNot', () => {
    expect(findMigrationRule({ reward_model: 'x' }, [{ delete: true }])).toBeNull()
  })
})

describe('migrateExperimentSpec', () => {
  const migrations = [
    {
      match: { reward_model: 'combo_all' },
      set: { reward_model: 'combo_unified', combo_sell: 1000 },
    },
  ]

  it('migrates spec.fixed in place so a dispatched run plans under the new shape', () => {
    const out = migrateExperimentSpec(
      { fixed: { reward_model: 'combo_all', lr: 0.1 }, seeds: [0] },
      migrations,
    )
    expect(out.fixed).toEqual({ reward_model: 'combo_unified', lr: 0.1, combo_sell: 1000 })
    expect(out.seeds).toEqual([0])
  })

  it('returns the SAME spec object (cheap pass-through) when nothing matches or no migrations', () => {
    const spec = { fixed: { reward_model: 'combo_unified' } }
    expect(migrateExperimentSpec(spec, migrations)).toBe(spec)
    expect(migrateExperimentSpec(spec, [])).toBe(spec)
    expect(migrateExperimentSpec(spec, undefined)).toBe(spec)
  })

  it('is a no-op when the spec has no fixed block (e.g. a pure sweep)', () => {
    const spec = { sweep: { reward_model: ['combo_all', 'profit_all2'] } }
    expect(migrateExperimentSpec(spec, migrations)).toBe(spec)
  })
})

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

  it('crosses each configuration with dataset bundles (applied together, not cartesian)', () => {
    const items = expandExperimentMatrix(
      manifest(),
      {
        sweep: { algo: ['a', 'b'] },
        datasets: [
          { lr: 0.001, steps: 10 },
          { lr: 0.002, steps: 20 },
        ],
      },
      hashByJson,
    )
    expect(items).toHaveLength(4)
    const ds0 = items.filter((i) => i.config.lr === 0.001)
    expect(ds0).toHaveLength(2)
    expect(ds0.every((i) => i.config.steps === 10)).toBe(true)
  })

  it('crosses datasets AND environments together (model × dataset × environment)', () => {
    const items = expandExperimentMatrix(
      manifest(),
      {
        sweep: { algo: ['a', 'b'] },
        datasets: [{ lr: 0.001 }, { lr: 0.002 }],
        environments: [{ steps: 10 }, { steps: 20 }],
      },
      hashByJson,
    )
    // 2 algos × 2 datasets × 2 environments = 8
    expect(items).toHaveLength(8)
  })

  it('rejects a dataset value that names no lever', () => {
    expect(() =>
      expandExperimentMatrix(manifest(), { datasets: [{ ghost: 1 }] }, hashByJson),
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

  it('drops data-gathering "hypotheses" (more seeds to establish an interval — not a test)', () => {
    const items = coerceHypothesisItems(
      [
        {
          title: 'Top Setup 1 Interval',
          rationale:
            'Running 4 more seeds for the best performing configuration to establish a trustworthy interval (≥5).',
          spec: { fixed: { lr: 0.5 } },
        },
        {
          title: 'Higher lr beats the baseline',
          rationale: 'test the lr effect',
          spec: { sweep: { lr: [0.2] } },
        },
      ],
      m,
    )
    expect(items).toHaveLength(1)
    expect(items[0].title).toBe('Higher lr beats the baseline')
  })
})

describe('looksLikeDataGathering', () => {
  it('flags add-more-seeds / establish-interval / reduce-noise phrasing', () => {
    expect(
      looksLikeDataGathering(
        'Top Setup 1 Interval',
        'Run 4 more seeds to establish a trustworthy interval (≥5).',
      ),
    ).toBe(true)
    expect(
      looksLikeDataGathering('Tighten estimate', 'additional seeds for a reliable interval'),
    ).toBe(true)
    expect(looksLikeDataGathering('Confirm', 'gather more runs to confirm the result')).toBe(true)
  })
  it('does NOT flag real test-claims (even when they mention variance)', () => {
    expect(
      looksLikeDataGathering('Higher lr beats baseline', 'a larger lr should raise the objective'),
    ).toBe(false)
    expect(
      looksLikeDataGathering(
        'Lower lr reduces return variance',
        'tests whether lr lowers variance',
      ),
    ).toBe(false)
    expect(
      looksLikeDataGathering('Attention beats LSTM', 'compare attn-ppo vs reppo on Sharpe'),
    ).toBe(false)
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

  it('xai-narrate system prompt names the project, objective and demands an honest per-run narrative', () => {
    const prompt = buildXaiNarrateSystemPrompt(m)
    expect(prompt).toContain(m.name)
    expect(prompt).toContain(m.objective.name)
    expect(prompt).toContain(m.objective.direction)
    expect(prompt).toMatch(/SHORT narrative/)
    expect(prompt).toMatch(/ONE specific run/)
    expect(prompt).toMatch(/sanity check/)
  })

  it('xai-narrate user content digests one run: decisions, attribution+sanity, reward, latent, sibling, context', () => {
    const content = buildXaiNarrateUserContent({
      runKey: 'abc123def456',
      config: { lr: 0.1, seed: 0 },
      objective: 12.34,
      criterion: { key: 'objective', direction: 'max', label: 'traded return' },
      rank: { position: 2, total: 9 },
      actionCounts: { hold: 80, buy: 15, sell: 5 },
      attribution: {
        topGroups: [
          ['1h', 0.42],
          ['1d', 0.18],
        ],
        method: 'integrated-gradients',
        sanityPassed: false,
        sanityRankCorr: 0.91,
      },
      rewardBreakdown: { base: 1.2, turnover_penalty: -0.3 },
      latent: { varianceExplained: 0.7, probeAccuracy: 0.82, probeBaseline: 0.6 },
      importances: [{ lever: 'lr', importance: 0.5, bestValue: '0.1' }],
      sibling: {
        key: 'sib9999',
        changed: 'lr 0.9→0.1',
        divergencePct: 23,
        qualityVerdict: 'better',
        qualitySummary: 'earned more where decisions changed',
      },
    })
    expect(content).toContain('abc123de') // 8-char run id, seed dropped from the config digest
    expect(content).not.toMatch(/seed=0/)
    expect(content).toMatch(/traded return: 12.34 \(ranks #2 of 9\)/)
    expect(content).toMatch(/Action mix: hold=80, buy=15, sell=5/)
    expect(content).toMatch(/integrated-gradients.*1h\(0.42\)/)
    expect(content).toMatch(/sanity check FAILED \(rank corr 0.91\)/)
    expect(content).toMatch(/turnover_penalty -0.3/)
    expect(content).toMatch(/linear probe.*82% vs a 60%/)
    expect(content).toMatch(/sibling sib9999 \(changed lr 0.9→0.1\): 23%.*better/)
    expect(content).toMatch(/lever importance.*lr 50% \(best≈0.1\)/)
  })

  it('xai-narrate user content omits empty sections and marks a PASSED sanity check', () => {
    const content = buildXaiNarrateUserContent({
      runKey: 'deadbeef',
      config: {},
      criterion: { key: 'objective', direction: 'min' },
      attribution: { topGroups: [['1h', 0.1]], sanityPassed: true, sanityRankCorr: 0.05 },
      importances: [],
    })
    expect(content).toContain('objective: n/a')
    expect(content).toMatch(/sanity check PASSED \(rank corr 0.05\)/)
    expect(content).not.toContain('Action mix')
    expect(content).not.toContain('Reward breakdown')
    expect(content).not.toContain('Latent')
    expect(content).not.toContain('nearest sibling')
    expect(content).not.toContain('lever importance')
  })

  it('xai-narrate user content handles attribution without a sanity check and latent without variance', () => {
    const content = buildXaiNarrateUserContent({
      runKey: 'cafe',
      config: { lr: 0.2 },
      criterion: { key: 'objective', direction: 'max' },
      attribution: { topGroups: [['1h', 0.3]] },
      latent: { probeAccuracy: 0.7 },
      importances: [],
    })
    expect(content).toMatch(/Input attribution \(saliency\): 1h\(0.3\)\./)
    expect(content).not.toContain('sanity check')
    expect(content).toMatch(/linear probe.*70% vs a 0% majority baseline\./)
    expect(content).not.toContain('variance')
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

describe('validateDecisionTrace', () => {
  it('returns undefined for a non-object (missing trace ≠ error)', () => {
    expect(validateDecisionTrace(undefined)).toBeUndefined()
    expect(validateDecisionTrace(null)).toBeUndefined()
    expect(validateDecisionTrace(42)).toBeUndefined()
    expect(validateDecisionTrace([])).toBeUndefined()
  })

  it('returns undefined when steps is missing or not an array', () => {
    expect(validateDecisionTrace({})).toBeUndefined()
    expect(validateDecisionTrace({ steps: 'nope' })).toBeUndefined()
  })

  it('returns undefined when no step is well-formed', () => {
    expect(validateDecisionTrace({ steps: [{ action: 'buy' }, { step: 1 }, 'x'] })).toBeUndefined()
  })

  it('keeps only well-formed steps and coerces the minimal shape', () => {
    const trace = validateDecisionTrace({
      steps: [
        { step: 0, action: 'hold' },
        { step: 'bad', action: 'buy' },
        { step: 1, action: 5 },
        { step: 2, action: 'sell' },
      ],
    })
    expect(trace?.steps).toEqual([
      { step: 0, action: 'hold' },
      { step: 2, action: 'sell' },
    ])
  })

  it('carries optional per-step fields only when well-typed', () => {
    const trace = validateDecisionTrace({
      steps: [
        {
          step: 3,
          action: 'buy',
          confidence: 0.8,
          actionValues: { hold: 1, buy: 2.5, bad: 'x' },
          alternativeAction: 'hold',
          forced: true,
          reward: -0.01,
          state: 'long',
          features: [0.1, 0.2, 'nope'],
        },
      ],
    })
    expect(trace?.steps[0]).toEqual({
      step: 3,
      action: 'buy',
      confidence: 0.8,
      actionValues: { hold: 1, buy: 2.5 },
      alternativeAction: 'hold',
      forced: true,
      reward: -0.01,
      state: 'long',
      features: [0.1, 0.2],
    })
  })

  it('drops non-finite numeric fields and empty coerced maps', () => {
    const trace = validateDecisionTrace({
      steps: [
        {
          step: 0,
          action: 'hold',
          confidence: Number.NaN,
          reward: Number.POSITIVE_INFINITY,
          actionValues: { bad: 'x' },
        },
      ],
    })
    expect(trace?.steps[0]).toEqual({ step: 0, action: 'hold' })
  })

  it('coerces actionCounts, totalSteps and featureAttribution', () => {
    const trace = validateDecisionTrace({
      steps: [{ step: 0, action: 'hold' }],
      actionCounts: { hold: 10, buy: 3, bad: 'x' },
      totalSteps: 13,
      featureAttribution: {
        perFeature: [0.5, 'x', 0.2],
        byGroup: { '1h': 0.7, '1d': 0.3, bad: null },
        method: 'gradient-saliency',
        samples: 12,
      },
    })
    expect(trace?.actionCounts).toEqual({ hold: 10, buy: 3 })
    expect(trace?.totalSteps).toBe(13)
    expect(trace?.featureAttribution).toEqual({
      perFeature: [0.5, 0.2],
      byGroup: { '1h': 0.7, '1d': 0.3 },
      method: 'gradient-saliency',
      samples: 12,
    })
  })

  it('coerces the saliency sanityCheck (Adebayo model-randomization)', () => {
    const trace = validateDecisionTrace({
      steps: [{ step: 0, action: 'hold' }],
      featureAttribution: {
        perFeature: [0.5, 0.2],
        sanityCheck: {
          method: 'model-randomization',
          rankCorrelation: 0.12,
          passed: true,
          junk: 1,
        },
      },
    })
    expect(trace?.featureAttribution?.sanityCheck).toEqual({
      method: 'model-randomization',
      rankCorrelation: 0.12,
      passed: true,
    })
  })

  it('drops an empty/garbage sanityCheck', () => {
    const trace = validateDecisionTrace({
      steps: [{ step: 0, action: 'hold' }],
      featureAttribution: {
        perFeature: [0.5],
        sanityCheck: { rankCorrelation: 'x', passed: 'no' },
      },
    })
    expect(trace?.featureAttribution?.sanityCheck).toBeUndefined()
  })

  it('coerces a rewardBreakdown (why this reward), dropping non-numeric entries', () => {
    const trace = validateDecisionTrace({
      steps: [{ step: 0, action: 'hold' }],
      rewardBreakdown: {
        base: 3.2,
        turnover_penalty: -0.4,
        noop_penalty: -0.1,
        total: 2.7,
        bad: 'x',
      },
    })
    expect(trace?.rewardBreakdown).toEqual({
      base: 3.2,
      turnover_penalty: -0.4,
      noop_penalty: -0.1,
      total: 2.7,
    })
  })

  it('coerces a latentMap, dropping malformed points and requiring ≥3', () => {
    const trace = validateDecisionTrace({
      steps: [{ step: 0, action: 'hold' }],
      latentMap: {
        points: [
          { x: 0.1, y: 0.2, action: 'buy' },
          { x: -0.3, y: 0.4, action: 'sell' },
          { x: 0.5, y: -0.1, action: 'hold' },
          { x: 'bad', y: 0, action: 'hold' },
        ],
        varianceExplained: 0.71,
        dim: 64,
        method: 'pca',
        probe: {
          accuracy: 0.88,
          baseline: 0.45,
          classes: 3,
          method: 'ridge-linear',
          testSize: 60,
          bad: 'x',
        },
      },
    })
    expect(trace?.latentMap?.points).toHaveLength(3)
    expect(trace?.latentMap?.varianceExplained).toBe(0.71)
    expect(trace?.latentMap?.dim).toBe(64)
    expect(trace?.latentMap?.probe).toEqual({
      accuracy: 0.88,
      baseline: 0.45,
      classes: 3,
      method: 'ridge-linear',
      testSize: 60,
    })
  })

  it('drops a latentMap with too few valid points', () => {
    const trace = validateDecisionTrace({
      steps: [{ step: 0, action: 'hold' }],
      latentMap: { points: [{ x: 0, y: 0, action: 'buy' }] },
    })
    expect(trace?.latentMap).toBeUndefined()
  })

  it('omits featureAttribution when it has no usable content', () => {
    const trace = validateDecisionTrace({
      steps: [{ step: 0, action: 'hold' }],
      featureAttribution: { method: 7, byGroup: { bad: 'x' } },
    })
    expect(trace?.featureAttribution).toBeUndefined()
    expect(trace?.actionCounts).toBeUndefined()
  })
})

describe('datasetAlignmentSignature', () => {
  const ds = (over: Record<string, unknown> = {}): TrainingRunSummary => ({
    objective: 0,
    dataset: { asset: 'BTC', timeframe: '1h', candles: 100, from: 'a', to: 'b', ...over },
  })

  it('builds a stable signature from the step-axis dataset fields', () => {
    expect(datasetAlignmentSignature(ds())).toBe('asset=BTC|timeframe=1h|candles=100|from=a|to=b')
  })

  it('ignores observation-only fields like fidelity_set so a fidelity tweak stays alignable', () => {
    expect(datasetAlignmentSignature(ds({ fidelity_set: '1h+1d', layers: ['1h', '1d'] }))).toBe(
      datasetAlignmentSignature(ds({ fidelity_set: '1h' })),
    )
  })

  it('returns empty when there is no dataset', () => {
    expect(datasetAlignmentSignature({ objective: 0 })).toBe('')
  })

  it('differs when the window differs', () => {
    expect(datasetAlignmentSignature(ds())).not.toBe(datasetAlignmentSignature(ds({ to: 'c' })))
  })
})

describe('diffDecisionTraces', () => {
  type StepSpec = { b: string; t: string; rb?: number; rt?: number; cb?: number; ct?: number }
  const DS = { asset: 'BTC', timeframe: '1h', candles: 100, from: 'a', to: 'b' }
  const run = (
    steps: Array<Record<string, unknown>>,
    extra: Partial<TrainingRunSummary> & { actionCounts?: Record<string, number> } = {},
  ): TrainingRunSummary => {
    const { actionCounts, ...rest } = extra
    return {
      objective: 0,
      dataset: DS,
      ...rest,
      artifacts: {
        decisionTrace: {
          steps,
          totalSteps: steps.length,
          ...(actionCounts ? { actionCounts } : {}),
        },
      },
    }
  }
  const pair = (specs: StepSpec[], opts: { objB?: number; objT?: number } = {}) => {
    const side = (k: 'b' | 't', rk: 'rb' | 'rt', ck: 'cb' | 'ct') =>
      specs.map((s, i) => ({
        step: i,
        action: s[k],
        ...(s[rk] !== undefined ? { reward: s[rk] } : {}),
        ...(s[ck] !== undefined ? { confidence: s[ck] } : {}),
      }))
    return {
      baseline: run(side('b', 'rb', 'cb'), { objective: opts.objB ?? 0 }),
      tweak: run(side('t', 'rt', 'ct'), { objective: opts.objT ?? 0 }),
    }
  }
  // n changed steps with a fixed tweak−baseline reward delta, then m unchanged steps with a control delta.
  const churn = (nChanged: number, changeDelta: number, mUnchanged: number, ctrlDelta: number) => [
    ...Array.from({ length: nChanged }, () => ({ b: 'hold', t: 'buy', rb: 0, rt: changeDelta })),
    ...Array.from({ length: mUnchanged }, () => ({ b: 'hold', t: 'hold', rb: 0, rt: ctrlDelta })),
  ]

  it('returns undefined when the baseline has no usable trace', () => {
    const { tweak } = pair([{ b: 'hold', t: 'buy' }])
    expect(diffDecisionTraces({ objective: 0, dataset: DS }, tweak)).toBeUndefined()
  })

  it('returns undefined when the tweak has no usable trace', () => {
    const { baseline } = pair([{ b: 'hold', t: 'buy' }])
    expect(diffDecisionTraces(baseline, { objective: 0, dataset: DS })).toBeUndefined()
  })

  it('reports aligned:false with a note when the dataset differs', () => {
    const { baseline, tweak } = pair([{ b: 'hold', t: 'buy' }])
    const other = { ...tweak, dataset: { ...DS, asset: 'ETH' } }
    const diff = diffDecisionTraces(baseline, other)
    expect(diff?.aligned).toBe(false)
    expect(diff?.alignmentNote).toMatch(/dataset/i)
  })

  it('reports aligned:false when totalSteps differ', () => {
    const baseline = run([{ step: 0, action: 'hold' }], { objective: 0 })
    const tweak = run([{ step: 0, action: 'buy' }], { objective: 0 })
    ;(tweak.artifacts!.decisionTrace as { totalSteps: number }).totalSteps = 99
    const diff = diffDecisionTraces(baseline, tweak)
    expect(diff?.aligned).toBe(false)
    expect(diff?.alignmentNote).toMatch(/totalSteps/i)
  })

  it('reports aligned:false when there are no shared step indices', () => {
    const baseline = run([{ step: 0, action: 'hold' }], { objective: 0 })
    const tweak = run([{ step: 7, action: 'buy' }], { objective: 0 })
    ;(tweak.artifacts!.decisionTrace as { totalSteps: number }).totalSteps = 1
    const diff = diffDecisionTraces(baseline, tweak)
    expect(diff?.aligned).toBe(false)
    expect(diff?.alignmentNote).toMatch(/shared/i)
  })

  it('computes alignment counts and divergence over shared steps', () => {
    const { baseline, tweak } = pair([
      { b: 'hold', t: 'hold' },
      { b: 'hold', t: 'buy' },
      { b: 'sell', t: 'buy' },
      { b: 'hold', t: 'hold' },
    ])
    const diff = diffDecisionTraces(baseline, tweak)!
    expect(diff.aligned).toBe(true)
    expect(diff.alignedSteps).toBe(4)
    expect(diff.changedSteps).toBe(2)
    expect(diff.divergenceRate).toBe(0.5)
    expect(diff.steps[1]).toMatchObject({
      step: 1,
      baselineAction: 'hold',
      tweakAction: 'buy',
      changed: true,
    })
  })

  it('records rewardDelta only when both steps have a reward', () => {
    const { baseline, tweak } = pair([
      { b: 'hold', t: 'buy', rb: 1, rt: 1.5 },
      { b: 'hold', t: 'buy', rt: 2 },
    ])
    const diff = diffDecisionTraces(baseline, tweak)!
    expect(diff.steps[0].rewardDelta).toBeCloseTo(0.5)
    expect(diff.steps[1].rewardDelta).toBeUndefined()
  })

  it('records confidenceDelta and the mean shift only over steps with both confidences', () => {
    const { baseline, tweak } = pair([
      { b: 'hold', t: 'buy', cb: 0.4, ct: 0.6 },
      { b: 'hold', t: 'buy', ct: 0.9 },
    ])
    const diff = diffDecisionTraces(baseline, tweak)!
    expect(diff.steps[0].confidenceDelta).toBeCloseTo(0.2)
    expect(diff.steps[1].confidenceDelta).toBeUndefined()
    expect(diff.meanConfidenceShift).toBeCloseTo(0.2)
  })

  it('tallies non-zero action-count deltas including labels in only one run', () => {
    const { baseline, tweak } = pair([{ b: 'hold', t: 'buy' }])
    baseline.artifacts!.decisionTrace = {
      ...(baseline.artifacts!.decisionTrace as object),
      actionCounts: { hold: 10, sell: 4 },
    }
    tweak.artifacts!.decisionTrace = {
      ...(tweak.artifacts!.decisionTrace as object),
      actionCounts: { hold: 7, buy: 3 },
    }
    const diff = diffDecisionTraces(baseline, tweak)!
    expect(diff.actionCountDeltas).toEqual({ hold: -3, sell: -4, buy: 3 })
  })

  it('folds in the objective delta as context', () => {
    const { baseline, tweak } = pair([{ b: 'hold', t: 'buy' }], { objB: 1, objT: 3 })
    expect(diffDecisionTraces(baseline, tweak)!.objectiveDelta).toBe(2)
  })

  it('reads quality "insufficient" with too few scored changed steps', () => {
    const { baseline, tweak } = pair(churn(3, 0.1, 4, 0))
    const q = diffDecisionTraces(baseline, tweak)!.quality
    expect(q.verdict).toBe('insufficient')
    expect(q.scoredChangedSteps).toBe(3)
  })

  it('reads quality "better" when changed steps beat the unchanged control', () => {
    const q = diffDecisionTraces(
      ...(Object.values(pair(churn(6, 0.1, 6, 0))) as [TrainingRunSummary, TrainingRunSummary]),
    )!.quality
    expect(q.verdict).toBe('better')
    expect(q.meanRewardDeltaOnChanges).toBeCloseTo(0.1)
    expect(q.meanRewardDeltaOnUnchanged).toBeCloseTo(0)
  })

  it('reads quality "worse" symmetrically', () => {
    const { baseline, tweak } = pair(churn(6, -0.1, 6, 0))
    expect(diffDecisionTraces(baseline, tweak)!.quality.verdict).toBe('worse')
  })

  it('reads quality "mixed" when the whole rollout shifted equally (control trips)', () => {
    const { baseline, tweak } = pair(churn(6, 0.1, 6, 0.1))
    expect(diffDecisionTraces(baseline, tweak)!.quality.verdict).toBe('mixed')
  })

  it('reads quality "unchanged" when the changed-step delta is ~0', () => {
    const { baseline, tweak } = pair(churn(6, 0, 6, 0))
    expect(diffDecisionTraces(baseline, tweak)!.quality.verdict).toBe('unchanged')
  })

  it('always labels the quality summary as a heuristic', () => {
    const { baseline, tweak } = pair(churn(6, 0.1, 6, 0))
    expect(diffDecisionTraces(baseline, tweak)!.quality.summary.toLowerCase()).toContain(
      'heuristic',
    )
  })

  it('is "insufficient" with no per-step rewards anywhere', () => {
    const { baseline, tweak } = pair([
      { b: 'hold', t: 'buy' },
      { b: 'sell', t: 'buy' },
    ])
    const q = diffDecisionTraces(baseline, tweak)!.quality
    expect(q.verdict).toBe('insufficient')
    expect(q.meanRewardDeltaOnChanges).toBeUndefined()
  })
})

describe('extractPaperText', () => {
  it('strips tags + scripts/styles and decodes common entities', () => {
    const html =
      '<html><head><style>x{}</style><script>bad()</script></head><body><h1>Title</h1><p>A &amp; B &lt; C</p></body></html>'
    const text = extractPaperText(html)
    expect(text).toContain('Title')
    expect(text).toContain('A & B < C')
    expect(text).not.toMatch(/<[a-z/]/i) // no HTML tags remain (a decoded "< C" is fine)
    expect(text).not.toContain('bad()')
  })
  it('caps very long input', () => {
    expect(extractPaperText('x'.repeat(20000)).length).toBe(12000)
  })
})

describe('buildAnalyzePaperSystemPrompt', () => {
  const m = {
    name: 'Demo',
    recordType: 'demo-run',
    run: '{configPath} {summaryOut}',
    objective: { name: 'obj', direction: 'max' },
    levers: { lr: { type: 'number' } },
  } as unknown as TrainerManifest
  it('asks for a single JSON object and includes the levers + notes', () => {
    const p = buildAnalyzePaperSystemPrompt(m, 'focus on fees')
    expect(p).toMatch(/JSON object/i)
    expect(p).toContain('lr')
    expect(p).toContain('focus on fees')
  })
  it('requests the testable hypotheses array', () => {
    const p = buildAnalyzePaperSystemPrompt(m)
    expect(p).toContain('hypotheses')
  })
})

describe('coercePaperDraft', () => {
  it('keeps a well-formed draft and drops ill-typed fields', () => {
    const d = coercePaperDraft({
      title: ' T ',
      claim: 'C',
      year: 2023,
      authors: 'A',
      claimedMetrics: { sharpe: 1.2, bad: 'x' },
      tags: ['a', 1],
      assumptions: { fees: false },
      junk: 'ignored',
    })
    expect(d).toMatchObject({ title: 'T', claim: 'C', year: 2023, authors: 'A' })
    expect(d?.claimedMetrics).toEqual({ sharpe: 1.2 })
    expect(d?.tags).toEqual(['a'])
    expect((d as Record<string, unknown>).junk).toBeUndefined()
  })
  it('drops replicateConfig (the paper no longer carries one — its hypotheses hold the specs)', () => {
    const d = coercePaperDraft({ title: 'T', claim: 'C', replicateConfig: { fixed: { lr: 1 } } })
    expect((d as Record<string, unknown>).replicateConfig).toBeUndefined()
  })
  it('returns undefined without a title or claim', () => {
    expect(coercePaperDraft({ title: 'T' })).toBeUndefined()
    expect(coercePaperDraft({ claim: 'C' })).toBeUndefined()
    expect(coercePaperDraft('nope')).toBeUndefined()
  })
})

describe('buildSuggestHypothesesSystemPrompt', () => {
  const m = {
    name: 'Demo',
    recordType: 'demo-run',
    run: '{configPath} {summaryOut}',
    objective: { name: 'obj', direction: 'max' },
    levers: { lr: { type: 'number' } },
  } as unknown as TrainerManifest
  it('asks to MATCH existing + SUGGEST new, and includes the levers', () => {
    const p = buildSuggestHypothesesSystemPrompt(m)
    expect(p).toMatch(/matchExistingIds/)
    expect(p).toMatch(/newHypotheses/)
    expect(p).toContain('lr')
  })
})

describe('the hypothesis prompts forbid data-gathering', () => {
  const m = {
    name: 'Demo',
    recordType: 'demo-run',
    run: '{configPath} {summaryOut}',
    objective: { name: 'obj', direction: 'max' },
    levers: { lr: { type: 'number' } },
  } as unknown as TrainerManifest
  it('every prompt that asks for hypotheses demands a FALSIFIABLE claim, not data-gathering', () => {
    for (const p of [
      buildProposeSystemPrompt(m, 5),
      buildAnalyzePaperSystemPrompt(m),
      buildSuggestHypothesesSystemPrompt(m),
    ]) {
      expect(p).toMatch(/FALSIFIABLE/)
      expect(p).toMatch(/DATA-GATHERING/)
      expect(p).toMatch(/trustworthy interval/)
    }
  })
})

describe('buildSuggestHypothesesUserContent', () => {
  it('serializes the paper + existing hypotheses, and text only when present', () => {
    const withText = JSON.parse(
      buildSuggestHypothesesUserContent({
        paper: { title: 'P' },
        existingHypotheses: [{ id: 'h1', title: 'X' }],
        text: 'body',
      }),
    )
    expect(withText).toMatchObject({ paper: { title: 'P' }, text: 'body' })
    expect(withText.existingHypotheses).toHaveLength(1)
    const noText = JSON.parse(
      buildSuggestHypothesesUserContent({ paper: { title: 'P' }, existingHypotheses: [] }),
    )
    expect('text' in noText).toBe(false)
  })
})

describe('coerceSuggestedHypotheses', () => {
  it('extracts string match ids and coerces valid new hypotheses', () => {
    const out = coerceSuggestedHypotheses(
      {
        matchExistingIds: ['h1', 'h2', 7, ''],
        newHypotheses: [
          { title: 'N', rationale: 'R', spec: { fixed: { lr: 0.5 } } },
          { title: 'bad', rationale: 'R', spec: { fixed: { unknown_lever: 1 } } },
        ],
      },
      manifest(),
    )
    expect(out.matchIds).toEqual(['h1', 'h2'])
    expect(out.newItems).toHaveLength(1)
    expect(out.newItems[0]).toMatchObject({ title: 'N' })
  })
  it('defaults to empty for a malformed response', () => {
    expect(coerceSuggestedHypotheses('nope', manifest())).toEqual({ matchIds: [], newItems: [] })
    expect(coerceSuggestedHypotheses({}, manifest())).toEqual({ matchIds: [], newItems: [] })
  })
})
