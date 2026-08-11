import { describe, expect, it } from 'vitest'
import type { TrainerManifest, TrainingRunSummary } from './modelTrainerTypes.js'
import { MAX_DECISION_TRACE_STEPS, MAX_SERIES_POINTS } from './modelTrainerConstants.js'
import {
  blendJudgeScore,
  extractSampleGame,
  renderReplayText,
  parseProgressMarker,
  buildAnalyzePaperSystemPrompt,
  buildJudgeSystemPrompt,
  buildJudgeUserContent,
  buildProposeSystemPrompt,
  buildProposeUserContent,
  buildSuggestHypothesesSystemPrompt,
  buildSuggestHypothesesUserContent,
  buildXaiNarrateSystemPrompt,
  buildReliabilitySystemPrompt,
  coerceReliabilityVerdict,
  buildXaiNarrateUserContent,
  applyMigrationRules,
  resolveCampaignParallelism,
  resolveModelDeviceForConfig,
  parseDeviceBenchmark,
  parseDataCatalog,
  catalogAssetIds,
  withCatalogChoices,
  parseDataLinkage,
  parseMineResult,
  buildDataDiscoveryGoal,
  coerceDataSourceCandidates,
  slugifyDataSource,
  missingCrossTestValues,
  THREAD_ENV_VARS,
  isRunAffectedByFidelityDesync,
  isSpecAffectedByFidelityDesync,
  isRunAffectedByFidelityLookahead,
  isSpecAffectedByFidelityLookahead,
  findMigrationRule,
  migrateExperimentSpec,
  validateSpecAgainstManifest,
  canonicalConfigString,
  estimateRemainingCampaignSeconds,
  coerceHypothesisItems,
  looksComparative,
  coercePaperDraft,
  coerceSuggestedHypotheses,
  coerceHypothesisWeights,
  coerceHypothesisCoverage,
  coerceVerdictRows,
  looksLikeDataGathering,
  extractPaperText,
  expandExperimentMatrix,
  normalizeObjectiveScores,
  pickBestRun,
  totalCampaignUnits,
  datasetAlignmentSignature,
  diffDecisionTraces,
  summarizeSnapshotTraces,
  summarizeStepAttribution,
  validateDecisionTrace,
  validateTrainerManifest,
  validateTrainingRunSummary,
  validateRunProvenance,
  describeRunFailures,
  capRunSummaryForStorage,
  plannedMatrixItemCount,
  modelSlug,
  inferModelCategory,
  humanizeModelName,
  modelBindingNames,
  discoverManifestModelCandidates,
  dedupeModelsBySlug,
  coerceScannedModels,
  buildScanModelsSystemPrompt,
  buildScanModelsUserContent,
  coerceAnalyzedPaperModels,
  mergeProposedImprovements,
  detectMissingPaperModels,
  buildAnalyzePaperModelsSystemPrompt,
  buildAnalyzePaperModelsUserContent,
  coerceConsolidationGroups,
  buildConsolidateModelsSystemPrompt,
  buildConsolidateModelsUserContent,
  appliesWhenMap,
  hypothesisConsolidationKey,
  mergeHypothesisSpecs,
  pickCanonicalHypothesis,
  planHypothesisSpecMigration,
  groupHypothesesForConsolidation,
  planHypothesisConsolidation,
  buildPaperResearchGoal,
  normalizeResearchUrl,
  coercePaperCandidates,
  dedupePaperCandidates,
  paperRelevanceClaim,
  isPaperVerdictAdmitted,
  paperHostAffinity,
  rankPaperCandidates,
  computeScorecard,
  scorecardRankValue,
  compareScorecards,
  primaryFitnessCriterion,
  deriveBackfillMetric,
  selectActiveScorecard,
} from './modelTrainerUtils.js'
import { hashTrainingConfig } from './modelTrainerHelpers.js'
import type { ProposedModel, TrainingRunSummary } from './modelTrainerTypes.js'

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

  it('strips keys named by `unset` and is idempotent once they are gone', () => {
    const stripRules = [{ unset: ['position_mode', 'trade_gate_mode'] }]
    const out = applyMigrationRules(
      { lr: 0.1, position_mode: 'x', trade_gate_mode: 'y' },
      stripRules,
    )
    expect(out).toEqual({ lr: 0.1 })
    // a config without any target key no longer matches the rule (no further rewrite)
    expect(applyMigrationRules({ lr: 0.1 }, stripRules)).toBeNull()
    // unset combines with set/match on the same rule
    const combo = applyMigrationRules({ reward_model: 'combo_unified', position_mode: 'x' }, [
      { match: { reward_model: 'combo_unified' }, unset: ['position_mode'] },
    ])
    expect(combo).toEqual({ reward_model: 'combo_unified' })
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

  it('migrates a sweep AXIS over a RETIRED lever that cleanly renames to a single replacement lever', () => {
    // The reported crash: a queued spec sweeps `use_indicators` (removed from the manifest in favour of the
    // `projection` dataset flavour), so expandExperimentMatrix throws "sweep names no manifest lever". The
    // manifest maps each swept value to a single projection value, so the axis rolls forward cleanly.
    const rules = [
      { match: { use_indicators: true }, set: { projection: 'with_indicators' }, unset: ['use_indicators'] },
      { match: { use_indicators: false }, set: { projection: 'standard' }, unset: ['use_indicators'] },
    ]
    const out = migrateExperimentSpec({ sweep: { use_indicators: [true, false] }, seeds: [0] }, rules)
    expect(out.sweep).toEqual({ projection: ['with_indicators', 'standard'] })
    expect(out.seeds).toEqual([0])
  })

  it('leaves a sweep axis untouched when its migration is not a clean single-lever rename (multi-key set)', () => {
    // combo_all → combo_unified ALSO sets combo_sell (two keys), so the axis can't collapse to one lever; the
    // lever name is still valid too, so nothing is rewritten (a same-object pass-through).
    const spec = { sweep: { reward_model: ['combo_all', 'profit_all2'] } }
    expect(migrateExperimentSpec(spec, migrations)).toBe(spec)
  })

  it('migrates only the retired sweep axis, leaving live axes (and their order) intact', () => {
    const rules = [
      { match: { use_indicators: true }, set: { projection: 'with_indicators' }, unset: ['use_indicators'] },
    ]
    const out = migrateExperimentSpec(
      { sweep: { learning_rate: [0.1, 0.01], use_indicators: [true] } },
      rules,
    )
    expect(out.sweep).toEqual({ learning_rate: [0.1, 0.01], projection: ['with_indicators'] })
  })

  it('migrates each spec.configs entry config while PRESERVING its key (re-run rolls old configs forward in place)', () => {
    const out = migrateExperimentSpec(
      {
        configs: [
          { config: { reward_model: 'combo_all', lr: 0.1 }, key: 'run-abc' },
          { config: { reward_model: 'combo_unified' }, key: 'run-def' },
        ],
      },
      migrations,
    )
    expect(out.configs).toEqual([
      { config: { reward_model: 'combo_unified', lr: 0.1, combo_sell: 1000 }, key: 'run-abc' },
      { config: { reward_model: 'combo_unified' }, key: 'run-def' },
    ])
  })

  it('returns the SAME spec object when its configs need no migration', () => {
    const spec = { configs: [{ config: { reward_model: 'combo_unified' }, key: 'run-def' }] }
    expect(migrateExperimentSpec(spec, migrations)).toBe(spec)
  })
})

describe('resolveCampaignParallelism', () => {
  it('SPEEDUP: auto-packs floor(cpus / maxThreadsPerRun) runs when concurrency is unset', () => {
    // 10 cores, 2 threads/run -> 5 parallel runs instead of the sequential default (8 idle cores).
    const r = resolveCampaignParallelism({ maxThreadsPerRun: 2, availableParallelism: 10 })
    expect(r.concurrency).toBe(5)
  })

  it('threads the per-run thread cap into every standard env knob (so a run cannot grab all cores)', () => {
    const r = resolveCampaignParallelism({ maxThreadsPerRun: 2, availableParallelism: 10 })
    for (const v of THREAD_ENV_VARS) expect(r.runEnv?.[v]).toBe('2')
    expect(r.runEnv?.BS_NUM_THREADS).toBe('2')
  })

  it('honours an explicit concurrency and gives each run its fair share of cores', () => {
    const r = resolveCampaignParallelism({
      concurrency: 3,
      maxThreadsPerRun: 2,
      availableParallelism: 10,
    })
    expect(r.concurrency).toBe(3)
    expect(r.runEnv?.OMP_NUM_THREADS).toBe('3') // floor(10/3) — more than the appetite, host fully used
  })

  it('SPEEDUP: a single/final run (concurrency 1) uses the WHOLE host, not the per-run appetite', () => {
    const r = resolveCampaignParallelism({
      concurrency: 1,
      maxThreadsPerRun: 2,
      availableParallelism: 10,
    })
    expect(r.concurrency).toBe(1)
    for (const v of THREAD_ENV_VARS) expect(r.runEnv?.[v]).toBe('10') // all 10 cores, not 2
  })

  it('caps per-run threads BELOW the appetite when over-packed, to avoid oversubscription', () => {
    const r = resolveCampaignParallelism({
      concurrency: 8,
      maxThreadsPerRun: 2,
      availableParallelism: 10,
    })
    expect(r.runEnv?.OMP_NUM_THREADS).toBe('1') // floor(10/8) — 8 runs share 10 cores, 1 thread each
  })

  it('keeps the safe sequential default and sets no env when the manifest declares no thread appetite', () => {
    const r = resolveCampaignParallelism({ availableParallelism: 10 })
    expect(r.concurrency).toBe(1)
    expect(r.runEnv).toBeUndefined()
  })

  it('never returns concurrency < 1 even on a single-core host with a multi-thread appetite', () => {
    const r = resolveCampaignParallelism({ maxThreadsPerRun: 4, availableParallelism: 1 })
    expect(r.concurrency).toBe(1)
  })

  it('floors a non-even cpu/threads ratio (10 cpus, 3 threads -> 3 runs)', () => {
    expect(
      resolveCampaignParallelism({ maxThreadsPerRun: 3, availableParallelism: 10 }).concurrency,
    ).toBe(3)
  })

  it('MEMORY: caps concurrency to floor(availableMemory / maxMemoryBytesPerRun) below the CPU pool', () => {
    // 10 cores, 2 threads/run → CPU wants 5 runs, but only 3 fit in RAM (7 GiB free / 2 GiB per run).
    const r = resolveCampaignParallelism({
      maxThreadsPerRun: 2,
      availableParallelism: 10,
      maxMemoryBytesPerRun: 2 * 1024 ** 3,
      availableMemoryBytes: 7 * 1024 ** 3,
    })
    expect(r.concurrency).toBe(3)
    expect(r.memoryCapped).toBe(true)
  })

  it('MEMORY: the RAM ceiling caps an EXPLICIT concurrency too (a hard ceiling, not a suggestion)', () => {
    const r = resolveCampaignParallelism({
      concurrency: 20,
      maxThreadsPerRun: 2,
      availableParallelism: 10,
      maxMemoryBytesPerRun: 2 * 1024 ** 3,
      availableMemoryBytes: 5 * 1024 ** 3, // fits 2
    })
    expect(r.concurrency).toBe(2)
    expect(r.memoryCapped).toBe(true)
  })

  it('MEMORY: applies no RAM cap when the manifest declares no per-run estimate (byte-compatible)', () => {
    const r = resolveCampaignParallelism({
      maxThreadsPerRun: 2,
      availableParallelism: 10,
      availableMemoryBytes: 1 * 1024 ** 3, // tiny, but no per-run estimate → ignored
    })
    expect(r.concurrency).toBe(5)
    expect(r.memoryCapped).toBeFalsy()
  })

  it('MEMORY: never caps below 1 even when a single run barely fits in RAM', () => {
    const r = resolveCampaignParallelism({
      concurrency: 4,
      maxThreadsPerRun: 2,
      availableParallelism: 10,
      maxMemoryBytesPerRun: 8 * 1024 ** 3,
      availableMemoryBytes: 2 * 1024 ** 3, // less than one run
    })
    expect(r.concurrency).toBe(1)
    expect(r.memoryCapped).toBe(true)
  })

  it('MEMORY: does NOT expand per-run threads when the RAM cap lowered concurrency', () => {
    // 10 cores, 2 threads/run, RAM caps 5→3 runs. Expanding to floor(10/3)=3 threads would raise each
    // run's RSS (more BLAS scratch / malloc arenas) above the estimate the cap was sized against — fighting
    // the very cap that fired. Hold per-run threads at the declared appetite (2).
    const r = resolveCampaignParallelism({
      maxThreadsPerRun: 2,
      availableParallelism: 10,
      maxMemoryBytesPerRun: 2 * 1024 ** 3,
      availableMemoryBytes: 7 * 1024 ** 3,
    })
    expect(r.concurrency).toBe(3)
    for (const v of THREAD_ENV_VARS) expect(r.runEnv?.[v]).toBe('2')
  })
})

describe('resolveModelDeviceForConfig', () => {
  const models = [
    {
      slug: 'reppo',
      flavors: [{ modelName: 'reppo-custom' }, { modelName: 'reppo' }],
      preferredDevice: 'mps' as const,
    },
    { slug: 'ppo', flavors: [{ modelName: 'ppo' }], preferredDevice: 'cpu' as const },
    { slug: 'dqn', flavors: [{ modelName: 'dqn' }] }, // never benchmarked -> no preferredDevice
  ]

  it('applies a model mps preference to a SINGLE run', () => {
    expect(
      resolveModelDeviceForConfig({
        config: { model_name: 'reppo-custom' },
        models,
        concurrency: 1,
      }),
    ).toBe('mps')
  })

  it('does NOT apply mps to a parallel sweep (one GPU cannot be shared)', () => {
    expect(
      resolveModelDeviceForConfig({
        config: { model_name: 'reppo-custom' },
        models,
        concurrency: 4,
      }),
    ).toBeUndefined()
  })

  it('applies a cpu preference even in a parallel sweep (cpu parallelises)', () => {
    expect(
      resolveModelDeviceForConfig({ config: { model_name: 'ppo' }, models, concurrency: 4 }),
    ).toBe('cpu')
  })

  it('respects an explicit device in the config over the model preference', () => {
    expect(
      resolveModelDeviceForConfig({
        config: { model_name: 'reppo-custom', device: 'cpu' },
        models,
        concurrency: 1,
      }),
    ).toBeUndefined()
  })

  it('returns undefined when no model matches or the model was never benchmarked', () => {
    expect(
      resolveModelDeviceForConfig({ config: { model_name: 'dqn' }, models, concurrency: 1 }),
    ).toBeUndefined()
    expect(
      resolveModelDeviceForConfig({ config: { model_name: 'unknown' }, models, concurrency: 1 }),
    ).toBeUndefined()
    expect(resolveModelDeviceForConfig({ config: {}, models, concurrency: 1 })).toBeUndefined()
  })
})

describe('parseDataCatalog', () => {
  it('returns the asset classes from a well-formed summary', () => {
    const classes = parseDataCatalog({
      assetClasses: [{ id: 'crypto', label: 'Crypto', directory: 'binance', instruments: [] }],
    })
    expect(classes).toHaveLength(1)
    expect(classes[0].id).toBe('crypto')
  })

  it('throws a clear error when assetClasses is missing or malformed', () => {
    expect(() => parseDataCatalog({})).toThrow(/assetClasses/)
    expect(() => parseDataCatalog(undefined)).toThrow(/assetClasses/)
    expect(() => parseDataCatalog({ assetClasses: 'nope' })).toThrow(/assetClasses/)
  })
})

describe('catalogAssetIds / withCatalogChoices (A6 — derive asset choices from disk)', () => {
  const inst = (symbol: string, onDisk: Record<string, unknown> = {}) => ({
    symbol, label: symbol, assetClass: 'crypto', source: 'binance', sourceSymbol: symbol,
    intervals: ['1h'], directory: 'binance', tier: 1, barCloseTz: 'UTC', onDisk,
  })
  const cls = (id: string, instruments: unknown[]) => ({ id, label: id, directory: id, instruments })
  const catalog = [cls('crypto', [inst('BTCUSDT', { '1h': {} }), inst('DOGEUSDT', { '1h': {} })])] as never

  it('catalogAssetIds returns the on-disk symbols, de-duplicated', () => {
    const classes = [
      cls('crypto', [inst('BTCUSDT', { '1h': {} }), inst('ETHUSDT', {})]), // ETH not on disk ⇒ excluded
      cls('stocks', [inst('NVDA', { '1d': {} }), inst('BTCUSDT', { '1h': {} })]), // dup BTC
    ] as never
    expect(catalogAssetIds(classes).sort()).toEqual(['BTCUSDT', 'NVDA'])
    expect(catalogAssetIds(undefined as never)).toEqual([])
    expect(catalogAssetIds([cls('crypto', [inst('BTCUSDT', {})])] as never)).toEqual([]) // nothing on disk
    // a null class / a class with no instruments array contributes nothing (never throws)
    expect(catalogAssetIds([{ id: 'x' }, null] as never)).toEqual([])
  })

  const manifest = {
    name: 'x', objective: { name: 'r', direction: 'max' },
    levers: {
      asset: { type: 'choice', choices: ['BTCUSDT', 'ETHUSDT'], choicesFrom: 'datacatalog', scope: 'dataset' },
      lr: { type: 'number', default: 0.01 },
    },
  } as unknown as TrainerManifest

  it('unions the on-disk catalog symbols into a choicesFrom:datacatalog lever, keeping the declared ones', () => {
    const out = withCatalogChoices(manifest, catalog)
    expect(out.levers.asset.choices).toEqual(['BTCUSDT', 'ETHUSDT', 'DOGEUSDT']) // declared kept, DOGE added, BTC deduped
    expect(out.levers.lr).toBe(manifest.levers.lr) // a lever without the flag is untouched (same ref)
  })

  it('leaves a lever WITHOUT the flag alone even if the catalog has assets', () => {
    const m = { ...manifest, levers: { asset: { type: 'choice', choices: ['BTCUSDT'] } } } as unknown as TrainerManifest
    expect(withCatalogChoices(m, catalog).levers.asset.choices).toEqual(['BTCUSDT'])
  })

  it('seeds a datacatalog lever that declares NO choices straight from the on-disk symbols', () => {
    const m = {
      name: 'x', objective: { name: 'r', direction: 'max' },
      levers: { asset: { type: 'choice', choicesFrom: 'datacatalog' } },
    } as unknown as TrainerManifest
    expect(withCatalogChoices(m, catalog).levers.asset.choices).toEqual(['BTCUSDT', 'DOGEUSDT'])
  })

  it('returns the SAME manifest when the catalog has no on-disk assets (no needless copy)', () => {
    expect(withCatalogChoices(manifest, [] as never)).toBe(manifest)
  })
})

describe('buildDataDiscoveryGoal', () => {
  const m = {
    name: 'BlackSwan',
    description: 'a crypto trader',
    objective: { name: 'return', direction: 'max' },
    levers: {},
  } as unknown as TrainerManifest

  it('composes the manifest, problem, and goal into a research query about datasets', () => {
    const q = buildDataDiscoveryGoal(m, 'price is too noisy', 'add macro context')
    expect(q).toContain('BlackSwan')
    expect(q).toContain('price is too noisy')
    expect(q).toContain('add macro context')
    expect(q.toLowerCase()).toContain('dataset')
  })

  it('works with no goal', () => {
    expect(buildDataDiscoveryGoal(m, 'just the problem')).toContain('just the problem')
  })
})

describe('coerceDataSourceCandidates', () => {
  it('coerces fields and dedupes by name (case-insensitive)', () => {
    const out = coerceDataSourceCandidates([
      { name: 'FRED macro', source: 'FRED', coverage: '1950-', cost: 'free', licence: 'attribution', description: 'macro' },
      { name: 'fred macro', source: 'FRED' },
      { nope: 1 },
    ])
    expect(out).toHaveLength(1)
    expect(out[0]).toMatchObject({
      name: 'FRED macro',
      source: 'FRED',
      coverage: '1950-',
      cost: 'free',
      licence: 'attribution',
    })
  })

  it('accepts alternate field names and a non-array input', () => {
    expect(coerceDataSourceCandidates('nope' as unknown as unknown[])).toEqual([])
    const out = coerceDataSourceCandidates([{ dataset: 'X', provider: 'P', license: 'MIT' }])
    expect(out[0]).toMatchObject({ name: 'X', source: 'P', licence: 'MIT' })
  })

  it('drops items with no name', () => {
    expect(coerceDataSourceCandidates([{ source: 'S' }, { description: 'd' }])).toEqual([])
  })
})

describe('missingCrossTestValues', () => {
  const universe = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']

  it('all = the lever universe minus the trained value and already-tested', () => {
    const { missing, unknown } = missingCrossTestValues('BTCUSDT', 'all', ['ETHUSDT'], universe)
    expect(missing).toEqual(['SOLUSDT', 'XRPUSDT'])
    expect(unknown).toEqual([])
  })

  it('undefined requested behaves like "all"', () => {
    expect(missingCrossTestValues('BTCUSDT', undefined, [], universe).missing).toEqual([
      'ETHUSDT',
      'SOLUSDT',
      'XRPUSDT',
    ])
  })

  it('an explicit list is filtered to the universe, minus trained + tested', () => {
    const { missing, unknown } = missingCrossTestValues(
      'BTCUSDT',
      ['ETHUSDT', 'SOLUSDT', 'NOPE'],
      ['SOLUSDT'],
      universe,
    )
    expect(missing).toEqual(['ETHUSDT'])
    expect(unknown).toEqual(['NOPE'])
  })

  it('excludes the trained value even if explicitly requested', () => {
    expect(missingCrossTestValues('BTCUSDT', ['BTCUSDT', 'ETHUSDT'], [], universe).missing).toEqual([
      'ETHUSDT',
    ])
  })

  it('dedupes and returns empty when everything is already tested', () => {
    expect(
      missingCrossTestValues('BTCUSDT', 'all', ['ETHUSDT', 'SOLUSDT', 'XRPUSDT'], universe).missing,
    ).toEqual([])
  })
})

describe('slugifyDataSource', () => {
  it('lowercases + dash-separates', () => {
    expect(slugifyDataSource('FRED Macro (US)')).toBe('fred-macro-us')
  })
  it('falls back to "source" for an empty name', () => {
    expect(slugifyDataSource('')).toBe('source')
    expect(slugifyDataSource('  ')).toBe('source')
  })
})

describe('parseDataLinkage', () => {
  it('returns the linkage edges from a catalog summary', () => {
    const edges = parseDataLinkage({
      linkage: { edges: [{ asset: 'JPM', proxy: 'T10Y2Y', edgeType: 'curve-slope' }] },
    })
    expect(edges).toHaveLength(1)
    expect(edges[0].asset).toBe('JPM')
  })

  it('defaults to an empty array when linkage is missing or malformed', () => {
    expect(parseDataLinkage({})).toEqual([])
    expect(parseDataLinkage(undefined)).toEqual([])
    expect(parseDataLinkage({ linkage: {} })).toEqual([])
    expect(parseDataLinkage({ linkage: { edges: 'nope' } })).toEqual([])
  })
})

describe('parseMineResult', () => {
  it('parses the mined outcome, unknowns, and through', () => {
    const r = parseMineResult({
      mined: [{ symbol: 'GOLD', source: 'yfinance', written: 2, skipped: 0, errors: [], gaps: [] }],
      unknown: ['NOPE'],
      through: '2026-06',
    })
    expect(r.mined[0].symbol).toBe('GOLD')
    expect(r.unknown).toEqual(['NOPE'])
    expect(r.through).toBe('2026-06')
  })

  it('defaults unknown/through when absent', () => {
    const r = parseMineResult({ mined: [] })
    expect(r.unknown).toEqual([])
    expect(r.through).toBe('')
  })

  it('throws when the mined array is missing', () => {
    expect(() => parseMineResult({})).toThrow(/mined/)
    expect(() => parseMineResult(undefined)).toThrow(/mined/)
  })
})

describe('parseDeviceBenchmark', () => {
  it('coerces a well-formed deviceBenchmark summary into a typed record', () => {
    const b = parseDeviceBenchmark(
      {
        deviceBenchmark: {
          bestDevice: 'mps',
          speedup: 1.45,
          usPerStep: { cpu: 100, mps: 69 },
          availableDevices: ['cpu', 'mps'],
        },
      },
      'T0',
    )
    expect(b).toEqual({
      bestDevice: 'mps',
      speedup: 1.45,
      usPerStep: { cpu: 100, mps: 69 },
      availableDevices: ['cpu', 'mps'],
      benchmarkedAt: 'T0',
    })
  })

  it('defaults safely to cpu on a missing/malformed summary (never sets a bogus device)', () => {
    expect(parseDeviceBenchmark(undefined, 'T').bestDevice).toBe('cpu')
    expect(parseDeviceBenchmark({}, 'T').bestDevice).toBe('cpu')
    expect(parseDeviceBenchmark({ deviceBenchmark: { bestDevice: 'gpu' } }, 'T').bestDevice).toBe(
      'cpu',
    )
  })

  it('drops non-positive usPerStep, clamps speedup to >= 1, derives availableDevices from usPerStep', () => {
    const b = parseDeviceBenchmark(
      { deviceBenchmark: { bestDevice: 'cpu', speedup: 0.4, usPerStep: { cpu: 100, mps: 0 } } },
      'T',
    )
    expect(b.speedup).toBe(1)
    expect(b.usPerStep).toEqual({ cpu: 100 })
    expect(b.availableDevices).toEqual(['cpu'])
  })

  it('keeps seconds/budget/errors when present and accepts cuda as best', () => {
    const b = parseDeviceBenchmark(
      {
        deviceBenchmark: {
          bestDevice: 'cuda',
          speedup: 2.1,
          usPerStep: { cpu: 100, cuda: 47 },
          seconds: { cpu: 15.5, cuda: 7.4, mps: 0 },
          budget: 300,
          errors: { mps: 'too slow / timed out' },
          availableDevices: ['cpu', 'cuda'],
        },
      },
      'T1',
    )
    expect(b.bestDevice).toBe('cuda')
    expect(b.budget).toBe(300)
    expect(b.seconds).toEqual({ cpu: 15.5, cuda: 7.4 }) // non-positive mps:0 dropped
    expect(b.errors).toEqual({ mps: 'too slow / timed out' })
  })

  it('omits seconds/budget/errors entirely when the summary lacks them', () => {
    const b = parseDeviceBenchmark(
      { deviceBenchmark: { bestDevice: 'cpu', usPerStep: { cpu: 50 } } },
      'T',
    )
    expect(b.seconds).toBeUndefined()
    expect(b.budget).toBeUndefined()
    expect(b.errors).toBeUndefined()
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

  it('rejects a benchmarkDevice template without {summaryOut}', () => {
    expect(() =>
      validateTrainerManifest({ ...manifest(), benchmarkDevice: 'python -m trainer.bench_device' }),
    ).toThrow(/summaryOut/)
  })

  it('accepts a valid benchmarkDevice template and preserves it', () => {
    const m = validateTrainerManifest({
      ...manifest(),
      benchmarkDevice: 'python -m trainer.bench_device --summary-out {summaryOut}',
    })
    expect(m.benchmarkDevice).toContain('bench_device')
  })

  it('rejects a play template without {configPath}', () => {
    expect(() =>
      validateTrainerManifest({ ...manifest(), play: 'python -m harness.play --serve --summary-out {summaryOut}' }),
    ).toThrow(/configPath/)
  })

  it('rejects a play template without {summaryOut}', () => {
    expect(() =>
      validateTrainerManifest({ ...manifest(), play: 'python -m harness.play --serve --config-json {configPath}' }),
    ).toThrow(/summaryOut/)
  })

  it('accepts a valid play template and preserves it', () => {
    const m = validateTrainerManifest({
      ...manifest(),
      play: 'python -m harness.play --serve --config-json {configPath} --summary-out {summaryOut}',
    })
    expect(m.play).toContain('--serve')
  })

  it('rejects a dataCatalog template without {summaryOut}', () => {
    expect(() =>
      validateTrainerManifest({ ...manifest(), dataCatalog: 'python -m trainer.data' }),
    ).toThrow(/summaryOut/)
  })

  it('accepts a valid dataCatalog template and preserves it', () => {
    const m = validateTrainerManifest({
      ...manifest(),
      dataCatalog: 'python -m trainer.data_catalog --summary-out {summaryOut}',
    })
    expect(m.dataCatalog).toContain('data_catalog')
  })

  it('rejects a mineData template without {configPath}', () => {
    expect(() =>
      validateTrainerManifest({ ...manifest(), mineData: 'python -m trainer.mine --summary-out {summaryOut}' }),
    ).toThrow(/configPath/)
  })

  it('rejects a mineData template without {summaryOut}', () => {
    expect(() =>
      validateTrainerManifest({ ...manifest(), mineData: 'python -m trainer.mine --config-json {configPath}' }),
    ).toThrow(/summaryOut/)
  })

  it('accepts a valid mineData template and preserves it', () => {
    const m = validateTrainerManifest({
      ...manifest(),
      mineData: 'python -m trainer.mine --config-json {configPath} --summary-out {summaryOut}',
    })
    expect(m.mineData).toContain('trainer.mine')
  })

  it('rejects a data declaration that is not an array', () => {
    expect(() => validateTrainerManifest({ ...manifest(), data: 'nope' })).toThrow(/array/)
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

  it('runs every value of a `compare` lever, holding the rest fixed', () => {
    const items = expandExperimentMatrix(
      manifest(),
      { fixed: { lr: 0.5 }, compare: { lever: 'algo', values: ['a', 'b'] } },
      hashByJson,
    )
    expect(items).toHaveLength(2)
    expect(items.map((i) => i.config.algo).sort()).toEqual(['a', 'b'])
    expect(items.every((i) => i.config.lr === 0.5)).toBe(true)
  })

  it('rejects a compare lever that names no manifest lever', () => {
    expect(() =>
      expandExperimentMatrix(
        manifest(),
        { compare: { lever: 'nope', values: ['a', 'b'] } },
        hashByJson,
      ),
    ).toThrow(/compare lever/)
  })

  it('rejects a valid compare lever that lists no values to compare', () => {
    expect(() =>
      expandExperimentMatrix(manifest(), { compare: { lever: 'algo', values: [] } }, hashByJson),
    ).toThrow(/must list values/)
  })

  it('MEMORY-SAFETY: rejects a cartesian blow-up BEFORE materializing it (fails fast, never OOMs)', () => {
    const levers: Record<string, { type: 'number'; default: number }> = {}
    const sweep: Record<string, number[]> = {}
    for (let i = 0; i < 8; i += 1) {
      levers[`l${i}`] = { type: 'number', default: 0 }
      sweep[`l${i}`] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
    // 10^8 planned items: building the product would exhaust the heap (this is the boot-crash-loop cause).
    // The guard computes the size and throws instantly — if it regressed, this test would OOM, not fail.
    expect(() => expandExperimentMatrix(manifest({ levers }), { sweep }, hashByJson)).toThrow(
      /100000000 items, exceeding the cap/,
    )
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

  it('plans exactly the explicit configs (each merged onto defaults), with no default base item', () => {
    const items = expandExperimentMatrix(
      manifest(),
      { configs: [{ config: { lr: 0.7, algo: 'b' } }, { config: { lr: 0.9 } }] },
      hashByJson,
    )
    expect(items).toHaveLength(2)
    expect(items[0].config).toEqual({ lr: 0.7, algo: 'b', steps: 100 })
    expect(items[1].config).toEqual({ lr: 0.9, algo: 'a', steps: 100 })
  })

  it('uses an explicit config key VERBATIM (so a re-run updates the same record), else hashes', () => {
    const items = expandExperimentMatrix(
      manifest(),
      { configs: [{ config: { lr: 0.5 }, key: 'pinned-key-123' }, { config: { lr: 0.6 } }] },
      hashByJson,
    )
    expect(items).toHaveLength(2)
    expect(items[0].key).toBe('pinned-key-123')
    expect(items[0].config).toEqual({ lr: 0.5, algo: 'a', steps: 100 })
    // no key supplied -> falls back to the injected hash of the merged config
    expect(items[1].key).toBe(hashByJson({ lr: 0.6, algo: 'a', steps: 100 }))
  })

  it('ignores sweep/seeds when explicit configs are given (configs define the matrix verbatim)', () => {
    const items = expandExperimentMatrix(
      manifest(),
      { sweep: { lr: [0.1, 0.2] }, seeds: [0, 1], configs: [{ config: { lr: 0.5, algo: 'b' } }] },
      hashByJson,
    )
    expect(items).toHaveLength(1)
    expect(items[0].config).toEqual({ lr: 0.5, algo: 'b', steps: 100 })
  })

  it('rejects an explicit config value that names no lever', () => {
    expect(() =>
      expandExperimentMatrix(manifest(), { configs: [{ config: { ghost: 1 } }] }, hashByJson),
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

  it('treats a non-finite score as the low bound (0), never NaN', () => {
    expect(blendJudgeScore(NaN, 0, 0.5)).toBe(0)
    expect(blendJudgeScore(100, NaN, 1)).toBe(0)
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

  it('defaults a missing / non-numeric score to 0 (keeping the keyed row)', () => {
    expect(coerceVerdictRows([{ key: 'a' }, { key: 'b', score: 'high' }])).toEqual([
      { key: 'a', score: 0, why: '' },
      { key: 'b', score: 0, why: '' },
    ])
  })
})

describe('validateSpecAgainstManifest', () => {
  const vm = manifest({
    levers: {
      lr: { type: 'number', default: 0.01, range: [0.0001, 0.1] },
      steps: { type: 'number', default: 100 },
      algo: { type: 'choice', choices: ['a', 'b'], default: 'a' },
      net_arch: {
        type: 'choice',
        choices: [
          [64, 64],
          [256, 256],
        ],
        default: [64, 64],
      },
      use_dropout: { type: 'boolean', default: false },
      dropout_p: {
        type: 'number',
        default: 0.1,
        range: [0, 1],
        appliesWhen: { use_dropout: [true] },
      },
    },
  })

  it('accepts a spec whose values are all in range / valid choices', () => {
    expect(
      validateSpecAgainstManifest(vm, { fixed: { algo: 'a', lr: 0.05 }, sweep: { steps: [50, 100] } }),
    ).toEqual([])
  })
  it('flags a number outside its declared range (fixed and sweep)', () => {
    expect(validateSpecAgainstManifest(vm, { fixed: { lr: 5 } })).toHaveLength(1)
    expect(validateSpecAgainstManifest(vm, { sweep: { lr: [0.05, 9] } })).toHaveLength(1)
  })
  it('flags a choice value not in the manifest choices', () => {
    expect(validateSpecAgainstManifest(vm, { fixed: { algo: 'c' } })).toHaveLength(1)
    expect(validateSpecAgainstManifest(vm, { fixed: { algo: 'a' } })).toEqual([])
  })
  it('compares array-valued choices by deep value (net_arch)', () => {
    expect(
      validateSpecAgainstManifest(vm, { fixed: { net_arch: [256, 256] } }),
    ).toEqual([])
    expect(validateSpecAgainstManifest(vm, { fixed: { net_arch: [1, 2] } })).toHaveLength(1)
  })
  it('flags a non-boolean value for a boolean lever', () => {
    expect(validateSpecAgainstManifest(vm, { fixed: { use_dropout: 1 } })).toHaveLength(1)
    expect(validateSpecAgainstManifest(vm, { fixed: { use_dropout: true } })).toEqual([])
  })
  it('validates compare values against the compared lever', () => {
    expect(
      validateSpecAgainstManifest(vm, { compare: { lever: 'algo', values: ['a', 'b'] } }),
    ).toEqual([])
    expect(
      validateSpecAgainstManifest(vm, { compare: { lever: 'algo', values: ['a', 'zzz'] } }),
    ).toHaveLength(1)
  })
  it('flags a lever pinned where its appliesWhen condition is not met (control pinned incompatibly)', () => {
    expect(
      validateSpecAgainstManifest(vm, { fixed: { dropout_p: 0.2, use_dropout: false } }),
    ).toHaveLength(1)
    expect(
      validateSpecAgainstManifest(vm, { fixed: { dropout_p: 0.2, use_dropout: true } }),
    ).toEqual([])
  })
  it('uses the control default when the control is unpinned (dropout_p needs use_dropout=true)', () => {
    // use_dropout defaults to false ⇒ dropout_p never applies ⇒ flagged.
    expect(validateSpecAgainstManifest(vm, { fixed: { dropout_p: 0.2 } })).toHaveLength(1)
    // but if the control is SWEPT to include a satisfying value, it can apply ⇒ not flagged.
    expect(
      validateSpecAgainstManifest(vm, {
        fixed: { dropout_p: 0.2 },
        sweep: { use_dropout: [true, false] },
      }),
    ).toEqual([])
  })
  it('flags a lever name that is not in the manifest', () => {
    expect(validateSpecAgainstManifest(vm, { fixed: { ghost: 1 } })).toHaveLength(1)
  })
  it('validates environment/dataset bundle values too', () => {
    const bm = manifest({
      levers: {
        algo: { type: 'choice', choices: ['a', 'b'], default: 'a' },
        asset: { type: 'choice', choices: ['btc', 'eth'], scope: 'dataset', default: 'btc' },
      },
    })
    expect(validateSpecAgainstManifest(bm, { datasets: [{ asset: 'btc' }, { asset: 'doge' }] })).toHaveLength(1)
    expect(validateSpecAgainstManifest(bm, { datasets: [{ asset: 'btc' }, { asset: 'eth' }] })).toEqual([])
  })
  it('appliesWhen control satisfied via a DATASET/ENVIRONMENT bundle is not flagged (bundles set levers too)', () => {
    const bm = manifest({
      levers: {
        model_name: { type: 'choice', choices: ['m'], default: 'm' },
        fidelity: { type: 'choice', choices: ['1m', '5m'], scope: 'dataset', default: '5m' },
        lookback: { type: 'number', default: 10, appliesWhen: { fidelity: ['1m'] } },
      },
    })
    // The only dataset bundle sets fidelity=1m, so lookback DOES apply — must not be dropped.
    expect(
      validateSpecAgainstManifest(bm, {
        fixed: { model_name: 'm', lookback: 20 },
        datasets: [{ fidelity: '1m' }],
      }),
    ).toEqual([])
    // No bundle → control falls back to the default 5m → genuinely inapplicable → flagged.
    expect(
      validateSpecAgainstManifest(bm, { fixed: { model_name: 'm', lookback: 20 } }),
    ).toHaveLength(1)
    // MIXED bundles (one satisfies, one missing the control → default): some context satisfies → not flagged.
    expect(
      validateSpecAgainstManifest(bm, {
        fixed: { model_name: 'm', lookback: 20 },
        datasets: [{ fidelity: '1m' }, {}],
      }),
    ).toEqual([])
  })
  it('coercion parity with specMatchesConfig: a string echo of a numeric choice / boolean is accepted', () => {
    const bm = manifest({
      levers: {
        fidelity: { type: 'choice', choices: [1, 5, 15], default: 5 },
        use_layernorm: { type: 'boolean', default: false },
      },
    })
    // specMatchesConfig compares String(a)===String(b), so '5' matches runs storing 5 — must not be dropped.
    expect(validateSpecAgainstManifest(bm, { fixed: { fidelity: '5' } })).toEqual([])
    expect(validateSpecAgainstManifest(bm, { fixed: { fidelity: '7' } })).toHaveLength(1)
    expect(validateSpecAgainstManifest(bm, { fixed: { use_layernorm: 'true' } })).toEqual([])
    expect(validateSpecAgainstManifest(bm, { fixed: { use_layernorm: 'yes' } })).toHaveLength(1)
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

  it('drops a proposal whose VALUE is off-manifest (choice not in choices) — generation-time validation', () => {
    // algo is a choice over ['a','b']; 'zzz' is not allowed, so the item never persists.
    expect(
      coerceHypothesisItems([{ title: 't', rationale: 'r', spec: { fixed: { algo: 'zzz' } } }], m),
    ).toEqual([])
    expect(
      coerceHypothesisItems([{ title: 't', rationale: 'r', spec: { fixed: { algo: 'b' } } }], m),
    ).toHaveLength(1)
  })

  it('folds a retired value via migrations before persisting (no stranded dead pin)', () => {
    const mm = manifest({
      levers: {
        algo: { type: 'choice', choices: ['a', 'b'], default: 'a' },
        lr: { type: 'number', default: 0.01 },
      },
      migrations: [{ match: { algo: 'old' }, set: { algo: 'b' } }],
    })
    const items = coerceHypothesisItems(
      [{ title: 't', rationale: 'r', spec: { fixed: { algo: 'old' } } }],
      mm,
    )
    expect(items).toHaveLength(1)
    expect(items[0].spec.fixed).toEqual({ algo: 'b' })
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

  it('carries the paper CLAIM label (the specific claim a hypothesis tests)', () => {
    const items = coerceHypothesisItems(
      [
        {
          title: 't',
          rationale: 'r',
          claim: '  Momentum predicts returns  ',
          spec: { fixed: { lr: 0.9 } },
        },
      ],
      m,
    )
    expect(items[0].claim).toBe('Momentum predicts returns')
  })

  it('omits the claim when blank or absent', () => {
    const items = coerceHypothesisItems(
      [
        { title: 'a', rationale: 'r', claim: '   ', spec: { fixed: { lr: 0.9 } } },
        { title: 'b', rationale: 'r', spec: { fixed: { lr: 0.9 } } },
      ],
      m,
    )
    expect(items[0].claim).toBeUndefined()
    expect(items[1].claim).toBeUndefined()
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

describe('looksComparative', () => {
  it('flags comparative phrasing', () => {
    expect(looksComparative('A outperforms B', '')).toBe(true)
    expect(looksComparative('Necessity of recurrent architectures', '')).toBe(true)
    expect(looksComparative('reppo vs ppo', '')).toBe(true)
  })
  it('does NOT flag a plain beats-buy-and-hold claim', () => {
    expect(looksComparative('reppo-custom beats buy-and-hold', 'tests one config')).toBe(false)
  })
})

describe('coerceHypothesisItems — precision guards + compare', () => {
  const mfModel = (): TrainerManifest =>
    manifest({
      levers: {
        model_name: {
          type: 'choice',
          choices: ['ppo-custom', 'reppo-custom', 'dqn'],
          default: 'ppo-custom',
        },
        timeframe: { type: 'choice', choices: ['1h', '1d'], default: '1h' },
        lr: { type: 'number', default: 0.01 },
        allow_shorting: { type: 'boolean', default: false, scope: 'environment' },
      },
    })
  it('GUARD A: drops a comparative claim expressed as a pooled model_name sweep', () => {
    expect(
      coerceHypothesisItems(
        [
          {
            title: 'reppo outperforms ppo',
            rationale: 'recurrence is better',
            spec: {
              fixed: { timeframe: '1h' },
              sweep: { model_name: ['ppo-custom', 'reppo-custom'] },
            },
          },
        ],
        mfModel(),
      ),
    ).toEqual([])
  })
  it('GUARD B: drops a single-context spec that does not pin model_name (too broad)', () => {
    expect(
      coerceHypothesisItems(
        [
          {
            title: 'beats hold',
            rationale: 'only timeframe pinned',
            spec: { fixed: { timeframe: '1h' } },
          },
        ],
        mfModel(),
      ),
    ).toEqual([])
  })
  it('clamps an out-of-range comparison.baselineIndex to 0 (an LLM index beyond the cells would mis-judge)', () => {
    const out = coerceHypothesisItems(
      [
        {
          title: 'reppo vs ppo',
          rationale: 'compare with a bad baseline index',
          spec: {
            fixed: { timeframe: '1h' },
            compare: { lever: 'model_name', values: ['ppo-custom', 'reppo-custom'] },
          },
          comparison: { kind: 'beats-baseline', baselineIndex: 7 },
        },
      ],
      mfModel(),
    )
    expect(out).toHaveLength(1)
    expect(out[0].comparison).toEqual({ kind: 'beats-baseline', baselineIndex: 0 })
  })
  it('keeps a valid in-range baselineIndex', () => {
    const out = coerceHypothesisItems(
      [
        {
          title: 'reppo vs ppo',
          rationale: 'compare with baseline 1',
          spec: {
            fixed: { timeframe: '1h' },
            compare: { lever: 'model_name', values: ['ppo-custom', 'reppo-custom'] },
          },
          comparison: { kind: 'beats-baseline', baselineIndex: 1 },
        },
      ],
      mfModel(),
    )
    expect(out[0].comparison).toEqual({ kind: 'beats-baseline', baselineIndex: 1 })
  })
  it('keeps a single-context spec that pins the model-identity lever', () => {
    const out = coerceHypothesisItems(
      [
        {
          title: 'reppo beats hold',
          rationale: 'one fully-pinned config',
          spec: { fixed: { model_name: 'reppo-custom', timeframe: '1h' } },
        },
      ],
      mfModel(),
    )
    expect(out).toHaveLength(1)
  })
  it('keeps + passes a `compare` spec through (exempt from the single-context guard)', () => {
    const out = coerceHypothesisItems(
      [
        {
          title: 'reppo vs ppo',
          rationale: 'compare the arms',
          comparison: { kind: 'beats-baseline' },
          spec: {
            fixed: { timeframe: '1h' },
            compare: { lever: 'model_name', values: ['ppo-custom', 'reppo-custom'] },
          },
        },
      ],
      mfModel(),
    )
    expect(out).toHaveLength(1)
    expect(out[0].spec.compare).toEqual({
      lever: 'model_name',
      values: ['ppo-custom', 'reppo-custom'],
    })
  })
  it('drops a compare naming an unknown lever or with fewer than two values', () => {
    const mf = mfModel()
    expect(
      coerceHypothesisItems(
        [{ title: 't', rationale: 'r', spec: { compare: { lever: 'nope', values: ['a', 'b'] } } }],
        mf,
      ),
    ).toEqual([])
    expect(
      coerceHypothesisItems(
        [
          {
            title: 't',
            rationale: 'r',
            spec: {
              fixed: { model_name: 'ppo-custom' },
              compare: { lever: 'model_name', values: ['ppo-custom'] },
            },
          },
        ],
        mf,
      ),
    ).toEqual([])
  })
})

describe('coerceHypothesisItems — context-spanning specs', () => {
  const cm = manifest({
    levers: {
      algo: { type: 'choice', choices: ['a', 'b'], default: 'a' },
      allow_shorting: { type: 'boolean', default: false, scope: 'environment' },
      asset: { type: 'choice', choices: ['BTC', 'ETH'], default: 'BTC', scope: 'dataset' },
    },
  })
  it('accepts environment bundles of environment-scoped levers (no sweep/fixed needed)', () => {
    const items = coerceHypothesisItems(
      [
        {
          title: 'long vs long+short',
          rationale: 'shorting should help in down markets',
          spec: { environments: [{ allow_shorting: false }, { allow_shorting: true }] },
        },
      ],
      cm,
    )
    expect(items).toHaveLength(1)
    expect(items[0].spec.environments).toEqual([
      { allow_shorting: false },
      { allow_shorting: true },
    ])
  })
  it('accepts dataset bundles of dataset-scoped levers', () => {
    const items = coerceHypothesisItems(
      [{ title: 't', rationale: 'r', spec: { datasets: [{ asset: 'BTC' }, { asset: 'ETH' }] } }],
      cm,
    )
    expect(items[0].spec.datasets).toEqual([{ asset: 'BTC' }, { asset: 'ETH' }])
  })
  it('drops a spec whose environment bundle names a non-environment lever', () => {
    expect(
      coerceHypothesisItems(
        [{ title: 't', rationale: 'r', spec: { environments: [{ algo: 'a' }] } }],
        cm,
      ),
    ).toEqual([])
  })
  it('drops a spec whose dataset bundle names a non-dataset lever', () => {
    expect(
      coerceHypothesisItems(
        [{ title: 't', rationale: 'r', spec: { datasets: [{ allow_shorting: true }] } }],
        cm,
      ),
    ).toEqual([])
  })
  it('passes through a valid comparison criterion', () => {
    const items = coerceHypothesisItems(
      [
        {
          title: 't',
          rationale: 'r',
          spec: { environments: [{ allow_shorting: false }, { allow_shorting: true }] },
          comparison: { kind: 'invariant', tolerance: 0.2 },
        },
      ],
      cm,
    )
    expect(items[0].comparison).toEqual({ kind: 'invariant', tolerance: 0.2 })
  })
  it('omits an invalid comparison kind', () => {
    const items = coerceHypothesisItems(
      [
        {
          title: 't',
          rationale: 'r',
          spec: { environments: [{ allow_shorting: true }, { allow_shorting: false }] },
          comparison: { kind: 'bogus' },
        },
      ],
      cm,
    )
    expect(items[0].comparison).toBeUndefined()
  })
  it('combines a fixed model lever with environment bundles', () => {
    const items = coerceHypothesisItems(
      [
        {
          title: 't',
          rationale: 'r',
          spec: {
            fixed: { algo: 'a' },
            environments: [{ allow_shorting: false }, { allow_shorting: true }],
          },
        },
      ],
      cm,
    )
    expect(items[0].spec.fixed).toEqual({ algo: 'a' })
    expect(items[0].spec.environments).toHaveLength(2)
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
  it('tolerates undefined title/rationale (empty text is not data-gathering)', () => {
    expect(looksLikeDataGathering(undefined as never, undefined as never)).toBe(false)
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

  it('reliability system prompt names the probabilistic levers and constrains the verdict vocabulary', () => {
    const withProb = {
      ...m,
      levers: { ...m.levers, prob_threshold: { type: 'number' as const, probabilistic: true } },
    }
    const prompt = buildReliabilitySystemPrompt(withProb)
    expect(prompt).toContain(m.name)
    expect(prompt).toContain('prob_threshold')
    expect(prompt).toMatch(/dubious/)
    expect(prompt).toMatch(/threshold-driven/)
    expect(prompt).toMatch(/JSON/)
  })

  it('coerceReliabilityVerdict accepts a valid verdict, trims the rationale, and rejects junk', () => {
    expect(
      coerceReliabilityVerdict({ level: 'dubious', rationale: '  threshold-tuned luck  ' }),
    ).toEqual({ level: 'dubious', rationale: 'threshold-tuned luck' })
    expect(coerceReliabilityVerdict({ level: 'ok' })).toEqual({ level: 'ok', rationale: '' })
    expect(coerceReliabilityVerdict({ level: 'nonsense' })).toBeNull()
    expect(coerceReliabilityVerdict(null)).toBeNull()
    expect(coerceReliabilityVerdict({ rationale: 'no level' })).toBeNull()
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

  it('surfaces per-step driver counts when present, and omits the line when absent', () => {
    const withDrivers = buildXaiNarrateUserContent({
      runKey: 'feed',
      config: {},
      criterion: { key: 'objective', direction: 'max' },
      attribution: { topGroups: [['layer:1d', 0.4]], driverCounts: [['layer:1d', 12], ['layer:1h', 3]] },
      importances: [],
    })
    expect(withDrivers).toMatch(/Per-step drivers.*layer:1d=12, layer:1h=3\./)
    const without = buildXaiNarrateUserContent({
      runKey: 'feed',
      config: {},
      criterion: { key: 'objective', direction: 'max' },
      attribution: { topGroups: [['layer:1d', 0.4]] },
      importances: [],
    })
    expect(without).not.toContain('Per-step drivers')
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

describe('describeRunFailures (Explore stop-message diagnosis)', () => {
  it('surfaces the newest failed run error + last log line', () => {
    const records = [
      { content: { status: 'completed', objective: 0.9 } },
      { content: { status: 'failed', error: 'unknown config keys: [device]', ranAt: '2026-08-10T09:00:00Z' } },
      { content: { status: 'failed', error: 'python3: command not found', ranAt: '2026-08-10T10:00:00Z', logTail: ['/bin/sh: python3: command not found'] } },
    ]
    const out = describeRunFailures(records)
    expect(out).toContain('python3: command not found') // newest first
    expect(out).toContain('unknown config keys')
    expect(out.startsWith(' Last failures:')).toBe(true)
  })

  it('collapses a multi-line error to its first line and appends the log tail once', () => {
    const out = describeRunFailures(
      [{ content: { status: 'failed', error: 'Traceback...\nValueError: bad', ranAt: 't', logTail: ['ValueError: bad'] } }],
      1,
    )
    expect(out).toBe(' Last failure:\n• Traceback... — ValueError: bad')
  })

  it('is empty when there are no failed runs', () => {
    expect(describeRunFailures([{ content: { status: 'completed' } }])).toBe('')
    expect(describeRunFailures([])).toBe('')
  })
})

describe('validateRunProvenance (§C.9 reproducibility soft-flag)', () => {
  it('flags every missing key when there is no provenance at all', () => {
    expect(validateRunProvenance({ objective: 0.6 } as never)).toEqual({
      complete: false,
      missing: ['gitCommit', 'configHash', 'seed', 'dataVersion'],
    })
  })

  it('is complete when the tuple is present (seed satisfied by the top-level seed)', () => {
    expect(
      validateRunProvenance({ seed: 3, provenance: { gitCommit: 'a', configHash: 'b', dataVersion: 'v1' } }),
    ).toEqual({ complete: true, missing: [] })
  })

  it('accepts a provenance-nested seed too, and flags only the genuinely-absent keys', () => {
    expect(
      validateRunProvenance({ provenance: { gitCommit: 'a', seed: 3 } }),
    ).toEqual({ complete: false, missing: ['configHash', 'dataVersion'] })
  })

  it('treats empty-string / null provenance values as missing', () => {
    expect(
      validateRunProvenance({ seed: 1, provenance: { gitCommit: '', configHash: null, dataVersion: 'v1' } }),
    ).toEqual({ complete: false, missing: ['gitCommit', 'configHash'] })
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

  it('carries per-step saliencyByGroup, keeping only finite entries', () => {
    const trace = validateDecisionTrace({
      steps: [{ step: 0, action: 'buy', saliencyByGroup: { 'layer:1d': 1, 'layer:1h': 'x' } }],
    })
    expect(trace?.steps[0]).toEqual({
      step: 0,
      action: 'buy',
      saliencyByGroup: { 'layer:1d': 1 },
    })
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

  it('coerces a well-formed attentionMatrix (2-D matrix attribution)', () => {
    const trace = validateDecisionTrace({
      steps: [{ step: 0, action: 'hold' }],
      attentionMatrix: {
        rows: ['t0', 't1'],
        cols: ['c0', 'c1'],
        grid: [
          [0.1, 0.2],
          [0.3, 0.4],
        ],
        method: 'attention-weights',
      },
    })
    expect(trace?.attentionMatrix).toEqual({
      rows: ['t0', 't1'],
      cols: ['c0', 'c1'],
      grid: [
        [0.1, 0.2],
        [0.3, 0.4],
      ],
      method: 'attention-weights',
    })
  })

  it('coerces the attentionMatrix sanityCheck (reuses the Adebayo shape)', () => {
    const trace = validateDecisionTrace({
      steps: [{ step: 0, action: 'hold' }],
      attentionMatrix: {
        rows: ['t0'],
        cols: ['c0'],
        grid: [[0.5]],
        sanityCheck: { method: 'model-randomization', rankCorrelation: 0.9, passed: true, junk: 1 },
      },
    })
    expect(trace?.attentionMatrix?.sanityCheck).toEqual({
      method: 'model-randomization',
      rankCorrelation: 0.9,
      passed: true,
    })
  })

  it('drops a ragged attentionMatrix grid but keeps the rest of the trace', () => {
    const trace = validateDecisionTrace({
      steps: [{ step: 0, action: 'hold' }],
      attentionMatrix: { rows: ['t0', 't1'], cols: ['c0', 'c1'], grid: [[0.1], [0.3, 0.4]] },
    })
    expect(trace).toBeDefined()
    expect(trace?.attentionMatrix).toBeUndefined()
  })

  it('drops an attentionMatrix whose grid row count ≠ rows.length', () => {
    const trace = validateDecisionTrace({
      steps: [{ step: 0, action: 'hold' }],
      attentionMatrix: { rows: ['t0', 't1'], cols: ['c0'], grid: [[0.1]] },
    })
    expect(trace?.attentionMatrix).toBeUndefined()
  })

  it('drops an attentionMatrix with a non-finite cell', () => {
    const trace = validateDecisionTrace({
      steps: [{ step: 0, action: 'hold' }],
      attentionMatrix: {
        rows: ['t0'],
        cols: ['c0', 'c1'],
        grid: [[0.1, Number.NaN]],
      },
    })
    expect(trace?.attentionMatrix).toBeUndefined()
  })

  it('drops an attentionMatrix missing rows or cols', () => {
    expect(
      validateDecisionTrace({
        steps: [{ step: 0, action: 'hold' }],
        attentionMatrix: { cols: ['c0'], grid: [[0.1]] },
      })?.attentionMatrix,
    ).toBeUndefined()
    expect(
      validateDecisionTrace({
        steps: [{ step: 0, action: 'hold' }],
        attentionMatrix: { rows: ['t0'], grid: [[0.1]] },
      })?.attentionMatrix,
    ).toBeUndefined()
  })
})

describe('summarizeSnapshotTraces (A6 mid-training snapshot index)', () => {
  it('shapes a well-formed index, sorted by training step, coercing keyMetrics', () => {
    const rows = summarizeSnapshotTraces([
      { step: 200, checkpointRef: 'm.step200', traceFile: 'checkpoints/m.snapshots.jsonl', keyMetrics: { actionCounts: { hold: 5, buy: 2, bad: 'x' }, totalSteps: 7 } },
      { step: 100, checkpointRef: 'm.step100', traceFile: 'checkpoints/m.snapshots.jsonl', keyMetrics: { actionCounts: { hold: 3 }, totalSteps: 3 } },
    ])
    expect(rows.map((r) => r.step)).toEqual([100, 200]) // ascending training order
    expect(rows[1]).toEqual({
      step: 200,
      checkpointRef: 'm.step200',
      traceFile: 'checkpoints/m.snapshots.jsonl',
      keyMetrics: { actionCounts: { hold: 5, buy: 2 }, totalSteps: 7 }, // non-numeric dropped
    })
  })

  it('drops malformed entries and returns [] for non-array input', () => {
    expect(summarizeSnapshotTraces(null)).toEqual([])
    expect(summarizeSnapshotTraces('nope')).toEqual([])
    const rows = summarizeSnapshotTraces([
      { step: 'bad', traceFile: 'f' }, // non-numeric step
      { step: 100, checkpointRef: 5 }, // no traceFile
      { step: 300, traceFile: 'f' }, // no keyMetrics -> {} + missing checkpointRef -> ''
    ])
    expect(rows).toEqual([{ step: 300, checkpointRef: '', traceFile: 'f', keyMetrics: {} }])
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

describe('summarizeStepAttribution', () => {
  it('returns undefined when no step carries per-step group saliency', () => {
    expect(
      summarizeStepAttribution({ steps: [{ step: 0, action: 'buy' }, { step: 1, action: 'hold' }] }),
    ).toBeUndefined()
  })

  it('summarizes groups, per-step dominant driver, dominance counts and sample count', () => {
    const summary = summarizeStepAttribution({
      steps: [
        { step: 0, action: 'buy', saliencyByGroup: { 'layer:1d': 3, 'layer:1h': 1 } },
        { step: 1, action: 'hold' },
        { step: 2, action: 'sell', saliencyByGroup: { 'layer:1d': 1, 'engineered:drawdown': 4 } },
      ],
    })
    expect(summary).toEqual({
      groups: ['engineered:drawdown', 'layer:1d', 'layer:1h'],
      perStep: [
        { step: 0, dominantGroup: 'layer:1d', byGroup: { 'layer:1d': 3, 'layer:1h': 1 } },
        { step: 2, dominantGroup: 'engineered:drawdown', byGroup: { 'layer:1d': 1, 'engineered:drawdown': 4 } },
      ],
      dominanceCounts: { 'layer:1d': 1, 'engineered:drawdown': 1 },
      samples: 2,
    })
  })

  it('breaks a dominance tie by group name for determinism', () => {
    const summary = summarizeStepAttribution({
      steps: [{ step: 0, action: 'buy', saliencyByGroup: { 'layer:1h': 2, 'layer:1d': 2 } }],
    })
    expect(summary?.perStep[0].dominantGroup).toBe('layer:1d')
    expect(summary?.dominanceCounts).toEqual({ 'layer:1d': 1 })
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
  it('teaches the cross-context form (environments/datasets bundles + comparison)', () => {
    const p = buildAnalyzePaperSystemPrompt(m)
    expect(p).toContain('environments')
    expect(p).toContain('datasets')
    expect(p).toContain('comparison')
    expect(p).toMatch(/beats-baseline|invariant|differs/)
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
      expect(p).toMatch(/more seeds|tighten an interval|reduce variance/)
    }
  })
})

describe('buildSuggestHypothesesUserContent', () => {
  it('serializes the paper + existing hypotheses + leverGuide, and text only when present', () => {
    const withText = JSON.parse(
      buildSuggestHypothesesUserContent({
        manifest: manifest(),
        paper: { title: 'P' },
        existingHypotheses: [{ id: 'h1', title: 'X' }],
        text: 'body',
      }),
    )
    expect(withText).toMatchObject({ paper: { title: 'P' }, text: 'body' })
    expect(withText.existingHypotheses).toHaveLength(1)
    // leverGuide names the lever roles so the prompt's precision rules are grounded in the manifest.
    expect(withText.leverGuide).toMatchObject({ identityLever: null }) // demo manifest has no model_name
    expect(Array.isArray(withText.leverGuide.modelLevers)).toBe(true)
    const noText = JSON.parse(
      buildSuggestHypothesesUserContent({
        manifest: manifest(),
        paper: { title: 'P' },
        existingHypotheses: [],
      }),
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

describe('coerceHypothesisWeights', () => {
  const linked = ['a', 'b', 'c']
  it('keeps only linked ids, rounds + clamps weight to 1..5, dedups', () => {
    const out = coerceHypothesisWeights(
      {
        weights: [
          { id: 'a', weight: 5, reason: 'central' },
          { id: 'b', weight: 9, reason: 'too high -> clamp 5' },
          { id: 'c', weight: 0, reason: 'too low -> clamp 1' },
          { id: 'a', weight: 2, reason: 'dup id ignored' },
          { id: 'x', weight: 3, reason: 'not linked -> dropped' },
        ],
      },
      linked,
    )
    expect(out).toEqual([
      { id: 'a', weight: 5, reason: 'central' },
      { id: 'b', weight: 5, reason: 'too high -> clamp 5' },
      { id: 'c', weight: 1, reason: 'too low -> clamp 1' },
    ])
  })
  it('rounds fractional weights and skips non-finite ones', () => {
    const out = coerceHypothesisWeights(
      {
        weights: [
          { id: 'a', weight: 3.4 },
          { id: 'b', weight: 'nope' },
        ],
      },
      linked,
    )
    expect(out).toEqual([{ id: 'a', weight: 3, reason: '' }])
  })
  it('resolves rows by 1-based index when the id is missing or wrong', () => {
    const out = coerceHypothesisWeights(
      {
        weights: [
          { index: 1, weight: 5, reason: 'by index' },
          { id: 'b', weight: 2, reason: 'by id' },
          { index: 3, id: 'wrong-id', weight: 4, reason: 'index resolves despite bad id' },
          { index: 99, weight: 1, reason: 'out of range -> dropped' },
        ],
      },
      linked,
    )
    expect(out).toEqual([
      { id: 'a', weight: 5, reason: 'by index' },
      { id: 'b', weight: 2, reason: 'by id' },
      { id: 'c', weight: 4, reason: 'index resolves despite bad id' },
    ])
  })
  it('accepts a bare array and defaults to empty on garbage', () => {
    expect(coerceHypothesisWeights([{ id: 'a', weight: 4, reason: 'r' }], linked)).toEqual([
      { id: 'a', weight: 4, reason: 'r' },
    ])
    expect(coerceHypothesisWeights('nope', linked)).toEqual([])
    expect(coerceHypothesisWeights({}, linked)).toEqual([])
  })
})

// --- Models catalog ----------------------------------------------------------

function modelManifest(
  choices: string[],
  leverExtras: Record<string, unknown> = {},
): TrainerManifest {
  return manifest({
    levers: {
      lr: { type: 'number', default: 0.01 },
      model_name: { type: 'choice', choices, default: choices[0], ...leverExtras },
    },
  })
}

describe('modelSlug', () => {
  it('kebab-cases a display name, dropping punctuation', () => {
    expect(modelSlug('Rainbow DQN (custom)')).toBe('rainbow-dqn-custom')
  })
  it('lowercases an acronym', () => {
    expect(modelSlug('IQN')).toBe('iqn')
  })
  it('trims surrounding whitespace + symbols to a clean slug', () => {
    expect(modelSlug('  A2C!! ')).toBe('a2c')
  })
  it('leaves an already-kebab identifier intact', () => {
    expect(modelSlug('duel-dqn-custom-lstm')).toBe('duel-dqn-custom-lstm')
  })
  it('is empty for a name with no alphanumerics', () => {
    expect(modelSlug('***')).toBe('')
  })
})

describe('inferModelCategory', () => {
  it('flags buy-and-hold as a baseline', () => {
    expect(inferModelCategory('hodl')).toBe('baseline')
  })
  it('flags a rule-based time strategy as a baseline', () => {
    expect(inferModelCategory('time-strategy')).toBe('baseline')
  })
  it('flags a supervised classifier as supervised', () => {
    expect(inferModelCategory('supervised-gbm')).toBe('supervised')
  })
  it('flags a features extractor as a component', () => {
    expect(inferModelCategory('sequence-features-extractor')).toBe('component')
  })
  it('defaults an RL algorithm name to rl', () => {
    expect(inferModelCategory('rainbow-dqn-custom')).toBe('rl')
  })
})

describe('humanizeModelName', () => {
  it('uppercases known acronyms and title-cases the rest', () => {
    expect(humanizeModelName('rainbow-dqn-custom')).toBe('Rainbow DQN Custom')
  })
  it('expands a known alias', () => {
    expect(humanizeModelName('hodl')).toBe('Buy-and-Hold')
  })
  it('keeps a single token', () => {
    expect(humanizeModelName('a2c')).toBe('A2C')
  })
})

describe('modelBindingNames', () => {
  it('reads model_name values from flavors', () => {
    expect(
      modelBindingNames({
        flavors: [{ modelName: 'a' }, { modelName: 'b' }, { name: 'x' } as never],
      }),
    ).toEqual(['a', 'b'])
  })
  it('falls back to a legacy flat modelNames[]', () => {
    expect(modelBindingNames({ modelNames: ['a', 'b'] })).toEqual(['a', 'b'])
  })
  it('is empty when neither is present', () => {
    expect(modelBindingNames({})).toEqual([])
  })
})

describe('discoverManifestModelCandidates', () => {
  it('returns lever choices not already covered by an existing model (by slug or binding)', () => {
    const out = discoverManifestModelCandidates(
      modelManifest(['dqn', 'ppo', 'rainbow-dqn-custom', 'hodl']),
      [
        { slug: 'dqn', modelNames: ['dqn'] },
        { slug: 'buy-and-hold', modelNames: ['hodl'] },
      ],
    )
    expect(out.map((c) => c.modelName)).toEqual(['ppo', 'rainbow-dqn-custom'])
  })
  it('carries a heuristic name + category + slug per candidate', () => {
    const out = discoverManifestModelCandidates(modelManifest(['rainbow-dqn-custom']), [])
    expect(out[0]).toEqual({
      modelName: 'rainbow-dqn-custom',
      slug: 'rainbow-dqn-custom',
      name: 'Rainbow DQN Custom',
      category: 'rl',
    })
  })
  it('is empty when the manifest has no model_name choice lever', () => {
    expect(discoverManifestModelCandidates(manifest(), [])).toEqual([])
  })
  it('is empty when model_name is not a choice lever', () => {
    const m = manifest({ levers: { model_name: { type: 'number', default: 1 } } })
    expect(discoverManifestModelCandidates(m, [])).toEqual([])
  })
})

describe('dedupeModelsBySlug', () => {
  it('keeps the first occurrence of each slug', () => {
    expect(
      dedupeModelsBySlug([
        { slug: 'a', v: 1 },
        { slug: 'b', v: 2 },
        { slug: 'a', v: 3 },
      ]),
    ).toEqual([
      { slug: 'a', v: 1 },
      { slug: 'b', v: 2 },
    ])
  })
})

describe('coerceScannedModels', () => {
  it('keeps only candidate slugs and filters fields + paper ids', () => {
    const out = coerceScannedModels(
      [
        {
          slug: 'ppo',
          name: 'Proximal Policy Optimization',
          description: 'On-policy.',
          category: 'rl',
          paperIds: ['p1', 'ghost'],
        },
        { slug: 'unknown', name: 'x' },
        { slug: 'rainbow-dqn-custom', category: 'nonsense' },
      ],
      new Set(['ppo', 'rainbow-dqn-custom']),
      new Set(['p1']),
    )
    expect(out.get('ppo')).toEqual({
      name: 'Proximal Policy Optimization',
      description: 'On-policy.',
      category: 'rl',
      paperIds: ['p1'],
    })
    expect(out.get('rainbow-dqn-custom')).toEqual({})
    expect(out.has('unknown')).toBe(false)
  })
  it('is an empty map for a non-array response', () => {
    expect(coerceScannedModels('nope', new Set(['a']), new Set()).size).toBe(0)
  })
})

describe('buildScanModelsSystemPrompt', () => {
  it('names the project and asks for a JSON array', () => {
    const p = buildScanModelsSystemPrompt(manifest())
    expect(p).toContain('demo')
    expect(p).toContain('JSON array')
  })
})

describe('buildScanModelsUserContent', () => {
  it('serialises candidates + papers as parseable JSON', () => {
    const content = buildScanModelsUserContent({
      candidates: [{ slug: 'ppo', modelName: 'ppo', name: 'PPO', category: 'rl' }],
      papers: [{ id: 'p1', title: 'A paper', claim: 'beats hold' }],
      leverDescription: 'the algorithm',
    })
    const parsed = JSON.parse(content)
    expect(parsed.candidates[0].slug).toBe('ppo')
    expect(parsed.papers[0].id).toBe('p1')
    expect(parsed.leverDescription).toBe('the algorithm')
  })
})

describe('coerceAnalyzedPaperModels', () => {
  it('keeps string match ids and coerces proposed models (slug from name, category default)', () => {
    const out = coerceAnalyzedPaperModels({
      matchModelIds: ['rainbow-dqn-custom', '', 5],
      proposedModels: [
        {
          name: 'C51 Distributional DQN',
          description: 'categorical value distribution',
          category: 'rl',
          proposal: 'add categorical DQN head',
        },
        { description: 'no name — dropped' },
        { name: 'NoisyNet', category: 'bogus' },
      ],
    })
    expect(out.matchModelIds).toEqual(['rainbow-dqn-custom'])
    expect(out.proposedModels).toEqual<ProposedModel[]>([
      {
        name: 'C51 Distributional DQN',
        slug: 'c51-distributional-dqn',
        description: 'categorical value distribution',
        category: 'rl',
        proposal: 'add categorical DQN head',
      },
      { name: 'NoisyNet', slug: 'noisynet', description: '', category: 'rl', proposal: '' },
    ])
  })
  it('coerces general proposedImprovements (functionality to build), classifying the kind', () => {
    const out = coerceAnalyzedPaperModels({
      proposedImprovements: [
        { title: 'Order-book feature stream', detail: 'ingest L2 depth', kind: 'data' },
        { title: '  Sortino metric  ', detail: '', kind: 'metric' },
        { title: 'Funding-rate env mechanic' }, // no kind → default capability, detail ''
        { detail: 'no title — dropped' },
        'not an object',
      ],
    })
    expect(out.proposedImprovements).toEqual([
      { title: 'Order-book feature stream', detail: 'ingest L2 depth', kind: 'data' },
      { title: 'Sortino metric', detail: '', kind: 'metric' },
      { title: 'Funding-rate env mechanic', detail: '', kind: 'capability' },
    ])
  })
  it('falls back to the capability kind for an unknown kind string', () => {
    const out = coerceAnalyzedPaperModels({
      proposedImprovements: [{ title: 'X', kind: 'bogus-kind' }],
    })
    expect(out.proposedImprovements[0].kind).toBe('capability')
  })
  it('defaults to empty for a malformed response', () => {
    expect(coerceAnalyzedPaperModels('nope')).toEqual({
      matchModelIds: [],
      proposedModels: [],
      proposedImprovements: [],
    })
  })
})

describe('mergeProposedImprovements', () => {
  const imp = (title, kind = 'model', inapplicable) => ({
    title,
    detail: '',
    kind,
    ...(inapplicable ? { inapplicable: true } : {}),
  })

  it('returns the incoming list unchanged when nothing was marked inapplicable', () => {
    const incoming = [imp('A'), imp('B')]
    expect(mergeProposedImprovements([imp('A')], incoming)).toEqual(incoming)
  })

  it('re-applies an existing inapplicable mark to a matching incoming item (case/space-insensitive)', () => {
    const existing = [imp("White's  Reality Check", 'model', true)]
    const incoming = [imp("white's reality check"), imp('B')]
    const out = mergeProposedImprovements(existing, incoming)
    expect(out.find((i) => i.title === "white's reality check").inapplicable).toBe(true)
    expect(out.find((i) => i.title === 'B').inapplicable).toBeUndefined()
  })

  it('keeps an inapplicable item the new run dropped, so it stays listed + referable', () => {
    const existing = [imp('Gone', 'model', true)]
    const incoming = [imp('Fresh')]
    const out = mergeProposedImprovements(existing, incoming)
    expect(out.map((i) => i.title)).toEqual(['Fresh', 'Gone'])
    expect(out.find((i) => i.title === 'Gone').inapplicable).toBe(true)
  })

  it('drops an existing APPLICABLE item absent from the incoming list (fresh list wins)', () => {
    const out = mergeProposedImprovements([imp('Stale')], [imp('New')])
    expect(out.map((i) => i.title)).toEqual(['New'])
  })

  it('tolerates missing/empty existing', () => {
    expect(mergeProposedImprovements(undefined, [imp('A')])).toEqual([imp('A')])
    expect(mergeProposedImprovements([], [imp('A')])).toEqual([imp('A')])
  })
})

describe('detectMissingPaperModels', () => {
  const proposed: ProposedModel[] = [
    { name: 'Rainbow DQN', slug: 'rainbow-dqn', description: '', category: 'rl', proposal: '' },
    { name: 'C51', slug: 'c51', description: '', category: 'rl', proposal: '' },
  ]
  it('drops a proposal already covered by a catalog binding', () => {
    const out = detectMissingPaperModels(proposed, [
      {
        slug: 'rainbow-dqn-custom',
        name: 'Rainbow DQN (custom)',
        modelNames: ['rainbow-dqn-custom', 'rainbow-dqn'],
      },
    ])
    expect(out.map((p) => p.slug)).toEqual(['c51'])
  })
  it('drops a proposal whose slug matches a catalog slug', () => {
    const out = detectMissingPaperModels(proposed, [
      { slug: 'rainbow-dqn', name: 'x', modelNames: [] },
    ])
    expect(out.map((p) => p.slug)).toEqual(['c51'])
  })
  it('returns all when the catalog is empty', () => {
    expect(detectMissingPaperModels(proposed, [])).toHaveLength(2)
  })
  it('matches a proposal against an un-kebab catalog slug (slug normalized on compare)', () => {
    const out = detectMissingPaperModels(
      [{ name: 'A2C', slug: 'a2c', description: '', category: 'rl', proposal: '' }],
      [{ slug: 'A2C', name: 'Different Name', modelNames: [] }],
    )
    expect(out).toHaveLength(0)
  })
  it('drops a proposal whose slug matches a catalog ALIAS (e.g. policy-gradient -> a2c)', () => {
    const pg: ProposedModel[] = [
      {
        name: 'Policy Gradient',
        slug: 'policy-gradient',
        description: '',
        category: 'rl',
        proposal: '',
      },
    ]
    const out = detectMissingPaperModels(pg, [
      { slug: 'a2c', name: 'A2C', modelNames: ['a2c'], aliases: ['policy-gradient'] },
    ])
    expect(out).toHaveLength(0)
  })
})

describe('buildAnalyzePaperModelsSystemPrompt', () => {
  it('names the project and asks for a JSON object', () => {
    const p = buildAnalyzePaperModelsSystemPrompt(manifest())
    expect(p).toContain('demo')
    expect(p).toContain('JSON object')
  })
})

describe('buildAnalyzePaperModelsUserContent', () => {
  it('serialises paper + existing models + optional text', () => {
    const content = buildAnalyzePaperModelsUserContent({
      paper: { title: 'P', claim: 'c' },
      existingModels: [
        { id: 'ppo', name: 'PPO', slug: 'ppo', category: 'rl', modelNames: ['ppo'] },
      ],
      text: 'abstract',
    })
    const parsed = JSON.parse(content)
    expect(parsed.paper.title).toBe('P')
    expect(parsed.existingModels[0].slug).toBe('ppo')
    expect(parsed.text).toBe('abstract')
  })
})

describe('isRunAffectedByFidelityDesync', () => {
  // SINGLE-timeline (SingleDataProvider) configs — UNAFFECTED by the observation/reward desync.
  it('treats 1d@1d as not affected (single 1d base)', () => {
    expect(isRunAffectedByFidelityDesync({ timeframe: '1d', fidelity_set: '1d' })).toBe(false)
  })
  it('treats 1h@1h as not affected (single 1h base)', () => {
    expect(isRunAffectedByFidelityDesync({ timeframe: '1h', fidelity_set: '1h' })).toBe(false)
  })
  it('treats auto@1d as not affected (auto resolves to single 1d)', () => {
    expect(isRunAffectedByFidelityDesync({ timeframe: '1d', fidelity_set: 'auto' })).toBe(false)
  })
  it('treats a missing fidelity_set at 1d as not affected (defaults to auto -> 1d)', () => {
    expect(isRunAffectedByFidelityDesync({ timeframe: '1d' })).toBe(false)
  })
  it('treats a fully-defaulted config as not affected (timeframe 1d, fidelity auto -> 1d)', () => {
    expect(isRunAffectedByFidelityDesync({ model_name: 'reppo-custom' })).toBe(false)
  })

  // MULTI-timeline (resolved-from-fidelity) configs — AFFECTED.
  it('treats auto@1h as affected (auto resolves to 1h+1d, multi)', () => {
    expect(isRunAffectedByFidelityDesync({ timeframe: '1h', fidelity_set: 'auto' })).toBe(true)
  })
  it('treats a missing fidelity_set at 1h as affected (defaults to auto -> 1h+1d)', () => {
    expect(isRunAffectedByFidelityDesync({ timeframe: '1h' })).toBe(true)
  })
  it.each(['1h+1d', '1h+1d+1w', '1d+1w'])('treats multi-layer set %s as affected', (fset) => {
    expect(isRunAffectedByFidelityDesync({ timeframe: '1h', fidelity_set: fset })).toBe(true)
    expect(isRunAffectedByFidelityDesync({ timeframe: '1d', fidelity_set: fset })).toBe(true)
  })
  it('treats a coarser single layer at a finer step (1d@1h) as affected', () => {
    expect(isRunAffectedByFidelityDesync({ timeframe: '1h', fidelity_set: '1d' })).toBe(true)
  })
  it('treats a finer base at a coarser step (1h@1d) as affected', () => {
    expect(isRunAffectedByFidelityDesync({ timeframe: '1d', fidelity_set: '1h' })).toBe(true)
  })

  // Guards.
  it('returns false for a missing/invalid config (never invalidate the unknown)', () => {
    expect(isRunAffectedByFidelityDesync(undefined)).toBe(false)
    expect(isRunAffectedByFidelityDesync(null as unknown as Record<string, unknown>)).toBe(false)
  })
})

describe('isSpecAffectedByFidelityDesync', () => {
  it('flags a fixed-only spec on the multi path', () => {
    expect(
      isSpecAffectedByFidelityDesync({ fixed: { timeframe: '1h', fidelity_set: '1h+1d' } }),
    ).toBe(true)
  })
  it('does not flag a fixed-only spec on the single path', () => {
    expect(isSpecAffectedByFidelityDesync({ fixed: { timeframe: '1d', fidelity_set: '1d' } })).toBe(
      false,
    )
  })
  it('flags a spec that SWEEPS fidelity_set into an affected value even when fixed is single', () => {
    expect(
      isSpecAffectedByFidelityDesync({
        fixed: { timeframe: '1d', fidelity_set: '1d' },
        sweep: { fidelity_set: ['1d', '1h+1d'] },
      }),
    ).toBe(true)
  })
  it('flags a spec that SWEEPS timeframe into an affected combination', () => {
    // fixed fidelity_set 1h; sweeping timeframe to 1d makes 1h@1d (a finer base at a coarser step) — affected.
    expect(
      isSpecAffectedByFidelityDesync({
        fixed: { fidelity_set: '1h' },
        sweep: { timeframe: ['1h', '1d'] },
      }),
    ).toBe(true)
  })
  it('does not flag a sweep whose every combination stays single', () => {
    expect(
      isSpecAffectedByFidelityDesync({
        fixed: { timeframe: '1d', fidelity_set: '1d' },
        sweep: { seed: [1, 2, 3] },
      }),
    ).toBe(false)
  })
  it('returns false for a missing spec', () => {
    expect(isSpecAffectedByFidelityDesync(undefined)).toBe(false)
  })
})

describe('isRunAffectedByFidelityLookahead', () => {
  // The v6 look-ahead: the run-cadence layer AND the base are BOTH absent from the observed layers
  // (an hourly/minute step observing ONLY coarser layers), so the provider never sliced its price series
  // to the decision cadence and get_price(step) trailed the observation by the warmup — the model saw
  // ~32 days of FUTURE relative to the traded price. AFFECTED iff every observed layer is strictly coarser
  // than the step.
  it('flags 1h@1d (the runs-audit config) as affected', () => {
    expect(isRunAffectedByFidelityLookahead({ timeframe: '1h', fidelity_set: '1d' })).toBe(true)
  })
  it('flags 1h@1d+1w (coarse-only at an hourly step) as affected', () => {
    expect(isRunAffectedByFidelityLookahead({ timeframe: '1h', fidelity_set: '1d+1w' })).toBe(true)
  })
  it.each(['1d', '1h', '1h+1d', '1d+1w'])('flags a minute step observing only %s as affected', (fset) => {
    expect(isRunAffectedByFidelityLookahead({ timeframe: '1m', fidelity_set: fset })).toBe(true)
  })

  // NOT affected — the base/run cadence IS observed (these v5 runs are correct; must NOT be invalidated).
  it.each(['1h+1d', '1h+1d+1w', '1h'])('does NOT flag %s at an hourly step (base 1h observed)', (fset) => {
    expect(isRunAffectedByFidelityLookahead({ timeframe: '1h', fidelity_set: fset })).toBe(false)
  })
  it('does NOT flag 1h@1d run-through the daily step (1d@1h: base 1h observed, divider handles it)', () => {
    expect(isRunAffectedByFidelityLookahead({ timeframe: '1d', fidelity_set: '1h' })).toBe(false)
  })
  it.each(['1d', '1d+1w', 'auto'])('does NOT flag a daily step observing %s (run cadence 1d observed)', (fset) => {
    expect(isRunAffectedByFidelityLookahead({ timeframe: '1d', fidelity_set: fset })).toBe(false)
  })
  it('does NOT flag single-timeline or auto (base always observed)', () => {
    expect(isRunAffectedByFidelityLookahead({ timeframe: '1h', fidelity_set: '1h' })).toBe(false)
    expect(isRunAffectedByFidelityLookahead({ timeframe: '1h', fidelity_set: 'auto' })).toBe(false)
    expect(isRunAffectedByFidelityLookahead({ timeframe: '1h' })).toBe(false)
    expect(isRunAffectedByFidelityLookahead({ timeframe: '1m', fidelity_set: '1m+1h' })).toBe(false)
  })
  it('returns false for a missing/invalid/unknown config (never invalidate the unknown)', () => {
    expect(isRunAffectedByFidelityLookahead(undefined)).toBe(false)
    expect(isRunAffectedByFidelityLookahead(null as unknown as Record<string, unknown>)).toBe(false)
    expect(isRunAffectedByFidelityLookahead({ timeframe: '1h', fidelity_set: 'bogus' })).toBe(false)
  })
})

describe('isSpecAffectedByFidelityLookahead', () => {
  it('flags a fixed-only spec on the look-ahead path (1h@1d)', () => {
    expect(isSpecAffectedByFidelityLookahead({ fixed: { timeframe: '1h', fidelity_set: '1d' } })).toBe(
      true,
    )
  })
  it('does NOT flag a fixed-only multi-layer spec that observes the base (1h+1d)', () => {
    expect(
      isSpecAffectedByFidelityLookahead({ fixed: { timeframe: '1h', fidelity_set: '1h+1d' } }),
    ).toBe(false)
  })
  it('flags a spec that SWEEPS fidelity_set into a coarse-only value even when fixed is safe', () => {
    expect(
      isSpecAffectedByFidelityLookahead({
        fixed: { timeframe: '1h', fidelity_set: '1h+1d' },
        sweep: { fidelity_set: ['1h+1d', '1d'] },
      }),
    ).toBe(true)
  })
  it('returns false for a missing spec', () => {
    expect(isSpecAffectedByFidelityLookahead(undefined)).toBe(false)
  })
})

describe('coerceConsolidationGroups', () => {
  const ids = ['a', 'b', 'c', 'd']

  it('parses a {groups:[...]} object and keeps valid groups', () => {
    const groups = coerceConsolidationGroups(
      { groups: [{ canonicalId: 'a', duplicateIds: ['b', 'c'], reason: ' same thing ' }] },
      ids,
    )
    expect(groups).toEqual([{ canonicalId: 'a', duplicateIds: ['b', 'c'], reason: 'same thing' }])
  })

  it('also accepts a bare array of groups', () => {
    const groups = coerceConsolidationGroups([{ canonicalId: 'a', duplicateIds: ['b'] }], ids)
    expect(groups).toEqual([{ canonicalId: 'a', duplicateIds: ['b'], reason: '' }])
  })

  it('drops a group whose canonicalId is not a real model', () => {
    expect(coerceConsolidationGroups([{ canonicalId: 'zzz', duplicateIds: ['b'] }], ids)).toEqual(
      [],
    )
  })

  it('filters out duplicateIds that are unknown, self, or repeated', () => {
    const groups = coerceConsolidationGroups(
      [{ canonicalId: 'a', duplicateIds: ['a', 'b', 'b', 'zzz', 'c'] }],
      ids,
    )
    expect(groups[0].duplicateIds).toEqual(['b', 'c'])
  })

  it('drops a group that has no valid duplicates left', () => {
    expect(
      coerceConsolidationGroups([{ canonicalId: 'a', duplicateIds: ['zzz', 'a'] }], ids),
    ).toEqual([])
  })

  it('never lets one model appear in two groups (cross-group de-dup)', () => {
    const groups = coerceConsolidationGroups(
      [
        { canonicalId: 'a', duplicateIds: ['b'] }, // consumes a, b
        { canonicalId: 'b', duplicateIds: ['c'] }, // b already consumed -> whole group dropped
        { canonicalId: 'c', duplicateIds: ['d'] }, // c + d still free -> kept
      ],
      ids,
    )
    expect(groups).toEqual([
      { canonicalId: 'a', duplicateIds: ['b'], reason: '' },
      { canonicalId: 'c', duplicateIds: ['d'], reason: '' },
    ])
  })

  it('returns [] for non-object / null / malformed input', () => {
    expect(coerceConsolidationGroups(null, ids)).toEqual([])
    expect(coerceConsolidationGroups('nope', ids)).toEqual([])
    expect(coerceConsolidationGroups({ groups: 'x' }, ids)).toEqual([])
    expect(coerceConsolidationGroups([{ canonicalId: 'a' }], ids)).toEqual([])
  })
})

describe('buildConsolidateModelsSystemPrompt', () => {
  it('asks for a single JSON object with the group schema + names the objective', () => {
    const prompt = buildConsolidateModelsSystemPrompt(manifest())
    expect(prompt).toContain('canonicalId')
    expect(prompt).toContain('duplicateIds')
    expect(prompt).toContain('groups')
    expect(prompt).toMatch(/JSON object/i)
    expect(prompt).toContain(manifest().objective.name)
  })
})

describe('buildConsolidateModelsUserContent', () => {
  it('round-trips the model summaries as JSON', () => {
    const models = [
      { id: 'a', name: 'A', slug: 'a', category: 'rl', description: 'x', modelNames: ['a-name'] },
    ]
    const parsed = JSON.parse(buildConsolidateModelsUserContent({ models }))
    expect(parsed.models).toEqual(models)
  })
})

describe('appliesWhenMap', () => {
  const manifest = {
    recordType: 'r',
    objective: { name: 'objective', direction: 'max' as const },
    run: 'x',
    levers: {
      model_name: { type: 'string' as const },
      forward_horizon: {
        type: 'number' as const,
        appliesWhen: { model_name: ['supervised-logreg', 'supervised-gbm'] },
      },
      momentum_lookback: { type: 'number' as const, appliesWhen: { model_name: ['momentum'] } },
      lr: { type: 'number' as const },
    },
  } as unknown as Parameters<typeof appliesWhenMap>[0]

  it('extracts only the levers that declare appliesWhen', () => {
    expect(appliesWhenMap(manifest)).toEqual({
      forward_horizon: { model_name: ['supervised-logreg', 'supervised-gbm'] },
      momentum_lookback: { model_name: ['momentum'] },
    })
  })

  it('is empty when no lever is conditional', () => {
    const m = {
      recordType: 'r',
      levers: { lr: { type: 'number' as const } },
    } as unknown as Parameters<typeof appliesWhenMap>[0]
    expect(appliesWhenMap(m)).toEqual({})
  })

  it('tolerates a manifest with no levers', () => {
    expect(
      appliesWhenMap({ recordType: 'r' } as unknown as Parameters<typeof appliesWhenMap>[0]),
    ).toEqual({})
  })
})

describe('hypothesisConsolidationKey', () => {
  const k = (spec: any) => hypothesisConsolidationKey(spec)
  it('two specs identical except sweep VALUES (one wider) share a key', () => {
    expect(k({ fixed: { model_name: 'ppo' }, sweep: { lr: [0.1] } })).toBe(
      k({ fixed: { model_name: 'ppo' }, sweep: { lr: [0.1, 0.2] } }),
    )
  })
  it('different sweep-KEY SETS differ', () => {
    expect(k({ fixed: { model_name: 'ppo' }, sweep: { lr: [0.1] } })).not.toBe(
      k({ fixed: { model_name: 'ppo' }, sweep: { batch: [32] } }),
    )
  })
  it('different fixed VALUES differ', () => {
    expect(k({ fixed: { model_name: 'ppo' } })).not.toBe(k({ fixed: { model_name: 'reppo' } }))
  })
  it('compare same lever but different VALUES share a key; present vs absent differ', () => {
    expect(k({ compare: { lever: 'model_name', values: ['a', 'b'] } })).toBe(
      k({ compare: { lever: 'model_name', values: ['a', 'b', 'c'] } }),
    )
    expect(k({ compare: { lever: 'model_name', values: ['a', 'b'] } })).not.toBe(k({ fixed: {} }))
  })
  it('same bundles in DIFFERENT order differ (bundle order is semantic)', () => {
    expect(k({ environments: [{ shorting: true }, { shorting: false }] })).not.toBe(
      k({ environments: [{ shorting: false }, { shorting: true }] }),
    )
  })
  it('seeds are NOT part of the key (wider seed set is the same hypothesis)', () => {
    expect(k({ fixed: { model_name: 'ppo' }, seeds: [0] })).toBe(
      k({ fixed: { model_name: 'ppo' }, seeds: [0, 1, 2] }),
    )
  })
  it('treats an undefined spec exactly like an empty one', () => {
    expect(hypothesisConsolidationKey(undefined)).toBe(hypothesisConsolidationKey({}))
  })
})

describe('mergeHypothesisSpecs', () => {
  it('unions sweep values (deduped + sorted), keeps fixed exact, drops configs/maxItems', () => {
    const merged = mergeHypothesisSpecs([
      { fixed: { model_name: 'ppo' }, sweep: { lr: [0.2, 0.1] }, maxItems: 5 },
      { fixed: { model_name: 'ppo' }, sweep: { lr: [0.1, 0.3] }, configs: [{ config: {} }] },
    ] as any)
    expect(merged.fixed).toEqual({ model_name: 'ppo' })
    expect(merged.sweep!.lr).toEqual([0.1, 0.2, 0.3])
    expect('maxItems' in merged).toBe(false)
    expect('configs' in merged).toBe(false)
  })
  it('is ORDER-INDEPENDENT: merging in either order yields the same id', () => {
    const a = { fixed: { m: 'x' }, sweep: { lr: [0.1] }, seeds: [1] }
    const b = { fixed: { m: 'x' }, sweep: { lr: [0.2] }, seeds: [0] }
    expect(hashTrainingConfig(mergeHypothesisSpecs([a, b] as any) as any)).toBe(
      hashTrainingConfig(mergeHypothesisSpecs([b, a] as any) as any),
    )
  })
  it('unions compare values keeping the lever; unions seeds ascending', () => {
    const merged = mergeHypothesisSpecs([
      { compare: { lever: 'model_name', values: ['b', 'a'] }, seeds: [2] },
      { compare: { lever: 'model_name', values: ['a', 'c'] }, seeds: [0, 1] },
    ] as any)
    expect(merged.compare!.lever).toBe('model_name')
    expect(merged.compare!.values).toEqual(['a', 'b', 'c'])
    expect(merged.seeds).toEqual([0, 1, 2])
  })
  it('idempotent fixed point: merging the merged spec again is a no-op (same id)', () => {
    const merged = mergeHypothesisSpecs([
      { fixed: { m: 'x' }, sweep: { lr: [0.1] } },
      { fixed: { m: 'x' }, sweep: { lr: [0.2] } },
    ] as any)
    expect(hashTrainingConfig(mergeHypothesisSpecs([merged]) as any)).toBe(
      hashTrainingConfig(merged as any),
    )
  })
  it('an empty member list merges to an empty spec', () => {
    expect(mergeHypothesisSpecs([])).toEqual({})
  })
  it('unions sweep DIMENSIONS across members that each sweep a different lever', () => {
    // A member missing a swept lever contributes [] to that lever's union — the axis is not lost.
    const merged = mergeHypothesisSpecs([{ sweep: { lr: [0.1] } }, { sweep: { batch: [32] } }] as any)
    expect(merged.sweep).toEqual({ batch: [32], lr: [0.1] })
  })
  it('unions compare values even when only one member declares the compare block', () => {
    const merged = mergeHypothesisSpecs([
      { compare: { lever: 'model_name', values: ['a'] } },
      { fixed: { x: 1 } },
    ] as any)
    expect(merged.compare).toEqual({ lever: 'model_name', values: ['a'] })
  })
  it('keeps the first member environments and datasets verbatim', () => {
    const merged = mergeHypothesisSpecs([
      { fixed: { m: 'x' }, environments: [{ shorting: true }], datasets: [{ asset: 'btc' }] },
      { fixed: { m: 'x' } },
    ] as any)
    expect(merged.environments).toEqual([{ shorting: true }])
    expect(merged.datasets).toEqual([{ asset: 'btc' }])
  })
})

describe('pickCanonicalHypothesis', () => {
  const h = (o: any) => ({
    id: o.id,
    spec: {},
    status: 'untested',
    verdictSource: 'auto',
    source: 'llm',
    ...o,
  })
  it('a single manual member wins', () => {
    const r = pickCanonicalHypothesis([
      h({ id: 'a', source: 'human' }),
      h({ id: 'b', verdictSource: 'manual', status: 'proven' }),
    ] as any)
    expect(r.conflict).toBe(false)
    expect(r.canonical!.id).toBe('b')
  })
  it('no manual: source priority human>paper>llm, then earliest createdAt', () => {
    const r = pickCanonicalHypothesis([
      h({ id: 'a', source: 'llm', createdAt: '2026-01-01' }),
      h({ id: 'b', source: 'paper', createdAt: '2026-02-01' }),
      h({ id: 'c', source: 'human', createdAt: '2026-03-01' }),
    ] as any)
    expect(r.canonical!.id).toBe('c')
  })
  it('conflicting manual statuses -> conflict signal, no silent pick', () => {
    const r = pickCanonicalHypothesis([
      h({ id: 'a', verdictSource: 'manual', status: 'proven' }),
      h({ id: 'b', verdictSource: 'manual', status: 'disproved' }),
    ] as any)
    expect(r.conflict).toBe(true)
    expect(r.canonical).toBeNull()
  })
  it('multiple manual with the SAME status ranks them (not a conflict)', () => {
    const r = pickCanonicalHypothesis([
      h({ id: 'a', verdictSource: 'manual', status: 'proven', source: 'llm', createdAt: '2026-02-01' }),
      h({ id: 'b', verdictSource: 'manual', status: 'proven', source: 'human', createdAt: '2026-03-01' }),
    ] as any)
    expect(r.conflict).toBe(false)
    expect(r.canonical!.id).toBe('b') // human outranks llm within the manual set
  })
  it('ranks an unknown source LAST, then by createdAt, then by id (rank tiebreaks)', () => {
    // unknown source sinks below a known one (priority default of 9)
    expect(
      pickCanonicalHypothesis([
        h({ id: 'x', source: 'mystery' }),
        h({ id: 'y', source: 'human' }),
      ] as any).canonical!.id,
    ).toBe('y')
    // same source + NO createdAt on either → both fall back to '' → pure id lexical tiebreak
    expect(
      pickCanonicalHypothesis([
        h({ id: 'zeta', source: 'mystery' }),
        h({ id: 'alpha', source: 'mystery' }),
      ] as any).canonical!.id,
    ).toBe('alpha')
    // same source, different createdAt → earliest wins, asserted in BOTH input orders so the
    // comparator exercises ta<tb in each direction
    expect(
      pickCanonicalHypothesis([
        h({ id: 'late', source: 'llm', createdAt: '2026-05-01' }),
        h({ id: 'early', source: 'llm', createdAt: '2026-01-01' }),
      ] as any).canonical!.id,
    ).toBe('early')
    expect(
      pickCanonicalHypothesis([
        h({ id: 'early', source: 'llm', createdAt: '2026-01-01' }),
        h({ id: 'late', source: 'llm', createdAt: '2026-05-01' }),
      ] as any).canonical!.id,
    ).toBe('early')
  })
  it('returns a null canonical for an empty member list', () => {
    expect(pickCanonicalHypothesis([])).toEqual({ canonical: null, conflict: false })
  })
})

describe('groupHypothesesForConsolidation', () => {
  const h = (id: string, spec: any, o: any = {}) => ({
    id,
    spec,
    status: 'untested',
    verdictSource: 'auto',
    source: 'llm',
    ...o,
  })
  it('groups genuine widening pairs; drops singletons + dismissed + already-shared-id', () => {
    const hyps = [
      h('h1', { fixed: { m: 'x' }, sweep: { lr: [0.1] } }),
      h('h2', { fixed: { m: 'x' }, sweep: { lr: [0.2] } }),
      h('lonely', { fixed: { m: 'y' } }),
      h('d1', { fixed: { m: 'z' }, sweep: { lr: [0.1] } }, { dismissed: true }),
      h('d2', { fixed: { m: 'z' }, sweep: { lr: [0.2] } }, { dismissed: true }),
    ]
    const groups = groupHypothesesForConsolidation(hyps as any, hashTrainingConfig)
    expect(groups.length).toBe(1)
    expect(new Set(groups[0].members.map((m: any) => m.id))).toEqual(new Set(['h1', 'h2']))
  })
  it('defers a group while a member has an in-flight (running) campaign', () => {
    const hyps = [
      h('h1', { fixed: { m: 'x' }, sweep: { lr: [0.1] } }, { campaign: { status: 'running' } }),
      h('h2', { fixed: { m: 'x' }, sweep: { lr: [0.2] } }),
    ]
    expect(groupHypothesesForConsolidation(hyps as any, hashTrainingConfig).length).toBe(0)
  })
  it('defers a group while a member has a QUEUED campaign', () => {
    const hyps = [
      h('h1', { fixed: { m: 'x' }, sweep: { lr: [0.1] } }, { campaign: { status: 'queued' } }),
      h('h2', { fixed: { m: 'x' }, sweep: { lr: [0.2] } }),
    ]
    expect(groupHypothesesForConsolidation(hyps as any, hashTrainingConfig).length).toBe(0)
  })
  it('skips a bucket whose members already share one id (already consolidated)', () => {
    // Same id + same consolidation key (sweep VALUES differ but keys match) → nothing to merge.
    const hyps = [
      h('same', { fixed: { m: 'x' }, sweep: { lr: [0.1] } }),
      h('same', { fixed: { m: 'x' }, sweep: { lr: [0.2] } }),
    ]
    expect(groupHypothesesForConsolidation(hyps as any, hashTrainingConfig).length).toBe(0)
  })
})

describe('planHypothesisConsolidation', () => {
  const h = (id: string, spec: any, o: any = {}) => ({
    id,
    spec,
    title: id,
    rationale: '',
    status: 'untested',
    verdictSource: 'auto',
    source: 'llm',
    createdAt: '2026-01-01',
    updatedAt: '2026-01-01',
    paperIds: [],
    ...o,
  })
  const plan = (members: any[], papers: any[] = [], models: any[] = [], experiments: any[] = []) =>
    planHypothesisConsolidation(
      { members } as any,
      papers,
      models,
      experiments,
      '2026-06-30T00:00:00Z',
      hashTrainingConfig,
    )
  const exp = (id: string, hypothesisId: string) => ({
    id,
    hypothesisId,
    thesis: 't',
    status: 'completed',
    matrix: {},
    cells: [],
    aggregate: null,
    provenance: { source: 'llm', createdAt: '2026-01-01', updatedAt: '2026-01-01' },
  })

  it('repoints an experiment linked to an absorbed hypothesis onto the survivor id', () => {
    const p = plan(
      [
        h('h1', { fixed: { m: 'x' }, sweep: { lr: [0.1] } }),
        h('h2', { fixed: { m: 'x' }, sweep: { lr: [0.2] } }),
      ],
      [],
      [],
      [exp('e1', 'h1'), exp('e2', 'unrelated')],
    )
    const newId = p!.unionRecord.id
    expect(p!.changedExperiments.map((e) => e.id)).toEqual(['e1'])
    expect(p!.changedExperiments[0].hypothesisId).toBe(newId)
    expect(p!.changedExperiments[0].provenance.updatedAt).toBe('2026-06-30T00:00:00Z')
  })

  it('builds a union record at the merged-spec id, lists the others as deleted, unions paperIds', () => {
    const p = plan([
      h('h1', { fixed: { m: 'x' }, sweep: { lr: [0.1] } }, { paperIds: ['pA'] }),
      h('h2', { fixed: { m: 'x' }, sweep: { lr: [0.2] } }, { paperIds: ['pB'] }),
    ])
    expect(p).not.toBeNull()
    const newId = hashTrainingConfig(
      mergeHypothesisSpecs([
        { fixed: { m: 'x' }, sweep: { lr: [0.1] } },
        { fixed: { m: 'x' }, sweep: { lr: [0.2] } },
      ]) as any,
    )
    expect(p!.unionRecord.id).toBe(newId)
    expect(p!.unionRecord.spec.sweep.lr).toEqual([0.1, 0.2])
    expect(new Set(p!.unionRecord.paperIds)).toEqual(new Set(['pA', 'pB']))
    expect(new Set(p!.deletedIds)).toEqual(new Set(['h1', 'h2']))
    expect(p!.unionRecord.verdictSource).toBe('auto')
    expect('evidence' in p!.unionRecord).toBe(false)
    expect('transitions' in p!.unionRecord).toBe(false)
  })

  it('repoints paper.hypothesisIds and PRESERVES the weight onto the survivor (max on collision)', () => {
    const paper = {
      id: 'pA',
      hypothesisIds: ['h1', 'h2', 'other'],
      hypothesisWeights: { h1: 2, h2: 5, other: 1 },
    }
    const p = plan(
      [
        h('h1', { fixed: { m: 'x' }, sweep: { lr: [0.1] } }, { paperIds: ['pA'] }),
        h('h2', { fixed: { m: 'x' }, sweep: { lr: [0.2] } }, { paperIds: ['pA'] }),
      ],
      [paper],
    )
    expect(p!.changedPapers.length).toBe(1)
    const cp = p!.changedPapers[0]
    const newId = p!.unionRecord.id
    expect(cp.hypothesisIds).toContain(newId)
    expect(cp.hypothesisIds).toContain('other')
    expect(cp.hypothesisIds).not.toContain('h1')
    // weight PRESERVED: max(2,5)=5 moved to the survivor, absorbed keys gone
    expect(cp.hypothesisWeights[newId]).toBe(5)
    expect('h1' in cp.hypothesisWeights).toBe(false)
    expect('h2' in cp.hypothesisWeights).toBe(false)
    expect(cp.hypothesisWeights.other).toBe(1)
  })

  it('repoints model + flavor hypothesisIds', () => {
    const model = {
      id: 'm1',
      hypothesisIds: ['h1'],
      flavors: [{ key: 'f', hypothesisIds: ['h2'] }],
    }
    const p = plan(
      [
        h('h1', { fixed: { m: 'x' }, sweep: { lr: [0.1] } }),
        h('h2', { fixed: { m: 'x' }, sweep: { lr: [0.2] } }),
      ],
      [],
      [model],
    )
    expect(p!.changedModels.length).toBe(1)
    const newId = p!.unionRecord.id
    expect(p!.changedModels[0].hypothesisIds).toEqual([newId])
    expect(p!.changedModels[0].flavors[0].hypothesisIds).toEqual([newId])
  })

  it('carries the canonical hypothesis CLAIM label onto the union (so the paper still groups it by claim)', () => {
    const p = plan([
      h('h1', { fixed: { m: 'x' }, sweep: { lr: [0.1] } }, { claim: 'Momentum predicts returns' }),
      h('h2', { fixed: { m: 'x' }, sweep: { lr: [0.2] } }),
    ])
    expect(p!.unionRecord.claim).toBe('Momentum predicts returns')
  })

  it('carries a manual canonical verdict onto the union but NO evidence/transitions', () => {
    const p = plan([
      h(
        'h1',
        { fixed: { m: 'x' }, sweep: { lr: [0.1] } },
        {
          verdictSource: 'manual',
          status: 'disproved',
          verdictNote: 'fails OOS',
          evidence: { matchedKeys: ['r1'] },
          transitions: [{ at: 't', from: 'untested', to: 'disproved' }],
        },
      ),
      h('h2', { fixed: { m: 'x' }, sweep: { lr: [0.2] } }),
    ])
    expect(p!.unionRecord.verdictSource).toBe('manual')
    expect(p!.unionRecord.status).toBe('disproved')
    expect(p!.unionRecord.verdictNote).toBe('fails OOS')
    expect('evidence' in p!.unionRecord).toBe(false)
    expect('transitions' in p!.unionRecord).toBe(false)
  })

  it('returns a conflict skip when members carry conflicting manual verdicts', () => {
    const p = plan([
      h(
        'h1',
        { fixed: { m: 'x' }, sweep: { lr: [0.1] } },
        { verdictSource: 'manual', status: 'proven' },
      ),
      h(
        'h2',
        { fixed: { m: 'x' }, sweep: { lr: [0.2] } },
        { verdictSource: 'manual', status: 'disproved' },
      ),
    ])
    expect(p && (p as any).skipped).toBe('conflict')
  })

  it('union-id collision: merges INTO an existing member that already equals the union (not deleted)', () => {
    const unionSpec = mergeHypothesisSpecs([
      { fixed: { m: 'x' }, sweep: { lr: [0.1] } },
      { fixed: { m: 'x' }, sweep: { lr: [0.2] } },
    ])
    const unionId = hashTrainingConfig(unionSpec as any)
    const p = plan([
      h(unionId, unionSpec, { paperIds: ['pA'] }),
      h('narrow', { fixed: { m: 'x' }, sweep: { lr: [0.1] } }, { paperIds: ['pB'] }),
    ])
    expect(p!.unionRecord.id).toBe(unionId)
    expect(p!.deletedIds).toEqual(['narrow'])
    expect(new Set(p!.unionRecord.paperIds)).toEqual(new Set(['pA', 'pB']))
  })
})

describe('planHypothesisSpecMigration', () => {
  const migrations = [{ match: { algo: 'old' }, set: { algo: 'b' } }]
  const h = (id: string, spec: any, o: any = {}) => ({
    id,
    spec,
    title: id,
    rationale: '',
    status: 'untested',
    verdictSource: 'auto',
    source: 'llm',
    createdAt: '2026-01-01',
    updatedAt: '2026-01-01',
    paperIds: [],
    ...o,
  })
  const idOf = (spec: any) => hashTrainingConfig(spec)
  const run = (
    hyps: any[],
    papers: any[] = [],
    models: any[] = [],
    rules: any = migrations,
    experiments: any[] = [],
  ) =>
    planHypothesisSpecMigration(
      hyps as any,
      papers as any,
      models as any,
      experiments as any,
      rules,
      '2026-07-16T00:00:00Z',
      hashTrainingConfig,
    )

  it('no-ops without migrations', () => {
    expect(
      planHypothesisSpecMigration(
        [h(idOf({ fixed: { algo: 'old' } }), { fixed: { algo: 'old' } })] as any,
        [],
        [],
        [],
        undefined,
        '2026-07-16T00:00:00Z',
        hashTrainingConfig,
      ),
    ).toEqual({ writes: [], deletes: [], changedPapers: [], changedModels: [], changedExperiments: [] })
  })

  it('repoints an experiment linked to a re-keyed hypothesis onto the survivor id', () => {
    const oldId = idOf({ fixed: { algo: 'old' } })
    const experiment = {
      id: 'e1',
      hypothesisId: oldId,
      thesis: 't',
      status: 'completed',
      matrix: {},
      cells: [],
      aggregate: null,
      provenance: { source: 'llm', createdAt: '2026-01-01', updatedAt: '2026-01-01' },
    }
    const p = run([h(oldId, { fixed: { algo: 'old' } })], [], [], migrations, [experiment])
    const newId = idOf({ fixed: { algo: 'b' } })
    expect(p.changedExperiments.map((e: any) => e.id)).toEqual(['e1'])
    expect(p.changedExperiments[0].hypothesisId).toBe(newId)
  })

  it('re-keys a hypothesis pinning a retired value to the migrated-spec id', () => {
    const oldId = idOf({ fixed: { algo: 'old' } })
    const p = run([h(oldId, { fixed: { algo: 'old' } })])
    expect(p.deletes).toEqual([oldId])
    expect(p.writes).toHaveLength(1)
    expect(p.writes[0].spec.fixed).toEqual({ algo: 'b' })
    expect(p.writes[0].id).toBe(idOf({ fixed: { algo: 'b' } }))
    expect(p.writes[0].id).not.toBe(oldId)
  })

  it('leaves an already-current hypothesis untouched', () => {
    const cur = idOf({ fixed: { algo: 'b' } })
    expect(run([h(cur, { fixed: { algo: 'b' } })])).toEqual({
      writes: [],
      deletes: [],
      changedPapers: [],
      changedModels: [],
      changedExperiments: [],
    })
  })

  it('consolidates two hypotheses that migrate to the SAME spec into one survivor', () => {
    const newId = idOf({ fixed: { algo: 'b' } })
    const a = h('A', { fixed: { algo: 'old' } }, { paperIds: ['p1'] }) // migrates → newId
    const bAtNew = h(newId, { fixed: { algo: 'b' } }, { paperIds: ['p2'] }) // already at newId
    const p = run([a, bAtNew])
    expect(p.writes).toHaveLength(1)
    expect(p.writes[0].id).toBe(newId)
    expect(new Set(p.writes[0].paperIds)).toEqual(new Set(['p1', 'p2']))
    expect(p.deletes).toEqual(['A'])
  })

  it('repoints paper hypothesisIds + weights off a deleted id (weights max-merged onto survivor)', () => {
    const newId = idOf({ fixed: { algo: 'b' } })
    const paper = { id: 'p1', hypothesisIds: ['A', newId], hypothesisWeights: { A: 5, [newId]: 2 } }
    const p = run([h('A', { fixed: { algo: 'old' } })], [paper])
    expect(p.changedPapers).toHaveLength(1)
    expect(p.changedPapers[0].hypothesisIds).toEqual([newId])
    expect(p.changedPapers[0].hypothesisWeights![newId]).toBe(5)
    expect('A' in (p.changedPapers[0].hypothesisWeights || {})).toBe(false)
  })

  it('defers a group with an in-flight campaign (would orphan its pending write)', () => {
    const a = h('A', { fixed: { algo: 'old' } }, { campaign: { status: 'running', activityId: 'x' } })
    expect(run([a]).writes).toEqual([])
  })

  it('skips a conflicting manual-verdict pair rather than silently pinning one', () => {
    const p = run([
      h('A', { fixed: { algo: 'old' } }, { verdictSource: 'manual', status: 'proven' }),
      h('B', { fixed: { algo: 'old' } }, { verdictSource: 'manual', status: 'disproved' }),
    ])
    expect(p.writes).toEqual([])
    expect(p.deletes).toEqual([])
  })

  it('drops stale evidence/transitions/campaign on a re-keyed survivor', () => {
    const a = h('A', { fixed: { algo: 'old' } }, {
      evidence: { matchedKeys: ['k'], status: 'proven' },
      transitions: [{ at: 't', from: 'untested', to: 'proven' }],
      campaign: { status: 'completed', activityId: 'x' },
    })
    const p = run([a])
    expect('evidence' in p.writes[0]).toBe(false)
    expect('transitions' in p.writes[0]).toBe(false)
    expect('campaign' in p.writes[0]).toBe(false)
  })

  it('CHAINED rules: a paper weight hops exactly ONE remap step (ids and weights stay in agreement)', () => {
    // Chained rule set: a→b and b→c (applyMigrationRules applies ONE hop per record). H1(a)→id B,
    // H2(b)→id C, H3(c) already at C. remap = {A→B, B→C}. A paper weighting A must land its weight on B
    // (where A's content now lives) — NOT double-hop through B→C onto a hypothesis it doesn't link.
    const chain = [
      { match: { algo: 'a' }, set: { algo: 'b' } },
      { match: { algo: 'b' }, set: { algo: 'c' } },
    ]
    const idA = idOf({ fixed: { algo: 'a' } })
    const idB = idOf({ fixed: { algo: 'b' } })
    const idC = idOf({ fixed: { algo: 'c' } })
    const paper = { id: 'p1', hypothesisIds: [idA], hypothesisWeights: { [idA]: 4 } }
    const p = run(
      [h(idA, { fixed: { algo: 'a' } }), h(idB, { fixed: { algo: 'b' } }), h(idC, { fixed: { algo: 'c' } })],
      [paper],
      [],
      chain,
    )
    // B is both written (survivor of A's group) and absorbed into C — the record guard keeps B alive.
    expect(p.deletes).toEqual([idA])
    const cp = p.changedPapers[0]
    expect(cp.hypothesisIds).toEqual([idB])
    expect(cp.hypothesisWeights![idB]).toBe(4)
    expect(idC in (cp.hypothesisWeights || {})).toBe(false)
  })
})

describe('coerceHypothesisCoverage (scrutinous weigh: weights + coverage gaps)', () => {
  const ids = ['h1', 'h2']
  it('extracts weights, uncovered claims, and the claim->hypotheses map (id or 1-based index)', () => {
    const out = coerceHypothesisCoverage(
      {
        claims: [
          { claim: 'Momentum predicts returns', hypothesisIds: ['h1'] },
          { claim: 'Vol-sizing lifts Sharpe', hypothesisIds: [2] },
        ],
        weights: [
          { id: 'h1', weight: 5, reason: 'central' },
          { index: 2, weight: 2, reason: 'supporting' },
        ],
        uncoveredClaims: ['Transaction costs are negligible', '  '],
      },
      ids,
    )
    expect(out.weights).toEqual([
      { id: 'h1', weight: 5, reason: 'central' },
      { id: 'h2', weight: 2, reason: 'supporting' },
    ])
    expect(out.uncoveredClaims).toEqual(['Transaction costs are negligible'])
    expect(out.coverageByClaim).toEqual([
      { claim: 'Momentum predicts returns', hypothesisIds: ['h1'] },
      { claim: 'Vol-sizing lifts Sharpe', hypothesisIds: ['h2'] },
    ])
  })
  it('is legacy-tolerant: a bare weights array yields weights with UNKNOWN coverage (not "no gaps")', () => {
    const out = coerceHypothesisCoverage([{ id: 'h1', weight: 3 }], ids)
    expect(out.weights).toEqual([{ id: 'h1', weight: 3, reason: '' }])
    expect(out.uncoveredClaims).toEqual([])
    expect(out.coverageByClaim).toEqual([])
  })
})

describe('estimateRemainingCampaignSeconds (remaining wall-clock from real per-run durations)', () => {
  it('returns undefined with no completed runs yet, or nothing remaining', () => {
    expect(
      estimateRemainingCampaignSeconds({ durationsMs: [], remaining: 5, concurrency: 1 }),
    ).toBeUndefined()
    expect(
      estimateRemainingCampaignSeconds({ durationsMs: [10000], remaining: 0, concurrency: 1 }),
    ).toBeUndefined()
  })
  it('serial (concurrency 1): avg per-run × remaining', () => {
    expect(
      estimateRemainingCampaignSeconds({ durationsMs: [10000], remaining: 5, concurrency: 1 }),
    ).toBe(50)
    expect(
      estimateRemainingCampaignSeconds({
        durationsMs: [10000, 20000],
        remaining: 2,
        concurrency: 1,
      }),
    ).toBe(30)
  })
  it('concurrency divides the wall-clock into waves (ceil)', () => {
    expect(
      estimateRemainingCampaignSeconds({ durationsMs: [10000], remaining: 5, concurrency: 5 }),
    ).toBe(10)
    expect(
      estimateRemainingCampaignSeconds({ durationsMs: [10000], remaining: 6, concurrency: 5 }),
    ).toBe(20)
    // concurrency never exceeds remaining
    expect(
      estimateRemainingCampaignSeconds({ durationsMs: [10000], remaining: 3, concurrency: 8 }),
    ).toBe(10)
  })
  it('ignores non-finite / non-positive durations (skipped runs reporting 0)', () => {
    expect(
      estimateRemainingCampaignSeconds({
        durationsMs: [0, NaN, -5, 10000],
        remaining: 2,
        concurrency: 1,
      }),
    ).toBe(20)
  })
  it('tolerates a missing durationsMs array (no estimate yet)', () => {
    expect(
      estimateRemainingCampaignSeconds({
        durationsMs: undefined as never,
        remaining: 5,
        concurrency: 1,
      }),
    ).toBeUndefined()
  })
  it('treats a zero / invalid concurrency as serial (1)', () => {
    // floor(0) || 1 → 1 wave-width, so 5 remaining × 10s = 50s.
    expect(
      estimateRemainingCampaignSeconds({ durationsMs: [10000], remaining: 5, concurrency: 0 }),
    ).toBe(50)
  })
})

describe('buildPaperResearchGoal', () => {
  const withFamilies = (overrides: Partial<TrainerManifest> = {}) =>
    manifest({
      name: 'BlackSwan',
      description: 'Trains an RL agent to trade BTCUSDT crypto and beat buy-and-hold.',
      objective: { name: 'traded_return', direction: 'max' },
      levers: {
        model_name: { type: 'choice', scope: 'model', choices: ['ppo-custom', 'reppo-custom'] },
        lr: { type: 'number' },
      },
      ...overrides,
    })

  it('composes a discovery goal from name, description, objective, and model families', () => {
    const goal = buildPaperResearchGoal(withFamilies())
    expect(goal).toContain('BlackSwan')
    expect(goal).toContain('beat buy-and-hold')
    expect(goal).toContain('traded_return')
    expect(goal).toMatch(/max is better/)
    expect(goal).toContain('ppo-custom')
    expect(goal).toContain('reppo-custom')
    // steers toward scholarly sources, single line
    expect(goal.toLowerCase()).toContain('research paper')
    expect(goal).not.toContain('\n')
  })

  it('omits the description clause when absent, without leaving dangling separators', () => {
    const goal = buildPaperResearchGoal(withFamilies({ description: undefined }))
    expect(goal).toContain('BlackSwan')
    expect(goal).toContain('traded_return')
    expect(goal).not.toContain('undefined')
    expect(goal).not.toMatch(/\s{2,}/)
  })

  it('omits the model-families clause when there is no identity lever / no choices', () => {
    const goal = buildPaperResearchGoal(
      manifest({ name: 'Demo', levers: { lr: { type: 'number' }, steps: { type: 'number' } } }),
    )
    expect(goal.toLowerCase()).not.toContain('model families')
    expect(goal).not.toContain('undefined')
  })

  it('appends notes verbatim when provided, and adds no trailing whitespace when absent', () => {
    expect(
      buildPaperResearchGoal(withFamilies(), { notes: 'focus on transformer policies' }),
    ).toContain('focus on transformer policies')
    expect(buildPaperResearchGoal(withFamilies())).toBe(
      buildPaperResearchGoal(withFamilies()).trim(),
    )
  })

  it('renders the objective direction (min is better) rather than hardcoding max', () => {
    const goal = buildPaperResearchGoal(
      withFamilies({ objective: { name: 'drawdown', direction: 'min' } }),
    )
    expect(goal).toMatch(/min is better/)
  })
})

describe('normalizeResearchUrl', () => {
  it('lowercases host and strips trailing slash, fragment, and utm params', () => {
    expect(normalizeResearchUrl('https://Example.COM/Path/#section')).toBe(
      'https://example.com/Path',
    )
    expect(normalizeResearchUrl('https://example.com/p/?utm_source=x&utm_medium=y')).toBe(
      'https://example.com/p',
    )
  })

  it('canonicalizes arxiv abs/pdf/versioned/.pdf variants to one key', () => {
    const canon = normalizeResearchUrl('https://arxiv.org/abs/2401.12345')
    expect(normalizeResearchUrl('https://arxiv.org/pdf/2401.12345')).toBe(canon)
    expect(normalizeResearchUrl('https://arxiv.org/pdf/2401.12345v2')).toBe(canon)
    expect(normalizeResearchUrl('https://arxiv.org/pdf/2401.12345.pdf')).toBe(canon)
    expect(normalizeResearchUrl('http://arxiv.org/abs/2401.12345v3')).toBe(canon)
  })

  it('leaves an unparseable string as a trimmed lowercase fallback', () => {
    expect(normalizeResearchUrl('  not a url  ')).toBe('not a url')
  })
})

describe('coercePaperCandidates', () => {
  it('maps well-formed records, accepting either name or title', () => {
    const out = coercePaperCandidates([
      { name: 'Deep RL Trading', url: 'https://arxiv.org/abs/1', hints: ['pdf'] },
      { title: 'Momentum Works', url: 'https://arxiv.org/abs/2' },
    ])
    expect(out).toEqual([
      { title: 'Deep RL Trading', url: 'https://arxiv.org/abs/1', hints: ['pdf'] },
      { title: 'Momentum Works', url: 'https://arxiv.org/abs/2' },
    ])
  })

  it('drops records missing a url or with a non-http url', () => {
    expect(
      coercePaperCandidates([
        { title: 'No url' },
        { title: 'Bad scheme', url: 'ftp://x/y' },
        { title: 'Ok', url: 'https://ok/1' },
      ]),
    ).toEqual([{ title: 'Ok', url: 'https://ok/1' }])
  })

  it('drops records with an empty/whitespace title', () => {
    expect(
      coercePaperCandidates([
        { title: '   ', url: 'https://x/1' },
        { name: '', url: 'https://x/2' },
      ]),
    ).toEqual([])
  })

  it('returns [] for a non-array or empty input', () => {
    expect(coercePaperCandidates([])).toEqual([])
    expect(coercePaperCandidates(undefined as unknown as unknown[])).toEqual([])
  })
})

describe('dedupePaperCandidates', () => {
  const c = (title: string, url: string) => ({ title, url })

  it('drops a candidate whose url matches an existing paper', () => {
    expect(
      dedupePaperCandidates(
        [c('A', 'https://arxiv.org/abs/1'), c('B', 'https://arxiv.org/abs/2')],
        [{ url: 'https://arxiv.org/pdf/1v2', title: 'Whatever' }],
      ),
    ).toEqual([c('B', 'https://arxiv.org/abs/2')])
  })

  it('drops a candidate whose title matches an existing one (case/space-insensitive)', () => {
    expect(
      dedupePaperCandidates(
        [c('Attention Is All You Need', 'https://x/new')],
        [{ url: 'https://other', title: '  attention is all you need ' }],
      ),
    ).toEqual([])
  })

  it('drops an intra-batch duplicate by normalized url', () => {
    expect(
      dedupePaperCandidates(
        [c('A', 'https://arxiv.org/abs/1'), c('A2', 'https://arxiv.org/pdf/1')],
        [],
      ),
    ).toEqual([c('A', 'https://arxiv.org/abs/1')])
  })

  it('returns all candidates unchanged (order preserved) when nothing matches', () => {
    const list = [c('A', 'https://x/1'), c('B', 'https://x/2')]
    expect(dedupePaperCandidates(list, [])).toEqual(list)
  })

  it('falls back to url-only matching when a candidate title is empty', () => {
    expect(
      dedupePaperCandidates(
        [
          { title: '', url: 'https://x/1' },
          { title: '', url: 'https://x/2' },
        ],
        [],
      ),
    ).toEqual([
      { title: '', url: 'https://x/1' },
      { title: '', url: 'https://x/2' },
    ])
  })
})

describe('paperRelevanceClaim', () => {
  it('grounds the claim in the project DOMAIN (name + description), not just the metric', () => {
    const claim = paperRelevanceClaim(
      { title: 'Deep RL for Trading', url: 'https://arxiv.org/abs/1' },
      manifest({
        name: 'BlackSwan',
        description: 'RL agent that trades BTCUSDT crypto.',
        objective: { name: 'traded_return', direction: 'max' },
      }),
    )
    expect(claim).toContain('Deep RL for Trading')
    expect(claim).toContain('BlackSwan')
    expect(claim).toContain('trades BTCUSDT crypto')
    expect(claim.toLowerCase()).toContain('research paper')
  })
})

describe('isPaperVerdictAdmitted', () => {
  const v = (status: string, confidence: number) =>
    ({ status, confidence, evidence: [], methodology: '', sourcesAttempted: [] }) as Parameters<
      typeof isPaperVerdictAdmitted
    >[0]

  it('admits confirmed/implied at or above the confidence floor', () => {
    expect(isPaperVerdictAdmitted(v('confirmed', 0.9), 0.5)).toBe(true)
    expect(isPaperVerdictAdmitted(v('implied', 0.5), 0.5)).toBe(true)
  })

  it('rejects unverifiable/refuted regardless of confidence', () => {
    expect(isPaperVerdictAdmitted(v('unverifiable', 0.99), 0.5)).toBe(false)
    expect(isPaperVerdictAdmitted(v('refuted', 0.99), 0.5)).toBe(false)
  })

  it('rejects a confirmed verdict below the confidence floor', () => {
    expect(isPaperVerdictAdmitted(v('confirmed', 0.49), 0.5)).toBe(false)
  })
})

describe('paperHostAffinity', () => {
  it('scores known paper venues highest', () => {
    expect(paperHostAffinity('https://arxiv.org/abs/2401.1')).toBeGreaterThan(
      paperHostAffinity('https://medium.com/some-post'),
    )
    expect(paperHostAffinity('https://openreview.net/forum?id=x')).toBeGreaterThan(0)
    expect(paperHostAffinity('https://proceedings.mlr.press/v1/x.html')).toBeGreaterThan(0)
    expect(paperHostAffinity('https://cs.stanford.edu/~x/paper')).toBeGreaterThan(0)
  })

  it('scores a bare .pdf link above a generic html page but below a paper venue', () => {
    const pdf = paperHostAffinity('https://someblog.com/report.pdf')
    const html = paperHostAffinity('https://someblog.com/report')
    const venue = paperHostAffinity('https://arxiv.org/pdf/2401.1')
    expect(pdf).toBeGreaterThan(html)
    expect(venue).toBeGreaterThan(pdf)
  })

  it('scores a generic host at zero and never throws on a bad url', () => {
    expect(paperHostAffinity('https://example.com/x')).toBe(0)
    expect(paperHostAffinity('not a url')).toBe(0)
  })
})

describe('rankPaperCandidates', () => {
  const c = (title: string, url: string) => ({ title, url })

  it('orders paper-venue candidates before generic ones', () => {
    const out = rankPaperCandidates([
      c('Blog', 'https://medium.com/p'),
      c('Paper', 'https://arxiv.org/abs/1'),
    ])
    expect(out.map((x) => x.title)).toEqual(['Paper', 'Blog'])
  })

  it('is stable — preserves input order within the same affinity tier', () => {
    const out = rankPaperCandidates([
      c('A', 'https://arxiv.org/abs/1'),
      c('B', 'https://openreview.net/forum?id=2'),
      c('C', 'https://blog.com/a'),
      c('D', 'https://blog.com/b'),
    ])
    expect(out.map((x) => x.title)).toEqual(['A', 'B', 'C', 'D'])
  })

  it('returns an all-generic list unchanged and never drops a candidate', () => {
    const list = [c('A', 'https://x.com/1'), c('B', 'https://y.com/2')]
    expect(rankPaperCandidates(list)).toEqual(list)
    expect(rankPaperCandidates([...list])).toHaveLength(2)
  })
})

describe('capRunSummaryForStorage', () => {
  const base = (extra: Partial<TrainingRunSummary> = {}): TrainingRunSummary =>
    ({ objective: 1, ...extra }) as TrainingRunSummary

  it('leaves a run whose series + trace are within the caps untouched (same reference)', () => {
    const summary = base({
      series: { equity: [1, 2, 3] },
      artifacts: { decisionTrace: { steps: [{ step: 0, action: 'hold' }] } },
    })
    expect(capRunSummaryForStorage(summary)).toBe(summary)
  })

  it('downsamples an over-long series metric to maxSeriesPoints, preserving the first and last point', () => {
    const equity = Array.from({ length: MAX_SERIES_POINTS + 5000 }, (_, i) => i)
    const out = capRunSummaryForStorage(base({ series: { equity } }))
    expect(out.series!.equity).toHaveLength(MAX_SERIES_POINTS)
    expect(out.series!.equity[0]).toBe(0)
    expect(out.series!.equity[out.series!.equity.length - 1]).toBe(equity.length - 1)
  })

  it('only downsamples the series arrays that exceed the cap, leaving short siblings intact', () => {
    const long = Array.from({ length: MAX_SERIES_POINTS + 1 }, (_, i) => i)
    const out = capRunSummaryForStorage(base({ series: { long, short: [1, 2, 3] } }))
    expect(out.series!.long).toHaveLength(MAX_SERIES_POINTS)
    expect(out.series!.short).toEqual([1, 2, 3])
  })

  it('truncates a runaway decisionTrace.steps to a contiguous prefix and records the pre-cap totalSteps', () => {
    const steps = Array.from({ length: MAX_DECISION_TRACE_STEPS + 100 }, (_, i) => ({ step: i, action: 'hold' }))
    const out = capRunSummaryForStorage(base({ artifacts: { decisionTrace: { steps } } }))
    const trace = (out.artifacts as { decisionTrace: { steps: unknown[]; totalSteps: number } }).decisionTrace
    expect(trace.steps).toHaveLength(MAX_DECISION_TRACE_STEPS)
    expect((trace.steps[0] as { step: number }).step).toBe(0) // contiguous prefix (adjacency preserved)
    expect(trace.totalSteps).toBe(steps.length)
  })

  it('preserves a producer-supplied totalSteps rather than overwriting it with the kept count', () => {
    const steps = Array.from({ length: MAX_DECISION_TRACE_STEPS + 1 }, (_, i) => ({ step: i, action: 'hold' }))
    const out = capRunSummaryForStorage(base({ artifacts: { decisionTrace: { steps, totalSteps: 999999 } } }))
    const trace = (out.artifacts as { decisionTrace: { totalSteps: number } }).decisionTrace
    expect(trace.totalSteps).toBe(999999)
  })

  it('does not mutate the input summary', () => {
    const equity = Array.from({ length: MAX_SERIES_POINTS + 1 }, (_, i) => i)
    const summary = base({ series: { equity } })
    capRunSummaryForStorage(summary)
    expect(summary.series!.equity).toHaveLength(MAX_SERIES_POINTS + 1)
  })

  it('preserves a trace-level attentionMatrix through the step-cap (grid is source-capped, never dropped here)', () => {
    const steps = Array.from({ length: MAX_DECISION_TRACE_STEPS + 10 }, (_, i) => ({ step: i, action: 'hold' }))
    const attentionMatrix = {
      rows: ['t0', 't1'],
      cols: ['c0', 'c1'],
      grid: [
        [0.1, 0.2],
        [0.3, 0.4],
      ],
      method: 'attention-weights',
    }
    const out = capRunSummaryForStorage(base({ artifacts: { decisionTrace: { steps, attentionMatrix } } }))
    const trace = (out.artifacts as { decisionTrace: { steps: unknown[]; attentionMatrix: unknown } }).decisionTrace
    expect(trace.steps).toHaveLength(MAX_DECISION_TRACE_STEPS)
    expect(trace.attentionMatrix).toEqual(attentionMatrix)
  })
})

describe('plannedMatrixItemCount', () => {
  it('is 1 for a bare spec (defaults only)', () => {
    expect(plannedMatrixItemCount({})).toBe(1)
  })

  it('multiplies each sweep lever value count (cartesian product)', () => {
    expect(plannedMatrixItemCount({ sweep: { a: [1, 2, 3], b: [1, 2] } })).toBe(6)
  })

  it('multiplies by seeds, datasets, environments, and compare values', () => {
    expect(
      plannedMatrixItemCount({
        sweep: { a: [1, 2] },
        seeds: [1, 2, 3],
        datasets: [{ d: 1 }, { d: 2 }],
        environments: [{ e: 1 }],
        compare: { lever: 'c', values: [1, 2, 3, 4] },
      }),
    ).toBe(2 * 4 * 2 * 1 * 3)
  })

  it('counts explicit configs as themselves and suppresses the swept base', () => {
    expect(
      plannedMatrixItemCount({ sweep: { a: [1, 2] }, configs: [{ config: {} }, { config: {} }] }),
    ).toBe(2)
  })

  it('reports a cartesian blow-up WITHOUT materializing it (the OOM guard)', () => {
    const sweep: Record<string, number[]> = {}
    for (let i = 0; i < 10; i += 1) sweep[`l${i}`] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    expect(plannedMatrixItemCount({ sweep })).toBe(1e10) // computed, never built
  })

  it('treats a non-array sweep value as a zero multiplier (collapses the product to 0)', () => {
    expect(plannedMatrixItemCount({ sweep: { a: 'oops' as never } })).toBe(0)
  })
})

describe('computeScorecard', () => {
  const objMax = { objective: { name: 'ret', direction: 'max' as const } }
  const objMin = { objective: { name: 'rmse', direction: 'min' as const } }

  it('with no gates and no fitness, accepts and ranks by the objective (the collapse)', () => {
    const card = computeScorecard(objMax, { objective: 42 })
    expect(card.gates).toEqual([])
    expect(card.accepted).toBe(true)
    expect(card.fitness).toEqual([{ metric: 'objective', direction: 'max', value: 42 }])
  })

  it('default fitness inherits the objective direction (min)', () => {
    const card = computeScorecard(objMin, { objective: 0.3 })
    expect(card.fitness).toEqual([{ metric: 'objective', direction: 'min', value: 0.3 }])
  })

  it('resolves a fitness metric from summary.metrics by key', () => {
    const card = computeScorecard(
      { ...objMax, fitness: [{ metric: 'oos_sharpe', direction: 'max' }] },
      { objective: 42, metrics: { oos_sharpe: 1.4 } },
    )
    expect(card.fitness).toEqual([{ metric: 'oos_sharpe', direction: 'max', value: 1.4 }])
  })

  it('carries a multi-objective (Pareto) fitness vector in declared order', () => {
    const card = computeScorecard(
      {
        ...objMax,
        fitness: [
          { metric: 'oos_sharpe', direction: 'max' },
          { metric: 'max_drawdown_pct', direction: 'max' },
        ],
      },
      { objective: 42, metrics: { oos_sharpe: 1.4, max_drawdown_pct: -12 } },
    )
    expect(card.fitness).toEqual([
      { metric: 'oos_sharpe', direction: 'max', value: 1.4 },
      { metric: 'max_drawdown_pct', direction: 'max', value: -12 },
    ])
  })

  it('a missing fitness metric resolves to NaN', () => {
    const card = computeScorecard({ ...objMax, fitness: [{ metric: 'gone', direction: 'max' }] }, { objective: 42 })
    expect(Number.isNaN(card.fitness[0].value)).toBe(true)
  })

  it('a passing literal gate accepts the run', () => {
    const card = computeScorecard(
      { ...objMax, gates: [{ metric: 'oos_return_pct', op: '>', value: 0 }] },
      { objective: 42, metrics: { oos_return_pct: 5 } },
    )
    expect(card.gates).toHaveLength(1)
    expect(card.gates[0]).toMatchObject({ metric: 'oos_return_pct', op: '>', bound: 0, actual: 5, pass: true })
    expect(card.accepted).toBe(true)
  })

  it('a failing literal gate rejects the run', () => {
    const card = computeScorecard(
      { ...objMax, gates: [{ metric: 'oos_return_pct', op: '>', value: 0 }] },
      { objective: 42, metrics: { oos_return_pct: -3 } },
    )
    expect(card.gates[0].pass).toBe(false)
    expect(card.accepted).toBe(false)
  })

  it('resolves a metric-vs-metric gate bound (beat buy-and-hold)', () => {
    const beatHold = { metric: 'oos_return_pct', op: '>' as const, value: { metric: 'hold_return_pct' } }
    const pass = computeScorecard({ ...objMax, gates: [beatHold] }, { objective: 1, metrics: { oos_return_pct: 10, hold_return_pct: 4 } })
    expect(pass.gates[0]).toMatchObject({ bound: 4, actual: 10, pass: true })
    const fail = computeScorecard({ ...objMax, gates: [beatHold] }, { objective: 1, metrics: { oos_return_pct: 2, hold_return_pct: 4 } })
    expect(fail.gates[0]).toMatchObject({ bound: 4, actual: 2, pass: false })
  })

  it('a gate on an ABSENT metric is SKIPPED (not applicable) and does not by itself REJECT the run', () => {
    // A run that predates a metric (the key isn't in `metrics` at all) can't be judged on that gate — skip it
    // rather than auto-fail. But when it's the ONLY gate, there is nothing we CAN evaluate, so the run is NOT
    // "accepted" (that would vacuously accept every failed/old run and make the accepted filter a no-op).
    const card = computeScorecard({ ...objMax, gates: [{ metric: 'gone', op: '>', value: 0 }] }, { objective: 42, metrics: { other: 1 } })
    expect(card.gates[0].applicable).toBe(false)
    expect(card.gates[0].pass).toBe(false)
    expect(card.accepted).toBe(false) // no applicable gate ⇒ can't verify ⇒ NOT accepted (no vacuous accept)
  })

  it('a skipped gate alongside a PASSING applicable gate still accepts (judged on what CAN be evaluated)', () => {
    const gates = [
      { metric: 'gone', op: '>' as const, value: 0 }, // absent ⇒ skipped
      { metric: 'ret', op: '>' as const, value: 0 }, // present + passes
    ]
    const card = computeScorecard({ ...objMax, gates }, { objective: 1, metrics: { ret: 5 } })
    expect(card.gates[0].applicable).toBe(false)
    expect(card.gates[1]).toMatchObject({ applicable: true, pass: true })
    expect(card.accepted).toBe(true)
  })

  it('a FAILED or degenerate run is never accepted, even if its applicable gates pass', () => {
    const gates = [{ metric: 'ret', op: '>' as const, value: 0 }]
    const metrics = { ret: 5 }
    expect(computeScorecard({ ...objMax, gates }, { objective: 1, metrics }).accepted).toBe(true)
    expect(computeScorecard({ ...objMax, gates }, { objective: 1, metrics, status: 'failed' }).accepted).toBe(false)
    expect(computeScorecard({ ...objMax, gates }, { objective: 1, metrics, status: 'invalid' }).accepted).toBe(false)
    expect(
      computeScorecard({ ...objMax, gates }, { objective: 1, metrics, health: { status: 'degenerate' } }).accepted,
    ).toBe(false)
  })

  it('a PRESENT but non-finite gate metric FAILS (a measured-garbage value is not skipped)', () => {
    const card = computeScorecard({ ...objMax, gates: [{ metric: 'm', op: '>', value: 0 }] }, { objective: 1, metrics: { m: NaN } })
    expect(card.gates[0].applicable).toBe(true)
    expect(card.gates[0].pass).toBe(false)
    expect(card.accepted).toBe(false)
  })

  it('rejects on a FAILING gate even when another gate is skipped', () => {
    const gates = [
      { metric: 'gone', op: '>' as const, value: 0 }, // absent ⇒ skipped
      { metric: 'ret', op: '>' as const, value: 0 }, // present + fails
    ]
    const card = computeScorecard({ ...objMax, gates }, { objective: 1, metrics: { ret: -5 } })
    expect(card.gates[0].applicable).toBe(false)
    expect(card.gates[1]).toMatchObject({ applicable: true, pass: false })
    expect(card.accepted).toBe(false)
  })

  it('a gate whose REFERENCED metric is missing fails', () => {
    const card = computeScorecard(
      { ...objMax, gates: [{ metric: 'oos_return_pct', op: '>', value: { metric: 'gone' } }] },
      { objective: 1, metrics: { oos_return_pct: 10 } },
    )
    expect(Number.isNaN(card.gates[0].bound)).toBe(true)
    expect(card.gates[0].pass).toBe(false)
  })

  it('accepts only when EVERY gate passes', () => {
    const gates = [
      { metric: 'a', op: '>' as const, value: 0 },
      { metric: 'b', op: '>' as const, value: 0 },
    ]
    expect(computeScorecard({ ...objMax, gates }, { objective: 1, metrics: { a: 1, b: 1 } }).accepted).toBe(true)
    expect(computeScorecard({ ...objMax, gates }, { objective: 1, metrics: { a: 1, b: -1 } }).accepted).toBe(false)
  })

  it('a gate can read the objective via the `objective` key', () => {
    const card = computeScorecard({ ...objMax, gates: [{ metric: 'objective', op: '>=', value: 40 }] }, { objective: 42 })
    expect(card.gates[0]).toMatchObject({ actual: 42, bound: 40, pass: true })
  })

  it.each([
    ['>', 5, 3, true],
    ['>', 3, 5, false],
    ['>=', 5, 5, true],
    ['>=', 4, 5, false],
    ['<', 3, 5, true],
    ['<', 5, 3, false],
    ['<=', 5, 5, true],
    ['<=', 6, 5, false],
    ['==', 5, 5, true],
    ['==', 5, 6, false],
    ['!=', 5, 6, true],
    ['!=', 5, 5, false],
  ] as const)('evaluates %s (actual=%d bound=%d) as %s', (op, actual, bound, expected) => {
    const card = computeScorecard({ ...objMax, gates: [{ metric: 'm', op, value: bound }] }, { objective: 1, metrics: { m: actual } })
    expect(card.gates[0].pass).toBe(expected)
  })

  it('a PRESENT NaN actual fails even the != operator', () => {
    const card = computeScorecard({ ...objMax, gates: [{ metric: 'm', op: '!=', value: 5 }] }, { objective: 1, metrics: { m: NaN } })
    expect(card.gates[0]).toMatchObject({ applicable: true, pass: false })
  })

  it('renders a default label from metric/op/literal when none is given', () => {
    const card = computeScorecard({ ...objMax, gates: [{ metric: 'oos_return_pct', op: '>', value: 0 }] }, { objective: 1, metrics: { oos_return_pct: 1 } })
    expect(card.gates[0].label).toBe('oos_return_pct > 0')
  })

  it('renders a default label naming the referenced metric', () => {
    const card = computeScorecard(
      { ...objMax, gates: [{ metric: 'oos_return_pct', op: '>', value: { metric: 'hold_return_pct' } }] },
      { objective: 1, metrics: { oos_return_pct: 5, hold_return_pct: 4 } },
    )
    expect(card.gates[0].label).toBe('oos_return_pct > hold_return_pct')
  })

  it('preserves a custom gate label', () => {
    const card = computeScorecard({ ...objMax, gates: [{ metric: 'm', op: '>', value: 0, label: 'is profitable' }] }, { objective: 1, metrics: { m: 1 } })
    expect(card.gates[0].label).toBe('is profitable')
  })

  it('treats a non-finite objective as NaN in the default fitness', () => {
    const card = computeScorecard(objMax, { objective: Infinity })
    expect(Number.isNaN(card.fitness[0].value)).toBe(true)
  })
})

describe('selectActiveScorecard', () => {
  const manifest = {
    objective: { name: 'ret', direction: 'max' as const },
    gates: [{ metric: 'a', op: '>' as const, value: 0 }],
    fitness: [{ metric: 'a', direction: 'max' as const }],
  }
  const cardX = { id: 'x', gates: [{ metric: 'b', op: '>' as const, value: 1 }], fitness: [{ metric: 'b', direction: 'min' as const }] }
  const cardY = { id: 'y', gates: [{ metric: 'c', op: '<' as const, value: 2 }], fitness: [{ metric: 'c', direction: 'max' as const }] }

  it('falls back to the manifest gates/fitness when there are no cards', () => {
    expect(selectActiveScorecard(manifest, [], null)).toEqual({
      objective: manifest.objective,
      gates: manifest.gates,
      fitness: manifest.fitness,
    })
  })

  it('returns the active card by id (grafting the manifest objective)', () => {
    const r = selectActiveScorecard(manifest, [cardX, cardY], 'y')
    expect(r).toEqual({ objective: manifest.objective, gates: cardY.gates, fitness: cardY.fitness })
  })

  it('falls back to the FIRST card when the active id is missing/unknown', () => {
    expect(selectActiveScorecard(manifest, [cardX, cardY], null).gates).toEqual(cardX.gates)
    expect(selectActiveScorecard(manifest, [cardX, cardY], 'gone').gates).toEqual(cardX.gates)
  })

  it('ignores card entries without an id', () => {
    expect(selectActiveScorecard(manifest, [{ gates: [], fitness: [] } as any, cardY], 'y').gates).toEqual(cardY.gates)
  })

  it('always keeps the manifest objective (cards carry no objective)', () => {
    expect(selectActiveScorecard(manifest, [cardX], 'x').objective).toEqual(manifest.objective)
  })

  it('defaults gates/fitness to empty for a card that declares neither', () => {
    const r = selectActiveScorecard(manifest, [{ id: 'bare' }], 'bare')
    expect(r.gates).toEqual([])
    expect(r.fitness).toEqual([])
  })
})

describe('deriveBackfillMetric', () => {
  const spec = {
    name: 'trades_per_day',
    ratePerDay: { count: 'n_trades', bars: 'oos_n_obs', barDaysLever: 'timeframe', barDays: { '1d': 1, '1h': 1 / 24 } },
  }

  it('derives a per-day rate as count / (bars × bar-days) at a daily step', () => {
    expect(deriveBackfillMetric(spec, { metrics: { n_trades: 10, oos_n_obs: 200 }, config: { timeframe: '1d' } })).toBeCloseTo(0.05)
  })

  it('scales by the timeframe bar-days (hourly step spans 1/24 day per bar)', () => {
    expect(deriveBackfillMetric(spec, { metrics: { n_trades: 10, oos_n_obs: 240 }, config: { timeframe: '1h' } })).toBeCloseTo(10 / (240 / 24))
  })

  it('returns undefined when a required input is missing', () => {
    expect(deriveBackfillMetric(spec, { metrics: { n_trades: 10 }, config: { timeframe: '1d' } })).toBeUndefined()
    expect(deriveBackfillMetric(spec, { metrics: { n_trades: 10, oos_n_obs: 200 }, config: {} })).toBeUndefined()
    expect(deriveBackfillMetric(spec, { metrics: { n_trades: 10, oos_n_obs: 200 }, config: { timeframe: '5m' } })).toBeUndefined()
  })

  it('returns 0 when there are no test bars (never divides by zero)', () => {
    expect(deriveBackfillMetric(spec, { metrics: { n_trades: 10, oos_n_obs: 0 }, config: { timeframe: '1d' } })).toBe(0)
  })

  it('resolves the bars count from the run dataset block when absent from metrics', () => {
    // The ~20k historical corpus predates oos_n_obs (a metric) but carries dataset.candles (bar count) — the
    // rate must derive from the dataset block so pre-existing runs backfill instead of staying "not recorded".
    const candleSpec = {
      name: 'trades_per_day',
      ratePerDay: { count: 'n_trades', bars: 'candles', barDaysLever: 'timeframe', barDays: { '1d': 1, '1h': 1 / 24 } },
    }
    expect(
      deriveBackfillMetric(candleSpec, { metrics: { n_trades: 10 }, dataset: { candles: 200 }, config: { timeframe: '1d' } }),
    ).toBeCloseTo(0.05)
  })

  it('prefers a metrics value over the dataset block when both carry the bars key', () => {
    const dualSpec = {
      name: 'trades_per_day',
      ratePerDay: { count: 'n_trades', bars: 'n_obs', barDaysLever: 'timeframe', barDays: { '1d': 1 } },
    }
    expect(
      deriveBackfillMetric(dualSpec, { metrics: { n_trades: 10, n_obs: 100 }, dataset: { n_obs: 999 }, config: { timeframe: '1d' } }),
    ).toBeCloseTo(0.1)
  })

  it('returns undefined for a spec with no supported derivation', () => {
    expect(deriveBackfillMetric({ name: 'x' }, { metrics: {}, config: {} })).toBeUndefined()
  })
})

describe('primaryFitnessCriterion', () => {
  it('falls back to the objective when no fitness is declared', () => {
    expect(primaryFitnessCriterion({ objective: { name: 'ret', direction: 'max' } })).toEqual({
      key: 'objective',
      direction: 'max',
    })
  })

  it('preserves the objective direction in the fallback (min)', () => {
    expect(primaryFitnessCriterion({ objective: { name: 'rmse', direction: 'min' } })).toEqual({
      key: 'objective',
      direction: 'min',
    })
  })

  it('uses the FIRST fitness objective as the ranking criterion', () => {
    expect(
      primaryFitnessCriterion({
        objective: { name: 'traded_return', direction: 'max' },
        fitness: [
          { metric: 'oos_sharpe', direction: 'max' },
          { metric: 'max_drawdown_pct', direction: 'max' },
        ],
      }),
    ).toEqual({ key: 'oos_sharpe', direction: 'max' })
  })

  it('honours a min-direction primary fitness objective', () => {
    expect(
      primaryFitnessCriterion({
        objective: { name: 'ret', direction: 'max' },
        fitness: [{ metric: 'max_drawdown_pct', direction: 'min' }],
      }),
    ).toEqual({ key: 'max_drawdown_pct', direction: 'min' })
  })
})

describe('scorecardRankValue', () => {
  it('returns the primary fitness value for a max objective', () => {
    expect(scorecardRankValue({ gates: [], accepted: true, fitness: [{ metric: 'x', direction: 'max', value: 7 }] })).toBe(7)
  })

  it('negates the value for a min objective so higher-is-better holds', () => {
    expect(scorecardRankValue({ gates: [], accepted: true, fitness: [{ metric: 'x', direction: 'min', value: 7 }] })).toBe(-7)
  })

  it('ranks only on the FIRST (primary) fitness objective', () => {
    const card = { gates: [], accepted: true, fitness: [{ metric: 'a', direction: 'max' as const, value: 3 }, { metric: 'b', direction: 'max' as const, value: 99 }] }
    expect(scorecardRankValue(card)).toBe(3)
  })

  it('returns -Infinity for a non-finite primary value', () => {
    expect(scorecardRankValue({ gates: [], accepted: true, fitness: [{ metric: 'x', direction: 'max', value: NaN }] })).toBe(-Infinity)
  })

  it('returns -Infinity when there is no fitness objective', () => {
    expect(scorecardRankValue({ gates: [], accepted: true, fitness: [] })).toBe(-Infinity)
  })
})

describe('compareScorecards', () => {
  const acc = (value: number) => ({ gates: [], accepted: true, fitness: [{ metric: 'x', direction: 'max' as const, value }] })
  const rej = (value: number) => ({ gates: [], accepted: false, fitness: [{ metric: 'x', direction: 'max' as const, value }] })

  it('orders an accepted run before a rejected one regardless of value', () => {
    expect(compareScorecards(rej(100), acc(1))).toBeGreaterThan(0)
    expect(compareScorecards(acc(1), rej(100))).toBeLessThan(0)
  })

  it('orders higher primary fitness first among accepted runs', () => {
    expect(compareScorecards(acc(9), acc(3))).toBeLessThan(0)
    expect(compareScorecards(acc(3), acc(9))).toBeGreaterThan(0)
  })

  it('sorts a mixed list best-first (accepted by fitness, then rejected)', () => {
    const cards = [rej(50), acc(2), acc(8), rej(90)]
    const sorted = [...cards].sort(compareScorecards)
    expect(sorted.map((c) => c.fitness[0].value)).toEqual([8, 2, 90, 50])
  })
})

describe('extractSampleGame (typed replay from a stored run record)', () => {
  const wellFormed = {
    sample_game: {
      opponent: 'random',
      model_seat: 0,
      winner: 1,
      moves: [
        { player: 0, action: 3, label: 'col 3' },
        { player: 1, action: 2, label: 'col 2' },
      ],
      frames: ['FRAME_A', 'FRAME_B', 'FRAME_C'],
    },
  }

  it('pulls a well-formed sample_game into a typed replay', () => {
    const game = extractSampleGame(wellFormed)
    expect(game).toBeDefined()
    expect(game?.model_seat).toBe(0)
    expect(game?.winner).toBe(1)
    expect(game?.opponent).toBe('random')
    expect(game?.frames).toEqual(['FRAME_A', 'FRAME_B', 'FRAME_C'])
    expect(game?.moves).toEqual([
      { player: 0, action: 3, label: 'col 3' },
      { player: 1, action: 2, label: 'col 2' },
    ])
  })

  it('returns undefined when the record carries no sample_game', () => {
    expect(extractSampleGame({})).toBeUndefined()
    expect(extractSampleGame({ sample_game: null })).toBeUndefined()
  })

  it('returns undefined when frames are missing or not a string array', () => {
    expect(extractSampleGame({ sample_game: { model_seat: 0, winner: null, moves: [] } })).toBeUndefined()
    expect(
      extractSampleGame({ sample_game: { model_seat: 0, winner: null, moves: [], frames: [1, 2, 3] } }),
    ).toBeUndefined()
  })

  it('preserves a draw (winner null) and drops malformed move entries', () => {
    const game = extractSampleGame({
      sample_game: {
        model_seat: 1,
        winner: null,
        moves: [{ player: 0, action: 4, label: 'col 4' }, { bogus: true }, 'nope'],
        frames: ['A', 'B'],
      },
    })
    expect(game?.winner).toBeNull()
    expect(game?.model_seat).toBe(1)
    expect(game?.moves).toEqual([{ player: 0, action: 4, label: 'col 4' }])
  })

  it('defaults model_seat to 0 when absent', () => {
    const game = extractSampleGame({ sample_game: { winner: 0, moves: [], frames: ['A'] } })
    expect(game?.model_seat).toBe(0)
  })
})

describe('renderReplayText (agent-facing move-by-move transcript)', () => {
  const game = {
    opponent: 'heuristic',
    model_seat: 0,
    winner: 0 as number | null,
    moves: [
      { player: 0, action: 3, label: 'col 3' },
      { player: 1, action: 2, label: 'col 2' },
    ],
    frames: ['FRAME_A', 'FRAME_B', 'FRAME_C'],
  }

  it('shows every frame, each move with its label, and the result', () => {
    const text = renderReplayText(game)
    expect(text).toContain('FRAME_A')
    expect(text).toContain('FRAME_B')
    expect(text).toContain('FRAME_C')
    expect(text).toContain('col 3')
    expect(text).toContain('col 2')
    expect(text).toContain('Move 1')
    expect(text).toContain('Move 2')
    expect(text).toContain('model wins') // winner 0 === model_seat 0
  })

  it('labels the opponent and reports a draw when winner is null', () => {
    const text = renderReplayText({ ...game, winner: null })
    expect(text).toContain('heuristic')
    expect(text).toContain('draw')
  })

  it('names the opponent as the winner when the model lost', () => {
    const text = renderReplayText({ ...game, winner: 1 })
    expect(text).toContain('heuristic wins')
  })
})
