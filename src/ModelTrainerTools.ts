import * as os from 'node:os'
import type {
  ClaimVerdict,
  ComputeRepoRef,
  ComputeRunner,
  DeepResearchTools,
  EvidencePassage,
  InferenceExecutor,
  LLMConfig,
  ModelSelection,
  ResearchBudget,
} from 'thefactory-tools/types'
import {
  deriveModelRef,
  estimateCampaignEtaSeconds,
  parseFirstValidJson,
  parseStructuredItems,
  runActivityWorkItems,
  uuidv4,
} from 'thefactory-tools/utils'
import type {
  AnalysisCriterion,
  AnalysisRun,
  AnalyzeConfigSpaceParams,
  AnalyzeConfigSpaceResult,
  AnalyzePaperFromUrlParams,
  AnalyzePaperFromUrlResult,
  ResearchTrainingPapersParams,
  ResearchTrainingPapersResult,
  ResearchPapersProgressEvent,
  PaperResearchVerdict,
  AnalyzePaperModelsParams,
  ConsolidateModelsParams,
  ConsolidateModelsResult,
  ConsolidateHypothesesParams,
  ConsolidateHypothesesResult,
  ConsolidatedHypothesisGroup,
  AnalyzePaperModelsResult,
  CalibrateTrainingParams,
  EvaluateTrainingRunParams,
  EvaluateTrainingRunResult,
  EvaluateTrainingRunsParams,
  EvaluateTrainingRunsResult,
  ExperimentSpec,
  GetRunDataParams,
  GetRunDataResult,
  GetRunXaiParams,
  GetRunXaiResult,
  JudgeTrainingRunsParams,
  JudgeTrainingRunsResult,
  MigrateTrainingRunsParams,
  MigrateTrainingRunsResult,
  InvalidateRunsParams,
  InvalidateRunsResult,
  ModelTrainerTools,
  ModelTrainerToolsDeps,
  PlannedTrainingItem,
  ProposeTrainingExperimentsParams,
  ProposeTrainingExperimentsResult,
  ProposeTrainingHypothesesParams,
  ProposeTrainingHypothesesResult,
  BenchmarkModelDeviceParams,
  BenchmarkModelDeviceResult,
  RunXaiDigest,
  ScanProjectModelsParams,
  ScanProjectModelsResult,
  SuggestPaperHypothesesParams,
  SuggestPaperHypothesesResult,
  WeighPaperHypothesesParams,
  WeighPaperHypothesesResult,
  WeighedHypothesis,
  TrainerLeverSpec,
  TrainerManifest,
  TrainingModel,
  TrainingCalibration,
  TrainingCampaignParams,
  TrainingCampaignProgress,
  TrainingCampaignResult,
  TrainingExperimentSuggestion,
  TrainingHypothesis,
  TrainingPaperRecord,
  TrainingRunSummary,
  TrainingVerdict,
  XaiNarrateParams,
  XaiNarrateResult,
} from './modelTrainerTypes.js'
import {
  DEFAULT_HYPOTHESIS_COUNT,
  DEFAULT_RAN_BY,
  HEAVY_RUN_FIELDS,
  DEFAULT_RESEARCH_PAPER_COUNT,
  MAX_RESEARCH_PAPER_COUNT,
  RESEARCH_DISCOVERY_OVERSCAN,
  PAPER_VERIFY_MIN_CONFIDENCE,
  JUDGE_LLM_WEIGHT,
  MAX_JUDGE_RUNS,
  MEMORY_BUDGET_FRACTION,
  DEFAULT_RUN_MEMORY_ESTIMATE_BYTES,
} from './modelTrainerConstants.js'
import {
  fetchPaperText,
  hashTrainingConfig,
  readTrainerManifest,
  setupKeyOf,
} from './modelTrainerHelpers.js'
import {
  applyMigrationRules,
  appliesWhenMap,
  blendJudgeScore,
  findMigrationRule,
  manifestDataFiles,
  migrateExperimentSpec,
  parseDeviceBenchmark,
  parseProgressMarker,
  resolveCampaignParallelism,
  resolveModelDeviceForConfig,
  buildAnalyzePaperModelsSystemPrompt,
  buildAnalyzePaperModelsUserContent,
  buildAnalyzePaperSystemPrompt,
  buildAnalyzePaperUserContent,
  buildJudgeSystemPrompt,
  buildJudgeUserContent,
  buildProposeSystemPrompt,
  buildProposeUserContent,
  buildScanModelsSystemPrompt,
  buildScanModelsUserContent,
  buildSuggestHypothesesSystemPrompt,
  buildSuggestHypothesesUserContent,
  buildWeighHypothesesSystemPrompt,
  buildWeighHypothesesUserContent,
  coerceHypothesisWeights,
  coerceHypothesisCoverage,
  buildXaiNarrateSystemPrompt,
  buildXaiNarrateUserContent,
  coerceAnalyzedPaperModels,
  buildConsolidateModelsSystemPrompt,
  buildConsolidateModelsUserContent,
  coerceConsolidationGroups,
  groupHypothesesForConsolidation,
  planHypothesisConsolidation,
  coerceHypothesisItems,
  coercePaperDraft,
  buildPaperResearchGoal,
  coercePaperCandidates,
  dedupePaperCandidates,
  rankPaperCandidates,
  paperRelevanceClaim,
  isPaperVerdictAdmitted,
  coerceScannedModels,
  coerceSuggestedHypotheses,
  coerceVerdictRows,
  detectMissingPaperModels,
  mergeProposedImprovements,
  diffDecisionTraces,
  summarizeStepAttribution,
  discoverManifestModelCandidates,
  modelBindingNames,
  expandExperimentMatrix,
  estimateRemainingCampaignSeconds,
  normalizeObjectiveScores,
  pickBestRun,
  totalCampaignUnits,
  validateDecisionTrace,
  validateTrainingRunSummary,
  capRunSummaryForStorage,
} from './modelTrainerUtils.js'
import {
  computeConfigSpaceAnalysis,
  criterionValueOf,
  leverImportances,
  normalizeConditionalLevers,
} from './xaiUtils.js'

/**
 * Record types the engine persists keyed by a RUN's key — its `-evaluation` (re-test), `-verdict`
 * (per-run judge score), and `-xai-narrative` (per-run narrative). When a run is deleted these are
 * removed alongside it so none orphan. The `-unrunnable` marker is keyed by SETUP key, not run key,
 * so it is handled separately.
 */
const RUN_KEYED_CHILD_SUFFIXES = ['-evaluation', '-verdict', '-xai-narrative'] as const

/** Drop a `decisionTrace` artifact that {@link validateDecisionTrace} can't use, leaving every other artifact intact. */
function sanitizeRunArtifacts(
  artifacts: Record<string, unknown> | undefined,
): Record<string, unknown> | undefined {
  if (!artifacts || typeof artifacts !== 'object' || !('decisionTrace' in artifacts)) {
    return artifacts
  }
  if (validateDecisionTrace(artifacts.decisionTrace)) return artifacts
  const rest = { ...artifacts }
  delete rest.decisionTrace
  return rest
}

export function createModelTrainerTools(deps: ModelTrainerToolsDeps): ModelTrainerTools {
  const now = deps.now ?? (() => new Date().toISOString())
  const logger = deps.logger

  const hostParallelism = (): number => {
    if (deps.availableParallelism) return deps.availableParallelism()
    try {
      return typeof os.availableParallelism === 'function'
        ? os.availableParallelism()
        : os.cpus().length
    } catch {
      return 1
    }
  }

  // Memory the run pool may budget for concurrent runs. Off TOTAL memory (× a headroom fraction), NOT
  // freemem — freemem excludes reclaimable page cache and reads chronically low after the trainer loads
  // multi-GB kline files, which would spuriously throttle the pool to 1. Injectable for deterministic tests.
  const hostMemoryBudget = (): number => {
    if (deps.availableMemoryBytes) return deps.availableMemoryBytes()
    try {
      return Math.floor(os.totalmem() * MEMORY_BUDGET_FRACTION)
    } catch {
      return 0
    }
  }

  function resolveRunner(target: string | undefined): ComputeRunner {
    if (!target) return deps.computeRunner
    const runner = deps.resolveComputeRunner?.(target)
    if (!runner) throw new Error(`unknown compute target "${target}"`)
    return runner
  }

  async function calibrateOnRunner(
    runner: ComputeRunner,
    params: CalibrateTrainingParams,
  ): Promise<TrainingCalibration | undefined> {
    if (!params.manifest.calibrate) return undefined
    const result = await runner.calibrate({
      repoRef: { kind: 'local', localPath: params.projectRoot },
      commandTemplate: params.manifest.calibrate,
      dataFiles: manifestDataFiles(params.manifest),
      abortSignal: params.abortSignal,
    })
    return {
      secondsObserved: result.secondsObserved,
      ...(result.unitsPerSecond !== undefined ? { unitsPerSecond: result.unitsPerSecond } : {}),
    }
  }

  async function calibrateTrainingThroughput(
    params: CalibrateTrainingParams,
  ): Promise<TrainingCalibration | undefined> {
    return calibrateOnRunner(deps.computeRunner, params)
  }

  async function runTrainingCampaign(
    params: TrainingCampaignParams,
  ): Promise<TrainingCampaignResult> {
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    const recordType = manifest.recordType
    // Roll the spec forward through the manifest's migrations BEFORE planning, so a run dispatched from
    // an old queued/pending config (e.g. a retired `reward_model` name) executes under the migrated
    // shape — runs can't start un-migrated regardless of who dispatched them. No-op when nothing matches.
    const spec = migrateExperimentSpec(params.spec, manifest.migrations)
    const items = expandExperimentMatrix(manifest, spec, hashTrainingConfig)
    const total = items.length
    // Pin conditional levers that don't apply (e.g. forward_horizon on a non-supervised model) to the
    // 'n/a' sentinel in the STORED config — so a record never carries a value for a lever it ignores and
    // xAI can't draw conclusions from it. The executor still receives the raw config (item.config); only
    // what we persist is canonicalised. `setupKey` is left as-is (an opaque dedup key, re-derived by the
    // analysis layer; canonicalising it would ripple into unrunnable/explored markers).
    const runAppliesWhen = appliesWhenMap(manifest)
    const canonicalRunConfig = (cfg: unknown): Record<string, unknown> | undefined =>
      cfg && typeof cfg === 'object'
        ? normalizeConditionalLevers(cfg as Record<string, unknown>, runAppliesWhen)
        : (cfg as undefined)
    const repoRef: ComputeRepoRef = { kind: 'local', localPath: params.projectRoot }
    const runner = resolveRunner(params.computeTarget)
    const dataFiles = manifestDataFiles(manifest)
    // Pack runs to the host: a manifest declaring maxThreadsPerRun lets an unset concurrency default to
    // floor(cpus / threadsPerRun) (vs the idle sequential default), and caps each run's threads so the
    // pool can't oversubscribe. Remote targets keep the host's own CPU math (the runner re-caps as needed).
    const memoryBudget = hostMemoryBudget()
    const {
      concurrency: runConcurrency,
      runEnv,
      memoryCapped,
    } = resolveCampaignParallelism({
      concurrency: params.concurrency,
      maxThreadsPerRun: manifest.maxThreadsPerRun,
      availableParallelism: hostParallelism(),
      // Default-on: fall back to a conservative per-run estimate so the host-RAM ceiling always bounds
      // the pool, even when the manifest declares no figure. Override per-manifest with a measured value.
      maxMemoryBytesPerRun: manifest.maxMemoryBytesPerRun ?? DEFAULT_RUN_MEMORY_ESTIMATE_BYTES,
      availableMemoryBytes: memoryBudget,
    })
    if (memoryCapped) {
      logger?.warn('campaign concurrency reduced to fit host memory', {
        recordType,
        concurrency: runConcurrency,
        requested: params.concurrency,
        maxMemoryBytesPerRun: manifest.maxMemoryBytesPerRun,
        memoryBudget,
      })
    }
    // Benchmarked models carry a `preferredDevice`; auto-apply it to each run's config (the device
    // benchmark's whole point). Concurrency-aware: resolveModelDeviceForConfig keeps an mps preference
    // OFF a parallel sweep (one shared GPU), and never overrides an explicit device.
    const benchmarkedModels = (
      await deps.storage.listRecords({ scope: params.scope, type: `${recordType}-model` })
    )
      .map(
        (r) =>
          r.content as {
            preferredDevice?: 'cpu' | 'mps'
            flavors?: { modelName?: string }[]
            modelNames?: string[]
          },
      )
      .filter((m) => !!m && !!m.preferredDevice)
    const ranBy = params.ranBy ?? params.computeTarget ?? DEFAULT_RAN_BY
    const thesisFields = {
      ...(params.thesis ? { thesis: params.thesis } : {}),
      ...(params.thesisTarget ? { thesisTarget: params.thesisTarget } : {}),
    }

    const emit = async (progress: TrainingCampaignProgress) => {
      await params.onProgress?.(progress)
    }

    let calibration: TrainingCalibration | undefined
    if (manifest.calibrate) {
      await emit({ phase: 'calibrate', done: 0, total, skipped: 0, failed: 0 })
      try {
        const measured = await calibrateOnRunner(runner, {
          projectRoot: params.projectRoot,
          manifest,
          abortSignal: params.abortSignal,
        })
        if (measured) {
          const units = totalCampaignUnits(items)
          const etaSeconds =
            units !== undefined && measured.unitsPerSecond !== undefined
              ? estimateCampaignEtaSeconds(units, measured.unitsPerSecond)
              : undefined
          calibration = { ...measured, ...(etaSeconds !== undefined ? { etaSeconds } : {}) }
        }
      } catch (err) {
        logger?.warn('calibration failed; training without an ETA', {
          error: err instanceof Error ? err.message : String(err),
        })
      }
    }

    // The pipeline version this campaign runs under. Comparability is by MAJOR version ("major.minor",
    // a bare "1" reads as 1.0): a breaking (MAJOR) bump makes every prior run incomparable, so it
    // re-explores everything regardless of skipExplored/unrunnable marks; a MINOR bump is additive and
    // comparable, so marks from the same major still apply. The full string is still stored per run.
    const pipelineVersion = manifest.pipelineVersion ?? '1'
    const majorOf = (v: string | undefined) => parseInt(String(v ?? '1'), 10) || 1
    const pipelineMajor = majorOf(pipelineVersion)
    const versionOf = (c: { pipelineVersion?: string } | undefined) => majorOf(c?.pipelineVersion)

    let exploredSetups: Set<string> | undefined
    if (params.skipExplored) {
      const priorRecords = await deps.storage.listRecords({
        scope: params.scope,
        type: recordType,
        omit: HEAVY_RUN_FIELDS,
      })
      exploredSetups = new Set(
        priorRecords
          .map(
            (r) =>
              r.content as {
                status?: string
                setupKey?: string
                pipelineVersion?: string
                config?: Record<string, unknown>
              },
          )
          .filter((c) => c?.status === 'completed' && versionOf(c) === pipelineMajor)
          .map((c) => c.setupKey ?? (c.config ? setupKeyOf(c.config) : undefined))
          .filter((k): k is string => typeof k === 'string'),
      )
    }

    // Setups the user marked unrunnable for THIS version: a hard skip unless `refresh` forces them.
    let unrunnableSetups = new Set<string>()
    if (!params.refresh) {
      const markers = await deps.storage.listRecords({
        scope: params.scope,
        type: `${recordType}-unrunnable`,
      })
      unrunnableSetups = new Set(
        markers
          .map(
            (r) =>
              r.content as {
                setupKey?: string
                unrunnable?: boolean
                pipelineVersion?: string
                config?: Record<string, unknown>
              },
          )
          .filter((c) => c?.unrunnable !== false && versionOf(c) === pipelineMajor)
          .map((c) => c?.setupKey ?? (c?.config ? setupKeyOf(c.config) : undefined))
          .filter((k): k is string => typeof k === 'string'),
      )
    }

    let lastKey: string | undefined
    // Real per-run durations of items completed THIS session — the basis for the remaining-time ETA. Runs
    // over the same data+model take a similar time, so the average × remaining concurrency waves is a stable
    // total estimate. NOT elapsed/done (which is diluted toward zero by instantly-skipped completed runs).
    const itemDurationsMs: number[] = []
    // Progress is a best-effort side-channel: a host's sink throwing synchronously or
    // rejecting (e.g. a transient progress-record write conflict under concurrency) must
    // never abort a training run, drop the terminal signal's siblings, or escape as an
    // unhandled rejection that could take the host process down.
    const emitItemProgress = (key: string, progress: Record<string, unknown>): Promise<void> => {
      if (!params.onItemProgress) return Promise.resolve()
      try {
        return Promise.resolve(params.onItemProgress(key, progress)).catch(() => {})
      } catch {
        return Promise.resolve()
      }
    }
    const summary = await runActivityWorkItems<
      PlannedTrainingItem,
      { key: string; objective: number }
    >({
      items,
      concurrency: runConcurrency,
      abortSignal: params.abortSignal,
      isFresh: async (item) => {
        if (params.refresh) return false
        const setupKey = setupKeyOf(item.config)
        if (unrunnableSetups.has(setupKey)) return true
        if (exploredSetups && exploredSetups.has(setupKey)) return true
        const existing = await deps.storage.readRecord({
          scope: params.scope,
          type: recordType,
          key: item.key,
        })
        const content = existing?.content as
          | { status?: string; pipelineVersion?: string }
          | undefined
        return content?.status === 'completed' && versionOf(content) === pipelineMajor
      },
      runItem: async (item) => {
        const device = resolveModelDeviceForConfig({
          config: item.config,
          models: benchmarkedModels,
          concurrency: runConcurrency,
        })
        const runConfig = device ? { ...item.config, device } : item.config
        const handle = runner.runJob({
          jobId: item.key,
          repoRef,
          commandTemplate: manifest.run,
          config: runConfig,
          dataFiles,
          ...(runEnv ? { env: runEnv } : {}),
          abortSignal: params.abortSignal,
        })
        if (params.onItemProgress) {
          // Subscribe to logs BEFORE the (awaited) starting emit so a synchronously
          // streamed marker can't slip past an unregistered listener.
          handle.onLog((line) => {
            const marker = parseProgressMarker(line)
            if (marker) void emitItemProgress(item.key, marker)
          })
          await emitItemProgress(item.key, { phase: 'starting' })
        }
        const result = await handle.done
        // Signal this item left the in-flight set (completed/failed/aborted) so a host
        // tracking concurrent runs can drop it from its live display.
        await emitItemProgress(item.key, { terminal: true, status: result.status })
        if (result.status !== 'completed') {
          const error = result.error ?? `training exited with code ${result.exitCode}`
          if (result.status !== 'aborted') {
            const partial = (result.summary ?? {}) as Record<string, unknown>
            await deps.storage.upsertRecord({
              scope: params.scope,
              type: recordType,
              key: item.key,
              content: {
                ...partial,
                status: 'failed',
                error,
                ...(result.logTail?.length ? { logTail: result.logTail } : {}),
                config: canonicalRunConfig(item.config),
                setupKey: setupKeyOf(item.config),
                pipelineVersion,
                ...thesisFields,
                ranAt: now(),
                ranBy,
                durationMs: result.durationMs,
              },
            })
            params.onRecordWritten?.(recordType, item.key)
          }
          throw new Error(error)
        }
        const runSummary = capRunSummaryForStorage(validateTrainingRunSummary(result.summary))
        const artifacts = sanitizeRunArtifacts(runSummary.artifacts)
        await deps.storage.upsertRecord({
          scope: params.scope,
          type: recordType,
          key: item.key,
          content: {
            ...runSummary,
            config: canonicalRunConfig(runSummary.config),
            ...(artifacts ? { artifacts } : {}),
            status: 'completed',
            setupKey: setupKeyOf(item.config),
            pipelineVersion,
            ...thesisFields,
            ranAt: now(),
            ranBy,
            durationMs: result.durationMs,
          },
        })
        params.onRecordWritten?.(recordType, item.key)
        lastKey = item.key
        if (typeof result.durationMs === 'number') itemDurationsMs.push(result.durationMs)
        return { key: item.key, objective: runSummary.objective }
      },
      onProgress: (progress) => {
        // Prefer the calibration-derived ETA; otherwise estimate the remaining wall-clock from the ACTUAL
        // durations of runs completed this session (avg per-run × remaining concurrency waves) — the total
        // predicted time for all remaining runs to finish.
        const calibratedEta =
          calibration?.etaSeconds !== undefined && progress.total > 0
            ? calibration.etaSeconds * ((progress.total - progress.done) / progress.total)
            : undefined
        const wallClockEta =
          calibratedEta === undefined
            ? estimateRemainingCampaignSeconds({
                durationsMs: itemDurationsMs,
                remaining: progress.total - progress.done,
                concurrency: runConcurrency,
              })
            : undefined
        const etaSeconds = calibratedEta ?? wallClockEta
        void emit({
          phase: 'train',
          done: progress.done,
          total: progress.total,
          skipped: progress.skipped,
          failed: progress.failed,
          ...(etaSeconds !== undefined ? { etaSeconds } : {}),
          ...(wallClockEta !== undefined && calibratedEta === undefined ? { etaApprox: true } : {}),
          ...(lastKey !== undefined ? { lastKey } : {}),
        })
      },
    })

    const completed = summary.outcomes.filter((o) => o.status === 'completed').length
    const failures = summary.outcomes
      .filter((o) => o.status === 'failed')
      .map((o) => ({ key: o.item.key, error: o.error ?? 'unknown failure' }))
    const records = await deps.storage.listRecords({
      scope: params.scope,
      type: recordType,
      omit: HEAVY_RUN_FIELDS,
    })
    const best = pickBestRun(
      records
        .map((r) => ({
          key: r.key ?? '',
          content: r.content as { status?: string; objective?: unknown },
        }))
        .filter(
          (r) =>
            r.key && r.content?.status === 'completed' && typeof r.content.objective === 'number',
        )
        .map((r) => ({ key: r.key, objective: r.content.objective as number })),
      manifest.objective.direction,
    )

    await emit({
      phase: 'done',
      done: summary.done,
      total,
      skipped: summary.skipped,
      failed: summary.failed,
    })
    logger?.info('training campaign finished', {
      recordType,
      planned: total,
      completed,
      skipped: summary.skipped,
      failed: summary.failed,
      aborted: summary.aborted,
    })

    return {
      recordType,
      planned: total,
      keys: items.map((item) => item.key),
      completed,
      skipped: summary.skipped,
      failed: summary.failed,
      aborted: summary.aborted,
      ...(failures.length > 0 ? { failures } : {}),
      ...(best ? { bestKey: best.key, bestObjective: best.objective } : {}),
      direction: manifest.objective.direction,
      ...(calibration ? { calibration } : {}),
      finishedAt: now(),
    }
  }

  async function evaluateOneRun(
    manifest: TrainerManifest,
    opts: {
      scope: string
      projectRoot: string
      runKey: string
      computeTarget?: string
      abortSignal?: AbortSignal
      onRecordWritten?: (type: string, key: string) => void
      activityId?: string
    },
  ): Promise<EvaluateTrainingRunResult> {
    const recordType = manifest.recordType
    const record = await deps.storage.readRecord({
      scope: opts.scope,
      type: recordType,
      key: opts.runKey,
    })
    if (!record) throw new Error(`no run record for key ${opts.runKey}`)
    const content = record.content as {
      config?: Record<string, unknown>
      artifacts?: { checkpoint?: unknown }
    }
    const checkpoint = content.artifacts?.checkpoint
    if (typeof checkpoint !== 'string' || !checkpoint) {
      throw new Error(`run ${opts.runKey} has no checkpoint artifact to evaluate`)
    }

    const { runEnv } = resolveCampaignParallelism({
      maxThreadsPerRun: manifest.maxThreadsPerRun,
      availableParallelism: hostParallelism(),
    })
    const handle = resolveRunner(opts.computeTarget).runJob({
      jobId: `eval-${opts.runKey}`,
      repoRef: { kind: 'local', localPath: opts.projectRoot },
      commandTemplate: manifest.evaluate!,
      config: { ...(content.config ?? {}), checkpoint },
      dataFiles: manifestDataFiles(manifest),
      ...(runEnv ? { env: runEnv } : {}),
      abortSignal: opts.abortSignal,
    })
    const result = await handle.done
    if (result.status !== 'completed') {
      throw new Error(result.error ?? `evaluation exited with code ${result.exitCode}`)
    }
    const summary = capRunSummaryForStorage(validateTrainingRunSummary(result.summary))
    const evaluatedAt = now()
    const evaluationType = `${recordType}-evaluation`
    await deps.storage.upsertRecord({
      scope: opts.scope,
      type: evaluationType,
      key: opts.runKey,
      content: {
        ...summary,
        runKey: opts.runKey,
        status: 'completed',
        evaluatedAt,
        ...(opts.activityId ? { activityId: opts.activityId } : {}),
      },
    })
    opts.onRecordWritten?.(evaluationType, opts.runKey)
    logger?.info('evaluated training run', { recordType, runKey: opts.runKey })
    return { recordType, runKey: opts.runKey, objective: summary.objective, evaluatedAt }
  }

  async function evaluateTrainingRun(
    params: EvaluateTrainingRunParams,
  ): Promise<EvaluateTrainingRunResult> {
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    if (!manifest.evaluate) {
      throw new Error('trainer manifest declares no evaluate command')
    }
    return evaluateOneRun(manifest, {
      scope: params.scope,
      projectRoot: params.projectRoot,
      runKey: params.runKey,
      computeTarget: params.computeTarget,
      abortSignal: params.abortSignal,
      onRecordWritten: params.onRecordWritten,
      activityId: params.activityId,
    })
  }

  async function evaluateTrainingRuns(
    params: EvaluateTrainingRunsParams,
  ): Promise<EvaluateTrainingRunsResult> {
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    if (!manifest.evaluate) {
      throw new Error('trainer manifest declares no evaluate command')
    }
    const recordType = manifest.recordType
    const { concurrency: evalConcurrency, memoryCapped: evalMemoryCapped } =
      resolveCampaignParallelism({
        concurrency: params.concurrency,
        maxThreadsPerRun: manifest.maxThreadsPerRun,
        availableParallelism: hostParallelism(),
        maxMemoryBytesPerRun: manifest.maxMemoryBytesPerRun ?? DEFAULT_RUN_MEMORY_ESTIMATE_BYTES,
        availableMemoryBytes: hostMemoryBudget(),
      })
    if (evalMemoryCapped) {
      logger?.warn('evaluation concurrency reduced to fit host memory', {
        recordType,
        concurrency: evalConcurrency,
        requested: params.concurrency,
      })
    }
    const results: EvaluateTrainingRunResult[] = []
    const summary = await runActivityWorkItems<string, EvaluateTrainingRunResult>({
      items: params.runKeys,
      concurrency: evalConcurrency,
      abortSignal: params.abortSignal,
      runItem: async (runKey) => {
        const result = await evaluateOneRun(manifest, {
          scope: params.scope,
          projectRoot: params.projectRoot,
          runKey,
          computeTarget: params.computeTarget,
          abortSignal: params.abortSignal,
          onRecordWritten: params.onRecordWritten,
          activityId: params.activityId,
        })
        results.push(result)
        return result
      },
      onProgress: (progress) =>
        params.onProgress?.({
          done: progress.done,
          total: progress.total,
          failed: progress.failed,
        }),
    })
    const failures = summary.outcomes
      .filter((o) => o.status === 'failed')
      .map((o) => ({ runKey: o.item, error: o.error ?? 'unknown failure' }))
    return {
      recordType,
      evaluated: summary.outcomes.filter((o) => o.status === 'completed').length,
      failed: summary.failed,
      results,
      ...(failures.length > 0 ? { failures } : {}),
    }
  }

  function requireInferenceExecutor(): InferenceExecutor {
    if (!deps.inferenceExecutor) {
      throw new Error('judging/proposing requires an inferenceExecutor')
    }
    return deps.inferenceExecutor
  }

  function requireDeepResearch(): DeepResearchTools {
    if (!deps.deepResearch) {
      throw new Error('researchTrainingPapers requires a deepResearch seam')
    }
    return deps.deepResearch
  }

  // `omitHeavy` sheds the unbounded per-step fields for callers that only read light fields (judging,
  // config-space analysis); the xAI digest leaves it off because it needs the focus run's decision trace.
  async function listCompletedRuns(scope: string, recordType: string, omitHeavy = false) {
    const records = await deps.storage.listRecords({
      scope,
      type: recordType,
      ...(omitHeavy ? { omit: HEAVY_RUN_FIELDS } : {}),
    })
    return records
      .filter(
        (r) =>
          r.key &&
          (r.content as { status?: string })?.status === 'completed' &&
          typeof (r.content as { objective?: unknown })?.objective === 'number',
      )
      .map((r) => ({ key: r.key as string, content: r.content as Record<string, unknown> }))
  }

  async function judgeTrainingRuns(
    params: JudgeTrainingRunsParams,
  ): Promise<JudgeTrainingRunsResult> {
    const executor = requireInferenceExecutor()
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    const recordType = manifest.recordType
    const judgedBy = deriveModelRef({ kind: 'api', llmConfig: params.llmConfig }).label
    const judgedAt = now()
    const allRuns = await listCompletedRuns(params.scope, recordType, true)
    const onlyKeys = params.runKeys && params.runKeys.length ? new Set(params.runKeys) : undefined
    const runs = onlyKeys ? allRuns.filter((run) => onlyKeys.has(run.key)) : allRuns

    const verdicts: TrainingVerdict[] = []
    const healthy: { key: string; objective: number; content: Record<string, unknown> }[] = []
    for (const run of runs) {
      const health = run.content.health as { status?: string; flags?: string[] } | undefined
      if (health?.status && health.status !== 'ok') {
        verdicts.push({
          key: run.key,
          score: 0,
          objectiveScore: 0,
          why: `auto-rejected on health: ${health.flags?.join(', ') || health.status}`,
          rejected: true,
          judgedBy,
          judgedAt,
        })
        continue
      }
      healthy.push({
        key: run.key,
        objective: run.content.objective as number,
        content: run.content,
      })
    }

    if (healthy.length > 0) {
      const direction = manifest.objective.direction
      healthy.sort((a, b) =>
        direction === 'max' ? b.objective - a.objective : a.objective - b.objective,
      )
      if (healthy.length > MAX_JUDGE_RUNS) {
        logger?.warn('judging only the best runs; the rest keep objective-only verdicts', {
          judged: MAX_JUDGE_RUNS,
          total: healthy.length,
        })
      }
      const sent = healthy.slice(0, MAX_JUDGE_RUNS)
      const objectiveScores = normalizeObjectiveScores(healthy, direction)
      const res = await executor.runInference({
        systemPrompt: buildJudgeSystemPrompt(manifest, params.instructions),
        userContent: buildJudgeUserContent(
          sent.map((r) => ({
            key: r.key,
            objective: r.objective,
            config: r.content.config as Record<string, unknown> | undefined,
            metrics: r.content.metrics as Record<string, number> | undefined,
            seed: r.content.seed as number | undefined,
          })),
        ),
        model: { kind: 'api', llmConfig: params.llmConfig },
        abortSignal: params.abortSignal,
      })
      const rowsByKey = new Map(
        coerceVerdictRows(parseStructuredItems(res.text)).map((r) => [r.key, r]),
      )
      for (const run of healthy) {
        const objectiveScore = objectiveScores.get(run.key) ?? 0
        const row = rowsByKey.get(run.key)
        verdicts.push(
          row
            ? {
                key: run.key,
                score: blendJudgeScore(objectiveScore, row.score, JUDGE_LLM_WEIGHT),
                objectiveScore,
                llmScore: row.score,
                why: row.why,
                judgedBy,
                judgedAt,
              }
            : {
                key: run.key,
                score: objectiveScore,
                objectiveScore,
                why: 'no LLM verdict; scored on the objective alone',
                judgedBy,
                judgedAt,
              },
        )
      }
    }

    const verdictType = `${recordType}-verdict`
    for (const verdict of verdicts) {
      await deps.storage.upsertRecord({
        scope: params.scope,
        type: verdictType,
        key: verdict.key,
        content: {
          ...(verdict as unknown as Record<string, unknown>),
          ...(params.activityId ? { activityId: params.activityId } : {}),
        },
      })
      params.onRecordWritten?.(verdictType, verdict.key)
    }
    logger?.info('judged training runs', {
      recordType,
      judged: verdicts.length,
      rejected: verdicts.filter((v) => v.rejected).length,
    })
    return {
      recordType,
      judged: verdicts.length,
      rejected: verdicts.filter((v) => v.rejected).length,
      verdicts,
      judgedBy,
      judgedAt,
    }
  }

  // Shared front half of both proposers: read history + verdicts, ask the LLM, and parse + validate the
  // returned items. The two proposers diverge only in how they PERSIST the items (hypothesis vs suggestion).
  async function proposeInferenceItems(
    params: ProposeTrainingHypothesesParams | ProposeTrainingExperimentsParams,
  ) {
    const executor = requireInferenceExecutor()
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    const recordType = manifest.recordType
    const proposedBy = deriveModelRef({ kind: 'api', llmConfig: params.llmConfig }).label
    const proposedAt = now()
    const count = params.count ?? DEFAULT_HYPOTHESIS_COUNT

    const runs = await listCompletedRuns(params.scope, recordType, true)
    const direction = manifest.objective.direction
    runs.sort((a, b) =>
      direction === 'max'
        ? (b.content.objective as number) - (a.content.objective as number)
        : (a.content.objective as number) - (b.content.objective as number),
    )
    const sentRuns = runs.slice(0, MAX_JUDGE_RUNS)
    const verdictRecords = await deps.storage.listRecords({
      scope: params.scope,
      type: `${recordType}-verdict`,
    })
    const verdicts = coerceVerdictRows(verdictRecords.map((r) => r.content))

    const res = await executor.runInference({
      systemPrompt: buildProposeSystemPrompt(manifest, count, params.instructions),
      userContent: buildProposeUserContent({
        manifest,
        runs: sentRuns.map((r) => ({
          key: r.key,
          objective: r.content.objective as number,
          config: r.content.config as Record<string, unknown> | undefined,
        })),
        verdicts,
        bestObjective: sentRuns[0]?.content.objective as number | undefined,
      }),
      model: { kind: 'api', llmConfig: params.llmConfig },
      abortSignal: params.abortSignal,
    })

    const items = coerceHypothesisItems(parseStructuredItems(res.text), manifest)
    return { items, recordType, proposedBy, proposedAt, count }
  }

  async function proposeTrainingHypotheses(
    params: ProposeTrainingHypothesesParams,
  ): Promise<ProposeTrainingHypothesesResult> {
    const { items, recordType, proposedBy, proposedAt, count } = await proposeInferenceItems(params)
    const hypothesisType = `${recordType}-hypothesis`
    const existing = await deps.storage.listRecords({ scope: params.scope, type: hypothesisType })
    const seenIds = new Set(existing.map((r) => r.key).filter((k): k is string => !!k))

    const hypotheses: TrainingHypothesis[] = []
    let skippedExisting = 0
    for (const item of items) {
      if (hypotheses.length >= count) break
      const id = hashTrainingConfig(item.spec as Record<string, unknown>)
      if (seenIds.has(id)) {
        skippedExisting += 1
        continue
      }
      seenIds.add(id)
      const hypothesis: TrainingHypothesis = {
        id,
        title: item.title,
        rationale: item.rationale,
        spec: item.spec,
        status: 'untested',
        verdictSource: 'auto',
        source: 'llm',
        proposedBy,
        createdAt: proposedAt,
        updatedAt: proposedAt,
      }
      await deps.storage.upsertRecord({
        scope: params.scope,
        type: hypothesisType,
        key: id,
        content: hypothesis as unknown as Record<string, unknown>,
      })
      params.onRecordWritten?.(hypothesisType, id)
      hypotheses.push(hypothesis)
    }
    logger?.info('proposed training hypotheses', {
      recordType,
      proposed: hypotheses.length,
      skippedExisting,
    })
    return {
      recordType,
      proposed: hypotheses.length,
      skippedExisting,
      hypotheses,
      proposedBy,
      proposedAt,
    }
  }

  async function proposeTrainingExperiments(
    params: ProposeTrainingExperimentsParams,
  ): Promise<ProposeTrainingExperimentsResult> {
    const { items, recordType, proposedBy, proposedAt, count } = await proposeInferenceItems(params)
    const suggestionType = `${recordType}-xai-suggestion`
    const existing = await deps.storage.listRecords({ scope: params.scope, type: suggestionType })
    const seenIds = new Set(existing.map((r) => r.key).filter((k): k is string => !!k))

    const suggestions: TrainingExperimentSuggestion[] = []
    let skippedExisting = 0
    for (const item of items) {
      if (suggestions.length >= count) break
      const id = hashTrainingConfig(item.spec as Record<string, unknown>)
      if (seenIds.has(id)) {
        skippedExisting += 1
        continue
      }
      seenIds.add(id)
      const suggestion: TrainingExperimentSuggestion = {
        id,
        title: item.title,
        rationale: item.rationale,
        spec: item.spec,
        source: 'llm',
        proposedBy,
        proposedAt,
      }
      await deps.storage.upsertRecord({
        scope: params.scope,
        type: suggestionType,
        key: id,
        content: suggestion as unknown as Record<string, unknown>,
      })
      params.onRecordWritten?.(suggestionType, id)
      suggestions.push(suggestion)
    }
    logger?.info('proposed training experiments', {
      recordType,
      proposed: suggestions.length,
      skippedExisting,
    })
    return {
      recordType,
      proposed: suggestions.length,
      skippedExisting,
      suggestions,
      proposedBy,
      proposedAt,
    }
  }

  // Shared paper-synthesis core, used by BOTH analyzePaperFromUrl (single pasted link) and
  // researchTrainingPapers (open-ended discovery): given the paper's fetched page TEXT, summarise it
  // into a draft record + its testable hypotheses. Returns undefined when the model produced no usable
  // draft (title+claim) — the caller decides whether that is fatal (analyze) or a skip (research).
  async function synthesizePaperFromText(input: {
    manifest: TrainerManifest
    url: string
    text: string
    notes?: string
    model: ModelSelection
    abortSignal?: AbortSignal
  }): Promise<
    | {
        paperDraft: Partial<TrainingPaperRecord>
        hypothesisItems: ReturnType<typeof coerceHypothesisItems>
      }
    | undefined
  > {
    const executor = requireInferenceExecutor()
    const res = await executor.runInference({
      systemPrompt: buildAnalyzePaperSystemPrompt(input.manifest, input.notes),
      userContent: buildAnalyzePaperUserContent({
        url: input.url,
        text: input.text,
        notes: input.notes,
      }),
      model: input.model,
      abortSignal: input.abortSignal,
    })
    const parsed = parseFirstValidJson(res.text)
    const paperDraft = coercePaperDraft(parsed)
    if (!paperDraft) return undefined
    const rawHyps =
      parsed &&
      typeof parsed === 'object' &&
      Array.isArray((parsed as Record<string, unknown>).hypotheses)
        ? ((parsed as Record<string, unknown>).hypotheses as unknown[])
        : []
    return { paperDraft, hypothesisItems: coerceHypothesisItems(rawHyps, input.manifest) }
  }

  // Persist a synthesised paper draft + its hypotheses. Hypotheses dedup by spec hash so identical specs
  // from any source (propose / manual / another paper) link to the ONE existing record rather than
  // duplicate; the paper is written as a DRAFT (status 'untested', source 'research') for the user to
  // verify. Shared by analyzePaperFromUrl + researchTrainingPapers.
  async function persistPaperWithHypotheses(input: {
    scope: string
    recordType: string
    paperDraft: Partial<TrainingPaperRecord>
    hypothesisItems: ReturnType<typeof coerceHypothesisItems>
    url: string
    proposedBy: string
    at: string
    /** Provenance of a research draft (why the verify gate admitted it); omitted for analyze/manual. */
    researchVerdict?: PaperResearchVerdict
    onRecordWritten?: (type: string, key: string) => void
  }): Promise<{
    paper: TrainingPaperRecord
    hypotheses: TrainingHypothesis[]
    linkedHypothesisIds: string[]
  }> {
    const paperId = uuidv4()
    const paperType = `${input.recordType}-paper`
    const hypothesisType = `${input.recordType}-hypothesis`
    const existing = await deps.storage.listRecords({ scope: input.scope, type: hypothesisType })
    const byId = new Map<string, TrainingHypothesis>()
    for (const r of existing) {
      const content = r.content as unknown as TrainingHypothesis
      if (content && typeof content.id === 'string') byId.set(content.id, content)
    }
    const hypotheses: TrainingHypothesis[] = []
    const linkedHypothesisIds: string[] = []
    const seen = new Set<string>()
    for (const item of input.hypothesisItems) {
      const hid = hashTrainingConfig(item.spec as Record<string, unknown>)
      if (seen.has(hid)) continue
      seen.add(hid)
      const prior = byId.get(hid)
      const hypothesis: TrainingHypothesis = prior
        ? {
            ...prior,
            paperIds: Array.from(new Set([...(prior.paperIds ?? []), paperId])),
            updatedAt: input.at,
          }
        : {
            id: hid,
            title: item.title,
            rationale: item.rationale,
            spec: item.spec,
            // Carry the criterion + claim the coercer extracted so a comparison/context hypothesis
            // keeps its success/failure definition (else it silently defaults to a pooled test).
            ...(item.comparison ? { comparison: item.comparison } : {}),
            ...(item.claim ? { claim: item.claim } : {}),
            status: 'untested',
            verdictSource: 'auto',
            source: 'paper',
            proposedBy: input.proposedBy,
            paperIds: [paperId],
            createdAt: input.at,
            updatedAt: input.at,
          }
      await deps.storage.upsertRecord({
        scope: input.scope,
        type: hypothesisType,
        key: hid,
        content: hypothesis as unknown as Record<string, unknown>,
      })
      input.onRecordWritten?.(hypothesisType, hid)
      hypotheses.push(hypothesis)
      linkedHypothesisIds.push(hid)
    }
    const paper: TrainingPaperRecord = {
      ...input.paperDraft,
      id: paperId,
      title: input.paperDraft.title as string,
      claim: input.paperDraft.claim as string,
      url: input.url,
      hypothesisIds: linkedHypothesisIds,
      status: 'untested',
      source: 'research',
      ...(input.researchVerdict ? { researchVerdict: input.researchVerdict } : {}),
      createdAt: input.at,
      updatedAt: input.at,
    }
    await deps.storage.upsertRecord({
      scope: input.scope,
      type: paperType,
      key: paperId,
      content: paper as unknown as Record<string, unknown>,
    })
    input.onRecordWritten?.(paperType, paperId)
    return { paper, hypotheses, linkedHypothesisIds }
  }

  async function analyzePaperFromUrl(
    params: AnalyzePaperFromUrlParams,
  ): Promise<AnalyzePaperFromUrlResult> {
    requireInferenceExecutor()
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    const recordType = manifest.recordType
    const analyzedBy = deriveModelRef({ kind: 'api', llmConfig: params.llmConfig }).label
    const analyzedAt = now()

    // The TOOL fetches the page text and hands it to the model — no web tools needed by the model.
    const fetchText = params.fetchPaperText ?? fetchPaperText
    const text = await fetchText(params.url, params.abortSignal)

    const synth = await synthesizePaperFromText({
      manifest,
      url: params.url,
      text,
      notes: params.notes,
      model: { kind: 'api', llmConfig: params.llmConfig },
      abortSignal: params.abortSignal,
    })
    if (!synth) throw new Error('the model did not return a usable paper summary for this link')

    const persisted = await persistPaperWithHypotheses({
      scope: params.scope,
      recordType,
      paperDraft: synth.paperDraft,
      hypothesisItems: synth.hypothesisItems,
      url: params.url,
      proposedBy: analyzedBy,
      at: analyzedAt,
      onRecordWritten: params.onRecordWritten,
    })
    logger?.info('analyzed paper from url', {
      recordType,
      url: params.url,
      id: persisted.paper.id,
      hypotheses: persisted.hypotheses.length,
    })
    return {
      recordType,
      paper: persisted.paper,
      hypotheses: persisted.hypotheses,
      linkedHypothesisIds: persisted.linkedHypothesisIds,
      analyzedBy,
      analyzedAt,
    }
  }

  // Open-ended paper discovery: derive a domain research goal, discover N candidates via the deep-research
  // seam, then per candidate — fetch its REAL page, verify (against that page) it is a real paper relevant
  // to this project, and synthesise the survivor into a DRAFT record + hypotheses. Reality/relevance is
  // grounded in the paper's own fetched text (like analyzePaperFromUrl), so nothing is fabricated; a
  // candidate that fails discovery/verify/fetch/synthesis is SKIPPED (counted), never invented.
  async function researchTrainingPapers(
    params: ResearchTrainingPapersParams,
  ): Promise<ResearchTrainingPapersResult> {
    const dr = requireDeepResearch()
    requireInferenceExecutor()
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    const recordType = manifest.recordType
    const researchedBy = deriveModelRef(params.model).label
    const researchedAt = now()
    const model = params.model
    const budgetOpt = params.researchBudget ? { budget: params.researchBudget } : {}
    const emit = (event: ResearchPapersProgressEvent) => {
      try {
        params.onProgress?.(event)
      } catch {
        // progress is best-effort — a bad sink must never fail the run.
      }
    }
    const throwIfAborted = () => {
      if (params.abortSignal?.aborted) throw new Error('research aborted')
    }

    const goal = buildPaperResearchGoal(manifest, { notes: params.notes })
    // `count` is the target number of DRAFTED papers; over-scan discovery so the paper-host ranker has a
    // pool to prefer from, then verify ranked candidates only until the target is hit (§ stop-at-target).
    const targetCount = Math.max(
      1,
      Math.min(params.count ?? DEFAULT_RESEARCH_PAPER_COUNT, MAX_RESEARCH_PAPER_COUNT),
    )
    const discoverLimit = targetCount * RESEARCH_DISCOVERY_OVERSCAN
    emit({ phase: 'discover', message: `Searching for papers (target ${targetCount})…` })
    const discoveredRaw = await dr.discoverSources({
      query: goal,
      limit: discoverLimit,
      model,
      abortSignal: params.abortSignal,
      ...budgetOpt,
    })
    // Rank paper-venue hosts first so the target is met from real papers before any blog/marketing tail.
    const candidates0 = rankPaperCandidates(coercePaperCandidates(discoveredRaw as unknown[]))

    // Dedup against the registry so a research run never re-drafts a paper already present (manual,
    // analyzePaperFromUrl, or an earlier research run).
    const existingPapers = await deps.storage.listRecords({
      scope: params.scope,
      type: `${recordType}-paper`,
    })
    const existing = existingPapers.map((r) => {
      const c = r.content as unknown as TrainingPaperRecord
      return { url: c?.url, title: c?.title }
    })
    const candidates = dedupePaperCandidates(candidates0, existing)
    const skippedDuplicate = candidates0.length - candidates.length

    const papers: TrainingPaperRecord[] = []
    const hypotheses: TrainingHypothesis[] = []
    let rejected = 0
    let failed = 0
    const fetchText = params.fetchPaperText ?? fetchPaperText

    emit({
      phase: 'verify',
      message: `${candidates.length} candidate paper(s) to verify.`,
      total: candidates.length,
    })
    for (let i = 0; i < candidates.length; i++) {
      // Stop once the target is drafted — the low-affinity tail is never fetched/verified (the cost win).
      if (papers.length >= targetCount) break
      throwIfAborted()
      const candidate = candidates[i]
      try {
        // Fetch the candidate's real page — it grounds BOTH the verify gate and the synthesis. A page
        // that won't fetch (404 / PDF) or reads empty is a failed candidate, not a fabricated one.
        const text = await fetchText(candidate.url, params.abortSignal)
        if (!text || !text.trim()) {
          failed++
          continue
        }
        throwIfAborted()
        emit({
          phase: 'verify',
          message: `Verifying "${candidate.title}"…`,
          index: i + 1,
          total: candidates.length,
        })
        const evidence: EvidencePassage[] = [
          { source: { title: candidate.title, url: candidate.url }, text },
        ]
        const verdict = await dr.verifyClaim({
          claim: paperRelevanceClaim(candidate, manifest),
          evidence,
          model,
          abortSignal: params.abortSignal,
          ...budgetOpt,
        })
        if (!isPaperVerdictAdmitted(verdict, PAPER_VERIFY_MIN_CONFIDENCE)) {
          rejected++
          continue
        }
        throwIfAborted()
        emit({
          phase: 'synthesize',
          message: `Summarising "${candidate.title}"…`,
          index: i + 1,
          total: candidates.length,
        })
        const synth = await synthesizePaperFromText({
          manifest,
          url: candidate.url,
          text,
          notes: params.notes,
          model,
          abortSignal: params.abortSignal,
        })
        if (!synth) {
          failed++
          continue
        }
        // Stamp the admitting verdict onto the draft so a reviewer sees WHY it was accepted.
        const researchVerdict: PaperResearchVerdict = {
          status: verdict.status,
          confidence: verdict.confidence,
          ...(Array.isArray(verdict.evidence) && verdict.evidence.length
            ? { quotes: verdict.evidence.slice(0, 3) }
            : {}),
        }
        const persisted = await persistPaperWithHypotheses({
          scope: params.scope,
          recordType,
          paperDraft: synth.paperDraft,
          hypothesisItems: synth.hypothesisItems,
          url: candidate.url,
          proposedBy: researchedBy,
          at: researchedAt,
          researchVerdict,
          onRecordWritten: params.onRecordWritten,
        })
        papers.push(persisted.paper)
        hypotheses.push(...persisted.hypotheses)
      } catch (err) {
        if (params.abortSignal?.aborted) throw err
        failed++
        logger?.warn('research candidate failed', { url: candidate.url, error: String(err) })
      }
    }

    logger?.info('researched training papers', {
      recordType,
      discovered: candidates0.length,
      drafted: papers.length,
      skippedDuplicate,
      rejected,
      failed,
    })
    return {
      recordType,
      papers,
      hypotheses,
      discovered: candidates0.length,
      skippedDuplicate,
      rejected,
      failed,
      researchedBy,
      researchedAt,
    }
  }

  async function suggestPaperHypotheses(
    params: SuggestPaperHypothesesParams,
  ): Promise<SuggestPaperHypothesesResult> {
    const executor = requireInferenceExecutor()
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    const recordType = manifest.recordType
    const suggestedBy = deriveModelRef({ kind: 'api', llmConfig: params.llmConfig }).label
    const suggestedAt = now()
    const paperType = `${recordType}-paper`
    const hypothesisType = `${recordType}-hypothesis`

    const paperRecord = await deps.storage.readRecord({
      scope: params.scope,
      type: paperType,
      key: params.paperId,
    })
    if (!paperRecord) throw new Error(`no paper "${params.paperId}" in this project`)
    const paper = paperRecord.content as unknown as TrainingPaperRecord

    const existingRecords = await deps.storage.listRecords({
      scope: params.scope,
      type: hypothesisType,
    })
    const existing = new Map<string, TrainingHypothesis>()
    for (const r of existingRecords) {
      const content = r.content as unknown as TrainingHypothesis
      if (content && typeof content.id === 'string') existing.set(content.id, content)
    }

    // The paper's URL text is helpful extra context, but optional — a fetch failure must not abort.
    let text: string | undefined
    if (typeof paper.url === 'string' && paper.url) {
      const fetchText = params.fetchPaperText ?? fetchPaperText
      try {
        text = await fetchText(paper.url, params.abortSignal)
      } catch {
        text = undefined
      }
    }

    const res = await executor.runInference({
      systemPrompt: buildSuggestHypothesesSystemPrompt(manifest),
      userContent: buildSuggestHypothesesUserContent({
        manifest,
        paper: {
          title: paper.title,
          claim: paper.claim,
          approach: paper.approach,
          claimedMetrics: paper.claimedMetrics,
          assumptions: paper.assumptions,
          url: paper.url,
        },
        existingHypotheses: [...existing.values()].map((h) => ({
          id: h.id,
          title: h.title,
          rationale: h.rationale,
          spec: h.spec,
        })),
        text,
      }),
      model: { kind: 'api', llmConfig: params.llmConfig },
      abortSignal: params.abortSignal,
    })

    const { matchIds, newItems } = coerceSuggestedHypotheses(
      parseFirstValidJson(res.text),
      manifest,
    )
    const linkedExistingIds = matchIds.filter((id) => existing.has(id))

    // Create the new hypotheses (spec-hash id, dedup — a "new" one that already exists links instead).
    const newHypotheses: TrainingHypothesis[] = []
    const linkedIds = new Set<string>(linkedExistingIds)
    const seen = new Set<string>()
    for (const item of newItems) {
      const hid = hashTrainingConfig(item.spec as Record<string, unknown>)
      if (seen.has(hid)) continue
      seen.add(hid)
      const prior = existing.get(hid)
      if (prior) {
        linkedIds.add(hid)
        continue
      }
      const hypothesis: TrainingHypothesis = {
        id: hid,
        title: item.title,
        rationale: item.rationale,
        spec: item.spec,
        // Carry the LLM's comparison criterion (else a context-spanning compare hypothesis silently defaults
        // to beats-baseline) and its claim label (which paper claim this tests, for multi-claim scoring).
        ...(item.comparison ? { comparison: item.comparison } : {}),
        ...(item.claim ? { claim: item.claim } : {}),
        status: 'untested',
        verdictSource: 'auto',
        source: 'paper',
        proposedBy: suggestedBy,
        paperIds: [params.paperId],
        createdAt: suggestedAt,
        updatedAt: suggestedAt,
      }
      newHypotheses.push(hypothesis)
      linkedIds.add(hid)
    }

    // Persist: every linked hypothesis gets this paperId; the paper gets every linked id.
    for (const id of linkedIds) {
      const created = newHypotheses.find((h) => h.id === id)
      const base = created ?? existing.get(id)
      if (!base) continue
      const content: TrainingHypothesis = created
        ? created
        : {
            ...base,
            paperIds: Array.from(new Set([...(base.paperIds ?? []), params.paperId])),
            updatedAt: suggestedAt,
          }
      await deps.storage.upsertRecord({
        scope: params.scope,
        type: hypothesisType,
        key: id,
        content: content as unknown as Record<string, unknown>,
      })
      params.onRecordWritten?.(hypothesisType, id)
    }

    const hypothesisIds = Array.from(new Set([...(paper.hypothesisIds ?? []), ...linkedIds]))
    const updatedPaper: TrainingPaperRecord = { ...paper, hypothesisIds, updatedAt: suggestedAt }
    await deps.storage.upsertRecord({
      scope: params.scope,
      type: paperType,
      key: params.paperId,
      content: updatedPaper as unknown as Record<string, unknown>,
    })
    params.onRecordWritten?.(paperType, params.paperId)
    logger?.info('suggested paper hypotheses', {
      recordType,
      paperId: params.paperId,
      matched: linkedExistingIds.length,
      created: newHypotheses.length,
    })
    // Auto-consolidate the suggestion's OWN near-duplicates: if a suggested spec shares the same main
    // parameters as an existing hypothesis (differing only in sweep breadth), fold them into one wider
    // hypothesis instead of leaving a duplicate. Scoped to THIS suggestion's hypotheses so it never silently
    // merges unrelated registry entries.
    if (linkedIds.size) {
      await applyHypothesisConsolidation({
        scope: params.scope,
        projectRoot: params.projectRoot,
        manifestRelPath: params.manifestRelPath,
        manifest,
        restrictToIds: [...linkedIds],
        onRecordWritten: params.onRecordWritten,
      })
    }

    return {
      recordType,
      paper: updatedPaper,
      linkedExistingIds,
      newHypotheses,
      linkedHypothesisIds: [...linkedIds],
      suggestedBy,
      suggestedAt,
    }
  }

  // Re-assess the IMPORTANCE WEIGHTS of a paper's linked hypotheses with the LLM and persist them, so the
  // paper verdict re-rolls-up by importance (a central claim outweighs minor ones). Mirrors the read →
  // infer → coerce → upsert shape of suggestPaperHypotheses; only edits `weight` on existing records.
  async function weighPaperHypotheses(
    params: WeighPaperHypothesesParams,
  ): Promise<WeighPaperHypothesesResult> {
    const executor = requireInferenceExecutor()
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    const recordType = manifest.recordType
    const weighedBy = deriveModelRef({ kind: 'api', llmConfig: params.llmConfig }).label
    const weighedAt = now()
    const paperType = `${recordType}-paper`
    const hypothesisType = `${recordType}-hypothesis`

    const paperRecord = await deps.storage.readRecord({
      scope: params.scope,
      type: paperType,
      key: params.paperId,
    })
    if (!paperRecord) throw new Error(`no paper "${params.paperId}" in this project`)
    const paper = paperRecord.content as unknown as TrainingPaperRecord

    const linkedIds = Array.isArray(paper.hypothesisIds) ? paper.hypothesisIds : []
    const linked: TrainingHypothesis[] = []
    for (const id of linkedIds) {
      const r = await deps.storage.readRecord({
        scope: params.scope,
        type: hypothesisType,
        key: id,
      })
      if (r) linked.push(r.content as unknown as TrainingHypothesis)
    }
    if (!linked.length)
      return { recordType, paperId: params.paperId, weighted: [], weighedBy, weighedAt }

    // Paper text is helpful extra context but optional — a fetch failure must not abort.
    let text: string | undefined
    if (typeof paper.url === 'string' && paper.url) {
      const fetchText = params.fetchPaperText ?? fetchPaperText
      try {
        text = await fetchText(paper.url, params.abortSignal)
      } catch {
        text = undefined
      }
    }

    const res = await executor.runInference({
      systemPrompt: buildWeighHypothesesSystemPrompt(manifest),
      userContent: buildWeighHypothesesUserContent({
        paper: {
          title: paper.title,
          claim: paper.claim,
          approach: paper.approach,
          assumptions: paper.assumptions,
          claimedMetrics: paper.claimedMetrics,
          url: paper.url,
        },
        hypotheses: linked.map((h) => ({
          id: h.id,
          title: h.title,
          rationale: h.rationale,
          spec: h.spec,
        })),
        text,
      }),
      model: { kind: 'api', llmConfig: params.llmConfig },
      abortSignal: params.abortSignal,
    })

    const coverage = coerceHypothesisCoverage(
      parseFirstValidJson(res.text),
      linked.map((h) => h.id),
    )
    const byId = new Map(linked.map((h) => [h.id, h]))
    const weighted: WeighedHypothesis[] = []
    const weights: Record<string, number> = { ...(paper.hypothesisWeights ?? {}) }
    for (const row of coverage.weights) {
      const base = byId.get(row.id)
      if (!base) continue
      weights[row.id] = row.weight
      weighted.push({ id: row.id, weight: row.weight, reason: row.reason, title: base.title })
    }
    // Persist the weights on THIS PAPER (its per-hypothesis importance), NOT the shared hypothesis records
    // — the same hypothesis can be central to one paper and minor to another — PLUS the coverage gaps (claims
    // no hypothesis covers, a warning signal). Persist if EITHER changed, so a gaps-only result isn't dropped.
    if (weighted.length || coverage.uncoveredClaims.length) {
      const updatedPaper: TrainingPaperRecord = {
        ...paper,
        hypothesisWeights: weights,
        coverageGaps: coverage.uncoveredClaims,
        updatedAt: weighedAt,
      }
      await deps.storage.upsertRecord({
        scope: params.scope,
        type: paperType,
        key: params.paperId,
        content: updatedPaper as unknown as Record<string, unknown>,
      })
      params.onRecordWritten?.(paperType, params.paperId)
    }
    logger?.info('weighed paper hypotheses', {
      recordType,
      paperId: params.paperId,
      weighed: weighted.length,
      gaps: coverage.uncoveredClaims.length,
    })
    return {
      recordType,
      paperId: params.paperId,
      weighted,
      coverageGaps: coverage.uncoveredClaims,
      coverageByClaim: coverage.coverageByClaim,
      weighedBy,
      weighedAt,
    }
  }

  // Map completed-run records to the engine's AnalysisRun shape (shared by the digest + whole-space bundle).
  function recordsToAnalysisRuns(
    records: { key: string; content: Record<string, unknown> }[],
  ): AnalysisRun[] {
    return records.map((r) => ({
      key: r.key,
      config: (r.content.config as Record<string, unknown>) || {},
      metrics: r.content.metrics as Record<string, number> | undefined,
      objective: r.content.objective as number,
      durationMs: r.content.durationMs as number | undefined,
      seed: r.content.seed as number | undefined,
      dataset: r.content.dataset as AnalysisRun['dataset'],
      status: 'completed',
      ranAt:
        ((r.content.provenance as { ranAt?: string } | undefined)?.ranAt ??
          (r.content.ranAt as string | undefined)) ||
        undefined,
    }))
  }

  async function analyzeConfigSpace(
    params: AnalyzeConfigSpaceParams,
  ): Promise<AnalyzeConfigSpaceResult> {
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    const recordType = manifest.recordType
    const criterion: AnalysisCriterion = {
      key: params.criterionKey ?? 'objective',
      direction: params.criterionDir ?? manifest.objective.direction,
    }
    // Environment + dataset levers are CONTEXT — the analysis scopes within one environment over the model
    // levers, never tuning these. Derived from the manifest's per-lever `scope`.
    const contextLevers = Object.entries(manifest.levers)
      .filter(([, spec]) => spec.scope === 'environment' || spec.scope === 'dataset')
      .map(([name]) => name)
    // Conditional levers (e.g. forward_horizon only applies to supervised models) → pinned n/a where they
    // don't apply, so they can't pollute the cross-model analysis.
    const appliesWhen: Record<string, Record<string, unknown[]>> = {}
    for (const [name, spec] of Object.entries(manifest.levers)) {
      if (spec.appliesWhen) appliesWhen[name] = spec.appliesWhen
    }
    const ignoreLevers = Object.entries(manifest.levers)
      .filter(([, spec]) => spec.scope === 'ignore')
      .map(([name]) => name)
    const records = await listCompletedRuns(params.scope, recordType, true)
    const analysis = computeConfigSpaceAnalysis(recordsToAnalysisRuns(records), criterion, {
      contextLevers,
      environment: params.environment,
      appliesWhen,
      ignoreLevers,
    })
    return { recordType, criterion, analysis }
  }

  // Build ONE run's structured deterministic xAI digest — its decisions, attribution + sanity, reward
  // breakdown, latent probe, the sibling decision-diff, and its standing among all completed runs. Shared by
  // the LLM narrative (xaiNarrate) and the agent-facing getRunXAI tool, so the facts live in ONE place.
  async function buildRunXaiDigest(
    scope: string,
    manifest: TrainerManifest,
    runKey: string,
    criterion: AnalysisCriterion,
    siblingKey?: string,
  ): Promise<{ digest: RunXaiDigest; runCount: number }> {
    // Rank/importances read the LEAN run set (heavy fields omitted) so this never bulk-loads every run's
    // decision trace; the trace-based parts re-read the focus (and sibling) run's FULL record by key.
    const recordType = manifest.recordType
    const records = await listCompletedRuns(scope, recordType, true)
    const focus = records.find((r) => r.key === runKey)
    if (!focus) throw new Error(`run "${runKey}" is not a completed run of this project`)
    const focusFull = await deps.storage.readRecord({ scope, type: recordType, key: runKey })
    const focusContent = (focusFull?.content ?? focus.content) as Record<string, unknown>
    const ignoreLevers = Object.entries(manifest.levers)
      .filter(([, spec]) => spec.scope === 'ignore')
      .map(([name]) => name)
    // Environment/dataset levers (e.g. transaction_fee, asset) DEFINE the environment — crucial constants,
    // never tunable knobs. They're excluded from the run's lever-importance analysis (you compare
    // environments, you don't tune them); `ignore` levers (device) are excluded everywhere.
    const nonModelLevers = Object.entries(manifest.levers)
      .filter(
        ([, spec]) =>
          spec.scope === 'environment' || spec.scope === 'dataset' || spec.scope === 'ignore',
      )
      .map(([name]) => name)
    const stripBy = (cfg: Record<string, unknown>, levers: string[]): Record<string, unknown> => {
      if (!levers.length) return cfg
      const out = { ...cfg }
      for (const k of levers) delete out[k]
      return out
    }
    // focusConfig keeps context levers (so a lever sweep can hold this run's environment fixed); drop only `ignore`.
    const focusConfig = stripBy(
      (focusContent.config as Record<string, unknown>) || {},
      ignoreLevers,
    )

    // The analysed run set drops context + ignore levers, so importances reflect only tunable MODEL knobs.
    const runs = recordsToAnalysisRuns(records).map((r) => ({
      ...r,
      config: stripBy(r.config, nonModelLevers),
    }))
    const ranked = runs
      .map((r) => ({ key: r.key, value: criterionValueOf(r, criterion) }))
      .filter((x): x is { key: string; value: number } => x.value !== undefined)
      .sort((a, b) => (criterion.direction === 'max' ? b.value - a.value : a.value - b.value))
    const rankPos = ranked.findIndex((x) => x.key === runKey)

    const trace = validateDecisionTrace(
      (focusContent.artifacts as { decisionTrace?: unknown } | undefined)?.decisionTrace,
    )
    const fa = trace?.featureAttribution
    const topGroups: [string, number][] = fa?.byGroup
      ? Object.entries(fa.byGroup)
          .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
          .slice(0, 5)
      : []
    const stepAttribution = trace ? summarizeStepAttribution(trace) : undefined
    const driverCounts: [string, number][] | undefined = stepAttribution
      ? Object.entries(stepAttribution.dominanceCounts)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 5)
      : undefined
    const probe = trace?.latentMap?.probe

    let sibling: RunXaiDigest['sibling']
    const sib = siblingKey ? records.find((r) => r.key === siblingKey) : undefined
    const sibFull = sib
      ? await deps.storage.readRecord({ scope, type: recordType, key: siblingKey as string })
      : undefined
    if (sib && sibFull) {
      const diff = diffDecisionTraces(
        sibFull.content as unknown as TrainingRunSummary,
        focusContent as unknown as TrainingRunSummary,
      )
      if (diff && diff.aligned) {
        const sibConfig = stripBy(
          (sib.content.config as Record<string, unknown>) || {},
          ignoreLevers,
        )
        const changed =
          Object.keys(manifest.levers)
            .filter((lk) => String(sibConfig[lk]) !== String(focusConfig[lk]))
            .map((lk) => `${lk} ${sibConfig[lk]}→${focusConfig[lk]}`)
            .join(', ') || 'seed/nondeterminism'
        sibling = {
          key: siblingKey as string,
          changed,
          divergencePct: Math.round(diff.divergenceRate * 100),
          qualityVerdict: diff.quality.verdict,
          qualitySummary: diff.quality.summary,
        }
      }
    }

    const digest: RunXaiDigest = {
      runKey,
      config: focusConfig,
      objective: focusContent.objective as number | undefined,
      criterion,
      rank: rankPos >= 0 ? { position: rankPos + 1, total: ranked.length } : undefined,
      actionCounts: trace?.actionCounts,
      attribution:
        fa || (driverCounts && driverCounts.length)
          ? {
              topGroups,
              ...(fa
                ? {
                    method: fa.method,
                    sanityPassed: fa.sanityCheck?.passed,
                    sanityRankCorr: fa.sanityCheck?.rankCorrelation,
                  }
                : {}),
              ...(driverCounts && driverCounts.length ? { driverCounts } : {}),
            }
          : undefined,
      rewardBreakdown: trace?.rewardBreakdown,
      latent: trace?.latentMap
        ? {
            varianceExplained: trace.latentMap.varianceExplained,
            probeAccuracy: probe?.accuracy,
            probeBaseline: probe?.baseline,
          }
        : undefined,
      importances: leverImportances(runs, criterion),
      sibling,
    }
    return { digest, runCount: runs.length }
  }

  // Resolve which registered training project a run id belongs to: search the host's
  // `trainer-project-manifest` records for one whose recordType holds a COMPLETED record keyed by `runKey`.
  async function resolveRunRecord(
    scope: string,
    runKey: string,
  ): Promise<
    { recordType: string; manifest: TrainerManifest; content: Record<string, unknown> } | undefined
  > {
    const manifests = await deps.storage.listRecords({ scope, type: 'trainer-project-manifest' })
    for (const mr of manifests) {
      const manifest = (mr.content as { manifest?: TrainerManifest } | undefined)?.manifest
      if (!manifest?.recordType) continue
      const rec = await deps.storage.readRecord({ scope, type: manifest.recordType, key: runKey })
      if (rec && (rec.content as { status?: string })?.status === 'completed') {
        return {
          recordType: manifest.recordType,
          manifest,
          content: rec.content as Record<string, unknown>,
        }
      }
    }
    return undefined
  }

  // Strip the heavy parts of a stored run summary (the per-step decision trace + the chart series) so an
  // agent gets a compact record; leave a small decision-trace digest in their place.
  function trimRunForAgent(content: Record<string, unknown>): Record<string, unknown> {
    const out: Record<string, unknown> = { ...content }
    delete out.series
    const artifacts = content.artifacts as { decisionTrace?: unknown } | undefined
    const trace = validateDecisionTrace(artifacts?.decisionTrace)
    if (artifacts) {
      const rest = { ...artifacts }
      delete rest.decisionTrace
      out.artifacts = rest
    }
    if (trace) {
      const fa = trace.featureAttribution
      out.decisionTraceDigest = {
        totalSteps: trace.totalSteps ?? trace.steps.length,
        actionCounts: trace.actionCounts,
        ...(fa
          ? { attributionMethod: fa.method, attributionSanityPassed: fa.sanityCheck?.passed }
          : {}),
        ...(trace.rewardBreakdown ? { rewardBreakdown: trace.rewardBreakdown } : {}),
        ...(trace.latentMap?.probe ? { latentProbeAccuracy: trace.latentMap.probe.accuracy } : {}),
      }
    }
    return out
  }

  async function benchmarkModelDevice(
    params: BenchmarkModelDeviceParams,
  ): Promise<BenchmarkModelDeviceResult> {
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    if (!manifest.benchmarkDevice) {
      throw new Error('trainer manifest declares no benchmarkDevice command')
    }
    const recordType = manifest.recordType
    const modelType = `${recordType}-model`
    const record = await deps.storage.readRecord({
      scope: params.scope,
      type: modelType,
      key: params.modelId,
    })
    if (!record) throw new Error(`no model record for id ${params.modelId}`)
    const model = record.content as unknown as TrainingModel
    const modelName = params.modelName ?? model.flavors?.[0]?.modelName ?? model.slug
    // Reuse the calibrate runner contract: one probe runs the benchmark command (which times BOTH devices
    // itself) with the model named via env, and returns the parsed {summaryOut} for us to read back.
    const probe = await resolveRunner(params.computeTarget).calibrate({
      repoRef: { kind: 'local', localPath: params.projectRoot },
      commandTemplate: manifest.benchmarkDevice,
      dataFiles: manifestDataFiles(manifest),
      env: { BENCH_MODEL_NAME: modelName },
      abortSignal: params.abortSignal,
    })
    const deviceBenchmark = parseDeviceBenchmark(probe.summary, now())
    const updated: TrainingModel = {
      ...model,
      preferredDevice: deviceBenchmark.bestDevice,
      deviceBenchmark,
      updatedAt: now(),
    }
    await deps.storage.upsertRecord({
      scope: params.scope,
      type: modelType,
      key: params.modelId,
      content: updated as unknown as Record<string, unknown>,
    })
    params.onRecordWritten?.(modelType, params.modelId)
    return {
      recordType,
      modelId: params.modelId,
      preferredDevice: deviceBenchmark.bestDevice,
      deviceBenchmark,
    }
  }

  async function scanProjectModels(
    params: ScanProjectModelsParams,
  ): Promise<ScanProjectModelsResult> {
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    const recordType = manifest.recordType
    const modelType = `${recordType}-model`
    const scannedAt = now()

    const existingRecords = await deps.storage.listRecords({ scope: params.scope, type: modelType })
    const existingModels = existingRecords
      .map((r) => r.content as unknown as TrainingModel)
      .filter((c): c is TrainingModel => !!c && typeof c.slug === 'string')

    const candidates = discoverManifestModelCandidates(manifest, existingModels)
    const lever = manifest.levers?.model_name as TrainerLeverSpec | undefined
    const totalChoices =
      lever && lever.type === 'choice' && Array.isArray(lever.choices)
        ? lever.choices.filter((c): c is string => typeof c === 'string' && !!c).length
        : 0
    const skippedExisting = Math.max(0, totalChoices - candidates.length)

    let enrichments = new Map<
      string,
      {
        name?: string
        description?: string
        category?: TrainingModel['category']
        paperIds?: string[]
      }
    >()
    let scannedBy: string | undefined
    if (params.llmConfig && candidates.length) {
      const executor = requireInferenceExecutor()
      scannedBy = deriveModelRef({ kind: 'api', llmConfig: params.llmConfig }).label
      const paperRecords = await deps.storage.listRecords({
        scope: params.scope,
        type: `${recordType}-paper`,
      })
      const papers = paperRecords
        .map((r) => r.content as unknown as TrainingPaperRecord)
        .filter((p): p is TrainingPaperRecord => !!p && typeof p.id === 'string')
        .map((p) => ({ id: p.id, title: p.title, claim: p.claim }))
      const res = await executor.runInference({
        systemPrompt: buildScanModelsSystemPrompt(manifest),
        userContent: buildScanModelsUserContent({
          candidates,
          papers,
          leverDescription: lever?.description,
        }),
        model: { kind: 'api', llmConfig: params.llmConfig },
        abortSignal: params.abortSignal,
      })
      enrichments = coerceScannedModels(
        parseFirstValidJson(res.text),
        new Set(candidates.map((c) => c.slug)),
        new Set(papers.map((p) => p.id)),
      )
    }

    const models: TrainingModel[] = []
    for (const c of candidates) {
      const e = enrichments.get(c.slug) ?? {}
      const model: TrainingModel = {
        id: c.slug,
        slug: c.slug,
        name: e.name ?? c.name,
        description: e.description ?? '',
        category: e.category ?? c.category,
        status: 'implemented',
        statusSource: 'auto',
        flavors: [{ modelName: c.modelName }],
        ...(e.paperIds && e.paperIds.length ? { paperIds: e.paperIds } : {}),
        source: params.llmConfig ? 'llm' : 'scan',
        ...(scannedBy ? { proposedBy: scannedBy } : {}),
        createdAt: scannedAt,
        updatedAt: scannedAt,
      }
      await deps.storage.upsertRecord({
        scope: params.scope,
        type: modelType,
        key: c.slug,
        content: model as unknown as Record<string, unknown>,
      })
      params.onRecordWritten?.(modelType, c.slug)
      models.push(model)
    }
    logger?.info('scanned project models', {
      recordType,
      discovered: candidates.length,
      created: models.length,
      skippedExisting,
    })
    return {
      recordType,
      discovered: candidates.length,
      created: models.length,
      skippedExisting,
      models,
      scannedBy,
      scannedAt,
    }
  }

  async function analyzePaperModels(
    params: AnalyzePaperModelsParams,
  ): Promise<AnalyzePaperModelsResult> {
    const executor = requireInferenceExecutor()
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    const recordType = manifest.recordType
    const paperType = `${recordType}-paper`
    const modelType = `${recordType}-model`
    const analyzedBy = deriveModelRef({ kind: 'api', llmConfig: params.llmConfig }).label
    const analyzedAt = now()

    const paperRecord = await deps.storage.readRecord({
      scope: params.scope,
      type: paperType,
      key: params.paperId,
    })
    if (!paperRecord) throw new Error(`no paper "${params.paperId}" in this project`)
    const paper = paperRecord.content as unknown as TrainingPaperRecord

    const existingRecords = await deps.storage.listRecords({ scope: params.scope, type: modelType })
    const existing = new Map<string, TrainingModel>()
    for (const r of existingRecords) {
      const content = r.content as unknown as TrainingModel
      if (content && typeof content.id === 'string') existing.set(content.id, content)
    }

    let text: string | undefined
    if (typeof paper.url === 'string' && paper.url) {
      const fetchText = params.fetchPaperText ?? fetchPaperText
      try {
        text = await fetchText(paper.url, params.abortSignal)
      } catch {
        text = undefined
      }
    }

    const res = await executor.runInference({
      systemPrompt: buildAnalyzePaperModelsSystemPrompt(manifest),
      userContent: buildAnalyzePaperModelsUserContent({
        paper: {
          title: paper.title,
          claim: paper.claim,
          approach: paper.approach,
          url: paper.url,
        },
        existingModels: [...existing.values()].map((m) => ({
          id: m.id,
          name: m.name,
          slug: m.slug,
          category: m.category,
          modelNames: modelBindingNames(m),
          aliases: m.aliases,
        })),
        text,
      }),
      model: { kind: 'api', llmConfig: params.llmConfig },
      abortSignal: params.abortSignal,
    })

    const { matchModelIds, proposedModels, proposedImprovements } = coerceAnalyzedPaperModels(
      parseFirstValidJson(res.text),
    )
    const linkedModelIds = matchModelIds.filter((id) => existing.has(id))

    for (const id of linkedModelIds) {
      const m = existing.get(id)!
      const paperIds = Array.from(new Set([...(m.paperIds ?? []), params.paperId]))
      if (paperIds.length !== (m.paperIds?.length ?? 0)) {
        const updated: TrainingModel = { ...m, paperIds, updatedAt: analyzedAt }
        await deps.storage.upsertRecord({
          scope: params.scope,
          type: modelType,
          key: id,
          content: updated as unknown as Record<string, unknown>,
        })
        params.onRecordWritten?.(modelType, id)
      }
    }

    const missingModels = detectMissingPaperModels(proposedModels, [...existing.values()])

    // Merge (not overwrite) so a user's `inapplicable` mark survives a re-run: a re-proposed item stays
    // inapplicable, and an inapplicable item this run dropped is kept + still listed for reference.
    const mergedImprovements = mergeProposedImprovements(
      paper.proposedImprovements,
      proposedImprovements,
    )

    const modelIds = Array.from(new Set([...(paper.modelIds ?? []), ...linkedModelIds]))
    const updatedPaper: TrainingPaperRecord = {
      ...paper,
      modelIds,
      proposedImprovements: mergedImprovements,
      updatedAt: analyzedAt,
    }
    await deps.storage.upsertRecord({
      scope: params.scope,
      type: paperType,
      key: params.paperId,
      content: updatedPaper as unknown as Record<string, unknown>,
    })
    params.onRecordWritten?.(paperType, params.paperId)
    logger?.info('analyzed paper models', {
      recordType,
      paperId: params.paperId,
      linked: linkedModelIds.length,
      missing: missingModels.length,
    })
    return {
      recordType,
      paper: updatedPaper,
      linkedModelIds,
      missingModels,
      proposedImprovements: mergedImprovements,
      analyzedBy,
      analyzedAt,
    }
  }

  async function consolidateModels(
    params: ConsolidateModelsParams,
  ): Promise<ConsolidateModelsResult> {
    const executor = requireInferenceExecutor()
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    const recordType = manifest.recordType
    const modelType = `${recordType}-model`
    const proposedBy = deriveModelRef({ kind: 'api', llmConfig: params.llmConfig }).label
    const proposedAt = now()

    const records = await deps.storage.listRecords({ scope: params.scope, type: modelType })
    const models: TrainingModel[] = []
    for (const r of records) {
      const content = r.content as unknown as TrainingModel
      if (content && typeof content.id === 'string' && !content.dismissed) models.push(content)
    }
    // Nothing to merge with fewer than two models — skip the LLM call entirely.
    if (models.length < 2) {
      return { recordType, groups: [], modelCount: models.length, proposedBy, proposedAt }
    }

    const res = await executor.runInference({
      systemPrompt: buildConsolidateModelsSystemPrompt(manifest),
      userContent: buildConsolidateModelsUserContent({
        models: models.map((m) => ({
          id: m.id,
          name: m.name,
          slug: m.slug,
          category: m.category,
          description: m.description,
          modelNames: modelBindingNames(m),
          aliases: m.aliases,
        })),
      }),
      model: { kind: 'api', llmConfig: params.llmConfig },
      abortSignal: params.abortSignal,
    })

    const groups = coerceConsolidationGroups(
      parseFirstValidJson(res.text),
      models.map((m) => m.id),
    )
    logger?.info('proposed model consolidations', {
      recordType,
      modelCount: models.length,
      groups: groups.length,
    })
    return { recordType, groups, modelCount: models.length, proposedBy, proposedAt }
  }

  // Deterministically fold hypotheses that share the same MAIN PARAMETERS (and differ only in sweep breadth)
  // into one wider hypothesis — repointing paper/model links + per-paper weights, deleting the absorbed
  // records. Shared by the manual "Consolidate" action and the auto-pass after suggest (the latter scoped via
  // `restrictToIds`). Pure grouping/planning in modelTrainerUtils; this performs the ordered writes.
  async function applyHypothesisConsolidation(
    params: ConsolidateHypothesesParams,
  ): Promise<ConsolidateHypothesesResult> {
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    const recordType = manifest.recordType
    const hypothesisType = `${recordType}-hypothesis`
    const paperType = `${recordType}-paper`
    const modelType = `${recordType}-model`
    const consolidatedAt = now()

    const hyps = (await deps.storage.listRecords({ scope: params.scope, type: hypothesisType }))
      .map((r) => r.content as unknown as TrainingHypothesis)
      .filter((h) => h && typeof h.id === 'string')
    const hypsById = new Map(hyps.map((h) => [h.id, h]))
    const papersById = new Map(
      (await deps.storage.listRecords({ scope: params.scope, type: paperType }))
        .map((r) => r.content as unknown as TrainingPaperRecord)
        .filter((p) => p && typeof p.id === 'string')
        .map((p) => [p.id, p] as const),
    )
    const modelsById = new Map(
      (await deps.storage.listRecords({ scope: params.scope, type: modelType }))
        .map((r) => r.content as unknown as TrainingModel)
        .filter((m) => m && typeof m.id === 'string')
        .map((m) => [m.id, m] as const),
    )

    const restrict = params.restrictToIds ? new Set(params.restrictToIds) : null
    const groups = groupHypothesesForConsolidation(hyps, hashTrainingConfig).filter(
      (g) => !restrict || g.members.some((m) => restrict.has(m.id)),
    )

    const merged: ConsolidatedHypothesisGroup[] = []
    const conflicts: { ids: string[] }[] = []
    const changedHyps = new Map<string, TrainingHypothesis>()
    const changedPapers = new Map<string, TrainingPaperRecord>()
    const changedModels = new Map<string, TrainingModel>()
    const toDelete = new Set<string>()

    for (const group of groups) {
      const members = group.members
        .map((m) => hypsById.get(m.id))
        .filter((m): m is TrainingHypothesis => !!m)
      if (members.length < 2) continue
      const plan = planHypothesisConsolidation(
        { members },
        [...papersById.values()],
        [...modelsById.values()],
        consolidatedAt,
        hashTrainingConfig,
      )
      if (!plan) continue
      if ('skipped' in plan) {
        conflicts.push({ ids: plan.members })
        continue
      }
      // Apply to the in-memory state so a later group sees the already-repointed papers/models.
      hypsById.set(plan.unionRecord.id, plan.unionRecord)
      changedHyps.set(plan.unionRecord.id, plan.unionRecord)
      toDelete.delete(plan.unionRecord.id)
      for (const p of plan.changedPapers) {
        papersById.set(p.id, p)
        changedPapers.set(p.id, p)
      }
      for (const m of plan.changedModels) {
        modelsById.set(m.id, m)
        changedModels.set(m.id, m)
      }
      for (const id of plan.deletedIds) {
        hypsById.delete(id)
        changedHyps.delete(id)
        toDelete.add(id)
      }
      merged.push({
        mergedId: plan.unionRecord.id,
        absorbedIds: plan.deletedIds,
        title: plan.unionRecord.title,
      })
    }

    // Write papers + models (the repoints) FIRST, then the union hypotheses, then DELETE absorbed records
    // last — so a mid-sequence failure never leaves a paper/model pointing at a deleted hypothesis.
    for (const p of changedPapers.values()) {
      await deps.storage.upsertRecord({
        scope: params.scope,
        type: paperType,
        key: p.id,
        content: p as unknown as Record<string, unknown>,
      })
      params.onRecordWritten?.(paperType, p.id)
    }
    for (const m of changedModels.values()) {
      await deps.storage.upsertRecord({
        scope: params.scope,
        type: modelType,
        key: m.id,
        content: m as unknown as Record<string, unknown>,
      })
      params.onRecordWritten?.(modelType, m.id)
    }
    for (const h of changedHyps.values()) {
      await deps.storage.upsertRecord({
        scope: params.scope,
        type: hypothesisType,
        key: h.id,
        content: h as unknown as Record<string, unknown>,
      })
      params.onRecordWritten?.(hypothesisType, h.id)
    }
    for (const id of toDelete) {
      if (changedHyps.has(id)) continue
      await deps.storage.deleteRecord({ scope: params.scope, type: hypothesisType, key: id })
      params.onRecordWritten?.(hypothesisType, id)
    }

    logger?.info('consolidated hypotheses', {
      recordType,
      merged: merged.length,
      absorbed: merged.reduce((n, g) => n + g.absorbedIds.length, 0),
      conflicts: conflicts.length,
    })
    return { recordType, merged, conflicts, hypothesisCount: hyps.length, consolidatedAt }
  }

  async function consolidateHypotheses(
    params: ConsolidateHypothesesParams,
  ): Promise<ConsolidateHypothesesResult> {
    return applyHypothesisConsolidation(params)
  }

  async function xaiNarrate(params: XaiNarrateParams): Promise<XaiNarrateResult> {
    const executor = requireInferenceExecutor()
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    const recordType = manifest.recordType
    const narratedBy = deriveModelRef({ kind: 'api', llmConfig: params.llmConfig }).label
    const narratedAt = now()
    const criterion = params.criterion ?? {
      key: 'objective',
      direction: manifest.objective.direction,
      label: manifest.objective.name,
    }
    const { digest, runCount } = await buildRunXaiDigest(
      params.scope,
      manifest,
      params.runKey,
      criterion,
      params.siblingKey,
    )
    const res = await executor.runInference({
      systemPrompt: buildXaiNarrateSystemPrompt(manifest),
      userContent: buildXaiNarrateUserContent(digest),
      model: { kind: 'api', llmConfig: params.llmConfig },
      abortSignal: params.abortSignal,
    })

    const narrativeType = `${recordType}-xai-narrative`
    await deps.storage.upsertRecord({
      scope: params.scope,
      type: narrativeType,
      key: params.runKey,
      content: {
        narrative: String(res.text || '').trim(),
        runKey: params.runKey,
        runCount,
        criterionKey: criterion.key,
        narratedBy,
        narratedAt,
      },
    })
    params.onRecordWritten?.(narrativeType, params.runKey)
    logger?.info('narrated xAI run', { recordType, runKey: params.runKey, runCount })
    return { recordType, runKey: params.runKey, runCount, narratedBy, narratedAt }
  }

  async function getRunData(params: GetRunDataParams): Promise<GetRunDataResult> {
    const resolved = await resolveRunRecord(params.scope, params.runKey)
    if (!resolved)
      return { found: false, error: `no completed run "${params.runKey}" found in this project` }
    return { found: true, recordType: resolved.recordType, run: trimRunForAgent(resolved.content) }
  }

  async function getRunXAI(params: GetRunXaiParams): Promise<GetRunXaiResult> {
    const resolved = await resolveRunRecord(params.scope, params.runKey)
    if (!resolved)
      return { found: false, error: `no completed run "${params.runKey}" found in this project` }
    const criterion = params.criterion ?? {
      key: 'objective',
      direction: resolved.manifest.objective.direction,
      label: resolved.manifest.objective.name,
    }
    const { digest, runCount } = await buildRunXaiDigest(
      params.scope,
      resolved.manifest,
      params.runKey,
      criterion,
      params.siblingKey,
    )
    return { found: true, recordType: resolved.recordType, runCount, analysis: digest }
  }

  async function migrateTrainingRuns(
    params: MigrateTrainingRunsParams,
  ): Promise<MigrateTrainingRunsResult> {
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    const recordType = manifest.recordType
    const rules = manifest.migrations ?? []
    // Conditional levers that don't apply (e.g. forward_horizon on a non-supervised model) are pinned to
    // 'n/a' on EVERY run/queue config — independently of the manifest's rule-based migrations — so stored
    // data stays canonical and xAI never reasons over a value a model ignores.
    const appliesWhen = appliesWhenMap(manifest)
    const hasConditional = Object.keys(appliesWhen).length > 0
    let examinedRuns = 0
    let migratedRuns = 0
    let deletedRuns = 0
    let examinedQueue = 0
    let migratedQueue = 0
    let deletedQueue = 0
    if (rules.length === 0 && !hasConditional) {
      return {
        recordType,
        examinedRuns,
        migratedRuns,
        deletedRuns,
        examinedQueue,
        migratedQueue,
        deletedQueue,
      }
    }

    // Delete a run AND its derived records (so nothing orphans), broadcasting each actual removal.
    const deleteRunAndDerived = async (runKey: string, setupKey: unknown): Promise<void> => {
      await deps.storage.deleteRecord({ scope: params.scope, type: recordType, key: runKey })
      params.onRecordWritten?.(recordType, runKey)
      for (const suffix of RUN_KEYED_CHILD_SUFFIXES) {
        const childType = recordType + suffix
        const removed = await deps.storage.deleteRecord({
          scope: params.scope,
          type: childType,
          key: runKey,
        })
        if (removed) params.onRecordWritten?.(childType, runKey)
      }
      if (typeof setupKey === 'string' && setupKey) {
        const unrunnableType = `${recordType}-unrunnable`
        const removed = await deps.storage.deleteRecord({
          scope: params.scope,
          type: unrunnableType,
          key: setupKey,
        })
        if (removed) params.onRecordWritten?.(unrunnableType, setupKey)
      }
    }

    // Scan LEAN (heavy fields omitted): this runs at boot over EVERY run, so materializing each run's
    // series/trace at once would OOM the process. Only the DECISION needs light fields (config/setupKey).
    const runRecords = await deps.storage.listRecords({
      scope: params.scope,
      type: recordType,
      omit: HEAVY_RUN_FIELDS,
    })
    for (const record of runRecords) {
      examinedRuns++
      const content = (record.content ?? {}) as Record<string, unknown>
      const config = content.config
      if (!record.key || !config || typeof config !== 'object') continue
      const cfg = config as Record<string, unknown>
      const rule = findMigrationRule(cfg, rules)
      if (rule?.delete) {
        await deleteRunAndDerived(record.key, content.setupKey)
        deletedRuns++
        continue
      }
      // Apply any matching rule, then pin inapplicable conditional levers to 'n/a'. Both passes converge,
      // so a config that's already migrated + normalised hashes identically and is left untouched.
      const ruled = rule ? (applyMigrationRules(cfg, rules) ?? cfg) : cfg
      const next = normalizeConditionalLevers(ruled, appliesWhen)
      if (hashTrainingConfig(next) === hashTrainingConfig(cfg)) continue
      // This run's config changes → rewrite. Read the FULL record by key (one at a time) so the omitted
      // heavy fields are preserved. setupKey stays RAW (from the pre-normalize config), exactly as the
      // write path + isFresh compute it, so canonicalising never desyncs the skipExplored/unrunnable dedup.
      const full = await deps.storage.readRecord({ scope: params.scope, type: recordType, key: record.key })
      const fullContent = (full?.content ?? content) as Record<string, unknown>
      await deps.storage.upsertRecord({
        scope: params.scope,
        type: recordType,
        key: record.key,
        content: { ...fullContent, config: next, setupKey: setupKeyOf(ruled) },
      })
      migratedRuns++
      params.onRecordWritten?.(recordType, record.key)
    }

    if (params.queueRecordType) {
      const queueRecords = await deps.storage.listRecords({
        scope: params.scope,
        type: params.queueRecordType,
      })
      for (const record of queueRecords) {
        examinedQueue++
        const content = (record.content ?? {}) as Record<string, unknown>
        const itemParams = content.params as Record<string, unknown> | undefined
        const spec = itemParams?.spec as { fixed?: Record<string, unknown> } | undefined
        const fixed = spec?.fixed
        if (!record.key || !fixed || typeof fixed !== 'object') continue
        const rule = findMigrationRule(fixed, rules)
        if (!rule) continue
        if (rule.delete) {
          await deps.storage.deleteRecord({
            scope: params.scope,
            type: params.queueRecordType,
            key: record.key,
          })
          deletedQueue++
          params.onRecordWritten?.(params.queueRecordType, record.key)
          continue
        }
        const migrated = applyMigrationRules(fixed, rules)
        if (!migrated) continue
        if (hashTrainingConfig(migrated) === hashTrainingConfig(fixed)) continue
        await deps.storage.upsertRecord({
          scope: params.scope,
          type: params.queueRecordType,
          key: record.key,
          content: { ...content, params: { ...itemParams, spec: { ...spec, fixed: migrated } } },
        })
        migratedQueue++
        params.onRecordWritten?.(params.queueRecordType, record.key)
      }
    }

    return {
      recordType,
      examinedRuns,
      migratedRuns,
      deletedRuns,
      examinedQueue,
      migratedQueue,
      deletedQueue,
    }
  }

  async function invalidateRuns(params: InvalidateRunsParams): Promise<InvalidateRunsResult> {
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    const recordType = manifest.recordType
    const majorOf = (v: unknown) => parseInt(String(v ?? '1'), 10) || 1
    let examinedRuns = 0
    let invalidatedRuns = 0
    let examinedQueue = 0
    let cancelledQueue = 0

    // Stamp affected, pre-fix runs as status='invalid'. Version-gated + skips already-invalid records, so
    // this is idempotent and never re-flags a re-run produced at/after the fix major.
    // Scan LEAN (heavy fields omitted) — the boot-time invalidation sweep touches EVERY run, so loading
    // every run's series/trace at once would OOM the process; the decision needs only light fields.
    const runRecords = await deps.storage.listRecords({
      scope: params.scope,
      type: recordType,
      omit: HEAVY_RUN_FIELDS,
    })
    for (const record of runRecords) {
      examinedRuns++
      const content = (record.content ?? {}) as Record<string, unknown>
      const config = content.config as Record<string, unknown> | undefined
      if (!record.key || !config || typeof config !== 'object') continue
      if (content.status === 'invalid') continue
      if (majorOf(content.pipelineVersion) >= params.beforePipelineMajor) continue
      if (!params.affectsRun(config)) continue
      // Read the FULL record by key (one at a time) so the omitted heavy fields survive the status stamp.
      const full = await deps.storage.readRecord({ scope: params.scope, type: recordType, key: record.key })
      const fullContent = (full?.content ?? content) as Record<string, unknown>
      await deps.storage.upsertRecord({
        scope: params.scope,
        type: recordType,
        key: record.key,
        content: {
          ...fullContent,
          status: 'invalid',
          invalidReason: params.reason,
          invalidatedBy: params.invalidationId,
          priorStatus: content.status ?? null,
          invalidatedAt: now(),
        },
      })
      invalidatedRuns++
      params.onRecordWritten?.(recordType, record.key)
    }

    // One-time cancellation of affected PENDING items, guarded by a marker keyed on invalidationId so a
    // post-fix reboot never cancels the re-runs the user has since queued.
    let pendingAlreadyApplied = false
    if (params.cancelPendingQueue && params.queueRecordType && params.affectsPending) {
      const markerType = `${recordType}-invalidation`
      const marker = await deps.storage.readRecord({
        scope: params.scope,
        type: markerType,
        key: params.invalidationId,
      })
      if (marker) {
        pendingAlreadyApplied = true
      } else {
        const queueRecords = await deps.storage.listRecords({
          scope: params.scope,
          type: params.queueRecordType,
        })
        for (const record of queueRecords) {
          examinedQueue++
          const content = (record.content ?? {}) as Record<string, unknown>
          const spec = (content.params as Record<string, unknown> | undefined)?.spec as
            | Record<string, unknown>
            | undefined
          if (!record.key || !spec || !params.affectsPending(spec)) continue
          await deps.storage.deleteRecord({
            scope: params.scope,
            type: params.queueRecordType,
            key: record.key,
          })
          cancelledQueue++
          params.onRecordWritten?.(params.queueRecordType, record.key)
        }
        await deps.storage.upsertRecord({
          scope: params.scope,
          type: markerType,
          key: params.invalidationId,
          content: { invalidationId: params.invalidationId, appliedAt: now(), cancelledQueue },
        })
        params.onRecordWritten?.(markerType, params.invalidationId)
      }
    }

    return {
      recordType,
      examinedRuns,
      invalidatedRuns,
      examinedQueue,
      cancelledQueue,
      pendingAlreadyApplied,
    }
  }

  return {
    readTrainerManifest: (projectRoot, manifestRelPath) =>
      readTrainerManifest(projectRoot, manifestRelPath),
    planTrainingMatrix: (manifest: TrainerManifest, spec: ExperimentSpec) =>
      expandExperimentMatrix(manifest, spec, hashTrainingConfig),
    calibrateTrainingThroughput,
    runTrainingCampaign,
    evaluateTrainingRun,
    evaluateTrainingRuns,
    judgeTrainingRuns,
    proposeTrainingHypotheses,
    proposeTrainingExperiments,
    analyzePaperFromUrl,
    researchTrainingPapers,
    suggestPaperHypotheses,
    weighPaperHypotheses,
    scanProjectModels,
    consolidateModels,
    consolidateHypotheses,
    benchmarkModelDevice,
    analyzePaperModels,
    xaiNarrate,
    getRunData,
    getRunXAI,
    analyzeConfigSpace,
    migrateTrainingRuns,
    invalidateRuns,
  }
}
