import type { ComputeRepoRef, ComputeRunner, InferenceExecutor } from 'thefactory-tools/types'
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
  AnalyzePaperFromUrlParams,
  AnalyzePaperFromUrlResult,
  AnalyzePaperModelsParams,
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
  ModelTrainerTools,
  ModelTrainerToolsDeps,
  PlannedTrainingItem,
  ProposeTrainingHypothesesParams,
  ProposeTrainingHypothesesResult,
  RunXaiDigest,
  ScanProjectModelsParams,
  ScanProjectModelsResult,
  SuggestPaperHypothesesParams,
  SuggestPaperHypothesesResult,
  TrainerLeverSpec,
  TrainerManifest,
  TrainingModel,
  TrainingCalibration,
  TrainingCampaignParams,
  TrainingCampaignProgress,
  TrainingCampaignResult,
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
  JUDGE_LLM_WEIGHT,
  MAX_JUDGE_RUNS,
} from './modelTrainerConstants.js'
import {
  fetchPaperText,
  hashTrainingConfig,
  readTrainerManifest,
  setupKeyOf,
} from './modelTrainerHelpers.js'
import {
  applyMigrationRules,
  blendJudgeScore,
  findMigrationRule,
  manifestDataFiles,
  migrateExperimentSpec,
  parseProgressMarker,
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
  buildXaiNarrateSystemPrompt,
  buildXaiNarrateUserContent,
  coerceAnalyzedPaperModels,
  coerceHypothesisItems,
  coercePaperDraft,
  coerceScannedModels,
  coerceSuggestedHypotheses,
  coerceVerdictRows,
  detectMissingPaperModels,
  diffDecisionTraces,
  discoverManifestModelCandidates,
  expandExperimentMatrix,
  normalizeObjectiveScores,
  pickBestRun,
  totalCampaignUnits,
  validateDecisionTrace,
  validateTrainingRunSummary,
} from './modelTrainerUtils.js'
import { criterionValueOf, leverImportances } from './xaiUtils.js'

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
    const repoRef: ComputeRepoRef = { kind: 'local', localPath: params.projectRoot }
    const runner = resolveRunner(params.computeTarget)
    const dataFiles = manifestDataFiles(manifest)
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
      const priorRecords = await deps.storage.listRecords({ scope: params.scope, type: recordType })
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
    // Wall-clock anchor for an ETA fallback: when there's no calibration (most projects
    // declare no `calibrate`), estimate remaining time from how long the completed items
    // have actually taken — so the UI shows a moving estimate instead of just a sweeping bar.
    const campaignStartMs = Date.parse(now())
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
      concurrency: params.concurrency,
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
        const handle = runner.runJob({
          jobId: item.key,
          repoRef,
          commandTemplate: manifest.run,
          config: item.config,
          dataFiles,
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
                config: item.config,
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
        const runSummary = validateTrainingRunSummary(result.summary)
        const artifacts = sanitizeRunArtifacts(runSummary.artifacts)
        await deps.storage.upsertRecord({
          scope: params.scope,
          type: recordType,
          key: item.key,
          content: {
            ...runSummary,
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
        return { key: item.key, objective: runSummary.objective }
      },
      onProgress: (progress) => {
        // Prefer the calibration-derived ETA; otherwise fall back to a wall-clock estimate
        // from elapsed-per-completed-item once at least one item has finished.
        const calibratedEta =
          calibration?.etaSeconds !== undefined && progress.total > 0
            ? calibration.etaSeconds * ((progress.total - progress.done) / progress.total)
            : undefined
        const elapsedSec = (Date.parse(now()) - campaignStartMs) / 1000
        const remaining = progress.total - progress.done
        const wallClockEta =
          calibratedEta === undefined && progress.done > 0 && remaining > 0 && elapsedSec > 0
            ? (elapsedSec / progress.done) * remaining
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
    const records = await deps.storage.listRecords({ scope: params.scope, type: recordType })
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

    const handle = resolveRunner(opts.computeTarget).runJob({
      jobId: `eval-${opts.runKey}`,
      repoRef: { kind: 'local', localPath: opts.projectRoot },
      commandTemplate: manifest.evaluate!,
      config: { ...(content.config ?? {}), checkpoint },
      dataFiles: manifestDataFiles(manifest),
      abortSignal: opts.abortSignal,
    })
    const result = await handle.done
    if (result.status !== 'completed') {
      throw new Error(result.error ?? `evaluation exited with code ${result.exitCode}`)
    }
    const summary = validateTrainingRunSummary(result.summary)
    const evaluatedAt = now()
    const evaluationType = `${recordType}-evaluation`
    await deps.storage.upsertRecord({
      scope: opts.scope,
      type: evaluationType,
      key: opts.runKey,
      content: { ...summary, runKey: opts.runKey, status: 'completed', evaluatedAt },
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
    const results: EvaluateTrainingRunResult[] = []
    const summary = await runActivityWorkItems<string, EvaluateTrainingRunResult>({
      items: params.runKeys,
      concurrency: params.concurrency,
      abortSignal: params.abortSignal,
      runItem: async (runKey) => {
        const result = await evaluateOneRun(manifest, {
          scope: params.scope,
          projectRoot: params.projectRoot,
          runKey,
          computeTarget: params.computeTarget,
          abortSignal: params.abortSignal,
          onRecordWritten: params.onRecordWritten,
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

  async function listCompletedRuns(scope: string, recordType: string) {
    const records = await deps.storage.listRecords({ scope, type: recordType })
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
    const allRuns = await listCompletedRuns(params.scope, recordType)
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
        content: verdict as unknown as Record<string, unknown>,
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

  async function proposeTrainingHypotheses(
    params: ProposeTrainingHypothesesParams,
  ): Promise<ProposeTrainingHypothesesResult> {
    const executor = requireInferenceExecutor()
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    const recordType = manifest.recordType
    const proposedBy = deriveModelRef({ kind: 'api', llmConfig: params.llmConfig }).label
    const proposedAt = now()
    const count = params.count ?? DEFAULT_HYPOTHESIS_COUNT

    const runs = await listCompletedRuns(params.scope, recordType)
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

    const hypothesisType = `${recordType}-hypothesis`
    const existing = await deps.storage.listRecords({ scope: params.scope, type: hypothesisType })
    const seenIds = new Set(existing.map((r) => r.key).filter((k): k is string => !!k))

    const items = coerceHypothesisItems(parseStructuredItems(res.text), manifest)
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

  async function analyzePaperFromUrl(
    params: AnalyzePaperFromUrlParams,
  ): Promise<AnalyzePaperFromUrlResult> {
    const executor = requireInferenceExecutor()
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    const recordType = manifest.recordType
    const analyzedBy = deriveModelRef({ kind: 'api', llmConfig: params.llmConfig }).label
    const analyzedAt = now()

    // The TOOL fetches the page text and hands it to the model — no web tools needed by the model.
    const fetchText = params.fetchPaperText ?? fetchPaperText
    const text = await fetchText(params.url, params.abortSignal)

    const res = await executor.runInference({
      systemPrompt: buildAnalyzePaperSystemPrompt(manifest, params.notes),
      userContent: buildAnalyzePaperUserContent({ url: params.url, text, notes: params.notes }),
      model: { kind: 'api', llmConfig: params.llmConfig },
      abortSignal: params.abortSignal,
    })

    const parsed = parseFirstValidJson(res.text)
    const draft = coercePaperDraft(parsed)
    if (!draft) throw new Error('the model did not return a usable paper summary for this link')

    const paperId = uuidv4()
    const paperType = `${recordType}-paper`
    const hypothesisType = `${recordType}-hypothesis`

    // Extract the paper's testable hypotheses; dedup by spec hash so identical specs from any source
    // (propose / manual / another paper) link to the ONE existing record rather than duplicate.
    const rawHyps =
      parsed &&
      typeof parsed === 'object' &&
      Array.isArray((parsed as Record<string, unknown>).hypotheses)
        ? ((parsed as Record<string, unknown>).hypotheses as unknown[])
        : []
    const items = coerceHypothesisItems(rawHyps, manifest)
    const existing = await deps.storage.listRecords({ scope: params.scope, type: hypothesisType })
    const byId = new Map<string, TrainingHypothesis>()
    for (const r of existing) {
      const content = r.content as unknown as TrainingHypothesis
      if (content && typeof content.id === 'string') byId.set(content.id, content)
    }
    const hypotheses: TrainingHypothesis[] = []
    const linkedHypothesisIds: string[] = []
    const seen = new Set<string>()
    for (const item of items) {
      const hid = hashTrainingConfig(item.spec as Record<string, unknown>)
      if (seen.has(hid)) continue
      seen.add(hid)
      const prior = byId.get(hid)
      const hypothesis: TrainingHypothesis = prior
        ? {
            ...prior,
            paperIds: Array.from(new Set([...(prior.paperIds ?? []), paperId])),
            updatedAt: analyzedAt,
          }
        : {
            id: hid,
            title: item.title,
            rationale: item.rationale,
            spec: item.spec,
            status: 'untested',
            verdictSource: 'auto',
            source: 'paper',
            proposedBy: analyzedBy,
            paperIds: [paperId],
            createdAt: analyzedAt,
            updatedAt: analyzedAt,
          }
      await deps.storage.upsertRecord({
        scope: params.scope,
        type: hypothesisType,
        key: hid,
        content: hypothesis as unknown as Record<string, unknown>,
      })
      params.onRecordWritten?.(hypothesisType, hid)
      hypotheses.push(hypothesis)
      linkedHypothesisIds.push(hid)
    }

    const paper: TrainingPaperRecord = {
      ...draft,
      id: paperId,
      title: draft.title as string,
      claim: draft.claim as string,
      url: params.url,
      hypothesisIds: linkedHypothesisIds,
      status: 'untested',
      source: 'research',
      createdAt: analyzedAt,
      updatedAt: analyzedAt,
    }
    await deps.storage.upsertRecord({
      scope: params.scope,
      type: paperType,
      key: paperId,
      content: paper as unknown as Record<string, unknown>,
    })
    params.onRecordWritten?.(paperType, paperId)
    logger?.info('analyzed paper from url', {
      recordType,
      url: params.url,
      id: paperId,
      hypotheses: hypotheses.length,
    })
    return { recordType, paper, hypotheses, linkedHypothesisIds, analyzedBy, analyzedAt }
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
    const records = await listCompletedRuns(scope, manifest.recordType)
    const focus = records.find((r) => r.key === runKey)
    if (!focus) throw new Error(`run "${runKey}" is not a completed run of this project`)
    const focusConfig = (focus.content.config as Record<string, unknown>) || {}

    const runs: AnalysisRun[] = records.map((r) => ({
      key: r.key,
      config: (r.content.config as Record<string, unknown>) || {},
      metrics: r.content.metrics as Record<string, number> | undefined,
      objective: r.content.objective as number,
      durationMs: r.content.durationMs as number | undefined,
      seed: r.content.seed as number | undefined,
      dataset: r.content.dataset as AnalysisRun['dataset'],
      status: 'completed',
    }))
    const ranked = runs
      .map((r) => ({ key: r.key, value: criterionValueOf(r, criterion) }))
      .filter((x): x is { key: string; value: number } => x.value !== undefined)
      .sort((a, b) => (criterion.direction === 'max' ? b.value - a.value : a.value - b.value))
    const rankPos = ranked.findIndex((x) => x.key === runKey)

    const trace = validateDecisionTrace(
      (focus.content.artifacts as { decisionTrace?: unknown } | undefined)?.decisionTrace,
    )
    const fa = trace?.featureAttribution
    const topGroups: [string, number][] = fa?.byGroup
      ? Object.entries(fa.byGroup)
          .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
          .slice(0, 5)
      : []
    const probe = trace?.latentMap?.probe

    let sibling: RunXaiDigest['sibling']
    const sib = siblingKey ? records.find((r) => r.key === siblingKey) : undefined
    if (sib) {
      const diff = diffDecisionTraces(
        sib.content as unknown as TrainingRunSummary,
        focus.content as unknown as TrainingRunSummary,
      )
      if (diff && diff.aligned) {
        const sibConfig = (sib.content.config as Record<string, unknown>) || {}
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
      objective: focus.content.objective as number | undefined,
      criterion,
      rank: rankPos >= 0 ? { position: rankPos + 1, total: ranked.length } : undefined,
      actionCounts: trace?.actionCounts,
      attribution: fa
        ? {
            topGroups,
            method: fa.method,
            sanityPassed: fa.sanityCheck?.passed,
            sanityRankCorr: fa.sanityCheck?.rankCorrelation,
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
        modelNames: [c.modelName],
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
          modelNames: m.modelNames,
        })),
        text,
      }),
      model: { kind: 'api', llmConfig: params.llmConfig },
      abortSignal: params.abortSignal,
    })

    const { matchModelIds, proposedModels } = coerceAnalyzedPaperModels(
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

    const modelIds = Array.from(new Set([...(paper.modelIds ?? []), ...linkedModelIds]))
    const updatedPaper: TrainingPaperRecord = { ...paper, modelIds, updatedAt: analyzedAt }
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
      analyzedBy,
      analyzedAt,
    }
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
    let examinedRuns = 0
    let migratedRuns = 0
    let deletedRuns = 0
    let examinedQueue = 0
    let migratedQueue = 0
    let deletedQueue = 0
    if (rules.length === 0) {
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

    const runRecords = await deps.storage.listRecords({ scope: params.scope, type: recordType })
    for (const record of runRecords) {
      examinedRuns++
      const content = (record.content ?? {}) as Record<string, unknown>
      const config = content.config
      if (!record.key || !config || typeof config !== 'object') continue
      const rule = findMigrationRule(config as Record<string, unknown>, rules)
      if (!rule) continue
      if (rule.delete) {
        await deleteRunAndDerived(record.key, content.setupKey)
        deletedRuns++
        continue
      }
      const migrated = applyMigrationRules(config as Record<string, unknown>, rules)
      if (!migrated) continue
      await deps.storage.upsertRecord({
        scope: params.scope,
        type: recordType,
        key: record.key,
        content: { ...content, config: migrated, setupKey: setupKeyOf(migrated) },
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
    analyzePaperFromUrl,
    suggestPaperHypotheses,
    scanProjectModels,
    analyzePaperModels,
    xaiNarrate,
    getRunData,
    getRunXAI,
    migrateTrainingRuns,
  }
}
