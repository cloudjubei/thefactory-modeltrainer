import type { ComputeRepoRef, ComputeRunner, InferenceExecutor } from 'thefactory-tools/types'
import {
  deriveModelRef,
  estimateCampaignEtaSeconds,
  parseStructuredItems,
  runActivityWorkItems,
} from 'thefactory-tools/utils'
import type {
  CalibrateTrainingParams,
  EvaluateTrainingRunParams,
  EvaluateTrainingRunResult,
  ExperimentSpec,
  JudgeTrainingRunsParams,
  JudgeTrainingRunsResult,
  ModelTrainerTools,
  ModelTrainerToolsDeps,
  PlannedTrainingItem,
  ProposeTrainingHypothesesParams,
  ProposeTrainingHypothesesResult,
  TrainerManifest,
  TrainingCalibration,
  TrainingCampaignParams,
  TrainingCampaignProgress,
  TrainingCampaignResult,
  TrainingHypothesis,
  TrainingVerdict,
} from './modelTrainerTypes.js'
import {
  DEFAULT_HYPOTHESIS_COUNT,
  DEFAULT_RAN_BY,
  JUDGE_LLM_WEIGHT,
  MAX_JUDGE_RUNS,
} from './modelTrainerConstants.js'
import { hashTrainingConfig, readTrainerManifest, setupKeyOf } from './modelTrainerHelpers.js'
import {
  blendJudgeScore,
  manifestDataFiles,
  parseProgressMarker,
  buildJudgeSystemPrompt,
  buildJudgeUserContent,
  buildProposeSystemPrompt,
  buildProposeUserContent,
  coerceHypothesisItems,
  coerceVerdictRows,
  expandExperimentMatrix,
  normalizeObjectiveScores,
  pickBestRun,
  totalCampaignUnits,
  validateTrainingRunSummary,
} from './modelTrainerUtils.js'

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
    const items = expandExperimentMatrix(manifest, params.spec, hashTrainingConfig)
    const total = items.length
    const repoRef: ComputeRepoRef = { kind: 'local', localPath: params.projectRoot }
    const runner = resolveRunner(params.computeTarget)
    const dataFiles = manifestDataFiles(manifest)
    const ranBy = params.ranBy ?? params.computeTarget ?? DEFAULT_RAN_BY

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
                config?: Record<string, unknown>
              },
          )
          .filter((c) => c?.status === 'completed')
          .map((c) => c.setupKey ?? (c.config ? setupKeyOf(c.config) : undefined))
          .filter((k): k is string => typeof k === 'string'),
      )
    }

    let lastKey: string | undefined
    const summary = await runActivityWorkItems<
      PlannedTrainingItem,
      { key: string; objective: number }
    >({
      items,
      concurrency: params.concurrency,
      abortSignal: params.abortSignal,
      isFresh: async (item) => {
        if (params.refresh) return false
        if (exploredSetups && exploredSetups.has(setupKeyOf(item.config))) return true
        const existing = await deps.storage.readRecord({
          scope: params.scope,
          type: recordType,
          key: item.key,
        })
        return (existing?.content as { status?: string } | undefined)?.status === 'completed'
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
          void params.onItemProgress(item.key, { phase: 'starting' })
          handle.onLog((line) => {
            const marker = parseProgressMarker(line)
            if (marker) void params.onItemProgress!(item.key, marker)
          })
        }
        const result = await handle.done
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
        await deps.storage.upsertRecord({
          scope: params.scope,
          type: recordType,
          key: item.key,
          content: {
            ...runSummary,
            status: 'completed',
            setupKey: setupKeyOf(item.config),
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
        const etaSeconds =
          calibration?.etaSeconds !== undefined && progress.total > 0
            ? calibration.etaSeconds * ((progress.total - progress.done) / progress.total)
            : undefined
        void emit({
          phase: 'train',
          done: progress.done,
          total: progress.total,
          skipped: progress.skipped,
          failed: progress.failed,
          ...(etaSeconds !== undefined ? { etaSeconds } : {}),
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

  async function evaluateTrainingRun(
    params: EvaluateTrainingRunParams,
  ): Promise<EvaluateTrainingRunResult> {
    const manifest =
      params.manifest ?? (await readTrainerManifest(params.projectRoot, params.manifestRelPath))
    if (!manifest.evaluate) {
      throw new Error('trainer manifest declares no evaluate command')
    }
    const recordType = manifest.recordType
    const record = await deps.storage.readRecord({
      scope: params.scope,
      type: recordType,
      key: params.runKey,
    })
    if (!record) throw new Error(`no run record for key ${params.runKey}`)
    const content = record.content as {
      config?: Record<string, unknown>
      artifacts?: { checkpoint?: unknown }
    }
    const checkpoint = content.artifacts?.checkpoint
    if (typeof checkpoint !== 'string' || !checkpoint) {
      throw new Error(`run ${params.runKey} has no checkpoint artifact to evaluate`)
    }

    const handle = resolveRunner(params.computeTarget).runJob({
      jobId: `eval-${params.runKey}`,
      repoRef: { kind: 'local', localPath: params.projectRoot },
      commandTemplate: manifest.evaluate,
      config: { ...(content.config ?? {}), checkpoint },
      dataFiles: manifestDataFiles(manifest),
      abortSignal: params.abortSignal,
    })
    const result = await handle.done
    if (result.status !== 'completed') {
      throw new Error(result.error ?? `evaluation exited with code ${result.exitCode}`)
    }
    const summary = validateTrainingRunSummary(result.summary)
    const evaluatedAt = now()
    const evaluationType = `${recordType}-evaluation`
    await deps.storage.upsertRecord({
      scope: params.scope,
      type: evaluationType,
      key: params.runKey,
      content: { ...summary, runKey: params.runKey, status: 'completed', evaluatedAt },
    })
    params.onRecordWritten?.(evaluationType, params.runKey)
    logger?.info('evaluated training run', { recordType, runKey: params.runKey })
    return { recordType, runKey: params.runKey, objective: summary.objective, evaluatedAt }
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
    const runs = await listCompletedRuns(params.scope, recordType)

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
        status: 'pending',
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

  return {
    readTrainerManifest: (projectRoot, manifestRelPath) =>
      readTrainerManifest(projectRoot, manifestRelPath),
    planTrainingMatrix: (manifest: TrainerManifest, spec: ExperimentSpec) =>
      expandExperimentMatrix(manifest, spec, hashTrainingConfig),
    calibrateTrainingThroughput,
    runTrainingCampaign,
    evaluateTrainingRun,
    judgeTrainingRuns,
    proposeTrainingHypotheses,
  }
}
