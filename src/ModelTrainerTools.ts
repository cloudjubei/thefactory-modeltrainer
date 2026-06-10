import type { ComputeRepoRef } from 'thefactory-tools/types'
import { estimateCampaignEtaSeconds, runActivityWorkItems } from 'thefactory-tools/utils'
import type {
  CalibrateTrainingParams,
  ExperimentSpec,
  ModelTrainerTools,
  ModelTrainerToolsDeps,
  PlannedTrainingItem,
  TrainerManifest,
  TrainingCalibration,
  TrainingCampaignParams,
  TrainingCampaignProgress,
  TrainingCampaignResult,
} from './modelTrainerTypes.js'
import { DEFAULT_RAN_BY } from './modelTrainerConstants.js'
import { hashTrainingConfig, readTrainerManifest } from './modelTrainerHelpers.js'
import {
  expandExperimentMatrix,
  pickBestRun,
  totalCampaignUnits,
  validateTrainingRunSummary,
} from './modelTrainerUtils.js'

export function createModelTrainerTools(deps: ModelTrainerToolsDeps): ModelTrainerTools {
  const now = deps.now ?? (() => new Date().toISOString())
  const logger = deps.logger

  async function calibrateTrainingThroughput(
    params: CalibrateTrainingParams,
  ): Promise<TrainingCalibration | undefined> {
    if (!params.manifest.calibrate) return undefined
    const result = await deps.computeRunner.calibrate({
      repoRef: { kind: 'local', localPath: params.projectRoot },
      commandTemplate: params.manifest.calibrate,
      abortSignal: params.abortSignal,
    })
    return {
      secondsObserved: result.secondsObserved,
      ...(result.unitsPerSecond !== undefined ? { unitsPerSecond: result.unitsPerSecond } : {}),
    }
  }

  async function runTrainingCampaign(
    params: TrainingCampaignParams,
  ): Promise<TrainingCampaignResult> {
    const manifest = params.manifest ?? (await readTrainerManifest(params.projectRoot))
    const recordType = manifest.recordType
    const items = expandExperimentMatrix(manifest, params.spec, hashTrainingConfig)
    const total = items.length
    const repoRef: ComputeRepoRef = { kind: 'local', localPath: params.projectRoot }
    const ranBy = params.ranBy ?? DEFAULT_RAN_BY

    const emit = async (progress: TrainingCampaignProgress) => {
      await params.onProgress?.(progress)
    }

    let calibration: TrainingCalibration | undefined
    if (manifest.calibrate) {
      await emit({ phase: 'calibrate', done: 0, total, skipped: 0, failed: 0 })
      try {
        const measured = await calibrateTrainingThroughput({
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

    let lastKey: string | undefined
    const summary = await runActivityWorkItems<
      PlannedTrainingItem,
      { key: string; objective: number }
    >({
      items,
      abortSignal: params.abortSignal,
      isFresh: async (item) => {
        if (params.refresh) return false
        const existing = await deps.storage.readRecord({
          scope: params.scope,
          type: recordType,
          key: item.key,
        })
        return (existing?.content as { status?: string } | undefined)?.status === 'completed'
      },
      runItem: async (item) => {
        const handle = deps.computeRunner.runJob({
          jobId: item.key,
          repoRef,
          commandTemplate: manifest.run,
          config: item.config,
          abortSignal: params.abortSignal,
        })
        const result = await handle.done
        if (result.status !== 'completed') {
          throw new Error(result.error ?? `training exited with code ${result.exitCode}`)
        }
        const runSummary = validateTrainingRunSummary(result.summary)
        await deps.storage.upsertRecord({
          scope: params.scope,
          type: recordType,
          key: item.key,
          content: { ...runSummary, status: 'completed', ranAt: now(), ranBy },
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

  return {
    readTrainerManifest: (projectRoot: string) => readTrainerManifest(projectRoot),
    planTrainingMatrix: (manifest: TrainerManifest, spec: ExperimentSpec) =>
      expandExperimentMatrix(manifest, spec, hashTrainingConfig),
    calibrateTrainingThroughput,
    runTrainingCampaign,
  }
}
