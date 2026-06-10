import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type {
  CalibrationProbe,
  CalibrationResult,
  ComputeJob,
  ComputeJobHandle,
  ComputeJobResult,
  ComputeRunner,
  DataRecord,
  DataRecordInput,
  DataStorage,
} from 'thefactory-tools/types'
import type { TrainerManifest, TrainingCampaignProgress } from './modelTrainerTypes.js'
import { createModelTrainerTools } from './ModelTrainerTools.js'

const NOW = '2026-06-10T12:00:00.000Z'

function manifest(overrides: Partial<TrainerManifest> = {}): TrainerManifest {
  return {
    name: 'demo',
    recordType: 'demo-run',
    run: 'bin/python -m trainer.run --config-json {configPath} --summary-out {summaryOut}',
    calibrate: 'bin/python -m trainer.run --calibrate --summary-out {summaryOut}',
    objective: { name: 'score', direction: 'max' },
    levers: {
      lr: { type: 'number', default: 0.01 },
      steps: { type: 'number', default: 100 },
    },
    eta: { unitsLever: 'steps' },
    ...overrides,
  }
}

interface MemoryStorage extends DataStorage {
  rows: Map<string, DataRecord>
}

function memoryStorage(): MemoryStorage {
  const rows = new Map<string, DataRecord>()
  const keyOf = (scope: string, type: string, key: string | null | undefined) =>
    `${scope}|${type}|${key ?? ''}`
  return {
    rows,
    async upsertRecord(input: DataRecordInput): Promise<DataRecord> {
      const record: DataRecord = {
        scope: input.scope,
        type: input.type,
        key: input.key ?? null,
        content: input.content,
        metadata: input.metadata,
        createdAt: NOW,
        updatedAt: NOW,
      }
      rows.set(keyOf(input.scope, input.type, input.key), record)
      return record
    },
    async readRecord(ref) {
      return rows.get(keyOf(ref.scope, ref.type, ref.key))
    },
    async listRecords(query) {
      return [...rows.values()].filter(
        (r) => r.scope === query.scope && (!query.type || r.type === query.type),
      )
    },
    async deleteRecord(ref) {
      return rows.delete(keyOf(ref.scope, ref.type, ref.key))
    },
  }
}

interface StubRunner extends ComputeRunner {
  jobs: ComputeJob[]
  probes: CalibrationProbe[]
}

function stubRunner(
  opts: {
    jobResult?: (job: ComputeJob) => Partial<ComputeJobResult>
    calibration?: CalibrationResult | (() => Promise<CalibrationResult>)
  } = {},
): StubRunner {
  const jobs: ComputeJob[] = []
  const probes: CalibrationProbe[] = []
  return {
    jobs,
    probes,
    async calibrate(probe) {
      probes.push(probe)
      const cal = opts.calibration ?? { secondsObserved: 2, unitsPerSecond: 500 }
      return typeof cal === 'function' ? cal() : cal
    },
    runJob(job): ComputeJobHandle {
      jobs.push(job)
      const partial = opts.jobResult?.(job) ?? {}
      const result: ComputeJobResult = {
        jobId: job.jobId,
        status: 'completed',
        exitCode: 0,
        summary: { objective: (job.config as { lr: number }).lr * 100 },
        logTail: [],
        durationMs: 5,
        ...partial,
      }
      return { jobId: job.jobId, onLog: () => {}, done: Promise.resolve(result), abort: () => {} }
    },
  }
}

function makeTools(
  runner: ComputeRunner,
  storage: DataStorage,
  logger = { info: vi.fn(), warn: vi.fn() },
) {
  return {
    tools: createModelTrainerTools({ computeRunner: runner, storage, logger, now: () => NOW }),
    logger,
  }
}

const tempDirs: string[] = []
afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })))
})

describe('planTrainingMatrix', () => {
  it('returns hash-keyed items', () => {
    const { tools } = makeTools(stubRunner(), memoryStorage())
    const items = tools.planTrainingMatrix(manifest(), { sweep: { lr: [0.1, 0.2] } })
    expect(items).toHaveLength(2)
    expect(items[0].key).toMatch(/^[0-9a-f]{12}$/)
    expect(items[0].key).not.toBe(items[1].key)
  })
})

describe('calibrateTrainingThroughput', () => {
  it('returns undefined when the manifest has no calibrate command', async () => {
    const { tools } = makeTools(stubRunner(), memoryStorage())
    const m = manifest()
    delete m.calibrate
    const result = await tools.calibrateTrainingThroughput({ projectRoot: '/repo', manifest: m })
    expect(result).toBeUndefined()
  })

  it('runs the calibrate template against the project root', async () => {
    const runner = stubRunner({ calibration: { secondsObserved: 3, unitsPerSecond: 250 } })
    const { tools } = makeTools(runner, memoryStorage())
    const result = await tools.calibrateTrainingThroughput({
      projectRoot: '/repo',
      manifest: manifest(),
    })
    expect(runner.probes[0].repoRef).toEqual({ kind: 'local', localPath: '/repo' })
    expect(runner.probes[0].commandTemplate).toContain('--calibrate')
    expect(result).toEqual({ secondsObserved: 3, unitsPerSecond: 250 })
  })
})

describe('runTrainingCampaign', () => {
  it('runs every planned item and persists completed records', async () => {
    const runner = stubRunner()
    const storage = memoryStorage()
    const { tools } = makeTools(runner, storage)
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: { sweep: { lr: [0.1, 0.2] } },
    })
    expect(result).toMatchObject({
      recordType: 'demo-run',
      planned: 2,
      completed: 2,
      skipped: 0,
      failed: 0,
      aborted: false,
      direction: 'max',
      finishedAt: NOW,
    })
    expect(runner.jobs).toHaveLength(2)
    const records = await storage.listRecords({ scope: 'proj', type: 'demo-run' })
    expect(records).toHaveLength(2)
    expect(records[0].content).toMatchObject({ status: 'completed', ranAt: NOW, ranBy: 'local' })
  })

  it('reports the best run by objective (max)', async () => {
    const storage = memoryStorage()
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: { sweep: { lr: [0.1, 0.4, 0.2] } },
    })
    expect(result.bestObjective).toBe(40)
  })

  it('reports the best run by objective (min)', async () => {
    const { tools } = makeTools(stubRunner(), memoryStorage())
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest({ objective: { name: 'score', direction: 'min' } }),
      spec: { sweep: { lr: [0.1, 0.4, 0.2] } },
    })
    expect(result.bestObjective).toBe(10)
    expect(result.direction).toBe('min')
  })

  it('counts the global best across pre-existing records', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'previous',
      content: { objective: 9000, status: 'completed' },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: {},
    })
    expect(result.bestKey).toBe('previous')
    expect(result.bestObjective).toBe(9000)
  })

  it('skips items whose record is already completed', async () => {
    const runner = stubRunner()
    const storage = memoryStorage()
    const { tools } = makeTools(runner, storage)
    const items = tools.planTrainingMatrix(manifest(), { sweep: { lr: [0.1, 0.2] } })
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: items[0].key,
      content: { objective: 1, status: 'completed' },
    })
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: { sweep: { lr: [0.1, 0.2] } },
    })
    expect(result.skipped).toBe(1)
    expect(result.completed).toBe(1)
    expect(runner.jobs).toHaveLength(1)
  })

  it('reruns fresh items when refresh is set', async () => {
    const runner = stubRunner()
    const storage = memoryStorage()
    const { tools } = makeTools(runner, storage)
    const items = tools.planTrainingMatrix(manifest(), {})
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: items[0].key,
      content: { objective: 1, status: 'completed' },
    })
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: {},
      refresh: true,
    })
    expect(result.skipped).toBe(0)
    expect(runner.jobs).toHaveLength(1)
  })

  it('marks a failed job failed, writes no record, and continues', async () => {
    const runner = stubRunner({
      jobResult: (job) =>
        (job.config as { lr: number }).lr === 0.1
          ? { status: 'failed', exitCode: 1, error: 'exit 1', summary: undefined }
          : {},
    })
    const storage = memoryStorage()
    const { tools } = makeTools(runner, storage)
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: { sweep: { lr: [0.1, 0.2] } },
    })
    expect(result.failed).toBe(1)
    expect(result.completed).toBe(1)
    expect(await storage.listRecords({ scope: 'proj', type: 'demo-run' })).toHaveLength(1)
  })

  it('marks a completed job with an invalid summary as failed', async () => {
    const runner = stubRunner({ jobResult: () => ({ summary: { nope: true } }) })
    const { tools } = makeTools(runner, memoryStorage())
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: {},
    })
    expect(result.failed).toBe(1)
    expect(result.completed).toBe(0)
  })

  it('stops at an abort signal and reports aborted', async () => {
    const controller = new AbortController()
    const runner = stubRunner({
      jobResult: () => {
        controller.abort()
        return {}
      },
    })
    const { tools } = makeTools(runner, memoryStorage())
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: { sweep: { lr: [0.1, 0.2, 0.3] } },
      abortSignal: controller.signal,
    })
    expect(result.aborted).toBe(true)
    expect(runner.jobs).toHaveLength(1)
  })

  it('streams calibrate, train and done progress with an ETA', async () => {
    const progress: TrainingCampaignProgress[] = []
    const runner = stubRunner({ calibration: { secondsObserved: 1, unitsPerSecond: 100 } })
    const { tools } = makeTools(runner, memoryStorage())
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: { sweep: { lr: [0.1, 0.2] } },
      onProgress: (p) => {
        progress.push(p)
      },
    })
    expect(progress[0]).toMatchObject({ phase: 'calibrate', done: 0, total: 2 })
    const trainTicks = progress.filter((p) => p.phase === 'train')
    expect(trainTicks).toHaveLength(2)
    expect(trainTicks[0].etaSeconds).toBe(1)
    expect(progress.at(-1)).toMatchObject({ phase: 'done', done: 2, total: 2 })
  })

  it('skips the calibrate phase when the manifest declares none', async () => {
    const progress: TrainingCampaignProgress[] = []
    const m = manifest()
    delete m.calibrate
    const { tools } = makeTools(stubRunner(), memoryStorage())
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: m,
      spec: {},
      onProgress: (p) => {
        progress.push(p)
      },
    })
    expect(progress.some((p) => p.phase === 'calibrate')).toBe(false)
    expect(result.calibration).toBeUndefined()
  })

  it('tolerates a calibrate failure and still trains', async () => {
    const runner = stubRunner({
      calibration: () => Promise.reject(new Error('boom')),
    })
    const storage = memoryStorage()
    const { tools, logger } = makeTools(runner, storage)
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: {},
    })
    expect(result.completed).toBe(1)
    expect(result.calibration).toBeUndefined()
    expect(logger.warn).toHaveBeenCalled()
  })

  it('notifies onRecordWritten for every persisted run', async () => {
    const written: string[] = []
    const { tools } = makeTools(stubRunner(), memoryStorage())
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: { sweep: { lr: [0.1, 0.2] } },
      onRecordWritten: (type, key) => {
        written.push(`${type}:${key}`)
      },
    })
    expect(written).toHaveLength(2)
    expect(written[0]).toMatch(/^demo-run:[0-9a-f]{12}$/)
  })

  it('surfaces per-item failures with their reasons', async () => {
    const runner = stubRunner({
      jobResult: (job) =>
        (job.config as { lr: number }).lr === 0.1
          ? { status: 'failed', exitCode: 3, error: undefined, summary: undefined }
          : {},
    })
    const { tools } = makeTools(runner, memoryStorage())
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: { sweep: { lr: [0.1, 0.2] } },
    })
    expect(result.failures).toHaveLength(1)
    expect(result.failures?.[0].error).toMatch(/exited with code 3/)
    expect(result.failures?.[0].key).toMatch(/^[0-9a-f]{12}$/)
  })

  it('omits failures when every run succeeds', async () => {
    const { tools } = makeTools(stubRunner(), memoryStorage())
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: {},
    })
    expect(result.failures).toBeUndefined()
  })

  it('keeps the calibration but omits the ETA when throughput is unknown', async () => {
    const runner = stubRunner({ calibration: { secondsObserved: 2 } })
    const { tools } = makeTools(runner, memoryStorage())
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: {},
    })
    expect(result.calibration).toEqual({ secondsObserved: 2 })
    expect(result.calibration?.etaSeconds).toBeUndefined()
  })

  it('exposes manifest reading on the toolset surface', async () => {
    const root = await mkdtemp(join(tmpdir(), 'modeltrainer-read-'))
    tempDirs.push(root)
    await mkdir(join(root, '.factory'), { recursive: true })
    await writeFile(join(root, '.factory', 'trainer.json'), JSON.stringify(manifest()))
    const { tools } = makeTools(stubRunner(), memoryStorage())
    const read = await tools.readTrainerManifest(root)
    expect(read.recordType).toBe('demo-run')
  })

  it('works without an injected clock or logger', async () => {
    const tools = createModelTrainerTools({ computeRunner: stubRunner(), storage: memoryStorage() })
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: {},
    })
    expect(result.completed).toBe(1)
    expect(result.finishedAt).toMatch(/^\d{4}-\d{2}-\d{2}T/)
  })

  it('stamps a custom ranBy label', async () => {
    const storage = memoryStorage()
    const { tools } = makeTools(stubRunner(), storage)
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: {},
      ranBy: 'runner-7',
    })
    const records = await storage.listRecords({ scope: 'proj', type: 'demo-run' })
    expect(records[0].content).toMatchObject({ ranBy: 'runner-7' })
  })

  it('reads the manifest from the project root when not provided', async () => {
    const root = await mkdtemp(join(tmpdir(), 'modeltrainer-tools-'))
    tempDirs.push(root)
    await mkdir(join(root, '.factory'), { recursive: true })
    await writeFile(join(root, '.factory', 'trainer.json'), JSON.stringify(manifest()))
    const { tools } = makeTools(stubRunner(), memoryStorage())
    const result = await tools.runTrainingCampaign({ scope: 'proj', projectRoot: root, spec: {} })
    expect(result.recordType).toBe('demo-run')
    expect(result.completed).toBe(1)
  })

  it('passes the run template and config through to the compute job', async () => {
    const runner = stubRunner()
    const { tools } = makeTools(runner, memoryStorage())
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: { fixed: { lr: 0.7 } },
    })
    expect(runner.jobs[0].commandTemplate).toContain('--config-json {configPath}')
    expect(runner.jobs[0].repoRef).toEqual({ kind: 'local', localPath: '/repo' })
    expect((runner.jobs[0].config as { lr: number }).lr).toBe(0.7)
  })
})
