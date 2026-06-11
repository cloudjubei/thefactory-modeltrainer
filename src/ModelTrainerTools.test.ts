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
  InferenceExecutor,
  InferenceRequest,
  LLMConfig,
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

const LLM: LLMConfig = { provider: 'openai', model: 'm', apiKey: 'k' } as LLMConfig

interface StubExecutor extends InferenceExecutor {
  requests: InferenceRequest[]
}

function stubExecutor(text: string | ((req: InferenceRequest) => string)): StubExecutor {
  const requests: InferenceRequest[] = []
  return {
    requests,
    async runInference(request) {
      requests.push(request)
      return { text: typeof text === 'function' ? text(request) : text }
    },
  }
}

function makeJudgeTools(executor: InferenceExecutor | undefined, storage: DataStorage) {
  const logger = { info: vi.fn(), warn: vi.fn() }
  return {
    tools: createModelTrainerTools({
      computeRunner: stubRunner(),
      storage,
      inferenceExecutor: executor,
      logger,
      now: () => NOW,
    }),
    logger,
  }
}

async function seedRun(
  storage: DataStorage,
  key: string,
  objective: number,
  extra: Record<string, unknown> = {},
) {
  await storage.upsertRecord({
    scope: 'proj',
    type: 'demo-run',
    key,
    content: {
      objective,
      status: 'completed',
      health: { status: 'ok', flags: [] },
      config: { lr: objective / 100 },
      ...extra,
    },
  })
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
    expect(records[0].content).toMatchObject({
      status: 'completed',
      ranAt: NOW,
      ranBy: 'local',
      durationMs: 5,
    })
  })

  it('reports the planned item keys on the result', async () => {
    const { tools } = makeTools(stubRunner(), memoryStorage())
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: { sweep: { lr: [0.1, 0.2] } },
    })
    expect(result.keys).toHaveLength(2)
    expect(result.keys.every((k) => /^[0-9a-f]{12}$/.test(k))).toBe(true)
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

describe('data files + compute targets', () => {
  const DATA = [
    {
      id: 'wine',
      files: [{ relPath: 'data/wine.csv', url: 'https://e.x/wine.csv' }],
    },
    {
      id: 'extra',
      files: [{ relPath: 'data/extra.csv', url: 'https://e.x/extra.csv', sha256: 'abc' }],
    },
  ]

  it('threads the manifest data files onto every training job and calibrate probe', async () => {
    const runner = stubRunner()
    const { tools } = makeTools(runner, memoryStorage())
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest({ data: DATA }),
      spec: {},
    })
    const expected = [
      { relPath: 'data/wine.csv', url: 'https://e.x/wine.csv' },
      { relPath: 'data/extra.csv', url: 'https://e.x/extra.csv', sha256: 'abc' },
    ]
    expect(runner.probes[0].dataFiles).toEqual(expected)
    expect(runner.jobs[0].dataFiles).toEqual(expected)
  })

  it('threads data files onto evaluate jobs', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'r1',
      content: {
        objective: 1,
        status: 'completed',
        config: {},
        artifacts: { checkpoint: 'c.zip' },
      },
    })
    const runner = stubRunner({ jobResult: () => ({ summary: { objective: 1 } }) })
    const { tools } = makeTools(runner, storage)
    await tools.evaluateTrainingRun({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: {
        ...manifest({ data: DATA }),
        evaluate: 'e --config-json {configPath} --summary-out {summaryOut}',
      },
      runKey: 'r1',
    })
    expect(runner.jobs[0].dataFiles).toHaveLength(2)
  })

  it('omits dataFiles when the manifest declares no data', async () => {
    const runner = stubRunner()
    const { tools } = makeTools(runner, memoryStorage())
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: {},
    })
    expect(runner.jobs[0].dataFiles).toBeUndefined()
  })

  it('runs on a resolved compute target and stamps it as ranBy', async () => {
    const local = stubRunner()
    const remote = stubRunner()
    const storage = memoryStorage()
    const tools = createModelTrainerTools({
      computeRunner: local,
      resolveComputeRunner: (target) => (target === 'gpu-1' ? remote : undefined),
      storage,
      now: () => NOW,
    })
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: {},
      computeTarget: 'gpu-1',
    })
    expect(remote.jobs).toHaveLength(1)
    expect(remote.probes).toHaveLength(1)
    expect(local.jobs).toHaveLength(0)
    expect(result.completed).toBe(1)
    const records = await storage.listRecords({ scope: 'proj', type: 'demo-run' })
    expect(records[0].content).toMatchObject({ ranBy: 'gpu-1' })
  })

  it('throws on an unresolvable compute target', async () => {
    const { tools } = makeTools(stubRunner(), memoryStorage())
    await expect(
      tools.runTrainingCampaign({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: manifest(),
        spec: {},
        computeTarget: 'ghost',
      }),
    ).rejects.toThrow(/ghost/)
  })

  it('evaluates on a resolved compute target', async () => {
    const local = stubRunner()
    const remote = stubRunner({ jobResult: () => ({ summary: { objective: 2 } }) })
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'r1',
      content: { objective: 1, status: 'completed', config: {}, artifacts: { checkpoint: 'c' } },
    })
    const tools = createModelTrainerTools({
      computeRunner: local,
      resolveComputeRunner: (t) => (t === 'gpu-1' ? remote : undefined),
      storage,
      now: () => NOW,
    })
    await tools.evaluateTrainingRun({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: { ...manifest(), evaluate: 'e {configPath} {summaryOut}' },
      runKey: 'r1',
      computeTarget: 'gpu-1',
    })
    expect(remote.jobs).toHaveLength(1)
    expect(local.jobs).toHaveLength(0)
  })
})

describe('evaluateTrainingRun', () => {
  function evalManifest(): TrainerManifest {
    return {
      ...manifest(),
      evaluate:
        'bin/python -m trainer.run --evaluate --config-json {configPath} --summary-out {summaryOut}',
    }
  }

  async function seedCompletedRun(storage: DataStorage, key: string) {
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key,
      content: {
        objective: 100,
        status: 'completed',
        config: { lr: 0.1, steps: 100 },
        artifacts: { checkpoint: `checkpoints/${key}.zip` },
      },
    })
  }

  it('re-runs the checkpoint via the evaluate template and persists the evaluation record', async () => {
    const storage = memoryStorage()
    await seedCompletedRun(storage, 'run1')
    const runner = stubRunner({
      jobResult: () => ({
        summary: {
          objective: 95,
          metrics: { eval_return_mean: 95 },
          evaluation: { checkpoint: 'checkpoints/run1.zip', episodes: 20 },
        },
      }),
    })
    const { tools } = makeTools(runner, storage)
    const result = await tools.evaluateTrainingRun({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: evalManifest(),
      runKey: 'run1',
    })
    expect(result).toEqual({
      recordType: 'demo-run',
      runKey: 'run1',
      objective: 95,
      evaluatedAt: NOW,
    })
    expect(runner.jobs[0].commandTemplate).toContain('--evaluate')
    expect(runner.jobs[0].repoRef).toEqual({ kind: 'local', localPath: '/repo' })
    expect(runner.jobs[0].config).toMatchObject({
      lr: 0.1,
      steps: 100,
      checkpoint: 'checkpoints/run1.zip',
    })
    const record = await storage.readRecord({
      scope: 'proj',
      type: 'demo-run-evaluation',
      key: 'run1',
    })
    expect(record?.content).toMatchObject({
      runKey: 'run1',
      objective: 95,
      status: 'completed',
      evaluatedAt: NOW,
    })
  })

  it('notifies onRecordWritten for the evaluation record', async () => {
    const storage = memoryStorage()
    await seedCompletedRun(storage, 'run1')
    const runner = stubRunner({ jobResult: () => ({ summary: { objective: 1 } }) })
    const { tools } = makeTools(runner, storage)
    const written: string[] = []
    await tools.evaluateTrainingRun({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: evalManifest(),
      runKey: 'run1',
      onRecordWritten: (type, key) => {
        written.push(`${type}:${key}`)
      },
    })
    expect(written).toEqual(['demo-run-evaluation:run1'])
  })

  it('throws when the manifest declares no evaluate command', async () => {
    const storage = memoryStorage()
    await seedCompletedRun(storage, 'run1')
    const { tools } = makeTools(stubRunner(), storage)
    await expect(
      tools.evaluateTrainingRun({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: manifest(),
        runKey: 'run1',
      }),
    ).rejects.toThrow(/evaluate/)
  })

  it('throws when the run record does not exist', async () => {
    const { tools } = makeTools(stubRunner(), memoryStorage())
    await expect(
      tools.evaluateTrainingRun({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: evalManifest(),
        runKey: 'ghost',
      }),
    ).rejects.toThrow(/ghost/)
  })

  it('throws when the run has no checkpoint artifact', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'nockpt',
      content: { objective: 1, status: 'completed', config: {} },
    })
    const { tools } = makeTools(stubRunner(), storage)
    await expect(
      tools.evaluateTrainingRun({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: evalManifest(),
        runKey: 'nockpt',
      }),
    ).rejects.toThrow(/checkpoint/)
  })

  it('throws when the evaluate job fails', async () => {
    const storage = memoryStorage()
    await seedCompletedRun(storage, 'run1')
    const runner = stubRunner({
      jobResult: () => ({
        status: 'failed',
        exitCode: 1,
        error: 'bad checkpoint',
        summary: undefined,
      }),
    })
    const { tools } = makeTools(runner, storage)
    await expect(
      tools.evaluateTrainingRun({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: evalManifest(),
        runKey: 'run1',
      }),
    ).rejects.toThrow('bad checkpoint')
    expect(
      await storage.readRecord({ scope: 'proj', type: 'demo-run-evaluation', key: 'run1' }),
    ).toBeUndefined()
  })

  it('throws when the evaluation summary is invalid', async () => {
    const storage = memoryStorage()
    await seedCompletedRun(storage, 'run1')
    const runner = stubRunner({ jobResult: () => ({ summary: { nope: true } }) })
    const { tools } = makeTools(runner, storage)
    await expect(
      tools.evaluateTrainingRun({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: evalManifest(),
        runKey: 'run1',
      }),
    ).rejects.toThrow(/objective/)
  })
})

describe('judgeTrainingRuns', () => {
  it('blends the normalised objective with the LLM verdict and persists verdict records', async () => {
    const storage = memoryStorage()
    await seedRun(storage, 'low', 10)
    await seedRun(storage, 'high', 90)
    const executor = stubExecutor(
      JSON.stringify([
        { key: 'low', score: 60, why: 'lucky seed' },
        { key: 'high', score: 80, why: 'stable' },
      ]),
    )
    const { tools } = makeJudgeTools(executor, storage)
    const result = await tools.judgeTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
    })
    expect(result).toMatchObject({
      recordType: 'demo-run',
      judged: 2,
      rejected: 0,
      judgedBy: 'openai/m',
      judgedAt: NOW,
    })
    const low = result.verdicts.find((v) => v.key === 'low')!
    expect(low).toMatchObject({ objectiveScore: 0, llmScore: 60, score: 30, why: 'lucky seed' })
    const high = result.verdicts.find((v) => v.key === 'high')!
    expect(high).toMatchObject({ objectiveScore: 100, llmScore: 80, score: 90 })
    const records = await storage.listRecords({ scope: 'proj', type: 'demo-run-verdict' })
    expect(records).toHaveLength(2)
    expect(records.find((r) => r.key === 'high')?.content).toMatchObject({
      score: 90,
      judgedBy: 'openai/m',
      judgedAt: NOW,
    })
  })

  it('auto-rejects health-flagged runs without consulting the LLM', async () => {
    const storage = memoryStorage()
    await seedRun(storage, 'good', 50)
    await seedRun(storage, 'bad', 99, {
      health: { status: 'degenerate', flags: ['degenerate_policy'] },
    })
    const executor = stubExecutor(JSON.stringify([{ key: 'good', score: 70, why: 'fine' }]))
    const { tools } = makeJudgeTools(executor, storage)
    const result = await tools.judgeTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
    })
    expect(result.rejected).toBe(1)
    expect(result.judged).toBe(2)
    const bad = result.verdicts.find((v) => v.key === 'bad')!
    expect(bad).toMatchObject({ rejected: true, score: 0 })
    expect(bad.why).toContain('degenerate_policy')
    const sent = JSON.parse(executor.requests[0].userContent) as { key: string }[]
    expect(sent.map((r) => r.key)).toEqual(['good'])
  })

  it('falls back to the objective score when the LLM returns no row for a run', async () => {
    const storage = memoryStorage()
    await seedRun(storage, 'scored', 10)
    await seedRun(storage, 'missed', 90)
    const executor = stubExecutor(JSON.stringify([{ key: 'scored', score: 40, why: 'ok' }]))
    const { tools } = makeJudgeTools(executor, storage)
    const result = await tools.judgeTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
    })
    const missed = result.verdicts.find((v) => v.key === 'missed')!
    expect(missed.llmScore).toBeUndefined()
    expect(missed.score).toBe(100)
    expect(missed.why).toMatch(/objective/i)
  })

  it('skips the LLM entirely when there are no completed runs', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'failed-run',
      content: { objective: 1, status: 'failed' },
    })
    const executor = stubExecutor('[]')
    const { tools } = makeJudgeTools(executor, storage)
    const result = await tools.judgeTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
    })
    expect(result.judged).toBe(0)
    expect(executor.requests).toHaveLength(0)
    expect(await storage.listRecords({ scope: 'proj', type: 'demo-run-verdict' })).toHaveLength(0)
  })

  it('threads extra instructions into the judge prompt', async () => {
    const storage = memoryStorage()
    await seedRun(storage, 'a', 1)
    const executor = stubExecutor('[]')
    const { tools } = makeJudgeTools(executor, storage)
    await tools.judgeTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
      instructions: 'prefer fewer trades',
    })
    expect(executor.requests[0].systemPrompt).toContain('prefer fewer trades')
  })

  it('notifies onRecordWritten per verdict', async () => {
    const storage = memoryStorage()
    await seedRun(storage, 'a', 1)
    const written: string[] = []
    const { tools } = makeJudgeTools(stubExecutor('[]'), storage)
    await tools.judgeTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
      onRecordWritten: (type, key) => {
        written.push(`${type}:${key}`)
      },
    })
    expect(written).toEqual(['demo-run-verdict:a'])
  })

  it('throws without an inference executor', async () => {
    const { tools } = makeJudgeTools(undefined, memoryStorage())
    await expect(
      tools.judgeTrainingRuns({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: manifest(),
        llmConfig: LLM,
      }),
    ).rejects.toThrow(/inference/i)
  })

  it('propagates an inference failure', async () => {
    const storage = memoryStorage()
    await seedRun(storage, 'a', 1)
    const executor: InferenceExecutor = {
      runInference: () => Promise.reject(new Error('llm down')),
    }
    const { tools } = makeJudgeTools(executor, storage)
    await expect(
      tools.judgeTrainingRuns({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: manifest(),
        llmConfig: LLM,
      }),
    ).rejects.toThrow('llm down')
  })
})

describe('proposeTrainingHypotheses', () => {
  const PROPOSALS = JSON.stringify([
    {
      title: 'Push lr higher',
      rationale: 'top runs cluster at high lr',
      spec: { sweep: { lr: [0.5, 0.9] } },
    },
    {
      title: 'Longer training',
      rationale: 'returns still climbing',
      spec: { fixed: { steps: 500 } },
    },
    { title: 'Bad one', rationale: 'names a ghost lever', spec: { sweep: { ghost: [1] } } },
  ])

  it('persists valid proposals as pending hypothesis records', async () => {
    const storage = memoryStorage()
    await seedRun(storage, 'a', 10)
    const executor = stubExecutor(PROPOSALS)
    const { tools } = makeJudgeTools(executor, storage)
    const result = await tools.proposeTrainingHypotheses({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
    })
    expect(result).toMatchObject({
      recordType: 'demo-run',
      proposed: 2,
      skippedExisting: 0,
      proposedBy: 'openai/m',
      proposedAt: NOW,
    })
    const records = await storage.listRecords({ scope: 'proj', type: 'demo-run-hypothesis' })
    expect(records).toHaveLength(2)
    expect(records[0].content).toMatchObject({
      status: 'pending',
      source: 'llm',
      proposedBy: 'openai/m',
      createdAt: NOW,
    })
    expect((records[0].content as { id: string }).id).toBe(records[0].key)
  })

  it('dedupes against existing hypotheses, preserving their status', async () => {
    const storage = memoryStorage()
    const executor = stubExecutor(PROPOSALS)
    const { tools } = makeJudgeTools(executor, storage)
    const first = await tools.proposeTrainingHypotheses({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
    })
    const existingKey = first.hypotheses[0].id
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run-hypothesis',
      key: existingKey,
      content: { ...first.hypotheses[0], status: 'accepted' },
    })
    const second = await tools.proposeTrainingHypotheses({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
    })
    expect(second.skippedExisting).toBe(2)
    expect(second.proposed).toBe(0)
    const record = await storage.readRecord({
      scope: 'proj',
      type: 'demo-run-hypothesis',
      key: existingKey,
    })
    expect((record?.content as { status: string }).status).toBe('accepted')
  })

  it('dedupes identical specs within one response', async () => {
    const storage = memoryStorage()
    const twice = JSON.stringify([
      { title: 'One', rationale: 'r', spec: { fixed: { lr: 0.5 } } },
      { title: 'Same spec again', rationale: 'r2', spec: { fixed: { lr: 0.5 } } },
    ])
    const { tools } = makeJudgeTools(stubExecutor(twice), storage)
    const result = await tools.proposeTrainingHypotheses({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
    })
    expect(result.proposed).toBe(1)
  })

  it('caps proposals at the requested count', async () => {
    const many = JSON.stringify(
      Array.from({ length: 7 }, (_, i) => ({
        title: `H${i}`,
        rationale: 'r',
        spec: { fixed: { lr: i / 10 } },
      })),
    )
    const { tools } = makeJudgeTools(stubExecutor(many), memoryStorage())
    const result = await tools.proposeTrainingHypotheses({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
      count: 2,
    })
    expect(result.proposed).toBe(2)
  })

  it('sends run history and verdicts to the proposer', async () => {
    const storage = memoryStorage()
    await seedRun(storage, 'a', 10)
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run-verdict',
      key: 'a',
      content: { key: 'a', score: 77, why: 'good' },
    })
    const executor = stubExecutor('[]')
    const { tools } = makeJudgeTools(executor, storage)
    await tools.proposeTrainingHypotheses({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
    })
    const sent = JSON.parse(executor.requests[0].userContent) as {
      runs: { key: string }[]
      verdicts: { score: number }[]
      bestObjective: number
    }
    expect(sent.runs[0].key).toBe('a')
    expect(sent.verdicts[0].score).toBe(77)
    expect(sent.bestObjective).toBe(10)
  })

  it('notifies onRecordWritten per hypothesis', async () => {
    const written: string[] = []
    const { tools } = makeJudgeTools(stubExecutor(PROPOSALS), memoryStorage())
    await tools.proposeTrainingHypotheses({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
      onRecordWritten: (type) => {
        written.push(type)
      },
    })
    expect(written).toEqual(['demo-run-hypothesis', 'demo-run-hypothesis'])
  })

  it('throws without an inference executor', async () => {
    const { tools } = makeJudgeTools(undefined, memoryStorage())
    await expect(
      tools.proposeTrainingHypotheses({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: manifest(),
        llmConfig: LLM,
      }),
    ).rejects.toThrow(/inference/i)
  })
})
