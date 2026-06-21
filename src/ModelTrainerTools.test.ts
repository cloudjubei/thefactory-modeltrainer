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
import { hashTrainingConfig, setupKeyOf } from './modelTrainerHelpers.js'

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

  it('strips an unusable decisionTrace artifact but keeps a valid one and other artifacts', async () => {
    const storage = memoryStorage()
    const runner = stubRunner({
      jobResult: (job) => ({
        summary: {
          objective: 1,
          artifacts:
            (job.config as { lr: number }).lr === 0.1
              ? { checkpoint: 'c.zip', decisionTrace: { steps: 'garbage' } }
              : { decisionTrace: { steps: [{ step: 0, action: 'hold' }] } },
        },
      }),
    })
    const { tools } = makeTools(runner, storage)
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: { sweep: { lr: [0.1, 0.2] } },
    })
    const records = await storage.listRecords({ scope: 'proj', type: 'demo-run' })
    const artifacts = records.map((r) => (r.content as { artifacts?: unknown }).artifacts)
    expect(artifacts).toContainEqual({ checkpoint: 'c.zip' })
    expect(artifacts).toContainEqual({ decisionTrace: { steps: [{ step: 0, action: 'hold' }] } })
  })

  it('stamps the launch thesis and target on every run record', async () => {
    const storage = memoryStorage()
    const { tools } = makeTools(stubRunner(), storage)
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: { sweep: { lr: [0.1, 0.2] } },
      thesis: 'fee-penalty reward',
      thesisTarget: 'lr',
    })
    const records = await storage.listRecords({ scope: 'proj', type: 'demo-run' })
    expect(records).toHaveLength(2)
    for (const r of records) {
      expect(r.content).toMatchObject({ thesis: 'fee-penalty reward', thesisTarget: 'lr' })
    }
  })

  it('omits the thesis fields when none is given', async () => {
    const storage = memoryStorage()
    const { tools } = makeTools(stubRunner(), storage)
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: { sweep: { lr: [0.1] } },
    })
    const [record] = await storage.listRecords({ scope: 'proj', type: 'demo-run' })
    expect(record.content).not.toHaveProperty('thesis')
    expect(record.content).not.toHaveProperty('thesisTarget')
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

  it('records a failed job (no silent loss) and continues', async () => {
    const runner = stubRunner({
      jobResult: (job) =>
        (job.config as { lr: number }).lr === 0.1
          ? {
              status: 'failed',
              exitCode: 1,
              error: 'exit 1',
              summary: undefined,
              logTail: ['traceback', 'boom'],
            }
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
    const records = await storage.listRecords({ scope: 'proj', type: 'demo-run' })
    expect(records).toHaveLength(2)
    const failed = records.find((r) => (r.content as { status?: string }).status === 'failed')!
    expect(failed.content).toMatchObject({
      status: 'failed',
      error: 'exit 1',
      logTail: ['traceback', 'boom'],
      config: { lr: 0.1 },
      ranAt: NOW,
    })
  })

  it('does not record a job that ended aborted', async () => {
    const runner = stubRunner({
      jobResult: () => ({
        status: 'aborted',
        exitCode: null,
        error: 'aborted',
        summary: undefined,
      }),
    })
    const storage = memoryStorage()
    const { tools } = makeTools(runner, storage)
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: {},
    })
    expect(result.completed).toBe(0)
    expect(await storage.listRecords({ scope: 'proj', type: 'demo-run' })).toHaveLength(0)
  })

  it('dispatches runs concurrently when concurrency > 1', async () => {
    let active = 0
    let peak = 0
    const runner: ComputeRunner = {
      async calibrate() {
        return { secondsObserved: 1, unitsPerSecond: 100 }
      },
      runJob(job: ComputeJob): ComputeJobHandle {
        return {
          jobId: job.jobId,
          onLog: () => {},
          abort: () => {},
          done: (async (): Promise<ComputeJobResult> => {
            active++
            peak = Math.max(peak, active)
            await Promise.resolve()
            await Promise.resolve()
            active--
            return {
              jobId: job.jobId,
              status: 'completed',
              exitCode: 0,
              summary: { objective: 1 },
              logTail: [],
              durationMs: 1,
            }
          })(),
        }
      },
    }
    const m = manifest()
    delete m.calibrate
    const { tools } = makeTools(runner, memoryStorage())
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: m,
      spec: { sweep: { lr: [0.1, 0.2, 0.3, 0.4] } },
      concurrency: 3,
    })
    expect(peak).toBe(3)
    expect(result.completed).toBe(4)
  })

  it('skips setups already run under any seed when skipExplored is set', async () => {
    const storage = memoryStorage()
    const { tools } = makeTools(stubRunner(), storage)
    const m = manifest()
    delete m.calibrate
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: m,
      spec: { fixed: { lr: 0.3 }, seeds: [0] },
    })
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: m,
      spec: { fixed: { lr: 0.3 }, seeds: [0, 1] },
      skipExplored: true,
    })
    expect(result.skipped).toBe(2)
    expect(result.completed).toBe(0)
  })

  it('runs a fresh seed of an explored setup when skipExplored is off', async () => {
    const storage = memoryStorage()
    const { tools } = makeTools(stubRunner(), storage)
    const m = manifest()
    delete m.calibrate
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: m,
      spec: { fixed: { lr: 0.3 }, seeds: [0] },
    })
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: m,
      spec: { fixed: { lr: 0.3 }, seeds: [0, 1] },
    })
    expect(result.skipped).toBe(1)
    expect(result.completed).toBe(1)
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

  it('forwards starting + streamed @@PROGRESS markers to onItemProgress', async () => {
    const events: { key: string; progress: Record<string, unknown> }[] = []
    const runner = stubRunner()
    // Make the stub emit a progress marker on its log stream before resolving.
    const baseRunJob = runner.runJob.bind(runner)
    runner.runJob = (job) => {
      const handle = baseRunJob(job)
      const logs: ((line: string) => void)[] = []
      return {
        ...handle,
        onLog: (cb: (line: string) => void) => logs.push(cb),
        done: Promise.resolve().then(() => {
          logs.forEach((cb) => cb('@@PROGRESS {"phase":"train","done":700,"total":1460}'))
          return handle.done
        }),
      }
    }
    const { tools } = makeTools(runner, memoryStorage())
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: {},
      onItemProgress: (key, progress) => {
        events.push({ key, progress })
      },
    })
    expect(events[0].progress).toEqual({ phase: 'starting' })
    expect(events.some((e) => e.progress.phase === 'train' && e.progress.done === 700)).toBe(true)
  })

  it('emits a terminal onItemProgress when an item settles (so a host can drop it from the in-flight set)', async () => {
    const events: { key: string; progress: Record<string, unknown> }[] = []
    const { tools } = makeTools(stubRunner(), memoryStorage())
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: { fixed: { lr: 0.1 }, seeds: [0] },
      onItemProgress: (key, progress) => events.push({ key, progress }),
    })
    const terminal = events.filter((e) => e.progress.terminal === true)
    expect(terminal).toHaveLength(1)
    expect(terminal[0].progress.status).toBe('completed')
  })

  it('isolates an onItemProgress that throws — progress is a best-effort side-channel', async () => {
    const storage = memoryStorage()
    const { tools } = makeTools(stubRunner(), storage)
    await expect(
      tools.runTrainingCampaign({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: manifest(),
        spec: { fixed: { lr: 0.1 }, seeds: [0] },
        onItemProgress: () => {
          throw new Error('host progress sink blew up')
        },
      }),
    ).resolves.toBeDefined()
    const records = await storage.listRecords({ scope: 'proj', type: 'demo-run' })
    expect(records.some((r) => r.content.status === 'completed')).toBe(true)
  })

  it('isolates an onItemProgress that rejects asynchronously', async () => {
    const storage = memoryStorage()
    const { tools } = makeTools(stubRunner(), storage)
    await expect(
      tools.runTrainingCampaign({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: manifest(),
        spec: { fixed: { lr: 0.1 }, seeds: [0] },
        onItemProgress: () => Promise.reject(new Error('host write failed')),
      }),
    ).resolves.toBeDefined()
    const records = await storage.listRecords({ scope: 'proj', type: 'demo-run' })
    expect(records.some((r) => r.content.status === 'completed')).toBe(true)
  })

  it('emits an approximate wall-clock ETA during training when there is no calibration', async () => {
    let t = Date.parse('2020-01-01T00:00:00.000Z')
    const now = () => {
      t += 5000
      return new Date(t).toISOString()
    }
    const tools = createModelTrainerTools({
      computeRunner: stubRunner(),
      storage: memoryStorage(),
      now,
    })
    const trainEvents: { done: number; etaSeconds?: number; etaApprox?: boolean }[] = []
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest({ calibrate: undefined }),
      spec: { fixed: { lr: 0.1 }, seeds: [0, 1] },
      onProgress: (p) => {
        if (p.phase === 'train')
          trainEvents.push({ done: p.done, etaSeconds: p.etaSeconds, etaApprox: p.etaApprox })
      },
    })
    const withEta = trainEvents.find((e) => e.done > 0 && e.etaSeconds !== undefined)
    expect(withEta).toBeDefined()
    expect(withEta!.etaApprox).toBe(true)
    expect(withEta!.etaSeconds!).toBeGreaterThan(0)
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

  describe('pipeline versioning + unrunnable', () => {
    it("tags each run record with the manifest's pipelineVersion", async () => {
      const storage = memoryStorage()
      const { tools } = makeTools(stubRunner(), storage)
      await tools.runTrainingCampaign({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: manifest({ pipelineVersion: '3' }),
        spec: { sweep: { lr: [0.1] } },
      })
      const [record] = await storage.listRecords({ scope: 'proj', type: 'demo-run' })
      expect(record.content).toMatchObject({ pipelineVersion: '3' })
    })

    it("defaults pipelineVersion to '1' when the manifest declares none", async () => {
      const storage = memoryStorage()
      const { tools } = makeTools(stubRunner(), storage)
      await tools.runTrainingCampaign({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: manifest(),
        spec: { sweep: { lr: [0.1] } },
      })
      const [record] = await storage.listRecords({ scope: 'proj', type: 'demo-run' })
      expect(record.content).toMatchObject({ pipelineVersion: '1' })
    })

    it('re-explores a setup completed under an OLDER pipelineVersion (a version bump invalidates explored)', async () => {
      const storage = memoryStorage()
      const { tools } = makeTools(stubRunner(), storage)
      const m1 = manifest({ pipelineVersion: '1' })
      delete m1.calibrate
      await tools.runTrainingCampaign({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: m1,
        spec: { fixed: { lr: 0.3 }, seeds: [0] },
      })
      const m2 = manifest({ pipelineVersion: '2' })
      delete m2.calibrate
      const result = await tools.runTrainingCampaign({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: m2,
        spec: { fixed: { lr: 0.3 }, seeds: [0] },
        skipExplored: true,
      })
      expect(result.skipped).toBe(0)
      expect(result.completed).toBe(1)
    })

    it('still skips a setup completed under the SAME pipelineVersion', async () => {
      const storage = memoryStorage()
      const { tools } = makeTools(stubRunner(), storage)
      const m = manifest({ pipelineVersion: '2' })
      delete m.calibrate
      await tools.runTrainingCampaign({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: m,
        spec: { fixed: { lr: 0.3 }, seeds: [0] },
      })
      const result = await tools.runTrainingCampaign({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: m,
        spec: { fixed: { lr: 0.3 }, seeds: [0] },
        skipExplored: true,
      })
      expect(result.skipped).toBe(1)
      expect(result.completed).toBe(0)
    })

    it('still skips a setup explored under an older MINOR of the same major (minor bumps stay comparable)', async () => {
      const storage = memoryStorage()
      const { tools } = makeTools(stubRunner(), storage)
      const m1 = manifest({ pipelineVersion: '2.0' })
      delete m1.calibrate
      await tools.runTrainingCampaign({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: m1,
        spec: { fixed: { lr: 0.3 }, seeds: [0] },
      })
      const m2 = manifest({ pipelineVersion: '2.3' })
      delete m2.calibrate
      const result = await tools.runTrainingCampaign({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: m2,
        spec: { fixed: { lr: 0.3 }, seeds: [0] },
        skipExplored: true,
      })
      expect(result.skipped).toBe(1)
      expect(result.completed).toBe(0)
    })

    it('skips a setup marked unrunnable (same version), and force-runs it on refresh', async () => {
      const storage = memoryStorage()
      const { tools } = makeTools(stubRunner(), storage)
      const m = manifest({ pipelineVersion: '1' })
      delete m.calibrate
      const setupKey = setupKeyOf({ lr: 0.3, steps: 100 })
      await storage.upsertRecord({
        scope: 'proj',
        type: 'demo-run-unrunnable',
        key: setupKey,
        content: { setupKey, pipelineVersion: '1', unrunnable: true },
      })
      const skipResult = await tools.runTrainingCampaign({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: m,
        spec: { fixed: { lr: 0.3, steps: 100 }, seeds: [0] },
        skipExplored: true,
      })
      expect(skipResult.skipped).toBe(1)
      const forceResult = await tools.runTrainingCampaign({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: m,
        spec: { fixed: { lr: 0.3, steps: 100 }, seeds: [0] },
        refresh: true,
      })
      expect(forceResult.completed).toBe(1)
    })

    it('ignores an unrunnable mark from a DIFFERENT pipelineVersion', async () => {
      const storage = memoryStorage()
      const { tools } = makeTools(stubRunner(), storage)
      const setupKey = setupKeyOf({ lr: 0.3, steps: 100 })
      await storage.upsertRecord({
        scope: 'proj',
        type: 'demo-run-unrunnable',
        key: setupKey,
        content: { setupKey, pipelineVersion: '1', unrunnable: true },
      })
      const m2 = manifest({ pipelineVersion: '2' })
      delete m2.calibrate
      const result = await tools.runTrainingCampaign({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: m2,
        spec: { fixed: { lr: 0.3, steps: 100 }, seeds: [0] },
        skipExplored: true,
      })
      expect(result.completed).toBe(1)
      expect(result.skipped).toBe(0)
    })
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

  describe('evaluateTrainingRuns (batch, parallel)', () => {
    it('evaluates every run and persists an evaluation record for each', async () => {
      const storage = memoryStorage()
      await seedCompletedRun(storage, 'run1')
      await seedCompletedRun(storage, 'run2')
      await seedCompletedRun(storage, 'run3')
      const runner = stubRunner({ jobResult: () => ({ summary: { objective: 42 } }) })
      const { tools } = makeTools(runner, storage)
      const result = await tools.evaluateTrainingRuns({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: evalManifest(),
        runKeys: ['run1', 'run2', 'run3'],
        concurrency: 3,
      })
      expect(result).toMatchObject({ recordType: 'demo-run', evaluated: 3, failed: 0 })
      expect(result.results.map((r) => r.runKey).sort()).toEqual(['run1', 'run2', 'run3'])
      for (const key of ['run1', 'run2', 'run3']) {
        const rec = await storage.readRecord({ scope: 'proj', type: 'demo-run-evaluation', key })
        expect(rec?.content).toMatchObject({ runKey: key, objective: 42, status: 'completed' })
      }
    })

    it('reports cumulative progress and notifies onRecordWritten per run', async () => {
      const storage = memoryStorage()
      await seedCompletedRun(storage, 'run1')
      await seedCompletedRun(storage, 'run2')
      const runner = stubRunner({ jobResult: () => ({ summary: { objective: 1 } }) })
      const { tools } = makeTools(runner, storage)
      const written: string[] = []
      const totals: number[] = []
      await tools.evaluateTrainingRuns({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: evalManifest(),
        runKeys: ['run1', 'run2'],
        onRecordWritten: (type, key) => written.push(`${type}:${key}`),
        onProgress: (p) => totals.push(p.done),
      })
      expect(written.sort()).toEqual(['demo-run-evaluation:run1', 'demo-run-evaluation:run2'])
      expect(totals.at(-1)).toBe(2)
    })

    it('isolates a failed run: the rest still evaluate and the failure is reported', async () => {
      const storage = memoryStorage()
      await seedCompletedRun(storage, 'good')
      await storage.upsertRecord({
        scope: 'proj',
        type: 'demo-run',
        key: 'nockpt',
        content: { objective: 1, status: 'completed', config: {} },
      })
      const runner = stubRunner({ jobResult: () => ({ summary: { objective: 7 } }) })
      const { tools } = makeTools(runner, storage)
      const result = await tools.evaluateTrainingRuns({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: evalManifest(),
        runKeys: ['good', 'nockpt'],
      })
      expect(result.evaluated).toBe(1)
      expect(result.failed).toBe(1)
      expect(result.failures).toEqual([
        { runKey: 'nockpt', error: expect.stringMatching(/checkpoint/) },
      ])
      expect(
        await storage.readRecord({ scope: 'proj', type: 'demo-run-evaluation', key: 'good' }),
      ).toBeDefined()
    })

    it('throws when the manifest declares no evaluate command', async () => {
      const storage = memoryStorage()
      await seedCompletedRun(storage, 'run1')
      const { tools } = makeTools(stubRunner(), storage)
      await expect(
        tools.evaluateTrainingRuns({
          scope: 'proj',
          projectRoot: '/repo',
          manifest: manifest(),
          runKeys: ['run1'],
        }),
      ).rejects.toThrow(/evaluate/)
    })
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

  it('judges only the selected runKeys when provided', async () => {
    const storage = memoryStorage()
    await seedRun(storage, 'a', 10)
    await seedRun(storage, 'b', 90)
    await seedRun(storage, 'c', 50)
    const executor = stubExecutor(
      JSON.stringify([
        { key: 'a', score: 60, why: 'x' },
        { key: 'c', score: 70, why: 'y' },
      ]),
    )
    const { tools } = makeJudgeTools(executor, storage)
    const result = await tools.judgeTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
      runKeys: ['a', 'c'],
    })
    expect(result.judged).toBe(2)
    expect(result.verdicts.map((v) => v.key).sort()).toEqual(['a', 'c'])
    const sent = JSON.parse(executor.requests[0].userContent) as { key: string }[]
    expect(sent.map((r) => r.key).sort()).toEqual(['a', 'c'])
    const records = await storage.listRecords({ scope: 'proj', type: 'demo-run-verdict' })
    expect(records.map((r) => r.key).sort()).toEqual(['a', 'c'])
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

  it('persists valid proposals as untested hypothesis records', async () => {
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
      status: 'untested',
      verdictSource: 'auto',
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

describe('xaiNarrate', () => {
  const TRACE = {
    steps: [{ step: 0, action: 'hold' }],
    actionCounts: { hold: 90, buy: 10 },
    featureAttribution: {
      byGroup: { '1h': 0.4, '1d': 0.1 },
      method: 'gradient-saliency',
      sanityCheck: { passed: true, rankCorrelation: 0.05 },
    },
    rewardBreakdown: { base: 1.5, turnover_penalty: -0.2 },
    latentMap: {
      varianceExplained: 0.7,
      points: [
        { x: 0, y: 0, action: 'hold' },
        { x: 1, y: 1, action: 'buy' },
        { x: 2, y: 0, action: 'hold' },
      ],
      probe: { accuracy: 0.8, baseline: 0.6 },
    },
  }

  it('narrates ONE run: keys the record by run, digests its trace, ranks it, and feeds the model the facts', async () => {
    const storage = memoryStorage()
    await seedRun(storage, 'aaa', 10, { config: { lr: 0.1 } })
    await seedRun(storage, 'bbb', 90, { config: { lr: 0.9 }, artifacts: { decisionTrace: TRACE } })
    const executor = stubExecutor('hold-dominant; saliency trustworthy.')
    const { tools } = makeJudgeTools(executor, storage)
    const result = await tools.xaiNarrate({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
      runKey: 'bbb',
    })
    expect(result).toMatchObject({
      recordType: 'demo-run',
      runKey: 'bbb',
      runCount: 2,
      narratedBy: 'openai/m',
      narratedAt: NOW,
    })
    const records = await storage.listRecords({ scope: 'proj', type: 'demo-run-xai-narrative' })
    expect(records).toHaveLength(1)
    expect(records[0].key).toBe('bbb')
    expect(records[0].content).toMatchObject({
      narrative: 'hold-dominant; saliency trustworthy.',
      runKey: 'bbb',
      runCount: 2,
      criterionKey: 'objective',
    })
    const uc = executor.requests[0].userContent
    expect(uc).toContain('Run bbb')
    expect(uc).toMatch(/Action mix: hold=90/)
    expect(uc).toMatch(/sanity check PASSED/)
    expect(uc).toMatch(/linear probe.*80% vs a 60%/)
    expect(uc).toMatch(/ranks #1 of 2/) // bbb's objective (90) is best under max
  })

  it('includes the sibling decision-diff when siblingKey is given', async () => {
    const aligned = (action: string) => ({
      steps: [{ step: 0, action, reward: 1 }],
      totalSteps: 1,
      actionCounts: { [action]: 1 },
    })
    const storage = memoryStorage()
    await seedRun(storage, 'base', 10, {
      config: { lr: 0.9 },
      dataset: { asset: 'BTC' },
      artifacts: { decisionTrace: aligned('hold') },
    })
    await seedRun(storage, 'tweak', 20, {
      config: { lr: 0.1 },
      dataset: { asset: 'BTC' },
      artifacts: { decisionTrace: aligned('buy') },
    })
    const executor = stubExecutor('the tweak buys.')
    const { tools } = makeJudgeTools(executor, storage)
    await tools.xaiNarrate({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
      runKey: 'tweak',
      siblingKey: 'base',
    })
    expect(executor.requests[0].userContent).toMatch(/nearest sibling base \(changed lr 0.9→0.1\)/)
  })

  it('honours a non-default criterion and fires onRecordWritten with the run key', async () => {
    const storage = memoryStorage()
    await seedRun(storage, 'aaa', 10, { durationMs: 1000 })
    const written: string[] = []
    const { tools } = makeJudgeTools(stubExecutor('fast.'), storage)
    await tools.xaiNarrate({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
      runKey: 'aaa',
      criterion: { key: 'runtime', direction: 'min', label: 'runtime' },
      onRecordWritten: (t, k) => written.push(`${t}/${k}`),
    })
    const records = await storage.listRecords({ scope: 'proj', type: 'demo-run-xai-narrative' })
    expect((records[0].content as { criterionKey: string }).criterionKey).toBe('runtime')
    expect(written).toContain('demo-run-xai-narrative/aaa')
  })

  it('throws for an unknown run and without an inference executor', async () => {
    const storage = memoryStorage()
    await seedRun(storage, 'aaa', 10)
    const { tools } = makeJudgeTools(stubExecutor('n'), storage)
    await expect(
      tools.xaiNarrate({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: manifest(),
        llmConfig: LLM,
        runKey: 'ghost',
      }),
    ).rejects.toThrow(/not a completed run/)

    const { tools: noLlm } = makeJudgeTools(undefined, storage)
    await expect(
      noLlm.xaiNarrate({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: manifest(),
        llmConfig: LLM,
        runKey: 'aaa',
      }),
    ).rejects.toThrow(/inferenceExecutor/)
  })
})

describe('getRunData / getRunXAI (agent read tools)', () => {
  // Register the training project so the read tools can resolve a run id → its recordType.
  async function seedProject(storage: DataStorage) {
    await storage.upsertRecord({
      scope: 'proj',
      type: 'trainer-project-manifest',
      key: 'blackswan',
      content: { manifest: manifest(), dir: 'BlackSwan' },
    })
  }

  it('getRunData resolves the recordType from the registered project and returns a trimmed record', async () => {
    const storage = memoryStorage()
    await seedProject(storage)
    await seedRun(storage, 'aaa', 12, {
      config: { lr: 0.1 },
      metrics: { total_return_pct: 12 },
      series: { equity: [1, 2, 3] },
      artifacts: {
        decisionTrace: {
          steps: [{ step: 0, action: 'hold' }],
          actionCounts: { hold: 90, buy: 10 },
          featureAttribution: {
            byGroup: { '1h': 0.4 },
            method: 'gradient-saliency',
            sanityCheck: { passed: true },
          },
        },
        checkpoint: 'ckpt',
      },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.getRunData({ scope: 'proj', runKey: 'aaa' })
    expect(result.found).toBe(true)
    expect(result.recordType).toBe('demo-run')
    const run = result.run as Record<string, unknown>
    expect(run.config).toEqual({ lr: 0.1 })
    // heavy parts stripped, compact digest kept
    expect(run.series).toBeUndefined()
    expect((run.artifacts as Record<string, unknown>).decisionTrace).toBeUndefined()
    expect((run.artifacts as Record<string, unknown>).checkpoint).toBe('ckpt')
    expect(run.decisionTraceDigest).toMatchObject({
      actionCounts: { hold: 90, buy: 10 },
      attributionMethod: 'gradient-saliency',
      attributionSanityPassed: true,
    })
  })

  it('getRunXAI returns the structured digest with the run ranked among all runs', async () => {
    const storage = memoryStorage()
    await seedProject(storage)
    await seedRun(storage, 'lo', 10, { config: { lr: 0.1 } })
    await seedRun(storage, 'hi', 90, { config: { lr: 0.9 } })
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.getRunXAI({ scope: 'proj', runKey: 'hi' })
    expect(result.found).toBe(true)
    expect(result.recordType).toBe('demo-run')
    expect(result.runCount).toBe(2)
    expect(result.analysis?.runKey).toBe('hi')
    expect(result.analysis?.rank).toEqual({ position: 1, total: 2 }) // hi (90) is best under max
    expect(result.analysis?.criterion.key).toBe('objective')
    expect(Array.isArray(result.analysis?.importances)).toBe(true)
  })

  it('both return found:false for an unknown run id', async () => {
    const storage = memoryStorage()
    await seedProject(storage)
    await seedRun(storage, 'aaa', 10)
    const { tools } = makeTools(stubRunner(), storage)
    const data = await tools.getRunData({ scope: 'proj', runKey: 'ghost' })
    const xai = await tools.getRunXAI({ scope: 'proj', runKey: 'ghost' })
    expect(data.found).toBe(false)
    expect(data.error).toMatch(/ghost/)
    expect(xai.found).toBe(false)
  })

  it('getRunData keeps a minimal trace digest when the trace has no attribution/reward/latent', async () => {
    const storage = memoryStorage()
    await seedProject(storage)
    await seedRun(storage, 'aaa', 10, {
      artifacts: {
        decisionTrace: { steps: [{ step: 0, action: 'hold' }], actionCounts: { hold: 5 } },
      },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.getRunData({ scope: 'proj', runKey: 'aaa' })
    expect(result.run?.decisionTraceDigest).toEqual({ totalSteps: 1, actionCounts: { hold: 5 } })
  })

  it('getRunData ignores a non-completed run (only completed runs resolve)', async () => {
    const storage = memoryStorage()
    await seedProject(storage)
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'failed1',
      content: { status: 'failed', config: { lr: 0.1 } },
    })
    const { tools } = makeTools(stubRunner(), storage)
    expect((await tools.getRunData({ scope: 'proj', runKey: 'failed1' })).found).toBe(false)
  })
})

describe('analyzePaperFromUrl', () => {
  const HYPS = [
    {
      title: 'Vol scaling helps',
      rationale: 'scale position by realized vol',
      spec: { fixed: { lr: 0.5 } },
    },
    {
      title: 'Longer training helps',
      rationale: 'more steps',
      spec: { sweep: { steps: [100, 200] } },
    },
  ]
  const draftWith = (hypotheses?: unknown) =>
    JSON.stringify({
      title: 'Deep RL for Trading',
      authors: 'A. Researcher',
      year: 2023,
      claim: 'RL beats buy-and-hold OOS',
      approach: 'PPO over price features',
      claimedMetrics: { sharpe: 1.3 },
      assumptions: { fees: false, notes: 'no costs modelled' },
      verdictNote: 'omits fees — likely fluff after costs',
      tags: ['rl', 'trading'],
      ...(hypotheses === undefined ? {} : { hypotheses }),
    })
  const DRAFT = draftWith(HYPS)
  const fetchStub = async () => 'the fetched paper text'
  const base = (overrides = {}) => ({
    scope: 'proj',
    projectRoot: '/repo',
    manifest: manifest(),
    url: 'https://arxiv.org/abs/1234.5678',
    llmConfig: LLM,
    fetchPaperText: fetchStub,
    ...overrides,
  })

  it('fetches, summarises, and upserts a draft paper record for review', async () => {
    const storage = memoryStorage()
    const executor = stubExecutor(DRAFT)
    const { tools } = makeJudgeTools(executor, storage)
    const written: string[] = []
    const result = await tools.analyzePaperFromUrl(
      base({ onRecordWritten: (t: string) => written.push(t) }),
    )
    expect(result).toMatchObject({
      recordType: 'demo-run',
      analyzedBy: 'openai/m',
      analyzedAt: NOW,
    })
    expect(result.paper).toMatchObject({
      title: 'Deep RL for Trading',
      claim: 'RL beats buy-and-hold OOS',
      url: 'https://arxiv.org/abs/1234.5678',
      status: 'untested',
      source: 'research',
      createdAt: NOW,
    })
    // the model is handed the fetched text — it does not browse
    expect(executor.requests[0].userContent).toContain('the fetched paper text')
    const records = await storage.listRecords({ scope: 'proj', type: 'demo-run-paper' })
    expect(records).toHaveLength(1)
    expect((records[0].content as { id: string }).id).toBe(records[0].key)
    // paper + its two hypotheses persisted and cross-linked
    expect(result.paper.hypothesisIds).toEqual(result.linkedHypothesisIds)
    expect(result.linkedHypothesisIds).toHaveLength(2)
    expect(written).toEqual(['demo-run-hypothesis', 'demo-run-hypothesis', 'demo-run-paper'])
  })

  it('extracts the testable hypotheses, links them to the paper, and marks them untested/auto', async () => {
    const storage = memoryStorage()
    const { tools } = makeJudgeTools(stubExecutor(DRAFT), storage)
    const result = await tools.analyzePaperFromUrl(base())
    const hyps = await storage.listRecords({ scope: 'proj', type: 'demo-run-hypothesis' })
    expect(hyps).toHaveLength(2)
    for (const r of hyps) {
      const h = r.content as {
        id: string
        status: string
        verdictSource: string
        source: string
        paperIds: string[]
      }
      expect(h.id).toBe(r.key)
      expect(h).toMatchObject({ status: 'untested', verdictSource: 'auto', source: 'paper' })
      expect(h.paperIds).toContain(result.paper.id)
    }
    expect([...result.paper.hypothesisIds!].sort()).toEqual(hyps.map((r) => r.key).sort())
  })

  it('dedups identical specs within one paper to a single hypothesis', async () => {
    const storage = memoryStorage()
    const dup = [HYPS[0], { ...HYPS[0], title: 'restated' }]
    const { tools } = makeJudgeTools(stubExecutor(draftWith(dup)), storage)
    const result = await tools.analyzePaperFromUrl(base())
    const hyps = await storage.listRecords({ scope: 'proj', type: 'demo-run-hypothesis' })
    expect(hyps).toHaveLength(1)
    expect(result.paper.hypothesisIds).toHaveLength(1)
  })

  it('links a PRE-EXISTING hypothesis (same spec) without duplicating or clobbering its verdict', async () => {
    const storage = memoryStorage()
    const priorId = hashTrainingConfig({ fixed: { lr: 0.5 } })
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run-hypothesis',
      key: priorId,
      content: {
        id: priorId,
        title: 'pre-existing',
        rationale: 'set earlier',
        spec: { fixed: { lr: 0.5 } },
        status: 'proven',
        verdictSource: 'manual',
        source: 'human',
        paperIds: [],
        createdAt: '2026-06-01T00:00:00.000Z',
        updatedAt: '2026-06-01T00:00:00.000Z',
      },
    })
    const { tools } = makeJudgeTools(stubExecutor(DRAFT), storage)
    const result = await tools.analyzePaperFromUrl(base())
    const hyps = await storage.listRecords({ scope: 'proj', type: 'demo-run-hypothesis' })
    expect(hyps).toHaveLength(2) // the prior + the one new (steps) hypothesis, not a duplicate
    const prior = hyps.find((r) => r.key === priorId)!.content as {
      status: string
      verdictSource: string
      paperIds: string[]
    }
    expect(prior).toMatchObject({ status: 'proven', verdictSource: 'manual' })
    expect(prior.paperIds).toContain(result.paper.id)
  })

  it('handles a paper with no runnable hypotheses (container stays empty)', async () => {
    const storage = memoryStorage()
    const { tools } = makeJudgeTools(stubExecutor(draftWith([])), storage)
    const result = await tools.analyzePaperFromUrl(base())
    expect(result.hypotheses).toEqual([])
    expect(result.paper.hypothesisIds).toEqual([])
    const hyps = await storage.listRecords({ scope: 'proj', type: 'demo-run-hypothesis' })
    expect(hyps).toHaveLength(0)
  })

  it('tolerates a response with no hypotheses field at all', async () => {
    const storage = memoryStorage()
    const { tools } = makeJudgeTools(stubExecutor(draftWith(undefined)), storage)
    const result = await tools.analyzePaperFromUrl(base())
    expect(result.hypotheses).toEqual([])
    expect(result.paper.hypothesisIds).toEqual([])
  })

  it('throws without an inference executor', async () => {
    const { tools } = makeJudgeTools(undefined, memoryStorage())
    await expect(tools.analyzePaperFromUrl(base())).rejects.toThrow(/inference/i)
  })

  it('throws when the model returns no usable summary', async () => {
    const { tools } = makeJudgeTools(stubExecutor('{"nope": true}'), memoryStorage())
    await expect(tools.analyzePaperFromUrl(base())).rejects.toThrow(/usable paper summary/i)
  })

  it('propagates a fetch failure', async () => {
    const { tools } = makeJudgeTools(stubExecutor(DRAFT), memoryStorage())
    await expect(
      tools.analyzePaperFromUrl(
        base({
          fetchPaperText: async () => {
            throw new Error('could not fetch paper (HTTP 404)')
          },
        }),
      ),
    ).rejects.toThrow(/404/)
  })
})

describe('suggestPaperHypotheses', () => {
  const seedPaper = async (storage: DataStorage, overrides: Record<string, unknown> = {}) =>
    storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run-paper',
      key: 'p1',
      content: {
        id: 'p1',
        title: 'Vol scaling paper',
        claim: 'scale by vol',
        status: 'untested',
        source: 'manual',
        createdAt: NOW,
        updatedAt: NOW,
        ...overrides,
      },
    })
  const seedHyp = async (storage: DataStorage, id: string, content: Record<string, unknown>) =>
    storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run-hypothesis',
      key: id,
      content: { id, ...content },
    })
  const base = (overrides = {}) => ({
    scope: 'proj',
    projectRoot: '/repo',
    manifest: manifest(),
    paperId: 'p1',
    llmConfig: LLM,
    ...overrides,
  })

  it('links matched existing hypotheses and creates+links the suggested new ones', async () => {
    const storage = memoryStorage()
    await seedPaper(storage)
    await seedHyp(storage, 'e1', {
      title: 'existing vol',
      rationale: 'r',
      spec: { fixed: { lr: 0.5 } },
      status: 'proven',
      verdictSource: 'manual',
      source: 'human',
      paperIds: [],
    })
    await seedHyp(storage, 'e2', {
      title: 'unrelated',
      spec: { fixed: { steps: 50 } },
      paperIds: [],
    })
    const exec = stubExecutor(
      JSON.stringify({
        matchExistingIds: ['e1', 'missing-id'],
        newHypotheses: [{ title: 'New', rationale: 'R', spec: { sweep: { steps: [100, 200] } } }],
      }),
    )
    const { tools } = makeJudgeTools(exec, storage)
    const written: string[] = []
    const result = await tools.suggestPaperHypotheses(
      base({ onRecordWritten: (t: string) => written.push(t) }),
    )
    expect(result.linkedExistingIds).toEqual(['e1']) // 'missing-id' filtered out
    expect(result.newHypotheses).toHaveLength(1)
    expect(result.newHypotheses[0]).toMatchObject({
      source: 'paper',
      status: 'untested',
      verdictSource: 'auto',
    })
    const newId = result.newHypotheses[0].id
    // paper links both
    expect([...result.paper.hypothesisIds!].sort()).toEqual(['e1', newId].sort())
    // e1 keeps its manual proven verdict, gains the paper link
    const e1 = (await storage.readRecord({
      scope: 'proj',
      type: 'demo-run-hypothesis',
      key: 'e1',
    }))!.content as { status: string; verdictSource: string; paperIds: string[] }
    expect(e1).toMatchObject({ status: 'proven', verdictSource: 'manual' })
    expect(e1.paperIds).toContain('p1')
    // e2 untouched (no paper link)
    const e2 = (await storage.readRecord({
      scope: 'proj',
      type: 'demo-run-hypothesis',
      key: 'e2',
    }))!.content as { paperIds: string[] }
    expect(e2.paperIds).toEqual([])
    expect(written).toContain('demo-run-paper')
  })

  it('dedups a suggested NEW hypothesis whose spec already exists (links, no duplicate)', async () => {
    const storage = memoryStorage()
    await seedPaper(storage)
    const dupId = hashTrainingConfig({ fixed: { lr: 0.5 } })
    await seedHyp(storage, dupId, {
      title: 'already here',
      spec: { fixed: { lr: 0.5 } },
      paperIds: [],
    })
    const exec = stubExecutor(
      JSON.stringify({
        matchExistingIds: [],
        newHypotheses: [{ title: 'restated', rationale: 'R', spec: { fixed: { lr: 0.5 } } }],
      }),
    )
    const { tools } = makeJudgeTools(exec, storage)
    const result = await tools.suggestPaperHypotheses(base())
    expect(result.newHypotheses).toHaveLength(0) // it already existed → linked, not created
    expect(result.linkedHypothesisIds).toEqual([dupId])
    const hyps = await storage.listRecords({ scope: 'proj', type: 'demo-run-hypothesis' })
    expect(hyps).toHaveLength(1)
  })

  it('tolerates a paper-URL fetch failure (uses stored fields instead)', async () => {
    const storage = memoryStorage()
    await seedPaper(storage, { url: 'https://arxiv.org/abs/1' })
    const exec = stubExecutor(JSON.stringify({ matchExistingIds: [], newHypotheses: [] }))
    const { tools } = makeJudgeTools(exec, storage)
    const result = await tools.suggestPaperHypotheses(
      base({
        fetchPaperText: async () => {
          throw new Error('HTTP 404')
        },
      }),
    )
    expect(result.linkedHypothesisIds).toEqual([])
  })

  it('throws when the paper does not exist', async () => {
    const { tools } = makeJudgeTools(stubExecutor('{}'), memoryStorage())
    await expect(tools.suggestPaperHypotheses(base())).rejects.toThrow(/no paper/i)
  })

  it('throws without an inference executor', async () => {
    const storage = memoryStorage()
    await seedPaper(storage)
    const { tools } = makeJudgeTools(undefined, storage)
    await expect(tools.suggestPaperHypotheses(base())).rejects.toThrow(/inference/i)
  })
})
