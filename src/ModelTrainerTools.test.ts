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
  DataQuery,
  DataRecord,
  DataRecordInput,
  DataStorage,
  DeepResearchTools,
  DiscoveredSource,
  ClaimVerdict,
  InferenceExecutor,
  InferenceRequest,
  LLMConfig,
} from 'thefactory-tools/types'
import type { TrainerManifest, TrainingCampaignProgress } from './modelTrainerTypes.js'
import { createModelTrainerTools } from './ModelTrainerTools.js'
import { initExplorationState } from './explorationUtils.js'
import { hashTrainingConfig, setupKeyOf } from './modelTrainerHelpers.js'
import { HEAVY_RUN_FIELDS } from './modelTrainerConstants.js'

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
  queries: DataQuery[]
}

// Faithfully drop `omit` paths (incl. nested `a.b`) from record content, so the fake mirrors the real
// backends' projection — a list that omits heavy fields genuinely lacks them, which is what forces the
// trace-dependent read paths to re-fetch the full record by key.
function applyOmit(content: unknown, omit: string[] | undefined): unknown {
  if (!omit?.length || !content || typeof content !== 'object') return content
  const out: Record<string, unknown> = { ...(content as Record<string, unknown>) }
  const nested = new Map<string, string[]>()
  for (const path of omit) {
    const dot = path.indexOf('.')
    if (dot === -1) delete out[path]
    else nested.set(path.slice(0, dot), [...(nested.get(path.slice(0, dot)) ?? []), path.slice(dot + 1)])
  }
  for (const [head, subs] of nested) {
    if (out[head] && typeof out[head] === 'object') out[head] = applyOmit(out[head], subs)
  }
  return out
}

function memoryStorage(): MemoryStorage {
  const rows = new Map<string, DataRecord>()
  const queries: DataQuery[] = []
  const keyOf = (scope: string, type: string, key: string | null | undefined) =>
    `${scope}|${type}|${key ?? ''}`
  return {
    rows,
    queries,
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
      queries.push(query)
      return [...rows.values()]
        .filter((r) => r.scope === query.scope && (!query.type || r.type === query.type))
        .map((r) => (query.omit ? { ...r, content: applyOmit(r.content, query.omit) } : r))
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
    tools: createModelTrainerTools({
      computeRunner: runner,
      storage,
      logger,
      now: () => NOW,
      // Take host RAM out of the concurrency math by default so CPU-concurrency assertions are
      // deterministic; tests that exercise the RAM ceiling inject a finite budget explicitly.
      availableMemoryBytes: () => Number.MAX_SAFE_INTEGER,
    }),
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
      availableMemoryBytes: () => Number.MAX_SAFE_INTEGER,
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

  it('migrates spec.fixed via manifest.migrations BEFORE planning, so a dispatched run executes under the new shape', async () => {
    const runner = stubRunner()
    const storage = memoryStorage()
    const { tools } = makeTools(runner, storage)
    const m = manifest({
      levers: {
        lr: { type: 'number', default: 0.01 },
        steps: { type: 'number', default: 100 },
        reward_model: {
          type: 'choice',
          choices: ['combo_all', 'combo_unified'],
          default: 'combo_unified',
        },
        combo_sell: { type: 'number', default: 0 },
      },
      migrations: [
        {
          match: { reward_model: 'combo_all' },
          set: { reward_model: 'combo_unified', combo_sell: 1000 },
        },
      ],
    })
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: m,
      // An OLD queued spec: a run dispatched from this must NOT execute as combo_all.
      spec: { fixed: { reward_model: 'combo_all', lr: 0.1 } },
    })
    expect(runner.jobs).toHaveLength(1)
    expect(runner.jobs[0].config).toMatchObject({
      reward_model: 'combo_unified',
      combo_sell: 1000,
      lr: 0.1,
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

  it('re-runs an existing run UNDER ITS KEY — updates that record in place, never a new one', async () => {
    const runner = stubRunner()
    const storage = memoryStorage()
    const { tools } = makeTools(runner, storage)
    // A pre-existing failed run keyed by an opaque id whose stored config does NOT re-hash to that key
    // (the real case: non-lever fields / defaults / rewrites) — so reconstructing+rehashing would land
    // on a DIFFERENT key and leak a new record. The batch re-run carries the run's key to prevent that.
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'orig-run-key',
      content: { objective: 1, status: 'failed', config: { lr: 0.1 } },
    })
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      spec: { configs: [{ config: { lr: 0.1 }, key: 'orig-run-key' }] },
      refresh: true,
    })
    expect(result.planned).toBe(1)
    expect(result.completed).toBe(1)
    expect(runner.jobs).toHaveLength(1)
    expect(runner.jobs[0].jobId).toBe('orig-run-key')
    const records = await storage.listRecords({ scope: 'proj', type: 'demo-run' })
    expect(records).toHaveLength(1) // NOT 2 — updated in place
    expect(records[0].key).toBe('orig-run-key')
    expect(records[0].content).toMatchObject({ status: 'completed' })
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

  it('SPEEDUP: auto-parallelizes to floor(cpus/maxThreadsPerRun) and threads the thread-cap env into each run', async () => {
    let active = 0
    let peak = 0
    const jobs: ComputeJob[] = []
    const runner: ComputeRunner = {
      async calibrate() {
        return { secondsObserved: 1, unitsPerSecond: 100 }
      },
      runJob(job: ComputeJob): ComputeJobHandle {
        jobs.push(job)
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
    const m = manifest({ maxThreadsPerRun: 2 })
    delete m.calibrate
    // 8 cores / 2 threads-per-run -> the campaign should run 4 at once, NOT sequentially.
    const tools = createModelTrainerTools({
      computeRunner: runner,
      storage: memoryStorage(),
      logger: { info: vi.fn(), warn: vi.fn() },
      now: () => NOW,
      availableParallelism: () => 8,
      availableMemoryBytes: () => Number.MAX_SAFE_INTEGER,
    })
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: m,
      spec: { sweep: { lr: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] } },
    })
    expect(result.completed).toBe(6)
    expect(peak).toBe(4) // floor(8 / 2), not 1 (sequential)
    expect(jobs[0].env).toMatchObject({ BS_NUM_THREADS: '2', OMP_NUM_THREADS: '2' })
  })

  it('MEMORY default-on: caps the pool by host RAM even when the manifest declares no per-run estimate', async () => {
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
    const m = manifest({ maxThreadsPerRun: 2 }) // no maxMemoryBytesPerRun → uses the conservative default
    delete m.calibrate
    const logger = { info: vi.fn(), warn: vi.fn() }
    // 8 cores/2-threads → CPU wants 4, but a 5 GiB budget / 2 GiB default per run fits only 2.
    const tools = createModelTrainerTools({
      computeRunner: runner,
      storage: memoryStorage(),
      logger,
      now: () => NOW,
      availableParallelism: () => 8,
      availableMemoryBytes: () => 5 * 1024 ** 3,
    })
    const result = await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: m,
      spec: { sweep: { lr: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] } },
    })
    expect(result.completed).toBe(6)
    expect(peak).toBe(2) // floor(5 / 2), not floor(8/2)=4 — RAM is the binding constraint
    expect(logger.warn).toHaveBeenCalledWith(
      'campaign concurrency reduced to fit host memory',
      expect.objectContaining({ concurrency: 2 }),
    )
  })

  it('keeps the sequential default and sets no per-run env when the manifest declares no thread appetite', async () => {
    const runner = stubRunner()
    const { tools } = makeTools(runner, memoryStorage())
    const m = manifest()
    delete m.calibrate
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: m,
      spec: { sweep: { lr: [0.1, 0.2] } },
    })
    expect(runner.jobs[0].env).toBeUndefined()
  })

  it('auto-applies a benchmarked model preferredDevice — mps for a single run, but NOT for a parallel sweep', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run-model',
      key: 'reppo-custom',
      content: {
        id: 'reppo-custom',
        slug: 'reppo-custom',
        flavors: [{ modelName: 'reppo-custom' }],
        preferredDevice: 'mps',
        createdAt: NOW,
        updatedAt: NOW,
      },
    })
    const m = manifest({
      levers: {
        lr: { type: 'number', default: 0.01 },
        steps: { type: 'number', default: 100 },
        model_name: { type: 'choice', choices: ['reppo-custom', 'ppo'], default: 'reppo-custom' },
      },
    })
    delete m.calibrate
    const runner = stubRunner()
    const { tools } = makeTools(runner, storage)
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: m,
      spec: { configs: [{ config: { model_name: 'reppo-custom' } }] },
      concurrency: 1,
      refresh: true,
    })
    expect((runner.jobs[0].config as { device?: string }).device).toBe('mps')

    runner.jobs.length = 0
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: m,
      spec: { configs: [{ config: { model_name: 'reppo-custom' } }] },
      concurrency: 3, // parallel -> one GPU can't be shared -> mps NOT applied
      refresh: true,
    })
    expect((runner.jobs[0].config as { device?: string }).device).toBeUndefined()
  })

  it('benchmarkModelDevice runs the benchmark, persists the winning device, and passes the model via env', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run-model',
      key: 'reppo-custom',
      content: {
        id: 'reppo-custom',
        slug: 'reppo-custom',
        name: 'Reppo (custom)',
        flavors: [{ modelName: 'reppo-custom' }],
        createdAt: NOW,
        updatedAt: NOW,
      },
    })
    const runner = stubRunner({
      calibration: {
        secondsObserved: 90,
        summary: {
          deviceBenchmark: {
            modelName: 'reppo-custom',
            bestDevice: 'mps',
            speedup: 1.45,
            usPerStep: { cpu: 67712, mps: 46647 },
            availableDevices: ['cpu', 'mps'],
          },
        },
      },
    })
    const { tools } = makeTools(runner, storage)
    const m = manifest({
      benchmarkDevice: 'bin/python -m trainer.bench_device --summary-out {summaryOut}',
    })
    const res = await tools.benchmarkModelDevice({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: m,
      modelId: 'reppo-custom',
    })
    expect(res.preferredDevice).toBe('mps')
    expect(res.deviceBenchmark.speedup).toBe(1.45)
    // the benchmark command received the model to run via env
    expect(runner.probes[0].env).toMatchObject({ BENCH_MODEL_NAME: 'reppo-custom' })
    // the winner is persisted onto the model record for the viewer to read
    const rec = await storage.readRecord({
      scope: 'proj',
      type: 'demo-run-model',
      key: 'reppo-custom',
    })
    const content = rec!.content as Record<string, unknown>
    expect(content.preferredDevice).toBe('mps')
    expect((content.deviceBenchmark as { bestDevice: string }).bestDevice).toBe('mps')
  })

  it('benchmarkModelDevice throws when the manifest declares no benchmarkDevice command', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run-model',
      key: 'm',
      content: {
        id: 'm',
        slug: 'm',
        flavors: [{ modelName: 'dqn' }],
        createdAt: NOW,
        updatedAt: NOW,
      },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const m = manifest()
    delete m.benchmarkDevice
    await expect(
      tools.benchmarkModelDevice({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: m,
        modelId: 'm',
      }),
    ).rejects.toThrow(/benchmarkDevice/)
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

  it('MEMORY-SAFETY: caps a runaway series in the persisted run record (producer safety valve)', async () => {
    const storage = memoryStorage()
    const runner = stubRunner({
      jobResult: () => ({
        summary: { objective: 5, series: { equity: Array.from({ length: 40000 }, (_, i) => i) } },
      }),
    })
    const { tools } = makeTools(runner, storage)
    const m = manifest()
    delete m.calibrate
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: m,
      spec: { fixed: { lr: 0.1 } },
    })
    const [record] = await storage.listRecords({ scope: 'proj', type: 'demo-run' })
    expect((record.content as { series: { equity: number[] } }).series.equity.length).toBeLessThanOrEqual(
      10000,
    )
  })

  it('MEMORY-SAFETY: the campaign dedup + pick-best scans omit heavy run fields (no full-content re-materialization)', async () => {
    const storage = memoryStorage()
    const { tools } = makeTools(stubRunner(), storage)
    const m = manifest()
    delete m.calibrate
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: m,
      spec: { sweep: { lr: [0.1, 0.2] } },
      skipExplored: true,
    })
    // The dedup scan (skipExplored) and the end-of-campaign pick-best scan both list the run type; neither
    // renders a chart, so both must shed the unbounded per-step fields — else 10 concurrent campaigns each
    // re-materialize the whole growing run-set into the Node heap.
    const runScans = storage.queries.filter((q) => q.type === 'demo-run')
    expect(runScans.length).toBeGreaterThan(0)
    for (const q of runScans) {
      expect(q.omit).toEqual(HEAVY_RUN_FIELDS)
    }
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

    it('stamps the originating activityId on each evaluation record (Run→Activity link)', async () => {
      const storage = memoryStorage()
      await seedCompletedRun(storage, 'run1')
      const runner = stubRunner({ jobResult: () => ({ summary: { objective: 7 } }) })
      const { tools } = makeTools(runner, storage)
      await tools.evaluateTrainingRuns({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: evalManifest(),
        runKeys: ['run1'],
        activityId: 'act-eval-1',
      })
      const rec = await storage.readRecord({ scope: 'proj', type: 'demo-run-evaluation', key: 'run1' })
      expect(rec?.content).toMatchObject({ runKey: 'run1', activityId: 'act-eval-1' })
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

  it('MEMORY-SAFETY: reads the completed runs with heavy fields omitted (judging never needs the trace)', async () => {
    const storage = memoryStorage()
    await seedRun(storage, 'a', 50)
    const executor = stubExecutor(JSON.stringify([{ key: 'a', score: 70, why: 'ok' }]))
    const { tools } = makeJudgeTools(executor, storage)
    await tools.judgeTrainingRuns({ scope: 'proj', projectRoot: '/repo', manifest: manifest(), llmConfig: LLM })
    const runScans = storage.queries.filter((q) => q.type === 'demo-run')
    expect(runScans.length).toBeGreaterThan(0)
    for (const q of runScans) {
      expect(q.omit).toEqual(HEAVY_RUN_FIELDS)
    }
  })

  it('stamps the originating activityId on each verdict record (Run→Activity link)', async () => {
    const storage = memoryStorage()
    await seedRun(storage, 'a', 50)
    const executor = stubExecutor(JSON.stringify([{ key: 'a', score: 70, why: 'ok' }]))
    const { tools } = makeJudgeTools(executor, storage)
    await tools.judgeTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
      activityId: 'act-judge-1',
    })
    const rec = await storage.readRecord({ scope: 'proj', type: 'demo-run-verdict', key: 'a' })
    expect(rec?.content).toMatchObject({ key: 'a', activityId: 'act-judge-1' })
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

describe('proposeTrainingExperiments', () => {
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

  it('persists valid proposals as xai-suggestion records, never hypotheses', async () => {
    const storage = memoryStorage()
    await seedRun(storage, 'a', 10)
    const { tools } = makeJudgeTools(stubExecutor(PROPOSALS), storage)
    const result = await tools.proposeTrainingExperiments({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
    })
    expect(result).toMatchObject({
      recordType: 'demo-run',
      proposed: 2, // the ghost-lever proposal is dropped
      skippedExisting: 0,
      proposedBy: 'openai/m',
      proposedAt: NOW,
    })
    const suggestions = await storage.listRecords({
      scope: 'proj',
      type: 'demo-run-xai-suggestion',
    })
    expect(suggestions).toHaveLength(2)
    expect(suggestions[0].content).toMatchObject({
      source: 'llm',
      proposedBy: 'openai/m',
      proposedAt: NOW,
    })
    expect((suggestions[0].content as { id: string }).id).toBe(suggestions[0].key)
    // The whole point: it must NOT write hypothesis records.
    const hyps = await storage.listRecords({ scope: 'proj', type: 'demo-run-hypothesis' })
    expect(hyps).toHaveLength(0)
  })

  it('dedupes identical specs against existing suggestions', async () => {
    const storage = memoryStorage()
    const { tools } = makeJudgeTools(stubExecutor(PROPOSALS), storage)
    await tools.proposeTrainingExperiments({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
    })
    const second = await tools.proposeTrainingExperiments({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
    })
    expect(second.skippedExisting).toBe(2)
    expect(second.proposed).toBe(0)
  })

  it('notifies onRecordWritten per suggestion', async () => {
    const written: string[] = []
    const { tools } = makeJudgeTools(stubExecutor(PROPOSALS), memoryStorage())
    await tools.proposeTrainingExperiments({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
      onRecordWritten: (type) => written.push(type),
    })
    expect(written).toEqual(['demo-run-xai-suggestion', 'demo-run-xai-suggestion'])
  })

  it('throws without an inference executor', async () => {
    const { tools } = makeJudgeTools(undefined, memoryStorage())
    await expect(
      tools.proposeTrainingExperiments({
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

  it('surfaces per-step drivers in the narration when steps carry saliencyByGroup', async () => {
    const storage = memoryStorage()
    await seedRun(storage, 'ccc', 50, {
      config: { lr: 0.5 },
      artifacts: {
        decisionTrace: {
          steps: [
            { step: 0, action: 'buy', saliencyByGroup: { 'layer:1d': 3, 'layer:1h': 1 } },
            { step: 1, action: 'sell', saliencyByGroup: { 'layer:1d': 2, 'layer:1h': 1 } },
          ],
          totalSteps: 2,
          actionCounts: { buy: 1, sell: 1 },
        },
      },
    })
    const executor = stubExecutor('attends to 1d.')
    const { tools } = makeJudgeTools(executor, storage)
    await tools.xaiNarrate({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
      runKey: 'ccc',
    })
    expect(executor.requests[0].userContent).toMatch(/Per-step drivers.*layer:1d=2/)
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

  it('MEMORY-SAFETY: getRunXAI reads the focus trace via a by-key full read, not by bulk-loading every run trace', async () => {
    const storage = memoryStorage()
    await seedProject(storage)
    await seedRun(storage, 'lo', 10, { config: { lr: 0.1 } })
    await seedRun(storage, 'hi', 90, {
      config: { lr: 0.9 },
      artifacts: { decisionTrace: { steps: [{ step: 0, action: 'hold' }], actionCounts: { hold: 5, buy: 2 } } },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.getRunXAI({ scope: 'proj', runKey: 'hi' })
    expect(result.found).toBe(true)
    // The trace lives in the omitted `artifacts.decisionTrace`; the digest can only show it if the focus
    // record was re-fetched IN FULL — the multi-run ranking list must never carry every run's trace.
    expect(result.analysis?.actionCounts).toEqual({ hold: 5, buy: 2 })
    const listScans = storage.queries.filter((q) => q.type === 'demo-run' && q.key === undefined)
    expect(listScans.length).toBeGreaterThan(0)
    for (const q of listScans) expect(q.omit).toEqual(HEAVY_RUN_FIELDS)
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

describe('migrateTrainingRuns', () => {
  const migrations = [
    {
      match: { reward_model: 'combo_all' },
      set: { reward_model: 'combo_unified', combo_sell: 1000, combo_fee_penalty: 0 },
    },
    {
      match: { reward_model: 'combo_all_fee' },
      set: { reward_model: 'combo_unified', combo_sell: 1000 },
      keepOrDefault: { combo_fee_penalty: 1.0 },
    },
  ]
  const withMigrations = () => manifest({ migrations })

  it('rewrites matching run configs in place and recomputes setupKey', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'r1',
      content: {
        objective: 1,
        config: { reward_model: 'combo_all', lr: 0.1, seed: 3 },
        setupKey: 'old',
      },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.migrateTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: withMigrations(),
    })
    expect(result).toMatchObject({ recordType: 'demo-run', examinedRuns: 1, migratedRuns: 1 })
    const rec = await storage.readRecord({ scope: 'proj', type: 'demo-run', key: 'r1' })
    const content = rec?.content as { config: Record<string, unknown>; setupKey: string }
    expect(content.config).toMatchObject({
      reward_model: 'combo_unified',
      combo_sell: 1000,
      lr: 0.1,
    })
    expect(content.setupKey).not.toBe('old')
  })

  it('MEMORY-SAFETY: scans lean (omits heavy fields) yet preserves them via a by-key read on rewrite', async () => {
    // The boot-time migration sweep runs over EVERY run; loading each run's full series/trace at once is
    // what OOMs the backend at startup. It must scan lean and only read a full record when it rewrites one.
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'r1',
      content: {
        objective: 1,
        config: { reward_model: 'combo_all', lr: 0.1 },
        series: { equity: [1, 2, 3] },
      },
    })
    const { tools } = makeTools(stubRunner(), storage)
    await tools.migrateTrainingRuns({ scope: 'proj', projectRoot: '/repo', manifest: withMigrations() })
    const listScans = storage.queries.filter((q) => q.type === 'demo-run')
    expect(listScans.length).toBeGreaterThan(0)
    for (const q of listScans) expect(q.omit).toEqual(HEAVY_RUN_FIELDS)
    const rec = await storage.readRecord({ scope: 'proj', type: 'demo-run', key: 'r1' })
    expect((rec?.content as { series: unknown }).series).toEqual({ equity: [1, 2, 3] })
    expect((rec?.content as { config: { reward_model: string } }).config.reward_model).toBe(
      'combo_unified',
    )
  })

  it('keeps an existing keepOrDefault value (per-run penalty preserved)', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'r1',
      content: { objective: 1, config: { reward_model: 'combo_all_fee', combo_fee_penalty: 0.5 } },
    })
    const { tools } = makeTools(stubRunner(), storage)
    await tools.migrateTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: withMigrations(),
    })
    const rec = await storage.readRecord({ scope: 'proj', type: 'demo-run', key: 'r1' })
    expect((rec?.content as { config: Record<string, unknown> }).config.combo_fee_penalty).toBe(0.5)
  })

  it('backfills a MISSING field via keepOrDefault and counts it migrated', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'r1',
      content: { objective: 1, config: { reward_model: 'combo_unified', lr: 0.1 } },
    })
    const backfill = [
      { match: { reward_model: 'combo_unified' }, keepOrDefault: { allow_shorting: false } },
    ]
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.migrateTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest({ migrations: backfill }),
    })
    expect(result.migratedRuns).toBe(1)
    const rec = await storage.readRecord({ scope: 'proj', type: 'demo-run', key: 'r1' })
    expect((rec?.content as { config: Record<string, unknown> }).config.allow_shorting).toBe(false)
  })

  it('skips a NO-OP rewrite (idempotent backfill on an already-set field does not churn)', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'r1',
      content: {
        objective: 1,
        config: { reward_model: 'combo_unified', allow_shorting: false },
        setupKey: 'keep',
      },
    })
    const backfill = [
      { match: { reward_model: 'combo_unified' }, keepOrDefault: { allow_shorting: false } },
    ]
    const written: string[] = []
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.migrateTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest({ migrations: backfill }),
      onRecordWritten: (_t: string, k: string) => written.push(k),
    })
    expect(result).toMatchObject({ examinedRuns: 1, migratedRuns: 0 })
    expect(written).toEqual([]) // no rewrite, no broadcast
    const rec = await storage.readRecord({ scope: 'proj', type: 'demo-run', key: 'r1' })
    expect((rec?.content as { setupKey: string }).setupKey).toBe('keep') // untouched
  })

  it('leaves already-migrated runs untouched and is idempotent', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'r1',
      content: { objective: 1, config: { reward_model: 'combo_unified', combo_sell: 1000 } },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const first = await tools.migrateTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: withMigrations(),
    })
    expect(first).toMatchObject({ examinedRuns: 1, migratedRuns: 0 })
  })

  it('migrates pending-queue items spec.fixed when a queueRecordType is given', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'trainer-queue',
      key: 'q1',
      content: {
        id: 'q1',
        activityType: 'train',
        params: {
          spec: { fixed: { reward_model: 'combo_all', lr: 0.2 }, seeds: [0] },
          concurrency: 1,
        },
      },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.migrateTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: withMigrations(),
      queueRecordType: 'trainer-queue',
    })
    expect(result).toMatchObject({ examinedQueue: 1, migratedQueue: 1 })
    const rec = await storage.readRecord({ scope: 'proj', type: 'trainer-queue', key: 'q1' })
    const fixed = (
      rec?.content as { params: { spec: { fixed: Record<string, unknown>; seeds: number[] } } }
    ).params.spec
    expect(fixed.fixed).toMatchObject({ reward_model: 'combo_unified', combo_sell: 1000, lr: 0.2 })
    expect(fixed.seeds).toEqual([0])
  })

  it('does nothing (no record reads needed) when the manifest declares no migrations', async () => {
    const storage = memoryStorage()
    await seedRun(storage, 'r1', 50)
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.migrateTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
    })
    expect(result).toMatchObject({
      examinedRuns: 0,
      migratedRuns: 0,
      examinedQueue: 0,
      migratedQueue: 0,
    })
  })

  it('skips run records that carry no config object', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'noconf',
      content: { objective: 1 },
    })
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'r1',
      content: { objective: 1, config: { reward_model: 'combo_all' } },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.migrateTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: withMigrations(),
    })
    expect(result).toMatchObject({ examinedRuns: 2, migratedRuns: 1 })
  })

  it('skips queue items with no spec.fixed (non-train tasks) and reads from disk when no manifest is passed', async () => {
    const storage = memoryStorage()
    const root = await mkdtemp(join(tmpdir(), 'mtq-'))
    tempDirs.push(root)
    await writeFile(join(root, 'trainer.json'), JSON.stringify(manifest({ migrations })))
    await storage.upsertRecord({
      scope: 'proj',
      type: 'trainer-queue',
      key: 'judge1',
      content: { id: 'judge1', activityType: 'judge', params: { runKeys: ['x'] } },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.migrateTrainingRuns({
      scope: 'proj',
      projectRoot: root,
      manifestRelPath: 'trainer.json',
      queueRecordType: 'trainer-queue',
    })
    expect(result).toMatchObject({ examinedQueue: 1, migratedQueue: 0 })
  })

  describe('delete rules', () => {
    // Rewrite rules first, then a trailing "delete anything that isn't combo_unified" rule.
    const pruneMigrations = [
      {
        match: { reward_model: 'combo_all' },
        set: { reward_model: 'combo_unified', combo_sell: 1000 },
      },
      { matchNot: { reward_model: 'combo_unified' }, delete: true },
    ]
    const withPrune = () => manifest({ migrations: pruneMigrations })

    async function seedRunRec(storage: DataStorage, key: string, config: Record<string, unknown>) {
      await storage.upsertRecord({
        scope: 'proj',
        type: 'demo-run',
        key,
        content: { objective: 1, config },
      })
    }

    it('deletes runs whose reward_model is not combo_unified, rewrites the migratable ones, keeps combo_unified', async () => {
      const storage = memoryStorage()
      await seedRunRec(storage, 'old', { reward_model: 'combo_all', lr: 0.1 }) // → rewritten
      await seedRunRec(storage, 'pp3', { reward_model: 'profit_percentage3', lr: 0.2 }) // → deleted
      await seedRunRec(storage, 'pa2', { reward_model: 'profit_all2', lr: 0.3 }) // → deleted
      await seedRunRec(storage, 'uni', { reward_model: 'combo_unified', lr: 0.4 }) // → kept
      const { tools } = makeTools(stubRunner(), storage)
      const result = await tools.migrateTrainingRuns({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: withPrune(),
      })

      expect(result).toMatchObject({ examinedRuns: 4, migratedRuns: 1, deletedRuns: 2 })
      expect(
        await storage.readRecord({ scope: 'proj', type: 'demo-run', key: 'pp3' }),
      ).toBeUndefined()
      expect(
        await storage.readRecord({ scope: 'proj', type: 'demo-run', key: 'pa2' }),
      ).toBeUndefined()
      expect(
        await storage.readRecord({ scope: 'proj', type: 'demo-run', key: 'uni' }),
      ).toBeDefined()
      const kept = await storage.readRecord({ scope: 'proj', type: 'demo-run', key: 'old' })
      expect((kept?.content as { config: Record<string, unknown> }).config.reward_model).toBe(
        'combo_unified',
      )
    })

    it('never deletes a run that lacks the matchNot field (e.g. hodl/supervised with no reward_model)', async () => {
      const storage = memoryStorage()
      await seedRunRec(storage, 'hodl', { model_name: 'hodl', lr: 0.1 })
      const { tools } = makeTools(stubRunner(), storage)
      const result = await tools.migrateTrainingRuns({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: withPrune(),
      })
      expect(result).toMatchObject({ examinedRuns: 1, migratedRuns: 0, deletedRuns: 0 })
      expect(
        await storage.readRecord({ scope: 'proj', type: 'demo-run', key: 'hodl' }),
      ).toBeDefined()
    })

    it('deletes pending-queue items whose spec.fixed reward is being removed', async () => {
      const storage = memoryStorage()
      await storage.upsertRecord({
        scope: 'proj',
        type: 'trainer-queue',
        key: 'q1',
        content: {
          id: 'q1',
          activityType: 'train',
          params: { spec: { fixed: { reward_model: 'profit_percentage3' } } },
        },
      })
      const { tools } = makeTools(stubRunner(), storage)
      const result = await tools.migrateTrainingRuns({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: withPrune(),
        queueRecordType: 'trainer-queue',
      })
      expect(result).toMatchObject({ examinedQueue: 1, deletedQueue: 1, migratedQueue: 0 })
      expect(
        await storage.readRecord({ scope: 'proj', type: 'trainer-queue', key: 'q1' }),
      ).toBeUndefined()
    })

    it('is idempotent — a second pass deletes/rewrites nothing', async () => {
      const storage = memoryStorage()
      await seedRunRec(storage, 'pp3', { reward_model: 'profit_percentage3' })
      await seedRunRec(storage, 'uni', { reward_model: 'combo_unified' })
      const { tools } = makeTools(stubRunner(), storage)
      await tools.migrateTrainingRuns({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: withPrune(),
      })
      const second = await tools.migrateTrainingRuns({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: withPrune(),
      })
      expect(second).toMatchObject({ examinedRuns: 1, migratedRuns: 0, deletedRuns: 0 })
    })

    it('cascades a deleted run to its derived records (evaluation/verdict/xai-narrative + unrunnable by setupKey)', async () => {
      const storage = memoryStorage()
      // A deleted run + ALL its derived records, plus a kept combo_unified run with its own derived records.
      await storage.upsertRecord({
        scope: 'proj',
        type: 'demo-run',
        key: 'pp3',
        content: {
          objective: 1,
          config: { reward_model: 'profit_percentage3' },
          setupKey: 'setupPP3',
        },
      })
      for (const type of ['demo-run-evaluation', 'demo-run-verdict', 'demo-run-xai-narrative']) {
        await storage.upsertRecord({ scope: 'proj', type, key: 'pp3', content: { runKey: 'pp3' } })
      }
      await storage.upsertRecord({
        scope: 'proj',
        type: 'demo-run-unrunnable',
        key: 'setupPP3',
        content: {},
      })
      await storage.upsertRecord({
        scope: 'proj',
        type: 'demo-run',
        key: 'uni',
        content: { objective: 2, config: { reward_model: 'combo_unified' }, setupKey: 'setupUNI' },
      })
      await storage.upsertRecord({
        scope: 'proj',
        type: 'demo-run-verdict',
        key: 'uni',
        content: { runKey: 'uni' },
      })

      const { tools } = makeTools(stubRunner(), storage)
      const written: string[] = []
      const result = await tools.migrateTrainingRuns({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: withPrune(),
        onRecordWritten: (type, key) => written.push(`${type}:${key}`),
      })

      expect(result).toMatchObject({ deletedRuns: 1 })
      // The run AND every derived record are gone.
      for (const type of [
        'demo-run',
        'demo-run-evaluation',
        'demo-run-verdict',
        'demo-run-xai-narrative',
      ]) {
        expect(await storage.readRecord({ scope: 'proj', type, key: 'pp3' })).toBeUndefined()
      }
      expect(
        await storage.readRecord({ scope: 'proj', type: 'demo-run-unrunnable', key: 'setupPP3' }),
      ).toBeUndefined()
      // The KEPT run and its verdict are untouched.
      expect(
        await storage.readRecord({ scope: 'proj', type: 'demo-run', key: 'uni' }),
      ).toBeDefined()
      expect(
        await storage.readRecord({ scope: 'proj', type: 'demo-run-verdict', key: 'uni' }),
      ).toBeDefined()
      // Each real deletion was broadcast (run + 3 children + unrunnable); a missing child isn't broadcast.
      expect(written).toEqual(
        expect.arrayContaining([
          'demo-run:pp3',
          'demo-run-evaluation:pp3',
          'demo-run-verdict:pp3',
          'demo-run-xai-narrative:pp3',
          'demo-run-unrunnable:setupPP3',
        ]),
      )
    })

    it('only broadcasts derived deletions that actually existed (no spurious data:updated)', async () => {
      const storage = memoryStorage()
      // A deleted run with NO derived records and no setupKey.
      await storage.upsertRecord({
        scope: 'proj',
        type: 'demo-run',
        key: 'pp3',
        content: { objective: 1, config: { reward_model: 'profit_percentage3' } },
      })
      const { tools } = makeTools(stubRunner(), storage)
      const written: string[] = []
      await tools.migrateTrainingRuns({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: withPrune(),
        onRecordWritten: (type, key) => written.push(`${type}:${key}`),
      })
      expect(written).toEqual(['demo-run:pp3'])
    })
  })
})

describe('scanProjectModels', () => {
  const mm = (choices: string[], desc?: string): TrainerManifest =>
    manifest({
      levers: {
        lr: { type: 'number', default: 0.01 },
        model_name: {
          type: 'choice',
          choices,
          default: choices[0],
          ...(desc ? { description: desc } : {}),
        },
      },
    })

  it('heuristically discovers uncatalogued model_name choices and persists them (no LLM)', async () => {
    const storage = memoryStorage()
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.scanProjectModels({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: mm(['dqn', 'rainbow-dqn-custom', 'hodl']),
    })
    expect(result).toMatchObject({
      recordType: 'demo-run',
      discovered: 3,
      created: 3,
      skippedExisting: 0,
      scannedAt: NOW,
    })
    expect(result.scannedBy).toBeUndefined()
    const recs = await storage.listRecords({ scope: 'proj', type: 'demo-run-model' })
    expect(recs).toHaveLength(3)
    const rainbow = recs.find((r) => r.key === 'rainbow-dqn-custom')!.content as Record<
      string,
      unknown
    >
    expect(rainbow).toMatchObject({
      id: 'rainbow-dqn-custom',
      slug: 'rainbow-dqn-custom',
      name: 'Rainbow DQN Custom',
      category: 'rl',
      status: 'implemented',
      statusSource: 'auto',
      source: 'scan',
      flavors: [{ modelName: 'rainbow-dqn-custom' }],
    })
    const hodl = recs.find((r) => r.key === 'hodl')!.content as Record<string, unknown>
    expect(hodl).toMatchObject({ name: 'Buy-and-Hold', category: 'baseline' })
  })

  it('enriches each candidate with the LLM (name/description/category/paper links)', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run-paper',
      key: 'p1',
      content: {
        id: 'p1',
        title: 'Rainbow',
        claim: 'combines six DQN improvements',
        status: 'untested',
        source: 'manual',
        createdAt: NOW,
        updatedAt: NOW,
      },
    })
    const enrich = JSON.stringify([
      {
        slug: 'rainbow-dqn-custom',
        name: 'Rainbow DQN',
        description: 'Combines six DQN improvements.',
        category: 'rl',
        paperIds: ['p1', 'ghost'],
      },
    ])
    const executor = stubExecutor(enrich)
    const { tools } = makeJudgeTools(executor, storage)
    const written: string[] = []
    const result = await tools.scanProjectModels({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: mm(['rainbow-dqn-custom'], 'the algorithm'),
      llmConfig: LLM,
      onRecordWritten: (t: string) => written.push(t),
    })
    expect(result.created).toBe(1)
    expect(result.scannedBy).toBe('openai/m')
    const m = (await storage.readRecord({
      scope: 'proj',
      type: 'demo-run-model',
      key: 'rainbow-dqn-custom',
    }))!.content as Record<string, unknown>
    expect(m).toMatchObject({
      name: 'Rainbow DQN',
      description: 'Combines six DQN improvements.',
      category: 'rl',
      source: 'llm',
      paperIds: ['p1'],
      proposedBy: 'openai/m',
    })
    expect(executor.requests[0].userContent).toContain('the algorithm')
    expect(written).toContain('demo-run-model')
  })

  it('does not re-discover a choice already in the catalog (counts it as skippedExisting)', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run-model',
      key: 'dqn',
      content: {
        id: 'dqn',
        slug: 'dqn',
        name: 'DQN',
        description: '',
        category: 'rl',
        status: 'implemented',
        statusSource: 'auto',
        modelNames: ['dqn'],
        source: 'manual',
        createdAt: NOW,
        updatedAt: NOW,
      },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.scanProjectModels({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: mm(['dqn', 'ppo']),
    })
    expect(result).toMatchObject({ discovered: 1, created: 1, skippedExisting: 1 })
    expect(result.models[0].slug).toBe('ppo')
  })

  it('returns nothing when the manifest has no model_name choice lever', async () => {
    const storage = memoryStorage()
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.scanProjectModels({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
    })
    expect(result).toMatchObject({ discovered: 0, created: 0, skippedExisting: 0, models: [] })
  })
})

describe('analyzePaperModels', () => {
  const mm = manifest({
    levers: {
      lr: { type: 'number', default: 0.01 },
      model_name: { type: 'choice', choices: ['dqn', 'rainbow-dqn-custom'], default: 'dqn' },
    },
  })
  const seedPaper = (storage: DataStorage, id = 'p1', extra: Record<string, unknown> = {}) =>
    storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run-paper',
      key: id,
      content: {
        id,
        title: 'Rainbow',
        claim: 'combine improvements',
        status: 'untested',
        source: 'manual',
        createdAt: NOW,
        updatedAt: NOW,
        ...extra,
      },
    })
  const seedModel = (storage: DataStorage, slug: string, extra: Record<string, unknown> = {}) =>
    storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run-model',
      key: slug,
      content: {
        id: slug,
        slug,
        name: slug,
        description: '',
        category: 'rl',
        status: 'implemented',
        statusSource: 'auto',
        modelNames: [slug],
        source: 'manual',
        createdAt: NOW,
        updatedAt: NOW,
        ...extra,
      },
    })

  it('links matched existing models both directions and returns missing proposals', async () => {
    const storage = memoryStorage()
    await seedPaper(storage)
    await seedModel(storage, 'rainbow-dqn-custom', {
      modelNames: ['rainbow-dqn-custom', 'rainbow-dqn'],
    })
    const resp = JSON.stringify({
      matchModelIds: ['rainbow-dqn-custom', 'ghost'],
      proposedModels: [
        {
          name: 'C51 Distributional DQN',
          description: 'categorical',
          category: 'rl',
          proposal: 'add C51 head',
        },
        { name: 'Rainbow DQN', slug: 'rainbow-dqn', description: '', category: 'rl', proposal: '' },
      ],
      proposedImprovements: [
        { title: 'C51 Distributional DQN', detail: 'add C51 head', kind: 'model' },
        {
          title: 'Order-book depth feature',
          detail: 'ingest L2 to test the microstructure claim',
          kind: 'data',
        },
      ],
    })
    const { tools } = makeJudgeTools(stubExecutor(resp), storage)
    const written: string[] = []
    const result = await tools.analyzePaperModels({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: mm,
      paperId: 'p1',
      llmConfig: LLM,
      onRecordWritten: (t: string, k: string) => written.push(`${t}:${k}`),
    })
    expect(result.linkedModelIds).toEqual(['rainbow-dqn-custom'])
    expect(result.missingModels.map((p) => p.slug)).toEqual(['c51-distributional-dqn'])
    expect(result.paper.modelIds).toEqual(['rainbow-dqn-custom'])
    // The general improvements ride back on the result AND persist on the paper for the Coverage-gaps panel.
    expect(result.proposedImprovements.map((p) => p.kind)).toEqual(['model', 'data'])
    expect(result.paper.proposedImprovements?.map((p) => p.title)).toEqual([
      'C51 Distributional DQN',
      'Order-book depth feature',
    ])
    const m = (await storage.readRecord({
      scope: 'proj',
      type: 'demo-run-model',
      key: 'rainbow-dqn-custom',
    }))!.content as { paperIds?: string[] }
    expect(m.paperIds).toContain('p1')
    expect(written).toContain('demo-run-model:rainbow-dqn-custom')
    expect(written).toContain('demo-run-paper:p1')
  })

  it('throws when the paper is not in the project', async () => {
    const { tools } = makeJudgeTools(stubExecutor('{}'), memoryStorage())
    await expect(
      tools.analyzePaperModels({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: mm,
        paperId: 'nope',
        llmConfig: LLM,
      }),
    ).rejects.toThrow(/no paper/)
  })

  it('stamps the paper (preserving prior modelIds) when nothing matches', async () => {
    const storage = memoryStorage()
    await seedPaper(storage, 'p1', { modelIds: ['existing-link'] })
    const { tools } = makeJudgeTools(
      stubExecutor(JSON.stringify({ matchModelIds: [], proposedModels: [] })),
      storage,
    )
    const result = await tools.analyzePaperModels({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: mm,
      paperId: 'p1',
      llmConfig: LLM,
    })
    expect(result.linkedModelIds).toEqual([])
    expect(result.missingModels).toEqual([])
    expect(result.paper.modelIds).toEqual(['existing-link'])
    expect(result.paper.updatedAt).toBe(NOW)
  })

  it('hands the model the fetched paper text when the paper has a url', async () => {
    const storage = memoryStorage()
    await seedPaper(storage, 'p1', { url: 'https://arxiv.org/abs/1' })
    const executor = stubExecutor(JSON.stringify({ matchModelIds: [], proposedModels: [] }))
    const { tools } = makeJudgeTools(executor, storage)
    await tools.analyzePaperModels({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: mm,
      paperId: 'p1',
      llmConfig: LLM,
      fetchPaperText: async () => 'ABSTRACT TEXT',
    })
    expect(executor.requests[0].userContent).toContain('ABSTRACT TEXT')
  })

  it('still analyses when the paper-text fetch fails (text omitted)', async () => {
    const storage = memoryStorage()
    await seedPaper(storage, 'p1', { url: 'https://arxiv.org/abs/1' })
    const executor = stubExecutor(JSON.stringify({ matchModelIds: [], proposedModels: [] }))
    const { tools } = makeJudgeTools(executor, storage)
    const result = await tools.analyzePaperModels({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: mm,
      paperId: 'p1',
      llmConfig: LLM,
      fetchPaperText: async () => {
        throw new Error('network down')
      },
    })
    expect(result.paper.modelIds).toEqual([])
    expect(executor.requests[0].userContent).not.toContain('ABSTRACT')
  })
})

describe('analyzeConfigSpace', () => {
  it('builds the whole-space bundle over every completed run, folding seeds into setups', async () => {
    const storage = memoryStorage()
    let k = 0
    for (const lr of [0.1, 0.5])
      for (const bs of [32, 64])
        for (const seed of [0, 1])
          await seedRun(storage, `r${k++}`, lr * 100 + bs, { config: { lr, batch_size: bs }, seed })
    const { tools } = makeJudgeTools(undefined, storage)
    const result = await tools.analyzeConfigSpace({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
    })
    expect(result.recordType).toBe('demo-run')
    expect(result.criterion.key).toBe('objective')
    expect(result.analysis).not.toBeNull()
    expect(result.analysis!.runCount).toBe(8)
    expect(result.analysis!.setupCount).toBe(4) // 2 lr × 2 batch_size, the 2 seeds folded
    expect(result.analysis!.surrogate.trees.length).toBeGreaterThan(0)
  })

  it('returns a null analysis when there are no completed runs', async () => {
    const { tools } = makeJudgeTools(undefined, memoryStorage())
    const result = await tools.analyzeConfigSpace({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
    })
    expect(result.analysis).toBeNull()
  })
})

describe('invalidateRuns', () => {
  // Generic engine logic — exercised with simple injected predicates (the fidelity detector is tested in
  // modelTrainerUtils.test.ts). affected = config.affected === true / spec.fixed.affected === true.
  const affectsRun = (c: Record<string, unknown>) => c.affected === true
  const affectsPending = (s: Record<string, unknown>) =>
    (s.fixed as Record<string, unknown> | undefined)?.affected === true
  const baseParams = {
    scope: 'proj',
    projectRoot: '/repo',
    invalidationId: 'bug-v5',
    reason: 'desync',
    beforePipelineMajor: 5,
    affectsRun,
  }

  it("marks an affected, pre-fix run status='invalid' (preserving priorStatus + reason)", async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'r1',
      content: {
        objective: 1,
        status: 'completed',
        pipelineVersion: '4.0',
        config: { affected: true },
      },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.invalidateRuns({ ...baseParams, manifest: manifest() })
    expect(result).toMatchObject({ examinedRuns: 1, invalidatedRuns: 1 })
    const rec = await storage.readRecord({ scope: 'proj', type: 'demo-run', key: 'r1' })
    const c = rec?.content as Record<string, unknown>
    expect(c.status).toBe('invalid')
    expect(c.priorStatus).toBe('completed')
    expect(c.invalidReason).toBe('desync')
    expect(c.invalidatedBy).toBe('bug-v5')
  })

  it('MEMORY-SAFETY: scans lean (omits heavy fields) yet preserves them when stamping invalid', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'r1',
      content: {
        objective: 1,
        status: 'completed',
        pipelineVersion: '4.0',
        config: { affected: true },
        series: { equity: [1, 2, 3] },
      },
    })
    const { tools } = makeTools(stubRunner(), storage)
    await tools.invalidateRuns({ ...baseParams, manifest: manifest() })
    const listScans = storage.queries.filter((q) => q.type === 'demo-run')
    expect(listScans.length).toBeGreaterThan(0)
    for (const q of listScans) expect(q.omit).toEqual(HEAVY_RUN_FIELDS)
    const rec = await storage.readRecord({ scope: 'proj', type: 'demo-run', key: 'r1' })
    expect((rec?.content as { series: unknown }).series).toEqual({ equity: [1, 2, 3] })
    expect((rec?.content as { status: string }).status).toBe('invalid')
  })

  it('does NOT mark an unaffected (single-path) run', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'r1',
      content: { status: 'completed', pipelineVersion: '4.0', config: { affected: false } },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.invalidateRuns({ ...baseParams, manifest: manifest() })
    expect(result.invalidatedRuns).toBe(0)
    const rec = await storage.readRecord({ scope: 'proj', type: 'demo-run', key: 'r1' })
    expect((rec?.content as Record<string, unknown>).status).toBe('completed')
  })

  it('does NOT mark a run at/after the fix major (a re-run with the fix is never re-flagged)', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'r1',
      content: { status: 'completed', pipelineVersion: '5.0', config: { affected: true } },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.invalidateRuns({ ...baseParams, manifest: manifest() })
    expect(result.invalidatedRuns).toBe(0)
  })

  it('is idempotent — an already-invalid run is skipped (no churn)', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'r1',
      content: { status: 'invalid', pipelineVersion: '4.0', config: { affected: true } },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.invalidateRuns({ ...baseParams, manifest: manifest() })
    expect(result.invalidatedRuns).toBe(0)
  })

  it('cancels affected pending-queue items ONCE and writes a marker; leaves unaffected ones', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'trainer-queue',
      key: 'q-bad',
      content: { params: { spec: { fixed: { affected: true } } } },
    })
    await storage.upsertRecord({
      scope: 'proj',
      type: 'trainer-queue',
      key: 'q-good',
      content: { params: { spec: { fixed: { affected: false } } } },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.invalidateRuns({
      ...baseParams,
      manifest: manifest(),
      affectsPending,
      queueRecordType: 'trainer-queue',
      cancelPendingQueue: true,
    })
    expect(result).toMatchObject({
      examinedQueue: 2,
      cancelledQueue: 1,
      pendingAlreadyApplied: false,
    })
    expect(
      await storage.readRecord({ scope: 'proj', type: 'trainer-queue', key: 'q-bad' }),
    ).toBeUndefined()
    expect(
      await storage.readRecord({ scope: 'proj', type: 'trainer-queue', key: 'q-good' }),
    ).toBeDefined()
    const marker = await storage.readRecord({
      scope: 'proj',
      type: 'demo-run-invalidation',
      key: 'bug-v5',
    })
    expect(marker).toBeDefined()
  })

  it('skips pending cancellation on a second run (marker present) so re-queued runs survive', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run-invalidation',
      key: 'bug-v5',
      content: { invalidationId: 'bug-v5', appliedAt: 'earlier' },
    })
    await storage.upsertRecord({
      scope: 'proj',
      type: 'trainer-queue',
      key: 'q-rerun',
      content: { params: { spec: { fixed: { affected: true } } } },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.invalidateRuns({
      ...baseParams,
      manifest: manifest(),
      affectsPending,
      queueRecordType: 'trainer-queue',
      cancelPendingQueue: true,
    })
    expect(result).toMatchObject({ cancelledQueue: 0, pendingAlreadyApplied: true })
    expect(
      await storage.readRecord({ scope: 'proj', type: 'trainer-queue', key: 'q-rerun' }),
    ).not.toBeNull()
  })
})

describe('consolidateModels', () => {
  async function seedModel(storage: DataStorage, id: string, extra: Record<string, unknown> = {}) {
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run-model',
      key: id,
      content: {
        id,
        slug: id,
        name: id,
        description: '',
        category: 'rl',
        status: 'implemented',
        statusSource: 'auto',
        flavors: [{ modelName: id }],
        source: 'paper',
        createdAt: NOW,
        updatedAt: NOW,
        ...extra,
      },
    })
  }

  it('proposes LLM-found groups, validated against the catalog ids', async () => {
    const storage = memoryStorage()
    await seedModel(storage, 'itransformer-ppo')
    await seedModel(storage, 'inverted-transformer-ppo')
    await seedModel(storage, 'dqn')
    const executor = stubExecutor(
      JSON.stringify({
        groups: [
          {
            canonicalId: 'itransformer-ppo',
            duplicateIds: ['inverted-transformer-ppo'],
            reason: 'same',
          },
          { canonicalId: 'ghost', duplicateIds: ['dqn'], reason: 'bogus' }, // unknown canonical -> dropped
        ],
      }),
    )
    const { tools } = makeJudgeTools(executor, storage)
    const result = await tools.consolidateModels({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
    })
    expect(result.groups).toEqual([
      {
        canonicalId: 'itransformer-ppo',
        duplicateIds: ['inverted-transformer-ppo'],
        reason: 'same',
      },
    ])
    expect(result.modelCount).toBe(3)
    expect(result.recordType).toBe('demo-run')
    expect(result.proposedAt).toBe(NOW)
  })

  it('excludes dismissed models from the candidate set', async () => {
    const storage = memoryStorage()
    await seedModel(storage, 'a')
    await seedModel(storage, 'b')
    await seedModel(storage, 'old', { dismissed: true })
    const executor = stubExecutor(JSON.stringify({ groups: [] }))
    const { tools } = makeJudgeTools(executor, storage)
    const result = await tools.consolidateModels({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
    })
    expect(result.modelCount).toBe(2)
    const sent = JSON.parse(executor.requests[0].userContent)
    expect(sent.models.map((m: { id: string }) => m.id).sort()).toEqual(['a', 'b'])
  })

  it('sends each model aliases to the LLM so a re-run is aware of prior merges', async () => {
    const storage = memoryStorage()
    await seedModel(storage, 'a2c', { aliases: ['policy-gradient'] })
    await seedModel(storage, 'ppo')
    const executor = stubExecutor(JSON.stringify({ groups: [] }))
    const { tools } = makeJudgeTools(executor, storage)
    await tools.consolidateModels({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
    })
    const sent = JSON.parse(executor.requests[0].userContent)
    const a2c = sent.models.find((m: { id: string }) => m.id === 'a2c')
    expect(a2c.aliases).toEqual(['policy-gradient'])
  })

  it('does NOT mutate or merge any model record (propose-only)', async () => {
    const storage = memoryStorage()
    await seedModel(storage, 'a')
    await seedModel(storage, 'b')
    const executor = stubExecutor(
      JSON.stringify({ groups: [{ canonicalId: 'a', duplicateIds: ['b'], reason: 'same' }] }),
    )
    const { tools } = makeJudgeTools(executor, storage)
    await tools.consolidateModels({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
    })
    const a = await storage.readRecord({ scope: 'proj', type: 'demo-run-model', key: 'a' })
    const b = await storage.readRecord({ scope: 'proj', type: 'demo-run-model', key: 'b' })
    expect((a!.content as { flavors: unknown[] }).flavors).toEqual([{ modelName: 'a' }])
    expect(b).not.toBeNull()
    expect((b!.content as { dismissed?: boolean }).dismissed).toBeUndefined()
  })

  it('returns empty groups without calling the LLM when there are fewer than 2 models', async () => {
    const storage = memoryStorage()
    await seedModel(storage, 'only')
    const executor = stubExecutor(JSON.stringify({ groups: [] }))
    const { tools } = makeJudgeTools(executor, storage)
    const result = await tools.consolidateModels({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: manifest(),
      llmConfig: LLM,
    })
    expect(result.groups).toEqual([])
    expect(result.modelCount).toBe(1)
    expect(executor.requests.length).toBe(0)
  })

  it('throws without an inference executor', async () => {
    const storage = memoryStorage()
    await seedModel(storage, 'a')
    const { tools } = makeJudgeTools(undefined, storage)
    await expect(
      tools.consolidateModels({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: manifest(),
        llmConfig: LLM,
      }),
    ).rejects.toThrow()
  })
})

describe('conditional-lever normalization (n/a hygiene)', () => {
  // forward_horizon applies only to the supervised model; on any other model it must read 'n/a'.
  const condManifest = (overrides: Partial<TrainerManifest> = {}) =>
    manifest({
      levers: {
        model_name: { type: 'choice', choices: ['rl', 'sup'], default: 'rl' },
        lr: { type: 'number', default: 0.01 },
        forward_horizon: { type: 'number', default: 1, appliesWhen: { model_name: ['sup'] } },
      },
      ...overrides,
    })
  // A runner that echoes the executed config back in its summary, like a real trainer.
  const echoRunner = () =>
    stubRunner({ jobResult: (job) => ({ summary: { objective: 1, config: job.config } }) })

  it('pins an inapplicable conditional lever to n/a in a COMPLETED run record (executor still gets the raw value)', async () => {
    const storage = memoryStorage()
    const runner = echoRunner()
    const { tools } = makeTools(runner, storage)
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: condManifest(),
      spec: { fixed: { model_name: 'rl', forward_horizon: 5, lr: 0.1 } },
    })
    const [rec] = await storage.listRecords({ scope: 'proj', type: 'demo-run' })
    expect((rec.content as { config: Record<string, unknown> }).config.forward_horizon).toBe('n/a')
    // the executor received the raw value (it only ignores it) — normalization is storage-only
    expect((runner.jobs[0].config as { forward_horizon: unknown }).forward_horizon).toBe(5)
  })

  it('keeps an APPLICABLE conditional lever as its real value', async () => {
    const storage = memoryStorage()
    const { tools } = makeTools(echoRunner(), storage)
    await tools.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: condManifest(),
      spec: { fixed: { model_name: 'sup', forward_horizon: 5, lr: 0.1 } },
    })
    const [rec] = await storage.listRecords({ scope: 'proj', type: 'demo-run' })
    expect((rec.content as { config: Record<string, unknown> }).config.forward_horizon).toBe(5)
  })

  it('pins n/a in a FAILED run record too', async () => {
    const storage = memoryStorage()
    const runner = stubRunner({ jobResult: () => ({ status: 'failed', error: 'boom' }) })
    const { tools } = makeTools(runner, storage)
    await tools
      .runTrainingCampaign({
        scope: 'proj',
        projectRoot: '/repo',
        manifest: condManifest(),
        spec: { fixed: { model_name: 'rl', forward_horizon: 7, lr: 0.1 } },
      })
      .catch(() => {})
    const [rec] = await storage.listRecords({ scope: 'proj', type: 'demo-run' })
    expect((rec.content as { status: string; config: Record<string, unknown> }).status).toBe(
      'failed',
    )
    expect((rec.content as { config: Record<string, unknown> }).config.forward_horizon).toBe('n/a')
  })

  it('migrates EXISTING completed runs to n/a even with no rule-based migrations, and is idempotent', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'r1',
      content: {
        objective: 1,
        status: 'completed',
        config: { model_name: 'rl', forward_horizon: 5, lr: 0.1 },
      },
    })
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'r2',
      content: {
        objective: 1,
        status: 'completed',
        config: { model_name: 'sup', forward_horizon: 5, lr: 0.1 },
      },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.migrateTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: condManifest(),
    })
    expect(result).toMatchObject({ examinedRuns: 2, migratedRuns: 1 })
    expect(
      (
        (await storage.readRecord({ scope: 'proj', type: 'demo-run', key: 'r1' }))!.content as {
          config: Record<string, unknown>
        }
      ).config.forward_horizon,
    ).toBe('n/a')
    // the supervised run is untouched (the lever applies there)
    expect(
      (
        (await storage.readRecord({ scope: 'proj', type: 'demo-run', key: 'r2' }))!.content as {
          config: Record<string, unknown>
        }
      ).config.forward_horizon,
    ).toBe(5)
    // second pass converges — nothing left to migrate
    const again = await tools.migrateTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: condManifest(),
    })
    expect(again.migratedRuns).toBe(0)
  })

  it('does NOT pin a pending queue spec to n/a (the spec drives execution; the trainer must get a real value)', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'trainer-queue',
      key: 'q1',
      content: {
        id: 'q1',
        activityType: 'train',
        params: { spec: { fixed: { model_name: 'rl', forward_horizon: 9, lr: 0.1 }, seeds: [0] } },
      },
    })
    const { tools } = makeTools(stubRunner(), storage)
    const result = await tools.migrateTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: condManifest(), // conditional levers but NO rule-based migrations
      queueRecordType: 'trainer-queue',
    })
    expect(result).toMatchObject({ examinedQueue: 1, migratedQueue: 0 })
    const rec = await storage.readRecord({ scope: 'proj', type: 'trainer-queue', key: 'q1' })
    const fixed = (rec?.content as { params: { spec: { fixed: Record<string, unknown> } } }).params
      .spec.fixed
    expect(fixed.forward_horizon).toBe(9) // untouched — a pending run is canonicalised in STORAGE on completion
  })

  it('migration keeps setupKey RAW so a canonicalised completed run still dedups against a fresh campaign', async () => {
    const storage = memoryStorage()
    const rawConfig = { model_name: 'rl', forward_horizon: 5, lr: 0.1, seed: 0 }
    const rawKey = setupKeyOf(rawConfig)
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run',
      key: 'r1',
      content: { objective: 1, status: 'completed', config: rawConfig, setupKey: rawKey },
    })
    const { tools } = makeTools(stubRunner(), storage)
    await tools.migrateTrainingRuns({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: condManifest(),
    })
    const rec = await storage.readRecord({ scope: 'proj', type: 'demo-run', key: 'r1' })
    const content = rec?.content as { config: Record<string, unknown>; setupKey: string }
    expect(content.config.forward_horizon).toBe('n/a') // stored config canonicalised
    expect(content.setupKey).toBe(rawKey) // …but setupKey stays RAW, matching setupKeyOf(item.config) in isFresh
    // A fresh skipExplored campaign for the SAME setup must therefore dedup it (no re-run).
    const runner = stubRunner({
      jobResult: (job) => ({ summary: { objective: 1, config: job.config } }),
    })
    const { tools: tools2 } = makeTools(runner, storage)
    await tools2.runTrainingCampaign({
      scope: 'proj',
      projectRoot: '/repo',
      manifest: condManifest(),
      spec: { fixed: { model_name: 'rl', forward_horizon: 5, lr: 0.1 }, seeds: [0] },
      skipExplored: true,
    })
    expect(runner.jobs).toHaveLength(0) // already explored — not re-run
  })
})

// ── researchTrainingPapers ────────────────────────────────────────────────
interface StubDeepResearch extends DeepResearchTools {
  discoverCalls: unknown[]
  verifyCalls: unknown[]
}

function stubDeepResearch(opts: {
  discovered?: DiscoveredSource[]
  verdict?: (claim: string) => Partial<ClaimVerdict>
}): StubDeepResearch {
  const discoverCalls: unknown[] = []
  const verifyCalls: unknown[] = []
  const okVerdict: ClaimVerdict = {
    status: 'confirmed',
    confidence: 0.9,
    evidence: [],
    methodology: 'stub',
    sourcesAttempted: [],
  }
  return {
    discoverCalls,
    verifyCalls,
    async planResearch(p) {
      return { goal: (p as { goal: string }).goal, subQuestions: [] }
    },
    async discoverSources(p) {
      discoverCalls.push(p)
      return opts.discovered ?? []
    },
    async gather() {
      return []
    },
    async extract() {
      return { items: [], sources: [] }
    },
    async verifyClaim(p) {
      verifyCalls.push(p)
      const claim = (p as { claim: string }).claim
      return { ...okVerdict, ...(opts.verdict ? opts.verdict(claim) : {}) }
    },
  }
}

function makeResearchTools(
  executor: InferenceExecutor | undefined,
  storage: DataStorage,
  deepResearch: DeepResearchTools | undefined,
) {
  const logger = { info: vi.fn(), warn: vi.fn() }
  return {
    tools: createModelTrainerTools({
      computeRunner: stubRunner(),
      storage,
      inferenceExecutor: executor,
      deepResearch,
      logger,
      now: () => NOW,
    }),
    logger,
  }
}

describe('researchTrainingPapers', () => {
  const PAPER_DRAFT = JSON.stringify({
    title: 'A Discovered Paper',
    claim: 'method X beats baseline',
    approach: 'X over features',
    hypotheses: [{ title: 'X helps', rationale: 'because X', spec: { fixed: { lr: 0.5 } } }],
  })
  const src = (title: string, url: string): DiscoveredSource => ({ name: title, url })
  const base = (overrides = {}) => ({
    scope: 'proj',
    projectRoot: '/repo',
    manifest: manifest(),
    model: { kind: 'api' as const, llmConfig: LLM },
    fetchPaperText: async () => 'the real fetched paper page text',
    ...overrides,
  })

  it('runs discovery, verify, and synthesis on the SELECTED model (e.g. a CLI agent)', async () => {
    const storage = memoryStorage()
    const executor = stubExecutor(PAPER_DRAFT)
    const dr = stubDeepResearch({ discovered: [src('Paper One', 'https://arxiv.org/abs/1')] })
    const { tools } = makeResearchTools(executor, storage, dr)
    const cliModel = { kind: 'cli' as const, cli: 'claude' as const }
    const result = await tools.researchTrainingPapers(base({ model: cliModel }))
    expect(result.papers).toHaveLength(1)
    // the CLI selection is threaded into every deep-research call AND the synthesis inference
    expect((dr.discoverCalls[0] as { model?: unknown }).model).toEqual(cliModel)
    expect((dr.verifyCalls[0] as { model?: unknown }).model).toEqual(cliModel)
    expect(executor.requests[0].model).toEqual(cliModel)
  })

  it('discovers, verifies, and drafts a research paper per admitted candidate', async () => {
    const storage = memoryStorage()
    const executor = stubExecutor(PAPER_DRAFT)
    const dr = stubDeepResearch({
      discovered: [
        src('Paper One', 'https://arxiv.org/abs/1'),
        src('Paper Two', 'https://arxiv.org/abs/2'),
      ],
    })
    const { tools } = makeResearchTools(executor, storage, dr)
    const written: string[] = []
    const result = await tools.researchTrainingPapers(
      base({ onRecordWritten: (t: string) => written.push(t) }),
    )
    expect(result).toMatchObject({
      recordType: 'demo-run',
      discovered: 2,
      skippedDuplicate: 0,
      rejected: 0,
      failed: 0,
      researchedBy: 'openai/m',
      researchedAt: NOW,
    })
    expect(result.papers).toHaveLength(2)
    for (const p of result.papers) {
      expect(p).toMatchObject({ status: 'untested', source: 'research', createdAt: NOW })
      // the admitting verify verdict is stamped on the draft as provenance
      expect(p.researchVerdict).toMatchObject({ status: 'confirmed', confidence: 0.9 })
    }
    // every deep-research call carried a model (the required ModelSelection)
    expect((dr.discoverCalls[0] as { model?: unknown }).model).toBeTruthy()
    expect((dr.verifyCalls[0] as { model?: unknown }).model).toBeTruthy()
    // synthesis was grounded in the fetched page text
    expect(executor.requests[0].userContent).toContain('the real fetched paper page text')
    const papers = await storage.listRecords({ scope: 'proj', type: 'demo-run-paper' })
    expect(papers).toHaveLength(2)
    expect(written).toContain('demo-run-paper')
  })

  it('throws when no deepResearch seam is injected', async () => {
    const { tools } = makeResearchTools(stubExecutor(PAPER_DRAFT), memoryStorage(), undefined)
    await expect(tools.researchTrainingPapers(base())).rejects.toThrow(/deepResearch/i)
  })

  it('returns an empty result and never calls the model when discovery finds nothing', async () => {
    const executor = stubExecutor(PAPER_DRAFT)
    const dr = stubDeepResearch({ discovered: [] })
    const { tools } = makeResearchTools(executor, memoryStorage(), dr)
    const result = await tools.researchTrainingPapers(base())
    expect(result.discovered).toBe(0)
    expect(result.papers).toEqual([])
    expect(executor.requests).toHaveLength(0)
  })

  it('rejects (does not draft) a candidate the verify gate does not admit', async () => {
    const executor = stubExecutor(PAPER_DRAFT)
    const dr = stubDeepResearch({
      discovered: [src('Bogus', 'https://x/1')],
      verdict: () => ({ status: 'unverifiable', confidence: 0.99 }),
    })
    const { tools } = makeResearchTools(executor, memoryStorage(), dr)
    const result = await tools.researchTrainingPapers(base())
    expect(result).toMatchObject({ discovered: 1, rejected: 1 })
    expect(result.papers).toEqual([])
    expect(executor.requests).toHaveLength(0) // no synthesis for a rejected candidate
  })

  it('skips a candidate already present in the papers registry (dedup)', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'demo-run-paper',
      key: 'existing',
      content: {
        id: 'existing',
        title: 'Old',
        url: 'https://arxiv.org/abs/1',
        status: 'untested',
        source: 'manual',
      },
    })
    const dr = stubDeepResearch({
      discovered: [
        src('Dup', 'https://arxiv.org/pdf/1v2'),
        src('Fresh', 'https://arxiv.org/abs/2'),
      ],
    })
    const { tools } = makeResearchTools(stubExecutor(PAPER_DRAFT), storage, dr)
    const result = await tools.researchTrainingPapers(base())
    expect(result).toMatchObject({ discovered: 2, skippedDuplicate: 1 })
    expect(result.papers).toHaveLength(1)
    expect(result.papers[0].url).toBe('https://arxiv.org/abs/2')
  })

  it('counts a candidate whose synthesis yields no usable draft as failed, without aborting the run', async () => {
    const executor = stubExecutor((req: InferenceRequest) =>
      req.userContent.includes('page-A') ? 'not json at all' : PAPER_DRAFT,
    )
    const dr = stubDeepResearch({
      discovered: [src('A', 'https://x/a'), src('B', 'https://x/b')],
    })
    const { tools } = makeResearchTools(executor, memoryStorage(), dr)
    const result = await tools.researchTrainingPapers(
      base({ fetchPaperText: async (u: string) => (u.endsWith('/a') ? 'page-A' : 'page-B') }),
    )
    expect(result).toMatchObject({ discovered: 2, failed: 1 })
    expect(result.papers).toHaveLength(1)
  })

  it('counts a candidate whose page cannot be fetched as failed, without aborting the run', async () => {
    const dr = stubDeepResearch({
      discovered: [src('A', 'https://x/a'), src('B', 'https://x/b')],
    })
    const { tools } = makeResearchTools(stubExecutor(PAPER_DRAFT), memoryStorage(), dr)
    const result = await tools.researchTrainingPapers(
      base({
        fetchPaperText: async (u: string) => {
          if (u.endsWith('/a')) throw new Error('404')
          return 'good page'
        },
      }),
    )
    expect(result).toMatchObject({ discovered: 2, failed: 1 })
    expect(result.papers).toHaveLength(1)
  })

  it('re-throws when aborted mid-run', async () => {
    const controller = new AbortController()
    const dr = stubDeepResearch({
      discovered: [src('A', 'https://x/a'), src('B', 'https://x/b')],
    })
    const { tools } = makeResearchTools(stubExecutor(PAPER_DRAFT), memoryStorage(), dr)
    await expect(
      tools.researchTrainingPapers(
        base({
          abortSignal: controller.signal,
          fetchPaperText: async () => {
            controller.abort()
            return 'page'
          },
        }),
      ),
    ).rejects.toThrow()
  })

  it('captures up to 3 cited quotes from the verify verdict onto the draft', async () => {
    const dr = stubDeepResearch({
      discovered: [src('Paper', 'https://arxiv.org/abs/1')],
      verdict: () => ({
        status: 'confirmed',
        confidence: 0.8,
        evidence: [
          { quote: 'q1', url: 'u1' },
          { quote: 'q2', url: 'u2' },
          { quote: 'q3', url: 'u3' },
          { quote: 'q4', url: 'u4' },
        ],
      }),
    })
    const { tools } = makeResearchTools(stubExecutor(PAPER_DRAFT), memoryStorage(), dr)
    const result = await tools.researchTrainingPapers(base())
    expect(result.papers[0].researchVerdict?.quotes).toEqual([
      { quote: 'q1', url: 'u1' },
      { quote: 'q2', url: 'u2' },
      { quote: 'q3', url: 'u3' },
    ])
  })

  it('verifies paper-venue hosts before generic ones (ranking)', async () => {
    const dr = stubDeepResearch({
      discovered: [src('Blog', 'https://medium.com/p'), src('Paper', 'https://arxiv.org/abs/1')],
    })
    const { tools } = makeResearchTools(stubExecutor(PAPER_DRAFT), memoryStorage(), dr)
    const result = await tools.researchTrainingPapers(base())
    // the arxiv candidate is verified + drafted first despite being discovered second
    expect(result.papers[0].url).toBe('https://arxiv.org/abs/1')
    expect((dr.verifyCalls[0] as { claim: string }).claim).toContain('https://arxiv.org/abs/1')
  })

  it('stops once the target count is drafted — the low-affinity tail is never verified', async () => {
    const dr = stubDeepResearch({
      discovered: [
        src('A', 'https://x/a'),
        src('B', 'https://x/b'),
        src('C', 'https://x/c'),
        src('D', 'https://x/d'),
        src('E', 'https://x/e'),
      ],
    })
    const { tools } = makeResearchTools(stubExecutor(PAPER_DRAFT), memoryStorage(), dr)
    const result = await tools.researchTrainingPapers(base({ count: 2 }))
    expect(result.papers).toHaveLength(2)
    expect(result.discovered).toBe(5)
    // only the first two candidates were fetched + verified; the rest were never touched
    expect(dr.verifyCalls).toHaveLength(2)
  })
})

describe('runExplorationCampaign (autopilot)', () => {
  // The synthetic 2-basin surface driven through the REAL campaign machinery (stub runner + memory storage):
  // A = global max (500 @ lr=0.5), B = local max (470 @ lr=0.3), C = baseline (20). noise_knob is inert.
  const SURFACE_MANIFEST = manifest({
    name: 'synthetic',
    recordType: 'synthetic-run',
    calibrate: undefined,
    eta: undefined,
    objective: { name: 'score', direction: 'max' },
    levers: {
      algo: { type: 'choice', choices: ['A', 'B', 'C'], default: 'A' },
      lr: { type: 'number', range: [0, 1], default: 0.1 },
      noise_knob: { type: 'number', range: [0, 1], default: 0.5 },
      seed: { type: 'number', default: 0 },
    },
  })
  function surface(c: Record<string, unknown>): number {
    const algo = String(c.algo)
    const lr = Number(c.lr ?? 0.1)
    const seed = Number(c.seed ?? 0)
    const jitter = (((seed * 37) % 7) - 3) * 0.4
    const base = algo === 'A' ? 500 - 1600 * (lr - 0.5) ** 2 : algo === 'B' ? 470 - 1600 * (lr - 0.3) ** 2 : 20
    return base + jitter
  }
  function surfaceRunner() {
    return stubRunner({
      jobResult: (job) => {
        const c = job.config as Record<string, unknown>
        const score = surface(c)
        return { summary: { objective: score, config: c, seed: Number(c.seed ?? 0), metrics: { score, baseline: 20 } } }
      },
    })
  }

  it('explores to convergence, enumerates both maxima, declares the global, and persists the map', async () => {
    const storage = memoryStorage()
    const { tools } = makeTools(surfaceRunner(), storage)
    const result = await tools.runExplorationCampaign({
      scope: 's',
      projectRoot: '/repo',
      manifest: SURFACE_MANIFEST,
      activityId: 'exp-1',
      budget: { maxRuns: 1200, maxConcurrent: 8 },
      targetObjective: 500,
      maxRounds: 300,
    })

    expect(result.state.done).toBe(true)
    expect(result.state.stage).toBe('converged')
    expect(result.basins.map((b) => String(b.region.algo)).sort()).toEqual(['A', 'B'])
    const declared = result.basins.find((b) => b.id === result.declaredBasinId)
    expect(declared?.region.algo).toBe('A')
    expect(declared?.peakObjective).toBeGreaterThan(485)

    // the exploration map is persisted for the viewer / pause-steer round-trip
    const rec = await storage.readRecord({ scope: 's', type: 'synthetic-run-exploration', key: 'exp-1' })
    expect((rec?.content as { stage?: string }).stage).toBe('converged')
    expect((rec?.content as { activityId?: string }).activityId).toBe('exp-1')

    // real training runs were persisted under the project's record type
    const runs = await storage.listRecords({ scope: 's', type: 'synthetic-run' })
    expect(runs.length).toBeGreaterThan(10)
  })

  it('halts immediately when the persisted state is paused (launches nothing)', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 's',
      type: 'synthetic-run-exploration',
      key: 'exp-2',
      content: { ...initExplorationState(SURFACE_MANIFEST), stage: 'global', paused: true },
    })
    const runner = surfaceRunner()
    const { tools } = makeTools(runner, storage)
    const result = await tools.runExplorationCampaign({
      scope: 's',
      projectRoot: '/repo',
      manifest: SURFACE_MANIFEST,
      activityId: 'exp-2',
      maxRounds: 50,
    })
    expect(result.state.paused).toBe(true)
    expect(runner.jobs).toHaveLength(0) // no training launched while paused
  })
})
