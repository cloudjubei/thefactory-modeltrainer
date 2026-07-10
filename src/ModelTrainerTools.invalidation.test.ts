import { describe, expect, it, vi } from 'vitest'
import type {
  ComputeJob,
  ComputeJobHandle,
  ComputeJobResult,
  ComputeRunner,
  DataQuery,
  DataRecord,
  DataRecordInput,
  DataStorage,
  TrainerManifest,
} from 'thefactory-tools/types'
import { createModelTrainerTools } from './ModelTrainerTools.js'
import {
  isRunAffectedByFidelityLookahead,
  isSpecAffectedByFidelityLookahead,
} from './modelTrainerUtils.js'
import { HEAVY_RUN_FIELDS } from './modelTrainerConstants.js'

// A4 — the invalidation FLOW wired against the REAL fidelity-look-ahead predicates. Where
// ModelTrainerTools.test.ts exercises invalidateRuns with trivial `affected === true` stubs, this suite
// proves the flow behaves on genuine BlackSwan configs: an hourly step observing ONLY coarser layers
// (1h@1d) is stale, while a step that observes its own/base cadence (1h+1d) is spared. The gate + stamp
// semantics are asserted end-to-end so a regression in either the predicate wiring or the engine surfaces.

const NOW = '2026-07-09T12:00:00.000Z'

function manifest(overrides: Partial<TrainerManifest> = {}): TrainerManifest {
  return {
    name: 'blackswan',
    recordType: 'trainer-run',
    run: 'bin/python -m trainer.run --config-json {configPath} --summary-out {summaryOut}',
    objective: { name: 'traded_return', direction: 'max' },
    levers: {
      timeframe: { type: 'choice', choices: ['1m', '1h', '1d'], default: '1h' },
      fidelity_set: { type: 'choice', choices: ['1d', '1h+1d', 'auto'], default: '1h+1d' },
    },
    ...overrides,
  }
}

// Drop `omit` paths (incl. nested `a.b`) from record content, mirroring the real backends' projection so a
// lean list genuinely lacks the heavy fields — which is what forces invalidateRuns to re-read the full
// record by key before stamping (the fidelity of that round-trip is what test 5 pins).
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

interface MemoryStorage extends DataStorage {
  rows: Map<string, DataRecord>
  queries: DataQuery[]
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

function stubRunner(): ComputeRunner {
  return {
    async calibrate() {
      return { secondsObserved: 1, unitsPerSecond: 100 }
    },
    runJob(job: ComputeJob): ComputeJobHandle {
      const result: ComputeJobResult = {
        jobId: job.jobId,
        status: 'completed',
        exitCode: 0,
        summary: { objective: 0 },
        logTail: [],
        durationMs: 1,
      }
      return { jobId: job.jobId, onLog: () => {}, done: Promise.resolve(result), abort: () => {} }
    },
  }
}

function makeTools(storage: DataStorage) {
  return createModelTrainerTools({ computeRunner: stubRunner(), storage, now: () => NOW })
}

async function seedRun(
  storage: DataStorage,
  key: string,
  config: Record<string, unknown>,
  overrides: Record<string, unknown> = {},
) {
  await storage.upsertRecord({
    scope: 'proj',
    type: 'trainer-run',
    key,
    content: {
      objective: 5,
      status: 'completed',
      pipelineVersion: '5.0',
      config,
      ...overrides,
    },
  })
}

const baseParams = {
  scope: 'proj',
  projectRoot: '/repo',
  invalidationId: 'fidelityLookahead-v6',
  reason: 'fidelity look-ahead (coarse-only observation trades on future price)',
  // The fix landed in pipeline major 6: everything BELOW 6 is stale, re-runs at/after 6 are trusted.
  beforePipelineMajor: 6,
  affectsRun: isRunAffectedByFidelityLookahead,
}

describe('invalidateRuns × fidelity look-ahead (real predicate)', () => {
  it("stamps a v5 1h@1d run status='invalid' with the full provenance envelope (I: affected + pre-fix)", async () => {
    // 1h step observing ONLY {1d} — coarse-only — is the audited leak family: RED if the predicate stopped
    // matching it, or if the engine failed to stamp any of the provenance fields.
    const storage = memoryStorage()
    await seedRun(storage, 'r-1h-1d', { timeframe: '1h', fidelity_set: '1d' })
    const tools = makeTools(storage)

    const result = await tools.invalidateRuns({ ...baseParams, manifest: manifest() })

    expect(result).toMatchObject({ examinedRuns: 1, invalidatedRuns: 1 })
    const c = (await storage.readRecord({ scope: 'proj', type: 'trainer-run', key: 'r-1h-1d' }))!
      .content as Record<string, unknown>
    expect(c.status).toBe('invalid')
    expect(c.invalidReason).toBe(baseParams.reason)
    expect(c.invalidatedBy).toBe('fidelityLookahead-v6')
    expect(c.priorStatus).toBe('completed')
    expect(c.invalidatedAt).toBe(NOW)
  })

  it('flags the two ACTUAL audited v5 runs (supervised-logreg + supervised-gbm at 1h@1d)', async () => {
    // The concrete runs the audit caught; both must be invalidated by the real predicate.
    const storage = memoryStorage()
    await seedRun(storage, 'audit-logreg', {
      timeframe: '1h',
      fidelity_set: '1d',
      model_name: 'supervised-logreg',
    })
    await seedRun(storage, 'audit-gbm', {
      timeframe: '1h',
      fidelity_set: '1d',
      model_name: 'supervised-gbm',
    })
    const tools = makeTools(storage)

    const result = await tools.invalidateRuns({ ...baseParams, manifest: manifest() })

    expect(result).toMatchObject({ examinedRuns: 2, invalidatedRuns: 2 })
    for (const key of ['audit-logreg', 'audit-gbm']) {
      const c = (await storage.readRecord({ scope: 'proj', type: 'trainer-run', key }))!
        .content as Record<string, unknown>
      expect(c.status).toBe('invalid')
    }
  })

  it('SPARES a v5 1h+1d run — the base cadence (1h) is observed, so the narrow predicate is false', async () => {
    // Guards against conflating this look-ahead invalidation with the BROAD desync predicate (all
    // multi-timeline). 1h+1d observes the 1h base, so get_price aligns; it must NOT be invalidated.
    const storage = memoryStorage()
    await seedRun(storage, 'r-1h+1d', { timeframe: '1h', fidelity_set: '1h+1d' })
    const tools = makeTools(storage)

    const result = await tools.invalidateRuns({ ...baseParams, manifest: manifest() })

    expect(result.invalidatedRuns).toBe(0)
    const c = (await storage.readRecord({ scope: 'proj', type: 'trainer-run', key: 'r-1h+1d' }))!
      .content as Record<string, unknown>
    expect(c.status).toBe('completed')
  })

  it('SPARES an affected 1h@1d run at/after the fix major (v6 re-run is trusted, version gate)', async () => {
    // Same coarse-only config, but produced under the fixed pipeline (major 6). The version gate must
    // exempt it — RED if beforePipelineMajor comparison regressed to invalidate re-runs of the fix.
    const storage = memoryStorage()
    await seedRun(storage, 'r-v6', { timeframe: '1h', fidelity_set: '1d' }, { pipelineVersion: '6.0' })
    const tools = makeTools(storage)

    const result = await tools.invalidateRuns({ ...baseParams, manifest: manifest() })

    expect(result).toMatchObject({ examinedRuns: 1, invalidatedRuns: 0 })
    const c = (await storage.readRecord({ scope: 'proj', type: 'trainer-run', key: 'r-v6' }))!
      .content as Record<string, unknown>
    expect(c.status).toBe('completed')
  })

  it('leaves an already-invalid run untouched (idempotent — no re-stamp, no churn)', async () => {
    const storage = memoryStorage()
    await seedRun(
      storage,
      'r-prior',
      { timeframe: '1h', fidelity_set: '1d' },
      { status: 'invalid', invalidatedBy: 'someone-else', priorStatus: 'failed' },
    )
    const tools = makeTools(storage)

    const result = await tools.invalidateRuns({ ...baseParams, manifest: manifest() })

    expect(result.invalidatedRuns).toBe(0)
    const c = (await storage.readRecord({ scope: 'proj', type: 'trainer-run', key: 'r-prior' }))!
      .content as Record<string, unknown>
    // The pre-existing invalidation provenance is preserved, not overwritten by this pass.
    expect(c.invalidatedBy).toBe('someone-else')
    expect(c.priorStatus).toBe('failed')
  })

  it('MEMORY-SAFETY: scans lean (heavy fields omitted) yet the stamp preserves them (re-read by key)', async () => {
    // The boot-time sweep touches every run; loading each run's series/trace at once would OOM. The scan
    // must omit HEAVY_RUN_FIELDS, but the stamped record must still carry them — proving the full re-read.
    const storage = memoryStorage()
    await seedRun(
      storage,
      'r-heavy',
      { timeframe: '1h', fidelity_set: '1d' },
      {
        series: { equity: [1, 2, 3] },
        artifacts: { checkpoint: 'c.zip', decisionTrace: { steps: [{ step: 0, action: 'hold' }] } },
      },
    )
    const tools = makeTools(storage)

    await tools.invalidateRuns({ ...baseParams, manifest: manifest() })

    const runScans = storage.queries.filter((q) => q.type === 'trainer-run')
    expect(runScans.length).toBeGreaterThan(0)
    for (const q of runScans) expect(q.omit).toEqual(HEAVY_RUN_FIELDS)
    const c = (await storage.readRecord({ scope: 'proj', type: 'trainer-run', key: 'r-heavy' }))!
      .content as Record<string, unknown>
    expect(c.status).toBe('invalid')
    expect((c.series as { equity: number[] }).equity).toEqual([1, 2, 3])
    expect((c.artifacts as { decisionTrace: unknown }).decisionTrace).toEqual({
      steps: [{ step: 0, action: 'hold' }],
    })
    expect((c.artifacts as { checkpoint: string }).checkpoint).toBe('c.zip')
  })

  it('mixed corpus: only the coarse-only pre-fix runs are stamped, everything else survives', async () => {
    const storage = memoryStorage()
    await seedRun(storage, 'bad-1h-1d', { timeframe: '1h', fidelity_set: '1d' })
    await seedRun(storage, 'bad-1h-1d1w', { timeframe: '1h', fidelity_set: '1d+1w' })
    await seedRun(storage, 'ok-1h-1d', { timeframe: '1h', fidelity_set: '1h+1d' })
    await seedRun(storage, 'ok-auto', { timeframe: '1h', fidelity_set: 'auto' })
    await seedRun(storage, 'ok-daily', { timeframe: '1d', fidelity_set: '1d+1w' })
    const tools = makeTools(storage)

    const result = await tools.invalidateRuns({ ...baseParams, manifest: manifest() })

    expect(result).toMatchObject({ examinedRuns: 5, invalidatedRuns: 2 })
    const statusOf = async (key: string) =>
      ((await storage.readRecord({ scope: 'proj', type: 'trainer-run', key }))!.content as {
        status: string
      }).status
    expect(await statusOf('bad-1h-1d')).toBe('invalid')
    expect(await statusOf('bad-1h-1d1w')).toBe('invalid')
    expect(await statusOf('ok-1h-1d')).toBe('completed')
    expect(await statusOf('ok-auto')).toBe('completed')
    expect(await statusOf('ok-daily')).toBe('completed')
  })
})

describe('invalidateRuns pending-queue cancellation × fidelity spec predicate', () => {
  const queueParams = {
    ...baseParams,
    affectsPending: isSpecAffectedByFidelityLookahead,
    queueRecordType: 'trainer-queue',
    cancelPendingQueue: true,
  }

  it('cancels a coarse-only pending spec ONCE, spares a safe one, and writes the one-time marker', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'trainer-queue',
      key: 'q-bad',
      content: { params: { spec: { fixed: { timeframe: '1h', fidelity_set: '1d' } } } },
    })
    // A queue item whose FIXED base is safe but whose SWEEP fans into a coarse-only combo must also be caught.
    await storage.upsertRecord({
      scope: 'proj',
      type: 'trainer-queue',
      key: 'q-bad-sweep',
      content: {
        params: { spec: { fixed: { timeframe: '1h' }, sweep: { fidelity_set: ['1h+1d', '1d'] } } },
      },
    })
    await storage.upsertRecord({
      scope: 'proj',
      type: 'trainer-queue',
      key: 'q-good',
      content: { params: { spec: { fixed: { timeframe: '1h', fidelity_set: '1h+1d' } } } },
    })
    const tools = makeTools(storage)

    const result = await tools.invalidateRuns({ ...queueParams, manifest: manifest() })

    expect(result).toMatchObject({
      examinedQueue: 3,
      cancelledQueue: 2,
      pendingAlreadyApplied: false,
    })
    expect(
      await storage.readRecord({ scope: 'proj', type: 'trainer-queue', key: 'q-bad' }),
    ).toBeUndefined()
    expect(
      await storage.readRecord({ scope: 'proj', type: 'trainer-queue', key: 'q-bad-sweep' }),
    ).toBeUndefined()
    expect(
      await storage.readRecord({ scope: 'proj', type: 'trainer-queue', key: 'q-good' }),
    ).toBeDefined()
    const marker = await storage.readRecord({
      scope: 'proj',
      type: 'trainer-run-invalidation',
      key: 'fidelityLookahead-v6',
    })
    expect(marker).toBeDefined()
  })

  it('skips pending cancellation on a second pass (marker present) so re-queued runs survive', async () => {
    const storage = memoryStorage()
    await storage.upsertRecord({
      scope: 'proj',
      type: 'trainer-run-invalidation',
      key: 'fidelityLookahead-v6',
      content: { invalidationId: 'fidelityLookahead-v6', appliedAt: 'earlier' },
    })
    await storage.upsertRecord({
      scope: 'proj',
      type: 'trainer-queue',
      key: 'q-rerun',
      content: { params: { spec: { fixed: { timeframe: '1h', fidelity_set: '1d' } } } },
    })
    const tools = makeTools(storage)

    const result = await tools.invalidateRuns({ ...queueParams, manifest: manifest() })

    expect(result).toMatchObject({ cancelledQueue: 0, pendingAlreadyApplied: true })
    expect(
      await storage.readRecord({ scope: 'proj', type: 'trainer-queue', key: 'q-rerun' }),
    ).toBeDefined()
  })
})
