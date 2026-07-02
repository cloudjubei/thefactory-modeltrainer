import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/badRuns.js is the no-build browser module for the "bad run" DEFINITION — which runs the Hide-bad
// toggle drops (a configurable list of criteria: failed, health-flagged, under-traded). Load it as CommonJS
// the same way datasetsViewer.test.ts loads viewer/datasets.js, so the ACTUAL viewer logic is unit-tested.
const require = createRequire(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'badRuns.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const B: any = mod.exports

const run = (over: any = {}) => ({
  key: over.key || 'r1',
  summary: {
    status: over.status || 'completed',
    health: over.health,
    metrics: over.metrics || {},
    ...(over.summary || {}),
  },
})

describe('defaultBadRunDefinition', () => {
  it('drops failed + degenerate + under-traded (≤2) by default', () => {
    expect(B.defaultBadRunDefinition()).toEqual({ failed: true, degenerate: true, minTrades: 2 })
  })
  it('returns a fresh object each call (no shared mutation)', () => {
    const a = B.defaultBadRunDefinition()
    a.minTrades = 99
    expect(B.defaultBadRunDefinition().minTrades).toBe(2)
  })
})

describe('normalizeBadRunDefinition', () => {
  it('fills missing fields from the default', () => {
    expect(B.normalizeBadRunDefinition({})).toEqual({
      failed: true,
      degenerate: true,
      minTrades: 2,
    })
  })
  it('coerces minTrades to a finite number or null', () => {
    expect(B.normalizeBadRunDefinition({ minTrades: '5' }).minTrades).toBe(5)
    expect(B.normalizeBadRunDefinition({ minTrades: null }).minTrades).toBe(null)
    expect(B.normalizeBadRunDefinition({ minTrades: 'x' }).minTrades).toBe(null)
  })
  it('keeps a minTrades of 0 as 0 (a real threshold), but treats "" as null (off)', () => {
    expect(B.normalizeBadRunDefinition({ minTrades: 0 }).minTrades).toBe(0)
    expect(B.normalizeBadRunDefinition({ minTrades: '' }).minTrades).toBe(null)
  })
  it('coerces the boolean flags', () => {
    const d = B.normalizeBadRunDefinition({ failed: false, degenerate: 0 as any })
    expect(d.failed).toBe(false)
    expect(d.degenerate).toBe(false)
  })
  it('treats a nullish input as the default', () => {
    expect(B.normalizeBadRunDefinition(null)).toEqual(B.defaultBadRunDefinition())
    expect(B.normalizeBadRunDefinition(undefined)).toEqual(B.defaultBadRunDefinition())
  })
})

describe('isBadRun', () => {
  const def = B.defaultBadRunDefinition()
  it('flags a failed run', () => {
    expect(B.isBadRun(def, run({ status: 'failed' }))).toBe(true)
  })
  it('flags a health-flagged (degenerate) run', () => {
    expect(B.isBadRun(def, run({ health: { status: 'degenerate' } }))).toBe(true)
  })
  it('flags an under-traded run (n_trades ≤ threshold)', () => {
    expect(B.isBadRun(def, run({ metrics: { n_trades: 2 } }))).toBe(true)
    expect(B.isBadRun(def, run({ metrics: { n_trades: 0 } }))).toBe(true)
  })
  it('keeps a healthy, sufficiently-traded run', () => {
    expect(B.isBadRun(def, run({ metrics: { n_trades: 50 }, health: { status: 'ok' } }))).toBe(
      false,
    )
  })
  it('keeps a run whose n_trades is absent/non-numeric (only finite ≤ threshold is bad)', () => {
    expect(B.isBadRun(def, run({ metrics: {} }))).toBe(false)
    expect(B.isBadRun(def, run({ metrics: { n_trades: 'na' } }))).toBe(false)
  })
  it('respects a disabled `failed` criterion', () => {
    expect(
      B.isBadRun({ failed: false, degenerate: true, minTrades: 2 }, run({ status: 'failed' })),
    ).toBe(false)
  })
  it('respects a disabled `degenerate` criterion', () => {
    expect(
      B.isBadRun(
        { failed: true, degenerate: false, minTrades: 2 },
        run({ health: { status: 'degenerate' } }),
      ),
    ).toBe(false)
  })
  it('respects a null minTrades (no under-trade criterion)', () => {
    expect(
      B.isBadRun(
        { failed: true, degenerate: true, minTrades: null },
        run({ metrics: { n_trades: 0 } }),
      ),
    ).toBe(false)
  })
  it('honours a raised minTrades threshold', () => {
    expect(
      B.isBadRun(
        { failed: true, degenerate: true, minTrades: 10 },
        run({ metrics: { n_trades: 8 } }),
      ),
    ).toBe(true)
  })
  it('a minTrades of 0 flags a zero-trade run but keeps a one-trade run', () => {
    const def = { failed: true, degenerate: true, minTrades: 0 }
    expect(B.isBadRun(def, run({ metrics: { n_trades: 0 } }))).toBe(true)
    expect(B.isBadRun(def, run({ metrics: { n_trades: 1 } }))).toBe(false)
  })
  it('keeps a run with an empty-string health.status (not a real degenerate flag)', () => {
    expect(B.isBadRun(B.defaultBadRunDefinition(), run({ health: { status: '' } }))).toBe(false)
  })
  it('normalizes a partial definition before testing', () => {
    expect(B.isBadRun({} as any, run({ status: 'failed' }))).toBe(true)
  })
})

describe('badRunWhere (server-side negation — keeps GOOD runs)', () => {
  it('combines all enabled criteria with and(), each exists-guarded so absent fields survive', () => {
    const w = B.badRunWhere(B.defaultBadRunDefinition())
    expect(w).toEqual({
      and: [
        { not: { field: 'status', op: '=', value: 'failed' } },
        {
          or: [
            { not: { field: 'health.status', op: 'exists' } },
            { field: 'health.status', op: '=', value: '' },
            { field: 'health.status', op: '=', value: 'ok' },
          ],
        },
        {
          or: [
            { not: { field: 'metrics.n_trades', op: 'exists' } },
            { not: { field: 'metrics.n_trades', op: '<=', value: 2 } },
          ],
        },
      ],
    })
  })
  it('under-trade keeps runs whose n_trades is ABSENT (not-exists OR not-<=), matching isBadRun', () => {
    // The whole point: a run with no metrics.n_trades must NOT be dropped server-side. isBadRun keeps it
    // (Number.isFinite(NaN) === false), so the where must include a not-exists branch — else NOT(NULL) in
    // SQL drops it. This asserts the exists-guard is present for the under-trade criterion.
    const w = B.badRunWhere({ failed: false, degenerate: false, minTrades: 5 })
    expect(w).toEqual({
      and: [
        {
          or: [
            { not: { field: 'metrics.n_trades', op: 'exists' } },
            { not: { field: 'metrics.n_trades', op: '<=', value: 5 } },
          ],
        },
      ],
    })
  })
  it('degenerate keeps an empty-string health.status (matching isBadRun, which needs a non-empty status)', () => {
    // isBadRun treats health.status === '' as NOT degenerate (falsy) → keeps it, so the where must too.
    const w = B.badRunWhere({ failed: false, degenerate: true, minTrades: null })
    expect(w).toEqual({
      and: [
        {
          or: [
            { not: { field: 'health.status', op: 'exists' } },
            { field: 'health.status', op: '=', value: '' },
            { field: 'health.status', op: '=', value: 'ok' },
          ],
        },
      ],
    })
  })
  it('returns undefined when no criteria are enabled', () => {
    expect(B.badRunWhere({ failed: false, degenerate: false, minTrades: null })).toBeUndefined()
  })
  it('a minTrades of 0 still emits the under-trade guard (0 is a real threshold, not "off")', () => {
    const w = B.badRunWhere({ failed: false, degenerate: false, minTrades: 0 })
    expect(w).toEqual({
      and: [
        {
          or: [
            { not: { field: 'metrics.n_trades', op: 'exists' } },
            { not: { field: 'metrics.n_trades', op: '<=', value: 0 } },
          ],
        },
      ],
    })
  })
  it('normalizes a partial definition', () => {
    expect(B.badRunWhere({})).toEqual(B.badRunWhere(B.defaultBadRunDefinition()))
  })
})
