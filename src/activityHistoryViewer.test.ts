import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/activityHistory.js is the no-build browser module for the Activity History popup's PURE logic
// (filter to this project's SETTLED activities, newest-finished first, duration + status class). Load it as
// CommonJS the same way crossTestViewer.test.ts loads viewer/crossTest.js so the ACTUAL logic is tested.
const require = createRequire(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'activityHistory.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const AH: any = mod.exports

const act = (o: Record<string, unknown> = {}) => ({
  activityId: 'a1',
  activityType: 'train',
  recordType: 'demo-run',
  status: 'completed',
  startedAt: '2026-07-30T10:00:00.000Z',
  finishedAt: '2026-07-30T10:03:12.000Z',
  ...o,
})

describe('historyRows', () => {
  it('keeps only SETTLED (completed/failed/aborted) activities of this project, newest-finished first', () => {
    const acts = [
      act({ activityId: 'run', status: 'running', finishedAt: undefined }),
      act({ activityId: 'q', status: 'queued', finishedAt: undefined }),
      act({
        activityId: 'other',
        status: 'completed',
        recordType: 'wine-run',
        finishedAt: '2026-07-30T11:00:00Z',
      }),
      act({ activityId: 'old', status: 'completed', finishedAt: '2026-07-30T09:00:00.000Z' }),
      act({
        activityId: 'new',
        status: 'failed',
        finishedAt: '2026-07-30T12:00:00.000Z',
        error: 'boom',
      }),
      act({ activityId: 'mid', status: 'aborted', finishedAt: '2026-07-30T10:30:00.000Z' }),
    ]
    const rows = AH.historyRows(acts, 'demo-run')
    expect(rows.map((r: any) => r.activityId)).toEqual(['new', 'mid', 'old']) // running/queued/other-project dropped, newest first
    expect(rows.find((r: any) => r.activityId === 'new').error).toBe('boom')
  })

  it('computes durationMs from started→finished (null when unparseable or negative)', () => {
    expect(AH.historyRows([act()], 'demo-run')[0].durationMs).toBe(192000) // 3m12s
    expect(AH.historyRows([act({ finishedAt: undefined })], 'demo-run')[0].durationMs).toBeNull()
    expect(
      AH.historyRows([act({ finishedAt: '2026-07-30T09:00:00Z' })], 'demo-run')[0].durationMs,
    ).toBeNull() // finished < started
  })

  it('carries the launch label (resumeToken.params._label), falling back to the type', () => {
    const withLabel = act({ resumeToken: { params: { _label: 'Auto cross-test' } } })
    expect(AH.historyRows([withLabel], 'demo-run')[0].label).toBe('Auto cross-test')
    expect(AH.historyRows([act()], 'demo-run')[0].label).toBe('train')
  })

  it('carries costUSD only when numeric', () => {
    expect(AH.historyRows([act({ costUSD: 0.42 })], 'demo-run')[0].costUSD).toBe(0.42)
    expect(AH.historyRows([act({ costUSD: 'x' })], 'demo-run')[0].costUSD).toBeNull()
  })

  it('returns [] for junk input', () => {
    expect(AH.historyRows(null, 'demo-run')).toEqual([])
    expect(AH.historyRows([null, undefined, {}], 'demo-run')).toEqual([])
  })

  it('with no recordType filter, keeps every project’s settled activities', () => {
    const rows = AH.historyRows([act({ recordType: 'demo-run' }), act({ recordType: 'wine-run' })])
    expect(rows).toHaveLength(2)
  })
})

describe('formatDuration', () => {
  it('formats seconds / minutes / hours, and — for null/negative', () => {
    expect(AH.formatDuration(45000)).toBe('45s')
    expect(AH.formatDuration(192000)).toBe('3m 12s')
    expect(AH.formatDuration(180000)).toBe('3m')
    expect(AH.formatDuration(3900000)).toBe('1h 5m')
    expect(AH.formatDuration(3600000)).toBe('1h')
    expect(AH.formatDuration(null)).toBe('—')
    expect(AH.formatDuration(-5)).toBe('—')
  })
})

describe('statusClass', () => {
  it('maps completed→ok, failed→bad, aborted→warn', () => {
    expect(AH.statusClass('completed')).toBe('is-ok')
    expect(AH.statusClass('failed')).toBe('is-bad')
    expect(AH.statusClass('aborted')).toBe('is-warn')
  })
})

describe('runKeysForActivity (A6 activity→runs jump)', () => {
  // caches are Maps runKey -> content; content.activityId is the producing/judging activity (stamped by the
  // tools). entriesLists = [evaluationsCache.entries(), verdictsCache.entries()].
  const evals = new Map<string, { activityId?: string }>([
    ['runA', { activityId: 'act-1' }],
    ['runB', { activityId: 'act-2' }],
    ['runC', {}], // never judged by an activity
  ])
  const verdicts = new Map<string, { activityId?: string }>([
    ['runA', { activityId: 'act-1' }], // same run judged + evaluated by act-1 → deduped
    ['runD', { activityId: 'act-1' }],
  ])
  const lists = () => [[...evals], [...verdicts]]

  it('collects the run keys any evaluation OR verdict stamped with this activity, deduped', () => {
    expect(AH.runKeysForActivity(lists(), 'act-1').sort()).toEqual(['runA', 'runD'])
    expect(AH.runKeysForActivity(lists(), 'act-2')).toEqual(['runB'])
  })

  it('returns [] for an unknown activity, a falsy id, or junk lists', () => {
    expect(AH.runKeysForActivity(lists(), 'act-none')).toEqual([])
    expect(AH.runKeysForActivity(lists(), '')).toEqual([])
    expect(AH.runKeysForActivity(null, 'act-1')).toEqual([])
  })
})

describe('stopOutcomeNote', () => {
  it.each([
    [{ outcome: 'stopped', status: 'aborted' }, 'ok', 'Stopped'],
    [{ outcome: 'noop', status: 'completed' }, 'muted', 'Already completed — nothing changed'],
    [{ outcome: 'not-stopped', status: 'completed' }, 'warn', 'Not stopped — it is completed'],
    [
      { outcome: 'failed', code: 'adapter-failed' },
      'warn',
      'Couldn’t confirm it stopped — it may still be running',
    ],
    [{ outcome: 'refused', code: 'not-found' }, 'warn', 'Not stopped — not-found'],
  ])('says what the host observed for %j', (result, tone, text) => {
    expect(AH.stopOutcomeNote(result)).toEqual({ tone, text })
  })

  it('counts what it started that was stopped with it, and warns when some may still run', () => {
    const c = (outcome: string) => ({ activityId: outcome, parentActivityId: 'a', outcome })
    expect(AH.stopOutcomeNote({ outcome: 'stopped', children: [c('stopped'), c('noop')] })).toEqual(
      {
        tone: 'ok',
        text: 'Stopped · also stopped 1 it started',
      },
    )
    expect(
      AH.stopOutcomeNote({ outcome: 'stopped', children: [c('stopped'), c('refused')] }),
    ).toEqual({ tone: 'warn', text: 'Stopped · 1 of 2 it started may still be running' })
  })

  it('says a stop the bridge stopped waiting on is still being carried out, and why one could not be sent', () => {
    const still = { tone: 'warn', text: 'Still stopping — refresh to see where it stands' }
    expect(AH.stopOutcomeNote(undefined)).toEqual(still)
    expect(AH.stopOutcomeNote(undefined, 'Overseer bridge timeout: activities.stop')).toEqual(still)
    expect(AH.stopOutcomeNote(undefined, 'host down')).toEqual({
      tone: 'warn',
      text: 'Couldn’t stop it — host down',
    })
  })

  it('reads a stop as having ended its activity only when the record has ended', () => {
    expect(AH.stopEnded({ outcome: 'stopped' })).toBe(true)
    expect(AH.stopEnded({ outcome: 'noop' })).toBe(true)
    expect(AH.stopEnded({ outcome: 'not-stopped' })).toBe(true)
    expect(AH.stopEnded({ outcome: 'failed' })).toBe(false)
    expect(AH.stopEnded({ outcome: 'refused' })).toBe(false)
    expect(AH.stopEnded(undefined)).toBe(false)
  })
})
