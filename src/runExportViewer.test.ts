import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/runExport.js is the no-build browser module that assembles the compare-mode audit export;
// load it as CommonJS the same way modelsViewer.test.ts loads viewer/models.js, so the ACTUAL viewer
// logic is tested here.
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'runExport.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const RE: any = mod.exports

// A summary carrying every field the genuineness audit needs: metrics (incl. return_vs_hold_pct),
// top-level regimes + benchmark (NOT under metrics), health, config, and artifacts.decisionTrace.
const summary = (over: Record<string, unknown> = {}) => ({
  config: { model_name: 'duel-dqn-custom', timeframe: '1h', reward_model: 'combo_unified' },
  objective: 60,
  status: 'completed',
  health: { status: 'ok', flags: [] },
  metrics: { traded_return: 60, return_vs_hold_pct: -12, blocked_signal_ratio: 0.95, n_trades: 25 },
  regimes: { trend: [{ regime: 'up', realized_pnl_pct: 70, bars_pct: 0.6 }], windows: [] },
  benchmark: { hold_return_pct: 72 },
  artifacts: {
    decisionTrace: { steps: [{ action: 1, confidence: 0.8 }], actionCounts: { hold: 99 } },
  },
  seed: 1,
  pipelineVersion: '3',
  ...over,
})

const run = (key: string, over?: Record<string, unknown>) => ({ key, summary: summary(over) })

describe('buildRunsAuditExport', () => {
  it('preserves each run summary VERBATIM — regimes, benchmark and decisionTrace are not dropped', () => {
    const r = run('run-1')
    const out = RE.buildRunsAuditExport([r], {})
    expect(out.runs[0].summary).toEqual(r.summary)
    expect(out.runs[0].summary.regimes).toBeDefined()
    expect(out.runs[0].summary.benchmark).toBeDefined()
    expect(out.runs[0].summary.artifacts.decisionTrace).toBeDefined()
  })

  it('stamps a schema tag and the run count', () => {
    const out = RE.buildRunsAuditExport([run('a'), run('b')], {})
    expect(out.schema).toBe('blackswan-runs-audit/v1')
    expect(out.count).toBe(2)
    expect(out.runs).toHaveLength(2)
  })

  it('preserves each run key in order', () => {
    const out = RE.buildRunsAuditExport([run('a'), run('b')], {})
    expect(out.runs.map((r: { key: string }) => r.key)).toEqual(['a', 'b'])
  })

  it('carries header meta (exportedAt, objective, project) through', () => {
    const out = RE.buildRunsAuditExport([run('a')], {
      exportedAt: '2026-06-26T00:00:00.000Z',
      objective: { name: 'traded_return', direction: 'max' },
      project: 'BlackSwan',
    })
    expect(out.exportedAt).toBe('2026-06-26T00:00:00.000Z')
    expect(out.objective).toEqual({ name: 'traded_return', direction: 'max' })
    expect(out.project).toBe('BlackSwan')
  })

  it('defaults header meta to null when not supplied', () => {
    const out = RE.buildRunsAuditExport([run('a')])
    expect(out.exportedAt).toBeNull()
    expect(out.objective).toBeNull()
    expect(out.project).toBeNull()
  })

  it('filters out null / summary-less runs', () => {
    const out = RE.buildRunsAuditExport([null, undefined, { key: 'x' }, run('a')], {})
    expect(out.count).toBe(1)
    expect(out.runs[0].key).toBe('a')
  })

  it('returns an empty export for a non-array input', () => {
    const out = RE.buildRunsAuditExport(undefined, {})
    expect(out.count).toBe(0)
    expect(out.runs).toEqual([])
  })
})
