import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/continueTrain.js is the no-build browser module for continued-training (extra-train) presentation:
// the parent<->child lineage derived from checkpoint provenance, and the per-set evaluation matrix rows.
// Load it as CommonJS the same way crossTestViewer.test.ts loads viewer/crossTest.js so the ACTUAL viewer
// logic is unit-tested directly.
const require = createRequire(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'continueTrain.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const CtT: any = mod.exports

const run = (
  key: string,
  opts: {
    checkpoint?: string
    continuedFrom?: string
    config?: Record<string, unknown>
    objective?: number
    status?: string
  } = {},
) => ({
  key,
  summary: {
    ...(opts.objective !== undefined ? { objective: opts.objective } : {}),
    ...(opts.status ? { status: opts.status } : {}),
    config: opts.config || {},
    artifacts: opts.checkpoint ? { checkpoint: opts.checkpoint } : {},
    provenance: opts.continuedFrom ? { continuedFrom: opts.continuedFrom } : {},
  },
})

describe('checkpointOf / continuedFromOf', () => {
  it('reads the kept checkpoint + the parent-checkpoint provenance', () => {
    const r = run('a', { checkpoint: 'ckpt/a.zip', continuedFrom: 'ckpt/parent.zip' })
    expect(CtT.checkpointOf(r)).toBe('ckpt/a.zip')
    expect(CtT.continuedFromOf(r)).toBe('ckpt/parent.zip')
  })
  it('returns empty strings when absent', () => {
    expect(CtT.checkpointOf(run('a'))).toBe('')
    expect(CtT.continuedFromOf(run('a'))).toBe('')
    expect(CtT.checkpointOf(null as any)).toBe('')
  })
})

describe('lineageIndex', () => {
  it('joins a child to its parent via the parent checkpoint path', () => {
    const parent = run('parent', { checkpoint: 'ckpt/parent.zip', config: { asset: 'BTCUSDT' } })
    const child = run('child', { continuedFrom: 'ckpt/parent.zip', config: { asset: 'ETHUSDT' } })
    const idx = CtT.lineageIndex([parent, child])
    expect(CtT.parentKeyOf(child, idx)).toBe('parent')
    expect(CtT.childKeysOf(parent, idx)).toEqual(['child'])
    // A run with no lineage has neither.
    expect(CtT.parentKeyOf(parent, idx)).toBe('')
    expect(CtT.childKeysOf(child, idx)).toEqual([])
  })
  it('collects multiple children of one parent', () => {
    const parent = run('p', { checkpoint: 'ckpt/p.zip' })
    const c1 = run('c1', { continuedFrom: 'ckpt/p.zip' })
    const c2 = run('c2', { continuedFrom: 'ckpt/p.zip' })
    const idx = CtT.lineageIndex([parent, c1, c2])
    expect(CtT.childKeysOf(parent, idx).sort()).toEqual(['c1', 'c2'])
  })
  it('ignores a dangling provenance (parent not in the set) and never self-links', () => {
    const orphan = run('o', { checkpoint: 'ckpt/o.zip', continuedFrom: 'ckpt/o.zip' })
    const dangling = run('d', { continuedFrom: 'ckpt/missing.zip' })
    const idx = CtT.lineageIndex([orphan, dangling])
    expect(CtT.parentKeyOf(orphan, idx)).toBe('')
    expect(CtT.parentKeyOf(dangling, idx)).toBe('')
  })
})

describe('continuedMatrixRows', () => {
  it('emits one row per continued child, labelled by its dataset lever values', () => {
    const parent = run('p', {
      checkpoint: 'ckpt/p.zip',
      config: { asset: 'BTCUSDT', walk_forward_window: '2024' },
    })
    const c1 = run('c1', {
      continuedFrom: 'ckpt/p.zip',
      config: { asset: 'ETHUSDT', walk_forward_window: '2025' },
      objective: 3.5,
      status: 'completed',
    })
    const c2 = run('c2', {
      continuedFrom: 'ckpt/p.zip',
      config: { asset: 'SOLUSDT', walk_forward_window: '2025' },
      objective: 1.2,
      status: 'completed',
    })
    const runsByKey = new Map([
      ['p', parent],
      ['c1', c1],
      ['c2', c2],
    ])
    const idx = CtT.lineageIndex([parent, c1, c2])
    const rows = CtT.continuedMatrixRows(parent, idx, runsByKey, ['asset', 'walk_forward_window'])
    expect(rows).toEqual([
      { key: 'c1', label: 'ETHUSDT · 2025', objective: 3.5, status: 'completed' },
      { key: 'c2', label: 'SOLUSDT · 2025', objective: 1.2, status: 'completed' },
    ])
  })
  it('is empty for a run with no continued children', () => {
    const parent = run('p', { checkpoint: 'ckpt/p.zip' })
    const idx = CtT.lineageIndex([parent])
    expect(CtT.continuedMatrixRows(parent, idx, new Map([['p', parent]]), ['asset'])).toEqual([])
  })
})
