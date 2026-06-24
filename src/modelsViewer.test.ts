import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/models.js is the no-build browser module for the Models catalog; load it as CommonJS the same
// way hypothesisViewer.test.ts loads viewer/hypothesis.js, so the ACTUAL viewer logic is tested here.
const require = createRequire(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'models.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const M: any = mod.exports

const run = (
  modelName: string,
  opts: { objective?: number; status?: string; flags?: string[] } = {},
) => ({
  key: `run-${modelName}-${opts.objective ?? 'x'}`,
  summary: {
    config: { model_name: modelName },
    objective: opts.objective,
    status: opts.status || 'completed',
    health: { status: opts.flags && opts.flags.length ? 'flagged' : 'ok', flags: opts.flags || [] },
  },
})

const manifest = (choices: string[]) => ({
  levers: { model_name: { type: 'choice', choices } },
})

const model = (over: Record<string, unknown> = {}) => ({
  id: 'rainbow-dqn-custom',
  slug: 'rainbow-dqn-custom',
  name: 'Rainbow DQN',
  category: 'rl',
  status: 'implemented',
  statusSource: 'auto',
  modelNames: ['rainbow-dqn-custom'],
  source: 'manual',
  ...over,
})

describe('modelMatchesRun', () => {
  it('matches when the run config model_name is one of the bindings', () => {
    expect(M.modelMatchesRun(model(), { model_name: 'rainbow-dqn-custom' })).toBe(true)
  })
  it('rejects a different model_name', () => {
    expect(M.modelMatchesRun(model(), { model_name: 'dqn' })).toBe(false)
  })
  it('matches nothing when the model has no bindings', () => {
    expect(M.modelMatchesRun(model({ modelNames: [] }), { model_name: 'rainbow-dqn-custom' })).toBe(
      false,
    )
  })
})

describe('runsForModel', () => {
  it('keeps only the runs whose config binds to the model', () => {
    const runs = [run('rainbow-dqn-custom', { objective: 1 }), run('dqn', { objective: 2 })]
    expect(M.runsForModel(model(), runs).map((r: any) => r.key)).toEqual([
      'run-rainbow-dqn-custom-1',
    ])
  })
})

describe('isModelImplemented', () => {
  it('is true when a binding is one of the manifest model_name choices', () => {
    expect(M.isModelImplemented(model(), manifest(['rainbow-dqn-custom', 'dqn']))).toBe(true)
  })
  it('is true when the model has an implPath even if not in the lever', () => {
    expect(
      M.isModelImplemented(
        model({ modelNames: ['agent57'], implPath: 'src/model/agent57.py' }),
        manifest(['dqn']),
      ),
    ).toBe(true)
  })
  it('is false when no binding is in the lever and there is no implPath', () => {
    expect(M.isModelImplemented(model({ modelNames: ['nope'] }), manifest(['dqn']))).toBe(false)
  })
})

describe('deriveModelStatus', () => {
  it('returns a manual pin unchanged', () => {
    expect(
      M.deriveModelStatus(
        model({ statusSource: 'manual', status: 'needs-improvement' }),
        [],
        manifest(['rainbow-dqn-custom']),
      ),
    ).toBe('needs-improvement')
  })
  it('is proposed when not implemented and has no runs', () => {
    expect(M.deriveModelStatus(model({ modelNames: [] }), [], manifest(['dqn']))).toBe('proposed')
  })
  it('is implemented when lever-bound but no runs yet', () => {
    expect(M.deriveModelStatus(model(), [], manifest(['rainbow-dqn-custom']))).toBe('implemented')
  })
  it('is failing when every matching run is health-flagged or errored', () => {
    const runs = [
      run('rainbow-dqn-custom', { flags: ['zero_trades'] }),
      run('rainbow-dqn-custom', { status: 'failed' }),
    ]
    expect(M.deriveModelStatus(model(), runs, manifest(['rainbow-dqn-custom']))).toBe('failing')
  })
  it('is implemented when at least one matching run is healthy', () => {
    const runs = [
      run('rainbow-dqn-custom', { objective: 1 }),
      run('rainbow-dqn-custom', { status: 'failed' }),
    ]
    expect(M.deriveModelStatus(model(), runs, manifest(['rainbow-dqn-custom']))).toBe('implemented')
  })
})

describe('modelRunSummary', () => {
  it('counts runs, the best objective (direction-aware), and the failing count', () => {
    const runs = [
      run('rainbow-dqn-custom', { objective: 1 }),
      run('rainbow-dqn-custom', { objective: 3 }),
      run('rainbow-dqn-custom', { flags: ['nan_loss'] }),
      run('dqn', { objective: 99 }),
    ]
    expect(M.modelRunSummary(model(), runs, 'max')).toEqual({ runs: 3, best: 3, failing: 1 })
  })
  it('reports a null best when no matching run carries an objective', () => {
    expect(M.modelRunSummary(model(), [], 'max')).toEqual({ runs: 0, best: null, failing: 0 })
  })
})

describe('buildProposedModelRecord', () => {
  it('builds a proposed, paper-linked catalog record from a ProposedModel', () => {
    const rec = M.buildProposedModelRecord(
      {
        name: 'C51',
        slug: 'c51',
        description: 'categorical',
        category: 'rl',
        proposal: 'add C51 head',
      },
      'p1',
      '2026-06-23T00:00:00.000Z',
    )
    expect(rec).toEqual({
      id: 'c51',
      slug: 'c51',
      name: 'C51',
      description: 'categorical',
      category: 'rl',
      status: 'proposed',
      statusSource: 'auto',
      modelNames: [],
      paperIds: ['p1'],
      proposal: 'add C51 head',
      source: 'paper',
      createdAt: '2026-06-23T00:00:00.000Z',
      updatedAt: '2026-06-23T00:00:00.000Z',
    })
  })
})

describe('modelsForPaper', () => {
  it('unions both link directions (paper.modelIds and model.paperIds)', () => {
    const models = [
      model({ id: 'a', slug: 'a', paperIds: ['p1'] }),
      model({ id: 'b', slug: 'b' }),
      model({ id: 'c', slug: 'c' }),
    ]
    const out = M.modelsForPaper({ id: 'p1', modelIds: ['b'] }, models)
    expect(out.map((m: any) => m.id).sort()).toEqual(['a', 'b'])
  })
})

describe('papersForModel', () => {
  it('unions both link directions (model.paperIds and paper.modelIds)', () => {
    const papers = [
      { id: 'p1', modelIds: [] },
      { id: 'p2', modelIds: ['rainbow-dqn-custom'] },
      { id: 'p3', modelIds: [] },
    ]
    const out = M.papersForModel(model({ paperIds: ['p1'] }), papers)
    expect(out.map((p: any) => p.id).sort()).toEqual(['p1', 'p2'])
  })
})

describe('status + category metadata', () => {
  it('labels and badges every status', () => {
    expect(M.MODEL_STATUSES).toEqual([
      'proposed',
      'implemented',
      'failing',
      'needs-improvement',
      'deprecated',
    ])
    expect(M.MODEL_STATUS_LABEL.failing).toBe('failing')
    expect(M.MODEL_STATUS_BADGE.implemented).toBe('is-done')
    expect(M.MODEL_STATUS_BADGE.failing).toBe('is-failed')
  })
  it('orders categories for grouping', () => {
    expect(M.MODEL_CATEGORIES).toEqual(['rl', 'supervised', 'baseline', 'component'])
  })
})

describe('seed re-sync (consolidation reaches already-imported catalogs)', () => {
  const seed = {
    id: 'dueling-dqn',
    slug: 'dueling-dqn',
    name: 'Dueling DQN',
    description: 'value + advantage streams',
    category: 'rl',
    status: 'implemented',
    modelNames: ['duel-dqn-custom', 'duel-dqn', 'duel-dqn-custom-lstm'],
    implPath: 'src/model/dueling_dqn/dueling_dqn.py',
    source: 'manual',
  }

  it('flags a seed with no existing record as differing (a new import)', () => {
    expect(M.seedDiffersFromModel(seed, undefined)).toBe(true)
  })
  it('flags a binding change (the consolidation) as differing', () => {
    const existing = { ...seed, modelNames: ['duel-dqn-custom'] }
    expect(M.seedDiffersFromModel(seed, existing)).toBe(true)
  })
  it('is false when the manifest-owned fields already match', () => {
    expect(M.seedDiffersFromModel(seed, { ...seed, statusSource: 'auto' })).toBe(false)
  })

  it('merges seed (manifest) fields while preserving a manual status + user notes/dismissed/links', () => {
    const existing = {
      id: 'dueling-dqn',
      slug: 'dueling-dqn',
      name: 'old name',
      modelNames: ['duel-dqn-custom'],
      category: 'rl',
      status: 'needs-improvement',
      statusSource: 'manual',
      statusNote: 'hand-pinned',
      notes: 'keep this',
      dismissed: true,
      hypothesisIds: ['h1'],
      paperIds: ['p-user'],
      createdAt: '2026-06-01T00:00:00.000Z',
      updatedAt: '2026-06-01T00:00:00.000Z',
    }
    const merged = M.mergeSeedIntoModel(
      { ...seed, paperIds: ['p-seed'] },
      existing,
      '2026-06-24T00:00:00.000Z',
    )
    // manifest-owned fields come from the seed
    expect(merged.name).toBe('Dueling DQN')
    expect(merged.modelNames).toEqual(['duel-dqn-custom', 'duel-dqn', 'duel-dqn-custom-lstm'])
    expect(merged.implPath).toBe('src/model/dueling_dqn/dueling_dqn.py')
    // user-owned fields are preserved
    expect(merged.status).toBe('needs-improvement')
    expect(merged.statusSource).toBe('manual')
    expect(merged.statusNote).toBe('hand-pinned')
    expect(merged.notes).toBe('keep this')
    expect(merged.dismissed).toBe(true)
    expect(merged.hypothesisIds).toEqual(['h1'])
    expect(merged.paperIds.sort()).toEqual(['p-seed', 'p-user'])
    expect(merged.createdAt).toBe('2026-06-01T00:00:00.000Z')
    expect(merged.updatedAt).toBe('2026-06-24T00:00:00.000Z')
  })
  it('defaults status to the seed value + auto source for a fresh import', () => {
    const merged = M.mergeSeedIntoModel(seed, undefined, '2026-06-24T00:00:00.000Z')
    expect(merged.status).toBe('implemented')
    expect(merged.statusSource).toBe('auto')
    expect(merged.createdAt).toBe('2026-06-24T00:00:00.000Z')
  })
})
