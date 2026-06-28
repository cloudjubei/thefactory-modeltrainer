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

// A run row for computeModelStats: { key, config, objective, status, health, ranAt }.
const row = (
  modelName: string,
  opts: {
    objective?: number
    status?: string
    flags?: string[]
    ranAt?: string
    config?: Record<string, unknown>
  } = {},
) => ({
  key: `run-${modelName}-${opts.objective ?? 'x'}`,
  config: { model_name: modelName, ...(opts.config || {}) },
  objective: opts.objective,
  status: opts.status || 'completed',
  health: { status: opts.flags && opts.flags.length ? 'flagged' : 'ok', flags: opts.flags || [] },
  ranAt: opts.ranAt,
})

const manifest = (choices: string[]) => ({ levers: { model_name: { type: 'choice', choices } } })

const model = (over: Record<string, unknown> = {}) => ({
  id: 'dueling-dqn',
  slug: 'dueling-dqn',
  name: 'Dueling DQN',
  category: 'rl',
  status: 'implemented',
  statusSource: 'auto',
  flavors: [
    { name: 'custom', modelName: 'duel-dqn-custom' },
    { name: 'vanilla', modelName: 'duel-dqn' },
  ],
  source: 'manual',
  ...over,
})

describe('modelFlavors', () => {
  it('returns the structured flavors', () => {
    expect(M.modelFlavors(model()).map((f: any) => f.modelName)).toEqual([
      'duel-dqn-custom',
      'duel-dqn',
    ])
  })
  it('derives one flavor per legacy modelNames entry', () => {
    expect(M.modelFlavors({ modelNames: ['a', 'b'] })).toEqual([
      { modelName: 'a' },
      { modelName: 'b' },
    ])
  })
  it('is empty for a flavorless proposal', () => {
    expect(M.modelFlavors({ flavors: [] })).toEqual([])
  })
})

describe('flavorMatchesConfig', () => {
  it('matches by model_name alone', () => {
    expect(M.flavorMatchesConfig({ modelName: 'duel-dqn' }, { model_name: 'duel-dqn' })).toBe(true)
  })
  it('rejects a different model_name', () => {
    expect(M.flavorMatchesConfig({ modelName: 'duel-dqn' }, { model_name: 'dqn' })).toBe(false)
  })
  it('narrows by an extra config value (loose-equal)', () => {
    const fl = { modelName: 'duel-dqn-custom-lstm', config: { lstm_hidden_size: 3 } }
    expect(
      M.flavorMatchesConfig(fl, { model_name: 'duel-dqn-custom-lstm', lstm_hidden_size: '3' }),
    ).toBe(true)
    expect(
      M.flavorMatchesConfig(fl, { model_name: 'duel-dqn-custom-lstm', lstm_hidden_size: 2 }),
    ).toBe(false)
  })
})

describe('flavorKey', () => {
  it('is model_name + a sorted config signature', () => {
    expect(M.flavorKey({ modelName: 'x', config: { b: 2, a: 1 } })).toBe('x|a=1&b=2')
    expect(M.flavorKey({ modelName: 'x' })).toBe('x|')
  })
})

describe('runMatchesModel + flavorModelNames', () => {
  it('matches a run binding any flavor', () => {
    expect(M.runMatchesModel(model(), { model_name: 'duel-dqn' })).toBe(true)
    expect(M.runMatchesModel(model(), { model_name: 'dqn' })).toBe(false)
  })
  it('lists the distinct flavor model_names', () => {
    expect(M.flavorModelNames(model())).toEqual(['duel-dqn-custom', 'duel-dqn'])
  })
})

describe('isModelImplemented', () => {
  it('is true when a flavor model_name is a manifest choice', () => {
    expect(M.isModelImplemented(model(), manifest(['duel-dqn-custom']))).toBe(true)
  })
  it('is true when a flavor carries an implPath even if not in the lever', () => {
    expect(
      M.isModelImplemented(
        { flavors: [{ modelName: 'agent57', implPath: 'src/model/agent57.py' }] },
        manifest(['dqn']),
      ),
    ).toBe(true)
  })
  it('is false when no flavor is in the lever and there is no implPath', () => {
    expect(M.isModelImplemented({ flavors: [{ modelName: 'nope' }] }, manifest(['dqn']))).toBe(
      false,
    )
  })
})

describe('computeModelStats', () => {
  const models = [
    model({
      flavors: [
        { name: 'custom', modelName: 'duel-dqn-custom' },
        { name: 'vanilla', modelName: 'duel-dqn' },
      ],
    }),
    {
      id: 'recurrent-ppo',
      slug: 'recurrent-ppo',
      flavors: [{ modelName: 'reppo-custom' }],
    },
  ]

  it('aggregates per-model + per-flavor over ALL runs (counts, best, failing, lastRunAt)', () => {
    const runs = [
      row('duel-dqn-custom', { objective: 1, ranAt: '2026-06-01' }),
      row('duel-dqn-custom', { objective: 5, ranAt: '2026-06-03' }),
      row('duel-dqn', { objective: 2, ranAt: '2026-06-02' }),
      row('duel-dqn-custom', { flags: ['nan_loss'], ranAt: '2026-06-04' }),
      row('reppo-custom', { objective: 9 }),
    ]
    const stats = M.computeModelStats(models, runs, 'max')
    expect(stats.totalRuns).toBe(5)
    expect(stats.newestRunAt).toBe('2026-06-04')
    const dd = stats.perModel['dueling-dqn']
    expect(dd.runs).toBe(4)
    expect(dd.best).toBe(5)
    expect(dd.failing).toBe(1)
    expect(dd.lastRunAt).toBe('2026-06-04')
    expect(dd.perFlavor['duel-dqn-custom|'].runs).toBe(3)
    expect(dd.perFlavor['duel-dqn|'].runs).toBe(1)
    expect(stats.perModel['recurrent-ppo'].runs).toBe(1)
    expect(stats.perModel['recurrent-ppo'].best).toBe(9)
  })

  it('honours the objective direction for best', () => {
    const stats = M.computeModelStats(
      models,
      [row('duel-dqn', { objective: 2 }), row('duel-dqn', { objective: 8 })],
      'min',
    )
    expect(stats.perModel['dueling-dqn'].best).toBe(2)
  })

  it('collects runs matching NO flavor as uncataloged (the missing-flavor signal)', () => {
    const runs = [
      row('duel-dqn-custom', { objective: 1 }),
      row('duel-dqn-custom-lstm3', { objective: 2 }),
      row('duel-dqn-custom-lstm3', { objective: 3 }),
      row('agent57', { objective: 4 }),
    ]
    const stats = M.computeModelStats(models, runs, 'max')
    expect(stats.uncataloged.map((u: any) => [u.modelName, u.runs])).toEqual([
      ['duel-dqn-custom-lstm3', 2],
      ['agent57', 1],
    ])
  })
})

describe('aggForModel + deriveModelStatus', () => {
  const mani = manifest(['duel-dqn-custom'])
  it('reads a model aggregate from a stats record', () => {
    const stats = { perModel: { 'dueling-dqn': { runs: 3, best: 5, failing: 0 } } }
    expect(M.aggForModel(stats, 'dueling-dqn').runs).toBe(3)
    expect(M.aggForModel(stats, 'missing')).toBe(null)
  })
  it('returns a manual pin unchanged', () => {
    expect(
      M.deriveModelStatus(
        model({ statusSource: 'manual', status: 'needs-improvement' }),
        null,
        mani,
      ),
    ).toBe('needs-improvement')
  })
  it('is proposed when unimplemented with no runs', () => {
    expect(M.deriveModelStatus({ flavors: [] }, null, mani)).toBe('proposed')
  })
  it('respects a PINNED status (deferred) even from an auto source, not just manual', () => {
    // A manifest seed can declare a model deferred; auto-derivation must not flip it back to proposed.
    expect(M.deriveModelStatus({ status: 'deferred', statusSource: 'auto', flavors: [] }, null, mani)).toBe(
      'deferred',
    )
    // and even if (hypothetically) runs exist, the deliberate pin stands
    expect(
      M.deriveModelStatus({ status: 'deferred', statusSource: 'auto', flavors: [] }, { runs: 3, failing: 0 }, mani),
    ).toBe('deferred')
  })
  it('is implemented when lever-bound but unrun', () => {
    expect(M.deriveModelStatus(model(), null, mani)).toBe('implemented')
  })
  it('is failing when every aggregated run is unhealthy', () => {
    expect(M.deriveModelStatus(model(), { runs: 4, failing: 4 }, mani)).toBe('failing')
  })
  it('is implemented when at least one aggregated run is healthy', () => {
    expect(M.deriveModelStatus(model(), { runs: 4, failing: 1 }, mani)).toBe('implemented')
  })
})

describe('buildProposedModelRecord', () => {
  it('builds a proposed, flavorless, paper-linked catalog record', () => {
    const rec = M.buildProposedModelRecord(
      {
        name: 'C51',
        slug: 'c51',
        description: 'categorical',
        category: 'rl',
        proposal: 'add C51 head',
      },
      'p1',
      '2026-06-24T00:00:00.000Z',
    )
    expect(rec).toEqual({
      id: 'c51',
      slug: 'c51',
      name: 'C51',
      description: 'categorical',
      category: 'rl',
      status: 'proposed',
      statusSource: 'auto',
      flavors: [],
      paperIds: ['p1'],
      proposal: 'add C51 head',
      source: 'paper',
      createdAt: '2026-06-24T00:00:00.000Z',
      updatedAt: '2026-06-24T00:00:00.000Z',
    })
  })
})

describe('modelsForPaper / papersForModel', () => {
  it('unions both link directions for a paper', () => {
    const models = [model({ id: 'a', paperIds: ['p1'] }), model({ id: 'b' }), model({ id: 'c' })]
    expect(
      M.modelsForPaper({ id: 'p1', modelIds: ['b'] }, models)
        .map((m: any) => m.id)
        .sort(),
    ).toEqual(['a', 'b'])
  })
  it('unions both link directions for a model', () => {
    const papers = [
      { id: 'p1', modelIds: [] },
      { id: 'p2', modelIds: ['dueling-dqn'] },
      { id: 'p3', modelIds: [] },
    ]
    expect(
      M.papersForModel(model({ paperIds: ['p1'] }), papers)
        .map((p: any) => p.id)
        .sort(),
    ).toEqual(['p1', 'p2'])
  })
})

describe('status + category metadata', () => {
  it('labels and badges every status', () => {
    expect(M.MODEL_STATUSES).toEqual([
      'proposed',
      'implemented',
      'failing',
      'needs-improvement',
      'deferred',
      'deprecated',
    ])
    expect(M.MODEL_STATUS_BADGE.implemented).toBe('is-done')
    expect(M.MODEL_STATUS_BADGE.failing).toBe('is-failed')
    expect(M.MODEL_STATUS_LABEL.deferred).toBe('deferred')
  })
  it('orders categories for grouping', () => {
    expect(M.MODEL_CATEGORIES).toEqual(['rl', 'supervised', 'baseline', 'component'])
  })
})

describe('seed re-sync (flavors)', () => {
  const seed = {
    id: 'dueling-dqn',
    slug: 'dueling-dqn',
    name: 'Dueling DQN',
    description: 'value + advantage streams',
    category: 'rl',
    status: 'implemented',
    flavors: [
      { name: 'custom', modelName: 'duel-dqn-custom' },
      { name: 'vanilla', modelName: 'duel-dqn' },
      { name: 'custom + LSTM', modelName: 'duel-dqn-custom-lstm' },
    ],
    implPath: 'src/model/dueling_dqn/dueling_dqn.py',
    source: 'manual',
  }
  it('flags a new seed as differing', () => {
    expect(M.seedDiffersFromModel(seed, undefined)).toBe(true)
  })
  it('flags a flavor change (the consolidation) as differing', () => {
    const existing = { ...seed, flavors: [{ modelName: 'duel-dqn-custom' }] }
    expect(M.seedDiffersFromModel(seed, existing)).toBe(true)
  })
  it('is false when flavors + scalar fields already match', () => {
    expect(M.seedDiffersFromModel(seed, { ...seed, statusSource: 'auto' })).toBe(false)
  })
  it('merges seed flavors while preserving a manual status + user notes/dismissed/links', () => {
    const existing = {
      id: 'dueling-dqn',
      slug: 'dueling-dqn',
      name: 'old',
      flavors: [{ modelName: 'duel-dqn-custom' }],
      category: 'rl',
      status: 'needs-improvement',
      statusSource: 'manual',
      statusNote: 'pinned',
      notes: 'keep',
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
    expect(merged.flavors.map((f: any) => f.modelName)).toEqual([
      'duel-dqn-custom',
      'duel-dqn',
      'duel-dqn-custom-lstm',
    ])
    expect(merged.status).toBe('needs-improvement')
    expect(merged.statusSource).toBe('manual')
    expect(merged.notes).toBe('keep')
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
  })
})

describe('mergeModelsForConsolidation', () => {
  const canonical = (over: Record<string, unknown> = {}) => ({
    id: 'itransformer-ppo',
    slug: 'itransformer-ppo',
    name: 'iTransformer-PPO',
    description: 'Inverted-attention PPO.',
    category: 'rl',
    status: 'needs-improvement',
    statusSource: 'manual',
    statusNote: 'pin',
    notes: 'mine',
    source: 'manual',
    flavors: [{ name: 'base', modelName: 'itransformer-ppo' }],
    paperIds: ['p-itransf'],
    hypothesisIds: ['h-a'],
    createdAt: '2026-06-01T00:00:00.000Z',
    updatedAt: '2026-06-01T00:00:00.000Z',
    ...over,
  })
  const dup = (over: Record<string, unknown> = {}) => ({
    id: 'inverted-transformer-ppo',
    slug: 'inverted-transformer-ppo',
    name: 'Inverted Transformer PPO',
    description: 'Same thing, other paper.',
    category: 'rl',
    status: 'implemented',
    statusSource: 'auto',
    source: 'paper',
    flavors: [{ name: 'variant', modelName: 'itransformer-ppo-v2' }],
    paperIds: ['p-inv'],
    hypothesisIds: ['h-b'],
    createdAt: '2026-06-02T00:00:00.000Z',
    updatedAt: '2026-06-02T00:00:00.000Z',
    ...over,
  })
  const NOW = '2026-06-27T00:00:00.000Z'

  it("unions flavors (canonical first) and de-dupes by flavorKey", () => {
    const merged = M.mergeModelsForConsolidation(
      canonical(),
      [dup(), dup({ id: 'x', slug: 'x', flavors: [{ modelName: 'itransformer-ppo' }] })],
      NOW,
    )
    // canonical's flavor first, dup's distinct flavor next, the third dup's flavor collides with
    // canonical's flavorKey so it is dropped.
    expect(merged.flavors.map((f: any) => f.modelName)).toEqual([
      'itransformer-ppo',
      'itransformer-ppo-v2',
    ])
  })

  it('unions paperIds + hypothesisIds across canonical and duplicates', () => {
    const merged = M.mergeModelsForConsolidation(canonical(), [dup()], NOW)
    expect(merged.paperIds.sort()).toEqual(['p-inv', 'p-itransf'])
    expect(merged.hypothesisIds.sort()).toEqual(['h-a', 'h-b'])
  })

  it('keeps the canonical identity + manual status/notes, refreshes updatedAt', () => {
    const merged = M.mergeModelsForConsolidation(canonical(), [dup()], NOW)
    expect(merged.id).toBe('itransformer-ppo')
    expect(merged.slug).toBe('itransformer-ppo')
    expect(merged.name).toBe('iTransformer-PPO')
    expect(merged.status).toBe('needs-improvement')
    expect(merged.statusSource).toBe('manual')
    expect(merged.statusNote).toBe('pin')
    expect(merged.notes).toBe('mine')
    expect(merged.createdAt).toBe('2026-06-01T00:00:00.000Z')
    expect(merged.updatedAt).toBe(NOW)
  })

  it('ignores a duplicate that is actually the canonical (same id)', () => {
    const merged = M.mergeModelsForConsolidation(canonical(), [canonical()], NOW)
    expect(merged.flavors.map((f: any) => f.modelName)).toEqual(['itransformer-ppo'])
    expect(merged.paperIds).toEqual(['p-itransf'])
  })

  it('derives flavors from legacy modelNames when flavors[] is absent', () => {
    const merged = M.mergeModelsForConsolidation(
      canonical({ flavors: undefined, modelNames: ['itransformer-ppo'] }),
      [dup({ flavors: undefined, modelNames: ['legacy-name'] })],
      NOW,
    )
    expect(merged.flavors.map((f: any) => f.modelName)).toEqual(['itransformer-ppo', 'legacy-name'])
    expect(merged.modelNames).toBeUndefined()
  })

  it('omits paperIds/hypothesisIds when none exist anywhere', () => {
    const merged = M.mergeModelsForConsolidation(
      canonical({ paperIds: undefined, hypothesisIds: undefined }),
      [dup({ paperIds: [], hypothesisIds: [] })],
      NOW,
    )
    expect(merged.paperIds).toBeUndefined()
    expect(merged.hypothesisIds).toBeUndefined()
  })
})

describe('repointPaperModelIds', () => {
  const paper = (id: string, modelIds: string[]) => ({ id, modelIds })

  it('replaces duplicate ids with the canonical id, only returning changed papers', () => {
    const papers = [
      paper('p1', ['inverted-transformer-ppo']),
      paper('p2', ['something-else']),
      paper('p3', ['itransformer-ppo', 'inverted-transformer-ppo']),
    ]
    const changed = M.repointPaperModelIds(papers, ['inverted-transformer-ppo'], 'itransformer-ppo')
    // p2 is untouched (not returned); p1 repointed; p3 de-duped to a single canonical id.
    expect(changed.map((p: any) => p.id)).toEqual(['p1', 'p3'])
    expect(changed[0].modelIds).toEqual(['itransformer-ppo'])
    expect(changed[1].modelIds).toEqual(['itransformer-ppo'])
  })

  it('handles papers with no modelIds and an empty fromIds set', () => {
    expect(M.repointPaperModelIds([{ id: 'p' }], ['a'], 'b')).toEqual([])
    expect(M.repointPaperModelIds([paper('p', ['a'])], [], 'b')).toEqual([])
  })
})
