import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/migrate.js is the no-build browser module that runs in the sandboxed hub viewer; it can't be
// `import`ed (the repo is type:module, the file is an IIFE that attaches to `window`/`module.exports`).
// Load it as CommonJS the same way scripts/xaiParityCheck.mjs loads viewer/xai.js, so the ACTUAL code
// that runs in the viewer is what's tested here — no parallel TS copy to drift from.
const require = createRequire(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'migrate.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const Migrate: any = mod.exports

describe('hold-fee migration', () => {
  it('recomputes a fee-free hold benchmark net of the round-trip fee and flags it', () => {
    const summary = {
      config: { transaction_fee: 0.001 },
      metrics: { hold_return_pct: 100, total_return_pct: 50 },
    }
    const patch = Migrate.holdFeeMetricsPatch(summary)
    const expectedHold = (2 * (1 - 0.001) ** 2 - 1) * 100 // 99.6002
    expect(patch.hold_return_pct).toBeCloseTo(expectedHold, 4)
    expect(patch.return_vs_hold_pct).toBeCloseTo(50 - expectedHold, 4)
    expect(patch.hold_net_of_fees).toBe(true)
  })

  it('defaults the fee to the env default (0.001) when the run carries no transaction_fee', () => {
    const patch = Migrate.holdFeeMetricsPatch({
      config: {},
      metrics: { hold_return_pct: 0, total_return_pct: 0 },
    })
    expect(patch.hold_net_of_fees).toBe(true)
    expect(patch.hold_return_pct).toBeCloseTo(((1 - 0.001) ** 2 - 1) * 100, 4) // -0.1999
  })

  it('is idempotent: returns null once a run is already net of fees', () => {
    expect(
      Migrate.holdFeeMetricsPatch({
        config: {},
        metrics: { hold_return_pct: 100, total_return_pct: 50, hold_net_of_fees: true },
      }),
    ).toBeNull()
  })

  it('returns null when there is no benchmark to adjust', () => {
    expect(
      Migrate.holdFeeMetricsPatch({ config: {}, metrics: { total_return_pct: 50 } }),
    ).toBeNull()
    expect(Migrate.holdFeeMetricsPatch({})).toBeNull()
  })

  it('migrationPatchFor returns the hold-fee metrics fix under a `metrics` key (no config change)', () => {
    const patch = Migrate.migrationPatchFor({
      config: { fidelity_set: '1h+1d' },
      metrics: { hold_return_pct: 100, total_return_pct: 50 },
    })
    expect(patch.config).toBeUndefined()
    expect(patch.metrics.hold_net_of_fees).toBe(true)
  })

  it('migrationPatchFor is null for a fully-migrated run', () => {
    expect(
      Migrate.migrationPatchFor({
        config: {},
        metrics: { hold_return_pct: 100, total_return_pct: 50, hold_net_of_fees: true },
      }),
    ).toBeNull()
  })
})

describe('autoFidelity', () => {
  it('mirrors fidelity.py: hourly step observes 1h+1d, any other step observes its own bar', () => {
    expect(Migrate.autoFidelity('1h')).toBe('1h+1d')
    expect(Migrate.autoFidelity('1d')).toBe('1d')
    expect(Migrate.autoFidelity(undefined)).toBe('1d')
  })
})

describe('hypothesisFromLegacyModel', () => {
  it('maps a model to a hypothesis whose spec.fixed pins the architecture levers', () => {
    const h = Migrate.hypothesisFromLegacyModel(
      {
        id: 'm1',
        name: 'Reppo-custom',
        match: { model_name: 'reppo-custom' },
        algo: 'RecurrentPPO',
        netArch: '512,64',
        rationale: 'recurrent core',
        status: 'untested',
      },
      'abc123',
      'T',
    )
    expect(h).toMatchObject({
      id: 'abc123',
      title: 'Reppo-custom',
      spec: { fixed: { model_name: 'reppo-custom' } },
      status: 'untested',
      verdictSource: 'auto',
      source: 'migrated-model',
      createdAt: 'T',
      updatedAt: 'T',
    })
    expect(h.rationale).toContain('recurrent core')
    expect(h.rationale).toContain('512,64')
  })
  it('preserves a human proven/disproved verdict as a MANUAL override', () => {
    const h = Migrate.hypothesisFromLegacyModel(
      { name: 'X', match: {}, status: 'proven' },
      'id',
      'T',
    )
    expect(h).toMatchObject({ status: 'proven', verdictSource: 'manual' })
  })
  it('keeps a research-sourced model as research; defaults an unknown status to untested/auto', () => {
    const h = Migrate.hypothesisFromLegacyModel(
      { name: 'X', source: 'research', status: 'weird' },
      'id',
      'T',
    )
    expect(h).toMatchObject({ source: 'research', status: 'untested', verdictSource: 'auto' })
    expect(h.spec).toEqual({ fixed: {} })
  })
})

describe('legacyHypothesisPatch', () => {
  it('migrates a pending hypothesis to untested/auto', () => {
    expect(Migrate.legacyHypothesisPatch({ status: 'pending' }, 'T')).toEqual({
      verdictSource: 'auto',
      status: 'untested',
      updatedAt: 'T',
    })
  })
  it('migrates accepted to untested/auto', () => {
    expect(Migrate.legacyHypothesisPatch({ status: 'accepted' }, 'T')).toMatchObject({
      status: 'untested',
      verdictSource: 'auto',
    })
  })
  it('migrates rejected to a dismissed untested card', () => {
    expect(Migrate.legacyHypothesisPatch({ status: 'rejected' }, 'T')).toEqual({
      verdictSource: 'auto',
      status: 'untested',
      dismissed: true,
      updatedAt: 'T',
    })
  })
  it('preserves an already-valid verdict status', () => {
    expect(Migrate.legacyHypothesisPatch({ status: 'proven' }, 'T')).toMatchObject({
      status: 'proven',
      verdictSource: 'auto',
    })
  })
  it('returns null for an already-migrated record', () => {
    expect(
      Migrate.legacyHypothesisPatch({ status: 'untested', verdictSource: 'auto' }, 'T'),
    ).toBeNull()
    expect(
      Migrate.legacyHypothesisPatch({ status: 'proven', verdictSource: 'manual' }, 'T'),
    ).toBeNull()
  })
})

describe('pipelineVersion normalization (bare integer → major.minor)', () => {
  it('normalizes a bare-integer string version to N.0', () => {
    expect(Migrate.pipelineVersionPatch({ pipelineVersion: '4' })).toEqual({
      pipelineVersion: '4.0',
    })
    expect(Migrate.pipelineVersionPatch({ pipelineVersion: '3' })).toEqual({
      pipelineVersion: '3.0',
    })
  })
  it('normalizes a numeric version to N.0', () => {
    expect(Migrate.pipelineVersionPatch({ pipelineVersion: 4 })).toEqual({ pipelineVersion: '4.0' })
  })
  it('is idempotent for an already major.minor version', () => {
    expect(Migrate.pipelineVersionPatch({ pipelineVersion: '4.0' })).toBeNull()
    expect(Migrate.pipelineVersionPatch({ pipelineVersion: '4.2' })).toBeNull()
  })
  it('leaves a missing or non-integer version alone', () => {
    expect(Migrate.pipelineVersionPatch({})).toBeNull()
    expect(Migrate.pipelineVersionPatch({ pipelineVersion: '' })).toBeNull()
    expect(Migrate.pipelineVersionPatch({ pipelineVersion: 'legacy:foo' })).toBeNull()
  })
  it('migrationPatchFor merges the version fix alongside the hold-fee fix', () => {
    const patch = Migrate.migrationPatchFor({
      pipelineVersion: '4',
      metrics: { hold_return_pct: 100, total_return_pct: 50 },
    })
    expect(patch.pipelineVersion).toBe('4.0')
    expect(patch.metrics.hold_net_of_fees).toBe(true)
  })
  it('migrationPatchFor returns the version fix alone when metrics are already net of fees', () => {
    const patch = Migrate.migrationPatchFor({
      pipelineVersion: '4',
      metrics: { hold_return_pct: 100, total_return_pct: 50, hold_net_of_fees: true },
    })
    expect(patch).toEqual({ pipelineVersion: '4.0' })
  })
})
