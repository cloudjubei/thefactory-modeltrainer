import { describe, expect, it } from 'vitest'
import {
  isRunAffectedByFidelityDesync,
  isRunAffectedByFidelityLookahead,
  isSpecAffectedByFidelityLookahead,
} from './modelTrainerUtils.js'

// ────────────────────────────────────────────────────────────────────────────────────────────────────
// AUTHOR A3 — exhaustive TS predicate + cross-consistency regression for the v6 fidelity LOOK-AHEAD bug.
//
// The BlackSwan multi-timeline provider leaked FUTURE price whenever an hourly/minute decision step
// observed ONLY coarser layers (neither the run cadence nor the base granularity is among the observed
// layers). `isRunAffectedByFidelityLookahead` is the pure TS mirror of the provider's own guard condition
// (`fidelity_run ∉ layers AND fidelity_input ∉ layers`) used to invalidate the tainted runs.
//
// This file is an INDEPENDENT re-derivation of that predicate straight from the ground-truth fidelity
// resolution in trainer/fidelity.py (resolve_fidelity + _LAYER_SETS + the 1m<1h<1d<1w cadence order).
// The oracle below is written from that spec, NOT copied from the implementation, so any drift between
// the predicate and the provider surfaces here as a failure. RED on the buggy tree: pre-fix the predicate
// did not exist / conflated look-ahead with the broad desync notion, so the truth-table true-rows and the
// strict-subset asymmetry (1h+1d is desync-affected yet look-ahead-SAFE) would fail.
// ────────────────────────────────────────────────────────────────────────────────────────────────────

// Coarsest-to-finest cadence rank — ground truth from trainer/fidelity.py (1m < 1h < 1d < 1w).
const RANK: Record<string, number> = { '1m': 0, '1h': 1, '1d': 2, '1w': 3 }

// fidelity_set label -> observed layer stack — ground truth from trainer/fidelity.py `_LAYER_SETS`.
const LAYER_SETS: Record<string, string[]> = {
  '1m': ['1m'],
  '1m+1h': ['1m', '1h'],
  '1m+1h+1d': ['1m', '1h', '1d'],
  '1d': ['1d'],
  '1h': ['1h'],
  '1h+1d': ['1h', '1d'],
  '1h+1d+1w': ['1h', '1d', '1w'],
  '1d+1w': ['1d', '1w'],
}

// The 8 canonical layer-set ids plus the resolved (`auto`), defaulted (`undefined`) and bogus inputs.
const FIDELITY_SET_IDS = [
  '1m',
  '1m+1h',
  '1m+1h+1d',
  '1d',
  '1h',
  '1h+1d',
  '1h+1d+1w',
  '1d+1w',
] as const
const FIDELITY_SET_INPUTS: Array<string | undefined> = [...FIDELITY_SET_IDS, 'auto', undefined, 'bogus']
const TIMEFRAMES = ['1h', '1d', '1m'] as const

/**
 * The OBSERVED layer stack for a (timeframe, fidelity_set), re-derived from trainer/fidelity.py
 * `resolve_fidelity`: `auto`/absent follows the step (1h -> 1h+1d, 1m -> 1m+1h, else 1d); a named set maps
 * through `_LAYER_SETS`; anything unknown observes nothing (empty).
 */
function resolveObservedLayers(timeframe: string, fidelitySet: string | undefined): string[] {
  const isAuto = fidelitySet === undefined || fidelitySet === null || fidelitySet === '' || fidelitySet === 'auto'
  if (isAuto) {
    if (timeframe === '1h') return ['1h', '1d']
    if (timeframe === '1m') return ['1m', '1h']
    return ['1d']
  }
  return LAYER_SETS[fidelitySet] ?? []
}

/**
 * The ORACLE: affected iff the step is a known cadence AND at least one layer is observed AND EVERY
 * observed layer is strictly COARSER than the step (so neither the run cadence nor the base is observed —
 * exactly the provider's `fidelity_run ∉ layers AND fidelity_input ∉ layers`).
 */
function oracleAffected(timeframe: string, fidelitySet: string | undefined): boolean {
  const stepRank = RANK[timeframe]
  if (stepRank === undefined) return false
  const layers = resolveObservedLayers(timeframe, fidelitySet)
  if (layers.length === 0) return false
  return layers.every((layer) => RANK[layer] !== undefined && RANK[layer] > stepRank)
}

// The family that leaks pre-fix (step strictly finer than EVERY observed layer), hardcoded to ANCHOR the
// oracle itself to the ground-truth audit so a mistaken oracle can't silently rubber-stamp a mistaken
// predicate. Keyed `${timeframe}@${fidelity_set}`.
const EXPECTED_AFFECTED = new Set([
  '1h@1d',
  '1h@1d+1w',
  '1m@1d',
  '1m@1h',
  '1m@1h+1d',
  '1m@1h+1d+1w',
  '1m@1d+1w',
])

describe('fidelity look-ahead oracle is anchored to the ground-truth audit', () => {
  // Meta-check: the independently-written oracle must reproduce the hand-audited affected family EXACTLY,
  // so the cross-consistency assertions below rest on a trustworthy reference. Pins invariant I2 (no
  // look-ahead) at the config layer.
  it('reproduces exactly the hand-audited affected (timeframe@fidelity_set) family', () => {
    const oracleAffectedKeys = new Set<string>()
    for (const timeframe of TIMEFRAMES) {
      for (const fidelitySet of FIDELITY_SET_IDS) {
        if (oracleAffected(timeframe, fidelitySet)) oracleAffectedKeys.add(`${timeframe}@${fidelitySet}`)
      }
    }
    expect([...oracleAffectedKeys].sort()).toEqual([...EXPECTED_AFFECTED].sort())
  })

  it('never flags any daily step (1d observes its own run cadence)', () => {
    for (const fidelitySet of FIDELITY_SET_INPUTS) {
      expect(oracleAffected('1d', fidelitySet)).toBe(false)
    }
  })
})

describe('isRunAffectedByFidelityLookahead — FULL truth table (timeframe × fidelity_set)', () => {
  // Every (timeframe in {1h,1d,1m}) × (8 layer-set ids + auto + undefined + bogus) with the required
  // boolean, generated from the ground-truth oracle. RED pre-fix: the true-rows (1h@1d, 1h@1d+1w and the
  // five 1m coarse-only rows) would not be flagged.
  const cases: Array<[string, string | undefined, boolean]> = []
  for (const timeframe of TIMEFRAMES) {
    for (const fidelitySet of FIDELITY_SET_INPUTS) {
      cases.push([timeframe, fidelitySet, oracleAffected(timeframe, fidelitySet)])
    }
  }

  it.each(cases)(
    'timeframe=%s fidelity_set=%s -> affected=%s',
    (timeframe, fidelitySet, expected) => {
      const config: Record<string, unknown> = { timeframe }
      if (fidelitySet !== undefined) config.fidelity_set = fidelitySet
      expect(isRunAffectedByFidelityLookahead(config)).toBe(expected)
    },
  )

  it('covers the full 3 × 11 grid', () => {
    expect(cases).toHaveLength(TIMEFRAMES.length * FIDELITY_SET_INPUTS.length)
    // At least one affected and one safe row exist, so a predicate stuck at a constant fails the grid.
    expect(cases.some(([, , affected]) => affected)).toBe(true)
    expect(cases.some(([, , affected]) => !affected)).toBe(true)
  })
})

describe('isRunAffectedByFidelityLookahead — cross-consistency with resolve_fidelity (no drift)', () => {
  // The predicate must equal "every observed layer strictly coarser than the step" for EVERY combo, so the
  // TS predicate and the python provider can never drift apart. Enforces invariant I2 at the config layer.
  for (const timeframe of TIMEFRAMES) {
    for (const fidelitySet of FIDELITY_SET_INPUTS) {
      const layers = resolveObservedLayers(timeframe, fidelitySet)
      const label = fidelitySet === undefined ? '(default)' : fidelitySet
      it(`timeframe=${timeframe} fidelity_set=${label} matches [${layers.join(',')}]`, () => {
        const config: Record<string, unknown> = { timeframe }
        if (fidelitySet !== undefined) config.fidelity_set = fidelitySet
        expect(isRunAffectedByFidelityLookahead(config)).toBe(oracleAffected(timeframe, fidelitySet))
      })
    }
  }
})

describe('isRunAffectedByFidelityLookahead ⊂ isRunAffectedByFidelityDesync (strict subset)', () => {
  // Every look-ahead-affected combo is ALSO desync-affected (the look-ahead family is a subset of the
  // broad multi-timeline family) — but NOT vice-versa. Guards against conflating the narrow v6 predicate
  // with the broad v5 one.
  for (const timeframe of TIMEFRAMES) {
    for (const fidelitySet of FIDELITY_SET_INPUTS) {
      if (!oracleAffected(timeframe, fidelitySet)) continue
      const label = fidelitySet === undefined ? '(default)' : fidelitySet
      it(`look-ahead ${timeframe}@${label} is also desync-affected`, () => {
        const config: Record<string, unknown> = { timeframe }
        if (fidelitySet !== undefined) config.fidelity_set = fidelitySet
        expect(isRunAffectedByFidelityLookahead(config)).toBe(true)
        expect(isRunAffectedByFidelityDesync(config)).toBe(true)
      })
    }
  }

  it('1h+1d is desync-affected yet look-ahead-SAFE (subset is strict, not equal)', () => {
    const config = { timeframe: '1h', fidelity_set: '1h+1d' }
    expect(isRunAffectedByFidelityDesync(config)).toBe(true)
    expect(isRunAffectedByFidelityLookahead(config)).toBe(false)
  })

  it('1h+1d+1w is desync-affected yet look-ahead-SAFE (1h base observed)', () => {
    const config = { timeframe: '1h', fidelity_set: '1h+1d+1w' }
    expect(isRunAffectedByFidelityDesync(config)).toBe(true)
    expect(isRunAffectedByFidelityLookahead(config)).toBe(false)
  })

  it('the two families are NOT equal across the grid (at least one desync-only combo)', () => {
    let desyncOnly = 0
    for (const timeframe of TIMEFRAMES) {
      for (const fidelitySet of FIDELITY_SET_INPUTS) {
        const config: Record<string, unknown> = { timeframe }
        if (fidelitySet !== undefined) config.fidelity_set = fidelitySet
        const look = isRunAffectedByFidelityLookahead(config)
        const desync = isRunAffectedByFidelityDesync(config)
        if (look) expect(desync).toBe(true)
        if (desync && !look) desyncOnly++
      }
    }
    expect(desyncOnly).toBeGreaterThan(0)
  })
})

describe('isRunAffectedByFidelityLookahead — the two ACTUALLY audited runs', () => {
  // The concrete configs the runs-audit flagged (+765% / +835% traded_return in a -65% bear window). Both
  // MUST be flagged so invalidation catches them. Pins invariant I2 for the shipped tainted runs.
  it('flags supervised-logreg 1h@1d (pipeline 5.0)', () => {
    expect(
      isRunAffectedByFidelityLookahead({
        timeframe: '1h',
        fidelity_set: '1d',
        model_name: 'supervised-logreg',
        pipelineVersion: '5.0',
      }),
    ).toBe(true)
  })

  it('flags supervised-gbm 1h@1d (pipeline 5.0)', () => {
    expect(
      isRunAffectedByFidelityLookahead({
        timeframe: '1h',
        fidelity_set: '1d',
        model_name: 'supervised-gbm',
        pipelineVersion: '5.0',
      }),
    ).toBe(true)
  })
})

describe('isSpecAffectedByFidelityLookahead — pending-spec sweep coverage', () => {
  it('flags a fixed-only spec whose base is coarse-only (1h@1d)', () => {
    expect(isSpecAffectedByFidelityLookahead({ fixed: { timeframe: '1h', fidelity_set: '1d' } })).toBe(true)
  })

  it('does not flag a fixed-only spec whose base observes the run cadence (1h@1h+1d)', () => {
    expect(
      isSpecAffectedByFidelityLookahead({ fixed: { timeframe: '1h', fidelity_set: '1h+1d' } }),
    ).toBe(false)
  })

  it('flags a spec that SWEEPS fidelity_set into a coarse-only value even when fixed is safe', () => {
    expect(
      isSpecAffectedByFidelityLookahead({
        fixed: { timeframe: '1h', fidelity_set: '1h+1d' },
        sweep: { fidelity_set: ['1h+1d', '1d'] },
      }),
    ).toBe(true)
  })

  it('flags a spec that SWEEPS timeframe from a safe daily step into an affected hourly one', () => {
    // fixed fidelity_set 1d; at 1d that observes its run cadence (safe), but sweeping timeframe to 1h makes
    // 1h@1d (coarse-only) — affected. Mirrors the run-level truth table under the sweep projection.
    expect(
      isSpecAffectedByFidelityLookahead({
        fixed: { fidelity_set: '1d' },
        sweep: { timeframe: ['1d', '1h'] },
      }),
    ).toBe(true)
  })

  it('does not flag a sweep whose every projected combination is safe', () => {
    expect(
      isSpecAffectedByFidelityLookahead({
        fixed: { timeframe: '1h' },
        sweep: { fidelity_set: ['1h', '1h+1d', '1h+1d+1w'] },
      }),
    ).toBe(false)
  })

  it('does not flag a purely daily-step sweep (no hourly/minute step to leak)', () => {
    expect(
      isSpecAffectedByFidelityLookahead({
        fixed: { timeframe: '1d' },
        sweep: { fidelity_set: ['1d', '1d+1w'] },
      }),
    ).toBe(false)
  })
})

describe('fidelity look-ahead predicates — guards (never invalidate the unknown)', () => {
  it('isRunAffectedByFidelityLookahead is false for null/undefined', () => {
    expect(isRunAffectedByFidelityLookahead(undefined)).toBe(false)
    expect(isRunAffectedByFidelityLookahead(null as unknown as Record<string, unknown>)).toBe(false)
  })

  it('isRunAffectedByFidelityLookahead is false for an unknown fidelity_set', () => {
    expect(isRunAffectedByFidelityLookahead({ timeframe: '1h', fidelity_set: 'bogus' })).toBe(false)
  })

  it('isRunAffectedByFidelityLookahead is false for an unknown timeframe (no known step cadence)', () => {
    expect(isRunAffectedByFidelityLookahead({ timeframe: '5m', fidelity_set: '1d' })).toBe(false)
  })

  it('isSpecAffectedByFidelityLookahead is false for null/undefined/empty', () => {
    expect(isSpecAffectedByFidelityLookahead(undefined)).toBe(false)
    expect(isSpecAffectedByFidelityLookahead(null as unknown as Record<string, unknown>)).toBe(false)
    expect(isSpecAffectedByFidelityLookahead({})).toBe(false)
  })
})
