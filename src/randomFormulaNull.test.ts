import { describe, it, expect } from 'vitest'
import {
  signalToWeights,
  formulaSharpe,
  nullPvalue,
  randomFormula,
  randomFormulaNull,
} from './randomFormulaNull.js'

function mulberry32(seed: number): () => number {
  let s = seed
  return () => {
    s |= 0
    s = (s + 0x6d2b79f5) | 0
    let t = Math.imul(s ^ (s >>> 15), 1 | s)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// signalToWeights / formulaSharpe / nullPvalue are golden-pinned to BlackSwan/trainer/random_formula_null.py on
// the same fixed inputs (no RNG in these deterministic scorers).
const SIG = [
  [0.0, 0.0, 0.0],
  [1.0, -1.0, 0.0],
  [0.0, 1.0, -1.0],
  [-1.0, 0.0, 1.0],
  [2.0, -1.0, -1.0],
  [0.5, 0.5, -1.0],
]
const RET = [
  [0.01, -0.01, 0.0],
  [0.02, 0.0, -0.01],
  [-0.01, 0.01, 0.02],
  [0.0, 0.03, -0.02],
  [0.01, 0.01, 0.01],
  [-0.02, 0.0, 0.02],
]

describe('randomFormulaNull deterministic scorers (Python-pinned)', () => {
  it('signalToWeights is dollar-neutral, unit-gross, matches Python', () => {
    expect(
      signalToWeights([
        [1, 2, 3],
        [-1, 0, 1],
      ]),
    ).toEqual([
      [-0.5, 0, 0.5],
      [-0.5, 0, 0.5],
    ])
  })

  it('formulaSharpe (fee 20 bps) matches Python golden', () => {
    expect(formulaSharpe(SIG, RET, 0.002, 252)).toBeCloseTo(-2.716916557618524, 10)
  })

  it('formulaSharpe (fee 0) matches Python golden (~0)', () => {
    expect(formulaSharpe(SIG, RET, 0.0, 252)).toBeCloseTo(0, 9)
  })

  it('nullPvalue continuity-corrected empirical p-value', () => {
    expect(nullPvalue(0.3, [-1, -0.5, 0, 0.5, 1])).toBeCloseTo(0.5, 12)
    expect(nullPvalue(100, [0, 0, 0, 0, 0, 0, 0, 0, 0])).toBeCloseTo(0.1, 12)
  })
})

describe('randomFormula generation (behaviour, JS-seeded)', () => {
  const feats = [
    [
      [0.1, -0.2, 0.3],
      [-0.1, 0.2, -0.3],
      [0.05, 0.0, -0.05],
      [0.2, -0.1, -0.1],
    ],
  ]

  it('is deterministic given the same seeded rng', () => {
    const a = randomFormula(mulberry32(7), feats, 3)
    const b = randomFormula(mulberry32(7), feats, 3)
    expect(a).toEqual(b)
    expect(a.length).toBe(4)
    expect(a[0].length).toBe(3)
    expect(a.every((row) => row.every((v) => Number.isFinite(v)))).toBe(true)
  })

  it('varies with the seed', () => {
    const a = randomFormula(mulberry32(7), feats, 3)
    const c = randomFormula(mulberry32(9), feats, 3)
    expect(a).not.toEqual(c)
  })

  it('randomFormulaNull returns k finite sharpes, deterministic per seed', () => {
    const r = [
      [0.01, -0.01, 0.0],
      [0.0, 0.01, -0.01],
      [-0.01, 0.0, 0.01],
      [0.01, 0.0, -0.01],
    ]
    const n1 = randomFormulaNull(mulberry32(3), feats, r, 2, 20, 0.0005)
    const n2 = randomFormulaNull(mulberry32(3), feats, r, 2, 20, 0.0005)
    expect(n1).toEqual(n2)
    expect(n1.length).toBe(20)
    expect(n1.every((v) => Number.isFinite(v))).toBe(true)
  })
})
