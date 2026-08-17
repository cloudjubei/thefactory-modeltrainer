import { describe, it, expect } from 'vitest'
import { turnoverSeries, portfolioGrossReturns, netReturnSeries } from './tradingCosts.js'

// Golden values generated from BlackSwan/trainer/trading_costs.py on the same fixed inputs.
const W = [
  [0.2, 0.8],
  [0.5, -0.5],
  [0.5, -0.5],
]
const R = [
  [0.01, -0.02],
  [0.03, 0.04],
  [-0.01, 0.02],
]

describe('tradingCosts (Python-pinned)', () => {
  it('turnoverSeries matches Python (entry from cash, then abs weight change)', () => {
    expect(turnoverSeries(W)).toEqual([1.0, 1.6, 0.0])
  })

  it('portfolioGrossReturns matches Python dot product', () => {
    const g = portfolioGrossReturns(W, R)
    ;[-0.014, -0.005, -0.015].forEach((v, i) => expect(g[i]).toBeCloseTo(v, 12))
  })

  it('netReturnSeries subtracts fee*turnover (10 bps)', () => {
    const net = netReturnSeries(W, R, 0.001)
    ;[-0.015, -0.0066, -0.015].forEach((v, i) => expect(net[i]).toBeCloseTo(v, 12))
  })

  it('net equals gross when fee is zero', () => {
    expect(netReturnSeries(W, R, 0)).toEqual(portfolioGrossReturns(W, R))
  })

  it('shape mismatch throws', () => {
    expect(() => portfolioGrossReturns([[0, 0]], [[0, 0, 0]])).toThrow()
  })

  it('empty is empty', () => {
    expect(turnoverSeries([])).toEqual([])
    expect(netReturnSeries([], [], 0.001)).toEqual([])
  })
})
