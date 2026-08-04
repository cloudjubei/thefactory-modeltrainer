import { describe, it, expect } from 'vitest'
import { normalPpf, psrFromStats, expectedMaxSharpe, deflatedSharpeFromStats } from './deflatedSharpe.js'

// The TS Deflated-Sharpe port MUST reproduce BlackSwan/trainer/sharpe.py's psr_from_stats / expected_max_sharpe
// / deflated_sharpe_ratio. Golden values below were generated from that Python reference (scipy); the TS
// normal CDF is Abramowitz-Stegun (~1.5e-7), so the PSR pins are to 6 decimals — plenty for a probability gate.

describe('normalPpf (Acklam inverse normal)', () => {
  it('matches known standard-normal quantiles', () => {
    expect(normalPpf(0.5)).toBeCloseTo(0, 12)
    expect(normalPpf(0.975)).toBeCloseTo(1.959963984540054, 8) // the 1.96 of a 95% CI
    expect(normalPpf(0.99)).toBeCloseTo(2.3263478740408408, 8)
    expect(normalPpf(0.001)).toBeCloseTo(-3.090232306167813, 8)
    expect(normalPpf(0.025)).toBeCloseTo(-1.959963984540054, 8) // symmetry
  })

  it('is monotone and handles the degenerate ends', () => {
    expect(normalPpf(0)).toBe(-Infinity)
    expect(normalPpf(1)).toBe(Infinity)
    expect(normalPpf(0.3)).toBeLessThan(normalPpf(0.7))
  })
})

describe('psrFromStats (TS twin of sharpe.py psr_from_stats)', () => {
  it('matches the Python golden vectors', () => {
    expect(psrFromStats(0.0, 0.0, 3.0, 100)).toBeCloseTo(0.5, 6) // SR = benchmark ⇒ Phi(0)
    expect(psrFromStats(0.1, 0.0, 3.0, 101)).toBeCloseTo(0.8407413278013518, 6) // normal moments
    expect(psrFromStats(0.15, -0.5, 4.0, 200, 0.05)).toBeCloseTo(0.911495153669269, 6)
  })

  it('is 0 when undefined (n < 2 or non-positive denominator)', () => {
    expect(psrFromStats(0.5, 0.0, 3.0, 1)).toBe(0)
    // A large negative skew with a big Sharpe can drive the denominator <= 0 ⇒ undefined ⇒ 0.
    expect(psrFromStats(5.0, 5.0, 3.0, 100)).toBe(0)
  })
})

describe('expectedMaxSharpe (TS twin of sharpe.py expected_max_sharpe)', () => {
  it('matches the Python golden vectors', () => {
    expect(expectedMaxSharpe(100, 0.5)).toBeCloseTo(1.2653014466008423, 7)
    expect(expectedMaxSharpe(1000, 1.0)).toBeCloseTo(3.255121513652723, 7)
  })

  it('is 0 for a single trial (no multiple testing) or no cross-trial spread', () => {
    expect(expectedMaxSharpe(1, 0.5)).toBe(0)
    expect(expectedMaxSharpe(0, 0.5)).toBe(0)
    expect(expectedMaxSharpe(100, 0)).toBe(0)
  })

  it('rises with more trials (a higher bar to clear)', () => {
    expect(expectedMaxSharpe(10, 1)).toBeLessThan(expectedMaxSharpe(100, 1))
  })
})

describe('deflatedSharpeFromStats (TS twin of sharpe.py deflated_sharpe_ratio)', () => {
  it('deflates a modest Sharpe toward 0 under heavy multiple testing', () => {
    // SR 0.2 against a deflation level ~1.27 (100 trials, std 0.5) ⇒ DSR ≈ 0 (Python golden 4.67e-123).
    expect(deflatedSharpeFromStats(0.2, 0.0, 3.0, 500, 100, 0.5)).toBeCloseTo(0, 9)
  })

  it('equals PSR(benchmark 0) when there is a single trial (no deflation)', () => {
    expect(deflatedSharpeFromStats(0.1, 0.0, 3.0, 101, 1, 0.5)).toBeCloseTo(psrFromStats(0.1, 0.0, 3.0, 101), 12)
  })
})
