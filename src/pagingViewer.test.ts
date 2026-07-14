import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/paging.js is a no-build browser module; load it as CommonJS the same way explorationViewer.test.ts
// loads viewer/exploration.js, so the ACTUAL page-accumulation logic (the fix for the silently-truncated
// "only 5500 of 20k runs" scan) is tested here.
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'paging.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const Paging: any = mod.exports

// A page source that hands out fixed-size pages from a flat list of {key}, so a short/empty final page marks
// the end. `failAt` maps an offset → number of times fetching THAT offset should throw before succeeding.
function pager(total: number, pageSize: number, failAt: Record<number, number> = {}) {
  const remaining: Record<number, number> = { ...failAt }
  const calls: number[] = []
  const fetchPage = async (offset: number) => {
    calls.push(offset)
    if (remaining[offset] > 0) {
      remaining[offset]--
      throw new Error('transient bridge timeout at ' + offset)
    }
    const page: Array<{ key: string }> = []
    for (let i = offset; i < Math.min(offset + pageSize, total); i++) page.push({ key: 'k' + i })
    return page
  }
  return { fetchPage, calls }
}

// A sleep that resolves immediately but records the requested delays, so tests never actually wait.
function fakeSleep() {
  const delays: number[] = []
  const sleep = async (ms: number) => {
    delays.push(ms)
  }
  return { sleep, delays }
}

describe('Paging.accumulatePages', () => {
  it('accumulates EVERY page and stops on the short final page', async () => {
    const { fetchPage, calls } = pager(1250, 500)
    const out = await Paging.accumulatePages({ fetchPage, pageSize: 500 })
    expect(out.length).toBe(1250)
    expect(out.map((r: any) => r.key)).toContain('k0')
    expect(out.map((r: any) => r.key)).toContain('k1249')
    expect(calls).toEqual([0, 500, 1000]) // third page (250 < 500) ends it, no wasted 4th fetch
  })

  it('stops on an EMPTY page when the total is an exact multiple of the page size', async () => {
    const { fetchPage, calls } = pager(1000, 500)
    const out = await Paging.accumulatePages({ fetchPage, pageSize: 500 })
    expect(out.length).toBe(1000)
    expect(calls).toEqual([0, 500, 1000]) // the empty page at 1000 is the terminator
  })

  it('dedups by key across pages (a row re-observed under a shifted window is not double-counted)', async () => {
    let n = 0
    const fetchPage = async () => {
      n++
      if (n === 1) return [{ key: 'a' }, { key: 'b' }]
      if (n === 2) return [{ key: 'b' }, { key: 'c' }] // 'b' repeats
      return []
    }
    const out = await Paging.accumulatePages({ fetchPage, pageSize: 2 })
    expect(out.map((r: any) => r.key).sort()).toEqual(['a', 'b', 'c'])
  })

  it('RETRIES a transient page failure and still returns the COMPLETE set (no silent truncation)', async () => {
    const { sleep, delays } = fakeSleep()
    const { fetchPage } = pager(1250, 500, { 500: 2 }) // the 2nd page fails twice, then succeeds
    const out = await Paging.accumulatePages({ fetchPage, pageSize: 500, retries: 4, sleep })
    expect(out.length).toBe(1250) // the fix: a mid-scan blip must NOT end the scan early
    expect(delays.length).toBe(2) // slept between the two failed attempts
  })

  it('RE-THROWS after exhausting retries rather than returning a truncated set', async () => {
    const { sleep } = fakeSleep()
    const { fetchPage } = pager(1250, 500, { 500: 99 }) // the 2nd page never recovers
    await expect(
      Paging.accumulatePages({ fetchPage, pageSize: 500, retries: 3, sleep }),
    ).rejects.toThrow(/transient bridge timeout at 500/)
  })

  it('reports cumulative UNIQUE progress after each page', async () => {
    const seen: number[] = []
    const { fetchPage } = pager(1250, 500)
    await Paging.accumulatePages({
      fetchPage,
      pageSize: 500,
      onProgress: (n: number) => seen.push(n),
    })
    expect(seen).toEqual([500, 1000, 1250])
  })
})

describe('Paging.withRetries', () => {
  it('returns on first success without sleeping', async () => {
    const { sleep, delays } = fakeSleep()
    const out = await Paging.withRetries(async () => 42, { retries: 3, sleep })
    expect(out).toBe(42)
    expect(delays.length).toBe(0)
  })

  it('retries a throwing fn then succeeds', async () => {
    const { sleep, delays } = fakeSleep()
    let n = 0
    const out = await Paging.withRetries(
      async () => {
        n++
        if (n < 3) throw new Error('boom')
        return 'ok'
      },
      { retries: 4, sleep },
    )
    expect(out).toBe('ok')
    expect(delays.length).toBe(2)
  })

  it('re-throws the last error after exhausting retries', async () => {
    const { sleep } = fakeSleep()
    await expect(
      Paging.withRetries(
        async () => {
          throw new Error('always')
        },
        { retries: 2, sleep },
      ),
    ).rejects.toThrow(/always/)
  })
})

describe('Paging.backoffMs', () => {
  it('grows with the attempt and never exceeds the cap', () => {
    const a0 = Paging.backoffMs(0)
    const a1 = Paging.backoffMs(1)
    const a2 = Paging.backoffMs(2)
    expect(a1).toBeGreaterThanOrEqual(a0)
    expect(a2).toBeGreaterThanOrEqual(a1)
    for (let i = 0; i < 20; i++) expect(Paging.backoffMs(i)).toBeLessThanOrEqual(5000)
  })
})
