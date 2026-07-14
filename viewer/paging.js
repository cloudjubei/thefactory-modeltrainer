// Resilient page-accumulation for unbounded record scans (every run of a 20k+ project). A single capped
// query returns ONE page; scanning the whole history means paging until exhausted. The trap this module
// closes: a mid-scan page failure (a bridge timeout under write contention while an exploration hammers the
// DB) must NOT be mistaken for the end of data. The old scan swallowed such a failure into an empty page and
// stopped early, reporting "verdicts over 5500 runs" for a 20k-run project. Here a failed page is RETRIED
// with backoff and, only if it never recovers, the whole scan THROWS — so a caller reports an honest error
// instead of a silently-truncated set. Pure + dual-loaded (browser window + Node testing), matching
// hypothesis.js / models.js / runExport.js.
;(function (root) {
  'use strict'

  // Capped exponential backoff between retry attempts (ms). Deterministic (no jitter) so it stays testable.
  function backoffMs(attempt) {
    return Math.min(5000, 300 * Math.pow(2, attempt))
  }

  // Run `fn` and, if it throws, retry up to `retries` more times, sleeping backoffMs(attempt) between tries.
  // Re-throws the LAST error once the retries are exhausted — a persistent failure is surfaced, never hidden.
  async function withRetries(fn, opts) {
    const retries = opts && Number.isFinite(opts.retries) ? opts.retries : 4
    const sleep = opts && opts.sleep
    let lastErr
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        return await fn(attempt)
      } catch (err) {
        lastErr = err
        if (attempt >= retries) break
        if (sleep) await sleep(backoffMs(attempt))
      }
    }
    throw lastErr
  }

  // Page through EVERY record via `fetchPage(offset) -> Promise<Array<{key}>>`, deduped by key. Each page is
  // fetched through withRetries so a transient failure is retried, not treated as end-of-data. The scan ends
  // ONLY on a genuinely empty or short (< pageSize) page. `onProgress(uniqueCount)` fires after each page.
  async function accumulatePages(opts) {
    const fetchPage = opts.fetchPage
    const pageSize = opts.pageSize
    const retries = Number.isFinite(opts.retries) ? opts.retries : 4
    const sleep = opts.sleep
    const onProgress = opts.onProgress
    const byKey = new Map()
    let offset = 0
    // Guard is a runaway backstop only — pageSize>=1 means it far exceeds any real project's page count.
    for (let guard = 0; guard < 1000000; guard++) {
      const at = offset
      const page = await withRetries(() => fetchPage(at), { retries: retries, sleep: sleep })
      if (!page || !page.length) break
      for (const r of page) if (r && r.key != null) byKey.set(r.key, r)
      if (onProgress) onProgress(byKey.size)
      if (page.length < pageSize) break
      offset += pageSize
    }
    return [...byKey.values()]
  }

  const Paging = {
    backoffMs: backoffMs,
    withRetries: withRetries,
    accumulatePages: accumulatePages,
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = Paging
  if (root) root.Paging = Paging
})(typeof window !== 'undefined' ? window : null)
