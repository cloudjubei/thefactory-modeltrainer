// Pure CROSS-TEST presentation logic — how a run's `{recordType}-settest` matrix (checkpoint replayed on
// other assets / an extended window, per-lever result maps) reduces to a robustness verdict, display
// rows, and the still-missing assets. The matrix itself is produced by the `cross-test` activity; this
// module only interprets it. Pure + dual-loaded (browser `window.CrossTest` + node `module.exports`) so
// the ACTUAL viewer logic is unit-tested directly (see src/crossTestViewer.test.ts).
;(function (root) {
  'use strict'

  function allCells(record) {
    const levers = (record && record.levers) || {}
    const out = []
    for (const lever of Object.keys(levers)) {
      const byValue = levers[lever] || {}
      for (const value of Object.keys(byValue)) {
        out.push({ lever: lever, cell: byValue[value] })
      }
    }
    return out
  }

  // Pooled robustness over every completed cell (all levers): 'robust' = every cell beats hold,
  // 'weak' = none do, 'mixed' otherwise, 'none' = nothing completed yet.
  function crossTestVerdict(record) {
    const cells = allCells(record)
    const completed = cells.filter((c) => c.cell && c.cell.status === 'completed')
    const failed = cells.length - completed.length
    const positive = completed.filter((c) => Number(c.cell.returnVsHold) > 0).length
    if (!completed.length) return { state: 'none', positive: 0, tested: 0, failed: failed }
    const state = positive === completed.length ? 'robust' : positive === 0 ? 'weak' : 'mixed'
    return { state: state, positive: positive, tested: completed.length, failed: failed }
  }

  // Flat, ordered display rows for the run-detail matrix.
  function crossTestRows(record) {
    return allCells(record).map((c) => ({
      lever: c.lever,
      value: c.cell.value,
      status: c.cell.status,
      objective: c.cell.objective,
      returnVsHold: c.cell.returnVsHold,
      error: c.cell.error,
      evaluatedAt: c.cell.evaluatedAt,
    }))
  }

  // The asset-lever values NOT yet cross-tested for a run: choices minus the trained asset minus the
  // already-tested cells. Mirrors the server's missingCrossTestValues so the button label is honest.
  function missingAssets(manifest, run, record) {
    const spec = manifest && manifest.levers && manifest.levers.asset
    const choices = (spec && spec.choices) || []
    const trained = String(
      ((run && run.summary && run.summary.config) || {}).asset === undefined
        ? ''
        : run.summary.config.asset,
    )
    const tested = new Set(Object.keys(((record && record.levers) || {}).asset || {}))
    return choices.map(String).filter((v) => v !== trained && !tested.has(v))
  }

  // The open-ended (test-through-latest) variant of a fixed walk-forward window, or null when the
  // window is already open-ended / unrecognised. '2024' -> 'oos-2024', 'alt-2025' -> 'alt-oos-2025'.
  function extendedWindowFor(windowId) {
    const id = String(windowId == null ? '' : windowId)
    if (id.indexOf('oos-') !== -1) return null
    if (/^\d{4}$/.test(id)) return 'oos-' + id
    const alt = /^alt-(\d{4})$/.exec(id)
    if (alt) return 'alt-oos-' + alt[1]
    return null
  }

  // The compact chip for the runs-table Robustness column.
  function verdictChip(verdict) {
    if (verdict.state === 'none') {
      return { label: verdict.failed ? verdict.failed + ' failed' : '—', cls: 'is-none' }
    }
    return {
      label: verdict.positive + '/' + verdict.tested + ' beat hold',
      cls: 'is-' + verdict.state,
    }
  }

  const api = {
    crossTestVerdict: crossTestVerdict,
    crossTestRows: crossTestRows,
    missingAssets: missingAssets,
    extendedWindowFor: extendedWindowFor,
    verdictChip: verdictChip,
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = api
  root.CrossTest = api
})(typeof globalThis !== 'undefined' ? globalThis : this)
