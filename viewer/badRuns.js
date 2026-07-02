// Pure "bad run" DEFINITION logic — which runs the Runs table's Hide-bad toggle drops. Badness is not one
// rule but a configurable LIST of criteria the user edits (like the custom numeric filters): a run is bad
// if it FAILED, or is health-flagged (degenerate — zero/few trades, degenerate policy, NaN metrics), or it
// UNDER-TRADED (finite n_trades ≤ a threshold). Each criterion can be turned off; the threshold is settable.
// The definition is persisted per project (a `<recordType>-bad-run-def` record) so it survives reloads.
// Pure + dual-loaded (browser `window.BadRuns` + node `module.exports`) so the ACTUAL viewer logic is
// unit-tested directly (see src/badRunsViewer.test.ts), matching datasets.js / bundleTable.js.
;(function (root) {
  'use strict'

  // summary.py's DEGENERATE_TRADE_COUNT — a run trading ≤ this many times isn't a real test of the setup.
  var DEFAULT_MIN_TRADES = 2

  function defaultBadRunDefinition() {
    return { failed: true, degenerate: true, minTrades: DEFAULT_MIN_TRADES }
  }

  // Coerce a persisted/partial definition to a full one: booleans truthy-coerced, minTrades to a finite
  // number or null (null ⇒ the under-trade criterion is off), missing fields filled from the default.
  function normalizeBadRunDefinition(def) {
    var d = def || {}
    var mt = d.minTrades
    var minTrades =
      mt === null || mt === undefined || mt === ''
        ? null
        : Number.isFinite(Number(mt))
          ? Number(mt)
          : null
    return {
      failed: d.failed === undefined ? true : !!d.failed,
      degenerate: d.degenerate === undefined ? true : !!d.degenerate,
      minTrades:
        def && Object.prototype.hasOwnProperty.call(def, 'minTrades')
          ? minTrades
          : DEFAULT_MIN_TRADES,
    }
  }

  // A run is health-flagged (degenerate) when summary.health.status is present and not 'ok'.
  function isDegenerate(run) {
    var h = run && run.summary && run.summary.health
    return !!(h && h.status && h.status !== 'ok')
  }

  // Does `run` match the (normalized) bad-run definition — i.e. should Hide-bad drop it?
  function isBadRun(def, run) {
    var d = normalizeBadRunDefinition(def)
    var s = (run && run.summary) || {}
    if (d.failed && s.status === 'failed') return true
    if (d.degenerate && isDegenerate(run)) return true
    if (d.minTrades !== null) {
      var n = s.metrics && Number(s.metrics.n_trades)
      if (Number.isFinite(n) && n <= d.minTrades) return true
    }
    return false
  }

  // The SERVER-side negation of isBadRun — a DataStorage `where` predicate that KEEPS the good runs, so the
  // paged Runs query stays full instead of being thinned client-side. Only the enabled criteria contribute;
  // no criteria ⇒ undefined (no filter). Each criterion is EXISTS-GUARDED so a run that simply LACKS the
  // field survives: under Postgres three-valued logic a bare `not(<=)` on a missing field is NOT(NULL)=NULL,
  // which a WHERE drops — the opposite of isBadRun, which keeps a run whose field is absent/non-numeric. So
  // the degenerate branch keeps absent / '' / 'ok' health.status (only a non-empty non-'ok' status is bad),
  // and the under-trade branch ORs a not-exists so an absent n_trades is kept.
  function badRunWhere(def) {
    var d = normalizeBadRunDefinition(def)
    var preds = []
    if (d.failed) preds.push({ not: { field: 'status', op: '=', value: 'failed' } })
    if (d.degenerate)
      preds.push({
        or: [
          { not: { field: 'health.status', op: 'exists' } },
          { field: 'health.status', op: '=', value: '' },
          { field: 'health.status', op: '=', value: 'ok' },
        ],
      })
    if (d.minTrades !== null)
      preds.push({
        or: [
          { not: { field: 'metrics.n_trades', op: 'exists' } },
          { not: { field: 'metrics.n_trades', op: '<=', value: d.minTrades } },
        ],
      })
    return preds.length ? { and: preds } : undefined
  }

  var api = {
    DEFAULT_MIN_TRADES: DEFAULT_MIN_TRADES,
    defaultBadRunDefinition: defaultBadRunDefinition,
    normalizeBadRunDefinition: normalizeBadRunDefinition,
    isBadRun: isBadRun,
    badRunWhere: badRunWhere,
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = api
  root.BadRuns = api
})(typeof globalThis !== 'undefined' ? globalThis : this)
