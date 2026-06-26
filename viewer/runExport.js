// Assembles the compare-mode "Download audit" payload: the FULL summary of each selected run, so the
// genuineness signals (return_vs_hold_pct, top-level regimes + benchmark, health, blocked_signal_ratio,
// and artifacts.decisionTrace) all survive into the JSON. Summaries are passed through VERBATIM rather
// than cherry-picked, so no audit field is ever silently dropped. Pure + dual-loaded (browser window +
// Node testing), matching hypothesis.js / models.js / xai.js.
;(function (root) {
  'use strict'

  function buildRunsAuditExport(runs, meta) {
    const list = Array.isArray(runs) ? runs : []
    const kept = list.filter(
      (r) => r && typeof r === 'object' && r.summary && typeof r.summary === 'object',
    )
    const m = meta || {}
    return {
      schema: 'blackswan-runs-audit/v1',
      exportedAt: m.exportedAt || null,
      objective: m.objective || null,
      project: m.project || null,
      count: kept.length,
      runs: kept.map((r) => ({ key: r.key, summary: r.summary })),
    }
  }

  const RunExport = {
    buildRunsAuditExport: buildRunsAuditExport,
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = RunExport
  if (root) root.RunExport = RunExport
})(typeof window !== 'undefined' ? window : null)
