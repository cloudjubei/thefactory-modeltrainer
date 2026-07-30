// Pure ACTIVITY-HISTORY presentation logic — reduce the raw `listActivities` result to the SETTLED
// (completed/failed/aborted) activities of the current project, newest-finished first, as display rows with a
// duration + a status class. The live/queued/paused activities render in the Activity tab; this is the
// browsable HISTORY the popup shows. Pure + dual-loaded (browser `window.ActivityHistory` + node
// `module.exports`) so the ACTUAL logic is unit-tested directly (see src/activityHistoryViewer.test.ts).
;(function (root) {
  'use strict'

  var TERMINAL = { completed: 1, failed: 1, aborted: 1 }

  function labelOf(a) {
    var l = a && a.resumeToken && a.resumeToken.params && a.resumeToken.params._label
    return (typeof l === 'string' && l) || (a && a.activityType) || 'activity'
  }

  function endMs(a) {
    var t = a && (a.finishedAt || a.startedAt)
    var v = t ? Date.parse(t) : NaN
    return isFinite(v) ? v : 0
  }

  function durationMsOf(a) {
    var s = a && a.startedAt ? Date.parse(a.startedAt) : NaN
    var f = a && a.finishedAt ? Date.parse(a.finishedAt) : NaN
    return isFinite(s) && isFinite(f) && f >= s ? f - s : null
  }

  // The settled activities for `recordType` (all projects when omitted), newest-finished first, as rows.
  function historyRows(activities, recordType) {
    var list = Array.isArray(activities) ? activities : []
    var out = []
    for (var i = 0; i < list.length; i++) {
      var a = list[i]
      if (!a || !TERMINAL[a.status]) continue
      if (recordType && a.recordType !== recordType) continue
      out.push({
        activityId: a.activityId,
        type: a.activityType,
        label: labelOf(a),
        status: a.status,
        startedAt: a.startedAt || null,
        finishedAt: a.finishedAt || null,
        durationMs: durationMsOf(a),
        error: a.error || null,
        costUSD: typeof a.costUSD === 'number' ? a.costUSD : null,
        recordType: a.recordType,
      })
    }
    out.sort(function (x, y) {
      return endMs(y) - endMs(x)
    })
    return out
  }

  function formatDuration(ms) {
    if (ms == null || !isFinite(ms) || ms < 0) return '—'
    var s = Math.round(ms / 1000)
    if (s < 60) return s + 's'
    var m = Math.floor(s / 60)
    if (m < 60) return m + 'm' + (s % 60 ? ' ' + (s % 60) + 's' : '')
    var h = Math.floor(m / 60)
    return h + 'h' + (m % 60 ? ' ' + (m % 60) + 'm' : '')
  }

  function statusClass(status) {
    return status === 'completed' ? 'is-ok' : status === 'failed' ? 'is-bad' : 'is-warn'
  }

  var api = {
    historyRows: historyRows,
    formatDuration: formatDuration,
    statusClass: statusClass,
    labelOf: labelOf,
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = api
  root.ActivityHistory = api
})(typeof globalThis !== 'undefined' ? globalThis : this)
