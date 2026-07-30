// Pure CONTINUED-TRAINING (extra-train) presentation logic — the parent<->child lineage and the per-set
// evaluation matrix for runs seeded from a parent checkpoint and trained further on shifted data. A CHILD
// stamps the PARENT's checkpoint path in `summary.provenance.continuedFrom`; the parent is the run whose
// `summary.artifacts.checkpoint` matches. The `continue-training` activity produces the children; this
// module only interprets the loaded run set. Pure + dual-loaded (browser `window.ContinueTrain` + node
// `module.exports`) so the ACTUAL viewer logic is unit-tested directly (see src/continueTrainViewer.test.ts).
;(function (root) {
  'use strict'

  function checkpointOf(run) {
    var a = run && run.summary && run.summary.artifacts
    return (a && a.checkpoint) || ''
  }

  function continuedFromOf(run) {
    var p = run && run.summary && run.summary.provenance
    return (p && p.continuedFrom) || ''
  }

  // Build parent<->child edges from continue-training provenance. Returns { parentOf, childrenOf } maps
  // keyed by run key. First checkpoint wins for a duplicated path; a dangling provenance (parent absent
  // from `runs`) and a self-link are ignored.
  function lineageIndex(runs) {
    var list = Array.isArray(runs) ? runs : []
    var keyByCheckpoint = new Map()
    for (var i = 0; i < list.length; i++) {
      var cp = checkpointOf(list[i])
      if (cp && !keyByCheckpoint.has(cp)) keyByCheckpoint.set(cp, list[i].key)
    }
    var parentOf = new Map()
    var childrenOf = new Map()
    for (var j = 0; j < list.length; j++) {
      var r = list[j]
      var from = continuedFromOf(r)
      if (!from) continue
      var parentKey = keyByCheckpoint.get(from)
      if (!parentKey || parentKey === r.key) continue
      parentOf.set(r.key, parentKey)
      var arr = childrenOf.get(parentKey) || []
      arr.push(r.key)
      childrenOf.set(parentKey, arr)
    }
    return { parentOf: parentOf, childrenOf: childrenOf }
  }

  function parentKeyOf(run, index) {
    return (index && run && index.parentOf.get(run.key)) || ''
  }

  function childKeysOf(run, index) {
    return (index && run && index.childrenOf.get(run.key)) || []
  }

  // The per-set evaluation matrix rows: one continued CHILD per row, labelled by the dataset it was
  // continued onto (its scope:'dataset' lever values), with the child's objective + status. Each child is
  // a full run judged on its own standardised test window — never the shifted train window.
  function continuedMatrixRows(run, index, runsByKey, datasetLeverKeys) {
    var keys = childKeysOf(run, index)
    var leverKeys = Array.isArray(datasetLeverKeys) ? datasetLeverKeys : []
    var rows = []
    for (var i = 0; i < keys.length; i++) {
      var child = runsByKey.get(keys[i])
      if (!child) continue
      var cfg = (child.summary && child.summary.config) || {}
      var parts = []
      for (var k = 0; k < leverKeys.length; k++) {
        var v = cfg[leverKeys[k]]
        if (v !== undefined && v !== null && String(v) !== '') parts.push(String(v))
      }
      rows.push({
        key: child.key,
        label: parts.length ? parts.join(' · ') : '—',
        objective: child.summary ? child.summary.objective : undefined,
        status: (child.summary && child.summary.status) || 'completed',
      })
    }
    return rows
  }

  var api = {
    checkpointOf: checkpointOf,
    continuedFromOf: continuedFromOf,
    lineageIndex: lineageIndex,
    parentKeyOf: parentKeyOf,
    childKeysOf: childKeysOf,
    continuedMatrixRows: continuedMatrixRows,
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = api
  root.ContinueTrain = api
})(typeof globalThis !== 'undefined' ? globalThis : this)
