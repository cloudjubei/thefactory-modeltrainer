// Sorts the Datasets / Environments management tables (named lever bundles) the same numeric-aware way the
// Runs table sorts, so a user can click a column header and compare values out of the box. A bundle row is
// { id, name, default, values: { <lever>: value } }; sortKey is 'name', 'default', or a lever key.
// Pure + dual-loaded (browser window + Node testing), matching hypothesis.js / models.js / xai.js.
;(function (root) {
  'use strict'

  // Two cells compared the way the Runs table compares: both numeric (a number or a numeric string) sort
  // numerically; anything else by case-insensitive string order; a missing cell (undefined / null / '')
  // always sorts LAST regardless of direction so blanks never crowd the top.
  function compareCells(a, b, dir) {
    var mul = dir === 'asc' ? 1 : -1
    var aMissing = a === undefined || a === null || a === ''
    var bMissing = b === undefined || b === null || b === ''
    if (aMissing && bMissing) return 0
    if (aMissing) return 1
    if (bMissing) return -1
    var an = typeof a === 'number' ? a : Number(typeof a === 'string' ? a.trim() : a)
    var bn = typeof b === 'number' ? b : Number(typeof b === 'string' ? b.trim() : b)
    if (Number.isFinite(an) && Number.isFinite(bn)) return (an - bn) * mul
    return String(a).localeCompare(String(b)) * mul
  }

  // The sortable value for a row under a given key. 'name' is lower-cased (case-insensitive order);
  // 'default' becomes 1/0 so descending floats the default row to the top; any other key reads a lever.
  function cellValue(row, sortKey) {
    if (!row) return undefined
    if (sortKey === 'name') return String(row.name == null ? '' : row.name).toLowerCase()
    if (sortKey === 'default') return row.default ? 1 : 0
    return (row.values || {})[sortKey]
  }

  // A stable sort (ties keep input order) over a copy — never mutates the caller's array.
  function sortRows(rows, sortKey, sortDir) {
    var list = Array.isArray(rows) ? rows.slice() : []
    var dir = sortDir === 'asc' ? 'asc' : 'desc'
    return list.sort(function (x, y) {
      return compareCells(cellValue(x, sortKey), cellValue(y, sortKey), dir)
    })
  }

  // Header-click transition: re-clicking the active column flips direction; a new column starts at its
  // default direction (ascending unless the caller passes 'desc', e.g. for "best score first" columns).
  function nextSort(curKey, curDir, clickedKey, defaultDir) {
    if (curKey === clickedKey) {
      return { key: clickedKey, dir: curDir === 'asc' ? 'desc' : 'asc' }
    }
    return { key: clickedKey, dir: defaultDir === 'desc' ? 'desc' : 'asc' }
  }

  var BundleTable = {
    compareCells: compareCells,
    cellValue: cellValue,
    sortRows: sortRows,
    nextSort: nextSort,
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = BundleTable
  if (root) root.BundleTable = BundleTable
})(typeof window !== 'undefined' ? window : null)
