// Pure logic behind the Runs tab's "By dataset" / "By environment" COMPARISON views. The point of these
// views is to compare the SAME setup across the varying axis: to see how one model holds up across datasets
// (or across environments), every OTHER lever must be pinned, else you'd be reading incomparable setups as
// one number. So an axis (the levers with scope === the axis) VARIES down the rows, while the LOCKED levers
// (every other lever except the ignored ones, e.g. seed) are held fixed via the toolbar dropdowns. Within a
// row (one axis value, all locks satisfied) the remaining runs are the seeds; they aggregate to min/avg/max.
// Pure + dual-loaded (browser `window.Comparison` + node `module.exports`) so the ACTUAL viewer logic is
// unit-tested directly (see src/comparisonViewer.test.ts), matching datasets.js / bundleTable.js / badRuns.js.
;(function (root) {
  'use strict'

  function leverEntries(manifest) {
    return Object.entries((manifest && manifest.levers) || {})
  }
  function leverScope(spec) {
    return (spec && spec.scope) || 'model'
  }

  // The levers that VARY down the rows for an axis — those whose scope IS the axis ('dataset' | 'environment').
  function axisLeverKeys(manifest, axis) {
    return leverEntries(manifest)
      .filter(function (e) {
        return leverScope(e[1]) === axis
      })
      .map(function (e) {
        return e[0]
      })
  }

  // The levers held FIXED for an axis — every lever that is neither the axis nor explicitly ignored (seed).
  // Locking these is what makes the across-axis comparison apples-to-apples.
  function lockedLeverKeys(manifest, axis) {
    return leverEntries(manifest)
      .filter(function (e) {
        var scope = leverScope(e[1])
        return scope !== axis && scope !== 'ignore'
      })
      .map(function (e) {
        return e[0]
      })
  }

  // Canonical signature of the axis a run sits on (its axis-lever values off summary.config), matching the
  // `key=value · key=value` shape datasets.js / env signatures use so a numeric 2024 and a choice '2024' tie.
  function runAxisSignature(manifest, axis, run) {
    var cfg = (run && run.summary && run.summary.config) || {}
    return axisLeverKeys(manifest, axis)
      .map(function (key) {
        return key + '=' + (cfg[key] === undefined ? '' : String(cfg[key]))
      })
      .join(' · ')
  }

  // Does a run satisfy every locked lever value (string-coerced)? An empty lock matches all runs.
  function matchesLock(lock, run) {
    var cfg = (run && run.summary && run.summary.config) || {}
    var keys = Object.keys(lock || {})
    for (var i = 0; i < keys.length; i++) {
      var k = keys[i]
      var want = lock[k]
      if (want === '' || want === undefined || want === null) continue
      if (String(cfg[k]) !== String(want)) return false
    }
    return true
  }

  // Numeric-aware ascending compare (both numeric ⇒ numeric order; else case-insensitive string order).
  function compareValues(a, b) {
    var an = typeof a === 'number' ? a : Number(typeof a === 'string' ? a.trim() : a)
    var bn = typeof b === 'number' ? b : Number(typeof b === 'string' ? b.trim() : b)
    if (Number.isFinite(an) && Number.isFinite(bn)) return an - bn
    return String(a).toLowerCase().localeCompare(String(b).toLowerCase())
  }

  // Per locked lever, the distinct values actually PRESENT across the runs (string-coerced, numeric-aware
  // sorted) — the options each lock dropdown offers, so numeric levers (no manifest `choices`) work too.
  function distinctLockValues(manifest, axis, runs) {
    var out = {}
    var keys = lockedLeverKeys(manifest, axis)
    for (var i = 0; i < keys.length; i++) {
      var key = keys[i]
      var seen = new Set()
      for (var j = 0; j < (runs || []).length; j++) {
        var cfg = (runs[j] && runs[j].summary && runs[j].summary.config) || {}
        var v = cfg[key]
        if (v !== undefined && v !== null && v !== '') seen.add(String(v))
      }
      out[key] = Array.from(seen).sort(compareValues)
    }
    return out
  }

  // The locked-lever values of the BEST run (by objective, honouring the direction) — the initial lock, so
  // the view opens comparing today's champion setup across the axis. Falls back to the first run; {} if none.
  function bestRunLock(manifest, axis, runs, direction) {
    var list = (runs || []).filter(function (r) {
      return r && r.summary
    })
    if (!list.length) return {}
    var withObj = list.filter(function (r) {
      return Number.isFinite(Number(r.summary.objective))
    })
    var best
    if (withObj.length) {
      best = withObj.reduce(function (acc, r) {
        var v = Number(r.summary.objective)
        var av = Number(acc.summary.objective)
        if (direction === 'min') return v < av ? r : acc
        return v > av ? r : acc
      })
    } else {
      best = list[0]
    }
    var cfg = best.summary.config || {}
    var lock = {}
    lockedLeverKeys(manifest, axis).forEach(function (key) {
      if (cfg[key] !== undefined && cfg[key] !== null) lock[key] = String(cfg[key])
    })
    return lock
  }

  function mean(nums) {
    if (!nums.length) return NaN
    var s = 0
    for (var i = 0; i < nums.length; i++) s += nums[i]
    return s / nums.length
  }

  // Group pre-extracted items by their axis signature into comparison rows. An item is
  // { key, axisSig, axisLabel, values: { <colId>: number } }; each column reduces its FINITE values to
  // { min, avg, max } (NaN all-round when a column has no finite value in the group). `count` = seeds/runs.
  function groupComparison(items) {
    var map = new Map()
    var order = []
    for (var i = 0; i < (items || []).length; i++) {
      var it = items[i]
      if (!map.has(it.axisSig)) {
        map.set(it.axisSig, { axisSig: it.axisSig, axisLabel: it.axisLabel, keys: [], _cols: {} })
        order.push(it.axisSig)
      }
      var g = map.get(it.axisSig)
      g.keys.push(it.key)
      var vals = it.values || {}
      for (var col in vals) {
        if (!Object.prototype.hasOwnProperty.call(vals, col)) continue
        var v = Number(vals[col])
        if (!g._cols[col]) g._cols[col] = []
        if (Number.isFinite(v)) g._cols[col].push(v)
      }
    }
    return order.map(function (sig) {
      var g = map.get(sig)
      var stats = {}
      for (var col in g._cols) {
        if (!Object.prototype.hasOwnProperty.call(g._cols, col)) continue
        var arr = g._cols[col]
        stats[col] = arr.length
          ? { min: Math.min.apply(null, arr), avg: mean(arr), max: Math.max.apply(null, arr) }
          : { min: NaN, avg: NaN, max: NaN }
      }
      return {
        axisSig: g.axisSig,
        axisLabel: g.axisLabel,
        keys: g.keys,
        count: g.keys.length,
        stats: stats,
      }
    })
  }

  // Sort comparison rows the way the Runs table sorts. sortKey: 'axis' (by label), '#runs' (by count), or a
  // column id (by that column's AVG). Missing/NaN always sort last. Never mutates the input.
  function sortComparisonGroups(groups, sortKey, sortDir) {
    var mul = sortDir === 'asc' ? 1 : -1
    var list = Array.isArray(groups) ? groups.slice() : []
    function valueOf(g) {
      if (sortKey === 'axis') return String(g.axisLabel == null ? '' : g.axisLabel).toLowerCase()
      if (sortKey === '#runs') return g.count
      var s = g.stats && g.stats[sortKey]
      return s ? s.avg : NaN
    }
    return list.sort(function (a, b) {
      var av = valueOf(a)
      var bv = valueOf(b)
      var aMissing =
        av === undefined ||
        av === null ||
        av === '' ||
        (typeof av === 'number' && !Number.isFinite(av))
      var bMissing =
        bv === undefined ||
        bv === null ||
        bv === '' ||
        (typeof bv === 'number' && !Number.isFinite(bv))
      if (aMissing && bMissing) return 0
      if (aMissing) return 1
      if (bMissing) return -1
      if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * mul
      return String(av).localeCompare(String(bv)) * mul
    })
  }

  var api = {
    axisLeverKeys: axisLeverKeys,
    lockedLeverKeys: lockedLeverKeys,
    runAxisSignature: runAxisSignature,
    matchesLock: matchesLock,
    distinctLockValues: distinctLockValues,
    bestRunLock: bestRunLock,
    groupComparison: groupComparison,
    sortComparisonGroups: sortComparisonGroups,
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = api
  root.Comparison = api
})(typeof globalThis !== 'undefined' ? globalThis : this)
