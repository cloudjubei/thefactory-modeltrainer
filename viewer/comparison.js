// Pure logic behind the "By dataset" / "By environment" views. An AXIS is the levers whose scope IS the axis
// ('dataset' | 'environment'); a run's axis SIGNATURE is those levers' values. Two consumers share this:
//   • the Runs tab POOLS every filtered run and groups by axis signature (which dataset/environment wins);
//   • the xAI current-run tabs hold the non-axis levers to one config (sameSetupExceptAxis) and group by axis
//     (how that one config holds up across the axis).
// groupComparison reduces each group's per-column values to { min, avg, max }; robustnessVerdict classifies a
// config's per-environment standings. Pure + dual-loaded (browser `window.Comparison` + node
// `module.exports`) so the ACTUAL viewer logic is unit-tested directly (see src/comparisonViewer.test.ts).
;(function (root) {
  'use strict'

  function leverEntries(manifest) {
    return Object.entries((manifest && manifest.levers) || {})
  }
  function leverScope(spec) {
    return (spec && spec.scope) || 'model'
  }
  // A SCOPE axis groups by every lever in a scope ('dataset' | 'environment'); anything else is a SINGLE
  // lever's key (the "By value" view — one chosen lever varies, everything else is held fixed).
  function isScopeAxis(axis) {
    return axis === 'dataset' || axis === 'environment'
  }
  // `seed` is a nuisance parameter pooled OVER, never a lever we lock or vary (matches setupKeyOfRun, which
  // drops seed from a setup's identity). Real manifests often leave it unscoped (→ 'model'), so exclude it
  // by NAME here rather than trusting the scope — otherwise the across-axis views pin the focus run's seed.
  function isNuisanceLever(key, spec) {
    return key === 'seed' || leverScope(spec) === 'ignore'
  }

  // The levers that VARY down the rows for an axis. Scope axis ⇒ every lever whose scope IS the axis; a
  // single-lever axis ⇒ just that lever (empty when it isn't a real, non-nuisance lever).
  function axisLeverKeys(manifest, axis) {
    return leverEntries(manifest)
      .filter(function (e) {
        if (isScopeAxis(axis)) return leverScope(e[1]) === axis
        return e[0] === axis && !isNuisanceLever(e[0], e[1])
      })
      .map(function (e) {
        return e[0]
      })
  }

  // The levers held FIXED for an axis — every non-nuisance lever that isn't itself the axis. Locking these is
  // what makes the comparison apples-to-apples (same setup, only the axis varies).
  function lockedLeverKeys(manifest, axis) {
    return leverEntries(manifest)
      .filter(function (e) {
        if (isNuisanceLever(e[0], e[1])) return false
        return isScopeAxis(axis) ? leverScope(e[1]) !== axis : e[0] !== axis
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

  // Canonical value for lever equality: unset / null / '' all collapse to the same 'n/a' bucket, so a lever
  // that is absent on one side and present on the other never counts as "equal", and two runs that both leave
  // it unset (or pinned 'n/a' by conditional normalisation) do.
  function canonLever(v) {
    return v === undefined || v === null || v === '' ? 'n/a' : String(v)
  }

  // Whether a run has the SAME setup as the focus config on every LOCKED lever for an axis — i.e. it differs
  // ONLY in the axis lever(s) (seed is always pooled, never locked). This is exhaustive over the locked
  // levers (not just the ones the focus happens to set), so a run that sets a lever the focus left unset is
  // correctly EXCLUDED — a By value / By dataset row is exactly one config's seeds, nothing bleeds in.
  function sameSetupExceptAxis(manifest, axis, focusCfg, runCfg) {
    var f = focusCfg || {}
    var r = runCfg || {}
    var keys = lockedLeverKeys(manifest, axis)
    for (var i = 0; i < keys.length; i++) {
      var k = keys[i]
      if (canonLever(f[k]) !== canonLever(r[k])) return false
    }
    return true
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
      if (sortKey === 'axis') {
        // A By value lever axis labels rows by the lever VALUE — sort those numerically (128 < 512 < 4096),
        // not lexically; a non-numeric label (dataset/env name, comma-list net_arch) stays a text sort.
        var label = String(g.axisLabel == null ? '' : g.axisLabel)
        var n = Number(label)
        return label !== '' && Number.isFinite(n) ? n : label.toLowerCase()
      }
      if (sortKey === '#runs') return g.count
      // 'standing' is a per-group robust-z the caller attaches (not a min/avg/max stat), so read it directly.
      if (sortKey === 'standing') return typeof g.standing === 'number' ? g.standing : NaN
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

  // Classify a config's ROBUSTNESS from its per-environment standings (robust z from
  // Xai.normalizeByEnvironment — how far it stands above/below each environment's TYPICAL config, higher
  // better). 'robust' = never below typical (min >= 0); 'weak' = never above typical (max <= 0); 'mixed' =
  // strong in some, below typical in others. Needs >= 2 environments to mean anything, else 'n/a'.
  function robustnessVerdict(standings) {
    var vals = (standings || []).filter(function (v) {
      return typeof v === 'number' && isFinite(v)
    })
    if (vals.length < 2) return { label: 'n/a', n: vals.length, min: NaN, max: NaN }
    var min = Math.min.apply(null, vals)
    var max = Math.max.apply(null, vals)
    var label = min >= 0 ? 'robust' : max <= 0 ? 'weak' : 'mixed'
    return { label: label, n: vals.length, min: min, max: max }
  }

  // Build a launch spec that PINS the non-axis levers to one config and SWEEPS an axis across supplied
  // values — the "fill the axis for this config" sweep. `fixed` = every locked lever the focus config sets
  // (skipping 'n/a'/blank); `sweep` = each axis lever mapped to its values (levers with no values dropped).
  // Returns null when nothing on the axis can be swept, so the caller can message instead of firing an empty
  // campaign. Pure — the seed strategy + first-time-only (refresh) behaviour is the caller's concern.
  function axisSweepSpec(manifest, axis, focusConfig, valuesByLever) {
    var cfg = focusConfig || {}
    var vals = valuesByLever || {}
    var fixed = {}
    var lockKeys = lockedLeverKeys(manifest, axis)
    for (var i = 0; i < lockKeys.length; i++) {
      var lk = lockKeys[i]
      var v = cfg[lk]
      if (v === undefined || v === null || v === '' || String(v) === 'n/a') continue
      fixed[lk] = v
    }
    var sweep = {}
    var axisKeys = axisLeverKeys(manifest, axis)
    for (var j = 0; j < axisKeys.length; j++) {
      var ak = axisKeys[j]
      var list = vals[ak]
      if (Array.isArray(list) && list.length) sweep[ak] = list
    }
    return Object.keys(sweep).length ? { fixed: fixed, sweep: sweep } : null
  }

  var api = {
    axisLeverKeys: axisLeverKeys,
    axisSweepSpec: axisSweepSpec,
    robustnessVerdict: robustnessVerdict,
    lockedLeverKeys: lockedLeverKeys,
    runAxisSignature: runAxisSignature,
    sameSetupExceptAxis: sameSetupExceptAxis,
    groupComparison: groupComparison,
    sortComparisonGroups: sortComparisonGroups,
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = api
  root.Comparison = api
})(typeof globalThis !== 'undefined' ? globalThis : this)
