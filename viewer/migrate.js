// Run-record normalization for the sandboxed hub viewer. The hub knows one BlackSwan-specific quirk:
// the `fidelity_set` DATASET lever is stored as the synonym "auto" (or, on pre-hub ledger imports, not
// at all), which fragments grouping against runs that name the SAME data explicitly (e.g. "1h+1d"). This
// module derives each run's CONCRETE dataset identity from data it already carries — never from domain
// guesses — so the on-start migration (see app.js migrateRunRecords) can regroup them. Pure + dual-loaded
// (browser `window.Migrate` + node `module.exports`) so the actual viewer code is unit-tested directly.
;(function (root) {
  'use strict'

  // Mirror of trainer/fidelity.py `_LAYER_SETS` (fidelity_set label -> observed layer stack, finest-first).
  // Keep in sync if the project adds runnable layer sets. Sub-hourly stacks are intentionally absent: the
  // current multi-timeline provider can't serve a finer-than-1h base, so they are retired, not runnable.
  var LAYER_SETS = {
    '1d': ['1d'],
    '1h': ['1h'],
    '1h+1d': ['1h', '1d'],
    '1h+1d+1w': ['1h', '1d', '1w'],
    '1d+1w': ['1d', '1w'],
  }

  // Sort key for a timeframe symbol (e.g. "15m", "1h") so layer stacks compare finest-first regardless of
  // the order they were written: minutes < hours < days < weeks, numeric within a unit.
  function timeframeRank(symbol) {
    var match = /^(\d+)([mhdw])$/.exec(String(symbol))
    if (!match) return Number.MAX_SAFE_INTEGER
    var unit = { m: 0, h: 1, d: 2, w: 3 }[match[2]]
    return unit * 100000 + Number(match[1])
  }

  function isSubHourly(symbol) {
    return /^\d+m$/.test(String(symbol))
  }

  // Map a layer stack to a fidelity_set identity. A stack that contains nothing finer than 1h AND matches a
  // runnable LAYER_SETS value gets that canonical label; anything else (a sub-hourly / retired stack) gets a
  // truthful, distinct `legacy:` label so it groups with its own kind and never merges into a runnable set.
  function fidelitySetFromLayers(layers) {
    if (!Array.isArray(layers) || !layers.length) return null
    var seen = {}
    var uniq = []
    for (var i = 0; i < layers.length; i++) {
      var s = String(layers[i])
      if (!seen[s]) {
        seen[s] = true
        uniq.push(s)
      }
    }
    uniq.sort(function (a, b) {
      return timeframeRank(a) - timeframeRank(b)
    })
    var subHourly = uniq.some(isSubHourly)
    if (!subHourly) {
      for (var label in LAYER_SETS) {
        if (!Object.prototype.hasOwnProperty.call(LAYER_SETS, label)) continue
        var set = LAYER_SETS[label]
        if (
          set.length === uniq.length &&
          set.every(function (x, j) {
            return x === uniq[j]
          })
        ) {
          return { fidelitySet: label, layers: uniq, legacy: false }
        }
      }
    }
    return { fidelitySet: 'legacy:' + uniq.join('+'), layers: uniq, legacy: true }
  }

  // The observed layer stack of a pre-hub ledger import, read off its raw `historical_data` tag. Separators
  // vary by era (`|`, `]`, `~` for `.`); the observation/input stacks are the MULTI-symbol (piped) groups,
  // so the union of their symbols is the data the run actually saw. Returns null when no such group exists
  // (a single-layer or unparseable tag) — the caller then leaves the run untouched rather than guess.
  function legacyLayersFromHistoricalData(historicalData) {
    if (typeof historicalData !== 'string' || !historicalData) return null
    var normalized = historicalData.replace(/\]/g, '|').replace(/~/g, '.')
    var tokens = normalized.split('_')
    var groupRe = /^\d+[mhdw](?:\|\d+[mhdw])*$/
    var seen = {}
    var symbols = []
    for (var i = 0; i < tokens.length; i++) {
      var token = tokens[i]
      if (!groupRe.test(token) || token.indexOf('|') < 0) continue
      var parts = token.split('|')
      for (var p = 0; p < parts.length; p++) {
        if (!seen[parts[p]]) {
          seen[parts[p]] = true
          symbols.push(parts[p])
        }
      }
    }
    return symbols.length ? symbols : null
  }

  // The walk-forward TEST window of a ledger import, read off the `...vsYYYY...` span in `historical_data`.
  function walkForwardWindowFromHistoricalData(historicalData) {
    if (typeof historicalData !== 'string') return null
    var match = /vs(\d{4})/.exec(historicalData)
    return match ? match[1] : null
  }

  // The record patch that normalizes ONE run's dataset identity, or null when nothing should change.
  // Three classes, all idempotent (a re-run sees a concrete fidelity_set and returns null):
  //   1. live "auto" run        -> concrete label from its own dataset.layers
  //   2. clean 1h+1d import     -> backfill fidelity_set + walk_forward_window + dataset.layers
  //   3. retired sub-hourly run -> backfill a `legacy:` fidelity_set + dataset.layers (distinct, truthful)
  function migrationPatchFor(summary) {
    var config = (summary && summary.config) || {}
    var current = config.fidelity_set

    if (current === 'auto') {
      var liveLayers = summary && summary.dataset && summary.dataset.layers
      var live = fidelitySetFromLayers(liveLayers)
      if (!live || live.fidelitySet === 'auto' || live.fidelitySet === current) return null
      return { config: { fidelity_set: live.fidelitySet } }
    }

    if (current === undefined || current === null || current === '') {
      var historicalData = config.historical_data
      var layers = legacyLayersFromHistoricalData(historicalData)
      var resolved = fidelitySetFromLayers(layers)
      if (!resolved) return null
      var patch = { config: { fidelity_set: resolved.fidelitySet }, dataset: { layers: resolved.layers } }
      if (config.walk_forward_window === undefined) {
        var window = walkForwardWindowFromHistoricalData(historicalData)
        if (window) patch.config.walk_forward_window = window
      }
      return patch
    }

    return null
  }

  var Migrate = {
    fidelitySetFromLayers: fidelitySetFromLayers,
    legacyLayersFromHistoricalData: legacyLayersFromHistoricalData,
    walkForwardWindowFromHistoricalData: walkForwardWindowFromHistoricalData,
    migrationPatchFor: migrationPatchFor,
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = Migrate
  if (root) root.Migrate = Migrate
})(typeof window !== 'undefined' ? window : null)
