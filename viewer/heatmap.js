// Pure HEATMAP presentation logic for a 2-D matrix attribution (e.g. a rollout-aggregated attention map).
// DOM-free: buildHeatmapModel(grid, opts) turns a number[][] into a positioned, colour-mapped cell model
// the thin app.js renderer emits as SVG. The colour scale is SEQUENTIAL for an all-nonnegative matrix and
// DIVERGING (centered at 0) when any value is negative; a degenerate min===max maps every cell to the
// neutral midpoint (never NaN); an oversized grid is bucketed (mean-pooled) to a per-axis cap so the SVG
// stays bounded. Labels are carried RAW — the renderer escapes them. Dual-loaded (browser `window.Heatmap`
// + node `module.exports`) so the ACTUAL logic is unit-tested directly (see src/heatmapViewer.test.ts).
;(function (root) {
  'use strict'

  var SEQ = [
    [240, 245, 255],
    [30, 58, 138],
  ]
  var DIV = [
    [33, 102, 172],
    [247, 247, 247],
    [178, 24, 43],
  ]

  function clamp01(t) {
    return t < 0 ? 0 : t > 1 ? 1 : t
  }

  function hex2(n) {
    var v = Math.max(0, Math.min(255, Math.round(n)))
    var s = v.toString(16)
    return s.length === 1 ? '0' + s : s
  }

  function mix(a, b, t) {
    return '#' + hex2(a[0] + (b[0] - a[0]) * t) + hex2(a[1] + (b[1] - a[1]) * t) + hex2(a[2] + (b[2] - a[2]) * t)
  }

  function rampColor(t, diverging) {
    var u = clamp01(t)
    if (!diverging) return mix(SEQ[0], SEQ[1], u)
    return u < 0.5 ? mix(DIV[0], DIV[1], u / 0.5) : mix(DIV[1], DIV[2], (u - 0.5) / 0.5)
  }

  // Mean-pool `grid` (a rectangular number[][]) so neither axis exceeds `maxDim`. Returns the pooled grid
  // plus the bucket sizes so labels can collapse to their first member.
  function bucket(grid, maxDim) {
    var R = grid.length
    var C = grid[0].length
    var rStride = Math.ceil(R / maxDim)
    var cStride = Math.ceil(C / maxDim)
    if (rStride <= 1 && cStride <= 1) return { grid: grid, rStride: 1, cStride: 1 }
    var outR = Math.ceil(R / rStride)
    var outC = Math.ceil(C / cStride)
    var out = []
    for (var i = 0; i < outR; i++) {
      var row = []
      for (var j = 0; j < outC; j++) {
        var sum = 0
        var n = 0
        for (var r = i * rStride; r < Math.min(R, (i + 1) * rStride); r++) {
          for (var c = j * cStride; c < Math.min(C, (j + 1) * cStride); c++) {
            sum += grid[r][c]
            n++
          }
        }
        row.push(n ? sum / n : 0)
      }
      out.push(row)
    }
    return { grid: out, rStride: rStride, cStride: cStride }
  }

  function labelsFor(supplied, count, stride) {
    var out = []
    var ok = Array.isArray(supplied) && supplied.length === count * stride
    for (var i = 0; i < count; i++) {
      out.push(ok ? String(supplied[i * stride]) : String(i))
    }
    return out
  }

  // Build the cell model. `opts`: { rowLabels, colLabels, cell, gutterLeft, gutterTop, maxDim }.
  function buildHeatmapModel(grid, opts) {
    opts = opts || {}
    var cell = opts.cell || 16
    var gutterLeft = opts.gutterLeft == null ? 44 : opts.gutterLeft
    var gutterTop = opts.gutterTop == null ? 16 : opts.gutterTop
    var maxDim = opts.maxDim || 64
    var origR = Array.isArray(grid) ? grid.length : 0
    var origC = origR && Array.isArray(grid[0]) ? grid[0].length : 0
    var total = origR * origC
    if (!origR || !origC) {
      return { cells: [], width: gutterLeft, height: gutterTop, rowLabels: [], colLabels: [], legend: { min: 0, max: 0, diverging: false }, shown: 0, total: 0 }
    }
    var pooled = bucket(grid, maxDim)
    var g = pooled.grid
    var R = g.length
    var C = g[0].length

    var min = Infinity
    var max = -Infinity
    for (var r = 0; r < R; r++) {
      for (var c = 0; c < C; c++) {
        var v = g[r][c]
        if (v < min) min = v
        if (v > max) max = v
      }
    }
    var diverging = min < 0
    var maxAbs = Math.max(Math.abs(min), Math.abs(max))
    var flat = min === max

    function tOf(v) {
      if (flat) return 0.5
      return diverging ? clamp01(0.5 + 0.5 * (v / maxAbs)) : clamp01((v - min) / (max - min))
    }

    var rowLabels = labelsFor(opts.rowLabels, R, pooled.rStride)
    var colLabels = labelsFor(opts.colLabels, C, pooled.cStride)

    var cells = []
    for (var i = 0; i < R; i++) {
      for (var j = 0; j < C; j++) {
        var value = g[i][j]
        var t = tOf(value)
        cells.push({
          r: i,
          c: j,
          x: gutterLeft + j * cell,
          y: gutterTop + i * cell,
          w: cell,
          h: cell,
          value: value,
          t: t,
          fill: rampColor(t, diverging),
          title: rowLabels[i] + ' → ' + colLabels[j] + ': ' + value,
        })
      }
    }
    return {
      cells: cells,
      width: gutterLeft + C * cell,
      height: gutterTop + R * cell,
      rowLabels: rowLabels.map(function (text, i) {
        return { y: gutterTop + i * cell + cell / 2, text: text }
      }),
      colLabels: colLabels.map(function (text, j) {
        return { x: gutterLeft + j * cell + cell / 2, text: text }
      }),
      legend: { min: min, max: max, diverging: diverging, mid: diverging ? 0 : (min + max) / 2 },
      shown: R * C,
      total: total,
    }
  }

  var api = { buildHeatmapModel: buildHeatmapModel, rampColor: rampColor }
  if (typeof module !== 'undefined' && module.exports) module.exports = api
  root.Heatmap = api
})(typeof globalThis !== 'undefined' ? globalThis : this)
