// Pure DATA-CATALOG presentation logic — how to summarise an instrument's on-disk coverage + status and
// build a mine (download) request. The catalog itself (asset classes → instruments, each with `onDisk`
// coverage) is produced by the project's `data-catalog` activity and read from the `{recordType}-datacatalog`
// record; this module only turns it into the compact status the Data tab shows and the request the Download
// button sends. Pure + dual-loaded (browser `window.DataCatalog` + node `module.exports`) so the ACTUAL
// viewer logic is unit-tested directly (see src/dataViewer.test.ts).
;(function (root) {
  'use strict'

  // The interval whose coverage best represents an instrument's status: prefer 1d (what stocks/commodities/
  // fx carry, and crypto's coarse view), else the finest interval actually on disk, else the first declared.
  function primaryInterval(inst) {
    const onDisk = (inst && inst.onDisk) || {}
    if (onDisk['1d']) return '1d'
    const present = Object.keys(onDisk).sort()
    if (present.length) return present[0]
    return ((inst && inst.intervals) || [])[0] || '1d'
  }

  // 'ready' (on disk, contiguous), 'gaps' (on disk with interior gaps), or 'available' (nothing on disk yet).
  function instrumentState(inst) {
    const onDisk = (inst && inst.onDisk) || {}
    const intervals = Object.keys(onDisk)
    if (!intervals.length) return 'available'
    const anyGaps = intervals.some((tf) => ((onDisk[tf] && onDisk[tf].gaps) || []).length > 0)
    return anyGaps ? 'gaps' : 'ready'
  }

  // A compact human coverage line, e.g. "1d · 2018-01 → 2026-06 · 102 mo" (+ " · N gaps"), or
  // "not downloaded" when nothing is on disk.
  function coverageLine(inst) {
    if (instrumentState(inst) === 'available') return 'not downloaded'
    const tf = primaryInterval(inst)
    const cov = ((inst && inst.onDisk) || {})[tf]
    if (!cov) return 'not downloaded'
    let line = tf + ' · ' + cov.start + ' → ' + cov.end + ' · ' + cov.months + ' mo'
    const gaps = (cov.gaps || []).length
    if (gaps) line += ' · ' + gaps + ' gap' + (gaps === 1 ? '' : 's')
    return line
  }

  // The walk-forward windows this asset's data range can actually run (inventory-driven, emitted per price
  // instrument by the catalog), e.g. "runs: 2024, 2026, alt-2024" — or '' when none (nothing on disk yet,
  // or a release series that isn't walk-forward-traded).
  function supportedWindowsLine(inst) {
    const windows = (inst && inst.supportedWindows) || []
    if (!windows.length) return ''
    return 'runs: ' + windows.join(', ')
  }

  // Every asset SYMBOL with data on disk (any interval), across all classes — deduped. [] when the catalog
  // is absent, so the Datasets cards NEVER dim everything just because no data-catalog run exists yet.
  function assetsWithData(catalog) {
    const classes = (catalog && catalog.assetClasses) || []
    const out = []
    const seen = {}
    for (const cls of classes) {
      for (const inst of (cls && cls.instruments) || []) {
        const sym = inst && inst.symbol
        if (sym && instrumentState(inst) !== 'available' && !seen[sym]) {
          seen[sym] = true
          out.push(sym)
        }
      }
    }
    return out
  }

  // Whether a manifest lever VALUE has no data on disk — true ONLY for the `asset` lever when the catalog is
  // present AND the symbol is absent from `assetsWithData`. Returns false (dim NOTHING) when the catalog is
  // null/empty (no data known) or the lever isn't `asset`, so a project without a data-catalog run is never
  // greyed out wholesale.
  function isLeverValueDataless(catalog, lever, value) {
    if (lever !== 'asset') return false
    const withData = assetsWithData(catalog)
    if (!withData.length) return false
    return withData.indexOf(String(value)) === -1
  }

  // { onDisk, total } for a class — how many instruments have any coverage.
  function classCounts(cls) {
    const instruments = (cls && cls.instruments) || []
    return {
      onDisk: instruments.filter((i) => instrumentState(i) !== 'available').length,
      total: instruments.length,
    }
  }

  // The mine request to download one instrument (all its intervals) or a whole class. The instrument
  // request is class-scoped so a fundamentals ticker resolves distinctly from the same stock ticker.
  function mineRequestForInstrument(inst) {
    return { symbols: [inst.symbol], class: inst.assetClass }
  }
  function mineRequestForClass(cls) {
    return { class: cls.id }
  }

  // The linkage edges FROM one asset — the related driver series a user can mine alongside it.
  function linkageForAsset(edges, symbol, assetClass) {
    return (edges || []).filter(
      (e) => e && e.asset === symbol && (!assetClass || e.assetClass === assetClass),
    )
  }

  const api = {
    primaryInterval: primaryInterval,
    instrumentState: instrumentState,
    coverageLine: coverageLine,
    supportedWindowsLine: supportedWindowsLine,
    assetsWithData: assetsWithData,
    isLeverValueDataless: isLeverValueDataless,
    classCounts: classCounts,
    mineRequestForInstrument: mineRequestForInstrument,
    mineRequestForClass: mineRequestForClass,
    linkageForAsset: linkageForAsset,
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = api
  root.DataCatalog = api
})(typeof globalThis !== 'undefined' ? globalThis : this)
