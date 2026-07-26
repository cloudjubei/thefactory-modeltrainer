// Pure dataset-IDENTITY logic — what makes two "datasets" the same, and which named dataset a run belongs
// to. A dataset is the pinned values of the manifest's scope:'dataset' levers (asset, timeframe, fidelity,
// walk-forward window…). Identity is derived ENTIRELY from those levers, so a lever is part of a dataset's
// identity IFF it carries scope:'dataset' in the manifest the viewer currently holds — a STALE manifest
// snapshot that predates a lever gaining scope:'dataset' silently drops it from identity, collapsing two
// datasets that differ only by it (keep the manifest fresh: see refreshProjectManifest in app.js).
// Pure + dual-loaded (browser `window.Datasets` + node `module.exports`) so the ACTUAL viewer logic is
// unit-tested directly (see src/datasetsViewer.test.ts).
;(function (root) {
  'use strict'

  // The ordered scope:'dataset' lever keys of a manifest — the levers a named dataset pins.
  function datasetLeverKeys(manifest) {
    const levers = (manifest && manifest.levers) || {}
    return Object.keys(levers).filter((k) => levers[k] && levers[k].scope === 'dataset')
  }

  // Canonical signature of a settings map over the dataset levers — the dedup/grouping key. A missing lever
  // is '' (so a partial old dataset never equals a complete new one); every value is String()-coerced so a
  // choice string '2024' and a numeric 2024 (from a run config) compare equal.
  function datasetSettingsSignature(manifest, settings) {
    const s = settings || {}
    return datasetLeverKeys(manifest)
      .map((key) => `${key}=${s[key] === undefined ? '' : String(s[key])}`)
      .join(' · ')
  }

  // The dataset in `datasets` that duplicates `name`/`settings` (same name case-insensitively, OR same
  // settings signature), excluding `exceptId` — or undefined when the dataset is unique.
  function findDuplicateDataset(manifest, datasets, name, settings, exceptId) {
    const sig = datasetSettingsSignature(manifest, settings)
    const lower = String(name == null ? '' : name)
      .trim()
      .toLowerCase()
    return (datasets || []).find(
      (d) =>
        d &&
        d.id !== exceptId &&
        (String(d.name == null ? '' : d.name)
          .trim()
          .toLowerCase() === lower ||
          datasetSettingsSignature(manifest, d.settings) === sig),
    )
  }

  // The signature of the dataset a RUN belongs to (its dataset-lever values, read off summary.config).
  function runDatasetSignature(manifest, run) {
    const cfg = (run && run.summary && run.summary.config) || {}
    return datasetLeverKeys(manifest)
      .map((key) => `${key}=${cfg[key] === undefined ? '' : String(cfg[key])}`)
      .join(' · ')
  }

  // The name of the named dataset a run matches (by signature), else 'Custom'.
  function runDatasetName(manifest, datasets, run) {
    const sig = runDatasetSignature(manifest, run)
    const match = (datasets || []).find(
      (d) => d && datasetSettingsSignature(manifest, d.settings) === sig,
    )
    return match ? match.name : 'Custom'
  }

  // The Cartesian product of per-lever value lists into a list of settings bundles — the ad-hoc dataset/
  // environment bundles built from the Launch form's ADVANCED per-lever fields (each lever set directly
  // instead of picking a named bundle). `perLever` is [[key, [v1, v2]], [key2, [v3]], ...]; a lever with
  // no values is skipped (falls back to its default at plan time). Empty input -> [{}] (one bundle, no
  // pins). A lever swept over N values multiplies the bundle count by N (an ablation grid).
  function cartesianBundles(perLever) {
    let bundles = [{}]
    for (const [key, values] of perLever || []) {
      if (!Array.isArray(values) || !values.length) continue
      const next = []
      for (const bundle of bundles) for (const value of values) next.push({ ...bundle, [key]: value })
      bundles = next
    }
    return bundles
  }

  const api = {
    datasetLeverKeys: datasetLeverKeys,
    datasetSettingsSignature: datasetSettingsSignature,
    findDuplicateDataset: findDuplicateDataset,
    runDatasetSignature: runDatasetSignature,
    runDatasetName: runDatasetName,
    cartesianBundles: cartesianBundles,
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = api
  root.Datasets = api
})(typeof globalThis !== 'undefined' ? globalThis : this)
