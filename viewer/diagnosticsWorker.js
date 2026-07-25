// Off-main-thread Research Diagnostician compute. Runs diagnostics.js's PURE `analyze` (diagnose + campaign
// generation over the 20k+ run corpus) so the Diagnosis tab never freezes the UI; the main thread only paints the
// slim result. diagnostics.js's dual-export IIFE assigns `root.Diagnostics` where root = window (absent in a
// Worker) — so shim `self.window = self` BEFORE importScripts; `injectStyles` still no-ops (no `document` here).
self.window = self
var loaded = false
self.onmessage = function (e) {
  var msg = e.data || {}
  try {
    if (!loaded) {
      // scriptUrl carries the same cache-bust the page used, so the worker analyzes with the deployed logic.
      importScripts(msg.scriptUrl)
      loaded = true
    }
    var analysis = self.Diagnostics.analyze({ runs: msg.runs, manifest: msg.manifest, settests: msg.settests })
    self.postMessage({ ok: true, analysis: analysis, token: msg.token })
  } catch (err) {
    self.postMessage({ ok: false, error: String((err && err.message) || err), token: msg.token })
  }
}
