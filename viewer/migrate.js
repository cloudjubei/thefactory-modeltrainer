// Record migrations for the sandboxed hub viewer: a hold-fee metrics fix and a legacy hypothesis/model
// normalization, applied on load. Plus two small shared helpers the launch UI uses to keep the input-only
// `fidelity_set: "auto"` synonym out of NEW datasets (it is not a record migration). Pure + dual-loaded
// (browser `window.Migrate` + node `module.exports`) so the actual viewer code is unit-tested directly.
;(function (root) {
  'use strict'

  // Mirror of trainer/fidelity.py: the "auto" fidelity FOLLOWS the run step (the `timeframe` lever) — an
  // hourly step observes 1h+1d, any other step observes just its own bar. Deterministic, and NEVER returns
  // the synonym; the launch form + synthetic-default seed use it to show a concrete value instead of 'auto'.
  function autoFidelity(timeframe) {
    return String(timeframe) === '1h' ? '1h+1d' : '1d'
  }

  // The DEFAULT per-trade fee for runs that predate an explicit `transaction_fee` lever — mirrors
  // src/conf/env_config.py `transaction_fee = 0.001`.
  var DEFAULT_TRANSACTION_FEE = 0.001

  // Recompute a run's hold benchmark NET OF FEES (the producer once stored it gross, so a pure buy-and-hold
  // scored below the benchmark by its own fee drag). Charges the same round-trip fee the model pays —
  // `(1 - fee)^2` on the gross price move — and re-derives `return_vs_hold_pct`. Idempotent: a run already
  // flagged `hold_net_of_fees` (migrated, or emitted by the fixed producer) returns null. Pure recompute
  // off stored fields — no re-run needed.
  function holdFeeMetricsPatch(summary) {
    var metrics = (summary && summary.metrics) || {}
    if (metrics.hold_net_of_fees) return null
    var hold = Number(metrics.hold_return_pct)
    var total = Number(metrics.total_return_pct)
    if (!isFinite(hold) || !isFinite(total)) return null
    var config = (summary && summary.config) || {}
    var fee = Number(config.transaction_fee)
    if (!isFinite(fee)) fee = DEFAULT_TRANSACTION_FEE
    var roundTrip = (1 - fee) * (1 - fee)
    var adjustedHold = ((1 + hold / 100) * roundTrip - 1) * 100
    return {
      hold_return_pct: adjustedHold,
      return_vs_hold_pct: total - adjustedHold,
      hold_net_of_fees: true,
    }
  }

  // Normalize a bare-integer pipeline version ("4", or the number 4) to "major.minor" ("4.0"), so once the
  // manifest adopts major.minor a run that ran under "4" reads as the SAME version "4.0" (same major ⇒ not
  // outdated) and doesn't fragment the version filter into "v4" + "v4.0". Idempotent (a version with a dot,
  // a non-integer label, or a missing version returns null).
  function pipelineVersionPatch(summary) {
    var v = summary && summary.pipelineVersion
    if (v === undefined || v === null || v === '') return null
    var s = String(v)
    if (s.indexOf('.') >= 0 || !/^\d+$/.test(s)) return null
    return { pipelineVersion: s + '.0' }
  }

  // The record patch for ONE run, or null. Each part is independently idempotent, so a fully-migrated run
  // returns null.
  function migrationPatchFor(summary) {
    var patch = null
    var metricsPatch = holdFeeMetricsPatch(summary)
    if (metricsPatch) patch = { metrics: metricsPatch }
    var versionPatch = pipelineVersionPatch(summary)
    if (versionPatch) {
      patch = patch || {}
      patch.pipelineVersion = versionPatch.pipelineVersion
    }
    return patch
  }

  // Map a legacy `-model` record to a unified hypothesis (its `spec.fixed` pins the architecture levers).
  // The id (spec hash) is computed by the caller (async subtle-crypto) and passed in. A model the user had
  // marked proven/disproved becomes a MANUAL verdict so the auto-refresh won't silently overwrite it.
  function hypothesisFromLegacyModel(model, id, at) {
    var m = model || {}
    var status = ['untested', 'proven', 'disproved'].indexOf(m.status) >= 0 ? m.status : 'untested'
    var rationale = [m.rationale, m.algo, m.netArch, m.policyInternals].filter(Boolean).join(' · ')
    return {
      id: id,
      title: m.name || m.modelName || 'Architecture',
      rationale: rationale,
      spec: { fixed: m.match && typeof m.match === 'object' ? m.match : {} },
      status: status,
      verdictSource: status !== 'untested' ? 'manual' : 'auto',
      verdictNote: m.verdictNote,
      claimedMetrics: m.claimedMetrics,
      tags: m.tags,
      source: m.source === 'research' ? 'research' : 'migrated-model',
      createdAt: at,
      updatedAt: at,
    }
  }

  // Patch an OLD hypothesis record (pending/accepted/rejected lifecycle) onto the verdict model, or null
  // if it's already migrated (carries a verdictSource). `rejected` becomes a dismissed untested card.
  function legacyHypothesisPatch(h, at) {
    var x = h || {}
    if (x.verdictSource === 'auto' || x.verdictSource === 'manual') return null
    var patch = { verdictSource: 'auto', updatedAt: at }
    if (x.status === 'rejected') {
      patch.dismissed = true
      patch.status = 'untested'
    } else if (['untested', 'proven', 'disproved'].indexOf(x.status) >= 0) {
      patch.status = x.status
    } else {
      patch.status = 'untested'
    }
    return patch
  }

  var Migrate = {
    // Input-only convenience values a lever accepts in the launch form but that are NEVER stored on a run.
    // The dataset form + runs filter drop these so the `fidelity_set: auto` synonym can't be picked or
    // filtered to — a named dataset always pins a concrete value.
    INPUT_SYNONYMS: ['auto'],
    autoFidelity: autoFidelity,
    holdFeeMetricsPatch: holdFeeMetricsPatch,
    pipelineVersionPatch: pipelineVersionPatch,
    migrationPatchFor: migrationPatchFor,
    hypothesisFromLegacyModel: hypothesisFromLegacyModel,
    legacyHypothesisPatch: legacyHypothesisPatch,
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = Migrate
  if (root) root.Migrate = Migrate
})(typeof window !== 'undefined' ? window : null)
