// Pure decision logic for the unified Hypotheses registry — a hypothesis is a claim that runs prove or
// disprove. Its `spec` both LAUNCHES the runs that test it AND identifies them: a run is evidence iff its
// config is consistent with the spec (every `fixed` lever matches; every swept lever's value is one of the
// options). The verdict is auto-derived from those runs (beats buy-and-hold OOS) and re-checked when runs
// land, recording which runs flipped it. Pure + dual-loaded (browser `window.Hypothesis` + node
// `module.exports`) so the ACTUAL viewer logic is unit-tested directly. A `RunRow` is `{ key, summary:{
// config, metrics, objective, status } }`.
;(function (root) {
  'use strict'

  const VERDICTS = ['untested', 'proven', 'disproved']

  // True iff a run config is consistent with `spec`: every `fixed` lever equals the run's value and every
  // swept lever's value is one of the options. An empty spec (no fixed + no sweep) matches NOTHING — a
  // blank hypothesis must not claim the whole backlog as evidence.
  function specMatchesConfig(spec, config) {
    const fixed = (spec && spec.fixed) || {}
    const sweep = (spec && spec.sweep) || {}
    const fixedKeys = Object.keys(fixed)
    const sweepKeys = Object.keys(sweep)
    if (!fixedKeys.length && !sweepKeys.length) return false
    const cfg = config || {}
    for (let i = 0; i < fixedKeys.length; i++) {
      const k = fixedKeys[i]
      if (String(cfg[k]) !== String(fixed[k])) return false
    }
    for (let j = 0; j < sweepKeys.length; j++) {
      const k = sweepKeys[j]
      const raw = sweep[k]
      const opts = Array.isArray(raw) ? raw : [raw]
      if (!opts.map(String).includes(String(cfg[k]))) return false
    }
    return true
  }

  // The runs (of `runs`) that are evidence for a hypothesis `spec`.
  function hypothesisMatchingRuns(spec, runs) {
    return (runs || []).filter((r) =>
      specMatchesConfig(spec, (r.summary && r.summary.config) || {}),
    )
  }

  // The aggregate read of matching runs: count, best objective (per `direction`), and whether any beats
  // buy-and-hold OOS (`return_vs_hold_pct > 0`). Null when no non-failed run is present.
  function measuredFromRuns(runs, direction) {
    const live = (runs || []).filter((r) => r.summary && r.summary.status !== 'failed')
    if (!live.length) return null
    const objectives = live
      .map((r) => Number(r.summary.objective))
      .filter((v) => Number.isFinite(v))
    const objective = objectives.length
      ? direction === 'min'
        ? Math.min.apply(null, objectives)
        : Math.max.apply(null, objectives)
      : NaN
    const vhs = live
      .map((r) => Number(((r.summary && r.summary.metrics) || {}).return_vs_hold_pct))
      .filter(Number.isFinite)
    return {
      runs: live.length,
      objective,
      beatsHold: vhs.length ? Math.max.apply(null, vhs) > 0 : null,
    }
  }

  // The verdict a measured read implies: untested with no beats-hold signal OR too few runs to trust
  // (fewer than `minRuns`), else proven/disproved. A single run can't adequately prove a hypothesis.
  function autoVerdictFor(measured, minRuns) {
    if (!measured || measured.beatsHold === null) return 'untested'
    if (minRuns && measured.runs < minRuns) return 'untested'
    return measured.beatsHold ? 'proven' : 'disproved'
  }

  // The verdict to SHOW: a manual override wins; otherwise the live auto-verdict from matching runs.
  function effectiveVerdict(h, runs, direction, minRuns) {
    if (h && h.verdictSource === 'manual' && VERDICTS.indexOf(h.status) >= 0) return h.status
    return autoVerdictFor(
      measuredFromRuns(hypothesisMatchingRuns(h && h.spec, runs), direction),
      minRuns,
    )
  }

  // Sorted keys of the runs matching a hypothesis (the evidence-set identity used to detect new runs).
  function matchedKeysOf(spec, runs) {
    return hypothesisMatchingRuns(spec, runs)
      .map((r) => r.key)
      .sort()
  }

  // Re-evaluate a hypothesis against the current runs. Returns `{ next, transition, changed }`:
  //   - manual verdicts are never auto-flipped (changed:false, no write).
  //   - `changed` is true iff the matched-run set OR the auto-status changed since `h.evidence` — the
  //     write guard, so a refresh with no new runs is a no-op.
  //   - a `transition` is produced ONLY when the auto-status flips; `byRunKeys` are the runs new since the
  //     last snapshot (what caused the flip).
  // `opts` carries `direction` (objective direction) and `at` (ISO timestamp for snapshots/transitions),
  // since this module is pure and cannot read the clock.
  function evaluateHypothesis(h, runs, opts) {
    const direction = (opts && opts.direction) || 'max'
    const at = (opts && opts.at) || ''
    const minRuns = (opts && opts.minRuns) || 0
    if (h && h.verdictSource === 'manual') return { next: h, transition: null, changed: false }
    const matchedKeys = matchedKeysOf(h && h.spec, runs)
    const measured = measuredFromRuns(hypothesisMatchingRuns(h && h.spec, runs), direction)
    const nextStatus = autoVerdictFor(measured, minRuns)
    const prev = (h && h.evidence) || { matchedKeys: [], status: 'untested' }
    const prevKeys = prev.matchedKeys || []
    const keysChanged =
      prevKeys.length !== matchedKeys.length || matchedKeys.some((k, i) => k !== prevKeys[i])
    const statusChanged = (prev.status || 'untested') !== nextStatus
    if (!keysChanged && !statusChanged) return { next: h, transition: null, changed: false }
    const newKeys = matchedKeys.filter((k) => prevKeys.indexOf(k) < 0)
    const evidence = { at, status: nextStatus, matchedKeys, measured }
    const next = Object.assign({}, h, {
      status: nextStatus,
      verdictSource: 'auto',
      evidence,
    })
    let transition = null
    if (statusChanged) {
      transition = {
        at,
        from: prev.status || 'untested',
        to: nextStatus,
        byRunKeys: newKeys,
        measured,
      }
      next.transitions = (h && h.transitions ? h.transitions : []).concat([transition])
    }
    return { next, transition, changed: true }
  }

  // A paper's verdict rolls up from its linked hypotheses: any proven ⇒ holds-up, all disproved ⇒ fluff,
  // else untested (no links, or a mix of untested/disproved).
  function rollupPaperVerdict(paper, hyps, runs, direction, minRuns) {
    const ids = {}
    const linkIds = (paper && paper.hypothesisIds) || []
    for (let i = 0; i < linkIds.length; i++) ids[linkIds[i]] = true
    const linked = (hyps || []).filter((h) => ids[h.id])
    if (!linked.length) return 'untested'
    const verdicts = linked.map((h) => effectiveVerdict(h, runs, direction, minRuns))
    if (verdicts.indexOf('proven') >= 0) return 'holds-up'
    if (verdicts.every((v) => v === 'disproved')) return 'fluff'
    return 'untested'
  }

  const Hypothesis = {
    VERDICTS: VERDICTS,
    specMatchesConfig: specMatchesConfig,
    hypothesisMatchingRuns: hypothesisMatchingRuns,
    matchedKeysOf: matchedKeysOf,
    measuredFromRuns: measuredFromRuns,
    autoVerdictFor: autoVerdictFor,
    effectiveVerdict: effectiveVerdict,
    evaluateHypothesis: evaluateHypothesis,
    rollupPaperVerdict: rollupPaperVerdict,
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = Hypothesis
  if (root) root.Hypothesis = Hypothesis
})(typeof window !== 'undefined' ? window : null)
