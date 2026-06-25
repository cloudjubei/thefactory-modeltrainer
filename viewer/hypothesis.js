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

  // The ordered CONTEXT cells a spec spans: the cartesian of its `environments` × `datasets` bundles
  // (env-major, mirroring expandExperimentMatrix). Empty ⇒ a single-context spec (the pooled path). A
  // context-spanning hypothesis compares its runs ACROSS these cells; runs in different cells are never
  // pooled (they ran in different environments/datasets — that's a different comparison).
  function contextCells(spec) {
    const envs = (spec && spec.environments) || []
    const dss = (spec && spec.datasets) || []
    if (!envs.length && !dss.length) return []
    const envCells = envs.length ? envs : [{}]
    const dsCells = dss.length ? dss : [{}]
    const cells = []
    for (let i = 0; i < envCells.length; i++) {
      for (let j = 0; j < dsCells.length; j++) {
        cells.push(Object.assign({}, envCells[i], dsCells[j]))
      }
    }
    return cells
  }

  // True iff a run config carries every context-lever value of `cell` (string-compared).
  function configMatchesCell(config, cell) {
    const cfg = config || {}
    const keys = Object.keys(cell || {})
    for (let i = 0; i < keys.length; i++) {
      if (String(cfg[keys[i]]) !== String(cell[keys[i]])) return false
    }
    return true
  }

  // The spec's matching runs partitioned by context cell — one group per cell, each holding only the runs
  // that ran in it (a run matching no declared cell is dropped). Null for a single-context spec.
  function groupRunsByContext(spec, runs) {
    const cells = contextCells(spec)
    if (!cells.length) return null
    const matching = hypothesisMatchingRuns(spec, runs)
    return cells.map((cell) => ({
      context: cell,
      runs: matching.filter((r) => configMatchesCell((r.summary && r.summary.config) || {}, cell)),
    }))
  }

  // Per-cell measured reads (count/objective/beats-hold) — each computed over ONLY that cell's runs.
  function measuredByContext(spec, runs, direction) {
    const groups = groupRunsByContext(spec, runs)
    if (!groups) return null
    return groups.map((g) => ({
      context: g.context,
      runKeys: g.runs.map((r) => r.key).sort(),
      measured: measuredFromRuns(g.runs, direction),
    }))
  }

  // The verdict a CROSS-CONTEXT comparison implies. `kind` is what the hypothesis claims:
  //   - beats-baseline: the best non-baseline cell's objective beats the baseline cell's (default).
  //   - invariant: the objective is stable across cells (spread ≤ tolerance) — a robustness thesis.
  //   - differs: the objective moves across cells (spread > tolerance) — a sensitivity thesis.
  // Untested until there are ≥2 cells each with a measured read of ≥minRuns finite-objective runs.
  function compareContexts(perContext, comparison, direction, minRuns) {
    const kind = (comparison && comparison.kind) || 'beats-baseline'
    const cells = perContext || []
    const ready = cells.every(
      (c) =>
        c.measured &&
        (!minRuns || c.measured.runs >= minRuns) &&
        Number.isFinite(c.measured.objective),
    )
    if (cells.length < 2 || !ready) return 'untested'
    const objs = cells.map((c) => c.measured.objective)
    const baselineIndex = (comparison && comparison.baselineIndex) || 0
    if (kind === 'beats-baseline') {
      const baseObj = objs[baselineIndex]
      const others = objs.filter((_, i) => i !== baselineIndex)
      const best = direction === 'min' ? Math.min.apply(null, others) : Math.max.apply(null, others)
      const better = direction === 'min' ? best < baseObj : best > baseObj
      return better ? 'proven' : 'disproved'
    }
    const tol = comparison && comparison.tolerance != null ? comparison.tolerance : 0.1
    const spread =
      (Math.max.apply(null, objs) - Math.min.apply(null, objs)) /
      (Math.abs(objs[baselineIndex]) || 1)
    if (kind === 'invariant') return spread <= tol ? 'proven' : 'disproved'
    if (kind === 'differs') return spread > tol ? 'proven' : 'disproved'
    return 'untested'
  }

  // The auto-verdict for a hypothesis: a context-spanning spec reads its cross-context comparison; a
  // single-context spec uses the pooled beats-hold rule.
  function autoVerdictForHypothesis(h, runs, direction, minRuns) {
    const spec = h && h.spec
    if (contextCells(spec).length) {
      return compareContexts(
        measuredByContext(spec, runs, direction),
        h && h.comparison,
        direction,
        minRuns,
      )
    }
    return autoVerdictFor(measuredFromRuns(hypothesisMatchingRuns(spec, runs), direction), minRuns)
  }

  // The verdict to SHOW: a manual override wins; otherwise the live auto-verdict from matching runs.
  function effectiveVerdict(h, runs, direction, minRuns) {
    if (h && h.verdictSource === 'manual' && VERDICTS.indexOf(h.status) >= 0) return h.status
    return autoVerdictForHypothesis(h, runs, direction, minRuns)
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
    const nextStatus = autoVerdictForHypothesis(h, runs, direction, minRuns)
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
    contextCells: contextCells,
    groupRunsByContext: groupRunsByContext,
    measuredByContext: measuredByContext,
    compareContexts: compareContexts,
    effectiveVerdict: effectiveVerdict,
    evaluateHypothesis: evaluateHypothesis,
    rollupPaperVerdict: rollupPaperVerdict,
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = Hypothesis
  if (root) root.Hypothesis = Hypothesis
})(typeof window !== 'undefined' ? window : null)
