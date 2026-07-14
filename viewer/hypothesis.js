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
    const cmp = spec && spec.compare
    const hasCompare = !!(cmp && cmp.lever && Array.isArray(cmp.values) && cmp.values.length)
    const fixedKeys = Object.keys(fixed)
    const sweepKeys = Object.keys(sweep)
    if (!fixedKeys.length && !sweepKeys.length && !hasCompare) return false
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
    // A `compare` restricts its lever to the compared values, so matching runs partition cleanly by value.
    if (hasCompare && !cmp.values.map(String).includes(String(cfg[cmp.lever]))) return false
    return true
  }

  // The runs (of `runs`) that are evidence for a hypothesis `spec`.
  function hypothesisMatchingRuns(spec, runs) {
    return (runs || []).filter((r) =>
      specMatchesConfig(spec, (r.summary && r.summary.config) || {}),
    )
  }

  // A per-snapshot inverted index over runs: leverKey -> String(value) -> [runs carrying that value]. Built
  // ONCE per run snapshot so matching can run over a small candidate SUPERSET rather than the whole set —
  // the accelerator behind a non-blocking verdict refresh. String() coercion is IDENTICAL to
  // specMatchesConfig, so buckets and lookups agree exactly. Holds only run REFERENCES (no duplication).
  function buildRunIndex(runs) {
    const index = new Map()
    const all = runs || []
    for (let i = 0; i < all.length; i++) {
      const cfg = (all[i].summary && all[i].summary.config) || {}
      const keys = Object.keys(cfg)
      for (let k = 0; k < keys.length; k++) {
        const key = keys[k]
        let byVal = index.get(key)
        if (!byVal) {
          byVal = new Map()
          index.set(key, byVal)
        }
        const v = String(cfg[key])
        let bucket = byVal.get(v)
        if (!bucket) {
          bucket = []
          byVal.set(v, bucket)
        }
        bucket.push(all[i])
      }
    }
    return index
  }

  // The smallest candidate SUPERSET of the runs matching `spec`, via the index. Every true match satisfies
  // EVERY constraint (fixed AND sweep AND compare are all required by specMatchesConfig), so it lives in
  // each constraint's bucket(s) — hence any single constraint's bucket(s) is a valid superset and the
  // SMALLEST is both safe and tightest. specMatchesConfig then verifies exactly, so matching over this
  // superset is byte-identical to a full scan. Returns [] when the spec has no fixed/sweep/compare —
  // mirroring specMatchesConfig, which matches nothing for such a spec (environments/datasets only PARTITION
  // already-matched runs; they never gate the pooled match).
  function candidateRunsFor(spec, index) {
    const fixed = (spec && spec.fixed) || {}
    const sweep = (spec && spec.sweep) || {}
    const cmp = spec && spec.compare
    const hasCompare = !!(cmp && cmp.lever && Array.isArray(cmp.values) && cmp.values.length)
    const fixedKeys = Object.keys(fixed)
    const sweepKeys = Object.keys(sweep)
    if (!fixedKeys.length && !sweepKeys.length && !hasCompare) return []
    const bucketFor = (key, value) => {
      const byVal = index && index.get(key)
      return (byVal && byVal.get(String(value))) || []
    }
    // Union of the buckets for a lever's allowed values (sweep options / compare values). Buckets for
    // DISTINCT string values are disjoint, but coerced duplicates (0.5 vs '0.5') collapse — a Set guards it.
    const unionBuckets = (key, values) => {
      if (values.length === 1) return bucketFor(key, values[0])
      const seen = new Set()
      const out = []
      for (let i = 0; i < values.length; i++) {
        const b = bucketFor(key, values[i])
        for (let j = 0; j < b.length; j++) {
          if (!seen.has(b[j])) {
            seen.add(b[j])
            out.push(b[j])
          }
        }
      }
      return out
    }
    let best = null
    const consider = (arr) => {
      if (best === null || arr.length < best.length) best = arr
    }
    for (let i = 0; i < fixedKeys.length; i++)
      consider(bucketFor(fixedKeys[i], fixed[fixedKeys[i]]))
    for (let i = 0; i < sweepKeys.length; i++) {
      const raw = sweep[sweepKeys[i]]
      consider(unionBuckets(sweepKeys[i], Array.isArray(raw) ? raw : [raw]))
    }
    if (hasCompare) consider(unionBuckets(cmp.lever, cmp.values))
    return best || []
  }

  // The single-context SUCCESS BENCHMARK: which run METRIC proves a hypothesis and the threshold it must
  // clear. The manifest declares it (`hypothesisBenchmark: { metric, threshold?, direction? }` — e.g.
  // CartPole: eval_return_mean ≥ 475); absent, the trading line's historical default applies
  // (return_vs_hold_pct > 0 — beats buy-and-hold OOS).
  const DEFAULT_HYPOTHESIS_BENCHMARK = { metric: 'return_vs_hold_pct', threshold: 0, direction: 'max' }
  function resolveBenchmark(benchmark) {
    const b = benchmark || DEFAULT_HYPOTHESIS_BENCHMARK
    return {
      metric: typeof b.metric === 'string' && b.metric ? b.metric : DEFAULT_HYPOTHESIS_BENCHMARK.metric,
      threshold: Number.isFinite(Number(b.threshold)) ? Number(b.threshold) : 0,
      direction: b.direction === 'min' ? 'min' : 'max',
    }
  }

  // The aggregate read of matching runs: count, best objective (per `direction`), and whether any run
  // clears the project's hypothesis benchmark. Null when no non-failed run is present. The `beatsHold`
  // field name is historical (the trading line's benchmark is buy-and-hold) — it means "meets the
  // benchmark" and is judged by the manifest's `hypothesisBenchmark` when one is declared.
  function measuredFromRuns(runs, direction, benchmark) {
    // Exclude failed AND invalid runs — an invalid run (since-fixed bug) must never move a verdict.
    const live = (runs || []).filter(
      (r) => r.summary && r.summary.status !== 'failed' && r.summary.status !== 'invalid',
    )
    if (!live.length) return null
    const objectives = live
      .map((r) => Number(r.summary.objective))
      .filter((v) => Number.isFinite(v))
    const objective = objectives.length
      ? direction === 'min'
        ? Math.min.apply(null, objectives)
        : Math.max.apply(null, objectives)
      : NaN
    const bench = resolveBenchmark(benchmark)
    const values = live
      .map((r) => Number(((r.summary && r.summary.metrics) || {})[bench.metric]))
      .filter(Number.isFinite)
    const best = values.length
      ? bench.direction === 'min'
        ? Math.min.apply(null, values)
        : Math.max.apply(null, values)
      : null
    return {
      runs: live.length,
      objective,
      beatsHold:
        best === null ? null : bench.direction === 'min' ? best < bench.threshold : best > bench.threshold,
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
    const cmp = spec && spec.compare
    // A `compare` lever contributes one cell per value — the judgeable form of "A vs B" — crossing the
    // environment/dataset bundles like another context dimension (compare-major).
    const cmpCells =
      cmp && cmp.lever && Array.isArray(cmp.values) && cmp.values.length
        ? cmp.values.map((v) => ({ [cmp.lever]: v }))
        : []
    if (!envs.length && !dss.length && !cmpCells.length) return []
    const compareCells = cmpCells.length ? cmpCells : [{}]
    const envCells = envs.length ? envs : [{}]
    const dsCells = dss.length ? dss : [{}]
    const cells = []
    for (let c = 0; c < compareCells.length; c++) {
      for (let i = 0; i < envCells.length; i++) {
        for (let j = 0; j < dsCells.length; j++) {
          cells.push(Object.assign({}, compareCells[c], envCells[i], dsCells[j]))
        }
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
  function measuredByContext(spec, runs, direction, benchmark) {
    const groups = groupRunsByContext(spec, runs)
    if (!groups) return null
    return groups.map((g) => ({
      context: g.context,
      runKeys: g.runs.map((r) => r.key).sort(),
      measured: measuredFromRuns(g.runs, direction, benchmark),
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
    // An out-of-range baseline must stay untested — comparing against undefined would silently mis-judge.
    if (baselineIndex >= objs.length) return 'untested'
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

  // Plain-language success/failure criterion for a context-spanning (environments / datasets / compare)
  // hypothesis — mirrors compareContexts EXACTLY, so the card states precisely how the verdict is decided
  // instead of just "no matching runs yet". opts: { objectiveName, direction:'min'|'max', minRuns }.
  function comparisonCriterion(spec, comparison, opts) {
    const o = opts || {}
    const objName = o.objectiveName || 'the objective'
    const minRuns = o.minRuns || 3
    const cells = contextCells(spec || {})
    const n = cells.length
    const kind = (comparison && comparison.kind) || 'beats-baseline'
    const cellLabel = (cell) =>
      Object.keys(cell || {})
        .map((k) => `${k}=${cell[k]}`)
        .join(', ') || 'the baseline'
    const gate = `Each of the ${n} contexts is run SEPARATELY (never pooled), and it stays UNTESTED until every context has at least ${minRuns} runs reporting ${objName}.`
    if (kind === 'invariant') {
      const tol = comparison && comparison.tolerance != null ? comparison.tolerance : 0.1
      return `${gate} PROVEN if ${objName} is INVARIANT across them — the spread (max−min, relative to the baseline) is ≤ ${tol}; DISPROVED if it exceeds ${tol}.`
    }
    if (kind === 'differs') {
      const tol = comparison && comparison.tolerance != null ? comparison.tolerance : 0.1
      return `${gate} PROVEN if ${objName} DIFFERS across them — the spread (max−min, relative to the baseline) exceeds ${tol}; DISPROVED if it is within ${tol}.`
    }
    const baselineIndex = (comparison && comparison.baselineIndex) || 0
    const baseLabel = cellLabel(cells[baselineIndex])
    const dir = o.direction === 'min' ? 'lowest' : 'highest'
    return `${gate} PROVEN if the ${dir} ${objName} among the other context${n - 1 === 1 ? '' : 's'} beats the baseline (${baseLabel}); DISPROVED if none beat it.`
  }

  // Whether a hypothesis can't yet be TESTED because a model it requires isn't implemented. `modelImplemented`
  // is an injected resolver name -> true (implemented) | false (known but unimplemented) | null (unknown), so
  // this module stays pure (no Models dependency). A fixed model_name is required; a compare over model_name
  // can run if ANY arm is implemented, so it's blocked only when none is and at least one is unimplemented.
  function requiresUnimplementedModel(spec, modelImplemented) {
    if (!spec || typeof modelImplemented !== 'function') return false
    if (
      spec.fixed &&
      spec.fixed.model_name != null &&
      modelImplemented(String(spec.fixed.model_name)) === false
    )
      return true
    if (spec.compare && spec.compare.lever === 'model_name' && Array.isArray(spec.compare.values)) {
      const states = spec.compare.values.map((v) => modelImplemented(String(v)))
      if (!states.some((s) => s === true) && states.some((s) => s === false)) return true
    }
    return false
  }

  // The auto-verdict for a hypothesis: a context-spanning spec reads its cross-context comparison; a
  // single-context spec uses the pooled beats-hold rule. Precedence: decided (proven/disproved) > proposed
  // (blocked on an unimplemented model) > untested.
  function autoVerdictForHypothesis(h, runs, direction, minRuns, modelImplemented, benchmark) {
    const spec = h && h.spec
    let verdict
    if (contextCells(spec).length) {
      verdict = compareContexts(
        measuredByContext(spec, runs, direction, benchmark),
        h && h.comparison,
        direction,
        minRuns,
      )
    } else {
      verdict = autoVerdictFor(
        measuredFromRuns(hypothesisMatchingRuns(spec, runs), direction, benchmark),
        minRuns,
      )
    }
    if (verdict === 'untested' && requiresUnimplementedModel(spec, modelImplemented))
      return 'proposed'
    return verdict
  }

  // The verdict to SHOW: a manual override wins; otherwise the live auto-verdict from matching runs.
  function effectiveVerdict(h, runs, direction, minRuns, modelImplemented, benchmark) {
    if (h && h.verdictSource === 'manual' && VERDICTS.indexOf(h.status) >= 0) return h.status
    return autoVerdictForHypothesis(h, runs, direction, minRuns, modelImplemented, benchmark)
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
    const modelImplemented = opts && opts.modelImplemented
    const benchmark = opts && opts.benchmark
    if (h && h.verdictSource === 'manual') return { next: h, transition: null, changed: false }
    const matchedKeys = matchedKeysOf(h && h.spec, runs)
    const measured = measuredFromRuns(hypothesisMatchingRuns(h && h.spec, runs), direction, benchmark)
    const nextStatus = autoVerdictForHypothesis(h, runs, direction, minRuns, modelImplemented, benchmark)
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

  // A paper's verdict rolls up from the WEIGHTED balance of its linked hypotheses' DECIDED verdicts —
  // not "any proven ⇒ holds-up" (which wrongly passed a paper with more disproved than proven). Over the
  // decided (proven|disproved) hypotheses, score = provenWeight / decidedWeight: ≥0.75 holds-up, ≤0.25
  // fluff, in between SHAKY (genuinely mixed), nothing decided ⇒ untested. Untested hypotheses are not yet
  // evidence (ignored in the score, surfaced in the explanation). A per-hypothesis `weight` (default 1)
  // lets ONE central claim outweigh minor ones. `paperVerdictDetail` returns the counts/score/explanation
  // the card shows; `rollupPaperVerdict` returns just the status (badge + back-compat).
  const PAPER_HOLDS_UP_AT = 0.75
  const PAPER_FLUFF_AT = 0.25
  // Score a set of linked hypotheses' verdicts+weights into a paper-style detail. `linked` is
  // [{verdict, weight, thesis?}] — the ONE place the holds-up/shaky/fluff thresholds live. A `proposed`
  // hypothesis (blocked on an unimplemented model) is COUNTED but EXCLUDED from the decided score, so it
  // signals pending work without diluting the verdict.
  function scoreLinkedBase(linked) {
    let proven = 0,
      disproved = 0,
      untested = 0,
      proposed = 0,
      provenW = 0,
      disprovedW = 0,
      hasWeights = false
    for (let i = 0; i < (linked || []).length; i++) {
      const it = linked[i]
      const w =
        typeof it.weight === 'number' && isFinite(it.weight) && it.weight > 0 ? it.weight : 1
      if (w !== 1) hasWeights = true
      if (it.verdict === 'proven') {
        proven++
        provenW += w
      } else if (it.verdict === 'disproved') {
        disproved++
        disprovedW += w
      } else if (it.verdict === 'proposed') proposed++
      else untested++
    }
    const decidedW = provenW + disprovedW
    const score = decidedW > 0 ? provenW / decidedW : null
    let status
    if (decidedW === 0) status = 'untested'
    else if (score >= PAPER_HOLDS_UP_AT) status = 'holds-up'
    else if (score <= PAPER_FLUFF_AT) status = 'fluff'
    else status = 'shaky'
    const detail = {
      status: status,
      score: score,
      counts: {
        proven: proven,
        disproved: disproved,
        untested: untested,
        proposed: proposed,
        total: (linked || []).length,
      },
      hasWeights: hasWeights,
      weighted: { proven: provenW, disproved: disprovedW, decided: decidedW },
    }
    detail.why = paperVerdictWhy(detail)
    return detail
  }
  // Group linked rows by their `claim` label (trimmed; null/empty = one untagged bucket), first-seen order.
  function groupHypothesesByClaim(linked) {
    const order = []
    const byKey = {}
    for (let i = 0; i < (linked || []).length; i++) {
      const it = linked[i]
      const label = typeof it.claim === 'string' && it.claim.trim() ? it.claim.trim() : null
      const key = label === null ? ' untagged' : label
      if (!byKey[key]) {
        byKey[key] = { claim: label, items: [] }
        order.push(key)
      }
      byKey[key].items.push(it)
    }
    return order.map((k) => byKey[k])
  }
  // The per-claim breakdown for a paper — an ADDITIVE lens. multiClaim when >1 DISTINCT non-empty label.
  function scorePaperClaims(linked) {
    const groups = groupHypothesesByClaim(linked)
    return {
      claims: groups.map((g) => ({ claim: g.claim, detail: scoreLinkedBase(g.items) })),
      multiClaim: groups.filter((g) => g.claim).length > 1,
    }
  }
  // The overall paper verdict (pooled over ALL linked hypotheses) PLUS the additive per-claim lens, so the
  // viewer's `paperVerdictInfo` (which calls this directly) gets `multiClaim`/`claims` too. For the status
  // chip, `totalClaims` counts the DISTINCT (labelled) claims and `passedClaims` those that hold up.
  function scorePaperVerdict(linked) {
    const detail = scoreLinkedBase(linked)
    const t = scorePaperClaims(linked)
    detail.multiClaim = t.multiClaim
    if (t.multiClaim) detail.claims = t.claims
    const labelled = t.claims.filter((c) => c.claim)
    detail.totalClaims = labelled.length
    detail.passedClaims = labelled.filter((c) => c.detail.status === 'holds-up').length
    return detail
  }
  function paperVerdictDetail(paper, hyps, runs, direction, minRuns, modelImplemented, benchmark) {
    const ids = {}
    const linkIds = (paper && paper.hypothesisIds) || []
    for (let i = 0; i < linkIds.length; i++) ids[linkIds[i]] = true
    // Weight is the PAPER's per-hypothesis importance (default 1) — NOT a property of the shared hypothesis.
    const weights = (paper && paper.hypothesisWeights) || {}
    const linked = (hyps || [])
      .filter((h) => ids[h.id])
      .map((h) => ({
        verdict: effectiveVerdict(h, runs, direction, minRuns, modelImplemented, benchmark),
        weight: weights[h.id],
        claim: h.claim,
      }))
    return scorePaperVerdict(linked)
  }

  function paperVerdictWhy(d) {
    const c = d.counts
    if (!c.total) return 'No hypotheses linked yet.'
    const propNote = c.proposed ? ', ' + c.proposed + ' awaiting model implementation' : ''
    if (d.status === 'untested')
      return (
        c.untested + c.proposed + ' of ' + c.total + ' hypotheses not yet decided' + propNote + '.'
      )
    const pct = Math.round(d.score * 100)
    const wnote = d.hasWeights ? ', weighted by importance' : ''
    const undecided = c.untested + c.proposed
    const tail = undecided ? ' (' + undecided + ' still undecided' + propNote + ')' : ''
    const evidence =
      c.proven +
      ' proven, ' +
      c.disproved +
      ' disproved' +
      wnote +
      ' — ' +
      pct +
      '% of the decided evidence supports the claim' +
      tail +
      '.'
    if (d.status === 'holds-up') return 'Holds up: ' + evidence
    if (d.status === 'fluff') return 'Fluff: ' + evidence
    return 'Shaky: ' + evidence + ' Mixed — neither side dominates.'
  }
  // A structured explainer for the expanded paper card: THIS paper's current weighted balance + the
  // threshold ladder (what balance flips the verdict). Pure → app.js wraps the strings in markup.
  function paperVerdictExplain(d) {
    const holdsPct = Math.round(PAPER_HOLDS_UP_AT * 100)
    const fluffPct = Math.round(PAPER_FLUFF_AT * 100)
    const c = d.counts || {}
    const w = d.weighted || { proven: 0, decided: 0 }
    const hasW = d.hasWeights
    const round2 = (n) => Math.round(n * 100) / 100
    const undecided = (c.untested || 0) + (c.proposed || 0)
    const notCounted = undecided
      ? ' ' +
        undecided +
        ' undecided' +
        (c.proposed ? ' (' + c.proposed + ' awaiting model implementation)' : '') +
        ' — not counted yet.'
      : ''
    let formula
    if (!w.decided) {
      formula =
        'No decided evidence yet — a hypothesis must be PROVEN or DISPROVED before there is a score.' +
        notCounted
    } else {
      const pct = Math.round((w.proven / w.decided) * 100)
      // Each hypothesis contributes its weight when proven, 0 when disproved; score = proven ÷ decided.
      const num = hasW ? round2(w.proven) + ' proven weight' : c.proven + ' proven'
      const den = hasW ? round2(w.decided) + ' decided weight' : c.proven + c.disproved + ' decided'
      formula =
        'Score = ' +
        num +
        ' ÷ ' +
        den +
        ' = ' +
        pct +
        '% (each hypothesis adds its weight when proven, 0 when disproved).' +
        notCounted
    }
    const ladder =
      '≥ ' +
      holdsPct +
      '% → Holds up · ≤ ' +
      fluffPct +
      '% → Fluff · in between → Shaky · nothing decided → Untested.'
    return { formula: formula, ladder: ladder }
  }

  function rollupPaperVerdict(paper, hyps, runs, direction, minRuns, modelImplemented, benchmark) {
    return paperVerdictDetail(paper, hyps, runs, direction, minRuns, modelImplemented, benchmark).status
  }

  // ——— HYGIENE: why is a hypothesis undecided? ———————————————————————————————————————————————
  // The registry-wide reality (e.g. 22 of 247 judged) is usually structural, not "not enough runs yet".
  // hypothesisHygiene diagnoses ONE hypothesis; hypothesisHygieneCensus aggregates. `manifest` supplies
  // levers (choices/range/appliesWhen) + migrations; both stay pure.

  // Faithful port of the engine's findMigrationRule + applyMigrationRules (modelTrainerUtils.ts) for the
  // pin census: returns the rewritten fixed config, or null when no rule fires / the rule is a delete.
  function migratedFixed(fixed, migrations) {
    const config = fixed || {}
    const rule = (migrations || []).find((r) => {
      const hasClause = !!(r.match || r.matchNot)
      const hasUnset = !!(r.unset && r.unset.length)
      if (!hasClause && !hasUnset) return false
      const eq = Object.entries(r.match || {}).every(
        ([k, v]) => k in config && String(config[k]) === String(v),
      )
      const neq = Object.entries(r.matchNot || {}).every(
        ([k, v]) => k in config && String(config[k]) !== String(v),
      )
      const clauseOk = hasClause ? eq && neq : true
      const unsetOk = hasUnset && !hasClause ? r.unset.some((k) => k in config) : true
      return clauseOk && unsetOk
    })
    if (!rule || rule.delete) return null
    const next = Object.assign({}, config, rule.set || {})
    Object.entries(rule.keepOrDefault || {}).forEach(([k, fb]) => {
      next[k] = k in config ? config[k] : fb
    })
    ;(rule.unset || []).forEach((k) => delete next[k])
    return next
  }

  // The CAUSE of a dead pin (matches === 0), most actionable first. STRUCTURAL causes (more runs can
  // never fix them): 'migrated' — a manifest migration rewrites the pinned value, so launches land under
  // the NEW value and stored configs never match; 'na-pinned' — the lever's appliesWhen is violated by
  // the spec's own pins, so stored configs hold the 'n/a' sentinel; 'off-manifest' — the value is outside
  // the lever's declared choices/range. LAUNCHABLE causes (running the spec fixes them): 'missing-key' —
  // no run config carries the lever (runs predate it); 'never-run' — the value simply hasn't run yet.
  function deadPinCause(lever, value, fixed, keyRuns, manifest, migrated) {
    if (migrated && String(migrated[lever]) !== String(value)) return 'migrated'
    const levers = (manifest && manifest.levers) || {}
    const spec = levers[lever]
    if (spec && spec.appliesWhen && String(value) !== 'n/a') {
      const violated = Object.entries(spec.appliesWhen).some(
        ([control, allowed]) =>
          control in (fixed || {}) &&
          !(Array.isArray(allowed) ? allowed : [allowed])
            .map(String)
            .includes(String(fixed[control])),
      )
      if (violated) return 'na-pinned'
    }
    if (spec) {
      if (Array.isArray(spec.choices) && !spec.choices.map(String).includes(String(value)))
        return 'off-manifest'
      const n = Number(value)
      if (
        Array.isArray(spec.range) &&
        spec.range.length === 2 &&
        Number.isFinite(n) &&
        (n < Number(spec.range[0]) || n > Number(spec.range[1]))
      )
        return 'off-manifest'
    }
    return keyRuns === 0 ? 'missing-key' : 'never-run'
  }

  const STRUCTURAL_PIN_CAUSES = ['migrated', 'na-pinned', 'off-manifest']

  // Diagnose ONE hypothesis against the full run set. Returns:
  //   { verdict, status: 'judged'|'blocked'|'starved', issues: [{kind, lever?, values?, detail}],
  //     pins: [{lever, value, keyRuns, matches, cause?}], cells|null, plannedItems, metricRuns, runsNeeded }
  // 'blocked' = structurally undecidable AS WRITTEN (a structural dead pin, an underplanned pinned-seed
  // spec, a single context cell, an out-of-range baseline, a metric the run family never reports);
  // 'starved' = sound, just needs runs (launch the spec / add seeds / fill cells).
  function hypothesisHygiene(h, runs, manifest, minRuns) {
    const spec = (h && h.spec) || {}
    const direction =
      (manifest && manifest.objective && manifest.objective.direction) === 'min' ? 'min' : 'max'
    const benchmark = resolveBenchmark(manifest && manifest.hypothesisBenchmark)
    const verdict = autoVerdictForHypothesis(h, runs, direction, minRuns, undefined, benchmark)
    const all = runs || []
    if (verdict === 'proven' || verdict === 'disproved') {
      return {
        verdict,
        status: 'judged',
        issues: [],
        pins: [],
        cells: null,
        plannedItems: 0,
        metricRuns: 0,
        runsNeeded: 0,
      }
    }
    const fixed = spec.fixed || {}
    const migrated = migratedFixed(fixed, manifest && manifest.migrations)
    const issues = []
    // Per-fixed-pin census against ALL runs (independent of the other pins), so a dead pin names the
    // exact lever+value that strands the hypothesis rather than just "0 matching runs".
    const pins = Object.entries(fixed).map(([lever, value]) => {
      let keyRuns = 0
      let matches = 0
      for (let i = 0; i < all.length; i++) {
        const cfg = (all[i].summary && all[i].summary.config) || {}
        if (lever in cfg) keyRuns++
        if (String(cfg[lever]) === String(value)) matches++
      }
      const pin = { lever, value, keyRuns, matches }
      if (matches === 0) {
        pin.cause = deadPinCause(lever, value, fixed, keyRuns, manifest, migrated)
        issues.push({
          kind: 'dead-pin',
          lever,
          detail: `fixed ${lever}=${value} matches no run (${pin.cause})`,
          structural: STRUCTURAL_PIN_CAUSES.indexOf(pin.cause) >= 0,
        })
      }
      return pin
    })
    // Per-sweep-option census: options nothing has run yet (launchable — the spec itself fills them).
    Object.entries(spec.sweep || {}).forEach(([lever, raw]) => {
      const opts = Array.isArray(raw) ? raw : [raw]
      const unrun = opts.filter(
        (v) =>
          !all.some(
            (r) => String(((r.summary && r.summary.config) || {})[lever]) === String(v),
          ),
      )
      if (unrun.length)
        issues.push({
          kind: 'sweep-unrun',
          lever,
          values: unrun.map(String),
          detail: `sweep ${lever}: ${unrun.length} of ${opts.length} option(s) never run`,
          structural: false,
        })
    })
    // Planned-item math for the spec AS WRITTEN. Only an EXPLICIT seed list caps re-launches (no seeds
    // declared ⇒ fresh seeds can always be added); explicit configs define their own count.
    const sweepProduct = Object.values(spec.sweep || {}).reduce(
      (n, v) => n * (Array.isArray(v) ? v.length : 1),
      1,
    )
    const cellList = contextCells(spec)
    const seedCount = Array.isArray(spec.seeds) && spec.seeds.length ? spec.seeds.length : 1
    const perCellPlanned = sweepProduct * seedCount
    const plannedItems =
      Array.isArray(spec.configs) && spec.configs.length
        ? spec.configs.length
        : perCellPlanned * (cellList.length || 1)
    if (
      Array.isArray(spec.seeds) &&
      spec.seeds.length &&
      !(Array.isArray(spec.configs) && spec.configs.length) &&
      minRuns &&
      perCellPlanned < minRuns
    ) {
      issues.push({
        kind: 'underplanned',
        detail: `the spec plans ${perCellPlanned} run(s) per cell but the verdict needs ${minRuns} — add seeds or sweep values`,
        structural: true,
      })
    }
    const matching = hypothesisMatchingRuns(spec, all)
    const live = matching.filter(
      (r) => r.summary && r.summary.status !== 'failed' && r.summary.status !== 'invalid',
    )
    const metricRuns = live.filter((r) =>
      Number.isFinite(Number(((r.summary && r.summary.metrics) || {})[benchmark.metric])),
    ).length
    if (matching.length && !live.length) {
      issues.push({
        kind: 'invalid-evidence',
        detail: `all ${matching.length} matching run(s) are failed/invalid — re-run to regenerate evidence`,
        structural: false,
      })
    }
    let cells = null
    let runsNeeded = 0
    if (cellList.length) {
      if (cellList.length < 2) {
        issues.push({
          kind: 'single-cell',
          detail: 'the context bundles collapse to ONE cell — nothing to compare across',
          structural: true,
        })
      }
      const baselineIndex = (h && h.comparison && h.comparison.baselineIndex) || 0
      if (baselineIndex >= cellList.length) {
        issues.push({
          kind: 'baseline-out-of-range',
          detail: `comparison.baselineIndex ${baselineIndex} exceeds the ${cellList.length} cell(s)`,
          structural: true,
        })
      }
      const perContext = measuredByContext(spec, all, direction) || []
      cells = perContext.map((c) => ({
        context: c.context,
        runs: c.measured ? c.measured.runs : 0,
        needed: minRuns || 0,
      }))
      const starvedCells = cells.filter((c) => c.runs < (minRuns || 0))
      if (starvedCells.length) {
        issues.push({
          kind: 'cell-starved',
          detail: `${starvedCells.length} of ${cells.length} cell(s) below ${minRuns} run(s)`,
          structural: false,
        })
      }
      runsNeeded = cells.reduce((n, c) => n + Math.max(0, (minRuns || 0) - c.runs), 0)
    } else {
      runsNeeded = Math.max(0, (minRuns || 0) - live.length)
      if (live.length && !metricRuns) {
        issues.push({
          kind: 'no-metric',
          detail:
            manifest && manifest.hypothesisBenchmark
              ? `matching runs report no ${benchmark.metric} — the manifest's hypothesisBenchmark metric this project judges by`
              : `matching runs report no ${benchmark.metric} — declare hypothesisBenchmark ({metric, threshold}) in the manifest so hypotheses are judged by THIS project's success metric`,
          structural: true,
        })
      }
    }
    const status = issues.some((i) => i.structural) ? 'blocked' : 'starved'
    return { verdict, status, issues, pins, cells, plannedItems, metricRuns, runsNeeded }
  }

  // Fold a PRECOMPUTED per-hypothesis diagnosis map (id -> hygiene) into the aggregate census. Split out so
  // a chunked/incremental sweep (the viewer, to stay non-blocking) and the one-shot hypothesisHygieneCensus
  // share ONE classification path. A manual non-untested verdict counts as judged regardless of the raw
  // hygiene status. `byId` may be a Map or a plain object.
  function foldHygieneCensus(hyps, byId) {
    const get =
      byId && typeof byId.get === 'function' ? (id) => byId.get(id) : (id) => byId && byId[id]
    const census = {
      total: (hyps || []).length,
      judged: 0,
      blocked: 0,
      starved: 0,
      byIssue: {},
      blockedIds: [],
    }
    for (const h of hyps || []) {
      if (h && h.verdictSource === 'manual' && VERDICTS.indexOf(h.status) >= 0 && h.status !== 'untested') {
        census.judged++
        continue
      }
      const d = get(h && h.id)
      if (!d) continue
      if (d.status === 'judged') census.judged++
      else if (d.status === 'blocked') {
        census.blocked++
        census.blockedIds.push(h && h.id)
      } else census.starved++
      for (const issue of d.issues) {
        census.byIssue[issue.kind] = (census.byIssue[issue.kind] || 0) + 1
      }
    }
    return census
  }

  // The registry-wide roll-up: how many hypotheses are judged / structurally blocked / merely starved,
  // issue-kind counts, and the blocked ids (the actionable fix list). Delegates classification to
  // foldHygieneCensus so the one-shot and the viewer's chunked warm can't diverge.
  function hypothesisHygieneCensus(hyps, runs, manifest, minRuns) {
    const byId = new Map()
    for (const h of hyps || []) {
      if (h && h.verdictSource === 'manual' && VERDICTS.indexOf(h.status) >= 0 && h.status !== 'untested')
        continue
      byId.set(h && h.id, hypothesisHygiene(h, runs, manifest, minRuns))
    }
    return foldHygieneCensus(hyps, byId)
  }

  const Hypothesis = {
    VERDICTS: VERDICTS,
    specMatchesConfig: specMatchesConfig,
    resolveBenchmark: resolveBenchmark,
    hypothesisMatchingRuns: hypothesisMatchingRuns,
    buildRunIndex: buildRunIndex,
    candidateRunsFor: candidateRunsFor,
    matchedKeysOf: matchedKeysOf,
    measuredFromRuns: measuredFromRuns,
    autoVerdictFor: autoVerdictFor,
    contextCells: contextCells,
    groupRunsByContext: groupRunsByContext,
    measuredByContext: measuredByContext,
    compareContexts: compareContexts,
    comparisonCriterion: comparisonCriterion,
    effectiveVerdict: effectiveVerdict,
    autoVerdictForHypothesis: autoVerdictForHypothesis,
    requiresUnimplementedModel: requiresUnimplementedModel,
    evaluateHypothesis: evaluateHypothesis,
    hypothesisHygiene: hypothesisHygiene,
    hypothesisHygieneCensus: hypothesisHygieneCensus,
    foldHygieneCensus: foldHygieneCensus,
    rollupPaperVerdict: rollupPaperVerdict,
    paperVerdictDetail: paperVerdictDetail,
    scorePaperVerdict: scorePaperVerdict,
    scorePaperClaims: scorePaperClaims,
    groupHypothesesByClaim: groupHypothesesByClaim,
    paperVerdictExplain: paperVerdictExplain,
    PAPER_HOLDS_UP_AT: PAPER_HOLDS_UP_AT,
    PAPER_FLUFF_AT: PAPER_FLUFF_AT,
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = Hypothesis
  if (root) root.Hypothesis = Hypothesis
})(typeof window !== 'undefined' ? window : null)
