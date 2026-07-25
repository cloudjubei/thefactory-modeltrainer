// Research Diagnostician — the no-build browser view. Given a project's completed run corpus + its
// TrainerManifest, it runs a domain-oblivious battery of checks and answers "why hasn't the search found a
// strong candidate, and what to do next". Dual-loaded: window.Diagnostics for the viewer, module.exports for
// src/diagnosticsViewer.test.ts (vitest only scans src/**). Keep it dependency-free at module top level.
;(function (root) {
  'use strict'

  const MIN_SEEDS = 5
  const DISCRIM_FLOOR = 1.5
  const CONFOUND_HIGH = 0.6
  const SEV_RANK = { blocker: 0, caution: 1, info: 2, ok: 3 }
  const CAT_ORDER = [
    'cohort-integrity',
    'objective-discriminability',
    'null-ceiling',
    'split-consistency',
    'incumbent-separation',
    'budget-coverage',
    'objective-confound',
  ]

  // --- primitives -------------------------------------------------------------------------------------
  const esc = (s) =>
    String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c])
  const num = (v) =>
    typeof v === 'number' && isFinite(v) ? v : v != null && v !== '' && isFinite(Number(v)) ? Number(v) : NaN
  const fmt = (v, d = 0) => (isFinite(v) ? Number(v).toFixed(d) : '—')
  const round = (v, d = 2) => (isFinite(v) ? Number(Number(v).toFixed(d)) : v)
  const mean = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : NaN)
  function std(a) {
    if (a.length < 2) return 0
    const m = mean(a)
    return Math.sqrt(a.reduce((s, x) => s + (x - m) * (x - m), 0) / (a.length - 1))
  }
  function iqm(a) {
    const v = a.filter((x) => isFinite(x)).slice().sort((x, y) => x - y)
    if (!v.length) return NaN
    if (v.length < 4) return mean(v)
    const cut = Math.floor(v.length * 0.25)
    return mean(v.slice(cut, v.length - cut))
  }
  function quantile(sorted, q) {
    if (!sorted.length) return NaN
    const s = sorted.slice().sort((a, b) => a - b)
    const pos = (s.length - 1) * q
    const lo = Math.floor(pos)
    const hi = Math.ceil(pos)
    return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (pos - lo)
  }
  const median = (a) => quantile(a.filter((x) => isFinite(x)), 0.5)
  function mulberry(seed) {
    let t = seed >>> 0
    return function () {
      t += 0x6d2b79f5
      let r = Math.imul(t ^ (t >>> 15), 1 | t)
      r ^= r + Math.imul(r ^ (r >>> 7), 61 | r)
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296
    }
  }
  function seedFrom(values) {
    let h = 2166136261
    for (const v of values) {
      h ^= Math.round((isFinite(v) ? v : 0) * 1000) | 0
      h = Math.imul(h, 16777619)
    }
    return h >>> 0
  }
  // deterministic bootstrap CI of the IQM
  function bootstrapCI(values, iters) {
    const v = values.filter((x) => isFinite(x))
    if (v.length < 2) return [v.length ? v[0] : NaN, v.length ? v[0] : NaN]
    const rng = mulberry(seedFrom(v))
    const n = v.length
    const stats = []
    const it = iters || 400
    for (let i = 0; i < it; i++) {
      const s = []
      for (let j = 0; j < n; j++) s.push(v[Math.floor(rng() * n)])
      stats.push(iqm(s))
    }
    stats.sort((a, b) => a - b)
    return [quantile(stats, 0.025), quantile(stats, 0.975)]
  }
  // deterministic bootstrap CI of iqm(a) - iqm(b), oriented so positive = a-better for 'max'
  function bootstrapDiffCI(a, b, iters) {
    const av = a.filter((x) => isFinite(x))
    const bv = b.filter((x) => isFinite(x))
    if (av.length < 2 || bv.length < 2) return [iqm(av) - iqm(bv), iqm(av) - iqm(bv)]
    const rng = mulberry(seedFrom(av.concat(bv)))
    const stats = []
    const it = iters || 400
    for (let i = 0; i < it; i++) {
      const sa = []
      const sb = []
      for (let j = 0; j < av.length; j++) sa.push(av[Math.floor(rng() * av.length)])
      for (let j = 0; j < bv.length; j++) sb.push(bv[Math.floor(rng() * bv.length)])
      stats.push(iqm(sa) - iqm(sb))
    }
    stats.sort((x, y) => x - y)
    return [quantile(stats, 0.025), quantile(stats, 0.975)]
  }
  function spearman(pairs) {
    const p = pairs.filter(([x, y]) => isFinite(x) && isFinite(y))
    if (p.length < 6) return NaN
    const rank = (vals) => {
      const idx = vals.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0])
      const r = new Array(vals.length)
      for (let i = 0; i < idx.length; i++) r[idx[i][1]] = i + 1
      return r
    }
    const rx = rank(p.map((q) => q[0]))
    const ry = rank(p.map((q) => q[1]))
    const dx = rx.map((v, i) => v - ry[i])
    const nn = p.length
    return 1 - (6 * dx.reduce((s, d) => s + d * d, 0)) / (nn * (nn * nn - 1))
  }
  function stableStr(obj) {
    const keys = Object.keys(obj || {}).sort()
    return keys.map((k) => k + '=' + String(obj[k])).join('|')
  }
  const distinct = (arr) => Array.from(new Set(arr))
  const objValue = (run) => num(run && run.objective)
  const metricValue = (run, key) => num(run && run.metrics && run.metrics[key])
  function withoutKeys(config, keys) {
    const c = {}
    for (const k of Object.keys(config || {})) if (!keys.includes(k)) c[k] = config[k]
    return c
  }
  // Exact config key INCLUDING seed — identifies a specific run. Used to drop already-run configs from a
  // recommended campaign so a launch never re-proposes work already on record ("nothing runs, nothing changes").
  function configKeyExact(config) {
    return JSON.stringify(
      Object.keys(config || {})
        .sort()
        .map((k) => [k, String(config[k])]),
    )
  }
  const _archiveCache = new WeakMap()
  function archiveKeySet(runs) {
    const arr = runs || []
    const hit = _archiveCache.get(arr)
    if (hit) return hit
    const s = new Set()
    for (const r of arr) s.add(configKeyExact(r.config || {}))
    _archiveCache.set(arr, s)
    return s
  }
  // Keep only configs not already in the archive AND not duplicated within the batch.
  function freshConfigs(configs, seen) {
    const out = []
    const batch = new Set()
    for (const c of configs) {
      const k = configKeyExact(c.config)
      if (seen.has(k) || batch.has(k)) continue
      batch.add(k)
      out.push(c)
    }
    return out
  }
  const finding = (code, category, severity, verdict, headline, evidence, action) => ({
    code,
    category,
    severity,
    verdict,
    headline,
    evidence: evidence || [],
    recommendedAction: action || null,
  })

  // --- spec resolution --------------------------------------------------------------------------------
  function inferSplitLevers(manifest) {
    const levers = (manifest && manifest.levers) || {}
    return Object.keys(levers).filter(
      (name) => levers[name] && levers[name].scope === 'dataset' && /window|fold|split/i.test(name),
    )
  }
  function resolveSpec(manifest) {
    const m = manifest || {}
    const obj = m.objective || {}
    const direction = obj.direction === 'min' ? 'min' : 'max'
    const hb = m.hypothesisBenchmark
    const dg = m.diagnostics || {}
    let target
    let nullBaseline
    if (dg.target != null) target = Number(dg.target)
    if (hb && hb.metric && obj.name && hb.metric === obj.name) {
      if (target == null && hb.threshold != null) target = Number(hb.threshold)
    } else if (dg.nullBaseline) {
      nullBaseline = dg.nullBaseline
    } else if (hb && hb.metric) {
      nullBaseline = {
        perRunMetric: hb.metric,
        constant: hb.threshold != null ? Number(hb.threshold) : 0,
        direction: hb.direction === 'min' ? 'min' : 'max',
      }
    }
    let splitLevers = []
    if (dg.splitAxis && Array.isArray(dg.splitAxis.levers)) splitLevers = dg.splitAxis.levers.slice()
    else splitLevers = inferSplitLevers(m)
    return {
      objectiveName: obj.name || 'objective',
      direction,
      target,
      nullBaseline,
      splitLevers,
      splitKind: (dg.splitAxis && dg.splitAxis.kind) || (splitLevers.length ? 'dataset' : null),
      confoundMetrics: (dg.confoundMetrics || []).slice(),
      riskMetrics: (dg.riskMetrics || []).slice(),
      degenerateWhen: (dg.degenerateWhen || []).slice(),
      levers: m.levers || {},
      minSeeds: MIN_SEEDS,
    }
  }

  // --- cohort + setups --------------------------------------------------------------------------------
  function matchesDegenerate(run, spec) {
    for (const rule of spec.degenerateWhen || []) {
      const v = metricValue(run, rule.metric)
      if (!isFinite(v)) continue
      const t = rule.metricRef != null ? metricValue(run, rule.metricRef) : Number(rule.value)
      if (rule.op === '==' && v === t) return true
      if (rule.op === '<' && v < t) return true
      if (rule.op === '>' && v > t) return true
    }
    return false
  }
  // partitionCohort + foldSetups are the O(n) hot path (foldSetups JSON-stringifies every run's config) and are
  // called 8+ times over the SAME array per render (each check + the composer + the campaign generators). On a
  // 22k-run project that was ~3s of repeated work that froze the tab. Memoize per (array identity, key-signature)
  // so a whole render partitions the corpus ONCE; a fresh query gives a new array → the WeakMap entry is GC'd.
  const _foldCache = new WeakMap()
  const _partCache = new WeakMap()
  function foldSetups(valid, exclude) {
    const arr = valid || []
    let per = _foldCache.get(arr)
    if (!per) {
      per = new Map()
      _foldCache.set(arr, per)
    }
    const sig = (exclude || []).join(',')
    const hit = per.get(sig)
    if (hit) return hit
    const res = foldSetupsRaw(arr, exclude)
    per.set(sig, res)
    return res
  }
  function foldSetupsRaw(valid, exclude) {
    const groups = new Map()
    for (const r of valid) {
      const k = stableStr(withoutKeys(r.config || {}, exclude))
      if (!groups.has(k)) groups.set(k, { key: k, config: r.config || {}, values: [], runs: [] })
      const g = groups.get(k)
      const v = objValue(r)
      if (isFinite(v)) g.values.push(v)
      g.runs.push(r)
    }
    const out = []
    for (const g of groups.values())
      out.push({ key: g.key, config: g.config, values: g.values, iqm: iqm(g.values), seeds: g.values.length, runs: g.runs })
    return out
  }
  function partitionCohort(records, spec) {
    const all = records || []
    let per = _partCache.get(all)
    if (!per) {
      per = new Map()
      _partCache.set(all, per)
    }
    const sig = spec.objectiveName + '|' + spec.direction + '|' + spec.minSeeds + '|' + JSON.stringify(spec.degenerateWhen || [])
    const hit = per.get(sig)
    if (hit) return hit
    const res = partitionCohortRaw(all, spec)
    per.set(sig, res)
    return res
  }
  function partitionCohortRaw(records, spec) {
    const all = records || []
    let failed = 0
    let degenerate = 0
    let invalid = 0
    const valid = []
    let completed = 0
    for (const r of all) {
      const status = r.status || 'completed'
      if (status === 'invalid') {
        invalid++
        continue
      }
      if (status !== 'completed') {
        failed++
        continue
      }
      completed++
      const h = r.health
      if ((h && h.status && h.status !== 'ok') || matchesDegenerate(r, spec) || !isFinite(objValue(r))) {
        degenerate++
        continue
      }
      valid.push(r)
    }
    const setups = foldSetups(valid, ['seed'])
    const decisionGradeN = setups.filter((s) => s.seeds >= spec.minSeeds).length
    return { total: all.length, completed, failed, invalid, degenerate, valid, setups, decisionGradeN }
  }
  function bestSetup(setups, direction) {
    return setups
      .filter((s) => isFinite(s.iqm))
      .slice()
      .sort((a, b) => (direction === 'max' ? b.iqm - a.iqm : a.iqm - b.iqm))[0]
  }
  function splitKey(run, levers) {
    return levers.map((l) => String((run.config || {})[l])).join('|')
  }
  function splitValObj(run, levers) {
    const o = {}
    for (const l of levers) o[l] = (run.config || {})[l]
    return o
  }
  // does a group of runs beat the declared null (per-run metric vs constant), or fall back to positive objective
  function beatsNullOn(runs, spec) {
    const nb = spec.nullBaseline
    if (nb) {
      const scores = runs.map((r) => (nb.perRunMetric ? metricValue(r, nb.perRunMetric) : objValue(r))).filter(isFinite)
      if (!scores.length) return false
      const c = nb.constant != null ? nb.constant : 0
      return nb.direction === 'min' ? iqm(scores) < c : iqm(scores) > c
    }
    return false
  }

  // --- the 7 checks -----------------------------------------------------------------------------------
  function checkCohortIntegrity(records, spec, cohort) {
    cohort = cohort || partitionCohort(records, spec)
    if (!cohort.valid.length)
      return finding(
        'cohort.empty',
        'cohort-integrity',
        'blocker',
        'no-runs',
        'No decision-grade runs — the corpus is empty or every run failed/degenerate/invalid.',
        [
          { stat: 'total', value: cohort.total },
          { stat: 'failed', value: cohort.failed },
          { stat: 'degenerate', value: cohort.degenerate },
        ],
        null,
      )
    const bad = cohort.failed + cohort.degenerate + cohort.invalid
    const badFrac = bad / Math.max(cohort.total, 1)
    const verdict = cohort.decisionGradeN === 0 ? 'thin' : badFrac > 0.3 ? 'high-waste' : 'ok'
    const severity = cohort.decisionGradeN === 0 ? 'caution' : badFrac > 0.3 ? 'caution' : 'ok'
    return finding(
      'cohort.' + verdict,
      'cohort-integrity',
      severity,
      verdict,
      `${cohort.valid.length} usable of ${cohort.total} runs · ${cohort.decisionGradeN} setups seeded ≥${spec.minSeeds}${bad ? ` · ${bad} failed/degenerate/invalid` : ''}.`,
      [
        { stat: 'decision-grade setups', value: cohort.decisionGradeN },
        { stat: 'failed+degenerate+invalid', value: bad },
      ],
      badFrac > 0.3
        ? { control: 'invalidate', text: 'A large fraction of runs are degenerate/failed — steer the search out of that region to reclaim budget.' }
        : null,
    )
  }
  function checkDiscriminability(records, spec) {
    const { setups } = partitionCohort(records, spec)
    if (setups.length < 2)
      return finding('discrim.single', 'objective-discriminability', 'info', 'single-setup', 'Too few setups to assess discriminability.', [])
    const seedCounts = setups.map((s) => s.seeds)
    const medSeeds = median(seedCounts)
    const multi = setups.filter((s) => s.seeds >= 2)
    const within = multi.length ? median(multi.map((s) => std(s.values))) : 0
    const between = std(setups.map((s) => s.iqm))
    const ratio = within > 0 ? between / within : setups.length > 1 ? Infinity : 0
    if (medSeeds < spec.minSeeds)
      return finding(
        'discrim.under-seeded',
        'objective-discriminability',
        'caution',
        'under-seeded',
        `Median ${fmt(medSeeds, 0)} seeds/setup (< ${spec.minSeeds}) — a single-run score can't be told from luck.`,
        [{ stat: 'median seeds', value: round(medSeeds, 1) }, { stat: 'S/N ratio', value: round(ratio, 2) }],
        { control: 'reseed', text: `Add seeds to the promising setups (target ≥${spec.minSeeds}).` },
      )
    if (ratio < DISCRIM_FLOOR)
      return finding(
        'discrim.noisy',
        'objective-discriminability',
        'caution',
        'noisy-objective',
        `Config spread (${fmt(between, 2)}) barely exceeds seed noise (${fmt(within, 2)}) — ranking is unreliable.`,
        [{ stat: 'S/N ratio', value: round(ratio, 2) }],
        { control: 'reseed', text: 'Add seeds, or use a less noisy objective.' },
      )
    return finding(
      'discrim.ok',
      'objective-discriminability',
      'ok',
      'discriminates',
      `Objective separates configs ${isFinite(ratio) ? fmt(ratio, 1) + '×' : 'well'} above seed noise.`,
      [{ stat: 'S/N ratio', value: round(ratio, 2) }, { stat: 'median seeds', value: round(medSeeds, 1) }],
      null,
    )
  }
  function checkNullCeiling(records, spec) {
    const { setups } = partitionCohort(records, spec)
    if (!setups.length) return finding('null.no-runs', 'null-ceiling', 'info', 'no-runs', 'No completed runs to evaluate.', [])
    const inc = bestSetup(setups, spec.direction)
    if (spec.target != null) {
      const ci = bootstrapCI(inc.values)
      const reached = spec.direction === 'max' ? ci[1] >= spec.target || inc.iqm >= spec.target : ci[0] <= spec.target || inc.iqm <= spec.target
      if (reached)
        return finding(
          'null.at-ceiling',
          'null-ceiling',
          'ok',
          'at-ceiling',
          `Incumbent reaches the declared target (${fmt(spec.target, 1)}) — the search is solved, not stuck.`,
          [{ stat: 'incumbent IQM', value: round(inc.iqm, 2) }, { stat: 'target', value: spec.target }],
          { control: 'change-objective', text: 'At the ceiling — declare success or raise the bar.' },
        )
    }
    if (spec.nullBaseline) {
      const nb = spec.nullBaseline
      const c = nb.constant != null ? nb.constant : 0
      const scores = inc.runs.map((r) => (nb.perRunMetric ? metricValue(r, nb.perRunMetric) : objValue(r))).filter(isFinite)
      if (scores.length >= 2) {
        const nci = bootstrapCI(scores)
        const beats = nb.direction === 'min' ? nci[1] < c : nci[0] > c
        const label = nb.perRunMetric || 'objective'
        if (beats)
          return finding(
            'null.beats',
            'null-ceiling',
            'ok',
            'beats-null',
            `Incumbent beats the null (${label} ${nb.direction === 'min' ? '<' : '>'} ${c}).`,
            [{ stat: label, value: round(iqm(scores), 2), ci: nci.map((x) => round(x, 2)) }],
            null,
          )
        return finding(
          'null.overlaps',
          'null-ceiling',
          'blocker',
          'no-signal',
          `Best result is not distinguishable from the null (${label} vs ${c}) — no candidate beats the baseline yet.`,
          [{ stat: label, value: round(iqm(scores), 2), ci: nci.map((x) => round(x, 2)), comparator: 'overlaps', threshold: c }],
          { control: 'change-objective', text: 'Change the approach (model / features / data), not just seeds.' },
        )
      }
    }
    return finding(
      'null.undeclared',
      'null-ceiling',
      'info',
      'no-null',
      'No null baseline or ceiling declared — add a hypothesisBenchmark or diagnostics.nullBaseline to test whether the best result beats a baseline.',
      [{ stat: 'incumbent IQM', value: round(inc.iqm, 2) }],
      null,
    )
  }
  function checkSplitConsistency(records, spec) {
    if (!spec.splitLevers.length)
      return finding(
        'split.unverifiable',
        'split-consistency',
        'info',
        'unverifiable',
        'No split axis declared — generalization across markets/folds/windows is not assessable. Declare diagnostics.splitAxis to enable this check.',
        [],
        null,
      )
    const { valid } = partitionCohort(records, spec)
    const exclude = ['seed'].concat(spec.splitLevers)
    const bases = foldSetups(valid, exclude)
    if (!bases.length) return finding('split.no-runs', 'split-consistency', 'info', 'no-runs', 'No runs.', [])
    const inc = bestSetup(bases, spec.direction)
    const splitVals = distinct(valid.map((r) => splitKey(r, spec.splitLevers)))
    let evaluated = 0
    let beating = 0
    for (const sv of splitVals) {
      const rs = inc.runs.filter((r) => splitKey(r, spec.splitLevers) === sv)
      if (!rs.length) continue
      evaluated++
      if (beatsNullOn(rs, spec) || (!spec.nullBaseline && aboveField(rs, bases, spec))) beating++
    }
    if (evaluated < 2)
      return finding(
        'split.not-replicated',
        'split-consistency',
        'caution',
        'not-replicated',
        `The incumbent was evaluated on only ${evaluated} of ${splitVals.length} splits — replicate it across the rest before trusting it.`,
        [{ stat: 'splits evaluated', value: evaluated, threshold: splitVals.length }],
        { control: 'split-fill', text: 'Replicate the incumbent across the remaining splits.' },
      )
    if (beating < evaluated)
      return finding(
        'split.single-split-luck',
        'split-consistency',
        'blocker',
        'single-split-luck',
        `The incumbent beats the null in only ${beating}/${evaluated} splits — single-split luck, not a robust edge.`,
        [{ stat: 'splits beating null', value: beating, threshold: evaluated }],
        { control: 'split-fill', text: 'Re-rank by worst-split objective; require consistency across splits before crowning a winner.' },
      )
    return finding(
      'split.robust',
      'split-consistency',
      'ok',
      'robust',
      `The incumbent holds across all ${evaluated} evaluated splits.`,
      [{ stat: 'splits beating null', value: beating, threshold: evaluated }],
      null,
    )
  }
  function aboveField(rs, bases, spec) {
    const med = median(bases.map((b) => b.iqm))
    const v = iqm(rs.map((r) => objValue(r)))
    return spec.direction === 'min' ? v < med : v > med
  }
  function checkIncumbentSeparation(records, spec) {
    const { setups } = partitionCohort(records, spec)
    if (setups.length < 2)
      return finding('sep.single', 'incumbent-separation', 'info', 'single-setup', 'Only one setup — nothing to separate.', [])
    const ranked = setups
      .filter((s) => isFinite(s.iqm))
      .slice()
      .sort((a, b) => (spec.direction === 'max' ? b.iqm - a.iqm : a.iqm - b.iqm))
    const first = ranked[0]
    const second = ranked[1]
    const diffCI = bootstrapDiffCI(first.values, second.values)
    const distinguishable = spec.direction === 'max' ? diffCI[0] > 0 : diffCI[1] < 0
    if (distinguishable)
      return finding(
        'sep.distinguishable',
        'incumbent-separation',
        'ok',
        'distinguishable',
        'The top setup is distinguishable from #2 (its confidence interval clears it).',
        [{ stat: '#1−#2 diff', ci: diffCI.map((x) => round(x, 3)) }],
        null,
      )
    if (spec.target != null) {
      const near =
        spec.direction === 'max' ? first.iqm >= spec.target && second.iqm >= spec.target : first.iqm <= spec.target && second.iqm <= spec.target
      if (near)
        return finding(
          'sep.multi-optima',
          'incumbent-separation',
          'ok',
          'multiple-optima',
          'Several configs reach the ceiling — pick any; there is no single winner to separate.',
          [{ stat: '#1−#2 diff', ci: diffCI.map((x) => round(x, 3)) }],
          null,
        )
    }
    return finding(
      'sep.tie',
      'incumbent-separation',
      'caution',
      'no-clear-winner',
      "#1 and #2 overlap — there's no distinguishable winner yet.",
      [{ stat: '#1−#2 diff', ci: diffCI.map((x) => round(x, 3)) }],
      { control: 'reseed', text: 'Add seeds to the tied top setups before crowning a champion.' },
    )
  }
  function checkBudgetCoverage(records, spec) {
    const { valid } = partitionCohort(records, spec)
    if (!valid.length) return finding('cov.no-runs', 'budget-coverage', 'info', 'no-runs', 'No runs.', [])
    const tunable = Object.keys(spec.levers).filter(
      (n) => n !== 'seed' && spec.levers[n] && spec.levers[n].scope !== 'ignore' && (spec.levers[n].active === undefined || spec.levers[n].active),
    )
    const constant = tunable.filter((n) => distinct(valid.map((r) => String((r.config || {})[n]))).length <= 1)
    if (constant.length)
      return finding(
        'cov.constant-levers',
        'budget-coverage',
        'caution',
        'constant-levers',
        `${constant.length} tunable lever${constant.length === 1 ? '' : 's'} never varied (${constant.slice(0, 4).join(', ')}${constant.length > 4 ? '…' : ''}) — a whole region of the space is unexplored.`,
        [{ stat: 'constant levers', value: constant.length }],
        { control: 'widen', text: `Open ${constant.slice(0, 3).join(', ')} in a sweep.` },
      )
    return finding('cov.ok', 'budget-coverage', 'ok', 'covered', 'Every tunable lever has been varied.', [{ stat: 'tunable levers', value: tunable.length }])
  }
  function checkObjectiveConfound(records, spec) {
    const { valid } = partitionCohort(records, spec)
    if (!valid.length || !spec.confoundMetrics.length)
      return finding('confound.none', 'objective-confound', 'info', 'not-declared', 'No confound metrics declared to screen the objective against.', [])
    let worst = null
    for (const cm of spec.confoundMetrics) {
      const r = spearman(valid.map((run) => [objValue(run), metricValue(run, cm)]))
      if (isFinite(r) && (!worst || Math.abs(r) > Math.abs(worst.r))) worst = { metric: cm, r }
    }
    if (worst && Math.abs(worst.r) >= CONFOUND_HIGH)
      return finding(
        'confound.high',
        'objective-confound',
        'caution',
        'confounded',
        `The objective correlates ${fmt(worst.r, 2)} with ${worst.metric} — it may be rewarding a proxy, not skill.`,
        [{ stat: 'spearman', value: round(worst.r, 2), comparator: '≥', threshold: CONFOUND_HIGH }],
        { control: 'change-objective', text: `Penalize / control for ${worst.metric}, or add it as a second Pareto axis.` },
      )
    return finding('confound.ok', 'objective-confound', 'ok', 'clean', 'The objective does not track the declared confound metrics.', [
      worst ? { stat: 'max |spearman|', value: round(Math.abs(worst.r), 2) } : { stat: 'confounds', value: spec.confoundMetrics.length },
    ])
  }

  // --- composer ---------------------------------------------------------------------------------------
  function orderFindings(findings) {
    return findings.slice().sort((a, b) => SEV_RANK[a.severity] - SEV_RANK[b.severity] || CAT_ORDER.indexOf(a.category) - CAT_ORDER.indexOf(b.category))
  }
  function incumbentOf(cohort, spec) {
    if (!cohort.setups.length) return null
    const inc = bestSetup(cohort.setups, spec.direction)
    if (!inc) return null
    return { config: withoutKeys(inc.config, ['seed']), iqm: round(inc.iqm, 3), seeds: inc.seeds, ci: bootstrapCI(inc.values).map((x) => round(x, 3)) }
  }
  function deriveVerdict(findings) {
    const block = findings.find((f) => f.severity === 'blocker')
    if (block) {
      if (block.category === 'split-consistency') return 'single-split-luck'
      if (block.category === 'null-ceiling') return 'no-signal'
      if (block.category === 'cohort-integrity') return 'under-powered'
      if (block.category === 'objective-confound') return 'confounded'
      return 'stalled'
    }
    const nc = findings.find((f) => f.category === 'null-ceiling')
    if (nc && nc.verdict === 'at-ceiling') return 'converged'
    const sep = findings.find((f) => f.category === 'incumbent-separation')
    const beats = nc && nc.verdict === 'beats-null'
    if (sep && (sep.verdict === 'distinguishable' || sep.verdict === 'multiple-optima')) return beats ? 'converged' : 'winner-emerging'
    const disc = findings.find((f) => f.category === 'objective-discriminability')
    if (disc && (disc.verdict === 'under-seeded' || disc.verdict === 'noisy-objective')) return 'under-powered'
    return 'stalled'
  }
  const VERDICT_NEXT = {
    'single-split-luck': 'Replicate the shortlist across every split and re-rank by the worst split.',
    'no-signal': 'No candidate beats the baseline — change the approach (model / features / data), not the seeds.',
    'under-powered': 'Add seeds to the promising setups so real differences surface.',
    confounded: 'Fix the objective (control the confound / add a risk term) before trusting the ranking.',
    converged: 'Converged — declare the winner or raise the bar.',
    'winner-emerging': 'A winner is emerging — replicate it to confirm, then declare.',
    stalled: 'Widen coverage into the unexplored levers.',
  }
  function deriveHeadline(findings, verdict) {
    const driver = findings.find((f) => f.severity === 'blocker') || findings.find((f) => f.severity === 'caution') || findings[0]
    const good = verdict === 'converged' || verdict === 'winner-emerging'
    return {
      stalledBecause: good ? '' : driver ? driver.headline : '',
      doNext: VERDICT_NEXT[verdict] || 'Review the findings below.',
    }
  }
  // Cross-asset robustness from the `-settest` matrices (kept checkpoints replayed on other assets —
  // cheap OOS evidence, no retraining): does the INCUMBENT's edge travel to markets it never trained on?
  // Complements split-consistency (across windows) with the across-assets axis.
  function checkCrossAssetRobustness(records, spec, settests) {
    const matrices = Array.isArray(settests) ? settests : []
    if (!matrices.length)
      return finding(
        'crossasset.unverifiable',
        'cross-asset',
        'info',
        'unverifiable',
        'No cross-test matrices — replay kept checkpoints on other assets (a run’s Cross-test section, or “Cross-test after training” in Launch) to verify the edge travels beyond its training market.',
        [],
        null,
      )
    const { valid } = partitionCohort(records, spec)
    const setups = foldSetups(valid, ['seed'])
    const inc = bestSetup(setups, spec.direction)
    if (!inc) return finding('crossasset.no-runs', 'cross-asset', 'info', 'no-runs', 'No runs.', [])
    const incKeys = new Set(inc.runs.map((r) => r.key))
    const cells = []
    for (const m of matrices) {
      if (!m || !incKeys.has(m.runKey)) continue
      const byValue = (m.levers || {}).asset || {}
      for (const value of Object.keys(byValue)) cells.push(byValue[value])
    }
    const completed = cells.filter((c) => c && c.status === 'completed')
    if (!completed.length)
      return finding(
        'crossasset.not-cross-tested',
        'cross-asset',
        'caution',
        'not-cross-tested',
        'The incumbent has no cross-asset evidence — cross-test its checkpoint before trusting the edge to travel.',
        [],
        { control: 'cross-test', text: 'Cross-test the incumbent’s runs on the other assets.' },
      )
    const beating = completed.filter((c) => Number(c.returnVsHold) > 0).length
    if (beating === completed.length)
      return finding(
        'crossasset.robust',
        'cross-asset',
        'ok',
        'robust',
        `The incumbent beats buy-and-hold on all ${completed.length} cross-tested asset cell${completed.length === 1 ? '' : 's'} — the edge travels.`,
        [{ stat: 'asset cells beating hold', value: beating, threshold: completed.length }],
        null,
      )
    if (beating === 0)
      return finding(
        'crossasset.asset-bound',
        'cross-asset',
        'blocker',
        'asset-bound',
        `The incumbent beats buy-and-hold on 0/${completed.length} other assets — it learned its training market, not a general edge.`,
        [{ stat: 'asset cells beating hold', value: 0, threshold: completed.length }],
        { control: 'cross-test', text: 'Treat the incumbent as asset-specific; re-rank candidates by worst cross-asset result.' },
      )
    return finding(
      'crossasset.partial',
      'cross-asset',
      'caution',
      'partial',
      `The incumbent beats buy-and-hold on ${beating}/${completed.length} cross-tested asset cells — partially transferable, weight the misses before declaring a winner.`,
      [{ stat: 'asset cells beating hold', value: beating, threshold: completed.length }],
      { control: 'cross-test', text: 'Cross-test the remaining assets and prefer candidates that hold everywhere.' },
    )
  }

  function diagnose(data) {
    const manifest = (data && data.manifest) || {}
    const spec = resolveSpec(manifest)
    const records = (data && data.runs) || []
    const cohort = partitionCohort(records, spec)
    const cohortFinding = checkCohortIntegrity(records, spec, cohort)
    if (!cohort.valid.length)
      return { cohort, incumbent: null, splitAxis: spec.splitLevers, findings: [cohortFinding], verdict: 'under-powered', headline: deriveHeadline([cohortFinding], 'under-powered'), spec }
    const findings = [
      cohortFinding,
      checkDiscriminability(records, spec),
      checkNullCeiling(records, spec),
      checkSplitConsistency(records, spec),
      checkIncumbentSeparation(records, spec),
      checkBudgetCoverage(records, spec),
      checkObjectiveConfound(records, spec),
    ]
    // Cross-asset evidence is optional input — only assessed when the host supplies the settest matrices.
    if (Array.isArray(data && data.settests)) {
      findings.push(checkCrossAssetRobustness(records, spec, data.settests))
    }
    const ordered = orderFindings(findings)
    const verdict = deriveVerdict(ordered)
    return { cohort, incumbent: incumbentOf(cohort, spec), splitAxis: spec.splitLevers, findings: ordered, verdict, headline: deriveHeadline(ordered, verdict), spec }
  }

  // --- campaign generators ----------------------------------------------------------------------------
  // reseed the top-N promising setups up to targetSeeds — one campaign, only the MISSING seeds.
  function reseedSpecs(opts) {
    const spec = opts.spec || resolveSpec(opts.manifest)
    const { setups } = partitionCohort(opts.runs || [], spec)
    const topN = opts.topN || 12
    const targetSeeds = opts.targetSeeds || MIN_SEEDS
    const promising = setups
      .filter((s) => isFinite(s.iqm))
      .slice()
      .sort((a, b) => (spec.direction === 'max' ? b.iqm - a.iqm : a.iqm - b.iqm))
      .slice(0, topN)
    const configs = []
    for (const s of promising) {
      if (s.seeds >= targetSeeds) continue
      const used = new Set(s.runs.map((r) => (r.config || {}).seed))
      const base = withoutKeys(s.config, ['seed'])
      let seed = 0
      let added = 0
      while (s.seeds + added < targetSeeds && seed < 100000) {
        if (!used.has(seed)) {
          configs.push({ config: Object.assign({}, base, { seed }) })
          added++
        }
        seed++
      }
    }
    const fresh = freshConfigs(configs, archiveKeySet(opts.runs))
    return fresh.length ? [{ configs: fresh }] : []
  }
  // replicate the shortlist across every split value — ONE campaign per split (efficient: shared data bundle).
  function replicateSpecs(opts) {
    const spec = opts.spec || resolveSpec(opts.manifest)
    if (!spec.splitLevers.length) return []
    const { valid } = partitionCohort(opts.runs || [], spec)
    const topN = opts.topN || 10
    const seeds = opts.seeds && opts.seeds.length ? opts.seeds : [0, 1, 2, 3, 4]
    const exclude = ['seed'].concat(spec.splitLevers)
    const bases = foldSetups(valid, exclude)
    const shortlist = bases
      .filter((s) => isFinite(s.iqm))
      .slice()
      .sort((a, b) => (spec.direction === 'max' ? b.iqm - a.iqm : a.iqm - b.iqm))
      .slice(0, topN)
    const seen = new Set()
    const splitVals = []
    for (const r of valid) {
      const k = splitKey(r, spec.splitLevers)
      if (seen.has(k)) continue
      seen.add(k)
      splitVals.push(splitValObj(r, spec.splitLevers))
    }
    const archived = archiveKeySet(opts.runs)
    const specs = []
    for (const sv of splitVals) {
      const configs = []
      for (const base of shortlist) {
        const baseCfg = withoutKeys(base.config, exclude)
        for (const seed of seeds) configs.push({ config: Object.assign({}, baseCfg, sv, { seed }) })
      }
      const fresh = freshConfigs(configs, archived)
      if (fresh.length) specs.push({ configs: fresh })
    }
    return specs
  }

  // --- render -----------------------------------------------------------------------------------------
  let stylesInjected = false
  function injectStyles() {
    if (stylesInjected || typeof document === 'undefined') return
    stylesInjected = true
    const css = `
    .diag{--d-line:#e2e7ef;--d-panel:#fff;--d-panel2:#f6f8fb;--d-text:#141d29;--d-muted:#5c6a7e;--d-faint:#8a97a9;--d-mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--d-bad:#c23b3b;--d-badbg:#fbeceb;--d-warn:#b9791b;--d-warnbg:#f8efdb;--d-ok:#12936a;--d-okbg:#e6f4ee;--d-info:#3a6ea5;--d-infobg:#eaf1f8;color:var(--d-text);padding:20px 22px 60px;font-feature-settings:"tnum" 1}
    @media (prefers-color-scheme:dark){.diag{--d-line:#1f2a3b;--d-panel:#111725;--d-panel2:#0d121d;--d-text:#dbe4f1;--d-muted:#8492a8;--d-faint:#5c6a80;--d-bad:#e0736e;--d-badbg:#2c1a18;--d-warn:#e0a94a;--d-warnbg:#2b2314;--d-ok:#4bd4a0;--d-okbg:#16281f;--d-info:#7fb1e0;--d-infobg:#15202e}}
    .diag h3{margin:0 0 4px;font-size:15px}
    .diag .d-verdict{border:1px solid var(--d-line);border-left:4px solid var(--d-accent,#b7791f);border-radius:12px;background:var(--d-panel);padding:18px 20px;margin-bottom:18px}
    .diag .d-verdict.good{border-left-color:var(--d-ok)} .diag .d-verdict.bad{border-left-color:var(--d-bad)} .diag .d-verdict.warn{border-left-color:var(--d-warn)}
    .diag .d-vtag{font-family:var(--d-mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--d-muted)}
    .diag .d-vtitle{font-size:20px;font-weight:700;margin:6px 0 4px}
    .diag .d-next{color:var(--d-muted);margin:0}
    .diag .d-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:1px;background:var(--d-line);border:1px solid var(--d-line);border-radius:10px;overflow:hidden;margin-bottom:18px}
    .diag .d-stat{background:var(--d-panel);padding:12px 14px}
    .diag .d-stat .v{font-family:var(--d-mono);font-size:20px;font-weight:700}
    .diag .d-stat .l{font-size:11.5px;color:var(--d-faint);margin-top:2px}
    .diag .d-cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;margin-bottom:20px}
    .diag .d-card{background:var(--d-panel);border:1px solid var(--d-line);border-radius:10px;padding:14px 15px}
    .diag .d-card .d-top{display:flex;align-items:center;gap:8px;margin-bottom:7px}
    .diag .d-sev{font-family:var(--d-mono);font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:2px 7px;border-radius:5px}
    .diag .d-sev.blocker{background:var(--d-badbg);color:var(--d-bad)} .diag .d-sev.caution{background:var(--d-warnbg);color:var(--d-warn)} .diag .d-sev.ok{background:var(--d-okbg);color:var(--d-ok)} .diag .d-sev.info{background:var(--d-infobg);color:var(--d-info)}
    .diag .d-card .d-cat{font-size:13px;font-weight:650;flex:1}
    .diag .d-card p{font-size:13px;color:var(--d-muted);margin:0 0 8px;line-height:1.5}
    .diag .d-ev{font-family:var(--d-mono);font-size:11.5px;background:var(--d-panel2);border:1px solid var(--d-line);border-radius:6px;padding:6px 8px;color:var(--d-text)}
    .diag .d-actions{display:flex;flex-wrap:wrap;gap:10px;margin:6px 0 22px}
    .diag .d-btn{font:inherit;font-size:13px;font-weight:600;cursor:pointer;border:1px solid var(--d-line);background:var(--d-panel);color:var(--d-text);border-radius:8px;padding:9px 14px}
    .diag .d-btn:hover{border-color:var(--d-muted)} .diag .d-btn:disabled{opacity:.5;cursor:default}
    .diag .d-btn small{display:block;font-weight:400;color:var(--d-faint);font-size:11px;margin-top:1px}
    .diag .d-sec{font-family:var(--d-mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--d-faint);margin:8px 0 10px;border-bottom:1px solid var(--d-line);padding-bottom:6px}
    .diag .d-empty{color:var(--d-faint);padding:24px 0}
    .diag .d-verdict .d-btn{margin-top:12px}
    .diag .d-plan{list-style:none;margin:0 0 22px;padding:0;display:flex;flex-direction:column;gap:10px}
    .diag .d-step{background:var(--d-panel);border:1px solid var(--d-line);border-left:3px solid var(--d-warn);border-radius:10px;padding:13px 15px}
    .diag .d-step.done{border-left-color:var(--d-ok);opacity:.75}
    .diag .d-step.strategic{border-left-color:var(--d-info);background:var(--d-infobg)}
    .diag .d-step-h{display:flex;align-items:center;gap:9px;margin-bottom:6px}
    .diag .d-stepn{font-family:var(--d-mono);font-size:12px;font-weight:700;color:var(--d-faint);min-width:16px}
    .diag .d-step-t{font-size:13.5px;font-weight:600;flex:1}
    .diag .d-step-a{margin-top:9px}
    .diag .d-done{font-size:13px;color:var(--d-ok);font-weight:600}
    .diag .d-note{font-size:12.5px;color:var(--d-muted);line-height:1.5;margin:0 0 8px}`
    const el = document.createElement('style')
    el.textContent = css
    document.head.appendChild(el)
  }
  const VERDICT_STYLE = {
    converged: ['good', 'Converged'],
    'winner-emerging': ['good', 'Winner emerging'],
    'single-split-luck': ['bad', 'Single-split luck'],
    'no-signal': ['bad', 'No signal'],
    'under-powered': ['warn', 'Under-powered'],
    confounded: ['warn', 'Confounded objective'],
    stalled: ['warn', 'Stalled'],
  }
  // Turn the diagnosis into an ORDERED, self-guided PLAN. Each actionable finding (blocker/caution) becomes a
  // step whose launchable action reports how many GENUINELY-NEW runs it would add (deduped against the archive) —
  // a step whose action adds 0 new runs is marked DONE, so the plan never re-recommends work already on record and
  // naturally advances toward the strategic "discuss with the AI" step once the mechanical steps are exhausted.
  function planSteps(d, campaigns) {
    const reseedN = campaigns.reseed.reduce((n, s) => n + s.configs.length, 0)
    const replN = campaigns.replicate.reduce((n, s) => n + s.configs.length, 0)
    const replCamps = campaigns.replicate.length
    const splitLabel = (d.splitAxis || []).join('+') || 'split'
    const steps = []
    for (const f of d.findings) {
      if (f.severity === 'ok' || f.severity === 'info') continue
      const step = { severity: f.severity, category: f.category, title: f.headline, evidence: f.evidence, action: { kind: 'discuss' }, newRuns: 0, done: false }
      if (
        (f.category === 'objective-discriminability' && (f.verdict === 'under-seeded' || f.verdict === 'noisy-objective')) ||
        (f.category === 'incumbent-separation' && f.verdict === 'no-clear-winner')
      ) {
        step.action = { kind: 'reseed', label: `Reseed the promising setups · ${reseedN} new run${reseedN === 1 ? '' : 's'}` }
        step.newRuns = reseedN
        step.done = reseedN === 0
        step.doneText = `Every promising setup is already seeded ≥${MIN_SEEDS}.`
      } else if (f.category === 'split-consistency' && (f.verdict === 'single-split-luck' || f.verdict === 'not-replicated')) {
        step.action = { kind: 'replicate', label: `Replicate the shortlist across every ${splitLabel} · ${replN} new run${replN === 1 ? '' : 's'} (${replCamps} campaign${replCamps === 1 ? '' : 's'})` }
        step.newRuns = replN
        step.done = replN === 0
        step.doneText = `The shortlist is already run across every ${splitLabel} — since it still doesn't generalize, this is now a strategic question, not a coverage gap.`
      } else if (f.category === 'budget-coverage' && f.verdict === 'constant-levers') {
        step.action = { kind: 'open-levers', label: 'Open the unexplored levers' }
      } else if (f.category === 'null-ceiling' && f.verdict === 'no-signal') {
        step.action = { kind: 'discuss', label: 'Rethink the model / features / data with the AI' }
      } else if (f.category === 'objective-confound' && f.verdict === 'confounded') {
        step.action = { kind: 'discuss', label: 'Fix the objective with the AI' }
      } else if (f.category === 'cohort-integrity') {
        step.action = { kind: 'discuss', label: 'Steer the search away from degenerate configs' }
      }
      steps.push(step)
    }
    return steps
  }

  // Pure, worker-safe compute: run the diagnosis + campaign generation + plan, and return a SLIM result (cohort
  // as counts, not the 20k run-object arrays) so it is cheap to structured-clone across a Web Worker boundary.
  // No DOM/globals here — the heavy work (partition/fold/campaigns) runs off the main thread; `paint` builds the
  // DOM from this result. `render` (analyze+paint) stays the synchronous entry for tests + a non-worker fallback.
  function analyze(data) {
    const dFull = diagnose(data)
    const runs = (data && data.runs) || []
    const spec = dFull.spec
    const campaigns = {
      reseed: reseedSpecs({ runs, manifest: data.manifest, spec, topN: 12, targetSeeds: MIN_SEEDS }),
      replicate: replicateSpecs({ runs, manifest: data.manifest, spec, topN: 10 }),
    }
    const steps = planSteps(dFull, campaigns)
    const d = {
      verdict: dFull.verdict,
      headline: dFull.headline,
      splitAxis: dFull.splitAxis,
      spec: { objectiveName: spec.objectiveName, direction: spec.direction, splitLevers: spec.splitLevers, minSeeds: spec.minSeeds },
      cohort: dFull.cohort
        ? {
            total: dFull.cohort.total,
            completed: dFull.cohort.completed,
            failed: dFull.cohort.failed,
            degenerate: dFull.cohort.degenerate,
            invalid: dFull.cohort.invalid,
            validCount: dFull.cohort.valid.length,
            decisionGradeN: dFull.cohort.decisionGradeN,
          }
        : null,
      incumbent: dFull.incumbent,
      findings: dFull.findings,
    }
    return { d, campaigns, steps }
  }

  function render(container, data, actions) {
    paint(container, analyze(data), data, actions)
  }

  // An instant, styled placeholder painted BEFORE the corpus loads / the analysis runs, so the tab is never a
  // frozen blank on a big project. `#diag-progress` is updated with scan/analyze progress by the host.
  function shell(container, message) {
    injectStyles()
    container.innerHTML = `<div class="diag">
      <div class="d-verdict warn">
        <div class="d-vtag">Diagnosis</div>
        <div class="d-vtitle">Analyzing the run corpus…</div>
        <div id="diag-progress" class="d-next">${esc(message || 'Loading runs…')}</div>
      </div>
      <div class="d-sec">Your plan — work through these in order</div>
      <div class="d-empty">The plan appears here once the runs are analyzed. This runs off the main thread, so the rest of the app stays responsive.</div>
    </div>`
  }

  function paint(container, analysis, data, actions) {
    injectStyles()
    const d = analysis.d
    const campaigns = analysis.campaigns
    const steps = analysis.steps
    const canDiscuss = !!(actions && actions.canDiscuss)
    const vs = VERDICT_STYLE[d.verdict] || ['warn', d.verdict]

    const stat = (v, l) => `<div class="d-stat"><div class="v">${esc(v)}</div><div class="l">${esc(l)}</div></div>`
    const stats = d.cohort
      ? `<div class="d-stats">
          ${stat(d.cohort.total, 'runs on record')}
          ${stat(d.cohort.validCount, 'decision-grade')}
          ${stat(d.cohort.decisionGradeN, `setups seeded ≥${MIN_SEEDS}`)}
          ${stat(d.incumbent ? fmt(d.incumbent.iqm, 2) : '—', 'incumbent (IQM)')}
        </div>`
      : ''
    const evHtml = (evidence) =>
      (evidence || [])
        .map((e) => `${esc(e.stat)}: <b>${esc(e.value != null ? e.value : e.ci ? '[' + e.ci.join(', ') + ']' : '')}</b>${e.ci && e.value != null ? ' [' + e.ci.join(', ') + ']' : ''}${e.threshold != null ? ' vs ' + esc(e.threshold) : ''}`)
        .join(' · ')

    const discussBtn = (label) => (canDiscuss ? `<button class="d-btn" data-diag-act="discuss">${esc(label)}</button>` : '')
    const stepAction = (s) => {
      if (s.done) return `<div class="d-done">✓ ${esc(s.doneText || 'Already done — no new runs to add.')}</div>`
      if (s.action.kind === 'reseed' || s.action.kind === 'replicate')
        return `<button class="d-btn" data-diag-act="${s.action.kind}">${esc(s.action.label)}</button>`
      if (s.action.kind === 'open-levers')
        return `<div class="d-note">Add values for the constant levers on the <b>Launch</b> tab (or a sweep), or ask the AI to propose them.</div>${discussBtn('Ask the AI which values to try')}`
      return discussBtn(s.action.label || 'Discuss with the AI') || `<div class="d-note">Open this project in the Overseer chat to discuss.</div>`
    }
    const stepLi = (s, i) => `<li class="d-step ${s.done ? 'done' : ''}">
        <div class="d-step-h"><span class="d-stepn">${i + 1}</span><span class="d-sev ${esc(s.severity)}">${esc(s.severity)}</span><span class="d-step-t">${esc(s.title)}</span></div>
        ${evHtml(s.evidence) ? `<div class="d-ev">${evHtml(s.evidence)}</div>` : ''}
        <div class="d-step-a">${stepAction(s)}</div>
      </li>`

    const strategicLi = `<li class="d-step strategic">
        <div class="d-step-h"><span class="d-stepn">${steps.length + 1}</span><span class="d-sev info">then</span><span class="d-step-t">Still no strong candidate after the steps above?</span></div>
        <p class="d-note">Work out the next move — different data, features, objective, or a strategic pivot — with the AI. It can read your runs (getTrainerState / getRunData / getRunXAI) and hand back concrete experiments.</p>
        <div class="d-step-a">${discussBtn('💬 Discuss the strategic next move with the AI') || '<div class="d-note">Open this project in the Overseer chat to discuss.</div>'}</div>
      </li>`

    const passed = d.findings.filter((f) => f.severity === 'ok' || f.severity === 'info')
    const passedCards = passed
      .map((f) => `<div class="d-card"><div class="d-top"><span class="d-sev ${esc(f.severity)}">${esc(f.severity)}</span><span class="d-cat">${esc(catLabel(f.category))}</span></div><p>${esc(f.headline)}</p></div>`)
      .join('')

    container.innerHTML = `<div class="diag">
      <div class="d-verdict ${vs[0]}">
        <div class="d-vtag">Diagnosis · ${esc(vs[1])}</div>
        <div class="d-vtitle">${esc(d.headline.stalledBecause || d.headline.doNext)}</div>
        ${d.headline.stalledBecause ? `<p class="d-next">Do next: ${esc(d.headline.doNext)}</p>` : ''}
        ${discussBtn('💬 Discuss this diagnosis with the AI')}
      </div>
      ${stats}
      <div class="d-sec">Your plan — work through these in order</div>
      <ol class="d-plan">${steps.map(stepLi).join('')}${strategicLi}</ol>
      ${passedCards ? `<div class="d-sec">Checks that passed</div><div class="d-cards">${passedCards}</div>` : ''}
    </div>`

    container.querySelectorAll('[data-diag-act]').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const kind = btn.getAttribute('data-diag-act')
        if (kind === 'discuss') {
          if (actions && actions.onDiscuss) actions.onDiscuss(d)
          return
        }
        const specs = kind === 'reseed' ? campaigns.reseed : campaigns.replicate
        const label = kind === 'reseed' ? 'Diagnose · reseed top setups' : 'Diagnose · replicate shortlist'
        btn.disabled = true
        if (actions && actions.onLaunchCampaigns) await actions.onLaunchCampaigns(specs, label, kind)
      })
    })
  }
  function catLabel(c) {
    return {
      'cohort-integrity': 'Cohort integrity',
      'objective-discriminability': 'Discriminability',
      'null-ceiling': 'Beats the null / ceiling',
      'split-consistency': 'Split-consistency',
      'incumbent-separation': 'Incumbent separation',
      'budget-coverage': 'Budget & coverage',
      'objective-confound': 'Objective confound',
    }[c] || c
  }

  const Diagnostics = {
    render,
    analyze,
    paint,
    shell,
    diagnose,
    resolveSpec,
    partitionCohort,
    checkCohortIntegrity,
    checkDiscriminability,
    checkNullCeiling,
    checkSplitConsistency,
    checkIncumbentSeparation,
    checkBudgetCoverage,
    checkObjectiveConfound,
    checkCrossAssetRobustness,
    planSteps,
    reseedSpecs,
    replicateSpecs,
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = Diagnostics
  if (root) root.Diagnostics = Diagnostics
})(typeof window !== 'undefined' ? window : null)
