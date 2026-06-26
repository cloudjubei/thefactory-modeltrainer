// Pure decision logic for the Models catalog — a Model is a model architecture/algorithm the project can
// train, the aggregating layer the Models tab renders. It OWNS its runs through its FLAVORS: each flavor
// maps to actual run configs by `model_name` (optionally narrowed by extra config values, e.g.
// `{lstm_hidden_size: 3}`), so a family like "Dueling DQN" records every variant precisely and a run
// resolves to exactly one flavor. Run counts/status come from an all-runs aggregate (`computeModelStats`),
// NOT from a page of runs. It LINKS the papers that introduce/improve it + the hypotheses that test it.
// Pure + dual-loaded (browser `window.Models` + node `module.exports`) so the ACTUAL viewer logic is
// unit-tested directly. A `RunRow` for the aggregate is `{ key, config, objective, status, health, ranAt }`.
;(function (root) {
  'use strict'

  const MODEL_STATUSES = ['proposed', 'implemented', 'failing', 'needs-improvement', 'deferred', 'deprecated']
  const MODEL_STATUS_LABEL = {
    proposed: 'proposed',
    implemented: 'implemented',
    failing: 'failing',
    'needs-improvement': 'needs work',
    deferred: 'deferred',
    deprecated: 'deprecated',
  }
  // Reuse the run-badge classes the Runs/Papers tabs already style (is-queued/is-running/is-done/is-failed).
  const MODEL_STATUS_BADGE = {
    proposed: 'is-queued',
    implemented: 'is-done',
    failing: 'is-failed',
    'needs-improvement': 'is-running',
    deferred: 'is-queued',
    deprecated: 'is-queued',
  }
  // Statuses that are NOT auto-derivable from runs (a deliberate lifecycle pin) — authoritative regardless
  // of statusSource, so a manifest seed can declare a model deferred/deprecated/needs-work and it sticks.
  const MODEL_PINNED_STATUSES = ['deferred', 'deprecated', 'needs-improvement']
  const MODEL_CATEGORIES = ['rl', 'supervised', 'baseline', 'component']
  const MODEL_CATEGORY_LABEL = {
    rl: 'Reinforcement learning',
    supervised: 'Supervised',
    baseline: 'Baselines',
    component: 'Components',
  }

  // The flavors of a model. Prefer the structured `flavors[]`; fall back to deriving one flavor per legacy
  // `modelNames[]` entry so records written before flavors still resolve. `[]` for a pure proposal.
  function modelFlavors(model) {
    if (model && Array.isArray(model.flavors) && model.flavors.length) return model.flavors
    if (model && Array.isArray(model.modelNames)) {
      return model.modelNames.map((n) => ({ modelName: n }))
    }
    return []
  }
  // A human label for a flavor (its own name, else its model_name).
  function flavorLabel(flavor) {
    return (flavor && (flavor.name || flavor.modelName)) || ''
  }
  // Canonical signature of a flavor's extra config matchers (sorted), for a stable per-flavor key.
  function configSignature(config) {
    if (!config || typeof config !== 'object') return ''
    return Object.keys(config)
      .sort()
      .map((k) => k + '=' + String(config[k]))
      .join('&')
  }
  // A stable identity for a flavor (model_name + its config matchers) — the per-flavor stats key.
  function flavorKey(flavor) {
    return (flavor ? flavor.modelName || '' : '') + '|' + configSignature(flavor && flavor.config)
  }
  // True iff a run config trains this flavor: its model_name matches AND every extra config matcher
  // loosely-equals the run's value (so JSON 3 matches a stored "3").
  function flavorMatchesConfig(flavor, config) {
    if (!flavor || !config) return false
    if (config.model_name !== flavor.modelName) return false
    const c = flavor.config
    if (c && typeof c === 'object') {
      for (const k of Object.keys(c)) {
        if (String(config[k]) !== String(c[k])) return false
      }
    }
    return true
  }
  // True iff a run config trains ANY of the model's flavors.
  function runMatchesModel(model, config) {
    return modelFlavors(model).some((fl) => flavorMatchesConfig(fl, config))
  }
  // The distinct model_name values across a model's flavors (for the lever check + a compact display).
  function flavorModelNames(model) {
    const seen = {}
    const out = []
    for (const fl of modelFlavors(model)) {
      if (fl.modelName && !seen[fl.modelName]) {
        seen[fl.modelName] = true
        out.push(fl.modelName)
      }
    }
    return out
  }
  // True iff the model is present in the project: it (or a flavor) names an implementation path, or a
  // flavor's model_name is one of the manifest's model_name choices. A proposed-but-unwired model is NOT.
  function isModelImplemented(model, manifest) {
    if (model && typeof model.implPath === 'string' && model.implPath) return true
    if (modelFlavors(model).some((fl) => typeof fl.implPath === 'string' && fl.implPath))
      return true
    const lever = manifest && manifest.levers && manifest.levers.model_name
    const choices =
      lever && lever.type === 'choice' && Array.isArray(lever.choices) ? lever.choices : []
    return flavorModelNames(model).some((n) => choices.indexOf(n) >= 0)
  }
  // A run is unhealthy when it errored or its RunSummary health is flagged.
  function runIsUnhealthy(row) {
    const s = row || {}
    if ((s.status || 'completed') === 'failed') return true
    const h = s.health
    if (h && h.status && h.status !== 'ok') return true
    if (h && Array.isArray(h.flags) && h.flags.length > 0) return true
    return false
  }

  function emptyBucket() {
    return { runs: 0, best: null, failing: 0, lastRunAt: null }
  }
  // Aggregate ALL runs across the catalog: per-model + per-flavor {runs, best, failing, lastRunAt}, plus
  // `uncataloged` (model_name values seen in runs that match NO flavor — the "missing flavor" signal) and
  // the overall newestRunAt/totalRuns. Pure: the viewer fetches every run and persists what this returns.
  function computeModelStats(models, runRows, direction) {
    const dir = direction === 'min' ? 'min' : 'max'
    const better = (a, b) => (dir === 'min' ? a < b : a > b)
    const bump = (bucket, obj, bad, ranAt) => {
      bucket.runs += 1
      if (bad) bucket.failing += 1
      if (obj !== null && (bucket.best === null || better(obj, bucket.best))) bucket.best = obj
      if (ranAt && (!bucket.lastRunAt || ranAt > bucket.lastRunAt)) bucket.lastRunAt = ranAt
    }
    const index = []
    const perModel = {}
    for (const m of models || []) {
      if (!m || !m.id) continue
      const entry = Object.assign(emptyBucket(), { perFlavor: {} })
      for (const fl of modelFlavors(m)) {
        const key = flavorKey(fl)
        entry.perFlavor[key] = Object.assign(emptyBucket(), {
          name: flavorLabel(fl),
          modelName: fl.modelName,
          config: fl.config,
        })
        index.push({ slug: m.id, flavor: fl, key: key })
      }
      perModel[m.id] = entry
    }
    const uncataloged = {}
    let newestRunAt = null
    const rows = runRows || []
    let counted = 0
    for (const r of rows) {
      // Invalid runs (produced by a since-fixed bug) are excluded from EVERY aggregation — run/flavor
      // counts, best, failing, total, and staleness — so they never influence model stats, hypothesis
      // verdicts, or xAI. They remain visible + filterable in the Runs tab via their status.
      if (r && r.status === 'invalid') continue
      counted += 1
      const config = (r && r.config) || {}
      const ranAt = (r && r.ranAt) || null
      if (ranAt && (!newestRunAt || ranAt > newestRunAt)) newestRunAt = ranAt
      const obj = r && typeof r.objective === 'number' && isFinite(r.objective) ? r.objective : null
      const bad = runIsUnhealthy(r)
      let matched = null
      for (const e of index) {
        if (flavorMatchesConfig(e.flavor, config)) {
          matched = e
          break
        }
      }
      if (matched) {
        bump(perModel[matched.slug], obj, bad, ranAt)
        bump(perModel[matched.slug].perFlavor[matched.key], obj, bad, ranAt)
      } else {
        const mn = config.model_name || '(unset)'
        if (!uncataloged[mn]) uncataloged[mn] = Object.assign(emptyBucket(), { modelName: mn })
        bump(uncataloged[mn], obj, bad, ranAt)
      }
    }
    return {
      perModel: perModel,
      uncataloged: Object.values(uncataloged).sort((a, b) => b.runs - a.runs),
      totalRuns: counted,
      newestRunAt: newestRunAt,
    }
  }
  // The aggregate for ONE model from a stats record (or null when not yet aggregated).
  function aggForModel(stats, slug) {
    return (stats && stats.perModel && stats.perModel[slug]) || null
  }
  // The auto-derived (or manually pinned) lifecycle status of a model, given its all-runs aggregate.
  function deriveModelStatus(model, agg, manifest) {
    // A manual override OR a pinned (non-auto-derivable) status is authoritative — otherwise derive from
    // whether the flavors bind a lever choice + the run health.
    if (model && (model.statusSource === 'manual' || MODEL_PINNED_STATUSES.indexOf(model.status) >= 0))
      return model.status
    const implemented = isModelImplemented(model, manifest)
    if (!agg || !agg.runs) return implemented ? 'implemented' : 'proposed'
    return agg.failing >= agg.runs ? 'failing' : 'implemented'
  }

  // Build a `proposed` catalog record from a paper's ProposedModel (the "Add to catalog" click). It has no
  // flavors yet (unimplemented); auto status-source so it tracks its lifecycle once flavors + runs land.
  function buildProposedModelRecord(proposed, paperId, nowIso) {
    return {
      id: proposed.slug,
      slug: proposed.slug,
      name: proposed.name,
      description: proposed.description || '',
      category: proposed.category || 'rl',
      status: 'proposed',
      statusSource: 'auto',
      flavors: [],
      paperIds: paperId ? [paperId] : [],
      proposal: proposed.proposal || '',
      source: 'paper',
      createdAt: nowIso,
      updatedAt: nowIso,
    }
  }

  // The catalog models a paper is about — union of both link directions (paper.modelIds and model.paperIds).
  function modelsForPaper(paper, models) {
    const direct = {}
    const ids = (paper && paper.modelIds) || []
    for (let i = 0; i < ids.length; i++) direct[ids[i]] = true
    return (models || []).filter((m) => direct[m.id] || (m.paperIds || []).indexOf(paper.id) >= 0)
  }
  // The papers a model is linked to — union of both link directions (model.paperIds and paper.modelIds).
  function papersForModel(model, papers) {
    const direct = {}
    const ids = (model && model.paperIds) || []
    for (let i = 0; i < ids.length; i++) direct[ids[i]] = true
    return (papers || []).filter((p) => direct[p.id] || (p.modelIds || []).indexOf(model.id) >= 0)
  }
  // The hypotheses a model is linked to (model.hypothesisIds).
  function hypothesesForModel(model, hyps) {
    const direct = {}
    const ids = (model && model.hypothesisIds) || []
    for (let i = 0; i < ids.length; i++) direct[ids[i]] = true
    return (hyps || []).filter((h) => direct[h.id])
  }

  // Manifest-owned SCALAR fields a seed re-sync overwrites onto an existing record (flavors compared
  // separately, deeply). The rest are user-owned: statusSource:manual status + statusNote, notes,
  // dismissed, hypothesisIds, createdAt.
  const MODEL_SEED_FIELDS = ['name', 'description', 'category', 'implPath', 'proposal', 'source']

  function flavorsEqual(a, b) {
    const x = a || []
    const y = b || []
    if (x.length !== y.length) return false
    for (let i = 0; i < x.length; i++) {
      if ((x[i].modelName || '') !== (y[i].modelName || '')) return false
      if ((x[i].name || '') !== (y[i].name || '')) return false
      if (configSignature(x[i].config) !== configSignature(y[i].config)) return false
    }
    return true
  }
  // True iff importing this seed would CHANGE the catalog — no existing record, or a manifest-owned field
  // (esp. the flavors / bindings — the consolidation) differs. Drives the "import / re-sync" banner.
  function seedDiffersFromModel(seed, existing) {
    if (!existing) return true
    if (!flavorsEqual(modelFlavors(seed), modelFlavors(existing))) return true
    for (let i = 0; i < MODEL_SEED_FIELDS.length; i++) {
      const f = MODEL_SEED_FIELDS[i]
      if ((seed[f] || '') !== (existing[f] || '')) return true
    }
    return false
  }
  // The record to persist when syncing a manifest seed: manifest-owned fields (incl. flavors) from the
  // seed, user-owned fields preserved from any existing record. A manual status pin is never overwritten.
  function mergeSeedIntoModel(seed, existing, nowIso) {
    const slug = seed.slug || seed.id
    const merged = {
      id: seed.id || slug,
      slug: slug,
      name: seed.name,
      description: seed.description || '',
      category: seed.category,
      flavors: Array.isArray(seed.flavors) ? seed.flavors : modelFlavors(seed),
      source: seed.source || 'manual',
    }
    if (seed.implPath) merged.implPath = seed.implPath
    if (seed.proposal) merged.proposal = seed.proposal
    const paperIds = []
    const seen = {}
    const allPapers = (seed.paperIds || []).concat((existing && existing.paperIds) || [])
    for (let i = 0; i < allPapers.length; i++) {
      if (!seen[allPapers[i]]) {
        seen[allPapers[i]] = true
        paperIds.push(allPapers[i])
      }
    }
    if (paperIds.length) merged.paperIds = paperIds
    if (existing && existing.hypothesisIds && existing.hypothesisIds.length) {
      merged.hypothesisIds = existing.hypothesisIds
    }
    if (existing && existing.statusSource === 'manual') {
      merged.status = existing.status
      merged.statusSource = 'manual'
      if (existing.statusNote) merged.statusNote = existing.statusNote
    } else {
      merged.status = seed.status || 'implemented'
      merged.statusSource = 'auto'
    }
    if (existing && existing.notes) merged.notes = existing.notes
    if (existing && existing.dismissed) merged.dismissed = existing.dismissed
    merged.createdAt = (existing && existing.createdAt) || nowIso
    merged.updatedAt = nowIso
    return merged
  }

  const Models = {
    MODEL_STATUSES: MODEL_STATUSES,
    MODEL_STATUS_LABEL: MODEL_STATUS_LABEL,
    MODEL_STATUS_BADGE: MODEL_STATUS_BADGE,
    MODEL_CATEGORIES: MODEL_CATEGORIES,
    MODEL_CATEGORY_LABEL: MODEL_CATEGORY_LABEL,
    modelFlavors: modelFlavors,
    flavorLabel: flavorLabel,
    flavorKey: flavorKey,
    flavorMatchesConfig: flavorMatchesConfig,
    runMatchesModel: runMatchesModel,
    flavorModelNames: flavorModelNames,
    isModelImplemented: isModelImplemented,
    runIsUnhealthy: runIsUnhealthy,
    computeModelStats: computeModelStats,
    aggForModel: aggForModel,
    deriveModelStatus: deriveModelStatus,
    buildProposedModelRecord: buildProposedModelRecord,
    modelsForPaper: modelsForPaper,
    papersForModel: papersForModel,
    hypothesesForModel: hypothesesForModel,
    seedDiffersFromModel: seedDiffersFromModel,
    mergeSeedIntoModel: mergeSeedIntoModel,
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = Models
  if (root) root.Models = Models
})(typeof window !== 'undefined' ? window : null)
