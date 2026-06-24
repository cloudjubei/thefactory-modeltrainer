// Pure decision logic for the Models catalog — a Model is a model architecture/algorithm the project can
// train, the aggregating layer the Models tab renders. It OWNS its runs by binding one or more model_name
// lever values (`modelNames`): a run trains this model iff its `config.model_name` is one of them, so the
// run roll-up, verdict and "is it failing" derive from the same substrate as the rest of the hub. It LINKS
// the papers that introduce/improve it and the hypotheses that test it. Pure + dual-loaded (browser
// `window.Models` + node `module.exports`) so the ACTUAL viewer logic is unit-tested directly. A `RunRow`
// is `{ key, summary:{ config, metrics, objective, status, health } }`.
;(function (root) {
  'use strict'

  const MODEL_STATUSES = ['proposed', 'implemented', 'failing', 'needs-improvement', 'deprecated']
  const MODEL_STATUS_LABEL = {
    proposed: 'proposed',
    implemented: 'implemented',
    failing: 'failing',
    'needs-improvement': 'needs work',
    deprecated: 'deprecated',
  }
  // Reuse the run-badge classes the Runs/Papers tabs already style (is-queued/is-running/is-done/is-failed).
  const MODEL_STATUS_BADGE = {
    proposed: 'is-queued',
    implemented: 'is-done',
    failing: 'is-failed',
    'needs-improvement': 'is-running',
    deprecated: 'is-queued',
  }
  const MODEL_CATEGORIES = ['rl', 'supervised', 'baseline', 'component']
  const MODEL_CATEGORY_LABEL = {
    rl: 'Reinforcement learning',
    supervised: 'Supervised',
    baseline: 'Baselines',
    component: 'Components',
  }

  // True iff a run's resolved config trains this model — i.e. its model_name is one of the model's
  // bindings. A model with no bindings (a pure proposal) matches NOTHING.
  function modelMatchesRun(model, config) {
    const names = (model && model.modelNames) || []
    if (!names.length || !config) return false
    return names.indexOf(config.model_name) >= 0
  }

  // The RunRows whose config binds to this model.
  function runsForModel(model, runs) {
    return (runs || []).filter((r) => r && r.summary && modelMatchesRun(model, r.summary.config))
  }

  // True iff the model is present in the project: a binding is one of the manifest's model_name choices,
  // or the model names an implementation path. A proposed-but-unwired model is NOT implemented.
  function isModelImplemented(model, manifest) {
    if (model && typeof model.implPath === 'string' && model.implPath) return true
    const names = (model && model.modelNames) || []
    const lever = manifest && manifest.levers && manifest.levers.model_name
    const choices =
      lever && lever.type === 'choice' && Array.isArray(lever.choices) ? lever.choices : []
    for (let i = 0; i < names.length; i++) if (choices.indexOf(names[i]) >= 0) return true
    return false
  }

  // A run is unhealthy when it errored or its RunSummary health is flagged.
  function runIsUnhealthy(r) {
    const s = (r && r.summary) || {}
    if ((s.status || 'completed') === 'failed') return true
    const h = s.health
    if (h && h.status && h.status !== 'ok') return true
    if (h && Array.isArray(h.flags) && h.flags.length > 0) return true
    return false
  }

  // The auto-derived lifecycle verdict: a `manual` status is returned as pinned. Otherwise — proposed when
  // not implemented and unrun; implemented when present but unrun; failing when every matching run is
  // unhealthy; implemented when at least one matching run is healthy.
  function deriveModelStatus(model, runs, manifest) {
    if (model && model.statusSource === 'manual') return model.status
    const matching = runsForModel(model, runs)
    if (!matching.length) return isModelImplemented(model, manifest) ? 'implemented' : 'proposed'
    const healthy = matching.filter((r) => !runIsUnhealthy(r))
    return healthy.length ? 'implemented' : 'failing'
  }

  // The aggregate read for a model's card chip: how many runs trained it, the best objective among them
  // (direction-aware), and how many were unhealthy.
  function modelRunSummary(model, runs, direction) {
    const matching = runsForModel(model, runs)
    let best = null
    let failing = 0
    for (let i = 0; i < matching.length; i++) {
      const r = matching[i]
      if (runIsUnhealthy(r)) failing += 1
      const o = r.summary && r.summary.objective
      if (typeof o === 'number' && isFinite(o)) {
        if (best === null || (direction === 'min' ? o < best : o > best)) best = o
      }
    }
    return { runs: matching.length, best: best, failing: failing }
  }

  // Build a `proposed` catalog record from a paper's ProposedModel (the "Add to catalog" click). Auto
  // status-source so it tracks its own lifecycle once implemented + run.
  function buildProposedModelRecord(proposed, paperId, nowIso) {
    return {
      id: proposed.slug,
      slug: proposed.slug,
      name: proposed.name,
      description: proposed.description || '',
      category: proposed.category || 'rl',
      status: 'proposed',
      statusSource: 'auto',
      modelNames: [],
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

  // Manifest-owned fields a seed re-sync overwrites onto an existing record (the rest are user-owned:
  // statusSource:manual status + statusNote, notes, dismissed, hypothesisIds, createdAt).
  const MODEL_SEED_FIELDS = ['name', 'description', 'category', 'implPath', 'proposal', 'source']

  function sameStringList(a, b) {
    const x = a || []
    const y = b || []
    if (x.length !== y.length) return false
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false
    return true
  }

  // True iff importing this seed would CHANGE the catalog — no existing record, or a manifest-owned
  // field (esp. the modelNames bindings — the consolidation) differs. Drives the "import / re-sync" banner.
  function seedDiffersFromModel(seed, existing) {
    if (!existing) return true
    if (!sameStringList(seed.modelNames, existing.modelNames)) return true
    for (let i = 0; i < MODEL_SEED_FIELDS.length; i++) {
      const f = MODEL_SEED_FIELDS[i]
      if ((seed[f] || '') !== (existing[f] || '')) return true
    }
    return false
  }

  // The record to persist when syncing a manifest seed: manifest-owned fields from the seed, user-owned
  // fields preserved from any existing record. A manual status pin is never overwritten by the re-sync.
  function mergeSeedIntoModel(seed, existing, nowIso) {
    const slug = seed.slug || seed.id
    const merged = {
      id: seed.id || slug,
      slug: slug,
      name: seed.name,
      description: seed.description || '',
      category: seed.category,
      modelNames: Array.isArray(seed.modelNames) ? seed.modelNames.slice() : [],
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

  // The hypotheses a model is linked to (model.hypothesisIds).
  function hypothesesForModel(model, hyps) {
    const direct = {}
    const ids = (model && model.hypothesisIds) || []
    for (let i = 0; i < ids.length; i++) direct[ids[i]] = true
    return (hyps || []).filter((h) => direct[h.id])
  }

  const Models = {
    MODEL_STATUSES: MODEL_STATUSES,
    MODEL_STATUS_LABEL: MODEL_STATUS_LABEL,
    MODEL_STATUS_BADGE: MODEL_STATUS_BADGE,
    MODEL_CATEGORIES: MODEL_CATEGORIES,
    MODEL_CATEGORY_LABEL: MODEL_CATEGORY_LABEL,
    modelMatchesRun: modelMatchesRun,
    runsForModel: runsForModel,
    isModelImplemented: isModelImplemented,
    runIsUnhealthy: runIsUnhealthy,
    deriveModelStatus: deriveModelStatus,
    modelRunSummary: modelRunSummary,
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
