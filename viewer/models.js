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
  // Per-model wall-clock training-run durations over the given run rows ({config.model_name, durationMs}),
  // for the Speed tab. Groups by raw model_name, ignores non-positive / non-numeric durations, and returns
  // [{ modelName, runs, meanMs, minMs, maxMs }] sorted fastest-first. A model with no valid duration is
  // dropped. Computed over EVERY run by the all-runs refresh + persisted in the model-stats record, so the
  // Speed table reflects all runs rather than the current page.
  function computeRunDurationsByModel(rows) {
    const by = {}
    for (const row of Array.isArray(rows) ? rows : []) {
      const d = row && row.durationMs
      if (typeof d !== 'number' || !(d > 0)) continue
      const mn = String((row.config || {}).model_name == null ? '?' : (row.config || {}).model_name)
      ;(by[mn] = by[mn] || []).push(d)
    }
    return Object.keys(by)
      .map((mn) => {
        const ds = by[mn]
        return {
          modelName: mn,
          runs: ds.length,
          meanMs: ds.reduce((a, b) => a + b, 0) / ds.length,
          minMs: Math.min.apply(null, ds),
          maxMs: Math.max.apply(null, ds),
        }
      })
      .sort((a, b) => a.meanMs - b.meanMs)
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

  // The component catalog entries a FLAVOR is composed of: its `components` slugs resolved to catalog
  // models (deduped, order-preserving). An unknown slug is returned as `{found:false}` so a typo still
  // shows (as a non-link chip) instead of vanishing. Pure.
  function flavorComponents(flavor, models) {
    const slugs = flavor && Array.isArray(flavor.components) ? flavor.components : []
    if (!slugs.length) return []
    const bySlug = {}
    for (const m of models || []) if (m && m.slug) bySlug[m.slug] = m
    const seen = {}
    const out = []
    for (const raw of slugs) {
      const s = slugify(raw)
      if (!s || seen[s]) continue
      seen[s] = true
      const m = bySlug[raw] || bySlug[s]
      out.push({ slug: m ? m.slug : raw, name: m ? m.name || m.slug : raw, found: !!m })
    }
    return out
  }

  // The catalog models that LIST this component slug in any flavor's `components` — the reverse of
  // flavorComponents, for a component entry's "used by". Excludes the component itself. Pure.
  function modelsUsingComponent(componentSlug, models) {
    const target = slugify(componentSlug)
    if (!target) return []
    const out = []
    for (const m of models || []) {
      if (!m || slugify(m.slug) === target) continue
      const used = modelFlavors(m).some(
        (fl) => Array.isArray(fl.components) && fl.components.some((c) => slugify(c) === target),
      )
      if (used) out.push(m)
    }
    return out
  }

  // Manifest-owned SCALAR fields a seed re-sync overwrites onto an existing record (flavors compared
  // separately, deeply). The rest are user-owned: statusSource:manual status + statusNote, notes,
  // dismissed, hypothesisIds, createdAt.
  const MODEL_SEED_FIELDS = ['name', 'description', 'category', 'implPath', 'proposal', 'source']

  // True iff importing this seed would CHANGE the catalog: no existing record, a manifest-owned scalar
  // differs, the seed introduces a NEW flavor binding, or it declares an alias the record lacks. Flavors +
  // aliases UNION on sync, so a flavor/alias the user or a consolidation added is NOT a reason to re-sync —
  // only something the manifest adds is. Drives the "import / re-sync" banner.
  function seedDiffersFromModel(seed, existing) {
    if (!existing) return true
    for (let i = 0; i < MODEL_SEED_FIELDS.length; i++) {
      const f = MODEL_SEED_FIELDS[i]
      if ((seed[f] || '') !== (existing[f] || '')) return true
    }
    const compKey = (fl) => (Array.isArray(fl.components) ? fl.components : []).map(slugify).join(',')
    const have = {}
    for (const fl of modelFlavors(existing)) have[flavorKey(fl)] = fl
    for (const fl of modelFlavors(seed)) {
      const ex = have[flavorKey(fl)]
      if (!ex) return true
      if (compKey(fl) !== compKey(ex)) return true
    }
    const ident = modelIdentitySlugs(existing)
    for (const a of seed.aliases || []) if (!ident.has(slugify(a))) return true
    return false
  }
  // The record to persist when syncing a manifest seed: manifest-owned scalars from the seed; flavors +
  // aliases UNIONED with the existing record (so a consolidation's absorbed bindings/aliases survive a
  // re-sync); other user-owned fields preserved. A manual status pin is never overwritten.
  // KNOWN LIMITATION of the union: the manifest can ADD a flavor but not REMOVE/RENAME one through sync —
  // a binding the record already has persists. This is intentional (it's what keeps a merge durable); for a
  // consolidation-absorbed flavor, continuing to attribute its runs to the canonical is the CORRECT result.
  // To truly drop a binding, delete the model and re-import. A run never matches two models (first flavor
  // wins in computeModelStats), so a lingering flavor cannot double-count.
  function mergeSeedIntoModel(seed, existing, nowIso) {
    const slug = seed.slug || seed.id
    const seedFlavors = Array.isArray(seed.flavors) ? seed.flavors : modelFlavors(seed)
    const merged = {
      id: seed.id || slug,
      slug: slug,
      name: seed.name,
      description: seed.description || '',
      category: seed.category,
      flavors: dedupeFlavors(seedFlavors.concat(modelFlavors(existing))),
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
    const own = modelIdentitySlugs({ id: merged.id, slug: merged.slug, name: merged.name })
    const aliases = []
    const seenAlias = {}
    const addAlias = (raw) => {
      const s = slugify(raw)
      if (s && !own.has(s) && !seenAlias[s]) {
        seenAlias[s] = true
        aliases.push(s)
      }
    }
    for (const a of (existing && existing.aliases) || []) addAlias(a)
    for (const a of seed.aliases || []) addAlias(a)
    if (aliases.length) merged.aliases = aliases
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
  // True iff this manifest seed should be SKIPPED on sync because it is an alias of another (non-dismissed)
  // model — i.e. it was merged away by a consolidation. A seed that still has its own live record is a model
  // in its own right and is never claimed. Stops "Sync" from re-adding a model the user merged.
  function seedClaimedByAlias(seed, models) {
    if (!seed || !seed.id) return false
    for (const m of models || []) if (m && !m.dismissed && m.id === seed.id) return false
    const tokens = modelIdentitySlugs(seed)
    for (const m of models || []) {
      if (!m || m.dismissed) continue
      for (const a of m.aliases || []) if (tokens.has(slugify(a))) return true
    }
    return false
  }

  // Kebab-normalize a slug/name for loose identity matching ("Inverted Transformer PPO" -> "inverted-transformer-ppo").
  function slugify(s) {
    return String(s == null ? '' : s)
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
  }
  // The slug-normalized identity tokens a model answers to: its slug, id, name, and declared aliases.
  function modelIdentitySlugs(model) {
    const out = new Set()
    if (!model) return out
    out.add(slugify(model.slug))
    out.add(slugify(model.id))
    out.add(slugify(model.name))
    for (const a of model.aliases || []) out.add(slugify(a))
    out.delete('')
    return out
  }
  // De-dupe a flavor list by flavorKey (model_name + config signature), keeping first occurrence.
  function dedupeFlavors(flavors) {
    const seen = {}
    const out = []
    for (const fl of flavors || []) {
      const k = flavorKey(fl)
      if (!seen[k]) {
        seen[k] = true
        out.push(fl)
      }
    }
    return out
  }
  // Order-preserving union of id lists (skips falsy + duplicates).
  function unionIds(lists) {
    const seen = {}
    const out = []
    for (const list of lists || []) {
      for (const id of list || []) {
        if (id && !seen[id]) {
          seen[id] = true
          out.push(id)
        }
      }
    }
    return out
  }
  // Fold one or more DUPLICATE models into the CANONICAL one (a consolidation): the canonical keeps its
  // identity + user pins (status/statusSource/notes/createdAt), and absorbs the duplicates' flavors (deduped
  // by flavorKey, canonical's first) and paper/hypothesis links (unioned). A duplicate that IS the canonical
  // (same id) is ignored. Legacy `modelNames[]` records resolve via modelFlavors and the stale field is
  // dropped. Pure: the viewer persists the result + deletes the duplicate records.
  function mergeModelsForConsolidation(canonical, duplicates, nowIso) {
    const dups = (duplicates || []).filter((d) => d && d.id !== canonical.id)
    const flavors = dedupeFlavors(
      modelFlavors(canonical).concat.apply(
        modelFlavors(canonical),
        dups.map((d) => modelFlavors(d)),
      ),
    )
    const paperIds = unionIds([canonical.paperIds].concat(dups.map((d) => d.paperIds)))
    const hypothesisIds = unionIds([canonical.hypothesisIds].concat(dups.map((d) => d.hypothesisIds)))
    // Record every name the merged-away models were known by as an alias of the canonical (skipping the
    // canonical's own identity), so a future paper/scan/seed referring to a duplicate resolves here and the
    // seed-sync never re-adds it.
    // `own` is the canonical's CORE identity (slug/id/name) only — deliberately NOT its existing aliases,
    // so those are re-added (preserved) below alongside each duplicate's slug/id/name + transitive aliases,
    // while we still never alias the canonical to its own slug/id/name.
    const own = new Set([slugify(canonical.slug), slugify(canonical.id), slugify(canonical.name)])
    own.delete('')
    const aliases = []
    const seenAlias = new Set()
    const addAlias = (raw) => {
      const s = slugify(raw)
      if (!s || own.has(s) || seenAlias.has(s)) return
      seenAlias.add(s)
      aliases.push(s)
    }
    for (const a of canonical.aliases || []) addAlias(a)
    for (const d of dups) {
      addAlias(d.slug)
      addAlias(d.id)
      addAlias(d.name)
      for (const a of d.aliases || []) addAlias(a)
    }
    const merged = Object.assign({}, canonical, { flavors: flavors, updatedAt: nowIso })
    delete merged.modelNames
    if (paperIds.length) merged.paperIds = paperIds
    else delete merged.paperIds
    if (hypothesisIds.length) merged.hypothesisIds = hypothesisIds
    else delete merged.hypothesisIds
    if (aliases.length) merged.aliases = aliases
    else delete merged.aliases
    return merged
  }
  // The ids of the members the user has checked to merge into the canonical — every member except the
  // canonical that is still in `checkedDuplicateIds`. The group object (not the DOM) is the source of truth,
  // so changing the canonical can never silently re-include a duplicate the user excluded.
  function selectedDuplicateIds(group) {
    return (group.members || [])
      .map((m) => m.id)
      .filter((id) => id !== group.canonicalId && group.checkedDuplicateIds.has(id))
  }
  // Make `newCanonicalId` the canonical of a group: the old canonical becomes a checked duplicate (default
  // include) and the new one drops out of the duplicate set — every other check the user made is preserved.
  function swapConsolidationCanonical(group, newCanonicalId) {
    group.checkedDuplicateIds.add(group.canonicalId)
    group.checkedDuplicateIds.delete(newCanonicalId)
    group.canonicalId = newCanonicalId
    return group
  }
  // Repoint papers that referenced any of `fromIds` (the duplicates being merged away) to `toId` (the
  // canonical) in their `modelIds`, de-duping. Returns ONLY the papers that changed (the viewer persists
  // those). Papers without modelIds — or with no reference to a duplicate — are left untouched.
  function repointPaperModelIds(papers, fromIds, toId) {
    const from = {}
    for (const id of fromIds || []) from[id] = true
    const changed = []
    for (const p of papers || []) {
      const ids = p && Array.isArray(p.modelIds) ? p.modelIds : null
      if (!ids || !ids.some((id) => from[id])) continue
      const seen = {}
      const next = []
      for (const id of ids) {
        const mapped = from[id] ? toId : id
        if (mapped && !seen[mapped]) {
          seen[mapped] = true
          next.push(mapped)
        }
      }
      changed.push(Object.assign({}, p, { modelIds: next }))
    }
    return changed
  }

  // The colour class for a device chip — cpu (blue) / mps (gray) / cuda (green); cpu is the default.
  function deviceChipClass(device) {
    const d = String(device == null ? '' : device).toLowerCase()
    return d === 'mps' ? 'device-chip-mps' : d === 'cuda' ? 'device-chip-cuda' : 'device-chip-cpu'
  }
  // A render-ready view of a model's deviceBenchmark: one row per standard device (cpu/mps/cuda, plus any
  // non-standard device the data carries), each with its µs/step + measured seconds + a per-device error
  // (e.g. "too slow / timed out") and whether it's the winner. Pure: both the Speed table and the model
  // card render from this, and the timings modal shows the raw per-device fields. null when not benchmarked.
  function deviceBenchmarkView(db, deviceOrder) {
    if (!db) return null
    const devices = (deviceOrder && deviceOrder.length ? deviceOrder : ['cpu', 'mps', 'cuda']).slice()
    const us = db.usPerStep || {}
    const secs = db.seconds || {}
    const errs = db.errors || {}
    for (const d of Object.keys(us).concat(Object.keys(secs), Object.keys(errs))) {
      if (devices.indexOf(d) < 0) devices.push(d)
    }
    const num = (m, d) => (typeof m[d] === 'number' && m[d] > 0 ? m[d] : null)
    const perDevice = devices.map((d) => ({
      device: d,
      usPerStep: num(us, d),
      seconds: num(secs, d),
      error: typeof errs[d] === 'string' && errs[d] ? errs[d] : null,
      isBest: d === db.bestDevice,
      chipClass: deviceChipClass(d),
    }))
    return {
      perDevice: perDevice,
      best: db.bestDevice || null,
      bestClass: deviceChipClass(db.bestDevice),
      speedup: typeof db.speedup === 'number' ? db.speedup : null,
      budget: typeof db.budget === 'number' ? db.budget : null,
      benchmarkedAt: db.benchmarkedAt || null,
    }
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
    computeRunDurationsByModel: computeRunDurationsByModel,
    deriveModelStatus: deriveModelStatus,
    buildProposedModelRecord: buildProposedModelRecord,
    modelsForPaper: modelsForPaper,
    papersForModel: papersForModel,
    hypothesesForModel: hypothesesForModel,
    flavorComponents: flavorComponents,
    modelsUsingComponent: modelsUsingComponent,
    seedDiffersFromModel: seedDiffersFromModel,
    mergeSeedIntoModel: mergeSeedIntoModel,
    mergeModelsForConsolidation: mergeModelsForConsolidation,
    repointPaperModelIds: repointPaperModelIds,
    selectedDuplicateIds: selectedDuplicateIds,
    swapConsolidationCanonical: swapConsolidationCanonical,
    slugify: slugify,
    modelIdentitySlugs: modelIdentitySlugs,
    seedClaimedByAlias: seedClaimedByAlias,
    deviceChipClass: deviceChipClass,
    deviceBenchmarkView: deviceBenchmarkView,
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = Models
  if (root) root.Models = Models
})(typeof window !== 'undefined' ? window : null)
