// Model Trainer hub — Overseer model-training dashboards (vanilla JS, no build).
// Runs embedded in Overseer's App view. The host project is just the hub: the
// user registers training-project directories as 'trainer-project' records, a
// backend 'inspect-trainer' activity writes each project's manifest into a
// 'trainer-project-manifest' record, and opening a project shows the usual
// runs/charts/hypotheses/launch/activity dashboard scoped to that manifest's
// record types, with the project's `dir` threaded into every
// train/judge/propose/evaluate activity.

const POLL_MS = 3000
const MAX_OBSERVE_MS = 6 * 60 * 60 * 1000
const MAX_QUICK_OBSERVE_MS = 10 * 60 * 1000
const ACTIVE_TAB_SS = 'trainer.activeTab'
const AUTO_EVAL_SS = 'trainer.autoEval'
const COMPUTE_TARGET_SS = 'trainer.computeTarget'
const CONCURRENCY_SS = 'trainer.concurrency'
const PROJECT_RECORD_TYPE = 'trainer-project'
const PROJECT_MANIFEST_RECORD_TYPE = 'trainer-project-manifest'
const QUEUE_RECORD_TYPE = 'trainer-queue'
const SEEN_RECORD_TYPE = 'trainer-seen'
const CHART_PALETTE = ['#3b82f6', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#14b8a6']
const JUDGE_HELP_TEXT =
  "Scores every completed run 0–100. Health-flagged runs are auto-rejected without using the LLM. For the rest, the objective is normalised (best=100) and blended 50/50 with an LLM verdict that weighs stability and how promising the configuration is — so a run can't win on prose alone. Results appear in the Judge column."
const PROPOSE_HELP_TEXT =
  "Sends the manifest's levers, the run history and the verdicts to the LLM and asks for new experiment specs likely to beat the best run. Proposals are validated against the levers, deduped by spec, and land below as pending hypotheses for you to accept or reject."
const NO_RUNNERS_HINT = 'No runners paired — manage them in the Compute Runners panel.'
const TABS = [
  { id: 'runs', label: 'Runs' },
  { id: 'charts', label: 'Charts' },
  { id: 'hypotheses', label: 'Hypotheses' },
  { id: 'launch', label: 'Launch' },
  { id: 'activity', label: 'Activity' },
]
const HYPOTHESIS_STATUSES = ['pending', 'accepted', 'rejected']
const HYPOTHESIS_SPEC_KEYS = ['sweep', 'fixed', 'seeds']
const HYPOTHESIS_SPEC_PLACEHOLDER = '{"sweep":{},"fixed":{},"seeds":[0]}'

let projectsCache = []
let manifestsCache = new Map()
const inspectingKeys = new Set()
// Home overview live state: the manifest recordTypes that currently have a
// running, live activity (drives each project card's spinner), each project's
// current run keys, and the per-project set of run keys the user has already
// seen (persisted as 'trainer-seen' records) — their difference is "unseen".
let liveRecordTypes = new Set()
let runKeysByProject = new Map()
let seenKeysByProject = new Map()
// Bumped whenever the home poll loop should stop (a project is opened); the loop
// captures it and exits when a newer session supersedes it.
let homePollSession = 0
let removeArmedKey = null
let currentProject = null
// Bumped on every home↔project navigation; long-lived async work captures it
// and stops when the user has navigated away.
let projectEpoch = 0
let manifest = null
let activeTabId = null
let runsCache = []
let verdictsCache = new Map()
let evaluationsCache = new Map()
let notesCache = new Map()
const evaluatingKeys = new Set()
let judgementSummary = null
let hypothesesCache = []
let proposalSummary = null
let selectedRunKey = null
let currentActivityId = null
let observing = false
let observeSession = 0
let lastActivityStatus = null
let lastProgress = null
let lastCampaign = null
let judging = false
let proposing = false
let queueCache = []
// In-memory double-dispatch guard for the persisted queue: only one pump loop
// drains the queue at a time, and `activeQueueItem` is the entry being run.
let queuePumping = false
let activeQueueItem = null
let runsFilterKeys = null
let runsFilterLabel = ''
let runsSortKey = null
let runsSortDir = 'desc'
let runsLeverFilter = {}
let runsTextFilter = ''
let runsCompareKeys = new Set()
let runsViewMode = 'runs'
// When drilled into a single setup's runs (via the by-setup view), this holds that
// setup's key so the runs view can show its conclusion-note editor (C4 ledger).
let runsDrillSetupKey = null
let chartSplits = {}
let itemBarTimer = null
// 1s ticker for the running item's elapsed timer (mm:ss) between the 3s polls.
let currentItemTimer = null
let runnersCache = []
// The runners offered in the Launch tab's "Run on" select, refreshed each time
// that tab is shown. Kept apart from runnersCache (the home panel's list) so the
// two surfaces never fight over each other's render state.
let launchRunnersCache = []
let runnerRemoveArmedId = null
let runnerPairing = null
// Bumped whenever the pairing sub-panel opens or closes; the observe + countdown
// loops capture it and stop the moment a newer pairing supersedes them.
let runnerPairingSession = 0
let runnerPairingKnownIds = null
let runnerCountdownTimer = null

// --- Utilities --------------------------------------------------------------
function escapeHtml(value) {
  return String(value ?? '').replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
  )
}
function truncate(value, max) {
  const s = String(value ?? '')
  return s.length > max ? s.slice(0, max - 1) + '…' : s
}
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
function byId(id) {
  return document.getElementById(id)
}
// Anti-flash assignment: the poll/observe loops re-render whole sections every
// 3s, and blindly re-assigning innerHTML replaces identical DOM (and resets any
// CSS animation, e.g. the indeterminate bar) every tick — which flashes. Only
// touch the DOM when the markup actually changed.
function setHtml(el, html) {
  if (!el || el.innerHTML === html) return
  el.innerHTML = html
}
function shortKey(key) {
  const k = String(key || '')
  return k.length > 10 ? k.slice(0, 10) : k
}
function formatObjective(value) {
  const v = Number(value)
  if (!Number.isFinite(v)) return '—'
  const a = Math.abs(v)
  if (a !== 0 && (a < 0.001 || a >= 1e6)) return v.toExponential(3)
  return String(Math.round(v * 10000) / 10000)
}
function formatEta(seconds) {
  const s = Math.max(0, Math.round(Number(seconds) || 0))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  if (h) return `${h}h ${m}m`
  if (m) return `${m}m ${sec}s`
  return `${sec}s`
}
function formatDuration(ms) {
  const v = Number(ms)
  if (!Number.isFinite(v) || v < 0) return '—'
  const s = Math.round(v / 1000)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  if (h) return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
  if (m) return `${m}:${String(sec).padStart(2, '0')}`
  return `${sec}s`
}
function formatWhen(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString()
}
// Coarse "n ago" phrasing for a recent timestamp (last-seen), falling back to the
// absolute time once it is more than a day old.
function formatRelative(iso) {
  if (!iso) return '—'
  const t = new Date(iso).getTime()
  if (Number.isNaN(t)) return String(iso)
  const sec = Math.max(0, Math.round((Date.now() - t) / 1000))
  if (sec < 45) return 'just now'
  const min = Math.round(sec / 60)
  if (min < 60) return `${min}m ago`
  const hr = Math.round(min / 60)
  if (hr < 24) return `${hr}h ago`
  return formatWhen(iso)
}
// mm:ss left until an ISO instant; clamped at 0:00 once it has passed.
function formatCountdown(iso) {
  const t = new Date(iso || '').getTime()
  const left = Number.isFinite(t) ? Math.max(0, Math.round((t - Date.now()) / 1000)) : 0
  const m = Math.floor(left / 60)
  const s = left % 60
  return `${m}:${String(s).padStart(2, '0')}`
}
// mm:ss elapsed since an ISO instant; clamped at 0:00 before it has started.
function formatElapsed(iso) {
  const t = new Date(iso || '').getTime()
  const sec = Number.isFinite(t) ? Math.max(0, Math.round((Date.now() - t) / 1000)) : 0
  const m = Math.floor(sec / 60)
  const s = sec % 60
  return `${m}:${String(s).padStart(2, '0')}`
}
function pairingExpired(pairing) {
  const t = new Date((pairing && pairing.expiresAt) || '').getTime()
  return !Number.isFinite(t) || t - Date.now() <= 0
}
function spinnerHtml() {
  return '<span class="spinner" aria-hidden="true"></span>'
}
// Small circular "?" button with a styled hover/focus callout (no native title).
function helpCalloutHtml(text) {
  return `<span class="help-callout"><button type="button" class="help-btn" aria-label="What does this do?">?</button><span class="help-pop" role="tooltip">${escapeHtml(text)}</span></span>`
}
function queuedStatusText(ahead) {
  return `Queued — ${ahead} ahead of it.`
}
function nowIso() {
  return new Date().toISOString()
}
function randomHexId() {
  const bytes = new Uint8Array(6)
  crypto.getRandomValues(bytes)
  return [...bytes].map((b) => b.toString(16).padStart(2, '0')).join('')
}
function slugifyProjectName(name) {
  return String(name || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}
function objectiveName() {
  return (manifest && manifest.objective && manifest.objective.name) || 'objective'
}
function objectiveDirection() {
  return (manifest && manifest.objective && manifest.objective.direction) === 'min' ? 'min' : 'max'
}

// --- Data layer (OverseerBridge DataStorage) ---------------------------------
function embedded() {
  return !!(window.OverseerBridge && window.OverseerBridge.embedded)
}
async function queryRecords(type, key) {
  if (!embedded()) return []
  try {
    const payload = key === undefined ? { type } : { type, key }
    const recs = await window.OverseerBridge.queryData(payload)
    return recs || []
  } catch {
    return []
  }
}
async function readRuns() {
  if (!manifest) return []
  const recs = await queryRecords(manifest.recordType)
  return recs
    .map((r) => {
      const summary = r.content || {}
      const key =
        r.key || (summary.provenance && summary.provenance.configHash) || summary.configHash || ''
      return { key, summary }
    })
    .filter((r) => r.key)
}
// Latest-record contents get the record-level timestamps merged in (when the
// content does not carry its own), so observers can anchor time estimates on
// when the record was actually written.
async function readLatestRecord(suffix) {
  if (!manifest) return null
  const recs = await queryRecords(manifest.recordType + suffix, 'latest')
  const rec = recs[0]
  if (!rec || !rec.content) return null
  const content = rec.content
  return {
    ...content,
    startedAt: content.startedAt || rec.createdAt,
    updatedAt: content.updatedAt || rec.updatedAt,
  }
}
async function readProgress() {
  return readLatestRecord('-progress')
}
async function readCampaign() {
  return readLatestRecord('-campaign')
}
async function readVerdicts() {
  if (!manifest) return new Map()
  const recs = await queryRecords(manifest.recordType + '-verdict')
  const map = new Map()
  for (const r of recs) {
    const content = r.content || {}
    const key = r.key || content.key || ''
    if (key) map.set(key, content)
  }
  return map
}
async function readJudgement() {
  return readLatestRecord('-judgement')
}
async function readEvaluations() {
  if (!manifest) return new Map()
  const recs = await queryRecords(manifest.recordType + '-evaluation')
  const map = new Map()
  for (const r of recs) {
    const content = r.content || {}
    const key = r.key || content.runKey || ''
    if (key) map.set(key, content)
  }
  return map
}
async function readProposal() {
  return readLatestRecord('-proposal')
}
// Per-setup user notes (your conclusion for a setup), keyed by setupKey — the user
// half of the ledger's "current conclusion" (the rest is score + LLM verdict).
async function readNotes() {
  if (!manifest) return new Map()
  const recs = await queryRecords(manifest.recordType + '-note')
  const map = new Map()
  for (const r of recs) {
    const key = r.key || (r.content && r.content.setupKey) || ''
    if (key) map.set(key, r.content || {})
  }
  return map
}
async function saveSetupNote(setupKey, note) {
  if (!manifest || !setupKey) return
  await window.OverseerBridge.putData({
    type: manifest.recordType + '-note',
    key: setupKey,
    content: { setupKey, note: String(note || ''), updatedAt: nowIso() },
  })
  notesCache = await readNotes()
}
async function readHypotheses() {
  if (!manifest) return []
  const recs = await queryRecords(manifest.recordType + '-hypothesis')
  return recs
    .map((r) => {
      const content = r.content || {}
      return { ...content, id: content.id || r.key || '' }
    })
    .filter((h) => h.id)
}
async function putHypothesis(content) {
  await window.OverseerBridge.putData({
    type: manifest.recordType + '-hypothesis',
    key: content.id,
    content,
  })
}
async function readProjects() {
  const recs = await queryRecords(PROJECT_RECORD_TYPE)
  return recs
    .map((r) => {
      const content = r.content || {}
      return { ...content, key: content.key || r.key || '' }
    })
    .filter((p) => p.key)
    .sort((a, b) => String(a.addedAt || '').localeCompare(String(b.addedAt || '')))
}
async function readManifestRecords() {
  const recs = await queryRecords(PROJECT_MANIFEST_RECORD_TYPE)
  const map = new Map()
  for (const r of recs) {
    const key = r.key || (r.content && r.content.projectKey) || ''
    if (key) map.set(key, r.content || {})
  }
  return map
}
// The set of run-record keys already SEEN by the user, per project (keyed by
// projectKey, content { keys, updatedAt }) — persisted so the unseen badge
// survives reloads and devices, not just this session.
async function readSeenKeys() {
  const recs = await queryRecords(SEEN_RECORD_TYPE)
  const map = new Map()
  for (const r of recs) {
    const key = r.key || (r.content && r.content.projectKey) || ''
    const keys = (r.content && Array.isArray(r.content.keys) && r.content.keys) || []
    if (key) map.set(key, new Set(keys))
  }
  return map
}
async function putSeenKeys(projectKey, keys) {
  await window.OverseerBridge.putData({
    type: SEEN_RECORD_TYPE,
    key: projectKey,
    content: { projectKey, keys: [...keys], updatedAt: nowIso() },
  })
}
// The current run-record keys for one project, read through its manifest's
// recordType (run records key on the configHash). Empty for an uninspected
// project or one with no manifest.
async function readProjectRunKeys(projectKey) {
  const rec = manifestsCache.get(projectKey)
  const recordType = rec && rec.manifest && rec.manifest.recordType
  if (!recordType) return []
  const recs = await queryRecords(recordType)
  return recs
    .map((r) => {
      const summary = r.content || {}
      return (
        r.key || (summary.provenance && summary.provenance.configHash) || summary.configHash || ''
      )
    })
    .filter(Boolean)
}
// Queue records are global ('trainer-queue') but project-scoped through the
// stored params.recordType; marker items (auto-eval intents) ride the same
// record type and are filtered out of the visible queue.
async function readQueueRecords() {
  if (!manifest) return []
  const recs = await queryRecords(QUEUE_RECORD_TYPE)
  return recs
    .map((r) => {
      const content = r.content || {}
      return { ...content, id: content.id || r.key || '' }
    })
    .filter((q) => q.id && q.params && q.params.recordType === manifest.recordType)
}
async function readQueue() {
  const items = await readQueueRecords()
  return items
    .filter((q) => !q.marker)
    .sort(
      (a, b) =>
        String(a.queuedAt || '').localeCompare(String(b.queuedAt || '')) ||
        a.id.localeCompare(b.id),
    )
}
async function putQueueItem(item) {
  await window.OverseerBridge.putData({ type: QUEUE_RECORD_TYPE, key: item.id, content: item })
}
async function deleteQueueItem(id) {
  try {
    await window.OverseerBridge.deleteData({ type: QUEUE_RECORD_TYPE, key: id })
  } catch {
    // a missing record is already gone — nothing to surface
  }
}

// --- Home (hub) ----------------------------------------------------------------
function projectHasManifest(projectKey) {
  const rec = manifestsCache.get(projectKey)
  return !!(rec && !rec.error && rec.manifest && rec.manifest.recordType)
}
function manifestStatusHtml(projectKey) {
  if (inspectingKeys.has(projectKey)) {
    return `<p class="project-status is-muted">${spinnerHtml()} Inspecting…</p>`
  }
  const rec = manifestsCache.get(projectKey)
  if (!rec) return '<p class="project-status is-muted">Not inspected yet.</p>'
  if (rec.error) return `<p class="project-status is-error">${escapeHtml(rec.error)}</p>`
  const m = rec.manifest
  if (!m || !m.recordType) {
    return '<p class="project-status is-error">Inspection found no usable manifest.</p>'
  }
  const obj = m.objective || {}
  const direction = obj.direction === 'min' ? 'min' : 'max'
  const leverCount = Object.keys(m.levers || {}).length
  return `<p class="project-status">${escapeHtml(obj.name || 'objective')} (${direction}) · ${leverCount} lever${leverCount === 1 ? '' : 's'}</p>`
}
// Whether a project's training has a live activity right now — its manifest's
// recordType is in the running-and-live set the home poll tracks.
function projectHasLiveActivity(projectKey) {
  const rec = manifestsCache.get(projectKey)
  const recordType = rec && rec.manifest && rec.manifest.recordType
  return !!recordType && liveRecordTypes.has(recordType)
}
// How many of a project's current run keys the user has not seen yet (its run
// keys minus the persisted seen set).
function projectUnseenCount(projectKey) {
  const keys = runKeysByProject.get(projectKey) || []
  if (!keys.length) return 0
  const seen = seenKeysByProject.get(projectKey) || new Set()
  return keys.reduce((n, k) => (seen.has(k) ? n : n + 1), 0)
}
function totalUnseenCount() {
  return projectsCache.reduce((n, p) => n + projectUnseenCount(p.key), 0)
}
function projectCardHtml(project) {
  const key = escapeHtml(project.key)
  const armed = removeArmedKey === project.key
  const live = projectHasLiveActivity(project.key)
  const unseen = projectUnseenCount(project.key)
  const corner =
    `${unseen > 0 ? `<span class="new-badge" title="Unseen runs">${unseen} new</span>` : ''}` +
    `${live ? `<span class="card-spinner" aria-label="Training">${spinnerHtml()}</span>` : ''}`
  return `<article class="project-card" data-key="${key}">
    ${corner ? `<div class="project-card-corner">${corner}</div>` : ''}
    <h3>${escapeHtml(project.name || project.key)}</h3>
    <p class="project-dir">${escapeHtml(project.dir || '')}</p>
    ${manifestStatusHtml(project.key)}
    <div class="project-actions">
      <button type="button" data-action="open" data-key="${key}"${projectHasManifest(project.key) ? '' : ' disabled'}>Open</button>
      <button type="button" data-action="inspect" data-key="${key}" class="ghost-btn"${inspectingKeys.has(project.key) ? ' disabled' : ''}>Re-inspect</button>
      <button type="button" data-action="remove" data-key="${key}" class="${armed ? 'danger-btn' : 'ghost-btn'}">${armed ? 'Confirm' : 'Remove'}</button>
    </div>
  </article>`
}
// The total-unseen pill next to the "Model Trainer" home header.
function renderHomeHeaderBadge() {
  const h = document.querySelector('.home-header h1')
  if (!h) return
  let badge = byId('home-unseen-badge')
  const total = totalUnseenCount()
  if (!total) {
    if (badge) badge.remove()
    return
  }
  if (!badge) {
    badge = document.createElement('span')
    badge.id = 'home-unseen-badge'
    badge.className = 'new-badge header-badge'
    h.insertAdjacentElement('afterend', badge)
  }
  badge.textContent = `${total} new`
}
// Refresh just the home-overview surfaces (project cards + header badge) from
// the current caches, without re-reading projects/manifests — the home poll
// calls this every tick, so it uses setHtml to avoid re-rendering identical DOM.
function renderHomeOverview() {
  const body = byId('home-projects')
  if (!body || !projectsCache.length) {
    renderHomeHeaderBadge()
    return
  }
  setHtml(body, `<div class="project-cards">${projectsCache.map(projectCardHtml).join('')}</div>`)
  renderHomeHeaderBadge()
}
async function renderHome() {
  const body = byId('home-projects')
  if (!body) return
  if (!embedded()) {
    setHtml(
      body,
      '<div class="empty-hint">Open inside the Overseer to manage training projects.</div>',
    )
    return
  }
  ;[projectsCache, manifestsCache, seenKeysByProject] = await Promise.all([
    readProjects(),
    readManifestRecords(),
    readSeenKeys(),
  ])
  if (!projectsCache.length) {
    setHtml(
      body,
      '<div class="empty-hint">No training projects yet — add one (try examples/cartpole).</div>',
    )
    renderHomeHeaderBadge()
    return
  }
  await refreshHomeLiveState()
  renderHomeOverview()
}
// Read the per-project live-activity set and current run keys that drive the card
// spinners + unseen badges, then re-render the overview. Cheap enough to poll.
async function refreshHomeLiveState() {
  liveRecordTypes = await readLiveRecordTypes()
  const entries = await Promise.all(
    projectsCache.map(async (p) => [p.key, await readProjectRunKeys(p.key)]),
  )
  runKeysByProject = new Map(entries)
}
// Every recordType with a running, live activity — matches a card to its project
// via the manifest's recordType.
async function readLiveRecordTypes() {
  try {
    const res = await window.OverseerBridge.listActivities()
    return new Set(
      ((res && res.activities) || [])
        .filter((a) => a.status === 'running' && a.isLive !== false && a.recordType)
        .map((a) => a.recordType),
    )
  } catch {
    return new Set()
  }
}
// Lightweight 3s poll while on the home screen: refresh the live-activity set +
// run keys so the card spinners and unseen badges stay current. A newer session
// (opening a project bumps homePollSession) or navigating away stops it; it is
// projectEpoch-guarded so a stale loop never paints over an opened project.
async function startHomePoll() {
  const session = ++homePollSession
  const epoch = projectEpoch
  while (session === homePollSession && epoch === projectEpoch && embedded()) {
    await sleep(POLL_MS)
    if (session !== homePollSession || epoch !== projectEpoch) return
    if (document.hidden || !projectsCache.length) continue
    await refreshHomeLiveState()
    if (session !== homePollSession || epoch !== projectEpoch) return
    seenKeysByProject = await readSeenKeys()
    if (session !== homePollSession || epoch !== projectEpoch) return
    renderHomeOverview()
  }
}
function stopHomePoll() {
  homePollSession += 1
}
// Run the backend inspect for one registered project: it reads the directory's
// trainer manifest and writes the trainer-project-manifest record this app
// re-reads (via renderHome) once the activity settles.
async function inspectProject(projectKey, dir, manifestRelPath) {
  if (inspectingKeys.has(projectKey)) return
  inspectingKeys.add(projectKey)
  setStatusLine('projects-status', '')
  await renderHome()
  try {
    const started = await window.OverseerBridge.startActivity('inspect-trainer', {
      projectKey,
      dir,
      ...(manifestRelPath ? { manifestRelPath } : {}),
    })
    const activityId = started && started.activityId
    if (!activityId) throw new Error('no activity id')
    const act = await observeQuickActivity(activityId)
    const failure = quickActivityFailureText(act, 'Inspecting')
    if (failure) setStatusLine('projects-status', failure, true)
  } catch {
    setStatusLine('projects-status', 'Could not start inspecting — please try again.', true)
  } finally {
    inspectingKeys.delete(projectKey)
    await renderHome()
  }
}
// Two-click remove: the first click arms the button ("Confirm"), the second
// deletes the trainer-project record. The manifest + run records are kept, so
// re-adding the same project picks its history right back up.
async function removeProject(projectKey) {
  if (removeArmedKey !== projectKey) {
    removeArmedKey = projectKey
    await renderHome()
    return
  }
  removeArmedKey = null
  try {
    await window.OverseerBridge.deleteData({ type: PROJECT_RECORD_TYPE, key: projectKey })
  } catch {
    setStatusLine('projects-status', 'Could not remove the project — please try again.', true)
  }
  await renderHome()
}
async function onAddProjectSubmit(event) {
  event.preventDefault()
  const form = byId('project-add-form')
  if (!form) return
  if (!embedded()) {
    setStatusLine('project-form-error', 'Open inside the Overseer to add training projects.', true)
    return
  }
  const name = String((form.elements.name && form.elements.name.value) || '').trim()
  const dir = String((form.elements.dir && form.elements.dir.value) || '').trim()
  const manifestRelPath = String(
    (form.elements.manifestRelPath && form.elements.manifestRelPath.value) || '',
  ).trim()
  const key = slugifyProjectName(name)
  if (!name || !key) {
    setStatusLine('project-form-error', 'Give the project a name with letters or digits.', true)
    return
  }
  if (!dir) {
    setStatusLine('project-form-error', 'Give the project a directory.', true)
    return
  }
  if (projectsCache.some((p) => p.key === key)) {
    setStatusLine('project-form-error', `A project with key "${key}" already exists.`, true)
    return
  }
  const saveBtn = byId('project-save-btn')
  if (saveBtn) saveBtn.disabled = true
  try {
    await window.OverseerBridge.putData({
      type: PROJECT_RECORD_TYPE,
      key,
      content: {
        key,
        name,
        dir,
        ...(manifestRelPath ? { manifestRelPath } : {}),
        addedAt: nowIso(),
      },
    })
  } catch {
    setStatusLine('project-form-error', 'Could not save the project — please try again.', true)
    return
  } finally {
    if (saveBtn) saveBtn.disabled = false
  }
  setStatusLine('project-form-error', '')
  form.reset()
  inspectProject(key, dir, manifestRelPath)
}
function onHomeProjectsClick(event) {
  const btn = event.target.closest('button[data-action]')
  if (!btn) return
  const { action, key } = btn.dataset
  if (removeArmedKey && (action !== 'remove' || key !== removeArmedKey)) removeArmedKey = null
  if (action === 'open') openProject(key)
  else if (action === 'inspect') {
    const project = projectsCache.find((p) => p.key === key)
    if (project) inspectProject(project.key, project.dir, project.manifestRelPath)
  } else if (action === 'remove') removeProject(key)
}
function setupHome() {
  const form = byId('project-add-form')
  if (form) form.addEventListener('submit', onAddProjectSubmit)
  const list = byId('home-projects')
  if (list) list.addEventListener('click', onHomeProjectsClick)
  const back = byId('back-btn')
  if (back) back.addEventListener('click', goHome)
}

// --- Compute Runners (home panel) --------------------------------------------
// Remote machines that poll the backend for training jobs. Pairing mints a
// short-lived PIN entered on the runner; this app only shows the PIN and watches
// the runner list for the new machine to appear. Every call goes through the
// host bridge — the host holds the credential, this app never does.
function runnerStatusHtml(runner) {
  if (runner.online) {
    const dot = '<span class="status-dot is-online" aria-hidden="true"></span>'
    const label = runner.busy ? 'Online — running a job' : 'Online — idle'
    return `${dot}${escapeHtml(label)}`
  }
  return '<span class="status-dot is-offline" aria-hidden="true"></span>Offline'
}
function runnerRowHtml(runner) {
  const id = escapeHtml(runner.id)
  const armed = runnerRemoveArmedId === runner.id
  const queued = Number(runner.queued) || 0
  const queuedBit = queued > 0 ? ` · <span class="runner-queued">${queued} queued</span>` : ''
  const seen = runner.lastSeenAt
    ? `last seen ${escapeHtml(formatRelative(runner.lastSeenAt))}`
    : 'never seen'
  return `<article class="runner-card" data-runner-id="${id}">
    <div class="runner-main">
      <h4>${escapeHtml(runner.name || runner.id)}</h4>
      <p class="runner-status">${runnerStatusHtml(runner)}${queuedBit}</p>
      <p class="card-sub runner-meta">Paired ${escapeHtml(formatWhen(runner.createdAt))} · ${seen}</p>
      <p class="runner-id">
        <code>${id}</code>
        <button type="button" class="runner-copy-id" data-copy="${id}">copy id</button>
      </p>
    </div>
    <div class="runner-actions">
      <button type="button" data-action="remove-runner" data-id="${id}" class="${armed ? 'danger-btn' : 'ghost-btn'}">${armed ? 'Confirm' : 'Remove'}</button>
    </div>
  </article>`
}
async function renderRunnersPanel(showSpinner) {
  const body = byId('runners-body')
  if (!body) return
  if (!embedded()) {
    body.innerHTML =
      '<div class="empty-hint">Open inside the Overseer to manage compute runners.</div>'
    return
  }
  if (showSpinner) {
    body.innerHTML = `<p class="card-sub">${spinnerHtml()} Loading runners…</p>`
  }
  let runners
  try {
    const res = await window.OverseerBridge.listRunners()
    runners = (res && res.runners) || []
  } catch {
    setStatusLine('runners-status', 'Could not reach the backend to list runners.', true)
    body.innerHTML = ''
    return
  }
  setStatusLine('runners-status', '')
  runnersCache = runners
  if (runnerRemoveArmedId && !runners.some((r) => r.id === runnerRemoveArmedId)) {
    runnerRemoveArmedId = null
  }
  if (!runners.length) {
    body.innerHTML =
      '<div class="empty-hint">No runners paired. Pair one to run training on another machine.</div>'
    return
  }
  const ordered = [...runners].sort((a, b) =>
    String(a.createdAt || '').localeCompare(String(b.createdAt || '')),
  )
  body.innerHTML = `<div class="runner-cards">${ordered.map(runnerRowHtml).join('')}</div>`
}
// Toggle the panel open/closed from the header button; opening loads the list
// (with a spinner) and closing tears down any open pairing sub-panel.
function toggleRunnersPanel(show) {
  const panel = byId('runners-panel')
  const btn = byId('runners-toggle')
  if (!panel) return
  const open = show === undefined ? panel.hidden : show
  panel.hidden = !open
  if (btn) btn.setAttribute('aria-expanded', String(open))
  if (open) {
    setStatusLine('runners-status', '')
    renderRunnersPanel(true)
  } else {
    closeRunnerPairing()
  }
}
// Two-click remove mirroring the projects list: first click arms ("Confirm"),
// second deletes the runner pairing server-side, then the list refreshes.
async function removeRunnerArmed(id) {
  if (runnerRemoveArmedId !== id) {
    runnerRemoveArmedId = id
    await renderRunnersPanel(false)
    return
  }
  runnerRemoveArmedId = null
  try {
    await window.OverseerBridge.removeRunner(id)
  } catch {
    setStatusLine('runners-status', 'Could not remove the runner — please try again.', true)
  }
  await renderRunnersPanel(false)
}
// Copyable setup commands shown to the user; the backend base IS this app's
// origin (the backend serves it), so the runner pairs straight back to it.
function pairingCommandsText(pin) {
  const origin = window.location.origin
  return (
    `node runner/agent.mjs pair --backend ${origin} --pin ${pin} --name my-runner\n` +
    'node runner/agent.mjs run'
  )
}
function renderRunnerPairing() {
  const el = byId('runner-pairing')
  if (!el) return
  if (!runnerPairing) {
    el.hidden = true
    el.innerHTML = ''
    return
  }
  const expired = pairingExpired(runnerPairing)
  const pin = String(runnerPairing.pin || '')
  const commands = pairingCommandsText(pin)
  const countdown = expired
    ? '<span class="runner-countdown is-expired">PIN expired</span><button type="button" id="runner-new-pin" class="ghost-btn">New PIN</button>'
    : `<span class="runner-countdown">Expires in <strong id="runner-countdown-value">${escapeHtml(formatCountdown(runnerPairing.expiresAt))}</strong></span>`
  el.hidden = false
  el.innerHTML = `
    <div class="card-head card-head-row">
      <div>
        <h3>Pair a new runner</h3>
        <p class="card-sub">
          Run these on the machine that will train (needs this repo + npm install). It pairs once,
          then polls for jobs.
        </p>
      </div>
      <button type="button" id="runner-pair-close" class="ghost-btn">Close</button>
    </div>
    <div class="runner-pin${expired ? ' is-expired' : ''}">${escapeHtml(pin)}</div>
    <p class="runner-countdown-row">${countdown}</p>
    <div class="runner-commands">
      <pre>${escapeHtml(commands)}</pre>
      <button type="button" id="runner-copy-cmd" class="ghost-btn">Copy commands</button>
    </div>`
}
// Local 1s ticker for the PIN countdown; on expiry it re-renders the sub-panel
// (showing "PIN expired" + a New PIN button) and stops the pairing observer.
function syncRunnerCountdown(active) {
  if (runnerCountdownTimer) {
    clearInterval(runnerCountdownTimer)
    runnerCountdownTimer = null
  }
  if (!active) return
  runnerCountdownTimer = setInterval(() => {
    if (!runnerPairing) {
      syncRunnerCountdown(false)
      return
    }
    if (pairingExpired(runnerPairing)) {
      syncRunnerCountdown(false)
      renderRunnerPairing()
      return
    }
    const value = byId('runner-countdown-value')
    if (value) value.textContent = formatCountdown(runnerPairing.expiresAt)
  }, 1000)
}
// While the pairing sub-panel is open and the PIN is live, poll the runner list
// every 3s; the first id that was not present when pairing started is the new
// machine — announce it, close the sub-panel and refresh. A newer pairing
// session (or a close) supersedes this loop via runnerPairingSession.
async function observeRunnerPairing() {
  const session = runnerPairingSession
  while (session === runnerPairingSession && runnerPairing && !pairingExpired(runnerPairing)) {
    await sleep(POLL_MS)
    if (session !== runnerPairingSession || !runnerPairing) return
    let runners
    try {
      const res = await window.OverseerBridge.listRunners()
      runners = (res && res.runners) || []
    } catch {
      continue
    }
    if (session !== runnerPairingSession || !runnerPairing) return
    const known = runnerPairingKnownIds || new Set()
    const fresh = runners.find((r) => !known.has(r.id))
    if (fresh) {
      setStatusLine('runners-status', `Paired ✓ ${fresh.name || fresh.id}`)
      closeRunnerPairing()
      runnersCache = runners
      await renderRunnersPanel(false)
      return
    }
  }
}
async function onPairRunner() {
  if (!embedded()) {
    setStatusLine('runners-status', 'Open inside the Overseer to pair compute runners.', false)
    return
  }
  const btn = byId('runner-pair-btn')
  if (btn) btn.disabled = true
  setStatusLine('runners-status', '')
  let pairing
  try {
    pairing = await window.OverseerBridge.createRunnerPairing()
  } catch {
    setStatusLine('runners-status', 'Could not start pairing — please try again.', true)
    if (btn) btn.disabled = false
    return
  }
  if (btn) btn.disabled = false
  if (!pairing || !pairing.pin) {
    setStatusLine('runners-status', 'Pairing did not return a PIN — please try again.', true)
    return
  }
  // Snapshot the runners present right now so the observer can tell the newly
  // paired machine apart from those already known.
  let known = runnersCache.map((r) => r.id)
  try {
    const res = await window.OverseerBridge.listRunners()
    known = ((res && res.runners) || []).map((r) => r.id)
  } catch {
    // fall back to the last rendered list — pairing detection stays best-effort
  }
  runnerPairing = pairing
  runnerPairingSession += 1
  runnerPairingKnownIds = new Set(known)
  renderRunnerPairing()
  syncRunnerCountdown(!pairingExpired(pairing))
  observeRunnerPairing()
}
function closeRunnerPairing() {
  runnerPairing = null
  runnerPairingSession += 1
  runnerPairingKnownIds = null
  syncRunnerCountdown(false)
  renderRunnerPairing()
}
// navigator.clipboard with a hidden-textarea fallback for sandboxed frames; the
// button briefly confirms either way.
async function copyText(text, button) {
  let ok = false
  try {
    await navigator.clipboard.writeText(text)
    ok = true
  } catch {
    try {
      const ta = document.createElement('textarea')
      ta.value = text
      ta.setAttribute('readonly', '')
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.append(ta)
      ta.select()
      ok = document.execCommand('copy')
      ta.remove()
    } catch {
      ok = false
    }
  }
  if (button) {
    const original = button.textContent
    button.textContent = ok ? 'Copied ✓' : 'Copy failed'
    setTimeout(() => {
      button.textContent = original
    }, 1500)
  }
}
function setupRunners() {
  const toggle = byId('runners-toggle')
  if (toggle) toggle.addEventListener('click', () => toggleRunnersPanel())
  const pairBtn = byId('runner-pair-btn')
  if (pairBtn) pairBtn.addEventListener('click', onPairRunner)
  const body = byId('runners-body')
  if (body) {
    body.addEventListener('click', (event) => {
      const copyBtn = event.target.closest('button[data-copy]')
      if (copyBtn) {
        copyText(copyBtn.dataset.copy, copyBtn)
        return
      }
      const actionBtn = event.target.closest('button[data-action="remove-runner"]')
      if (actionBtn) {
        if (runnerRemoveArmedId && actionBtn.dataset.id !== runnerRemoveArmedId) {
          runnerRemoveArmedId = null
        }
        removeRunnerArmed(actionBtn.dataset.id)
      }
    })
  }
  const pairing = byId('runner-pairing')
  if (pairing) {
    pairing.addEventListener('click', (event) => {
      if (event.target.closest('#runner-pair-close')) {
        closeRunnerPairing()
        return
      }
      if (event.target.closest('#runner-new-pin')) {
        onPairRunner()
        return
      }
      const copyCmd = event.target.closest('#runner-copy-cmd')
      if (copyCmd && runnerPairing) {
        copyText(pairingCommandsText(String(runnerPairing.pin)), copyCmd)
      }
    })
  }
}

// --- Navigation (home ↔ project dashboard) ----------------------------------
function showView(view) {
  const home = byId('view-home')
  if (home) home.hidden = view !== 'home'
  const dash = byId('view-dashboard')
  if (dash) dash.hidden = view !== 'dashboard'
}
function resetDashboardState() {
  runsCache = []
  verdictsCache = new Map()
  evaluationsCache = new Map()
  evaluatingKeys.clear()
  judgementSummary = null
  hypothesesCache = []
  proposalSummary = null
  selectedRunKey = null
  currentActivityId = null
  lastActivityStatus = null
  lastProgress = null
  lastCampaign = null
  judging = false
  proposing = false
  queueCache = []
  runsFilterKeys = null
  runsFilterLabel = ''
  chartSplits = {}
  launchRunnersCache = []
  syncItemBarTimer(false)
  syncCurrentItemTimer(null)
  closeRunDetail()
  toggleHypothesisForm(false)
  setStatusLine('judge-status', '')
  setStatusLine('hypotheses-status', '')
  const live = byId('runs-live')
  if (live) {
    live.innerHTML = ''
    live.hidden = true
  }
  const runsBody = byId('runs-body')
  if (runsBody) runsBody.innerHTML = ''
  const spark = byId('runs-sparkline')
  if (spark) {
    spark.innerHTML = ''
    spark.hidden = true
  }
  const hypothesesBody = byId('hypotheses-body')
  if (hypothesesBody) hypothesesBody.innerHTML = ''
  const chartsBody = byId('charts-body')
  if (chartsBody) chartsBody.innerHTML = ''
  const activityBody = byId('activity-body')
  if (activityBody) activityBody.innerHTML = ''
}
async function openProject(projectKey) {
  const project = projectsCache.find((p) => p.key === projectKey)
  if (!project || !projectHasManifest(projectKey)) return
  stopHomePoll()
  projectEpoch += 1
  const epoch = projectEpoch
  currentProject = project
  manifest = manifestsCache.get(projectKey).manifest
  removeArmedKey = null
  closeRunnerPairing()
  resetDashboardState()
  applyManifestChrome()
  renderLaunchForm()
  showView('dashboard')
  showTab(savedTabId() || TABS[0].id)
  await renderRuns()
  await resumeRunningActivity()
  if (epoch !== projectEpoch) return
  await resumeRunningQuickActivity()
  if (epoch !== projectEpoch) return
  await refreshQueue()
  await processSettledCampaignEffects()
  if (epoch !== projectEpoch) return
  pumpQueue()
}
function goHome() {
  projectEpoch += 1
  currentProject = null
  manifest = null
  syncItemBarTimer(false)
  syncCurrentItemTimer(null)
  document.title = 'Model Trainer'
  showView('home')
  startHomePoll()
  renderHome()
}
function applyManifestChrome() {
  const name = (currentProject && currentProject.name) || (manifest && manifest.name) || 'Trainer'
  document.title = `${name} — Model Trainer`
  const head = byId('dash-project-name')
  if (head) head.textContent = name
  const foot = byId('foot-label')
  if (foot) foot.textContent = `${name} — trainer viewer`
  const sub = byId('runs-sub')
  if (sub) {
    const dir = objectiveDirection() === 'min' ? 'lowest' : 'highest'
    sub.textContent = `Every training run, ${dir} ${objectiveName()} first.`
  }
}
function setBanner(text) {
  const banner = byId('app-banner')
  if (!banner) return
  banner.textContent = text || ''
  banner.hidden = !text
}
// Base params for every per-project activity: the manifest's record type plus
// the selected project's directory, so the backend works on the right project.
function trainerActivityParams(extra) {
  const params = {
    recordType: manifest.recordType,
    dir: currentProject ? currentProject.dir : undefined,
  }
  if (currentProject && currentProject.manifestRelPath) {
    params.manifestRelPath = currentProject.manifestRelPath
  }
  return extra ? { ...params, ...extra } : params
}
// Params for the activities that execute the project's training code — 'train'
// and 'evaluate' — which also carry the remote compute target when one is set.
// Judge / propose never take a compute target.
function trainerComputeParams(extra) {
  const computeTarget = remoteComputeTarget(savedComputeTarget())
  return trainerActivityParams(computeTarget ? { ...extra, computeTarget } : extra)
}

// --- Activity queue (persisted, project-scoped) -------------------------------
// Every trainer activity goes through startOrEnqueue: while another activity for
// this project is live the request lands in the 'trainer-queue' records instead,
// and pumpQueue drains the queue head-first whenever the live one settles.
async function findLiveTrainerActivity() {
  if (!manifest) return null
  try {
    const res = await window.OverseerBridge.listActivities()
    return (
      ((res && res.activities) || []).find(
        (a) => a.recordType === manifest.recordType && a.status === 'running' && a.isLive !== false,
      ) || null
    )
  } catch {
    return null
  }
}
async function readLiveActivityIds() {
  try {
    const res = await window.OverseerBridge.listActivities()
    return new Set(
      ((res && res.activities) || [])
        .filter((a) => a.status === 'running' && a.isLive !== false)
        .map((a) => a.activityId),
    )
  } catch {
    return new Set()
  }
}
function trainerBusyLocally() {
  return queuePumping || !!activeQueueItem || (observing && lastActivityStatus === 'running')
}
async function startOrEnqueue(activityType, params, label, extra) {
  const live = await findLiveTrainerActivity()
  if (!live && !trainerBusyLocally()) {
    const started = await window.OverseerBridge.startActivity(activityType, params)
    const activityId = started && started.activityId
    if (!activityId) throw new Error('no activity id')
    return { started: true, activityId }
  }
  const queue = await readQueue()
  const item = {
    id: randomHexId(),
    activityType,
    params,
    label,
    queuedAt: nowIso(),
    ...(extra || {}),
  }
  await putQueueItem(item)
  await refreshQueue()
  return { queued: true, id: item.id, ahead: queue.length + 1 }
}
async function refreshQueue() {
  queueCache = await readQueue()
  renderActivity()
}
async function onQueueRemove(id) {
  const item = queueCache.find((q) => q.id === id)
  await deleteQueueItem(id)
  if (item && item.hypothesisId && item.params && item.params.recordType) {
    await clearHypothesisCampaign(item.params.recordType, item.hypothesisId, id)
    if (activeTabId === 'hypotheses') await renderHypotheses()
  }
  await refreshQueue()
}
// Drain the queue: while nothing for this project is live, pop the head record,
// start its stored activity and observe it to settlement before the next pop.
// Double-dispatch is guarded by the in-memory queuePumping flag; navigation is
// guarded by projectEpoch.
async function pumpQueue() {
  if (queuePumping || !embedded() || !manifest) return
  const epoch = projectEpoch
  queuePumping = true
  try {
    while (epoch === projectEpoch) {
      if (lastActivityStatus === 'paused') return
      if (await findLiveTrainerActivity()) return
      const queue = await readQueue()
      if (epoch !== projectEpoch || !queue.length) return
      const head = queue[0]
      await deleteQueueItem(head.id)
      queueCache = queueCache.filter((q) => q.id !== head.id)
      const settled = await dispatchQueueItem(head, epoch)
      if (!settled) return
    }
  } finally {
    queuePumping = false
    activeQueueItem = null
  }
}
// Start one dequeued item and observe it until it settles. Returns true when
// the pump should continue with the next item.
async function dispatchQueueItem(item, epoch) {
  activeQueueItem = item
  renderActivity()
  let activityId = null
  try {
    const started = await window.OverseerBridge.startActivity(item.activityType, item.params)
    activityId = started && started.activityId
    if (!activityId) throw new Error('no activity id')
  } catch {
    activeQueueItem = null
    if (epoch === projectEpoch) renderActivity()
    return true
  }
  if (item.autoEval) await putAutoEvalMarker(activityId, item.params)
  if (item.hypothesisId && item.params && item.params.recordType) {
    await stampHypothesisCampaign(item.params.recordType, item.hypothesisId, {
      activityId,
      launchedAt: nowIso(),
      status: 'running',
    })
  }
  if (item.activityType === 'train') {
    activeQueueItem = null
    if (epoch !== projectEpoch) return false
    currentActivityId = activityId
    lastProgress = null
    lastCampaign = null
    lastActivityStatus = 'running'
    renderActivity()
    const status = await observeActivityUntilDone(activityId)
    return status === 'completed' || status === 'failed' || status === 'aborted'
  }
  applyQuickDispatchState(item, true)
  const act = await observeQuickActivity(activityId)
  applyQuickDispatchState(item, false)
  activeQueueItem = null
  if (epoch !== projectEpoch) return false
  renderActivity()
  await refreshAfterQuickDispatch(item, act)
  return !!act
}
// Mirror the busy state the direct start sites set, so the same buttons and
// spinners light up when the queue dispatches a judge / propose / evaluate.
function applyQuickDispatchState(item, busy) {
  if (item.activityType === 'judge') {
    judging = busy
    renderJudgeControls()
  } else if (item.activityType === 'propose') {
    proposing = busy
    renderProposeControls()
  } else if (item.activityType === 'evaluate') {
    const key = (item.params && item.params.runKey) || ''
    if (!key) return
    if (busy) evaluatingKeys.add(key)
    else evaluatingKeys.delete(key)
    if (selectedRunKey === key) renderRunDetail(key)
  }
}
async function refreshAfterQuickDispatch(item, act) {
  if (item.activityType === 'judge') {
    setStatusLine('judge-status', quickActivityFailureText(act, 'Judging'), true)
    await renderRuns()
  } else if (item.activityType === 'propose') {
    setStatusLine('hypotheses-status', quickActivityFailureText(act, 'Proposing'), true)
    await renderHypotheses()
  } else if (item.activityType === 'evaluate') {
    await renderRuns()
  }
}
// The auto-eval intent survives reloads as a hidden marker record in the queue:
// it names the campaign activity it belongs to, and is consumed (evaluations of
// the campaign's completed-with-checkpoint runs get enqueued) once that
// campaign's record lands.
async function putAutoEvalMarker(activityId, params) {
  await putQueueItem({
    id: randomHexId(),
    marker: 'auto-eval',
    activityId,
    params: {
      recordType: params.recordType,
      dir: params.dir,
      ...(params.computeTarget ? { computeTarget: params.computeTarget } : {}),
    },
    queuedAt: nowIso(),
  })
}
async function processAutoEvalMarkers() {
  const markers = (await readQueueRecords()).filter((q) => q.marker === 'auto-eval')
  if (!markers.length) return
  const campaign = await readCampaign()
  for (const marker of markers) {
    if (campaign && campaign.activityId === marker.activityId) {
      if (!campaign.finishedAt) continue
      if (!campaign.aborted) await enqueueMissingEvaluations(campaign, marker.params)
      await deleteQueueItem(marker.id)
      continue
    }
    const act = await getActivity(marker.activityId)
    if (act && (act.status === 'running' || act.status === 'paused')) continue
    await deleteQueueItem(marker.id)
  }
  await refreshQueue()
}
async function enqueueMissingEvaluations(campaign, baseParams) {
  const keys = Array.isArray(campaign.keys) ? campaign.keys : []
  if (!keys.length) return
  const [runs, evaluations, queue] = await Promise.all([readRuns(), readEvaluations(), readQueue()])
  const queuedKeys = new Set(
    queue
      .filter((q) => q.activityType === 'evaluate')
      .map((q) => (q.params && q.params.runKey) || ''),
  )
  for (const key of keys) {
    const run = runs.find((r) => r.key === key)
    if (!run) continue
    const s = run.summary
    if (s.status && s.status !== 'completed') continue
    if (!(s.artifacts && s.artifacts.checkpoint)) continue
    if (evaluations.has(key) || queuedKeys.has(key) || evaluatingKeys.has(key)) continue
    await putQueueItem({
      id: randomHexId(),
      activityType: 'evaluate',
      params: { ...baseParams, runKey: key },
      label: `Evaluate ${shortKey(key)}`,
      queuedAt: nowIso(),
    })
  }
}
// Settle-time bookkeeping shared by the observe loop and project open: stamp
// finished campaigns into their hypotheses, consume auto-eval markers.
async function processSettledCampaignEffects() {
  await stampHypothesisCampaignResults()
  await processAutoEvalMarkers()
  if (activeTabId === 'hypotheses') await renderHypotheses()
}

// --- Runs tab -----------------------------------------------------------------
function runRanAt(summary) {
  return (summary.provenance && summary.provenance.ranAt) || summary.ranAt || ''
}
// Run records stamp ranBy with the compute target ('local' when run here) —
// remote runs get a small muted suffix next to their ran-at time.
function ranBySuffixHtml(summary) {
  const ranBy = String(summary.ranBy || '')
  if (!ranBy || ranBy === 'local') return ''
  return ` <span class="ran-by">on ${escapeHtml(ranBy)}</span>`
}
function sortRunsByObjective(runs) {
  const dir = objectiveDirection()
  return [...runs].sort((a, b) => {
    const va = Number(a.summary.objective)
    const vb = Number(b.summary.objective)
    const fa = Number.isFinite(va)
    const fb = Number.isFinite(vb)
    if (fa && fb) return dir === 'min' ? va - vb : vb - va
    if (fa) return -1
    if (fb) return 1
    return 0
  })
}
function healthBadgeHtml(health) {
  const status = (health && health.status) || 'unknown'
  const cls = status === 'ok' ? 'is-ok' : 'is-bad'
  return `<span class="badge ${cls}">${escapeHtml(status)}</span>`
}
// Hand-rolled SVG sparkline of objective over ranAt order (no chart libs).
function sparklineSvg(runs) {
  const ordered = [...runs].sort((a, b) =>
    String(runRanAt(a.summary)).localeCompare(String(runRanAt(b.summary))),
  )
  const values = ordered.map((r) => Number(r.summary.objective)).filter((v) => Number.isFinite(v))
  if (values.length < 2) return ''
  const W = 600
  const H = 72
  const PAD = 6
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || 1
  const stepX = (W - PAD * 2) / (values.length - 1)
  const pts = values.map((v, i) => {
    const x = PAD + i * stepX
    const y = H - PAD - ((v - min) / span) * (H - PAD * 2)
    return [Math.round(x * 10) / 10, Math.round(y * 10) / 10]
  })
  const poly = pts.map((p) => p.join(',')).join(' ')
  const last = pts[pts.length - 1]
  return (
    `<svg class="sparkline" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="${escapeHtml(objectiveName())} over time">` +
    `<polyline points="${poly}" fill="none" stroke="currentColor" stroke-width="2" vector-effect="non-scaling-stroke"></polyline>` +
    `<circle cx="${last[0]}" cy="${last[1]}" r="3" fill="currentColor"></circle>` +
    `</svg>` +
    `<div class="sparkline-range"><span>min ${escapeHtml(formatObjective(min))}</span><span>${escapeHtml(objectiveName())} per run, oldest → newest</span><span>max ${escapeHtml(formatObjective(max))}</span></div>`
  )
}
function verdictChipHtml(verdict) {
  if (!verdict) return '<span class="judge-none">—</span>'
  if (verdict.rejected) return '<span class="badge is-rejected">rejected</span>'
  const score = Number(verdict.score)
  if (!Number.isFinite(score)) return '<span class="judge-none">—</span>'
  const cls = score >= 70 ? 'is-ok' : score >= 40 ? 'is-warn' : 'is-bad'
  return `<span class="badge score-chip ${cls}">${escapeHtml(String(Math.round(score)))}</span>`
}
// Green when the re-test held up the training result (respecting the
// objective's direction), amber when it came back worse.
function evalChipHtml(run) {
  const evaluation = evaluationsCache.get(run.key)
  const value = evaluation ? Number(evaluation.objective) : NaN
  if (!Number.isFinite(value)) return '<span class="judge-none">—</span>'
  const train = Number(run.summary.objective)
  const heldUp =
    !Number.isFinite(train) || (objectiveDirection() === 'min' ? value <= train : value >= train)
  return `<span class="badge eval-chip ${heldUp ? 'is-ok' : 'is-warn'}">${escapeHtml(formatObjective(value))}</span>`
}
// "asset · timeframe" for the runs-table Data column, from the run's dataset
// descriptor (preferred) or its config — so multi-asset/timeframe runs separate.
function datasetLabel(s) {
  const d = (s && s.dataset) || {}
  const cfg = (s && s.config) || {}
  const bits = [d.asset || cfg.asset, d.timeframe || cfg.timeframe].filter(Boolean)
  return bits.length ? bits.join(' · ') : '—'
}
function runRowHtml(run, cols) {
  const s = run.summary
  const rowClasses = [
    run.key === selectedRunKey ? 'is-selected' : '',
    s.status === 'failed'
      ? 'is-failed-row'
      : s.health && s.health.status && s.health.status !== 'ok'
        ? 'is-degenerate-row'
        : '',
  ]
    .filter(Boolean)
    .join(' ')
  const classAttr = rowClasses ? ` class="${rowClasses}"` : ''
  const tds = cols.map((c) => `<td class="${c.num ? 'num' : ''}">${c.get(run)}</td>`).join('')
  return `<tr data-key="${escapeHtml(run.key)}"${classAttr}>${tds}</tr>`
}
function bestObjectiveOf(runs) {
  const values = runs.map((r) => Number(r.summary.objective)).filter((v) => Number.isFinite(v))
  if (!values.length) return NaN
  return objectiveDirection() === 'min' ? Math.min(...values) : Math.max(...values)
}
// Small live badge in the Runs header while a campaign is training.
function renderRunsLive() {
  const el = byId('runs-live')
  if (!el) return
  const live = lastActivityStatus === 'running'
  setHtml(el, live ? `<span class="run-badge is-running">${spinnerHtml()} training…</span>` : '')
  el.hidden = !live
}
function clearRunsFilter() {
  runsFilterKeys = null
  runsFilterLabel = ''
  runsLeverFilter = {}
  runsTextFilter = ''
  runsDrillSetupKey = null
  renderRunsTable()
}
// Drill from a by-setup row into that setup's individual runs (filter + switch view).
function drillIntoSetup(key) {
  const group = aggregateBySetup(applyRunsFilters(runsCache)).find((g) => g.key === key)
  if (!group) return
  runsFilterKeys = new Set(group.runs.map((r) => r.key))
  runsFilterLabel = group.label
  runsDrillSetupKey = key
  runsViewMode = 'runs'
  renderRunsTable()
}
// Drill from a by-experiment row into that thesis's individual runs.
function drillIntoExperiment(thesis) {
  const group = aggregateByExperiment(applyRunsFilters(runsCache)).find((g) => g.thesis === thesis)
  if (!group) return
  runsFilterKeys = new Set(group.runs.map((r) => r.key))
  runsFilterLabel = thesis
  runsDrillSetupKey = null
  runsViewMode = 'runs'
  renderRunsTable()
}
// Viewing the Runs tab clears the project's unseen badge: mark every current run
// key as seen, persisting the union when any are new. No-op when nothing changed,
// so the poll-driven renderRuns does not write every tick.
async function markRunsSeen() {
  if (!embedded() || !currentProject || !manifest) return
  const projectKey = currentProject.key
  const seen = seenKeysByProject.get(projectKey) || new Set()
  const unseen = runsCache.map((r) => r.key).filter((k) => k && !seen.has(k))
  if (!unseen.length) return
  const next = new Set(seen)
  for (const k of unseen) next.add(k)
  seenKeysByProject.set(projectKey, next)
  try {
    await putSeenKeys(projectKey, next)
  } catch {
    // best-effort: the badge resyncs from the persisted record on next read
  }
}
// --- Results workbench: dynamic metric columns + sort + filter -----------------
// Common metric keys first (the rest follow alphabetically) so the table reads
// well for trading (sharpe/return/drawdown…) and the dip line (f1/precision…).
const RUN_METRIC_ORDER = [
  'sharpe',
  'total_return_pct',
  'cagr_pct',
  'max_drawdown_pct',
  'win_pct',
  'n_trades',
  'stop_losses',
  'final_net_worth',
  'f1',
  'precision',
  'recall',
  'accuracy',
  'negative_recall',
  'positive_rate',
  'simple_ratio',
]
// Plain-language help per metric (shown as a "?" on the column header) so a newcomer
// knows what each means + which direction is good.
const METRIC_INFO = {
  sharpe:
    'Risk-adjusted return (return ÷ volatility, annualised). Higher is better; above ~1 is strong.',
  total_return_pct: 'Total % return over the test window. Higher is better.',
  cagr_pct: 'Annualised growth rate implied by the run. Higher is better.',
  max_drawdown_pct: 'Worst peak-to-trough fall (negative %). Closer to 0 is better.',
  win_pct: 'Share of trades that were profitable.',
  n_trades: 'Number of trades taken — very high can mean churn (fees eat returns).',
  stop_losses: 'How many trades were closed by the stop-loss.',
  final_net_worth: 'Ending portfolio value.',
  f1: 'Dip classifier: balance of precision & recall (0–1, higher better).',
  precision: 'Of predicted dips, how many were real (higher better).',
  recall: 'Of real dips, how many were caught (higher better).',
  accuracy: 'Balanced accuracy across the two classes.',
  negative_recall: 'Of non-dips, how many were correctly skipped.',
  positive_rate: 'Share of samples that are positive — the class balance.',
  simple_ratio: 'Correct vs incorrect positive predictions.',
}
const VSHOLD_INFO =
  'Run return minus buy-and-hold over the same window. Positive = beat just holding. Hold is a control to judge against — never the optimisation target.'
function runMetricKeys() {
  const keys = new Set()
  for (const r of runsCache) {
    const m = r.summary && r.summary.metrics
    if (m && typeof m === 'object') for (const k of Object.keys(m)) keys.add(k)
  }
  const known = RUN_METRIC_ORDER.filter((k) => keys.has(k))
  const rest = [...keys].filter((k) => !RUN_METRIC_ORDER.includes(k)).sort()
  return [...known, ...rest]
}
function anyBenchmark() {
  return runsCache.some(
    (r) =>
      r.summary &&
      r.summary.benchmark &&
      Number.isFinite(Number(r.summary.benchmark.hold_return_pct)),
  )
}
// Run return % minus the buy-and-hold control over the same window (display only).
function vsHoldValue(s) {
  const ret = Number(s && s.metrics && s.metrics.total_return_pct)
  const hold = Number(s && s.benchmark && s.benchmark.hold_return_pct)
  return Number.isFinite(ret) && Number.isFinite(hold) ? ret - hold : NaN
}
// The column model the table renders + sorts from; metric columns are derived.
function runsColumns() {
  const cols = [
    {
      id: 'compare',
      label: 'pick',
      num: false,
      noSort: true,
      help: 'Tick 2+ runs to compare them below — config diff, side-by-side metrics, and overlaid return curves vs buy-and-hold.',
      get: (r) =>
        `<input type="checkbox" class="run-compare-cb" data-key="${escapeHtml(r.key)}"${runsCompareKeys.has(r.key) ? ' checked' : ''} aria-label="select to compare" />`,
    },
    {
      id: 'key',
      label: 'Run',
      num: false,
      get: (r) => `<code>${escapeHtml(shortKey(r.key))}</code>`,
      sort: (r) => r.key,
    },
    {
      id: 'data',
      label: 'Data',
      num: false,
      get: (r) => escapeHtml(datasetLabel(r.summary)),
      sort: (r) => datasetLabel(r.summary),
    },
  ]
  for (const mk of runMetricKeys()) {
    cols.push({
      id: 'm:' + mk,
      label: mk.replace(/_pct$/, ' %').replace(/_/g, ' '),
      num: true,
      help: METRIC_INFO[mk],
      get: (r) => {
        const v = r.summary.metrics && r.summary.metrics[mk]
        return escapeHtml(
          typeof v === 'number' ? formatObjective(v) : v === undefined ? '—' : String(v),
        )
      },
      sort: (r) => {
        const v = r.summary.metrics && r.summary.metrics[mk]
        return typeof v === 'number' ? v : NaN
      },
    })
  }
  if (anyBenchmark()) {
    cols.push({
      id: 'vshold',
      label: 'vs hold',
      num: true,
      help: VSHOLD_INFO,
      get: (r) => {
        const d = vsHoldValue(r.summary)
        return Number.isFinite(d)
          ? `<span class="${d >= 0 ? 'delta-pos' : 'delta-neg'}">${d >= 0 ? '+' : ''}${d.toFixed(1)}</span>`
          : '—'
      },
      sort: (r) => vsHoldValue(r.summary),
    })
  }
  cols.push(
    {
      id: 'seed',
      label: 'Seed',
      num: true,
      get: (r) => escapeHtml(r.summary.seed === undefined ? '—' : String(r.summary.seed)),
      sort: (r) => Number(r.summary.seed),
    },
    {
      id: 'health',
      label: 'Status',
      num: false,
      get: (r) =>
        r.summary.status === 'failed'
          ? '<span class="badge is-bad">failed</span>'
          : healthBadgeHtml(r.summary.health),
      sort: (r) =>
        r.summary.status === 'failed'
          ? 'failed'
          : (r.summary.health && r.summary.health.status) || '',
    },
    {
      id: 'judge',
      label: 'Judge',
      num: true,
      get: (r) => verdictChipHtml(verdictsCache.get(r.key)),
      sort: (r) => {
        const v = verdictsCache.get(r.key)
        return v ? Number(v.score) : NaN
      },
    },
    {
      id: 'eval',
      label: 'Eval',
      num: true,
      get: (r) => evalChipHtml(r),
      sort: (r) => {
        const e = evaluationsCache.get(r.key)
        return e ? Number(e.objective) : NaN
      },
    },
    {
      id: 'took',
      label: 'Took',
      num: true,
      get: (r) => escapeHtml(formatDuration(r.summary.durationMs)),
      sort: (r) => Number(r.summary.durationMs),
    },
    {
      id: 'ran',
      label: 'Ran at',
      num: false,
      get: (r) => `${escapeHtml(formatWhen(runRanAt(r.summary)))}${ranBySuffixHtml(r.summary)}`,
      sort: (r) => String(runRanAt(r.summary)),
    },
  )
  return cols
}
function compareNumeric(a, b, dir) {
  const fa = Number.isFinite(a)
  const fb = Number.isFinite(b)
  if (fa && fb) return (a - b) * dir
  if (fa) return -1
  if (fb) return 1
  return 0
}
function sortRuns(runs) {
  const col = runsSortKey ? runsColumns().find((c) => c.id === runsSortKey) : null
  if (!col) return sortRunsByObjective(runs)
  const dir = runsSortDir === 'asc' ? 1 : -1
  return [...runs].sort((a, b) => {
    const va = col.sort(a)
    const vb = col.sort(b)
    if (typeof va === 'number' || typeof vb === 'number')
      return compareNumeric(Number(va), Number(vb), dir)
    return String(va).localeCompare(String(vb)) * dir
  })
}
function applyRunsFilters(runs) {
  let out = runsFilterKeys ? runs.filter((r) => runsFilterKeys.has(r.key)) : runs
  for (const [lever, val] of Object.entries(runsLeverFilter)) {
    if (val) out = out.filter((r) => String((r.summary.config || {})[lever]) === String(val))
  }
  const q = runsTextFilter.trim().toLowerCase()
  if (q) {
    out = out.filter(
      (r) =>
        r.key.toLowerCase().includes(q) ||
        JSON.stringify(r.summary.config || {})
          .toLowerCase()
          .includes(q),
    )
  }
  return out
}
// --- Per-setup aggregation (a SETUP = config minus seed; a result is a setup, not a
// single run). Group runs by setup and report min/avg/max across its seeds. ---------
const SETUP_LABEL_PRIORITY = [
  'model_name',
  'net_arch',
  'reward_model',
  'timeframe',
  'asset',
  'loss_fn',
]
function setupKeyOfRun(run) {
  if (run.summary && run.summary.setupKey) return run.summary.setupKey
  const cfg = { ...((run.summary && run.summary.config) || {}) }
  delete cfg.seed
  return JSON.stringify(
    Object.keys(cfg)
      .sort()
      .map((k) => [k, cfg[k]]),
  )
}
function setupConfigLabel(config) {
  const cfg = config || {}
  const all = Object.keys((manifest && manifest.levers) || {}).filter(
    (k) => k !== 'seed' && cfg[k] !== undefined,
  )
  const ordered = [
    ...SETUP_LABEL_PRIORITY.filter((k) => all.includes(k)),
    ...all.filter((k) => !SETUP_LABEL_PRIORITY.includes(k)),
  ]
  return ordered.length
    ? ordered
        .slice(0, 5)
        .map((k) => `${k}=${cfg[k]}`)
        .join(' · ')
    : '—'
}
function median(xs) {
  if (!xs.length) return NaN
  const s = [...xs].sort((a, b) => a - b)
  const m = Math.floor(s.length / 2)
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}
function mean(xs) {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN
}
// Linear-interpolated quantile (q in [0,1]) over an unsorted array — for IQR spread.
function quantile(xs, q) {
  if (!xs.length) return NaN
  const s = [...xs].sort((a, b) => a - b)
  const pos = (s.length - 1) * q
  const lo = Math.floor(pos)
  const hi = Math.ceil(pos)
  return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (pos - lo)
}
function aggregateBySetup(runs) {
  const groups = new Map()
  for (const r of runs) {
    const k = setupKeyOfRun(r)
    if (!groups.has(k)) groups.set(k, [])
    groups.get(k).push(r)
  }
  const dir = objectiveDirection()
  const out = []
  for (const [key, rs] of groups) {
    const objs = rs.map((r) => Number(r.summary.objective)).filter(Number.isFinite)
    const vsh = rs.map((r) => vsHoldValue(r.summary)).filter(Number.isFinite)
    let bestRun = null
    for (const r of rs) {
      const o = Number(r.summary.objective)
      if (!Number.isFinite(o)) continue
      const bo = bestRun ? Number(bestRun.summary.objective) : NaN
      if (!bestRun || (dir === 'min' ? o < bo : o > bo)) bestRun = r
    }
    // RB3 — seed robustness: how tightly the objective clusters across seeds (IQR), how often it
    // lands on the good side of zero (fraction-positive), and whether it flips sign at all (a
    // seed-fragile setup you should not trust on a single lucky run).
    const positive = objs.filter((o) => (dir === 'min' ? o < 0 : o > 0)).length
    out.push({
      key,
      runs: rs,
      bestRun,
      label: setupConfigLabel(rs[0].summary.config),
      count: rs.length,
      objMin: objs.length ? Math.min(...objs) : NaN,
      objMax: objs.length ? Math.max(...objs) : NaN,
      objAvg: mean(objs),
      objMedian: median(objs),
      objIqr: objs.length ? quantile(objs, 0.75) - quantile(objs, 0.25) : NaN,
      fractionPositive: objs.length ? positive / objs.length : NaN,
      positiveCount: positive,
      objCount: objs.length,
      unstable: objs.length >= 2 && Math.min(...objs) < 0 && Math.max(...objs) > 0,
      vsHoldAvg: mean(vsh),
      failed: rs.filter((r) => r.summary.status === 'failed').length,
    })
  }
  return out
}
// The LLM's one-line verdict for a setup's best run (if a judge has scored it) — the
// machine half of the ledger's "current conclusion".
function setupLlmNote(group) {
  const v = group.bestRun && verdictsCache.get(group.bestRun.key)
  return (v && (v.why || v.summary)) || ''
}
// Group runs by the THESIS they tested (set at launch) so experiments compare
// head-to-head — including theses outside the levers (data prep, code changes).
function aggregateByExperiment(runs) {
  const groups = new Map()
  for (const r of runs) {
    const t = (r.summary && r.summary.thesis) || '(untagged)'
    if (!groups.has(t)) groups.set(t, [])
    groups.get(t).push(r)
  }
  const out = []
  for (const [thesis, rs] of groups) {
    const objs = rs.map((r) => Number(r.summary.objective)).filter(Number.isFinite)
    const targets = [...new Set(rs.map((r) => r.summary && r.summary.thesisTarget).filter(Boolean))]
    out.push({
      thesis,
      target: targets.join(', '),
      runs: rs,
      count: rs.length,
      setups: new Set(rs.map(setupKeyOfRun)).size,
      objMin: objs.length ? Math.min(...objs) : NaN,
      objMax: objs.length ? Math.max(...objs) : NaN,
      objAvg: mean(objs),
    })
  }
  return out
}
function runsToolbarHtml(shownCount, total) {
  const dropdowns = leverEntries()
    .filter(([, spec]) => spec.type === 'choice')
    .map(([key, spec]) => {
      const opts = [`<option value="">${escapeHtml(key)}: any</option>`]
        .concat(
          (spec.choices || []).map(
            (c) =>
              `<option value="${escapeHtml(String(c))}"${String(runsLeverFilter[key] || '') === String(c) ? ' selected' : ''}>${escapeHtml(String(c))}</option>`,
          ),
        )
        .join('')
      return `<select class="runs-filter-lever" data-lever="${escapeHtml(key)}">${opts}</select>`
    })
    .join('')
  const active = runsFilterKeys || runsTextFilter || Object.values(runsLeverFilter).some(Boolean)
  const label = runsFilterLabel ? ` (${escapeHtml(runsFilterLabel)})` : ''
  const toggle = `<div class="runs-viewmode">
    <button type="button" class="runs-view-btn${runsViewMode === 'runs' ? ' is-active' : ''}" data-view="runs">Runs</button><button type="button" class="runs-view-btn${runsViewMode === 'setup' ? ' is-active' : ''}" data-view="setup">By setup ${helpCalloutHtml('Group runs by SETUP (config ignoring seed) and show the spread across seeds — what a setup concluded, not one lucky run.')}</button><button type="button" class="runs-view-btn${runsViewMode === 'experiment' ? ' is-active' : ''}" data-view="experiment">By experiment ${helpCalloutHtml('Group runs by the THESIS set at launch, so experiments compare head-to-head (incl. theses outside the levers).')}</button>
  </div>`
  return `<div class="runs-toolbar">
    ${toggle}
    ${dropdowns}
    <input type="search" id="runs-filter-text" class="runs-filter-text" placeholder="filter config / key…" value="${escapeHtml(runsTextFilter)}" />
    <span class="runs-count">${shownCount}/${total} runs${label}</span>
    ${active ? '<button type="button" id="runs-filter-clear" class="ghost-btn">clear</button>' : ''}
  </div>`
}
function toggleRunsSort(id) {
  if (runsSortKey === id) runsSortDir = runsSortDir === 'asc' ? 'desc' : 'asc'
  else {
    runsSortKey = id
    runsSortDir = 'desc'
  }
  renderRunsTable()
}
// One row per SETUP (config minus seed): seed count + min/avg/max/median objective +
// avg vs-hold. Click a row to drill into that setup's individual runs.
function bySetupTableHtml(filtered) {
  const dir = objectiveDirection()
  const groups = aggregateBySetup(filtered).sort((a, b) => {
    const fa = Number.isFinite(a.objAvg)
    const fb = Number.isFinite(b.objAvg)
    if (fa && fb) return dir === 'min' ? a.objAvg - b.objAvg : b.objAvg - a.objAvg
    return fa ? -1 : fb ? 1 : 0
  })
  const on = escapeHtml(objectiveName())
  const rows = groups
    .map((g) => {
      const range = Number.isFinite(g.objMin)
        ? `${escapeHtml(formatObjective(g.objMin))} – ${escapeHtml(formatObjective(g.objMax))}`
        : '—'
      const vsh = Number.isFinite(g.vsHoldAvg)
        ? `<span class="${g.vsHoldAvg >= 0 ? 'delta-pos' : 'delta-neg'}">${g.vsHoldAvg >= 0 ? '+' : ''}${g.vsHoldAvg.toFixed(1)}</span>`
        : '—'
      const failed = g.failed ? ` <span class="card-sub">(${g.failed} failed)</span>` : ''
      const llm = setupLlmNote(g)
      const note = (notesCache.get(g.key) || {}).note || ''
      return `<tr data-setup-key="${escapeHtml(g.key)}" class="setup-row${g.unstable ? ' is-unstable' : ''}">
        <td>${escapeHtml(g.label)}</td>
        <td class="num">${g.count}${failed}</td>
        <td class="num">${setupStabilityCell(g)}</td>
        <td class="num">${escapeHtml(formatObjective(g.objAvg))}</td>
        <td class="num">${range}</td>
        <td class="num">${escapeHtml(formatObjective(g.objMedian))}</td>
        <td class="num">${vsh}</td>
        <td class="ledger-note" title="${escapeHtml(llm)}">${llm ? escapeHtml(truncate(llm, 70)) : '<span class="card-sub">—</span>'}</td>
        <td class="ledger-note ${note ? '' : 'is-empty'}" title="${escapeHtml(note)}">${note ? escapeHtml(truncate(note, 70)) : '<span class="card-sub">add note ✎</span>'}</td>
      </tr>`
    })
    .join('')
  return `<div class="table-wrap"><table class="runs-table">
    <thead><tr><th>Setup</th><th class="num">seeds</th><th class="num">stability ${helpCalloutHtml('Seed robustness: how many seeds land on the good side of 0 (↑), and ⚠ if the objective flips sign across seeds — a fragile setup you should not trust on one lucky run. Hover a cell for the IQR (spread). Needs ≥2 seeds.')}</th><th class="num">${on} avg</th><th class="num">${on} range</th><th class="num">${on} median</th><th class="num">vs hold avg</th><th>LLM verdict ${helpCalloutHtml("The judge's one-line verdict for this setup's best run (if scored).")}</th><th>Your conclusion ${helpCalloutHtml('Your note for this setup — open the setup to edit. The ledger of everything tried.')}</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`
}
// RB3 cell: seed-robustness at a glance — "k/n ↑" positive seeds, ⚠ when sign-unstable, IQR on
// hover. Needs ≥2 seeds to mean anything; single-seed setups show a muted note instead.
function setupStabilityCell(g) {
  if (g.objCount < 2) return '<span class="card-sub">1 seed</span>'
  const iqr = Number.isFinite(g.objIqr) ? formatObjective(g.objIqr) : '—'
  const warn = g.unstable ? '<span class="seed-unstable">⚠</span> ' : ''
  const tip = `${g.positiveCount} of ${g.objCount} seeds on the good side of 0 · IQR (spread) ${iqr}${g.unstable ? ' · objective flips sign across seeds — fragile' : ''}`
  return `<span title="${escapeHtml(tip)}">${warn}${g.positiveCount}/${g.objCount} ↑ <span class="card-sub">· IQR ${escapeHtml(iqr)}</span></span>`
}
// One row per EXPERIMENT/thesis: how many runs + setups it spans + its objective
// spread, so theses compare head-to-head. Click a row to drill into its runs.
function byExperimentTableHtml(filtered) {
  const dir = objectiveDirection()
  const groups = aggregateByExperiment(filtered).sort((a, b) => {
    const fa = Number.isFinite(a.objMax)
    const fb = Number.isFinite(b.objMax)
    if (fa && fb) return dir === 'min' ? a.objMin - b.objMin : b.objMax - a.objMax
    return fa ? -1 : fb ? 1 : 0
  })
  const on = escapeHtml(objectiveName())
  const rows = groups
    .map((g) => {
      const range = Number.isFinite(g.objMin)
        ? `${escapeHtml(formatObjective(g.objMin))} – ${escapeHtml(formatObjective(g.objMax))}`
        : '—'
      return `<tr data-experiment-key="${escapeHtml(g.thesis)}" class="setup-row">
        <td>${escapeHtml(g.thesis)}</td>
        <td>${escapeHtml(g.target || '—')}</td>
        <td class="num">${g.count}</td>
        <td class="num">${g.setups}</td>
        <td class="num">${escapeHtml(formatObjective(g.objAvg))}</td>
        <td class="num">${range}</td>
      </tr>`
    })
    .join('')
  return `<div class="table-wrap"><table class="runs-table">
    <thead><tr><th>Experiment / thesis</th><th>Target</th><th class="num">runs</th><th class="num">setups</th><th class="num">${on} avg</th><th class="num">${on} best–worst</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`
}
// When drilled into a single setup, an editor for that setup's conclusion note —
// the user half of the ledger (LLM verdict + score being the other halves).
function setupNoteEditorHtml() {
  if (!runsDrillSetupKey) return ''
  const note = (notesCache.get(runsDrillSetupKey) || {}).note || ''
  return `<div class="setup-note-editor">
    <label class="setup-note-label">Your conclusion for this setup ${helpCalloutHtml('What did this setup teach you? Saved against the setup (not one run) so it survives re-runs — your ledger of everything tried.')}</label>
    <textarea id="setup-note-text" class="setup-note-text" rows="2" placeholder="e.g. learns but overfits past episode 20; vs-hold only on 1h data…">${escapeHtml(note)}</textarea>
    <div class="setup-note-actions"><button type="button" id="setup-note-save" class="ghost-btn">Save conclusion</button><span id="setup-note-status" class="card-sub"></span></div>
  </div>`
}
// Render the table from the in-memory caches (no refetch) — used by sort/filter/view.
function renderRunsTable() {
  const body = byId('runs-body')
  const spark = byId('runs-sparkline')
  if (!body) return
  if (!runsCache.length) {
    if (spark) spark.hidden = true
    setHtml(body, '<div class="empty-hint">No runs yet — launch a campaign.</div>')
    closeRunDetail()
    return
  }
  const filtered = applyRunsFilters(runsCache)
  if (spark) {
    const svg = sparklineSvg(filtered)
    setHtml(spark, svg)
    spark.hidden = !svg
  }
  if (!filtered.length) {
    setHtml(
      body,
      `${runsToolbarHtml(0, runsCache.length)}<div class="empty-hint">No runs match the filter.</div>`,
    )
    return
  }
  const toolbar = runsToolbarHtml(filtered.length, runsCache.length)
  if (runsViewMode === 'setup') {
    const legend = `<p class="runs-legend">Each row is a SETUP (config ignoring seed) — the ledger of everything tried. Click one to drill into its runs <em>and write your conclusion</em> · <span class="delta-pos">green</span>/<span class="delta-neg">red</span> = beat / lagged buy-and-hold (avg) · "LLM verdict" needs the judge to have run · "—" = not yet scored / noted.</p>`
    setHtml(body, `${toolbar}${bySetupTableHtml(filtered)}${legend}`)
    renderCompare()
    return
  }
  if (runsViewMode === 'experiment') {
    const groups = aggregateByExperiment(filtered)
    const onlyUntagged = groups.length <= 1 && (!groups[0] || groups[0].thesis === '(untagged)')
    if (onlyUntagged) {
      setHtml(
        body,
        `${toolbar}<div class="empty-hint">No experiments tagged yet. Set a <strong>Thesis</strong> when you launch a campaign (e.g. "fee-penalty reward" or "1m data prep") — its runs group here so you can compare theses head-to-head, even ones that don't map to a lever.</div>`,
      )
      renderCompare()
      return
    }
    const legend = `<p class="runs-legend">Each row is an EXPERIMENT — the thesis set at launch. Click one to drill into its runs. Untagged runs group under "(untagged)".</p>`
    setHtml(body, `${toolbar}${byExperimentTableHtml(filtered)}${legend}`)
    renderCompare()
    return
  }
  const shown = sortRuns(filtered)
  const cols = runsColumns()
  const header = cols
    .map((c) => {
      const help = c.help ? helpCalloutHtml(c.help) : ''
      if (c.noSort) return `<th class="${c.num ? 'num' : ''}">${escapeHtml(c.label)}${help}</th>`
      const arrow = runsSortKey === c.id ? (runsSortDir === 'asc' ? ' ▲' : ' ▼') : ''
      return `<th class="runs-th${c.num ? ' num' : ''}" data-sort="${c.id}">${escapeHtml(c.label)}${help}${arrow}</th>`
    })
    .join('')
  const rows = shown.map((r) => runRowHtml(r, cols)).join('')
  const legend = `<p class="runs-legend">Click a header to sort · hover <span class="help-legend">?</span> for what a column means · <span class="delta-pos">green</span>/<span class="delta-neg">red</span> = beat / lagged buy-and-hold · greyed = failed/degenerate · "—" = not recorded (re-run to populate).</p>`
  setHtml(
    body,
    `${toolbar}${setupNoteEditorHtml()}<div class="table-wrap"><table class="runs-table">
    <thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table></div>${legend}`,
  )
  if (selectedRunKey && !shown.some((r) => r.key === selectedRunKey)) closeRunDetail()
  else if (selectedRunKey) renderRunDetail(selectedRunKey)
  renderCompare()
}
async function renderRuns() {
  if (!byId('runs-body')) return
  ;[runsCache, verdictsCache, judgementSummary, evaluationsCache, notesCache] = await Promise.all([
    readRuns(),
    readVerdicts(),
    readJudgement(),
    readEvaluations(),
    readNotes(),
  ])
  renderJudgeControls()
  renderRunsLive()
  await markRunsSeen()
  renderRunsTable()
}
// Multi-select comparison: a config diff (only differing levers), metrics
// side-by-side, and overlaid %-return curves (+ the buy-and-hold control) for the
// runs ticked in the table. Hidden until ≥2 are selected; pruned of stale keys.
function renderCompare() {
  const card = byId('run-compare')
  if (!card) return
  runsCompareKeys = new Set([...runsCompareKeys].filter((k) => runsCache.some((r) => r.key === k)))
  const runs = [...runsCompareKeys].map((k) => runsCache.find((r) => r.key === k)).filter(Boolean)
  if (runs.length < 2) {
    setHtml(card, '')
    card.hidden = true
    return
  }
  const headRow = runs.map((r) => `<th><code>${escapeHtml(shortKey(r.key))}</code></th>`).join('')
  const diffRows = Object.keys((manifest && manifest.levers) || {})
    .map((lk) => {
      const vals = runs.map((r) => (r.summary.config || {})[lk])
      if (new Set(vals.map((v) => String(v))).size <= 1) return ''
      const cells = vals
        .map((v) => `<td>${escapeHtml(v === undefined ? '—' : String(v))}</td>`)
        .join('')
      return `<tr><th>${escapeHtml(lk)}</th>${cells}</tr>`
    })
    .filter(Boolean)
    .join('')
  const metricRows = runMetricKeys()
    .map((mk) => {
      const cells = runs
        .map((r) => {
          const v = r.summary.metrics && r.summary.metrics[mk]
          return `<td class="num">${escapeHtml(typeof v === 'number' ? formatObjective(v) : v === undefined ? '—' : String(v))}</td>`
        })
        .join('')
      return `<tr><th>${escapeHtml(mk.replace(/_pct$/, ' %').replace(/_/g, ' '))}</th>${cells}</tr>`
    })
    .join('')
  setHtml(
    card,
    `<div class="card-head card-head-row">
      <h3>Compare ${runs.length} runs</h3>
      <button type="button" id="compare-clear" class="ghost-btn">clear</button>
    </div>
    ${compareEquityChartHtml(runs)}
    <h3>Config diff</h3>
    ${diffRows ? `<table class="kv-table compare-table"><thead><tr><th></th>${headRow}</tr></thead><tbody>${diffRows}</tbody></table>` : '<p class="card-sub">Selected runs share the same config.</p>'}
    <h3>Metrics</h3>
    <table class="kv-table compare-table"><thead><tr><th></th>${headRow}</tr></thead><tbody>${metricRows}</tbody></table>`,
  )
  card.hidden = false
}
// Overlay each selected run's equity as a % -return curve, plus the buy-and-hold
// control derived from a run's price series — so the lines are comparable across
// runs with different absolute net worth.
function compareEquityChartHtml(runs) {
  const points = []
  for (const r of runs) {
    const eq = r.summary.series && r.summary.series.equity
    if (!Array.isArray(eq) || eq.length < 2) continue
    const base = Number(eq[0])
    if (!Number.isFinite(base) || base === 0) continue
    eq.forEach((v, i) =>
      points.push({ x: i, y: (Number(v) / base - 1) * 100, group: shortKey(r.key) }),
    )
  }
  const withPrice = runs.find(
    (r) =>
      r.summary.artifacts &&
      r.summary.artifacts.runChart &&
      Array.isArray(r.summary.artifacts.runChart.price),
  )
  if (withPrice) {
    const pr = withPrice.summary.artifacts.runChart.price.map(Number)
    const p0 = pr[0]
    if (Number.isFinite(p0) && p0 !== 0) {
      pr.forEach((v, i) => points.push({ x: i, y: (v / p0 - 1) * 100, group: 'buy & hold' }))
    }
  }
  if (points.length < 2) return ''
  return `<div class="chart-wrap">${buildLineChart({
    points,
    xLabel: 'step',
    yLabel: 'return %',
    width: 680,
    height: 240,
    markers: false,
    ariaLabel: 'compared returns vs buy and hold',
  })}</div>`
}
function metricsTableHtml(metrics) {
  const entries = Object.entries(metrics || {})
  if (!entries.length) return '<p class="card-sub">No metrics recorded.</p>'
  const rows = entries
    .map(
      ([k, v]) =>
        `<tr><th>${escapeHtml(k)}</th><td class="num">${escapeHtml(typeof v === 'number' ? formatObjective(v) : String(v))}</td></tr>`,
    )
    .join('')
  return `<table class="kv-table"><tbody>${rows}</tbody></table>`
}
function verdictSectionHtml(verdict) {
  if (!verdict) return '<h3>Verdict</h3><p class="card-sub">Not judged yet.</p>'
  const judgedLine = `<p class="card-sub">Judged${verdict.judgedBy ? ` by ${escapeHtml(verdict.judgedBy)}` : ''} · ${escapeHtml(formatWhen(verdict.judgedAt))}</p>`
  if (verdict.rejected) {
    return `<h3>Verdict</h3>
      <p class="badges-row">${verdictChipHtml(verdict)}</p>
      <p class="verdict-rejected">${escapeHtml(verdict.why || 'Health-flagged — auto-rejected.')}</p>
      ${judgedLine}`
  }
  const scoreRow = (label, value) =>
    `<tr><th>${escapeHtml(label)}</th><td class="num">${escapeHtml(
      Number.isFinite(Number(value)) ? String(Math.round(Number(value))) : '—',
    )}</td></tr>`
  return `<h3>Verdict</h3>
    <p class="badges-row">${verdictChipHtml(verdict)}</p>
    <table class="kv-table"><tbody>
      ${scoreRow('Blended score', verdict.score)}
      ${scoreRow('Objective score', verdict.objectiveScore)}
      ${scoreRow('LLM score', verdict.llmScore)}
    </tbody></table>
    ${verdict.why ? `<p class="verdict-why">${escapeHtml(verdict.why)}</p>` : ''}
    ${judgedLine}`
}
// The run's live re-test of its saved checkpoint: an Evaluate button while no
// evaluation exists, a spinner while the evaluate activity runs, and the stored
// evaluation record (eval vs train objective, episodes, health) once it has.
function evaluationSectionHtml(run) {
  const s = run.summary
  const statusLine = '<p id="run-eval-status" class="form-status" role="status" hidden></p>'
  if (evaluatingKeys.has(run.key)) {
    return `<h3>Evaluation</h3>
      <p class="card-sub">${spinnerHtml()} Evaluating — re-running the saved checkpoint…</p>
      ${statusLine}`
  }
  const checkpoint = (s.artifacts && s.artifacts.checkpoint) || ''
  const canEvaluate = !!(manifest && manifest.evaluate)
  const evaluation = evaluationsCache.get(run.key)
  const button =
    checkpoint && canEvaluate
      ? `<div class="form-actions"><button type="button" data-action="evaluate" data-key="${escapeHtml(run.key)}">${evaluation ? 'Re-evaluate' : 'Evaluate'}</button></div>`
      : ''
  if (!evaluation) {
    const hint = !canEvaluate
      ? 'This trainer declares no evaluate command — re-testing a saved checkpoint is not available.'
      : checkpoint
        ? 'Not evaluated yet — re-test the saved checkpoint.'
        : 'No saved checkpoint — this run cannot be evaluated.'
    return `<h3>Evaluation</h3><p class="card-sub">${escapeHtml(hint)}</p>${button}${statusLine}`
  }
  const evalObjective = Number(evaluation.objective)
  const train = Number(s.objective)
  const delta =
    Number.isFinite(evalObjective) && Number.isFinite(train) ? evalObjective - train : NaN
  const deltaText = Number.isFinite(delta)
    ? `${delta >= 0 ? '+' : ''}${formatObjective(delta)}`
    : '—'
  const episodes = Number(
    (evaluation.evaluation || {}).episodes ?? (evaluation.metrics || {}).episodes_evaluated,
  )
  const statusBit =
    evaluation.status && evaluation.status !== 'completed' ? ` · ${evaluation.status}` : ''
  return `<h3>Evaluation</h3>
    <p class="badges-row">${healthBadgeHtml(evaluation.health)}</p>
    <table class="kv-table"><tbody>
      <tr><th>Eval ${escapeHtml(objectiveName())}</th><td class="num">${escapeHtml(formatObjective(evalObjective))}</td></tr>
      <tr><th>Train ${escapeHtml(objectiveName())}</th><td class="num">${escapeHtml(formatObjective(train))}</td></tr>
      <tr><th>Delta (eval − train)</th><td class="num">${escapeHtml(deltaText)}</td></tr>
      <tr><th>Episodes</th><td class="num">${escapeHtml(Number.isFinite(episodes) ? String(episodes) : '—')}</td></tr>
    </tbody></table>
    <p class="card-sub">Evaluated ${escapeHtml(formatWhen(evaluation.evaluatedAt))}${escapeHtml(statusBit)}</p>
    ${button}${statusLine}`
}
function trainingCurveSectionHtml(summary) {
  const series = summary.series
  if (!series || typeof series !== 'object') return ''
  // Render the first numeric series the project emitted (episode_return for
  // RL, val_rmse for regression, equity for trading…), labelled by its key.
  const seriesKey = Object.keys(series).find(
    (k) => Array.isArray(series[k]) && series[k].length >= 2,
  )
  if (!seriesKey) return ''
  const label = seriesKey.replace(/_/g, ' ')
  const points = series[seriesKey]
    .map((v, i) => ({ x: i, y: Number(v), label: `${label} · step ${i} · ${formatObjective(v)}` }))
    .filter((p) => Number.isFinite(p.y))
  if (points.length < 2) return ''
  const svg = buildLineChart({
    points,
    xLabel: 'step',
    yLabel: label,
    width: 640,
    height: 180,
    markers: points.length <= 80,
    ariaLabel: `Training curve (${label})`,
  })
  return `<h3>Training curve (${escapeHtml(label)})</h3><div class="chart-wrap">${svg}</div>`
}
// A project-agnostic "what data did this run use" badge, shown when a run emits
// a `dataset` descriptor (asset / timeframe / sample count / date span).
function datasetBadgeHtml(dataset) {
  if (!dataset || typeof dataset !== 'object') return ''
  const bits = []
  if (dataset.asset) bits.push(escapeHtml(String(dataset.asset)))
  if (dataset.timeframe) bits.push(escapeHtml(String(dataset.timeframe)))
  if (Number.isFinite(Number(dataset.candles))) bits.push(`${Number(dataset.candles)} candles`)
  if (!bits.length) return ''
  const span =
    dataset.from && dataset.to
      ? ` · ${escapeHtml(String(dataset.from).slice(0, 10))} → ${escapeHtml(String(dataset.to).slice(0, 10))}`
      : ''
  return `<span class="run-dataset" title="data this run trained on">${bits.join(' · ')}${span}</span>`
}
// Price line + trade markers (buy/sell/TP/SL), re-surfacing the repo's old
// matplotlib action-on-price plot as serialised data drawn on our SVG engine.
// Markers carry their own index into the downsampled price array, so none are lost.
function buildPriceActionChart(chart, opts) {
  const price = (chart.price || []).map(Number).filter(Number.isFinite)
  if (price.length < 2) return ''
  const markers = Array.isArray(chart.markers) ? chart.markers : []
  const W = (opts && opts.width) || 640
  const H = (opts && opts.height) || 200
  const pad = { top: 14, right: 16, bottom: 30, left: 60 }
  const innerW = W - pad.left - pad.right
  const innerH = H - pad.top - pad.bottom
  const ys = price.slice()
  for (const m of markers) if (Number.isFinite(Number(m.price))) ys.push(Number(m.price))
  const xScale = linearScale(price.map((_, i) => i))
  const yScale = linearScale(ys)
  const px = (v) => Math.round((pad.left + xScale.pos(v) * innerW) * 10) / 10
  const py = (v) => Math.round((pad.top + (1 - yScale.pos(v)) * innerH) * 10) / 10
  const parts = []
  for (const t of yScale.ticks) {
    const y = py(t)
    parts.push(
      `<line class="chart-grid" x1="${pad.left}" y1="${y}" x2="${W - pad.right}" y2="${y}"></line>`,
      `<text class="chart-tick" x="${pad.left - 6}" y="${y + 3}" text-anchor="end">${escapeHtml(formatTickValue(t))}</text>`,
    )
  }
  parts.push(
    `<line class="chart-axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${H - pad.bottom}"></line>`,
    `<line class="chart-axis" x1="${pad.left}" y1="${H - pad.bottom}" x2="${W - pad.right}" y2="${H - pad.bottom}"></line>`,
    `<polyline class="chart-line" points="${price.map((v, i) => `${px(i)},${py(v)}`).join(' ')}"></polyline>`,
  )
  for (const m of markers) {
    const i = Number(m.i)
    const p = Number(m.price)
    if (!Number.isFinite(i) || !Number.isFinite(p)) continue
    const x = px(i)
    const y = py(p)
    const up = m.type === 'buy' || m.type === 'tp'
    const d = up
      ? `M ${x} ${y - 5} L ${x - 4} ${y + 3} L ${x + 4} ${y + 3} Z`
      : `M ${x} ${y + 5} L ${x - 4} ${y - 3} L ${x + 4} ${y - 3} Z`
    parts.push(
      `<path class="run-mark run-mark-${escapeHtml(String(m.type))}" d="${d}"><title>${escapeHtml(`${m.type} · ${formatTickValue(p)}`)}</title></path>`,
    )
  }
  if (opts && opts.xLabel) {
    parts.push(
      `<text class="chart-label" x="${pad.left + innerW / 2}" y="${H - 4}" text-anchor="middle">${escapeHtml(opts.xLabel)}</text>`,
    )
  }
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeHtml((opts && opts.ariaLabel) || 'price with trade actions')}">${parts.join('')}</svg>`
}
// The custom per-run result view: a price chart with the run's buy/sell/TP/SL
// markers + a count legend. Gated on `artifacts.runChart`, so non-trading
// projects (cartpole, regression) keep the generic run-detail view.
function priceActionSectionHtml(summary) {
  const chart = summary && summary.artifacts && summary.artifacts.runChart
  if (!chart || !Array.isArray(chart.price) || chart.price.length < 2) return ''
  const markers = Array.isArray(chart.markers) ? chart.markers : []
  const counts = {}
  for (const m of markers) counts[m.type] = (counts[m.type] || 0) + 1
  const legend = ['buy', 'sell', 'tp', 'sl']
    .filter((t) => counts[t])
    .map((t) => `<span class="run-mark-key run-mark-${t}">${escapeHtml(t)} ${counts[t]}</span>`)
    .join(' ')
  const svg = buildPriceActionChart(chart, {
    xLabel: 'step',
    ariaLabel: 'price with trade actions',
    width: 640,
    height: 200,
  })
  return `<h3>Price &amp; actions</h3><div class="chart-wrap">${svg}</div>${legend ? `<p class="badges-row run-mark-legend">${legend}</p>` : ''}`
}
// Model equity vs the buy-and-hold control: what the portfolio would be worth if
// you'd simply bought at step 0 and held. Hold is a comparison CONTROL (not a
// reward target) — derived from the price the run already emits, so no producer
// change. Shows the model curve, the hold curve, and the end-of-window delta.
function equityVsHoldSectionHtml(summary) {
  const series = summary && summary.series
  const chart = summary && summary.artifacts && summary.artifacts.runChart
  const equity = series && Array.isArray(series.equity) ? series.equity.map(Number) : []
  const price = chart && Array.isArray(chart.price) ? chart.price.map(Number) : []
  if (equity.length < 2 || price.length < 2) return ''
  const n = Math.min(equity.length, price.length)
  const start = Number(equity[0])
  const p0 = Number(price[0])
  if (!Number.isFinite(start) || !Number.isFinite(p0) || p0 === 0) return ''
  const points = []
  for (let i = 0; i < n; i++) {
    const hold = start * (Number(price[i]) / p0)
    points.push({ x: i, y: Number(equity[i]), group: 'model' })
    points.push({ x: i, y: hold, group: 'buy & hold' })
  }
  const svg = buildLineChart({
    points,
    xLabel: 'step',
    yLabel: 'net worth',
    width: 640,
    height: 200,
    markers: false,
    ariaLabel: 'model equity vs buy and hold',
  })
  const modelPct = ((Number(equity[n - 1]) - start) / start) * 100
  const holdPct = ((start * (Number(price[n - 1]) / p0) - start) / start) * 100
  const delta = modelPct - holdPct
  const fmt = (v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`
  const cls = delta >= 0 ? 'is-ok' : 'is-warn'
  return `<h3>Equity vs buy &amp; hold <span class="card-sub">— control</span></h3>
    <div class="chart-wrap">${svg}</div>
    <p class="badges-row">model ${escapeHtml(fmt(modelPct))} · hold ${escapeHtml(fmt(holdPct))}
      · <span class="badge ${cls}">${escapeHtml(`${delta >= 0 ? '+' : ''}${delta.toFixed(1)} pts vs hold`)}</span></p>`
}
// Older trading runs (an `equity` series but no `runChart`/`dataset`) predate the
// price-action + hold-comparison view; tell the user a re-run will surface them.
function oldRunChartHintHtml(s) {
  const hasEquity = s && s.series && Array.isArray(s.series.equity) && s.series.equity.length >= 2
  const hasChart = s && s.artifacts && s.artifacts.runChart
  if (hasEquity && !hasChart && !s.dataset) {
    return `<p class="card-sub run-rerun-hint">This run predates the price &amp; action view — re-run the config to see the price chart and the buy-and-hold comparison.</p>`
  }
  return ''
}
// A failed run's diagnostics: the error line + the captured stdout/stderr tail, so a bare "exit
// code 1" is actually explainable. The engine records `error` + `logTail` on any non-completed run.
function failureDetailHtml(s) {
  const err = s.error
    ? `<p class="verdict-rejected"><strong>Failed:</strong> ${escapeHtml(String(s.error))}</p>`
    : '<p class="verdict-rejected">Run failed (no error message recorded).</p>'
  const tail =
    Array.isArray(s.logTail) && s.logTail.length
      ? `<details class="log-tail" open><summary>Last output — ${s.logTail.length} lines (stdout + stderr)</summary><pre class="log-tail-pre">${escapeHtml(s.logTail.join('\n'))}</pre></details>`
      : '<p class="card-sub">No log output was captured for this run.</p>'
  return `<div class="failure-detail">${err}${tail}</div>`
}
function renderRunDetail(key) {
  const panel = byId('run-detail')
  if (!panel) return
  const run = runsCache.find((r) => r.key === key)
  if (!run) {
    closeRunDetail()
    return
  }
  const s = run.summary
  const failed = s.status === 'failed'
  const flags = (s.health && Array.isArray(s.health.flags) && s.health.flags) || []
  const flagChips = flags.length
    ? flags.map((f) => `<span class="badge is-bad">${escapeHtml(f)}</span>`).join(' ')
    : '<span class="card-sub">none</span>'
  const checkpoint = (s.artifacts && s.artifacts.checkpoint) || ''
  const datasetBadge = datasetBadgeHtml(s.dataset)
  const headline = failed
    ? '<span class="badge is-bad">failed</span>'
    : `${escapeHtml(objectiveName())} ${escapeHtml(formatObjective(s.objective))} · ${healthBadgeHtml(s.health)}`
  const html = `
    <div class="card-head card-head-row">
      <div>
        <h2>Run <code>${escapeHtml(shortKey(run.key))}</code></h2>
        <p class="card-sub">${headline} · seed ${escapeHtml(s.seed === undefined ? '—' : String(s.seed))}
          · ${escapeHtml(formatWhen(runRanAt(s)))}${datasetBadge ? ` · ${datasetBadge}` : ''}</p>
      </div>
      <div class="head-actions">
        <button type="button" data-action="clone" data-key="${escapeHtml(run.key)}" class="ghost-btn">Clone to Launch</button>
        <button type="button" id="run-detail-close" class="ghost-btn">Close</button>
      </div>
    </div>
    ${failed ? failureDetailHtml(s) : ''}
    <h3>Health flags</h3>
    <p class="badges-row">${flagChips}</p>
    ${verdictSectionHtml(verdictsCache.get(run.key))}
    ${evaluationSectionHtml(run)}
    <h3>Metrics</h3>
    ${metricsTableHtml(s.metrics)}
    ${oldRunChartHintHtml(s)}
    ${priceActionSectionHtml(s)}
    ${equityVsHoldSectionHtml(s) || trainingCurveSectionHtml(s)}
    <h3>Config</h3>
    <pre class="json">${escapeHtml(JSON.stringify(s.config || {}, null, 2))}</pre>
    <h3>Artifacts</h3>
    <p class="mono">${checkpoint ? escapeHtml(checkpoint) : '—'}</p>
    <p class="card-sub">configHash <code>${escapeHtml(run.key)}</code></p>`
  setHtml(panel, html)
  panel.hidden = false
}
function openRunDetail(key) {
  selectedRunKey = key
  for (const row of document.querySelectorAll('#runs-body tr[data-key]')) {
    row.classList.toggle('is-selected', row.dataset.key === key)
  }
  renderRunDetail(key)
}
function closeRunDetail() {
  selectedRunKey = null
  const panel = byId('run-detail')
  if (panel) {
    panel.hidden = true
    panel.innerHTML = ''
  }
  for (const row of document.querySelectorAll('#runs-body tr.is-selected')) {
    row.classList.remove('is-selected')
  }
}
// Pre-fill the Launch form with a run's exact settings, so it's easy to sweep NEAR
// a known result (tweak one lever, run the neighbours).
function cloneRunToLaunch(key) {
  const run = runsCache.find((r) => r.key === key)
  if (!run) return
  showTab('launch')
  applyPresetFixed(run.summary.config || {})
}
function busyButtonHtml(label) {
  return `${spinnerHtml()} ${escapeHtml(label)}`
}
function setStatusLine(id, text, isError) {
  const el = byId(id)
  if (!el) return
  el.textContent = text || ''
  el.hidden = !text
  el.classList.toggle('is-error', !!isError)
}
function renderJudgeControls() {
  const btn = byId('judge-btn')
  if (btn) {
    btn.disabled = judging
    btn.innerHTML = judging ? busyButtonHtml('Judging…') : 'Judge runs'
  }
  renderTabLiveIndicator()
  const last = byId('judge-last')
  if (!last) return
  if (judgementSummary && judgementSummary.judgedAt) {
    const j = judgementSummary
    const by = j.judgedBy ? ` by ${j.judgedBy}` : ''
    last.textContent = `Last judged ${formatWhen(j.judgedAt)}${by} — ${Number(j.judged) || 0} judged, ${Number(j.rejected) || 0} rejected.`
    last.hidden = false
  } else {
    last.hidden = true
  }
}
// Shared observe loop for the quick one-LLM-call activities (judge / propose):
// poll the activity list until the run leaves 'running', then hand back its
// final entry so the caller can surface a failure's error state.
async function observeQuickActivity(activityId) {
  const start = Date.now()
  let missing = 0
  while (Date.now() - start < MAX_QUICK_OBSERVE_MS) {
    if (!document.hidden) {
      const act = await getActivity(activityId)
      if (act && act.status && act.status !== 'running') return act
      if (!act && ++missing >= 3) return null
    }
    await sleep(POLL_MS)
  }
  return null
}
function quickActivityFailureText(act, what) {
  if (!act) return `${what} did not settle — re-open this tab later to see results.`
  if (act.status === 'completed') return ''
  if (act.error) return `${what} ${act.status}: ${act.error}`
  if (act.status === 'failed') return `${what} failed — check the agent LLM config.`
  return `${what} ${act.status}.`
}
async function onJudgeClick() {
  if (judging) return
  if (!embedded()) {
    setStatusLine('judge-status', 'Open inside the Overseer to judge runs.', false)
    return
  }
  const epoch = projectEpoch
  setStatusLine('judge-status', '')
  try {
    const result = await startOrEnqueue('judge', trainerActivityParams(), 'Judge runs')
    if (result.queued) {
      if (epoch === projectEpoch) setStatusLine('judge-status', queuedStatusText(result.ahead))
      return
    }
    judging = true
    renderJudgeControls()
    const act = await observeQuickActivity(result.activityId)
    if (epoch === projectEpoch) {
      setStatusLine('judge-status', quickActivityFailureText(act, 'Judging'), true)
    }
  } catch {
    if (epoch === projectEpoch) {
      setStatusLine('judge-status', 'Could not start judging — please try again.', true)
    }
  } finally {
    if (judging) {
      judging = false
      renderJudgeControls()
      if (epoch === projectEpoch) await renderRuns()
      pumpQueue()
    }
  }
}
// Re-test one run's saved checkpoint via the quick 'evaluate' activity (no
// LLM): start it, observe until it settles, then re-read the evaluation record
// through renderRuns so the table chip and the detail section both refresh.
async function onEvaluateRun(key) {
  if (evaluatingKeys.has(key) || !embedded()) return
  if (!runsCache.some((r) => r.key === key)) return
  if (
    queueCache.some((q) => q.activityType === 'evaluate' && q.params && q.params.runKey === key)
  ) {
    if (selectedRunKey === key) setStatusLine('run-eval-status', 'Already queued.')
    return
  }
  const epoch = projectEpoch
  let started = false
  let failure = ''
  try {
    const result = await startOrEnqueue(
      'evaluate',
      trainerComputeParams({ runKey: key }),
      `Evaluate ${shortKey(key)}`,
    )
    if (result.queued) {
      if (epoch === projectEpoch && selectedRunKey === key) {
        setStatusLine('run-eval-status', queuedStatusText(result.ahead))
      }
      return
    }
    started = true
    evaluatingKeys.add(key)
    if (selectedRunKey === key) renderRunDetail(key)
    const act = await observeQuickActivity(result.activityId)
    failure = quickActivityFailureText(act, 'Evaluating')
  } catch {
    failure = 'Could not start evaluating — please try again.'
  } finally {
    if (started || failure) {
      evaluatingKeys.delete(key)
      if (epoch === projectEpoch) {
        await renderRuns()
        if (failure && selectedRunKey === key) setStatusLine('run-eval-status', failure, true)
      }
      if (started) pumpQueue()
    }
  }
}
function setupRuns() {
  const body = byId('runs-body')
  if (body) {
    body.addEventListener('click', (event) => {
      if (event.target.closest('#runs-filter-clear')) {
        clearRunsFilter()
        return
      }
      if (event.target.closest('#setup-note-save')) {
        const ta = byId('setup-note-text')
        const status = byId('setup-note-status')
        if (status) status.textContent = 'Saving…'
        saveSetupNote(runsDrillSetupKey, ta ? ta.value : '').then(() => {
          const s = byId('setup-note-status')
          if (s) s.textContent = 'Saved ✓'
        })
        return
      }
      if (event.target.closest('.help-btn')) return // a "?" tooltip, not an action
      const viewBtn = event.target.closest('.runs-view-btn')
      if (viewBtn) {
        runsViewMode = viewBtn.dataset.view
        renderRunsTable()
        return
      }
      const th = event.target.closest('.runs-th[data-sort]')
      if (th) {
        toggleRunsSort(th.dataset.sort)
        return
      }
      if (event.target.closest('.run-compare-cb')) return // checkbox toggles compare, not detail
      const setupRow = event.target.closest('tr[data-setup-key]')
      if (setupRow) {
        drillIntoSetup(setupRow.dataset.setupKey)
        return
      }
      const expRow = event.target.closest('tr[data-experiment-key]')
      if (expRow) {
        drillIntoExperiment(expRow.dataset.experimentKey)
        return
      }
      const row = event.target.closest('tr[data-key]')
      if (row) openRunDetail(row.dataset.key)
    })
    body.addEventListener('change', (event) => {
      const cb = event.target.closest('.run-compare-cb')
      if (cb) {
        if (cb.checked) runsCompareKeys.add(cb.dataset.key)
        else runsCompareKeys.delete(cb.dataset.key)
        renderCompare()
        return
      }
      const sel = event.target.closest('.runs-filter-lever')
      if (sel) {
        runsLeverFilter[sel.dataset.lever] = sel.value
        renderRunsTable()
      }
    })
    body.addEventListener('input', (event) => {
      if (event.target.id !== 'runs-filter-text') return
      runsTextFilter = event.target.value
      const pos = event.target.selectionStart
      renderRunsTable()
      const el = byId('runs-filter-text')
      if (el) {
        el.focus()
        try {
          el.setSelectionRange(pos, pos)
        } catch {
          // some input types disallow setSelectionRange — focus alone is fine
        }
      }
    })
  }
  const panel = byId('run-detail')
  if (panel) {
    panel.addEventListener('click', (event) => {
      if (event.target.closest('#run-detail-close')) closeRunDetail()
      const evalBtn = event.target.closest('button[data-action="evaluate"]')
      if (evalBtn) onEvaluateRun(evalBtn.dataset.key)
      const cloneBtn = event.target.closest('button[data-action="clone"]')
      if (cloneBtn) cloneRunToLaunch(cloneBtn.dataset.key)
    })
  }
  const compareCard = byId('run-compare')
  if (compareCard) {
    compareCard.addEventListener('click', (event) => {
      if (event.target.closest('#compare-clear')) {
        runsCompareKeys = new Set()
        renderRunsTable()
      }
    })
  }
  const judgeBtn = byId('judge-btn')
  if (judgeBtn) {
    judgeBtn.addEventListener('click', onJudgeClick)
    judgeBtn.insertAdjacentHTML('afterend', helpCalloutHtml(JUDGE_HELP_TEXT))
  }
}

// --- SVG chart helpers (hand-rolled, no libs) ----------------------------------
// One shared XY chart builder behind two thin wrappers: buildLineChart and
// buildScatterChart take { points: [{ x, y, label?, marker?, group? }], xLabel,
// yLabel, logX?, xTime?, refDiagonal?, markers?, width?, height?, ariaLabel?,
// groupColors? } and return an SVG string with axes, grid lines, ticks and
// per-point <title> tooltips. Scales come from three tiny tick generators
// below. Points carrying a `group` label are colour-coded from a fixed palette
// (or the caller's groupColors map); grouped line charts draw one polyline per
// group.
function groupColorMap(labels) {
  const distinct = [...new Set(labels.map((l) => String(l)))].sort()
  const map = new Map()
  distinct.forEach((label, i) => map.set(label, CHART_PALETTE[i % CHART_PALETTE.length]))
  return map
}
function chartLegendHtml(groupColors) {
  if (!groupColors || !groupColors.size) return ''
  const items = [...groupColors]
    .map(
      ([label, color]) =>
        `<span class="chart-legend-item"><span class="chart-legend-swatch" style="background:${color}"></span>${escapeHtml(label)}</span>`,
    )
    .join('')
  return `<div class="chart-legend">${items}</div>`
}
function formatTickValue(value) {
  const v = Number(value)
  if (!Number.isFinite(v)) return ''
  const a = Math.abs(v)
  if (a !== 0 && (a < 0.001 || a >= 1e5)) return v.toExponential(1)
  return String(Math.round(v * 1000) / 1000)
}
function timeTickFormat(values) {
  const span = Math.max(...values) - Math.min(...values)
  if (span < 24 * 60 * 60 * 1000) {
    return (t) => new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }
  return (t) => new Date(t).toLocaleDateString()
}
function niceStep(rough) {
  const base = Math.pow(10, Math.floor(Math.log10(rough)))
  const f = rough / base
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * base
}
// Linear scale snapped to "nice" tick steps; always spans at least one step.
function linearScale(values) {
  let min = Math.min(...values)
  let max = Math.max(...values)
  if (min === max) {
    const bump = (Math.abs(min) || 1) / 2
    min -= bump
    max += bump
  }
  const step = niceStep((max - min) / 4)
  const lo = Math.floor(min / step) * step
  const hi = Math.ceil(max / step) * step
  const ticks = []
  for (let v = lo; v <= hi + step / 2; v += step) ticks.push(v)
  return { ticks, pos: (v) => (v - lo) / (hi - lo) }
}
// Log10 scale (positive values only) with a tick per decade.
function logScale(values) {
  const lo = Math.floor(Math.log10(Math.min(...values)))
  const rawHi = Math.ceil(Math.log10(Math.max(...values)))
  const hi = rawHi === lo ? lo + 1 : rawHi
  const ticks = []
  for (let e = lo; e <= hi; e++) ticks.push(Math.pow(10, e))
  return { ticks, pos: (v) => (Math.log10(v) - lo) / (hi - lo) }
}
// Raw-span scale with evenly spaced ticks — used for time axes, where "nice"
// millisecond steps land on meaningless instants.
function spanScale(values, tickCount) {
  let min = Math.min(...values)
  let max = Math.max(...values)
  if (min === max) {
    min -= 1
    max += 1
  }
  const n = Math.max(2, tickCount || 4)
  const ticks = Array.from({ length: n }, (_, i) => min + ((max - min) * i) / (n - 1))
  return { ticks, pos: (v) => (v - min) / (max - min) }
}
function buildXyChart(opts) {
  const points = (opts.points || []).filter(
    (p) => Number.isFinite(p.x) && Number.isFinite(p.y) && (!opts.logX || p.x > 0),
  )
  if (!points.length) return ''
  const W = opts.width || 460
  const H = opts.height || 240
  const pad = { top: 14, right: 16, bottom: 38, left: 58 }
  const innerW = W - pad.left - pad.right
  const innerH = H - pad.top - pad.bottom
  let xValues = points.map((p) => p.x)
  let yValues = points.map((p) => p.y)
  if (opts.refDiagonal) {
    xValues = xValues.concat(yValues)
    yValues = xValues
  }
  const xScale = opts.logX
    ? logScale(xValues)
    : opts.xTime
      ? spanScale(xValues)
      : linearScale(xValues)
  const yScale = linearScale(yValues)
  const px = (v) => Math.round((pad.left + xScale.pos(v) * innerW) * 10) / 10
  const py = (v) => Math.round((pad.top + (1 - yScale.pos(v)) * innerH) * 10) / 10
  const fmtX = opts.xTickFormat || formatTickValue
  const parts = []
  for (const t of yScale.ticks) {
    const y = py(t)
    parts.push(
      `<line class="chart-grid" x1="${pad.left}" y1="${y}" x2="${W - pad.right}" y2="${y}"></line>`,
      `<text class="chart-tick" x="${pad.left - 6}" y="${y + 3}" text-anchor="end">${escapeHtml(formatTickValue(t))}</text>`,
    )
  }
  for (const t of xScale.ticks) {
    const x = px(t)
    parts.push(
      `<line class="chart-grid" x1="${x}" y1="${pad.top}" x2="${x}" y2="${H - pad.bottom}"></line>`,
      `<text class="chart-tick" x="${x}" y="${H - pad.bottom + 14}" text-anchor="middle">${escapeHtml(fmtX(t))}</text>`,
    )
  }
  parts.push(
    `<line class="chart-axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${H - pad.bottom}"></line>`,
    `<line class="chart-axis" x1="${pad.left}" y1="${H - pad.bottom}" x2="${W - pad.right}" y2="${H - pad.bottom}"></line>`,
  )
  if (opts.xLabel) {
    parts.push(
      `<text class="chart-label" x="${pad.left + innerW / 2}" y="${H - 4}" text-anchor="middle">${escapeHtml(opts.xLabel)}</text>`,
    )
  }
  if (opts.yLabel) {
    parts.push(
      `<text class="chart-label" transform="rotate(-90)" x="${-(pad.top + innerH / 2)}" y="12" text-anchor="middle">${escapeHtml(opts.yLabel)}</text>`,
    )
  }
  if (opts.refDiagonal) {
    const lo = xScale.ticks[0]
    const hi = xScale.ticks[xScale.ticks.length - 1]
    parts.push(
      `<line class="chart-ref" x1="${px(lo)}" y1="${py(lo)}" x2="${px(hi)}" y2="${py(hi)}"></line>`,
    )
  }
  const grouped = points.some((p) => p.group !== undefined)
  const groupColors = grouped
    ? opts.groupColors ||
      groupColorMap(points.filter((p) => p.group !== undefined).map((p) => p.group))
    : null
  const pointColor = (p) =>
    groupColors && p.group !== undefined ? groupColors.get(String(p.group)) || '' : ''
  if (opts.line) {
    if (groupColors) {
      for (const [label, color] of groupColors) {
        const path = points
          .filter((p) => p.group !== undefined && String(p.group) === label)
          .map((p) => `${px(p.x)},${py(p.y)}`)
          .join(' ')
        if (path) {
          parts.push(
            `<polyline class="chart-line" style="stroke:${color}" points="${path}"></polyline>`,
          )
        }
      }
    } else {
      const path = points.map((p) => `${px(p.x)},${py(p.y)}`).join(' ')
      parts.push(`<polyline class="chart-line" points="${path}"></polyline>`)
    }
  }
  if (opts.markers !== false) {
    for (const p of points) {
      const x = px(p.x)
      const y = py(p.y)
      const title = p.label ? `<title>${escapeHtml(p.label)}</title>` : ''
      const color = pointColor(p)
      if (p.marker === 'cross') {
        parts.push(
          `<g class="chart-cross"><path${color ? ` style="stroke:${color}"` : ''} d="M ${x - 4} ${y - 4} L ${x + 4} ${y + 4} M ${x - 4} ${y + 4} L ${x + 4} ${y - 4}"></path>${title}</g>`,
        )
      } else {
        parts.push(
          `<circle class="chart-point"${color ? ` style="fill:${color}"` : ''} cx="${x}" cy="${y}" r="3.5">${title}</circle>`,
        )
      }
    }
  }
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeHtml(opts.ariaLabel || '')}">${parts.join('')}</svg>`
}
function buildLineChart(opts) {
  return buildXyChart({ ...opts, line: true })
}
function buildScatterChart(opts) {
  return buildXyChart({ ...opts, line: false })
}

// --- Charts tab ------------------------------------------------------------------
// Each chart carries a per-chart "Split by" selector over the manifest's choice
// levers; selections live in chartSplits while the dashboard is open. Split
// charts colour-group their points by the lever's value in each run's config.
function chartSplitKey(chartId) {
  const lever = chartSplits[chartId]
  const spec = lever && manifest && manifest.levers && manifest.levers[lever]
  return spec && spec.type === 'choice' ? lever : ''
}
function runSplitGroup(run, lever) {
  const config = run.summary.config || {}
  return config[lever] === undefined ? '—' : String(config[lever])
}
function splitSelectHtml(chartId) {
  const choices = leverEntries().filter(([, spec]) => spec.type === 'choice')
  if (!choices.length) return ''
  const selected = chartSplitKey(chartId)
  const options = ['<option value="">None</option>']
    .concat(
      choices.map(
        ([key]) =>
          `<option value="${escapeHtml(key)}"${key === selected ? ' selected' : ''}>${escapeHtml(key)}</option>`,
      ),
    )
    .join('')
  return `<label class="chart-split"><span>Split by</span> <select data-chart-split="${escapeHtml(chartId)}">${options}</select></label>`
}
function chartSectionHtml(title, svg, emptyText, extras) {
  const controls = (extras && extras.controls) || ''
  const legend = (extras && extras.legend) || ''
  const head = `<div class="chart-section-head"><h3>${escapeHtml(title)}</h3>${controls}</div>`
  const content =
    (extras && extras.content) ||
    (svg
      ? `<div class="chart-wrap">${svg}</div>`
      : `<div class="empty-hint">${escapeHtml(emptyText)}</div>`)
  return `<section class="chart-section">${head}${legend}${content}</section>`
}
function chartFigureHtml(title, svg) {
  return `<figure class="chart-figure"><figcaption>${escapeHtml(title)}</figcaption><div class="chart-wrap">${svg}</div></figure>`
}
function spansTwoOrders(values) {
  const positive = values.filter((v) => v > 0)
  if (positive.length < 2) return false
  return Math.max(...positive) / Math.min(...positive) >= 100
}
// Group colours for one split lever, computed across every run so the same
// value keeps the same colour in every chart of the dashboard.
function splitGroupColors(split) {
  return split ? groupColorMap(runsCache.map((r) => runSplitGroup(r, split))) : null
}
function objectiveTimelineSvg(split, groupColors) {
  const points = runsCache
    .map((r) => ({
      x: new Date(runRanAt(r.summary)).getTime(),
      y: Number(r.summary.objective),
      label: `${shortKey(r.key)} · ${objectiveName()} ${formatObjective(r.summary.objective)} · ${formatWhen(runRanAt(r.summary))}`,
      group: split ? runSplitGroup(r, split) : undefined,
    }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
    .sort((a, b) => a.x - b.x)
  if (points.length < 2) return ''
  return buildLineChart({
    points,
    xLabel: 'ran at',
    yLabel: objectiveName(),
    xTime: true,
    xTickFormat: timeTickFormat(points.map((p) => p.x)),
    width: 920,
    height: 280,
    ariaLabel: `${objectiveName()} over time`,
    groupColors,
  })
}
function timelineChartSectionHtml() {
  const split = chartSplitKey('timeline')
  const groupColors = splitGroupColors(split)
  const svg = objectiveTimelineSvg(split, groupColors)
  return chartSectionHtml(`${objectiveName()} over time`, svg, 'Not enough runs yet.', {
    controls: splitSelectHtml('timeline'),
    legend: svg ? chartLegendHtml(groupColors) : '',
  })
}
// Small-multiple scatters: one per numeric lever with ≥2 distinct swept values,
// log-x when the values span at least two orders of magnitude.
function leverScatterFiguresHtml(split, groupColors) {
  const figures = []
  for (const [key, spec] of leverEntries()) {
    if (spec.type !== 'number') continue
    const points = runsCache
      .map((r) => {
        const value = Number(r.summary.config && r.summary.config[key])
        return {
          x: value,
          y: Number(r.summary.objective),
          label: `${shortKey(r.key)} · ${key} ${formatObjective(value)} · ${objectiveName()} ${formatObjective(r.summary.objective)}`,
          group: split ? runSplitGroup(r, split) : undefined,
        }
      })
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
    if (new Set(points.map((p) => p.x)).size < 2) continue
    const svg = buildScatterChart({
      points,
      xLabel: key,
      yLabel: objectiveName(),
      logX: spansTwoOrders(points.map((p) => p.x)),
      width: 320,
      height: 220,
      ariaLabel: `${objectiveName()} vs ${key}`,
      groupColors,
    })
    figures.push(chartFigureHtml(key, svg))
  }
  return figures
}
function leverChartsSectionHtml() {
  const split = chartSplitKey('levers')
  const groupColors = splitGroupColors(split)
  const hasNumericLevers = leverEntries().some(([, spec]) => spec.type === 'number')
  let content
  let legend = ''
  if (!hasNumericLevers) {
    content = '<div class="empty-hint">No numeric levers in this manifest.</div>'
  } else {
    const figures = leverScatterFiguresHtml(split, groupColors)
    if (figures.length) {
      content = `<div class="chart-multiples">${figures.join('')}</div>`
      legend = chartLegendHtml(groupColors)
    } else {
      content =
        '<div class="empty-hint">Not enough runs yet — sweep a lever to see its effect.</div>'
    }
  }
  return chartSectionHtml(`${objectiveName()} vs levers`, null, '', {
    controls: splitSelectHtml('levers'),
    legend,
    content,
  })
}
function judgeScatterSvg(split, groupColors) {
  const points = []
  for (const r of runsCache) {
    const verdict = verdictsCache.get(r.key)
    if (!verdict) continue
    const objective = Number(r.summary.objective)
    if (!Number.isFinite(objective)) continue
    const score = Number(verdict.score)
    const group = split ? runSplitGroup(r, split) : undefined
    if (verdict.rejected) {
      points.push({
        x: objective,
        y: Number.isFinite(score) ? score : 0,
        marker: 'cross',
        label: `${shortKey(r.key)} · rejected${verdict.why ? ` — ${verdict.why}` : ''}`,
        group,
      })
    } else if (Number.isFinite(score)) {
      points.push({
        x: objective,
        y: score,
        label: `${shortKey(r.key)} · score ${Math.round(score)} · ${objectiveName()} ${formatObjective(objective)}`,
        group,
      })
    }
  }
  if (!points.length) return ''
  return buildScatterChart({
    points,
    xLabel: objectiveName(),
    yLabel: 'blended score',
    width: 460,
    height: 260,
    ariaLabel: `Judge score vs ${objectiveName()}`,
    groupColors,
  })
}
function judgeChartSectionHtml() {
  const split = chartSplitKey('judge')
  const groupColors = splitGroupColors(split)
  const svg = judgeScatterSvg(split, groupColors)
  return chartSectionHtml(`Judge score vs ${objectiveName()}`, svg, 'No verdicts yet.', {
    controls: splitSelectHtml('judge'),
    legend: svg ? chartLegendHtml(groupColors) : '',
  })
}
function trainEvalScatterSvg(split, groupColors) {
  const points = []
  for (const r of runsCache) {
    const evaluation = evaluationsCache.get(r.key)
    if (!evaluation) continue
    const train = Number(r.summary.objective)
    const evalObjective = Number(evaluation.objective)
    if (!Number.isFinite(train) || !Number.isFinite(evalObjective)) continue
    points.push({
      x: train,
      y: evalObjective,
      label: `${shortKey(r.key)} · train ${formatObjective(train)} · eval ${formatObjective(evalObjective)}`,
      group: split ? runSplitGroup(r, split) : undefined,
    })
  }
  if (!points.length) return ''
  return buildScatterChart({
    points,
    xLabel: `train ${objectiveName()}`,
    yLabel: `eval ${objectiveName()}`,
    refDiagonal: true,
    width: 460,
    height: 260,
    ariaLabel: `Eval vs train ${objectiveName()}`,
    groupColors,
  })
}
function trainEvalChartSectionHtml() {
  const split = chartSplitKey('train-eval')
  const groupColors = splitGroupColors(split)
  const svg = trainEvalScatterSvg(split, groupColors)
  return chartSectionHtml('Train vs eval', svg, 'No evaluations yet.', {
    controls: splitSelectHtml('train-eval'),
    legend: svg ? chartLegendHtml(groupColors) : '',
  })
}
async function renderCharts() {
  const body = byId('charts-body')
  if (!body) return
  if (!embedded()) {
    setHtml(body, '<div class="empty-hint">Open inside the Overseer to see charts.</div>')
    return
  }
  ;[runsCache, verdictsCache, evaluationsCache] = await Promise.all([
    readRuns(),
    readVerdicts(),
    readEvaluations(),
  ])
  setHtml(
    body,
    [
      timelineChartSectionHtml(),
      leverChartsSectionHtml(),
      judgeChartSectionHtml(),
      trainEvalChartSectionHtml(),
    ].join(''),
  )
}
function setupCharts() {
  const body = byId('charts-body')
  if (!body) return
  body.addEventListener('change', (event) => {
    const select = event.target.closest('select[data-chart-split]')
    if (!select) return
    chartSplits[select.dataset.chartSplit] = select.value
    renderCharts()
  })
}

// --- Hypotheses tab --------------------------------------------------------------
function specSummaryHtml(spec) {
  const s = spec || {}
  const items = []
  for (const [lever, values] of Object.entries(s.sweep || {})) {
    const list = Array.isArray(values) ? values.map((v) => String(v)).join(', ') : String(values)
    items.push(`<li><code>${escapeHtml(lever)}</code>: ${escapeHtml(list)}</li>`)
  }
  for (const [lever, value] of Object.entries(s.fixed || {})) {
    items.push(`<li><code>${escapeHtml(lever)}</code> = ${escapeHtml(String(value))}</li>`)
  }
  const seeds = Array.isArray(s.seeds) ? s.seeds.length : 0
  if (seeds) items.push(`<li>${seeds} seed${seeds === 1 ? '' : 's'}</li>`)
  if (!items.length) return '<p class="card-sub">Default config.</p>'
  return `<ul class="spec-list">${items.join('')}</ul>`
}
function hypothesisActionsHtml(h) {
  const id = escapeHtml(h.id)
  const c = h.campaign
  const viewRuns =
    c && c.finishedAt && Array.isArray(c.keys) && c.keys.length
      ? `<button type="button" data-action="view-runs" data-id="${id}" class="ghost-btn">View runs</button>`
      : ''
  if (h.status === 'accepted') {
    const busy = c && (c.status === 'running' || c.status === 'queued')
    return `<button type="button" data-action="run" data-id="${id}"${busy ? ' disabled' : ''}>Run campaign</button>${viewRuns}`
  }
  if (h.status === 'rejected') {
    return `<button type="button" data-action="restore" data-id="${id}" class="ghost-btn">Restore</button>${viewRuns}`
  }
  return `<button type="button" data-action="accept" data-id="${id}">Accept</button>
    <button type="button" data-action="reject" data-id="${id}" class="ghost-btn">Reject</button>${viewRuns}`
}
// The hypothesis's linked campaign: live status while queued/running, and once
// finished its OWN results (best among its keys, completed/failed counts) next
// to the project's overall best, so hypotheses are comparable.
function hypothesisCampaignHtml(h, liveIds) {
  const c = h.campaign
  if (!c) return ''
  if (c.status === 'queued') {
    return '<p class="badges-row"><span class="run-badge is-queued">campaign queued</span></p>'
  }
  if (c.status === 'running') {
    const live = liveIds && liveIds.has(c.activityId)
    return `<p class="badges-row"><span class="run-badge is-running">${live ? `${spinnerHtml()} ` : ''}campaign running</span></p>`
  }
  const keys = Array.isArray(c.keys) ? c.keys : []
  const keySet = new Set(keys)
  const own = runsCache.filter((r) => keySet.has(r.key))
  const ownBest = bestObjectiveOf(own)
  const best = Number.isFinite(ownBest) ? ownBest : Number(c.bestObjective)
  const projectBest = bestObjectiveOf(runsCache)
  const completed = Number(c.completed) || 0
  const failed = Number(c.failed) || 0
  const badge = `<span class="run-badge ${c.status === 'completed' ? 'is-done' : 'is-failed'}">campaign ${escapeHtml(c.status || 'completed')}</span>`
  return `<div class="hypothesis-campaign">
    <p class="badges-row">${badge}</p>
    <p class="card-sub">Own best ${escapeHtml(objectiveName())} <strong>${escapeHtml(formatObjective(best))}</strong> · project best ${escapeHtml(formatObjective(projectBest))}</p>
    <p class="card-sub">${completed} completed${failed ? ` · ${failed} failed` : ''} · finished ${escapeHtml(formatWhen(c.finishedAt))}</p>
  </div>`
}
function hypothesisCardHtml(h, liveIds) {
  const source = h.source === 'llm' ? 'LLM' : 'human'
  const by = h.proposedBy ? ` · ${escapeHtml(h.proposedBy)}` : ''
  return `<article class="hypothesis-card${h.status === 'rejected' ? ' is-muted' : ''}" data-id="${escapeHtml(h.id)}">
    <h4>${escapeHtml(h.title || h.id)}</h4>
    ${h.rationale ? `<p class="hypothesis-rationale">${escapeHtml(h.rationale)}</p>` : ''}
    ${specSummaryHtml(h.spec)}
    ${hypothesisCampaignHtml(h, liveIds)}
    <p class="card-sub">${escapeHtml(source)}${by} · created ${escapeHtml(formatWhen(h.createdAt))} · updated ${escapeHtml(formatWhen(h.updatedAt))}</p>
    <div class="hypothesis-actions">${hypothesisActionsHtml(h)}</div>
  </article>`
}
function hypothesisGroupHtml(status, items, liveIds) {
  const label = status[0].toUpperCase() + status.slice(1)
  const cards = items.length
    ? `<div class="hypothesis-cards">${items.map((h) => hypothesisCardHtml(h, liveIds)).join('')}</div>`
    : '<p class="card-sub">None.</p>'
  return `<section class="hypothesis-group">
    <h3>${escapeHtml(label)} <span class="group-count">${items.length}</span></h3>
    ${cards}
  </section>`
}
function renderProposeControls() {
  const btn = byId('propose-btn')
  if (btn) {
    btn.disabled = proposing
    btn.innerHTML = proposing ? busyButtonHtml('Proposing…') : 'Propose experiments'
  }
  renderTabLiveIndicator()
  const last = byId('propose-last')
  if (!last) return
  if (proposalSummary && proposalSummary.proposedAt) {
    const p = proposalSummary
    const by = p.proposedBy ? ` by ${p.proposedBy}` : ''
    const skipped = Number(p.skippedExisting) || 0
    last.textContent = `Last proposed ${formatWhen(p.proposedAt)}${by} — ${Number(p.proposed) || 0} new${skipped ? `, ${skipped} already known` : ''}.`
    last.hidden = false
  } else {
    last.hidden = true
  }
}
async function renderHypotheses() {
  const body = byId('hypotheses-body')
  if (!body) return
  if (!embedded()) {
    renderProposeControls()
    body.innerHTML = '<div class="empty-hint">Open inside the Overseer to manage hypotheses.</div>'
    return
  }
  ;[hypothesesCache, proposalSummary, runsCache] = await Promise.all([
    readHypotheses(),
    readProposal(),
    readRuns(),
  ])
  const liveIds = hypothesesCache.some((h) => h.campaign && h.campaign.status === 'running')
    ? await readLiveActivityIds()
    : new Set()
  renderProposeControls()
  if (!hypothesesCache.length) {
    body.innerHTML =
      '<div class="empty-hint">No hypotheses yet — propose some or add your own.</div>'
    return
  }
  const sorted = [...hypothesesCache].sort((a, b) =>
    String(b.createdAt || '').localeCompare(String(a.createdAt || '')),
  )
  const groups = { pending: [], accepted: [], rejected: [] }
  for (const h of sorted) (groups[h.status] || groups.pending).push(h)
  body.innerHTML = HYPOTHESIS_STATUSES.map((status) =>
    hypothesisGroupHtml(status, groups[status], liveIds),
  ).join('')
}
async function onProposeClick() {
  if (proposing) return
  if (!embedded()) {
    setStatusLine('hypotheses-status', 'Open inside the Overseer to propose experiments.', false)
    return
  }
  const epoch = projectEpoch
  setStatusLine('hypotheses-status', '')
  try {
    const result = await startOrEnqueue('propose', trainerActivityParams(), 'Propose experiments')
    if (result.queued) {
      if (epoch === projectEpoch) {
        setStatusLine('hypotheses-status', queuedStatusText(result.ahead))
      }
      return
    }
    proposing = true
    renderProposeControls()
    const act = await observeQuickActivity(result.activityId)
    if (epoch === projectEpoch) {
      setStatusLine('hypotheses-status', quickActivityFailureText(act, 'Proposing'), true)
    }
  } catch {
    if (epoch === projectEpoch) {
      setStatusLine('hypotheses-status', 'Could not start proposing — please try again.', true)
    }
  } finally {
    if (proposing) {
      proposing = false
      renderProposeControls()
      if (epoch === projectEpoch) await renderHypotheses()
      pumpQueue()
    }
  }
}
async function setHypothesisStatus(id, status) {
  const h = hypothesesCache.find((x) => x.id === id)
  if (!h) return
  setStatusLine('hypotheses-status', '')
  try {
    await putHypothesis({ ...h, status, updatedAt: nowIso() })
  } catch {
    setStatusLine('hypotheses-status', 'Could not update the hypothesis — please try again.', true)
    return
  }
  await renderHypotheses()
}
// Stamp a hypothesis record's `campaign` block (preserving every other field),
// keyed by the explicit recordType so queue dispatches survive navigation.
async function stampHypothesisCampaign(recordType, hypothesisId, campaign) {
  const recs = await queryRecords(recordType + '-hypothesis', hypothesisId)
  const existing = (recs[0] && recs[0].content) || null
  if (!existing) return
  await window.OverseerBridge.putData({
    type: recordType + '-hypothesis',
    key: hypothesisId,
    content: { ...existing, campaign, updatedAt: nowIso() },
  })
}
// Removing a queued campaign from the queue clears its hypothesis stamp again.
async function clearHypothesisCampaign(recordType, hypothesisId, queueId) {
  const recs = await queryRecords(recordType + '-hypothesis', hypothesisId)
  const existing = (recs[0] && recs[0].content) || null
  if (!existing || !existing.campaign) return
  if (queueId && existing.campaign.queueId !== queueId) return
  const { campaign, ...rest } = existing
  await window.OverseerBridge.putData({
    type: recordType + '-hypothesis',
    key: hypothesisId,
    content: { ...rest, updatedAt: nowIso() },
  })
}
// Finalise hypotheses whose campaign has settled: copy the campaign record's
// results in when its activityId matches, or fall back to the activity's final
// status when the campaign record was superseded before we saw it.
async function stampHypothesisCampaignResults() {
  if (!manifest) return
  const recordType = manifest.recordType
  const hyps = await readHypotheses()
  const open = hyps.filter((h) => h.campaign && h.campaign.activityId && !h.campaign.finishedAt)
  if (!open.length) return
  const campaign = await readCampaign()
  for (const h of open) {
    if (campaign && campaign.activityId === h.campaign.activityId && campaign.finishedAt) {
      await stampHypothesisCampaign(recordType, h.id, {
        activityId: campaign.activityId,
        launchedAt: h.campaign.launchedAt,
        status: campaign.aborted ? 'aborted' : 'completed',
        keys: Array.isArray(campaign.keys) ? campaign.keys : [],
        bestKey: campaign.bestKey,
        bestObjective: campaign.bestObjective,
        completed: campaign.completed,
        failed: campaign.failed,
        finishedAt: campaign.finishedAt,
      })
      continue
    }
    const act = await getActivity(h.campaign.activityId)
    if (act && act.status && act.status !== 'running' && act.status !== 'paused') {
      await stampHypothesisCampaign(recordType, h.id, {
        ...h.campaign,
        status: act.status,
        finishedAt: nowIso(),
      })
    }
  }
}
// Switch to the Runs tab filtered down to the hypothesis campaign's own runs.
function viewHypothesisRuns(id) {
  const h = hypothesesCache.find((x) => x.id === id)
  if (!h || !h.campaign || !Array.isArray(h.campaign.keys) || !h.campaign.keys.length) return
  runsFilterKeys = new Set(h.campaign.keys)
  runsFilterLabel = h.title || shortKey(id)
  showTab('runs')
}
// Launch the accepted hypothesis as a training campaign, exactly like the
// Launch tab: start 'train' with its spec (or queue it behind the live
// activity), stamp the hypothesis with the campaign link, then observe.
async function runHypothesisCampaign(id, button) {
  const h = hypothesesCache.find((x) => x.id === id)
  if (!h) return
  const epoch = projectEpoch
  const recordType = manifest.recordType
  setStatusLine('hypotheses-status', '')
  if (button) button.disabled = true
  try {
    const result = await startOrEnqueue(
      'train',
      trainerComputeParams({ spec: h.spec }),
      `Campaign: ${h.title || h.id}`,
      { hypothesisId: h.id },
    )
    if (result.queued) {
      await stampHypothesisCampaign(recordType, h.id, {
        status: 'queued',
        queueId: result.id,
        queuedAt: nowIso(),
      })
      if (epoch !== projectEpoch) return
      setStatusLine('hypotheses-status', queuedStatusText(result.ahead))
      await renderHypotheses()
      return
    }
    await stampHypothesisCampaign(recordType, h.id, {
      activityId: result.activityId,
      launchedAt: nowIso(),
      status: 'running',
    })
    if (epoch !== projectEpoch) return
    currentActivityId = result.activityId
    lastProgress = null
    lastCampaign = null
    lastActivityStatus = 'running'
    showTab('activity')
    observeActivityUntilDone(result.activityId)
  } catch {
    if (epoch === projectEpoch) {
      setStatusLine('hypotheses-status', 'Could not start the campaign — please try again.', true)
    }
  } finally {
    if (button) button.disabled = false
  }
}
function validateHypothesisSpec(text) {
  let parsed
  try {
    parsed = JSON.parse(text)
  } catch {
    return { error: 'Spec must be valid JSON.' }
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { error: 'Spec must be a JSON object.' }
  }
  const unknownKeys = Object.keys(parsed).filter((k) => !HYPOTHESIS_SPEC_KEYS.includes(k))
  if (unknownKeys.length) {
    return {
      error: `Unknown spec ${unknownKeys.length === 1 ? 'key' : 'keys'} ${unknownKeys.join(', ')} — only sweep, fixed and seeds are allowed.`,
    }
  }
  for (const part of ['sweep', 'fixed']) {
    const value = parsed[part]
    if (value === undefined) continue
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return { error: `"${part}" must be an object of lever values.` }
    }
  }
  if (parsed.seeds !== undefined && !Array.isArray(parsed.seeds)) {
    return { error: '"seeds" must be an array.' }
  }
  const known = new Set(Object.keys((manifest && manifest.levers) || {}))
  const unknownLevers = [
    ...Object.keys(parsed.sweep || {}),
    ...Object.keys(parsed.fixed || {}),
  ].filter((k) => !known.has(k))
  if (unknownLevers.length) {
    return {
      error: `Unknown ${unknownLevers.length === 1 ? 'lever' : 'levers'} ${unknownLevers.join(', ')} — this manifest declares: ${[...known].join(', ') || 'none'}.`,
    }
  }
  return { spec: parsed }
}
function renderHypothesisForm() {
  const form = byId('hypothesis-form')
  if (!form) return
  form.innerHTML = `
    <fieldset class="lever">
      <legend>New hypothesis</legend>
      <div class="hypothesis-form-fields">
        <label class="field"><span>Title</span>
          <input type="text" name="title" placeholder="e.g. Higher learning rate with more epochs" />
        </label>
        <label class="field"><span>Rationale</span>
          <textarea name="rationale" rows="3"></textarea>
        </label>
        <label class="field"><span>Spec <em>(JSON with sweep / fixed / seeds)</em></span>
          <textarea name="spec" rows="3" spellcheck="false">${escapeHtml(HYPOTHESIS_SPEC_PLACEHOLDER)}</textarea>
        </label>
      </div>
    </fieldset>
    <div class="form-actions">
      <button type="submit" id="hypothesis-save-btn">Save hypothesis</button>
      <button type="button" id="hypothesis-cancel-btn" class="ghost-btn">Cancel</button>
    </div>
    <p id="hypothesis-form-error" class="form-status is-error" role="alert" hidden></p>`
}
function toggleHypothesisForm(show) {
  const form = byId('hypothesis-form')
  if (!form) return
  form.hidden = !show
  if (show) {
    renderHypothesisForm()
    if (form.elements.title) form.elements.title.focus()
  } else {
    form.innerHTML = ''
  }
}
async function onHypothesisSave(event) {
  event.preventDefault()
  const form = byId('hypothesis-form')
  if (!form) return
  if (!embedded()) {
    setStatusLine('hypothesis-form-error', 'Open inside the Overseer to add hypotheses.', true)
    return
  }
  const title = String((form.elements.title && form.elements.title.value) || '').trim()
  if (!title) {
    setStatusLine('hypothesis-form-error', 'Give the hypothesis a title.', true)
    return
  }
  const { spec, error } = validateHypothesisSpec(
    (form.elements.spec && form.elements.spec.value) || '',
  )
  if (error) {
    setStatusLine('hypothesis-form-error', error, true)
    return
  }
  const now = nowIso()
  const content = {
    id: randomHexId(),
    title,
    rationale: String((form.elements.rationale && form.elements.rationale.value) || '').trim(),
    spec,
    status: 'pending',
    source: 'human',
    createdAt: now,
    updatedAt: now,
  }
  const saveBtn = byId('hypothesis-save-btn')
  if (saveBtn) saveBtn.disabled = true
  try {
    await putHypothesis(content)
  } catch {
    setStatusLine(
      'hypothesis-form-error',
      'Could not save the hypothesis — please try again.',
      true,
    )
    if (saveBtn) saveBtn.disabled = false
    return
  }
  toggleHypothesisForm(false)
  await renderHypotheses()
}
function setupHypotheses() {
  const proposeBtn = byId('propose-btn')
  if (proposeBtn) {
    proposeBtn.addEventListener('click', onProposeClick)
    proposeBtn.insertAdjacentHTML('afterend', helpCalloutHtml(PROPOSE_HELP_TEXT))
  }
  const addToggle = byId('hypothesis-add-toggle')
  if (addToggle) {
    addToggle.addEventListener('click', () => {
      const form = byId('hypothesis-form')
      toggleHypothesisForm(!!(form && form.hidden))
    })
  }
  const form = byId('hypothesis-form')
  if (form) {
    form.addEventListener('submit', onHypothesisSave)
    form.addEventListener('click', (event) => {
      if (event.target.closest('#hypothesis-cancel-btn')) toggleHypothesisForm(false)
    })
  }
  const body = byId('hypotheses-body')
  if (body) {
    body.addEventListener('click', (event) => {
      const btn = event.target.closest('button[data-action]')
      if (!btn) return
      const { action, id } = btn.dataset
      if (action === 'accept') setHypothesisStatus(id, 'accepted')
      else if (action === 'reject') setHypothesisStatus(id, 'rejected')
      else if (action === 'restore') setHypothesisStatus(id, 'pending')
      else if (action === 'run') runHypothesisCampaign(id, btn)
      else if (action === 'view-runs') viewHypothesisRuns(id)
    })
  }
}

// --- Launch tab ----------------------------------------------------------------
function leverEntries() {
  return Object.entries((manifest && manifest.levers) || {})
}
function leverRange(spec) {
  const r = spec.range
  if (Array.isArray(r) && r.length >= 2) return { min: r[0], max: r[1] }
  if (r && typeof r === 'object') return { min: r.min, max: r.max }
  return { min: undefined, max: undefined }
}
function numberLeverHtml(key, spec) {
  const { min, max } = leverRange(spec)
  const minAttr = Number.isFinite(Number(min)) ? ` min="${Number(min)}"` : ''
  const maxAttr = Number.isFinite(Number(max)) ? ` max="${Number(max)}"` : ''
  const value = spec.default === undefined ? '' : escapeHtml(String(spec.default))
  return `<div class="lever-grid">
    <label class="field"><span>Value</span>
      <input type="number" step="any" name="fixed:${escapeHtml(key)}" value="${value}"${minAttr}${maxAttr} />
    </label>
    <label class="field"><span>Sweep values <em>(comma-separated, overrides value)</em></span>
      <input type="text" name="sweep:${escapeHtml(key)}" placeholder="e.g. 0.01, 0.05, 0.1" />
    </label>
  </div>`
}
// Best objective seen so far per value of a choice lever (marginal, across all
// non-failed runs) — used to annotate the launch options with "best <obj>" + a ★.
function leverBestSoFar(leverKey) {
  const dir = objectiveDirection()
  const best = new Map()
  for (const r of runsCache) {
    const v = (r.summary.config || {})[leverKey]
    const obj = Number(r.summary.objective)
    if (v === undefined || !Number.isFinite(obj) || r.summary.status === 'failed') continue
    const k = String(v)
    if (!best.has(k) || (dir === 'min' ? obj < best.get(k) : obj > best.get(k))) best.set(k, obj)
  }
  return best
}
function choiceLeverHtml(key, spec) {
  const choices = Array.isArray(spec.choices) ? spec.choices : []
  const best = leverBestSoFar(key)
  let topValue
  for (const [v, o] of best) {
    if (
      topValue === undefined ||
      (objectiveDirection() === 'min' ? o < best.get(topValue) : o > best.get(topValue))
    ) {
      topValue = v
    }
  }
  const optionText = (c) => {
    const b = best.get(String(c))
    if (b === undefined) return escapeHtml(String(c))
    return `${escapeHtml(String(c))}${escapeHtml(` — best ${formatObjective(b)}`)}${String(c) === topValue ? ' ★' : ''}`
  }
  const fixedOptions = choices
    .map(
      (c) =>
        `<option value="${escapeHtml(String(c))}"${String(c) === String(spec.default) ? ' selected' : ''}>${optionText(c)}</option>`,
    )
    .join('')
  const sweepOptions = choices
    .map((c) => `<option value="${escapeHtml(String(c))}">${optionText(c)}</option>`)
    .join('')
  const size = Math.max(2, Math.min(choices.length, 4))
  return `<div class="lever-grid">
    <label class="field"><span>Value</span>
      <select name="fixed:${escapeHtml(key)}">${fixedOptions}</select>
    </label>
    <label class="field"><span>Sweep choices <em>(select several, overrides value)</em></span>
      <select name="sweep:${escapeHtml(key)}" multiple size="${size}">${sweepOptions}</select>
    </label>
  </div>`
}
function booleanLeverHtml(key, spec) {
  return `<label class="check-row">
    <input type="checkbox" name="fixed:${escapeHtml(key)}"${spec.default ? ' checked' : ''} />
    <span>Enabled</span>
  </label>`
}
function leverFieldsetHtml(key, spec) {
  const inner =
    spec.type === 'number'
      ? numberLeverHtml(key, spec)
      : spec.type === 'choice'
        ? choiceLeverHtml(key, spec)
        : booleanLeverHtml(key, spec)
  return `<fieldset class="lever">
    <legend>${escapeHtml(key)} <span class="lever-type">${escapeHtml(spec.type || '')}</span>${spec.description ? helpCalloutHtml(spec.description) : ''}</legend>
    ${inner}
  </fieldset>`
}
function savedAutoEval() {
  try {
    const v = sessionStorage.getItem(AUTO_EVAL_SS)
    return v === null ? true : v === '1'
  } catch {
    return true
  }
}
function rememberAutoEval(on) {
  try {
    sessionStorage.setItem(AUTO_EVAL_SS, on ? '1' : '')
  } catch {
    // storage may be unavailable in a sandboxed frame — purely best-effort
  }
}
// Max parallel runs is an Activity-level setting (not a per-launch field): how many
// runs of a campaign may execute at once. Persisted; read at launch.
function savedConcurrency() {
  try {
    const n = Math.floor(Number(sessionStorage.getItem(CONCURRENCY_SS)))
    return Number.isFinite(n) && n >= 1 ? n : 1
  } catch {
    return 1
  }
}
function rememberConcurrency(n) {
  try {
    sessionStorage.setItem(CONCURRENCY_SS, String(Math.max(1, Math.floor(Number(n)) || 1)))
  } catch {
    // storage may be unavailable in a sandboxed frame — purely best-effort
  }
}
function savedComputeTarget() {
  try {
    return sessionStorage.getItem(COMPUTE_TARGET_SS) || ''
  } catch {
    return ''
  }
}
function rememberComputeTarget(value) {
  try {
    sessionStorage.setItem(COMPUTE_TARGET_SS, value)
  } catch {
    // storage may be unavailable in a sandboxed frame — purely best-effort
  }
}
// The remote target a "Run on" value names: empty or "local" → run locally.
function remoteComputeTarget(value) {
  const target = String(value || '').trim()
  return target && target !== 'local' ? target : ''
}
// The "Run on" <option>s: a "Local" default, one per known runner labelled with
// its online/offline state, and — when the persisted selection names a runner no
// longer in the list (a queued or older target) — a kept "(unavailable)" option
// so the selection still shows rather than silently reverting to Local.
function computeTargetOptionsHtml(runners, selected) {
  const list = Array.isArray(runners) ? runners : []
  const sel = String(selected || '')
  const options = [
    `<option value=""${sel === '' || sel === 'local' ? ' selected' : ''}>Local</option>`,
  ]
  for (const runner of list) {
    const id = String(runner.id || '')
    if (!id) continue
    const label = `${runner.name || id} (${runner.online ? 'online' : 'offline'})`
    options.push(
      `<option value="${escapeHtml(id)}"${id === sel ? ' selected' : ''}>${escapeHtml(label)}</option>`,
    )
  }
  if (sel && sel !== 'local' && !list.some((r) => String(r.id || '') === sel)) {
    options.push(
      `<option value="${escapeHtml(sel)}" selected>(unavailable) ${escapeHtml(sel)}</option>`,
    )
  }
  return options.join('')
}
// The "Run on" field's inner markup (the select + hint), rebuilt whenever the
// runner list refreshes. The hint nudges toward the Compute Runners panel when
// nothing is paired.
function computeTargetFieldHtml(runners, selected) {
  const list = Array.isArray(runners) ? runners : []
  const hint = list.length
    ? 'A paired runner runs the campaign; pick Local to run here. Manage runners in the Compute Runners panel.'
    : NO_RUNNERS_HINT
  return `<span>Run on</span>
    <select name="computeTarget">${computeTargetOptionsHtml(list, selected)}</select>
    <em class="field-hint">${escapeHtml(hint)}</em>`
}
// Refresh the Launch tab's "Run on" select from the live runner list, rebuilding
// it in place via setHtml so it never flashes. Guarded by projectEpoch so a slow
// list never paints over an unloaded or navigated-away launch form.
async function refreshLaunchRunners() {
  const field = byId('launch-target-field')
  if (!field || !embedded()) return
  const epoch = projectEpoch
  let runners = []
  try {
    const res = await window.OverseerBridge.listRunners()
    runners = (res && res.runners) || []
  } catch {
    runners = []
  }
  if (epoch !== projectEpoch) return
  launchRunnersCache = runners
  setHtml(field, computeTargetFieldHtml(runners, savedComputeTarget()))
}
function renderLaunchForm() {
  const form = byId('launch-form')
  if (!form) return
  const levers = leverEntries()
    .map(([key, spec]) => leverFieldsetHtml(key, spec))
    .join('')
  form.innerHTML = `
    ${presetsSelectHtml()}
    ${levers || '<p class="card-sub">This manifest declares no levers — the campaign runs the default config.</p>'}
    <fieldset class="lever">
      <legend>Campaign</legend>
      <div class="lever-grid">
        <label class="field"><span>Thesis ${helpCalloutHtml('What this campaign tests, e.g. "fee-penalty reward" or "1m data prep". Stamped on every run so you can group + compare by experiment in the By-experiment view. Optional.')}</span>
          <input type="text" name="thesis" placeholder="what are you testing? (optional)" />
        </label>
        <label class="field"><span>Testing which setting? ${helpCalloutHtml('Optional: the lever this thesis varies, so the by-experiment view can highlight it. Leave blank for theses outside the levers (e.g. a new data prep or code change).')}</span>
          <select name="thesisTarget"><option value="">—</option>${leverEntries()
            .map(([k]) => `<option value="${escapeHtml(k)}">${escapeHtml(k)}</option>`)
            .join('')}</select>
        </label>
        <label class="field"><span>Seeds ${helpCalloutHtml('How many seeds to run per config (0…N−1). Keep at 1 while exploring; raise it when homing in to measure variance across seeds (the by-setup view then shows the spread).')}</span>
          <input type="number" name="seeds" min="1" step="1" value="1" />
        </label>
        <label class="check-row launch-refresh">
          <input type="checkbox" name="refresh" />
          <span>Refresh — re-run configs that already have a result ${helpCalloutHtml('Off (default): a config with a completed run is skipped. On: re-run it anyway, e.g. after changing code or data.')}</span>
        </label>
        <label class="check-row launch-skip-explored">
          <input type="checkbox" name="skipExplored" checked />
          <span>Exploration — skip setups already tried (any seed) ${helpCalloutHtml('On by default: if a setup (the config ignoring seed) was already run, skip it — exploration should not re-test the same idea. Turn OFF when homing in to run more seeds of a setup.')}</span>
        </label>
        <label class="check-row launch-autoeval">
          <input type="checkbox" name="autoEval"${savedAutoEval() ? ' checked' : ''} />
          <span>Auto-evaluate completed runs ${helpCalloutHtml('After each run finishes, automatically re-test its saved checkpoint (shown in the Eval column).')}</span>
        </label>
        <label class="field launch-target" id="launch-target-field">${computeTargetFieldHtml(launchRunnersCache, savedComputeTarget())}</label>
      </div>
    </fieldset>
    <p class="launch-summary" id="launch-summary"></p>
    <div class="form-actions">
      <button type="submit" id="launch-btn">Launch campaign</button>
    </div>
    <p id="launch-status" class="form-status" role="status"></p>`
  updateLaunchSummary()
}
// All launch presets in order: the quick-start (if any) first, then the manifest's
// curated known-good setups. The "Load a setup" select indexes into this list.
function launchPresets() {
  const list = []
  if (manifest && manifest.quickStart && manifest.quickStart.fixed) {
    list.push({
      label: manifest.quickStart.label || 'Quick start',
      fixed: manifest.quickStart.fixed,
    })
  }
  if (manifest && Array.isArray(manifest.presets)) {
    for (const p of manifest.presets)
      if (p && (p.fixed || p.sweep))
        list.push({
          label: p.label || 'Preset',
          fixed: p.fixed,
          sweep: p.sweep,
          seeds: p.seeds,
          thesis: p.thesis,
          thesisTarget: p.thesisTarget,
          isExperiment: !!p.sweep,
        })
  }
  return list
}
function presetsSelectHtml() {
  const presets = launchPresets()
  if (!presets.length) return ''
  const opt = (p, i) => `<option value="${i}">${escapeHtml(p.label)}</option>`
  const indexed = presets.map((p, i) => [p, i])
  const experiments = indexed.filter(([p]) => p.isExperiment)
  const setups = indexed.filter(([p]) => !p.isExperiment)
  const groups = []
  if (experiments.length)
    groups.push(
      `<optgroup label="Experiments — one-click campaigns">${experiments.map(([p, i]) => opt(p, i)).join('')}</optgroup>`,
    )
  if (setups.length)
    groups.push(
      `<optgroup label="Known-good single setups">${setups.map(([p, i]) => opt(p, i)).join('')}</optgroup>`,
    )
  return `<fieldset class="lever launch-presets">
    <legend>Load a setup or experiment</legend>
    <label class="field"><span>Presets <em>(an experiment fills a whole sweep + seeds + thesis; a setup pins one config to seed your own sweep)</em></span>
      <select id="launch-preset-select"><option value="">— choose —</option>${groups.join('')}</select>
    </label>
  </fieldset>`
}
// Load the launch form from a preset. A preset may set `fixed` lever values, a `sweep`
// (lever → candidate list, for a full experiment), a `seeds` count, and a `thesis`/`thesisTarget`
// tag. The form is first reset to defaults so the preset fully determines it, then the preset's
// fixed values, sweep selections, seeds and thesis are applied.
function applyPreset(preset) {
  const form = byId('launch-form')
  if (!form || !preset) return
  for (const [key, spec] of leverEntries()) {
    const sweepEl = form.elements['sweep:' + key]
    if (sweepEl) {
      if (sweepEl.multiple) for (const o of sweepEl.options) o.selected = false
      else sweepEl.value = ''
    }
    const fixedEl = form.elements['fixed:' + key]
    if (fixedEl) {
      if (fixedEl.type === 'checkbox') fixedEl.checked = !!spec.default
      else fixedEl.value = spec.default === undefined ? '' : String(spec.default)
    }
  }
  for (const [key, value] of Object.entries(preset.fixed || {})) {
    const fixedEl = form.elements['fixed:' + key]
    if (fixedEl) {
      if (fixedEl.type === 'checkbox') fixedEl.checked = !!value
      else fixedEl.value = String(value)
    }
  }
  for (const [key, values] of Object.entries(preset.sweep || {})) {
    const sweepEl = form.elements['sweep:' + key]
    if (!sweepEl || !Array.isArray(values)) continue
    if (sweepEl.multiple) {
      const want = new Set(values.map(String))
      for (const o of sweepEl.options) o.selected = want.has(o.value)
    } else {
      sweepEl.value = values.join(', ')
    }
  }
  if (preset.seeds && form.elements.seeds) form.elements.seeds.value = String(preset.seeds)
  if (form.elements.thesis) form.elements.thesis.value = preset.thesis || ''
  if (form.elements.thesisTarget) form.elements.thesisTarget.value = preset.thesisTarget || ''
  updateLaunchSummary()
}
// Clone-to-Launch and other fixed-only callers: a preset that only pins lever values.
function applyPresetFixed(fixed) {
  applyPreset({ fixed: fixed || {} })
}
function parseNumberList(text) {
  return String(text || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number)
    .filter((n) => Number.isFinite(n))
}
function readSweepValues(form, key, spec) {
  const el = form.elements['sweep:' + key]
  if (!el) return []
  if (spec.type === 'number') return parseNumberList(el.value)
  if (spec.type === 'choice') return [...el.selectedOptions].map((o) => o.value)
  return []
}
function readFixedValue(form, key, spec) {
  const el = form.elements['fixed:' + key]
  if (!el) return spec.default
  if (spec.type === 'boolean') return !!el.checked
  if (spec.type === 'number') {
    const v = Number(el.value)
    return el.value !== '' && Number.isFinite(v) ? v : spec.default
  }
  return el.value !== '' ? el.value : spec.default
}
function readSeedCount(form) {
  const el = form.elements.seeds
  const n = Math.floor(Number(el && el.value))
  return Number.isFinite(n) && n >= 1 ? n : 1
}
function buildSpecFromForm(form) {
  const sweep = {}
  const fixed = {}
  for (const [key, spec] of leverEntries()) {
    const values = readSweepValues(form, key, spec)
    if (values.length) sweep[key] = values
    else fixed[key] = readFixedValue(form, key, spec)
  }
  const seedCount = readSeedCount(form)
  return { sweep, fixed, seeds: Array.from({ length: seedCount }, (_, i) => i) }
}
function updateLaunchSummary() {
  const form = byId('launch-form')
  const line = byId('launch-summary')
  if (!form || !line) return
  const spec = buildSpecFromForm(form)
  const configs = Object.values(spec.sweep).reduce((acc, values) => acc * values.length, 1)
  const seeds = spec.seeds.length
  const total = configs * seeds
  const target = remoteComputeTarget(savedComputeTarget())
  line.textContent = `${configs} configuration${configs === 1 ? '' : 's'} × ${seeds} seed${seeds === 1 ? '' : 's'} = ${total} run${total === 1 ? '' : 's'}${target ? ` on ${target}` : ''}`
}
function campaignLabel(spec) {
  const sweeps = Object.entries(spec.sweep || {}).map(
    ([key, values]) => `${key} × ${values.length}`,
  )
  const seeds = Array.isArray(spec.seeds) ? spec.seeds.length : 1
  return `Campaign: ${sweeps.length ? `${sweeps.join(', ')}, ` : ''}seeds ${seeds}`
}
async function onLaunchSubmit(event) {
  event.preventDefault()
  const form = byId('launch-form')
  const status = byId('launch-status')
  const button = byId('launch-btn')
  if (!form) return
  if (!embedded()) {
    if (status) status.textContent = 'Open inside the Overseer to launch campaigns.'
    return
  }
  const epoch = projectEpoch
  const spec = buildSpecFromForm(form)
  const refresh = !!(form.elements.refresh && form.elements.refresh.checked)
  const autoEval = !!(form.elements.autoEval && form.elements.autoEval.checked)
  const concurrency = savedConcurrency()
  const skipExplored = !!(form.elements.skipExplored && form.elements.skipExplored.checked)
  const thesis = String((form.elements.thesis && form.elements.thesis.value) || '').trim()
  const thesisTarget = String(
    (form.elements.thesisTarget && form.elements.thesisTarget.value) || '',
  ).trim()
  if (button) button.disabled = true
  if (status) status.textContent = 'Starting campaign…'
  try {
    const params = trainerComputeParams({
      spec,
      refresh,
      ...(concurrency > 1 ? { concurrency } : {}),
      ...(skipExplored ? { skipExplored: true } : {}),
      ...(thesis ? { thesis } : {}),
      ...(thesis && thesisTarget ? { thesisTarget } : {}),
    })
    const result = await startOrEnqueue(
      'train',
      params,
      campaignLabel(spec),
      autoEval ? { autoEval: true } : undefined,
    )
    if (result.queued) {
      if (epoch === projectEpoch && status) status.textContent = queuedStatusText(result.ahead)
      return
    }
    if (autoEval) await putAutoEvalMarker(result.activityId, params)
    if (epoch !== projectEpoch) return
    currentActivityId = result.activityId
    lastProgress = null
    lastCampaign = null
    lastActivityStatus = 'running'
    if (status) status.textContent = ''
    showTab('activity')
    observeActivityUntilDone(result.activityId)
  } catch {
    if (epoch === projectEpoch && status) {
      status.textContent = 'Could not start the campaign — please try again.'
    }
  } finally {
    if (button) button.disabled = false
  }
}
function setupLaunch() {
  const form = byId('launch-form')
  if (!form) return
  form.addEventListener('submit', onLaunchSubmit)
  form.addEventListener('input', (event) => {
    if (event.target && event.target.name === 'computeTarget') {
      rememberComputeTarget(event.target.value)
    }
    updateLaunchSummary()
  })
  form.addEventListener('change', (event) => {
    if (event.target && event.target.id === 'launch-preset-select') {
      const idx = Number(event.target.value)
      const presets = launchPresets()
      if (Number.isInteger(idx) && presets[idx]) applyPreset(presets[idx])
      return
    }
    if (event.target && event.target.name === 'computeTarget') {
      rememberComputeTarget(event.target.value)
    }
    if (event.target && event.target.name === 'autoEval') rememberAutoEval(event.target.checked)
    updateLaunchSummary()
  })
}

// --- Activity tab ----------------------------------------------------------------
// Look up one activity from the host project's list (or null). The list spans
// every training project, so an entry carrying a recordType only matches when
// it is the selected project's. The run object carries { status, isLive } so
// observers can tell a live run from an orphaned one.
async function getActivity(activityId) {
  try {
    const res = await window.OverseerBridge.listActivities()
    const act = ((res && res.activities) || []).find((a) => a.activityId === activityId) || null
    if (act && manifest && act.recordType && act.recordType !== manifest.recordType) return null
    return act
  } catch {
    return null
  }
}
// The campaign still in flight per the latest progress record, with its activity
// (null when nothing is running or everything already settled).
async function findRunningActivity() {
  const progress = await readProgress()
  if (!progress || !progress.activityId) return null
  if (progress.phase === 'done') return null
  const campaign = await readCampaign()
  if (campaign && campaign.activityId === progress.activityId && campaign.finishedAt) return null
  const activity = await getActivity(progress.activityId)
  return { activityId: progress.activityId, activity, progress }
}
// On opening a project, re-attach to a campaign still marked running. Live →
// observe; orphaned or paused (not live) → resume it once (carfinder's
// resume-on-load logic), and fall back to a Resume button when that fails.
async function resumeRunningActivity() {
  const epoch = projectEpoch
  const progress = await readProgress()
  const campaign = await readCampaign()
  const found = await findRunningActivity()
  if (epoch !== projectEpoch) return
  lastProgress = progress
  lastCampaign = campaign
  if (!found) {
    if (lastCampaign || (lastProgress && lastProgress.phase === 'done')) {
      lastActivityStatus = 'completed'
    }
    renderActivity()
    return
  }
  currentActivityId = found.activityId
  lastProgress = found.progress
  const act = found.activity
  if (act && act.status && act.status !== 'running' && act.status !== 'paused') {
    lastActivityStatus = act.status
    renderActivity()
    await renderRuns()
    return
  }
  if (act && act.status === 'running' && act.isLive !== false) {
    lastActivityStatus = 'running'
    renderActivity()
    observeActivityUntilDone(found.activityId)
    return
  }
  try {
    await window.OverseerBridge.resumeActivity(found.activityId)
    if (epoch !== projectEpoch) return
    lastActivityStatus = 'running'
    renderActivity()
    observeActivityUntilDone(found.activityId)
  } catch {
    if (epoch !== projectEpoch) return
    lastActivityStatus = 'paused'
    renderActivity()
  }
}
// On opening a project (after re-attaching any running train campaign), re-attach
// to a quick judge / propose activity that is still live for this project — the
// piece a page reload would otherwise drop, leaving the Judge / Propose button
// looking idle mid-run. The activity list carries the activity's type in its
// resumeToken, so judge and propose are told apart precisely and each reuses its
// own settle handler (button spinner via judging/proposing, then refresh
// verdicts / hypotheses). Guarded by projectEpoch so navigating away cancels it,
// and skipped when that button is already observing.
async function resumeRunningQuickActivity() {
  const epoch = projectEpoch
  if (judging || proposing) return
  let activities
  try {
    const res = await window.OverseerBridge.listActivities()
    activities = (res && res.activities) || []
  } catch {
    return
  }
  if (epoch !== projectEpoch || !manifest) return
  const live = activities.filter(
    (a) => a.recordType === manifest.recordType && a.status === 'running' && a.isLive !== false,
  )
  const judge = live.find((a) => quickActivityType(a) === 'judge')
  const propose = live.find((a) => quickActivityType(a) === 'propose')
  if (judge) observeResumedJudge(judge.activityId, epoch)
  if (propose) observeResumedPropose(propose.activityId, epoch)
}
// The activity's type from its resume token (the host carries `{activityType,…}`
// there), used to tell a running judge / propose / train apart in the list.
function quickActivityType(activity) {
  const token = activity && activity.resumeToken
  return token && typeof token.activityType === 'string' ? token.activityType : ''
}
// Re-attach to a live judge: light the Judge button's spinner, observe until it
// settles, then drop the spinner, refresh the verdicts in the Runs table and let
// the queue dispatch — mirroring onJudgeClick's settle path.
async function observeResumedJudge(activityId, epoch) {
  judging = true
  renderJudgeControls()
  const act = await observeQuickActivity(activityId)
  if (epoch !== projectEpoch) return
  judging = false
  renderJudgeControls()
  setStatusLine('judge-status', quickActivityFailureText(act, 'Judging'), true)
  await renderRuns()
  pumpQueue()
}
// Re-attach to a live propose: light the Propose button's spinner, observe until
// it settles, then drop the spinner, refresh the hypotheses and let the queue
// dispatch — mirroring onProposeClick's settle path.
async function observeResumedPropose(activityId, epoch) {
  proposing = true
  renderProposeControls()
  const act = await observeQuickActivity(activityId)
  if (epoch !== projectEpoch) return
  proposing = false
  renderProposeControls()
  setStatusLine('hypotheses-status', quickActivityFailureText(act, 'Proposing'), true)
  await renderHypotheses()
  pumpQueue()
}
// Poll loop: every 3s while the page is visible, read the progress record + the
// activity status; settle when the activity leaves 'running'. An orphaned run
// (server restart) gets one resume attempt, then shows as paused. A newer loop
// (another launch, another project) or navigating home cancels this one.
async function observeActivityUntilDone(activityId) {
  const session = ++observeSession
  const epoch = projectEpoch
  observing = true
  currentActivityId = activityId
  const start = Date.now()
  let resumeTries = 0
  try {
    while (Date.now() - start < MAX_OBSERVE_MS) {
      if (session !== observeSession || epoch !== projectEpoch) return 'cancelled'
      if (!document.hidden) {
        const [progress, act, campaign] = await Promise.all([
          readProgress(),
          getActivity(activityId),
          readCampaign(),
        ])
        if (session !== observeSession || epoch !== projectEpoch) return 'cancelled'
        if (progress && progress.activityId === activityId) lastProgress = progress
        if (campaign) lastCampaign = campaign
        if (act && act.status && act.status !== 'running') {
          await settleActivity(act.status)
          return act.status
        }
        if (!act || act.isLive === false) {
          if (resumeTries < 1) {
            resumeTries += 1
            try {
              await window.OverseerBridge.resumeActivity(activityId)
            } catch {
              // keep observing; the records may settle on their own
            }
          } else {
            lastActivityStatus = 'paused'
            renderActivity()
            return 'paused'
          }
        } else {
          lastActivityStatus = 'running'
        }
        renderActivity()
      }
      await sleep(POLL_MS)
    }
    return 'running'
  } finally {
    if (session === observeSession) observing = false
  }
}
// The activity settled: stamp the final status, re-read the campaign result,
// refresh the Runs tab so new run records show up, run the settle bookkeeping
// (hypothesis stamps + auto-eval markers) and let the queue dispatch its head.
async function settleActivity(status) {
  lastActivityStatus = status
  lastProgress = await readProgress()
  lastCampaign = await readCampaign()
  renderActivity()
  await renderRuns()
  await processSettledCampaignEffects()
  pumpQueue()
}
async function abortCurrentActivity() {
  if (!currentActivityId) return
  const btn = byId('activity-abort')
  if (btn) btn.disabled = true
  try {
    await window.OverseerBridge.abortActivity(currentActivityId)
  } catch {
    if (btn) btn.disabled = false
    return
  }
  if (!observing) await settleActivity('aborted')
}
async function resumeCurrentActivity() {
  if (!currentActivityId) return
  const btn = byId('activity-resume')
  if (btn) btn.disabled = true
  try {
    await window.OverseerBridge.resumeActivity(currentActivityId)
    lastActivityStatus = 'running'
    renderActivity()
    observeActivityUntilDone(currentActivityId)
  } catch {
    if (btn) btn.disabled = false
  }
}
const STATUS_META = {
  running: { label: 'Running', cls: 'is-running' },
  paused: { label: 'Paused', cls: 'is-warn' },
  completed: { label: 'Completed', cls: 'is-ok' },
  failed: { label: 'Failed', cls: 'is-bad' },
  aborted: { label: 'Aborted', cls: 'is-bad' },
}
const PHASE_LABEL = { calibrate: 'Calibrating', train: 'Training', done: 'Done' }
function progressBarHtml(progress, running, labelPrefix) {
  const done = Number(progress && progress.done) || 0
  const total = Number(progress && progress.total) || 0
  const pct = total > 0 ? Math.max(0, Math.min(100, (done / total) * 100)) : 0
  const indeterminate = running && total === 0
  return `<div class="build-progress">
    <div class="build-progress-bar"><span class="${indeterminate ? 'is-indeterminate' : ''}" style="width:${indeterminate ? '' : pct + '%'}"></span></div>
    <span class="build-progress-label">${labelPrefix ? `${escapeHtml(labelPrefix)} ` : ''}${done} / ${total || '?'}</span>
  </div>`
}
// Per-item ETA derived from the whole-campaign ETA: etaSeconds covers the
// remaining items, so one item ≈ etaSeconds / (total - done) — in ms.
function perItemEtaMs(progress) {
  const eta = Number(progress && progress.etaSeconds)
  const total = Number(progress && progress.total) || 0
  const done = Number(progress && progress.done) || 0
  const remaining = total - done
  if (!(eta > 0) || remaining <= 0) return 0
  return (eta / remaining) * 1000
}
// Time-estimated completion of the CURRENT item, animated from the progress
// record's updatedAt and capped at 95% until the next tick advances done.
// Negative → not estimable (indeterminate bar).
function estimatedItemPct(progress) {
  const per = perItemEtaMs(progress)
  if (!per) return -1
  const updated = new Date(progress.updatedAt || '').getTime()
  const elapsed = Number.isFinite(updated) ? Math.max(0, Date.now() - updated) : 0
  return Math.min(95, (elapsed / per) * 100)
}
// While running, the Activity tab shows TWO bars for multi-item campaigns: the
// time-estimated current item ("Experiment X of N") above the campaign total.
function activityProgressHtml(progress, running) {
  const total = Number(progress && progress.total) || 0
  if (!running || total < 1) return progressBarHtml(progress, running)
  const done = Number(progress && progress.done) || 0
  const itemIndex = Math.min(done + 1, total)
  const pct = estimatedItemPct(progress)
  const indeterminate = pct < 0
  const label = `Experiment ${itemIndex} of ${total}${indeterminate ? '' : ' · ~estimated'}`
  const itemBar = `<div class="build-progress">
    <div class="build-progress-bar"><span id="activity-item-bar" class="${indeterminate ? 'is-indeterminate' : ''}" style="width:${indeterminate ? '' : `${pct.toFixed(1)}%`}"></span></div>
    <span class="build-progress-label">${label}</span>
  </div>`
  if (total === 1) return itemBar
  return itemBar + progressBarHtml(progress, running, 'Total')
}
// Local 1s ticker that advances the current-item bar between the 3s polls.
function syncItemBarTimer(active) {
  if (itemBarTimer) {
    clearInterval(itemBarTimer)
    itemBarTimer = null
  }
  if (!active) return
  itemBarTimer = setInterval(() => {
    const el = byId('activity-item-bar')
    if (!el || !lastProgress) return
    const pct = estimatedItemPct(lastProgress)
    if (pct >= 0) el.style.width = `${pct.toFixed(1)}%`
  }, 1000)
}
const CURRENT_PHASE_LABEL = {
  loading: 'loading data + model…',
  starting: 'starting…',
  train: 'training',
  test: 'testing',
  summarize: 'summarizing',
}
// The within-run sub-progress carried by progress.current while a single
// experiment executes: the running item's key + phase with a spinner, an elapsed
// timer, and EITHER a real done/total bar (data-driven runs) or — since this
// run's model differs from the calibration model, making the campaign ETA
// unreliable for it — an honest indeterminate striped bar. The item-count "k of
// N" total bar above it stays accurate and is rendered separately.
function currentItemHtml(current) {
  if (!current || !current.key) return ''
  const phase = String(current.phase || '')
  const phaseLabel = CURRENT_PHASE_LABEL[phase] || phase || 'running'
  const done = Number(current.done)
  const total = Number(current.total)
  const hasCount = Number.isFinite(done) && Number.isFinite(total) && total > 0
  const pct = hasCount ? Math.max(0, Math.min(100, (done / total) * 100)) : 0
  const bar = hasCount
    ? `<div class="build-progress-bar"><span style="width:${pct.toFixed(1)}%"></span></div>`
    : '<div class="build-progress-bar"><span class="is-indeterminate"></span></div>'
  const count = hasCount ? `${done} / ${total} · ${Math.round(pct)}%` : ''
  // The elapsed value is filled by the ticker (immediately + every 1s), not baked
  // in here, so this markup stays identical across the 3s polls while the same
  // item runs — letting setHtml skip the re-render and keep the indeterminate
  // bar's animation (and the spinner) from restarting every tick.
  return `<div class="current-item">
    <p class="current-item-head">${spinnerHtml()} Run <code>${escapeHtml(shortKey(current.key))}</code> · ${escapeHtml(phaseLabel)}<span class="current-item-elapsed" id="activity-current-elapsed"></span><span class="current-item-eta" id="activity-current-eta"></span></p>
    <div class="build-progress">
      ${bar}
      ${count ? `<span class="build-progress-label">${escapeHtml(count)}</span>` : ''}
    </div>
  </div>`
}
// Drive the running item's elapsed timer (mm:ss) + a live time-left estimate from
// this run's own training progress (elapsed × remaining/done). Setup phases
// (loading/starting) show no ETA — that time is genuinely indeterminate.
function syncCurrentItemTimer(current) {
  if (currentItemTimer) {
    clearInterval(currentItemTimer)
    currentItemTimer = null
  }
  if (!current || !current.startedAt) return
  const startedAt = current.startedAt
  const done = Number(current.done)
  const total = Number(current.total)
  const determinate =
    String(current.phase) === 'train' &&
    Number.isFinite(done) &&
    Number.isFinite(total) &&
    done > 0 &&
    total > done
  const tick = () => {
    const el = byId('activity-current-elapsed')
    if (el) el.textContent = formatElapsed(startedAt)
    const etaEl = byId('activity-current-eta')
    if (etaEl) {
      if (determinate) {
        const elapsedMs = Date.now() - new Date(startedAt).getTime()
        const remainingS = (elapsedMs * ((total - done) / done)) / 1000
        etaEl.textContent = ` · ~${formatEta(remainingS)} left`
      } else {
        etaEl.textContent = ''
      }
    }
  }
  tick()
  currentItemTimer = setInterval(tick, 1000)
}
// Runs that failed across the settled campaign: a short count plus a collapsible
// per-run {key, error} list, surfaced once the campaign carries failures[].
function campaignFailuresHtml(campaign) {
  const failures = (campaign && Array.isArray(campaign.failures) && campaign.failures) || []
  if (!failures.length) return ''
  const items = failures
    .map(
      (f) =>
        `<li><code>${escapeHtml(shortKey(f.key))}</code> — ${escapeHtml(f.error || 'failed')}</li>`,
    )
    .join('')
  return `<details class="run-failures">
    <summary>${failures.length} run${failures.length === 1 ? '' : 's'} failed</summary>
    <ul class="run-failures-list">${items}</ul>
  </details>`
}
function activityCountsHtml(progress) {
  const skipped = Number(progress && progress.skipped) || 0
  const failed = Number(progress && progress.failed) || 0
  const bits = []
  if (skipped) bits.push(`<span class="badge is-warn">${skipped} skipped</span>`)
  if (failed) bits.push(`<span class="badge is-bad">${failed} failed</span>`)
  return bits.length ? `<p class="badges-row">${bits.join(' ')}</p>` : ''
}
function bestLineHtml(campaign) {
  if (!campaign || !campaign.bestKey) return ''
  return `<p class="activity-best">Best: <code>${escapeHtml(shortKey(campaign.bestKey))}</code> @ ${escapeHtml(formatObjective(campaign.bestObjective))}</p>`
}
// The persisted queue under the current activity: the item being dispatched
// (spinner) plus every waiting entry with its type chip and a remove button.
function queueSectionHtml() {
  const dispatching = activeQueueItem
    ? `<p class="queue-dispatching">${spinnerHtml()} ${escapeHtml(activeQueueItem.label || activeQueueItem.activityType)}…</p>`
    : ''
  if (!queueCache.length && !dispatching) return ''
  const rows = queueCache
    .map(
      (item) => `<li class="queue-item">
      <span class="badge queue-chip">${escapeHtml(item.activityType)}</span>
      <span class="queue-label">${escapeHtml(item.label || item.activityType)}</span>
      <button type="button" class="queue-remove" data-queue-remove="${escapeHtml(item.id)}" aria-label="Remove from queue">✕</button>
    </li>`,
    )
    .join('')
  return `<div class="queue-section">
    <h3>Queue <span class="group-count">${queueCache.length}</span></h3>
    ${dispatching}
    ${rows ? `<ul class="queue-list">${rows}</ul>` : ''}
  </div>`
}
// Activity-level parallelism control (not a per-launch field): how many runs of a
// campaign may execute at once. Persisted; applied to campaigns launched after.
function activitySettingsHtml() {
  return `<div class="activity-settings">
    <label class="field"><span>Max parallel runs ${helpCalloutHtml('How many runs of a campaign run at once. Set this BEFORE you launch — it applies to the NEXT campaign you start. It does NOT resize a campaign already running (you would relaunch to change that). The real ceiling is host CPU/GPU/RAM; default 1 = sequential.')}</span>
      <input type="number" id="activity-concurrency" min="1" step="1" value="${savedConcurrency()}" />
    </label>
  </div>`
}
function renderActivity() {
  const body = byId('activity-body')
  if (!body) return
  renderRunsLive()
  renderTabLiveIndicator()
  if (!embedded()) {
    setHtml(body, '<div class="empty-hint">Open inside the Overseer to follow campaigns.</div>')
    return
  }
  const queueHtml = queueSectionHtml()
  if (!lastProgress && !lastCampaign && !lastActivityStatus) {
    setHtml(
      body,
      activitySettingsHtml() +
        (queueHtml ||
          '<div class="empty-hint">No campaign yet — launch one from the Launch tab.</div>'),
    )
    syncItemBarTimer(false)
    syncCurrentItemTimer(null)
    return
  }
  const status = lastActivityStatus || 'completed'
  const meta = STATUS_META[status] || { label: status, cls: '' }
  const p = lastProgress
  const phase = p && PHASE_LABEL[p.phase] ? PHASE_LABEL[p.phase] : ''
  const running = status === 'running'
  const eta =
    running && p && Number(p.etaSeconds) > 0
      ? `<p class="activity-eta">ETA ~${escapeHtml(formatEta(p.etaSeconds))}</p>`
      : ''
  const times = p
    ? `<p class="card-sub">Started ${escapeHtml(formatWhen(p.startedAt))} · Updated ${escapeHtml(formatWhen(p.updatedAt))}${p.lastKey ? ` · Last run <code>${escapeHtml(shortKey(p.lastKey))}</code>` : ''}</p>`
    : ''
  const actions = running
    ? '<div class="form-actions"><button type="button" id="activity-abort" class="danger-btn">Abort</button></div>'
    : status === 'paused'
      ? '<div class="form-actions"><button type="button" id="activity-resume">Resume</button></div>'
      : ''
  // The running item's sub-progress only applies mid-run, while the campaign is
  // training (not calibrating, where the model differs from the run model).
  const current = running && p && p.phase === 'train' ? p.current : null
  const currentHtml = current ? currentItemHtml(current) : ''
  setHtml(
    body,
    `
    ${activitySettingsHtml()}
    <div class="activity-status-row">
      <span class="status-pill ${meta.cls}">${running ? `${spinnerHtml()} ` : ''}${escapeHtml(meta.label)}</span>
      ${phase ? `<span class="activity-phase">${escapeHtml(phase)}</span>` : ''}
      ${currentActivityId ? `<code class="activity-id">${escapeHtml(shortKey(currentActivityId))}</code>` : ''}
    </div>
    ${p ? activityProgressHtml(p, running) : ''}
    ${currentHtml}
    ${activityCountsHtml(p)}
    ${eta}
    ${times}
    ${bestLineHtml(lastCampaign)}
    ${campaignFailuresHtml(lastCampaign)}
    ${actions}
    ${queueHtml}`,
  )
  syncItemBarTimer(running && !!p && perItemEtaMs(p) > 0)
  syncCurrentItemTimer(currentHtml ? current : null)
}
function setupActivity() {
  const body = byId('activity-body')
  if (!body) return
  body.addEventListener('click', (event) => {
    if (event.target.closest('#activity-abort')) abortCurrentActivity()
    else if (event.target.closest('#activity-resume')) resumeCurrentActivity()
    else {
      const removeBtn = event.target.closest('button[data-queue-remove]')
      if (removeBtn) onQueueRemove(removeBtn.dataset.queueRemove)
    }
  })
  body.addEventListener('change', (event) => {
    if (event.target.id === 'activity-concurrency') rememberConcurrency(event.target.value)
  })
}

// --- Tabs --------------------------------------------------------------------
function showTab(id) {
  const target = TABS.some((t) => t.id === id) ? id : TABS[0].id
  activeTabId = target
  for (const tab of TABS) {
    const panel = byId(`tab-${tab.id}`)
    if (panel) panel.hidden = tab.id !== target
    const btn = document.querySelector(`.tab-btn[data-tab="${tab.id}"]`)
    if (btn) {
      btn.classList.toggle('is-active', tab.id === target)
      btn.setAttribute('aria-selected', String(tab.id === target))
    }
  }
  try {
    sessionStorage.setItem(ACTIVE_TAB_SS, target)
  } catch {
    // storage may be unavailable in a sandboxed frame — purely best-effort
  }
  renderTabLiveIndicator()
  if (target === 'runs') renderRuns()
  if (target === 'charts') renderCharts()
  if (target === 'hypotheses') renderHypotheses()
  if (target === 'launch') refreshLaunchRunners()
  if (target === 'activity') {
    renderActivity()
    refreshQueue()
  }
}
// A tiny spinner on the ACTIVE tab while the open project has a live (running)
// activity, so the in-flight campaign is visible from any tab — not just
// Activity. The spinner lives in a fixed `.tab-live` slot on each button so
// toggling it never disturbs the label text.
// Which tabs currently have work happening WITHIN them (so a spinner shows on the
// tab no matter where the user is): a running campaign touches Activity/Runs/Charts;
// judging writes verdicts to Runs; proposing writes Hypotheses.
function tabHasLiveWork(id) {
  const campaign = lastActivityStatus === 'running'
  if (id === 'activity') return campaign
  if (id === 'runs') return campaign || judging
  if (id === 'charts') return campaign
  if (id === 'hypotheses') return proposing
  return false
}
function renderTabLiveIndicator() {
  for (const tab of TABS) {
    const slot = document.querySelector(`.tab-btn[data-tab="${tab.id}"] .tab-live`)
    if (!slot) continue
    setHtml(slot, tabHasLiveWork(tab.id) ? spinnerHtml() : '')
  }
}
function setupTabs() {
  const bar = byId('tabbar')
  if (!bar) return
  bar.innerHTML = ''
  for (const tab of TABS) {
    const btn = document.createElement('button')
    btn.type = 'button'
    btn.className = 'tab-btn'
    btn.dataset.tab = tab.id
    btn.setAttribute('role', 'tab')
    btn.innerHTML = `<span class="tab-label">${escapeHtml(tab.label)}</span><span class="tab-live" aria-hidden="true"></span>`
    btn.addEventListener('click', () => showTab(tab.id))
    bar.append(btn)
  }
}
function savedTabId() {
  try {
    return sessionStorage.getItem(ACTIVE_TAB_SS)
  } catch {
    return null
  }
}

// --- Init ----------------------------------------------------------------------
async function init() {
  setupHome()
  setupRunners()
  setupRuns()
  setupCharts()
  setupHypotheses()
  setupLaunch()
  setupActivity()
  setupTabs()
  showView('home')
  if (!embedded()) {
    setBanner('Open inside the Overseer to use the viewer.')
    renderHome()
    return
  }
  await renderHome()
  startHomePoll()
}

init()
