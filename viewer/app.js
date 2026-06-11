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
const PROJECT_RECORD_TYPE = 'trainer-project'
const PROJECT_MANIFEST_RECORD_TYPE = 'trainer-project-manifest'
const QUEUE_RECORD_TYPE = 'trainer-queue'
const CHART_PALETTE = ['#3b82f6', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#14b8a6']
const JUDGE_HELP_TEXT =
  "Scores every completed run 0–100. Health-flagged runs are auto-rejected without using the LLM. For the rest, the objective is normalised (best=100) and blended 50/50 with an LLM verdict that weighs stability and how promising the configuration is — so a run can't win on prose alone. Results appear in the Judge column."
const PROPOSE_HELP_TEXT =
  "Sends the manifest's levers, the run history and the verdicts to the LLM and asks for new experiment specs likely to beat the best run. Proposals are validated against the levers, deduped by spec, and land below as pending hypotheses for you to accept or reject."
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
let chartSplits = {}
let itemBarTimer = null

// --- Utilities --------------------------------------------------------------
function escapeHtml(value) {
  return String(value ?? '').replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
  )
}
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
function byId(id) {
  return document.getElementById(id)
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
function projectCardHtml(project) {
  const key = escapeHtml(project.key)
  const armed = removeArmedKey === project.key
  return `<article class="project-card" data-key="${key}">
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
async function renderHome() {
  const body = byId('home-projects')
  if (!body) return
  if (!embedded()) {
    body.innerHTML =
      '<div class="empty-hint">Open inside the Overseer to manage training projects.</div>'
    return
  }
  ;[projectsCache, manifestsCache] = await Promise.all([readProjects(), readManifestRecords()])
  if (!projectsCache.length) {
    body.innerHTML =
      '<div class="empty-hint">No training projects yet — add one (try examples/cartpole).</div>'
    return
  }
  body.innerHTML = `<div class="project-cards">${projectsCache.map(projectCardHtml).join('')}</div>`
}
// Run the backend inspect for one registered project: it reads the directory's
// trainer manifest and writes the trainer-project-manifest record this app
// re-reads (via renderHome) once the activity settles.
async function inspectProject(projectKey, dir) {
  if (inspectingKeys.has(projectKey)) return
  inspectingKeys.add(projectKey)
  setStatusLine('projects-status', '')
  await renderHome()
  try {
    const started = await window.OverseerBridge.startActivity('inspect-trainer', {
      projectKey,
      dir,
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
      content: { key, name, dir, addedAt: nowIso() },
    })
  } catch {
    setStatusLine('project-form-error', 'Could not save the project — please try again.', true)
    return
  } finally {
    if (saveBtn) saveBtn.disabled = false
  }
  setStatusLine('project-form-error', '')
  form.reset()
  inspectProject(key, dir)
}
function onHomeProjectsClick(event) {
  const btn = event.target.closest('button[data-action]')
  if (!btn) return
  const { action, key } = btn.dataset
  if (removeArmedKey && (action !== 'remove' || key !== removeArmedKey)) removeArmedKey = null
  if (action === 'open') openProject(key)
  else if (action === 'inspect') {
    const project = projectsCache.find((p) => p.key === key)
    if (project) inspectProject(project.key, project.dir)
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
  syncItemBarTimer(false)
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
  projectEpoch += 1
  const epoch = projectEpoch
  currentProject = project
  manifest = manifestsCache.get(projectKey).manifest
  removeArmedKey = null
  resetDashboardState()
  applyManifestChrome()
  renderLaunchForm()
  showView('dashboard')
  showTab(savedTabId() || TABS[0].id)
  await renderRuns()
  await resumeRunningActivity()
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
  document.title = 'Model Trainer'
  showView('home')
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
  return extra ? { ...params, ...extra } : params
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
    params: { recordType: params.recordType, dir: params.dir },
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
function configSummaryText(config) {
  const cfg = config || {}
  const keys = Object.keys((manifest && manifest.levers) || {})
    .filter((k) => cfg[k] !== undefined)
    .slice(0, 3)
  if (!keys.length) return '—'
  return keys.map((k) => `${k}=${String(cfg[k])}`).join(' · ')
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
function runRowHtml(run) {
  const s = run.summary
  const selected = run.key === selectedRunKey ? ' class="is-selected"' : ''
  return `<tr data-key="${escapeHtml(run.key)}"${selected}>
    <td><code>${escapeHtml(shortKey(run.key))}</code></td>
    <td class="num">${escapeHtml(formatObjective(s.objective))}</td>
    <td>${healthBadgeHtml(s.health)}</td>
    <td>${verdictChipHtml(verdictsCache.get(run.key))}</td>
    <td>${evalChipHtml(run)}</td>
    <td class="num">${escapeHtml(s.seed === undefined ? '—' : String(s.seed))}</td>
    <td class="config-cell">${escapeHtml(configSummaryText(s.config))}</td>
    <td class="num">${escapeHtml(formatDuration(s.durationMs))}</td>
    <td class="when-cell">${escapeHtml(formatWhen(runRanAt(s)))}</td>
  </tr>`
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
  el.innerHTML = live ? `<span class="run-badge is-running">${spinnerHtml()} training…</span>` : ''
  el.hidden = !live
}
function runsFilterBarHtml(shownCount) {
  if (!runsFilterKeys) return ''
  const label = runsFilterLabel ? ` (${escapeHtml(runsFilterLabel)})` : ''
  return `<p class="runs-filter-bar">Showing ${shownCount} campaign run${shownCount === 1 ? '' : 's'}${label} — <button type="button" id="runs-filter-clear">clear filter</button></p>`
}
function clearRunsFilter() {
  runsFilterKeys = null
  runsFilterLabel = ''
  renderRuns()
}
async function renderRuns() {
  const body = byId('runs-body')
  const spark = byId('runs-sparkline')
  if (!body) return
  ;[runsCache, verdictsCache, judgementSummary, evaluationsCache] = await Promise.all([
    readRuns(),
    readVerdicts(),
    readJudgement(),
    readEvaluations(),
  ])
  renderJudgeControls()
  renderRunsLive()
  const shown = runsFilterKeys ? runsCache.filter((r) => runsFilterKeys.has(r.key)) : runsCache
  if (spark) {
    const svg = sparklineSvg(shown)
    spark.innerHTML = svg
    spark.hidden = !svg
  }
  if (!runsCache.length) {
    body.innerHTML = '<div class="empty-hint">No runs yet — launch a campaign.</div>'
    closeRunDetail()
    return
  }
  if (!shown.length) {
    body.innerHTML = `${runsFilterBarHtml(0)}<div class="empty-hint">No runs recorded for this campaign yet.</div>`
    closeRunDetail()
    return
  }
  const rows = sortRunsByObjective(shown).map(runRowHtml).join('')
  body.innerHTML = `${runsFilterBarHtml(shown.length)}<div class="table-wrap"><table class="runs-table">
    <thead><tr>
      <th>Key</th><th class="num">${escapeHtml(objectiveName())}</th><th>Health</th>
      <th>Judge</th><th>Eval</th><th class="num">Seed</th><th>Config</th><th class="num">Took</th><th>Ran at</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table></div>`
  if (selectedRunKey && !shown.some((r) => r.key === selectedRunKey)) closeRunDetail()
  else if (selectedRunKey) renderRunDetail(selectedRunKey)
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
  const evaluation = evaluationsCache.get(run.key)
  const button = checkpoint
    ? `<div class="form-actions"><button type="button" data-action="evaluate" data-key="${escapeHtml(run.key)}">${evaluation ? 'Re-evaluate' : 'Evaluate'}</button></div>`
    : ''
  if (!evaluation) {
    const hint = checkpoint
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
  const values =
    summary.series && Array.isArray(summary.series.episode_return)
      ? summary.series.episode_return
      : []
  const points = values
    .map((v, i) => ({ x: i, y: Number(v), label: `episode ${i} · return ${formatObjective(v)}` }))
    .filter((p) => Number.isFinite(p.y))
  if (points.length < 2) return ''
  const svg = buildLineChart({
    points,
    xLabel: 'episode',
    yLabel: 'return',
    width: 640,
    height: 180,
    markers: points.length <= 80,
    ariaLabel: 'Training curve (episode return)',
  })
  return `<h3>Training curve (episode return)</h3><div class="chart-wrap">${svg}</div>`
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
  const flags = (s.health && Array.isArray(s.health.flags) && s.health.flags) || []
  const flagChips = flags.length
    ? flags.map((f) => `<span class="badge is-bad">${escapeHtml(f)}</span>`).join(' ')
    : '<span class="card-sub">none</span>'
  const checkpoint = (s.artifacts && s.artifacts.checkpoint) || ''
  panel.innerHTML = `
    <div class="card-head card-head-row">
      <div>
        <h2>Run <code>${escapeHtml(shortKey(run.key))}</code></h2>
        <p class="card-sub">${escapeHtml(objectiveName())} ${escapeHtml(formatObjective(s.objective))}
          · ${healthBadgeHtml(s.health)} · seed ${escapeHtml(s.seed === undefined ? '—' : String(s.seed))}
          · ${escapeHtml(formatWhen(runRanAt(s)))}</p>
      </div>
      <button type="button" id="run-detail-close" class="ghost-btn">Close</button>
    </div>
    <h3>Health flags</h3>
    <p class="badges-row">${flagChips}</p>
    ${verdictSectionHtml(verdictsCache.get(run.key))}
    ${evaluationSectionHtml(run)}
    <h3>Metrics</h3>
    ${metricsTableHtml(s.metrics)}
    ${trainingCurveSectionHtml(s)}
    <h3>Config</h3>
    <pre class="json">${escapeHtml(JSON.stringify(s.config || {}, null, 2))}</pre>
    <h3>Artifacts</h3>
    <p class="mono">${checkpoint ? escapeHtml(checkpoint) : '—'}</p>
    <p class="card-sub">configHash <code>${escapeHtml(run.key)}</code></p>`
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
      trainerActivityParams({ runKey: key }),
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
      const row = event.target.closest('tr[data-key]')
      if (row) openRunDetail(row.dataset.key)
    })
  }
  const panel = byId('run-detail')
  if (panel) {
    panel.addEventListener('click', (event) => {
      if (event.target.closest('#run-detail-close')) closeRunDetail()
      const evalBtn = event.target.closest('button[data-action="evaluate"]')
      if (evalBtn) onEvaluateRun(evalBtn.dataset.key)
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
    body.innerHTML = '<div class="empty-hint">Open inside the Overseer to see charts.</div>'
    return
  }
  ;[runsCache, verdictsCache, evaluationsCache] = await Promise.all([
    readRuns(),
    readVerdicts(),
    readEvaluations(),
  ])
  body.innerHTML = [
    timelineChartSectionHtml(),
    leverChartsSectionHtml(),
    judgeChartSectionHtml(),
    trainEvalChartSectionHtml(),
  ].join('')
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
      trainerActivityParams({ spec: h.spec }),
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
function choiceLeverHtml(key, spec) {
  const choices = Array.isArray(spec.choices) ? spec.choices : []
  const fixedOptions = choices
    .map(
      (c) =>
        `<option value="${escapeHtml(String(c))}"${String(c) === String(spec.default) ? ' selected' : ''}>${escapeHtml(String(c))}</option>`,
    )
    .join('')
  const sweepOptions = choices
    .map((c) => `<option value="${escapeHtml(String(c))}">${escapeHtml(String(c))}</option>`)
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
    <legend>${escapeHtml(key)} <span class="lever-type">${escapeHtml(spec.type || '')}</span></legend>
    ${inner}
  </fieldset>`
}
function savedAutoEval() {
  try {
    return sessionStorage.getItem(AUTO_EVAL_SS) === '1'
  } catch {
    return false
  }
}
function rememberAutoEval(on) {
  try {
    sessionStorage.setItem(AUTO_EVAL_SS, on ? '1' : '')
  } catch {
    // storage may be unavailable in a sandboxed frame — purely best-effort
  }
}
function renderLaunchForm() {
  const form = byId('launch-form')
  if (!form) return
  const levers = leverEntries()
    .map(([key, spec]) => leverFieldsetHtml(key, spec))
    .join('')
  form.innerHTML = `
    ${levers || '<p class="card-sub">This manifest declares no levers — the campaign runs the default config.</p>'}
    <fieldset class="lever">
      <legend>Campaign</legend>
      <div class="lever-grid">
        <label class="field"><span>Seeds <em>(each config runs once per seed 0…N−1)</em></span>
          <input type="number" name="seeds" min="1" step="1" value="1" />
        </label>
        <label class="check-row launch-refresh">
          <input type="checkbox" name="refresh" />
          <span>Refresh — re-run configs that already have results</span>
        </label>
        <label class="check-row launch-autoeval">
          <input type="checkbox" name="autoEval"${savedAutoEval() ? ' checked' : ''} />
          <span>Auto-evaluate completed runs</span>
        </label>
      </div>
    </fieldset>
    <p class="launch-summary" id="launch-summary"></p>
    <div class="form-actions">
      <button type="submit" id="launch-btn">Launch campaign</button>
    </div>
    <p id="launch-status" class="form-status" role="status"></p>`
  updateLaunchSummary()
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
  line.textContent = `${configs} configuration${configs === 1 ? '' : 's'} × ${seeds} seed${seeds === 1 ? '' : 's'} = ${total} run${total === 1 ? '' : 's'}`
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
  if (button) button.disabled = true
  if (status) status.textContent = 'Starting campaign…'
  try {
    const params = trainerActivityParams({ spec, refresh })
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
  form.addEventListener('input', updateLaunchSummary)
  form.addEventListener('change', (event) => {
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
function renderActivity() {
  const body = byId('activity-body')
  if (!body) return
  renderRunsLive()
  if (!embedded()) {
    body.innerHTML = '<div class="empty-hint">Open inside the Overseer to follow campaigns.</div>'
    return
  }
  const queueHtml = queueSectionHtml()
  if (!lastProgress && !lastCampaign && !lastActivityStatus) {
    body.innerHTML =
      queueHtml || '<div class="empty-hint">No campaign yet — launch one from the Launch tab.</div>'
    syncItemBarTimer(false)
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
  body.innerHTML = `
    <div class="activity-status-row">
      <span class="status-pill ${meta.cls}">${running ? `${spinnerHtml()} ` : ''}${escapeHtml(meta.label)}</span>
      ${phase ? `<span class="activity-phase">${escapeHtml(phase)}</span>` : ''}
      ${currentActivityId ? `<code class="activity-id">${escapeHtml(shortKey(currentActivityId))}</code>` : ''}
    </div>
    ${p ? activityProgressHtml(p, running) : ''}
    ${activityCountsHtml(p)}
    ${eta}
    ${times}
    ${bestLineHtml(lastCampaign)}
    ${actions}
    ${queueHtml}`
  syncItemBarTimer(running && !!p && perItemEtaMs(p) > 0)
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
  if (target === 'runs') renderRuns()
  if (target === 'charts') renderCharts()
  if (target === 'hypotheses') renderHypotheses()
  if (target === 'activity') {
    renderActivity()
    refreshQueue()
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
    btn.textContent = tab.label
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
}

init()
