// Model Trainer hub — Overseer model-training dashboards (vanilla JS, no build).
// Runs embedded in Overseer's App view. The host project is just the hub: the
// user registers training-project directories as 'trainer-project' records, a
// backend 'inspect-trainer' activity writes each project's manifest into a
// 'trainer-project-manifest' record, and opening a project shows the usual
// runs/hypotheses/launch/activity dashboard scoped to that manifest's record
// types, with the project's `dir` threaded into every train/judge/propose
// activity.

const POLL_MS = 3000
const MAX_OBSERVE_MS = 6 * 60 * 60 * 1000
const MAX_QUICK_OBSERVE_MS = 10 * 60 * 1000
const ACTIVE_TAB_SS = 'trainer.activeTab'
const PROJECT_RECORD_TYPE = 'trainer-project'
const PROJECT_MANIFEST_RECORD_TYPE = 'trainer-project-manifest'
const TABS = [
  { id: 'runs', label: 'Runs' },
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
function formatWhen(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString()
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
async function readLatestRecord(suffix) {
  if (!manifest) return null
  const recs = await queryRecords(manifest.recordType + suffix, 'latest')
  return (recs[0] && recs[0].content) || null
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

// --- Home (hub) ----------------------------------------------------------------
function projectHasManifest(projectKey) {
  const rec = manifestsCache.get(projectKey)
  return !!(rec && !rec.error && rec.manifest && rec.manifest.recordType)
}
function manifestStatusHtml(projectKey) {
  if (inspectingKeys.has(projectKey)) {
    return '<p class="project-status is-muted"><span class="spinner" aria-hidden="true"></span> Inspecting…</p>'
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
  closeRunDetail()
  toggleHypothesisForm(false)
  setStatusLine('judge-status', '')
  setStatusLine('hypotheses-status', '')
  const runsBody = byId('runs-body')
  if (runsBody) runsBody.innerHTML = ''
  const spark = byId('runs-sparkline')
  if (spark) {
    spark.innerHTML = ''
    spark.hidden = true
  }
  const hypothesesBody = byId('hypotheses-body')
  if (hypothesesBody) hypothesesBody.innerHTML = ''
  const activityBody = byId('activity-body')
  if (activityBody) activityBody.innerHTML = ''
}
async function openProject(projectKey) {
  const project = projectsCache.find((p) => p.key === projectKey)
  if (!project || !projectHasManifest(projectKey)) return
  projectEpoch += 1
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
}
function goHome() {
  projectEpoch += 1
  currentProject = null
  manifest = null
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
function runRowHtml(run) {
  const s = run.summary
  const selected = run.key === selectedRunKey ? ' class="is-selected"' : ''
  return `<tr data-key="${escapeHtml(run.key)}"${selected}>
    <td><code>${escapeHtml(shortKey(run.key))}</code></td>
    <td class="num">${escapeHtml(formatObjective(s.objective))}</td>
    <td>${healthBadgeHtml(s.health)}</td>
    <td>${verdictChipHtml(verdictsCache.get(run.key))}</td>
    <td class="num">${escapeHtml(s.seed === undefined ? '—' : String(s.seed))}</td>
    <td class="config-cell">${escapeHtml(configSummaryText(s.config))}</td>
    <td class="when-cell">${escapeHtml(formatWhen(runRanAt(s)))}</td>
  </tr>`
}
async function renderRuns() {
  const body = byId('runs-body')
  const spark = byId('runs-sparkline')
  if (!body) return
  ;[runsCache, verdictsCache, judgementSummary] = await Promise.all([
    readRuns(),
    readVerdicts(),
    readJudgement(),
  ])
  renderJudgeControls()
  if (spark) {
    const svg = sparklineSvg(runsCache)
    spark.innerHTML = svg
    spark.hidden = !svg
  }
  if (!runsCache.length) {
    body.innerHTML = '<div class="empty-hint">No runs yet — launch a campaign.</div>'
    closeRunDetail()
    return
  }
  const rows = sortRunsByObjective(runsCache).map(runRowHtml).join('')
  body.innerHTML = `<div class="table-wrap"><table class="runs-table">
    <thead><tr>
      <th>Key</th><th class="num">${escapeHtml(objectiveName())}</th><th>Health</th>
      <th>Judge</th><th class="num">Seed</th><th>Config</th><th>Ran at</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table></div>`
  if (selectedRunKey && !runsCache.some((r) => r.key === selectedRunKey)) closeRunDetail()
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
    <h3>Metrics</h3>
    ${metricsTableHtml(s.metrics)}
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
  return `<span class="spinner" aria-hidden="true"></span> ${escapeHtml(label)}`
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
  judging = true
  setStatusLine('judge-status', '')
  renderJudgeControls()
  try {
    const started = await window.OverseerBridge.startActivity('judge', trainerActivityParams())
    const activityId = started && started.activityId
    if (!activityId) throw new Error('no activity id')
    const act = await observeQuickActivity(activityId)
    if (epoch === projectEpoch) {
      setStatusLine('judge-status', quickActivityFailureText(act, 'Judging'), true)
    }
  } catch {
    if (epoch === projectEpoch) {
      setStatusLine('judge-status', 'Could not start judging — please try again.', true)
    }
  } finally {
    judging = false
    renderJudgeControls()
    if (epoch === projectEpoch) await renderRuns()
  }
}
function setupRuns() {
  const body = byId('runs-body')
  if (body) {
    body.addEventListener('click', (event) => {
      const row = event.target.closest('tr[data-key]')
      if (row) openRunDetail(row.dataset.key)
    })
  }
  const panel = byId('run-detail')
  if (panel) {
    panel.addEventListener('click', (event) => {
      if (event.target.closest('#run-detail-close')) closeRunDetail()
    })
  }
  const judgeBtn = byId('judge-btn')
  if (judgeBtn) judgeBtn.addEventListener('click', onJudgeClick)
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
  if (h.status === 'accepted') {
    return `<button type="button" data-action="run" data-id="${id}">Run campaign</button>`
  }
  if (h.status === 'rejected') {
    return `<button type="button" data-action="restore" data-id="${id}" class="ghost-btn">Restore</button>`
  }
  return `<button type="button" data-action="accept" data-id="${id}">Accept</button>
    <button type="button" data-action="reject" data-id="${id}" class="ghost-btn">Reject</button>`
}
function hypothesisCardHtml(h) {
  const source = h.source === 'llm' ? 'LLM' : 'human'
  const by = h.proposedBy ? ` · ${escapeHtml(h.proposedBy)}` : ''
  return `<article class="hypothesis-card${h.status === 'rejected' ? ' is-muted' : ''}" data-id="${escapeHtml(h.id)}">
    <h4>${escapeHtml(h.title || h.id)}</h4>
    ${h.rationale ? `<p class="hypothesis-rationale">${escapeHtml(h.rationale)}</p>` : ''}
    ${specSummaryHtml(h.spec)}
    <p class="card-sub">${escapeHtml(source)}${by} · created ${escapeHtml(formatWhen(h.createdAt))} · updated ${escapeHtml(formatWhen(h.updatedAt))}</p>
    <div class="hypothesis-actions">${hypothesisActionsHtml(h)}</div>
  </article>`
}
function hypothesisGroupHtml(status, items) {
  const label = status[0].toUpperCase() + status.slice(1)
  const cards = items.length
    ? `<div class="hypothesis-cards">${items.map(hypothesisCardHtml).join('')}</div>`
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
  ;[hypothesesCache, proposalSummary] = await Promise.all([readHypotheses(), readProposal()])
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
    hypothesisGroupHtml(status, groups[status]),
  ).join('')
}
async function onProposeClick() {
  if (proposing) return
  if (!embedded()) {
    setStatusLine('hypotheses-status', 'Open inside the Overseer to propose experiments.', false)
    return
  }
  const epoch = projectEpoch
  proposing = true
  setStatusLine('hypotheses-status', '')
  renderProposeControls()
  try {
    const started = await window.OverseerBridge.startActivity('propose', trainerActivityParams())
    const activityId = started && started.activityId
    if (!activityId) throw new Error('no activity id')
    const act = await observeQuickActivity(activityId)
    if (epoch === projectEpoch) {
      setStatusLine('hypotheses-status', quickActivityFailureText(act, 'Proposing'), true)
    }
  } catch {
    if (epoch === projectEpoch) {
      setStatusLine('hypotheses-status', 'Could not start proposing — please try again.', true)
    }
  } finally {
    proposing = false
    renderProposeControls()
    if (epoch === projectEpoch) await renderHypotheses()
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
// Launch the accepted hypothesis as a training campaign, exactly like the
// Launch tab: start 'train' with its spec, then observe from the Activity tab.
async function runHypothesisCampaign(id, button) {
  const h = hypothesesCache.find((x) => x.id === id)
  if (!h) return
  const epoch = projectEpoch
  setStatusLine('hypotheses-status', '')
  if (button) button.disabled = true
  try {
    const started = await window.OverseerBridge.startActivity(
      'train',
      trainerActivityParams({ spec: h.spec }),
    )
    const activityId = started && started.activityId
    if (!activityId) throw new Error('no activity id')
    if (epoch !== projectEpoch) return
    currentActivityId = activityId
    lastProgress = null
    lastCampaign = null
    lastActivityStatus = 'running'
    showTab('activity')
    observeActivityUntilDone(activityId)
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
  if (proposeBtn) proposeBtn.addEventListener('click', onProposeClick)
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
  if (button) button.disabled = true
  if (status) status.textContent = 'Starting campaign…'
  try {
    const started = await window.OverseerBridge.startActivity(
      'train',
      trainerActivityParams({ spec, refresh }),
    )
    const activityId = started && started.activityId
    if (!activityId) throw new Error('no activity id')
    if (epoch !== projectEpoch) return
    currentActivityId = activityId
    lastProgress = null
    lastCampaign = null
    lastActivityStatus = 'running'
    if (status) status.textContent = ''
    showTab('activity')
    observeActivityUntilDone(activityId)
  } catch {
    if (status) status.textContent = 'Could not start the campaign — please try again.'
  } finally {
    if (button) button.disabled = false
  }
}
function setupLaunch() {
  const form = byId('launch-form')
  if (!form) return
  form.addEventListener('submit', onLaunchSubmit)
  form.addEventListener('input', updateLaunchSummary)
  form.addEventListener('change', updateLaunchSummary)
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
// The activity settled: stamp the final status, re-read the campaign result, and
// always refresh the Runs tab so new run records show up.
async function settleActivity(status) {
  lastActivityStatus = status
  lastProgress = await readProgress()
  lastCampaign = await readCampaign()
  renderActivity()
  await renderRuns()
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
function progressBarHtml(progress, running) {
  const done = Number(progress && progress.done) || 0
  const total = Number(progress && progress.total) || 0
  const pct = total > 0 ? Math.max(0, Math.min(100, (done / total) * 100)) : 0
  const indeterminate = running && total === 0
  return `<div class="build-progress">
    <div class="build-progress-bar"><span class="${indeterminate ? 'is-indeterminate' : ''}" style="width:${indeterminate ? '' : pct + '%'}"></span></div>
    <span class="build-progress-label">${done} / ${total || '?'}</span>
  </div>`
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
function renderActivity() {
  const body = byId('activity-body')
  if (!body) return
  if (!embedded()) {
    body.innerHTML = '<div class="empty-hint">Open inside the Overseer to follow campaigns.</div>'
    return
  }
  if (!lastProgress && !lastCampaign && !lastActivityStatus) {
    body.innerHTML =
      '<div class="empty-hint">No campaign yet — launch one from the Launch tab.</div>'
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
      <span class="status-pill ${meta.cls}">${escapeHtml(meta.label)}</span>
      ${phase ? `<span class="activity-phase">${escapeHtml(phase)}</span>` : ''}
      ${currentActivityId ? `<code class="activity-id">${escapeHtml(shortKey(currentActivityId))}</code>` : ''}
    </div>
    ${p ? progressBarHtml(p, running) : ''}
    ${activityCountsHtml(p)}
    ${eta}
    ${times}
    ${bestLineHtml(lastCampaign)}
    ${actions}`
}
function setupActivity() {
  const body = byId('activity-body')
  if (!body) return
  body.addEventListener('click', (event) => {
    if (event.target.closest('#activity-abort')) abortCurrentActivity()
    else if (event.target.closest('#activity-resume')) resumeCurrentActivity()
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
  if (target === 'hypotheses') renderHypotheses()
  if (target === 'activity') renderActivity()
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
