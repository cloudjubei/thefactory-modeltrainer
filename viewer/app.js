// Trainer viewer — Overseer model-training dashboard (vanilla JS, no build).
// Runs embedded in Overseer's App view: reads the project's .factory/trainer.json
// manifest, lists run records via window.OverseerBridge, launches 'train'
// activities and observes them through run-scoped progress/campaign records.

const POLL_MS = 3000
const MAX_OBSERVE_MS = 6 * 60 * 60 * 1000
const ACTIVE_TAB_SS = 'trainer.activeTab'
const TABS = [
  { id: 'runs', label: 'Runs' },
  { id: 'launch', label: 'Launch' },
  { id: 'activity', label: 'Activity' },
]

let manifest = null
let activeTabId = null
let runsCache = []
let selectedRunKey = null
let currentActivityId = null
let observing = false
let lastActivityStatus = null
let lastProgress = null
let lastCampaign = null

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

// --- Manifest -----------------------------------------------------------------
function viewToken() {
  try {
    return new URLSearchParams(window.location.search).get('viewToken') || ''
  } catch {
    return ''
  }
}
async function loadManifest() {
  try {
    const res = await fetch('./.factory/trainer.json?viewToken=' + encodeURIComponent(viewToken()))
    if (!res.ok) return null
    const json = await res.json()
    if (!json || typeof json !== 'object' || !json.recordType) return null
    return json
  } catch {
    return null
  }
}
function renderNoManifest() {
  const topbar = document.querySelector('.topbar')
  if (topbar) topbar.hidden = true
  const main = byId('app-main')
  if (!main) return
  main.innerHTML = `
    <section class="card dead-state">
      <h2>Not a trainer project</h2>
      <p>This project has no <code>.factory/trainer.json</code> — not a trainer-conformant project.</p>
      <p class="card-sub">Add a manifest describing the run record type, objective and levers, then reopen this view.</p>
    </section>`
}
function applyManifestChrome() {
  const name = manifest.name || 'Trainer'
  document.title = `${name} — Trainer`
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
function runRowHtml(run) {
  const s = run.summary
  const selected = run.key === selectedRunKey ? ' class="is-selected"' : ''
  return `<tr data-key="${escapeHtml(run.key)}"${selected}>
    <td><code>${escapeHtml(shortKey(run.key))}</code></td>
    <td class="num">${escapeHtml(formatObjective(s.objective))}</td>
    <td>${healthBadgeHtml(s.health)}</td>
    <td class="num">${escapeHtml(s.seed === undefined ? '—' : String(s.seed))}</td>
    <td class="config-cell">${escapeHtml(configSummaryText(s.config))}</td>
    <td class="when-cell">${escapeHtml(formatWhen(runRanAt(s)))}</td>
  </tr>`
}
async function renderRuns() {
  const body = byId('runs-body')
  const spark = byId('runs-sparkline')
  if (!body) return
  runsCache = await readRuns()
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
      <th class="num">Seed</th><th>Config</th><th>Ran at</th>
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
  const spec = buildSpecFromForm(form)
  const refresh = !!(form.elements.refresh && form.elements.refresh.checked)
  if (button) button.disabled = true
  if (status) status.textContent = 'Starting campaign…'
  try {
    const started = await window.OverseerBridge.startActivity('train', {
      recordType: manifest.recordType,
      spec,
      refresh,
    })
    const activityId = started && started.activityId
    if (!activityId) throw new Error('no activity id')
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
// Look up one activity from the project's list (or null). The run object carries
// { status, isLive } so observers can tell a live run from an orphaned one.
async function getActivity(activityId) {
  try {
    const res = await window.OverseerBridge.listActivities()
    return ((res && res.activities) || []).find((a) => a.activityId === activityId) || null
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
// On load, re-attach to a campaign still marked running. Live → observe; orphaned
// or paused (not live) → resume it once (carfinder's resume-on-load logic), and
// fall back to a Resume button when that fails.
async function resumeRunningActivity() {
  lastProgress = await readProgress()
  lastCampaign = await readCampaign()
  const found = await findRunningActivity()
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
    lastActivityStatus = 'running'
    renderActivity()
    observeActivityUntilDone(found.activityId)
  } catch {
    lastActivityStatus = 'paused'
    renderActivity()
  }
}
// Poll loop: every 3s while the page is visible, read the progress record + the
// activity status; settle when the activity leaves 'running'. An orphaned run
// (server restart) gets one resume attempt, then shows as paused.
async function observeActivityUntilDone(activityId) {
  if (observing) return
  observing = true
  currentActivityId = activityId
  const start = Date.now()
  let resumeTries = 0
  try {
    while (Date.now() - start < MAX_OBSERVE_MS) {
      if (!document.hidden) {
        const [progress, act, campaign] = await Promise.all([
          readProgress(),
          getActivity(activityId),
          readCampaign(),
        ])
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
    observing = false
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
  let saved = null
  try {
    saved = sessionStorage.getItem(ACTIVE_TAB_SS)
  } catch {
    // storage unavailable — fall through to the default tab
  }
  showTab(saved || TABS[0].id)
}

// --- Init ----------------------------------------------------------------------
async function init() {
  manifest = await loadManifest()
  if (!manifest) {
    renderNoManifest()
    return
  }
  applyManifestChrome()
  setupRuns()
  setupLaunch()
  setupActivity()
  renderLaunchForm()
  setupTabs()
  if (!embedded()) {
    setBanner('Open inside the Overseer to use the viewer.')
    renderRuns()
    renderActivity()
    return
  }
  await renderRuns()
  await resumeRunningActivity()
}

init()
