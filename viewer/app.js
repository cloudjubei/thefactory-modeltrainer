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
// How long an activity must read as not-live (controller gone) AND show no progress
// advancement before we treat it as genuinely dead. `isLive` is an in-memory flag with
// no heartbeat, so a momentary backend blip (a slow GET, a dev reload) can flip it for a
// poll or two while the campaign is still training — pausing on that strands a live run.
const DEAD_CONFIRM_MS = 45000
const MAX_QUICK_OBSERVE_MS = 10 * 60 * 1000
const ACTIVE_TAB_SS = 'trainer.activeTab'
const AUTO_EVAL_SS = 'trainer.autoEval'
const COMPUTE_TARGET_SS = 'trainer.computeTarget'
const CONCURRENCY_SS = 'trainer.concurrency'
const ACTIVITY_BUDGET_SS = 'trainer.activityBudget'
const DEFAULT_ACTIVITY_BUDGET = 3
const PROJECT_RECORD_TYPE = 'trainer-project'
const PROJECT_MANIFEST_RECORD_TYPE = 'trainer-project-manifest'
const ENVIRONMENT_RECORD_SUFFIX = '-environment'
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
  { id: 'hypotheses', label: 'Hypotheses' },
  { id: 'versions', label: 'Versions' },
  { id: 'environments', label: 'Environments' },
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
// Run keys the user dismissed from the Activity failures list (persisted so a
// reload doesn't resurface them); the failure record itself is left intact.
let dismissedFailures = new Set()
// setupKeys the user marked UNRUNNABLE for the current pipeline version — skipped
// on re-run (alongside skipExplored) unless force-rerun or a version bump clears them.
let unrunnableCache = new Set()
const evaluatingKeys = new Set()
let judgementSummary = null
let hypothesesCache = []
let proposalSummary = null
let selectedRunKey = null
// Activities run CONCURRENTLY up to a budget. Each live activity (campaign / judge /
// propose / evaluate) is tracked here by id with its own observed state; the Activity tab
// renders one block per entry. Shape: { activityId, activityType, label, status, progress,
// campaign, startedAt, session, item }.
let liveActivities = new Map()
let activitySession = 0
// The most-recently-settled campaign, kept only so the Activity tab can show its result
// summary when nothing is live (its runs are in the Runs tab regardless).
let lastSettledCampaign = null
let judging = false
let proposing = false
let queueCache = []
// In-memory double-dispatch guard for the persisted queue: only one pump loop drains it
// at a time (it dispatches up to the budget, then returns).
let queuePumping = false
let runsFilterKeys = null
let runsFilterLabel = ''
let runsSortKey = null
let runsSortDir = 'desc'
let runsLeverFilter = {}
let runsTextFilter = ''
let runsHideBad = false
let runsVersionFilter = ''
let runsCompareKeys = new Set()
// First click on Delete arms it (in-app confirm — window.confirm is blocked in the
// embedding iframe sandbox); the second click within the timeout performs the delete.
let runsDeleteArmed = false
let runsDeleteArmTimer = null
let runsViewMode = 'runs'
// Named environments (env-lever bundles) the user defined for this project.
let environmentsCache = []
// When drilled into a single setup's runs (via the by-setup view), this holds that
// setup's key so the runs view can show its conclusion-note editor (C4 ledger).
let runsDrillSetupKey = null
// 1s ticker for the in-flight runs' elapsed timers (mm:ss) between the 3s polls.
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
// The app-wide delete/trash icon (mirrors thefactory-ui's IconDelete) for icon-only
// delete buttons — used instead of an emoji so the viewer matches the rest of the app.
function iconDeleteSvg() {
  return (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<polyline points="3 6 5 6 21 6" stroke="#6366F1" stroke-width="2"/>' +
    '<path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" stroke="#EF4444" stroke-width="2"/>' +
    '<path d="M10 11v6" stroke="#A855F7" stroke-width="2"/>' +
    '<path d="M14 11v6" stroke="#A855F7" stroke-width="2"/>' +
    '<path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" stroke="#F59E0B" stroke-width="2"/>' +
    '</svg>'
  )
}
function iconCheckSvg() {
  return (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<polyline points="20 6 9 17 4 12"/></svg>'
  )
}
function iconCrossSvg() {
  return (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>'
  )
}
function iconChatSvg() {
  return (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M21 11.5a8.38 8.38 0 0 1-8.5 8.5 8.5 8.5 0 0 1-3.9-.9L3 21l1.9-5.6a8.5 8.5 0 0 1-.9-3.9 8.38 8.38 0 0 1 8.5-8.5 8.38 8.38 0 0 1 8.5 8.5z"/></svg>'
  )
}
// A balance-scale glyph for the Judge button (the LLM weighs runs against each other).
function iconJudgeSvg() {
  return (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M12 3v18"/>' +
    '<path d="M7 21h10"/>' +
    '<path d="M5 6h14"/>' +
    '<path d="M5 6l-3 6a3 3 0 0 0 6 0z"/>' +
    '<path d="M19 6l-3 6a3 3 0 0 0 6 0z"/>' +
    '</svg>'
  )
}
// Small circular "?" button with a styled hover/focus callout (no native title).
// Help is shown by HOVERING (or focusing) the item it describes — no "?" buttons. Any
// element carrying a `data-help` attribute is a trigger; `helpAttr(text)` produces that
// attribute fragment to splice into the item's opening tag.
function helpAttr(text) {
  return text ? ` data-help="${escapeHtml(text)}"` : ''
}
let helpTooltipEl = null
let helpTooltipFor = null
function ensureHelpTooltip() {
  if (helpTooltipEl) return helpTooltipEl
  helpTooltipEl = document.createElement('div')
  helpTooltipEl.className = 'app-tooltip'
  helpTooltipEl.setAttribute('role', 'tooltip')
  helpTooltipEl.hidden = true
  document.body.appendChild(helpTooltipEl)
  return helpTooltipEl
}
// Position a single fixed tooltip next to its trigger, CLAMPED to the viewport so it is
// never cut off near an edge or by a scroll container (prefers below, flips above).
function positionHelpTooltip(target) {
  const tip = helpTooltipEl
  if (!tip || tip.hidden) return
  const r = target.getBoundingClientRect()
  const margin = 8
  const tw = tip.offsetWidth
  const th = tip.offsetHeight
  let top = r.bottom + 6
  if (top + th + margin > window.innerHeight) top = r.top - th - 6
  if (top < margin) top = margin
  let left = r.left + r.width / 2 - tw / 2
  left = Math.max(margin, Math.min(left, window.innerWidth - tw - margin))
  tip.style.left = `${Math.round(left)}px`
  tip.style.top = `${Math.round(top)}px`
}
function showHelpTooltip(target) {
  const text = target.getAttribute('data-help')
  if (!text) return
  const tip = ensureHelpTooltip()
  helpTooltipFor = target
  tip.textContent = text
  tip.hidden = false
  positionHelpTooltip(target)
}
function hideHelpTooltip() {
  helpTooltipFor = null
  if (helpTooltipEl) helpTooltipEl.hidden = true
}
function setupHelpTooltips() {
  document.addEventListener('mouseover', (e) => {
    const t = e.target.closest && e.target.closest('[data-help]')
    if (t && t !== helpTooltipFor) showHelpTooltip(t)
  })
  document.addEventListener('mouseout', (e) => {
    if (!helpTooltipFor) return
    if (e.relatedTarget && helpTooltipFor.contains(e.relatedTarget)) return
    if (e.target.closest && e.target.closest('[data-help]') === helpTooltipFor) hideHelpTooltip()
  })
  document.addEventListener('focusin', (e) => {
    const t = e.target.closest && e.target.closest('[data-help]')
    if (t) showHelpTooltip(t)
  })
  document.addEventListener('focusout', hideHelpTooltip)
  // A scroll moves the trigger out from under a fixed tooltip — just hide it; it reappears
  // on the next hover, already clamped.
  window.addEventListener('scroll', hideHelpTooltip, true)
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
// Failures the user dismissed from the Activity list, persisted as one record per
// dismissed run key so the dismissal survives reloads + other clients.
async function readDismissedFailures() {
  if (!manifest) return new Set()
  const recs = await queryRecords(manifest.recordType + '-dismissed-failure')
  const set = new Set()
  for (const r of recs) {
    const key = r.key || (r.content && r.content.runKey) || ''
    if (key) set.add(key)
  }
  return set
}
async function dismissFailure(key) {
  if (!manifest || !key) return
  dismissedFailures.add(key)
  renderActivity()
  await window.OverseerBridge.putData({
    type: manifest.recordType + '-dismissed-failure',
    key,
    content: { runKey: key, dismissedAt: nowIso() },
  })
}
// The pipeline version a fresh run would carry; unrunnable marks + explored-skips
// are scoped to it, so a version bump re-opens everything.
function currentPipelineVersion() {
  return (manifest && manifest.pipelineVersion) || '1'
}
// setupKeys marked unrunnable for the CURRENT pipeline version (older-version marks
// are ignored — a breaking change re-opens the setup).
async function readUnrunnable() {
  if (!manifest) return new Set()
  const recs = await queryRecords(manifest.recordType + '-unrunnable')
  const version = currentPipelineVersion()
  const set = new Set()
  for (const r of recs) {
    const c = r.content || {}
    const key = r.key || c.setupKey || ''
    if (key && c.unrunnable !== false && (c.pipelineVersion || '1') === version) set.add(key)
  }
  return set
}
function setupKeyForRun(run) {
  return (run && run.summary && run.summary.setupKey) || setupKeyOfRun(run)
}
async function toggleUnrunnable(runKey) {
  if (!manifest || !runKey) return
  const run = runsCache.find((r) => r.key === runKey)
  if (!run) return
  const setupKey = setupKeyForRun(run)
  if (!setupKey) return
  const nowUnrunnable = !unrunnableCache.has(setupKey)
  if (nowUnrunnable) unrunnableCache.add(setupKey)
  else unrunnableCache.delete(setupKey)
  if (selectedRunKey === runKey) renderRunDetail(runKey)
  await window.OverseerBridge.putData({
    type: manifest.recordType + '-unrunnable',
    key: setupKey,
    content: {
      setupKey,
      pipelineVersion: currentPipelineVersion(),
      unrunnable: nowUnrunnable,
      markedAt: nowIso(),
    },
  })
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
// On opening a project, refresh its manifest record from disk in the background (the
// `inspect-trainer` activity re-reads trainer.json) so an edit — e.g. dropping the
// `evaluate` command for an RL project — takes effect without a manual Re-inspect.
// Re-applies + re-renders only if the manifest actually changed.
async function refreshProjectManifest(project) {
  if (!embedded() || !project || !window.OverseerBridge) return
  const epoch = projectEpoch
  try {
    const started = await window.OverseerBridge.startActivity('inspect-trainer', {
      projectKey: project.key,
      dir: project.dir,
      ...(project.manifestRelPath ? { manifestRelPath: project.manifestRelPath } : {}),
    })
    const activityId = started && started.activityId
    if (!activityId) return
    await observeQuickActivity(activityId)
    if (epoch !== projectEpoch || !currentProject || currentProject.key !== project.key) return
    manifestsCache = await readManifestRecords()
    const rec = manifestsCache.get(project.key)
    if (!rec || !rec.manifest) return
    if (JSON.stringify(rec.manifest) === JSON.stringify(manifest)) return
    manifest = rec.manifest
    applyManifestChrome()
    renderLaunchForm()
    if (activeTabId === 'runs') renderRunsTable()
  } catch {
    // best-effort — a stale manifest just persists until the next Re-inspect
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
  liveActivities = new Map()
  lastSettledCampaign = null
  judging = false
  proposing = false
  queueCache = []
  runsFilterKeys = null
  runsFilterLabel = ''
  launchRunnersCache = []
  syncInFlightTimer([])
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
  const versionsBody = byId('versions-body')
  if (versionsBody) versionsBody.innerHTML = ''
  const environmentsBody = byId('environments-body')
  if (environmentsBody) environmentsBody.innerHTML = ''
  environmentsCache = []
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
  // Load saved environments before the launch form so its environment picker is populated.
  environmentsCache = hasEnvLevers() ? await readEnvironments() : []
  if (epoch !== projectEpoch) return
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
  // Pick up any edit to the project's trainer.json (e.g. removing the `evaluate` command)
  // without a manual Re-inspect — refreshes the cached manifest record in the background.
  void refreshProjectManifest(project)
}
function goHome() {
  projectEpoch += 1
  currentProject = null
  manifest = null
  syncInFlightTimer([])
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
// Trainer activities run CONCURRENTLY up to a budget: startOrEnqueue starts one if a slot is
// free, else parks it in the 'trainer-queue' records, and pumpQueue dispatches queued items
// as slots free up. Each activity observes itself; the Activity tab renders one block each.
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
// How many compute slots are in use — only RUNNING/starting activities count (a paused or
// stalled-waiting-to-resume entry isn't consuming compute, so it shouldn't block new launches).
function liveSlotCount() {
  let n = 0
  for (const a of liveActivities.values())
    if (a.status === 'running' || a.status === 'starting') n++
  return n
}
function anyActivityRunning() {
  for (const a of liveActivities.values())
    if (a.status === 'running' || a.status === 'starting') return true
  return false
}
// Start now if a slot is free (up to the activity budget), else enqueue. The caller only
// handles the queued message + its own status line — launchActivity owns dispatch + observe.
async function startOrEnqueue(activityType, params, label, extra) {
  const item = {
    id: randomHexId(),
    activityType,
    params,
    label,
    queuedAt: nowIso(),
    ...(extra || {}),
  }
  if (liveSlotCount() < savedActivityBudget()) {
    const activityId = await launchActivity(item)
    if (activityId) return { started: true, activityId }
  }
  await putQueueItem(item)
  await refreshQueue()
  const queue = await readQueue()
  return { queued: true, id: item.id, ahead: queue.length }
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
// Drain the queue until the activity budget is full. Each dispatched activity observes
// itself (non-blocking) and re-pumps when it settles, freeing its slot for the next item.
async function pumpQueue() {
  if (queuePumping || !embedded() || !manifest) return
  const epoch = projectEpoch
  queuePumping = true
  try {
    while (epoch === projectEpoch && liveSlotCount() < savedActivityBudget()) {
      const queue = await readQueue()
      if (epoch !== projectEpoch || !queue.length) break
      const head = queue[0]
      await deleteQueueItem(head.id)
      queueCache = queueCache.filter((q) => q.id !== head.id)
      const activityId = await launchActivity(head)
      if (!activityId) {
        // Transient backend failure — put it back and stop draining so we don't lose the
        // whole queue; a later pump (settle / focus) retries.
        await putQueueItem(head)
        queueCache = await readQueue()
        break
      }
    }
  } finally {
    queuePumping = false
  }
}
// Start one activity and observe it CONCURRENTLY. The entry is registered immediately (so
// the budget + Activity tab reflect it before the backend confirms), then its own observe
// loop runs and untracks + re-pumps on settle.
async function launchActivity(item) {
  const localId = item.id || randomHexId()
  const entry = {
    localId,
    activityId: null,
    activityType: item.activityType,
    label: item.label || item.activityType,
    status: 'starting',
    progress: null,
    campaign: null,
    startedAt: Date.now(),
    item,
  }
  liveActivities.set(localId, entry)
  renderActivity()
  try {
    const started = await window.OverseerBridge.startActivity(item.activityType, item.params)
    entry.activityId = started && started.activityId
    if (!entry.activityId) throw new Error('no activity id')
  } catch {
    liveActivities.delete(localId)
    renderActivity()
    return null
  }
  entry.status = 'running'
  // Best-effort bookkeeping: a marker/stamp write failure must NOT strand the entry (slot
  // leak → stalled queue) or skip the observe wiring (a live campaign that never settles).
  try {
    if (item.autoEval) await putAutoEvalMarker(entry.activityId, item.params)
    if (item.hypothesisId && item.params && item.params.recordType) {
      await stampHypothesisCampaign(item.params.recordType, item.hypothesisId, {
        activityId: entry.activityId,
        launchedAt: nowIso(),
        status: 'running',
      })
    }
  } catch {
    // ignore — the campaign is running; the marker/stamp are non-critical
  }
  renderActivity()
  void observeActivity(entry)
  return entry.activityId
}
// Re-attach an observe loop to an ALREADY-RUNNING backend activity (on reload / resume) by
// registering a tracking entry for it and observing — without starting anything new.
function trackExistingActivity(activityId, activityType, label) {
  for (const a of liveActivities.values()) if (a.activityId === activityId) return a
  const entry = {
    localId: randomHexId(),
    activityId,
    activityType,
    label: label || activityType,
    status: 'running',
    progress: null,
    campaign: null,
    startedAt: Date.now(),
    item: { activityType, label },
  }
  liveActivities.set(entry.localId, entry)
  void observeActivity(entry)
  return entry
}
// An evaluate item carries either a batch of `runKeys` (parallel) or a single
// legacy `runKey`; both collapse to the list of keys it evaluates.
function evaluateKeysOf(item) {
  const params = (item && item.params) || {}
  if (Array.isArray(params.runKeys)) return params.runKeys.filter((k) => typeof k === 'string' && k)
  return params.runKey ? [params.runKey] : []
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
    const keys = evaluateKeysOf(item)
    if (!keys.length) return
    for (const key of keys) {
      if (busy) evaluatingKeys.add(key)
      else evaluatingKeys.delete(key)
    }
    if (selectedRunKey && keys.includes(selectedRunKey)) renderRunDetail(selectedRunKey)
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
      if (!campaign.aborted) await enqueueMissingEvaluations(campaign.keys, marker.params)
      await deleteQueueItem(marker.id)
      continue
    }
    const act = await getActivity(marker.activityId)
    if (act && (act.status === 'running' || act.status === 'paused')) continue
    // Settled, but a concurrent campaign overwrote the single 'latest' campaign record, so we
    // can't read this one's keys — fall back to evaluating every completed run missing an eval.
    if (act && act.status === 'completed') await enqueueMissingEvaluations(null, marker.params)
    await deleteQueueItem(marker.id)
  }
  await refreshQueue()
}
// Queue an evaluate for each completed run still missing one. `keys` scopes it to a
// campaign's runs; when null (its campaign record was overwritten by a concurrent campaign)
// it falls back to ALL completed runs missing an eval, so auto-eval is never silently lost.
async function enqueueMissingEvaluations(keys, baseParams) {
  if (!evalEnabled()) return
  const [runs, evaluations, queue] = await Promise.all([readRuns(), readEvaluations(), readQueue()])
  const scope = Array.isArray(keys) && keys.length ? keys : runs.map((r) => r.key)
  const queuedKeys = new Set(
    queue.filter((q) => q.activityType === 'evaluate').flatMap(evaluateKeysOf),
  )
  const pending = []
  for (const key of scope) {
    const run = runs.find((r) => r.key === key)
    if (!run) continue
    const s = run.summary
    if (s.status && s.status !== 'completed') continue
    // Degenerate runs (zero/few trades, NaN, etc.) have no result worth re-testing — auto-eval skips them.
    if (runIsDegenerate(run)) continue
    if (!(s.artifacts && s.artifacts.checkpoint)) continue
    if (evaluations.has(key) || queuedKeys.has(key) || evaluatingKeys.has(key)) continue
    pending.push(key)
  }
  if (!pending.length) return
  // One batch activity re-tests every checkpoint in parallel (a bounded pool),
  // instead of one serial activity per run.
  const concurrency = savedConcurrency()
  await putQueueItem({
    id: randomHexId(),
    activityType: 'evaluate',
    params: { ...baseParams, runKeys: pending, concurrency },
    label:
      pending.length === 1 ? `Evaluate ${shortKey(pending[0])}` : `Evaluate ${pending.length} runs`,
    queuedAt: nowIso(),
  })
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
  let training = false
  for (const a of liveActivities.values()) {
    if (a.activityType === 'train' && (a.status === 'running' || a.status === 'starting'))
      training = true
  }
  setHtml(
    el,
    training ? `<span class="run-badge is-running">${spinnerHtml()} training…</span>` : '',
  )
  el.hidden = !training
}
function clearRunsFilter() {
  runsFilterKeys = null
  runsFilterLabel = ''
  runsLeverFilter = {}
  runsTextFilter = ''
  runsVersionFilter = ''
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
// Drill from a by-environment row into that environment's runs.
function drillIntoEnvironment(sig) {
  const group = aggregateByEnvironment(applyRunsFilters(runsCache)).find((g) => g.sig === sig)
  if (!group) return
  runsFilterKeys = new Set(group.runs.map((r) => r.key))
  runsFilterLabel = group.name
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
// %return + #trades lead (the two the user reads first); both are colour-coded like
// vs-hold (see metricColorClass). The rest follow; unknown metrics come after, sorted.
const RUN_METRIC_ORDER = [
  'total_return_pct',
  'n_trades',
  'win_pct',
  'sharpe',
  'cagr_pct',
  'max_drawdown_pct',
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
// Threshold below which a trade count reads as degenerate (≈ buy-and-hold); mirrors
// summary.py's DEGENERATE_TRADE_COUNT. Used for the #trades colour + the bad-run filter.
const DEGENERATE_TRADE_COUNT = 2
// vs-hold-style green/red for the lead metrics: %return by sign, #trades by whether the
// run actually traded (more than a near-hold count).
function metricColorClass(mk, v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return ''
  if (mk === 'total_return_pct') return v >= 0 ? 'delta-pos' : 'delta-neg'
  if (mk === 'n_trades') return v > DEGENERATE_TRADE_COUNT ? 'delta-pos' : 'delta-neg'
  return ''
}
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
// Short column headers for the runs TABLE (the detail view keeps the raw metric key).
const METRIC_LABEL = {
  total_return_pct: 'return %',
  n_trades: '#trades',
  win_pct: 'win %',
  stop_losses: 'SLs',
  final_net_worth: 'Final $',
  max_drawdown_pct: 'drawdown',
}
// Metrics shown to a fixed 2 decimals in the table (money + ratios read cleaner that way).
const TWO_DP_METRICS = new Set([
  'win_pct',
  'final_net_worth',
  'sharpe',
  'cagr_pct',
  'max_drawdown_pct',
])
// Metrics surfaced only in a single run's DETAIL (too granular for the table); kept
// out of the table's metric columns but still rendered by metricsTableHtml.
const TABLE_HIDDEN_METRICS = new Set([
  'trade_gate',
  'traded_return',
  'windows_profitable_pct',
  'worst_window_return_pct',
])
function metricLabel(mk) {
  return METRIC_LABEL[mk] || mk.replace(/_pct$/, ' %').replace(/_/g, ' ')
}
// Some metrics read cleaner at a fixed 2dp (win %, money, ratios); the rest use the
// objective formatter.
function formatMetricValue(mk, v) {
  if (TWO_DP_METRICS.has(mk)) return v.toFixed(2)
  return formatObjective(v)
}
function runMetricKeys() {
  const keys = new Set()
  for (const r of runsCache) {
    const m = r.summary && r.summary.metrics
    if (m && typeof m === 'object') for (const k of Object.keys(m)) keys.add(k)
  }
  const known = RUN_METRIC_ORDER.filter((k) => keys.has(k) && !TABLE_HIDDEN_METRICS.has(k))
  const rest = [...keys]
    .filter((k) => !RUN_METRIC_ORDER.includes(k) && !TABLE_HIDDEN_METRICS.has(k))
    .sort()
  return [...known, ...rest]
}
// A status DOT (no text) for the Run column — green healthy / amber degenerate or
// health-flagged / red failed / grey unknown. The Run header's "?" explains it.
function statusDotHtml(s) {
  let cls = 'is-ok'
  let label = 'healthy'
  if (s.status === 'failed') {
    cls = 'is-bad'
    label = 'failed'
  } else if (s.health && s.health.status && s.health.status !== 'ok') {
    cls = 'is-warn'
    label = `health: ${s.health.status}`
  } else if (!s.health || !s.health.status) {
    cls = 'is-unknown'
    label = 'unknown'
  }
  return `<span class="run-status-dot ${cls}" title="${escapeHtml(label)}" aria-label="status: ${escapeHtml(label)}"></span>`
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
// Whether this project supports re-testing a saved checkpoint ("Eval"). The switch is the
// manifest declaring an `evaluate` command: classification/regression projects do; RL
// projects don't (the real test is a live environment/market), so all Eval UI hides.
function evalEnabled() {
  return !!(manifest && manifest.evaluate)
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
      help: 'The dot is run status: green = healthy · amber = degenerate / health-flagged (e.g. too few trades) · red = failed · grey = unknown. The code is the run id (config hash).',
      get: (r) => `${statusDotHtml(r.summary)}<code>${escapeHtml(shortKey(r.key))}</code>`,
      sort: (r) => r.key,
    },
    {
      id: 'data',
      label: 'Data',
      num: false,
      get: (r) => escapeHtml(datasetLabel(r.summary)),
      sort: (r) => datasetLabel(r.summary),
    },
    {
      id: 'version',
      label: 'v',
      num: false,
      help: 'Pipeline version this run ran under. Runs from different versions are NOT comparable (a breaking version changed how data is fed/scored). Filter by it to clear out old runs.',
      get: (r) => escapeHtml(String((r.summary && r.summary.pipelineVersion) || '1')),
      sort: (r) => String((r.summary && r.summary.pipelineVersion) || '1'),
    },
  ]
  for (const mk of runMetricKeys()) {
    cols.push({
      id: 'm:' + mk,
      label: metricLabel(mk),
      num: true,
      help: METRIC_INFO[mk],
      get: (r) => {
        const v = r.summary.metrics && r.summary.metrics[mk]
        if (typeof v !== 'number') return escapeHtml(v === undefined ? '—' : String(v))
        const cls = metricColorClass(mk, v)
        const text = formatMetricValue(mk, v)
        return cls ? `<span class="${cls}">${escapeHtml(text)}</span>` : escapeHtml(text)
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
  cols.push({
    id: 'judge',
    label: 'Judge',
    num: true,
    help: "The LLM judge's blended score (0–100) for this run — objective rank + the model's qualitative read of the config/metrics. Run “Judge” on selected runs to fill it. “—” = not judged.",
    get: (r) => verdictChipHtml(verdictsCache.get(r.key)),
    sort: (r) => {
      const v = verdictsCache.get(r.key)
      return v ? Number(v.score) : NaN
    },
  })
  // Eval (re-test a saved checkpoint) only applies to projects that declare an `evaluate`
  // command — RL projects don't (you test on the live environment/market), so the column,
  // detail section and auto-eval option all disappear for them.
  if (evalEnabled()) {
    cols.push({
      id: 'eval',
      label: 'Eval',
      num: true,
      help: 'Re-tests the run’s saved checkpoint (no retraining). green = held up, amber = came back worse.',
      get: (r) => evalChipHtml(r),
      sort: (r) => {
        const e = evaluationsCache.get(r.key)
        return e ? Number(e.objective) : NaN
      },
    })
  }
  cols.push(
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
// A run is "bad" — failed/errored, or a degenerate result (health-flagged: zero/few
// trades ≤ DEGENERATE_TRADE_COUNT, degenerate policy, NaN). The Hide-bad toggle drops these.
function runIsBad(r) {
  const s = r.summary || {}
  if (s.status === 'failed') return true
  if (runIsDegenerate(r)) return true
  const n = s.metrics && Number(s.metrics.n_trades)
  return Number.isFinite(n) && n <= DEGENERATE_TRADE_COUNT
}
function runVersionOf(r) {
  return String((r.summary && r.summary.pipelineVersion) || '1')
}
function applyRunsFilters(runs) {
  let out = runsFilterKeys ? runs.filter((r) => runsFilterKeys.has(r.key)) : runs
  if (runsHideBad) out = out.filter((r) => !runIsBad(r))
  if (runsVersionFilter) out = out.filter((r) => runVersionOf(r) === runsVersionFilter)
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
// A run is degenerate when its health is flagged (zero/few trades, degenerate policy, NaN metrics):
// its objective is a fluke, not a real test of the setup, so it must not be averaged in blind or
// chosen as a setup's representative "best".
function runIsDegenerate(r) {
  const h = r && r.summary && r.summary.health
  return !!(h && h.status && h.status !== 'ok')
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
    const healthy = rs.filter((r) => !runIsDegenerate(r))
    const degenerateCount = rs.length - healthy.length
    // Aggregate + pick the representative best from HEALTHY runs only, so a setup's headline
    // numbers + its surfaced verdict/conclusion reflect real trades, not a lucky degenerate run.
    const scored = healthy.length ? healthy : []
    const objs = scored.map((r) => Number(r.summary.objective)).filter(Number.isFinite)
    const vsh = scored.map((r) => vsHoldValue(r.summary)).filter(Number.isFinite)
    let bestRun = null
    for (const r of scored) {
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
      degenerateCount,
      allDegenerate: rs.length > 0 && healthy.length === 0,
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
// The project's pipeline version + changelog. A BREAKING version changed how data is
// fed/scored, so runs across versions are NOT comparable; each run is tagged with the
// version it ran under and a bump re-opens skipExplored/unrunnable. Shown above the runs.
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
  const active =
    runsFilterKeys ||
    runsTextFilter ||
    runsVersionFilter ||
    Object.values(runsLeverFilter).some(Boolean)
  const label = runsFilterLabel ? ` (${escapeHtml(runsFilterLabel)})` : ''
  const envViewBtn = hasEnvLevers()
    ? `<button type="button" class="runs-view-btn${runsViewMode === 'environment' ? ' is-active' : ''}" data-view="environment"${helpAttr('Group runs by the ENVIRONMENT they ran in (fee / TP-SL regime), so you can see how a model holds up across regimes.')}>By environment</button>`
    : ''
  const toggle = `<div class="runs-viewmode">
    <button type="button" class="runs-view-btn${runsViewMode === 'runs' ? ' is-active' : ''}" data-view="runs">Runs</button><button type="button" class="runs-view-btn${runsViewMode === 'setup' ? ' is-active' : ''}" data-view="setup"${helpAttr('Group runs by SETUP (config ignoring seed) and show the spread across seeds — what a setup concluded, not one lucky run.')}>By setup</button><button type="button" class="runs-view-btn${runsViewMode === 'experiment' ? ' is-active' : ''}" data-view="experiment"${helpAttr('Group runs by the THESIS set at launch, so experiments compare head-to-head (incl. theses outside the levers).')}>By experiment</button>${envViewBtn}
  </div>`
  const hideBad = `<label class="runs-hidebad" title="Hide failed/errored runs and degenerate results (≤${DEGENERATE_TRADE_COUNT} trades or health-flagged).">
    <input type="checkbox" id="runs-hide-bad"${runsHideBad ? ' checked' : ''} /> Hide bad runs
  </label>`
  const versions = [
    ...new Set([
      ...runsCache.map(runVersionOf),
      String((manifest && manifest.pipelineVersion) || '1'),
    ]),
  ].sort()
  const versionFilter = `<select class="runs-filter-lever" id="runs-version-filter"${helpAttr("Show only runs from one pipeline version — cross-version scores aren't comparable. Set automatically when you open a version from the Versions tab.")}>
          <option value="">version: any</option>
          ${versions.map((v) => `<option value="${escapeHtml(v)}"${runsVersionFilter === v ? ' selected' : ''}>v${escapeHtml(v)}</option>`).join('')}
        </select>`
  return `<div class="runs-toolbar">
    ${toggle}
    ${versionFilter}
    ${dropdowns}
    <input type="search" id="runs-filter-text" class="runs-filter-text" placeholder="filter config / key…" value="${escapeHtml(runsTextFilter)}" />
    ${hideBad}
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
      const degen = g.degenerateCount
        ? ` <span class="card-sub" title="Health-flagged runs (zero/few trades, degenerate policy, NaN) — excluded from this setup's averages, best run and verdict.">(${g.degenerateCount} degenerate)</span>`
        : ''
      const llm = setupLlmNote(g)
      const llmCell = g.allDegenerate
        ? '<span class="card-sub" title="Every run for this setup was health-flagged; no real result to judge.">all runs degenerate</span>'
        : llm
          ? escapeHtml(truncate(llm, 70))
          : '<span class="card-sub">—</span>'
      const note = (notesCache.get(g.key) || {}).note || ''
      return `<tr data-setup-key="${escapeHtml(g.key)}" class="setup-row${g.unstable ? ' is-unstable' : ''}${g.allDegenerate ? ' is-degenerate-row' : ''}">
        <td>${escapeHtml(g.label)}</td>
        <td class="num">${g.count}${failed}${degen}</td>
        <td class="num">${setupStabilityCell(g)}</td>
        <td class="num">${escapeHtml(formatObjective(g.objAvg))}</td>
        <td class="num">${range}</td>
        <td class="num">${escapeHtml(formatObjective(g.objMedian))}</td>
        <td class="num">${vsh}</td>
        <td class="ledger-note" title="${escapeHtml(llm)}">${llmCell}</td>
        <td class="ledger-note ${note ? '' : 'is-empty'}" title="${escapeHtml(note)}">${note ? escapeHtml(truncate(note, 70)) : '<span class="card-sub">add note ✎</span>'}</td>
      </tr>`
    })
    .join('')
  return `<div class="table-wrap"><table class="runs-table">
    <thead><tr><th>Setup</th><th class="num">seeds</th><th class="num"${helpAttr('Seed robustness: how many seeds land on the good side of 0 (↑), and ⚠ if the objective flips sign across seeds — a fragile setup you should not trust on one lucky run. Hover a cell for the IQR (spread). Needs ≥2 seeds.')}>stability</th><th class="num">${on} avg</th><th class="num">${on} range</th><th class="num">${on} median</th><th class="num">vs hold avg</th><th${helpAttr("The judge's one-line verdict for this setup's best run (if scored).")}>LLM verdict</th><th${helpAttr('Your note for this setup — open the setup to edit. The ledger of everything tried.')}>Your conclusion</th></tr></thead>
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
// Group runs by the ENVIRONMENT (fee / TP-SL regime) they ran in — so one model can be compared
// across regimes. The environment is matched from each run's env-lever values to a named environment.
function aggregateByEnvironment(runs) {
  const groups = new Map()
  for (const r of runs) {
    const sig = runEnvSignature(r)
    if (!groups.has(sig)) groups.set(sig, { name: runEnvName(r), sig, runs: [] })
    groups.get(sig).runs.push(r)
  }
  const out = []
  for (const g of groups.values()) {
    const objs = g.runs.map((r) => Number(r.summary.objective)).filter(Number.isFinite)
    out.push({
      name: g.name,
      sig: g.sig,
      runs: g.runs,
      count: g.runs.length,
      setups: new Set(g.runs.map(setupKeyOfRun)).size,
      objMin: objs.length ? Math.min(...objs) : NaN,
      objMax: objs.length ? Math.max(...objs) : NaN,
      objAvg: mean(objs),
    })
  }
  return out
}
function byEnvironmentTableHtml(filtered) {
  const dir = objectiveDirection()
  const groups = aggregateByEnvironment(filtered).sort((a, b) => {
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
      return `<tr data-environment-sig="${escapeHtml(g.sig)}" class="setup-row">
        <td>${escapeHtml(g.name)}</td>
        <td class="card-sub">${escapeHtml(g.sig)}</td>
        <td class="num">${g.count}</td>
        <td class="num">${g.setups}</td>
        <td class="num">${escapeHtml(formatObjective(g.objAvg))}</td>
        <td class="num">${range}</td>
      </tr>`
    })
    .join('')
  return `<div class="table-wrap"><table class="runs-table">
    <thead><tr><th>Environment</th><th>Settings</th><th class="num">runs</th><th class="num">setups</th><th class="num">${on} avg</th><th class="num">${on} best–worst</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`
}
// When drilled into a single setup, an editor for that setup's conclusion note —
// the user half of the ledger (LLM verdict + score being the other halves).
function setupNoteEditorHtml() {
  if (!runsDrillSetupKey) return ''
  const note = (notesCache.get(runsDrillSetupKey) || {}).note || ''
  return `<div class="setup-note-editor">
    <label class="setup-note-label"${helpAttr('What did this setup teach you? Saved against the setup (not one run) so it survives re-runs — your ledger of everything tried.')}>Your conclusion for this setup</label>
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
  if (runsViewMode === 'environment') {
    const legend = `<p class="runs-legend">Each row is an ENVIRONMENT (fee / TP-SL regime) a run trained in. Click one to drill into its runs. "Custom" = env values matching no saved environment.</p>`
    setHtml(body, `${toolbar}${byEnvironmentTableHtml(filtered)}${legend}`)
    renderCompare()
    return
  }
  const shown = sortRuns(filtered)
  const cols = runsColumns()
  const allSelected = shown.length > 0 && shown.every((r) => runsCompareKeys.has(r.key))
  const header = cols
    .map((c) => {
      // The compare column header is a "select all visible" checkbox.
      if (c.id === 'compare') {
        return `<th><input type="checkbox" id="runs-select-all"${allSelected ? ' checked' : ''} aria-label="Select all visible runs" title="Select all visible runs" /></th>`
      }
      if (c.noSort)
        return `<th class="${c.num ? 'num' : ''}"${helpAttr(c.help)}>${escapeHtml(c.label)}</th>`
      const arrow = runsSortKey === c.id ? (runsSortDir === 'asc' ? ' ▲' : ' ▼') : ''
      return `<th class="runs-th${c.num ? ' num' : ''}" data-sort="${c.id}"${helpAttr(c.help)}>${escapeHtml(c.label)}${arrow}</th>`
    })
    .join('')
  const rows = shown.map((r) => runRowHtml(r, cols)).join('')
  const legend = `<p class="runs-legend">Click a header to sort · hover a column header for what it means · <span class="delta-pos">green</span>/<span class="delta-neg">red</span> = beat / lagged buy-and-hold · greyed = failed/degenerate · "—" = not recorded (re-run to populate).</p>`
  setHtml(
    body,
    `${toolbar}${setupNoteEditorHtml()}<div class="table-wrap"><table class="runs-table">
    <thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table></div>${legend}`,
  )
  if (selectedRunKey && !shown.some((r) => r.key === selectedRunKey)) closeRunDetail()
  else if (selectedRunKey) renderRunDetail(selectedRunKey)
  renderCompare()
  syncRunsSelectionUI()
}
// The two selection-gated run actions in the runs head — Judge selected + Delete
// selected — share one look (icon + text) and one rule: disabled until ≥1 run is
// ticked. Judge also reflects the live judging state.
function syncRunsSelectionUI() {
  const n = runsCompareKeys.size
  const judge = byId('judge-btn')
  if (judge) {
    judge.disabled = judging || n === 0
    judge.innerHTML = judging
      ? busyButtonHtml('Judging…')
      : `${iconJudgeSvg()}<span>Judge${n ? ` (${n})` : ''}</span>`
  }
  const del = byId('runs-delete-selected')
  if (del) {
    del.disabled = n === 0
    del.classList.toggle('is-armed', runsDeleteArmed && n > 0)
    del.innerHTML =
      runsDeleteArmed && n > 0
        ? `${iconDeleteSvg()}<span>Confirm? (${n})</span>`
        : `${iconDeleteSvg()}<span>(${n})</span>`
  }
}
function disarmRunsDelete() {
  if (runsDeleteArmTimer) {
    clearTimeout(runsDeleteArmTimer)
    runsDeleteArmTimer = null
  }
  if (runsDeleteArmed) {
    runsDeleteArmed = false
    syncRunsSelectionUI()
  }
}
async function renderRuns() {
  if (!byId('runs-body')) return
  ;[
    runsCache,
    verdictsCache,
    judgementSummary,
    evaluationsCache,
    notesCache,
    dismissedFailures,
    unrunnableCache,
  ] = await Promise.all([
    readRuns(),
    readVerdicts(),
    readJudgement(),
    readEvaluations(),
    readNotes(),
    readDismissedFailures(),
    readUnrunnable(),
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
    syncRunsMdLayout()
    return
  }
  // Selecting ≥2 runs is a COMPARE gesture — collapse any single-run detail (and its
  // row highlight) so only the compare pane shows.
  if (selectedRunKey) closeRunDetail()
  // One stable colour per run, reused in the id headers AND the return curves below.
  const runColors = new Map(
    runs.map((r, i) => [shortKey(r.key), CHART_PALETTE[i % CHART_PALETTE.length]]),
  )
  const colspan = runs.length + 1
  const headRow = runs
    .map((r) => {
      const c = runColors.get(shortKey(r.key))
      return `<th><code class="cmp-id" style="color:${c};border-color:${c}">${escapeHtml(shortKey(r.key))}</code></th>`
    })
    .join('')
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
          return `<td class="num">${escapeHtml(typeof v === 'number' ? formatMetricValue(mk, v) : v === undefined ? '—' : String(v))}</td>`
        })
        .join('')
      return `<tr><th>${escapeHtml(metricLabel(mk))}</th>${cells}</tr>`
    })
    .join('')
  // Config diff FIRST, then metrics — one table, run ids as colour-coded columns. The
  // wrapper scrolls horizontally so many runs don't bleed past the card.
  const sectionRow = (label) =>
    `<tr class="cmp-section"><th colspan="${colspan}">${escapeHtml(label)}</th></tr>`
  const comparisonTable = `<div class="compare-table-wrap"><table class="kv-table compare-table">
      <thead><tr><th></th>${headRow}</tr></thead>
      <tbody>
        ${sectionRow('Config diff')}
        ${diffRows || `<tr><td colspan="${colspan}" class="card-sub">Selected runs share the same config.</td></tr>`}
        ${sectionRow('Metrics')}
        ${metricRows}
      </tbody>
    </table></div>`
  const versions = new Set(runs.map((r) => String((r.summary && r.summary.pipelineVersion) || '1')))
  const versionWarn =
    versions.size > 1
      ? `<p class="compare-version-warn"><span class="badge is-bad">heads-up</span> These runs span pipeline versions ${[...versions].map((v) => `v${escapeHtml(v)}`).join(', ')} — a breaking version changed how data is fed/scored, so their scores are NOT directly comparable.</p>`
      : ''
  setHtml(
    card,
    `<div class="card-head card-head-row">
      <h3>Compare ${runs.length} runs</h3>
      <button type="button" id="compare-clear" class="icon-btn" title="Clear selection" aria-label="Clear selection">✕</button>
    </div>
    ${versionWarn}
    ${comparisonTable}
    ${compareEquityChartHtml(runs, runColors)}
    <h3>Charts</h3>
    <div class="charts-body">${chartsSectionsHtml(runs, runColors)}</div>`,
  )
  card.hidden = false
  syncRunsMdLayout()
}
// Overlay each selected run's equity as a % -return curve, plus the buy-and-hold
// control derived from a run's price series — so the lines are comparable across
// runs with different absolute net worth.
function compareEquityChartHtml(runs, runColors) {
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
  // Reuse the run-id colours from the comparison table; buy & hold gets a muted grey so it
  // reads as the control, not another run.
  const groupColors = new Map(runColors || [])
  if (points.some((p) => p.group === 'buy & hold')) groupColors.set('buy & hold', '#94a3b8')
  return `<h3>Returns vs buy &amp; hold</h3>${chartLegendHtml(groupColors)}<div class="chart-wrap">${buildLineChart(
    {
      points,
      xLabel: 'step',
      yLabel: 'return %',
      width: 680,
      height: 240,
      markers: false,
      groupColors,
      ariaLabel: 'compared returns vs buy and hold',
    },
  )}</div>`
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
  // Always label the price line; then each action marker that occurred (with its count).
  const markerKeys = ['buy', 'sell', 'tp', 'sl']
    .filter((t) => counts[t])
    .map((t) => `<span class="run-mark-key run-mark-${t}">${escapeHtml(t)} ${counts[t]}</span>`)
    .join(' ')
  const legend = `<p class="badges-row run-mark-legend"><span class="run-mark-key run-mark-price">price</span>${markerKeys ? ` ${markerKeys}` : ''}</p>`
  const svg = buildPriceActionChart(chart, {
    xLabel: 'step',
    ariaLabel: 'price with trade actions',
    width: 640,
    height: 200,
  })
  return `<h3>Price &amp; actions</h3>${legend}<div class="chart-wrap">${svg}</div>`
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
  const groupColors = new Map([
    ['model', CHART_PALETTE[0]],
    ['buy & hold', '#94a3b8'],
  ])
  const svg = buildLineChart({
    points,
    xLabel: 'step',
    yLabel: 'net worth',
    width: 640,
    height: 200,
    markers: false,
    groupColors,
    ariaLabel: 'model equity vs buy and hold',
  })
  const modelPct = ((Number(equity[n - 1]) - start) / start) * 100
  const holdPct = ((start * (Number(price[n - 1]) / p0) - start) / start) * 100
  const delta = modelPct - holdPct
  const fmt = (v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`
  const cls = delta >= 0 ? 'is-ok' : 'is-warn'
  return `<h3>Equity vs buy &amp; hold <span class="card-sub">— control</span></h3>
    ${chartLegendHtml(groupColors)}
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
function failureDetailHtml(s, key) {
  const err = s.error
    ? `<p class="verdict-rejected"><strong>Failed:</strong> ${escapeHtml(String(s.error))}</p>`
    : '<p class="verdict-rejected">Run failed (no error message recorded).</p>'
  const tail =
    Array.isArray(s.logTail) && s.logTail.length
      ? `<details class="log-tail" open><summary>Last output — ${s.logTail.length} lines (stdout + stderr)</summary><pre class="log-tail-pre">${escapeHtml(s.logTail.join('\n'))}</pre></details>`
      : '<p class="card-sub">No log output was captured for this run.</p>'
  // Re-run + diagnose are the header icons now (⧉ clone-to-Launch and the chat button),
  // shown for EVERY run — so the failure block is just the error + log.
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
  const degenerate = !failed && !!(s.health && s.health.status && s.health.status !== 'ok')
  // Verdict only makes sense for a healthy run (degenerate auto-rejects, failed has no
  // result). Evaluation needs that AND a project that supports eval at all (not RL).
  const showVerdict = !failed && !degenerate
  const showEval = showVerdict && evalEnabled()
  const flags = (s.health && Array.isArray(s.health.flags) && s.health.flags) || []
  const flagChips = flags.length
    ? flags.map((f) => `<span class="badge is-bad">${escapeHtml(f)}</span>`).join(' ')
    : '<span class="card-sub">none</span>'
  const checkpoint = (s.artifacts && s.artifacts.checkpoint) || ''
  const datasetBadge = datasetBadgeHtml(s.dataset)
  const isUnrunnable = unrunnableCache.has(setupKeyForRun(run))
  const unrunnableBadge = isUnrunnable
    ? ' · <span class="badge is-bad" title="This setup is marked unrunnable — skipped on re-run for this pipeline version unless forced.">unrunnable</span>'
    : ''
  // The pipeline version this run was produced under; runs across versions aren't comparable.
  const versionBit = s.pipelineVersion
    ? ` · <span title="Pipeline version this run ran under — runs from different versions aren't comparable.">pipeline v${escapeHtml(String(s.pipelineVersion))}</span>`
    : ''
  const envBit = hasEnvLevers()
    ? ` · <span title="${escapeHtml(runEnvSignature(run))}">env ${escapeHtml(runEnvName(run))}</span>`
    : ''
  const headline = failed
    ? '<span class="badge is-bad">failed</span>'
    : `${escapeHtml(objectiveName())} ${escapeHtml(formatObjective(s.objective))} · ${healthBadgeHtml(s.health)}`
  const html = `
    <div class="card-head card-head-row">
      <div>
        <h2>Run <code>${escapeHtml(shortKey(run.key))}</code></h2>
        <p class="card-sub">${headline} · seed ${escapeHtml(s.seed === undefined ? '—' : String(s.seed))}
          · ${escapeHtml(formatWhen(runRanAt(s)))}${datasetBadge ? ` · ${datasetBadge}` : ''}${versionBit}${envBit}${unrunnableBadge}</p>
      </div>
      <div class="head-actions">
        <button type="button" data-action="clone" data-key="${escapeHtml(run.key)}" class="icon-btn" title="Clone to Launch" aria-label="Clone to Launch">⧉</button>
        <button type="button" data-action="chat" data-key="${escapeHtml(run.key)}" class="icon-btn"${chatAboutRunAvailable() ? '' : ' disabled'} title="Discuss this run with the AI" aria-label="Discuss this run">${iconChatSvg()}</button>
        <button type="button" data-action="toggle-unrunnable" data-key="${escapeHtml(run.key)}" class="icon-btn" title="${isUnrunnable ? 'Allow this setup to run again' : 'Mark unrunnable — skip on re-run (this pipeline version) unless forced'}" aria-label="${isUnrunnable ? 'Mark runnable' : 'Mark unrunnable'}">${isUnrunnable ? '⊙' : '⊘'}</button>
        <button type="button" data-action="delete-run" data-key="${escapeHtml(run.key)}" class="icon-btn icon-btn-danger" title="Delete this run (and its evaluation/verdict)" aria-label="Delete run">${iconDeleteSvg()}</button>
        <button type="button" id="run-detail-close" class="icon-btn" title="Close" aria-label="Close">✕</button>
      </div>
    </div>
    ${failed ? failureDetailHtml(s, run.key) : ''}
    ${flags.length ? `<h3>Health flags</h3><p class="badges-row">${flagChips}</p>` : ''}
    ${showVerdict ? verdictSectionHtml(verdictsCache.get(run.key)) : ''}
    ${showEval ? evaluationSectionHtml(run) : ''}
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
  syncRunsMdLayout()
}
// Split the Runs master-detail into two panes only when a detail/compare is open;
// otherwise the list spans the full width.
function syncRunsMdLayout() {
  const md = byId('runs-md')
  if (!md) return
  const detail = byId('run-detail')
  const compare = byId('run-compare')
  const hasDetail = !!((detail && !detail.hidden) || (compare && !compare.hidden))
  md.classList.toggle('has-detail', hasDetail)
}
function openRunDetail(key) {
  // While ≥2 runs are ticked for compare, the compare pane owns the detail column —
  // a stray row click must not pop a single-run detail back open.
  if (runsCompareKeys.size >= 2) return
  selectedRunKey = key
  for (const row of document.querySelectorAll('#runs-body tr[data-key]')) {
    row.classList.toggle('is-selected', row.dataset.key === key)
  }
  renderRunDetail(key)
  syncRunsMdLayout()
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
  syncRunsMdLayout()
}
// Pre-fill the Launch form with a run's exact settings, so it's easy to sweep NEAR
// a known result (tweak one lever, run the neighbours).
function cloneRunToLaunch(key) {
  const run = runsCache.find((r) => r.key === key)
  if (!run) return
  showTab('launch')
  applyPresetFixed(run.summary.config || {})
}
function chatAboutRunAvailable() {
  return embedded() && !!window.OverseerBridge && !!window.OverseerBridge.discussTopic
}
// A topic-chat system prompt describing THIS training project — name, the manifest's plain-language
// description, and the objective in human terms — so the in-app agent is grounded in the actual
// project (e.g. BlackSwan), not the generic model trainer.
function projectChatPreamble() {
  if (!manifest) return ''
  const dir = objectiveDirection() === 'min' ? 'lower is better' : 'higher is better'
  return [
    `You are an expert assistant for the "${manifest.name}" model-training project, helping the user understand and improve its training runs.`,
    manifest.description ? String(manifest.description) : '',
    `The optimisation objective is "${objectiveName()}" (${dir}).`,
  ]
    .filter(Boolean)
    .join('\n\n')
}
// Open the host chat (a project topic) whose SYSTEM PROMPT is preloaded with the project context +
// everything about THIS run — config, metrics, health, objective, and (for a failure) the error +
// log tail — so the user can discuss or diagnose ANY run, grounded in the project, without copying.
async function chatAboutRun(key) {
  const run = runsCache.find((r) => r.key === key)
  if (!run || !chatAboutRunAvailable()) return
  const s = run.summary
  const failed = s.status === 'failed'
  const logTail = Array.isArray(s.logTail) ? s.logTail.slice(-40).join('\n') : ''
  const verdict = verdictsCache.get(key)
  const runContext = [
    failed
      ? `The user is investigating training run ${shortKey(key)}, which FAILED.`
      : `The user is discussing training run ${shortKey(key)}.`,
    `Pipeline v${String(s.pipelineVersion || '1')}.`,
    failed
      ? ''
      : `Objective (${objectiveName()}): ${formatObjective(s.objective)} · health: ${(s.health && s.health.status) || 'unknown'}.`,
    s.metrics ? `Metrics:\n${JSON.stringify(s.metrics, null, 2)}` : '',
    verdict && verdict.why ? `Judge verdict: ${verdict.why}` : '',
    s.error ? `Error:\n${s.error}` : '',
    logTail
      ? `Recent log output (last ${Math.min(40, (s.logTail || []).length)} lines):\n${logTail}`
      : '',
    `Config:\n${JSON.stringify(s.config || {}, null, 2)}`,
  ]
    .filter(Boolean)
    .join('\n\n')
  const systemPrompt = [projectChatPreamble(), runContext].filter(Boolean).join('\n\n')
  const seed = failed
    ? 'This run failed — help me diagnose the cause and propose a fix.'
    : 'Help me understand what this run did and how to improve it.'
  try {
    await window.OverseerBridge.discussTopic({ title: `Run ${shortKey(key)}`, seed, systemPrompt })
  } catch {
    if (selectedRunKey === key)
      setStatusLine('run-eval-status', 'Could not open chat — please try again.', true)
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
  syncRunsSelectionUI()
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
    // Poll regardless of document visibility: the queue pump advances off this settling, so skipping
    // the check while backgrounded stalls the whole queue until the user re-opens the app.
    const act = await getActivity(activityId)
    if (act && act.status && act.status !== 'running') return act
    if (!act && ++missing >= 3) return null
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
  const runKeys = [...runsCompareKeys].filter((k) => runsCache.some((r) => r.key === k))
  if (!runKeys.length) {
    setStatusLine('judge-status', 'Select one or more runs to judge.', false)
    return
  }
  const epoch = projectEpoch
  setStatusLine('judge-status', '')
  try {
    // launchActivity (inside startOrEnqueue) observes the judge: lights the button spinner,
    // refreshes verdicts on settle, and re-pumps the queue. The caller only shows queued.
    const result = await startOrEnqueue('judge', trainerActivityParams({ runKeys }), 'Judge runs')
    if (result.queued && epoch === projectEpoch) {
      setStatusLine('judge-status', queuedStatusText(result.ahead))
    }
  } catch {
    if (epoch === projectEpoch) {
      setStatusLine('judge-status', 'Could not start judging — please try again.', true)
    }
  }
}
// Re-test one run's saved checkpoint via the quick 'evaluate' activity (no
// LLM): start it, observe until it settles, then re-read the evaluation record
// through renderRuns so the table chip and the detail section both refresh.
async function onEvaluateRun(key) {
  if (evaluatingKeys.has(key) || !embedded()) return
  if (!runsCache.some((r) => r.key === key)) return
  if (queueCache.some((q) => q.activityType === 'evaluate' && evaluateKeysOf(q).includes(key))) {
    if (selectedRunKey === key) setStatusLine('run-eval-status', 'Already queued.')
    return
  }
  const epoch = projectEpoch
  try {
    // launchActivity observes the evaluate: lights the per-run spinner (evaluatingKeys) and
    // refreshes the Runs table on settle. The caller only shows the queued message.
    const result = await startOrEnqueue(
      'evaluate',
      trainerComputeParams({ runKeys: [key] }),
      `Evaluate ${shortKey(key)}`,
    )
    if (result.queued && epoch === projectEpoch && selectedRunKey === key) {
      setStatusLine('run-eval-status', queuedStatusText(result.ahead))
    }
  } catch {
    if (epoch === projectEpoch && selectedRunKey === key) {
      setStatusLine('run-eval-status', 'Could not start evaluating — please try again.', true)
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
      const envRow = event.target.closest('tr[data-environment-sig]')
      if (envRow) {
        drillIntoEnvironment(envRow.dataset.environmentSig)
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
        disarmRunsDelete()
        renderCompare()
        syncRunsSelectionUI()
        return
      }
      if (event.target.id === 'runs-select-all') {
        const shownKeys = applyRunsFilters(runsCache).map((r) => r.key)
        if (event.target.checked) for (const k of shownKeys) runsCompareKeys.add(k)
        else for (const k of shownKeys) runsCompareKeys.delete(k)
        disarmRunsDelete()
        renderRunsTable()
        return
      }
      if (event.target.id === 'runs-version-filter') {
        runsVersionFilter = event.target.value
        renderRunsTable()
        return
      }
      const sel = event.target.closest('.runs-filter-lever')
      if (sel) {
        runsLeverFilter[sel.dataset.lever] = sel.value
        renderRunsTable()
        return
      }
      if (event.target.id === 'runs-hide-bad') {
        runsHideBad = event.target.checked
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
      const chatBtn = event.target.closest('button[data-action="chat"]')
      if (chatBtn) chatAboutRun(chatBtn.dataset.key)
      const unrunBtn = event.target.closest('button[data-action="toggle-unrunnable"]')
      if (unrunBtn) toggleUnrunnable(unrunBtn.dataset.key)
      const delBtn = event.target.closest('button[data-action="delete-run"]')
      if (delBtn) deleteRun(delBtn.dataset.key)
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
    judgeBtn.setAttribute('data-help', JUDGE_HELP_TEXT)
  }
  const deleteSelectedBtn = byId('runs-delete-selected')
  if (deleteSelectedBtn) deleteSelectedBtn.addEventListener('click', deleteSelectedRuns)
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

// --- Comparison charts -----------------------------------------------------------
// Shown in the multi-run COMPARE pane, scoped to the SELECTED runs and colour-coded by
// run (the same colours as the comparison table + return curves).
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
function objectiveTimelineSvg(runs, runColors) {
  const points = runs
    .map((r) => ({
      x: new Date(runRanAt(r.summary)).getTime(),
      y: Number(r.summary.objective),
      label: `${shortKey(r.key)} · ${objectiveName()} ${formatObjective(r.summary.objective)} · ${formatWhen(runRanAt(r.summary))}`,
      group: shortKey(r.key),
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
    groupColors: runColors,
  })
}
function timelineChartSectionHtml(runs, runColors) {
  const svg = objectiveTimelineSvg(runs, runColors)
  return chartSectionHtml(`${objectiveName()} over time`, svg, 'Not enough selected runs.', {
    legend: svg ? chartLegendHtml(runColors) : '',
  })
}
// Small-multiple scatters: one per numeric lever on which the selected runs DIFFER,
// log-x when the values span at least two orders of magnitude. Dots coloured by run.
function leverScatterFiguresHtml(runs, runColors) {
  const figures = []
  for (const [key, spec] of leverEntries()) {
    if (spec.type !== 'number') continue
    const points = runs
      .map((r) => {
        const value = Number(r.summary.config && r.summary.config[key])
        return {
          x: value,
          y: Number(r.summary.objective),
          label: `${shortKey(r.key)} · ${key} ${formatObjective(value)} · ${objectiveName()} ${formatObjective(r.summary.objective)}`,
          group: shortKey(r.key),
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
      groupColors: runColors,
    })
    figures.push(chartFigureHtml(key, svg))
  }
  return figures
}
function leverChartsSectionHtml(runs, runColors) {
  const hasNumericLevers = leverEntries().some(([, spec]) => spec.type === 'number')
  let content
  let legend = ''
  if (!hasNumericLevers) {
    content = '<div class="empty-hint">No numeric levers in this manifest.</div>'
  } else {
    const figures = leverScatterFiguresHtml(runs, runColors)
    if (figures.length) {
      content = `<div class="chart-multiples">${figures.join('')}</div>`
      legend = chartLegendHtml(runColors)
    } else {
      content = '<div class="empty-hint">Selected runs don’t differ on a numeric lever.</div>'
    }
  }
  return chartSectionHtml(`${objectiveName()} vs levers`, null, '', { legend, content })
}
function judgeScatterSvg(runs, runColors) {
  const points = []
  for (const r of runs) {
    const verdict = verdictsCache.get(r.key)
    if (!verdict) continue
    const objective = Number(r.summary.objective)
    if (!Number.isFinite(objective)) continue
    const score = Number(verdict.score)
    const group = shortKey(r.key)
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
    groupColors: runColors,
  })
}
function judgeChartSectionHtml(runs, runColors) {
  const svg = judgeScatterSvg(runs, runColors)
  return chartSectionHtml(`Judge score vs ${objectiveName()}`, svg, 'No verdicts for these runs.', {
    legend: svg ? chartLegendHtml(runColors) : '',
  })
}
// The analysis charts (objective timeline, lever effects, judge agreement) over the
// SELECTED runs only, colour-coded by run. Rendered inside the multi-run COMPARE pane.
function chartsSectionsHtml(runs, runColors) {
  return [
    timelineChartSectionHtml(runs, runColors),
    leverChartsSectionHtml(runs, runColors),
    judgeChartSectionHtml(runs, runColors),
  ].join('')
}
// The Versions tab: the pipeline changelog + how runs fared per version. Each
// version's run-count links into the Runs tab filtered to that version.
async function renderVersions() {
  const body = byId('versions-body')
  if (!body) return
  if (!embedded()) {
    setHtml(body, '<div class="empty-hint">Open inside the Overseer to see versions.</div>')
    return
  }
  runsCache = await readRuns()
  const changelog =
    manifest && Array.isArray(manifest.pipelineChangelog) ? manifest.pipelineChangelog : []
  const current = String((manifest && manifest.pipelineVersion) || '1')
  const byVersion = new Map()
  for (const r of runsCache) {
    const v = runVersionOf(r)
    if (!byVersion.has(v)) byVersion.set(v, [])
    byVersion.get(v).push(r)
  }
  const dir = objectiveDirection()
  const versions = [
    ...new Set([...changelog.map((e) => String(e.version)), ...byVersion.keys()]),
  ].sort((a, b) => b.localeCompare(a, undefined, { numeric: true }))
  if (!versions.length) {
    setHtml(
      body,
      '<div class="empty-hint">No pipeline versions yet — declare <code>pipelineVersion</code> + <code>pipelineChangelog</code> in the trainer manifest.</div>',
    )
    return
  }
  const cards = versions
    .map((v) => {
      const entry = changelog.find((e) => String(e.version) === v)
      const runs = byVersion.get(v) || []
      const objs = runs.map((r) => Number(r.summary && r.summary.objective)).filter(Number.isFinite)
      const best = objs.length ? (dir === 'min' ? Math.min(...objs) : Math.max(...objs)) : NaN
      const isCur = v === current
      const breaking = entry && entry.breaking ? '<span class="badge is-bad">breaking</span> ' : ''
      const date =
        entry && entry.date ? `<span class="card-sub">${escapeHtml(String(entry.date))}</span>` : ''
      const summary = entry
        ? escapeHtml(String(entry.summary || ''))
        : '<span class="card-sub">(no changelog entry for this version)</span>'
      const stats = runs.length
        ? `View ${runs.length} run${runs.length === 1 ? '' : 's'} · best ${escapeHtml(objectiveName())} ${escapeHtml(formatObjective(best))} →`
        : 'no runs'
      return `<div class="version-card${isCur ? ' is-current' : ''}">
        <p class="version-card-head"><strong>v${escapeHtml(v)}</strong>${isCur ? ' <span class="badge">current</span>' : ''} ${breaking}${date}</p>
        <p class="version-summary">${summary}</p>
        <p class="card-sub">${runs.length ? `<button type="button" class="link-btn" data-version-filter="${escapeHtml(v)}">${stats}</button>` : stats}</p>
      </div>`
    })
    .join('')
  setHtml(body, `<div class="versions-list">${cards}</div>`)
}
function setupVersions() {
  const body = byId('versions-body')
  if (!body) return
  body.addEventListener('click', (event) => {
    const btn = event.target.closest('button[data-version-filter]')
    if (!btn) return
    runsVersionFilter = btn.dataset.versionFilter
    showTab('runs')
  })
}

// --- Environments tab ------------------------------------------------------------
function environmentCardHtml(env, editable) {
  const rows = envLeverEntries()
    .map(
      ([k]) =>
        `<tr><th>${escapeHtml(k)}</th><td class="num">${escapeHtml(env.settings[k] === undefined ? '—' : String(env.settings[k]))}</td></tr>`,
    )
    .join('')
  const actions = editable
    ? `<div class="head-actions">
        <button type="button" class="icon-btn" data-env-edit="${escapeHtml(env.id)}" title="Edit" aria-label="Edit">✎</button>
        <button type="button" class="icon-btn icon-btn-danger" data-env-delete="${escapeHtml(env.id)}" title="Delete" aria-label="Delete">${iconDeleteSvg()}</button>
      </div>`
    : `<div class="head-actions"><button type="button" class="icon-btn" data-env-clone="${escapeHtml(env.id)}" title="Duplicate to a new environment" aria-label="Duplicate">⧉</button></div>`
  return `<div class="environment-card">
    <div class="card-head card-head-row">
      <h3>${escapeHtml(env.name)}${editable ? '' : ' <span class="card-sub">(manifest defaults)</span>'}</h3>
      ${actions}
    </div>
    <table class="kv-table"><tbody>${rows}</tbody></table>
  </div>`
}
async function renderEnvironments() {
  const body = byId('environments-body')
  if (!body) return
  if (!embedded()) {
    setHtml(body, '<div class="empty-hint">Open inside the Overseer to manage environments.</div>')
    return
  }
  if (!hasEnvLevers()) {
    setHtml(
      body,
      '<div class="empty-hint">This project declares no environment settings. Tag levers with <code>"scope": "environment"</code> in the manifest (e.g. fees, take-profit / stop-loss) to manage them as environments here.</div>',
    )
    return
  }
  environmentsCache = await readEnvironments()
  setHtml(
    body,
    environmentCardHtml(defaultEnvironment(), false) +
      environmentsCache.map((e) => environmentCardHtml(e, true)).join(''),
  )
}
// The create/edit form: a name + a number field per environment lever.
function environmentFormHtml(env) {
  const isNew = !env || env.id === 'default'
  const settings = (env && env.settings) || defaultEnvironment().settings
  const fields = envLeverEntries()
    .map(([k, spec]) => {
      const { min, max } = leverRange(spec)
      const minAttr = Number.isFinite(Number(min)) ? ` min="${Number(min)}"` : ''
      const maxAttr = Number.isFinite(Number(max)) ? ` max="${Number(max)}"` : ''
      const val = settings[k] === undefined ? '' : escapeHtml(String(settings[k]))
      return `<label class="field"><span${helpAttr(spec.description || '')}>${escapeHtml(k)}</span>
        <input type="number" step="any" name="env:${escapeHtml(k)}" value="${val}"${minAttr}${maxAttr} /></label>`
    })
    .join('')
  return `<input type="hidden" name="id" value="${escapeHtml(isNew ? randomHexId() : env.id)}" />
    <label class="field"><span>Name</span>
      <input type="text" name="name" value="${escapeHtml(isNew ? '' : env.name)}" placeholder="e.g. Low fee · tight SL" /></label>
    <div class="lever-grid">${fields}</div>
    <div class="form-actions">
      <button type="submit">Save environment</button>
      <button type="button" id="environment-cancel" class="ghost-btn">Cancel</button>
    </div>`
}
function toggleEnvironmentForm(show, env) {
  const form = byId('environment-form')
  if (!form) return
  setStatusLine('environments-status', '')
  if (show) {
    form.innerHTML = environmentFormHtml(env)
    form.hidden = false
  } else {
    form.innerHTML = ''
    form.hidden = true
  }
}
async function onSaveEnvironment(form) {
  const id = form.elements.id.value
  const name = String(form.elements.name.value || '').trim()
  if (!name) {
    setStatusLine('environments-status', 'Give the environment a name.', true)
    return
  }
  const settings = {}
  for (const [k] of envLeverEntries()) {
    const el = form.querySelector(`input[name="env:${k}"]`)
    if (el && el.value !== '') settings[k] = Number(el.value)
  }
  try {
    await putEnvironment({ id, name, settings })
  } catch {
    setStatusLine('environments-status', 'Could not save — please try again.', true)
    return
  }
  toggleEnvironmentForm(false)
  await renderEnvironments()
  renderLaunchForm()
}
async function onDeleteEnvironment(id) {
  try {
    await deleteEnvironmentRecord(id)
  } catch {
    setStatusLine('environments-status', 'Could not delete — please try again.', true)
    return
  }
  environmentsCache = environmentsCache.filter((e) => e.id !== id)
  await renderEnvironments()
  renderLaunchForm()
}
function setupEnvironments() {
  const addToggle = byId('environment-add-toggle')
  if (addToggle) addToggle.addEventListener('click', () => toggleEnvironmentForm(true))
  const form = byId('environment-form')
  if (form) {
    form.addEventListener('submit', (event) => {
      event.preventDefault()
      onSaveEnvironment(form)
    })
    form.addEventListener('click', (event) => {
      if (event.target.closest('#environment-cancel')) toggleEnvironmentForm(false)
    })
  }
  const body = byId('environments-body')
  if (body) {
    body.addEventListener('click', (event) => {
      const edit = event.target.closest('button[data-env-edit]')
      if (edit) {
        const env = environmentsCache.find((x) => x.id === edit.dataset.envEdit)
        if (env) toggleEnvironmentForm(true, env)
        return
      }
      const clone = event.target.closest('button[data-env-clone]')
      if (clone) {
        toggleEnvironmentForm(true)
        return
      }
      const del = event.target.closest('button[data-env-delete]')
      if (del) onDeleteEnvironment(del.dataset.envDelete)
    })
  }
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
  return `<button type="button" data-action="accept" data-id="${id}" class="icon-btn icon-btn-accept" aria-label="Accept"${helpAttr('Accept this experiment — moves it to the backlog so you can run it.')}>${iconCheckSvg()}</button>
    <button type="button" data-action="reject" data-id="${id}" class="icon-btn icon-btn-reject" aria-label="Reject"${helpAttr('Reject this experiment — hides it from the backlog (you can restore it later).')}>${iconCrossSvg()}</button>${viewRuns}`
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
  // Settled (accepted/rejected) cards get a corner delete button to clear them out.
  const settled = h.status === 'accepted' || h.status === 'rejected'
  const deleteBtn = settled
    ? `<button type="button" class="icon-btn icon-btn-danger hypothesis-delete" data-action="delete" data-id="${escapeHtml(h.id)}" title="Delete hypothesis" aria-label="Delete hypothesis">${iconDeleteSvg()}</button>`
    : ''
  return `<article class="hypothesis-card${h.status === 'rejected' ? ' is-muted' : ''}" data-id="${escapeHtml(h.id)}">
    ${deleteBtn}
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
    // launchActivity observes the propose: lights the button spinner, refreshes hypotheses on
    // settle, and re-pumps the queue. The caller only shows the queued message.
    const result = await startOrEnqueue('propose', trainerActivityParams(), 'Propose experiments')
    if (result.queued && epoch === projectEpoch) {
      setStatusLine('hypotheses-status', queuedStatusText(result.ahead))
    }
  } catch {
    if (epoch === projectEpoch) {
      setStatusLine('hypotheses-status', 'Could not start proposing — please try again.', true)
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
// Permanently remove a settled (accepted/rejected) hypothesis card.
async function deleteHypothesis(id) {
  if (!manifest || !id) return
  setStatusLine('hypotheses-status', '')
  try {
    await window.OverseerBridge.deleteData({ type: manifest.recordType + '-hypothesis', key: id })
  } catch {
    setStatusLine('hypotheses-status', 'Could not delete the hypothesis — please try again.', true)
    return
  }
  hypothesesCache = hypothesesCache.filter((x) => x.id !== id)
  await renderHypotheses()
}
// Also drop a run's derived records so a deleted run fully disappears (no orphan
// evaluation / verdict / unrunnable marker keyed by the same run/setup key).
async function deleteRelatedRunRecords(key, setupKey) {
  const types = [manifest.recordType + '-evaluation', manifest.recordType + '-verdict']
  for (const type of types) {
    try {
      await window.OverseerBridge.deleteData({ type, key })
    } catch {
      // best-effort: a missing derived record is fine
    }
  }
  if (setupKey) {
    try {
      await window.OverseerBridge.deleteData({
        type: manifest.recordType + '-unrunnable',
        key: setupKey,
      })
    } catch {
      // best-effort
    }
  }
}
// Permanently delete a run record + its derived records. Used to clear out old
// (pre-fix) runs so only runs with correct stored values remain.
async function deleteRun(key) {
  if (!manifest || !key) return
  const run = runsCache.find((r) => r.key === key)
  setStatusLine('run-eval-status', '')
  try {
    await window.OverseerBridge.deleteData({ type: manifest.recordType, key })
  } catch {
    if (selectedRunKey === key)
      setStatusLine('run-eval-status', 'Could not delete the run — please try again.', true)
    return
  }
  await deleteRelatedRunRecords(key, run ? setupKeyForRun(run) : undefined)
  runsCache = runsCache.filter((r) => r.key !== key)
  runsCompareKeys.delete(key)
  if (selectedRunKey === key) closeRunDetail()
  await renderRuns()
}
// Delete the SELECTED runs (the ticked checkboxes) after a confirmation popup —
// e.g. select old runs (or all via the header checkbox) and clear them.
async function deleteSelectedRuns() {
  if (!manifest) return
  const keys = [...runsCompareKeys].filter((k) => runsCache.some((r) => r.key === k))
  if (!keys.length) return
  // In-app confirm: window.confirm is blocked in the embedding iframe sandbox (no
  // allow-modals), so the first click arms the button and the second deletes.
  if (!runsDeleteArmed) {
    runsDeleteArmed = true
    syncRunsSelectionUI()
    runsDeleteArmTimer = setTimeout(disarmRunsDelete, 4000)
    return
  }
  disarmRunsDelete()
  for (const key of keys) {
    const run = runsCache.find((r) => r.key === key)
    try {
      await window.OverseerBridge.deleteData({ type: manifest.recordType, key })
      await deleteRelatedRunRecords(key, run ? setupKeyForRun(run) : undefined)
    } catch {
      // best-effort: keep going
    }
  }
  const removed = new Set(keys)
  runsCache = runsCache.filter((r) => !removed.has(r.key))
  for (const k of keys) runsCompareKeys.delete(k)
  if (selectedRunKey && removed.has(selectedRunKey)) closeRunDetail()
  await renderRuns()
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
    // launchActivity already stamped the hypothesis + is observing the campaign.
    if (epoch !== projectEpoch) return
    showTab('activity')
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
    proposeBtn.setAttribute('data-help', PROPOSE_HELP_TEXT)
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
      else if (action === 'delete') deleteHypothesis(id)
    })
  }
}

// --- Launch tab ----------------------------------------------------------------
function leverEntries() {
  return Object.entries((manifest && manifest.levers) || {})
}
// A lever configures the ENVIRONMENT (market mechanics — fees, TP/SL) rather than the MODEL when its
// manifest spec sets scope:'environment'. Environment levers are managed as named environments a model
// runs AGAINST, so they're split out of the model launch form.
function isEnvLever(spec) {
  return !!spec && spec.scope === 'environment'
}
function modelLeverEntries() {
  return leverEntries().filter(([, spec]) => !isEnvLever(spec))
}
function envLeverEntries() {
  return leverEntries().filter(([, spec]) => isEnvLever(spec))
}
function hasEnvLevers() {
  return envLeverEntries().length > 0
}
// The implicit "Default" environment from the manifest's env-lever defaults — always available,
// never stored unless the user edits it into a named environment.
function defaultEnvironment() {
  const settings = {}
  for (const [key, spec] of envLeverEntries())
    if (spec.default !== undefined) settings[key] = spec.default
  return { id: 'default', name: 'Default', settings }
}
// Default first, then the user's saved environments (the launch picker + Environments tab order).
function allEnvironments() {
  return [defaultEnvironment(), ...environmentsCache]
}
async function readEnvironments() {
  if (!manifest) return []
  const recs = await queryRecords(manifest.recordType + ENVIRONMENT_RECORD_SUFFIX)
  return recs.map((r) => r.content).filter((c) => c && c.id && c.id !== 'default')
}
async function putEnvironment(env) {
  await window.OverseerBridge.putData({
    type: manifest.recordType + ENVIRONMENT_RECORD_SUFFIX,
    key: env.id,
    content: { ...env, updatedAt: nowIso() },
  })
}
async function deleteEnvironmentRecord(id) {
  await window.OverseerBridge.deleteData({
    type: manifest.recordType + ENVIRONMENT_RECORD_SUFFIX,
    key: id,
  })
}
// Canonical signature of a run's environment (its env-lever values), for grouping + naming.
function runEnvSignature(run) {
  const cfg = (run && run.summary && run.summary.config) || {}
  return envLeverEntries()
    .map(([key]) => `${key}=${cfg[key] === undefined ? '' : String(cfg[key])}`)
    .join(' · ')
}
function envSettingsSignature(settings) {
  return envLeverEntries()
    .map(([key]) => `${key}=${settings[key] === undefined ? '' : String(settings[key])}`)
    .join(' · ')
}
// The named environment a run matches (by env-value signature), else 'Custom'.
function runEnvName(run) {
  const sig = runEnvSignature(run)
  const match = allEnvironments().find((e) => envSettingsSignature(e.settings) === sig)
  return match ? match.name : 'Custom'
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
// Best objective seen so far per value of a choice lever — used to annotate the launch options with
// "best <obj>" + a ★. When `selection` is given (the other currently-chosen CHOICE levers), only runs
// matching that context count, so the ★ reflects the best value CONDITIONAL on the rest of the form
// (B1). With no selection it's the marginal best across all non-failed runs.
function leverBestSoFar(leverKey, selection) {
  const dir = objectiveDirection()
  const best = new Map()
  for (const r of runsCache) {
    const cfg = r.summary.config || {}
    const v = cfg[leverKey]
    const obj = Number(r.summary.objective)
    if (v === undefined || !Number.isFinite(obj) || r.summary.status === 'failed') continue
    if (selection && !runMatchesSelection(cfg, selection, leverKey)) continue
    const k = String(v)
    if (!best.has(k) || (dir === 'min' ? obj < best.get(k) : obj > best.get(k))) best.set(k, obj)
  }
  return best
}
// True when a run's config matches every selected choice-lever value except the one being annotated.
function runMatchesSelection(cfg, selection, exceptKey) {
  for (const [k, val] of Object.entries(selection)) {
    if (k === exceptKey || val === '' || val === undefined) continue
    if (String(cfg[k]) !== String(val)) return false
  }
  return true
}
// The current choice-lever values picked in the launch form (used to condition leverBestSoFar).
function currentChoiceSelection(form) {
  const sel = {}
  if (!form) return sel
  for (const [key, spec] of leverEntries()) {
    if (spec.type !== 'choice') continue
    const el = form.elements['fixed:' + key]
    if (el && el.value) sel[key] = el.value
  }
  return sel
}
// Recompute the conditional best-so-far annotations in place (no form re-render, so sweep/seed/thesis
// selections are preserved) — called after the form builds and on every lever change.
function refreshLeverAnnotations(form) {
  if (!form) return
  const dir = objectiveDirection()
  const selection = currentChoiceSelection(form)
  for (const [key, spec] of leverEntries()) {
    if (spec.type !== 'choice') continue
    const best = leverBestSoFar(key, selection)
    let topValue
    for (const [v, o] of best) {
      if (
        topValue === undefined ||
        (dir === 'min' ? o < best.get(topValue) : o > best.get(topValue))
      ) {
        topValue = v
      }
    }
    for (const elName of ['fixed:' + key, 'sweep:' + key]) {
      const el = form.elements[elName]
      if (!el || !el.options) continue
      for (const opt of el.options) {
        const b = best.get(String(opt.value))
        opt.textContent =
          b === undefined
            ? String(opt.value)
            : `${opt.value} — best ${formatObjective(b)}${String(opt.value) === topValue ? ' ★' : ''}`
      }
    }
  }
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
  return `<div class="lever-grid">
    <label class="check-row">
      <input type="checkbox" name="fixed:${escapeHtml(key)}"${spec.default ? ' checked' : ''} />
      <span>Enabled</span>
    </label>
    <label class="check-row">
      <input type="checkbox" name="sweep:${escapeHtml(key)}" />
      <span${helpAttr('Run BOTH off and on in one campaign to compare them side by side. Overrides the value on the left.')}>Sweep both</span>
    </label>
  </div>`
}
function leverFieldsetHtml(key, spec) {
  const inner =
    spec.type === 'number'
      ? numberLeverHtml(key, spec)
      : spec.type === 'choice'
        ? choiceLeverHtml(key, spec)
        : booleanLeverHtml(key, spec)
  return `<fieldset class="lever">
    <legend${helpAttr(spec.description || '')}>${escapeHtml(key)} <span class="lever-type">${escapeHtml(spec.type || '')}</span></legend>
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
// Persisted in localStorage (NOT sessionStorage): the embedded iframe gets a fresh
// session on every host reload / cache-bust, which would silently reset the chosen
// parallelism back to 1 — the cause of "I configured 12 but only 1 ran".
function savedConcurrency() {
  try {
    const n = Math.floor(Number(localStorage.getItem(CONCURRENCY_SS)))
    return Number.isFinite(n) && n >= 1 ? n : 1
  } catch {
    return 1
  }
}
function rememberConcurrency(n) {
  try {
    localStorage.setItem(CONCURRENCY_SS, String(Math.max(1, Math.floor(Number(n)) || 1)))
  } catch {
    // storage may be unavailable in a sandboxed frame — purely best-effort
  }
}
// How many ACTIVITIES (campaigns / judge / propose / evaluate) may run at once. A judge no
// longer blocks a campaign, and several campaigns can run together. Each campaign still has
// its own "Max parallel runs", so total training processes ≈ the sum — keep both modest.
function savedActivityBudget() {
  try {
    const n = Math.floor(Number(localStorage.getItem(ACTIVITY_BUDGET_SS)))
    return Number.isFinite(n) && n >= 1 ? n : DEFAULT_ACTIVITY_BUDGET
  } catch {
    return DEFAULT_ACTIVITY_BUDGET
  }
}
function rememberActivityBudget(n) {
  try {
    localStorage.setItem(ACTIVITY_BUDGET_SS, String(Math.max(1, Math.floor(Number(n)) || 1)))
  } catch {
    // best-effort
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
// Which ENVIRONMENTS to run the model against (checkboxes; Default pre-checked). Picking several
// tests the same model across regimes in one campaign. Hidden when the manifest has no env levers.
function environmentPickerHtml() {
  if (!hasEnvLevers()) return ''
  const rows = allEnvironments()
    .map((e) => {
      const summary = envLeverEntries()
        .map(([k]) => `${k} ${e.settings[k] === undefined ? '—' : e.settings[k]}`)
        .join(' · ')
      return `<label class="check-row env-pick">
        <input type="checkbox" name="env" value="${escapeHtml(e.id)}"${e.id === 'default' ? ' checked' : ''} />
        <span><strong>${escapeHtml(e.name)}</strong> <span class="card-sub">${escapeHtml(summary)}</span></span>
      </label>`
    })
    .join('')
  return `<fieldset class="lever env-picker">
    <legend${helpAttr('Which ENVIRONMENTS (fee / TP-SL regimes) to run this model against. Pick several to test one model across regimes in a single campaign — runs = configs × environments × seeds. Define + tweak them in the Environments tab.')}>Run against environments</legend>
    ${rows}
  </fieldset>`
}
function selectedEnvironments(form) {
  const ids = new Set([...form.querySelectorAll('input[name="env"]:checked')].map((el) => el.value))
  return allEnvironments().filter((e) => ids.has(e.id))
}
function renderLaunchForm() {
  const form = byId('launch-form')
  if (!form) return
  const levers = modelLeverEntries()
    .map(([key, spec]) => leverFieldsetHtml(key, spec))
    .join('')
  form.innerHTML = `
    ${presetsSelectHtml()}
    ${environmentPickerHtml()}
    ${levers || '<p class="card-sub">This manifest declares no model levers — the campaign runs the default config.</p>'}
    <fieldset class="lever">
      <legend>Campaign</legend>
      <div class="lever-grid">
        <label class="field"><span${helpAttr('What this campaign tests, e.g. "fee-penalty reward" or "1m data prep". Stamped on every run so you can group + compare by experiment in the By-experiment view. Optional.')}>Thesis</span>
          <input type="text" name="thesis" placeholder="what are you testing? (optional)" />
        </label>
        <label class="field"><span${helpAttr('Optional: the lever this thesis varies, so the by-experiment view can highlight it. Leave blank for theses outside the levers (e.g. a new data prep or code change).')}>Testing which setting?</span>
          <select name="thesisTarget"><option value="">—</option>${modelLeverEntries()
            .map(([k]) => `<option value="${escapeHtml(k)}">${escapeHtml(k)}</option>`)
            .join('')}</select>
        </label>
        <label class="field"><span${helpAttr('How many seeds to run per config (0…N−1). Keep at 1 while exploring; raise it when homing in to measure variance across seeds (the by-setup view then shows the spread).')}>Seeds</span>
          <input type="number" name="seeds" min="1" step="1" value="1" />
        </label>
        <label class="field"><span${helpAttr('How many runs of this campaign train at once (a bounded worker pool). Higher finishes the sweep faster but uses more CPU/RAM. 1 = strictly sequential.')}>Max parallel runs</span>
          <input type="number" name="concurrency" min="1" step="1" value="${savedConcurrency()}" />
        </label>
        <label class="check-row launch-refresh">
          <input type="checkbox" name="refresh" />
          <span${helpAttr('Off (default): a config with a completed run is skipped. On: re-run it anyway, e.g. after changing code or data.')}>Refresh — re-run configs that already have a result</span>
        </label>
        <label class="check-row launch-skip-explored">
          <input type="checkbox" name="skipExplored" checked />
          <span${helpAttr('On by default: if a setup (the config ignoring seed) was already run, skip it — exploration should not re-test the same idea. Turn OFF when homing in to run more seeds of a setup.')}>Exploration — skip setups already tried (any seed)</span>
        </label>
        ${
          evalEnabled()
            ? `<label class="check-row launch-autoeval">
          <input type="checkbox" name="autoEval"${savedAutoEval() ? ' checked' : ''} />
          <span${helpAttr('After each run finishes, automatically re-test its saved checkpoint (shown in the Eval column).')}>Auto-evaluate completed runs</span>
        </label>`
            : ''
        }
        <label class="field launch-target" id="launch-target-field">${computeTargetFieldHtml(launchRunnersCache, savedComputeTarget())}</label>
      </div>
    </fieldset>
    <p class="launch-summary" id="launch-summary"></p>
    <div class="form-actions">
      <button type="submit" id="launch-btn">Launch campaign</button>
    </div>
    <p id="launch-status" class="form-status" role="status"></p>`
  refreshLeverAnnotations(form)
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
      else if (sweepEl.type === 'checkbox') sweepEl.checked = false
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
    } else if (sweepEl.type === 'checkbox') {
      sweepEl.checked = values.length > 1
    } else {
      sweepEl.value = values.join(', ')
    }
  }
  if (preset.seeds && form.elements.seeds) form.elements.seeds.value = String(preset.seeds)
  if (form.elements.thesis) form.elements.thesis.value = preset.thesis || ''
  if (form.elements.thesisTarget) form.elements.thesisTarget.value = preset.thesisTarget || ''
  refreshLeverAnnotations(form)
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
  if (spec.type === 'boolean') return el.checked ? [false, true] : []
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
  // Only MODEL levers — environment levers come from the selected environments, not the form.
  for (const [key, spec] of modelLeverEntries()) {
    const values = readSweepValues(form, key, spec)
    if (values.length) sweep[key] = values
    else fixed[key] = readFixedValue(form, key, spec)
  }
  const seedCount = readSeedCount(form)
  const out = { sweep, fixed, seeds: Array.from({ length: seedCount }, (_, i) => i) }
  if (hasEnvLevers()) {
    const envs = selectedEnvironments(form)
    if (envs.length) out.environments = envs.map((e) => e.settings)
  }
  return out
}
function updateLaunchSummary() {
  const form = byId('launch-form')
  const line = byId('launch-summary')
  if (!form || !line) return
  const spec = buildSpecFromForm(form)
  const configs = Object.values(spec.sweep).reduce((acc, values) => acc * values.length, 1)
  const envs = Array.isArray(spec.environments) ? spec.environments.length : 1
  const seeds = spec.seeds.length
  const total = configs * envs * seeds
  const envBit = Array.isArray(spec.environments)
    ? ` × ${envs} environment${envs === 1 ? '' : 's'}`
    : ''
  const target = remoteComputeTarget(savedComputeTarget())
  line.textContent = `${configs} configuration${configs === 1 ? '' : 's'}${envBit} × ${seeds} seed${seeds === 1 ? '' : 's'} = ${total} run${total === 1 ? '' : 's'}${target ? ` on ${target}` : ''}`
}
function campaignLabel(spec) {
  const sweeps = Object.entries(spec.sweep || {}).map(
    ([key, values]) => `${key} × ${values.length}`,
  )
  const envs = Array.isArray(spec.environments) ? spec.environments.length : 0
  const seeds = Array.isArray(spec.seeds) ? spec.seeds.length : 1
  const envBit = envs > 1 ? `${envs} envs, ` : ''
  return `Campaign: ${sweeps.length ? `${sweeps.join(', ')}, ` : ''}${envBit}seeds ${seeds}`
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
  const concurrency = Math.max(
    1,
    Math.floor(Number(form.elements.concurrency && form.elements.concurrency.value)) || 1,
  )
  rememberConcurrency(concurrency)
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
      concurrency,
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
    // launchActivity recorded the auto-eval marker (if any) + is observing the campaign.
    if (epoch !== projectEpoch) return
    if (status) status.textContent = ''
    showTab('activity')
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
    if (event.target && event.target.name === 'concurrency') {
      rememberConcurrency(event.target.value)
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
    refreshLeverAnnotations(form)
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
function activityTypeLabel(type) {
  if (type === 'train') return 'Campaign'
  if (type === 'judge') return 'Judge runs'
  if (type === 'propose') return 'Propose experiments'
  if (type === 'evaluate') return 'Evaluate runs'
  return type
}
// On opening a project (or regaining focus), re-attach an observer to EVERY backend-live
// activity for this project we're not already tracking — up to the budget — so concurrent
// campaigns + a judge all reconnect. When nothing is live, surface the last campaign result.
async function resumeRunningActivity() {
  const epoch = projectEpoch
  let activities = []
  try {
    const res = await window.OverseerBridge.listActivities()
    activities = (res && res.activities) || []
  } catch {
    activities = []
  }
  if (epoch !== projectEpoch || !manifest) return
  const mine = activities.filter((a) => a.recordType === manifest.recordType)
  const tracked = (id) => [...liveActivities.values()].some((e) => e.activityId === id)
  const live = mine.filter((a) => a.status === 'running' && a.isLive !== false)
  for (const a of live) {
    if (liveSlotCount() >= savedActivityBudget()) break
    if (tracked(a.activityId)) continue
    const type = quickActivityType(a) || 'train'
    trackExistingActivity(a.activityId, type, activityTypeLabel(type))
  }
  // STALLED: 'running' in the store but NOT live in the backend (it restarted mid-run). Surface
  // each as PAUSED + Resume — resuming re-launches it and the trainer's completed-record skip
  // re-runs only the PENDING runs (a 4-run campaign with 2 done resumes just the other 2).
  for (const a of mine) {
    if (a.status !== 'running' || a.isLive !== false) continue
    if (!a.resumeToken || !a.resumeToken.activityType) continue
    if (tracked(a.activityId)) continue
    const type = a.resumeToken.activityType || 'train'
    const localId = 'stalled:' + a.activityId
    liveActivities.set(localId, {
      localId,
      activityId: a.activityId,
      activityType: type,
      label: activityTypeLabel(type),
      status: 'paused',
      progress: null,
      campaign: null,
      startedAt: Date.parse(a.updatedAt || a.createdAt || '') || Date.now(),
      item: { activityType: type, label: activityTypeLabel(type) },
    })
  }
  if (!liveActivities.size) {
    const campaign = await readCampaign()
    const progress = await readProgress()
    if (epoch !== projectEpoch) return
    if (campaign || (progress && progress.phase === 'done')) {
      lastSettledCampaign = campaign
    }
  }
  renderActivity()
}
// The activity's type from its resume token (the host carries `{activityType,…}`
// there), used to tell a running judge / propose / train apart in the list.
function quickActivityType(activity) {
  const token = activity && activity.resumeToken
  return token && typeof token.activityType === 'string' ? token.activityType : ''
}
// Orchestrate ONE activity's lifecycle: light its quick-button state, observe to settlement,
// run the type-specific settle, then untrack + re-pump to free its slot.
async function observeActivity(entry) {
  const epoch = projectEpoch
  const item = entry.item
  applyQuickDispatchState(item, true)
  try {
    if (item.activityType === 'train') {
      await observeTrainActivity(entry, epoch)
    } else {
      const act = await observeQuickActivity(entry.activityId)
      entry.status = (act && act.status) || 'completed'
      if (epoch === projectEpoch) await refreshAfterQuickDispatch(item, act)
    }
  } finally {
    applyQuickDispatchState(item, false)
    // A paused campaign keeps its slot + block (with a Resume button) until resumed/aborted.
    if (entry.status !== 'paused') liveActivities.delete(entry.localId)
    if (epoch === projectEpoch) {
      renderActivity()
      pumpQueue()
    }
  }
}
// Poll a training campaign to settlement, writing its OWN progress/campaign/status into its
// `entry` (filtered to its activityId — the backend keys these records 'latest', so with two
// same-project campaigns a block's live progress is best-effort; results are unaffected).
async function observeTrainActivity(entry, epoch) {
  const activityId = entry.activityId
  const start = Date.now()
  let resumeTries = 0
  let deadSince = 0
  let lastSig = ''
  while (Date.now() - start < MAX_OBSERVE_MS) {
    if (epoch !== projectEpoch || !liveActivities.has(entry.localId)) return
    const [progress, act, campaign] = await Promise.all([
      readProgress(),
      getActivity(activityId),
      readCampaign(),
    ])
    if (epoch !== projectEpoch || !liveActivities.has(entry.localId)) return
    const mineProgress = progress && progress.activityId === activityId ? progress : null
    const mineCampaign = campaign && campaign.activityId === activityId ? campaign : null
    if (mineProgress) entry.progress = mineProgress
    if (mineCampaign) entry.campaign = mineCampaign
    if (act && act.status && act.status !== 'running') {
      entry.status = act.status
      await settleTrainActivity(entry)
      return
    }
    const sig = activityProgressSig(mineProgress, mineCampaign)
    const advanced = sig !== lastSig
    lastSig = sig
    if ((!act || act.isLive === false) && !advanced && document.hidden) {
      // Backgrounded: throttled timers make a stalled-looking poll most likely us, not the
      // run — never declare paused while hidden; the visibility-regain handler re-verifies.
      entry.status = 'running'
    } else if ((!act || act.isLive === false) && !advanced) {
      const now = Date.now()
      if (!deadSince) deadSince = now
      if (now - deadSince >= DEAD_CONFIRM_MS) {
        // Sustained-dead: one relaunch (never while advancing — would double-run), then pause.
        if (resumeTries < 1) {
          resumeTries += 1
          deadSince = now
          try {
            await window.OverseerBridge.resumeActivity(activityId)
          } catch {
            // keep observing; the records may settle on their own
          }
        } else {
          entry.status = 'paused'
          renderActivity()
          return
        }
      } else {
        entry.status = 'running'
      }
    } else {
      deadSince = 0
      entry.status = 'running'
    }
    renderActivity()
    await sleep(POLL_MS)
  }
  entry.status = 'running'
}
// A cheap fingerprint of campaign progress; when it changes between polls the run is
// demonstrably alive (so we never pause/relaunch it even if `isLive` momentarily lies).
function activityProgressSig(progress, campaign) {
  const p = progress || {}
  const c = campaign || {}
  return [
    p.updatedAt,
    p.done,
    p.total,
    p.phase,
    Array.isArray(p.inFlight) ? p.inFlight.length : 0,
    c.updatedAt,
    c.done,
  ].join('|')
}
// A campaign settled: re-read its result, refresh the Runs tab so new run records show,
// and run the settle bookkeeping (hypothesis stamps + auto-eval markers).
async function settleTrainActivity(entry) {
  const activityId = entry.activityId
  const progress = await readProgress()
  const campaign = await readCampaign()
  if (progress && progress.activityId === activityId) entry.progress = progress
  if (campaign && campaign.activityId === activityId) entry.campaign = campaign
  if (entry.campaign) lastSettledCampaign = entry.campaign
  renderActivity()
  await renderRuns()
  await processSettledCampaignEffects()
}
// Abort ONE activity by id — the observe loop's next poll sees 'aborted' and settles it.
async function abortActivityById(activityId) {
  if (!activityId) return
  const btn = document.querySelector(`[data-abort="${escapeHtml(activityId)}"]`)
  if (btn) btn.disabled = true
  try {
    await window.OverseerBridge.abortActivity(activityId)
  } catch {
    if (btn) btn.disabled = false
  }
  // A stalled (paused, not-observed) entry has no observe loop to settle it — drop it here so the
  // discarded campaign disappears. A live (running) entry is left for its observe loop to settle.
  let droppedPaused = false
  for (const [k, e] of liveActivities) {
    if (e.activityId === activityId && e.status === 'paused') {
      liveActivities.delete(k)
      droppedPaused = true
    }
  }
  if (droppedPaused) {
    renderActivity()
    pumpQueue()
  }
}
// Resume ONE paused activity by id — re-fire its observe loop.
async function resumeActivityById(activityId) {
  const entry = [...liveActivities.values()].find((a) => a.activityId === activityId)
  if (!entry) return
  const btn = document.querySelector(`[data-resume="${escapeHtml(activityId)}"]`)
  if (btn) btn.disabled = true
  try {
    await window.OverseerBridge.resumeActivity(activityId)
    entry.status = 'running'
    renderActivity()
    void observeActivity(entry)
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
// The whole-campaign bar: how many of the planned runs have finished (k of N).
// Each concurrently-running run shows its own within-run bar + ETA separately
// (inFlightHtml), and the campaign ETA is shown on its own line.
function activityProgressHtml(progress, running) {
  return progressBarHtml(progress, running, 'Runs')
}
const CURRENT_PHASE_LABEL = {
  loading: 'loading data + model…',
  starting: 'starting…',
  train: 'training',
  test: 'testing',
  summarize: 'summarizing',
}
// One row PER in-flight run (the backend tracks all concurrent runs in
// `progress.inFlight`): key + phase with a spinner, a live elapsed timer + per-run
// ETA, and EITHER a real within-run done/total bar (data-driven runs) or an honest
// indeterminate striped bar. The campaign "k of N runs" total bar is separate.
// Elapsed/ETA are filled by the ticker (data-*-key spans), not baked in, so the
// markup stays stable across 3s polls and the striped animation doesn't restart.
function inFlightRowHtml(run) {
  if (!run || !run.key) return ''
  const phase = String(run.phase || '')
  const phaseLabel = CURRENT_PHASE_LABEL[phase] || phase || 'running'
  const done = Number(run.done)
  const total = Number(run.total)
  const hasCount = Number.isFinite(done) && Number.isFinite(total) && total > 0
  const pct = hasCount ? Math.max(0, Math.min(100, (done / total) * 100)) : 0
  const bar = hasCount
    ? `<div class="build-progress-bar"><span style="width:${pct.toFixed(1)}%"></span></div>`
    : '<div class="build-progress-bar"><span class="is-indeterminate"></span></div>'
  const count = hasCount ? `${done} / ${total} · ${Math.round(pct)}%` : ''
  const k = escapeHtml(run.key)
  return `<div class="current-item">
    <p class="current-item-head">${spinnerHtml()} Run <code>${escapeHtml(shortKey(run.key))}</code> · ${escapeHtml(phaseLabel)}<span class="current-item-elapsed" data-elapsed-key="${k}"></span><span class="current-item-eta" data-eta-key="${k}"></span></p>
    <div class="build-progress">
      ${bar}
      ${count ? `<span class="build-progress-label">${escapeHtml(count)}</span>` : ''}
    </div>
  </div>`
}
function inFlightHtml(runs) {
  if (!runs.length) return ''
  const head = `<p class="card-sub inflight-head">${runs.length} run${runs.length === 1 ? '' : 's'} running now</p>`
  return `<div class="inflight-runs">${head}${runs.map(inFlightRowHtml).join('')}</div>`
}
// Drive every in-flight run's elapsed timer (mm:ss) + a live time-left estimate from
// that run's OWN training progress (elapsed × remaining/done). Setup phases
// (loading/starting) show no ETA — that time is genuinely indeterminate. One shared
// interval updates all rows by their data-*-key spans.
function syncInFlightTimer(runs) {
  if (currentItemTimer) {
    clearInterval(currentItemTimer)
    currentItemTimer = null
  }
  if (!runs || !runs.length) return
  const tick = () => {
    for (const run of runs) {
      if (!run || !run.key || !run.startedAt) continue
      const el = document.querySelector(`[data-elapsed-key="${run.key}"]`)
      if (el) el.textContent = ` · ${formatElapsed(run.startedAt)}`
      const etaEl = document.querySelector(`[data-eta-key="${run.key}"]`)
      if (!etaEl) continue
      const done = Number(run.done)
      const total = Number(run.total)
      if (String(run.phase) === 'train' && done > 0 && total > done) {
        const elapsedMs = Date.now() - new Date(run.startedAt).getTime()
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
// Runs that failed across the settled campaign: a collapsible per-run {key, error}
// list where each entry is INSPECTABLE (opens the run detail with the full error +
// logTail + "Ask AI for help"), RE-RUNNABLE (clones its config to Launch), and
// DISMISSABLE (hidden from the list, persisted). Dismissed entries drop out.
function campaignFailuresHtml(campaign) {
  const all = (campaign && Array.isArray(campaign.failures) && campaign.failures) || []
  const failures = all.filter((f) => !dismissedFailures.has(f.key))
  const dismissed = all.length - failures.length
  if (!failures.length) {
    return dismissed
      ? `<p class="card-sub run-failures-empty">${dismissed} failed run${dismissed === 1 ? '' : 's'} dismissed.</p>`
      : ''
  }
  const items = failures
    .map(
      (f) =>
        `<li class="run-failure-item">
          <span class="run-failure-text"><code>${escapeHtml(shortKey(f.key))}</code> — ${escapeHtml(f.error || 'failed')}</span>
          <span class="run-failure-actions">
            <button type="button" class="link-btn" data-failure-action="inspect" data-key="${escapeHtml(f.key)}">See error</button>
            <button type="button" class="link-btn" data-failure-action="rerun" data-key="${escapeHtml(f.key)}">Re-run</button>
            <button type="button" class="link-btn" data-failure-action="dismiss" data-key="${escapeHtml(f.key)}">Dismiss</button>
          </span>
        </li>`,
    )
    .join('')
  const dismissedNote = dismissed ? ` <span class="card-sub">· ${dismissed} dismissed</span>` : ''
  return `<details class="run-failures" open>
    <summary>${failures.length} run${failures.length === 1 ? '' : 's'} failed${dismissedNote}</summary>
    <ul class="run-failures-list">${items}</ul>
  </details>`
}
// Open a failed run's full detail (error + logTail + AI-help) from the Activity list.
function inspectFailedRun(key) {
  if (!key) return
  showTab('runs')
  openRunDetail(key)
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
  if (!queueCache.length) return ''
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
    <ul class="queue-list">${rows}</ul>
  </div>`
}
// Two activity-level knobs (persisted, applied to activities started after): how many
// ACTIVITIES (campaigns / judge / …) run at once, and how many RUNS each campaign runs at once.
function activitySettingsHtml() {
  return `<div class="activity-settings">
    <label class="field"><span${helpAttr('How many ACTIVITIES (campaigns, judge, propose, evaluate) run at the same time. A judge no longer blocks a campaign, and several campaigns can run together. Each campaign also has its own “Max parallel runs”, so total training processes ≈ the sum — keep both modest for your host.')}>Max concurrent activities</span>
      <input type="number" id="activity-budget" min="1" step="1" value="${savedActivityBudget()}" />
    </label>
    <label class="field"><span${helpAttr('How many runs of a campaign run at once. Set this BEFORE you launch — it applies to the NEXT campaign you start. It does NOT resize a campaign already running (you would relaunch to change that). The real ceiling is host CPU/GPU/RAM; default 1 = sequential.')}>Max parallel runs</span>
      <input type="number" id="activity-concurrency" min="1" step="1" value="${savedConcurrency()}" />
    </label>
  </div>`
}
// One block per LIVE activity (campaign / judge / …); campaigns show full progress, quick
// activities just a status pill. Each campaign has its own Abort / Resume.
function activityBlockHtml(entry) {
  const status = entry.status === 'starting' ? 'running' : entry.status
  const meta = STATUS_META[status] || { label: status, cls: '' }
  const running = status === 'running'
  const isTrain = entry.activityType === 'train'
  const p = isTrain ? entry.progress : null
  const phase = p && PHASE_LABEL[p.phase] ? PHASE_LABEL[p.phase] : ''
  const eta =
    running && p && Number(p.etaSeconds) > 0
      ? `<p class="activity-eta">ETA ~${escapeHtml(formatEta(p.etaSeconds))}${p.etaApprox ? ' (est.)' : ''}</p>`
      : ''
  const times = p
    ? `<p class="card-sub">Started ${escapeHtml(formatWhen(p.startedAt))} · Updated ${escapeHtml(formatWhen(p.updatedAt))}${p.lastKey ? ` · Last run <code>${escapeHtml(shortKey(p.lastKey))}</code>` : ''}</p>`
    : ''
  const inFlight =
    running && p && p.phase === 'train'
      ? Array.isArray(p.inFlight)
        ? p.inFlight
        : p.current
          ? [p.current]
          : []
      : []
  const actions =
    running && entry.activityId
      ? `<div class="form-actions"><button type="button" class="danger-btn" data-abort="${escapeHtml(entry.activityId)}">Abort</button></div>`
      : status === 'paused' && entry.activityId
        ? `<div class="form-actions"><button type="button" data-resume="${escapeHtml(entry.activityId)}">Resume</button><button type="button" class="ghost-btn" data-abort="${escapeHtml(entry.activityId)}">Discard</button></div>`
        : ''
  const stalledNote =
    status === 'paused' && isTrain
      ? '<p class="card-sub">Stalled (the backend restarted while it ran). Resume re-runs only the unfinished runs — completed ones are kept.</p>'
      : ''
  return `<div class="activity-block">
    <div class="activity-status-row">
      <span class="status-pill ${meta.cls}">${running ? `${spinnerHtml()} ` : ''}${escapeHtml(meta.label)}</span>
      ${isTrain ? '' : `<span class="activity-kind">${escapeHtml(entry.label)}</span>`}
      ${phase ? `<span class="activity-phase">${escapeHtml(phase)}</span>` : ''}
      ${entry.activityId ? `<code class="activity-id">${escapeHtml(shortKey(entry.activityId))}</code>` : ''}
    </div>
    ${stalledNote}
    ${isTrain && p ? activityProgressHtml(p, running) : ''}
    ${inFlightHtml(inFlight)}
    ${isTrain ? activityCountsHtml(p) : ''}
    ${eta}
    ${isTrain ? times : ''}
    ${isTrain ? bestLineHtml(entry.campaign) : ''}
    ${isTrain ? campaignFailuresHtml(entry.campaign) : ''}
    ${actions}
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
  const entries = [...liveActivities.values()].sort((a, b) => b.startedAt - a.startedAt)
  const blocks = entries.map(activityBlockHtml).join('')
  // Every in-flight run across all live campaigns, for the shared elapsed-timer ticker.
  const allInFlight = []
  for (const entry of entries) {
    const p = entry.activityType === 'train' && entry.status === 'running' ? entry.progress : null
    if (p && p.phase === 'train') {
      const inf = Array.isArray(p.inFlight) ? p.inFlight : p.current ? [p.current] : []
      for (const r of inf) allInFlight.push(r)
    }
  }
  if (!blocks) {
    const last = lastSettledCampaign ? bestLineHtml(lastSettledCampaign) : ''
    const lastFailures = lastSettledCampaign ? campaignFailuresHtml(lastSettledCampaign) : ''
    setHtml(
      body,
      activitySettingsHtml() +
        (last
          ? `<div class="activity-block"><div class="activity-status-row"><span class="status-pill is-ok">Last campaign</span></div>${last}${lastFailures}</div>${queueHtml}`
          : queueHtml ||
            '<div class="empty-hint">No campaign yet — launch one from the Launch tab.</div>'),
    )
    syncInFlightTimer([])
    return
  }
  setHtml(body, `${activitySettingsHtml()}${blocks}${queueHtml}`)
  syncInFlightTimer(allInFlight)
}
function setupActivity() {
  const body = byId('activity-body')
  if (!body) return
  body.addEventListener('click', (event) => {
    const abortBtn = event.target.closest('button[data-abort]')
    if (abortBtn) {
      abortActivityById(abortBtn.dataset.abort)
      return
    }
    const resumeBtn = event.target.closest('button[data-resume]')
    if (resumeBtn) {
      resumeActivityById(resumeBtn.dataset.resume)
      return
    }
    const failBtn = event.target.closest('button[data-failure-action]')
    if (failBtn) {
      const action = failBtn.dataset.failureAction
      const key = failBtn.dataset.key
      if (action === 'inspect') inspectFailedRun(key)
      else if (action === 'rerun') cloneRunToLaunch(key)
      else if (action === 'dismiss') dismissFailure(key)
      return
    }
    const removeBtn = event.target.closest('button[data-queue-remove]')
    if (removeBtn) onQueueRemove(removeBtn.dataset.queueRemove)
  })
  body.addEventListener('change', (event) => {
    if (event.target.id === 'activity-concurrency') rememberConcurrency(event.target.value)
    else if (event.target.id === 'activity-budget') {
      rememberActivityBudget(event.target.value)
      pumpQueue()
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
  // Runs is the only full-width, own-scroll master-detail tab.
  const main = document.querySelector('.tab-main')
  if (main) main.classList.toggle('is-fullwidth', target === 'runs')
  try {
    sessionStorage.setItem(ACTIVE_TAB_SS, target)
  } catch {
    // storage may be unavailable in a sandboxed frame — purely best-effort
  }
  renderTabLiveIndicator()
  if (target === 'runs') renderRuns()
  if (target === 'versions') renderVersions()
  if (target === 'environments') renderEnvironments()
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
// Which tabs show a spinner. Activity is the single home for "what's pending" — it spins for ANY live
// work (campaign / judge / propose / evaluate). Runs spins ONLY for run-specific work that lands on a
// run (a judgement or an evaluation). Hypotheses spins while proposing. Versions is view-only — never.
function tabHasLiveWork(id) {
  if (id === 'activity') {
    return anyActivityRunning() || judging || proposing || evaluatingKeys.size > 0
  }
  if (id === 'runs') return judging || evaluatingKeys.size > 0
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
// When the viewer regains visibility/focus after being backgrounded (timers throttled),
// re-verify any running activity and keep the queue draining — so a campaign + its queued
// follow-ups don't appear to "stop between runs" just because we were in the background.
// (Fully UNATTENDED progression while the app is closed needs a server-side queue drain.)
function onViewerVisible() {
  if (!embedded() || !manifest) return
  if (anyActivityRunning()) {
    pumpQueue()
    return
  }
  void resumeRunningActivity().then(() => pumpQueue())
}
function setupVisibilityResume() {
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) onViewerVisible()
  })
  window.addEventListener('focus', onViewerVisible)
}
async function init() {
  setupHome()
  setupRunners()
  setupRuns()
  setupVersions()
  setupEnvironments()
  setupHypotheses()
  setupLaunch()
  setupActivity()
  setupTabs()
  setupVisibilityResume()
  setupHelpTooltips()
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
