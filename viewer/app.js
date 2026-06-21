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
// Two independent concurrency lanes. EXPERIMENT activities (training campaigns + checkpoint
// evaluations) execute the project's training code on compute and share one budget; the lighter
// TASK activities (judge / propose / analyze-paper) get their own budget so a quick judge or paper
// import is never blocked behind a long campaign. The experiment key is kept stable so an existing
// 'activityBudget' setting carries over as the experiment budget.
const EXPERIMENT_BUDGET_SS = 'trainer.activityBudget'
const DEFAULT_EXPERIMENT_BUDGET = 3
const TASK_BUDGET_SS = 'trainer.taskBudget'
const DEFAULT_TASK_BUDGET = 3
// Activity types that run on compute — the experiment lane. Everything else is a task.
const EXPERIMENT_ACTIVITY_TYPES = new Set(['train', 'evaluate'])
// Whether the 2nd ('Tasks') column is collapsed (persisted per session).
const TASKS_COLLAPSED_SS = 'trainer.tasksCollapsed'
// activityIds the user PAUSED (vs a backend-down stall). Persisted so a paused campaign still
// surfaces as Resume-able after a reload; resuming re-launches it and the trainer's completed-run
// skip continues from the last finished run. Cleared on resume/discard.
const PAUSED_IDS_SS = 'trainer.pausedActivities'
// The campaign currently showing the inline "kill the process?" pause confirmation (survives the
// activity block's frequent re-renders, since it lives here, not in the DOM).
let pausePromptId = null
const PROJECT_RECORD_TYPE = 'trainer-project'
const PROJECT_MANIFEST_RECORD_TYPE = 'trainer-project-manifest'
const ENVIRONMENT_RECORD_SUFFIX = '-environment'
const DATASET_RECORD_SUFFIX = '-dataset'
const PAPER_RECORD_SUFFIX = '-paper'
// Approach/paper verdict lifecycle (matches TrainingPaperRecord.status). Drives the verdict badge,
// the verdict filter, and the auto-suggested verdict from measured-vs-hold.
const PAPER_STATUSES = ['untested', 'replicating', 'holds-up', 'fluff']
const QUEUE_RECORD_TYPE = 'trainer-queue'
const SEEN_RECORD_TYPE = 'trainer-seen'
const CHART_PALETTE = ['#3b82f6', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#14b8a6']
const JUDGE_HELP_TEXT =
  "Scores every completed run 0–100. Health-flagged runs are auto-rejected without using the LLM. For the rest, the objective is normalised (best=100) and blended 50/50 with an LLM verdict that weighs stability and how promising the configuration is — so a run can't win on prose alone. Results appear in the Judge column."
const PROPOSE_HELP_TEXT =
  "Sends the manifest's levers, the run history and the verdicts to the LLM and asks for new experiment specs likely to beat the best run. Proposals are validated against the levers, deduped by spec, and land below as untested hypotheses that auto-verify against your runs."
const NO_RUNNERS_HINT = 'No runners paired — manage them in the Compute Runners panel.'
const TABS = [
  { id: 'runs', label: 'Runs' },
  { id: 'hypotheses', label: 'Hypotheses' },
  { id: 'papers', label: 'Papers' },
  { id: 'versions', label: 'Versions' },
  { id: 'datasets', label: 'Datasets' },
  { id: 'environments', label: 'Environments' },
  { id: 'launch', label: 'Launch' },
  { id: 'activity', label: 'Activity' },
  { id: 'xai', label: 'xAI' },
]
// xAI tab state: the selectable analysis criterion + direction, the focused run (Model internals), the
// OFAT lever, and the last recommendation set (so a Run-batch click can launch it by index).
let xaiCriterionKey = 'objective'
let xaiCriterionDir = null
let xaiFocusKey = null
let xaiLever = null
let xaiRecsCache = []
// Per-run LLM narrative records, keyed by run key → {narrative, runKey, runCount, criterionKey, narratedBy, narratedAt}.
const xaiNarrativeCache = new Map()
let narrating = false
// Phase 3: the two levers crossed in the interaction grid (default to the top-2 by surrogate importance).
let xaiInterA = null
let xaiInterB = null
// Hypothesis verdict (auto-derived from matching runs, manually overridable). Drives the badge + filter.
const HYPOTHESIS_VERDICTS = ['untested', 'proven', 'disproved']
const HYPOTHESIS_VERDICT_BADGE = {
  untested: 'is-queued',
  proven: 'is-done',
  disproved: 'is-failed',
}
const HYPOTHESIS_VERDICT_LABEL = { untested: 'untested', proven: 'proven', disproved: 'disproved' }
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
// The lever/version dropdowns collapse under a single header; collapsed, only the
// dropdowns with a non-default selection stay visible (highlighted). Starts collapsed.
let runsDropdownsCollapsed = true
// User-defined numeric filter rules (e.g. "return % > 0"), persisted per project as
// recordType + '-filter-rule' records so they survive reloads. Each: { id, field, op,
// value, active }. Rendered as toggle chips alongside "Hide bad runs".
let customRulesCache = []
// When the add/edit custom-rule popup is open, the id being edited (null = creating new).
let editingCustomRuleId = null
// The flat Runs table renders one page at a time so a large run set doesn't build thousands of
// DOM rows at once (the main cost of "Runs takes a while to show"). Grouped views aggregate to
// few rows, so they stay unpaginated. `runsVisibleKeys` = the keys on the current page (for
// "select all visible").
const RUNS_PAGE_SIZE = 50
let runsPage = 0
let runsVisibleKeys = []
// Total server-side matches for the current pushed filter (the flat view paginates server-side, so
// the count comes from the backend, not runsCache.length).
let runsTotalCount = 0
// Full records for runs that scrolled off the current server page while still open/selected, so a
// detail/compare view resolves them even though they aren't in the page cache.
const runExtraCache = new Map()
let runsCompareKeys = new Set()
// First click on Delete arms it (in-app confirm — window.confirm is blocked in the
// embedding iframe sandbox); the second click within the timeout performs the delete.
let runsDeleteArmed = false
let runsDeleteArmTimer = null
let runsViewMode = 'runs'
// Named environments (env-lever bundles) the user defined for this project.
let environmentsCache = []
// Named datasets (dataset-lever bundles — asset / window / fidelity) the user defined.
let datasetsCache = []
// Approach/paper library records + the active rolled-up-verdict filter ('all' | a paper verdict).
let papersCache = []
let paperVerdictFilter = 'all'
// The active hypothesis verdict filter ('all' | a HYPOTHESIS_VERDICTS value), and the paper-scoped
// add/link sub-form state (`null` | { paperId, mode: 'add' | 'link' }).
let hypothesisVerdictFilter = 'all'
let hypothesisOverrideId = null
let paperSubform = null
// Dataset bundles supplied by an applied preset (an experiment that sweeps datasets); when set they
// override the launch picker's selection. Cleared on reset / manual picker change.
let launchPresetDatasets = []
// Environment bundles supplied by an applied preset (an experiment that sweeps exit/fee regimes);
// mirrors launchPresetDatasets — overrides the environment picker until reset / manual change.
let launchPresetEnvironments = []
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
let toastTimer = null
// A transient in-app popup (window.alert is blocked in the embedding iframe sandbox).
function showToast(message) {
  let el = byId('app-toast')
  if (!el) {
    el = document.createElement('div')
    el.id = 'app-toast'
    el.className = 'app-toast'
    document.body.appendChild(el)
  }
  el.textContent = message
  el.classList.add('is-visible')
  if (toastTimer) clearTimeout(toastTimer)
  toastTimer = setTimeout(() => el.classList.remove('is-visible'), 3500)
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
function recordToRun(r) {
  const summary = r.content || {}
  const key =
    r.key || (summary.provenance && summary.provenance.configHash) || summary.configHash || ''
  return { key, summary }
}
// Run records via the bridge, with the server-side filter/sort/pagination passed through (`extra` =
// { where?, orderBy?, limit?, offset? }). The flat view sends a page; group-by/drill send no limit.
async function queryRunRecords(extra) {
  if (!embedded() || !manifest) return []
  try {
    const payload = { type: manifest.recordType }
    if (extra && extra.where) payload.where = extra.where
    if (extra && extra.orderBy) payload.orderBy = extra.orderBy
    if (extra && extra.limit !== undefined) payload.limit = extra.limit
    if (extra && extra.offset !== undefined) payload.offset = extra.offset
    const recs = await window.OverseerBridge.queryData(payload)
    return (recs || []).map(recordToRun).filter((r) => r.key)
  } catch {
    return []
  }
}
// Total runs matching `where` (ignores limit/offset), for the flat view's pager. Degrades to the
// loaded length when the host predates the count verb.
async function countRunRecords(where) {
  if (!embedded() || !manifest || !window.OverseerBridge.countData) return null
  try {
    const payload = { type: manifest.recordType }
    if (where) payload.where = where
    const res = await window.OverseerBridge.countData(payload)
    return Number((res && res.count) || 0)
  } catch {
    return null
  }
}
async function readRuns() {
  if (!manifest) return []
  const where = buildRunsServerWhere()
  if (runsServerPaged()) {
    const [page, total] = await Promise.all([
      queryRunRecords({
        where,
        orderBy: runsServerOrderBy(),
        limit: RUNS_PAGE_SIZE,
        offset: runsPage * RUNS_PAGE_SIZE,
      }),
      countRunRecords(where),
    ])
    runsTotalCount = total === null ? page.length : total
    return page
  }
  const all = await queryRunRecords({ where })
  runsTotalCount = all.length
  return all
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
  return (manifest && manifest.pipelineVersion) || '1.0'
}
// Pipeline versions are "major.minor" (a bare "1" reads as 1.0). MAJOR is breaking — a data/scoring
// change that makes prior runs incomparable, so they need a re-run; MINOR is additive and comparable.
function parsePipelineVersion(v) {
  const m = /^(\d+)(?:\.(\d+))?/.exec(String(v == null ? '1' : v))
  return { major: m ? Number(m[1]) : 1, minor: m && m[2] !== undefined ? Number(m[2]) : 0 }
}
// A run is OUTDATED (worth re-running under the latest pipeline) only when its MAJOR trails the current
// one — a minor gap stays comparable, so it doesn't.
function runIsOutdated(r) {
  return (
    parsePipelineVersion(runVersionOf(r)).major <
    parsePipelineVersion(currentPipelineVersion()).major
  )
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
// User-defined numeric filter rules, one record each (keyed by rule id) so they persist
// across reloads + clients. Returns them sorted by creation order for a stable chip layout.
async function readCustomRules() {
  if (!manifest) return []
  const recs = await queryRecords(manifest.recordType + '-filter-rule')
  return recs
    .map((r) => {
      const c = r.content || {}
      return { ...c, id: c.id || r.key || '' }
    })
    .filter((c) => c.id && c.field && c.op)
    .sort((a, b) => String(a.createdAt || '').localeCompare(String(b.createdAt || '')))
}
async function saveCustomRule(rule) {
  if (!manifest || !rule || !rule.id) return
  await window.OverseerBridge.putData({
    type: manifest.recordType + '-filter-rule',
    key: rule.id,
    content: rule,
  })
  customRulesCache = await readCustomRules()
}
async function deleteCustomRule(id) {
  if (!manifest || !id) return
  await window.OverseerBridge.deleteData({ type: manifest.recordType + '-filter-rule', key: id })
  customRulesCache = customRulesCache.filter((r) => r.id !== id)
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
    if (activeTabId === 'runs') refreshRuns()
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
  xaiNarrativeCache.clear()
  narrating = false
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
  datasetsCache = []
  papersCache = []
  paperSubform = null
  launchPresetDatasets = []
  launchPresetEnvironments = []
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
  datasetsCache = hasDatasetLevers() ? await readDatasets() : []
  if (epoch !== projectEpoch) return
  // On-load record migrations: the hold-fee metrics fix over every run + the legacy hypothesis/model
  // normalization.
  await migrateAllRunRecords()
  await migrateModelsToHypotheses()
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
// Whether an activity type runs on compute (experiment lane) vs the lighter task lane.
function isExperimentActivityType(type) {
  return EXPERIMENT_ACTIVITY_TYPES.has(type)
}
// How many slots one lane is using — only RUNNING/starting activities count (a paused or
// stalled-waiting-to-resume entry isn't consuming compute, so it shouldn't block new launches).
function laneSlotCount(experiment) {
  let n = 0
  for (const a of liveActivities.values()) {
    if (a.status !== 'running' && a.status !== 'starting') continue
    if (isExperimentActivityType(a.activityType) === experiment) n++
  }
  return n
}
function experimentSlotCount() {
  return laneSlotCount(true)
}
function taskSlotCount() {
  return laneSlotCount(false)
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
  const experiment = isExperimentActivityType(activityType)
  const slots = experiment ? experimentSlotCount() : taskSlotCount()
  const budget = experiment ? savedExperimentBudget() : savedTaskBudget()
  if (slots < budget) {
    const activityId = await launchActivity(item)
    if (activityId) return { started: true, activityId }
  }
  await putQueueItem(item)
  await refreshQueue()
  const queue = await readQueue()
  const ahead = queue.filter((q) => isExperimentActivityType(q.activityType) === experiment).length
  return { queued: true, id: item.id, ahead }
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
// Drain BOTH lanes until each is full to its own budget. The lanes are independent: a full
// experiment lane never holds back a queued task, and vice versa. Each dispatched activity
// observes itself (non-blocking) and re-pumps when it settles, freeing its slot for the next item.
async function pumpQueue() {
  if (queuePumping || !embedded() || !manifest) return
  const epoch = projectEpoch
  queuePumping = true
  try {
    // A lane that hits a transient launch failure is skipped for the rest of this pump so we
    // never lose the queue or hammer a failing launch; a later pump (settle / focus) retries.
    const lanes = [
      {
        isExp: true,
        slotCount: experimentSlotCount,
        budget: savedExperimentBudget,
        blocked: false,
      },
      { isExp: false, slotCount: taskSlotCount, budget: savedTaskBudget, blocked: false },
    ]
    let progressing = true
    while (progressing && epoch === projectEpoch) {
      progressing = false
      const queue = await readQueue()
      if (epoch !== projectEpoch || !queue.length) break
      for (const lane of lanes) {
        if (epoch !== projectEpoch || lane.blocked) continue
        if (lane.slotCount() >= lane.budget()) continue
        const head = queue.find((q) => isExperimentActivityType(q.activityType) === lane.isExp)
        if (!head) continue
        await deleteQueueItem(head.id)
        queueCache = queueCache.filter((q) => q.id !== head.id)
        const activityId = await launchActivity(head)
        if (activityId) {
          progressing = true
        } else {
          await putQueueItem(head)
          queueCache = await readQueue()
          lane.blocked = true
        }
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
    if (activeTabId === 'xai') renderXai()
  } else if (item.activityType === 'xai-narrate') {
    narrating = busy
    if (activeTabId === 'xai') renderXai()
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
    if (activeTabId === 'xai') renderXai()
  } else if (item.activityType === 'xai-narrate') {
    await loadXaiNarrative(item.params && item.params.runKey)
    if (activeTabId === 'xai') renderXai()
    setStatusLine('xai-status', quickActivityFailureText(act, 'Narrating'), true)
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
// Settle-time bookkeeping shared by the observe loop and project open: stamp finished campaigns into
// their hypotheses, consume auto-eval markers, then re-evaluate every hypothesis against the now-current
// runs so a settled campaign flips verdicts (and records the transitions) even off the Hypotheses tab.
async function processSettledCampaignEffects() {
  await stampHypothesisCampaignResults()
  await processAutoEvalMarkers()
  runsCache = await readRuns()
  await refreshHypothesisVerdicts(await readHypotheses())
  if (activeTabId === 'hypotheses') await renderHypotheses()
  if (activeTabId === 'papers') await renderPapers()
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
  runsPage = 0
  refreshRuns()
}
// Drill from a by-experiment row into that thesis's individual runs.
function drillIntoExperiment(thesis) {
  const group = aggregateByExperiment(applyRunsFilters(runsCache)).find((g) => g.thesis === thesis)
  if (!group) return
  runsFilterKeys = new Set(group.runs.map((r) => r.key))
  runsFilterLabel = thesis
  runsViewMode = 'runs'
  runsPage = 0
  refreshRuns()
}
// Drill from a by-environment row into that environment's runs.
function drillIntoEnvironment(sig) {
  const group = aggregateByEnvironment(applyRunsFilters(runsCache)).find((g) => g.sig === sig)
  if (!group) return
  runsFilterKeys = new Set(group.runs.map((r) => r.key))
  runsFilterLabel = group.name
  runsViewMode = 'runs'
  runsPage = 0
  refreshRuns()
}
// Drill from a by-dataset row into that dataset's runs.
function drillIntoDataset(sig) {
  const group = aggregateByDataset(applyRunsFilters(runsCache)).find((g) => g.sig === sig)
  if (!group) return
  runsFilterKeys = new Set(group.runs.map((r) => r.key))
  runsFilterLabel = group.name
  runsViewMode = 'runs'
  runsPage = 0
  refreshRuns()
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
  'stop_losses',
  'final_net_worth',
  'hold_return_pct',
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
  if (mk === 'total_return_pct' || mk === 'return_vs_hold_pct')
    return v >= 0 ? 'delta-pos' : 'delta-neg'
  if (mk === 'n_trades') return v > DEGENERATE_TRADE_COUNT ? 'delta-pos' : 'delta-neg'
  return ''
}
// Plain-language help per metric (shown as a "?" on the column header) so a newcomer
// knows what each means + which direction is good.
const METRIC_INFO = {
  total_return_pct:
    'Realized return over the test window — the SUM of per-trade P&L (a position still open at the end is closed at the last price). Higher is better.',
  traded_return:
    'The objective: realized return gated by trade frequency (under-trading is punished).',
  win_pct: 'Share of trades that were profitable.',
  n_trades: 'Number of trades (executed exits — agent sells plus auto TP/trailing/SL closes).',
  stop_losses: 'How many trades were closed by the stop-loss.',
  final_net_worth: 'Ending portfolio value (initial + realized P&L).',
  hold_return_pct:
    'Buy-and-hold over the same window — a yardstick to beat, never the optimisation target.',
  return_vs_hold_pct:
    'Realized return minus buy-and-hold (indicative; the capital base differs from the fixed-stake strategy).',
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
  hold_return_pct: 'hold %',
  return_vs_hold_pct: 'vs hold',
  realized_cost_bps: 'cost (bps)',
}
// Metrics shown to a fixed 2 decimals in the table (money + ratios read cleaner that way).
const TWO_DP_METRICS = new Set([
  'win_pct',
  'final_net_worth',
  'hold_return_pct',
  'return_vs_hold_pct',
  'realized_cost_bps',
])
// Metrics surfaced only in a single run's DETAIL (too granular for the table); kept
// out of the table's metric columns but still rendered by metricsTableHtml.
// Retired metrics — no longer produced (summary.py dropped Sharpe/CAGR/max-drawdown + the window
// breakdown as noise for the trade-frequency objective). Hidden EVERYWHERE so OLD runs that still
// carry them in storage don't resurrect dead columns/rows. New runs don't emit them at all.
const RETIRED_METRICS = [
  'sharpe',
  'cagr_pct',
  'max_drawdown_pct',
  'sharpe_alpha',
  'worst_window_return_pct',
  'windows_profitable_pct',
]
// `return_vs_hold_pct` has its own dedicated, formatted "vs hold" column (see runsColumns); `hold_net_of_fees`
// is a provenance flag, not a number. Both are kept out of the generic metric columns so neither double-shows.
const TABLE_HIDDEN_METRICS = new Set([
  'trade_gate',
  'traded_return',
  'return_vs_hold_pct',
  'hold_net_of_fees',
  ...RETIRED_METRICS,
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
  return String((r.summary && r.summary.pipelineVersion) || '1.0')
}
// The comparison operators a custom numeric rule can use, in menu order.
const CUSTOM_RULE_OPS = ['<', '<=', '=', '>=', '>']
function compareWithOp(value, op, target) {
  switch (op) {
    case '<':
      return value < target
    case '<=':
      return value <= target
    case '=':
      return value === target
    case '>=':
      return value >= target
    case '>':
      return value > target
    default:
      return false
  }
}
// Every run field a custom rule can test — those carrying NUMERIC values: the objective,
// each measured metric, vs-hold (when a benchmark exists), and every numeric-typed lever.
// Each entry is { key (stable id), label (for chips/menus), get(run) -> number }.
function numericRunFields() {
  const fields = [
    {
      key: 'objective',
      label: objectiveName(),
      get: (r) => Number(r.summary && r.summary.objective),
    },
  ]
  for (const mk of runMetricKeys()) {
    fields.push({
      key: 'metric:' + mk,
      label: metricLabel(mk),
      get: (r) => Number(r.summary && r.summary.metrics && r.summary.metrics[mk]),
    })
  }
  if (anyBenchmark()) {
    fields.push({ key: 'vs_hold', label: 'vs hold', get: (r) => vsHoldValue(r.summary) })
  }
  for (const [key, spec] of leverEntries()) {
    if (spec && spec.type === 'number') {
      fields.push({
        key: 'config:' + key,
        label: key,
        get: (r) => Number((r.summary && r.summary.config && r.summary.config[key]) ?? NaN),
      })
    }
  }
  return fields
}
function customRuleFieldByKey(key) {
  return numericRunFields().find((f) => f.key === key)
}
// A run passes a rule when its field value is a finite number satisfying the comparison;
// a missing/non-numeric value (e.g. a failed run with no metrics) never passes.
function runMatchesCustomRule(run, rule) {
  const field = customRuleFieldByKey(rule.field)
  if (!field) return true
  const v = field.get(run)
  if (!Number.isFinite(v)) return false
  return compareWithOp(v, rule.op, Number(rule.value))
}
// A short human label for a rule, e.g. "return % > 0".
function customRuleLabel(rule) {
  const field = customRuleFieldByKey(rule.field)
  return `${field ? field.label : rule.field} ${rule.op} ${rule.value}`
}
// A custom-rule field maps to a stored content dot-path when it can be pushed to the server; a
// computed field (vs_hold) returns null and stays a client-side filter.
function customRuleServerField(field) {
  if (field === 'objective') return 'objective'
  if (field.startsWith('metric:')) return 'metrics.' + field.slice(7)
  if (field.startsWith('config:')) return 'config.' + field.slice(7)
  return null
}
// The filters that push DOWN to the server query: lever-equals, pipeline version, and the pushable
// numeric rules. Text search, Hide-bad, vs-hold rules, and a group-by drill stay client-side.
function buildRunsServerWhere() {
  const preds = []
  for (const [lever, val] of Object.entries(runsLeverFilter)) {
    if (val) preds.push({ field: 'config.' + lever, op: '=', value: String(val) })
  }
  if (runsVersionFilter) {
    preds.push({ field: 'pipelineVersion', op: '=', value: String(runsVersionFilter) })
  }
  for (const rule of customRulesCache) {
    if (!rule.active) continue
    const field = customRuleServerField(rule.field)
    if (field) preds.push({ field, op: rule.op, value: rule.value })
  }
  // Hide-bad is an internal criterion, pushed server-side like the custom rules so each page stays
  // full instead of being thinned after the fact.
  if (runsHideBad) preds.push(runsHideBadWhere())
  return preds.length ? { and: preds } : undefined
}
// The server-side negation of runIsBad: keep runs that did NOT fail, are NOT health-flagged, and
// did NOT under-trade (n_trades > DEGENERATE_TRADE_COUNT). `not(<=)` also keeps runs whose n_trades
// is absent/non-numeric — matching the client's "finite n and n<=2" badness test.
function runsHideBadWhere() {
  return {
    and: [
      { not: { field: 'status', op: '=', value: 'failed' } },
      {
        or: [
          { not: { field: 'health.status', op: 'exists' } },
          { field: 'health.status', op: '=', value: 'ok' },
        ],
      },
      { not: { field: 'metrics.n_trades', op: '<=', value: DEGENERATE_TRADE_COUNT } },
    ],
  }
}
// Server ordering for the flat view. The "Ran at" column must sort by the run's actual ran-at (what
// it displays: provenance.ranAt, else ranAt) — NOT the entity's updated_at, which a later judge/eval
// bumps without re-running, which is why most rows look right but edited ones don't. Metric/version/
// duration columns map to their stored field; id/data have no server-sortable key (recency stands).
function runsServerOrderBy() {
  if (!runsSortKey) return undefined
  if (runsSortKey === 'ran') {
    return [
      { field: 'provenance.ranAt', direction: runsSortDir, numeric: false },
      { field: 'ranAt', direction: runsSortDir, numeric: false },
    ]
  }
  let field = null
  if (runsSortKey.startsWith('m:')) field = 'metrics.' + runsSortKey.slice(2)
  else if (runsSortKey === 'version') field = 'pipelineVersion'
  else if (runsSortKey === 'durationMs') field = 'durationMs'
  if (!field) return undefined
  return [{ field, direction: runsSortDir, numeric: field !== 'pipelineVersion' }]
}
// The flat Runs view paginates server-side; the group-by views and a setup/experiment drill need
// the full matching set in memory, so they fetch unpaginated.
function runsServerPaged() {
  return runsViewMode === 'runs' && !runsFilterKeys
}
// Resolve a run by key from the current page cache, falling back to records stashed when a run was
// opened/selected (so detail + compare survive paging away from the run).
function findRun(key) {
  return runsCache.find((r) => r.key === key) || runExtraCache.get(key)
}
function rememberRun(key) {
  const run = runsCache.find((r) => r.key === key)
  if (run) runExtraCache.set(key, run)
}
function applyRunsFilters(runs) {
  let out = runsFilterKeys ? runs.filter((r) => runsFilterKeys.has(r.key)) : runs
  if (runsHideBad) out = out.filter((r) => !runIsBad(r))
  if (runsVersionFilter) out = out.filter((r) => runVersionOf(r) === runsVersionFilter)
  for (const [lever, val] of Object.entries(runsLeverFilter)) {
    if (val) out = out.filter((r) => String((r.summary.config || {})[lever]) === String(val))
  }
  for (const rule of customRulesCache) {
    if (rule.active) out = out.filter((r) => runMatchesCustomRule(r, rule))
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
// One toggle chip per saved custom numeric rule + a trailing "+" to add one. A chip is a
// checkbox (activate/deactivate), the rule text (click to edit), and an ✕ (delete).
function customTogglesHtml() {
  const chips = customRulesCache
    .map(
      (rule) =>
        `<span class="filter-chip${rule.active ? ' is-on' : ''}" data-rule-edit="${escapeHtml(rule.id)}" title="Click to edit this filter">
          <input type="checkbox" class="filter-chip-cb"${rule.active ? ' checked' : ''} data-rule-toggle="${escapeHtml(rule.id)}" aria-label="Activate filter ${escapeHtml(customRuleLabel(rule))}" />
          <span class="filter-chip-text">${escapeHtml(customRuleLabel(rule))}</span>
          <button type="button" class="filter-chip-x" data-rule-del="${escapeHtml(rule.id)}" title="Delete this filter" aria-label="Delete filter">✕</button>
        </span>`,
    )
    .join('')
  return `<span class="runs-custom-toggles">${chips}<button type="button" id="runs-add-toggle" class="runs-add-toggle" title="Add a custom numeric filter (e.g. return % > 0)" aria-label="Add a custom filter">+</button></span>`
}
// The add/edit popup for a custom numeric rule: pick a field, a comparison, a value. Opens
// blank to CREATE, or pre-filled (editingCustomRuleId set) to EDIT an existing rule.
function openCustomRulePopup(ruleId) {
  editingCustomRuleId = ruleId || null
  renderCustomRulePopup()
}
function closeCustomRulePopup() {
  editingCustomRuleId = null
  const m = byId('custom-rule-modal')
  if (m) m.hidden = true
}
function renderCustomRulePopup() {
  const fields = numericRunFields()
  if (!fields.length) return
  const editing = editingCustomRuleId
    ? customRulesCache.find((r) => r.id === editingCustomRuleId)
    : null
  const selectedField = editing ? editing.field : fields[0].key
  const selectedOp = editing ? editing.op : '>'
  const selectedValue = editing ? String(editing.value) : ''
  let modal = byId('custom-rule-modal')
  if (!modal) {
    modal = document.createElement('div')
    modal.id = 'custom-rule-modal'
    modal.className = 'chart-modal'
    document.body.appendChild(modal)
    modal.addEventListener('click', (event) => {
      if (event.target === modal || event.target.closest('[data-rule-cancel]')) {
        return closeCustomRulePopup()
      }
    })
    modal.addEventListener('submit', (event) => {
      if (event.target.closest('#custom-rule-form')) {
        event.preventDefault()
        submitCustomRulePopup()
      }
    })
  }
  const fieldOpts = fields
    .map(
      (f) =>
        `<option value="${escapeHtml(f.key)}"${f.key === selectedField ? ' selected' : ''}>${escapeHtml(f.label)}</option>`,
    )
    .join('')
  const opOpts = CUSTOM_RULE_OPS.map(
    (op) => `<option value="${op}"${op === selectedOp ? ' selected' : ''}>${op}</option>`,
  ).join('')
  modal.innerHTML = `<div class="chart-modal__backdrop" data-rule-cancel></div>
    <div class="chart-modal__panel custom-rule-panel" role="dialog" aria-label="${editing ? 'Edit' : 'Add'} a custom filter">
      <div class="chart-modal__head">
        <strong>${editing ? 'Edit' : 'Add'} a filter</strong>
        <button type="button" class="icon-btn" data-rule-cancel title="Close" aria-label="Close">✕</button>
      </div>
      <form id="custom-rule-form" class="custom-rule-form">
        <p class="card-sub">Keep only runs whose value satisfies this rule. Applies to numeric fields.</p>
        <div class="custom-rule-row">
          <select id="custom-rule-field" aria-label="Field">${fieldOpts}</select>
          <select id="custom-rule-op" aria-label="Comparison">${opOpts}</select>
          <input type="number" id="custom-rule-value" step="any" placeholder="value" value="${escapeHtml(selectedValue)}" aria-label="Value" />
        </div>
        <div class="custom-rule-actions">
          <button type="button" class="ghost-btn" data-rule-cancel>Cancel</button>
          <button type="submit" class="ghost-btn is-primary">${editing ? 'Save' : 'Create'}</button>
        </div>
      </form>
    </div>`
  modal.hidden = false
  const valueInput = byId('custom-rule-value')
  if (valueInput) valueInput.focus()
}
async function submitCustomRulePopup() {
  const fieldEl = byId('custom-rule-field')
  const opEl = byId('custom-rule-op')
  const valueEl = byId('custom-rule-value')
  if (!fieldEl || !opEl || !valueEl) return
  const value = Number(valueEl.value)
  if (valueEl.value === '' || !Number.isFinite(value)) {
    valueEl.focus()
    return
  }
  const editing = editingCustomRuleId
    ? customRulesCache.find((r) => r.id === editingCustomRuleId)
    : null
  const rule = {
    id: editing ? editing.id : randomHexId(),
    field: fieldEl.value,
    op: opEl.value,
    value,
    active: editing ? editing.active !== false : true,
    createdAt: editing ? editing.createdAt || nowIso() : nowIso(),
    updatedAt: nowIso(),
  }
  await saveCustomRule(rule)
  closeCustomRulePopup()
  runsPage = 0
  refreshRuns()
}
function runsToolbarHtml(shownCount, total) {
  // Lever choice dropdowns + the pipeline-version dropdown live in the collapsible panel.
  // Each carries `is-changed` (highlighted) when it has a non-default selection; collapsed,
  // CSS shows only those.
  // Filter options = the manifest choices UNION the values actually present in the runs (so migration's
  // `legacy:…` dataset tags are filterable), minus input-only synonyms (e.g. `fidelity_set: auto`) that no
  // stored run carries — unless one actually does. So the dropdown reflects what you can really filter to.
  const synonyms = new Set((window.Migrate && window.Migrate.INPUT_SYNONYMS) || [])
  const leverDropdowns = leverEntries()
    .filter(([, spec]) => spec.type === 'choice')
    .map(([key, spec]) => {
      const selected = String(runsLeverFilter[key] || '')
      const present = new Set(
        runsCache
          .map((r) => (r.summary.config || {})[key])
          .filter((v) => v !== undefined && v !== null)
          .map(String),
      )
      const values = [...new Set([...(spec.choices || []).map(String), ...present])].filter(
        (v) => present.has(v) || !synonyms.has(v),
      )
      const opts = [`<option value="">${escapeHtml(key)}: any</option>`]
        .concat(
          values.map(
            (c) =>
              `<option value="${escapeHtml(c)}"${selected === c ? ' selected' : ''}>${escapeHtml(c)}</option>`,
          ),
        )
        .join('')
      return `<select class="runs-filter-lever${selected ? ' is-changed' : ''}" data-lever="${escapeHtml(key)}">${opts}</select>`
    })
    .join('')
  const versions = [
    ...new Set([
      ...runsCache.map(runVersionOf),
      String((manifest && manifest.pipelineVersion) || '1'),
    ]),
  ].sort()
  const versionFilter = `<select class="runs-filter-lever${runsVersionFilter ? ' is-changed' : ''}" id="runs-version-filter"${helpAttr("Show only runs from one pipeline version — cross-version scores aren't comparable. Set automatically when you open a version from the Versions tab.")}>
          <option value="">version: any</option>
          ${versions.map((v) => `<option value="${escapeHtml(v)}"${runsVersionFilter === v ? ' selected' : ''}>v${escapeHtml(v)}</option>`).join('')}
        </select>`
  const changedDropdowns =
    (runsVersionFilter ? 1 : 0) + Object.values(runsLeverFilter).filter(Boolean).length
  const dropdownsToggle = `<button type="button" id="runs-dropdowns-toggle" class="runs-dropdowns-toggle" aria-expanded="${runsDropdownsCollapsed ? 'false' : 'true'}">
    <span class="caret">${runsDropdownsCollapsed ? '▸' : '▾'}</span> ${runsDropdownsCollapsed ? 'More filter options' : 'Hide filter options'}${runsDropdownsCollapsed && changedDropdowns ? ` <span class="runs-dropdowns-count">${changedDropdowns}</span>` : ''}
  </button>`
  const dropdownsPanel = `<div id="runs-dropdowns" class="runs-dropdowns${runsDropdownsCollapsed ? ' is-collapsed' : ''}">
    ${dropdownsToggle}
    <div class="runs-dropdowns-body">${versionFilter}${leverDropdowns}</div>
  </div>`

  const active =
    runsFilterKeys ||
    runsTextFilter ||
    runsVersionFilter ||
    Object.values(runsLeverFilter).some(Boolean)
  const label = runsFilterLabel ? ` (${escapeHtml(runsFilterLabel)})` : ''
  const envViewBtn = hasEnvLevers()
    ? `<button type="button" class="runs-view-btn${runsViewMode === 'environment' ? ' is-active' : ''}" data-view="environment"${helpAttr('Group runs by the ENVIRONMENT they ran in (fee / TP-SL regime), so you can see how a model holds up across regimes.')}>By environment</button>`
    : ''
  const datasetViewBtn = hasDatasetLevers()
    ? `<button type="button" class="runs-view-btn${runsViewMode === 'dataset' ? ' is-active' : ''}" data-view="dataset"${helpAttr('Group runs by the DATASET they ran on (asset / walk-forward window / fidelity stack), so you can see how a model holds up across datasets.')}>By dataset</button>`
    : ''
  const toggle = `<div class="runs-viewmode">
    <button type="button" class="runs-view-btn${runsViewMode === 'runs' ? ' is-active' : ''}" data-view="runs">Runs</button><button type="button" class="runs-view-btn${runsViewMode === 'experiment' ? ' is-active' : ''}" data-view="experiment"${helpAttr('Group runs by the THESIS set at launch, so experiments compare head-to-head (incl. theses outside the levers).')}>By experiment</button>${datasetViewBtn}${envViewBtn}
  </div>`
  const hideBad = `<label class="runs-hidebad" title="Hide failed/errored runs and degenerate results (≤${DEGENERATE_TRADE_COUNT} trades or health-flagged).">
    <input type="checkbox" id="runs-hide-bad"${runsHideBad ? ' checked' : ''} /> Hide bad runs
  </label>`
  return `<div class="runs-toolbar">
    ${toggle}
    ${dropdownsPanel}
    <div class="runs-filters">
      <input type="search" id="runs-filter-text" class="runs-filter-text" placeholder="filter config / key…" value="${escapeHtml(runsTextFilter)}" />
      ${hideBad}
      ${customTogglesHtml()}
      <span class="runs-count">${shownCount}/${total} runs${label}</span>
      ${active ? '<button type="button" id="runs-filter-clear" class="ghost-btn">clear</button>' : ''}
    </div>
  </div>`
}
function toggleRunsSort(id) {
  if (runsSortKey === id) runsSortDir = runsSortDir === 'asc' ? 'desc' : 'asc'
  else {
    runsSortKey = id
    runsSortDir = 'desc'
  }
  runsPage = 0
  refreshRuns()
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
function runMetricValue(run, key) {
  const v = Number(((run && run.summary && run.summary.metrics) || {})[key])
  return Number.isFinite(v) ? v : NaN
}
function formatSignedPct(v) {
  if (!Number.isFinite(v)) return '—'
  return `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`
}
function aggregateByDataset(runs) {
  const groups = new Map()
  for (const r of runs) {
    const sig = runDatasetSignature(r)
    if (!groups.has(sig)) groups.set(sig, { name: runDatasetName(r), sig, runs: [] })
    groups.get(sig).runs.push(r)
  }
  const out = []
  for (const g of groups.values()) {
    const objs = g.runs.map((r) => Number(r.summary.objective)).filter(Number.isFinite)
    const vh = g.runs.map((r) => runMetricValue(r, 'return_vs_hold_pct')).filter(Number.isFinite)
    out.push({
      name: g.name,
      sig: g.sig,
      runs: g.runs,
      count: g.runs.length,
      setups: new Set(g.runs.map(setupKeyOfRun)).size,
      objMin: objs.length ? Math.min(...objs) : NaN,
      objMax: objs.length ? Math.max(...objs) : NaN,
      objAvg: mean(objs),
      vhAvg: vh.length ? mean(vh) : NaN,
      vhWorst: vh.length ? Math.min(...vh) : NaN,
    })
  }
  return out
}
// A run's MODEL signature: its config minus seed AND minus dataset levers, so the same model config
// run across several datasets (e.g. walk-forward windows) shares one signature — the key for judging
// how a config holds up ACROSS datasets rather than being tuned to one lucky one.
function modelSignatureOfRun(run) {
  const cfg = { ...((run.summary && run.summary.config) || {}) }
  delete cfg.seed
  for (const [key] of datasetLeverEntries()) delete cfg[key]
  return JSON.stringify(
    Object.keys(cfg)
      .sort()
      .map((k) => [k, cfg[k]]),
  )
}
// Per model-config robustness ACROSS the datasets it ran on: each dataset is averaged first (so seed
// counts don't bias), then reduced to a mean and a WORST dataset. Only configs spanning ≥2 datasets
// are returned. "worst" follows the objective direction (the weakest window for a max objective).
function aggregateRobustnessAcrossDatasets(runs) {
  const dir = objectiveDirection()
  const bySig = new Map()
  for (const r of runs) {
    if (!r.summary || r.summary.status === 'failed') continue
    const sig = modelSignatureOfRun(r)
    if (!bySig.has(sig)) bySig.set(sig, [])
    bySig.get(sig).push(r)
  }
  const out = []
  for (const rs of bySig.values()) {
    const byDataset = new Map()
    for (const r of rs) {
      const ds = runDatasetSignature(r)
      if (!byDataset.has(ds)) byDataset.set(ds, { objs: [], vh: [] })
      const obj = Number(r.summary.objective)
      if (Number.isFinite(obj)) byDataset.get(ds).objs.push(obj)
      const v = runMetricValue(r, 'return_vs_hold_pct')
      if (Number.isFinite(v)) byDataset.get(ds).vh.push(v)
    }
    if (byDataset.size < 2) continue
    const objMeans = [...byDataset.values()].map((d) => mean(d.objs)).filter(Number.isFinite)
    const vhMeans = [...byDataset.values()].map((d) => mean(d.vh)).filter(Number.isFinite)
    out.push({
      config: (rs[0].summary && rs[0].summary.config) || {},
      datasets: byDataset.size,
      runs: rs.length,
      objMean: mean(objMeans),
      objWorst: objMeans.length
        ? dir === 'min'
          ? Math.max(...objMeans)
          : Math.min(...objMeans)
        : NaN,
      vhMean: vhMeans.length ? mean(vhMeans) : NaN,
      vhWorst: vhMeans.length ? Math.min(...vhMeans) : NaN,
    })
  }
  out.sort((a, b) => {
    const fa = Number.isFinite(a.objWorst)
    const fb = Number.isFinite(b.objWorst)
    if (fa && fb) return dir === 'min' ? a.objWorst - b.objWorst : b.objWorst - a.objWorst
    return fa ? -1 : fb ? 1 : 0
  })
  return out
}
// A warning shown when the runs in view span more than one dataset (e.g. several walk-forward
// windows): they are separate out-of-sample samples and must not be read as a single number.
function mixedDatasetBannerHtml(filtered) {
  if (!hasDatasetLevers()) return ''
  const withSummary = filtered.filter((r) => r.summary)
  const sigs = new Set(withSummary.map(runDatasetSignature))
  if (sigs.size < 2) return ''
  const names = [...new Set(withSummary.map(runDatasetName))]
  const shown = names.slice(0, 6).join(', ') + (names.length > 6 ? '…' : '')
  return `<p class="mixed-dataset-banner">⚠ These runs span ${sigs.size} datasets (${escapeHtml(shown)}) — each is a separate out-of-sample sample. Compare per-dataset; don't read across them as one number.</p>`
}
// Compact "how robust is each config across datasets" table for the By-dataset view.
function robustnessAcrossDatasetsHtml(filtered) {
  const rows = aggregateRobustnessAcrossDatasets(filtered)
  if (!rows.length) return ''
  const on = escapeHtml(objectiveName())
  const hasVh = rows.some((r) => Number.isFinite(r.vhMean))
  const vhHead = hasVh ? '<th class="num">vs hold avg</th><th class="num">vs hold worst</th>' : ''
  const body = rows
    .slice(0, 12)
    .map((r) => {
      const vhCells = hasVh
        ? `<td class="num ${metricColorClass('return_vs_hold_pct', r.vhMean)}">${escapeHtml(formatSignedPct(r.vhMean))}</td><td class="num ${metricColorClass('return_vs_hold_pct', r.vhWorst)}">${escapeHtml(formatSignedPct(r.vhWorst))}</td>`
        : ''
      return `<tr>
        <td class="card-sub">${escapeHtml(setupConfigLabel(r.config))}</td>
        <td class="num">${r.datasets}</td>
        <td class="num">${escapeHtml(formatObjective(r.objMean))}</td>
        <td class="num">${escapeHtml(formatObjective(r.objWorst))}</td>
        ${vhCells}</tr>`
    })
    .join('')
  return `<div class="robustness-block">
    <h4 class="robustness-title">Robustness across datasets</h4>
    <p class="runs-legend">Each row is one model config aggregated across the datasets it ran on (each dataset averaged first, so seed counts don't bias). "worst" = its weakest dataset — a config tuned to one lucky window shows a strong avg but a weak worst.</p>
    <div class="table-wrap"><table class="runs-table">
      <thead><tr><th>Config</th><th class="num">datasets</th><th class="num">${on} avg</th><th class="num">${on} worst</th>${vhHead}</tr></thead>
      <tbody>${body}</tbody></table></div>
  </div>`
}
function byDatasetTableHtml(filtered) {
  const dir = objectiveDirection()
  const groups = aggregateByDataset(filtered).sort((a, b) => {
    const fa = Number.isFinite(a.objMax)
    const fb = Number.isFinite(b.objMax)
    if (fa && fb) return dir === 'min' ? a.objMin - b.objMin : b.objMax - a.objMax
    return fa ? -1 : fb ? 1 : 0
  })
  const on = escapeHtml(objectiveName())
  const hasVh = groups.some((g) => Number.isFinite(g.vhAvg))
  const rows = groups
    .map((g) => {
      const range = Number.isFinite(g.objMin)
        ? `${escapeHtml(formatObjective(g.objMin))} – ${escapeHtml(formatObjective(g.objMax))}`
        : '—'
      const vhCell = hasVh
        ? `<td class="num ${metricColorClass('return_vs_hold_pct', g.vhAvg)}">${escapeHtml(formatSignedPct(g.vhAvg))}</td>`
        : ''
      return `<tr data-dataset-sig="${escapeHtml(g.sig)}" class="setup-row">
        <td>${escapeHtml(g.name)}</td>
        <td class="card-sub">${escapeHtml(g.sig)}</td>
        <td class="num">${g.count}</td>
        <td class="num">${g.setups}</td>
        <td class="num">${escapeHtml(formatObjective(g.objAvg))}</td>
        <td class="num">${range}</td>
        ${vhCell}
      </tr>`
    })
    .join('')
  const vhHead = hasVh ? `<th class="num"${helpAttr(VSHOLD_INFO)}>vs hold avg</th>` : ''
  return `<div class="table-wrap"><table class="runs-table">
    <thead><tr><th>Dataset</th><th>Settings</th><th class="num">runs</th><th class="num">setups</th><th class="num">${on} avg</th><th class="num">${on} best–worst</th>${vhHead}</tr></thead>
    <tbody>${rows}</tbody></table></div>`
}
// When drilled into a single setup, an editor for that setup's conclusion note —
// the user half of the ledger (LLM verdict + score being the other halves).
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
  // Keep the off-page run cache bounded to runs still referenced by an open detail / compare set.
  for (const k of [...runExtraCache.keys()]) {
    if (k !== selectedRunKey && !runsCompareKeys.has(k)) runExtraCache.delete(k)
  }
  const serverPaged = runsServerPaged()
  // Server total when the flat view paginates server-side; otherwise the loaded set's size.
  const total = serverPaged ? runsTotalCount : runsCache.length
  const filtered = applyRunsFilters(runsCache)
  if (spark) {
    const svg = sparklineSvg(filtered)
    setHtml(spark, svg)
    spark.hidden = !svg
  }
  if (!filtered.length) {
    setHtml(
      body,
      `${runsToolbarHtml(0, total)}<div class="empty-hint">No runs match the filter.</div>`,
    )
    return
  }
  const toolbar = runsToolbarHtml(filtered.length, total)
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
  if (runsViewMode === 'dataset') {
    const legend = `<p class="runs-legend">Each row is a DATASET (asset / walk-forward window / fidelity stack) a run trained on. Click one to drill into its runs. "Custom" = dataset values matching no saved dataset.</p>`
    setHtml(
      body,
      `${toolbar}${mixedDatasetBannerHtml(filtered)}${robustnessAcrossDatasetsHtml(filtered)}${byDatasetTableHtml(filtered)}${legend}`,
    )
    renderCompare()
    return
  }
  // Server-paged: the backend already filtered + sorted + sliced this page, so render it as-is
  // (client text/Hide-bad already refined `filtered`). Otherwise (group-by drill) sort + slice here.
  let shown
  let pageCount
  let start
  if (serverPaged) {
    pageCount = Math.max(1, Math.ceil(total / RUNS_PAGE_SIZE))
    start = runsPage * RUNS_PAGE_SIZE
    shown = filtered
  } else {
    const sorted = sortRuns(filtered)
    pageCount = Math.max(1, Math.ceil(sorted.length / RUNS_PAGE_SIZE))
    runsPage = Math.min(Math.max(0, runsPage), pageCount - 1)
    start = runsPage * RUNS_PAGE_SIZE
    shown = sorted.slice(start, start + RUNS_PAGE_SIZE)
  }
  runsVisibleKeys = shown.map((r) => r.key)
  const cols = runsColumns()
  // "Select all visible" reflects the runs on THIS page.
  const allSelected = shown.length > 0 && shown.every((r) => runsCompareKeys.has(r.key))
  const header = cols
    .map((c) => {
      // The compare column header is a "select all visible" checkbox.
      if (c.id === 'compare') {
        return `<th><input type="checkbox" id="runs-select-all"${allSelected ? ' checked' : ''} aria-label="Select all visible runs" title="Select all runs on this page" /></th>`
      }
      if (c.noSort)
        return `<th class="${c.num ? 'num' : ''}"${helpAttr(c.help)}>${escapeHtml(c.label)}</th>`
      const arrow = runsSortKey === c.id ? (runsSortDir === 'asc' ? ' ▲' : ' ▼') : ''
      return `<th class="runs-th${c.num ? ' num' : ''}" data-sort="${c.id}"${helpAttr(c.help)}>${escapeHtml(c.label)}${arrow}</th>`
    })
    .join('')
  const rows = shown.map((r) => runRowHtml(r, cols)).join('')
  const pager = runsPagerHtml(total, start, shown.length, pageCount)
  const legend = `<p class="runs-legend">Click a header to sort · hover a column header for what it means · <span class="delta-pos">green</span>/<span class="delta-neg">red</span> = beat / lagged buy-and-hold · greyed = failed/degenerate · "—" = not recorded (re-run to populate).</p>`
  setHtml(
    body,
    `${toolbar}<div class="table-wrap"><table class="runs-table">
    <thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table></div>${pager}${legend}`,
  )
  // Keep an open detail/compare even when its run scrolled to another server page (resolved via the
  // remembered-run cache), only closing when the run is genuinely gone.
  if (selectedRunKey && !findRun(selectedRunKey)) closeRunDetail()
  else if (selectedRunKey) renderRunDetail(selectedRunKey)
  renderCompare()
  syncRunsSelectionUI()
}
// Prev/next pager for the flat Runs view; hidden when everything fits on one page.
function runsPagerHtml(total, start, count, pageCount) {
  if (pageCount <= 1) return ''
  const from = total === 0 ? 0 : start + 1
  const to = start + count
  return `<div class="runs-pager">
    <button type="button" class="ghost-btn" id="runs-prev"${runsPage <= 0 ? ' disabled' : ''}>‹ Prev</button>
    <span class="runs-pager-info">${from}–${to} of ${total} · page ${runsPage + 1}/${pageCount}</span>
    <button type="button" class="ghost-btn" id="runs-next"${runsPage >= pageCount - 1 ? ' disabled' : ''}>Next ›</button>
  </div>`
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
  // Custom rules drive the server-side `where`, so they must be loaded BEFORE readRuns builds it.
  customRulesCache = await readCustomRules()
  ;[
    runsCache,
    verdictsCache,
    judgementSummary,
    evaluationsCache,
    dismissedFailures,
    unrunnableCache,
  ] = await Promise.all([
    readRuns(),
    readVerdicts(),
    readJudgement(),
    readEvaluations(),
    readDismissedFailures(),
    readUnrunnable(),
  ])
  await normalizeRunRecords(runsCache)
  renderJudgeControls()
  renderRunsLive()
  await markRunsSeen()
  renderRunsTable()
}
// Re-fetch only the runs for the current view/filters/sort/page and re-render the table. Server-side
// filtering + pagination mean a filter/sort/page/view change needs a fresh query, not just a client
// re-render (the other caches — verdicts/notes/etc. — are unaffected, so they aren't refetched).
async function refreshRuns() {
  if (!byId('runs-body')) return
  runsCache = await readRuns()
  await normalizeRunRecords(runsCache)
  renderRunsTable()
  // Close the xAI analyse→run→re-analyse loop: when records change (e.g. a launched batch lands), the
  // open xAI tab recomputes its effects + recommendations off the fresh runs.
  if (activeTabId === 'xai') renderXai()
}
// The generic on-load run-record migration applier (see normalizeRunRecords): each run gets
// window.Migrate.migrationPatchFor (currently the hold-fee metrics fix), written back idempotently, with
// the in-memory run updated so the render reflects it. Runs on project open over EVERY run (the backlog)
// and on each runs refresh over the loaded set. setupKey is recomputed only when a patch changes the config.
async function recomputeSetupKey(config) {
  const rest = { ...config }
  delete rest.seed
  const canonical = window.Xai.canonicalConfigString(rest)
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(canonical))
  return [...new Uint8Array(buf)]
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('')
    .slice(0, 12)
}
// A hypothesis's canonical identity = the spec hash (the seed-inclusive sibling of recomputeSetupKey),
// matching the backend `hashTrainingConfig` so the same spec dedupes across viewer + LLM/paper sources.
// Async (subtle-crypto) — await it at every id site (form save, add-to-paper, migration, seed dedup).
async function hashTrainingConfig(spec) {
  const canonical = window.Xai.canonicalConfigString(spec || {})
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(canonical))
  return [...new Uint8Array(buf)]
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('')
    .slice(0, 12)
}
// Drop empty sweep/fixed/seeds so a hand-built spec hashes the SAME as the backend's minimal spec.
function normalizeSpec(spec) {
  const out = {}
  const sweep = (spec && spec.sweep) || {}
  const fixed = (spec && spec.fixed) || {}
  if (Object.keys(sweep).length) out.sweep = sweep
  if (Object.keys(fixed).length) out.fixed = fixed
  const seeds = (spec && spec.seeds) || []
  if (Array.isArray(seeds) && seeds.length) out.seeds = seeds
  return out
}
async function normalizeRunRecords(runs) {
  if (!manifest || !embedded() || !window.Migrate || !Array.isArray(runs)) return 0
  let changed = 0
  for (const run of runs) {
    const patch = window.Migrate.migrationPatchFor(run.summary)
    if (!patch) continue
    const content = { ...run.summary, config: { ...run.summary.config, ...patch.config } }
    if (patch.dataset) content.dataset = { ...(run.summary.dataset || {}), ...patch.dataset }
    if (patch.metrics) content.metrics = { ...(run.summary.metrics || {}), ...patch.metrics }
    if (patch.pipelineVersion) content.pipelineVersion = patch.pipelineVersion
    if (patch.config && run.summary.setupKey)
      content.setupKey = await recomputeSetupKey(content.config)
    try {
      await window.OverseerBridge.putData({ type: manifest.recordType, key: run.key, content })
      run.summary = content
      changed++
    } catch {
      // A transient write failure shouldn't abort the batch — the next pass retries this record.
    }
  }
  return changed
}
// Normalize EVERY run of the open project (ignores the flat view's pagination, since the backlog spans
// pages). Run once on open; the per-refresh pass keeps newly-landed runs in step thereafter.
async function migrateAllRunRecords() {
  if (!manifest || !embedded() || !window.Migrate) return 0
  return normalizeRunRecords(await queryRunRecords())
}
// One-time on project open: fold any legacy `-model` records into the unified Hypotheses registry
// (spec = {fixed: match}, deduped by spec hash), deleting the `-model` record. Idempotent — once the
// `-model` records are gone it no-ops. Also migrates old pending/accepted/rejected hypotheses to the
// verdict model (rejected → dismissed). The pure mappings live in migrate.js so they're node-tested.
async function migrateModelsToHypotheses() {
  if (!manifest || !embedded() || !window.Migrate) return
  const modelType = manifest.recordType + '-model'
  const modelRecs = await queryRecords(modelType)
  const models = modelRecs.map((r) => r.content).filter((c) => c && c.id)
  if (models.length) {
    const have = new Set((await readHypotheses()).map((h) => h.id))
    const now = nowIso()
    for (const m of models) {
      const spec = normalizeSpec({ fixed: m.match || {} })
      const id = await hashTrainingConfig(spec)
      if (!have.has(id)) {
        const h = window.Migrate.hypothesisFromLegacyModel(m, id, now)
        h.spec = spec
        await putHypothesis(h)
        have.add(id)
      }
      try {
        await window.OverseerBridge.deleteData({ type: modelType, key: m.id })
      } catch {
        // best-effort: a failed delete just leaves the legacy record to be retried next open
      }
    }
  }
  // Migrate any old-lifecycle hypotheses (pending/accepted/rejected) onto the verdict model.
  const now = nowIso()
  for (const h of await readHypotheses()) {
    const patch = window.Migrate.legacyHypothesisPatch(h, now)
    if (patch) {
      try {
        await putHypothesis({ ...h, ...patch })
      } catch {
        // best-effort
      }
    }
  }
}
// Multi-select comparison: a config diff (only differing levers), metrics
// side-by-side, and overlaid %-return curves (+ the buy-and-hold control) for the
// runs ticked in the table. Hidden until ≥2 are selected; pruned of stale keys.
function renderCompare() {
  const card = byId('run-compare')
  if (!card) return
  runsCompareKeys = new Set([...runsCompareKeys].filter((k) => findRun(k)))
  const runs = [...runsCompareKeys].map((k) => findRun(k)).filter(Boolean)
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
  // Only a MAJOR-version spread breaks comparability; runs differing only by minor stay comparable.
  const majors = new Set(
    runs.map((r) => parsePipelineVersion((r.summary && r.summary.pipelineVersion) || '1').major),
  )
  const versionWarn =
    majors.size > 1
      ? `<p class="compare-version-warn"><span class="badge is-bad">heads-up</span> These runs span pipeline versions ${[...versions].map((v) => `v${escapeHtml(v)}`).join(', ')} — a breaking (MAJOR) version changed how data is fed/scored, so their scores are NOT directly comparable.</p>`
      : ''
  setHtml(
    card,
    `<div class="card-head card-head-row">
      <h3>Compare ${runs.length} runs</h3>
      <div class="head-actions">
        ${embedded() && runs.every((r) => r.summary && r.summary.status === 'failed') ? `<button type="button" id="compare-rerun-all" class="ghost-btn" title="Queue all ${runs.length} failed runs again">Re-run all (${runs.length})</button>` : ''}
        ${runs.length === 2 && chatAboutRunAvailable() ? `<button type="button" id="compare-discuss" class="icon-btn" title="Discuss these two runs (incl. the decision diff)" aria-label="Discuss these two runs">${iconChatSvg()}</button>` : ''}
        <button type="button" id="compare-clear" class="icon-btn" title="Clear selection" aria-label="Clear selection">✕</button>
      </div>
    </div>
    <div class="card-scroll">
    ${versionWarn}
    ${comparisonTable}
    ${compareEquityChartHtml(runs, runColors)}
    ${runs.length === 2 ? decisionDiffSectionHtml(runs[0], runs[1]) : ''}
    <h3>Charts</h3>
    <div class="charts-body">${chartsSectionsHtml(runs, runColors)}</div></div>`,
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
// Surfaced via the objective headline already (traded_return), plus the retired risk metrics that
// old runs still carry — both hidden so the detail table shows only live, meaningful metrics.
const DETAIL_HIDDEN_METRICS = new Set(['traded_return', ...RETIRED_METRICS])
function metricsTableHtml(metrics) {
  const entries = Object.entries(metrics || {}).filter(([k]) => !DETAIL_HIDDEN_METRICS.has(k))
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
// Marker taxonomy for the price chart: executed opens (buy/short) & agent closes (sell/cover),
// auto closes (tp/trailing/sl), and no-op ATTEMPTED requests (*_attempt). Bullish marks point up.
const UP_MARKS = new Set(['buy', 'cover', 'tp', 'trailing', 'buy_attempt', 'cover_attempt'])
const ATTEMPT_MARKS = new Set(['buy_attempt', 'sell_attempt', 'short_attempt', 'cover_attempt'])
const MARK_LABEL = {
  buy: 'buy',
  sell: 'sell',
  short: 'short',
  cover: 'cover',
  tp: 'TP',
  trailing: 'trail',
  sl: 'SL',
  buy_attempt: 'buy⊘',
  sell_attempt: 'sell⊘',
  short_attempt: 'short⊘',
  cover_attempt: 'cover⊘',
}
const MARK_ORDER = [
  'buy',
  'sell',
  'short',
  'cover',
  'tp',
  'trailing',
  'sl',
  'buy_attempt',
  'sell_attempt',
  'short_attempt',
  'cover_attempt',
]
// Price line + trade markers, re-surfacing the repo's old matplotlib action-on-price plot as
// serialised data drawn on our SVG engine. Markers carry their own index into the downsampled
// price array, so none are lost. Attempted (no-op) requests render hollow (see run-mark CSS).
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
    const up = UP_MARKS.has(m.type)
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
  const sizeAttr =
    opts && opts.fixedSize ? ` width="${W}" height="${H}" style="width:${W}px;height:${H}px"` : ''
  return `<svg class="chart"${sizeAttr} viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeHtml((opts && opts.ariaLabel) || 'price with trade actions')}">${parts.join('')}</svg>`
}
// The custom per-run result view: a price chart with the run's buy/sell/TP/SL
// markers + a count legend. Gated on `artifacts.runChart`, so non-trading
// projects (cartpole, regression) keep the generic run-detail view.
function priceActionLegendHtml(chart) {
  const markers = Array.isArray(chart.markers) ? chart.markers : []
  // Prefer the producer's AUTHORITATIVE counts (tallied before the draw-only downsample dedup, so
  // they match the metrics/ledger); fall back to tallying drawn markers for runs that predate counts.
  let counts = chart.counts && typeof chart.counts === 'object' ? chart.counts : null
  if (!counts) {
    counts = {}
    for (const m of markers) counts[m.type] = (counts[m.type] || 0) + 1
  }
  const markerKeys = MARK_ORDER.filter((t) => counts[t])
    .map(
      (t) =>
        `<span class="run-mark-key run-mark-${t}" title="${ATTEMPT_MARKS.has(t) ? 'requested but did not execute (no-op)' : 'executed'}">${escapeHtml(MARK_LABEL[t] || t)} ${counts[t]}</span>`,
    )
    .join(' ')
  return `<p class="badges-row run-mark-legend"><span class="run-mark-key run-mark-price">price</span>${markerKeys ? ` ${markerKeys}` : ''}</p>`
}
function priceActionSectionHtml(summary, key) {
  const chart = summary && summary.artifacts && summary.artifacts.runChart
  if (!chart || !Array.isArray(chart.price) || chart.price.length < 2) return ''
  const legend = priceActionLegendHtml(chart)
  const svg = buildPriceActionChart(chart, {
    xLabel: 'step',
    ariaLabel: 'price with trade actions',
    width: 640,
    height: 200,
  })
  const expand = key
    ? ` <button type="button" class="icon-btn chart-expand-btn" data-action="expand-chart" data-key="${escapeHtml(key)}" title="Open larger — zoom &amp; scroll" aria-label="Expand chart">🔍</button>`
    : ''
  return `<h3>Price &amp; actions${expand}</h3>${legend}<div class="chart-wrap">${svg}</div>`
}
// Expanded Price & actions: a popup of the same chart at large size with zoom (widens the plot so
// dense marker runs separate) and horizontal scroll. Uses the run's downsampled chart data.
let chartModalData = null
let chartModalZoom = 1
function expandPriceActionChart(key) {
  const run = findRun(key)
  const chart = run && run.summary && run.summary.artifacts && run.summary.artifacts.runChart
  if (!chart || !Array.isArray(chart.price) || chart.price.length < 2) return
  chartModalData = chart
  chartModalZoom = 1
  renderChartModal()
}
function setChartZoom(z) {
  chartModalZoom = Math.max(1, Math.min(12, z))
  renderChartModal()
}
function closeChartModal() {
  const m = byId('chart-modal')
  if (m) m.hidden = true
}
function renderChartModal() {
  if (!chartModalData) return
  let modal = byId('chart-modal')
  if (!modal) {
    modal = document.createElement('div')
    modal.id = 'chart-modal'
    modal.className = 'chart-modal'
    document.body.appendChild(modal)
    modal.addEventListener('click', (event) => {
      if (event.target === modal || event.target.closest('[data-chart-close]'))
        return closeChartModal()
      if (event.target.closest('[data-chart-zoom="in"]')) return setChartZoom(chartModalZoom * 1.5)
      if (event.target.closest('[data-chart-zoom="out"]')) return setChartZoom(chartModalZoom / 1.5)
      if (event.target.closest('[data-chart-zoom="reset"]')) return setChartZoom(1)
    })
  }
  const width = Math.round(1100 * chartModalZoom)
  const svg = buildPriceActionChart(chartModalData, {
    xLabel: 'step',
    ariaLabel: 'price with trade actions (expanded)',
    width,
    height: 420,
    fixedSize: true,
  })
  modal.innerHTML = `<div class="chart-modal__backdrop" data-chart-close></div>
    <div class="chart-modal__panel" role="dialog" aria-label="Price and actions (expanded)">
      <div class="chart-modal__head">
        <strong>Price &amp; actions <span class="card-sub">— ${Math.round(chartModalZoom * 100)}%</span></strong>
        <div class="chart-modal__tools">
          <button type="button" class="icon-btn" data-chart-zoom="out" title="Zoom out" aria-label="Zoom out">−</button>
          <button type="button" class="icon-btn" data-chart-zoom="reset" title="Reset zoom" aria-label="Reset zoom">100%</button>
          <button type="button" class="icon-btn" data-chart-zoom="in" title="Zoom in" aria-label="Zoom in">+</button>
          <button type="button" class="icon-btn" data-chart-close title="Close (Esc)" aria-label="Close">✕</button>
        </div>
      </div>
      ${priceActionLegendHtml(chartModalData)}
      <div class="chart-modal__scroll">${svg}</div>
    </div>`
  modal.hidden = false
}
// Human-readable label for an exit reason, shared by the exits table and the ledger.
const EXIT_REASON_LABEL = {
  sell: 'agent sell (long)',
  cover: 'agent cover (short)',
  tp: 'take-profit',
  trailing: 'trailing TP',
  sl: 'stop-loss',
  open: 'open at end (implied)',
}
function fmtReportPct(v) {
  const n = Number(v)
  return Number.isFinite(n) ? `${n.toFixed(2)}%` : '—'
}
// Exit-reason breakdown: how each position was closed — agent decisions (sell/cover) vs the
// auto TP/trailing/SL rules, plus an implied close for a position still open at the end. Directly
// answers "does the model decide to sell, or do the rules close for it?".
function exitsSectionHtml(summary) {
  const exits = summary && summary.exits
  if (!exits || typeof exits !== 'object') return ''
  const order = ['sell', 'cover', 'tp', 'trailing', 'sl', 'open']
  const keys = Object.keys(exits).sort((a, b) => {
    const ia = order.indexOf(a)
    const ib = order.indexOf(b)
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib)
  })
  if (!keys.length) return ''
  const rows = keys
    .map((k) => {
      const e = exits[k] || {}
      const pnl = Number(e.total_pnl_pct)
      const cls = Number.isFinite(pnl) ? (pnl >= 0 ? 'delta-pos' : 'delta-neg') : ''
      return `<tr><th>${escapeHtml(EXIT_REASON_LABEL[k] || k)}</th>
        <td class="num">${Number(e.count) || 0}</td>
        <td class="num">${escapeHtml(fmtReportPct(e.win_pct))}</td>
        <td class="num ${cls}">${escapeHtml(fmtReportPct(e.total_pnl_pct))}</td>
        <td class="num">${escapeHtml(fmtReportPct(e.avg_pnl_pct))}</td></tr>`
    })
    .join('')
  return `<h3>Exits <span class="card-sub">— how positions closed</span></h3>
    <table class="kv-table report-table"><thead><tr><th>reason</th><th class="num">#</th><th class="num">win %</th><th class="num">total P&amp;L %</th><th class="num">avg P&amp;L %</th></tr></thead>
    <tbody>${rows}</tbody></table>`
}
// Skill-vs-luck: per equal-time window and per trailing-trend regime, the market's move next to the
// model's realized trading P&L. Profit that only appears where the market rose is beta, not timing.
function regimesSectionHtml(summary) {
  const r = summary && summary.regimes
  if (!r || typeof r !== 'object') return ''
  let out = ''
  if (Array.isArray(r.windows) && r.windows.length) {
    const rows = r.windows
      .map((w, i) => {
        const mkt = Number(w.market_return_pct)
        const pnl = Number(w.realized_pnl_pct)
        return `<tr><th>window ${i + 1}</th>
          <td class="num ${mkt >= 0 ? 'delta-pos' : 'delta-neg'}">${escapeHtml(fmtReportPct(mkt))}</td>
          <td class="num ${pnl >= 0 ? 'delta-pos' : 'delta-neg'}">${escapeHtml(fmtReportPct(pnl))}</td>
          <td class="num">${Number(w.n_trades) || 0}</td>
          <td class="num">${escapeHtml(fmtReportPct(w.win_pct))}</td></tr>`
      })
      .join('')
    out += `<table class="kv-table report-table"><thead><tr><th>by time</th><th class="num">market %</th><th class="num">model P&amp;L %</th><th class="num">#</th><th class="num">win %</th></tr></thead><tbody>${rows}</tbody></table>`
  }
  if (r.trend && typeof r.trend === 'object') {
    const rows = ['up', 'flat', 'down']
      .filter((k) => r.trend[k])
      .map((k) => {
        const t = r.trend[k]
        const pnl = Number(t.realized_pnl_pct)
        return `<tr><th>${escapeHtml(k)} market</th>
          <td class="num ${pnl >= 0 ? 'delta-pos' : 'delta-neg'}">${escapeHtml(fmtReportPct(pnl))}</td>
          <td class="num">${Number(t.n_trades) || 0}</td>
          <td class="num">${escapeHtml(fmtReportPct(t.win_pct))}</td>
          <td class="num">${escapeHtml(fmtReportPct(t.bars_pct))}</td></tr>`
      })
      .join('')
    if (rows) {
      out += `<table class="kv-table report-table"><thead><tr><th>by regime</th><th class="num">model P&amp;L %</th><th class="num">#</th><th class="num">win %</th><th class="num">% of window</th></tr></thead><tbody>${rows}</tbody></table>`
    }
  }
  if (!out) return ''
  return `<h3>Regimes <span class="card-sub">— skill vs riding the market</span></h3>${out}`
}
// The trade ledger: every reconstructed round-trip, so a run is readable rather than a black box.
// Long ledgers collapse behind a disclosure.
const LEDGER_INLINE_MAX = 60
function ledgerRowHtml(t) {
  const pnl = Number(t.pnl_pct)
  const cls = Number.isFinite(pnl) ? (pnl >= 0 ? 'delta-pos' : 'delta-neg') : ''
  return `<tr><td class="num">${Number(t.entry_step)}→${Number(t.exit_step)}</td>
    <td>${escapeHtml(String(t.side || ''))}</td>
    <td>${escapeHtml(EXIT_REASON_LABEL[t.reason] || String(t.reason || ''))}</td>
    <td class="num">${escapeHtml(formatTickValue(Number(t.entry_price)))}→${escapeHtml(formatTickValue(Number(t.exit_price)))}</td>
    <td class="num">${Number(t.bars_held)}</td>
    <td class="num ${cls}">${escapeHtml(fmtReportPct(t.pnl_pct))}</td></tr>`
}
function ledgerSectionHtml(summary) {
  const ledger = summary && Array.isArray(summary.ledger) ? summary.ledger : []
  if (!ledger.length) return ''
  const head = `<thead><tr><th class="num">steps</th><th>side</th><th>exit</th><th class="num">price</th><th class="num">bars</th><th class="num">P&amp;L %</th></tr></thead>`
  const body = `<tbody>${ledger.map(ledgerRowHtml).join('')}</tbody>`
  const table = `<table class="kv-table report-table ledger-table">${head}${body}</table>`
  const title = `<h3>Trade ledger <span class="card-sub">— ${ledger.length} round-trips</span></h3>`
  if (ledger.length <= LEDGER_INLINE_MAX) return `${title}${table}`
  return `${title}<details class="ledger-details"><summary>Show all ${ledger.length} round-trips</summary>${table}</details>`
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
// --- Explain: the decision trace (xAI) -------------------------------------
// A generic, domain-oblivious read of WHY the model acted, from summary.artifacts.decisionTrace (a
// DecisionTrace; arbitrary action labels, no trading vocabulary): an action-distribution diagnostic
// that flags anomalies + dormant actions, the per-action VALUE the policy assigned over time (so
// "why so few sells?" is answerable — sell-value vs hold-value), confidence over time, and any input
// attribution. The timeline shares the downsampled step axis with the price/equity charts above.
const ACTION_VALUE_PALETTE = new Map([
  ['hold', '#94a3b8'],
  ['buy', '#3b82f6'],
  ['sell', '#ef4444'],
  ['short', '#8b5cf6'],
  ['cover', '#14b8a6'],
])
function readDecisionTrace(summary) {
  const t = summary && summary.artifacts && summary.artifacts.decisionTrace
  if (!t || typeof t !== 'object') return null
  const steps = Array.isArray(t.steps)
    ? t.steps.filter((s) => s && typeof s === 'object' && typeof s.action === 'string')
    : []
  if (!steps.length) return null
  return { ...t, steps }
}
function decisionActionColors(labels) {
  const map = new Map()
  labels.forEach((label, i) =>
    map.set(label, ACTION_VALUE_PALETTE.get(label) || CHART_PALETTE[i % CHART_PALETTE.length]),
  )
  return map
}
// Possible actions = every action the policy ever SCORED (a key in any step's actionValues) unioned
// with actions actually taken, so a never-taken-but-scorable action surfaces as dormant — the crux of
// the sparse-sell question.
function decisionTraceDigest(trace) {
  const steps = trace.steps
  const counts =
    trace.actionCounts && typeof trace.actionCounts === 'object'
      ? { ...trace.actionCounts }
      : steps.reduce((m, s) => ((m[s.action] = (m[s.action] || 0) + 1), m), {})
  const total = Number(trace.totalSteps) || steps.length
  const scorable = new Set()
  for (const s of steps)
    if (s.actionValues) for (const k of Object.keys(s.actionValues)) scorable.add(k)
  const taken = Object.keys(counts)
  const possible = [...new Set([...taken, ...scorable])]
  const entries = taken.map((a) => [a, Number(counts[a]) || 0]).sort((x, y) => y[1] - x[1])
  const dominant = entries.length ? entries[0][0] : null
  const dominantCount = entries.length ? entries[0][1] : 0
  const dormant = possible.filter((a) => !counts[a])
  const anomalies = []
  if (dominant) {
    for (const [a, c] of entries) {
      if (a !== dominant && dominantCount >= 10 * Math.max(c, 1) && dominantCount >= 5)
        anomalies.push(`${dominant} ≫ ${a} (${dominantCount} vs ${c})`)
    }
  }
  for (const a of dormant) anomalies.push(`${a} never taken — scored but dormant`)
  return { total, counts, entries, dominant, dormant, possible, anomalies }
}
// Mean (value[a] − value[b]) over steps where both were scored — e.g. how much "hold" outscores
// "sell", a quantitative read on why an action stays dormant. Null when never co-scored.
function avgDecisionValueGap(trace, a, b) {
  let sum = 0
  let n = 0
  for (const s of trace.steps) {
    if (!s.actionValues) continue
    const va = Number(s.actionValues[a])
    const vb = Number(s.actionValues[b])
    if (Number.isFinite(va) && Number.isFinite(vb)) {
      sum += va - vb
      n += 1
    }
  }
  return n ? sum / n : null
}
function decisionDistributionHtml(digest) {
  const total = digest.total || 1
  const rows = digest.entries
    .map(
      ([a, c]) =>
        `<tr><th>${escapeHtml(a)}</th><td class="num">${c}</td><td class="num">${((100 * c) / total).toFixed(1)}%</td></tr>`,
    )
    .join('')
  const dormantRows = digest.dormant
    .map(
      (a) => `<tr><th>${escapeHtml(a)} <span class="card-sub">dormant</span></th>
        <td class="num">0</td><td class="num">0.0%</td></tr>`,
    )
    .join('')
  const anomalyChips = digest.anomalies.length
    ? `<p class="badges-row">${digest.anomalies.map((a) => `<span class="badge is-warn">${escapeHtml(a)}</span>`).join(' ')}</p>`
    : '<p class="card-sub">No distribution anomalies.</p>'
  return `<h4 class="card-sub">Action distribution — ${digest.total} decisions</h4>
    ${anomalyChips}
    <table class="kv-table report-table"><thead><tr><th>action</th><th class="num">#</th><th class="num">share</th></tr></thead>
    <tbody>${rows}${dormantRows}</tbody></table>`
}
// The sparse-sell deep-dive: one value line per scorable action over the step axis — makes it
// self-evident whether "hold" is persistently worth more than "sell" (value never learned) or the
// signal just never wins.
function decisionValueChartHtml(trace, digest) {
  const lines = digest.possible.filter((a) =>
    trace.steps.some((s) => s.actionValues && Number.isFinite(Number(s.actionValues[a]))),
  )
  if (lines.length < 2) return ''
  const points = []
  trace.steps.forEach((s, i) => {
    if (!s.actionValues) return
    for (const a of lines) {
      const v = Number(s.actionValues[a])
      if (Number.isFinite(v)) points.push({ x: i, y: v, group: a })
    }
  })
  if (points.length < 2) return ''
  const groupColors = decisionActionColors(lines)
  const svg = buildLineChart({
    points,
    xLabel: 'step',
    yLabel: 'value',
    width: 640,
    height: 200,
    markers: false,
    groupColors,
    ariaLabel: 'per-action value over time',
  })
  return `<h4 class="card-sub">Per-action value over time — is the dormant signal ever worth more than holding?</h4>
    ${chartLegendHtml(groupColors)}<div class="chart-wrap">${svg}</div>`
}
function decisionConfidenceChartHtml(trace) {
  const points = []
  trace.steps.forEach((s, i) => {
    if (Number.isFinite(Number(s.confidence)))
      points.push({ x: i, y: Number(s.confidence), group: 'confidence' })
  })
  if (points.length < 2) return ''
  const groupColors = new Map([['confidence', CHART_PALETTE[4]]])
  const svg = buildLineChart({
    points,
    xLabel: 'step',
    yLabel: 'confidence',
    width: 640,
    height: 160,
    markers: false,
    groupColors,
    ariaLabel: 'policy confidence over time',
  })
  return `<h4 class="card-sub">Confidence in the chosen action over time</h4><div class="chart-wrap">${svg}</div>`
}
// Reward decomposition — "why this reward": the named additive contributions (base earnings vs the
// penalties that dragged it down), so the user sees what's driving the score. `total` shown last.
function decisionRewardBreakdownHtml(trace) {
  const bd = trace.rewardBreakdown
  if (!bd || typeof bd !== 'object') return ''
  const entries = Object.entries(bd).filter(([, v]) => Number.isFinite(Number(v)))
  if (!entries.length) return ''
  const sorted = entries
    .filter(([k]) => k !== 'total')
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
  if (bd.total !== undefined && Number.isFinite(Number(bd.total)))
    sorted.push(['total', Number(bd.total)])
  const rows = sorted
    .map(([k, v]) => {
      const isTotal = k === 'total'
      const cls = Number(v) >= 0 ? 'delta-pos' : 'delta-neg'
      return `<tr><th>${isTotal ? '<strong>total</strong>' : escapeHtml(k)}</th><td class="num ${cls}">${Number(v) >= 0 ? '+' : ''}${escapeHtml(formatTickValue(v))}</td></tr>`
    })
    .join('')
  return `<h4 class="card-sub">Reward breakdown <span class="card-sub">— why this reward (named contributions sum to the total)</span></h4>
    <table class="kv-table report-table"><thead><tr><th>component</th><th class="num">contribution</th></tr></thead><tbody>${rows}</tbody></table>`
}
// Latent state map — a 2-D PCA of the policy's penultimate-layer activations, coloured by the action
// taken. Clusters = how the model organises states by decision (its INTERNAL representation).
function decisionLatentMapHtml(trace) {
  const lm = trace.latentMap
  if (!lm || !Array.isArray(lm.points)) return ''
  const points = lm.points
    .map((p) => ({ x: Number(p.x), y: Number(p.y), group: String(p.action) }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
  if (points.length < 3) return ''
  const actions = [...new Set(points.map((p) => p.group))]
  const groupColors = new Map(actions.map((a, i) => [a, CHART_PALETTE[i % CHART_PALETTE.length]]))
  const svg = buildScatterChart({
    points,
    xLabel: 'PC1',
    yLabel: 'PC2',
    width: 460,
    height: 320,
    groupColors,
    ariaLabel: 'latent state map coloured by action',
  })
  const varPct = Number.isFinite(Number(lm.varianceExplained))
    ? `${Math.round(Number(lm.varianceExplained) * 100)}% of variance`
    : ''
  return `<h4 class="card-sub">Latent state map <span class="card-sub">— penultimate-layer activations (PCA${varPct ? `, ${escapeHtml(varPct)}` : ''}); clusters = how it organises states by decision</span></h4>
    ${chartLegendHtml(groupColors)}<div class="chart-wrap">${svg}</div>
    ${xaiLatentProbeHtml(lm.probe)}`
}
// The linear-probe read: does the latent linearly encode the action? Accuracy vs the majority baseline.
function xaiLatentProbeHtml(probe) {
  if (!probe || typeof probe !== 'object' || !Number.isFinite(Number(probe.accuracy))) return ''
  const acc = Math.round(Number(probe.accuracy) * 100)
  const base = Number.isFinite(Number(probe.baseline))
    ? Math.round(Number(probe.baseline) * 100)
    : null
  const beats = base !== null && acc > base + 5
  const cls = beats ? 'is-ok' : ''
  return `<p class="badges-row"><span class="badge ${cls}">linear probe ${acc}%</span> <span class="card-sub">— a linear classifier predicts the action from the latent at ${acc}% accuracy${base !== null ? ` (vs ${base}% majority baseline)` : ''}${beats ? ' — the representation linearly encodes the decision' : base !== null ? ' — barely above chance, so the decision isn’t linearly separable in the latent' : ''}.</span></p>`
}
function decisionAttributionHtml(trace) {
  const fa = trace.featureAttribution
  if (!fa || typeof fa !== 'object') return ''
  let entries = []
  let label = 'feature'
  if (fa.byGroup && typeof fa.byGroup === 'object') {
    entries = Object.entries(fa.byGroup).filter(([, v]) => Number.isFinite(Number(v)))
    label = 'group'
  } else if (Array.isArray(fa.perFeature)) {
    entries = fa.perFeature
      .map((v, i) => [`feature ${i}`, Number(v)])
      .filter(([, v]) => Number.isFinite(v))
  }
  if (!entries.length) return ''
  entries = entries.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 12)
  const rows = entries
    .map(
      ([k, v]) =>
        `<tr><th>${escapeHtml(k)}</th><td class="num">${escapeHtml(formatTickValue(v))}</td></tr>`,
    )
    .join('')
  const meta = `${escapeHtml(String(fa.method || 'saliency'))}${fa.samples ? `, ${Number(fa.samples)} decisions` : ''}`
  return `<h4 class="card-sub">Input attribution — ${meta}</h4>
    ${xaiSanityBadgeHtml(fa.sanityCheck)}
    <table class="kv-table report-table"><thead><tr><th>${label}</th><th class="num">saliency</th></tr></thead><tbody>${rows}</tbody></table>`
}
// The Adebayo saliency sanity-check verdict: a faithful map CHANGES when the model's weights are
// randomized (low rank correlation). A failed/absent check tells the user not to over-trust the map.
function xaiSanityBadgeHtml(sc) {
  if (!sc || typeof sc !== 'object') return ''
  const corr = Number.isFinite(Number(sc.rankCorrelation))
    ? Number(sc.rankCorrelation).toFixed(2)
    : '?'
  if (sc.passed === true) {
    return `<p class="badges-row"><span class="badge is-ok">sanity ✓ faithful</span> <span class="card-sub">— saliency changed under weight-randomization (rank corr ${escapeHtml(corr)}); Adebayo check.</span></p>`
  }
  return `<p class="badges-row"><span class="badge is-warn">sanity ⚠ unreliable</span> <span class="card-sub">— saliency barely changed when weights were randomized (rank corr ${escapeHtml(corr)}), so it may reflect the input/architecture, not what the model learned. Treat with caution.</span></p>`
}
function explainSectionHtml(summary) {
  const trace = readDecisionTrace(summary)
  if (!trace) return ''
  const digest = decisionTraceDigest(trace)
  const inner = [
    decisionDistributionHtml(digest),
    decisionRewardBreakdownHtml(trace),
    decisionValueChartHtml(trace, digest),
    decisionConfidenceChartHtml(trace),
    decisionAttributionHtml(trace),
    decisionLatentMapHtml(trace),
  ]
    .filter(Boolean)
    .join('')
  if (!inner) return ''
  const showing =
    trace.steps.length < digest.total ? ` · showing ${trace.steps.length} of ${digest.total}` : ''
  return `<h3>Explain <span class="card-sub">— why the model acted${escapeHtml(showing)}</span></h3>${inner}`
}
// A compact text read of the decision trace for the chat system prompt, so "why so few sells?" has
// the action counts, dormant signals, value gaps and top attributed inputs already in context.
function decisionTraceChatSummary(summary) {
  const trace = readDecisionTrace(summary)
  if (!trace) return ''
  const d = decisionTraceDigest(trace)
  const parts = [
    `Action counts over ${d.total} steps: ${d.entries.map(([a, c]) => `${a}=${c}`).join(', ')}.`,
  ]
  if (d.anomalies.length) parts.push(`Distribution flags: ${d.anomalies.join('; ')}.`)
  if (d.dominant) {
    for (const a of d.dormant) {
      const gap = avgDecisionValueGap(trace, d.dominant, a)
      if (gap != null)
        parts.push(
          `On average "${d.dominant}" is worth ${formatTickValue(gap)} more than the dormant "${a}" — why "${a}" stays unused.`,
        )
    }
  }
  const fa = trace.featureAttribution
  if (fa && fa.byGroup && typeof fa.byGroup === 'object') {
    const top = Object.entries(fa.byGroup)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, 5)
      .map(([k, v]) => `${k}(${formatTickValue(v)})`)
      .join(', ')
    if (top) parts.push(`Top input groups by saliency: ${top}.`)
  } else if (fa && Array.isArray(fa.perFeature)) {
    const top = fa.perFeature
      .map((v, i) => [i, v])
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, 5)
      .map(([i, v]) => `f${i}(${formatTickValue(v)})`)
      .join(', ')
    if (top) parts.push(`Top input features by saliency: ${top}.`)
  }
  const bd = trace.rewardBreakdown
  if (bd && typeof bd === 'object') {
    const comps = Object.entries(bd)
      .filter(([k, v]) => k !== 'total' && Number.isFinite(Number(v)))
      .map(([k, v]) => `${k} ${Number(v) >= 0 ? '+' : ''}${formatTickValue(v)}`)
      .join(', ')
    if (comps) parts.push(`Reward breakdown (why this reward): ${comps}.`)
  }
  return `Decision trace summary:\n${parts.join('\n')}`
}
// --- Data-influence: the decision DIFF between two runs (xAI) ----------------
// A counterfactual read of how a lever tweak (the "new information") changed the model's DECISIONS,
// not just the score. Mirrors the engine's diffDecisionTraces (modelTrainerUtils.ts) over the compact
// traces both runs already carry — the engine util stays the source of truth. Domain-oblivious.
const ALIGNMENT_DATASET_KEYS = ['asset', 'timeframe', 'candles', 'from', 'to']
function datasetAlignmentSignature(summary) {
  const d = summary && summary.dataset
  if (!d || typeof d !== 'object') return ''
  return ALIGNMENT_DATASET_KEYS.filter((k) => d[k] !== undefined && d[k] !== null)
    .map((k) => `${k}=${d[k]}`)
    .join('|')
}
function meanOf(xs) {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : undefined
}
function classifyDecisionQuality(changed, unchanged) {
  const MIN = 5
  const EPS = 1e-6
  const onChanges = meanOf(changed)
  const onUnchanged = meanOf(unchanged)
  const out = {
    scoredChangedSteps: changed.length,
    meanRewardDeltaOnChanges: onChanges,
    meanRewardDeltaOnUnchanged: onUnchanged,
    verdict: 'insufficient',
  }
  if (changed.length < MIN) return out
  const ch = onChanges || 0
  const un = onUnchanged || 0
  if (Math.abs(ch) <= EPS) out.verdict = 'unchanged'
  else if (ch > EPS) out.verdict = ch > un + EPS ? 'better' : 'mixed'
  else out.verdict = ch < un - EPS ? 'worse' : 'mixed'
  return out
}
function diffDecisionTraces(baseline, tweak) {
  const a = readDecisionTrace(baseline)
  const b = readDecisionTrace(tweak)
  if (!a || !b) return null
  const sigA = datasetAlignmentSignature(baseline)
  const sigB = datasetAlignmentSignature(tweak)
  if (!sigA || !sigB || sigA !== sigB)
    return { aligned: false, alignmentNote: 'different dataset — not step-comparable' }
  const totalA = Number(a.totalSteps) || a.steps.length
  const totalB = Number(b.totalSteps) || b.steps.length
  if (totalA !== totalB)
    return { aligned: false, alignmentNote: `different step counts (${totalA} vs ${totalB})` }
  const mapA = new Map(a.steps.map((s) => [s.step, s]))
  const mapB = new Map(b.steps.map((s) => [s.step, s]))
  const shared = [...mapA.keys()].filter((k) => mapB.has(k)).sort((x, y) => x - y)
  if (!shared.length) return { aligned: false, alignmentNote: 'no shared steps' }
  const changedDeltas = []
  const unchangedDeltas = []
  const confDeltas = []
  const stepFlags = []
  let changedSteps = 0
  for (const step of shared) {
    const sa = mapA.get(step)
    const sb = mapB.get(step)
    const changed = sa.action !== sb.action
    if (changed) changedSteps += 1
    stepFlags.push({ step, changed })
    if (typeof sa.reward === 'number' && typeof sb.reward === 'number')
      (changed ? changedDeltas : unchangedDeltas).push(sb.reward - sa.reward)
    if (typeof sa.confidence === 'number' && typeof sb.confidence === 'number')
      confDeltas.push(sb.confidence - sa.confidence)
  }
  const labels = new Set([
    ...Object.keys(a.actionCounts || {}),
    ...Object.keys(b.actionCounts || {}),
  ])
  const actionCountDeltas = {}
  for (const k of labels) {
    const d = ((b.actionCounts || {})[k] || 0) - ((a.actionCounts || {})[k] || 0)
    if (d) actionCountDeltas[k] = d
  }
  return {
    aligned: true,
    alignedSteps: shared.length,
    changedSteps,
    divergenceRate: changedSteps / shared.length,
    stepFlags,
    actionCountDeltas,
    meanConfidenceShift: meanOf(confDeltas),
    objectiveDelta: Number(tweak.objective) - Number(baseline.objective),
    quality: classifyDecisionQuality(changedDeltas, unchangedDeltas),
  }
}
const DECISION_VERDICT_BADGE = {
  better: 'is-ok',
  worse: 'is-bad',
  mixed: 'is-warn',
  unchanged: '',
  insufficient: '',
}
function fmtSignedNum(v) {
  if (typeof v !== 'number' || !Number.isFinite(v)) return 'n/a'
  const r = Number(v.toFixed(4))
  return r >= 0 ? `+${r}` : `${r}`
}
function decisionDivergenceStripHtml(diff, runColors, labelA, labelB) {
  const pts = diff.stepFlags.map((f, i) => ({ x: i, y: f.changed ? 1 : 0, group: 'changed' }))
  if (pts.length < 2) return ''
  const svg = buildLineChart({
    points: pts,
    xLabel: 'step',
    yLabel: 'changed',
    width: 640,
    height: 90,
    markers: false,
    groupColors: new Map([['changed', CHART_PALETTE[3]]]),
    ariaLabel: 'where the decisions diverged over the step axis',
  })
  return `<p class="card-sub">Where decisions diverged (${escapeHtml(labelA)} → ${escapeHtml(labelB)}), on the same step axis as the equity overlay:</p><div class="chart-wrap">${svg}</div>`
}
// The compare-pane decision-diff section (exactly 2 runs). baseline = the first selected, tweak = the
// second; the config diff above already shows WHICH lever changed (the "new information").
function decisionDiffSectionHtml(baseline, tweak) {
  const diff = diffDecisionTraces(baseline, tweak)
  if (!diff) return '' // one or both runs have no decision trace
  const labelA = shortKey(baseline.key)
  const labelB = shortKey(tweak.key)
  if (!diff.aligned) {
    return `<h3>Decision diff <span class="card-sub">— ${escapeHtml(labelA)} → ${escapeHtml(labelB)}</span></h3>
      <p class="card-sub">${escapeHtml(diff.alignmentNote)} — the two runs aren't step-comparable, so their decisions can't be diffed.</p>`
  }
  const q = diff.quality
  const badge = DECISION_VERDICT_BADGE[q.verdict] || ''
  const verdictChip = `<span class="badge ${badge}">decisions: ${escapeHtml(q.verdict)}${q.verdict === 'better' || q.verdict === 'worse' || q.verdict === 'mixed' ? ' — heuristic' : ''}</span>`
  const divergence = `<p class="badges-row">${verdictChip} · ${(diff.divergenceRate * 100).toFixed(0)}% of ${diff.alignedSteps} aligned steps changed</p>`
  const control =
    q.scoredChangedSteps >= 5
      ? `<p class="card-sub">At the ${q.scoredChangedSteps} changed steps the tweak averaged ${fmtSignedNum(q.meanRewardDeltaOnChanges)} per-step reward vs baseline; on unchanged steps ${fmtSignedNum(q.meanRewardDeltaOnUnchanged)} (the control). A real decision gain shows AT the changes, not everywhere — heuristic, not causal.</p>`
      : `<p class="card-sub">Too few changed steps carry a reward (${q.scoredChangedSteps}) to read decision quality.</p>`
  const objDelta =
    typeof diff.objectiveDelta === 'number'
      ? `<p class="card-sub">objective Δ ${fmtSignedNum(diff.objectiveDelta)} <span class="card-sub">— context, not the verdict${(q.verdict === 'better' && diff.objectiveDelta <= 0) || (q.verdict === 'worse' && diff.objectiveDelta >= 0) ? '; note it DISAGREES with the decision read — steer by the decisions' : ''}</span></p>`
      : ''
  const deltaEntries = Object.entries(diff.actionCountDeltas).sort(
    (x, y) => Math.abs(y[1]) - Math.abs(x[1]),
  )
  const countTable = deltaEntries.length
    ? `<table class="kv-table report-table"><thead><tr><th>action</th><th class="num">count Δ</th></tr></thead><tbody>${deltaEntries
        .map(
          ([k, v]) =>
            `<tr><th>${escapeHtml(k)}</th><td class="num ${v >= 0 ? 'delta-pos' : 'delta-neg'}">${v >= 0 ? '+' : ''}${v}</td></tr>`,
        )
        .join('')}</tbody></table>`
    : '<p class="card-sub">No change in the action mix.</p>'
  const confShift =
    typeof diff.meanConfidenceShift === 'number'
      ? `<p class="card-sub">mean confidence shift ${fmtSignedNum(diff.meanConfidenceShift)}</p>`
      : ''
  return `<h3>Decision diff <span class="card-sub">— how the tweak changed the decisions, not just the score</span></h3>
    ${divergence}
    ${control}
    ${objDelta}
    ${confShift}
    <h4 class="card-sub">Action mix shift (full rollout)</h4>
    ${countTable}
    ${decisionDivergenceStripHtml(diff, null, labelA, labelB)}`
}
// A one-line decision-diff read for the 2-run Discuss seed, so the agent can reason about the delta.
function decisionDiffChatSummary(baseline, tweak) {
  const diff = diffDecisionTraces(baseline, tweak)
  if (!diff || !diff.aligned) return ''
  const q = diff.quality
  const deltas = Object.entries(diff.actionCountDeltas)
    .sort((x, y) => Math.abs(y[1]) - Math.abs(x[1]))
    .map(([k, v]) => `${k}${v >= 0 ? '+' : ''}${v}`)
    .join(', ')
  return `Decision diff vs the other run: ${(diff.divergenceRate * 100).toFixed(0)}% of ${diff.alignedSteps} aligned decisions changed; decision-quality reads "${q.verdict}" (reward Δ ${fmtSignedNum(q.meanRewardDeltaOnChanges)} at changed steps vs ${fmtSignedNum(q.meanRewardDeltaOnUnchanged)} control — heuristic, not causal); objective Δ ${fmtSignedNum(diff.objectiveDelta)}; action-mix Δ ${deltas || 'none'}.`
}
// --- xAI tab: model internals + cross-run config effects + the experiment recommender -------------
// All analysis runs in the browser via window.Xai (a parity-tested mirror of the TS engine) over the
// run records already in runsCache — deterministic, non-LLM, re-runnable anytime.
function xaiRuns() {
  return runsCache
    .filter(
      (r) => r.summary && r.summary.status !== 'failed' && typeof r.summary.objective === 'number',
    )
    .map((r) => ({
      key: r.key,
      config: r.summary.config || {},
      metrics: r.summary.metrics,
      objective: r.summary.objective,
      durationMs: r.summary.durationMs,
      seed: r.summary.seed,
      dataset: r.summary.dataset,
      status: 'completed',
    }))
}
function xaiCriteria() {
  const list = [{ key: 'objective', label: objectiveName(), dir: objectiveDirection() }]
  for (const k of runMetricKeys())
    if (k !== 'objective') list.push({ key: k, label: metricLabel(k), dir: 'max' })
  list.push({ key: 'durationMs', label: 'runtime', dir: 'min' })
  return list
}
function currentXaiCriterion() {
  const all = xaiCriteria()
  const found = all.find((c) => c.key === xaiCriterionKey) || all[0]
  return { key: found.key, direction: xaiCriterionDir || found.dir, label: found.label }
}
function renderXai() {
  const body = byId('xai-body')
  if (!body) return
  if (!manifest) {
    setHtml(
      body,
      '<div class="card"><p class="card-sub">Open a training project to analyse its runs.</p></div>',
    )
    return
  }
  if (!window.Xai) {
    setHtml(body, '<div class="card"><p class="card-sub">xAI engine failed to load.</p></div>')
    return
  }
  const runs = xaiRuns()
  const criterion = currentXaiCriterion()
  setHtml(
    body,
    [
      xaiHeaderHtml(criterion, runs.length),
      xaiNarrativeHtml(runs, criterion),
      xaiFocusKey ? xaiModelInternalsHtml(xaiFocusKey) : '',
      xaiConfigEffectsHtml(runs, criterion),
      xaiSurrogateHtml(runs, criterion),
      xaiRecommenderHtml(runs, criterion),
    ].join(''),
  )
}
function xaiHeaderHtml(criterion, nRuns) {
  const opts = xaiCriteria()
    .map(
      (c) =>
        `<option value="${escapeHtml(c.key)}"${c.key === xaiCriterionKey ? ' selected' : ''}>${escapeHtml(c.label)}</option>`,
    )
    .join('')
  return `<div class="card">
    <div class="card-head card-head-row"><h2>xAI <span class="card-sub">— ${nRuns} runs · deterministic, non-LLM</span></h2></div>
    <p class="badges-row">
      <label class="card-sub">Criterion <select id="xai-criterion">${opts}</select></label>
      <label class="card-sub">Better when <select id="xai-direction">
        <option value="max"${criterion.direction === 'max' ? ' selected' : ''}>higher</option>
        <option value="min"${criterion.direction === 'min' ? ' selected' : ''}>lower</option>
      </select></label>
    </p>
  </div>`
}
// The one-shot LLM narrative of the FOCUSED run — what this model does, why, how trustworthy, vs its
// sibling, what to try next. Per run (keyed by run key); the button becomes "Refresh (N new runs)" as the
// cross-run context drifts. With no run focused, a slim hint points the user at how to focus one.
function xaiNarrativeHtml(runs, criterion) {
  if (!xaiFocusKey) {
    return `<div class="card"><div class="card-head card-head-row"><h3>Narrative <span class="card-sub">— LLM synthesis, per run</span></h3></div>
      <p class="card-sub">Focus a run (open it in Runs → “Analyze in xAI”) to generate a one-shot narrative of what that model is doing and why.</p></div>`
  }
  const rec = xaiNarrativeCache.get(xaiFocusKey)
  const hasNarrative = !!(rec && rec.narrative)
  const since = hasNarrative ? Math.max(0, runs.length - (Number(rec.runCount) || 0)) : 0
  const staleCriterion =
    hasNarrative && rec.criterionKey && rec.criterionKey !== criterion.key ? rec.criterionKey : ''
  const label = !hasNarrative
    ? 'Generate narrative'
    : since > 0
      ? `Refresh (${since} new run${since === 1 ? '' : 's'})`
      : 'Refresh'
  const btn = `<button type="button" data-xai-narrate${narrating || !embedded() ? ' disabled' : ''}>${
    narrating ? `${spinnerHtml()} Narrating…` : escapeHtml(label)
  }</button>`
  const body = hasNarrative
    ? `<p class="xai-narrative">${escapeHtml(rec.narrative)}</p>
       <p class="card-sub">${escapeHtml(rec.narratedBy || 'AI')}${rec.narratedAt ? ` · ${escapeHtml(formatWhen(rec.narratedAt))}` : ''}${
         staleCriterion ? ` · generated for the “${escapeHtml(staleCriterion)}” criterion` : ''
       }${since > 0 ? ` · <strong>${since}</strong> new run${since === 1 ? '' : 's'} since — refresh to update` : ' · up to date'}</p>`
    : `<p class="card-sub">A one-shot LLM read of THIS run — its decisions, what drives them, how trustworthy the explanation is, and what to try next. Synthesises the analysis below; it doesn't replace it.</p>`
  return `<div class="card">
    <div class="card-head card-head-row"><h3>Narrative <span class="card-sub">— run ${escapeHtml(shortKey(xaiFocusKey))}</span></h3>${btn}</div>
    ${body}
    <p id="xai-status" class="form-status" role="status" hidden></p>
  </div>`
}
// Model internals for the focused run: the Explain panels + the decision-internals reads + a decision
// diff against the nearest comparable run (same data, differs by a lever).
function xaiModelInternalsHtml(focusKey) {
  const run = runsCache.find((r) => r.key === focusKey)
  if (!run) return ''
  const s = run.summary
  const sibling = xaiBestSibling(run)
  return `<div class="card">
    <div class="card-head card-head-row"><h3>Model internals <span class="card-sub">— run ${escapeHtml(shortKey(focusKey))}</span></h3>
      <div class="head-actions">
        ${chatAboutRunAvailable() ? `<button type="button" class="ghost-btn" data-xai-discuss="${escapeHtml(focusKey)}" title="Discuss the FULL xAI analysis of this run with the AI">${iconChatSvg()} Discuss xAI</button>` : ''}
        <button type="button" class="ghost-btn" data-xai-clear-focus title="Stop focusing this run">✕ Clear run</button>
      </div></div>
    <div class="card-scroll">
    ${explainSectionHtml(s)}
    ${decisionInternalsHtml(readDecisionTrace(s))}
    ${sibling ? decisionDiffSectionHtml(sibling.summary, s) : ''}
    </div></div>`
}
// The nearest comparable run for a decision diff: same dataset/window (step-alignable), has a trace,
// differs in config — the best such by objective.
function xaiBestSibling(run) {
  const sig = datasetAlignmentSignature(run.summary)
  if (!sig) return null
  const candidates = runsCache.filter(
    (r) =>
      r.key !== run.key &&
      r.summary &&
      r.summary.status !== 'failed' &&
      datasetAlignmentSignature(r.summary) === sig &&
      readDecisionTrace(r.summary),
  )
  if (!candidates.length) return null
  const dir = objectiveDirection()
  return candidates.sort((a, b) =>
    dir === 'max'
      ? b.summary.objective - a.summary.objective
      : a.summary.objective - b.summary.objective,
  )[0]
}
// Decision INTERNALS from the trace: decisiveness (top-2 action-value gap), policy entropy over time
// (normalised), and a confidence-vs-realised-reward calibration table ("is its confidence trustworthy?").
function decisionInternalsHtml(trace) {
  if (!trace) return ''
  const withVals = trace.steps.filter((s) => s.actionValues)
  if (withVals.length < 2) return ''
  const gap = []
  const entropy = []
  withVals.forEach((s, i) => {
    const vals = Object.values(s.actionValues).map(Number).filter(Number.isFinite)
    if (vals.length < 2) return
    const sorted = [...vals].sort((a, b) => b - a)
    gap.push({ x: i, y: sorted[0] - sorted[1], group: 'top-2 gap' })
    const mx = Math.max(...vals)
    const ex = vals.map((v) => Math.exp(v - mx))
    const z = ex.reduce((a, b) => a + b, 0)
    const ps = ex.map((e) => e / z)
    const h = -ps.reduce((a, p) => a + (p > 0 ? p * Math.log(p) : 0), 0) / Math.log(vals.length)
    entropy.push({ x: i, y: h, group: 'entropy' })
  })
  const decisiveChart =
    gap.length >= 2
      ? `<h4 class="card-sub">Decisiveness — gap between the best and 2nd-best action value (low ⇒ indecision)</h4>
       <div class="chart-wrap">${buildLineChart({ points: gap, xLabel: 'step', yLabel: 'gap', width: 640, height: 140, markers: false, groupColors: new Map([['top-2 gap', CHART_PALETTE[2]]]), ariaLabel: 'decisiveness over time' })}</div>`
      : ''
  const entropyChart =
    entropy.length >= 2
      ? `<h4 class="card-sub">Policy entropy over time (normalised 0–1; high ⇒ uncertain, ~0 ⇒ degenerate)</h4>
       <div class="chart-wrap">${buildLineChart({ points: entropy, xLabel: 'step', yLabel: 'entropy', width: 640, height: 140, markers: false, groupColors: new Map([['entropy', CHART_PALETTE[5]]]), ariaLabel: 'policy entropy over time' })}</div>`
      : ''
  return `${decisiveChart}${entropyChart}${calibrationTableHtml(trace)}`
}
// Confidence calibration: bin steps by confidence, show the mean realised reward per bin — if higher
// confidence doesn't track better realised reward, the policy's confidence isn't trustworthy.
function calibrationTableHtml(trace) {
  const pts = trace.steps.filter(
    (s) => typeof s.confidence === 'number' && typeof s.reward === 'number',
  )
  if (pts.length < 10) return ''
  const bins = [0.2, 0.4, 0.6, 0.8, 1.01]
  const rows = bins
    .map((hi, bi) => {
      const lo = bi === 0 ? 0 : bins[bi - 1]
      const inBin = pts.filter((s) => s.confidence >= lo && s.confidence < hi)
      if (!inBin.length) return ''
      const meanR = inBin.reduce((a, s) => a + s.reward, 0) / inBin.length
      const cls = meanR >= 0 ? 'delta-pos' : 'delta-neg'
      return `<tr><th>${lo.toFixed(1)}–${(hi > 1 ? 1 : hi).toFixed(1)}</th><td class="num">${inBin.length}</td>
        <td class="num ${cls}">${escapeHtml(formatTickValue(meanR))}</td></tr>`
    })
    .join('')
  if (!rows) return ''
  return `<h4 class="card-sub">Confidence calibration — mean realised reward by confidence bin (heuristic)</h4>
    <table class="kv-table report-table"><thead><tr><th>confidence</th><th class="num">steps</th><th class="num">mean reward</th></tr></thead><tbody>${rows}</tbody></table>`
}
function xaiConfigEffectsHtml(runs, criterion) {
  const importances = window.Xai.leverImportances(runs, criterion)
  if (!importances.length) {
    return `<div class="card"><div class="card-head card-head-row"><h3>Config effects</h3></div>
      <p class="card-sub">Need ≥2 runs that vary a lever (on the same data) to analyse config effects.</p></div>`
  }
  if (!xaiLever || !importances.some((i) => i.lever === xaiLever)) xaiLever = importances[0].lever
  const leverOpts = importances
    .map(
      (i) =>
        `<option value="${escapeHtml(i.lever)}"${i.lever === xaiLever ? ' selected' : ''}>${escapeHtml(i.lever)}</option>`,
    )
    .join('')
  const contrasts = window.Xai.ofatContrasts(runs, xaiLever, criterion)
  const contrastHtml = contrasts.length
    ? contrasts.map((c) => xaiOfatContrastHtml(c)).join('')
    : `<p class="card-sub">No clean one-factor contrast for <code>${escapeHtml(xaiLever)}</code> yet — no runs vary only this lever with everything else fixed. The recommender below can fill the gap.</p>`
  return `<div class="card">
    <div class="card-head card-head-row"><h3>Config effects <span class="card-sub">— which levers move ${escapeHtml(criterion.label)}</span></h3></div>
    <div class="card-scroll">
    ${xaiMethodNoteHtml()}
    <h4 class="card-sub">Lever importance <span class="card-sub">(screening — spread of each lever's marginal; confounded, use the contrast below for the controlled read)</span></h4>
    ${xaiImportanceTableHtml(importances)}
    <h4 class="card-sub">One-factor effect — <label>lever <select id="xai-lever">${leverOpts}</select></label> <span class="card-sub">holding everything else fixed</span></h4>
    ${contrastHtml}
    </div></div>`
}
// How the numbers are computed + the honesty caveat, so a user doesn't over-trust a 2-run estimate.
function xaiMethodNoteHtml() {
  return `<details class="xai-note"><summary class="card-sub">How these numbers are computed</summary>
    <p class="card-sub">Each value's score is the <strong>interquartile mean (IQM)</strong> of its runs (robust to a lucky/unlucky seed); the 95% CI is a deterministic <strong>bootstrap</strong>. An effect is <strong>“significant”</strong> only when the bootstrap CI of the <em>difference</em> excludes 0 (after a Benjamini-Hochberg multiple-comparison correction) — overlapping value-CIs do <em>not</em> by themselves mean “no effect”. <strong>Importance</strong> is how much a lever's marginal score spreads versus the others — a screening hint only (it ignores interactions and is confounded when runs differ on other levers; trust the controlled one-factor contrast). ⚠ Any estimate built from fewer than ${5} runs per value is unreliable — use <strong>Suggested experiments</strong> below to add seeds first.</p>
  </details>`
}
function xaiImportanceTableHtml(importances) {
  const rows = importances
    .map((i) => {
      const pct = `${Math.round(i.importance * 100)}%`
      const impCell = i.confident
        ? `<td class="num">${pct}</td>`
        : `<td class="num card-sub" title="only ${i.minRuns} run(s) for some value — unreliable">⚠ ${pct}</td>`
      const dataCell = i.confident
        ? `<td class="num">${i.minRuns}</td>`
        : `<td class="num"><button type="button" class="ghost-btn" data-xai-scroll-recs title="Jump to Suggested experiments">${i.minRuns} ⤓ run more</button></td>`
      return `<tr><th><code>${escapeHtml(i.lever)}</code></th>${impCell}
        <td>${escapeHtml(i.bestValue)}</td><td>${escapeHtml(i.worstValue)}</td><td class="num">${i.values}</td>${dataCell}</tr>`
    })
    .join('')
  return `<table class="kv-table report-table"><thead><tr><th>lever</th><th class="num">importance</th><th>best</th><th>worst</th><th class="num">#vals</th><th class="num">min runs/val</th></tr></thead><tbody>${rows}</tbody></table>`
}
function xaiOfatContrastHtml(c) {
  const fmt = (v) => escapeHtml(formatTickValue(v))
  const thin = c.levels.some((l) => l.seeds < 5)
  const levelRows = c.levels
    .map((l) => {
      const seedCell =
        l.seeds < 5
          ? `<td class="num card-sub" title="too few seeds to trust">⚠ ${l.seeds}</td>`
          : `<td class="num">${l.seeds}</td>`
      return `<tr><th>${escapeHtml(l.value)}</th><td class="num">${fmt(l.aggregate.iqm)}</td>
        <td class="num">[${fmt(l.aggregate.ci[0])}, ${fmt(l.aggregate.ci[1])}]</td>${seedCell}</tr>`
    })
    .join('')
  const effectRows = c.effects
    .map((e) => {
      const cls = e.delta >= 0 ? 'delta-pos' : 'delta-neg'
      const verdict = e.significant
        ? '<span class="badge is-ok">significant</span>'
        : '<span class="badge">not significant</span>'
      return `<tr><th>${escapeHtml(e.to)} vs ${escapeHtml(e.from)}</th>
        <td class="num ${cls}">${e.delta >= 0 ? '+' : ''}${fmt(e.delta)}</td>
        <td class="num">[${fmt(e.diffCi[0])}, ${fmt(e.diffCi[1])}]</td><td>${verdict}</td></tr>`
    })
    .join('')
  const ctx = (c.controlSignature || '').split('||')[0]
  const thinNote = thin
    ? `<p class="card-sub">⚠ Some values have &lt;5 seeds — these effects aren't trustworthy yet. <button type="button" class="ghost-btn" data-xai-scroll-recs>⤓ Suggested batch</button></p>`
    : ''
  return `<div class="xai-contrast">
    <p class="card-sub">held fixed: <code>${escapeHtml(ctx || '(only this lever varies)')}</code></p>
    <table class="kv-table report-table"><thead><tr><th>${escapeHtml(c.lever)}</th><th class="num">IQM</th><th class="num">95% CI</th><th class="num">seeds</th></tr></thead><tbody>${levelRows}</tbody></table>
    <table class="kv-table report-table"><thead><tr><th>effect</th><th class="num">Δ</th><th class="num">diff CI (excl. 0 ⇒ real)</th><th>verdict</th></tr></thead><tbody>${effectRows}</tbody></table>
    ${thinNote}
  </div>`
}
// Phase 3: a seeded random-forest surrogate over (config → criterion) drives a global fANOVA importance,
// the ablation TREE (worst→best, one change at a time), and a 2-lever interaction heatmap — the
// across-the-whole-space view (predicts unobserved configs, so it fills the OFAT gaps). Deterministic.
function xaiSurrogateHtml(runs, criterion) {
  const surrogate = window.Xai.fitConfigSurrogate(runs, criterion)
  if (!surrogate.trees.length) return '' // <2 runs or no levers — nothing to model
  const fanova = window.Xai.fanovaImportances(surrogate, runs, criterion)
  const path = window.Xai.ablationPath(surrogate, runs, criterion)
  const leverNames = surrogate.levers.map((l) => l.name)
  const ranked = fanova.map((f) => f.lever)
  if (!xaiInterA || !leverNames.includes(xaiInterA)) xaiInterA = ranked[0] || leverNames[0] || null
  if (!xaiInterB || !leverNames.includes(xaiInterB) || xaiInterB === xaiInterA)
    xaiInterB =
      ranked.find((l) => l !== xaiInterA) || leverNames.find((l) => l !== xaiInterA) || null
  const grid =
    xaiInterA && xaiInterB
      ? window.Xai.interactionGrid(surrogate, runs, criterion, xaiInterA, xaiInterB)
      : null
  return `<div class="card">
    <div class="card-head card-head-row"><h3>Surrogate model <span class="card-sub">— global importance · ablation tree · interactions (predicts unobserved configs)</span></h3></div>
    <div class="card-scroll">
    <p class="card-sub">A seeded random forest fit on every run's (config → ${escapeHtml(criterion.label)}). It predicts the criterion for configs you HAVEN'T run, so it can rank levers globally and walk an ablation path — a model, so treat it as a hypothesis to confirm with real runs, not ground truth.</p>
    ${xaiFanovaHtml(fanova)}
    ${xaiAblationTreeHtml(path, criterion)}
    ${xaiInteractionHtml(grid, leverNames)}
    </div></div>`
}
function xaiFanovaHtml(fanova) {
  if (!fanova.length) return ''
  const rows = fanova
    .map(
      (f) =>
        `<tr><th><code>${escapeHtml(f.lever)}</code></th><td class="num">${Math.round(f.importance * 100)}%</td><td class="num">${f.values}</td></tr>`,
    )
    .join('')
  return `<h4 class="card-sub">fANOVA importance <span class="card-sub">— variance each lever explains in the surrogate (captures interactions)</span></h4>
    <table class="kv-table report-table"><thead><tr><th>lever</th><th class="num">importance</th><th class="num">#vals</th></tr></thead><tbody>${rows}</tbody></table>`
}
function xaiAblationTreeHtml(path, criterion) {
  if (!path || !path.steps.length) return ''
  const fmt = (v) => escapeHtml(formatTickValue(v))
  const steps = path.steps
    .map((s, i) => {
      const cls = s.gain >= 0 ? 'delta-pos' : 'delta-neg'
      return `<li class="xai-abl-step"><span class="card-sub">${i + 1}.</span> <code>${escapeHtml(s.lever)}</code> ${escapeHtml(s.from)} → <strong>${escapeHtml(s.to)}</strong>
        <span class="num ${cls}">${s.gain >= 0 ? '+' : ''}${fmt(s.gain)}</span> <span class="card-sub">→ ${fmt(s.predicted)}</span></li>`
    })
    .join('')
  return `<h4 class="card-sub">Ablation tree <span class="card-sub">— worst→best config, the single change at each step that helps most (predicted)</span></h4>
    <p class="card-sub">baseline ${fmt(path.baselinePredicted)} → incumbent ${fmt(path.incumbentPredicted)}</p>
    <ol class="xai-abl">${steps}</ol>`
}
function xaiInteractionHtml(grid, leverNames) {
  if (leverNames.length < 2) return ''
  const sel = (id, current) =>
    `<select id="${id}">${leverNames.map((l) => `<option value="${escapeHtml(l)}"${l === current ? ' selected' : ''}>${escapeHtml(l)}</option>`).join('')}</select>`
  const picker = `<h4 class="card-sub">Interaction — <label>rows ${sel('xai-inter-a', xaiInterA)}</label> × <label>cols ${sel('xai-inter-b', xaiInterB)}</label> <span class="card-sub">does one help universally or only at some value of the other?</span></h4>`
  if (!grid)
    return `${picker}<p class="card-sub">Pick two different levers to see their interaction.</p>`
  const all = grid.cells.filter((v) => Number.isFinite(v))
  const lo = Math.min(...all)
  const hi = Math.max(...all)
  const shade = (v) => {
    const t = hi > lo ? (v - lo) / (hi - lo) : 0.5
    // green (better=high) gradient; the criterion direction is already baked into "higher prediction".
    return `background:rgba(34,197,94,${(0.12 + t * 0.5).toFixed(2)})`
  }
  const head = `<tr><th></th>${grid.valuesB.map((b) => `<th class="num">${escapeHtml(b)}</th>`).join('')}</tr>`
  const body = grid.valuesA
    .map((a, i) => {
      const cells = grid.valuesB
        .map((b, j) => {
          const v = grid.cells[i * grid.valuesB.length + j]
          return `<td class="num" style="${shade(v)}" title="${escapeHtml(grid.leverA)}=${escapeHtml(a)}, ${escapeHtml(grid.leverB)}=${escapeHtml(b)}">${escapeHtml(formatTickValue(v))}</td>`
        })
        .join('')
      return `<tr><th>${escapeHtml(a)}</th>${cells}</tr>`
    })
    .join('')
  return `${picker}<div class="compare-table-wrap"><table class="kv-table report-table xai-heat"><thead>${head}</thead><tbody>${body}</tbody></table></div>
    <p class="card-sub">cells = surrogate-predicted ${escapeHtml(grid.leverA)} (rows) × ${escapeHtml(grid.leverB)} (cols); greener = higher predicted value.</p>`
}
function xaiRecommenderHtml(runs, criterion) {
  xaiRecsCache = window.Xai.recommendExperiments(runs, criterion)
  if (!xaiRecsCache.length) {
    return `<div class="card" id="xai-recommender"><div class="card-head card-head-row"><h3>Suggested experiments</h3>
      <button type="button" class="ghost-btn" data-xai-propose${proposing || !embedded() ? ' disabled' : ''} title="Ask the AI to propose NEW experiments from the xAI analysis (lands in Hypotheses)">${proposing ? `${spinnerHtml()} Proposing…` : 'Propose with AI'}</button></div>
      <p class="card-sub">No deterministic gaps — the grids you've explored are complete and well-seeded. 🎉 Ask the AI to propose experiments beyond the explored grid.</p></div>`
  }
  const total = xaiRecsCache.reduce((a, r) => a + r.runCount, 0)
  const cards = xaiRecsCache.map((r, i) => xaiRecCardHtml(r, i)).join('')
  return `<div class="card" id="xai-recommender">
    <div class="card-head card-head-row"><h3>Suggested experiments <span class="card-sub">— ${xaiRecsCache.length} gaps · ${total} runs</span></h3>
      <div class="head-actions">
        <label class="card-sub">parallel <input type="number" id="xai-batch-concurrency" min="1" step="1" value="${savedConcurrency()}" style="width:3.2em" /></label>
        <button type="button" class="ghost-btn" data-xai-propose${proposing || !embedded() ? ' disabled' : ''} title="Ask the AI to propose NEW experiments from the xAI analysis (lands in Hypotheses)">${proposing ? `${spinnerHtml()} Proposing…` : 'Propose with AI'}</button>
        <button type="button" class="ghost-btn" data-xai-run-all>Run all (${total})</button>
      </div></div>
    <div class="card-scroll">${cards}</div></div>`
}
function xaiRecCardHtml(r, i) {
  return `<div class="xai-rec badges-row">
    <span class="badge">${escapeHtml(r.kind)}</span>
    <span>${escapeHtml(r.reason)}</span>
    <span class="card-sub">${r.runCount} run${r.runCount === 1 ? '' : 's'}</span>
    <button type="button" class="ghost-btn" data-xai-run-rec="${i}">Run batch</button>
  </div>`
}
async function xaiLaunchBatch(specs, label) {
  const input = byId('xai-batch-concurrency')
  const concurrency = Math.max(1, Math.floor(Number(input && input.value)) || savedConcurrency())
  try {
    for (const spec of specs) {
      await startOrEnqueue('train', trainerComputeParams({ spec, concurrency }), label)
    }
    showTab('activity')
  } catch {
    setStatusLine('xai-status', 'Could not launch the batch — please try again.', true)
  }
}
// Load ONE run's narrative record into the cache (or drop it when none).
async function loadXaiNarrative(runKey) {
  if (!manifest || !runKey) return
  const recs = await queryRecords(`${manifest.recordType}-xai-narrative`, runKey)
  if (recs && recs[0] && recs[0].content) xaiNarrativeCache.set(runKey, recs[0].content)
  else xaiNarrativeCache.delete(runKey)
}
async function refreshXai() {
  if (xaiFocusKey) await loadXaiNarrative(xaiFocusKey)
  renderXai()
}
async function onXaiNarrateClick() {
  if (narrating || !xaiFocusKey) return
  if (!embedded()) {
    setStatusLine('xai-status', 'Open inside the Overseer to generate a narrative.', false)
    return
  }
  const criterion = currentXaiCriterion()
  const sibling = xaiBestSibling(findRun(xaiFocusKey))
  setStatusLine('xai-status', '')
  try {
    const result = await startOrEnqueue(
      'xai-narrate',
      trainerActivityParams({
        runKey: xaiFocusKey,
        siblingKey: sibling ? sibling.key : undefined,
        criterionKey: criterion.key,
        criterionDir: criterion.direction,
        criterionLabel: criterion.label,
      }),
      'xAI narrative',
    )
    if (result.queued) setStatusLine('xai-status', queuedStatusText(result.ahead))
  } catch {
    setStatusLine('xai-status', 'Could not start the narrative — please try again.', true)
  }
}
// Enrich the existing LLM proposer with the deterministic xAI signal (top levers + the criterion +
// the recommender's gaps), then route the user to Hypotheses where proposals land.
function xaiProposeInstructions() {
  const runs = xaiRuns()
  const criterion = currentXaiCriterion()
  const top = window.Xai.leverImportances(runs, criterion)
    .slice(0, 4)
    .map(
      (i) =>
        `${i.lever} (${Math.round(i.importance * 100)}% importance${i.confident ? '' : ', low data'}, best≈${i.bestValue})`,
    )
    .join(', ')
  const gaps = xaiRecsCache
    .slice(0, 4)
    .map((r) => r.reason)
    .join('; ')
  return [
    `Optimise for the "${criterion.label}" criterion (${criterion.direction} is better).`,
    top
      ? `Deterministic xAI screening of the ${runs.length} runs so far ranks the levers: ${top}.`
      : '',
    gaps ? `Under-explored regions the deterministic recommender flagged: ${gaps}.` : '',
    `Prioritise experiments that exploit the best-known region AND test promising untried values; avoid configs already run.`,
  ]
    .filter(Boolean)
    .join(' ')
}
async function onXaiProposeClick() {
  if (!embedded()) {
    setStatusLine('xai-status', 'Open inside the Overseer to propose experiments.', false)
    return
  }
  if (proposing) {
    showTab('hypotheses')
    return
  }
  setStatusLine('xai-status', '')
  try {
    const result = await startOrEnqueue(
      'propose',
      trainerActivityParams({ instructions: xaiProposeInstructions() }),
      'Propose experiments',
    )
    showTab('hypotheses')
    if (result.queued) setStatusLine('hypotheses-status', queuedStatusText(result.ahead))
  } catch {
    setStatusLine('xai-status', 'Could not start proposing — please try again.', true)
  }
}
// Open the xAI tab focused on a run's internals (the run-detail entry point).
function analyzeInXai(key) {
  xaiFocusKey = key
  showTab('xai')
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
  // Re-run queues the EXACT same config again to be tried (header ⧉ clones to Launch instead,
  // for tweaking before running). Only shown embedded, where queueing is possible.
  const rerun = embedded()
    ? `<p class="failure-rerun"><button type="button" class="ghost-btn" data-action="rerun" data-key="${escapeHtml(key)}" title="Queue this exact run again">Re-run</button></p>`
    : ''
  return `<div class="failure-detail">${err}${tail}${rerun}</div>`
}
function renderRunDetail(key) {
  const panel = byId('run-detail')
  if (!panel) return
  const run = findRun(key)
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
  // The pipeline version this run was produced under; a MAJOR-version gap means its scores aren't
  // comparable to the current pipeline (re-run to refresh) — a minor gap stays comparable.
  const outdated = runIsOutdated(run)
  const versionBit = s.pipelineVersion
    ? ` · <span class="${outdated ? 'badge is-bad' : ''}" title="Pipeline version this run ran under${outdated ? ` — a BREAKING (major) change has landed since v${escapeHtml(String(currentPipelineVersion()))}, so this run's scores aren't comparable. Re-run with the latest version.` : ' — runs from different MAJOR versions aren’t comparable.'}">pipeline v${escapeHtml(String(s.pipelineVersion))}${outdated ? ' · outdated' : ''}</span>`
    : ''
  const envBit = hasEnvLevers()
    ? ` · <span title="${escapeHtml(runEnvSignature(run))}">env ${escapeHtml(runEnvName(run))}</span>`
    : ''
  const datasetNameBit = hasDatasetLevers()
    ? ` · <span title="${escapeHtml(runDatasetSignature(run))}">dataset ${escapeHtml(runDatasetName(run))}</span>`
    : ''
  const headline = failed
    ? '<span class="badge is-bad">failed</span>'
    : `${escapeHtml(objectiveName())} ${escapeHtml(formatObjective(s.objective))} · ${healthBadgeHtml(s.health)}`
  const html = `
    <div class="card-head card-head-row">
      <div>
        <h2>Run <code>${escapeHtml(shortKey(run.key))}</code></h2>
        <p class="card-sub">${headline} · seed ${escapeHtml(s.seed === undefined ? '—' : String(s.seed))}
          · ${escapeHtml(formatWhen(runRanAt(s)))}${datasetBadge ? ` · ${datasetBadge}` : ''}${versionBit}${datasetNameBit}${envBit}${unrunnableBadge}</p>
      </div>
      <div class="head-actions">
        ${embedded() && outdated ? `<button type="button" data-action="rerun" data-key="${escapeHtml(run.key)}" class="ghost-btn" title="Re-run this exact config under the current pipeline version (v${escapeHtml(String(currentPipelineVersion()))}) — a breaking version has landed since, so its scores are outdated">↻ Re-run with latest version</button>` : ''}
        <button type="button" data-action="clone" data-key="${escapeHtml(run.key)}" class="icon-btn" title="Clone to Launch" aria-label="Clone to Launch">⧉</button>
        <button type="button" data-action="xai" data-key="${escapeHtml(run.key)}" class="icon-btn" title="Analyze in xAI — internals, config effects, suggested experiments" aria-label="Analyze in xAI">🔬</button>
        <button type="button" data-action="chat" data-key="${escapeHtml(run.key)}" class="icon-btn"${chatAboutRunAvailable() ? '' : ' disabled'} title="Discuss this run with the AI" aria-label="Discuss this run">${iconChatSvg()}</button>
        <button type="button" data-action="toggle-unrunnable" data-key="${escapeHtml(run.key)}" class="icon-btn" title="${isUnrunnable ? 'Allow this setup to run again' : 'Mark unrunnable — skip on re-run (this pipeline version) unless forced'}" aria-label="${isUnrunnable ? 'Mark runnable' : 'Mark unrunnable'}">${isUnrunnable ? '⊙' : '⊘'}</button>
        <button type="button" data-action="delete-run" data-key="${escapeHtml(run.key)}" class="icon-btn icon-btn-danger" title="Delete this run (and its evaluation/verdict)" aria-label="Delete run">${iconDeleteSvg()}</button>
        <button type="button" id="run-detail-close" class="icon-btn" title="Close" aria-label="Close">✕</button>
      </div>
    </div>
    <div class="card-scroll">
    ${failed ? failureDetailHtml(s, run.key) : ''}
    ${flags.length ? `<h3>Health flags</h3><p class="badges-row">${flagChips}</p>` : ''}
    ${showVerdict ? verdictSectionHtml(verdictsCache.get(run.key)) : ''}
    ${showEval ? evaluationSectionHtml(run) : ''}
    <h3>Metrics</h3>
    ${metricsTableHtml(s.metrics)}
    ${oldRunChartHintHtml(s)}
    ${priceActionSectionHtml(s, run.key)}
    ${exitsSectionHtml(s)}
    ${regimesSectionHtml(s)}
    ${equityVsHoldSectionHtml(s) || trainingCurveSectionHtml(s)}
    ${explainSectionHtml(s)}
    ${ledgerSectionHtml(s)}
    <h3>Config</h3>
    <pre class="json">${escapeHtml(JSON.stringify(s.config || {}, null, 2))}</pre>
    <h3>Artifacts</h3>
    <p class="mono">${checkpoint ? escapeHtml(checkpoint) : '—'}</p>
    <p class="card-sub">configHash <code>${escapeHtml(run.key)}</code></p></div>`
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
  rememberRun(key)
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
  const run = findRun(key)
  if (!run) return
  showTab('launch')
  applyPresetFixed(run.summary.config || {})
}
// Build a one-config campaign spec that reproduces a run EXACTLY: every manifest lever it
// carried, fixed (non-lever keys dropped so the planner doesn't reject them; seed is itself a
// lever so it rides along). expandSpec layers the manifest defaults underneath, so the planned
// config — and its hash — matches the original run.
function reRunSpecForConfig(config) {
  const cfg = config || {}
  const leverKeys = new Set(leverEntries().map(([k]) => k))
  const fixed = {}
  for (const [k, v] of Object.entries(cfg)) {
    if (leverKeys.has(k) && v !== undefined) fixed[k] = v
  }
  return { sweep: {}, fixed }
}
// Queue the exact same run(s) again to be tried — refresh=true so an existing (e.g. failed) result
// doesn't skip them. The config — and therefore its hash + record key — matches the original, so each
// re-run UPDATES that run in place (it never creates a new record) and the fresh result carries the
// current pipelineVersion. Each becomes its own single-config 'train' activity; any beyond the compute
// budget park in the queue. Stays put (a toast confirms) rather than yanking the user to Activity.
async function reRunRuns(keys) {
  if (!embedded()) return
  const runs = keys.map((k) => findRun(k)).filter(Boolean)
  if (!runs.length) return
  for (const run of runs) {
    const params = trainerComputeParams({
      spec: reRunSpecForConfig(run.summary.config),
      refresh: true,
      concurrency: 1,
    })
    await startOrEnqueue('train', params, `Re-run ${shortKey(run.key)}`)
  }
  showToast(`Queued ${runs.length} run${runs.length === 1 ? '' : 's'} to re-run — see the Activity tab.`)
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
  const run = findRun(key)
  if (!run || !chatAboutRunAvailable()) return
  const s = run.summary
  const failed = s.status === 'failed'
  const logTail = Array.isArray(s.logTail) ? s.logTail.slice(-40).join('\n') : ''
  const verdict = verdictsCache.get(key)
  const runContext = [
    `You are discussing ONE specific, already-selected training run — run id "${shortKey(key)}"${failed ? ', which FAILED' : ''}. Its full configuration, metrics${verdict && verdict.why ? ', judge verdict' : ''}${failed ? ', error and recent logs' : ''} are all given below, so do NOT ask the user for the run id or any of these details — work directly from what follows.`,
    `Pipeline v${String(s.pipelineVersion || '1')}.`,
    failed
      ? ''
      : `Objective (${objectiveName()}): ${formatObjective(s.objective)} · health: ${(s.health && s.health.status) || 'unknown'}.`,
    s.metrics ? `Metrics:\n${JSON.stringify(s.metrics, null, 2)}` : '',
    failed ? '' : decisionTraceChatSummary(s),
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
// Discuss TWO selected runs: the config diff (which lever changed = the "new information"), both runs'
// objective/metrics, and the DECISION DIFF — so the agent can reason about whether the tweak's decision
// changes look good even when the score hasn't moved.
async function chatAboutRuns(keyA, keyB) {
  const a = runsCache.find((r) => r.key === keyA)
  const b = runsCache.find((r) => r.key === keyB)
  if (!a || !b || !chatAboutRunAvailable()) return
  const configDiff = Object.keys((manifest && manifest.levers) || {})
    .map((lk) => [lk, (a.summary.config || {})[lk], (b.summary.config || {})[lk]])
    .filter(([, va, vb]) => String(va) !== String(vb))
    .map(([lk, va, vb]) => `${lk}: ${va === undefined ? '—' : va} → ${vb === undefined ? '—' : vb}`)
  const ctx = [
    `You are comparing TWO training runs of this project — baseline "${shortKey(keyA)}" vs tweak "${shortKey(keyB)}". Work from the details below; do not ask for the run ids.`,
    configDiff.length
      ? `What changed (the "new information"):\n${configDiff.join('\n')}`
      : 'The two runs share the same config (compare seeds / nondeterminism).',
    `Objective (${objectiveName()}): ${formatObjective(a.summary.objective)} → ${formatObjective(b.summary.objective)}.`,
    decisionDiffChatSummary(a.summary, b.summary),
    `Help me judge whether the tweak improved the model's DECISIONS — not just the score. Steer by the decision diff (divergence, the reward delta AT the changed steps vs the control, the action-mix shift), treating it as a heuristic, not proof of causation.`,
  ].filter(Boolean)
  const systemPrompt = [projectChatPreamble(), ...ctx].filter(Boolean).join('\n\n')
  try {
    await window.OverseerBridge.discussTopic({
      title: `Runs ${shortKey(keyA)} vs ${shortKey(keyB)}`,
      seed: 'Did this tweak improve the decisions? Walk me through the decision diff.',
      systemPrompt,
    })
  } catch {
    setStatusLine('run-eval-status', 'Could not open chat — please try again.', true)
  }
}
// The comprehensive xAI seed for ONE run — far more than the Runs-detail chat: the decision-trace digest
// PLUS the attribution faithfulness (sanity check), the latent representation + probe, the run's standing
// + lever importances among all runs, and the decision diff vs its nearest comparable run.
function xaiRunChatSummary(run) {
  const s = run.summary
  const parts = []
  const traceSummary = decisionTraceChatSummary(s)
  if (traceSummary) parts.push(traceSummary)
  const trace = readDecisionTrace(s)
  const fa = trace && trace.featureAttribution
  if (fa && fa.sanityCheck) {
    const sc = fa.sanityCheck
    parts.push(
      `Attribution faithfulness (Adebayo model-randomization check on ${fa.method || 'saliency'}): ${sc.passed ? 'PASSED' : 'FAILED'} (rank corr ${formatTickValue(sc.rankCorrelation)}) — ${sc.passed ? 'the attribution changes when the weights are randomized, so it reflects the learned function' : 'the attribution barely changes when the weights are randomized, so it likely reflects the input/architecture, NOT what the model learned — do not over-trust it'}.`,
    )
  }
  const lm = trace && trace.latentMap
  if (lm && lm.probe && Number.isFinite(Number(lm.probe.accuracy))) {
    const acc = Math.round(Number(lm.probe.accuracy) * 100)
    const base = Math.round(Number(lm.probe.baseline || 0) * 100)
    parts.push(
      `Latent representation: the penultimate-layer activations project to 2-D retaining ${Math.round(Number(lm.varianceExplained || 0) * 100)}% of variance; a linear probe predicts the action from the latent at ${acc}% vs a ${base}% majority baseline — the representation ${acc > base + 5 ? 'linearly encodes the decision' : 'barely separates the actions linearly'}.`,
    )
  }
  if (window.Xai) {
    const runs = xaiRuns()
    const criterion = currentXaiCriterion()
    const ranked = runs
      .map((r) => ({ key: r.key, v: window.Xai.criterionValueOf(r, criterion) }))
      .filter((r) => r.v != null)
      .sort((a, b) => (criterion.direction === 'max' ? b.v - a.v : a.v - b.v))
    const rank = ranked.findIndex((r) => r.key === run.key)
    if (rank >= 0)
      parts.push(`This run ranks #${rank + 1} of ${ranked.length} by ${criterion.label}.`)
    const importances = window.Xai.leverImportances(runs, criterion)
    if (importances.length) {
      const top = importances
        .slice(0, 4)
        .map(
          (i) => `${i.lever} ${Math.round(i.importance * 100)}%${i.confident ? '' : ' (low data)'}`,
        )
        .join(', ')
      parts.push(
        `Lever importance for ${criterion.label} across all runs (which knobs move it — confounded screening): ${top}.`,
      )
    }
  }
  const sibling = xaiBestSibling(run)
  if (sibling) {
    const dd = decisionDiffChatSummary(sibling.summary, s)
    if (dd) parts.push(`Vs the nearest comparable run ${shortKey(sibling.key)} — ${dd}`)
  }
  return parts.join('\n\n')
}
// Discuss the FULL xAI analysis of one run (the xAI-tab entry point — richer than the Runs-detail chat).
async function chatAboutRunXai(key) {
  const run = findRun(key)
  if (!run || !chatAboutRunAvailable()) return
  const s = run.summary
  const ctx = [
    `You are discussing the FULL xAI analysis of ONE training run — id "${shortKey(key)}". Everything below is provided (config, metrics, the decision trace, input attribution + its faithfulness check, the reward breakdown, the latent representation + probe, the run's standing + lever importances among all runs, and the decision diff vs its nearest comparable run), so work directly from it — don't ask for the run id or these details. The xAI computations are DETERMINISTIC + heuristic; treat the attribution and decision-quality reads as EVIDENCE, not proof, and say so when a signal is weak (e.g. an attribution that FAILED its sanity check).`,
    `Pipeline v${String(s.pipelineVersion || '1')}. Objective (${objectiveName()}): ${formatObjective(s.objective)} · health: ${(s.health && s.health.status) || 'unknown'}.`,
    s.metrics ? `Metrics:\n${JSON.stringify(s.metrics, null, 2)}` : '',
    `Config:\n${JSON.stringify(s.config || {}, null, 2)}`,
    xaiRunChatSummary(run),
  ].filter(Boolean)
  const systemPrompt = [projectChatPreamble(), ...ctx].filter(Boolean).join('\n\n')
  try {
    await window.OverseerBridge.discussTopic({
      title: `xAI ${shortKey(key)}`,
      seed: 'Walk me through what this model is doing and why — the decisions it makes, what drives them, how trustworthy the explanation is, and what to try next.',
      systemPrompt,
    })
  } catch {
    setStatusLine('xai-status', 'Could not open chat — please try again.', true)
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
  const runKeys = [...runsCompareKeys].filter((k) => findRun(k))
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
      if (event.target.closest('#runs-dropdowns-toggle')) {
        runsDropdownsCollapsed = !runsDropdownsCollapsed
        renderRunsTable()
        return
      }
      if (event.target.closest('#runs-prev')) {
        runsPage = Math.max(0, runsPage - 1)
        refreshRuns()
        return
      }
      if (event.target.closest('#runs-next')) {
        runsPage += 1
        refreshRuns()
        return
      }
      if (event.target.closest('#runs-add-toggle')) {
        openCustomRulePopup(null)
        return
      }
      const delChip = event.target.closest('[data-rule-del]')
      if (delChip) {
        deleteCustomRule(delChip.dataset.ruleDel).then(() => {
          runsPage = 0
          refreshRuns()
        })
        return
      }
      // A click anywhere on the chip body (but not its checkbox/✕) edits the rule.
      const editChip = event.target.closest('[data-rule-edit]')
      if (editChip && !event.target.closest('.filter-chip-cb')) {
        openCustomRulePopup(editChip.dataset.ruleEdit)
        return
      }
      const viewBtn = event.target.closest('.runs-view-btn')
      if (viewBtn) {
        runsViewMode = viewBtn.dataset.view
        runsPage = 0
        refreshRuns()
        return
      }
      const th = event.target.closest('.runs-th[data-sort]')
      if (th) {
        toggleRunsSort(th.dataset.sort)
        return
      }
      if (event.target.closest('.run-compare-cb')) return // checkbox toggles compare, not detail
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
      const dsRow = event.target.closest('tr[data-dataset-sig]')
      if (dsRow) {
        drillIntoDataset(dsRow.dataset.datasetSig)
        return
      }
      const row = event.target.closest('tr[data-key]')
      if (row) openRunDetail(row.dataset.key)
    })
    body.addEventListener('change', (event) => {
      const cb = event.target.closest('.run-compare-cb')
      if (cb) {
        if (cb.checked) {
          runsCompareKeys.add(cb.dataset.key)
          rememberRun(cb.dataset.key)
        } else runsCompareKeys.delete(cb.dataset.key)
        disarmRunsDelete()
        renderCompare()
        syncRunsSelectionUI()
        return
      }
      if (event.target.id === 'runs-select-all') {
        const shownKeys = runsVisibleKeys
        if (event.target.checked)
          for (const k of shownKeys) {
            runsCompareKeys.add(k)
            rememberRun(k)
          }
        else for (const k of shownKeys) runsCompareKeys.delete(k)
        disarmRunsDelete()
        renderRunsTable()
        return
      }
      if (event.target.id === 'runs-version-filter') {
        runsVersionFilter = event.target.value
        runsPage = 0
        refreshRuns()
        return
      }
      const sel = event.target.closest('.runs-filter-lever')
      if (sel) {
        runsLeverFilter[sel.dataset.lever] = sel.value
        runsPage = 0
        refreshRuns()
        return
      }
      if (event.target.id === 'runs-hide-bad') {
        runsHideBad = event.target.checked
        runsPage = 0
        refreshRuns()
        return
      }
      const ruleCb = event.target.closest('[data-rule-toggle]')
      if (ruleCb) {
        const rule = customRulesCache.find((r) => r.id === ruleCb.dataset.ruleToggle)
        if (rule) {
          rule.active = ruleCb.checked
          runsPage = 0
          saveCustomRule(rule)
          refreshRuns()
        }
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
      const rerunBtn = event.target.closest('button[data-action="rerun"]')
      if (rerunBtn) reRunRuns([rerunBtn.dataset.key])
      const chatBtn = event.target.closest('button[data-action="chat"]')
      if (chatBtn) chatAboutRun(chatBtn.dataset.key)
      const xaiBtn = event.target.closest('button[data-action="xai"]')
      if (xaiBtn) analyzeInXai(xaiBtn.dataset.key)
      const unrunBtn = event.target.closest('button[data-action="toggle-unrunnable"]')
      if (unrunBtn) toggleUnrunnable(unrunBtn.dataset.key)
      const delBtn = event.target.closest('button[data-action="delete-run"]')
      if (delBtn) deleteRun(delBtn.dataset.key)
      const expandBtn = event.target.closest('button[data-action="expand-chart"]')
      if (expandBtn) expandPriceActionChart(expandBtn.dataset.key)
    })
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        closeChartModal()
        closeCustomRulePopup()
      }
    })
  }
  const compareCard = byId('run-compare')
  if (compareCard) {
    compareCard.addEventListener('click', (event) => {
      if (event.target.closest('#compare-clear')) {
        runsCompareKeys = new Set()
        renderRunsTable()
      }
      if (event.target.closest('#compare-discuss')) {
        const keys = [...runsCompareKeys]
        if (keys.length === 2) chatAboutRuns(keys[0], keys[1])
      }
      if (event.target.closest('#compare-rerun-all')) {
        reRunRuns([...runsCompareKeys])
      }
    })
  }
  const xaiBody = byId('xai-body')
  if (xaiBody) {
    xaiBody.addEventListener('change', (event) => {
      const t = event.target
      if (t.id === 'xai-criterion') {
        xaiCriterionKey = t.value
        xaiCriterionDir = null
        xaiLever = null
        renderXai()
      } else if (t.id === 'xai-direction') {
        xaiCriterionDir = t.value
        renderXai()
      } else if (t.id === 'xai-lever') {
        xaiLever = t.value
        renderXai()
      } else if (t.id === 'xai-inter-a') {
        xaiInterA = t.value
        renderXai()
      } else if (t.id === 'xai-inter-b') {
        xaiInterB = t.value
        renderXai()
      }
    })
    xaiBody.addEventListener('click', (event) => {
      const discussBtn = event.target.closest('[data-xai-discuss]')
      if (discussBtn) {
        chatAboutRunXai(discussBtn.dataset.xaiDiscuss)
        return
      }
      if (event.target.closest('[data-xai-narrate]')) {
        onXaiNarrateClick()
        return
      }
      if (event.target.closest('[data-xai-propose]')) {
        onXaiProposeClick()
        return
      }
      if (event.target.closest('[data-xai-clear-focus]')) {
        xaiFocusKey = null
        renderXai()
        return
      }
      if (event.target.closest('[data-xai-scroll-recs]')) {
        const recs = byId('xai-recommender')
        if (recs) {
          recs.scrollIntoView({ behavior: 'smooth', block: 'start' })
          recs.classList.add('xai-flash')
          setTimeout(() => recs.classList.remove('xai-flash'), 1600)
        }
        return
      }
      const recBtn = event.target.closest('[data-xai-run-rec]')
      if (recBtn) {
        const rec = xaiRecsCache[Number(recBtn.dataset.xaiRunRec)]
        if (rec) xaiLaunchBatch([rec.spec], `xAI: ${rec.kind}`)
        return
      }
      if (event.target.closest('[data-xai-run-all]')) {
        xaiLaunchBatch(
          xaiRecsCache.map((r) => r.spec),
          'xAI: fill gaps',
        )
        return
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
function environmentCardHtml(env, editable, isDefault) {
  const rows = envLeverEntries()
    .map(
      ([k]) =>
        `<tr><th>${escapeHtml(k)}</th><td class="num">${escapeHtml(env.settings[k] === undefined ? '—' : String(env.settings[k]))}</td></tr>`,
    )
    .join('')
  const setDefaultBtn =
    editable && !isDefault
      ? `<button type="button" class="icon-btn" data-env-default="${escapeHtml(env.id)}" title="Set as default environment" aria-label="Set as default">☆</button>`
      : ''
  const actions = editable
    ? `<div class="head-actions">
        ${setDefaultBtn}
        <button type="button" class="icon-btn" data-env-edit="${escapeHtml(env.id)}" title="Edit" aria-label="Edit">✎</button>
        <button type="button" class="icon-btn icon-btn-danger" data-env-delete="${escapeHtml(env.id)}" title="Delete" aria-label="Delete">${iconDeleteSvg()}</button>
      </div>`
    : `<div class="head-actions"><button type="button" class="icon-btn" data-env-clone="${escapeHtml(env.id)}" title="Duplicate to a new environment" aria-label="Duplicate">⧉</button></div>`
  const note = isDefault
    ? ' <span class="badge">★ default</span>'
    : editable
      ? ''
      : ' <span class="card-sub">(manifest defaults — clone to start)</span>'
  return `<div class="environment-card">
    <div class="card-head card-head-row">
      <h3>${escapeHtml(env.name)}${note}</h3>
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
  const defId = defaultEnvironmentId()
  // The manifest-defaults card is only a clone-to-start SEED for a project with no environments yet; once
  // the user has defined any, it's hidden (the chosen default lives among their records).
  const seed = environmentsCache.length ? '' : environmentCardHtml(defaultEnvironment(), false)
  setHtml(
    body,
    seed + environmentsCache.map((e) => environmentCardHtml(e, true, e.id === defId)).join(''),
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
  if (environmentDuplicateOf(name, settings, id)) {
    setStatusLine(
      'environments-status',
      'An environment with the same name or settings already exists.',
      true,
    )
    return
  }
  // Editing preserves the default flag; the FIRST environment a project gets becomes its default.
  const existing = environmentsCache.find((e) => e.id === id)
  const isDefault = existing ? !!existing.default : environmentsCache.length === 0
  try {
    await putEnvironment({ id, name, settings, default: isDefault })
  } catch {
    setStatusLine('environments-status', 'Could not save — please try again.', true)
    return
  }
  toggleEnvironmentForm(false)
  await renderEnvironments()
  renderLaunchForm()
}
async function onSetDefaultEnvironment(id) {
  await setDefaultEnvironment(id)
  await renderEnvironments()
  renderLaunchForm()
}
async function onDeleteEnvironment(id) {
  const wasDefault = !!(environmentsCache.find((e) => e.id === id) || {}).default
  try {
    await deleteEnvironmentRecord(id)
  } catch {
    setStatusLine('environments-status', 'Could not delete — please try again.', true)
    return
  }
  environmentsCache = environmentsCache.filter((e) => e.id !== id)
  // Removing the default promotes the next available environment (the picker always has a default).
  if (wasDefault && environmentsCache.length) await setDefaultEnvironment(environmentsCache[0].id)
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
      const setDef = event.target.closest('button[data-env-default]')
      if (setDef) {
        onSetDefaultEnvironment(setDef.dataset.envDefault)
        return
      }
      const del = event.target.closest('button[data-env-delete]')
      if (del) onDeleteEnvironment(del.dataset.envDelete)
    })
  }
}

// --- Datasets tab (named dataset-lever bundles — asset / window / fidelity) ------
function datasetCardHtml(ds, editable, isDefault) {
  const rows = datasetLeverEntries()
    .map(
      ([k]) =>
        `<tr><th>${escapeHtml(k)}</th><td>${escapeHtml(ds.settings[k] === undefined ? '—' : String(ds.settings[k]))}</td></tr>`,
    )
    .join('')
  const setDefaultBtn =
    editable && !isDefault
      ? `<button type="button" class="icon-btn" data-ds-default="${escapeHtml(ds.id)}" title="Set as default dataset" aria-label="Set as default">☆</button>`
      : ''
  const actions = editable
    ? `<div class="head-actions">
        ${setDefaultBtn}
        <button type="button" class="icon-btn" data-ds-edit="${escapeHtml(ds.id)}" title="Edit" aria-label="Edit">✎</button>
        <button type="button" class="icon-btn icon-btn-danger" data-ds-delete="${escapeHtml(ds.id)}" title="Delete" aria-label="Delete">${iconDeleteSvg()}</button>
      </div>`
    : `<div class="head-actions"><button type="button" class="icon-btn" data-ds-clone="${escapeHtml(ds.id)}" title="Duplicate to a new dataset" aria-label="Duplicate">⧉</button></div>`
  const note = isDefault
    ? ' <span class="badge">★ default</span>'
    : editable
      ? ''
      : ' <span class="card-sub">(manifest defaults — clone to start)</span>'
  return `<div class="environment-card">
    <div class="card-head card-head-row">
      <h3>${escapeHtml(ds.name)}${note}</h3>
      ${actions}
    </div>
    <table class="kv-table"><tbody>${rows}</tbody></table>
  </div>`
}
async function renderDatasets() {
  const body = byId('datasets-body')
  if (!body) return
  if (!embedded()) {
    setHtml(body, '<div class="empty-hint">Open inside the Overseer to manage datasets.</div>')
    return
  }
  if (!hasDatasetLevers()) {
    setHtml(
      body,
      '<div class="empty-hint">This project declares no dataset settings. Tag levers with <code>"scope": "dataset"</code> in the manifest (e.g. asset, walk-forward window, fidelity stack) to manage them as datasets here.</div>',
    )
    return
  }
  datasetsCache = await readDatasets()
  const defId = defaultDatasetId()
  // The manifest-defaults card is only a clone-to-start SEED for a project with no datasets yet; once the
  // user has defined any, it's hidden (the chosen default lives among their records, not this synthetic one).
  const seed = datasetsCache.length ? '' : datasetCardHtml(defaultDataset(), false)
  setHtml(body, seed + datasetsCache.map((d) => datasetCardHtml(d, true, d.id === defId)).join(''))
}
// A single-value, type-aware field for one dataset lever (choice → select, number → number input,
// boolean → true/false select), pre-filled with the dataset's saved value. "" = use the default.
function datasetFieldHtml(key, spec, value) {
  // A named dataset must pin a CONCRETE value for every lever — there is no "— default —" escape that would
  // leave it unpinned (and so resolve to a synonym at run time). A concrete option is always pre-selected:
  // the saved value when editing/cloning, else the manifest default, else the first concrete choice.
  const synonyms = new Set((window.Migrate && window.Migrate.INPUT_SYNONYMS) || [])
  let input
  if (spec.type === 'choice') {
    const choices = (spec.choices || []).map(String).filter((c) => !synonyms.has(c))
    const current = String(value)
    const selected = choices.includes(current)
      ? current
      : choices.includes(String(spec.default))
        ? String(spec.default)
        : choices[0]
    const opts = choices
      .map(
        (c) =>
          `<option value="${escapeHtml(c)}"${c === selected ? ' selected' : ''}>${escapeHtml(c)}</option>`,
      )
      .join('')
    input = `<select name="dataset:${escapeHtml(key)}" required>${opts}</select>`
  } else if (spec.type === 'boolean') {
    const sel =
      value === true || value === 'true'
        ? 'true'
        : value === false || value === 'false'
          ? 'false'
          : spec.default
            ? 'true'
            : 'false'
    input = `<select name="dataset:${escapeHtml(key)}" required>
      <option value="true"${sel === 'true' ? ' selected' : ''}>true</option>
      <option value="false"${sel === 'false' ? ' selected' : ''}>false</option></select>`
  } else {
    const { min, max } = leverRange(spec)
    const minAttr = Number.isFinite(Number(min)) ? ` min="${Number(min)}"` : ''
    const maxAttr = Number.isFinite(Number(max)) ? ` max="${Number(max)}"` : ''
    const v =
      value === undefined || value === '' ? (spec.default === undefined ? '' : spec.default) : value
    input = `<input type="number" required step="any" name="dataset:${escapeHtml(key)}" value="${escapeHtml(String(v))}"${minAttr}${maxAttr} />`
  }
  return `<label class="field"><span${helpAttr(spec.description || '')}>${escapeHtml(key)}</span>${input}</label>`
}
function datasetFormHtml(ds) {
  const isNew = !ds || ds.id === 'default'
  const settings = (ds && ds.settings) || defaultDataset().settings
  const fields = datasetLeverEntries()
    .map(([k, spec]) => datasetFieldHtml(k, spec, settings[k]))
    .join('')
  return `<input type="hidden" name="id" value="${escapeHtml(isNew ? randomHexId() : ds.id)}" />
    <label class="field"><span>Name</span>
      <input type="text" name="name" value="${escapeHtml(isNew ? '' : ds.name)}" placeholder="e.g. 1h+1d · 2024" /></label>
    <div class="lever-grid">${fields}</div>
    <div class="form-actions">
      <button type="submit">Save dataset</button>
      <button type="button" id="dataset-cancel" class="ghost-btn">Cancel</button>
    </div>`
}
function toggleDatasetForm(show, ds) {
  const form = byId('dataset-form')
  if (!form) return
  setStatusLine('datasets-status', '')
  if (show) {
    form.innerHTML = datasetFormHtml(ds)
    form.hidden = false
  } else {
    form.innerHTML = ''
    form.hidden = true
  }
}
async function onSaveDataset(form) {
  const id = form.elements.id.value
  const name = String(form.elements.name.value || '').trim()
  if (!name) {
    setStatusLine('datasets-status', 'Give the dataset a name.', true)
    return
  }
  const settings = {}
  for (const [k, spec] of datasetLeverEntries()) {
    const el = form.querySelector(`[name="dataset:${k}"]`)
    if (!el || el.value === '') continue
    settings[k] =
      spec.type === 'number'
        ? Number(el.value)
        : spec.type === 'boolean'
          ? el.value === 'true'
          : el.value
  }
  if (datasetDuplicateOf(name, settings, id)) {
    setStatusLine(
      'datasets-status',
      'A dataset with the same name or settings already exists.',
      true,
    )
    return
  }
  // Editing preserves the default flag; the FIRST dataset a project gets becomes its default.
  const existing = datasetsCache.find((d) => d.id === id)
  const isDefault = existing ? !!existing.default : datasetsCache.length === 0
  try {
    await putDataset({ id, name, settings, default: isDefault })
  } catch {
    setStatusLine('datasets-status', 'Could not save — please try again.', true)
    return
  }
  toggleDatasetForm(false)
  await renderDatasets()
  renderLaunchForm()
}
async function onSetDefaultDataset(id) {
  await setDefaultDataset(id)
  await renderDatasets()
  renderLaunchForm()
}
async function onDeleteDataset(id) {
  const wasDefault = !!(datasetsCache.find((d) => d.id === id) || {}).default
  try {
    await deleteDatasetRecord(id)
  } catch {
    setStatusLine('datasets-status', 'Could not delete — please try again.', true)
    return
  }
  datasetsCache = datasetsCache.filter((d) => d.id !== id)
  // Removing the default promotes the next available dataset (the launch picker always has a default).
  if (wasDefault && datasetsCache.length) await setDefaultDataset(datasetsCache[0].id)
  await renderDatasets()
  renderLaunchForm()
}
function setupDatasets() {
  const addToggle = byId('dataset-add-toggle')
  if (addToggle) addToggle.addEventListener('click', () => toggleDatasetForm(true))
  const form = byId('dataset-form')
  if (form) {
    form.addEventListener('submit', (event) => {
      event.preventDefault()
      onSaveDataset(form)
    })
    form.addEventListener('click', (event) => {
      if (event.target.closest('#dataset-cancel')) toggleDatasetForm(false)
    })
  }
  const body = byId('datasets-body')
  if (body) {
    body.addEventListener('click', (event) => {
      const edit = event.target.closest('button[data-ds-edit]')
      if (edit) {
        const ds = datasetsCache.find((x) => x.id === edit.dataset.dsEdit)
        if (ds) toggleDatasetForm(true, ds)
        return
      }
      const clone = event.target.closest('button[data-ds-clone]')
      if (clone) {
        toggleDatasetForm(true)
        return
      }
      const setDef = event.target.closest('button[data-ds-default]')
      if (setDef) {
        onSetDefaultDataset(setDef.dataset.dsDefault)
        return
      }
      const del = event.target.closest('button[data-ds-delete]')
      if (del) onDeleteDataset(del.dataset.dsDelete)
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
// The runs that are evidence for a hypothesis (spec-consistent), and the measured read / effective
// verdict derived from them — thin wrappers over the pure `window.Hypothesis` module.
function hypothesisMatchedRuns(h) {
  return window.Hypothesis.hypothesisMatchingRuns(h && h.spec, runsCache)
}
function hypothesisMeasured(h) {
  return window.Hypothesis.measuredFromRuns(hypothesisMatchedRuns(h), objectiveDirection())
}
function effectiveHypothesisVerdict(h) {
  return window.Hypothesis.effectiveVerdict(h, runsCache, objectiveDirection())
}
function hypothesisActionsHtml(h) {
  const id = escapeHtml(h.id)
  const c = h.campaign
  const busy = c && (c.status === 'running' || c.status === 'queued')
  const matched = hypothesisMatchedRuns(h).length
  const viewRuns = matched
    ? `<button type="button" data-action="view-runs" data-id="${id}" class="ghost-btn"${helpAttr('Show the runs matching this hypothesis in the Runs tab.')}>View ${matched} run${matched === 1 ? '' : 's'}</button>`
    : ''
  return `<button type="button" data-action="run" data-id="${id}"${busy ? ' disabled' : ''}${helpAttr('Launch this hypothesis’s spec to gather more evidence — the verdict updates as runs land.')}>Launch more runs</button>${viewRuns}
    <button type="button" class="icon-btn" data-action="override" data-id="${id}" title="Override verdict" aria-label="Override verdict">✎</button>
    <button type="button" class="icon-btn" data-action="${h.dismissed ? 'restore' : 'dismiss'}" data-id="${id}" title="${h.dismissed ? 'Restore' : 'Dismiss'}" aria-label="${h.dismissed ? 'Restore' : 'Dismiss'}">${h.dismissed ? '↺' : iconCrossSvg()}</button>
    <button type="button" class="icon-btn icon-btn-danger" data-action="delete" data-id="${id}" title="Delete" aria-label="Delete">${iconDeleteSvg()}</button>`
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
// Claimed (if any) vs the measured read from the matching runs — mirrors the old model card table.
function hypothesisClaimedVsMeasuredHtml(h) {
  const claimed = h.claimedMetrics && Object.keys(h.claimedMetrics).length ? h.claimedMetrics : null
  const m = hypothesisMeasured(h)
  if (!claimed && !m) return ''
  const claimedBits = claimed
    ? Object.entries(claimed)
        .map(([k, v]) => `${escapeHtml(k)} ${escapeHtml(String(v))}`)
        .join(' · ')
    : '—'
  const measuredBits = m
    ? `${escapeHtml(objectiveName())} ${escapeHtml(formatObjective(m.objective))} · ${m.runs} run${m.runs === 1 ? '' : 's'}${m.beatsHold === null ? '' : m.beatsHold ? ' · beats hold' : ' · trails hold'}`
    : 'no matching runs yet'
  return `<table class="kv-table paper-cvm"><tbody>
    <tr><th>claimed</th><td>${claimedBits}</td></tr>
    <tr><th>measured</th><td>${measuredBits}</td></tr>
  </tbody></table>`
}
// The history of auto-verdict flips — each names the runs new since the prior snapshot and the read.
function transitionsHtml(transitions) {
  const list = Array.isArray(transitions) ? transitions : []
  if (!list.length) return ''
  const recent = list.slice(-3).reverse()
  const rows = recent
    .map((t) => {
      const n = Array.isArray(t.byRunKeys) ? t.byRunKeys.length : 0
      const obj =
        t.measured && Number.isFinite(t.measured.objective)
          ? ` · ${escapeHtml(formatObjective(t.measured.objective))}`
          : ''
      return `<li>${escapeHtml(formatWhen(t.at))}: <strong>${escapeHtml(t.from)}</strong> → <strong>${escapeHtml(t.to)}</strong> · ${n} run${n === 1 ? '' : 's'}${obj}</li>`
    })
    .join('')
  const more = list.length > 3 ? `<li class="card-sub">+${list.length - 3} earlier</li>` : ''
  return `<details class="hypothesis-transitions"><summary>${list.length} verdict change${list.length === 1 ? '' : 's'}</summary><ul>${rows}${more}</ul></details>`
}
// Chips for the Papers that link this hypothesis (click → Papers tab).
function linkedPaperChipsHtml(h) {
  const ids = Array.isArray(h.paperIds) ? h.paperIds : []
  if (!ids.length) return ''
  const chips = ids
    .map((pid) => {
      const p = papersCache.find((x) => x.id === pid)
      return `<span class="paper-chip" data-action="goto-paper" data-id="${escapeHtml(pid)}">${escapeHtml((p && p.title) || 'paper')}</span>`
    })
    .join('')
  return `<div class="paper-chips">${chips}</div>`
}
// The inline verdict-override form (one open at a time, tracked by hypothesisOverrideId).
function hypothesisVerdictFormHtml(h) {
  const id = escapeHtml(h.id)
  const cur = effectiveHypothesisVerdict(h)
  const opts = HYPOTHESIS_VERDICTS.map(
    (v) =>
      `<option value="${v}"${cur === v ? ' selected' : ''}>${escapeHtml(HYPOTHESIS_VERDICT_LABEL[v])}</option>`,
  ).join('')
  return `<div class="hypothesis-override">
    <div class="lever-grid">
      <label class="field"><span>Override verdict</span><select id="hyp-override-verdict">${opts}</select></label>
      <label class="field"><span>Note</span><input type="text" id="hyp-override-note" value="${escapeHtml(h.verdictNote || '')}" placeholder="why?" /></label>
    </div>
    <div class="form-actions">
      <button type="button" data-action="save-override" data-id="${id}">Save override</button>
      ${h.verdictSource === 'manual' ? `<button type="button" class="ghost-btn" data-action="clear-override" data-id="${id}">Clear override (auto)</button>` : ''}
      <button type="button" class="ghost-btn" data-action="cancel-override" data-id="${id}">Cancel</button>
    </div>
  </div>`
}
function hypothesisCardHtml(h, liveIds) {
  const verdict = effectiveHypothesisVerdict(h)
  const badge = `<span class="run-badge ${HYPOTHESIS_VERDICT_BADGE[verdict]}">${escapeHtml(HYPOTHESIS_VERDICT_LABEL[verdict])}</span>`
  const source =
    h.source === 'llm'
      ? 'LLM'
      : h.source === 'paper'
        ? 'paper'
        : h.source === 'migrated-model'
          ? 'architecture'
          : 'human'
  const by = h.proposedBy ? ` · ${escapeHtml(h.proposedBy)}` : ''
  const overrideNote =
    h.verdictSource === 'manual'
      ? `<p class="paper-suggest"${helpAttr('Manually set — auto-refresh won’t change it. Clear the override to auto-derive again.')}>manual override — auto suggests: ${escapeHtml(HYPOTHESIS_VERDICT_LABEL[window.Hypothesis.autoVerdictFor(hypothesisMeasured(h))])}</p>`
      : ''
  const overrideForm = hypothesisOverrideId === h.id ? hypothesisVerdictFormHtml(h) : ''
  return `<article class="hypothesis-card${h.dismissed ? ' is-dismissed' : ''}" data-id="${escapeHtml(h.id)}">
    <div class="card-head card-head-row">
      <div><h4>${escapeHtml(h.title || h.id)} ${badge}</h4></div>
      <div class="head-actions">${hypothesisActionsHtml(h)}</div>
    </div>
    ${h.rationale ? `<p class="hypothesis-rationale">${escapeHtml(h.rationale)}</p>` : ''}
    ${specSummaryHtml(h.spec)}
    ${hypothesisClaimedVsMeasuredHtml(h)}
    ${overrideNote}
    ${transitionsHtml(h.transitions)}
    ${linkedPaperChipsHtml(h)}
    ${hypothesisCampaignHtml(h, liveIds)}
    ${h.verdictNote ? `<p class="card-sub paper-note">${escapeHtml(h.verdictNote)}</p>` : ''}
    ${overrideForm}
    <p class="card-sub">${escapeHtml(source)}${by} · created ${escapeHtml(formatWhen(h.createdAt))} · updated ${escapeHtml(formatWhen(h.updatedAt))}</p>
  </article>`
}
function hypothesisGroupHtml(verdict, items, liveIds) {
  const label = HYPOTHESIS_VERDICT_LABEL[verdict] || verdict
  const cards = items.length
    ? `<div class="hypothesis-cards">${items.map((h) => hypothesisCardHtml(h, liveIds)).join('')}</div>`
    : '<p class="card-sub">None.</p>'
  return `<section class="hypothesis-group">
    <h3>${escapeHtml(label[0].toUpperCase() + label.slice(1))} <span class="group-count">${items.length}</span></h3>
    ${cards}
  </section>`
}
function hypothesisFilterBarHtml(visible) {
  const btns = ['all', ...HYPOTHESIS_VERDICTS]
    .map((v) => {
      const label = v === 'all' ? 'all' : HYPOTHESIS_VERDICT_LABEL[v]
      const count =
        v === 'all'
          ? visible.length
          : visible.filter((h) => effectiveHypothesisVerdict(h) === v).length
      return `<button type="button" class="paper-filter-btn${hypothesisVerdictFilter === v ? ' is-active' : ''}" data-hyp-verdict="${escapeHtml(v)}">${escapeHtml(label)} (${count})</button>`
    })
    .join('')
  return `<div class="paper-filter">${btns}</div>`
}
// Starter hypotheses the manifest ships (plus legacy models[] converted to {fixed: match} specs);
// dedup is by spec hash (async), so a seed already present as a migrated/manual hypothesis isn't re-offered.
function manifestHypothesisSeeds() {
  const out = []
  for (const h of (manifest && manifest.hypotheses) || []) if (h && h.spec) out.push(h)
  for (const m of (manifest && manifest.models) || []) {
    if (m && (m.spec || m.match))
      out.push({
        ...m,
        title: m.name || m.modelName || m.title,
        spec: m.spec || { fixed: m.match || {} },
      })
  }
  return out
}
async function pendingSeedHypotheses() {
  const seeds = manifestHypothesisSeeds()
  if (!seeds.length) return []
  const have = new Set(hypothesesCache.map((h) => h.id))
  const pending = []
  for (const s of seeds) {
    if (!s.spec) continue
    const id = await hashTrainingConfig(normalizeSpec(s.spec))
    if (!have.has(id)) pending.push({ seed: s, id })
  }
  return pending
}
function pendingSeedHypothesesBannerHtml(pending) {
  if (!pending.length) return ''
  return `<div class="paper-seed-banner">${pending.length} starter hypothes${pending.length === 1 ? 'is' : 'es'} from the manifest aren’t in your library yet. <button type="button" class="ghost-btn" data-action="import-hyp-seeds">Import ${pending.length}</button></div>`
}
async function importSeedHypotheses() {
  const pending = await pendingSeedHypotheses()
  if (!pending.length) return
  try {
    const now = nowIso()
    for (const { seed, id } of pending) {
      await putHypothesis({
        id,
        title: seed.title || 'Hypothesis',
        rationale: seed.rationale || '',
        spec: normalizeSpec(seed.spec),
        status: HYPOTHESIS_VERDICTS.includes(seed.status) ? seed.status : 'untested',
        verdictSource: seed.status && seed.status !== 'untested' ? 'manual' : 'auto',
        claimedMetrics: seed.claimedMetrics,
        tags: seed.tags,
        source: seed.source || 'human',
        createdAt: now,
        updatedAt: now,
      })
    }
  } catch {
    setStatusLine(
      'hypotheses-status',
      'Could not import starter hypotheses — please try again.',
      true,
    )
    return
  }
  setStatusLine(
    'hypotheses-status',
    `Imported ${pending.length} hypothes${pending.length === 1 ? 'is' : 'es'}.`,
    false,
  )
  await renderHypotheses()
}
// Re-evaluate every hypothesis against the current runs, persisting only material changes (a new
// matched-run set OR an auto-verdict flip), recording a transition on each flip. Runs on tab open + on
// campaign settle — NOT the bare poll — so persistence is debounced.
async function refreshHypothesisVerdicts(hyps) {
  if (!manifest || !embedded() || !window.Hypothesis) return false
  const direction = objectiveDirection()
  const at = nowIso()
  let wrote = false
  for (const h of hyps || []) {
    const { next, changed } = window.Hypothesis.evaluateHypothesis(h, runsCache, { direction, at })
    if (!changed) continue
    try {
      await putHypothesis(next)
      Object.assign(h, next)
      wrote = true
    } catch {
      // best-effort: a failed verdict write must not break rendering
    }
  }
  return wrote
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
  ;[hypothesesCache, proposalSummary, papersCache, runsCache] = await Promise.all([
    readHypotheses(),
    readProposal(),
    readPapers(),
    readRuns(),
  ])
  await refreshHypothesisVerdicts(hypothesesCache)
  const liveIds = hypothesesCache.some((h) => h.campaign && h.campaign.status === 'running')
    ? await readLiveActivityIds()
    : new Set()
  renderProposeControls()
  const pending = await pendingSeedHypotheses()
  const seedBanner = pendingSeedHypothesesBannerHtml(pending)
  const visible = hypothesesCache.filter((h) => !h.dismissed)
  if (!visible.length) {
    body.innerHTML =
      seedBanner +
      '<div class="empty-hint">No hypotheses yet — propose some, add your own, or import the starter set.</div>'
    return
  }
  const shown =
    hypothesisVerdictFilter === 'all'
      ? visible
      : visible.filter((h) => effectiveHypothesisVerdict(h) === hypothesisVerdictFilter)
  const sorted = [...shown].sort((a, b) =>
    String(b.updatedAt || '').localeCompare(String(a.updatedAt || '')),
  )
  const groups = { untested: [], proven: [], disproved: [] }
  for (const h of sorted) (groups[effectiveHypothesisVerdict(h)] || groups.untested).push(h)
  body.innerHTML =
    seedBanner +
    hypothesisFilterBarHtml(visible) +
    HYPOTHESIS_VERDICTS.map((v) => hypothesisGroupHtml(v, groups[v], liveIds)).join('')
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
// Dismiss / restore a hypothesis (hide a bad proposal without deleting it).
async function setHypothesisDismissed(id, dismissed) {
  const h = hypothesesCache.find((x) => x.id === id)
  if (!h) return
  setStatusLine('hypotheses-status', '')
  try {
    await putHypothesis({ ...h, dismissed, updatedAt: nowIso() })
  } catch {
    setStatusLine('hypotheses-status', 'Could not update the hypothesis — please try again.', true)
    return
  }
  await renderHypotheses()
}
// Save a MANUAL verdict override (sticks through auto-refresh), keyed by the inline form's inputs.
async function saveHypothesisOverride(id) {
  const h = hypothesesCache.find((x) => x.id === id)
  if (!h) return
  const verdict = String((byId('hyp-override-verdict') || {}).value || 'untested')
  const note = String((byId('hyp-override-note') || {}).value || '').trim()
  if (!HYPOTHESIS_VERDICTS.includes(verdict)) return
  try {
    await putHypothesis({
      ...h,
      status: verdict,
      verdictSource: 'manual',
      verdictNote: note || undefined,
      updatedAt: nowIso(),
    })
  } catch {
    setStatusLine('hypotheses-status', 'Could not save the override — please try again.', true)
    return
  }
  hypothesisOverrideId = null
  await renderHypotheses()
}
// Clear a manual override → re-derive the verdict from the runs on the next refresh.
async function clearHypothesisOverride(id) {
  const h = hypothesesCache.find((x) => x.id === id)
  if (!h) return
  try {
    await putHypothesis({ ...h, verdictSource: 'auto', updatedAt: nowIso() })
  } catch {
    setStatusLine('hypotheses-status', 'Could not clear the override — please try again.', true)
    return
  }
  hypothesisOverrideId = null
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
  const run = findRun(key)
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
  const keys = [...runsCompareKeys].filter((k) => findRun(k))
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
    const run = findRun(key)
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
// Switch to the Runs tab filtered down to the runs MATCHING this hypothesis (its evidence).
function viewHypothesisRuns(id) {
  const h = hypothesesCache.find((x) => x.id === id)
  if (!h) return
  const runs = hypothesisMatchedRuns(h)
  if (!runs.length) return
  runsFilterKeys = new Set(runs.map((r) => r.key))
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
  const saveBtn = byId('hypothesis-save-btn')
  if (saveBtn) saveBtn.disabled = true
  try {
    await createOrLinkHypothesis({
      title,
      rationale: String((form.elements.rationale && form.elements.rationale.value) || '').trim(),
      spec,
      source: 'human',
    })
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
// Create a hypothesis (spec-hash identity, so an identical spec from any source dedupes to one record),
// or link an EXISTING one — and, when `paperId` is given, link it to that paper. Returns the id.
async function createOrLinkHypothesis({ title, rationale, spec, source, paperId }) {
  const norm = normalizeSpec(spec)
  const id = await hashTrainingConfig(norm)
  const now = nowIso()
  const existing =
    hypothesesCache.find((x) => x.id === id) || (await readHypotheses()).find((x) => x.id === id)
  const content = existing
    ? {
        ...existing,
        paperIds: paperId
          ? Array.from(new Set([...(existing.paperIds || []), paperId]))
          : existing.paperIds,
        updatedAt: now,
      }
    : {
        id,
        title: title || 'Hypothesis',
        rationale: rationale || '',
        spec: norm,
        status: 'untested',
        verdictSource: 'auto',
        source: source || 'human',
        paperIds: paperId ? [paperId] : [],
        createdAt: now,
        updatedAt: now,
      }
  await putHypothesis(content)
  if (paperId) await linkHypothesisToPaper(paperId, id)
  return id
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
      const seedBtn = event.target.closest('button[data-action="import-hyp-seeds"]')
      if (seedBtn) {
        importSeedHypotheses()
        return
      }
      const filterBtn = event.target.closest('button[data-hyp-verdict]')
      if (filterBtn) {
        hypothesisVerdictFilter = filterBtn.dataset.hypVerdict
        renderHypotheses()
        return
      }
      const chip = event.target.closest('[data-action="goto-paper"]')
      if (chip) {
        showTab('papers')
        return
      }
      const btn = event.target.closest('button[data-action]')
      if (!btn) return
      const { action, id } = btn.dataset
      if (action === 'run') runHypothesisCampaign(id, btn)
      else if (action === 'view-runs') viewHypothesisRuns(id)
      else if (action === 'override') {
        hypothesisOverrideId = hypothesisOverrideId === id ? null : id
        renderHypotheses()
      } else if (action === 'save-override') saveHypothesisOverride(id)
      else if (action === 'clear-override') clearHypothesisOverride(id)
      else if (action === 'cancel-override') {
        hypothesisOverrideId = null
        renderHypotheses()
      } else if (action === 'dismiss') setHypothesisDismissed(id, true)
      else if (action === 'restore') setHypothesisDismissed(id, false)
      else if (action === 'delete') deleteHypothesis(id)
    })
  }
}
// --- Papers / Library tab --------------------------------------------------------
// A registry of approaches/papers to prove out or falsify — claim + assumptions + a verdict, with
// claimed-vs-measured read from linked runs and one-click Replicate into the Launch form. Generic; the
// trading line's first consumers are the published methods it replicates under real 0.1% fees.
const PAPER_VERDICT_BADGE = {
  untested: 'is-queued',
  replicating: 'is-running',
  'holds-up': 'is-done',
  fluff: 'is-failed',
}
const PAPER_VERDICT_LABEL = {
  untested: 'untested',
  replicating: 'replicating',
  'holds-up': 'holds up',
  fluff: 'fluff',
}
async function readPapers() {
  if (!manifest) return []
  const recs = await queryRecords(manifest.recordType + PAPER_RECORD_SUFFIX)
  return recs.map((r) => r.content).filter((c) => c && c.id)
}
async function putPaper(paper) {
  await window.OverseerBridge.putData({
    type: manifest.recordType + PAPER_RECORD_SUFFIX,
    key: paper.id,
    content: { ...paper, updatedAt: nowIso() },
  })
}
async function deletePaperRecord(id) {
  await window.OverseerBridge.deleteData({
    type: manifest.recordType + PAPER_RECORD_SUFFIX,
    key: id,
  })
}
// A paper's verdict ROLLS UP from its linked hypotheses (any proven ⇒ holds-up; all disproved ⇒ fluff).
function paperVerdict(paper) {
  return window.Hypothesis.rollupPaperVerdict(
    paper,
    hypothesesCache,
    runsCache,
    objectiveDirection(),
  )
}
// The linked hypotheses, each as a row (title + its live verdict badge + Unlink), with the add/link picker.
function paperLinkedHypothesesHtml(paper) {
  const ids = Array.isArray(paper.hypothesisIds) ? paper.hypothesisIds : []
  const id = escapeHtml(paper.id)
  const rows = ids
    .map((hid) => {
      const h = hypothesesCache.find((x) => x.id === hid)
      if (!h) return ''
      const v = effectiveHypothesisVerdict(h)
      const badge = `<span class="run-badge ${HYPOTHESIS_VERDICT_BADGE[v]}">${escapeHtml(HYPOTHESIS_VERDICT_LABEL[v])}</span>`
      return `<li><span class="paper-hyp-title" data-action="goto-hyp" data-id="${escapeHtml(hid)}">${escapeHtml(h.title || hid)}</span> ${badge} <button type="button" class="icon-btn" data-action="unlink-hyp" data-paper="${id}" data-id="${escapeHtml(hid)}" title="Unlink" aria-label="Unlink">×</button></li>`
    })
    .join('')
  const list = rows
    ? `<ul class="paper-hyp-list">${rows}</ul>`
    : '<p class="card-sub">No hypotheses linked yet — Extract from the link, add one, or link an existing one.</p>'
  return `<div class="paper-hyps">${list}${paperSubformHtml(paper)}</div>`
}
// The per-card inline sub-form: add a NEW hypothesis, or pick an EXISTING one to link.
function paperSubformHtml(paper) {
  if (!paperSubform || paperSubform.paperId !== paper.id) return ''
  const id = escapeHtml(paper.id)
  if (paperSubform.mode === 'add') {
    return `<div class="paper-subform">
      <label class="field"><span>Title</span><input type="text" id="paper-hyp-title" placeholder="hypothesis title" /></label>
      <label class="field"><span>Rationale</span><input type="text" id="paper-hyp-rationale" placeholder="why?" /></label>
      <label class="field"><span>Spec <em>(JSON: sweep / fixed / seeds)</em></span><textarea id="paper-hyp-spec" rows="2" spellcheck="false">${escapeHtml(HYPOTHESIS_SPEC_PLACEHOLDER)}</textarea></label>
      <div class="form-actions">
        <button type="button" data-action="create-hyp" data-paper="${id}">Create &amp; link</button>
        <button type="button" class="ghost-btn" data-action="close-subform" data-paper="${id}">Cancel</button>
      </div>
    </div>`
  }
  const linked = new Set(Array.isArray(paper.hypothesisIds) ? paper.hypothesisIds : [])
  const opts = hypothesesCache
    .filter((h) => !linked.has(h.id))
    .map((h) => `<option value="${escapeHtml(h.id)}">${escapeHtml(h.title || h.id)}</option>`)
    .join('')
  if (!opts) {
    return `<div class="paper-subform"><p class="card-sub">No other hypotheses to link.</p><div class="form-actions"><button type="button" class="ghost-btn" data-action="close-subform" data-paper="${id}">Close</button></div></div>`
  }
  return `<div class="paper-subform">
    <div class="paper-link-row">
      <select id="paper-link-hyp-select">${opts}</select>
      <button type="button" data-action="do-link-existing" data-paper="${id}">Link</button>
      <button type="button" class="ghost-btn" data-action="close-subform" data-paper="${id}">Cancel</button>
    </div>
  </div>`
}
async function linkHypothesisToPaper(paperId, hid) {
  const fresh = await readPapers()
  const p = fresh.find((x) => x.id === paperId) || papersCache.find((x) => x.id === paperId)
  if (!p) return
  const ids = Array.from(new Set([...(p.hypothesisIds || []), hid]))
  await putPaper({ ...p, hypothesisIds: ids })
}
async function unlinkHypothesisFromPaper(paperId, hid) {
  const p = papersCache.find((x) => x.id === paperId)
  if (!p) return
  const ids = (p.hypothesisIds || []).filter((x) => x !== hid)
  try {
    await putPaper({ ...p, hypothesisIds: ids })
  } catch {
    setStatusLine('papers-status', 'Could not unlink — please try again.', true)
    return
  }
  await renderPapers()
}
// Create a hypothesis from the per-card add sub-form and link it to the paper.
async function createPaperHypothesis(paperId) {
  const title = String((byId('paper-hyp-title') || {}).value || '').trim()
  if (!title) {
    setStatusLine('papers-status', 'Give the hypothesis a title.', true)
    return
  }
  const { spec, error } = validateHypothesisSpec(String((byId('paper-hyp-spec') || {}).value || ''))
  if (error) {
    setStatusLine('papers-status', error, true)
    return
  }
  try {
    await createOrLinkHypothesis({
      title,
      rationale: String((byId('paper-hyp-rationale') || {}).value || '').trim(),
      spec,
      source: 'human',
      paperId,
    })
  } catch {
    setStatusLine('papers-status', 'Could not add the hypothesis — please try again.', true)
    return
  }
  paperSubform = null
  await renderPapers()
}
async function linkExistingHypothesisToPaper(paperId) {
  const hid = String((byId('paper-link-hyp-select') || {}).value || '').trim()
  if (!hid) return
  try {
    await linkHypothesisToPaper(paperId, hid)
  } catch {
    setStatusLine('papers-status', 'Could not link — please try again.', true)
    return
  }
  paperSubform = null
  await renderPapers()
}
function paperAssumptionChips(a) {
  if (!a) return ''
  const chips = []
  if (a.frictionless === true) chips.push('<span class="paper-chip is-warn">frictionless</span>')
  if (a.fees === false) chips.push('<span class="paper-chip is-warn">no fees</span>')
  else if (a.fees === true) chips.push('<span class="paper-chip">fees modelled</span>')
  if (a.netOfCosts === false) chips.push('<span class="paper-chip is-warn">gross returns</span>')
  else if (a.netOfCosts === true) chips.push('<span class="paper-chip">net of costs</span>')
  if (a.multiAsset === true) chips.push('<span class="paper-chip">multi-asset</span>')
  if (a.retrainCadence)
    chips.push(`<span class="paper-chip">retrain: ${escapeHtml(String(a.retrainCadence))}</span>`)
  return chips.length ? `<div class="paper-chips">${chips.join('')}</div>` : ''
}
function paperCardHtml(paper) {
  const verdict = paperVerdict(paper)
  const badge = `<span class="run-badge ${PAPER_VERDICT_BADGE[verdict]}">${escapeHtml(PAPER_VERDICT_LABEL[verdict])}</span>`
  const title = paper.url
    ? `<a href="${escapeHtml(paper.url)}" target="_blank" rel="noopener">${escapeHtml(paper.title || 'Untitled')}</a>`
    : escapeHtml(paper.title || 'Untitled')
  const meta = [paper.authors, paper.year]
    .filter(Boolean)
    .map((x) => escapeHtml(String(x)))
    .join(' · ')
  const id = escapeHtml(paper.id)
  const linkedCount = Array.isArray(paper.hypothesisIds) ? paper.hypothesisIds.length : 0
  return `<article class="paper-card" data-id="${id}">
    <div class="card-head card-head-row">
      <div>
        <h3>${title} ${badge}</h3>
        ${meta ? `<p class="card-sub">${meta}</p>` : ''}
      </div>
      <div class="head-actions">
        <button type="button" data-action="add-hyp" data-id="${id}" class="ghost-btn"${helpAttr('Add a hypothesis manually and link it to this paper.')}>Add hypothesis</button>
        <button type="button" data-action="link-existing" data-id="${id}" class="ghost-btn"${helpAttr('Link an existing hypothesis to this paper.')}>Link existing</button>
        ${linkedCount ? `<button type="button" data-action="replicate" data-id="${id}" class="ghost-btn"${helpAttr('Launch every linked hypothesis’s spec.')}>Replicate</button>` : ''}
        <button type="button" class="icon-btn" data-action="edit" data-id="${id}" title="Edit" aria-label="Edit">✎</button>
        <button type="button" class="icon-btn icon-btn-danger" data-action="delete" data-id="${id}" title="Delete" aria-label="Delete">${iconDeleteSvg()}</button>
      </div>
    </div>
    ${paper.claim ? `<p class="paper-claim">${escapeHtml(paper.claim)}</p>` : ''}
    ${paperAssumptionChips(paper.assumptions)}
    ${paperLinkedHypothesesHtml(paper)}
    ${paper.verdictNote ? `<p class="card-sub paper-note">${escapeHtml(paper.verdictNote)}</p>` : ''}
  </article>`
}
function paperFilterBarHtml() {
  const opts = ['all', 'untested', 'holds-up', 'fluff']
  const btns = opts
    .map((s) => {
      const label = s === 'all' ? 'all' : PAPER_VERDICT_LABEL[s]
      const count =
        s === 'all' ? papersCache.length : papersCache.filter((p) => paperVerdict(p) === s).length
      return `<button type="button" class="paper-filter-btn${paperVerdictFilter === s ? ' is-active' : ''}" data-verdict="${escapeHtml(s)}">${escapeHtml(label)} (${count})</button>`
    })
    .join('')
  return `<div class="paper-filter">${btns}</div>`
}
// Starter approaches the manifest ships (parallel to presets); imported into the registry once by id.
function manifestPaperSeeds() {
  return manifest && Array.isArray(manifest.papers) ? manifest.papers : []
}
function pendingSeedPapers() {
  const have = new Set(papersCache.map((p) => p.id))
  return manifestPaperSeeds().filter((s) => s && s.id && !have.has(s.id))
}
function pendingSeedPapersHtml() {
  const pending = pendingSeedPapers()
  if (!pending.length) return ''
  return `<div class="paper-seed-banner">${pending.length} curated starter approach${pending.length === 1 ? '' : 'es'} from the manifest aren’t in your library yet. <button type="button" class="ghost-btn" data-action="import-seeds">Import ${pending.length}</button></div>`
}
async function importSeedPapers() {
  const pending = pendingSeedPapers()
  if (!pending.length) return
  try {
    for (const s of pending) {
      await putPaper({
        ...s,
        status: PAPER_STATUSES.includes(s.status) ? s.status : 'untested',
        source: s.source || 'manual',
        createdAt: nowIso(),
      })
    }
  } catch {
    setStatusLine('papers-status', 'Could not import starter papers — please try again.', true)
    return
  }
  setStatusLine(
    'papers-status',
    `Imported ${pending.length} starter approach${pending.length === 1 ? '' : 'es'}.`,
    false,
  )
  await renderPapers()
}
async function renderPapers() {
  const body = byId('papers-body')
  if (!body) return
  if (!embedded()) {
    setHtml(
      body,
      '<div class="empty-hint">Open inside the Overseer to manage the approach library.</div>',
    )
    return
  }
  // Load hypotheses + runs alongside papers so the linked-hypothesis verdicts + paper roll-up are fresh.
  ;[papersCache, hypothesesCache, runsCache] = await Promise.all([
    readPapers(),
    readHypotheses(),
    readRuns(),
  ])
  await refreshHypothesisVerdicts(hypothesesCache)
  const seedBanner = pendingSeedPapersHtml()
  if (!papersCache.length) {
    setHtml(
      body,
      seedBanner +
        '<div class="empty-hint">No approaches yet — add a paper/method, or import the curated starter set.</div>',
    )
    return
  }
  const shown =
    paperVerdictFilter === 'all'
      ? papersCache
      : papersCache.filter((p) => paperVerdict(p) === paperVerdictFilter)
  const sorted = [...shown].sort((a, b) =>
    String(b.updatedAt || '').localeCompare(String(a.updatedAt || '')),
  )
  setHtml(
    body,
    seedBanner +
      paperFilterBarHtml() +
      (sorted.length
        ? sorted.map(paperCardHtml).join('')
        : '<div class="empty-hint">No approaches match this verdict.</div>'),
  )
}
// New entries open as a CHOOSER: just the link + two paths. "Manual Entry" reveals the full form;
// "Automatic Fill" (deferred) would read the link with an LLM. Editing skips straight to the full form.
function paperChooserButtonsHtml() {
  return `<button type="button" id="paper-manual-entry">Manual Entry</button>
    <button type="button" id="paper-auto-fill" class="ghost-btn"${helpAttr('Read the link with an LLM, draft the paper, and extract its testable hypotheses.')}>Extract hypotheses</button>
    <button type="button" id="paper-cancel" class="ghost-btn">Cancel</button>`
}
function paperChooserHtml() {
  return `<label class="field"><span>Paper / source link</span>
    <input type="text" name="url" id="paper-chooser-url" placeholder="https://arxiv.org/abs/… (optional for manual entry)" /></label>
  <p class="card-sub">Enter a link, then extract its hypotheses automatically — or skip the link and enter it manually.</p>
  <div id="paper-chooser-actions" class="form-actions">${paperChooserButtonsHtml()}</div>`
}
function paperFormHtml(paper) {
  return paper ? paperFullFormHtml(paper) : paperChooserHtml()
}
function showPaperManualForm() {
  const form = byId('paper-form')
  if (!form) return
  const url = String((byId('paper-chooser-url') || {}).value || '').trim()
  form.innerHTML = paperFullFormHtml(url ? { url } : undefined)
}
// Extract: an LLM reads the link, drafts the paper, AND extracts its testable hypotheses (linked back).
// The backend 'analyze-paper' activity fetches the page text, summarises it, and writes a DRAFT
// <recordType>-paper + its <recordType>-hypothesis records; on completion we re-render Papers for review.
async function onPaperAutoFill() {
  const input = byId('paper-chooser-url')
  const url = String((input && input.value) || '').trim()
  if (!/^https?:\/\/\S+/i.test(url)) {
    setStatusLine('papers-status', 'Enter a valid link (https://…) to extract.', true)
    return
  }
  if (!embedded()) {
    setStatusLine('papers-status', 'Open inside the Overseer to use Extract.', true)
    return
  }
  const restore = () => {
    const a = byId('paper-chooser-actions')
    if (a) setHtml(a, paperChooserButtonsHtml())
  }
  const actions = byId('paper-chooser-actions')
  if (actions)
    setHtml(actions, `<span class="paper-autofill-busy">${spinnerHtml()} Reading paper…</span>`)
  setStatusLine('papers-status', '')
  const epoch = projectEpoch
  try {
    const result = await startOrEnqueue(
      'analyze-paper',
      trainerActivityParams({ url }),
      'Analyze paper',
    )
    if (result.queued) {
      if (epoch === projectEpoch) {
        restore()
        setStatusLine(
          'papers-status',
          queuedStatusText(result.ahead) + ' — the draft will appear here when it finishes.',
        )
      }
      return
    }
    const act = await observeQuickActivity(result.activityId)
    if (epoch !== projectEpoch) return
    if (act && act.status === 'completed') {
      togglePaperForm(false)
      await renderPapers()
      showToast('Extracted the paper + its hypotheses — review them below (marked “untested”).')
    } else {
      restore()
      setStatusLine('papers-status', quickActivityFailureText(act, 'Extract'), true)
    }
  } catch {
    if (epoch === projectEpoch) {
      restore()
      setStatusLine('papers-status', 'Extract failed — please try again or use Manual Entry.', true)
    }
  }
}
function paperFullFormHtml(paper) {
  const p = paper || {}
  const a = p.assumptions || {}
  const isNew = !p.id
  const cb = (name, on, label) =>
    `<label class="check-row"><input type="checkbox" name="${name}"${on ? ' checked' : ''} /><span>${label}</span></label>`
  return `<input type="hidden" name="id" value="${escapeHtml(isNew ? randomHexId() : p.id)}" />
    <label class="field"><span>Title</span><input type="text" name="title" value="${escapeHtml(p.title || '')}" placeholder="e.g. Deep RL for crypto trading" /></label>
    <div class="lever-grid">
      <label class="field"><span>URL</span><input type="text" name="url" value="${escapeHtml(p.url || '')}" placeholder="https://…" /></label>
      <label class="field"><span>Authors</span><input type="text" name="authors" value="${escapeHtml(p.authors || '')}" /></label>
      <label class="field"><span>Year</span><input type="number" name="year" value="${escapeHtml(p.year ? String(p.year) : '')}" /></label>
    </div>
    <label class="field"><span>Claim</span><textarea name="claim" rows="2" placeholder="what does it claim to achieve?">${escapeHtml(p.claim || '')}</textarea></label>
    <label class="field"><span>Approach</span><textarea name="approach" rows="2" placeholder="how does it work?">${escapeHtml(p.approach || '')}</textarea></label>
    <label class="field"><span>Claimed metrics <em>(JSON, optional)</em></span><input type="text" name="claimedMetrics" value="${escapeHtml(p.claimedMetrics ? JSON.stringify(p.claimedMetrics) : '')}" placeholder='{"return_pct":30,"sharpe":1.5}' /></label>
    <fieldset class="paper-assumptions"><legend>Assumptions (honesty checklist)</legend>
      ${cb('fees', a.fees === true, 'Realistic fees modelled')}
      ${cb('netOfCosts', a.netOfCosts === true, 'Returns net of costs')}
      ${cb('frictionless', a.frictionless === true, 'Assumes frictionless execution')}
      ${cb('multiAsset', a.multiAsset === true, 'Needs a multi-asset universe')}
      <label class="field"><span>Retrain cadence</span><input type="text" name="retrainCadence" value="${escapeHtml(a.retrainCadence || '')}" placeholder="e.g. monthly" /></label>
      <label class="field"><span>Notes</span><input type="text" name="assumptionNotes" value="${escapeHtml(a.notes || '')}" /></label>
    </fieldset>
    <label class="field"><span>Tags <em>(comma-sep)</em></span><input type="text" name="tags" value="${escapeHtml(Array.isArray(p.tags) ? p.tags.join(', ') : '')}" /></label>
    <label class="field"><span>Verdict note</span><textarea name="verdictNote" rows="2" placeholder="overall read on the paper (its verdict rolls up from the linked hypotheses)">${escapeHtml(p.verdictNote || '')}</textarea></label>
    <div class="form-actions"><button type="submit">Save</button><button type="button" id="paper-cancel" class="ghost-btn">Cancel</button></div>`
}
function togglePaperForm(show, paper) {
  const form = byId('paper-form')
  if (!form) return
  setStatusLine('papers-status', '')
  if (show) {
    form.innerHTML = paperFormHtml(paper)
    form.hidden = false
  } else {
    form.innerHTML = ''
    form.hidden = true
  }
}
// Parse an optional JSON-object form field: '' → undefined (cleared); invalid → null (signals error).
function parseJsonObjectField(form, name) {
  const raw = String((form.elements[name] && form.elements[name].value) || '').trim()
  if (!raw) return undefined
  try {
    const v = JSON.parse(raw)
    return v && typeof v === 'object' && !Array.isArray(v) ? v : null
  } catch {
    return null
  }
}
async function onSavePaper(form) {
  // Submitting from the CHOOSER (Enter in the link field, before any fields exist) advances to the
  // full manual form rather than crashing on the missing title field.
  if (!form.elements.title) {
    showPaperManualForm()
    return
  }
  const title = String(form.elements.title.value || '').trim()
  if (!title) {
    setStatusLine('papers-status', 'Give the approach a title.', true)
    return
  }
  const claimedMetrics = parseJsonObjectField(form, 'claimedMetrics')
  if (claimedMetrics === null) {
    setStatusLine('papers-status', 'Claimed metrics must be a valid JSON object (or blank).', true)
    return
  }
  const id = form.elements.id.value
  const existing = papersCache.find((x) => x.id === id) || {}
  const yearVal = Number(form.elements.year.value)
  const tags = String(form.elements.tags.value || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
  const assumptions = {
    fees: form.elements.fees.checked,
    netOfCosts: form.elements.netOfCosts.checked,
    frictionless: form.elements.frictionless.checked,
    multiAsset: form.elements.multiAsset.checked,
    retrainCadence: String(form.elements.retrainCadence.value || '').trim() || undefined,
    notes: String(form.elements.assumptionNotes.value || '').trim() || undefined,
  }
  const paper = {
    ...existing,
    id,
    title,
    url: String(form.elements.url.value || '').trim() || undefined,
    authors: String(form.elements.authors.value || '').trim() || undefined,
    year: Number.isFinite(yearVal) && yearVal > 0 ? yearVal : undefined,
    claim: String(form.elements.claim.value || '').trim(),
    approach: String(form.elements.approach.value || '').trim() || undefined,
    claimedMetrics,
    assumptions,
    hypothesisIds: Array.isArray(existing.hypothesisIds) ? existing.hypothesisIds : [],
    status: existing.status || 'untested',
    verdictNote: String(form.elements.verdictNote.value || '').trim() || undefined,
    tags: tags.length ? tags : undefined,
    source: existing.source || 'manual',
    createdAt: existing.createdAt || nowIso(),
  }
  try {
    await putPaper(paper)
  } catch {
    setStatusLine('papers-status', 'Could not save — please try again.', true)
    return
  }
  togglePaperForm(false)
  await renderPapers()
}
async function onDeletePaper(id) {
  try {
    await deletePaperRecord(id)
  } catch {
    setStatusLine('papers-status', 'Could not delete — please try again.', true)
    return
  }
  papersCache = papersCache.filter((p) => p.id !== id)
  await renderPapers()
}
// Replicate = launch every linked hypothesis's spec as a campaign (each gathers its own evidence).
async function replicatePaper(id) {
  const p = papersCache.find((x) => x.id === id)
  const ids = (p && Array.isArray(p.hypothesisIds) ? p.hypothesisIds : []).filter((hid) =>
    hypothesesCache.some((h) => h.id === hid),
  )
  if (!ids.length) {
    setStatusLine('papers-status', 'No linked hypotheses to launch — add or link one first.', true)
    return
  }
  for (const hid of ids) await runHypothesisCampaign(hid)
  showToast(`Launching ${ids.length} linked hypothes${ids.length === 1 ? 'is' : 'es'}.`)
}
function setupPapers() {
  const addToggle = byId('paper-add-toggle')
  if (addToggle) {
    addToggle.addEventListener('click', () => {
      const form = byId('paper-form')
      togglePaperForm(!!(form && form.hidden))
    })
  }
  const form = byId('paper-form')
  if (form) {
    form.addEventListener('submit', (event) => {
      event.preventDefault()
      onSavePaper(form)
    })
    form.addEventListener('click', (event) => {
      if (event.target.closest('#paper-cancel')) togglePaperForm(false)
      else if (event.target.closest('#paper-manual-entry')) showPaperManualForm()
      else if (event.target.closest('#paper-auto-fill')) onPaperAutoFill()
    })
  }
  const body = byId('papers-body')
  if (body) {
    body.addEventListener('click', (event) => {
      const filterBtn = event.target.closest('button[data-verdict]')
      if (filterBtn) {
        paperVerdictFilter = filterBtn.dataset.verdict
        renderPapers()
        return
      }
      const chip = event.target.closest('[data-action="goto-hyp"]')
      if (chip) {
        showTab('hypotheses')
        return
      }
      const btn = event.target.closest('button[data-action]')
      if (!btn) return
      const { action, id } = btn.dataset
      const paperId = btn.dataset.paper
      if (action === 'import-seeds') importSeedPapers()
      else if (action === 'replicate') replicatePaper(id)
      else if (action === 'add-hyp') {
        paperSubform =
          paperSubform && paperSubform.paperId === id ? null : { paperId: id, mode: 'add' }
        renderPapers()
      } else if (action === 'link-existing') {
        paperSubform =
          paperSubform && paperSubform.paperId === id ? null : { paperId: id, mode: 'link' }
        renderPapers()
      } else if (action === 'create-hyp') createPaperHypothesis(paperId)
      else if (action === 'do-link-existing') linkExistingHypothesisToPaper(paperId)
      else if (action === 'unlink-hyp') unlinkHypothesisFromPaper(paperId, id)
      else if (action === 'close-subform') {
        paperSubform = null
        renderPapers()
      } else if (action === 'edit')
        togglePaperForm(
          true,
          papersCache.find((p) => p.id === id),
        )
      else if (action === 'delete') onDeletePaper(id)
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
  return leverEntries().filter(([, spec]) => !isEnvLever(spec) && !isDatasetLever(spec))
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
// The user's CHOSEN default environment: the saved record flagged `default`, else the first saved one
// (the fallback after a default is removed). Null when the project HAS env levers but the user hasn't
// defined any environment yet — launch is gated on this, since there's nothing to run against.
function defaultEnvironmentId() {
  if (!environmentsCache.length) return null
  return (environmentsCache.find((e) => e.default) || environmentsCache[0]).id
}
// Make `id` the sole default among saved environments (clearing the flag on the rest); persists each
// record whose flag changed. No-op for an unknown id.
async function setDefaultEnvironment(id) {
  if (!environmentsCache.some((e) => e.id === id)) return
  for (const env of environmentsCache) {
    const shouldBe = env.id === id
    if (!!env.default !== shouldBe) {
      env.default = shouldBe
      await putEnvironment(env)
    }
  }
}
// Another saved environment with the same name (case-insensitive) or the exact same settings already
// exists — used to refuse a duplicate. `exceptId` skips the record being edited.
function environmentDuplicateOf(name, settings, exceptId) {
  const sig = envSettingsSignature(settings)
  const lower = name.trim().toLowerCase()
  return environmentsCache.find(
    (e) =>
      e.id !== exceptId &&
      (e.name.trim().toLowerCase() === lower || envSettingsSignature(e.settings) === sig),
  )
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
// A lever configures the DATASET (which data the model trains/tests on — asset, time window,
// fidelity stack) rather than the MODEL when its spec sets scope:'dataset'. Dataset levers are
// managed as named datasets a model runs AGAINST, so they're split out of the model launch form
// (mirrors environments).
function isDatasetLever(spec) {
  return !!spec && spec.scope === 'dataset'
}
function datasetLeverEntries() {
  return leverEntries().filter(([, spec]) => isDatasetLever(spec))
}
function hasDatasetLevers() {
  return datasetLeverEntries().length > 0
}
// The implicit "Default" dataset from the manifest's dataset-lever defaults.
function defaultDataset() {
  const synonyms = (window.Migrate && window.Migrate.INPUT_SYNONYMS) || []
  const tfSpec = (manifest && manifest.levers && manifest.levers.timeframe) || {}
  const settings = {}
  for (const [key, spec] of datasetLeverEntries()) {
    if (spec.default === undefined) continue
    let value = spec.default
    // The synthetic 'manifest defaults' bundle must be CONCRETE — never carry an input synonym like
    // fidelity_set 'auto'. Resolve fidelity_set via the auto rule on the manifest's default timeframe;
    // any other synonym falls back to the first concrete choice.
    if (synonyms.includes(String(value))) {
      value =
        key === 'fidelity_set' && window.Migrate && window.Migrate.autoFidelity
          ? window.Migrate.autoFidelity(tfSpec.default)
          : (spec.choices || []).map(String).find((c) => !synonyms.includes(c))
    }
    if (value !== undefined) settings[key] = value
  }
  return { id: 'default', name: 'Default', settings }
}
function allDatasets() {
  return [defaultDataset(), ...datasetsCache]
}
// The user's CHOSEN default dataset: the saved record flagged `default`, else the first saved one (the
// fallback after a default is removed). Null when the project HAS dataset levers but the user hasn't
// defined any dataset yet — launch is gated on this, since there's nothing to run against.
function defaultDatasetId() {
  if (!datasetsCache.length) return null
  return (datasetsCache.find((d) => d.default) || datasetsCache[0]).id
}
// Make `id` the sole default among saved datasets (clearing the flag on the rest); persists each record
// whose flag changed. No-op for an unknown id.
async function setDefaultDataset(id) {
  if (!datasetsCache.some((d) => d.id === id)) return
  for (const ds of datasetsCache) {
    const shouldBe = ds.id === id
    if (!!ds.default !== shouldBe) {
      ds.default = shouldBe
      await putDataset(ds)
    }
  }
}
// Another saved dataset with the same name (case-insensitive) or the exact same settings already exists
// — used to refuse a duplicate. `exceptId` skips the record being edited.
function datasetDuplicateOf(name, settings, exceptId) {
  const sig = datasetSettingsSignature(settings)
  const lower = name.trim().toLowerCase()
  return datasetsCache.find(
    (d) =>
      d.id !== exceptId &&
      (d.name.trim().toLowerCase() === lower || datasetSettingsSignature(d.settings) === sig),
  )
}
async function readDatasets() {
  if (!manifest) return []
  const recs = await queryRecords(manifest.recordType + DATASET_RECORD_SUFFIX)
  return recs.map((r) => r.content).filter((c) => c && c.id && c.id !== 'default')
}
async function putDataset(ds) {
  await window.OverseerBridge.putData({
    type: manifest.recordType + DATASET_RECORD_SUFFIX,
    key: ds.id,
    content: { ...ds, updatedAt: nowIso() },
  })
}
async function deleteDatasetRecord(id) {
  await window.OverseerBridge.deleteData({
    type: manifest.recordType + DATASET_RECORD_SUFFIX,
    key: id,
  })
}
// Canonical signature of a run's dataset (its dataset-lever values), for grouping + naming.
function runDatasetSignature(run) {
  const cfg = (run && run.summary && run.summary.config) || {}
  return datasetLeverEntries()
    .map(([key]) => `${key}=${cfg[key] === undefined ? '' : String(cfg[key])}`)
    .join(' · ')
}
function datasetSettingsSignature(settings) {
  return datasetLeverEntries()
    .map(([key]) => `${key}=${settings[key] === undefined ? '' : String(settings[key])}`)
    .join(' · ')
}
// The named dataset a run matches (by dataset-value signature), else 'Custom'.
function runDatasetName(run) {
  const sig = runDatasetSignature(run)
  const match = allDatasets().find((d) => datasetSettingsSignature(d.settings) === sig)
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
// Human-readable "applies only when reward_model = combo_all_noop" note for a conditional lever.
function appliesWhenLabel(cond) {
  return (
    'Applies only when ' +
    Object.entries(cond)
      .map(([k, vals]) => `${k} = ${(Array.isArray(vals) ? vals : [vals]).join(' / ')}`)
      .join('; ')
  )
}
function leverFieldsetHtml(key, spec) {
  const inner =
    spec.type === 'number'
      ? numberLeverHtml(key, spec)
      : spec.type === 'choice'
        ? choiceLeverHtml(key, spec)
        : booleanLeverHtml(key, spec)
  const naNote = spec.appliesWhen
    ? `<p class="lever-na-note">${escapeHtml(appliesWhenLabel(spec.appliesWhen))}</p>`
    : ''
  return `<fieldset id="lever-${escapeHtml(key)}" class="lever${spec.appliesWhen ? ' lever-conditional' : ''}">
    <legend${helpAttr(spec.description || '')}>${escapeHtml(key)} <span class="lever-type">${escapeHtml(spec.type || '')}</span></legend>
    ${inner}
    ${naNote}
  </fieldset>`
}
// A lever's currently-effective values in the form: the swept selection if any, else the fixed value.
function currentLeverValues(form, key) {
  const spec = (manifest && manifest.levers && manifest.levers[key]) || null
  if (!spec) return []
  const sweep = readSweepValues(form, key, spec)
  if (sweep.length) return sweep.map(String)
  const fixed = readFixedValue(form, key, spec)
  return fixed === undefined || fixed === '' ? [] : [String(fixed)]
}
// True unless an `appliesWhen` condition is unmet — every named lever must currently hold one of the
// listed values (matching either the fixed value or any swept value).
function leverApplies(spec, form) {
  if (!spec.appliesWhen) return true
  for (const [k, allowed] of Object.entries(spec.appliesWhen)) {
    const allowedStr = (Array.isArray(allowed) ? allowed : [allowed]).map(String)
    const current = currentLeverValues(form, k)
    if (!current.some((v) => allowedStr.includes(v))) return false
  }
  return true
}
// Grey out + disable conditional levers whose `appliesWhen` isn't satisfied, so a setting only some
// reward models use isn't pinned/swept where it does nothing. Re-run whenever a controlling lever changes.
function refreshLeverApplicability(form) {
  if (!form) return
  for (const [key, spec] of modelLeverEntries()) {
    if (!spec.appliesWhen) continue
    const fieldset = byId('lever-' + key)
    if (!fieldset) continue
    const applies = leverApplies(spec, form)
    fieldset.classList.toggle('is-na', !applies)
    for (const el of fieldset.querySelectorAll('input, select, textarea')) el.disabled = !applies
  }
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
// How many EXPERIMENTS (campaigns + evaluations) and how many TASKS (judge / propose / paper) may
// run at once — two independent lanes so a quick task never waits behind a long campaign. Each
// campaign still has its own "Max parallel runs", so total training processes ≈ the sum.
function savedLaneBudget(key, fallback) {
  try {
    const n = Math.floor(Number(localStorage.getItem(key)))
    return Number.isFinite(n) && n >= 1 ? n : fallback
  } catch {
    return fallback
  }
}
function savedExperimentBudget() {
  return savedLaneBudget(EXPERIMENT_BUDGET_SS, DEFAULT_EXPERIMENT_BUDGET)
}
function savedTaskBudget() {
  return savedLaneBudget(TASK_BUDGET_SS, DEFAULT_TASK_BUDGET)
}
function rememberLaneBudget(key, n) {
  try {
    localStorage.setItem(key, String(Math.max(1, Math.floor(Number(n)) || 1)))
  } catch {
    // best-effort
  }
}
function rememberExperimentBudget(n) {
  rememberLaneBudget(EXPERIMENT_BUDGET_SS, n)
}
function rememberTaskBudget(n) {
  rememberLaneBudget(TASK_BUDGET_SS, n)
}
// Whether the 2nd ('Tasks') column is collapsed — a per-session UI preference.
function tasksColCollapsed() {
  try {
    return sessionStorage.getItem(TASKS_COLLAPSED_SS) === '1'
  } catch {
    return false
  }
}
function setTasksColCollapsed(collapsed) {
  try {
    sessionStorage.setItem(TASKS_COLLAPSED_SS, collapsed ? '1' : '0')
  } catch {
    // best-effort
  }
}
function loadPausedIds() {
  try {
    const v = JSON.parse(localStorage.getItem(PAUSED_IDS_SS) || '[]')
    return new Set(Array.isArray(v) ? v.map(String) : [])
  } catch {
    return new Set()
  }
}
function isPausedByUser(activityId) {
  return loadPausedIds().has(String(activityId))
}
function setPausedByUser(activityId, paused) {
  try {
    const ids = loadPausedIds()
    if (paused) ids.add(String(activityId))
    else ids.delete(String(activityId))
    localStorage.setItem(PAUSED_IDS_SS, JSON.stringify([...ids]))
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
// Read-only display of the bundles an applied experiment preset sweeps (its own environments /
// datasets), so the launch form SHOWS what the preset will run rather than leaving the picker
// looking empty. Purely informational (no inputs) — buildSpecFromForm reads the launchPreset* state.
function presetBundleListHtml(bundles, entries, slotId, noun) {
  const rows = bundles
    .map((b) => {
      const summary = entries.map(([k]) => `${k} ${b[k] === undefined ? '—' : b[k]}`).join(' · ')
      return `<div class="check-row preset-bundle-row"><span class="preset-bundle-tag">preset</span><span class="card-sub">${escapeHtml(summary)}</span></div>`
    })
    .join('')
  return `<div id="${slotId}" class="preset-bundles">
    <p class="preset-bundles-note">This experiment runs ${bundles.length} ${noun}${bundles.length === 1 ? '' : 's'} from the selected preset (read-only). Pick your own below to override.</p>
    ${rows}
  </div>`
}
function environmentPickerHtml() {
  if (!hasEnvLevers()) return ''
  const presetActive = launchPresetEnvironments.length > 0
  const presetBlock = presetActive
    ? presetBundleListHtml(
        launchPresetEnvironments,
        envLeverEntries(),
        'env-preset-info',
        'environment',
      )
    : ''
  const defId = defaultEnvironmentId()
  const rows = environmentsCache
    .map((e) => {
      const summary = envLeverEntries()
        .map(([k]) => `${k} ${e.settings[k] === undefined ? '—' : e.settings[k]}`)
        .join(' · ')
      const checked = !presetActive && e.id === defId ? ' checked' : ''
      const star = e.id === defId ? ' <span class="card-sub" title="default">★</span>' : ''
      return `<label class="check-row env-pick">
        <input type="checkbox" name="env" value="${escapeHtml(e.id)}"${checked} />
        <span><strong>${escapeHtml(e.name)}</strong>${star} <span class="card-sub">${escapeHtml(summary)}</span></span>
      </label>`
    })
    .join('')
  const rowsOrHint = environmentsCache.length
    ? rows
    : '<p class="card-sub">No environments defined yet — add one in the Environments tab before launching.</p>'
  return `<fieldset id="launch-env-picker" class="lever env-picker">
    <legend${helpAttr('Which ENVIRONMENTS (fee / TP-SL regimes) to run this model against. Pick several to test one model across regimes in a single campaign — runs = configs × environments × seeds. An applied experiment preset supplies its own environments (shown read-only); define + tweak your own (and set the default) in the Environments tab.')}>Run against environments</legend>
    ${presetBlock}
    ${rowsOrHint}
  </fieldset>`
}
function selectedEnvironments(form) {
  const ids = new Set([...form.querySelectorAll('input[name="env"]:checked')].map((el) => el.value))
  return allEnvironments().filter((e) => ids.has(e.id))
}
function datasetPickerHtml() {
  if (!hasDatasetLevers()) return ''
  const presetActive = launchPresetDatasets.length > 0
  const presetBlock = presetActive
    ? presetBundleListHtml(launchPresetDatasets, datasetLeverEntries(), 'ds-preset-info', 'dataset')
    : ''
  const defId = defaultDatasetId()
  const rows = datasetsCache
    .map((d) => {
      const summary = datasetLeverEntries()
        .map(([k]) => `${k} ${d.settings[k] === undefined ? '—' : d.settings[k]}`)
        .join(' · ')
      const checked = !presetActive && d.id === defId ? ' checked' : ''
      const star = d.id === defId ? ' <span class="card-sub" title="default">★</span>' : ''
      return `<label class="check-row env-pick">
        <input type="checkbox" name="ds" value="${escapeHtml(d.id)}"${checked} />
        <span><strong>${escapeHtml(d.name)}</strong>${star} <span class="card-sub">${escapeHtml(summary)}</span></span>
      </label>`
    })
    .join('')
  const rowsOrHint = datasetsCache.length
    ? rows
    : '<p class="card-sub">No datasets defined yet — add one in the Datasets tab before launching.</p>'
  return `<fieldset id="launch-ds-picker" class="lever env-picker">
    <legend${helpAttr('Which DATASETS (asset / walk-forward window / fidelity stack) to run this model against. Pick several to test one model across datasets in a single campaign — runs = configs × datasets × environments × seeds. An applied experiment preset supplies its own datasets (shown read-only); define + tweak your own (and set the default) in the Datasets tab.')}>Run against datasets</legend>
    ${presetBlock}
    ${rowsOrHint}
  </fieldset>`
}
// Re-render the dataset + environment picker fieldsets in place from current state (preset bundles +
// default checks). Safe because the form's change listener is delegated, so swapping a child fieldset
// keeps it wired.
function refreshLaunchPickers() {
  const env = byId('launch-env-picker')
  if (env) env.outerHTML = environmentPickerHtml()
  const ds = byId('launch-ds-picker')
  if (ds) ds.outerHTML = datasetPickerHtml()
}
function selectedDatasets(form) {
  const ids = new Set([...form.querySelectorAll('input[name="ds"]:checked')].map((el) => el.value))
  return allDatasets().filter((d) => ids.has(d.id))
}
function renderLaunchForm() {
  const form = byId('launch-form')
  if (!form) return
  const levers = modelLeverEntries()
    .map(([key, spec]) => leverFieldsetHtml(key, spec))
    .join('')
  form.innerHTML = `
    ${presetsSelectHtml()}
    ${datasetPickerHtml()}
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
  refreshLeverApplicability(form)
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
      if (p && (p.fixed || p.sweep || p.datasets || p.environments))
        list.push({
          label: p.label || 'Preset',
          fixed: p.fixed,
          sweep: p.sweep,
          datasets: p.datasets,
          environments: p.environments,
          seeds: p.seeds,
          thesis: p.thesis,
          thesisTarget: p.thesisTarget,
          isExperiment: !!(p.sweep || p.datasets || p.environments),
        })
  }
  list.push(...bestRunPresets())
  return list
}
// Up to 3 one-click presets derived from the best runs SO FAR (by objective), each reproducing that
// run's exact config — model levers as `fixed`, dataset/environment levers as their bundles — so a
// proven setup is re-runnable (and a seed for your own sweep) without hand-copying it. Deduped by
// setup so three seeds of one winner don't crowd out genuinely distinct configs.
function bestRunPresets() {
  if (!manifest || !Array.isArray(runsCache) || !runsCache.length) return []
  const dir = objectiveDirection()
  const scored = runsCache
    .filter(
      (r) =>
        r.summary && r.summary.status !== 'failed' && Number.isFinite(Number(r.summary.objective)),
    )
    .sort((a, b) =>
      dir === 'min'
        ? Number(a.summary.objective) - Number(b.summary.objective)
        : Number(b.summary.objective) - Number(a.summary.objective),
    )
  const seen = new Set()
  const presets = []
  for (const r of scored) {
    const key = setupKeyOfRun(r)
    if (seen.has(key)) continue
    seen.add(key)
    const cfg = { ...(r.summary.config || {}) }
    delete cfg.seed
    const fixed = {}
    const ds = {}
    const env = {}
    for (const [k, v] of Object.entries(cfg)) {
      const spec = (manifest.levers || {})[k]
      if (!spec) continue
      if (spec.scope === 'dataset') ds[k] = v
      else if (spec.scope === 'environment') env[k] = v
      else fixed[k] = v
    }
    const preset = {
      label: `★ Best #${presets.length + 1}: ${objectiveName()} ${formatObjective(r.summary.objective)} · ${setupConfigLabel(cfg)}`,
      fixed,
      fromResults: true,
    }
    if (Object.keys(ds).length) preset.datasets = [ds]
    if (Object.keys(env).length) preset.environments = [env]
    presets.push(preset)
    if (presets.length >= 3) break
  }
  return presets
}
function presetsSelectHtml() {
  const presets = launchPresets()
  if (!presets.length) return ''
  const opt = (p, i) => `<option value="${i}">${escapeHtml(p.label)}</option>`
  const indexed = presets.map((p, i) => [p, i])
  const fromResults = indexed.filter(([p]) => p.fromResults)
  const experiments = indexed.filter(([p]) => p.isExperiment && !p.fromResults)
  const setups = indexed.filter(([p]) => !p.isExperiment && !p.fromResults)
  const groups = []
  if (fromResults.length)
    groups.push(
      `<optgroup label="Top results so far — one-click re-run">${fromResults.map(([p, i]) => opt(p, i)).join('')}</optgroup>`,
    )
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
  // A fresh preset replaces any prior preset-supplied dataset/environment bundles (set below) and
  // any pending paper auto-link (a manual preset isn't replicating a paper).
  launchPresetDatasets = []
  launchPresetEnvironments = []
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
  // A preset may sweep DATASETS (walk-forward windows / fidelity stacks) and/or ENVIRONMENTS (exit/fee
  // regimes) as bundles; those override the pickers for this launch. Re-render the pickers so the
  // swept bundles show read-only (and Default unchecks) — the count alone isn't enough to see them.
  if (Array.isArray(preset.datasets) && preset.datasets.length)
    launchPresetDatasets = preset.datasets
  if (Array.isArray(preset.environments) && preset.environments.length)
    launchPresetEnvironments = preset.environments
  refreshLaunchPickers()
  refreshLeverAnnotations(form)
  refreshLeverApplicability(form)
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
    // A conditional lever that doesn't currently apply is left out of the spec entirely — it falls back
    // to its manifest default at plan time, so a setting only some reward models use never pins/sweeps
    // where it does nothing.
    if (spec.appliesWhen && !leverApplies(spec, form)) continue
    const values = readSweepValues(form, key, spec)
    if (values.length) sweep[key] = values
    else fixed[key] = readFixedValue(form, key, spec)
  }
  const seedCount = readSeedCount(form)
  const out = { sweep, fixed, seeds: Array.from({ length: seedCount }, (_, i) => i) }
  if (hasEnvLevers()) {
    // A preset that sweeps environments wins; otherwise the picker selection.
    const envBundles = launchPresetEnvironments.length
      ? launchPresetEnvironments
      : selectedEnvironments(form).map((e) => e.settings)
    if (envBundles.length) out.environments = envBundles
  }
  if (hasDatasetLevers()) {
    // A preset that sweeps datasets wins; otherwise the picker selection.
    const dsBundles = launchPresetDatasets.length
      ? launchPresetDatasets
      : selectedDatasets(form).map((d) => d.settings)
    if (dsBundles.length) out.datasets = dsBundles
  }
  return out
}
function updateLaunchSummary() {
  const form = byId('launch-form')
  const line = byId('launch-summary')
  if (!form || !line) return
  const spec = buildSpecFromForm(form)
  const configs = Object.values(spec.sweep).reduce((acc, values) => acc * values.length, 1)
  // For a project WITH dataset/env levers the count is the SELECTED bundles (0 when none picked → nothing
  // runs); a project without those levers has no such dimension (treated as 1).
  const datasets = hasDatasetLevers()
    ? Array.isArray(spec.datasets)
      ? spec.datasets.length
      : 0
    : 1
  const envs = hasEnvLevers()
    ? Array.isArray(spec.environments)
      ? spec.environments.length
      : 0
    : 1
  const seeds = spec.seeds.length
  if (hasDatasetLevers() && !datasets) {
    line.textContent = 'Select at least one dataset to run against.'
    return
  }
  if (hasEnvLevers() && !envs) {
    line.textContent = 'Select at least one environment to run against.'
    return
  }
  const total = configs * datasets * envs * seeds
  const dsBit = hasDatasetLevers()
    ? ` × ${datasets} dataset${datasets === 1 ? '' : 's'}${launchPresetDatasets.length ? ' (from preset)' : ''}`
    : ''
  const envBit = hasEnvLevers()
    ? ` × ${envs} environment${envs === 1 ? '' : 's'}${launchPresetEnvironments.length ? ' (from preset)' : ''}`
    : ''
  const target = remoteComputeTarget(savedComputeTarget())
  line.textContent = `${configs} configuration${configs === 1 ? '' : 's'}${dsBit}${envBit} × ${seeds} seed${seeds === 1 ? '' : 's'} = ${total} run${total === 1 ? '' : 's'}${target ? ` on ${target}` : ''}`
}
function campaignLabel(spec) {
  const sweeps = Object.entries(spec.sweep || {}).map(
    ([key, values]) => `${key} × ${values.length}`,
  )
  const datasets = Array.isArray(spec.datasets) ? spec.datasets.length : 0
  const envs = Array.isArray(spec.environments) ? spec.environments.length : 0
  const seeds = Array.isArray(spec.seeds) ? spec.seeds.length : 1
  const dsBit = datasets > 1 ? `${datasets} datasets, ` : ''
  const envBit = envs > 1 ? `${envs} envs, ` : ''
  return `Campaign: ${sweeps.length ? `${sweeps.join(', ')}, ` : ''}${dsBit}${envBit}seeds ${seeds}`
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
  // A project with dataset/environment levers can't launch with NONE selected (nothing to run against).
  // An applied preset supplying its own bundles satisfies this; otherwise it's the picker selection — and
  // with none DEFINED at all (a valid just-started-project state) the message points at the tab.
  if (hasDatasetLevers() && !launchPresetDatasets.length && !selectedDatasets(form).length) {
    if (status) {
      status.textContent = datasetsCache.length
        ? 'Select at least one dataset to run against.'
        : 'Define a dataset in the Datasets tab before launching.'
    }
    return
  }
  if (hasEnvLevers() && !launchPresetEnvironments.length && !selectedEnvironments(form).length) {
    if (status) {
      status.textContent = environmentsCache.length
        ? 'Select at least one environment to run against.'
        : 'Define an environment in the Environments tab before launching.'
    }
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
    const extra = {
      ...(autoEval ? { autoEval: true } : {}),
    }
    const result = await startOrEnqueue(
      'train',
      params,
      campaignLabel(spec),
      Object.keys(extra).length ? extra : undefined,
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
      // The '— choose —' placeholder has value '' (→ Number('')===0); guard it so re-selecting the
      // placeholder no-ops instead of applying presets[0].
      const idx = Number(event.target.value)
      const presets = launchPresets()
      if (event.target.value !== '' && Number.isInteger(idx) && presets[idx]) {
        applyPreset(presets[idx])
      }
      return
    }
    // Manually touching the dataset / environment picker takes over from any preset-supplied bundles;
    // drop the now-stale read-only preset rows (keeping the checkbox the user just toggled).
    if (event.target && event.target.name === 'ds') {
      launchPresetDatasets = []
      const info = byId('ds-preset-info')
      if (info) info.remove()
    }
    if (event.target && event.target.name === 'env') {
      launchPresetEnvironments = []
      const info = byId('env-preset-info')
      if (info) info.remove()
    }
    if (event.target && event.target.name === 'computeTarget') {
      rememberComputeTarget(event.target.value)
    }
    if (event.target && event.target.name === 'autoEval') rememberAutoEval(event.target.checked)
    refreshLeverAnnotations(form)
    refreshLeverApplicability(form)
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
  if (type === 'xai-narrate') return 'xAI narrative'
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
    if (tracked(a.activityId)) continue
    const type = quickActivityType(a) || 'train'
    const experiment = isExperimentActivityType(type)
    const slots = experiment ? experimentSlotCount() : taskSlotCount()
    const budget = experiment ? savedExperimentBudget() : savedTaskBudget()
    if (slots >= budget) continue
    trackExistingActivity(a.activityId, type, activityTypeLabel(type))
  }
  // STALLED: 'running' in the store but NOT live in the backend (it restarted mid-run). Surface
  // each as PAUSED + Resume — resuming re-launches it and the trainer's completed-record skip
  // re-runs only the PENDING runs (a 4-run campaign with 2 done resumes just the other 2).
  for (const a of mine) {
    // Surface as PAUSED: an orphaned 'running' record (backend restarted mid-run), OR one the user
    // explicitly paused (now 'aborted' but flagged) — both resume from the last completed run.
    const orphanedRunning = a.status === 'running' && a.isLive === false
    const userPaused = isPausedByUser(a.activityId) && a.isLive !== true
    if (!orphanedRunning && !userPaused) continue
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
      // A user pause aborts the process but must stay Resume-able — keep it 'paused', don't settle.
      if (isPausedByUser(activityId)) {
        entry.status = 'paused'
        renderActivity()
        return
      }
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
// PAUSE ONE running campaign — kills the process but keeps it resumable: marks it user-paused (so the
// observe loop + reload detection keep it as a Resume-able block), then aborts the live process. The
// trainer's completed-run skip means a later Resume continues from the last finished run.
async function pauseActivityById(activityId) {
  if (!activityId) return
  pausePromptId = null
  setPausedByUser(activityId, true)
  const entry = [...liveActivities.values()].find((a) => a.activityId === activityId)
  if (entry) entry.status = 'paused'
  renderActivity()
  try {
    await window.OverseerBridge.abortActivity(activityId)
  } catch {
    // best-effort — the entry stays paused (user-marked) regardless
  }
}
// Abort ONE activity by id — the observe loop's next poll sees 'aborted' and settles it.
async function abortActivityById(activityId) {
  if (!activityId) return
  setPausedByUser(activityId, false)
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
    setPausedByUser(activityId, false)
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
// A persisted queue for one lane: each waiting entry with its type chip and a remove button.
function queueSectionHtml(items, title) {
  if (!items.length) return ''
  const rows = items
    .map(
      (item) => `<li class="queue-item">
      <span class="badge queue-chip">${escapeHtml(item.activityType)}</span>
      <span class="queue-label">${escapeHtml(item.label || item.activityType)}</span>
      <button type="button" class="queue-remove" data-queue-remove="${escapeHtml(item.id)}" aria-label="Remove from queue">✕</button>
    </li>`,
    )
    .join('')
  return `<div class="queue-section">
    <h3>${escapeHtml(title)} <span class="group-count">${items.length}</span></h3>
    <ul class="queue-list">${rows}</ul>
  </div>`
}
// The experiment lane's knobs (persisted, applied to activities started after): how many
// EXPERIMENTS (campaigns + evaluations) run at once, and how many RUNS each campaign runs at once.
function activitySettingsHtml() {
  return `<div class="activity-settings">
    <label class="field"><span${helpAttr('How many EXPERIMENTS (training campaigns + checkpoint evaluations) run at the same time. These execute the project training code on compute. Each campaign also has its own “Max parallel runs”, so total training processes ≈ the sum — keep both modest for your host.')}>Max concurrent experiments</span>
      <input type="number" id="experiment-budget" min="1" step="1" value="${savedExperimentBudget()}" />
    </label>
    <label class="field"><span${helpAttr('How many runs of a campaign run at once. Set this BEFORE you launch — it applies to the NEXT campaign you start. It does NOT resize a campaign already running (you would relaunch to change that). The real ceiling is host CPU/GPU/RAM; default 1 = sequential.')}>Max parallel runs</span>
      <input type="number" id="activity-concurrency" min="1" step="1" value="${savedConcurrency()}" />
    </label>
  </div>`
}
// The task lane's only knob: how many light TASKS (judge / propose / paper import) run at once.
function taskSettingsHtml() {
  return `<div class="activity-settings">
    <label class="field"><span${helpAttr('How many TASKS (judge, propose experiments, import paper) run at the same time. These are light LLM activities that do not use the training compute, so they run in their own lane and never wait behind a campaign.')}>Max concurrent tasks</span>
      <input type="number" id="task-budget" min="1" step="1" value="${savedTaskBudget()}" />
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
  const id = escapeHtml(entry.activityId || '')
  let actions = ''
  if (running && entry.activityId) {
    if (isTrain) {
      // Campaigns Pause (resumable); a pause kills the live process but keeps completed runs, so it
      // resumes from the last finished run. Confirm inline (window.confirm is blocked in the iframe).
      actions =
        pausePromptId === entry.activityId
          ? `<div class="form-actions activity-confirm"><span class="card-sub">Kill the running process and pause? It will resume from the last completed run.</span><button type="button" class="danger-btn" data-pause-confirm="${id}">Pause</button><button type="button" class="ghost-btn" data-pause-cancel="${id}">Cancel</button></div>`
          : `<div class="form-actions"><button type="button" data-pause="${id}">Pause</button></div>`
    } else {
      actions = `<div class="form-actions"><button type="button" class="danger-btn" data-abort="${id}">Abort</button></div>`
    }
  } else if (status === 'paused' && entry.activityId) {
    actions = `<div class="form-actions"><button type="button" data-resume="${id}">Resume</button><button type="button" class="ghost-btn" data-abort="${id}">Discard</button></div>`
  }
  const stalledNote =
    status === 'paused' && isTrain
      ? '<p class="card-sub">Paused — Resume re-runs only the unfinished runs; completed ones are kept. (A backend restart pauses a campaign here too.)</p>'
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
// The Activity tab is split into two columns by lane: EXPERIMENTS (campaigns + evaluations) on the
// left and TASKS (judge / propose / paper import) on the right. The tasks column is collapsible and
// shows an empty view when nothing is live or queued there.
function experimentColumnHtml(entries, queue) {
  const blocks = entries.map(activityBlockHtml).join('')
  const queueHtml = queueSectionHtml(queue, 'Queue')
  let inner
  if (blocks) {
    inner = `${blocks}${queueHtml}`
  } else {
    const last = lastSettledCampaign ? bestLineHtml(lastSettledCampaign) : ''
    const lastFailures = lastSettledCampaign ? campaignFailuresHtml(lastSettledCampaign) : ''
    inner = last
      ? `<div class="activity-block"><div class="activity-status-row"><span class="status-pill is-ok">Last campaign</span></div>${last}${lastFailures}</div>${queueHtml}`
      : queueHtml ||
        '<div class="empty-hint">No campaign yet — launch one from the Launch tab.</div>'
  }
  const count = entries.length + queue.length
  return `<section class="activity-col activity-col-experiments">
    <header class="activity-col-head"><span class="activity-col-title">Experiments${count ? ` <span class="group-count">${count}</span>` : ''}</span></header>
    ${activitySettingsHtml()}
    ${inner}
  </section>`
}
function taskColumnHtml(entries, queue, collapsed) {
  const count = entries.length + queue.length
  const blocks = entries.map(activityBlockHtml).join('')
  const queueHtml = queueSectionHtml(queue, 'Queued')
  const bodyContent = count
    ? `${blocks}${queueHtml}`
    : '<div class="empty-hint">No tasks running. Judging, proposing experiments, and importing papers appear here — they run without waiting on training.</div>'
  // The toggle only appears on the IDLE lane (to declutter the empty view); once there is work the
  // header is a plain title so the body — with its Abort/Resume/remove controls — always shows.
  const header = count
    ? `<span class="activity-col-title">Tasks <span class="group-count">${count}</span></span>`
    : `<button type="button" class="activity-col-toggle" data-tasks-toggle aria-expanded="${collapsed ? 'false' : 'true'}"><span class="twisty">${collapsed ? '▸' : '▾'}</span> Tasks</button>`
  // The task-budget control mirrors the experiment column's settings and stays visible in every
  // state (including collapsed) so the lane's concurrency is always adjustable.
  return `<section class="activity-col activity-col-tasks">
    <header class="activity-col-head">${header}</header>
    ${taskSettingsHtml()}
    ${collapsed ? '' : `<div class="activity-col-body">${bodyContent}</div>`}
  </section>`
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
  const entries = [...liveActivities.values()].sort((a, b) => b.startedAt - a.startedAt)
  const experimentEntries = entries.filter((e) => isExperimentActivityType(e.activityType))
  const taskEntries = entries.filter((e) => !isExperimentActivityType(e.activityType))
  const experimentQueue = queueCache.filter((q) => isExperimentActivityType(q.activityType))
  const taskQueue = queueCache.filter((q) => !isExperimentActivityType(q.activityType))
  // The Tasks column only collapses when its lane is IDLE — any live/paused block or queued item
  // keeps it shown so the Abort/Resume/remove controls stay reachable. Collapsing the idle column
  // reclaims its width (via the grid modifier) so Experiments fills the row.
  const tasksCollapsed = tasksColCollapsed() && taskEntries.length + taskQueue.length === 0
  setHtml(
    body,
    `<div class="activity-cols${tasksCollapsed ? ' tasks-collapsed' : ''}">${experimentColumnHtml(experimentEntries, experimentQueue)}${taskColumnHtml(taskEntries, taskQueue, tasksCollapsed)}</div>`,
  )
  // Every in-flight run across all live campaigns, for the shared elapsed-timer ticker.
  const allInFlight = []
  for (const entry of experimentEntries) {
    const p = entry.activityType === 'train' && entry.status === 'running' ? entry.progress : null
    if (p && p.phase === 'train') {
      const inf = Array.isArray(p.inFlight) ? p.inFlight : p.current ? [p.current] : []
      for (const r of inf) allInFlight.push(r)
    }
  }
  syncInFlightTimer(allInFlight)
}
function setupActivity() {
  const body = byId('activity-body')
  if (!body) return
  body.addEventListener('click', (event) => {
    const tasksToggle = event.target.closest('[data-tasks-toggle]')
    if (tasksToggle) {
      setTasksColCollapsed(!tasksColCollapsed())
      renderActivity()
      return
    }
    const pauseBtn = event.target.closest('button[data-pause]')
    if (pauseBtn) {
      pausePromptId = pauseBtn.dataset.pause
      renderActivity()
      return
    }
    const pauseConfirmBtn = event.target.closest('button[data-pause-confirm]')
    if (pauseConfirmBtn) {
      pauseActivityById(pauseConfirmBtn.dataset.pauseConfirm)
      return
    }
    const pauseCancelBtn = event.target.closest('button[data-pause-cancel]')
    if (pauseCancelBtn) {
      pausePromptId = null
      renderActivity()
      return
    }
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
    else if (event.target.id === 'experiment-budget') {
      rememberExperimentBudget(event.target.value)
      pumpQueue()
    } else if (event.target.id === 'task-budget') {
      rememberTaskBudget(event.target.value)
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
  // Runs is the only full-width, own-scroll master-detail tab. Scope to the DASHBOARD's tab-main —
  // the home view has its own `.tab-main` that would otherwise match `querySelector` first.
  const main = document.querySelector('#view-dashboard .tab-main')
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
  if (target === 'datasets') renderDatasets()
  if (target === 'hypotheses') renderHypotheses()
  if (target === 'papers') renderPapers()
  if (target === 'launch') refreshLaunchRunners()
  if (target === 'activity') {
    renderActivity()
    refreshQueue()
  }
  if (target === 'xai') void refreshXai()
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
  setupDatasets()
  setupHypotheses()
  setupPapers()
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
