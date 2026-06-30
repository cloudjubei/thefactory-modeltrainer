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
const LAUNCH_DEVICE_SS = 'trainer.launchDevice'
const CONCURRENCY_SS = 'trainer.concurrency'
// Input-only convenience lever values that are NEVER stored on a run (e.g. fidelity_set 'auto', which
// resolves to a concrete value at run time). The dataset form + runs filter drop these so a named dataset
// always pins a concrete value.
const INPUT_SYNONYMS = ['auto']
// Mirror of trainer/fidelity.py: the "auto" fidelity follows the run step (the `timeframe` lever) — an
// hourly step observes 1h+1d, any other step observes its own bar. Used so the synthetic-default dataset
// seed shows a concrete value instead of 'auto'.
function autoFidelity(timeframe) {
  return String(timeframe) === '1h' ? '1h+1d' : '1d'
}
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
const MODEL_RECORD_SUFFIX = '-model'
// Approach/paper verdict lifecycle (matches TrainingPaperRecord.status). Drives the verdict badge,
// the verdict filter, and the auto-suggested verdict from measured-vs-hold.
const PAPER_STATUSES = ['untested', 'replicating', 'holds-up', 'shaky', 'fluff']
const QUEUE_RECORD_TYPE = 'trainer-queue'
const SEEN_RECORD_TYPE = 'trainer-seen'
const CHART_PALETTE = ['#3b82f6', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#14b8a6']
const JUDGE_HELP_TEXT =
  "Scores every completed run 0–100. Health-flagged runs are auto-rejected without using the LLM. For the rest, the objective is normalised (best=100) and blended 50/50 with an LLM verdict that weighs stability and how promising the configuration is — so a run can't win on prose alone. Results appear in the Judge column."
const PROPOSE_HELP_TEXT =
  "Sends the manifest's levers, the run history and the verdicts to the LLM and asks for new experiment specs likely to beat the best run. Proposals are validated against the levers, deduped by spec, and land below as untested hypotheses that auto-verify against your runs."
const NO_RUNNERS_HINT = 'No runners paired — manage them in the Compute Runners panel.'
const TABS = [
  { id: 'runs', label: 'Runs', icon: iconRunsSvg },
  { id: 'hypotheses', label: 'Hypotheses', icon: iconHypothesisSvg },
  { id: 'papers', label: 'Papers', icon: iconPaperSvg },
  { id: 'models', label: 'Models', icon: iconModelSvg },
  { id: 'versions', label: 'Versions', icon: iconVersionSvg },
  { id: 'datasets', label: 'Datasets', icon: iconDatasetSvg },
  { id: 'environments', label: 'Environments', icon: iconEnvironmentSvg },
  { id: 'launch', label: 'Launch', icon: iconRunSvg },
  { id: 'speed', label: 'Speed', icon: iconSpeedSvg },
  { id: 'xai', label: 'xAI', icon: iconXaiSvg },
  { id: 'activity', label: 'Activity' },
]
// xAI tab state: the selectable analysis criterion + direction, the focused run (Model internals), the
// OFAT lever, and the last recommendation set (so a Run-batch click can launch it by index).
let xaiCriterionKey = 'objective'
let xaiCriterionDir = null
let xaiFocusKey = null
let xaiLever = null
let xaiRecsCache = []
// Canonical config keys of suggested batches already launched this session (queued/running). Keyed by
// the spec's fixed config so a re-render keeps the button locked; the rec itself disappears once its runs
// land (the config becomes observed), so no explicit clear is needed beyond the per-project reset.
let xaiLaunchedSpecs = new Set()
// LLM-proposed experiment suggestions ({recordType}-xai-suggestion records), surfaced in the recommender.
let xaiSuggestionsCache = []
// Busy flag for the xAI tab's "Propose with AI" (the propose-experiments activity), kept independent of
// the Hypotheses-tab proposer.
let proposingExperiments = false
// Which scope the xAI tab analyses: 'all' (the whole-space bundle over EVERY run) or 'current' (one focused
// run's deterministic digest). Both are computed on demand server-side and cached — never over the page.
let xaiScope = 'all'
// The active 2nd-level tab within the xAI view (a vertical icon-rail). Per-scope; reset if invalid.
let xaiTab = 'environments'
// The icon rail collapses to icons-only (like the main nav), persisted for the session.
let xaiRailCollapsed = false
// Environments tab (a Runs-style columnar table): free-text filter + the sorted column + direction.
let xaiEnvSearch = ''
let xaiEnvSortKey = 'best'
let xaiEnvSortDir = 'desc'
// Which "configuration map" is shown in the Maps tab: 'parallel', 'pca', or 'pareto' (trade-off frontier).
let xaiMapKind = 'parallel'
// PCA colour mode: 'rank' (performance) or 'model' (cluster by model_name).
let xaiPcaColor = 'rank'
// Pareto (trade-off) map axes: which two metrics + the better-direction for each. An optional 3rd metric
// (xaiParetoZ, null = off) turns the map 3-D with a true 3-D Pareto frontier; it stays 2-D when unset.
let xaiParetoX = null
let xaiParetoXDir = 'max'
let xaiParetoY = null
let xaiParetoYDir = 'max'
let xaiParetoZ = null
let xaiParetoZDir = 'max'
// Parallel-map axis filter: lever -> the picked value (a config must match every pick to stay highlit).
let xaiPcPick = {}
// The view whose "?" explainer callout is open (null = none), toggled by the per-view "?" button.
let xaiHelpOpen = null
// Whether the shared scope header (env + criterion + direction + Re-run) is expanded.
let xaiScopeHeaderExpanded = true
// Ids of collapsed collapsible-cards (default = expanded); persists across re-renders.
const xaiCollapsed = new Set(['surr-coupling', 'surr-interactions'])
// Cached whole-space analysis bundles ({recordType}-config-space records), keyed by criterion key. The
// heavy surrogate/fANOVA/coupling/PCA work runs server-side over ALL runs; the cards render purely from this
// — no browser fit, no page-limited picture.
let xaiConfigSpaceCache = new Map()
let analyzingConfigSpace = false
// Cached per-run xAI digests ({recordType}-run-xai records), keyed by run key — the server keeps only the 5
// most-recent (LRU), so this mirrors that. The run whose digest is currently being computed, for the spinner.
let xaiRunAnalysisCache = new Map()
let analyzingRunKey = null
// Per-run LLM narrative records, keyed by run key → {narrative, runKey, runCount, criterionKey, narratedBy, narratedAt}.
const xaiNarrativeCache = new Map()
let narrating = false
// The two levers crossed in the whole-space interaction grid (default to the top-2 by fANOVA importance).
let xaiInterA = null
let xaiInterB = null
// Hypothesis verdict (auto-derived from matching runs, manually overridable). Drives the badge + filter.
const HYPOTHESIS_VERDICTS = ['untested', 'proven', 'disproved']
const HYPOTHESIS_VERDICT_BADGE = {
  untested: 'is-queued',
  proposed: 'is-running',
  proven: 'is-done',
  disproved: 'is-failed',
  hidden: 'is-queued',
}
const HYPOTHESIS_VERDICT_LABEL = {
  untested: 'untested',
  proposed: 'proposed',
  proven: 'proven',
  disproved: 'disproved',
  hidden: 'hidden',
}
const HYPOTHESIS_SPEC_KEYS = ['sweep', 'fixed', 'seeds', 'environments', 'datasets', 'compare']
const HYPOTHESIS_SPEC_PLACEHOLDER = '{"sweep":{},"fixed":{},"seeds":[0]}'
const DEFAULT_HYPOTHESIS_MIN_RUNS = 3
const HYPOTHESIS_CONFIG_SUFFIX = '-hypothesis-config'

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
// Status filter: '' (any), or one of the keys in RUN_STATUS_FILTERS (completed / degenerate / failed).
let runsStatusFilter = ''
// The lever/version dropdowns collapse under a single header; collapsed, only the
// dropdowns with a non-default selection stay visible (highlighted). Starts collapsed.
let runsDropdownsCollapsed = true
// User-defined numeric filter rules (e.g. "return % > 0"), persisted per project as
// recordType + '-filter-rule' records so they survive reloads. Each: { id, field, op,
// value, active }. Rendered as toggle chips alongside "Hide bad runs".
let customRulesCache = []
let favoritesCache = new Set()
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
// Heavy per-bar / per-step / per-trade content fields a LIST/aggregate fetch never reads. They are
// omitted server-side (projection) so a page of run records can't blow the JSON response past V8's
// max-string limit. Detail / compare / xAI / audit DO need them, so they re-fetch the FULL record by
// key via ensureRunFull. `artifacts.checkpoint` / `artifacts.decisionTraceFile` are small and kept.
const HEAVY_RUN_FIELDS = [
  'series',
  'ledger',
  'regimes',
  'artifacts.runChart',
  'artifacts.decisionTrace',
]
// Keys whose `runExtraCache` entry holds the FULL (un-projected) record — so findRun prefers it over a
// lean page entry, and rememberRun won't downgrade it.
const fullRunKeys = new Set()
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
// Free-text papers filter (name/title/authors/claim/abstract/tags); applied on top of the verdict filter.
let paperSearch = ''
// How the Papers list is sorted: 'status' (rolled-up verdict, then creation date — the default), 'name',
// or 'year'. User-picked via the sort dropdown; the list only re-sorts on a full render, not on each update.
let paperSortKey = 'status'
let hypothesisSortKey = 'updated'
let modelSortKey = 'name'
// Per-paper LLM ops in flight ('analyze-paper-models:<id>' / 'suggest-paper-hypotheses:<id>'), set at click
// (covering the queued wait) and cleared when the activity settles — drives the per-button spinner so the
// user can keep launching on OTHER papers while one is queued/running.
const pendingPaperOps = new Set()
// Which verdict section is expanded (accordion — only one at a time). `undefined` = not yet chosen
// (render default-opens the first non-empty); `null` = the user collapsed them all. Plus the paper-scoped
// add/link sub-form state (`null` | { paperId, mode: 'add' | 'link' }).
let hypothesisOpenSection
let hypothesisOverrideId = null
// Minimum matching runs before a hypothesis can be judged proven/disproved (fewer ⇒ stays untested — a
// single run can't adequately prove a claim). User-settable at the top of the Hypotheses tab; persisted
// per project. Default 3.
let hypothesisMinRuns = DEFAULT_HYPOTHESIS_MIN_RUNS
// Ids of the hypothesis/paper cards the user has expanded (collapsible rows), so expansion survives re-render.
const hypothesisExpanded = new Set()
const paperExpanded = new Set()
let paperSubform = null
// Models catalog records + the active category filter + which cards are expanded (survives re-render).
let modelsCache = []
let modelCategoryFilter = 'all'
const modelExpanded = new Set()
// Proposed model-consolidation groups (near-duplicates the LLM found), resolved to catalog models, while
// the review modal is open. Each: { canonicalId, reason, members:[model], checkedDuplicateIds:Set } — the
// group object is the source of truth for which member is canonical + which are checked to merge in.
let consolidationGroups = []
// The project epoch captured when the consolidation modal opened — accepting aborts if the project changed.
let consolidationModalEpoch = null
// Models whose CPU-vs-MPS device benchmark is currently running (disables the per-model button).
const benchmarkingModels = new Set()
// The persisted all-runs aggregate (a `<recordType>-model-stats` record): per-model + per-flavor run
// counts/best/failing computed over EVERY run, not the current Runs page. `modelStatsStale` flags that
// newer runs exist past the aggregate; `modelStatsRefreshing` spins the Refresh buttons while recomputing.
let modelStatsCache = null
let modelStatsStale = false
let modelStatsRefreshing = false
// A SHARED in-memory snapshot of EVERY run, populated by the all-runs refresh. Both the Models stats and
// the Hypotheses verdicts derive from it (one scan updates both). Empty until refreshed / after a reload —
// the tabs then render from each record's PERSISTED result (model-stats record; hypothesis status/evidence).
let allRunsCache = []
// Missing-model proposals from the last "Find models" run, keyed by paperId — the Papers card turns
// these into one-click "Add to catalog" buttons until the user acts on them.
const paperMissingModels = new Map()
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
  // Skip when the markup is unchanged (anti-flash: re-assigning innerHTML restarts CSS animations like the
  // spinner). Compare against the LAST STRING WE SET, cached on the node — NOT `el.innerHTML`, because
  // reading innerHTML forces a full DOM-subtree SERIALIZATION on every call (the dominant main-thread cost
  // on the 3s live-poll for the big Activities/Runs subtrees), and the browser normalizes parsed HTML so
  // `el.innerHTML === html` almost never matched anyway — paying the serialization for nothing.
  if (!el || el.__lastHtml === html) return
  el.__lastHtml = html
  el.innerHTML = html
}
function shortKey(key) {
  const k = String(key || '')
  return k.length > 10 ? k.slice(0, 10) : k
}
// The favorites picker option label: "model · dataset · profit%" (e.g. "duel-dqn-custom · BTCUSDT · 1h ·
// +12.5%"), so a favorite reads by what it IS, not an opaque run key. Falls back to the short key when the
// run isn't in the loaded page (favorites can outlive a page of results).
function favoriteOptionLabel(key) {
  const run = runsCache.find((r) => r.key === key)
  const s = run && run.summary
  if (!s) return shortKey(key)
  const model = (s.config && s.config.model_name) || shortKey(key)
  const ds = datasetLabel(s)
  const p = Number(s.metrics && s.metrics.total_return_pct)
  const profit = Number.isFinite(p) ? `${p >= 0 ? '+' : ''}${p.toFixed(1)}%` : '—'
  return `${model} · ${ds} · ${profit}`
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
// Show a transient toast. When `onClick` is given the toast is clickable (and lingers a little longer) —
// clicking runs it then dismisses, so an "activity done" toast can take the user straight to the result.
function showToast(message, onClick) {
  let el = byId('app-toast')
  if (!el) {
    el = document.createElement('div')
    el.id = 'app-toast'
    el.className = 'app-toast'
    document.body.appendChild(el)
  }
  el.textContent = message
  el.classList.toggle('is-clickable', !!onClick)
  el.onclick = onClick
    ? () => {
        el.classList.remove('is-visible')
        if (toastTimer) clearTimeout(toastTimer)
        try {
          onClick()
        } catch {
          // a navigation failure must not throw out of the toast handler
        }
      }
    : null
  el.classList.add('is-visible')
  if (toastTimer) clearTimeout(toastTimer)
  toastTimer = setTimeout(() => el.classList.remove('is-visible'), onClick ? 6000 : 3500)
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
// A layered-stack glyph — the "model" icon (a model architecture). Used on the Papers card's Find-models button.
function iconModelSvg(size) {
  const s = size || 14
  return (
    `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">` +
    '<path d="M12 2 2 7l10 5 10-5-10-5Z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/></svg>'
  )
}
// A circle-slash glyph — the "not wanted" icon (hide without deleting). Used on the Papers card.
function iconBanSvg(size) {
  const s = size || 14
  return (
    `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">` +
    '<circle cx="12" cy="12" r="9"/><line x1="5.6" y1="5.6" x2="18.4" y2="18.4"/></svg>'
  )
}
function iconChatSvg(size) {
  const s = size || 18
  return (
    `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">` +
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
// A flask glyph — the "hypothesis" icon (a claim tested by experiment). Used in the tab + composed into
// the paper/hypothesis action buttons so a button reads "<verb> a hypothesis".
function iconHypothesisSvg(size) {
  const s = size || 16
  return (
    `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">` +
    '<path d="M9 3h6"/><path d="M10 3v6l-5 9a1 1 0 0 0 1 1.5h12a1 1 0 0 0 1-1.5l-5-9V3"/><line x1="8" y1="15" x2="16" y2="15"/></svg>'
  )
}
// A lightbulb — "suggest / propose ideas".
function iconLightbulbSvg(size) {
  const s = size || 16
  return (
    `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">` +
    '<path d="M9 18h6"/><path d="M10 21h4"/><path d="M12 3a6 6 0 0 0-4 10.4c.7.6 1 1.4 1 2.1h6c0-.7.3-1.5 1-2.1A6 6 0 0 0 12 3z"/></svg>'
  )
}
// A chain-link — "link an existing one".
function iconLinkSvg(size) {
  const s = size || 16
  return (
    `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">` +
    '<path d="M10 13a5 5 0 0 0 7 0l2-2a5 5 0 0 0-7-7l-1 1"/><path d="M14 11a5 5 0 0 0-7 0l-2 2a5 5 0 0 0 7 7l1-1"/></svg>'
  )
}
// A play triangle — "run / launch".
function iconRunSvg(size) {
  const s = size || 16
  return (
    `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="currentColor" stroke="none" aria-hidden="true">` +
    '<path d="M7 4.5v15a1 1 0 0 0 1.5.86l12-7.5a1 1 0 0 0 0-1.72l-12-7.5A1 1 0 0 0 7 4.5z"/></svg>'
  )
}
// A plus — "add a new one".
function iconPlusSvg(size) {
  const s = size || 16
  return (
    `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true">` +
    '<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>'
  )
}
// A pencil — "edit".
function iconEditSvg(size) {
  const s = size || 16
  return (
    `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">` +
    '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>'
  )
}
// --- Tab icons -------------------------------------------------------------------
// One glyph per section in the tab bar (Activity stays text-only). Stroke=currentColor so each inherits
// the tab's active/idle colour; sized from the tab bar via icon(14).
// A list of result rows — the Runs tab (every run is a row in the table).
function iconRunsSvg(size) {
  const s = size || 16
  return (
    `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">` +
    '<line x1="9" y1="6" x2="21" y2="6"/><line x1="9" y1="12" x2="21" y2="12"/><line x1="9" y1="18" x2="21" y2="18"/>' +
    '<circle cx="4" cy="6" r="1.2" fill="currentColor" stroke="none"/><circle cx="4" cy="12" r="1.2" fill="currentColor" stroke="none"/><circle cx="4" cy="18" r="1.2" fill="currentColor" stroke="none"/></svg>'
  )
}
// A document with text lines — the Papers tab.
function iconPaperSvg(size) {
  const s = size || 16
  return (
    `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">` +
    '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>' +
    '<line x1="9" y1="13" x2="15" y2="13"/><line x1="9" y1="17" x2="14" y2="17"/></svg>'
  )
}
// A price-tag — the Versions tab (each pipeline version is a tagged release).
function iconVersionSvg(size) {
  const s = size || 16
  return (
    `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">` +
    '<path d="M20.6 13.4 13.4 20.6a2 2 0 0 1-2.83 0L3 13V3h10l7.59 7.59a2 2 0 0 1 0 2.82z"/>' +
    '<circle cx="7.5" cy="7.5" r="1.4" fill="currentColor" stroke="none"/></svg>'
  )
}
// A database cylinder — the Datasets tab (named data bundles).
function iconDatasetSvg(size) {
  const s = size || 16
  return (
    `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">` +
    '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.66 3.58 3 8 3s8-1.34 8-3V5"/><path d="M4 11v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6"/></svg>'
  )
}
// A globe — the Environments tab (the market a model trades in).
function iconEnvironmentSvg(size) {
  const s = size || 16
  return (
    `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">` +
    '<circle cx="12" cy="12" r="9"/><line x1="3" y1="12" x2="21" y2="12"/><path d="M12 3a14 14 0 0 1 0 18 14 14 0 0 1 0-18z"/></svg>'
  )
}
// A gauge/speedometer — the Speed tab (device benchmarks).
function iconSpeedSvg(size) {
  const s = size || 16
  return (
    `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">` +
    '<path d="M4 18a8 8 0 0 1 16 0"/><path d="M12 18l5-3.5"/><circle cx="12" cy="18" r="1.4" fill="currentColor" stroke="none"/></svg>'
  )
}
// Sparkles — the xAI tab (model-explanation / insight).
function iconXaiSvg(size) {
  const s = size || 16
  return (
    `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">` +
    '<path d="M12 3l1.7 4.6L18 9l-4.3 1.4L12 15l-1.7-4.6L6 9l4.3-1.4z"/><path d="M18 14l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z"/></svg>'
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
  const content = r.content || {}
  const key =
    r.key || (content.provenance && content.provenance.configHash) || content.configHash || ''
  // Pin conditional levers that don't apply to this run's model (e.g. forward_horizon on a PPO run) to
  // the 'n/a' sentinel, so compare/detail show n/a and xAI importance excludes them — matching the
  // server-side config-space analysis. The stored record is untouched; only this view is normalised.
  const levers = (manifest && manifest.levers) || null
  const summary =
    levers && content.config
      ? { ...content, config: window.Xai.normalizeConditionalConfig(content.config, levers) }
      : content
  return { key, summary }
}
// Run records via the bridge, with the server-side filter/sort/pagination passed through (`extra` =
// { where?, orderBy?, limit?, offset? }). The flat view sends a page; group-by/drill send no limit.
async function queryRunRecords(extra) {
  if (!embedded() || !manifest) return []
  try {
    // Every list/aggregate fetch drops the heavy per-bar fields — they're re-fetched by key on demand.
    const payload = { type: manifest.recordType, omit: HEAVY_RUN_FIELDS }
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
// EVERY run record matching `where` (omit it for all) — independent of any page cap. An unbounded query
// can't be served (a single oversized page would blow the response), so page through with a fixed limit
// until exhausted, deduped by key. `onProgress(n)` fires after each page so a long scan shows progress.
async function queryAllRunRecords(onProgress, where) {
  const PAGE = 500
  const byKey = new Map()
  let offset = 0
  for (let guard = 0; guard < 10000; guard++) {
    const extra = where ? { where, limit: PAGE, offset } : { limit: PAGE, offset }
    const page = await queryRunRecords(extra)
    if (!page.length) break
    for (const r of page) byKey.set(r.key, r)
    if (onProgress) onProgress(byKey.size)
    if (page.length < PAGE) break
    offset += PAGE
  }
  return [...byKey.values()]
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
  // Group-by / drill mode wants every matching run — page through it rather than one unbounded fetch.
  const all = await queryAllRunRecords(undefined, where)
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
// Load the per-project min-runs-to-judge setting (default DEFAULT_HYPOTHESIS_MIN_RUNS) into the module var.
async function loadHypothesisMinRuns() {
  hypothesisMinRuns = DEFAULT_HYPOTHESIS_MIN_RUNS
  if (!manifest) return
  try {
    const recs = await queryRecords(manifest.recordType + HYPOTHESIS_CONFIG_SUFFIX, 'latest')
    const v = Number(recs[0] && recs[0].content && recs[0].content.minRuns)
    if (Number.isFinite(v) && v >= 1) hypothesisMinRuns = Math.floor(v)
  } catch {
    // keep the default
  }
}
async function saveHypothesisMinRuns(n) {
  const v = Math.max(1, Math.floor(Number(n) || DEFAULT_HYPOTHESIS_MIN_RUNS))
  hypothesisMinRuns = v
  await window.OverseerBridge.putData({
    type: manifest.recordType + HYPOTHESIS_CONFIG_SUFFIX,
    key: 'latest',
    content: { minRuns: v, updatedAt: nowIso() },
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
// Favorite runs: a single record holding the set of favorited run keys, so they can be quick-picked in xAI.
async function readFavorites() {
  if (!manifest) return new Set()
  const recs = await queryRecords(`${manifest.recordType}-favorites`, 'favorites')
  const keys = recs && recs[0] && recs[0].content && Array.isArray(recs[0].content.keys) ? recs[0].content.keys : []
  return new Set(keys)
}
async function toggleFavorite(runKey) {
  if (!manifest || !runKey) return
  const next = new Set(favoritesCache)
  if (next.has(runKey)) next.delete(runKey)
  else next.add(runKey)
  favoritesCache = next
  await window.OverseerBridge.putData({
    type: `${manifest.recordType}-favorites`,
    key: 'favorites',
    content: { keys: [...next] },
  })
  if (selectedRunKey === runKey) renderRunDetail(runKey)
  if (activeTabId === 'xai') renderXai()
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
    // The manifest's lever SCOPES drive the Datasets/Environments/Hypotheses classification, so a changed
    // manifest must re-render the ACTIVE tab (not just Launch) — otherwise the tab keeps classifying against
    // the stale snapshot (e.g. a lever that just gained scope:'dataset' stays invisible to the Datasets tab,
    // collapsing datasets that differ only by it). showTab reloads that tab's caches from the fresh manifest.
    if (activeTabId) showTab(activeTabId)
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
  const refresh = byId('dash-refresh')
  if (refresh) {
    refresh.addEventListener('click', (event) => {
      if (event.target.closest('#dash-refresh-latest')) return void refreshLatestRunsDerived()
      if (event.target.closest('#dash-refresh-all')) return void refreshAllRunsDerived()
    })
  }
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
  runExtraCache.clear()
  fullRunKeys.clear()
  fullRunFetches.clear()
  fullRunFetchFailed.clear()
  verdictsCache = new Map()
  evaluationsCache = new Map()
  evaluatingKeys.clear()
  judgementSummary = null
  hypothesesCache = []
  proposalSummary = null
  xaiNarrativeCache.clear()
  xaiLaunchedSpecs.clear()
  xaiSuggestionsCache = []
  proposingExperiments = false
  xaiScope = 'all'
  xaiTab = 'environments'
  xaiEnvSearch = ''
  xaiConfigSpaceCache = new Map()
  analyzingConfigSpace = false
  xaiRunAnalysisCache = new Map()
  analyzingRunKey = null
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
    setHtml(live, '') // via setHtml so the __lastHtml cache stays consistent with the DOM
    live.hidden = true
  }
  const runsBody = byId('runs-body')
  if (runsBody) setHtml(runsBody, '')
  const spark = byId('runs-sparkline')
  if (spark) {
    setHtml(spark, '')
    spark.hidden = true
  }
  // Clear via setHtml so the __lastHtml cache stays in sync with the DOM (these bodies are re-rendered
  // through setHtml, which would otherwise skip a later identical re-render after a direct innerHTML reset).
  const hypothesesBody = byId('hypotheses-body')
  if (hypothesesBody) setHtml(hypothesesBody, '')
  const versionsBody = byId('versions-body')
  if (versionsBody) setHtml(versionsBody, '')
  const environmentsBody = byId('environments-body')
  if (environmentsBody) setHtml(environmentsBody, '')
  environmentsCache = []
  datasetsCache = []
  papersCache = []
  paperSubform = null
  launchPresetDatasets = []
  launchPresetEnvironments = []
  const activityBody = byId('activity-body')
  if (activityBody) setHtml(activityBody, '')
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
  await loadHypothesisMinRuns()
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
  void initGlobalRefresh()
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
// Reorder a pending item within its OWN lane (experiments/tasks pump independently). `queuedAt` is the
// sort-only, never-displayed order key, so a move re-stamps the whole lane to a fresh increasing
// sequence — the cleanest way to persist the new order without timestamp-collision math.
async function moveQueueItem(id, dir) {
  const item = queueCache.find((q) => q.id === id)
  if (!item) return
  const lane = isExperimentActivityType(item.activityType)
  const laneItems = queueCache
    .filter((q) => !q.marker && isExperimentActivityType(q.activityType) === lane)
    .sort(
      (a, b) =>
        String(a.queuedAt || '').localeCompare(String(b.queuedAt || '')) ||
        String(a.id).localeCompare(String(b.id)),
    )
  const i = laneItems.findIndex((q) => q.id === id)
  if (i < 0) return
  const j = dir === 'top' ? 0 : dir === 'up' ? i - 1 : dir === 'down' ? i + 1 : i
  if (j === i || j < 0 || j >= laneItems.length) return
  laneItems.splice(i, 1)
  laneItems.splice(j, 0, item)
  const base = Date.now() - laneItems.length * 1000
  await Promise.all(
    laneItems.map((q, idx) => {
      q.queuedAt = new Date(base + idx * 1000).toISOString()
      return putQueueItem(q)
    }),
  )
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
  } else if (item.activityType === 'propose-experiments') {
    proposingExperiments = busy
    if (activeTabId === 'xai') renderXai()
  } else if (item.activityType === 'config-space-analyze') {
    analyzingConfigSpace = busy
    if (activeTabId === 'xai') renderXai()
  } else if (item.activityType === 'run-xai-analyze') {
    analyzingRunKey = busy ? (item.params && item.params.runKey) || analyzingRunKey : null
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
  } else if (
    item.activityType === 'analyze-paper-models' ||
    item.activityType === 'suggest-paper-hypotheses' ||
    item.activityType === 'weigh-paper-hypotheses'
  ) {
    const pid = item.params && item.params.paperId
    if (!pid) return
    const key = paperOpKey(item.activityType, pid)
    if (busy) pendingPaperOps.add(key)
    else pendingPaperOps.delete(key)
    if (activeTabId === 'papers') updatePaperCard(pid)
  } else if (item.activityType === 'benchmark-model-device') {
    const mid = item.params && item.params.modelId
    if (!mid) return
    if (busy) benchmarkingModels.add(mid)
    else benchmarkingModels.delete(mid)
    if (activeTabId === 'models') renderModels()
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
  } else if (item.activityType === 'propose-experiments') {
    if (act && act.status === 'completed') await loadXaiSuggestions()
    if (activeTabId === 'xai') renderXai()
    setStatusLine('xai-status', quickActivityFailureText(act, 'Proposing'), true)
  } else if (item.activityType === 'config-space-analyze') {
    if (act && act.status === 'completed') await loadXaiConfigSpace()
    if (activeTabId === 'xai') renderXai()
    setStatusLine('xai-status', quickActivityFailureText(act, 'Analysing'), true)
  } else if (item.activityType === 'run-xai-analyze') {
    analyzingRunKey = null
    if (act && act.status === 'completed') await loadXaiRunAnalyses()
    if (activeTabId === 'xai') renderXai()
    setStatusLine('xai-status', quickActivityFailureText(act, 'Analysing'), true)
  } else if (item.activityType === 'xai-narrate') {
    await loadXaiNarrative(item.params && item.params.runKey)
    if (activeTabId === 'xai') renderXai()
    setStatusLine('xai-status', quickActivityFailureText(act, 'Narrating'), true)
  } else if (item.activityType === 'evaluate') {
    await renderRuns()
  } else if (item.activityType === 'analyze-paper-models') {
    const pid = item.params && item.params.paperId
    if (act && act.status === 'completed' && pid) await loadPaperModelsResult(pid)
    else setStatusLine('papers-status', quickActivityFailureText(act, 'Find models'), true)
  } else if (item.activityType === 'suggest-paper-hypotheses') {
    if (act && act.status === 'completed') {
      papersCache = await readPapers()
      hypothesesCache = await readHypotheses()
      await refreshHypothesisVerdicts(hypothesesCache)
      const pid = item.params && item.params.paperId
      showToast('Suggested + linked hypotheses — view', pid ? () => focusPaper(pid) : undefined)
      // Suggesting LINKS hypotheses to the paper but assigns NO weights — so the verdict roll-up would stay
      // unweighted (and no ⚖ chips show). Chain a weigh pass so the paper's hypotheses get importance weights
      // immediately, without the user having to click Re-weigh separately.
      const paper = pid && papersCache.find((p) => p.id === pid)
      if (paper && Array.isArray(paper.hypothesisIds) && paper.hypothesisIds.length) {
        void onReweighPaperHypotheses(pid)
      }
    } else {
      setStatusLine('papers-status', quickActivityFailureText(act, 'Suggest hypotheses'), true)
    }
  } else if (item.activityType === 'weigh-paper-hypotheses') {
    if (act && act.status === 'completed') {
      // The weights are written onto the PAPER (hypothesisWeights) — re-read papers so the roll-up reflects them.
      papersCache = await readPapers()
      const pid = item.params && item.params.paperId
      showToast(
        'Re-weighed hypotheses by importance — view',
        pid ? () => focusPaper(pid) : undefined,
      )
    } else {
      setStatusLine('papers-status', quickActivityFailureText(act, 'Re-weigh hypotheses'), true)
    }
  } else if (item.activityType === 'benchmark-model-device') {
    // Re-read the model record so its newly-measured preferredDevice chip replaces the spinner.
    if (act && act.status === 'completed') modelsCache = await readModels()
    else setStatusLine('models-status', quickActivityFailureText(act, 'Benchmark'), true)
    if (activeTabId === 'models') await renderModels()
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
  // A settled campaign means new runs exist past the last aggregate — reflect that in the topbar control.
  void checkModelStatsStale()
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
    s.status === 'invalid'
      ? 'is-invalid-row'
      : s.status === 'failed'
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
// Whether any runs filter is active — incl. active custom rules (which can be pushed SERVER-side and so
// empty the loaded page). Used to keep the toolbar reachable + show the clear button.
function hasActiveRunsFilters() {
  return !!(
    runsFilterKeys ||
    runsTextFilter ||
    runsVersionFilter ||
    runsStatusFilter ||
    Object.values(runsLeverFilter).some(Boolean) ||
    customRulesCache.some((r) => r.active)
  )
}
function clearRunsFilter() {
  runsFilterKeys = null
  runsFilterLabel = ''
  runsLeverFilter = {}
  runsTextFilter = ''
  runsVersionFilter = ''
  runsStatusFilter = ''
  for (const rule of customRulesCache) {
    if (rule.active) {
      rule.active = false
      saveCustomRule(rule)
    }
  }
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
  'blocked_signal_ratio',
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
  // blocked-signal share: lower is better (a clean, actionable signal stream); green/red at half.
  if (mk === 'blocked_signal_ratio') return v <= 0.5 ? 'delta-pos' : 'delta-neg'
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
  realized_cost_bps:
    'Total trading fees paid over the run, in basis points of the stake (1 bp = 0.01%). Each round-trip pays the per-trade fee twice, so more trading means more fee drag. Lower is better.',
  blocked_signal_ratio:
    "Share of the agent's buy/sell signals that were NO-OPS — a buy emitted while already in position, or a sell while flat — that the environment could not act on. High means the raw per-step signal stream is mostly unusable even when the executed trades are good. Lower is better (0% = every emitted signal was actionable).",
  oos_sharpe:
    'Sharpe ratio of the per-bar equity returns over the out-of-sample test window (mean / stdev of returns). A skill-vs-noise yardstick that feeds the Wave-2 PSR/DSR verdict layer — NOT the optimisation target. Higher is better.',
  oos_n_obs:
    'Number of out-of-sample return observations (equity steps) the OOS statistics are computed over — more observations make the Sharpe/skew/kurtosis more reliable.',
  oos_ret_skew:
    'Skew of the out-of-sample per-bar returns. Positive = a few large gains; negative = a fat left tail (a few large losses).',
  oos_ret_kurt:
    'Excess kurtosis of the out-of-sample per-bar returns — tail-heaviness. High = more frequent extreme moves than a normal distribution.',
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
  realized_cost_bps: 'fees (bps)',
  blocked_signal_ratio: 'blocked %',
}
// Metrics shown to a fixed 2 decimals in the table (money + ratios read cleaner that way).
const TWO_DP_METRICS = new Set([
  'win_pct',
  'final_net_worth',
  'hold_return_pct',
  'return_vs_hold_pct',
  'realized_cost_bps',
])
// Stored as a 0–1 fraction but read as a percentage in the table (e.g. 0.95 → "95.0%").
const PERCENT_RATIO_METRICS = new Set(['blocked_signal_ratio'])
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
  // Signal-noise breakdown is detail-only; the Runs table shows just blocked_signal_ratio (as "blocked %").
  'blocked_signals',
  'executed_signals',
  'signal_noise_pct',
  // OOS return statistics (Sharpe/skew/kurtosis/n) feed the Wave-2 PSR/DSR verdict layer — detail-only,
  // too granular for the table.
  'oos_sharpe',
  'oos_n_obs',
  'oos_ret_skew',
  'oos_ret_kurt',
  ...RETIRED_METRICS,
])
function metricLabel(mk) {
  return METRIC_LABEL[mk] || mk.replace(/_pct$/, ' %').replace(/_/g, ' ')
}
// Some metrics read cleaner at a fixed 2dp (win %, money, ratios); the rest use the
// objective formatter.
function formatMetricValue(mk, v) {
  if (PERCENT_RATIO_METRICS.has(mk)) return (v * 100).toFixed(1) + '%'
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
  if (s.status === 'invalid') {
    cls = 'is-bad'
    label = s.invalidReason ? `invalid — ${s.invalidReason}` : 'invalid'
  } else if (s.status === 'failed') {
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
      id: 'model',
      label: 'Model',
      num: false,
      help: 'The model architecture this run trained — its model_name lever.',
      get: (r) => escapeHtml(String((r.summary.config || {}).model_name ?? '—')),
      sort: (r) => String((r.summary.config || {}).model_name ?? ''),
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
// blocked_signal_ratio is stored as a 0–1 fraction but typed + shown as a % in filters (matching the
// "blocked %" column), so the user enters "10" for 10% instead of "0.1". The conversion lives ONLY at the
// editor's input/label boundary — the stored rule value stays a fraction so the client predicate and the
// server-pushed `where` keep comparing it against the raw metric value.
function ruleFieldIsPercent(fieldKey) {
  return (
    typeof fieldKey === 'string' &&
    fieldKey.startsWith('metric:') &&
    PERCENT_RATIO_METRICS.has(fieldKey.slice(7))
  )
}
function ruleDisplayValue(fieldKey, storedValue) {
  return ruleFieldIsPercent(fieldKey) ? Number((storedValue * 100).toFixed(6)) : storedValue
}
function ruleStoredValue(fieldKey, displayValue) {
  return ruleFieldIsPercent(fieldKey) ? Number((displayValue / 100).toFixed(8)) : displayValue
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
  return `${field ? field.label : rule.field} ${rule.op} ${ruleDisplayValue(rule.field, Number(rule.value))}`
}
// A custom-rule field maps to a stored content dot-path when it can be pushed to the server; a
// computed field (vs_hold) returns null and stays a client-side filter.
function customRuleServerField(field) {
  if (field === 'objective') return 'objective'
  if (field.startsWith('metric:')) return 'metrics.' + field.slice(7)
  if (field.startsWith('config:')) return 'config.' + field.slice(7)
  return null
}
// The selectable run-status filters: each carries the server `where` predicate AND the client predicate so
// the flat (server-paged) view and the in-memory group-by views filter identically. "degenerate" is the
// health-flagged subset (derived, not a stored status).
const RUN_STATUS_FILTERS = {
  completed: {
    label: 'completed',
    where: { field: 'status', op: '=', value: 'completed' },
    test: (r) => (r.summary && r.summary.status) === 'completed',
  },
  degenerate: {
    label: 'degenerate',
    where: {
      and: [
        { field: 'health.status', op: 'exists' },
        { not: { field: 'health.status', op: '=', value: 'ok' } },
      ],
    },
    test: (r) => runIsDegenerate(r),
  },
  failed: {
    label: 'failed',
    where: { field: 'status', op: '=', value: 'failed' },
    test: (r) => (r.summary && r.summary.status) === 'failed',
  },
  invalid: {
    label: 'invalid',
    where: { field: 'status', op: '=', value: 'invalid' },
    test: (r) => (r.summary && r.summary.status) === 'invalid',
  },
}
// The filters that push DOWN to the server query: lever-equals, pipeline version, status, and the pushable
// numeric rules. Text search, Hide-bad, vs-hold rules, and a group-by drill stay client-side.
function buildRunsServerWhere() {
  const preds = []
  for (const [lever, val] of Object.entries(runsLeverFilter)) {
    if (val) preds.push({ field: 'config.' + lever, op: '=', value: String(val) })
  }
  if (runsVersionFilter) {
    preds.push({ field: 'pipelineVersion', op: '=', value: String(runsVersionFilter) })
  }
  if (RUN_STATUS_FILTERS[runsStatusFilter]) preds.push(RUN_STATUS_FILTERS[runsStatusFilter].where)
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
  else if (runsSortKey === 'model') field = 'config.model_name'
  if (!field) return undefined
  const numeric = field !== 'pipelineVersion' && field !== 'config.model_name'
  return [{ field, direction: runsSortDir, numeric }]
}
// The flat Runs view paginates server-side; the group-by views and a setup/experiment drill need
// the full matching set in memory, so they fetch unpaginated.
// Filters applied ONLY on the client (the text search + any custom rule whose field can't be pushed
// server-side via customRuleServerField) trim the page AFTER the server has counted + sliced it, so the
// server count/pager no longer matches the rows shown (phantom/short pages once ≥1 such filter is active).
// When any is present, fall back to the unpaged client-paginated path (the one the group-by views use),
// which filters the full matching set THEN slices — keeping pages + rows consistent.
function hasClientOnlyRunsFilter() {
  if (runsTextFilter && runsTextFilter.trim()) return true
  return customRulesCache.some((rule) => rule.active && !customRuleServerField(rule.field))
}
function runsServerPaged() {
  return runsViewMode === 'runs' && !runsFilterKeys && !hasClientOnlyRunsFilter()
}
// Resolve a run by key from the current page cache, falling back to records stashed when a run was
// opened/selected (so detail + compare survive paging away from the run).
function findRun(key) {
  // Prefer a cached FULL record (has the heavy fields) over the lean page/off-page copy.
  const extra = runExtraCache.get(key)
  if (extra && fullRunKeys.has(key)) return extra
  return runsCache.find((r) => r.key === key) || extra
}
function rememberRun(key) {
  // Don't downgrade an already-full cache entry to the lean page copy.
  if (fullRunKeys.has(key)) return
  const run = runsCache.find((r) => r.key === key)
  if (run) runExtraCache.set(key, run)
}
// In-flight keyed full-fetches (key → Promise), so concurrent callers share one request.
const fullRunFetches = new Map()
// Keys we've already attempted to fully fetch but that did NOT upgrade (empty/error). Tracked so a
// resolves-but-never-upgrades key can't drive an unbounded warm→fetch→re-render loop; cleared on an
// explicit user re-trigger (rearmRunFull) and on data refresh / project reset.
const fullRunFetchFailed = new Set()
// Fetch a run's FULL content by key (a single keyed query, no `omit`) and cache it so findRun and the
// heavy views resolve it. Falls back to whatever (lean) copy exists if the fetch can't run or fails.
async function ensureRunFull(key) {
  if (!key) return null
  if (fullRunKeys.has(key)) return runExtraCache.get(key) || findRun(key)
  if (!embedded() || !manifest) return findRun(key)
  if (fullRunFetches.has(key)) return fullRunFetches.get(key)
  const p = (async () => {
    try {
      const recs = await window.OverseerBridge.queryData({ type: manifest.recordType, key })
      const rec = (recs || [])[0]
      if (rec) {
        const run = recordToRun(rec)
        runExtraCache.set(key, run)
        fullRunKeys.add(key)
        fullRunFetchFailed.delete(key)
        return run
      }
      fullRunFetchFailed.add(key) // no such record (e.g. deleted) — don't keep retrying on re-render
    } catch {
      fullRunFetchFailed.add(key) // transient failure — rearmed on the next data refresh / re-trigger
    } finally {
      fullRunFetches.delete(key)
    }
    return findRun(key)
  })()
  fullRunFetches.set(key, p)
  return p
}
// Clear the "already attempted, didn't upgrade" mark for a key so an explicit user action re-arms a
// fresh full-fetch (e.g. re-opening a run after a transient error).
function rearmRunFull(key) {
  fullRunFetchFailed.delete(key)
}
// Drop every cache trace of a run (used when it's deleted) so a stale full/lean entry can't resolve via findRun.
function forgetRun(key) {
  runExtraCache.delete(key)
  fullRunKeys.delete(key)
  fullRunFetchFailed.delete(key)
  fullRunFetches.delete(key)
}
// Render-time seam for the synchronous heavy views: fetch any of `keys` that isn't full yet (and hasn't
// already failed to upgrade), then re-render — but ONLY if a key actually became full, so a key that
// resolves-but-never-upgrades can't loop. The caller renders with the lean copy now and upgrades on arrival.
function warmRunsForRender(keys, rerender) {
  const missing = (keys || []).filter(
    (k) => k && !fullRunKeys.has(k) && !fullRunFetchFailed.has(k) && !fullRunFetches.has(k),
  )
  if (!missing.length) return
  Promise.all(missing.map((k) => ensureRunFull(k).then(() => fullRunKeys.has(k)))).then((becameFull) => {
    if (becameFull.some(Boolean)) rerender()
  })
}
function applyRunsFilters(runs) {
  let out = runsFilterKeys ? runs.filter((r) => runsFilterKeys.has(r.key)) : runs
  if (runsHideBad) out = out.filter((r) => !runIsBad(r))
  if (runsVersionFilter) out = out.filter((r) => runVersionOf(r) === runsVersionFilter)
  if (RUN_STATUS_FILTERS[runsStatusFilter])
    out = out.filter(RUN_STATUS_FILTERS[runsStatusFilter].test)
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
  const selectedValue = editing
    ? String(ruleDisplayValue(editing.field, Number(editing.value)))
    : ''
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
    value: ruleStoredValue(fieldEl.value, value),
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
  const synonyms = new Set(INPUT_SYNONYMS)
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
  const versionFilter = `<select class="runs-filter-lever${runsVersionFilter ? ' is-changed' : ''}" id="runs-version-filter"${helpAttr("Show only runs from one pipeline version — cross-version scores aren't comparable.")}>
          <option value="">version: any</option>
          ${versions.map((v) => `<option value="${escapeHtml(v)}"${runsVersionFilter === v ? ' selected' : ''}>v${escapeHtml(v)}</option>`).join('')}
        </select>`
  const statusFilter = `<select class="runs-filter-lever${runsStatusFilter ? ' is-changed' : ''}" id="runs-status-filter"${helpAttr('Show only runs in one state — completed, degenerate (health-flagged), or failed.')}>
          <option value="">status: any</option>
          ${Object.entries(RUN_STATUS_FILTERS)
            .map(
              ([k, s]) =>
                `<option value="${escapeHtml(k)}"${runsStatusFilter === k ? ' selected' : ''}>${escapeHtml(s.label)}</option>`,
            )
            .join('')}
        </select>`
  const changedDropdowns =
    (runsVersionFilter ? 1 : 0) +
    (runsStatusFilter ? 1 : 0) +
    Object.values(runsLeverFilter).filter(Boolean).length
  const dropdownsToggle = `<button type="button" id="runs-dropdowns-toggle" class="runs-dropdowns-toggle" aria-expanded="${runsDropdownsCollapsed ? 'false' : 'true'}">
    <span class="caret">${runsDropdownsCollapsed ? '▸' : '▾'}</span> ${runsDropdownsCollapsed ? 'More filter options' : 'Hide filter options'}${runsDropdownsCollapsed && changedDropdowns ? ` <span class="runs-dropdowns-count">${changedDropdowns}</span>` : ''}
  </button>`
  const dropdownsPanel = `<div id="runs-dropdowns" class="runs-dropdowns${runsDropdownsCollapsed ? ' is-collapsed' : ''}">
    ${dropdownsToggle}
    <div class="runs-dropdowns-body">${statusFilter}${versionFilter}${leverDropdowns}</div>
  </div>`

  const active = hasActiveRunsFilters()
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
    if (!r.summary || r.summary.status === 'failed' || r.summary.status === 'invalid') continue
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
    if (hasActiveRunsFilters()) {
      // A filter (often a server-pushed custom rule) excluded everything on this page — keep the toolbar so
      // the user can clear it, instead of stranding them on "No runs yet" with no way back.
      const total = runsServerPaged() ? runsTotalCount : 0
      setHtml(
        body,
        `${runsToolbarHtml(0, total)}<div class="empty-hint">No runs match the active filter — use \u201cclear\u201d (or untick the filter chip) above to see your runs.</div>`,
      )
    } else {
      setHtml(body, '<div class="empty-hint">No runs yet — launch a campaign.</div>')
    }
    closeRunDetail()
    return
  }
  // Keep the off-page run cache bounded to runs still referenced by an open detail / compare / xAI view.
  // Skip keys with an in-flight full-fetch — it would re-populate the cache right after eviction.
  for (const k of [...runExtraCache.keys()]) {
    if (
      k !== selectedRunKey &&
      k !== xaiFocusKey &&
      !runsCompareKeys.has(k) &&
      !fullRunFetches.has(k)
    ) {
      runExtraCache.delete(k)
      fullRunKeys.delete(k)
      fullRunFetchFailed.delete(k)
    }
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
  favoritesCache = await readFavorites()
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
  renderRunsTable()
  // Close the xAI analyse→run→re-analyse loop: when records change (e.g. a launched batch lands), the
  // open xAI tab recomputes its effects + recommendations off the fresh runs.
  if (activeTabId === 'xai') renderXai()
}
// A hypothesis's canonical identity = the spec hash (matching how the engine hashes a run's config),
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
  const compare = spec && spec.compare
  if (compare && compare.lever && Array.isArray(compare.values) && compare.values.length)
    out.compare = { lever: compare.lever, values: compare.values }
  const environments = (spec && spec.environments) || []
  if (Array.isArray(environments) && environments.length) out.environments = environments
  const datasets = (spec && spec.datasets) || []
  if (Array.isArray(datasets) && datasets.length) out.datasets = datasets
  const seeds = (spec && spec.seeds) || []
  if (Array.isArray(seeds) && seeds.length) out.seeds = seeds
  return out
}
// Compare-mode "Download audit": a JSON of the selected runs' FULL summaries (config, metrics,
// top-level regimes + benchmark, health, decision trace) for an offline / hand-to-agent genuineness
// audit. The viewer is a sandboxed iframe: a Blob+anchor download is a SILENT no-op unless the host
// grants `allow-downloads` (and the click never throws), so we always ALSO copy the JSON to the
// clipboard — the reliable in-sandbox path via allow-same-origin — and confirm on the button.
async function downloadRunsAudit(keys, button) {
  // The audit carries each run's FULL summary verbatim (regimes, decision trace, runChart) — fetch the
  // full records by key before assembling it, since the list cache is lean.
  await Promise.all((keys || []).map(ensureRunFull))
  const runs = (keys || []).map((k) => findRun(k)).filter(Boolean)
  if (!runs.length) return
  const payload = window.RunExport.buildRunsAuditExport(runs, {
    exportedAt: new Date().toISOString(),
    objective: (manifest && manifest.objective) || null,
    project: (manifest && manifest.name) || null,
  })
  const text = JSON.stringify(payload, null, 2)
  const stamp = new Date().toISOString().slice(0, 10)
  const filename = `runs-audit-${runs.map((r) => shortKey(r.key)).join('-')}-${stamp}.json`
  try {
    const blob = new Blob([text], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  } catch {
    // sandbox / API gap — the clipboard copy below still delivers the data
  }
  copyText(text, button)
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
  // The equity overlay + decision diff need the heavy fields; warm the compared runs and re-render.
  warmRunsForRender([...runsCompareKeys], renderCompare)
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
  // Batch re-run affordances are mutually exclusive: ALL-failed gets "Re-run all"; otherwise ALL-outdated
  // (and on the same version) gets "Re-run all with latest version". Never both.
  const allFailed = runs.every((r) => r.summary && r.summary.status === 'failed')
  const allOutdated =
    !allFailed && runs.every(runIsOutdated) && new Set(runs.map(runVersionOf)).size === 1
  setHtml(
    card,
    `<div class="card-head card-head-row">
      <h3>Compare ${runs.length} runs</h3>
      <div class="head-actions">
        ${embedded() && allFailed ? `<button type="button" id="compare-rerun-all" class="ghost-btn" title="Queue all ${runs.length} failed runs again">Re-run all (${runs.length})</button>` : ''}
        ${embedded() && allOutdated ? `<button type="button" id="compare-rerun-latest" class="ghost-btn" title="Re-run all ${runs.length} runs under the current pipeline version (v${escapeHtml(String(currentPipelineVersion()))}) — they ran under an older, incomparable version">↻ Re-run all with latest version (${runs.length})</button>` : ''}
        ${runs.length === 2 && chatAboutRunAvailable() ? `<button type="button" id="compare-discuss" class="icon-btn" title="Discuss these two runs (incl. the decision diff)" aria-label="Discuss these two runs">${iconChatSvg()}</button>` : ''}
        <button type="button" id="compare-download" class="icon-btn" title="Download + copy to clipboard a JSON audit of these ${runs.length} runs (config, metrics, regimes, health, decision trace)" aria-label="Download audit export">⬇</button>
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
// blocked_signal_ratio is the Runs-table "blocked %" column only; the detail shows the breakdown
// (blocked_signals / executed_signals / signal_noise_pct) instead, so the ratio is hidden here.
const DETAIL_HIDDEN_METRICS = new Set(['traded_return', 'blocked_signal_ratio', ...RETIRED_METRICS])
function metricsTableHtml(metrics) {
  const entries = Object.entries(metrics || {}).filter(([k]) => !DETAIL_HIDDEN_METRICS.has(k))
  if (!entries.length) return '<p class="card-sub">No metrics recorded.</p>'
  const rows = entries
    .map(
      ([k, v]) =>
        `<tr><th${helpAttr(METRIC_INFO[k])}>${escapeHtml(k)}</th><td class="num">${escapeHtml(typeof v === 'number' ? formatObjective(v) : String(v))}</td></tr>`,
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
async function expandPriceActionChart(key) {
  const run = await ensureRunFull(key)
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
  // The current-run internals (price/equity/explain/decision-diff) need the heavy fields for the focus
  // run and its compared sibling; warm them and re-render once they arrive.
  if (xaiScope === 'current' && xaiFocusKey) {
    const focusRun = findRun(xaiFocusKey)
    const sibling = focusRun ? xaiBestSibling(focusRun) : null
    const keys = sibling ? [xaiFocusKey, sibling.key] : [xaiFocusKey]
    warmRunsForRender(keys, () => {
      if (xaiScope === 'current' && xaiFocusKey) renderXai()
    })
  }
  const criterion = currentXaiCriterion()
  setHtml(body, xaiShellHtml(criterion))
}
// Small inline icons for the xAI tab rail (kept self-contained so the rail doesn't depend on the nav set).
function xaiSvg(inner) {
  return (s) =>
    `<svg width="${s || 16}" height="${s || 16}" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">${inner}</svg>`
}
const xaiIconEnv = xaiSvg(
  '<path d="M8 1.5 14.5 5 8 8.5 1.5 5 8 1.5Z"/><path d="M1.5 8 8 11.5 14.5 8"/><path d="M1.5 11 8 14.5 14.5 11"/>',
)
const xaiIconSliders = xaiSvg(
  '<path d="M2 4h7"/><path d="M12 4h2"/><circle cx="10.5" cy="4" r="1.5"/><path d="M2 8h2"/><path d="M7 8h7"/><circle cx="5.5" cy="8" r="1.5"/><path d="M2 12h9"/><path d="M14 12h0.5"/><circle cx="12.5" cy="12" r="1.5"/>',
)
const xaiIconSurrogate = xaiSvg(
  '<circle cx="8" cy="3" r="1.6"/><circle cx="4" cy="12" r="1.6"/><circle cx="12" cy="12" r="1.6"/><path d="M8 4.6 4.8 10.6"/><path d="M8 4.6 11.2 10.6"/>',
)
const xaiIconMap = xaiSvg(
  '<circle cx="4" cy="11" r="1.3"/><circle cx="7.5" cy="6.5" r="1.3"/><circle cx="11" cy="9" r="1.3"/><circle cx="12.5" cy="4" r="1.3"/><path d="M2 14h12"/><path d="M2 14V2"/>',
)
const xaiIconRank = xaiSvg(
  '<path d="M3 14V8h3v6"/><path d="M6.5 14V4h3v10"/><path d="M10 14v-4h3v4"/>',
)
// The xAI shell: a full-height left rail (scope + criterion + direction, then the view tabs), and the active
// view filling the rest — each view a FIXED header (title + chat + "?" + its controls) over a body that
// scrolls within itself, not the page.
function xaiShellHtml(criterion) {
  let tabs = []
  let prompt = ''
  if (xaiScope === 'current') {
    if (!xaiFocusKey) {
      prompt = `<div class="card"><p class="card-sub">Focus a run (Runs → “Analyze in xAI”) to analyse one model in depth — its decisions, what drives them, and how it ranks among all runs.</p></div>`
    } else if (!findRun(xaiFocusKey)) {
      prompt = `<div class="card"><div class="card-head card-head-row"><h3>Run ${escapeHtml(shortKey(xaiFocusKey))} not loaded</h3><button type="button" class="ghost-btn" data-xai-clear-focus>✕ Clear run</button></div>
        <p class="card-sub">Open it from the Runs tab and choose “Analyze in xAI”.</p></div>`
    } else tabs = xaiCurrentTabs(criterion)
  } else {
    const bundle = xaiResolveBundle(criterion)
    if (!bundle) prompt = xaiNoBundleHtml(criterion)
    else tabs = xaiAllTabs(bundle, criterion)
  }
  if (tabs.length && !tabs.some((t) => t.id === xaiTab)) xaiTab = tabs[0].id
  const active = tabs.find((t) => t.id === xaiTab)
  const railBtn = (t) =>
    `<button type="button" class="xai-rail-btn${t.id === xaiTab ? ' active' : ''}" data-xai-tab="${escapeHtml(t.id)}" title="${escapeHtml(t.label)}"><span class="xai-rail-ico" aria-hidden="true">${t.icon(16)}</span><span class="xai-rail-lbl">${escapeHtml(t.label)}</span></button>`
  const rail = `<nav class="xai-rail" aria-label="xAI controls and views">
    ${xaiRailControlsHtml()}
    <div class="xai-rail-tabs">${tabs.map(railBtn).join('')}</div>
    <button type="button" class="xai-rail-btn xai-rail-toggle" data-xai-rail-toggle title="${xaiRailCollapsed ? 'Expand panel' : 'Collapse panel'}"><span class="xai-rail-ico" aria-hidden="true">${xaiRailCollapsed ? '»' : '«'}</span><span class="xai-rail-lbl">Collapse</span></button>
  </nav>`
  return `<div class="xai-shell${xaiRailCollapsed ? ' rail-collapsed' : ''}">${rail}
    <div class="xai-tab-content">${xaiScopeHeaderHtml(criterion)}${active ? xaiViewHtml(active) : prompt}</div></div>`
}
// The rail's top controls: scope toggle, then criterion + direction (hidden when the rail is collapsed).
function xaiRailControlsHtml() {
  const scopeBtn = (s, l) =>
    `<button type="button" class="ghost-btn xai-scope-btn${xaiScope === s ? ' active' : ''}" data-xai-scope="${s}">${l}</button>`
  return `<div class="xai-rail-controls">
    <div class="xai-scope-switch">${scopeBtn('all', 'All runs')}${scopeBtn('current', 'Current run')}</div>
    <p id="xai-status" class="form-status" role="status" hidden></p>
  </div>`
}
// One view = a sticky header (title · the tab's own controls · Discuss · "?") over a body that scrolls within.
// The shared scope header shown above EVERY view: what is being analysed (environment + criterion +
// direction, with Re-run) for the all-runs scope, or which run is in focus for the current scope — so the
// analysis context is always visible, never hidden behind the Environments tab.
function xaiScopeHeaderHtml(criterion) {
  const critSel = `<label class="card-sub xai-scope-field">Criterion <select id="xai-criterion" class="app-select">${xaiCriteria()
    .map(
      (c) =>
        `<option value="${escapeHtml(c.key)}"${c.key === xaiCriterionKey ? ' selected' : ''}>${escapeHtml(c.label)}</option>`,
    )
    .join('')}</select></label>`
  const dirSel = `<label class="card-sub xai-scope-field">Better when <select id="xai-direction" class="app-select"><option value="max"${criterion.direction === 'max' ? ' selected' : ''}>higher</option><option value="min"${criterion.direction === 'min' ? ' selected' : ''}>lower</option></select></label>`
  if (xaiScope === 'current') {
    const favs = [...favoritesCache]
    const favPicker = favs.length
      ? `<label class="card-sub xai-scope-field">\u2605 Favorites <select id="xai-favorite-pick" class="app-select"><option value="">jump to a favorite\u2026</option>${favs.map((k) => `<option value="${escapeHtml(k)}"${k === xaiFocusKey ? ' selected' : ''}>${escapeHtml(favoriteOptionLabel(k))}</option>`).join('')}</select></label>`
      : ''
    if (!xaiFocusKey)
      return favPicker
        ? `<div class="card xai-scope-header"><div class="xai-scope-body"><span class="xai-scope-tag">No run focused</span>${favPicker}<span class="xai-scope-spacer"></span></div></div>`
        : ''
    return `<div class="card xai-scope-header">
      <div class="xai-scope-body">
        <span class="xai-scope-tag">Run in focus</span><code class="xai-scope-value">${escapeHtml(shortKey(xaiFocusKey))}</code>
        ${favPicker}${critSel}${dirSel}
        <span class="xai-scope-spacer"></span>
        <button type="button" class="ghost-btn" data-xai-clear-focus>\u2715 Clear run</button>
      </div>
    </div>`
  }
  const bundle = xaiResolveBundle(criterion)
  if (!bundle) {
    return `<div class="card xai-scope-header"><div class="xai-scope-body">
      <span class="xai-scope-tag">No analysis yet</span>${critSel}${dirSel}
      <span class="xai-scope-spacer"></span>${xaiAnalyzeAllBtnHtml('Run analysis')}
    </div></div>`
  }
  const a = bundle.analysis
  const when = bundle.generatedAt ? formatWhen(bundle.generatedAt) : 'recently'
  const envs = a.environments || []
  const currentSig = window.Xai.canonicalConfigString(a.environment || {})
  const envLabel = a.environment ? xaiEnvLabel(a.environment) : 'the whole space'
  const envControl =
    envs.length > 1
      ? `<label class="card-sub xai-scope-field">Environment <select id="xai-env-switch" class="app-select" title="Switch the environment being analysed">${envs
          .map(
            (e) =>
              `<option value="${escapeHtml(e.signature)}"${e.signature === currentSig ? ' selected' : ''}>${escapeHtml(xaiEnvLabel(e.values))} (${e.runCount})</option>`,
          )
          .join('')}</select></label>`
      : `<span class="xai-scope-tag">Analysing</span><code class="xai-scope-value">${escapeHtml(envLabel)}</code>`
  const ctx = [...(a.contextImportances || [])].sort((x, y) => y.importance - x.importance).slice(0, 3)
  const ctxBadge = ctx.length
    ? `<span class="card-sub" title="Context (environment/dataset) levers that move the score most">\ud83d\udd12 ${ctx
        .map((sc) => `<code>${escapeHtml(xaiAbbrevLever(sc.lever))}</code>`)
        .join(', ')}</span>`
    : ''
  const expanded = xaiScopeHeaderExpanded
  return `<div class="card xai-scope-header${expanded ? '' : ' is-collapsed'}">
    <button type="button" class="xai-scope-toggle" data-xai-scope-header-toggle aria-expanded="${expanded ? 'true' : 'false'}">
      <span class="xai-collapse-chevron" aria-hidden="true">\u25be</span>
      <span class="xai-scope-tag">Analysing</span><span class="xai-scope-value-inline">${escapeHtml(envLabel)}</span>
      <span class="card-sub">\u00b7 ${escapeHtml(criterion.label)} (${criterion.direction === 'min' ? 'lower' : 'higher'} better) \u00b7 ${a.runCount} runs \u00b7 ${escapeHtml(when)}</span>
    </button>
    <div class="xai-scope-body">
      ${envControl}${critSel}${dirSel}${ctxBadge}
      <span class="xai-scope-spacer"></span>
      ${xaiAnalyzeAllBtnHtml('Re-run analysis')}
    </div>
  </div>`
}
function xaiViewHtml(tab) {
  const controls = tab.controls ? tab.controls() : ''
  const chatBtn = chatAboutRunAvailable()
    ? `<button type="button" class="icon-btn" data-xai-chat="${escapeHtml(tab.id)}" title="Discuss what you see in “${escapeHtml(tab.label)}” with the AI" aria-label="Discuss this view">${iconChatSvg(15)}</button>`
    : ''
  const helpBtn = `<button type="button" class="icon-btn xai-help-btn${xaiHelpOpen === tab.id ? ' active' : ''}" data-xai-help="${escapeHtml(tab.id)}" title="${escapeHtml(xaiHelp(tab.id).title)}" aria-label="Explain this view">?</button>`
  const callout =
    xaiHelpOpen === tab.id
      ? `<div class="xai-help-callout"><p>${xaiHelp(tab.id).body}</p></div>`
      : ''
  return `<section class="xai-view">
    <header class="xai-view-head"><h3>${escapeHtml(tab.label)}</h3><div class="xai-view-actions">${controls}${chatBtn}${helpBtn}</div></header>
    ${callout}
    <div class="xai-view-body">${tab.render()}</div>
  </section>`
}
function xaiAllTabs(bundle, criterion) {
  return [
    {
      id: 'environments',
      label: 'Environments',
      icon: xaiIconEnv,
      render: () => xaiEnvironmentsTabHtml(bundle, criterion),
    },
    {
      id: 'effects',
      label: 'Config effects',
      icon: xaiIconSliders,
      render: () => xaiConfigEffectsHtml(bundle, criterion),
    },
    {
      id: 'surrogate',
      label: 'Surrogate',
      icon: xaiIconSurrogate,
      render: () => xaiSurrogateHtml(bundle, criterion),
    },
    {
      id: 'maps',
      label: 'Maps',
      icon: xaiIconMap,
      controls: () =>
        [
          ['parallel', 'Parallel'],
          ['pca', 'PCA'],
          ['pareto', 'Trade-off'],
        ]
          .map(
            ([k, l]) =>
              `<button type="button" class="xai-map-tab${xaiMapKind === k ? ' active' : ''}" data-xai-map="${k}">${l}</button>`,
          )
          .join(''),
      render: () =>
        xaiMapKind === 'pareto'
          ? xaiParetoHtml(bundle, criterion)
          : xaiMapKind === 'pca'
            ? xaiPcaHtml(bundle, criterion)
            : xaiParallelCoordsHtml(bundle, criterion),
    },
    { id: 'progress', label: 'Progress', icon: xaiIconRank, render: () => xaiConvergenceHtml(bundle, criterion) + xaiLeaderboardHtml(bundle, criterion) },
    {
      id: 'recommender',
      label: 'Suggested',
      icon: iconLightbulbSvg,
      controls: () => xaiRecommenderControlsHtml(bundle),
      render: () => xaiRecommenderHtml(criterion, bundle),
    },
  ]
}
function xaiCurrentTabs(criterion) {
  return [
    {
      id: 'standing',
      label: 'Standing',
      icon: xaiIconRank,
      render: () => xaiRunStandingHtml(xaiFocusKey, criterion),
    },
    {
      id: 'map',
      label: 'Map',
      icon: xaiIconMap,
      controls: () => xaiCurrentMapControlsHtml(),
      render: () => xaiCurrentMapHtml(xaiFocusKey, criterion),
    },
    {
      id: 'narrative',
      label: 'Narrative',
      icon: iconChatSvg,
      render: () => xaiNarrativeHtml(xaiTotalRuns(), criterion),
    },
    {
      id: 'internals',
      label: 'Internals',
      icon: xaiIconSurrogate,
      render: () => xaiModelInternalsHtml(xaiFocusKey),
    },
  ]
}
// CURRENT-RUN map: the same configuration maps as All-runs, scoped to the focused run's environment, with
// THIS run ringed so you can see where it sits among similar runs. Falls back to hints when the whole-space
// analysis hasn't been computed, or is scoped to a different environment than this run.
function xaiCurrentMapHtml(focusKey, criterion) {
  const run = findRun(focusKey)
  const rec = xaiRunAnalysisCache.get(focusKey)
  // The digest config (rec.analysis.config) only strips ignore levers; the run-summary fallback is already
  // conditional-normalized (recordToRun). Normalize here too so present-but-inactive conditional levers read
  // 'n/a' exactly as the bundle's setup configs do — otherwise the "this run" match silently fails.
  let focusCfg =
    (rec && rec.analysis && rec.analysis.config) || (run && run.summary && run.summary.config) || null
  if (focusCfg && manifest && manifest.levers)
    focusCfg = window.Xai.normalizeConditionalConfig(focusCfg, manifest.levers)
  const bundle = xaiResolveBundle(criterion)
  const card = (inner, head) =>
    `<div class="card"><div class="card-head card-head-row"><h3>Map <span class="card-sub">\u2014 where this run sits among similar runs</span></h3>${head || ''}</div>${inner}</div>`
  if (!focusCfg)
    return card(`<p class="card-sub">This run isn\u2019t loaded \u2014 open it from the <strong>Runs</strong> tab.</p>`)
  if (!bundle)
    return card(
      `<p class="card-sub">Run the whole-space analysis to place this run on the configuration map among every other run.</p>`,
      xaiAnalyzeAllBtnHtml('Run analysis'),
    )
  const a = bundle.analysis
  const env = a.environment
  const inEnv = !env || (a.contextLevers || []).every((lev) => String(focusCfg[lev]) === String(env[lev]))
  if (!inEnv) {
    const envVals = {}
    for (const lev of a.contextLevers || []) if (focusCfg[lev] !== undefined) envVals[lev] = focusCfg[lev]
    const sig = window.Xai.canonicalConfigString(envVals)
    const match = (a.environments || []).find((e) => e.signature === sig)
    const switchBtn = match
      ? `<button type="button" class="ghost-btn" data-xai-env="${escapeHtml(sig)}">Scope analysis to this run\u2019s environment</button>`
      : ''
    return card(
      `<p class="card-sub">The whole-space analysis is currently scoped to <strong>${escapeHtml(xaiEnvLabel(env))}</strong>, but this run is in <strong>${escapeHtml(xaiEnvLabel(envVals))}</strong>. ${match ? 'Scope it to this run\u2019s environment to place it on the map.' : 'Re-run the analysis scoped to this run\u2019s environment to place it on the map.'}</p>`,
      switchBtn,
    )
  }
  const intro = `<p class="card-sub">The same configuration maps as <strong>All runs</strong>, scoped to this run\u2019s environment, with <strong style="color:#1d4ed8">this run ringed in blue</strong>. Switch the map type at the top-right; on the trade-off map you can add a 3rd metric for a 3-D view.</p>`
  const body =
    xaiMapKind === 'pareto'
      ? xaiParetoHtml(bundle, criterion, focusCfg)
      : xaiMapKind === 'pca'
        ? xaiPcaHtml(bundle, criterion, focusCfg)
        : xaiParallelCoordsHtml(bundle, criterion, focusCfg)
  return `${card(intro)}${body}`
}
// The map-kind switch (Parallel / PCA / Trade-off) for the current-run Map tab \u2014 shares xaiMapKind +
// the data-xai-map handler with the All-runs Maps tab, so the chosen map type stays consistent across scopes.
function xaiCurrentMapControlsHtml() {
  return [
    ['parallel', 'Parallel'],
    ['pca', 'PCA'],
    ['pareto', 'Trade-off'],
  ]
    .map(
      ([k, l]) =>
        `<button type="button" class="xai-map-tab${xaiMapKind === k ? ' active' : ''}" data-xai-map="${k}">${l}</button>`,
    )
    .join('')
}
// The best total-run count the viewer knows without re-querying — the analysed bundle's count, else the
// runs-tab total, else the loaded page. Used for the narrative's "N new runs since" hint.
function xaiTotalRuns() {
  let max = 0
  for (const c of xaiConfigSpaceCache.values()) max = Math.max(max, c.runCount || 0)
  return max || runsTotalCount || runsCache.length
}
// Plain-language explainer for each view's "?" button — title is the hover tooltip, body the click callout.
// Maps explains the CURRENTLY-selected map (parallel vs PCA), per the user's ask.
function xaiHelp(viewId) {
  if (viewId === 'maps') {
    if (xaiMapKind === 'pareto')
      return {
        title: 'What the Trade-off (Pareto) map shows',
        body: 'Every point is a config plotted on TWO metrics you choose (e.g. return vs drawdown). A config is on the <strong>Pareto frontier</strong> (green) when no other config beats it on <em>both</em> axes at once; grey points are <em>dominated</em> (something is better on both, so never pick them). There is no single "best" \u2014 choose from the green frontier by how much of one metric you\u2019ll trade for the other. Set each axis\u2019s metric + whether higher or lower is better. Add an optional <strong>Z</strong> metric to make it a 3-D trade-off (the frontier then needs no config to beat it on all three axes); leave Z on \u201cnone\u201d to stay 2-D.',
      }
    return xaiMapKind === 'pca'
      ? {
          title: 'What the PCA map shows',
          body: 'A 2-D <em>sketch</em> of every explored config: numeric levers z-scored, categorical one-hot, then squashed to two axes that capture the most variance. The axes are <em>blends</em> of levers, not knobs — so read it only for <strong>clusters</strong> (configs that behave alike sit together) and <strong>outliers</strong>. Colour = performance rank (green = top). It can’t tell you which lever to turn — use Parallel or the Surrogate for that.',
        }
      : {
          title: 'What the Parallel map shows',
          body: 'Every line is one config threading its values across the lever axes (ordered most-decisive-first by fANOVA); colour = performance rank, green = top. <strong>Read it directly:</strong> where the green lines <em>bunch</em> at one value on an axis, that value is good; where they fan across the axis, that lever doesn’t decide the outcome. Real axes, so no "don’t read the directions" caveat.',
        }
  }
  const H = {
    progress: {
      title: 'About Search progress',
      body: 'The <strong>best score</strong> reached so far plotted against how many runs you have done, in <em>time order</em>. Each step up is a new best config. A long <strong>flat tail</strong> means more runs of the same kind are unlikely to help \u2014 time to stop, widen the search, or try a different approach. A still-rising curve means keep going. The <strong>Top configs</strong> table below ranks the best configs with confidence intervals, flags which are statistically tied (overlapping CIs), and flags any that score implausibly well (a leakage guardrail).',
    },
    environments: {
      title: 'What environments are',
      body: 'An <strong>environment</strong> is a fixed combination of market-mechanics + dataset settings (fee, stop-loss, asset, walk-forward window, sizing…). Each is analysed <em>separately</em> and never tuned — a config that wins in one can lose in another, so they’re never blended. Click one to scope every other tab to it. The 🔒 list shows which context settings move the score most.',
    },
    effects: {
      title: 'What Config effects show',
      body: '<strong>Lever importance</strong> screens, model-free, how much each lever’s value swings the score (confounded — a hint, not proof). <strong>One-factor effect</strong> is the controlled read: it compares a lever’s values <em>holding everything else fixed</em>, with a bootstrap CI and significance, so you can trust it. Use importance to pick what to look at, the contrast to decide.',
    },
    surrogate: {
      title: 'What the Surrogate model is',
      body: 'A seeded random forest fit on every config → score, so it can predict the score of configs you <em>haven’t</em> run. From it: <strong>fANOVA</strong> (each lever’s global importance — main effect vs total, which includes interactions), <strong>coupling</strong> (lever pairs whose best value depends on each other), and the <strong>ablation tree</strong> (worst→best, the single change that helps most at each step). It’s a model — treat it as a hypothesis to confirm with real runs.',
    },
    recommender: {
      title: 'What Suggested experiments are',
      body: 'The next configs worth running — <strong>▲ climb</strong> picks are the surrogate’s highest Expected-Improvement unrun configs (toward the optimum), <strong>gap</strong>/<strong>pair</strong> fill untested factorial cells, <strong>seeds</strong> firm up a thin top setup, and <strong>✦ AI</strong> are model-proposed ideas beyond the grid. They never change the environment’s settings — only model levers. Run a batch to close the analyse → run → re-analyse loop.',
    },
    map: {
      title: 'Where this run sits',
      body: 'The same configuration maps as <strong>All runs</strong> (Parallel / PCA / Trade-off), scoped to this run\u2019s environment, with <strong>this run ringed in blue</strong> so you can see where it sits among similar runs \u2014 and, on the trade-off map, whether it\u2019s on the frontier or dominated. Needs the whole-space analysis computed for this run\u2019s environment.',
    },
    standing: {
      title: 'How this run ranks',
      body: 'This run’s standing among <strong>every</strong> run (not just the page) by the chosen criterion, plus its action mix and whether its input-attribution passed its faithfulness check. Computed on demand server-side; only the 5 most-recently analysed runs are kept.',
    },
    narrative: {
      title: 'About the narrative',
      body: 'A one-shot LLM read of THIS run — what it’s doing, what drives its decisions, how trustworthy the explanation is, and what to try next. It synthesises the deterministic analysis; it doesn’t replace it.',
    },
    internals: {
      title: 'About model internals',
      body: 'This run’s own decision internals from its trace — decisiveness, policy entropy over time, a confidence-vs-realised-reward calibration table, and (vs its nearest comparable run) how its decisions differed, not just its score.',
    },
  }
  return H[viewId] || { title: viewId, body: '' }
}
// The "Run/Re-run analysis" button for the whole-space scope — drives config-space-analyze.
function xaiAnalyzeAllBtnHtml(label) {
  const disabled = analyzingConfigSpace || !embedded() ? ' disabled' : ''
  return `<button type="button" class="ghost-btn" data-xai-refresh-analysis${disabled} title="Compute the whole-space surrogate / fANOVA / coupling / PCA / config-effects over EVERY completed run, server-side (runs in the activity queue)">${analyzingConfigSpace ? `${spinnerHtml()} Analysing…` : escapeHtml(label)}</button>`
}
// ALL-RUNS scope: render purely from the server-cached whole-space bundle — never the current page.
// All-runs scope, nothing cached yet — the prompt to compute the whole-space analysis.
function xaiNoBundleHtml(criterion) {
  return `<div class="card"><div class="card-head card-head-row"><h3>Whole-space analysis</h3>${xaiAnalyzeAllBtnHtml('Run analysis')}</div>
    <p class="card-sub">Runs the surrogate, fANOVA, coupling, config-effects, maps and recommender over <strong>EVERY completed run</strong> (not just this page) — server-side, in the activity queue — and caches the result here. ${analyzingConfigSpace ? 'Analysing now…' : `Click <strong>Run analysis</strong> for the “${escapeHtml(criterion.label)}” criterion.`}</p></div>`
}
// ENVIRONMENTS tab: the analysis freshness + re-run, the scope note, and the (consolidated) environment list.
function xaiEnvironmentsTabHtml(bundle, criterion) {
  const a = bundle.analysis
  const scopeNote = a.environment
    ? `Scoped to <strong>${escapeHtml(xaiEnvLabel(a.environment))}</strong> — <strong>model levers only</strong>; its environment/dataset settings are held fixed (never tuned). The other tabs analyse THIS environment.`
    : `No environment/dataset levers declared — the whole space is analysed together.`
  return (a.environments || []).length
    ? `<p class="card-sub">${scopeNote}</p>${xaiCompareEnvironmentsHtml(a, criterion)}`
    : `<div class="card"><p class="card-sub">${scopeNote}</p></div>`
}
// A human-readable label for an environment (its context-lever values).
function xaiEnvLabel(values) {
  const parts = Object.entries(values || {}).map(([k, v]) => `${k}=${xaiFmtLeverValue(v)}`)
  return parts.length ? parts.join(' · ') : 'default'
}
// The column model for the environment table: one column per context lever that VARIES across environments
// (constant ones are dead columns), then runs + best. Mirrors runsColumns() so the table reads like Runs.
function xaiEnvColumns(a, criterion) {
  const envs = a.environments || []
  const declared =
    a.contextLevers && a.contextLevers.length
      ? a.contextLevers
      : Array.from(new Set(envs.flatMap((e) => Object.keys(e.values || {}))))
  const varying = declared.filter(
    (k) => new Set(envs.map((e) => xaiFmtLeverValue((e.values || {})[k]))).size > 1,
  )
  const constant = declared.filter((k) => !varying.includes(k))
  const cols = varying.map((k) => ({
    id: 'lv:' + k,
    label: xaiAbbrevLever(k),
    help: `Environment/dataset setting “${k}” (🔒 context — compared, never tuned).`,
    num: false,
    get: (e) => escapeHtml(xaiFmtLeverValue((e.values || {})[k])),
    sort: (e) => (e.values || {})[k],
  }))
  cols.push({
    id: 'runs',
    label: 'runs',
    num: true,
    get: (e) => String(e.runCount),
    sort: (e) => e.runCount,
  })
  cols.push({
    id: 'best',
    label: `best ${criterion.label}`,
    num: true,
    get: (e) => escapeHtml(formatTickValue(e.best)),
    sort: (e) => e.best,
  })
  return { cols, constant }
}
// Cross-environment comparison as a Runs-style columnar table: context levers are columns, headers sort,
// clicking a row scopes the other tabs to that environment. Each environment is analysed separately + never
// tuned; the 🔒 list shows which context settings move the score most.
function xaiCompareEnvironmentsHtml(a, criterion) {
  const all = a.environments || []
  const currentSig = window.Xai.canonicalConfigString(a.environment || {})
  const { cols, constant } = xaiEnvColumns(a, criterion)
  const q = xaiEnvSearch.trim().toLowerCase()
  let envs = q ? all.filter((e) => xaiEnvLabel(e.values).toLowerCase().includes(q)) : [...all]
  const col = cols.find((c) => c.id === xaiEnvSortKey) || cols[cols.length - 1]
  const dir = xaiEnvSortDir === 'asc' ? 1 : -1
  envs.sort((x, y) => {
    const xv = col.sort(x)
    const yv = col.sort(y)
    if (typeof xv === 'number' && typeof yv === 'number') return (xv - yv) * dir
    return String(xv).localeCompare(String(yv)) * dir
  })
  const header = cols
    .map((c) => {
      const arrow = xaiEnvSortKey === c.id ? (xaiEnvSortDir === 'asc' ? ' ▲' : ' ▼') : ''
      return `<th class="runs-th${c.num ? ' num' : ''}" data-xai-env-sort="${escapeHtml(c.id)}"${helpAttr(c.help)}>${escapeHtml(c.label)}${arrow}</th>`
    })
    .join('')
  const rows = envs
    .map((e) => {
      const cur = e.signature === currentSig
      const cells = cols.map((c) => `<td class="${c.num ? 'num' : ''}">${c.get(e)}</td>`).join('')
      return `<tr class="xai-env-row${cur ? ' is-selected xai-env-current' : ''}" data-xai-env="${escapeHtml(e.signature)}" title="Scope the analysis to this environment">${cells}</tr>`
    })
    .join('')
  const constNote = constant.length
    ? `<p class="card-sub">Held the same across every environment: ${constant.map((k) => `<code>${escapeHtml(k)}</code>`).join(', ')}.</p>`
    : ''
  const ctx = [...(a.contextImportances || [])].sort((x, y) => y.importance - x.importance)
  const ctxList = ctx.length
    ? `<h4 class="card-sub">Which environment/dataset settings move the score most <span class="card-sub">(🔒 context — compare only, never tuned)</span></h4>
       <ul class="xai-coupling">${ctx.map((s) => `<li>🔒 <code>${escapeHtml(s.lever)}</code> <span class="num">${Math.round(s.importance * 100)}%</span> <span class="card-sub">best ${escapeHtml(String(s.bestValue))}</span></li>`).join('')}</ul>`
    : ''
  return `<div class="card"><div class="card-head card-head-row"><h3>Compare environments <span class="card-sub">— ${all.length} environments · click a row to scope to it</span></h3></div>
    <p class="card-sub">Each environment (market mechanics + which data) is analysed <em>separately</em> — a config that's great in one can be poor in another, so they're never blended. ${constNote}</p>
    <div class="badges-row xai-env-controls">
      <input type="search" id="xai-env-search" class="xai-env-search" placeholder="Filter environments…" value="${escapeHtml(xaiEnvSearch)}" aria-label="Filter environments" />
      <span class="card-sub">${envs.length} of ${all.length} · click a header to sort</span>
    </div>
    <div class="xai-env-scroll table-wrap"><table class="runs-table">
      <thead><tr>${header}</tr></thead>
      <tbody>${rows || `<tr><td colspan="${cols.length}" class="card-sub">no environments match “${escapeHtml(xaiEnvSearch)}”</td></tr>`}</tbody></table></div>
    ${ctxList}</div>`
}
// The focused run's standing among ALL runs (rank + action mix + attribution sanity), from the on-demand
// LRU-cached digest. The button computes/recomputes it server-side over every run.
// The closest runs to the focused one IN ITS ENVIRONMENT (from the whole-space bundle's setups), with the
// one or two levers that differ + their score delta — \u201cwhy did this run land here vs its neighbours\u201d.
// Returns '' when the cached bundle is for a different environment than this run (so it never misleads).
function xaiNeighboursHtml(digest, bundle, criterion) {
  const a = bundle.analysis
  if (!digest || !digest.config) return ''
  const env = a.environment
  const inEnv = !env || (a.contextLevers || []).every((lev) => String(digest.config[lev]) === String(env[lev]))
  if (!inEnv) return ''
  const focus = digest.config
  const all = (a.setups || []).filter((sp) => sp.config)
  if (!all.length) return ''
  const leverKeys = (a.levers && a.levers.length ? a.levers : Object.keys(all[0].config)).filter((k) => k !== 'seed')
  const ranges = {}
  for (const k of leverKeys) {
    const nums = [focus[k]].concat(all.map((sp) => sp.config[k])).map(Number).filter((x) => Number.isFinite(x))
    if (nums.length) {
      const lo = Math.min.apply(null, nums)
      const hi = Math.max.apply(null, nums)
      ranges[k] = hi > lo ? hi - lo : 0
    }
  }
  const dist = (cfg) => {
    let d = 0
    for (const k of leverKeys) {
      const rng = ranges[k]
      const fk = Number(focus[k])
      const ck = Number(cfg[k])
      if (rng && Number.isFinite(fk) && Number.isFinite(ck)) d += Math.abs((fk - ck) / rng)
      else if (String(focus[k]) !== String(cfg[k])) d += 1
    }
    return d
  }
  const val = (sp) => xaiMetricVal(sp, criterion.key)
  const scored = all.map((sp) => ({ sp, d: dist(sp.config) }))
  // The focus run's OWN setup has distance 0 (identical model levers) \u2014 use it for the baseline value and
  // exclude it from the neighbour list. (Setups are keyed by their first seed, so a key compare misses it.)
  const self = scored.find((x) => x.d <= 1e-9)
  const fval =
    self && typeof val(self.sp) === 'number'
      ? val(self.sp)
      : typeof digest.objective === 'number'
        ? digest.objective
        : null
  const neigh = scored
    .filter((x) => x.d > 1e-9)
    .sort((x, y) => x.d - y.d)
    .slice(0, 5)
  const diffOf = (cfg) =>
    leverKeys
      .filter((k) => String(focus[k]) !== String(cfg[k]))
      .map((k) => `<code>${escapeHtml(xaiAbbrevLever(k))}: ${escapeHtml(xaiFmtLeverValue(focus[k]))}\u2192${escapeHtml(xaiFmtLeverValue(cfg[k]))}</code>`)
      .join(' ') || '<span class="card-sub">same levers (seed only)</span>'
  const rows = neigh
    .map(({ sp }) => {
      const sv = val(sp)
      const delta = fval != null && typeof sv === 'number' ? sv - fval : null
      const dcls = delta == null ? '' : (criterion.direction === 'min' ? delta < 0 : delta > 0) ? 'delta-pos' : 'delta-neg'
      const dstr = delta == null ? '\u2014' : `<span class="${dcls}">${delta >= 0 ? '+' : ''}${escapeHtml(formatTickValue(delta))}</span>`
      return `<tr class="xai-lb-row" data-xai-focus-config="${escapeHtml(sp.key)}" title="Open this run"><td>${diffOf(sp.config)}</td><td class="num">${escapeHtml(formatTickValue(sv))}</td><td class="num">${dstr}</td></tr>`
    })
    .join('')
  return `<div class="card"><div class="card-head card-head-row"><h3>Nearest configs <span class="card-sub">\u2014 the closest runs in this environment + how they differ</span></h3></div>
    <p class="card-sub">The runs whose settings are most similar to this one, the lever(s) that differ, and their score vs this run \u2014 a controlled view of what moved the result.</p>
    <table class="kv-table report-table"><thead><tr><th>differs by</th><th class="num">${escapeHtml(criterion.label)}</th><th class="num">vs this run</th></tr></thead><tbody>${rows}</tbody></table></div>`
}
function xaiRunStandingHtml(runKey, criterion) {
  const busy = analyzingRunKey === runKey
  const rec = xaiRunAnalysisCache.get(runKey)
  const digest = rec && rec.analysis ? rec.analysis : null
  const disabled = busy || !embedded() ? ' disabled' : ''
  const btn = `<button type="button" class="ghost-btn" data-xai-analyze-run="${escapeHtml(runKey)}"${disabled} title="Compute this run's standing among EVERY run (rank, where it sits on each lever, its closest comparable run, what drove it), server-side">${busy ? `${spinnerHtml()} Analysing…` : digest ? 'Re-analyse' : 'Analyse among all runs'}</button>`
  if (!digest) {
    return `<div class="card"><div class="card-head card-head-row"><h3>Why this run got this result</h3><div class="head-actions">${btn}</div></div>
      <p class="card-sub">${busy ? 'Analysing this run against every other run…' : 'Compute, server-side, how this run ranks among <strong>all</strong> runs, where it sits on each decisive lever, its closest comparable run, and what drove its result — to explain why it worked (or did not).'}</p></div>`
  }
  const when = rec.generatedAt ? formatWhen(rec.generatedAt) : ''
  const pos = digest.rank ? digest.rank.position : null
  const tot = digest.rank ? digest.rank.total : null
  const topPct = pos && tot ? Math.max(1, Math.ceil((pos / tot) * 100)) : null
  const objLine = digest.objective != null ? `${escapeHtml(criterion.label)} ${escapeHtml(formatTickValue(digest.objective))}` : ''
  const actions = digest.actionCounts
    ? Object.entries(digest.actionCounts).map(([k, v]) => `${escapeHtml(k)} ${v}`).join(' · ')
    : ''
  const sanity =
    digest.attribution && typeof digest.attribution.sanityPassed === 'boolean'
      ? digest.attribution.sanityPassed
        ? '<span class="delta-pos">attribution sanity ✓</span>'
        : '<span class="delta-neg">attribution sanity ✗ (untrustworthy)</span>'
      : ''
  const stabBtn = chatAboutRunAvailable()
    ? `<button type="button" class="ghost-btn" data-xai-stabilize-current title="Run more seeds of this exact config to check the result is repeatable, not seed luck">Verify seeds \u21bb</button>`
    : ''
  const standingCard = `<div class="card"><div class="card-head card-head-row"><h3>Standing${objLine ? ` <span class="card-sub">— ${objLine}</span>` : ''}</h3><div class="head-actions">${stabBtn}${btn}</div></div>
    <table class="kv-table"><tbody>
      <tr><th>Rank by ${escapeHtml(criterion.label)}</th><td>${pos != null ? `#${pos} of ${tot}${topPct != null ? ` <span class="card-sub">(top ${topPct}%)</span>` : ''}` : '—'}</td></tr>
      ${actions ? `<tr><th>Action mix</th><td>${actions}</td></tr>` : ''}
    </tbody></table>
    <p class="card-sub">Analysed ${escapeHtml(when)} over ${rec.runCount || 0} runs.</p></div>`
  const leverRows = (digest.importances || [])
    .filter((i) => (i.lever === 'model_name' || i.importance >= 0.02) && String(digest.config[i.lever]) !== 'n/a')
    .slice(0, 6)
    .map((i) => {
      const mine = digest.config ? digest.config[i.lever] : undefined
      const mineStr = mine === undefined ? '—' : xaiFmtLeverValue(mine)
      const atBest = mine !== undefined && String(mine) === String(i.bestValue)
      const verdict =
        mine === undefined
          ? ''
          : atBest
            ? '<span class="delta-pos">✓ at best-seen</span>'
            : `<span class="delta-neg">→ try ${escapeHtml(xaiFmtLeverValue(i.bestValue))}</span>`
      const explore =
        i.lever === 'model_name'
          ? ''
          : `<button type="button" class="ghost-btn xai-explore-btn" data-xai-explore-lever="${escapeHtml(i.lever)}" title="Launch a campaign sweeping ${escapeHtml(i.lever)} across its values, holding this run's other settings fixed">sweep \u2197</button>`
      return `<tr><th><code>${escapeHtml(xaiAbbrevLever(i.lever))}</code></th><td class="num">${Math.round(i.importance * 100)}%</td><td>${escapeHtml(mineStr)}</td><td>${escapeHtml(xaiFmtLeverValue(i.bestValue))}</td><td>${verdict} ${explore}</td></tr>`
    })
    .join('')
  const leversCard = leverRows
    ? `<div class="card"><div class="card-head card-head-row"><h3>Where this run sits <span class="card-sub">— the most decisive levers (screening, confounded) and whether this run uses the best-seen value</span></h3></div>
      <table class="kv-table report-table"><thead><tr><th>lever</th><th class="num">importance</th><th>this run</th><th>best seen</th><th></th></tr></thead><tbody>${leverRows}</tbody></table></div>`
    : ''
  const sib = digest.sibling
  const sibCard = sib
    ? `<div class="card"><div class="card-head card-head-row"><h3>Closest comparable run <span class="card-sub">— ${escapeHtml(shortKey(sib.key))}</span></h3></div>
      <p class="card-sub">Differs by <code>${escapeHtml(sib.changed)}</code>; their decisions diverged <strong>${Math.round(sib.divergencePct)}%</strong>${sib.qualityVerdict ? ` · ${escapeHtml(sib.qualityVerdict)}` : ''}.${sib.qualitySummary ? ` ${escapeHtml(sib.qualitySummary)}` : ''}</p></div>`
    : ''
  const reward = digest.rewardBreakdown
    ? Object.entries(digest.rewardBreakdown).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 4)
    : []
  const rewardLine = reward.length
    ? `<p class="card-sub"><strong>Reward drivers:</strong> ${reward.map(([k, v]) => `${escapeHtml(k)} ${v >= 0 ? '+' : ''}${escapeHtml(formatTickValue(v))}`).join(' · ')}</p>`
    : ''
  const attrLine =
    digest.attribution && digest.attribution.topGroups && digest.attribution.topGroups.length
      ? `<p class="card-sub"><strong>Decision drivers (saliency):</strong> ${digest.attribution.topGroups.slice(0, 4).map((g) => `<code>${escapeHtml(g[0])}</code>`).join(', ')} ${sanity}</p>`
      : sanity
        ? `<p class="card-sub">${sanity}</p>`
        : ''
  const latentLine =
    digest.latent && digest.latent.probeAccuracy != null
      ? `<p class="card-sub"><strong>Representation:</strong> probe accuracy ${escapeHtml(formatTickValue(digest.latent.probeAccuracy))}${digest.latent.probeBaseline != null ? ` (baseline ${escapeHtml(formatTickValue(digest.latent.probeBaseline))})` : ''}</p>`
      : ''
  const whyCard =
    rewardLine || attrLine || latentLine
      ? `<div class="card"><div class="card-head card-head-row"><h3>Why this result</h3></div>${rewardLine}${attrLine}${latentLine}</div>`
      : `<div class="card"><div class="card-head card-head-row"><h3>Why this result</h3></div><p class="card-sub">This run didn\u2019t record decision attribution, a reward breakdown, or a latent probe, so the deterministic \u201cwhy\u201d is limited \u2014 the <strong>Internals</strong> tab still shows its decision trace, regimes, exits, and trade ledger. To check the result is real (not seed luck), use <strong>Verify seeds</strong> above; the trustworthiness flag in the narrative reflects this missing attribution.</p></div>`
  const nbBundle = xaiResolveBundle(criterion)
  const neighboursCard = nbBundle ? xaiNeighboursHtml(digest, nbBundle, criterion) : ''
  // Seed robustness: is this exact config repeatable, or does the score swing with the seed?
  let seedCard = ''
  const selfSetup =
    nbBundle && digest.config
      ? (nbBundle.analysis.setups || []).find((sp) =>
          Object.keys(sp.config).every((k) => String(sp.config[k]) === String(digest.config[k])),
        )
      : null
  if (selfSetup) {
    const n = selfSetup.seeds || 1
    if (n < 2) {
      seedCard = `<div class="card"><div class="card-head card-head-row"><h3>Seed robustness <span class="card-sub">— 1 seed</span></h3></div><p class="card-sub">Only <strong>one seed</strong> has run this config — you can't tell skill from initialisation luck. Use <strong>Verify seeds</strong> above to run more and get a confidence interval.</p></div>`
    } else if (selfSetup.ci) {
      const ci = selfSetup.ci
      const span = Math.abs(ci[1] - ci[0])
      const rel = digest.objective ? span / Math.max(1e-9, Math.abs(digest.objective)) : span
      const verdict =
        rel < 0.1
          ? 'tight — the result looks <strong>repeatable</strong>'
          : rel < 0.3
            ? 'moderate — somewhat seed-sensitive'
            : '<strong>wide — seed-sensitive</strong> (the score swings with the seed; treat the ranking cautiously)'
      seedCard = `<div class="card"><div class="card-head card-head-row"><h3>Seed robustness <span class="card-sub">— ${n} seeds</span></h3></div><p class="card-sub">95% CI across seeds: <strong>[${escapeHtml(formatTickValue(ci[0]))}, ${escapeHtml(formatTickValue(ci[1]))}]</strong> — ${verdict}.</p></div>`
    }
  }
  return `${standingCard}${seedCard}${leversCard}${neighboursCard}${sibCard}${whyCard}`
}
// The one-shot LLM narrative of the FOCUSED run — what this model does, why, how trustworthy, vs its
// sibling, what to try next. Per run (keyed by run key); the button becomes "Refresh (N new runs)" as the
// cross-run context drifts. With no run focused, a slim hint points the user at how to focus one.
function xaiNarrativeHtml(nRuns, criterion) {
  if (!xaiFocusKey) {
    return `<div class="card"><div class="card-head card-head-row"><h3>Narrative <span class="card-sub">— LLM synthesis, per run</span></h3></div>
      <p class="card-sub">Focus a run (open it in Runs → “Analyze in xAI”) to generate a one-shot narrative of what that model is doing and why.</p></div>`
  }
  const rec = xaiNarrativeCache.get(xaiFocusKey)
  const hasNarrative = !!(rec && rec.narrative)
  const since = hasNarrative ? Math.max(0, nRuns - (Number(rec.runCount) || 0)) : 0
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
  </div>`
}
// Model internals for the focused run: the Explain panels + the decision-internals reads + a decision
// diff against the nearest comparable run (same data, differs by a lever).
function xaiModelInternalsHtml(focusKey) {
  const run = findRun(focusKey)
  if (!run)
    return `<div class="card"><p class="card-sub">This run isn't loaded \u2014 open it from the <strong>Runs</strong> tab (it may be off the current page).</p></div>`
  const s = run.summary
  const sibling = xaiBestSibling(run)
  return `<div class="card">
    <div class="card-head card-head-row"><h3>Model internals <span class="card-sub">— run ${escapeHtml(shortKey(focusKey))}</span></h3>
      <div class="head-actions">
        ${chatAboutRunAvailable() ? `<button type="button" class="ghost-btn" data-xai-discuss="${escapeHtml(focusKey)}" title="Discuss the FULL xAI analysis of this run with the AI">${iconChatSvg()} Discuss xAI</button>` : ''}
      </div></div>
    <div class="card-scroll">
    <h3>Metrics</h3>
    ${metricsTableHtml(s.metrics)}
    ${priceActionSectionHtml(s, run.key)}
    ${exitsSectionHtml(s)}
    ${regimesSectionHtml(s)}
    ${equityVsHoldSectionHtml(s) || trainingCurveSectionHtml(s)}
    ${explainSectionHtml(s)}
    ${decisionInternalsHtml(readDecisionTrace(s))}
    ${ledgerSectionHtml(s)}
    ${sibling ? decisionDiffSectionHtml(sibling.summary, s) : ''}
    </div></div>`
}
// The nearest comparable run for a decision diff: same dataset/window (step-alignable), differs in
// config — the best such by objective. Selected from the LEAN list cache (the trace can't be required
// here — it's omitted from list records — so the caller warms the chosen sibling's full record before
// reading its trace; the diff degrades to empty if that sibling has no trace).
function xaiBestSibling(run) {
  const sig = datasetAlignmentSignature(run.summary)
  if (!sig) return null
  const candidates = runsCache.filter(
    (r) =>
      r.key !== run.key &&
      r.summary &&
      r.summary.status !== 'failed' &&
      datasetAlignmentSignature(r.summary) === sig,
  )
  if (!candidates.length) return null
  const dir = objectiveDirection()
  const best = candidates.sort((a, b) =>
    dir === 'max'
      ? b.summary.objective - a.summary.objective
      : a.summary.objective - b.summary.objective,
  )[0]
  // Resolve through findRun so that, once the chosen sibling's full record is warmed, callers reading
  // its decisionTrace get the FULL copy rather than this lean runsCache entry.
  return findRun(best.key) || best
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
// A reusable collapsible card: a full-width clickable header (chevron + title) over a body that hides when
// collapsed. State lives in xaiCollapsed so a collapse survives the next renderXai().
function xaiCollapsibleCard(id, title, subHtml, bodyHtml, options) {
  const opts = options || {}
  const titleHtml = `<h3>${escapeHtml(title)}${subHtml ? ` <span class="card-sub">\u2014 ${subHtml}</span>` : ''}</h3>`
  // Nothing to show \u2192 a static, non-interactive card (no expander to click into emptiness).
  if (opts.isEmpty) {
    return `<div class="card xai-collapse-card is-empty">
      <div class="xai-collapse-head is-static">${titleHtml}</div>
      <div class="xai-collapse-body is-empty-note">${bodyHtml}</div>
    </div>`
  }
  const collapsed = xaiCollapsed.has(id)
  const summary = opts.collapsedSummary ? `<p class="xai-collapse-summary">${opts.collapsedSummary}</p>` : ''
  return `<div class="card xai-collapse-card${collapsed ? ' is-collapsed' : ''}" data-collapse-card="${escapeHtml(id)}">
    <button type="button" class="xai-collapse-head" data-xai-collapse="${escapeHtml(id)}" aria-expanded="${collapsed ? 'false' : 'true'}">
      <span class="xai-collapse-chevron" aria-hidden="true">\u25be</span>
      <div class="xai-collapse-headtext">${titleHtml}${summary}</div>
    </button>
    <div class="xai-collapse-body">${bodyHtml}</div>
  </div>`
}
// CONFIG EFFECTS tab: two collapsible cards over the same data — Lever importance (model-free screening,
// confounded) and One-factor effect (the controlled, CI-backed contrast).
function xaiConfigEffectsHtml(bundle, criterion) {
  const importances = (bundle.analysis.screening || []).slice()
  if (!importances.length) {
    return `<div class="card"><p class="card-sub">Need \u22652 runs that vary a lever (on the same data) to analyse config effects.</p></div>`
  }
  if (!xaiLever || !importances.some((i) => i.lever === xaiLever)) xaiLever = importances[0].lever
  const leverOpts = importances
    .map(
      (i) =>
        `<option value="${escapeHtml(i.lever)}"${i.lever === xaiLever ? ' selected' : ''}>${escapeHtml(i.lever)}</option>`,
    )
    .join('')
  const contrasts = (bundle.analysis.ofat && bundle.analysis.ofat[xaiLever]) || []
  const contrastHtml = contrasts.length
    ? contrasts.map((c) => xaiOfatContrastHtml(c)).join('')
    : `<p class="card-sub">No clean one-factor contrast for <code>${escapeHtml(xaiLever)}</code> yet \u2014 no runs vary only this lever with everything else fixed. The <strong>Suggested</strong> tab can fill the gap.</p>`
  const importanceBody = xaiImportanceTableHtml(importances)
  const oneFactorBody = `<p class="card-sub"><label>Lever <select id="xai-lever" class="app-select">${leverOpts}</select></label> \u2014 its values compared with <strong>everything else held fixed</strong>.</p>${contrastHtml}`
  const impSummary = importances
    .slice(0, 3)
    .map((i) => `${escapeHtml(xaiAbbrevLever(i.lever))} ${Math.round(i.importance * 100)}%`)
    .join(' \u00b7 ')
  return `${xaiMethodNoteHtml()}
    ${xaiCollapsibleCard('effects-importance', 'Lever importance', `which levers move ${escapeHtml(criterion.label)} most \u2014 model-free screening, confounded`, importanceBody, { collapsedSummary: impSummary })}
    ${xaiCollapsibleCard('effects-onefactor', 'One-factor effect', 'the controlled read \u2014 one lever varied, the rest held fixed', oneFactorBody, { collapsedSummary: `lever: ${escapeHtml(xaiAbbrevLever(xaiLever))}${contrasts.length ? '' : ' \u2014 no clean contrast yet'}` })}`
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
// pairwise coupling, the ablation TREE (worst→best, one change at a time), and a 2-lever interaction
// heatmap — the across-the-whole-space view (predicts unobserved configs). Rendered + the grid marginalised
// live from the server-cached surrogate + setups. Deterministic.
// The one-line \u201cbottom line\u201d distilled from the surrogate: the most decisive lever, which act through
// interactions, which are inert, and how many changes the ablation path needs.
function xaiSurrogateTakeawayHtml(a, criterion) {
  const imp = [...(a.importances || [])].sort((x, y) => (y.total || 0) - (x.total || 0))
  if (!imp.length) return ''
  const top = imp[0]
  const inert = imp.filter((f) => (f.total || 0) < XAI_NEGLIGIBLE_TOTAL).map((f) => f.lever)
  const interactive = imp
    .filter((f) => {
      const i = (f.total || 0) - f.importance
      return i > f.importance && i > 0.05
    })
    .map((f) => f.lever)
  const parts = [
    `<strong>${escapeHtml(xaiAbbrevLever(top.lever))}</strong> moves ${escapeHtml(criterion.label)} most \u2014 ${Math.round((top.total || 0) * 100)}% of the variance.`,
  ]
  if (interactive.length)
    parts.push(
      `${interactive.slice(0, 3).map((l) => `<code>${escapeHtml(xaiAbbrevLever(l))}</code>`).join(', ')} act mainly through interactions \u2014 tune with their partner (see Coupling).`,
    )
  if (inert.length)
    parts.push(
      `${inert.slice(0, 4).map((l) => `<code>${escapeHtml(xaiAbbrevLever(l))}</code>`).join(', ')} ${inert.length === 1 ? 'is' : 'are'} inert \u2014 stop sweeping ${inert.length === 1 ? 'it' : 'them'}.`,
    )
  if (a.ablation && a.ablation.steps && a.ablation.steps.length)
    parts.push(
      `The ablation path reaches the predicted best in ${a.ablation.steps.length} change${a.ablation.steps.length === 1 ? '' : 's'}.`,
    )
  return `<div class="xai-conclusions"><strong class="card-sub">Bottom line</strong><ul class="card-sub">${parts.map((x) => `<li>${x}</li>`).join('')}</ul></div>`
}
function xaiSurrogateHtml(bundle, criterion) {
  const a = bundle.analysis
  const surrogate = a.surrogate
  if (!surrogate || !surrogate.trees.length)
    return `<div class="card"><p class="card-sub">Need \u22652 runs with varying levers to fit a surrogate model.</p></div>`
  const setups = a.setups || []
  const leverNames = surrogate.levers.map((l) => l.name)
  const ranked = a.importances.map((f) => f.lever)
  if (!xaiInterA || !leverNames.includes(xaiInterA)) xaiInterA = ranked[0] || leverNames[0] || null
  if (!xaiInterB || !leverNames.includes(xaiInterB) || xaiInterB === xaiInterA)
    xaiInterB =
      ranked.find((l) => l !== xaiInterA) || leverNames.find((l) => l !== xaiInterA) || null
  // Conditional levers (those with `appliesWhen`) must not be crossed against the values they don't apply to
  // (e.g. forward_horizon × an RL model that ignores it) — pass the map so those cells read 'n/a', not a guess.
  const appliesWhen = {}
  if (manifest && manifest.levers)
    for (const [ln, spec] of Object.entries(manifest.levers))
      if (spec && spec.appliesWhen) appliesWhen[ln] = spec.appliesWhen
  const grid =
    xaiInterA && xaiInterB && setups.length
      ? window.Xai.interactionGrid(surrogate, setups, criterion, xaiInterA, xaiInterB, appliesWhen)
      : null
  const na = 'Not enough data yet \u2014 run more configs.'
  const fanovaBody = xaiFanovaHtml(a.importances)
  const ablationBody = xaiAblationTreeHtml(a.ablation, criterion)
  const couplingBody = xaiCouplingHtml(a.couplings)
  const interBody = xaiInteractionHtml(grid, leverNames, setups)
  const fanovaRanked = [...(a.importances || [])].sort((x, y) => (y.total || 0) - (x.total || 0))
  const strongCoupling = (a.couplings || []).filter((c) => c.strength >= 0.05).length
  const ablSteps = a.ablation && a.ablation.steps ? a.ablation.steps.length : 0
  return `${xaiSurrogateTakeawayHtml(a, criterion)}
    ${xaiCollapsibleCard('surr-fanova', 'Global importance', 'each lever\u2019s share of the variance \u2014 main vs total', fanovaBody || na, { isEmpty: !fanovaBody, collapsedSummary: fanovaRanked.length ? `${fanovaRanked.length} levers \u00b7 top ${escapeHtml(xaiAbbrevLever(fanovaRanked[0].lever))}` : '' })}
    ${xaiCollapsibleCard('surr-ablation', 'Ablation path', 'worst \u2192 best, the single best change at each step', ablationBody || na, { isEmpty: !ablationBody, collapsedSummary: ablSteps ? `${ablSteps} step${ablSteps === 1 ? '' : 's'}` : '' })}
    ${xaiCollapsibleCard('surr-coupling', 'Coupling', 'lever pairs whose best value depends on each other', couplingBody || na, { isEmpty: !couplingBody, collapsedSummary: strongCoupling ? `${strongCoupling} coupled pair${strongCoupling === 1 ? '' : 's'}` : 'levers act independently' })}
    ${xaiCollapsibleCard('surr-interactions', 'Interactions', 'a 2-lever what-if heatmap from the surrogate', interBody || na, { isEmpty: !interBody, collapsedSummary: xaiInterA && xaiInterB ? `${escapeHtml(xaiAbbrevLever(xaiInterA))} \u00d7 ${escapeHtml(xaiAbbrevLever(xaiInterB))}` : '' })}`
}
// Below this TOTAL-effect fraction a lever is effectively inert across the explored range ("stop sweeping").
const XAI_NEGLIGIBLE_TOTAL = 0.03
// Classify a lever by its main vs total effect: inert (drop it), interactive (tune with its partner), or direct.
function fanovaEffectTag(f) {
  if ((f.total || 0) < XAI_NEGLIGIBLE_TOTAL) {
    return {
      label: '✕ inert',
      cls: 'fanova-inert',
      title: 'No measurable effect across the explored range — stop sweeping this lever.',
    }
  }
  const interactive = (f.total || 0) - f.importance
  if (interactive > f.importance && interactive > 0.05) {
    return {
      label: '↔ interactive',
      cls: 'fanova-interactive',
      title:
        'Most of its effect comes through interactions — tune it together with its partner (see Coupling).',
    }
  }
  return {
    label: '→ direct',
    cls: '',
    title: 'Mostly a direct (main) effect, independent of the other levers.',
  }
}
// Order lever values numerically when both look numeric, else alphabetically — so any value series shown is sorted.
function xaiCmpValues(a, b) {
  const na = Number(a)
  const nb = Number(b)
  const aNum = a !== '' && a != null && Number.isFinite(na)
  const bNum = b !== '' && b != null && Number.isFinite(nb)
  if (aNum && bNum) return na - nb
  return String(a).localeCompare(String(b))
}
// Display a lever value: numbers (incl. numeric strings) via the tick formatter; everything else verbatim
// (so categorical levers like model_name render their names, not blank).
function xaiFmtLeverValue(v) {
  if (typeof v === 'number') return formatTickValue(v)
  const n = Number(v)
  if (typeof v === 'string' && v.trim() !== '' && Number.isFinite(n)) return formatTickValue(n)
  return String(v)
}
// Render a lever's distinct values as run-links, sorted; long/many values collapse to #1,#2… with the full
// value on hover so the table stays readable.
function xaiValueLinksHtml(lever, valueList) {
  const sorted = [...(valueList || [])].sort(xaiCmpValues)
  const abbreviate = sorted.length > 6 || sorted.some((v) => String(v).length > 10)
  return sorted
    .map((v, i) => {
      const shown = abbreviate ? `#${i + 1}` : xaiFmtLeverValue(v)
      return `<button type="button" class="link-btn" data-xai-runs data-l1="${escapeHtml(lever)}" data-v1="${escapeHtml(String(v))}" title="${escapeHtml(lever)} = ${escapeHtml(String(v))} — view its runs">${escapeHtml(shown)}</button>`
    })
    .join('<span class="card-sub">, </span>')
}
function xaiFanovaHtml(fanova) {
  if (!fanova.length) return ''
  const rows = fanova
    .map((f) => {
      const vals = xaiValueLinksHtml(f.lever, f.valueList)
      const tag = fanovaEffectTag(f)
      return `<tr><th><code>${escapeHtml(f.lever)}</code></th><td class="num">${Math.round(f.importance * 100)}%</td><td class="num">${Math.round((f.total || 0) * 100)}%</td><td><span class="${tag.cls}" title="${escapeHtml(tag.title)}">${tag.label}</span></td><td>${vals}</td></tr>`
    })
    .join('')
  return `<h4 class="card-sub">fANOVA importance <span class="card-sub">— how much each lever moves the score. Click a value to see its runs.</span></h4>
    <p class="card-sub"><strong>main</strong> = the lever's effect <em>on its own</em> (varying just it, averaging the rest). <strong>total</strong> = main <em>plus</em> everything it does <em>through interactions</em> with other levers. So: <strong>total ≈ main</strong> ⇒ acts independently; <strong>total ≫ main</strong> ⇒ its best value depends on other levers (↔ interactive); <strong>total ≈ 0</strong> ⇒ no measurable effect (✕ inert — stop sweeping it).</p>
    <table class="kv-table report-table"><thead><tr><th>lever</th><th class="num" title="The lever's own first-order effect (Sobol main effect)">main</th><th class="num" title="Main effect + all interaction effects (Sobol total effect)">total</th><th>effect</th><th>values</th></tr></thead><tbody>${rows}</tbody></table>`
}
// The strongly COUPLED lever pairs (one's best value depends on the other) — read alongside the interaction grid.
function xaiCouplingHtml(couplings) {
  const strong = (couplings || []).filter((c) => c.strength >= 0.05).slice(0, 6)
  if (!strong.length) {
    return `<p class="card-sub"><strong>Coupling</strong> — none notable: the levers act independently (their effects add up), so you can tune them one at a time.</p>`
  }
  const items = strong
    .map(
      (c) =>
        `<li><code>${escapeHtml(c.leverA)}</code> × <code>${escapeHtml(c.leverB)}</code> <span class="num">${Math.round(c.strength * 100)}%</span></li>`,
    )
    .join('')
  return `<h4 class="card-sub">Coupling <span class="card-sub">— lever PAIRS whose best value depends on each other (tune together, not one-at-a-time); % = interaction variance explained</span></h4>
    <ul class="xai-coupling">${items}</ul>`
}
function xaiAbbrevLever(name) {
  const s = String(name)
  return s.length > 13 ? s.slice(0, 12) + '…' : s
}
// Parallel-coordinates "configuration map": every explored config is a line threading its lever values,
// coloured by performance rank — readable axes, so you can SEE which lever values the good configs share.
function xaiParallelCoordsHtml(bundle, criterion, focusConfig) {
  const a = bundle.analysis
  const setupVal = (s) =>
    criterion.key === 'objective'
      ? s.objective
      : criterion.key === 'durationMs'
        ? s.durationMs
        : s.metrics && s.metrics[criterion.key]
  const items = (a.setups || [])
    .map((s) => ({ s, v: setupVal(s) }))
    .filter((x) => typeof x.v === 'number' && Number.isFinite(x.v))
  const card = (inner) =>
    `<div class="card"><div class="card-head card-head-row"><h3>Parallel coordinates</h3></div><div class="card-scroll">${inner}</div></div>`
  if (items.length < 3)
    return card(`<p class="card-sub">Need ≥3 explored configs to draw the parallel map.</p>`)
  // Axes = the varying model levers, ordered by fANOVA total effect (most decisive first), capped for legibility.
  const relevant = xaiRelevantLevers(bundle)
  const ranked = [...(a.importances || [])]
    .sort((x, y) => (y.total || 0) - (x.total || 0))
    .map((f) => f.lever)
    .filter((l) => relevant.has(l))
  const ordered = [...ranked, ...(a.levers || []).filter((l) => relevant.has(l) && !ranked.includes(l))]
  const info = {}
  for (const lev of ordered) {
    const vals = items.map((x) => x.s.config[lev])
    const distinct = [...new Set(vals.map((v) => String(v)))]
    if (distinct.length < 2) continue // doesn't vary in this environment → no axis
    const allNum = vals.every(
      (v) => typeof v === 'number' || (v !== '' && v != null && Number.isFinite(Number(v))),
    )
    info[lev] = {
      allNum,
      sorted: distinct.sort(xaiCmpValues),
      nums: allNum ? vals.map(Number) : null,
    }
  }
  const levers = Object.keys(info).slice(0, 8)
  if (levers.length < 2)
    return card(
      `<p class="card-sub">Not enough <em>varying</em> model levers in this environment to draw a parallel map (try a different environment, or run more configs).</p>`,
    )
  // Performance rank → hue (green=best), like the PCA map, so the colour always spans the full range.
  const vs = items.map((x) => x.v)
  const order = vs
    .map((v, i) => [v, i])
    .sort((p, q) => (criterion.direction === 'min' ? q[0] - p[0] : p[0] - q[0]))
  const rank = new Array(vs.length).fill(0.5)
  order.forEach(([, idx], k) => (rank[idx] = vs.length > 1 ? k / (vs.length - 1) : 0.5))
  const W = 660,
    H = 340,
    padL = 28,
    padR = 28,
    padT = 22,
    padB = 66
  const plotW = W - padL - padR,
    plotH = H - padT - padB
  const axisX = (i) => padL + (levers.length === 1 ? plotW / 2 : (i / (levers.length - 1)) * plotW)
  const yOf = (lev, v) => {
    const d = info[lev]
    let t
    if (d.allNum) {
      const lo = Math.min(...d.nums),
        hi = Math.max(...d.nums)
      t = hi === lo ? 0.5 : (Number(v) - lo) / (hi - lo)
    } else {
      const i = d.sorted.indexOf(String(v))
      t = d.sorted.length === 1 ? 0.5 : i / (d.sorted.length - 1)
    }
    return padT + (1 - t) * plotH
  }
  const axes = levers
    .map((lev, i) => {
      const x = axisX(i)
      const d = info[lev]
      const ticks = d.allNum
        ? `<text x="${x}" y="${padT - 6}" text-anchor="middle" class="xai-pc-tick">${escapeHtml(formatTickValue(Math.max(...d.nums)))}</text><text x="${x}" y="${padT + plotH + 13}" text-anchor="middle" class="xai-pc-tick">${escapeHtml(formatTickValue(Math.min(...d.nums)))}</text>`
        : ''
      return `<line x1="${x}" y1="${padT}" x2="${x}" y2="${padT + plotH}" class="xai-pc-axis-line" />${ticks}<text x="${x}" y="${H - padB + 20}" text-anchor="middle" class="xai-pc-axis"><title>${escapeHtml(lev)}</title>${escapeHtml(xaiAbbrevLever(lev))}</text>`
    })
    .join('')
  const compactCfg = (cfg) =>
    levers
      .map((l) => `${l}=${xaiFmtLeverValue(cfg[l])}`)
      .join(' · ')
      .slice(0, 160)
  // Drop any picked lever that isn't an axis here (env/criterion changed) so a stale filter can't dim everything.
  for (const lev of Object.keys(xaiPcPick)) if (!levers.includes(lev)) delete xaiPcPick[lev]
  const anyPick = Object.keys(xaiPcPick).length > 0
  const lineMatches = (cfg) => Object.entries(xaiPcPick).every(([lev, val]) => String(cfg[lev]) === val)
  const polys = items
    .map((x, i) => ({ x, r: rank[i] }))
    .sort((p, q) => p.r - q.r) // draw best (green) last → on top
    .map(({ x, r }) => {
      const pts = levers
        .map((lev, j) => `${axisX(j).toFixed(1)},${yOf(lev, x.s.config[lev]).toFixed(1)}`)
        .join(' ')
      const matched = !anyPick || lineMatches(x.s.config)
      const op = matched ? (0.22 + r * 0.55).toFixed(2) : '0.05'
      const tip = `${compactCfg(x.s.config)} · ${criterion.label} ${formatTickValue(x.v)} · top ${100 - Math.round(r * 100)}%`
      return `<polyline class="xai-pc-line" data-xai-focus-config="${escapeHtml(x.s.key)}" points="${pts}" fill="none" stroke="hsl(${Math.round(r * 120)},70%,45%)" stroke-width="${matched ? '1.4' : '0.8'}" opacity="${op}"><title>${escapeHtml(tip)} · click to open this run</title></polyline>`
    })
    .join('')
  // The focused run (current-run scope): a bold blue halo + white core line drawn last, so it reads as
  // "this run" against the field. Reuses the same click-to-open behaviour as the other lines.
  let focusOverlay = ''
  if (focusConfig) {
    const fi = items.find((x) => xaiSetupMatchesConfig(x.s.config, focusConfig))
    if (fi) {
      const fpts = levers
        .map((lev, j) => `${axisX(j).toFixed(1)},${yOf(lev, fi.s.config[lev]).toFixed(1)}`)
        .join(' ')
      focusOverlay = `<polyline points="${fpts}" fill="none" stroke="#1d4ed8" stroke-width="4.5" opacity="0.85"/><polyline class="xai-pc-line" data-xai-focus-config="${escapeHtml(fi.s.key)}" points="${fpts}" fill="none" stroke="#fff" stroke-width="1.6" opacity="0.95"><title>THIS RUN · ${escapeHtml(compactCfg(fi.s.config))} · click to open</title></polyline>`
    }
  }
  // Clickable value ticks on each low-cardinality axis: click one to keep only the configs at that value.
  const valueMarkers = levers
    .map((lev, i) => {
      const d = info[lev]
      if (d.sorted.length > 10) return ''
      const x = axisX(i)
      return d.sorted
        .map((val) => {
          const y = yOf(lev, d.allNum ? Number(val) : val)
          const picked = xaiPcPick[lev] === String(val)
          return `<circle class="xai-pc-pick${picked ? ' is-picked' : ''}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${picked ? 5 : 3.2}" data-xai-pc-pick="${escapeHtml(lev)}|||${escapeHtml(String(val))}"><title>${escapeHtml(xaiAbbrevLever(lev))}=${escapeHtml(String(val))} — click to filter</title></circle>`
        })
        .join('')
    })
    .join('')
  const matchedCount = anyPick ? items.filter((x) => lineMatches(x.s.config)).length : items.length
  const clearBar = anyPick
    ? `<p class="card-sub">Filtered to <strong>${matchedCount}</strong> of ${items.length} configs (${Object.entries(xaiPcPick)
        .map(([l, v]) => `<code>${escapeHtml(xaiAbbrevLever(l))}=${escapeHtml(v)}</code>`)
        .join(', ')}). <button type="button" class="ghost-btn" data-xai-pc-clear>✕ Clear axis filter</button></p>`
    : ''
  return card(`<p class="card-sub">Every line is one config; <strong style="color:hsl(120,70%,40%)">green = top</strong>, <strong style="color:hsl(0,70%,45%)">red = bottom</strong> by ${escapeHtml(criterion.label)}. <strong>Hover</strong> a line to read its config, <strong>click</strong> it to open that run, or <strong>click a tick</strong> on an axis to keep only the configs at that value. Axes are ordered most-decisive-first (by fANOVA).</p>
    ${clearBar}
    <div class="chart-wrap"><svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" class="xai-pc" role="img" aria-label="Parallel-coordinates configuration map">${axes}${polys}${focusOverlay}${valueMarkers}</svg></div>
    ${xaiParallelConclusionsHtml(items, rank, levers, info, criterion)}`)
}
// PCA "configuration map": a 2-D sketch of the explored setups coloured by performance (green=better).
// Intuition only — the PC axes are lever mixes, not knobs. Reuses the shared scatter builder.
function xaiPcaHtml(bundle, criterion, focusConfig) {
  const pca = bundle.analysis.pca
  if (!pca || pca.points.length < 3)
    return `<div class="card"><div class="card-head card-head-row"><h3>Configuration map (PCA)</h3></div><div class="card-scroll"><p class="card-sub">Need ≥3 explored configs to draw the PCA map.</p></div></div>`
  const values = pca.points.map((p) => p.value)
  // Colour by PERFORMANCE RANK, not raw value: sort points worst→best and map position to a red→green hue.
  // This always uses the full colour range — so a field where most configs perform similarly still shows a
  // clear best/worst spread (raw-value colouring washed out to "all green" when one outlier set the scale).
  const order = values
    .map((v, i) => [v, i])
    .sort((a, b) => (criterion.direction === 'min' ? b[0] - a[0] : a[0] - b[0])) // worst first
  const rankFrac = new Array(values.length).fill(0.5)
  order.forEach(([, idx], k) => {
    rankFrac[idx] = values.length > 1 ? k / (values.length - 1) : 0.5
  })
  // Each PCA point's representative key matches a setup's key (both the first run of the setup group), so
  // we can label points with their config from the bundle's setups.
  const setupByKey = new Map((bundle.analysis.setups || []).map((s) => [s.key, s]))
  const relevant = xaiRelevantLevers(bundle)
  const compactConfig = (cfg) =>
    Object.entries(cfg || {})
      .filter(([k, v]) => relevant.has(k) && String(v) !== 'n/a')
      .map(([k, v]) => `${k}=${xaiFmtLeverValue(v)}`)
      .join(' ')
      .slice(0, 140)
  const colorMode = xaiPcaColor === 'model' ? 'model' : 'rank'
  const modelOf = (i) => {
    const sp = setupByKey.get(pca.points[i].key)
    return sp ? String(sp.config.model_name ?? '?') : '?'
  }
  const models = [...new Set(pca.points.map((_, i) => modelOf(i)))].sort()
  const points = pca.points.map((p, i) => {
    const setup = setupByKey.get(p.key)
    const cfg = setup ? compactConfig(setup.config) : ''
    const seeds = p.runKeys.length > 1 ? ` (${p.runKeys.length} seeds)` : ''
    const pct = Math.round(rankFrac[i] * 100)
    const isFocus = !!(focusConfig && setup && xaiSetupMatchesConfig(setup.config, focusConfig))
    return {
      x: p.x,
      y: p.y,
      focus: isFocus,
      color:
        colorMode === 'model'
          ? `hsl(${xaiModelHue(modelOf(i))}, 62%, 50%)`
          : `hsl(${Math.round(rankFrac[i] * 120)}, 70%, 45%)`,
      label: `${isFocus ? '▶ THIS RUN · ' : ''}${modelOf(i)} · ${cfg ? cfg + ' · ' : ''}${criterion.label} ${formatTickValue(p.value)} · top ${100 - pct}%${seeds}`,
    }
  })
  const ev = (i) => Math.round((pca.explainedVariance[i] || 0) * 100)
  const svg = buildScatterChart({
    points,
    xLabel: `PC1 · ${ev(0)}% var`,
    yLabel: `PC2 · ${ev(1)}% var`,
    width: 480,
    height: 300,
    ariaLabel: 'PCA projection of explored configs coloured by performance rank',
  })
  return `<div class="card"><div class="card-head card-head-row"><h3>Configuration map (PCA) <span class="card-sub">— ${pca.points.length} setups · ${pca.features} features</span></h3></div>
    <div class="card-scroll">
    <p class="card-sub">A 2-D <em>sketch</em> of the explored configs (numeric levers z-scored, categorical one-hot), coloured by <strong>${escapeHtml(criterion.label)} rank</strong> — <strong style="color:hsl(120,70%,40%)">green = top</strong>, <strong style="color:hsl(0,70%,45%)">red = bottom</strong>, evenly across all configs. <strong>How to use it:</strong> nearby points have similar configs, so look for a <em>region where the green points cluster</em> — that's a promising part of the space to explore more (use the recommender). Scattered green with no cluster means performance isn't driven by where you are in this 2-D mix (check the lever importances instead). The PC axes are blends of levers, not knobs — don't read them as directions. PC1+PC2 capture ${ev(0) + ev(1)}% of the configuration variance.</p>
    ${(() => {
      const cb = (m, l) =>
        `<button type="button" class="xai-map-tab${colorMode === m ? ' active' : ''}" data-xai-pca-color="${m}">${l}</button>`
      return `<div class="badges-row" style="gap:.4rem;align-items:center;margin:.2rem 0 .5rem"><span class="card-sub">Colour by</span>${cb('rank', 'Performance')}${cb('model', 'Model')}</div>`
    })()}
    ${
      colorMode === 'model'
        ? `<div class="badges-row" style="gap:.6rem;flex-wrap:wrap;margin-bottom:.4rem">${models
            .map(
              (m) =>
                `<span class="card-sub"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:hsl(${xaiModelHue(m)},62%,50%);margin-right:4px;vertical-align:middle"></span>${escapeHtml(xaiAbbrevLever(m))}</span>`,
            )
            .join('')}</div>`
        : ''
    }
    <div class="chart-wrap">${svg}</div>
    ${colorMode === 'model' ? xaiPcaModelClustersHtml(pca, modelOf) : xaiPcaConclusionsHtml(pca, rankFrac, criterion)}
    </div></div>`
}
// Which models cluster tightly (consistent configs) vs spread across the PCA map (config-sensitive).
function xaiPcaModelClustersHtml(pca, modelOf) {
  const byModel = {}
  pca.points.forEach((p, i) => {
    const m = modelOf(i)
    ;(byModel[m] = byModel[m] || []).push(p)
  })
  const std = (arr, sel) => {
    if (arr.length < 2) return 0
    const mn = arr.reduce((a, p) => a + sel(p), 0) / arr.length
    return Math.sqrt(arr.reduce((a, p) => a + (sel(p) - mn) ** 2, 0) / arr.length)
  }
  const allSpread = Math.hypot(std(pca.points, (p) => p.x), std(pca.points, (p) => p.y)) || 1
  const tight = []
  const spread = []
  for (const [m, ps] of Object.entries(byModel)) {
    if (ps.length < 2) continue
    const sp = Math.hypot(std(ps, (p) => p.x), std(ps, (p) => p.y)) / allSpread
    if (sp < 0.5) tight.push(m)
    else spread.push(m)
  }
  const parts = []
  if (tight.length)
    parts.push(
      `${tight.slice(0, 5).map((m) => `<code>${escapeHtml(xaiAbbrevLever(m))}</code>`).join(', ')} form <strong>tight clusters</strong> \u2014 their configs behave consistently (a clear model signature).`,
    )
  if (spread.length)
    parts.push(
      `${spread.slice(0, 5).map((m) => `<code>${escapeHtml(xaiAbbrevLever(m))}</code>`).join(', ')} are <strong>spread out</strong> \u2014 highly sensitive to their config.`,
    )
  if (!parts.length)
    parts.push('Each model has too few configs to judge clustering \u2014 run more to see model groupings.')
  return `<div class="xai-conclusions"><strong class="card-sub">Model clusters</strong><ul class="card-sub">${parts.map((x) => `<li>${x}</li>`).join('')}</ul></div>`
}
// Deterministic \u201cwhat this map shows\u201d for the parallel map: the lever values the top configs concentrate
// at, and the levers that don't separate good from bad (their lines fan across the axis).
function xaiParallelConclusionsHtml(items, rank, levers, info, criterion) {
  const n = items.length
  let top = items.map((_, i) => i).filter((i) => rank[i] >= 0.67)
  if (top.length < 3)
    top = items
      .map((_, i) => i)
      .sort((x, y) => rank[y] - rank[x])
      .slice(0, Math.max(3, Math.ceil(n / 3)))
  const decisive = []
  const neutral = []
  for (const lev of levers) {
    const counts = {}
    for (const i of top) {
      const v = xaiFmtLeverValue(items[i].s.config[lev])
      counts[v] = (counts[v] || 0) + 1
    }
    const entries = Object.entries(counts).sort((x, y) => y[1] - x[1])
    if (!entries.length) continue
    const frac = entries[0][1] / top.length
    if (frac >= 0.6) decisive.push({ lev, val: entries[0][0], pct: Math.round(frac * 100) })
    else neutral.push(lev)
  }
  const parts = []
  if (decisive.length)
    parts.push(
      `The top ${top.length} configs concentrate at ${decisive
        .slice(0, 4)
        .map((d) => `<code>${escapeHtml(xaiAbbrevLever(d.lev))}=${escapeHtml(String(d.val))}</code> (${d.pct}%)`)
        .join(', ')} \u2014 good starting points.`,
    )
  if (neutral.length)
    parts.push(
      `${neutral
        .slice(0, 5)
        .map((l) => `<code>${escapeHtml(xaiAbbrevLever(l))}</code>`)
        .join(', ')} ${neutral.length === 1 ? "doesn't" : "don't"} separate good from bad here (lines fan across the axis).`,
    )
  if (!decisive.length && !neutral.length)
    parts.push(
      'No axis clearly separates the top configs \u2014 performance is driven by interactions or by factors outside these levers.',
    )
  return `<div class="xai-conclusions"><strong class="card-sub">What this map shows</strong><ul class="card-sub">${parts.map((x) => `<li>${x}</li>`).join('')}</ul></div>`
}
// Deterministic \u201cwhat this map shows\u201d for PCA: whether the top configs cluster in one region, plus how
// faithful the 2-D sketch is (variance captured by PC1+PC2).
function xaiPcaConclusionsHtml(pca, rankFrac, criterion) {
  const pts = pca.points.map((pt, i) => ({ x: pt.x, y: pt.y, r: rankFrac[i] }))
  const top = pts.filter((pt) => pt.r >= 0.67)
  const std = (arr, sel) => {
    if (arr.length < 2) return 0
    const m = arr.reduce((a, pt) => a + sel(pt), 0) / arr.length
    return Math.sqrt(arr.reduce((a, pt) => a + (sel(pt) - m) ** 2, 0) / arr.length)
  }
  const allSpread = Math.hypot(std(pts, (pt) => pt.x), std(pts, (pt) => pt.y))
  const ev2 = Math.round(((pca.explainedVariance[0] || 0) + (pca.explainedVariance[1] || 0)) * 100)
  const parts = []
  if (top.length >= 3 && allSpread > 0) {
    const ratio = Math.hypot(std(top, (pt) => pt.x), std(top, (pt) => pt.y)) / allSpread
    if (ratio < 0.7)
      parts.push(
        `The top configs <strong>cluster</strong> in one region (~${Math.round((1 - ratio) * 100)}% tighter than the field) \u2014 a promising neighbourhood; use <strong>Suggested</strong> to explore around it.`,
      )
    else
      parts.push(
        'The top configs are <strong>scattered</strong>, not clustered \u2014 where a config sits in this 2-D mix doesn\u2019t drive performance; read the lever importances (Surrogate / Config effects) instead.',
      )
  } else parts.push('Too few top configs to judge clustering yet.')
  if (ev2 < 55)
    parts.push(
      `This is a rough sketch \u2014 PC1+PC2 capture only ${ev2}% of the configuration variation, so distances are approximate.`,
    )
  return `<div class="xai-conclusions"><strong class="card-sub">What this map shows</strong><ul class="card-sub">${parts.map((x) => `<li>${x}</li>`).join('')}</ul></div>`
}
// A metric's value on a setup: the criterion 'objective', 'durationMs', or a named entry in metrics.
// Deterministic hue [0,360) from a model name, so each model gets a stable colour in the PCA model view.
function xaiModelHue(name) {
  let h = 0
  const str = String(name)
  for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) % 360
  return h
}
function xaiMetricVal(s, key) {
  return key === 'objective' ? s.objective : key === 'durationMs' ? s.durationMs : s.metrics && s.metrics[key]
}
// Guess a sensible better-direction from a metric's name (risk/drawdown/error \u2192 lower is better).
function xaiInferMetricDir(key) {
  return /draw|risk|loss|vol|cost|\bdd\b|error|mae|rmse|slippage/i.test(String(key)) ? 'min' : 'max'
}
// Does a bundle setup's config match the focused run's config? Setups carry only the model levers (context
// stripped), so compare on the setup's own keys, ignoring seed. Used to ring "this run" on the maps.
function xaiSetupMatchesConfig(setupConfig, focusConfig) {
  if (!setupConfig || !focusConfig) return false
  const keys = Object.keys(setupConfig).filter((k) => k !== 'seed')
  if (!keys.length) return false
  return keys.every((k) => String(setupConfig[k]) === String(focusConfig[k]))
}
// TRADE-OFF (Pareto) map: every config on two (or, with an optional 3rd axis, three) chosen metrics, with
// the non-dominated frontier highlighted. `focusConfig` (current-run scope) rings "this run" on the map.
function xaiParetoHtml(bundle, criterion, focusConfig) {
  const a = bundle.analysis
  const setups = a.setups || []
  const card = (inner) =>
    `<div class="card"><div class="card-head card-head-row"><h3>Trade-off frontier <span class="card-sub">— the best achievable balance across two or three metrics</span></h3></div><div class="card-scroll">${inner}</div></div>`
  const metricKeys = ['objective'].concat(
    Array.from(new Set(setups.flatMap((s) => Object.keys(s.metrics || {})))).sort(),
  )
  if (metricKeys.length < 2 || setups.length < 3)
    return card(
      `<p class="card-sub">Need ≥3 configs with at least two recorded metrics to draw a trade-off frontier.</p>`,
    )
  if (!xaiParetoX || !metricKeys.includes(xaiParetoX)) {
    xaiParetoX = 'objective'
    xaiParetoXDir = criterion.direction
  }
  if (!xaiParetoY || !metricKeys.includes(xaiParetoY) || xaiParetoY === xaiParetoX) {
    xaiParetoY = metricKeys.find((k) => k !== xaiParetoX) || metricKeys[0]
    xaiParetoYDir = xaiInferMetricDir(xaiParetoY)
  }
  // The 3rd axis is active only when it names a real metric distinct from X and Y. Drop a stale/colliding Z
  // outright (not just hide it) so the 2-D/3-D mode always matches the Z control and can't resurrect later.
  if (
    xaiParetoZ &&
    (!metricKeys.includes(xaiParetoZ) || xaiParetoZ === xaiParetoX || xaiParetoZ === xaiParetoY)
  )
    xaiParetoZ = null
  const is3d = !!xaiParetoZ
  const relevant = xaiRelevantLevers(bundle)
  const compactCfg = (cfg) =>
    Object.entries(cfg || {})
      .filter(([k, v]) => relevant.has(k) && String(v) !== 'n/a')
      .map(([k, v]) => `${k}=${xaiFmtLeverValue(v)}`)
      .join(' ')
      .slice(0, 120)
  const dims = is3d ? [xaiParetoX, xaiParetoY, xaiParetoZ] : [xaiParetoX, xaiParetoY]
  const dirs = is3d
    ? [xaiParetoXDir, xaiParetoYDir, xaiParetoZDir]
    : [xaiParetoXDir, xaiParetoYDir]
  const rows = []
  const owner = []
  for (const sp of setups) {
    const vals = dims.map((d) => xaiMetricVal(sp, d))
    if (vals.every((v) => typeof v === 'number' && Number.isFinite(v))) {
      rows.push(vals)
      owner.push(sp)
    }
  }
  const sel = (id, cur) =>
    `<select id="${id}" class="app-select">${metricKeys.map((k) => `<option value="${escapeHtml(k)}"${k === cur ? ' selected' : ''}>${escapeHtml(k)}</option>`).join('')}</select>`
  const zSel = `<select id="xai-pareto-z" class="app-select"><option value="">— none (2-D)</option>${metricKeys
    .filter((k) => k !== xaiParetoX && k !== xaiParetoY)
    .map(
      (k) =>
        `<option value="${escapeHtml(k)}"${k === xaiParetoZ ? ' selected' : ''}>${escapeHtml(k)}</option>`,
    )
    .join('')}</select>`
  const dirSel = (id, cur) =>
    `<select id="${id}" class="app-select"><option value="max"${cur === 'max' ? ' selected' : ''}>higher better</option><option value="min"${cur === 'min' ? ' selected' : ''}>lower better</option></select>`
  const pickers = `<div class="xai-pareto-pickers">
    <label class="card-sub xai-scope-field">X ${sel('xai-pareto-x', xaiParetoX)} ${dirSel('xai-pareto-xdir', xaiParetoXDir)}</label>
    <label class="card-sub xai-scope-field">Y ${sel('xai-pareto-y', xaiParetoY)} ${dirSel('xai-pareto-ydir', xaiParetoYDir)}</label>
    <label class="card-sub xai-scope-field">Z ${zSel} ${is3d ? dirSel('xai-pareto-zdir', xaiParetoZDir) : ''}</label>
  </div>`
  if (rows.length < 3)
    return card(
      `${pickers}<p class="card-sub">Need ≥3 configs with ${is3d ? 'all three' : 'both'} of <code>${escapeHtml(dims.join('</code>, <code>'))}</code> recorded — pick different metrics or run more configs.</p>`,
    )
  const frontier = new Set(window.Xai.paretoFrontier(rows, dirs))
  const focusIdx = focusConfig
    ? owner.findIndex((sp) => xaiSetupMatchesConfig(sp.config, focusConfig))
    : -1
  const dirWord = (d) => (d === 'min' ? 'lower' : 'higher')
  const valStr = (i) => dims.map((d, k) => `${escapeHtml(d)} ${formatTickValue(rows[i][k])}`).join(' · ')
  const labelFor = (i) =>
    `${i === focusIdx ? '▶ THIS RUN · ' : ''}${compactCfg(owner[i].config)} · ${valStr(i)}${frontier.has(i) ? ' · Pareto-optimal' : ''}`
  let svg
  if (is3d) {
    const points = rows.map((r, i) => ({
      x: r[0],
      y: r[1],
      z: r[2],
      color: frontier.has(i) ? 'hsl(140, 70%, 42%)' : 'rgba(127,127,127,0.5)',
      label: labelFor(i),
      key: owner[i].key,
      frontier: frontier.has(i),
      focus: i === focusIdx,
    }))
    svg = buildScatter3dChart({
      points,
      xLabel: `${xaiParetoX} (${dirWord(xaiParetoXDir)})`,
      yLabel: `${xaiParetoY} (${dirWord(xaiParetoYDir)})`,
      zLabel: `${xaiParetoZ} (${dirWord(xaiParetoZDir)})`,
      width: 560,
      height: 440,
      ariaLabel: '3-D Pareto trade-off scatter coloured by frontier membership',
    })
  } else {
    const points = rows.map((r, i) => ({
      x: r[0],
      y: r[1],
      color: frontier.has(i) ? 'hsl(140, 70%, 42%)' : 'rgba(127,127,127,0.45)',
      label: labelFor(i),
      focus: i === focusIdx,
    }))
    svg = buildScatterChart({
      points,
      xLabel: `${xaiParetoX} (${dirWord(xaiParetoXDir)} better)`,
      yLabel: `${xaiParetoY} (${dirWord(xaiParetoYDir)} better)`,
      width: 520,
      height: 320,
      ariaLabel: 'Pareto trade-off scatter coloured by frontier membership',
    })
  }
  const focusLine =
    focusIdx >= 0
      ? `<li><strong style="color:#1d4ed8">This run</strong> ${frontier.has(focusIdx) ? 'is <strong>on the frontier</strong> — nothing beats it across these metrics.' : 'is <strong>dominated</strong> — a frontier config beats it on every axis; compare it to the green configs below.'}</li>`
      : ''
  const concl = `<div class="xai-conclusions"><strong class="card-sub">What this shows</strong><ul class="card-sub">
    <li><strong style="color:hsl(140,70%,38%)">${frontier.size}</strong> of ${rows.length} configs are <strong>Pareto-optimal</strong> (green) — no other config beats them on <em>${is3d ? 'all three' : 'both'}</em> of ${escapeHtml(dims.join(', '))}. Grey points are dominated (something is better on every axis), so never pick them.</li>
    <li>There is no single best — choose along the green frontier by how much of one metric you’ll trade for another.</li>
    ${focusLine}
  </ul></div>`
  const frontierRows = [...frontier]
    .sort((m, n) => (xaiParetoXDir === 'min' ? rows[m][0] - rows[n][0] : rows[n][0] - rows[m][0]))
    .map(
      (i) =>
        `<tr class="xai-lb-row${i === focusIdx ? ' is-focus' : ''}" data-xai-focus-config="${escapeHtml(owner[i].key)}" title="Open this run"><td><code>${compactCfg(owner[i].config)}</code>${i === focusIdx ? ' <span class="card-sub" style="color:#1d4ed8">◀ this run</span>' : ''}</td>${dims.map((d, k) => `<td class="num">${escapeHtml(formatTickValue(rows[i][k]))}</td>`).join('')}</tr>`,
    )
    .join('')
  const frontierTable = `<h4 class="card-sub">The ${frontier.size} Pareto-optimal config${frontier.size === 1 ? '' : 's'} (green) — click one to open it</h4>
    <div class="xai-env-scroll"><table class="runs-table"><thead><tr><th>config</th>${dims.map((d) => `<th class="num">${escapeHtml(d)}</th>`).join('')}</tr></thead><tbody>${frontierRows}</tbody></table></div>`
  return card(
    `${pickers}<p class="card-sub">Each point is a config. <strong style="color:hsl(140,70%,38%)">Green</strong> = on the trade-off frontier (best achievable); grey = dominated.${is3d ? ' Isometric 3-D — drop-lines show each config’s height on the vertical axis.' : ' Add a <strong>Z</strong> metric to make it 3-D.'}${focusIdx >= 0 ? ' <strong style="color:#1d4ed8">This run is ringed in blue.</strong>' : ''}</p><div class="chart-wrap">${svg}</div>${concl}${frontierTable}`,
  )
}
// The levers worth showing: model_name + levers with real impact (fANOVA total >= 0.03, else screening
// importance >= 0.02). scope:'ignore' levers (device) are already stripped by the engine; low-impact noise
// (e.g. gamma) falls below threshold; per-config 'n/a' conditional levers are dropped by the callers.
// 1-2-5 \u201cnice\u201d values within [lo, hi] for a log-friendly numeric sweep (e.g. learning_rate \u2192 5e-5, 1e-4, 2e-4, ...).
function xaiNiceLogValues(lo, hi) {
  const out = []
  if (!(lo > 0) || !(hi > lo)) return out
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, e)
      if (v >= lo * 0.999 && v <= hi * 1.001) out.push(Number(v.toPrecision(4)))
    }
  return out
}
// Sane values to sweep a lever over: its declared choices, else a 1-2-5 (or linear) spread across its range
// (or, with no range, two orders of magnitude around the current value), always including the current value.
function xaiSweepValues(lever, current, observed) {
  const spec = manifest && manifest.levers ? manifest.levers[lever] : null
  const m = new Map()
  const add = (v) => {
    if (v === undefined || v === null || String(v) === 'n/a') return
    if (!m.has(String(v))) m.set(String(v), v)
  }
  if (spec && Array.isArray(spec.choices)) {
    for (const c of spec.choices) add(c)
  } else {
    const cur = Number(current)
    let lo, hi
    if (spec && Array.isArray(spec.range) && spec.range.length === 2) {
      lo = Number(spec.range[0])
      hi = Number(spec.range[1])
    } else if (Number.isFinite(cur) && cur !== 0) {
      lo = Math.abs(cur) / 50
      hi = Math.abs(cur) * 50
    }
    if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) {
      let vals = lo > 0 ? xaiNiceLogValues(lo, hi) : []
      if (!vals.length) for (let k = 0; k <= 6; k++) vals.push(lo + ((hi - lo) * k) / 6)
      if (Number.isInteger(cur) && Number.isInteger(lo) && Number.isInteger(hi))
        vals = [...new Set(vals.map((v) => Math.round(v)))].filter((v) => v >= lo && v <= hi)
      for (const v of vals.slice(0, 12)) add(v)
    }
  }
  for (const v of observed || []) add(v)
  add(typeof current === 'number' ? current : Number.isFinite(Number(current)) ? Number(current) : current)
  return [...m.values()]
}
// Smallest non-negative seed numbers not already used \u2014 for stabilize/explore launches.
function xaiFreshSeeds(used, need) {
  const u = new Set(used || [])
  const out = []
  let v = 0
  while (out.length < Math.max(0, need)) {
    if (!u.has(v)) out.push(v)
    v++
  }
  return out
}
function xaiRelevantLevers(bundle) {
  const a = bundle.analysis
  const set = new Set(['model_name'])
  const fanova = a.importances || []
  if (fanova.length) {
    for (const f of fanova) if ((f.total || 0) >= 0.03) set.add(f.lever)
  } else {
    for (const sc of a.screening || []) if (sc.importance >= 0.02) set.add(sc.lever)
  }
  return set
}
// Top configs by the criterion, with bootstrap CIs — flags which are within seed-noise of #1 (tied) and
// which score implausibly well (a leakage/degenerate guardrail). Both read from the per-setup CI the engine
// now retains.
function xaiLeaderboardHtml(bundle, criterion) {
  const a = bundle.analysis
  const val = (sp) => xaiMetricVal(sp, criterion.key)
  const setups = (a.setups || []).filter((sp) => typeof val(sp) === 'number' && Number.isFinite(val(sp)))
  if (setups.length < 2) return ''
  const sorted = setups.slice().sort((x, y) => (criterion.direction === 'min' ? val(x) - val(y) : val(y) - val(x)))
  const top = sorted.slice(0, 12)
  const bestCi = top[0].ci
  const overlapsBest = (ci) => !!(ci && bestCi && ci[0] <= bestCi[1] && ci[1] >= bestCi[0])
  // direction-aware robust outlier fence for a "too-good-to-be-true" flag.
  const vals = setups.map(val).slice().sort((m, n) => m - n)
  const qAt = (pp) => vals[Math.max(0, Math.min(vals.length - 1, Math.floor(pp * (vals.length - 1))))]
  const iqr = qAt(0.75) - qAt(0.25)
  const suspicious = (v) =>
    iqr > 0 && (criterion.direction === 'min' ? v < qAt(0.25) - 3 * iqr : v > qAt(0.75) + 3 * iqr)
  const relevant = xaiRelevantLevers(bundle)
  const compactCfg = (cfg) =>
    Object.entries(cfg || {})
      .filter(([k, v]) => relevant.has(k) && String(v) !== 'n/a')
      .map(([k, v]) => `${escapeHtml(xaiAbbrevLever(k))}=${escapeHtml(xaiFmtLeverValue(v))}`)
      .join(' ')
      .slice(0, 90)
  const rows = top
    .map((sp, i) => {
      const v = val(sp)
      const tied = i > 0 && overlapsBest(sp.ci)
      const ciStr = sp.ci ? `[${escapeHtml(formatTickValue(sp.ci[0]))}, ${escapeHtml(formatTickValue(sp.ci[1]))}]` : '\u2014'
      const flags = [
        tied ? '<span class="badge">tied #1</span>' : '',
        suspicious(v)
          ? `<button type="button" class="ghost-btn xai-stabilize-btn" data-xai-stabilize="${escapeHtml(sp.key)}" title="Strong outlier \u2014 run more seeds to confirm it is real, not a fluke or leak">\u26a0 verify \u2192 run seeds</button>`
          : '',
      ]
        .filter(Boolean)
        .join(' ')
      return `<tr class="xai-lb-row" data-xai-focus-config="${escapeHtml(sp.key)}" title="Open this run"><td class="num">${i + 1}</td><td><code>${compactCfg(sp.config)}</code></td><td class="num">${escapeHtml(formatTickValue(v))}</td><td class="num card-sub">${ciStr}</td><td class="num">${sp.seeds || 1}</td><td>${flags}</td></tr>`
    })
    .join('')
  let tiedTop = 1
  for (let i = 1; i < top.length; i++) {
    if (overlapsBest(top[i].ci)) tiedTop++
    else break
  }
  const notes = []
  if (tiedTop > 1)
    notes.push(
      `<strong>#1\u2013#${tiedTop} are statistically tied</strong> \u2014 their 95% CIs overlap, so the ranking among them is within seed-noise. Add seeds (Suggested) to separate them.`,
    )
  const suspCount = setups.filter((sp) => suspicious(val(sp))).length
  if (suspCount)
    notes.push(
      `\u26a0 <strong>${suspCount} config${suspCount === 1 ? '' : 's'} score implausibly ${criterion.direction === 'min' ? 'low' : 'high'}</strong> \u2014 verify they are not leaking future information or degenerate (a recurring trap).`,
    )
  const noteHtml = notes.length
    ? `<div class="xai-conclusions"><strong class="card-sub">Read</strong><ul class="card-sub">${notes.map((n) => `<li>${n}</li>`).join('')}</ul></div>`
    : ''
  return `<div class="card"><div class="card-head card-head-row"><h3>Top configs <span class="card-sub">\u2014 best ${escapeHtml(criterion.label)}, with confidence intervals</span></h3></div><div class="card-scroll">
    <table class="kv-table report-table"><thead><tr><th class="num">#</th><th>config</th><th class="num">${escapeHtml(criterion.label)}</th><th class="num">95% CI</th><th class="num">seeds</th><th>flags</th></tr></thead><tbody>${rows}</tbody></table>
    ${noteHtml}</div></div>`
}
// PROGRESS tab: the best-so-far convergence curve over the environment's runs in time order, with a
// plateau read so the user knows whether to keep going or stop.
function xaiConvergenceHtml(bundle, criterion) {
  const a = bundle.analysis
  const conv = a.convergence || []
  const card = (inner) =>
    `<div class="card"><div class="card-head card-head-row"><h3>Search progress <span class="card-sub">\u2014 best ${escapeHtml(criterion.label)} so far vs runs</span></h3></div><div class="card-scroll">${inner}</div></div>`
  if (conv.length < 2)
    return card(
      `<p class="card-sub">Need \u22652 timestamped runs to chart progress${conv.length === 0 ? ' \u2014 these runs may predate run timestamps (re-run to record them)' : ''}.</p>`,
    )
  const points = conv.map((pt) => ({ x: pt.index, y: pt.best }))
  const svg = buildXyChart({
    points,
    line: true,
    xLabel: 'runs',
    yLabel: `best ${criterion.label}`,
    width: 540,
    height: 300,
    ariaLabel: 'Best-so-far convergence over runs',
  })
  const finalBest = conv[conv.length - 1].best
  let sinceImproved = 0
  for (let i = conv.length - 1; i > 0; i--) {
    if (conv[i].best === conv[i - 1].best) sinceImproved++
    else break
  }
  const frac = sinceImproved / conv.length
  const verdict =
    sinceImproved === 0
      ? 'The most recent run set a new best \u2014 the search is <strong>still improving</strong>, keep going.'
      : frac >= 0.3
        ? `No improvement for the last <strong>${sinceImproved}</strong> runs (${Math.round(frac * 100)}% of the search) \u2014 likely <strong>plateaued</strong>; consider stopping or changing the approach.`
        : `No improvement for the last ${sinceImproved} run${sinceImproved === 1 ? '' : 's'} \u2014 watch for a plateau.`
  const concl = `<div class="xai-conclusions"><strong class="card-sub">When to stop</strong><ul class="card-sub">
    <li>Best ${escapeHtml(criterion.label)} so far: <strong>${escapeHtml(formatTickValue(finalBest))}</strong> after ${conv.length} runs.</li>
    <li>${verdict}</li>
  </ul></div>`
  return card(
    `<p class="card-sub">Each step up is a new best config found as you ran more experiments (in time order). A long flat tail means you\u2019ve likely converged.</p><div class="chart-wrap">${svg}</div>${concl}`,
  )
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
// The 2-lever interaction heatmap: the surrogate's predicted criterion for every (leverA × leverB) cell,
// marginalised over the explored setups. Cells with a real explored config link to those runs.
function xaiInteractionHtml(grid, leverNames, setups) {
  if (leverNames.length < 2) return ''
  const sel = (id, current) =>
    `<select id="${id}" class="app-select">${leverNames.map((l) => `<option value="${escapeHtml(l)}"${l === current ? ' selected' : ''}>${escapeHtml(l)}</option>`).join('')}</select>`
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
  // Does an EXPLORED config sit at this (leverA=a, leverB=b) cell? Only those cells link to the Runs tab;
  // a surrogate-PREDICTED cell with no explored config stays plain (nothing to open).
  const explored = (a, b) =>
    (setups || []).some(
      (s) =>
        String((s.config || {})[grid.leverA]) === String(a) &&
        String((s.config || {})[grid.leverB]) === String(b),
    )
  // Sort both axes (numeric or alphabetic) via an index permutation, remapping cells to match.
  const aIdx = grid.valuesA
    .map((_, i) => i)
    .sort((i, j) => xaiCmpValues(grid.valuesA[i], grid.valuesA[j]))
  const bIdx = grid.valuesB
    .map((_, i) => i)
    .sort((i, j) => xaiCmpValues(grid.valuesB[i], grid.valuesB[j]))
  const valuesA = aIdx.map((i) => grid.valuesA[i])
  const valuesB = bIdx.map((i) => grid.valuesB[i])
  const cellAt = (ai, bj) => grid.cells[aIdx[ai] * grid.valuesB.length + bIdx[bj]]
  const head = `<tr><th></th>${valuesB.map((b) => `<th class="num">${escapeHtml(xaiFmtLeverValue(b))}</th>`).join('')}</tr>`
  const body = valuesA
    .map((a, i) => {
      const cells = valuesB
        .map((b, j) => {
          const v = cellAt(i, j)
          const where = `${escapeHtml(grid.leverA)}=${escapeHtml(String(a))}, ${escapeHtml(grid.leverB)}=${escapeHtml(String(b))}`
          if (v == null) {
            return `<td class="num xai-cell-na" title="${where} — these levers don\u2019t apply together (a conditional lever is inert here)">n/a</td>`
          }
          if (!explored(a, b)) {
            return `<td class="num" style="${shade(v)}" title="${where} (surrogate-predicted; not yet run)">${escapeHtml(formatTickValue(v))}</td>`
          }
          return `<td class="num xai-cell-link" style="${shade(v)}" data-xai-runs data-l1="${escapeHtml(grid.leverA)}" data-v1="${escapeHtml(String(a))}" data-l2="${escapeHtml(grid.leverB)}" data-v2="${escapeHtml(String(b))}" title="${where} — view the runs here in the Runs tab">${escapeHtml(formatTickValue(v))}</td>`
        })
        .join('')
      return `<tr><th>${escapeHtml(xaiFmtLeverValue(a))}</th>${cells}</tr>`
    })
    .join('')
  return `${picker}<div class="compare-table-wrap"><table class="kv-table report-table xai-heat"><thead>${head}</thead><tbody>${body}</tbody></table></div>
    <p class="card-sub">cells = surrogate-predicted ${escapeHtml(grid.leverA)} (rows) × ${escapeHtml(grid.leverB)} (cols); greener = higher predicted value; <strong>n/a</strong> cells are combinations that can\u2019t occur (a conditional lever doesn\u2019t apply there). Click an explored cell to see its runs.</p>`
}
// The "Propose with AI" button — shared by the empty + populated recommender. Drives the
// propose-experiments activity; results land as runnable ✦ AI suggestions in THIS list.
function xaiProposeButtonHtml() {
  const disabled = proposingExperiments || !embedded() ? ' disabled' : ''
  return `<button type="button" class="ghost-btn" data-xai-propose${disabled} title="Ask the AI to propose NEW experiments beyond the explored grid — they appear here as runnable suggestions">${proposingExperiments ? `${spinnerHtml()} Proposing…` : 'Propose with AI'}</button>`
}
// How many runs an LLM suggestion's spec expands to (sweep cartesian × seeds), for the run-count chip.
function xaiSpecRunCount(spec) {
  const s = spec || {}
  const seeds = Array.isArray(s.seeds) && s.seeds.length ? s.seeds.length : 1
  let combos = 1
  for (const v of Object.values(s.sweep || {})) if (Array.isArray(v) && v.length) combos *= v.length
  return combos * seeds
}
function xaiSuggestionToRec(sug) {
  return {
    kind: 'llm',
    reason: sug.rationale ? `${sug.title} — ${sug.rationale}` : sug.title,
    runCount: xaiSpecRunCount(sug.spec),
    spec: sug.spec,
    priority: 100,
  }
}
// Merge LLM suggestions (first) with the bundle's deterministic recommendations (computed off the
// whole-space surrogate over EVERY run), deduped by spec so an AI pick that coincides with a grid gap
// shows once.
function xaiBuildRecs(bundle) {
  const merged = []
  const seen = new Set()
  for (const r of [
    ...xaiSuggestionsCache.map(xaiSuggestionToRec),
    ...bundle.analysis.recommendations,
  ]) {
    const k = xaiSpecKey(r.spec)
    if (seen.has(k)) continue
    seen.add(k)
    merged.push(r)
  }
  return merged
}
// The Suggested view's header controls (Propose + Run-all) — concurrency is set in Activity, not here.
function xaiRecommenderControlsHtml(bundle) {
  const recs = xaiBuildRecs(bundle)
  const total = recs.reduce((a, r) => a + r.runCount, 0)
  const runAll = recs.length
    ? `<button type="button" class="ghost-btn" data-xai-run-all>Run all (${total})</button>`
    : ''
  return `${xaiProposeButtonHtml()}${runAll}`
}
function xaiRecommenderHtml(criterion, bundle) {
  xaiRecsCache = xaiBuildRecs(bundle)
  if (!xaiRecsCache.length) {
    return `<div class="card" id="xai-recommender"><p class="card-sub">No deterministic gaps — the grids you've explored are complete and well-seeded. 🎉 Use <strong>Propose</strong> (above) to ask the AI for experiments beyond the explored grid.</p></div>`
  }
  const total = xaiRecsCache.reduce((a, r) => a + r.runCount, 0)
  const climbs = xaiRecsCache.filter((r) => r.kind === 'acquisition').length
  const llms = xaiRecsCache.filter((r) => r.kind === 'llm').length
  const cards = xaiRecsCache.map((r, i) => xaiRecCardHtml(r, i)).join('')
  return `<div class="card" id="xai-recommender">
    <p class="card-sub"><strong>${xaiRecsCache.length} suggestions · ${total} runs.</strong> ${climbs ? `<strong>▲ climb</strong> picks are the surrogate's highest Expected-Improvement unrun configs — the next steps toward the optimum. ` : ''}${llms ? `<strong>✦ AI</strong> picks are model-proposed experiments beyond the grid. ` : ''}<strong>seeds</strong> firm up a thin top setup; <strong>gap</strong>/<strong>pair</strong> fill untested factorial cells.</p>
    ${cards}</div>`
}
// Friendly badge labels for each recommendation kind (the raw kind is kept as the badge's tooltip).
const REC_KIND_LABELS = {
  acquisition: '▲ climb',
  llm: '✦ AI',
  'thin-seeds': 'seeds',
  'missing-cell': 'gap',
  interaction: 'pair',
}
function xaiRecCardHtml(r, i) {
  const label = REC_KIND_LABELS[r.kind] || r.kind
  const cls =
    r.kind === 'acquisition'
      ? 'badge xai-rec-climb'
      : r.kind === 'llm'
        ? 'badge xai-rec-llm'
        : 'badge'
  const launched = xaiLaunchedSpecs.has(xaiSpecKey(r.spec))
  const action = launched
    ? `<button type="button" class="ghost-btn" data-xai-view-activity title="This batch is queued or running">View in Activity →</button>`
    : `<button type="button" class="ghost-btn" data-xai-run-rec="${i}">Run batch</button>`
  return `<div class="xai-rec badges-row">
    <span class="${cls}" title="${escapeHtml(r.kind)}">${escapeHtml(label)}</span>
    <span>${escapeHtml(r.reason)}</span>
    <span class="card-sub">${r.runCount} run${r.runCount === 1 ? '' : 's'}</span>
    ${action}
  </div>`
}
// A content key for a spec covering BOTH a single cell (fixed) and an LLM sweep, so the launch lock + the
// merge dedup distinguish sweep suggestions from each other and from grid cells.
function xaiSpecKey(spec) {
  const s = spec || {}
  return `${window.Xai.canonicalConfigString(s.fixed || {})}#${window.Xai.canonicalConfigString(s.sweep || {})}`
}
async function xaiLaunchBatch(specs, label) {
  const concurrency = savedConcurrency()
  // Skip any batch already launched this session (its button is locked) so Run-all never double-fires.
  const pending = specs.filter((spec) => !xaiLaunchedSpecs.has(xaiSpecKey(spec)))
  if (!pending.length) {
    showTab('activity')
    return
  }
  try {
    for (const spec of pending) {
      await startOrEnqueue('train', trainerComputeParams({ spec, concurrency }), label)
      xaiLaunchedSpecs.add(xaiSpecKey(spec)) // lock its button + Run-all until its runs land
    }
    if (activeTabId === 'xai') renderXai() // reflect the now-locked buttons
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
// Load the LLM-proposed experiment suggestions so the recommender can surface them as runnable batches.
async function loadXaiSuggestions() {
  if (!manifest) {
    xaiSuggestionsCache = []
    return
  }
  const recs = await queryRecords(`${manifest.recordType}-xai-suggestion`)
  xaiSuggestionsCache = (recs || [])
    .map((r) => r.content)
    .filter((c) => c && c.spec && (c.spec.fixed || c.spec.sweep))
}
// Load the cached whole-space analysis bundles (one per criterion) into the cache, keyed by criterion.
async function loadXaiConfigSpace() {
  xaiConfigSpaceCache = new Map()
  if (!manifest) return
  const recs = await queryRecords(`${manifest.recordType}-config-space`)
  for (const r of recs || []) {
    const c = r.content
    if (c && c.criterion && c.criterion.key) xaiConfigSpaceCache.set(c.criterion.key, c)
  }
}
// The server-cached whole-space bundle for a criterion (null if none computed yet). Carries the run count +
// when it was computed, so the header can say "analysed N runs · <when>".
function xaiResolveBundle(criterion) {
  const cached = xaiConfigSpaceCache.get(criterion.key)
  if (cached && cached.analysis) {
    return {
      analysis: cached.analysis,
      cachedRunCount: cached.runCount || 0,
      generatedAt: cached.generatedAt,
    }
  }
  return null
}
// Load the LRU-cached per-run xAI digests ({recordType}-run-xai records) into the cache, keyed by run key.
async function loadXaiRunAnalyses() {
  xaiRunAnalysisCache = new Map()
  if (!manifest) return
  const recs = await queryRecords(`${manifest.recordType}-run-xai`)
  for (const r of recs || []) {
    const c = r.content
    if (c && c.runKey) xaiRunAnalysisCache.set(c.runKey, c)
  }
}
async function refreshXai() {
  if (xaiFocusKey) await loadXaiNarrative(xaiFocusKey)
  await loadXaiSuggestions()
  await loadXaiConfigSpace()
  await loadXaiRunAnalyses()
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
  if (proposingExperiments) {
    showTab('activity')
    return
  }
  setStatusLine('xai-status', '')
  try {
    // Lands the proposals as runnable suggestions in THIS tab's recommender (not as hypotheses); the
    // observe loop reloads them on settle. Stay on the xAI tab.
    const result = await startOrEnqueue(
      'propose-experiments',
      trainerActivityParams({ instructions: xaiProposeInstructions() }),
      'Propose experiments',
    )
    if (result.queued) setStatusLine('xai-status', queuedStatusText(result.ahead))
    else setStatusLine('xai-status', 'Proposing experiments — they’ll appear below as suggestions.')
  } catch {
    setStatusLine('xai-status', 'Could not start proposing — please try again.', true)
  }
}
// Re-scope the whole-space analysis to a chosen environment (by its signature) — looks up its context
// values from the cached bundle and triggers a fresh, environment-scoped analysis.
function xaiAnalyzeEnvironment(signature) {
  const bundle = xaiResolveBundle(currentXaiCriterion())
  const env = bundle && (bundle.analysis.environments || []).find((e) => e.signature === signature)
  if (env) onXaiRefreshAnalysisClick(env.values)
}
async function onXaiRefreshAnalysisClick(environment) {
  if (!embedded()) {
    setStatusLine(
      'xai-status',
      'Open inside the Overseer to compute the whole-space analysis.',
      false,
    )
    return
  }
  if (analyzingConfigSpace) return
  const criterion = currentXaiCriterion()
  setStatusLine('xai-status', '')
  try {
    const result = await startOrEnqueue(
      'config-space-analyze',
      trainerActivityParams({
        criterionKey: criterion.key,
        criterionDir: criterion.direction,
        // The environment to scope within (its context-lever values); omit ⇒ the most-run environment.
        environment: environment || undefined,
      }),
      'Analyse config space',
    )
    if (result.queued) setStatusLine('xai-status', queuedStatusText(result.ahead))
  } catch {
    setStatusLine('xai-status', 'Could not start the analysis — please try again.', true)
  }
}
async function onXaiAnalyzeRunClick(runKey) {
  if (!runKey || analyzingRunKey) return
  if (!embedded()) {
    setStatusLine('xai-status', 'Open inside the Overseer to analyse a run.', false)
    return
  }
  const criterion = currentXaiCriterion()
  const sibling = xaiBestSibling(findRun(runKey))
  analyzingRunKey = runKey
  renderXai()
  setStatusLine('xai-status', '')
  try {
    const result = await startOrEnqueue(
      'run-xai-analyze',
      trainerActivityParams({
        runKey,
        siblingKey: sibling ? sibling.key : undefined,
        criterionKey: criterion.key,
        criterionDir: criterion.direction,
      }),
      `Analyse run ${shortKey(runKey)}`,
    )
    if (result.queued) setStatusLine('xai-status', queuedStatusText(result.ahead))
  } catch {
    analyzingRunKey = null
    renderXai()
    setStatusLine('xai-status', 'Could not start the analysis — please try again.', true)
  }
}
// Open the xAI tab focused on ONE run (the current-run scope) — the run-detail entry point.
function analyzeInXai(key) {
  xaiFocusKey = key
  rearmRunFull(key) // an explicit re-focus retries a previously-failed full-fetch
  xaiScope = 'current'
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
  // The list copy is lean (heavy chart/trace fields omitted); pull the full record and re-render the
  // detail once it arrives so the price/equity/explain charts populate.
  warmRunsForRender([key], () => {
    if (selectedRunKey === key) renderRunDetail(key)
  })
  const s = run.summary
  const failed = s.status === 'failed'
  const invalid = s.status === 'invalid'
  const degenerate =
    !failed && !invalid && !!(s.health && s.health.status && s.health.status !== 'ok')
  // Verdict only makes sense for a healthy run (degenerate auto-rejects; failed has no result; an INVALID
  // run was produced by a since-fixed bug and must not count toward anything). Evaluation needs that AND a
  // project that supports eval at all (not RL).
  const showVerdict = !failed && !invalid && !degenerate
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
  const headline = invalid
    ? `<span class="badge is-bad" title="${escapeHtml(s.invalidReason || 'invalid')}">invalid</span>${s.invalidReason ? ` <span class="card-sub">${escapeHtml(s.invalidReason)}</span>` : ''}`
    : failed
      ? '<span class="badge is-bad">failed</span>'
      : `${escapeHtml(objectiveName())} ${escapeHtml(formatObjective(s.objective))} · ${healthBadgeHtml(s.health)}`
  const html = `
    <div class="card-head card-head-row">
      <div>
        <h2>Run <code>${escapeHtml(shortKey(run.key))}</code>${s.config && s.config.model_name ? ` · <span class="run-detail-model">${escapeHtml(String(s.config.model_name))}</span>` : ''}</h2>
        <p class="card-sub">${headline} · seed ${escapeHtml(s.seed === undefined ? '—' : String(s.seed))}
          · ${escapeHtml(formatWhen(runRanAt(s)))}${datasetBadge ? ` · ${datasetBadge}` : ''}${versionBit}${datasetNameBit}${envBit}${unrunnableBadge}</p>
      </div>
      <div class="head-actions">
        ${embedded() && outdated ? `<button type="button" data-action="rerun" data-key="${escapeHtml(run.key)}" class="ghost-btn" title="Re-run this exact config under the current pipeline version (v${escapeHtml(String(currentPipelineVersion()))}) — a breaking version has landed since, so its scores are outdated">↻ Re-run with latest version</button>` : ''}
        <button type="button" data-action="clone" data-key="${escapeHtml(run.key)}" class="icon-btn" title="Clone to Launch" aria-label="Clone to Launch">⧉</button>
        <button type="button" data-action="xai" data-key="${escapeHtml(run.key)}" class="icon-btn" title="Analyze in xAI — internals, config effects, suggested experiments" aria-label="Analyze in xAI">🔬</button>
        <button type="button" data-action="toggle-favorite" data-key="${escapeHtml(run.key)}" class="icon-btn${favoritesCache.has(run.key) ? ' is-fav' : ''}" title="${favoritesCache.has(run.key) ? 'Remove from favorites' : 'Mark as favorite — quick-pick it in xAI'}" aria-label="Toggle favorite">${favoritesCache.has(run.key) ? '★' : '☆'}</button>
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
  rearmRunFull(key) // an explicit re-open retries a previously-failed full-fetch
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
// The lever-only config to re-run for a stored run — the planner's expected shape, used as one entry in
// the batch re-run's spec.configs.
function reRunConfigForRun(config) {
  const cfg = config || {}
  const leverKeys = new Set(leverEntries().map(([k]) => k))
  const fixed = {}
  for (const [k, v] of Object.entries(cfg)) {
    if (leverKeys.has(k) && v !== undefined) fixed[k] = v
  }
  return fixed
}
// Queue the exact same run(s) again to be tried, as ONE campaign/activity (a spec.configs batch, like a
// Launch sweep) — never one activity per run. refresh=true so an existing (e.g. failed/outdated) result
// doesn't skip them. Each config — and therefore its hash + record key — matches the original, so a re-run
// UPDATES that run in place (it never creates a new record) and the fresh result carries the current
// pipelineVersion. Stays put (a toast confirms) rather than yanking the user to Activity.
async function reRunRuns(keys) {
  if (!embedded()) return
  const runs = keys.map((k) => findRun(k)).filter(Boolean)
  if (!runs.length) return
  const params = trainerComputeParams({
    spec: {
      configs: runs.map((run) => ({ config: reRunConfigForRun(run.summary.config), key: run.key })),
    },
    refresh: true,
    concurrency: savedConcurrency(),
  })
  const label = runs.length === 1 ? `Re-run ${shortKey(runs[0].key)}` : `Re-run ${runs.length} runs`
  await startOrEnqueue('train', params, label)
  showToast(
    `Queued ${runs.length} run${runs.length === 1 ? '' : 's'} to re-run as one activity — see the Activity tab.`,
  )
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
  const run = await ensureRunFull(key)
  if (!run || !chatAboutRunAvailable()) return
  const s = run.summary
  const failed = s.status === 'failed'
  const invalid = s.status === 'invalid'
  const logTail = Array.isArray(s.logTail) ? s.logTail.slice(-40).join('\n') : ''
  const verdict = verdictsCache.get(key)
  const runContext = [
    `You are discussing ONE specific, already-selected training run — run id "${shortKey(key)}"${failed ? ', which FAILED' : ''}${invalid ? `, which is INVALID${s.invalidReason ? ` (${s.invalidReason})` : ''} and must NOT be treated as a real result` : ''}. Its full configuration, metrics${verdict && verdict.why ? ', judge verdict' : ''}${failed ? ', error and recent logs' : ''} are all given below, so do NOT ask the user for the run id or any of these details — work directly from what follows.`,
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
  await Promise.all([ensureRunFull(keyA), ensureRunFull(keyB)])
  const a = findRun(keyA)
  const b = findRun(keyB)
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
function xaiRunChatSummary(run, presuppliedSibling) {
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
  // Use the sibling the caller already warmed (so its decision trace is the full record), falling back
  // to a fresh pick only when none was supplied.
  const sibling = presuppliedSibling || xaiBestSibling(run)
  if (sibling) {
    const dd = decisionDiffChatSummary(sibling.summary, s)
    if (dd) parts.push(`Vs the nearest comparable run ${shortKey(sibling.key)} — ${dd}`)
  }
  return parts.join('\n\n')
}
// Discuss the FULL xAI analysis of one run (the xAI-tab entry point — richer than the Runs-detail chat).
async function chatAboutRunXai(key) {
  const run = await ensureRunFull(key)
  if (!run || !chatAboutRunAvailable()) return
  // xaiRunChatSummary reads the nearest sibling's decision trace too — warm it, then re-resolve so the
  // sibling we hand it is the FULL record (and the same one we warmed, not a re-picked lean candidate).
  const leanSibling = xaiBestSibling(run)
  if (leanSibling) await ensureRunFull(leanSibling.key)
  const sibling = leanSibling ? findRun(leanSibling.key) : null
  const s = run.summary
  const ctx = [
    `You are discussing the FULL xAI analysis of ONE training run — id "${shortKey(key)}". Everything below is provided (config, metrics, the decision trace, input attribution + its faithfulness check, the reward breakdown, the latent representation + probe, the run's standing + lever importances among all runs, and the decision diff vs its nearest comparable run), so work directly from it — don't ask for the run id or these details. The xAI computations are DETERMINISTIC + heuristic; treat the attribution and decision-quality reads as EVIDENCE, not proof, and say so when a signal is weak (e.g. an attribution that FAILED its sanity check).`,
    `Pipeline v${String(s.pipelineVersion || '1')}. Objective (${objectiveName()}): ${formatObjective(s.objective)} · health: ${(s.health && s.health.status) || 'unknown'}.`,
    s.metrics ? `Metrics:\n${JSON.stringify(s.metrics, null, 2)}` : '',
    `Config:\n${JSON.stringify(s.config || {}, null, 2)}`,
    xaiRunChatSummary(run, sibling),
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
// Discuss the CURRENT xAI view with the AI — a topic-chat whose system prompt is preloaded with exactly the
// data the user is looking at (per view), so the agent reasons from it without the user copying anything.
async function xaiDiscussView(viewId) {
  if (!chatAboutRunAvailable()) return
  if (xaiScope === 'current') {
    if (xaiFocusKey) await chatAboutRunXai(xaiFocusKey) // the focused run's full xAI covers standing/narrative/internals
    return
  }
  const criterion = currentXaiCriterion()
  const bundle = xaiResolveBundle(criterion)
  if (!bundle) return
  const a = bundle.analysis
  const envLabel = a.environment
    ? xaiEnvLabel(a.environment)
    : 'the whole space (no environment levers)'
  const J = (x) => JSON.stringify(x, null, 2)
  const setupVal = (s) =>
    criterion.key === 'objective'
      ? s.objective
      : criterion.key === 'durationMs'
        ? s.durationMs
        : s.metrics && s.metrics[criterion.key]
  let label, seed, ctx
  if (viewId === 'environments') {
    label = 'Environments'
    seed =
      'Which environment looks most promising, and how much do the environment/dataset settings actually matter?'
    ctx = `The user is comparing ENVIRONMENTS (each = a fixed combo of market-mechanics + data; analysed separately, never tuned).\nEnvironments (best ${criterion.label} + run count):\n${J((a.environments || []).slice(0, 40).map((e) => ({ env: xaiEnvLabel(e.values), runs: e.runCount, best: e.best })))}\nHow much each context (env/dataset) lever moves the score:\n${J((a.contextImportances || []).map((s) => ({ lever: s.lever, importance: s.importance, best: s.bestValue })))}`
  } else if (viewId === 'effects') {
    label = 'Config effects'
    seed = 'Which model levers matter most here and which values should I prefer?'
    ctx = `Scoped to environment: ${envLabel}.\nLever-importance screening (model-free, confounded):\n${J((a.screening || []).map((s) => ({ lever: s.lever, importance: s.importance, best: s.bestValue, worst: s.worstValue, minRuns: s.minRuns, confident: s.confident })))}`
  } else if (viewId === 'surrogate') {
    label = 'Surrogate'
    seed =
      'Read the surrogate — which levers matter, what interacts, and what the ablation path suggests?'
    ctx = `Scoped to ${envLabel}.\nfANOVA importances (main vs total; total≫main ⇒ interactive, total≈0 ⇒ inert):\n${J((a.importances || []).map((f) => ({ lever: f.lever, main: f.importance, total: f.total })))}\nStrong couplings (lever pairs whose best value depends on each other):\n${J((a.couplings || []).slice(0, 8))}\nAblation path (worst→best, best single change each step):\n${a.ablation ? J(a.ablation) : 'none (too few runs for an ablation path)'}`
  } else if (viewId === 'maps') {
    label = 'Maps'
    seed = 'What does the configuration map tell me about which configs are good?'
    ctx = `Scoped to ${envLabel}. The user is viewing the ${xaiMapKind === 'pca' ? 'PCA cluster' : 'parallel-coordinates'} map of ${a.setupCount} distinct configs, coloured by ${criterion.label}.\nThe configs (config → ${criterion.label}):\n${J((a.setups || []).slice(0, 60).map((s) => ({ config: s.config, value: setupVal(s) })))}`
  } else if (viewId === 'recommender') {
    label = 'Suggested'
    seed = 'Which of these suggested experiments should I run first, and why?'
    ctx = `Scoped to ${envLabel}. Suggested next experiments:\n${J((a.recommendations || []).map((r) => ({ kind: r.kind, reason: r.reason, spec: r.spec })))}`
  } else return
  const systemPrompt = [
    projectChatPreamble(),
    `You are discussing the xAI "${label}" view of the DETERMINISTIC whole-space analysis for objective ${objectiveName()} / criterion "${criterion.label}" (better when ${criterion.direction === 'min' ? 'lower' : 'higher'}). Everything the user sees is given below — work directly from it, don't ask for it. These reads are heuristic; flag weak signals (e.g. low run counts) honestly.`,
    ctx,
  ]
    .filter(Boolean)
    .join('\n\n')
  try {
    await window.OverseerBridge.discussTopic({
      title: `xAI ${label}${a.environment ? ` · ${xaiEnvLabel(a.environment).slice(0, 40)}` : ''}`,
      seed,
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
    // Only bail on 3 CONSECUTIVE misses (the activity genuinely vanished) — a transient null mid-run (e.g.
    // a just-queued activity not yet visible) must not accumulate into a spurious "did not settle".
    if (act) missing = 0
    else if (++missing >= 3) return null
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
      if (event.target.id === 'runs-status-filter') {
        runsStatusFilter = event.target.value
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
      const favBtn = event.target.closest('button[data-action="toggle-favorite"]')
      if (favBtn) toggleFavorite(favBtn.dataset.key)
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
        closeConsolidateModal()
        closeDeviceTimingsModal()
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
      const downloadBtn = event.target.closest('#compare-download')
      if (downloadBtn) {
        downloadRunsAudit([...runsCompareKeys], downloadBtn)
      }
      if (
        event.target.closest('#compare-rerun-all') ||
        event.target.closest('#compare-rerun-latest')
      ) {
        reRunRuns([...runsCompareKeys])
      }
    })
  }
  const xaiBody = byId('xai-body')
  if (xaiBody) {
    xaiBody.addEventListener('change', (event) => {
      const t = event.target
      if (t.id === 'xai-favorite-pick') {
        if (t.value) {
          xaiFocusKey = t.value
          xaiScope = 'current'
          xaiTab = 'standing'
          renderXai()
        }
      } else if (t.id === 'xai-env-switch') {
        xaiAnalyzeEnvironment(t.value)
      } else if (t.id === 'xai-pareto-x') {
        xaiParetoX = t.value
        renderXai()
      } else if (t.id === 'xai-pareto-y') {
        xaiParetoY = t.value
        renderXai()
      } else if (t.id === 'xai-pareto-xdir') {
        xaiParetoXDir = t.value === 'min' ? 'min' : 'max'
        renderXai()
      } else if (t.id === 'xai-pareto-ydir') {
        xaiParetoYDir = t.value === 'min' ? 'min' : 'max'
        renderXai()
      } else if (t.id === 'xai-pareto-z') {
        xaiParetoZ = t.value || null
        if (xaiParetoZ) xaiParetoZDir = xaiInferMetricDir(xaiParetoZ)
        renderXai()
      } else if (t.id === 'xai-pareto-zdir') {
        xaiParetoZDir = t.value === 'min' ? 'min' : 'max'
        renderXai()
      } else if (t.id === 'xai-criterion') {
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
    // Live-filter the environment table in place (no re-render, so the search box keeps focus).
    xaiBody.addEventListener('input', (event) => {
      if (event.target.id !== 'xai-env-search') return
      xaiEnvSearch = event.target.value
      const q = xaiEnvSearch.trim().toLowerCase()
      let shown = 0
      xaiBody.querySelectorAll('.xai-env-scroll tbody tr[data-xai-env]').forEach((tr) => {
        const match = !q || tr.textContent.toLowerCase().includes(q)
        tr.hidden = !match
        if (match) shown++
      })
      const total = (xaiResolveBundle(currentXaiCriterion())?.analysis.environments || []).length
      const count = xaiBody.querySelector('.xai-env-controls .card-sub:last-child')
      if (count) count.textContent = `${shown} of ${total} · click a header to sort`
    })
    xaiBody.addEventListener('click', (event) => {
      const runsLink = event.target.closest('[data-xai-runs]')
      if (runsLink) {
        const d = runsLink.dataset
        const pairs = [[d.l1, d.v1]]
        let label = `${d.l1} = ${d.v1}`
        if (d.l2 != null) {
          pairs.push([d.l2, d.v2])
          label = `${d.l1}=${d.v1} × ${d.l2}=${d.v2}`
        }
        viewRunsByLeverValues(pairs, label)
        return
      }
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
      if (event.target.closest('[data-xai-refresh-analysis]')) {
        // Re-run keeps the currently-scoped environment (don't snap back to the default).
        const cur = xaiResolveBundle(currentXaiCriterion())
        onXaiRefreshAnalysisClick((cur && cur.analysis.environment) || undefined)
        return
      }
      const envSortTh = event.target.closest('[data-xai-env-sort]')
      if (envSortTh) {
        const key = envSortTh.dataset.xaiEnvSort
        if (xaiEnvSortKey === key) xaiEnvSortDir = xaiEnvSortDir === 'asc' ? 'desc' : 'asc'
        else {
          xaiEnvSortKey = key
          xaiEnvSortDir = 'desc'
        }
        renderXai()
        return
      }
      const envBtn = event.target.closest('[data-xai-env]')
      if (envBtn) {
        xaiAnalyzeEnvironment(envBtn.dataset.xaiEnv)
        return
      }
      const tabBtn = event.target.closest('[data-xai-tab]')
      if (tabBtn) {
        xaiTab = tabBtn.dataset.xaiTab
        renderXai()
        return
      }
      if (event.target.closest('[data-xai-rail-toggle]')) {
        xaiRailCollapsed = !xaiRailCollapsed
        renderXai()
        return
      }
      const pcPick = event.target.closest('[data-xai-pc-pick]')
      if (pcPick) {
        const raw = pcPick.dataset.xaiPcPick
        const sep = raw.indexOf('|||')
        const lev = raw.slice(0, sep)
        const val = raw.slice(sep + 3)
        if (xaiPcPick[lev] === val) delete xaiPcPick[lev]
        else xaiPcPick[lev] = val
        renderXai()
        return
      }
      if (event.target.closest('[data-xai-pc-clear]')) {
        xaiPcPick = {}
        renderXai()
        return
      }
      const exLev = event.target.closest('[data-xai-explore-lever]')
      if (exLev) {
        const lever = exLev.dataset.xaiExploreLever
        const rec = xaiRunAnalysisCache.get(xaiFocusKey)
        const cfg = rec && rec.analysis ? rec.analysis.config : null
        if (cfg) {
          const bundle = xaiResolveBundle(currentXaiCriterion())
          const observed = bundle ? (bundle.analysis.setups || []).map((sp) => sp.config[lever]) : []
          const values = xaiSweepValues(lever, cfg[lever], observed)
          if (values.length < 2) {
            setStatusLine('xai-status', `Can't sweep ${lever} — it has no declared range or choices to vary.`, true)
          } else {
            const fixed = {}
            for (const [k, v] of Object.entries(cfg)) if (k !== lever && k !== 'seed' && String(v) !== 'n/a') fixed[k] = v
            xaiLaunchBatch([{ fixed, sweep: { [lever]: values }, seeds: [0] }], `Sweep ${lever}`)
          }
        }
        return
      }
      if (event.target.closest('[data-xai-stabilize-current]')) {
        const rec = xaiRunAnalysisCache.get(xaiFocusKey)
        const cfg = rec && rec.analysis ? rec.analysis.config : null
        const bundle = xaiResolveBundle(currentXaiCriterion())
        if (cfg && bundle) {
          const setup = (bundle.analysis.setups || []).find((sp) =>
            Object.keys(sp.config).every((k) => String(sp.config[k]) === String(cfg[k])),
          )
          const usedList = setup ? setup.seedList || [] : []
          const cur = usedList.length || 1
          const fresh = xaiFreshSeeds(usedList, Math.max(2, Math.max(5, cur + 2) - cur))
          const fixed = {}
          for (const [k, v] of Object.entries(cfg)) if (k !== 'seed' && String(v) !== 'n/a') fixed[k] = v
          xaiLaunchBatch([{ fixed, seeds: fresh }], 'Verify seeds')
        }
        return
      }
      const stab = event.target.closest('[data-xai-stabilize]')
      if (stab) {
        const bundle = xaiResolveBundle(currentXaiCriterion())
        const setup = bundle && (bundle.analysis.setups || []).find((sp) => sp.key === stab.dataset.xaiStabilize)
        if (setup) {
          const cur = setup.seeds || 1
          const target = Math.max(5, cur + 3) // enough seeds for a trustworthy interval (+a few for outliers)
          const fresh = xaiFreshSeeds(setup.seedList || [], target - cur)
          const fixed = { ...(bundle.analysis.environment || {}), ...setup.config }
          xaiLaunchBatch([{ fixed, seeds: fresh }], `Stabilize ${shortKey(setup.key)}`)
        }
        return
      }
      const pcFocus = event.target.closest('[data-xai-focus-config]')
      if (pcFocus) {
        xaiFocusKey = pcFocus.dataset.xaiFocusConfig
        xaiScope = 'current'
        xaiTab = 'standing'
        renderXai()
        return
      }
      const pcaColor = event.target.closest('[data-xai-pca-color]')
      if (pcaColor) {
        xaiPcaColor = pcaColor.dataset.xaiPcaColor === 'model' ? 'model' : 'rank'
        renderXai()
        return
      }
      const mapBtn = event.target.closest('[data-xai-map]')
      if (mapBtn) {
        const k = mapBtn.dataset.xaiMap
        xaiMapKind = k === 'pca' || k === 'pareto' ? k : 'parallel'
        renderXai()
        return
      }
      const helpBtn = event.target.closest('[data-xai-help]')
      if (helpBtn) {
        const id = helpBtn.dataset.xaiHelp
        xaiHelpOpen = xaiHelpOpen === id ? null : id
        renderXai()
        return
      }
      const chatBtn2 = event.target.closest('[data-xai-chat]')
      if (chatBtn2) {
        xaiDiscussView(chatBtn2.dataset.xaiChat)
        return
      }
      if (event.target.closest('[data-xai-scope-header-toggle]')) {
        xaiScopeHeaderExpanded = !xaiScopeHeaderExpanded
        renderXai()
        return
      }
      const collapseBtn = event.target.closest('[data-xai-collapse]')
      if (collapseBtn) {
        const id = collapseBtn.dataset.xaiCollapse
        const card = collapseBtn.closest('[data-collapse-card]')
        const nowCollapsed = !xaiCollapsed.has(id)
        if (nowCollapsed) xaiCollapsed.add(id)
        else xaiCollapsed.delete(id)
        if (card) card.classList.toggle('is-collapsed', nowCollapsed)
        collapseBtn.setAttribute('aria-expanded', String(!nowCollapsed))
        return
      }
      const scopeBtn = event.target.closest('[data-xai-scope]')
      if (scopeBtn) {
        xaiScope = scopeBtn.dataset.xaiScope === 'current' ? 'current' : 'all'
        // Land on the first tab of the new scope (the old tab id may not exist there).
        xaiTab = xaiScope === 'current' ? 'standing' : 'environments'
        renderXai()
        return
      }
      const analyzeRunBtn = event.target.closest('[data-xai-analyze-run]')
      if (analyzeRunBtn) {
        onXaiAnalyzeRunClick(analyzeRunBtn.dataset.xaiAnalyzeRun)
        return
      }
      if (event.target.closest('[data-xai-clear-focus]')) {
        xaiFocusKey = null
        renderXai()
        return
      }
      if (event.target.closest('[data-xai-scroll-recs]')) {
        xaiTab = 'recommender'
        renderXai()
        return
      }
      if (event.target.closest('[data-xai-view-activity]')) {
        showTab('activity')
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
    p.color || (groupColors && p.group !== undefined ? groupColors.get(String(p.group)) || '' : '')
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
    // Focus rings drawn last so the highlighted point (e.g. "this run") always sits on top.
    for (const p of points.filter((q) => q.focus)) {
      parts.push(
        `<circle class="chart-focus-ring" cx="${px(p.x)}" cy="${py(p.y)}" r="7.5" fill="none">${p.label ? `<title>${escapeHtml(p.label)}</title>` : ''}</circle>`,
      )
    }
  }
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeHtml(opts.ariaLabel || '')}">${parts.join('')}</svg>`
}
// A 3-D scatter rendered with an isometric projection: a faint wireframe cube for orientation, faint
// drop-lines to the floor for height, and points coloured by the caller. Each point = {x,y,z,color,label,
// key?,focus?}. Clickable when it carries a `key` (data-xai-focus-config). Used by the 3-D trade-off map.
function buildScatter3dChart(opts) {
  const pts = (opts.points || []).filter(
    (p) => Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z),
  )
  if (!pts.length) return ''
  const W = opts.width || 560
  const H = opts.height || 440
  const pad = 52
  const plotW = W - 2 * pad
  const plotH = H - 2 * pad
  const axis = (sel) => {
    const vs = pts.map(sel)
    const lo = Math.min(...vs)
    const hi = Math.max(...vs)
    return { lo, hi, n: (v) => (hi > lo ? (v - lo) / (hi - lo) : 0.5) }
  }
  const AX = axis((p) => p.x)
  const AY = axis((p) => p.y)
  const AZ = axis((p) => p.z)
  const a = Math.PI / 6
  const cos = Math.cos(a)
  const sin = Math.sin(a)
  // Isometric: x → right+down, z → left+down, y (height) → up. The 8 unit-cube corners bound the screen
  // extents to [-cos, cos] × [-1, 1], so the projection auto-fits with fixed padding.
  const iso = (nx, ny, nz) => [(nx - nz) * cos, (nx + nz) * sin - ny]
  const sx = (ix) => pad + ((ix + cos) / (2 * cos)) * plotW
  const sy = (iy) => pad + ((iy + 1) / 2) * plotH
  const project = (nx, ny, nz) => {
    const [ix, iy] = iso(nx, ny, nz)
    return [sx(ix), sy(iy)]
  }
  const corner = (cx, cy, cz) => project(cx, cy, cz)
  const line = (p, q, cls) => `<line class="${cls}" x1="${p[0].toFixed(1)}" y1="${p[1].toFixed(1)}" x2="${q[0].toFixed(1)}" y2="${q[1].toFixed(1)}" />`
  const corners = [
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
  ]
  const parts = []
  // Floor (y=0) parallelogram, faint, to ground the cube.
  const floor = [corner(0, 0, 0), corner(1, 0, 0), corner(1, 0, 1), corner(0, 0, 1)]
  parts.push(`<polygon class="chart3d-floor" points="${floor.map((c) => `${c[0].toFixed(1)},${c[1].toFixed(1)}`).join(' ')}" />`)
  // All 12 cube edges (corners one unit-step apart), faint.
  for (let i = 0; i < corners.length; i++)
    for (let j = i + 1; j < corners.length; j++) {
      const d = corners[i].reduce((s, v, k) => s + Math.abs(v - corners[j][k]), 0)
      if (d === 1) parts.push(line(corner(...corners[i]), corner(...corners[j]), 'chart3d-edge'))
    }
  // The three labelled axes from the origin corner + their min/max ticks.
  const O = corner(0, 0, 0)
  const axEnd = corner(1, 0, 0)
  const ayEnd = corner(0, 1, 0)
  const azEnd = corner(0, 0, 1)
  parts.push(line(O, axEnd, 'chart3d-axis'), line(O, ayEnd, 'chart3d-axis'), line(O, azEnd, 'chart3d-axis'))
  const tick = (c, txt, anchor, dy) => `<text class="chart-tick" x="${(c[0]).toFixed(1)}" y="${(c[1] + (dy || 0)).toFixed(1)}" text-anchor="${anchor}">${escapeHtml(formatTickValue(txt))}</text>`
  const axisLabel = (c, txt, anchor, dy) => `<text class="chart3d-axis-label" x="${(c[0]).toFixed(1)}" y="${(c[1] + (dy || 0)).toFixed(1)}" text-anchor="${anchor}">${escapeHtml(txt)}</text>`
  parts.push(
    tick(axEnd, AX.hi, 'start', 14), tick(O, AX.lo, 'end', 14), axisLabel(axEnd, opts.xLabel || 'X', 'start', 27),
    tick(azEnd, AZ.hi, 'end', 14), axisLabel(azEnd, opts.zLabel || 'Z', 'end', 27),
    tick(ayEnd, AY.hi, 'middle', -8), axisLabel(ayEnd, opts.yLabel || 'Y', 'middle', -20),
  )
  // Drop-lines from each point to the floor (height cue); stronger for frontier points.
  const placed = pts.map((p) => {
    const nx = AX.n(p.x)
    const ny = AY.n(p.y)
    const nz = AZ.n(p.z)
    return { p, nx, ny, nz, top: project(nx, ny, nz), foot: project(nx, 0, nz) }
  })
  // Draw far points (smaller nx+nz) first so nearer ones sit on top.
  placed.sort((m, n) => m.nx + m.nz - (n.nx + n.nz))
  for (const it of placed)
    parts.push(`<line class="chart3d-drop${it.p.frontier ? ' is-frontier' : ''}" x1="${it.top[0].toFixed(1)}" y1="${it.top[1].toFixed(1)}" x2="${it.foot[0].toFixed(1)}" y2="${it.foot[1].toFixed(1)}" />`)
  for (const it of placed) {
    const [x, y] = it.top
    const title = it.p.label ? `<title>${escapeHtml(it.p.label)}</title>` : ''
    const open = it.p.key ? ` data-xai-focus-config="${escapeHtml(it.p.key)}" style="cursor:pointer;fill:${it.p.color}"` : ` style="fill:${it.p.color}"`
    parts.push(`<circle class="chart3d-point" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${it.p.frontier ? 5 : 3.4}"${open}>${title}</circle>`)
  }
  for (const it of placed.filter((q) => q.p.focus))
    parts.push(`<circle class="chart-focus-ring" cx="${it.top[0].toFixed(1)}" cy="${it.top[1].toFixed(1)}" r="8.5" fill="none">${it.p.label ? `<title>${escapeHtml(it.p.label)}</title>` : ''}</circle>`)
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
      const isCur = v === current
      const breaking = entry && entry.breaking ? '<span class="badge is-bad">breaking</span> ' : ''
      const date =
        entry && entry.date ? `<span class="card-sub">${escapeHtml(String(entry.date))}</span>` : ''
      const summary = entry
        ? escapeHtml(String(entry.summary || ''))
        : '<span class="card-sub">(no changelog entry for this version)</span>'
      return `<div class="version-card${isCur ? ' is-current' : ''}">
        <p class="version-card-head"><strong>v${escapeHtml(v)}</strong>${isCur ? ' <span class="badge">current</span>' : ''} ${breaking}${date}</p>
        <p class="version-summary">${summary}</p>
      </div>`
    })
    .join('')
  setHtml(body, `<div class="versions-list">${cards}</div>`)
}

// --- Datasets / Environments tables ----------------------------------------------
// Both tabs render their named lever bundles as the reusable Runs-style sortable table (one column per
// lever + a trailing actions column), so values line up and any column sorts on a header click. State is
// a {key, dir} per tab, defaulting to name ascending.
let environmentsSort = { key: 'name', dir: 'asc' }
let datasetsSort = { key: 'name', dir: 'asc' }
// A sortable Runs-style table for named lever bundles. `rows` are normalised
// ({ id, name, default, editable, values, raw }); `leverEntries` is [k, spec] pairs (the columns, in
// manifest order); `actionsFn(row)` supplies the trailing buttons. Header cells carry data-<sortAttr> so
// the tab's click handler can re-sort; the default row is highlighted + badged.
function bundleTableHtml(rows, leverEntries, sort, sortAttr, actionsFn) {
  const cols = [{ id: 'name', label: 'Name', num: false, help: 'The name you gave this bundle.' }].concat(
    leverEntries.map(([k, spec]) => ({
      id: k,
      label: k,
      num: spec.type === 'number',
      help: spec.description || '',
    })),
  )
  const sorted = window.BundleTable.sortRows(rows, sort.key, sort.dir)
  const header =
    cols
      .map((c) => {
        const arrow = sort.key === c.id ? (sort.dir === 'asc' ? ' ▲' : ' ▼') : ''
        return `<th class="runs-th${c.num ? ' num' : ''}" data-${sortAttr}="${escapeHtml(c.id)}"${helpAttr(c.help)}>${escapeHtml(c.label)}${arrow}</th>`
      })
      .join('') + '<th class="runs-th bundle-actions-col" aria-label="Actions"></th>'
  const body = sorted
    .map((r) => {
      const note = r.editable
        ? ''
        : ' <span class="card-sub">(manifest defaults — clone to start)</span>'
      const star = r.default ? ' <span class="badge">★ default</span>' : ''
      const leverCells = leverEntries
        .map(
          ([k, spec]) =>
            `<td class="${spec.type === 'number' ? 'num' : ''}">${escapeHtml(r.values[k] === undefined ? '—' : String(r.values[k]))}</td>`,
        )
        .join('')
      return `<tr class="bundle-row${r.default ? ' is-selected' : ''}"><td class="bundle-name-cell">${escapeHtml(r.name)}${star}${note}</td>${leverCells}<td class="bundle-actions-cell"><div class="head-actions">${actionsFn(r)}</div></td></tr>`
    })
    .join('')
  return `<div class="table-wrap"><table class="runs-table bundle-table"><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table></div>`
}

// --- Environments tab ------------------------------------------------------------
function environmentActionsHtml(env, editable, isDefault) {
  const cloneBtn = `<button type="button" class="icon-btn" data-env-clone="${escapeHtml(env.id)}" title="Clone to a new environment" aria-label="Clone">⧉</button>`
  const setDefaultBtn =
    editable && !isDefault
      ? `<button type="button" class="icon-btn" data-env-default="${escapeHtml(env.id)}" title="Set as default environment" aria-label="Set as default">☆</button>`
      : ''
  return editable
    ? `${cloneBtn}${setDefaultBtn}<button type="button" class="icon-btn" data-env-edit="${escapeHtml(env.id)}" title="Edit" aria-label="Edit">✎</button><button type="button" class="icon-btn icon-btn-danger" data-env-delete="${escapeHtml(env.id)}" title="Delete" aria-label="Delete">${iconDeleteSvg()}</button>`
    : cloneBtn
}
function environmentRows() {
  const defId = defaultEnvironmentId()
  // The manifest-defaults row is only a clone-to-start SEED for a project with no environments yet; once
  // the user has defined any, it's hidden (the chosen default lives among their records).
  if (!environmentsCache.length) {
    const d = defaultEnvironment()
    return [{ id: d.id, name: d.name, default: false, editable: false, values: d.settings || {}, raw: d }]
  }
  return environmentsCache.map((e) => ({
    id: e.id,
    name: e.name,
    default: e.id === defId,
    editable: true,
    values: e.settings || {},
    raw: e,
  }))
}
function renderEnvironmentsTable() {
  const body = byId('environments-body')
  if (!body) return
  setHtml(
    body,
    bundleTableHtml(environmentRows(), envLeverEntries(), environmentsSort, 'env-sort', (r) =>
      environmentActionsHtml(r.raw, r.editable, r.default),
    ),
  )
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
  renderEnvironmentsTable()
}
// The create/edit form: a name + a typed field per environment lever (checkbox for boolean levers like
// allow_shorting, a select for choices, else a number).
function environmentFormHtml(env, isClone) {
  const isNew = isClone || !env || env.id === 'default'
  const settings = (env && env.settings) || defaultEnvironment().settings
  const nameVal = isClone
    ? env && env.name
      ? `${env.name} (copy)`
      : ''
    : isNew
      ? ''
      : env.name
  const fields = envLeverEntries()
    .map(([k, spec]) => {
      const cur = settings[k]
      const labelSpan = `<span${helpAttr(spec.description || '')}>${escapeHtml(k)}</span>`
      if (spec.type === 'boolean') {
        const checked = cur === true || String(cur) === 'true' ? ' checked' : ''
        return `<label class="check-row"><input type="checkbox" name="env:${escapeHtml(k)}"${checked} />${labelSpan}</label>`
      }
      if (spec.type === 'choice') {
        const opts = (spec.choices || [])
          .map(
            (c) =>
              `<option value="${escapeHtml(String(c))}"${String(cur) === String(c) ? ' selected' : ''}>${escapeHtml(String(c))}</option>`,
          )
          .join('')
        return `<label class="field">${labelSpan}<select name="env:${escapeHtml(k)}">${opts}</select></label>`
      }
      const { min, max } = leverRange(spec)
      const minAttr = Number.isFinite(Number(min)) ? ` min="${Number(min)}"` : ''
      const maxAttr = Number.isFinite(Number(max)) ? ` max="${Number(max)}"` : ''
      const val = cur === undefined ? '' : escapeHtml(String(cur))
      return `<label class="field">${labelSpan}
        <input type="number" step="any" name="env:${escapeHtml(k)}" value="${val}"${minAttr}${maxAttr} /></label>`
    })
    .join('')
  return `<input type="hidden" name="id" value="${escapeHtml(isNew ? randomHexId() : env.id)}" />
    <label class="field"><span>Name</span>
      <input type="text" name="name" value="${escapeHtml(nameVal)}" placeholder="e.g. Low fee · tight SL" /></label>
    <div class="lever-grid">${fields}</div>
    <div class="form-actions">
      <button type="submit">Save environment</button>
      <button type="button" class="ghost-btn" data-env-form-cancel>Cancel</button>
    </div>`
}
// The add/edit/clone form lives in a modal (built lazily, mirrors the custom-filter popup) so it overlays
// the table instead of pushing it down. Backdrop / ✕ / Cancel all close it.
function environmentModal() {
  let modal = byId('environment-modal')
  if (!modal) {
    modal = document.createElement('div')
    modal.id = 'environment-modal'
    modal.className = 'chart-modal'
    document.body.appendChild(modal)
    modal.addEventListener('click', (event) => {
      if (event.target === modal || event.target.closest('[data-env-form-cancel]')) {
        toggleEnvironmentForm(false)
      }
    })
    modal.addEventListener('submit', (event) => {
      const form = event.target.closest('#environment-form')
      if (form) {
        event.preventDefault()
        onSaveEnvironment(form)
      }
    })
  }
  return modal
}
function toggleEnvironmentForm(show, env, isClone) {
  const modal = environmentModal()
  if (!show) {
    modal.hidden = true
    modal.innerHTML = ''
    return
  }
  const editing = !!(env && env.id && env.id !== 'default' && !isClone)
  const title = editing ? 'Edit environment' : isClone ? 'Clone environment' : 'Add environment'
  modal.innerHTML = `<div class="chart-modal__backdrop" data-env-form-cancel></div>
    <div class="chart-modal__panel bundle-form-panel" role="dialog" aria-label="${title}">
      <div class="chart-modal__head">
        <strong>${title}</strong>
        <button type="button" class="icon-btn" data-env-form-cancel title="Close" aria-label="Close">✕</button>
      </div>
      <div class="chart-modal__scroll">
        <p id="environment-form-status" class="form-status" role="status" hidden></p>
        <form id="environment-form" class="bundle-form" autocomplete="off">${environmentFormHtml(env, isClone)}</form>
      </div>
    </div>`
  modal.hidden = false
  const nameInput = modal.querySelector('input[name="name"]')
  if (nameInput) nameInput.focus()
}
async function onSaveEnvironment(form) {
  const id = form.elements.id.value
  const name = String(form.elements.name.value || '').trim()
  if (!name) {
    setStatusLine('environment-form-status', 'Give the environment a name.', true)
    return
  }
  const settings = {}
  for (const [k, spec] of envLeverEntries()) {
    const el = form.querySelector(`[name="env:${k}"]`)
    if (!el) continue
    if (spec.type === 'boolean') settings[k] = el.checked
    else if (spec.type === 'choice') {
      if (el.value !== '') settings[k] = el.value
    } else if (el.value !== '') settings[k] = Number(el.value)
  }
  if (environmentDuplicateOf(name, settings, id)) {
    setStatusLine(
      'environment-form-status',
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
    setStatusLine('environment-form-status', 'Could not save — please try again.', true)
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
  const body = byId('environments-body')
  if (body) {
    body.addEventListener('click', (event) => {
      const th = event.target.closest('.runs-th[data-env-sort]')
      if (th) {
        environmentsSort = window.BundleTable.nextSort(
          environmentsSort.key,
          environmentsSort.dir,
          th.dataset.envSort,
          'asc',
        )
        renderEnvironmentsTable()
        return
      }
      const edit = event.target.closest('button[data-env-edit]')
      if (edit) {
        const env = environmentsCache.find((x) => x.id === edit.dataset.envEdit)
        if (env) toggleEnvironmentForm(true, env)
        return
      }
      const clone = event.target.closest('button[data-env-clone]')
      if (clone) {
        const env =
          environmentsCache.find((x) => x.id === clone.dataset.envClone) || defaultEnvironment()
        toggleEnvironmentForm(true, env, true)
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
function datasetActionsHtml(ds, editable, isDefault) {
  const cloneBtn = `<button type="button" class="icon-btn" data-ds-clone="${escapeHtml(ds.id)}" title="Clone to a new dataset" aria-label="Clone">⧉</button>`
  const setDefaultBtn =
    editable && !isDefault
      ? `<button type="button" class="icon-btn" data-ds-default="${escapeHtml(ds.id)}" title="Set as default dataset" aria-label="Set as default">☆</button>`
      : ''
  return editable
    ? `${cloneBtn}${setDefaultBtn}<button type="button" class="icon-btn" data-ds-edit="${escapeHtml(ds.id)}" title="Edit" aria-label="Edit">✎</button><button type="button" class="icon-btn icon-btn-danger" data-ds-delete="${escapeHtml(ds.id)}" title="Delete" aria-label="Delete">${iconDeleteSvg()}</button>`
    : cloneBtn
}
function datasetRows() {
  const defId = defaultDatasetId()
  // The manifest-defaults row is only a clone-to-start SEED for a project with no datasets yet; once the
  // user has defined any, it's hidden (the chosen default lives among their records, not this synthetic one).
  if (!datasetsCache.length) {
    const d = defaultDataset()
    return [{ id: d.id, name: d.name, default: false, editable: false, values: d.settings || {}, raw: d }]
  }
  return datasetsCache.map((d) => ({
    id: d.id,
    name: d.name,
    default: d.id === defId,
    editable: true,
    values: d.settings || {},
    raw: d,
  }))
}
function renderDatasetsTable() {
  const body = byId('datasets-body')
  if (!body) return
  setHtml(
    body,
    bundleTableHtml(datasetRows(), datasetLeverEntries(), datasetsSort, 'ds-sort', (r) =>
      datasetActionsHtml(r.raw, r.editable, r.default),
    ),
  )
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
  renderDatasetsTable()
}
// A single-value, type-aware field for one dataset lever (choice → select, number → number input,
// boolean → true/false select), pre-filled with the dataset's saved value. "" = use the default.
function datasetFieldHtml(key, spec, value) {
  // A named dataset must pin a CONCRETE value for every lever — there is no "— default —" escape that would
  // leave it unpinned (and so resolve to a synonym at run time). A concrete option is always pre-selected:
  // the saved value when editing/cloning, else the manifest default, else the first concrete choice.
  const synonyms = new Set(INPUT_SYNONYMS)
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
function datasetFormHtml(ds, isClone) {
  const isNew = isClone || !ds || ds.id === 'default'
  const settings = (ds && ds.settings) || defaultDataset().settings
  const nameVal = isClone ? (ds && ds.name ? `${ds.name} (copy)` : '') : isNew ? '' : ds.name
  const fields = datasetLeverEntries()
    .map(([k, spec]) => datasetFieldHtml(k, spec, settings[k]))
    .join('')
  return `<input type="hidden" name="id" value="${escapeHtml(isNew ? randomHexId() : ds.id)}" />
    <label class="field"><span>Name</span>
      <input type="text" name="name" value="${escapeHtml(nameVal)}" placeholder="e.g. 1h+1d · 2024" /></label>
    <div class="lever-grid">${fields}</div>
    <div class="form-actions">
      <button type="submit">Save dataset</button>
      <button type="button" class="ghost-btn" data-ds-form-cancel>Cancel</button>
    </div>`
}
// The add/edit/clone form lives in a modal (mirrors the Environments popup) so it overlays the table
// instead of pushing it down. Backdrop / ✕ / Cancel all close it.
function datasetModal() {
  let modal = byId('dataset-modal')
  if (!modal) {
    modal = document.createElement('div')
    modal.id = 'dataset-modal'
    modal.className = 'chart-modal'
    document.body.appendChild(modal)
    modal.addEventListener('click', (event) => {
      if (event.target === modal || event.target.closest('[data-ds-form-cancel]')) {
        toggleDatasetForm(false)
      }
    })
    modal.addEventListener('submit', (event) => {
      const form = event.target.closest('#dataset-form')
      if (form) {
        event.preventDefault()
        onSaveDataset(form)
      }
    })
  }
  return modal
}
function toggleDatasetForm(show, ds, isClone) {
  const modal = datasetModal()
  if (!show) {
    modal.hidden = true
    modal.innerHTML = ''
    return
  }
  const editing = !!(ds && ds.id && ds.id !== 'default' && !isClone)
  const title = editing ? 'Edit dataset' : isClone ? 'Clone dataset' : 'Add dataset'
  modal.innerHTML = `<div class="chart-modal__backdrop" data-ds-form-cancel></div>
    <div class="chart-modal__panel bundle-form-panel" role="dialog" aria-label="${title}">
      <div class="chart-modal__head">
        <strong>${title}</strong>
        <button type="button" class="icon-btn" data-ds-form-cancel title="Close" aria-label="Close">✕</button>
      </div>
      <div class="chart-modal__scroll">
        <p id="dataset-form-status" class="form-status" role="status" hidden></p>
        <form id="dataset-form" class="bundle-form" autocomplete="off">${datasetFormHtml(ds, isClone)}</form>
      </div>
    </div>`
  modal.hidden = false
  const nameInput = modal.querySelector('input[name="name"]')
  if (nameInput) nameInput.focus()
}
async function onSaveDataset(form) {
  const id = form.elements.id.value
  const name = String(form.elements.name.value || '').trim()
  if (!name) {
    setStatusLine('dataset-form-status', 'Give the dataset a name.', true)
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
      'dataset-form-status',
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
    setStatusLine('dataset-form-status', 'Could not save — please try again.', true)
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
  const body = byId('datasets-body')
  if (body) {
    body.addEventListener('click', (event) => {
      const th = event.target.closest('.runs-th[data-ds-sort]')
      if (th) {
        datasetsSort = window.BundleTable.nextSort(
          datasetsSort.key,
          datasetsSort.dir,
          th.dataset.dsSort,
          'asc',
        )
        renderDatasetsTable()
        return
      }
      const edit = event.target.closest('button[data-ds-edit]')
      if (edit) {
        const ds = datasetsCache.find((x) => x.id === edit.dataset.dsEdit)
        if (ds) toggleDatasetForm(true, ds)
        return
      }
      const clone = event.target.closest('button[data-ds-clone]')
      if (clone) {
        const ds = datasetsCache.find((x) => x.id === clone.dataset.dsClone) || defaultDataset()
        toggleDatasetForm(true, ds, true)
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
  if (s.compare && s.compare.lever && Array.isArray(s.compare.values)) {
    items.push(
      `<li>compare <code>${escapeHtml(String(s.compare.lever))}</code>: ${escapeHtml(s.compare.values.map((v) => String(v)).join(' vs '))}</li>`,
    )
  }
  const seeds = Array.isArray(s.seeds) ? s.seeds.length : 0
  if (seeds) items.push(`<li>${seeds} seed${seeds === 1 ? '' : 's'}</li>`)
  if (!items.length) return '<p class="card-sub">Default config.</p>'
  return `<ul class="spec-list">${items.join('')}</ul>`
}
// The runs that are evidence for a hypothesis (spec-consistent), and the measured read / effective verdict
// derived from them — thin wrappers over the pure `window.Hypothesis` module. They use the ALL-runs
// snapshot when present (after a refresh); otherwise they fall back to the hypothesis's PERSISTED evidence
// / status, so a paged Runs view (only the current page loaded) never undercounts a verdict.
function hypothesisMatchedRuns(h) {
  return window.Hypothesis.hypothesisMatchingRuns(h && h.spec, allRunsCache)
}
function hypothesisMatchedCount(h) {
  if (allRunsCache.length) return hypothesisMatchedRuns(h).length
  return h && h.evidence && Array.isArray(h.evidence.matchedKeys)
    ? h.evidence.matchedKeys.length
    : 0
}
function hypothesisMeasured(h) {
  if (allRunsCache.length)
    return window.Hypothesis.measuredFromRuns(hypothesisMatchedRuns(h), objectiveDirection())
  return (h && h.evidence && h.evidence.measured) || null
}
// The verdict the runs imply IGNORING any manual override — the "auto suggests" hint. Context-aware: a
// context-spanning hypothesis reads its cross-context comparison, not the pooled beats-hold.
// Resolver for the 'proposed' derivation: is the catalog model bound to `name` implemented? Returns
// true (implemented) | false (a known catalog model that isn't implemented) | null (catalog not loaded, or
// no/ambiguous model). null SUPPRESSES the derivation, so an unloaded catalog never spuriously flips a
// verdict ('proposed' is display-derived, never persisted — refresh never writes it).
function modelNameImplemented(name) {
  if (!modelsCache || !modelsCache.length) return null
  const cfg = { model_name: String(name) }
  const model = modelsCache.find((m) => window.Models.runMatchesModel(m, cfg))
  if (!model) return null
  return window.Models.isModelImplemented(model, manifest)
}
function autoSuggestedVerdict(h) {
  return window.Hypothesis.autoVerdictForHypothesis(
    h,
    allRunsCache,
    objectiveDirection(),
    hypothesisMinRuns,
    modelNameImplemented,
  )
}
function effectiveHypothesisVerdict(h) {
  if (!h) return 'untested'
  if (allRunsCache.length)
    return window.Hypothesis.effectiveVerdict(
      h,
      allRunsCache,
      objectiveDirection(),
      hypothesisMinRuns,
      modelNameImplemented,
    )
  // No run snapshot loaded: use the persisted status, but still surface 'proposed' for an auto hypothesis
  // whose required model isn't implemented (it depends on the model lifecycle, not on runs).
  const base = h.status || 'untested'
  if (
    h.verdictSource !== 'manual' &&
    base === 'untested' &&
    window.Hypothesis.requiresUnimplementedModel(h.spec, modelNameImplemented)
  )
    return 'proposed'
  return base
}
// Icon-only action buttons (same size, tooltips on hover) — shown in the card summary so they're
// available collapsed AND expanded. "Launch more runs" is the play button; it stays enabled below
// minRuns so the user can gather the data a verdict needs.
function hypothesisActionsHtml(h) {
  const id = escapeHtml(h.id)
  const c = h.campaign
  const busy = c && (c.status === 'running' || c.status === 'queued')
  return (
    `<button type="button" class="card-btn" data-action="run" data-id="${id}"${busy ? ' disabled' : ''}${helpAttr('Launch more runs — gather the evidence the verdict needs (the verdict updates as runs land).')}>${iconRunSvg(15)}</button>` +
    `<button type="button" class="card-btn" data-action="override" data-id="${id}"${helpAttr('Override the verdict manually.')}>${iconEditSvg(15)}</button>` +
    `<button type="button" class="card-btn" data-action="${h.dismissed ? 'restore' : 'dismiss'}" data-id="${id}"${helpAttr(h.dismissed ? 'Restore this hypothesis.' : 'Dismiss this hypothesis (hide without deleting).')}>${h.dismissed ? '↺' : iconCrossSvg()}</button>` +
    `<button type="button" class="card-btn card-btn-danger" data-action="delete" data-id="${id}"${helpAttr('Delete this hypothesis.')}>${iconDeleteSvg()}</button>`
  )
}
// The run-count chip shown next to the verdict — click to view the matching runs. Highlights "n/min"
// when there aren't enough runs to judge it yet.
function hypothesisRunChipHtml(h) {
  const n = hypothesisMatchedCount(h)
  const enough = n >= hypothesisMinRuns
  const label = enough ? `${n} run${n === 1 ? '' : 's'}` : `${n}/${hypothesisMinRuns} runs`
  const help = enough
    ? 'Runs matching this hypothesis — click to view them in the Runs tab.'
    : `Only ${n} of ${hypothesisMinRuns} runs needed to judge it — launch more.`
  return `<button type="button" class="run-count-chip${enough ? '' : ' is-low'}" data-action="view-runs" data-id="${escapeHtml(h.id)}"${helpAttr(help)}>${label}</button>`
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
  const own = (allRunsCache.length ? allRunsCache : runsCache).filter((r) => keySet.has(r.key))
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
// The EVIDENCE behind a hypothesis's verdict: a plain-language basis line (on what grounds it's
// proven/disproved/untested), the source's claimed metrics if any, and the list of matching runs (each
// with its objective + return-vs-hold) with a link that opens the Runs tab filtered to exactly them.
// The plain-language rule the auto-verdict applies, so an "untested"/"disproved" card explains itself.
function hypothesisCriteriaText() {
  const m = hypothesisMinRuns
  return `How it's judged: among the runs whose config MATCHES this spec, PROVEN if the best beats buy-and-hold out-of-sample (a positive return-vs-hold), DISPROVED if at least ${m} report a result and none beat it, otherwise UNTESTED — which also holds while fewer than ${m} matching runs REPORT return-vs-hold (older runs that don't are skipped).`
}
// The verdict basis from the PERSISTED evidence — used when the full run snapshot isn't loaded on this tab,
// so the view stays consistent with the run-count chip + verdict badge (which also read persisted state).
function hypothesisPersistedBasis(h) {
  const ev = (h && h.evidence) || {}
  const n = Array.isArray(ev.matchedKeys) ? ev.matchedKeys.length : 0
  const m = ev.measured || {}
  const verdict = (h && h.status) || 'untested'
  if (!n) return 'No matching runs recorded yet — launch runs to test it.'
  const runs = `${n} matching run${n === 1 ? '' : 's'}`
  if (m.beatsHold === null || m.beatsHold === undefined)
    return `${runs}, but none report return-vs-hold — can't judge against buy-and-hold yet, so it stays <strong>untested</strong>.`
  if (verdict === 'proven')
    return `<strong>Proven</strong> — the best of ${runs} beats buy-and-hold out-of-sample.`
  if (verdict === 'disproved')
    return `<strong>Disproved</strong> — none of ${runs} beat buy-and-hold out-of-sample.`
  if (Number(m.runs || n) < hypothesisMinRuns)
    return `<strong>Untested</strong> — only ${Number(m.runs || n)}/${hypothesisMinRuns} matching runs report a result.`
  return `<strong>${escapeHtml(verdict)}</strong> — ${runs}.`
}
function hypothesisEvidenceHtml(h) {
  const id = escapeHtml(h.id)
  const claimed = h.claimedMetrics && Object.keys(h.claimedMetrics).length ? h.claimedMetrics : null
  const claimedRow = claimed
    ? `<p class="card-sub hyp-claimed">Source claims: ${escapeHtml(
        Object.entries(claimed)
          .map(([k, v]) => `${k} ${v}`)
          .join(' · '),
      )}</p>`
    : ''
  const criteria = `<p class="card-sub hyp-criteria">${hypothesisCriteriaText()}</p>`
  // The full run snapshot is loaded by the shared Refresh, not on this tab. When it isn't loaded, show the
  // PERSISTED evidence (consistent with the chip + badge) rather than recomputing against an empty cache —
  // which would wrongly read "no matching runs" even though the verdict was computed over thousands.
  if (!allRunsCache.length) {
    return `<div class="hyp-evidence">${claimedRow}
      <p class="card-sub hyp-basis">${hypothesisPersistedBasis(h)}</p>
      ${criteria}
      <p class="card-sub hyp-stale-note">This reflects the last full <strong>Refresh</strong>. After launching runs, click <strong>Refresh</strong> (top-right) to re-evaluate the verdict against them.</p>
    </div>`
  }
  const runs = hypothesisMatchedRuns(h)
  if (!runs.length) {
    return `<div class="hyp-evidence">${claimedRow}<p class="card-sub">No runs match this spec yet — launch runs to test it.</p>${criteria}</div>`
  }
  const pct = (v) => `${v > 0 ? '+' : ''}${v.toFixed(1)}%`
  const rows = runs
    .map((r) => ({
      key: r.key,
      obj: Number(r.summary && r.summary.objective),
      vh: runMetricValue(r, 'return_vs_hold_pct'),
    }))
    .sort(
      (a, b) =>
        (Number.isFinite(b.vh) ? b.vh : -Infinity) - (Number.isFinite(a.vh) ? a.vh : -Infinity),
    )
  const vhVals = rows.map((r) => r.vh).filter(Number.isFinite)
  const beat = vhVals.filter((v) => v > 0).length
  const verdict = effectiveHypothesisVerdict(h)
  let basis
  if (!vhVals.length) {
    basis = `${rows.length} matching run${rows.length === 1 ? '' : 's'}, but none report return-vs-hold — can't judge against buy-and-hold yet.`
  } else if (verdict === 'proven') {
    basis = `<strong>Proven</strong> — ${beat} of ${rows.length} matching runs beat buy-and-hold OOS (best ${pct(Math.max(...vhVals))} vs hold).`
  } else if (verdict === 'disproved') {
    basis = `<strong>Disproved</strong> — 0 of ${rows.length} matching runs beat buy-and-hold OOS (best ${pct(Math.max(...vhVals))} vs hold).`
  } else if (rows.length < hypothesisMinRuns) {
    basis = `<strong>Untested</strong> — only ${rows.length}/${hypothesisMinRuns} runs needed to judge (best ${pct(Math.max(...vhVals))} vs hold so far). Launch more.`
  } else {
    basis = `<strong>Untested</strong> — ${rows.length} runs, best ${pct(Math.max(...vhVals))} vs hold.`
  }
  const shown = rows.slice(0, 6)
  const runRows = shown
    .map(
      (r) =>
        `<tr><td><code>${escapeHtml(shortKey(r.key))}</code></td><td>${Number.isFinite(r.obj) ? escapeHtml(formatObjective(r.obj)) : '—'}</td><td>${Number.isFinite(r.vh) ? escapeHtml(pct(r.vh)) : '—'}</td></tr>`,
    )
    .join('')
  const more =
    rows.length > shown.length ? `<p class="card-sub">+${rows.length - shown.length} more</p>` : ''
  return `<div class="hyp-evidence">
    ${claimedRow}
    <p class="card-sub hyp-basis">${basis}</p>
    ${criteria}
    <table class="kv-table hyp-runs"><thead><tr><th>run</th><th>${escapeHtml(objectiveName())}</th><th>vs hold</th></tr></thead><tbody>${runRows}</tbody></table>
    ${more}
    <button type="button" class="ghost-btn" data-action="view-runs" data-id="${id}"${helpAttr('Open the Runs tab filtered to exactly these matching runs.')}>View ${rows.length} run${rows.length === 1 ? '' : 's'} in Runs</button>
  </div>`
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
// Whether a hypothesis spans more than one context (environments/datasets) — its runs are compared
// ACROSS contexts rather than pooled into one beats-hold read.
function hypothesisIsContextSpanning(h) {
  return !!(h && h.spec && window.Hypothesis.contextCells(h.spec).length)
}
function contextCellLabel(cell) {
  const parts = Object.entries(cell || {}).map(
    ([k, v]) => `${escapeHtml(k)}=${escapeHtml(String(v))}`,
  )
  return parts.length ? parts.join(', ') : '—'
}
function comparisonKindLabel(kind) {
  return kind === 'invariant'
    ? 'invariant across contexts'
    : kind === 'differs'
      ? 'differs across contexts'
      : 'thesis beats baseline'
}
// The cross-context comparison behind a context-spanning hypothesis: one row per context cell (its
// context values, run count, best objective, beats-hold) — never pooled — plus the criterion + verdict.
function hypothesisComparisonHtml(h) {
  const cmp = h.comparison || { kind: 'beats-baseline' }
  const baselineIndex = cmp.baselineIndex || 0
  const cellCount = window.Hypothesis.contextCells(h.spec).length
  const perContext = allRunsCache.length
    ? window.Hypothesis.measuredByContext(h.spec, allRunsCache, objectiveDirection())
    : null
  const cells = perContext || []
  const verdict = effectiveHypothesisVerdict(h)
  const tolNote =
    cmp.kind === 'beats-baseline'
      ? ''
      : ` (tolerance ${cmp.tolerance != null ? cmp.tolerance : 0.1})`
  const basis = `<strong>${escapeHtml(HYPOTHESIS_VERDICT_LABEL[verdict])}</strong> — comparison: ${escapeHtml(comparisonKindLabel(cmp.kind))}${escapeHtml(tolNote)}.`
  // Spell out EXACTLY what proves/disproves this comparison (which value beats which baseline, the spread
  // tolerance, the readiness gate) — mirrors compareContexts, shown whether or not runs exist yet.
  const criterion = `<p class="card-sub hyp-criteria">How it's judged: ${escapeHtml(
    window.Hypothesis.comparisonCriterion(h.spec, cmp, {
      objectiveName: objectiveName(),
      direction: objectiveDirection(),
      minRuns: hypothesisMinRuns,
    }),
  )}</p>`
  const totalRuns = cells.reduce((n, c) => n + (c.measured ? c.measured.runs : 0), 0)
  if (!totalRuns) {
    return `<div class="hyp-evidence"><p class="card-sub">No matching runs yet across the ${cellCount} contexts — launch runs to test it.</p>${criterion}</div>`
  }
  const rows = cells
    .map((c, i) => {
      const m = c.measured
      const obj = m && Number.isFinite(m.objective) ? formatObjective(m.objective) : '—'
      const beats = m && m.beatsHold === true ? 'yes' : m && m.beatsHold === false ? 'no' : '—'
      const tag =
        cmp.kind === 'beats-baseline' && i === baselineIndex
          ? ' <span class="badge">baseline</span>'
          : ''
      return `<tr><td>${contextCellLabel(c.context)}${tag}</td><td>${m ? m.runs : 0}</td><td>${escapeHtml(obj)}</td><td>${beats}</td></tr>`
    })
    .join('')
  return `<div class="hyp-evidence">
    <p class="card-sub hyp-basis">${basis}</p>
    ${criterion}
    <table class="kv-table hyp-runs"><thead><tr><th>context</th><th>runs</th><th>${escapeHtml(objectiveName())}</th><th>vs hold</th></tr></thead><tbody>${rows}</tbody></table>
    <button type="button" class="ghost-btn" data-action="view-runs" data-id="${escapeHtml(h.id)}"${helpAttr('Open the Runs tab filtered to these matching runs (grouped by environment there).')}>View runs in Runs</button>
  </div>`
}
// A small chip showing THIS PAPER's importance weight for a linked hypothesis (1 = minor … 5 = central) —
// shown in the paper's linked-hypotheses list once the paper has assigned weights (via "Re-weigh
// hypotheses"). The weight is per-paper, so it never appears on a standalone hypothesis card.
function hypothesisWeightChipHtml(weight) {
  const w = typeof weight === 'number' && isFinite(weight) ? weight : null
  if (w === null) return ''
  return `<span class="hyp-weight-chip"${helpAttr('THIS paper’s importance weight for the hypothesis (1 = minor / supporting … 5 = the paper’s central claim) — used when the paper verdict rolls up. Set via "Re-weigh hypotheses".')}>⚖ ${escapeHtml(String(w))}</span>`
}
function hypothesisCardHtml(h, liveIds) {
  const verdict = effectiveHypothesisVerdict(h)
  const badge = `<span class="run-badge ${HYPOTHESIS_VERDICT_BADGE[verdict]}">${escapeHtml(HYPOTHESIS_VERDICT_LABEL[verdict])}</span>`
  const live =
    h.campaign && h.campaign.status === 'running' && liveIds && liveIds.has(h.campaign.activityId)
  const running = live ? `<span class="hyp-running">${spinnerHtml()}</span>` : ''
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
      ? `<p class="paper-suggest"${helpAttr('Manually set — auto-refresh won’t change it. Clear the override to auto-derive again.')}>manual override — auto suggests: ${escapeHtml(HYPOTHESIS_VERDICT_LABEL[autoSuggestedVerdict(h)])}</p>`
      : ''
  const overrideForm = hypothesisOverrideId === h.id ? hypothesisVerdictFormHtml(h) : ''
  const open = hypothesisExpanded.has(h.id) ? ' open' : ''
  return `<details class="hypothesis-card${h.dismissed ? ' is-dismissed' : ''}" data-id="${escapeHtml(h.id)}"${open}>
    <summary class="hypothesis-summary">
      ${badge}
      ${hypothesisRunChipHtml(h)}
      <span class="hyp-title">${escapeHtml(h.title || h.id)}</span>
      ${hypothesisThesisChipHtml(h)}
      ${running}
      <span class="card-actions">${hypothesisActionsHtml(h)}</span>
    </summary>
    <div class="hypothesis-body">
      ${h.rationale ? `<p class="hypothesis-rationale">${escapeHtml(h.rationale)}</p>` : ''}
      ${specSummaryHtml(h.spec)}
      ${hypothesisIsContextSpanning(h) ? hypothesisComparisonHtml(h) : hypothesisEvidenceHtml(h)}
      ${overrideNote}
      ${transitionsHtml(h.transitions)}
      ${linkedPaperChipsHtml(h)}
      ${hypothesisCampaignHtml(h, liveIds)}
      ${h.verdictNote ? `<p class="card-sub paper-note">${escapeHtml(h.verdictNote)}</p>` : ''}
      ${overrideForm}
      <p class="card-sub">${escapeHtml(source)}${by} · created ${escapeHtml(formatWhen(h.createdAt))} · updated ${escapeHtml(formatWhen(h.updatedAt))}</p>
    </div>
  </details>`
}
// Each verdict view is its own collapsible, internally-scrolling section (accordion: only one open at a
// time, tracked by `hypothesisOpenSection`), so all three views stay visible at once and the open one
// scrolls within rather than pushing the page.
function hypothesisSectionHtml(verdict, items, liveIds) {
  const label = HYPOTHESIS_VERDICT_LABEL[verdict] || verdict
  const open = hypothesisOpenSection === verdict ? ' open' : ''
  const cards = items.length
    ? `<div class="hypothesis-cards">${items.map((h) => hypothesisCardHtml(h, liveIds)).join('')}</div>`
    : '<p class="card-sub">None.</p>'
  return `<details class="hypothesis-section" data-section="${escapeHtml(verdict)}"${open}>
    <summary class="hypothesis-section-head">
      <span class="run-badge ${HYPOTHESIS_VERDICT_BADGE[verdict]}">${escapeHtml(label[0].toUpperCase() + label.slice(1))}</span>
      <span class="group-count">${items.length}</span>
    </summary>
    <div class="hypothesis-section-body">${cards}</div>
  </details>`
}
// Re-evaluate every hypothesis against the ALL-runs snapshot, persisting only material changes (a new
// matched-run set OR an auto-verdict flip), recording a transition on each flip. Driven by the shared
// all-runs refresh (`refreshAllRunsDerived`), so the verdict reflects EVERY run, not a page.
async function refreshHypothesisVerdicts(hyps) {
  if (!manifest || !embedded() || !window.Hypothesis) return false
  // Evaluate ONLY when the all-runs snapshot is loaded — never against an empty/partial set, which would
  // wrongly zero every verdict to untested. Without a snapshot the persisted verdicts stand.
  if (!allRunsCache.length) return false
  const direction = objectiveDirection()
  const at = nowIso()
  let wrote = false
  for (const h of hyps || []) {
    const { next, changed } = window.Hypothesis.evaluateHypothesis(h, allRunsCache, {
      direction,
      at,
      minRuns: hypothesisMinRuns,
    })
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
    // Lightbulb + hypothesis = "propose new experiment ideas". Icon-only; the tooltip explains it.
    btn.innerHTML = proposing ? spinnerHtml() : iconLightbulbSvg(15) + iconHypothesisSvg(15)
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
  // Load the persisted aggregate (for the freshness note) + hypotheses (whose verdicts are persisted) +
  // a page of runs (only for the per-campaign best display). Verdicts are NOT recomputed here from the
  // page — the shared all-runs refresh (`refreshAllRunsDerived`) owns that.
  ;[hypothesesCache, proposalSummary, papersCache, runsCache, modelStatsCache, modelsCache] =
    await Promise.all([
      readHypotheses(),
      readProposal(),
      readPapers(),
      readRuns(),
      readModelStats(),
      readModels(),
    ])
  const minRunsInput = byId('hypothesis-min-runs')
  if (minRunsInput && document.activeElement !== minRunsInput)
    minRunsInput.value = hypothesisMinRuns
  const liveIds = hypothesesCache.some((h) => h.campaign && h.campaign.status === 'running')
    ? await readLiveActivityIds()
    : new Set()
  renderProposeControls()
  const controls = hypothesisStatsControlsHtml()
  const visible = hypothesesCache.filter((h) => !h.dismissed)
  const hidden = hypothesesCache.filter((h) => h.dismissed)
  if (!visible.length && !hidden.length) {
    body.innerHTML =
      controls +
      '<div class="empty-hint">No hypotheses yet — propose some (the lightbulb above) or add your own.</div>'
    void checkModelStatsStale()
    return
  }
  const sorted = sortHypotheses(visible)
  const groups = { untested: [], proven: [], disproved: [] }
  for (const h of sorted) (groups[effectiveHypothesisVerdict(h)] || groups.untested).push(h)
  // First load (open section not yet chosen): default-open the first non-empty verdict section so it isn't
  // blank — falling back to Hidden when everything has been dismissed.
  if (hypothesisOpenSection === undefined) {
    hypothesisOpenSection =
      HYPOTHESIS_VERDICTS.find((v) => groups[v].length) || (hidden.length ? 'hidden' : null)
  }
  // Hidden (dismissed) hypotheses get their own collapsed section at the bottom, so they stay viewable +
  // restorable without cluttering the live verdict sections.
  const hiddenSection = hidden.length
    ? hypothesisSectionHtml('hidden', sortHypotheses(hidden), liveIds)
    : ''
  body.innerHTML =
    controls +
    HYPOTHESIS_VERDICTS.map((v) => hypothesisSectionHtml(v, groups[v], liveIds)).join('') +
    hiddenSection
  void checkModelStatsStale()
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
// evaluation / verdict / xai-narrative / unrunnable marker keyed by the same run/setup key).
async function deleteRelatedRunRecords(key, setupKey) {
  const types = [
    manifest.recordType + '-evaluation',
    manifest.recordType + '-verdict',
    manifest.recordType + '-xai-narrative',
  ]
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
  forgetRun(key)
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
  for (const k of keys) {
    runsCompareKeys.delete(k)
    forgetRun(k)
  }
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
  // Use the all-runs snapshot when loaded, else the hypothesis's persisted matched-run keys.
  const keys = allRunsCache.length
    ? hypothesisMatchedRuns(h).map((r) => r.key)
    : (h.evidence && Array.isArray(h.evidence.matchedKeys) && h.evidence.matchedKeys) || []
  if (!keys.length) return
  runsFilterKeys = new Set(keys)
  runsFilterLabel = h.title || shortKey(id)
  showTab('runs')
}
// Open the Runs tab drilled to the analysed runs whose config matches EVERY (lever, value) pair — the
// runs behind an xAI fANOVA value or interaction cell. No-op when the pairs name no actual run (e.g. a
// surrogate-predicted cell). `pairs` is an array of [lever, value]; `label` becomes the runs chip.
function viewRunsByLeverValues(pairs, label) {
  const matches = xaiRuns().filter((r) =>
    pairs.every(([lever, value]) => String((r.config || {})[lever]) === String(value)),
  )
  if (!matches.length) return
  runsFilterKeys = new Set(matches.map((r) => r.key))
  runsFilterLabel = label
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
      error: `Unknown spec ${unknownKeys.length === 1 ? 'key' : 'keys'} ${unknownKeys.join(', ')} — only sweep, fixed, seeds, environments, datasets and compare are allowed.`,
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
  for (const part of ['environments', 'datasets']) {
    const value = parsed[part]
    if (value === undefined) continue
    if (
      !Array.isArray(value) ||
      value.some((b) => !b || typeof b !== 'object' || Array.isArray(b))
    ) {
      return { error: `"${part}" must be an array of context-lever bundles (objects).` }
    }
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
  // environments/datasets bundles must carry CONTEXT levers of the matching scope — they're held-fixed
  // context (managed as named environments/datasets), not model knobs.
  const scopeError = (part, wantScope) => {
    const keys = (parsed[part] || []).flatMap((b) => Object.keys(b))
    const offenders = keys.filter((k) => {
      const spec = (manifest && manifest.levers && manifest.levers[k]) || null
      return !spec || spec.scope !== wantScope
    })
    return offenders.length
      ? `"${part}" may only set ${wantScope}-scoped levers — ${[...new Set(offenders)].join(', ')} ${offenders.length === 1 ? 'is' : 'are'} not.`
      : ''
  }
  const envErr = scopeError('environments', 'environment')
  if (envErr) return { error: envErr }
  const dsErr = scopeError('datasets', 'dataset')
  if (dsErr) return { error: dsErr }
  // `compare` pits the values of ONE lever against each other (the judgeable form of "A vs B").
  if (parsed.compare !== undefined) {
    const c = parsed.compare
    if (!c || typeof c !== 'object' || Array.isArray(c) || typeof c.lever !== 'string' || !c.lever) {
      return { error: '"compare" must be an object { lever, values }.' }
    }
    if (!Array.isArray(c.values) || c.values.length < 2) {
      return { error: '"compare" needs a "values" array with at least two values to compare.' }
    }
    if (!known.has(c.lever)) {
      return { error: `compare lever "${c.lever}" is not a lever this manifest declares.` }
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
        <label class="field"><span>Spec <em>(JSON with sweep / fixed / seeds / environments / datasets / compare)</em></span>
          <textarea name="spec" rows="3" spellcheck="false">${escapeHtml(HYPOTHESIS_SPEC_PLACEHOLDER)}</textarea>
        </label>
        <div class="lever-grid" ${helpAttr('Optional — COMPARE values of one lever (A vs B) holding the rest of the spec fixed; the runs are partitioned by value and judged against each other by the Comparison below (baseline = the FIRST value). Leave blank for a plain beats-hold hypothesis.')}>
          <label class="field"><span>Compare lever <em>(optional A-vs-B)</em></span>
            <input type="text" name="compare-lever" placeholder="e.g. model_name" />
          </label>
          <label class="field"><span>Compare values <em>(comma-separated, ≥2)</em></span>
            <input type="text" name="compare-values" placeholder="e.g. ppo-custom, reppo-custom" />
          </label>
        </div>
        <div class="lever-grid" ${helpAttr('How a context-spanning (environments/datasets) OR a compare hypothesis is judged across its cells.')}>
          <label class="field"><span>Comparison <em>(context-spanning)</em></span>
            <select name="comparison-kind">
              <option value="beats-baseline">thesis beats baseline</option>
              <option value="invariant">invariant across contexts</option>
              <option value="differs">differs across contexts</option>
            </select>
          </label>
          <label class="field"><span>Baseline cell #</span>
            <input type="number" name="comparison-baseline" value="0" min="0" step="1" />
          </label>
          <label class="field"><span>Tolerance</span>
            <input type="number" name="comparison-tolerance" value="0.1" min="0" step="any" />
          </label>
        </div>
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
  const compareErr = applyCompareFromForm(form, spec)
  if (compareErr) {
    setStatusLine('hypothesis-form-error', compareErr, true)
    return
  }
  const saveBtn = byId('hypothesis-save-btn')
  if (saveBtn) saveBtn.disabled = true
  try {
    await createOrLinkHypothesis({
      title,
      rationale: String((form.elements.rationale && form.elements.rationale.value) || '').trim(),
      spec,
      comparison: readComparisonFromForm(form, spec),
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
// Merge the dedicated "Compare" inputs into the spec (so a comparison needn't be hand-written as JSON).
// Returns an error string, or null on success (mutating spec.compare). No-op when both fields are blank.
function applyCompareFromForm(form, spec) {
  const lever = String((form.elements['compare-lever'] && form.elements['compare-lever'].value) || '').trim()
  const valuesRaw = String(
    (form.elements['compare-values'] && form.elements['compare-values'].value) || '',
  ).trim()
  if (!lever && !valuesRaw) return null
  if (!lever) return 'Compare values given but no compare lever — name the lever to compare.'
  const lv = manifest && manifest.levers && manifest.levers[lever]
  if (!lv) return `Compare lever "${lever}" is not a lever this manifest declares.`
  const coerce = (v) => (lv.type === 'number' ? Number(v) : lv.type === 'boolean' ? v === 'true' : v)
  const values = valuesRaw
    .split(',')
    .map((v) => v.trim())
    .filter(Boolean)
    .map(coerce)
  if (values.length < 2) return 'Compare needs at least two comma-separated values.'
  spec.compare = { lever, values }
  return null
}
// The comparison criterion read off the form — for a context-spanning OR compare spec (else undefined, so
// a plain beats-hold hypothesis carries no comparison block).
function readComparisonFromForm(form, spec) {
  const spans =
    (spec.environments && spec.environments.length) ||
    (spec.datasets && spec.datasets.length) ||
    (spec.compare && spec.compare.lever)
  if (!spans) return undefined
  const el = (name) => form.elements[name]
  const kind = String((el('comparison-kind') && el('comparison-kind').value) || 'beats-baseline')
  const baselineIndex = Math.max(
    0,
    Math.floor(Number((el('comparison-baseline') && el('comparison-baseline').value) || 0)) || 0,
  )
  const tolerance = Number(el('comparison-tolerance') && el('comparison-tolerance').value)
  const comparison = { kind }
  if (baselineIndex) comparison.baselineIndex = baselineIndex
  if (kind !== 'beats-baseline' && Number.isFinite(tolerance)) comparison.tolerance = tolerance
  return comparison
}
// Create a hypothesis (spec-hash identity, so an identical spec from any source dedupes to one record),
// or link an EXISTING one — and, when `paperId` is given, link it to that paper. Returns the id.
async function createOrLinkHypothesis({ title, rationale, spec, comparison, source, paperId }) {
  const norm = normalizeSpec(spec)
  const id = await hashTrainingConfig(norm)
  const now = nowIso()
  const existing =
    hypothesesCache.find((x) => x.id === id) || (await readHypotheses()).find((x) => x.id === id)
  const content = existing
    ? {
        ...existing,
        comparison: comparison !== undefined ? comparison : existing.comparison,
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
        ...(comparison ? { comparison } : {}),
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
    addToggle.innerHTML = iconPlusSvg(13) + iconHypothesisSvg(14)
    addToggle.setAttribute('data-help', 'Add a hypothesis manually.')
    addToggle.addEventListener('click', () => {
      const form = byId('hypothesis-form')
      toggleHypothesisForm(!!(form && form.hidden))
    })
  }
  const minRunsInput = byId('hypothesis-min-runs')
  if (minRunsInput) {
    minRunsInput.addEventListener('change', async () => {
      await saveHypothesisMinRuns(minRunsInput.value)
      await renderHypotheses()
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
    // Track expand/collapse so it survives re-render. The `toggle` event doesn't bubble — capture it.
    // A verdict SECTION is an accordion (only one open): opening one closes the rest via re-render.
    body.addEventListener(
      'toggle',
      (event) => {
        const d = event.target
        if (!d || !d.classList || !d.dataset) return
        if (d.classList.contains('hypothesis-section') && d.dataset.section) {
          const next = d.open ? d.dataset.section : null
          if (next !== hypothesisOpenSection) {
            hypothesisOpenSection = next
            renderHypotheses()
          }
        } else if (d.classList.contains('hypothesis-card') && d.dataset.id) {
          if (d.open) hypothesisExpanded.add(d.dataset.id)
          else hypothesisExpanded.delete(d.dataset.id)
        }
      },
      true,
    )
    body.addEventListener('change', (event) => {
      const sort = event.target.closest('#hypothesis-sort-select')
      if (sort) {
        hypothesisSortKey = sort.value
        renderHypotheses()
      }
    })
    body.addEventListener('click', (event) => {
      // A control inside a <summary> must not toggle the card — it runs its own action.
      if (event.target.closest('summary') && event.target.closest('[data-action]')) {
        event.preventDefault()
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
        // Open the inline override form AND expand the card so it's visible from the collapsed state.
        if (hypothesisOverrideId === id) {
          hypothesisOverrideId = null
        } else {
          hypothesisOverrideId = id
          hypothesisExpanded.add(id)
        }
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
  shaky: 'is-shaky',
  fluff: 'is-failed',
}
const PAPER_VERDICT_LABEL = {
  untested: 'untested',
  replicating: 'replicating',
  'holds-up': 'holds up',
  shaky: 'shaky',
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
// A paper's verdict ROLLS UP from the WEIGHTED balance of its linked hypotheses' (effective) verdicts —
// holds-up / shaky / fluff / untested via the shared scorer (one set of thresholds for viewer + tests).
// `paperVerdictInfo` returns the full {status, score, counts, why} the card explains; `paperVerdict` the
// status alone (filtering + badge). The PAPER's per-hypothesis weight (default 1) lets a central claim
// dominate — weight lives on the paper, not the (shared, standalone) hypothesis.
function paperVerdictInfo(paper) {
  const ids = new Set((paper && paper.hypothesisIds) || [])
  const weights = (paper && paper.hypothesisWeights) || {}
  const linked = hypothesesCache
    .filter((h) => ids.has(h.id))
    .map((h) => ({ verdict: effectiveHypothesisVerdict(h), weight: weights[h.id], thesis: h.thesis }))
  return window.Hypothesis.scorePaperVerdict(linked)
}
function paperVerdict(paper) {
  return paperVerdictInfo(paper).status
}
// One linked-hypothesis row: title + its live verdict badge + the paper's weight chip + Unlink. `showThesis`
// adds the thesis chip (on for the flat list, off in the grouped multi-thesis view where the heading is it).
function paperHypRowHtml(paper, hid, showThesis) {
  const h = hypothesesCache.find((x) => x.id === hid)
  if (!h) return ''
  const v = effectiveHypothesisVerdict(h)
  const badge = `<span class="run-badge ${HYPOTHESIS_VERDICT_BADGE[v]}">${escapeHtml(HYPOTHESIS_VERDICT_LABEL[v])}</span>`
  const thesis = showThesis === false ? '' : hypothesisThesisChipHtml(h)
  return `<li><span class="paper-hyp-title" data-action="goto-hyp" data-id="${escapeHtml(hid)}">${escapeHtml(h.title || hid)}</span> ${badge} ${hypothesisWeightChipHtml(paper.hypothesisWeights && paper.hypothesisWeights[hid])} ${thesis} <button type="button" class="icon-btn" data-action="unlink-hyp" data-paper="${escapeHtml(paper.id)}" data-id="${escapeHtml(hid)}" title="Unlink" aria-label="Unlink">×</button></li>`
}
// The linked hypotheses, each as a row (title + its live verdict badge + Unlink), with the add/link picker.
function paperLinkedHypothesesHtml(paper) {
  const ids = Array.isArray(paper.hypothesisIds) ? paper.hypothesisIds : []
  const rows = ids.map((hid) => paperHypRowHtml(paper, hid, true)).join('')
  const list = rows
    ? `<ul class="paper-hyp-list">${rows}</ul>`
    : '<p class="card-sub">No hypotheses linked yet — Extract from the link, add one, or link an existing one.</p>'
  return `<div class="paper-hyps">${list}${paperSubformHtml(paper)}</div>`
}
// The per-paper verdict EXPLAINER: this paper's current weighted balance + the threshold ladder (what flips
// it). Pure logic in hypothesis.js (paperVerdictExplain); this wraps it. Gated by the caller on counts.total.
function paperVerdictExplainHtml(info) {
  const ex = window.Hypothesis.paperVerdictExplain(info)
  return `<div class="paper-verdict-explain card-sub"><div class="paper-verdict-formula">${escapeHtml(ex.formula)}</div><div class="paper-verdict-ladder">${escapeHtml(ex.ladder)}</div></div>`
}
// A chip flagging hypotheses blocked on an unimplemented model ("N proposals") — they can't be tested until
// the model is built, so they don't yet count toward the verdict.
function paperProposalsChipHtml(info) {
  const n = (info && info.counts && info.counts.proposed) || 0
  if (!n) return ''
  return `<span class="hyp-count-chip is-proposed"${helpAttr('Hypotheses whose required model is NOT implemented yet — they can’t be tested (or counted in the score) until it’s built.')}>${n} proposal${n === 1 ? '' : 's'}</span>`
}
// A small chip naming the PAPER thesis (claim) a hypothesis tests — so the thesis↔hypothesis link is visible
// in the flat list + the standalone Hypotheses card (the multi-thesis view groups by it instead).
function hypothesisThesisChipHtml(h) {
  const t = h && typeof h.thesis === 'string' && h.thesis.trim() ? h.thesis.trim() : ''
  if (!t) return ''
  return `<span class="hyp-thesis-chip"${helpAttr('Paper thesis this hypothesis tests: ' + t)}>◆ ${escapeHtml(t)}</span>`
}
// A warning listing paper claims NO linked hypothesis covers — found by the scrutinous re-weigh pass. A
// signal only (never gates the verdict); the user can Suggest more hypotheses to close the gap.
function paperCoverageGapsHtml(paper) {
  const gaps = Array.isArray(paper.coverageGaps) ? paper.coverageGaps.filter(Boolean) : []
  if (!gaps.length) return ''
  const items = gaps.map((g) => `<li>${escapeHtml(g)}</li>`).join('')
  return `<div class="paper-coverage-gaps card-sub"${helpAttr('Paper claims with no covering hypothesis (found when re-weighing). Use Suggest to add hypotheses that close the gap — coherent coverage makes the verdict trustworthy.')}><strong>⚠ Coverage gaps</strong> — claims no hypothesis tests yet:<ul>${items}</ul></div>`
}
// A chip flagging a MULTI-THESIS paper (the paper makes >1 distinct claim — see the per-thesis breakdown).
function paperMultiThesisChipHtml(info) {
  if (!info || !info.multiThesis) return ''
  const n = (info.theses || []).filter((t) => t.thesis).length
  return `<span class="hyp-count-chip"${helpAttr('This paper makes several distinct theses — each is scored separately below.')}>${n} theses</span>`
}
// The multi-thesis inner view: the paper's hypotheses partitioned by thesis, each group with its own verdict
// badge + why, plus an "Other (untagged)" section so no hypothesis is hidden. Only rendered for a
// multi-thesis paper; single-thesis/untagged papers use the flat paperLinkedHypothesesHtml.
function paperThesesHtml(paper, info) {
  const order = []
  const byKey = {}
  for (const hid of Array.isArray(paper.hypothesisIds) ? paper.hypothesisIds : []) {
    const h = hypothesesCache.find((x) => x.id === hid)
    if (!h) continue
    const label = typeof h.thesis === 'string' && h.thesis.trim() ? h.thesis.trim() : null
    const key = label === null ? ' untagged' : label
    if (!byKey[key]) {
      byKey[key] = { label, hids: [] }
      order.push(key)
    }
    byKey[key].hids.push(hid)
  }
  const detailByKey = {}
  for (const t of info.theses || [])
    detailByKey[t.thesis === null ? ' untagged' : t.thesis] = t.detail
  const sections = order
    .map((key) => {
      const g = byKey[key]
      const detail = detailByKey[key]
      const verdict = detail ? detail.status : 'untested'
      const badge = `<span class="run-badge ${PAPER_VERDICT_BADGE[verdict]}"${detail ? helpAttr(detail.why) : ''}>${escapeHtml(PAPER_VERDICT_LABEL[verdict])}</span>`
      const heading = g.label || 'Other (untagged hypotheses)'
      // The per-thesis weighted score (its own hypotheses only) — the same proven÷decided formula as the paper.
      const score =
        detail && detail.counts && detail.counts.total ? paperVerdictExplainHtml(detail) : ''
      // Rows WITHOUT the thesis chip — the heading already names the thesis.
      const rows = g.hids.map((hid) => paperHypRowHtml(paper, hid, false)).join('')
      return `<div class="paper-thesis"><div class="paper-thesis-head"><span class="paper-thesis-label">${escapeHtml(heading)}</span> ${badge}</div>${score}<ul class="paper-hyp-list">${rows}</ul></div>`
    })
    .join('')
  return `<div class="paper-hyps paper-theses"><p class="card-sub paper-theses-intro">This paper makes ${order.filter((k) => byKey[k].label).length} distinct theses — each scored separately below:</p>${sections}${paperSubformHtml(paper)}</div>`
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
// Chip showing how many hypotheses the paper links — highlighted when none, so an empty paper stands out.
// Drop links to hypotheses that no longer exist (deleted) from every paper, persisting the cleaned list
// + updating papersCache so the count chip + rolled-up verdict are correct. Idempotent (a no-op once clean).
async function prunePaperHypothesisLinks() {
  const live = new Set(hypothesesCache.map((h) => h.id))
  for (const p of papersCache) {
    const ids = Array.isArray(p.hypothesisIds) ? p.hypothesisIds : []
    const kept = ids.filter((id) => live.has(id))
    if (kept.length === ids.length) continue
    p.hypothesisIds = kept
    try {
      await putPaper({ ...p, hypothesisIds: kept })
    } catch {
      // best-effort: a failed cleanup write just retries next render
    }
  }
}
function paperHypCountChipHtml(paper) {
  const n = Array.isArray(paper.hypothesisIds) ? paper.hypothesisIds.length : 0
  const label = `${n} hypoth${n === 1 ? 'esis' : 'eses'}`
  const help = n
    ? 'Hypotheses linked to this paper — the verdict rolls up from them.'
    : 'No hypotheses yet — Suggest, Add, or Link one so this paper can be judged.'
  return `<span class="hyp-count-chip${n ? '' : ' is-empty'}"${helpAttr(help)}>${iconHypothesisSvg(12)} ${label}</span>`
}
function paperCardHtml(paper) {
  const info = paperVerdictInfo(paper)
  const verdict = info.status
  const badge = `<span class="run-badge ${PAPER_VERDICT_BADGE[verdict]}"${helpAttr(info.why)}>${escapeHtml(PAPER_VERDICT_LABEL[verdict])}</span>`
  const title = paper.url
    ? `<a href="${escapeHtml(paper.url)}" target="_blank" rel="noopener">${escapeHtml(paper.title || 'Untitled')}</a>`
    : escapeHtml(paper.title || 'Untitled')
  const meta = [paper.authors, paper.year]
    .filter(Boolean)
    .map((x) => escapeHtml(String(x)))
    .join(' · ')
  const id = escapeHtml(paper.id)
  const linkedCount = Array.isArray(paper.hypothesisIds) ? paper.hypothesisIds.length : 0
  const open = paperExpanded.has(paper.id) ? ' open' : ''
  const suggestPending = paperOpPending('suggest-paper-hypotheses', paper.id)
  const findPending = paperOpPending('analyze-paper-models', paper.id)
  const reweighPending = paperOpPending('weigh-paper-hypotheses', paper.id)
  // Icon-only action buttons (verb + hypothesis glyph), shown collapsed AND expanded; tooltips on hover.
  // A per-paper LLM op (suggest / find-models / re-weigh) shows a spinner + disables ITS button until it
  // settles — the user can still launch the other ops, or the same op on other papers (those queue).
  const actions =
    `<button type="button" class="card-btn combo" data-action="suggest-hyp" data-id="${id}"${suggestPending ? ' disabled' : ''}${helpAttr('Suggest hypotheses — an LLM matches existing ones to this paper and proposes new ones, all auto-linked.')}>${suggestPending ? spinnerHtml() : iconLightbulbSvg(13) + iconHypothesisSvg(14)}</button>` +
    (linkedCount
      ? `<button type="button" class="card-btn combo" data-action="reweigh-hyps" data-id="${id}"${reweighPending ? ' disabled' : ''}${helpAttr('Re-weigh hypotheses — an LLM rates each linked hypothesis by importance (the paper’s central claim heavy, supporting ones light); the verdict then rolls up by weight.')}>${reweighPending ? spinnerHtml() : '<span class="card-btn-glyph">⚖</span>' + iconHypothesisSvg(14)}</button>`
      : '') +
    `<button type="button" class="card-btn combo" data-action="add-hyp" data-id="${id}"${helpAttr('Add a hypothesis manually and link it to this paper.')}>${iconPlusSvg(13)}${iconHypothesisSvg(14)}</button>` +
    `<button type="button" class="card-btn combo" data-action="link-existing" data-id="${id}"${helpAttr('Link an existing hypothesis to this paper.')}>${iconLinkSvg(13)}${iconHypothesisSvg(14)}</button>` +
    `<button type="button" class="card-btn" data-action="find-models" data-id="${id}"${findPending ? ' disabled' : ''}${helpAttr('Find models — an LLM links this paper to the catalog models it introduces/improves and proposes any missing ones to add.')}>${findPending ? spinnerHtml() : iconModelSvg(14)}</button>` +
    (linkedCount
      ? `<button type="button" class="card-btn" data-action="replicate" data-id="${id}"${helpAttr('Replicate — launch every linked hypothesis’s spec.')}>${iconRunSvg(15)}</button>`
      : '') +
    `<button type="button" class="card-btn" data-action="edit" data-id="${id}"${helpAttr('Edit this paper.')}>${iconEditSvg(15)}</button>` +
    `<button type="button" class="card-btn" data-action="toggle-dismiss" data-id="${id}"${helpAttr(paper.dismissed ? 'Restore — show this paper in the list again.' : 'Not wanted — hide this paper from the list (without deleting it).')}>${iconBanSvg(14)}</button>` +
    `<button type="button" class="card-btn card-btn-danger" data-action="delete" data-id="${id}"${helpAttr('Delete this paper.')}>${iconDeleteSvg()}</button>`
  return `<details class="paper-card" data-id="${id}"${open}>
    <summary class="paper-summary">
      ${badge}
      ${paperMultiThesisChipHtml(info)}
      ${paperHypCountChipHtml(paper)}
      ${paperProposalsChipHtml(info)}
      <span class="paper-summary-title">${title}</span>
      <span class="card-actions">${actions}</span>
    </summary>
    <div class="paper-body">
      ${meta ? `<p class="card-sub">${meta}</p>` : ''}
      ${paper.claim ? `<p class="paper-claim">${escapeHtml(paper.claim)}</p>` : ''}
      ${paperAssumptionChips(paper.assumptions)}
      ${info.counts.total ? `<p class="card-sub paper-verdict-why">${escapeHtml(info.why)}</p>` : ''}
      ${info.counts.total ? paperVerdictExplainHtml(info) : ''}
      ${paperCoverageGapsHtml(paper)}
      ${info.multiThesis ? paperThesesHtml(paper, info) : paperLinkedHypothesesHtml(paper)}
      ${paperModelsHtml(paper)}
      ${paper.verdictNote ? `<p class="card-sub paper-note">${escapeHtml(paper.verdictNote)}</p>` : ''}
    </div>
  </details>`
}
// Order papers for a FULL render. Default 'status' = rolled-up verdict bucket, then newest-first within it
// (stable across an update, so a changed card never jumps); 'name' / 'year' are explicit user picks.
const PAPER_VERDICT_SORT = { 'holds-up': 0, shaky: 1, replicating: 2, untested: 3, fluff: 4 }
function sortPapers(list) {
  const arr = [...list]
  if (paperSortKey === 'name') {
    arr.sort((a, b) => String(a.title || '').localeCompare(String(b.title || '')))
  } else if (paperSortKey === 'year') {
    arr.sort(
      (a, b) =>
        (Number(b.year) || 0) - (Number(a.year) || 0) ||
        String(a.title || '').localeCompare(String(b.title || '')),
    )
  } else {
    arr.sort((a, b) => {
      const va = PAPER_VERDICT_SORT[paperVerdict(a)] ?? 9
      const vb = PAPER_VERDICT_SORT[paperVerdict(b)] ?? 9
      if (va !== vb) return va - vb
      return String(b.createdAt || '').localeCompare(String(a.createdAt || ''))
    })
  }
  return arr
}
// Reusable labelled sort dropdown (the shared .app-select look) — "Sort <select>".
function sortControlHtml(id, options, current) {
  const opts = options
    .map(([v, l]) => `<option value="${escapeHtml(v)}"${v === current ? ' selected' : ''}>${escapeHtml(l)}</option>`)
    .join('')
  return `<label class="sort-control">Sort <select id="${escapeHtml(id)}" class="app-select">${opts}</select></label>`
}
function sortHypotheses(list) {
  const arr = [...list]
  if (hypothesisSortKey === 'name')
    arr.sort((a, b) => String(a.title || a.id).localeCompare(String(b.title || b.id)))
  else if (hypothesisSortKey === 'oldest')
    arr.sort((a, b) => String(a.updatedAt || '').localeCompare(String(b.updatedAt || '')))
  else arr.sort((a, b) => String(b.updatedAt || '').localeCompare(String(a.updatedAt || '')))
  return arr
}
// Comparator for the model catalog (sorts WITHIN each category group).
function modelSortComparator() {
  const byName = (a, b) => String(a.name || a.id).localeCompare(String(b.name || b.id))
  if (modelSortKey === 'runs')
    return (a, b) => {
      const ra = modelAgg(a)
      const rb = modelAgg(b)
      return ((rb ? rb.runs : 0) || 0) - ((ra ? ra.runs : 0) || 0) || byName(a, b)
    }
  if (modelSortKey === 'best')
    return (a, b) => {
      const aa = modelAgg(a)
      const bb = modelAgg(b)
      const va = aa && aa.best != null ? aa.best : -Infinity
      const vb = bb && bb.best != null ? bb.best : -Infinity
      return vb - va || byName(a, b)
    }
  if (modelSortKey === 'status')
    return (a, b) => String(modelStatusOf(a)).localeCompare(String(modelStatusOf(b))) || byName(a, b)
  return byName
}
function paperFilterBarHtml() {
  const opts = ['all', 'untested', 'holds-up', 'shaky', 'fluff']
  const active = papersCache.filter((p) => !p.dismissed)
  const btns = opts
    .map((s) => {
      const label = s === 'all' ? 'all' : PAPER_VERDICT_LABEL[s]
      const count = s === 'all' ? active.length : active.filter((p) => paperVerdict(p) === s).length
      return `<button type="button" class="paper-filter-btn${paperVerdictFilter === s ? ' is-active' : ''}" data-verdict="${escapeHtml(s)}">${escapeHtml(label)} (${count})</button>`
    })
    .join('')
  const dismissedCount = papersCache.filter((p) => p.dismissed).length
  const notWanted = dismissedCount
    ? `<button type="button" class="paper-filter-btn${paperVerdictFilter === 'dismissed' ? ' is-active' : ''}" data-verdict="dismissed">not wanted (${dismissedCount})</button>`
    : ''
  const sortOpts = [
    ['status', 'Status'],
    ['name', 'Name'],
    ['year', 'Year'],
  ]
    .map(([v, l]) => `<option value="${v}"${paperSortKey === v ? ' selected' : ''}>${l}</option>`)
    .join('')
  const sort = `<label class="sort-control">Sort <select id="paper-sort-select" class="app-select" aria-label="Sort papers">${sortOpts}</select></label>`
  const search = `<input type="search" id="paper-search-text" class="paper-search-text" placeholder="Search name, title, abstract…" value="${escapeHtml(paperSearch)}" aria-label="Search papers" />`
  return `<div class="paper-filter">${search}${btns}${notWanted}${sort}</div>`
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
  // Load hypotheses + models alongside papers for the roll-up + linked-models section. The paper verdict
  // rolls up from each hypothesis's PERSISTED verdict (the shared all-runs refresh keeps those fresh).
  ;[papersCache, hypothesesCache, modelsCache] = await Promise.all([
    readPapers(),
    readHypotheses(),
    readModels(),
  ])
  // Self-heal: a paper may reference hypotheses since deleted — drop those stale links (persisted) so the
  // count chip + rolled-up verdict reflect reality.
  await prunePaperHypothesisLinks()
  renderPapersList()
}
// The blob a paper is searched against — id (so "bysik" finds it), title, authors, year, claim/abstract,
// approach, verdict note, tags, and the assumptions notes.
function paperSearchText(p) {
  return [
    p.id,
    p.title,
    p.authors,
    p.year,
    p.claim,
    p.approach,
    p.verdictNote,
    Array.isArray(p.tags) ? p.tags.join(' ') : '',
    (p.assumptions && p.assumptions.notes) || '',
  ]
    .filter(Boolean)
    .join(' ')
    .toLowerCase()
}
// Render the list from the CURRENT caches (no fetch) — used by renderPapers, the verdict filter, and the
// live search box, so typing re-filters instantly without re-querying the store.
function renderPapersList() {
  const body = byId('papers-body')
  if (!body) return
  const seedBanner = pendingSeedPapersHtml()
  if (!papersCache.length) {
    setHtml(
      body,
      seedBanner +
        '<div class="empty-hint">No approaches yet — add a paper/method, or import the curated starter set.</div>',
    )
    return
  }
  const q = paperSearch.trim().toLowerCase()
  const matchesSearch = (p) => !q || paperSearchText(p).indexOf(q) >= 0
  // "not wanted" papers are hidden from every verdict view except the dedicated 'dismissed' filter.
  const shown = (
    paperVerdictFilter === 'dismissed'
      ? papersCache.filter((p) => p.dismissed)
      : papersCache.filter(
          (p) =>
            !p.dismissed &&
            (paperVerdictFilter === 'all' || paperVerdict(p) === paperVerdictFilter),
        )
  ).filter(matchesSearch)
  const sorted = sortPapers(shown)
  setHtml(
    body,
    seedBanner +
      paperFilterBarHtml() +
      (sorted.length
        ? sorted.map(paperCardHtml).join('')
        : `<div class="empty-hint">${q ? 'No approaches match your search.' : 'No approaches match this view.'}</div>`),
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
// Suggest hypotheses: an LLM matches existing hypotheses to the paper + proposes new ones (all
// auto-linked). Triggers the backend `suggest-paper-hypotheses` activity, then re-renders Papers.
// A per-paper LLM op keyed for the in-flight Set, so its button can spin until the activity settles.
function paperOpKey(activityType, paperId) {
  return activityType + ':' + paperId
}
function paperOpPending(activityType, paperId) {
  return pendingPaperOps.has(paperOpKey(activityType, paperId))
}
// Launch a per-paper LLM op WITHOUT blocking: mark it pending now (covers the queued wait), start/enqueue,
// and let the shared observe lifecycle (applyQuickDispatchState / refreshAfterQuickDispatch) clear it +
// update the card on settle. The user can keep launching on other papers freely (they queue past the limit).
async function launchPaperOp(activityType, paperId, label) {
  if (!embedded()) {
    setStatusLine('papers-status', 'Open inside the Overseer to run this.', true)
    return
  }
  if (paperOpPending(activityType, paperId)) return
  const epoch = projectEpoch
  setStatusLine('papers-status', '')
  pendingPaperOps.add(paperOpKey(activityType, paperId))
  if (activeTabId === 'papers') updatePaperCard(paperId)
  try {
    const result = await startOrEnqueue(activityType, trainerActivityParams({ paperId }), label)
    if (result.queued && epoch === projectEpoch) {
      setStatusLine('papers-status', queuedStatusText(result.ahead))
    }
  } catch {
    pendingPaperOps.delete(paperOpKey(activityType, paperId))
    if (activeTabId === 'papers') updatePaperCard(paperId)
    if (epoch === projectEpoch)
      setStatusLine('papers-status', 'Could not start — please try again.', true)
  }
}
async function onSuggestPaperHypotheses(id) {
  await launchPaperOp('suggest-paper-hypotheses', id, 'Suggest hypotheses')
}
async function onReweighPaperHypotheses(id) {
  await launchPaperOp('weigh-paper-hypotheses', id, 'Re-weigh hypotheses')
}
async function onFindPaperModels(id) {
  await launchPaperOp('analyze-paper-models', id, 'Find models')
}
// Settle handler for Find-models: reload the paper + models, surface the missing-model proposals, and
// toast a clickable "view" that focuses the card.
async function loadPaperModelsResult(pid) {
  papersCache = await readPapers()
  modelsCache = await readModels()
  const recs = await queryRecords(manifest.recordType + '-model-analysis', 'latest')
  const analysis = recs && recs[0] && recs[0].content
  const missing =
    analysis && analysis.paperId === pid && Array.isArray(analysis.missingModels)
      ? analysis.missingModels
      : []
  paperMissingModels.set(pid, missing)
  paperExpanded.add(pid)
  showToast(
    missing.length
      ? `Found ${missing.length} model${missing.length === 1 ? '' : 's'} to add on this paper — view`
      : 'Linked the paper to its catalog models — view',
    () => focusPaper(pid),
  )
}
// Re-render ONE paper card in place (no list re-sort, no full-tab flash) — used when an async op settles.
function updatePaperCard(pid) {
  const body = byId('papers-body')
  if (!body) return
  const paper = papersCache.find((p) => p.id === pid)
  const el = [...body.querySelectorAll('details.paper-card')].find((e) => e.dataset.id === pid)
  if (!el) return
  if (!paper || (paper.dismissed && paperVerdictFilter !== 'dismissed')) {
    el.remove()
    return
  }
  const wasOpen = el.open || paperExpanded.has(pid)
  const tmp = document.createElement('div')
  tmp.innerHTML = paperCardHtml(paper)
  const fresh = tmp.firstElementChild
  if (fresh) {
    el.replaceWith(fresh)
    fresh.open = wasOpen
  }
}
// Bring a paper into view: expand it + (on the Papers tab) update + scroll to it; else switch tabs.
function focusPaper(pid) {
  paperExpanded.add(pid)
  if (activeTabId !== 'papers') {
    showTab('papers')
    return
  }
  updatePaperCard(pid)
  const body = byId('papers-body')
  const el =
    body && [...body.querySelectorAll('details.paper-card')].find((e) => e.dataset.id === pid)
  if (el) {
    el.open = true
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }
}
// Bring a model card into view: clear any category filter (so a component card isn't hidden), expand it,
// re-render the Models tab + scroll to it. Used by the flavor block chips / component "used by" chips.
async function focusModel(slug) {
  modelExpanded.add(slug)
  if (activeTabId !== 'models') {
    showTab('models')
    return
  }
  modelCategoryFilter = 'all'
  await renderModels()
  const body = byId('models-body')
  const el =
    body && [...body.querySelectorAll('details.model-card')].find((e) => e.dataset.id === slug)
  if (el) {
    el.open = true
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }
}
async function onToggleDismissPaper(id) {
  const p = papersCache.find((x) => x.id === id)
  if (!p) return
  try {
    await putPaper({ ...p, dismissed: !p.dismissed })
  } catch {
    setStatusLine('papers-status', 'Could not update — please try again.', true)
    return
  }
  p.dismissed = !p.dismissed
  await renderPapers()
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
    // Track expand/collapse so it survives re-render. The `toggle` event doesn't bubble — capture it.
    body.addEventListener(
      'toggle',
      (event) => {
        const d = event.target
        if (!d || !d.classList || !d.classList.contains('paper-card') || !d.dataset.id) return
        if (d.open) paperExpanded.add(d.dataset.id)
        else paperExpanded.delete(d.dataset.id)
      },
      true,
    )
    body.addEventListener('change', (event) => {
      const sort = event.target.closest('#paper-sort-select')
      if (sort) {
        paperSortKey = sort.value
        renderPapers()
      }
    })
    body.addEventListener('input', (event) => {
      if (event.target.id !== 'paper-search-text') return
      paperSearch = event.target.value
      const pos = event.target.selectionStart
      renderPapersList()
      const el = byId('paper-search-text')
      if (el) {
        el.focus()
        try {
          el.setSelectionRange(pos, pos)
        } catch {
          // some input types disallow setSelectionRange — focus alone is fine
        }
      }
    })
    body.addEventListener('click', (event) => {
      // A control inside a <summary> must not toggle the card — it runs its own action.
      if (event.target.closest('summary') && event.target.closest('[data-action]')) {
        event.preventDefault()
      }
      const filterBtn = event.target.closest('button[data-verdict]')
      if (filterBtn) {
        paperVerdictFilter = filterBtn.dataset.verdict
        renderPapersList()
        return
      }
      const chip = event.target.closest('[data-action="goto-hyp"]')
      if (chip) {
        showTab('hypotheses')
        return
      }
      const modelChip = event.target.closest('[data-action="goto-model"]')
      if (modelChip) {
        showTab('models')
        return
      }
      const btn = event.target.closest('button[data-action]')
      if (!btn) return
      const { action, id } = btn.dataset
      const paperId = btn.dataset.paper
      if (action === 'import-seeds') importSeedPapers()
      else if (action === 'replicate') replicatePaper(id)
      else if (action === 'find-models') onFindPaperModels(id)
      else if (action === 'add-model') addProposedModelToCatalog(paperId, id)
      else if (action === 'toggle-dismiss') onToggleDismissPaper(id)
      else if (action === 'suggest-hyp') onSuggestPaperHypotheses(id)
      else if (action === 'reweigh-hyps') onReweighPaperHypotheses(id)
      else if (action === 'add-hyp') {
        // Open the add sub-form AND expand the card so it's visible from the collapsed state.
        if (paperSubform && paperSubform.paperId === id && paperSubform.mode === 'add') {
          paperSubform = null
        } else {
          paperSubform = { paperId: id, mode: 'add' }
          paperExpanded.add(id)
        }
        renderPapers()
      } else if (action === 'link-existing') {
        if (paperSubform && paperSubform.paperId === id && paperSubform.mode === 'link') {
          paperSubform = null
        } else {
          paperSubform = { paperId: id, mode: 'link' }
          paperExpanded.add(id)
        }
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

// --- Models tab ----------------------------------------------------------------
// The catalog of model architectures the project can train. A model OWNS its runs by binding model_name
// lever values; its status (proposed/implemented/failing) auto-derives from those runs (viewer/models.js).
// It LINKS the papers that introduce/improve it + the hypotheses that test it. "Discuss" seeds a project
// chat to implement / improve / fix it; "Scan Project" + the Papers tab's "Find models" populate it.
async function readModels() {
  if (!manifest) return []
  const recs = await queryRecords(manifest.recordType + MODEL_RECORD_SUFFIX)
  return recs.map((r) => r.content).filter((c) => c && c.id)
}
async function putModel(model) {
  await window.OverseerBridge.putData({
    type: manifest.recordType + MODEL_RECORD_SUFFIX,
    key: model.id,
    content: { ...model, updatedAt: nowIso() },
  })
}
async function deleteModelRecord(id) {
  await window.OverseerBridge.deleteData({
    type: manifest.recordType + MODEL_RECORD_SUFFIX,
    key: id,
  })
}
// Starter models the manifest ships (parallel to papers); imported into the catalog once by id/slug.
function manifestModelSeeds() {
  return manifest && Array.isArray(manifest.models) ? manifest.models : []
}
// Manifest seeds that are NEW or OUT OF SYNC (a manifest-owned field — esp. the model_name bindings —
// differs from the imported record). Syncing re-writes those fields while preserving the user's status
// override / notes / dismissed / hypothesis links — so a consolidation reaches already-imported catalogs.
function pendingSeedModels() {
  const byId = new Map(modelsCache.map((m) => [m.id, m]))
  return manifestModelSeeds().filter(
    (s) =>
      s &&
      s.id &&
      // A seed merged away into another model (now one of its aliases) is never re-added.
      !window.Models.seedClaimedByAlias(s, modelsCache) &&
      window.Models.seedDiffersFromModel(s, byId.get(s.id)),
  )
}
function pendingSeedModelsHtml() {
  const pending = pendingSeedModels()
  if (!pending.length) return ''
  const plural = pending.length === 1 ? '' : 's'
  return `<div class="paper-seed-banner">${pending.length} model${plural} from the manifest ${pending.length === 1 ? 'is' : 'are'} new or out of sync (e.g. updated run bindings). <button type="button" class="ghost-btn" data-action="import-seeds">Sync ${pending.length}</button></div>`
}
async function importSeedModels() {
  const pending = pendingSeedModels()
  if (!pending.length) return
  const byId = new Map(modelsCache.map((m) => [m.id, m]))
  try {
    for (const s of pending) {
      await putModel(window.Models.mergeSeedIntoModel(s, byId.get(s.id), nowIso()))
    }
  } catch {
    setStatusLine('models-status', 'Could not sync models — please try again.', true)
    return
  }
  setStatusLine(
    'models-status',
    `Synced ${pending.length} model${pending.length === 1 ? '' : 's'} from the manifest.`,
    false,
  )
  await renderModels()
}
// The model's all-runs aggregate ({runs,best,failing,lastRunAt,perFlavor}) from the persisted stats, or null.
function modelAgg(model) {
  return window.Models.aggForModel(modelStatsCache, model.id)
}
// The auto-derived (or manually pinned) lifecycle status of a model, from its all-runs aggregate.
function modelStatusOf(model) {
  return window.Models.deriveModelStatus(model, modelAgg(model), manifest)
}
function modelStatusBadgeHtml(model) {
  const st = modelStatusOf(model)
  const cls = window.Models.MODEL_STATUS_BADGE[st] || 'is-queued'
  const label = window.Models.MODEL_STATUS_LABEL[st] || st
  return `<span class="run-badge ${cls}">${escapeHtml(label)}</span>`
}
function modelRunChipHtml(model) {
  // A component's runs are attributed to the models that compose it, never to the component itself — so it
  // carries no runs chip (use the model card's "used by" to see where it's consumed).
  if (model.category === 'component') return ''
  const s = modelAgg(model)
  if (!modelStatsCache) {
    return `<span class="hyp-count-chip is-empty"${helpAttr('Run counts are computed over ALL runs — press Refresh.')}>— runs</span>`
  }
  if (!s || !s.runs) {
    return `<span class="hyp-count-chip is-empty"${helpAttr('No runs have trained this model yet.')}>0 runs</span>`
  }
  const parts = [`${s.runs} run${s.runs === 1 ? '' : 's'}`]
  if (s.best !== null && s.best !== undefined) parts.push(`best ${formatObjective(s.best)}`)
  if (s.failing) parts.push(`${s.failing} failing`)
  return `<span class="hyp-count-chip"${helpAttr('Every run binding this model (all flavors), aggregated over ALL runs.')}>${escapeHtml(parts.join(' · '))}</span>`
}
// The model's flavors, each with its per-flavor run count (from the aggregate) + any flavor-specific links.
function modelFlavorsHtml(model) {
  const flavors = window.Models.modelFlavors(model)
  if (!flavors.length) return ''
  const agg = modelAgg(model)
  const rows = flavors
    .map((fl) => {
      const per = agg && agg.perFlavor && agg.perFlavor[window.Models.flavorKey(fl)]
      const count = per ? per.runs : modelStatsCache ? 0 : null
      const chip =
        count === null
          ? ''
          : ` <span class="flavor-runs${count ? '' : ' is-empty'}">${count} run${count === 1 ? '' : 's'}</span>`
      const cfg =
        fl.config && Object.keys(fl.config).length
          ? ` <span class="card-sub">(${escapeHtml(
              Object.entries(fl.config)
                .map(([k, v]) => `${k}=${v}`)
                .join(', '),
            )})</span>`
          : ''
      const label = escapeHtml(fl.name || fl.modelName)
      return `<li><code>${escapeHtml(fl.modelName)}</code> ${label}${cfg}${chip}${flavorComponentsHtml(fl)}</li>`
    })
    .join('')
  return `<div class="model-flavors"><span class="card-sub">Flavors</span><ul class="paper-hyp-list">${rows}</ul></div>`
}
// The building blocks a flavor is composed of, as chips — known components link to their catalog entry,
// unknown slugs render as a plain chip. Empty when the flavor declares no `components`.
function flavorComponentsHtml(flavor) {
  const comps = window.Models.flavorComponents(flavor, modelsCache)
  if (!comps.length) return ''
  const chips = comps
    .map((c) =>
      c.found
        ? `<button type="button" class="block-chip" data-action="open-model" data-id="${escapeHtml(c.slug)}"${helpAttr('Building block — open its catalog entry')}>${escapeHtml(c.name)}</button>`
        : `<span class="block-chip is-missing"${helpAttr('Declared block with no catalog entry')}>${escapeHtml(c.name)}</span>`,
    )
    .join('')
  return `<div class="flavor-components"><span class="block-chip-label">blocks</span>${chips}</div>`
}
// On a component card: the models whose flavors are built from it (reverse of the flavor block chips).
function modelUsedByHtml(model) {
  if (!model || model.category !== 'component') return ''
  const users = window.Models.modelsUsingComponent(model.slug, modelsCache)
  if (!users.length) return ''
  const chips = users
    .map(
      (m) =>
        `<button type="button" class="block-chip" data-action="open-model" data-id="${escapeHtml(m.slug)}"${helpAttr('Open this model')}>${escapeHtml(m.name || m.slug)}</button>`,
    )
    .join('')
  return `<div class="model-used-by"><span class="card-sub">Used by</span><div class="flavor-components">${chips}</div></div>`
}
function modelLinkedPapersHtml(model) {
  const papers = window.Models.papersForModel(model, papersCache)
  if (!papers.length) return ''
  const rows = papers
    .map(
      (p) =>
        `<li><span class="paper-hyp-title" data-action="goto-paper" data-id="${escapeHtml(p.id)}">${escapeHtml(p.title || p.id)}</span></li>`,
    )
    .join('')
  return `<div class="model-links"><span class="card-sub">Papers</span><ul class="paper-hyp-list">${rows}</ul></div>`
}
function modelLinkedHypothesesHtml(model) {
  const hyps = window.Models.hypothesesForModel(model, hypothesesCache)
  if (!hyps.length) return ''
  const rows = hyps
    .map((h) => {
      const v = effectiveHypothesisVerdict(h)
      const badge = `<span class="run-badge ${HYPOTHESIS_VERDICT_BADGE[v]}">${escapeHtml(HYPOTHESIS_VERDICT_LABEL[v])}</span>`
      return `<li><span class="paper-hyp-title" data-action="goto-hyp" data-id="${escapeHtml(h.id)}">${escapeHtml(h.title || h.id)}</span> ${badge}</li>`
    })
    .join('')
  return `<div class="model-links"><span class="card-sub">Hypotheses</span><ul class="paper-hyp-list">${rows}</ul></div>`
}
// A manual status override (or 'auto' to let runs drive it) — the "what needs adding/improving" control.
function modelStatusSelectHtml(model) {
  const cur = model.statusSource === 'manual' ? model.status : 'auto'
  const opts = ['auto'].concat(window.Models.MODEL_STATUSES)
  const sel = opts
    .map(
      (s) =>
        `<option value="${escapeHtml(s)}"${s === cur ? ' selected' : ''}>${s === 'auto' ? 'auto (from runs)' : escapeHtml(window.Models.MODEL_STATUS_LABEL[s] || s)}</option>`,
    )
    .join('')
  return `<label class="field model-status-field"><span>Status</span><select data-action="set-status" data-id="${escapeHtml(model.id)}">${sel}</select></label>`
}
// Whether this model has a CPU-vs-MPS device benchmark in flight — RUNNING (busy, set by the activity
// lifecycle in applyQuickDispatchState) OR still QUEUED (waiting for a slot). Both must spin the chip.
function modelBenchmarkPending(modelId) {
  if (benchmarkingModels.has(modelId)) return true
  return queueCache.some(
    (q) => q.activityType === 'benchmark-model-device' && q.params && q.params.modelId === modelId,
  )
}
// The device chip shown next to the implementation-status badge: the measured CPU-vs-MPS winner, a
// spinner while a benchmark is queued/running, or a "check" affordance when never measured. The chip IS
// the trigger (click to (re-)benchmark) — there is no separate button.
function modelDeviceChipHtml(model) {
  // A component is a building block composed INTO models — it has no run/device of its own (those belong to
  // the models/baselines that use it), so it shows no device chip.
  if (model.category === 'component') return ''
  const id = escapeHtml(model.id)
  if (modelBenchmarkPending(model.id)) {
    return `<span class="run-badge is-running" title="Benchmarking CPU vs MPS…">${spinnerHtml()} device</span>`
  }
  if (!embedded()) {
    return model.preferredDevice
      ? `<span class="run-badge">${escapeHtml(String(model.preferredDevice).toUpperCase())}</span>`
      : ''
  }
  const b = model.deviceBenchmark
  if (model.preferredDevice) {
    const dev = String(model.preferredDevice).toUpperCase()
    const cls = window.Models.deviceChipClass(model.preferredDevice)
    const margin = b && b.speedup && b.speedup > 1.01 ? `, ${b.speedup}× faster` : ''
    const per =
      b && b.usPerStep
        ? ' — ' +
          Object.entries(b.usPerStep)
            .map(([d, us]) => `${d} ${Math.round(us)}µs/step`)
            .join(', ')
        : ''
    const title = `Fastest device (measured): ${dev}${margin}.${per} Click for the exact per-device timings.`
    return `<button type="button" class="run-badge model-device-chip ${cls}" data-action="device-timings" data-id="${id}"${helpAttr(title)}>${escapeHtml(dev)}</button>`
  }
  return `<button type="button" class="run-badge is-queued model-device-chip" data-action="benchmark-device" data-id="${id}"${helpAttr('Benchmark this model on CPU vs MPS to set its faster device')}>⚡ check device</button>`
}
// The other names this model is known by (from a merge or a manifest seed) — papers/scans/seeds that refer
// to one resolve to this model instead of creating a duplicate.
function modelAliasesHtml(model) {
  const aliases = Array.isArray(model.aliases) ? model.aliases.filter(Boolean) : []
  if (!aliases.length) return ''
  return `<p class="card-sub model-aliases"><strong>Also known as:</strong> ${aliases
    .map((a) => `<code>${escapeHtml(a)}</code>`)
    .join(' ')}</p>`
}
function modelCardHtml(model) {
  const id = escapeHtml(model.id)
  const open = modelExpanded.has(model.id) ? ' open' : ''
  const flavorCount = window.Models.modelFlavors(model).length
  const st = modelStatusOf(model)
  const discussTitle =
    st === 'failing'
      ? 'Discuss + fix this failing model with the AI'
      : st === 'proposed'
        ? 'Discuss implementing this model with the AI'
        : 'Discuss / improve this model with the AI'
  const actions =
    (chatAboutRunAvailable()
      ? `<button type="button" class="card-btn" data-action="discuss-model" data-id="${id}"${helpAttr(discussTitle)}>${iconChatSvg()}</button>`
      : '') +
    `<button type="button" class="card-btn card-btn-danger" data-action="delete-model" data-id="${id}"${helpAttr('Remove this model from the catalog.')}>${iconDeleteSvg()}</button>`
  return `<details class="paper-card model-card" data-id="${id}"${open}>
    <summary class="paper-summary">
      ${modelStatusBadgeHtml(model)}
      ${modelDeviceChipHtml(model)}
      ${modelRunChipHtml(model)}
      <span class="paper-summary-title">${escapeHtml(model.name || model.id)}</span>
      <span class="card-actions">${actions}</span>
    </summary>
    <div class="paper-body">
      ${model.description ? `<p class="paper-claim">${escapeHtml(model.description)}</p>` : ''}
      ${modelAliasesHtml(model)}
      ${model.proposal ? `<p class="card-sub paper-note"><strong>To add:</strong> ${escapeHtml(model.proposal)}</p>` : ''}
      ${flavorCount || model.category === 'component' ? '' : '<p class="card-sub">No flavor yet — a proposal not wired into any run config.</p>'}
      ${model.implPath ? `<p class="card-sub">Code: <code>${escapeHtml(model.implPath)}</code></p>` : ''}
      ${modelFlavorsHtml(model)}
      ${modelUsedByHtml(model)}
      ${modelStatusSelectHtml(model)}
      ${modelLinkedPapersHtml(model)}
      ${modelLinkedHypothesesHtml(model)}
    </div>
  </details>`
}
function modelCategoryFilterBarHtml() {
  const cats = ['all'].concat(window.Models.MODEL_CATEGORIES)
  const btns = cats
    .map((c) => {
      const count =
        c === 'all' ? modelsCache.length : modelsCache.filter((m) => m.category === c).length
      if (c !== 'all' && !count) return ''
      const label = c === 'all' ? 'all' : window.Models.MODEL_CATEGORY_LABEL[c] || c
      return `<button type="button" class="paper-filter-btn${modelCategoryFilter === c ? ' is-active' : ''}" data-category="${escapeHtml(c)}">${escapeHtml(label)} (${count})</button>`
    })
    .join('')
  const sort = sortControlHtml(
    'model-sort-select',
    [
      ['name', 'Name'],
      ['runs', 'Runs'],
      ['best', 'Best'],
      ['status', 'Status'],
    ],
    modelSortKey,
  )
  return `<div class="paper-filter">${btns}${sort}</div>`
}
async function readModelStats() {
  return readLatestRecord('-model-stats')
}
// A run record → the compact row computeModelStats needs.
function modelRunRow(run) {
  const s = (run && run.summary) || {}
  return {
    key: run.key,
    config: s.config || {},
    objective: s.objective,
    status: s.status,
    health: s.health,
    durationMs: s.durationMs,
    ranAt: (s.provenance && s.provenance.ranAt) || s.ranAt,
  }
}
// The Refresh-stats control row + freshness note. Refresh recomputes the all-runs aggregate (a process
// that scans EVERY run, so it spins); a stale flag appears when newer runs exist past the aggregate.
function modelStatsControlsHtml() {
  const total = modelStatsCache ? modelStatsCache.totalRuns : null
  const at =
    modelStatsCache && modelStatsCache.aggregatedAt
      ? ' · ' + String(modelStatsCache.aggregatedAt).slice(0, 16).replace('T', ' ')
      : ''
  const note = modelStatsCache
    ? `Run counts aggregated over ${total} run${total === 1 ? '' : 's'}${at}.`
    : 'Run counts are computed over ALL runs (not the current page) — use Refresh, top right.'
  const stale = modelStatsStale
    ? '<span class="model-stats-stale">newer runs exist — Refresh latest, top right</span>'
    : ''
  return `<div class="model-stats-controls"><span class="card-sub">${escapeHtml(note)}</span> ${stale}</div>`
}
// Runs whose model_name matches no catalog flavor — the "a flavor is missing" signal.
function uncatalogedModelsHtml() {
  const unc = (modelStatsCache && modelStatsCache.uncataloged) || []
  if (!unc.length) return ''
  const rows = unc
    .map(
      (u) =>
        `<li><code>${escapeHtml(u.modelName)}</code> <span class="flavor-runs">${u.runs} run${u.runs === 1 ? '' : 's'}</span></li>`,
    )
    .join('')
  return `<div class="model-group"><h3 class="model-group-h">Unmapped runs — missing flavors</h3><p class="card-sub">These <code>model_name</code> values appear in runs but match no catalog flavor — add a flavor (or Scan) so they roll up to a model.</p><ul class="paper-hyp-list">${rows}</ul></div>`
}
// The Refresh control row for the Hypotheses tab — same shared all-runs refresh, worded for verdicts.
function hypothesisStatsControlsHtml() {
  const total = modelStatsCache ? modelStatsCache.totalRuns : null
  const at =
    modelStatsCache && modelStatsCache.aggregatedAt
      ? ' · ' + String(modelStatsCache.aggregatedAt).slice(0, 16).replace('T', ' ')
      : ''
  const note = modelStatsCache
    ? `Verdicts evaluated over ${total} run${total === 1 ? '' : 's'}${at}.`
    : 'Verdicts are evaluated over ALL runs (not the current page) — use Refresh, top right.'
  const stale = modelStatsStale
    ? '<span class="model-stats-stale">newer runs exist — Refresh latest, top right</span>'
    : ''
  const sort = sortControlHtml(
    'hypothesis-sort-select',
    [
      ['updated', 'Recent'],
      ['name', 'Name'],
      ['oldest', 'Oldest'],
    ],
    hypothesisSortKey,
  )
  return `<div class="model-stats-controls"><span class="card-sub">${escapeHtml(note)}</span> ${stale} ${sort}</div>`
}
// Re-render whichever run-derived tab is active (to repaint the spinner / fresh results).
function rerenderRunDerivedTab() {
  if (activeTabId === 'hypotheses') return renderHypotheses()
  if (activeTabId === 'papers') return renderPapers()
  if (activeTabId === 'speed') return renderSpeed()
  return renderModels()
}
// Run-derived data is refreshed through ONE registry of updaters, each consuming the shared run set and
// refreshing one body of derived data. Add an entry here to make a new artifact participate in the global
// Refresh (both the full scan and the latest-only path) — that's the single seam for "things that need
// updating when runs land".
const RUN_DERIVED_UPDATERS = [
  { label: 'model stats', run: updateModelStatsFromRuns },
  {
    label: 'hypothesis verdicts',
    run: async () => {
      await refreshHypothesisVerdicts(await readHypotheses())
    },
  },
]
async function applyRunDerivedUpdaters(allRuns) {
  for (const u of RUN_DERIVED_UPDATERS) await u.run(allRuns)
}
// Recompute + persist the model-stats aggregate (the `<recordType>-model-stats` record), the first
// run-derived updater. `newestRunAt`/`totalRuns` it records are the frontier the latest-only path extends.
async function updateModelStatsFromRuns(allRuns) {
  const models = await readModels()
  const rows = allRuns.map(modelRunRow)
  const stats = window.Models.computeModelStats(models, rows, objectiveDirection())
  // Per-model run durations ride along in the same persisted record so the Speed tab reflects EVERY run
  // (refreshed through this one all-runs scan), not just the current Runs page.
  const content = {
    ...stats,
    durations: window.Models.computeRunDurationsByModel(rows),
    aggregatedAt: nowIso(),
  }
  await window.OverseerBridge.putData({
    type: manifest.recordType + '-model-stats',
    key: 'latest',
    content,
  })
  modelStatsCache = content
}
// The newer tail: page newest-first (ranAt desc) and collect runs past the last aggregate's frontier,
// stopping at the first run that's already covered. The cheap path — it never re-pages the whole history.
async function fetchRunsNewerThan(newestRunAt, onProgress) {
  const PAGE = 500
  const fresh = []
  let offset = 0
  for (let guard = 0; guard < 10000; guard++) {
    const page = await queryRunRecords({
      orderBy: [{ field: 'provenance.ranAt', direction: 'desc', numeric: false }],
      limit: PAGE,
      offset,
    })
    if (!page.length) break
    let hitOld = false
    for (const r of page) {
      const ranAt = r.summary && r.summary.provenance && r.summary.provenance.ranAt
      if (ranAt && newestRunAt && ranAt <= newestRunAt) {
        hitOld = true
        break
      }
      fresh.push(r)
    }
    if (onProgress) onProgress(fresh.length)
    if (hitOld || page.length < PAGE) break
    offset += PAGE
  }
  return fresh
}
function setRefreshProgress(text) {
  const btn = byId('dash-refresh-all')
  if (btn) setHtml(btn, `${spinnerHtml()} ${escapeHtml(text)}`)
}
// Shared refresh core. `mode` 'all' re-scans EVERY run; 'latest' fetches only the newer tail and merges it
// into the in-memory cache (deduped by key) — both then run every run-derived updater over the full set.
// Latest needs a prior aggregate (a base cache + a frontier) to extend; without one it falls back to all.
async function runDerivedRefresh(mode) {
  if (!embedded() || !manifest || modelStatsRefreshing) return
  const canLatest =
    mode === 'latest' && allRunsCache.length && modelStatsCache && modelStatsCache.newestRunAt
  const epoch = projectEpoch
  modelStatsRefreshing = true
  renderGlobalRefresh()
  await rerenderRunDerivedTab()
  let ok = false
  try {
    let allRuns
    if (canLatest) {
      const fresh = await fetchRunsNewerThan(modelStatsCache.newestRunAt, (n) =>
        setRefreshProgress(`Fetching ${n} new run${n === 1 ? '' : 's'}…`),
      )
      const byKey = new Map(allRunsCache.map((r) => [r.key, r]))
      for (const r of fresh) byKey.set(r.key, r)
      allRuns = [...byKey.values()]
    } else {
      allRuns = await queryAllRunRecords((n) => setRefreshProgress(`Scanning ${n} runs…`))
    }
    allRunsCache = allRuns
    await applyRunDerivedUpdaters(allRuns)
    modelStatsStale = false
    ok = true
  } catch {
    if (epoch === projectEpoch) {
      const id = activeTabId === 'hypotheses' ? 'hypotheses-status' : 'models-status'
      setStatusLine(id, 'Could not refresh from runs — please try again.', true)
    }
  } finally {
    modelStatsRefreshing = false
    if (epoch === projectEpoch) {
      renderGlobalRefresh()
      await rerenderRunDerivedTab()
      if (ok) showToast(canLatest ? 'Refreshed latest runs.' : 'Refreshed over all runs.')
    }
  }
}
// Both Refresh paths the topbar control exposes; the in-memory `allRunsCache` is shared so a single scan
// feeds every updater.
async function refreshAllRunsDerived() {
  return runDerivedRefresh('all')
}
async function refreshLatestRunsDerived() {
  return runDerivedRefresh('latest')
}
// On project open: load the persisted aggregate (so freshness shows from ANY tab, not just the run-derived
// ones), render the topbar control, and run the cheap staleness check so "Refresh latest" can appear.
async function initGlobalRefresh() {
  renderGlobalRefresh()
  if (!modelStatsCache) {
    try {
      modelStatsCache = await readModelStats()
    } catch {
      // no aggregate yet — the control just offers a full Refresh
    }
  }
  renderGlobalRefresh()
  await checkModelStatsStale()
}
// The global Refresh control (dashboard topbar, top-right): "Refresh" (full scan, always) + "Refresh
// latest" (the newer-tail path, only when newer runs exist and there's a prior aggregate to extend).
function renderGlobalRefresh() {
  const el = byId('dash-refresh')
  if (!el) return
  if (!embedded() || !manifest) {
    el.innerHTML = ''
    return
  }
  const busy = modelStatsRefreshing
  const canLatest = !busy && modelStatsStale && !!modelStatsCache && allRunsCache.length > 0
  const at =
    modelStatsCache && modelStatsCache.aggregatedAt
      ? String(modelStatsCache.aggregatedAt).slice(0, 16).replace('T', ' ')
      : ''
  const note = modelStatsStale
    ? '<span class="dash-refresh-note is-stale">new runs since last refresh</span>'
    : at
      ? `<span class="dash-refresh-note">updated ${escapeHtml(at)}</span>`
      : ''
  const latestBtn = canLatest
    ? `<button type="button" id="dash-refresh-latest" class="ghost-btn"${helpAttr('Fetch only the runs added since the last refresh and update — the fast path.')}>Refresh latest</button>`
    : ''
  el.innerHTML =
    note +
    `<button type="button" id="dash-refresh-all" class="ghost-btn"${busy ? ' disabled' : ''}${helpAttr('Re-scan EVERY run and recompute all run-derived data (model stats, hypothesis verdicts).')}>${busy ? `${spinnerHtml()} Refreshing…` : 'Refresh'}</button>` +
    latestBtn
}
// Cheap freshness check: is the live newest run / count past what the aggregate covered? Updates ONLY the
// control row in place (no full re-render) so it can run after each render without looping.
async function checkModelStatsStale() {
  const was = modelStatsStale
  if (!modelStatsCache) {
    modelStatsStale = false
  } else {
    try {
      const newest = await queryRunRecords({
        orderBy: [{ field: 'provenance.ranAt', direction: 'desc', numeric: false }],
        limit: 1,
      })
      const liveNewest =
        newest[0] && newest[0].summary && newest[0].summary.provenance
          ? newest[0].summary.provenance.ranAt
          : null
      const total = await countRunRecords(undefined)
      modelStatsStale =
        (!!liveNewest && liveNewest !== modelStatsCache.newestRunAt) ||
        (total !== null && total !== modelStatsCache.totalRuns)
    } catch {
      modelStatsStale = false
    }
  }
  if (modelStatsStale !== was) {
    renderGlobalRefresh()
    const bodyId =
      activeTabId === 'hypotheses'
        ? 'hypotheses-body'
        : activeTabId === 'models'
          ? 'models-body'
          : null
    const body = bodyId && byId(bodyId)
    const el = body && body.querySelector('.model-stats-controls')
    if (el) {
      const tmp = document.createElement('div')
      tmp.innerHTML =
        activeTabId === 'hypotheses' ? hypothesisStatsControlsHtml() : modelStatsControlsHtml()
      if (tmp.firstElementChild) el.replaceWith(tmp.firstElementChild)
    }
  }
}
// SPEED tab: per-model device benchmark (cpu vs mps µs/step + speedup) and wall-clock run durations by
// model. device is used here only — never as an analysis lever.
async function renderSpeed() {
  const body = byId('speed-body')
  if (!body) return
  if (!embedded()) {
    setHtml(body, '<div class="empty-hint">Open inside the Overseer to see speed data.</div>')
    return
  }
  // Delegated once (the body's innerHTML is replaced each render): a benchmark row opens its exact timings.
  if (!body.dataset.timingsWired) {
    body.dataset.timingsWired = '1'
    body.addEventListener('click', (event) => {
      const row = event.target.closest('.device-row[data-action="device-timings"]')
      if (row) showDeviceTimingsModal(row.dataset.id)
    })
  }
  ;[modelsCache, modelStatsCache] = await Promise.all([readModels(), readModelStats()])
  void checkModelStatsStale()
  const benchmarked = (modelsCache || []).filter(
    (m) =>
      m.deviceBenchmark &&
      (m.deviceBenchmark.usPerStep || m.deviceBenchmark.errors) &&
      window.Models.isSpeedEligibleModel(m, modelStatusOf(m)),
  )
  // One column per device that appears across all benchmarks (cpu/mps/cuda first, others after).
  const colSet = {}
  for (const m of benchmarked) {
    for (const d of Object.keys(m.deviceBenchmark.usPerStep || {})) colSet[d] = true
    for (const d of Object.keys(m.deviceBenchmark.errors || {})) colSet[d] = true
  }
  const stdOrder = ['cpu', 'mps', 'cuda']
  const deviceCols = stdOrder
    .filter((d) => colSet[d])
    .concat(
      Object.keys(colSet)
        .filter((d) => stdOrder.indexOf(d) < 0)
        .sort(),
    )
  const benchRows = benchmarked
    .map((m) => {
      const view = window.Models.deviceBenchmarkView(m.deviceBenchmark, deviceCols)
      const byDev = {}
      for (const pd of view.perDevice) byDev[pd.device] = pd
      const cells = deviceCols
        .map((d) => {
          const pd = byDev[d]
          if (pd && pd.usPerStep != null)
            return `<td class="num device-cell${pd.isBest ? ' is-best' : ''}">${Math.round(pd.usPerStep)}</td>`
          if (pd && pd.error)
            return `<td class="num device-cell device-slow"${helpAttr(pd.error)}>slow</td>`
          return `<td class="num device-cell">\u2014</td>`
        })
        .join('')
      const bestChip = view.best
        ? `<span class="run-badge ${view.bestClass}">${escapeHtml(String(view.best).toUpperCase())}</span>`
        : '\u2014'
      return `<tr class="device-row" data-action="device-timings" data-id="${escapeHtml(m.id)}"${helpAttr('Click for the exact per-device timings')}><th>${escapeHtml(m.name || m.id)}</th>${cells}<td>${bestChip}</td><td class="num">${view.speedup && view.speedup > 1.01 ? view.speedup.toFixed(2) + '\u00d7' : '\u2014'}</td><td class="card-sub">${escapeHtml(view.benchmarkedAt ? formatWhen(view.benchmarkedAt) : '\u2014')}</td></tr>`
    })
    .join('')
  const deviceHeaders = deviceCols.map((d) => `<th class="num">${escapeHtml(d)}</th>`).join('')
  const benchCard = benchRows
    ? `<div class="card"><div class="card-head card-head-row"><h3>Device benchmark <span class="card-sub">\u2014 \u00b5s/step per device (lower = faster); click a row for exact timings</span></h3></div>
      <table class="kv-table report-table device-table"><thead><tr><th>model</th>${deviceHeaders}<th>best</th><th class="num">speedup</th><th>benchmarked</th></tr></thead><tbody>${benchRows}</tbody></table></div>`
    : `<div class="card"><p class="card-sub">No device benchmarks yet \u2014 run \u201cBenchmark device\u201d on a model (Models tab) to measure cpu vs mps speed.</p></div>`
  // Durations come from the persisted all-runs aggregate (refreshed through the shared global Refresh, like
  // model stats + hypothesis verdicts) so the table reflects EVERY run. Before the first refresh of this
  // record we fall back to whatever runs are in memory (the all-runs snapshot, else the current page) and
  // say so, so the scope is never misread as "all" when it isn't.
  const persistedDur =
    modelStatsCache && Array.isArray(modelStatsCache.durations) ? modelStatsCache.durations : null
  const durList =
    persistedDur ||
    window.Models.computeRunDurationsByModel((allRunsCache.length ? allRunsCache : []).map(modelRunRow))
  const durAt =
    persistedDur && modelStatsCache.aggregatedAt
      ? ' \u00b7 updated ' + String(modelStatsCache.aggregatedAt).slice(0, 16).replace('T', ' ')
      : ''
  const durScope = persistedDur
    ? `wall-clock per training run \u2014 over ${modelStatsCache.totalRuns} run${modelStatsCache.totalRuns === 1 ? '' : 's'}${durAt}`
    : allRunsCache.length
      ? 'wall-clock per training run \u2014 all loaded runs'
      : 'wall-clock per training run \u2014 Refresh (top right) to aggregate over all runs'
  const durStale = persistedDur && modelStatsStale ? ' <span class="model-stats-stale">newer runs exist</span>' : ''
  const durRows = durList
    .map(
      (d) =>
        `<tr><th>${escapeHtml(d.modelName)}</th><td class="num">${d.runs}</td><td class="num">${escapeHtml(formatDuration(Math.round(d.meanMs)))}</td><td class="num">${escapeHtml(formatDuration(d.minMs))}</td><td class="num">${escapeHtml(formatDuration(d.maxMs))}</td></tr>`,
    )
    .join('')
  const durCard = durRows
    ? `<div class="card"><div class="card-head card-head-row"><h3>Run duration by model <span class="card-sub">\u2014 ${durScope}</span>${durStale}</h3></div>
      <table class="kv-table report-table"><thead><tr><th>model</th><th class="num">runs</th><th class="num">mean</th><th class="num">fastest</th><th class="num">slowest</th></tr></thead><tbody>${durRows}</tbody></table></div>`
    : `<div class="card"><div class="card-head card-head-row"><h3>Run duration by model</h3></div><p class="card-sub">${escapeHtml(durScope)}</p></div>`
  setHtml(body, benchCard + durCard)
}
async function renderModels() {
  const body = byId('models-body')
  if (!body) return
  if (!embedded()) {
    setHtml(
      body,
      '<div class="empty-hint">Open inside the Overseer to manage the model catalog.</div>',
    )
    return
  }
  // Load the persisted all-runs STATS (not a page of runs) + papers + hypotheses for the links. Hypothesis
  // verdicts render from their PERSISTED status (refreshed by the shared all-runs refresh, not here).
  ;[modelsCache, modelStatsCache, papersCache, hypothesesCache] = await Promise.all([
    readModels(),
    readModelStats(),
    readPapers(),
    readHypotheses(),
  ])
  const seedBanner = pendingSeedModelsHtml()
  if (!modelsCache.length) {
    setHtml(
      body,
      seedBanner +
        modelStatsControlsHtml() +
        '<div class="empty-hint">No models yet — Scan Project to discover them, or import the curated set.</div>',
    )
    void checkModelStatsStale()
    return
  }
  const shown =
    modelCategoryFilter === 'all'
      ? modelsCache
      : modelsCache.filter((m) => m.category === modelCategoryFilter)
  const order = window.Models.MODEL_CATEGORIES
  const cmp = modelSortComparator()
  const groups = order
    .map((cat) => ({ cat, models: shown.filter((m) => m.category === cat).sort(cmp) }))
    .filter((g) => g.models.length)
  const other = shown.filter((m) => order.indexOf(m.category) < 0).sort(cmp)
  if (other.length) groups.push({ cat: 'other', models: other })
  const html = groups
    .map((g) => {
      const label = window.Models.MODEL_CATEGORY_LABEL[g.cat] || 'Other'
      return `<div class="model-group"><h3 class="model-group-h">${escapeHtml(label)}</h3>${g.models.map(modelCardHtml).join('')}</div>`
    })
    .join('')
  setHtml(
    body,
    seedBanner +
      modelStatsControlsHtml() +
      modelCategoryFilterBarHtml() +
      (html || '<div class="empty-hint">No models in this category.</div>') +
      uncatalogedModelsHtml(),
  )
  void checkModelStatsStale()
}
// Open the host chat preloaded with this model's full context (status, bindings, runs, linked papers +
// hypotheses); the seed adapts to status — implement (proposed), fix (failing), or improve.
async function chatAboutModel(id) {
  const model = modelsCache.find((m) => m.id === id)
  if (!model || !chatAboutRunAvailable()) return
  const status = modelStatusOf(model)
  const agg = modelAgg(model)
  const papers = window.Models.papersForModel(model, papersCache)
  const hyps = window.Models.hypothesesForModel(model, hypothesesCache)
  const bindings = window.Models.flavorModelNames(model)
  const runsLine = agg
    ? `Training runs so far: ${agg.runs}${agg.best !== null && agg.best !== undefined ? `, best ${objectiveName()} ${formatObjective(agg.best)}` : ''}${agg.failing ? `, ${agg.failing} health-flagged/failed` : ''}.`
    : 'Run counts not yet aggregated.'
  const ctx = [
    `You are discussing ONE specific MODEL in this project's catalog — "${model.name || model.id}" (status: ${status}). Work from the details below; don't ask the user to restate them.`,
    `Category: ${model.category}.`,
    model.description ? `What it is: ${model.description}` : '',
    model.proposal ? `What's asked: ${model.proposal}` : '',
    bindings.length
      ? `It is trained by runs whose model_name is one of: ${bindings.join(', ')}.`
      : 'It is NOT yet wired to any run config (a pure proposal).',
    model.implPath ? `Implemented at: ${model.implPath}` : 'No implementation path on record yet.',
    runsLine,
    papers.length ? `Linked papers: ${papers.map((p) => p.title || p.id).join('; ')}.` : '',
    hyps.length ? `Linked hypotheses: ${hyps.map((h) => h.title || h.id).join('; ')}.` : '',
  ]
    .filter(Boolean)
    .join('\n\n')
  const systemPrompt = [projectChatPreamble(), ctx].filter(Boolean).join('\n\n')
  const seed =
    status === 'proposed'
      ? 'Help me implement this model in the project — outline the files, classes and the model_name wiring needed to add it.'
      : status === 'failing'
        ? 'This model is failing — help me diagnose why its runs are degenerate/erroring and propose a fix.'
        : status === 'needs-improvement'
          ? 'Help me improve this model — what changes are most likely to raise its objective?'
          : 'Help me work on this model — what would most improve it from here?'
  try {
    await window.OverseerBridge.discussTopic({
      title: `Model ${model.name || model.id}`,
      seed,
      systemPrompt,
    })
  } catch {
    setStatusLine('models-status', 'Could not open chat — please try again.', true)
  }
}
// Scan Project: discover declared models (model_name choices) not yet catalogued, enrich with the LLM,
// and persist them. Triggers the backend `scan-models` activity, then re-renders Models.
async function onScanModels() {
  if (!embedded()) {
    setStatusLine('models-status', 'Open inside the Overseer to scan for models.', true)
    return
  }
  const epoch = projectEpoch
  setStatusLine('models-status', '')
  const btn = byId('models-scan-btn')
  if (btn) btn.disabled = true
  try {
    const result = await startOrEnqueue('scan-models', trainerActivityParams({}), 'Scan models')
    if (result.queued) {
      if (epoch === projectEpoch) setStatusLine('models-status', queuedStatusText(result.ahead))
      return
    }
    const act = await observeQuickActivity(result.activityId)
    if (epoch !== projectEpoch) return
    if (act && act.status === 'completed') {
      await renderModels()
      showToast('Scanned the project for models.')
    } else {
      setStatusLine('models-status', quickActivityFailureText(act, 'Scan'), true)
    }
  } catch {
    if (epoch === projectEpoch) {
      setStatusLine('models-status', 'Could not scan — please try again.', true)
    }
  } finally {
    if (btn) btn.disabled = false
  }
}
// Consolidate: ask the LLM to find near-duplicate models (often the same model proposed from several
// papers) and review merging each group into ONE canonical model — the rest fold in as flavors and are
// deleted. Triggers the backend `consolidate-models` activity, then opens the review modal.
async function onConsolidateModels() {
  if (!embedded()) {
    setStatusLine('models-status', 'Open inside the Overseer to consolidate models.', true)
    return
  }
  const epoch = projectEpoch
  setStatusLine('models-status', '')
  const btn = byId('models-consolidate-btn')
  if (btn) btn.disabled = true
  try {
    const result = await startOrEnqueue(
      'consolidate-models',
      trainerActivityParams({}),
      'Consolidate models',
    )
    if (result.queued) {
      if (epoch === projectEpoch) setStatusLine('models-status', queuedStatusText(result.ahead))
      return
    }
    const act = await observeQuickActivity(result.activityId)
    if (epoch !== projectEpoch) return
    if (act && act.status === 'completed') {
      const groups = await loadConsolidationGroups()
      if (!groups.length) {
        showToast('No near-duplicate models to consolidate.')
        return
      }
      openConsolidateModal(groups)
    } else {
      setStatusLine('models-status', quickActivityFailureText(act, 'Consolidate'), true)
    }
  } catch {
    if (epoch === projectEpoch) {
      setStatusLine('models-status', 'Could not consolidate — please try again.', true)
    }
  } finally {
    if (btn) btn.disabled = false
  }
}
// Consolidate hypotheses: DETERMINISTIC (no LLM) — fold hypotheses sharing the same main parameters into one
// wider hypothesis (unioning their sweeps). Applied server-side by the activity (repoints paper/model links +
// per-paper weights, deletes absorbed records); we re-read + report. Conflicting manual verdicts are left.
async function onConsolidateHypotheses() {
  if (!embedded()) {
    setStatusLine('hypotheses-status', 'Open inside the Overseer to consolidate hypotheses.', true)
    return
  }
  const epoch = projectEpoch
  setStatusLine('hypotheses-status', '')
  const btn = byId('hypotheses-consolidate-btn')
  if (btn) btn.disabled = true
  try {
    const result = await startOrEnqueue(
      'consolidate-hypotheses',
      trainerActivityParams({}),
      'Consolidate hypotheses',
    )
    if (result.queued) {
      if (epoch === projectEpoch) setStatusLine('hypotheses-status', queuedStatusText(result.ahead))
      return
    }
    const act = await observeQuickActivity(result.activityId)
    if (epoch !== projectEpoch) return
    if (act && act.status === 'completed') {
      const summary = await loadHypothesisConsolidationResult()
      hypothesesCache = await readHypotheses()
      papersCache = await readPapers()
      await refreshHypothesisVerdicts(hypothesesCache)
      if (activeTabId === 'hypotheses') await renderHypotheses()
      const groups = (summary && summary.merged) || []
      const absorbed = groups.reduce((n, g) => n + ((g.absorbedIds && g.absorbedIds.length) || 0), 0)
      const conflicts = ((summary && summary.conflicts) || []).length
      if (!groups.length && !conflicts) {
        showToast('No similar hypotheses to consolidate.')
      } else {
        showToast(
          `Consolidated ${absorbed} hypothes${absorbed === 1 ? 'is' : 'es'} into ${groups.length} wider one${groups.length === 1 ? '' : 's'}${conflicts ? ` — ${conflicts} left (conflicting manual verdicts)` : ''}.`,
        )
      }
    } else {
      setStatusLine('hypotheses-status', quickActivityFailureText(act, 'Consolidate'), true)
    }
  } catch {
    if (epoch === projectEpoch)
      setStatusLine('hypotheses-status', 'Could not consolidate — please try again.', true)
  } finally {
    const b = byId('hypotheses-consolidate-btn')
    if (b) b.disabled = false
  }
}
async function loadHypothesisConsolidationResult() {
  try {
    const recs = await queryRecords(manifest.recordType + '-hypothesis-consolidation', 'latest')
    return (recs && recs[0] && recs[0].content) || null
  } catch {
    return null
  }
}
// Read the latest `-consolidation` proposal, resolve each group's ids to live catalog models (dropping any
// that no longer exist), and keep only groups that still have a canonical + at least one duplicate.
async function loadConsolidationGroups() {
  modelsCache = await readModels()
  const byModelId = new Map(modelsCache.map((m) => [m.id, m]))
  const recs = await queryRecords(manifest.recordType + '-consolidation', 'latest')
  const content = recs && recs[0] && recs[0].content
  const groups = content && Array.isArray(content.groups) ? content.groups : []
  const resolved = []
  for (const g of groups) {
    const canonical = byModelId.get(g.canonicalId)
    if (!canonical) continue
    const duplicates = (g.duplicateIds || []).map((id) => byModelId.get(id)).filter(Boolean)
    if (!duplicates.length) continue
    resolved.push({
      canonicalId: g.canonicalId,
      reason: g.reason || '',
      members: [canonical, ...duplicates],
      checkedDuplicateIds: new Set(duplicates.map((d) => d.id)),
    })
  }
  return resolved
}
function openConsolidateModal(groups) {
  consolidationGroups = groups
  consolidationModalEpoch = projectEpoch
  renderConsolidateModal()
}
function closeConsolidateModal() {
  const m = byId('consolidate-modal')
  if (m) m.hidden = true
  consolidationGroups = []
}
// One member row: a radio to pick THIS entry as the canonical to keep, and (for non-canonical members) a
// checkbox to fold it in. The canonical keeps its identity, runs, papers + flavors; the others are deleted.
function consolidateMemberHtml(gi, model, group) {
  const isCanon = model.id === group.canonicalId
  const names = window.Models.flavorModelNames(model)
  const sub = names.length ? ` <span class="card-sub">(${names.map(escapeHtml).join(', ')})</span>` : ''
  const checked = group.checkedDuplicateIds.has(model.id) ? ' checked' : ''
  const control = isCanon
    ? '<span class="consolidate-keep">keeps its runs &amp; flavors</span>'
    : `<label class="consolidate-merge"><input type="checkbox" data-consolidate-merge value="${escapeHtml(model.id)}"${checked} /> merge in</label>`
  return (
    `<li class="consolidate-member${isCanon ? ' is-canonical' : ''}">` +
    `<label class="consolidate-canon"><input type="radio" name="consolidate-canon-${gi}" value="${escapeHtml(model.id)}"${isCanon ? ' checked' : ''} /> canonical</label>` +
    `<span class="consolidate-member-name"><strong>${escapeHtml(model.name || model.id)}</strong>${sub}</span>` +
    `<span class="consolidate-member-ctl">${control}</span>` +
    `</li>`
  )
}
function consolidateGroupHtml(group, gi) {
  const list = group.members.map((m) => consolidateMemberHtml(gi, m, group)).join('')
  return (
    `<div class="consolidate-group" data-index="${gi}">` +
    (group.reason ? `<p class="consolidate-reason">${escapeHtml(group.reason)}</p>` : '') +
    `<ul class="consolidate-members">${list}</ul>` +
    `<div class="consolidate-actions">` +
    `<button type="button" class="consolidate-merge-btn" data-consolidate-accept="${gi}">Merge</button>` +
    `<button type="button" class="ghost-btn" data-consolidate-reject="${gi}">Reject</button>` +
    `</div></div>`
  )
}
function renderConsolidateModal() {
  let modal = byId('consolidate-modal')
  if (!modal) {
    modal = document.createElement('div')
    modal.id = 'consolidate-modal'
    modal.className = 'chart-modal'
    document.body.appendChild(modal)
    modal.addEventListener('click', (event) => {
      if (event.target === modal || event.target.closest('[data-consolidate-close]'))
        return closeConsolidateModal()
      const accept = event.target.closest('[data-consolidate-accept]')
      if (accept) return onAcceptConsolidation(Number(accept.dataset.consolidateAccept))
      const reject = event.target.closest('[data-consolidate-reject]')
      if (reject) return onRejectConsolidation(Number(reject.dataset.consolidateReject))
    })
    modal.addEventListener('change', (event) => {
      const radio = event.target.closest('input[type="radio"][name^="consolidate-canon-"]')
      if (radio) {
        onChangeConsolidationCanonical(
          Number(radio.name.replace('consolidate-canon-', '')),
          radio.value,
        )
        return
      }
      const check = event.target.closest('input[type="checkbox"][data-consolidate-merge]')
      if (check) {
        const groupEl = check.closest('.consolidate-group')
        const group = groupEl && consolidationGroups[Number(groupEl.dataset.index)]
        if (!group) return
        if (check.checked) group.checkedDuplicateIds.add(check.value)
        else group.checkedDuplicateIds.delete(check.value)
      }
    })
  }
  const groups = consolidationGroups
  const body = groups.length
    ? groups.map((g, i) => consolidateGroupHtml(g, i)).join('')
    : '<p class="consolidate-empty">All suggestions handled.</p>'
  modal.innerHTML =
    `<div class="chart-modal__backdrop" data-consolidate-close></div>` +
    `<div class="chart-modal__panel consolidate-panel" role="dialog" aria-label="Consolidate models">` +
    `<div class="chart-modal__head">` +
    `<strong>Consolidate models <span class="card-sub">— ${groups.length} suggestion${groups.length === 1 ? '' : 's'}</span></strong>` +
    `<button type="button" class="icon-btn" data-consolidate-close title="Close (Esc)" aria-label="Close">✕</button>` +
    `</div>` +
    `<p class="card-sub consolidate-help">Each group is, per the LLM, the same model proposed more than once. Pick the entry to KEEP as canonical; the ones you "merge in" fold into it as flavors (their papers/hypotheses are preserved) and are then deleted.</p>` +
    `<div class="chart-modal__scroll">${body}</div>` +
    `</div>`
  modal.hidden = false
}
// Re-render one group when its canonical changes so the new canonical loses its "merge in" checkbox. The
// user's per-duplicate checks live in `group.checkedDuplicateIds` (kept current by the change listener), so
// they survive the re-render; the newly-demoted old canonical defaults to checked, the new one drops out.
function onChangeConsolidationCanonical(gi, modelId) {
  const group = consolidationGroups[gi]
  if (!group) return
  window.Models.swapConsolidationCanonical(group, modelId)
  const modal = byId('consolidate-modal')
  const el = modal && modal.querySelector(`.consolidate-group[data-index="${gi}"]`)
  if (!el) return
  const tmp = document.createElement('div')
  tmp.innerHTML = consolidateGroupHtml(group, gi)
  const fresh = tmp.firstElementChild
  if (fresh) el.replaceWith(fresh)
}
function onRejectConsolidation(gi) {
  consolidationGroups.splice(gi, 1)
  renderConsolidateModal()
}
// Apply one accepted merge: fold the checked duplicates into the chosen canonical (flavors + paper/hyp
// links unioned), persist the canonical, delete the duplicates, and repoint papers that referenced them.
async function onAcceptConsolidation(gi) {
  const group = consolidationGroups[gi]
  if (!group) return
  // Abort if the project changed under the open modal — the group ids belong to the old project.
  if (consolidationModalEpoch !== projectEpoch) {
    closeConsolidateModal()
    setStatusLine('models-status', 'Project changed — reopen Consolidate to continue.', true)
    return
  }
  const canonicalId = group.canonicalId
  // The selection lives in the group object (the source of truth), not the DOM — so changing the canonical
  // never silently re-checks a duplicate the user excluded.
  const duplicateIds = window.Models.selectedDuplicateIds(group)
  if (!duplicateIds.length) {
    setStatusLine('models-status', 'Select at least one model to merge in (or Reject).', true)
    return
  }
  const canonical = modelsCache.find((m) => m.id === canonicalId)
  const duplicates = duplicateIds.map((id) => modelsCache.find((m) => m.id === id)).filter(Boolean)
  if (!canonical || !duplicates.length) return
  try {
    const merged = window.Models.mergeModelsForConsolidation(canonical, duplicates, nowIso())
    // Order matters: repoint papers + write the (un-deleted) canonical BEFORE deleting any duplicate, so a
    // mid-sequence failure can never leave a paper or the catalog pointing at a model that's already gone.
    const changedPapers = window.Models.repointPaperModelIds(
      await readPapers(),
      duplicateIds,
      canonicalId,
    )
    for (const p of changedPapers) await putPaper(p)
    await putModel(merged)
    for (const d of duplicates) await deleteModelRecord(d.id)
  } catch {
    setStatusLine('models-status', 'Could not merge — please try again.', true)
    return
  }
  consolidationGroups.splice(gi, 1)
  showToast(
    `Merged ${duplicates.length} model${duplicates.length === 1 ? '' : 's'} into ${canonical.name || canonicalId}.`,
  )
  renderConsolidateModal()
  await renderModels()
}
// The exact per-device timings behind a model's benchmark (µs/step + measured seconds + why a device was
// skipped) — opened from the device chip (Models tab) or a Speed-tab row, with a Re-benchmark affordance.
function closeDeviceTimingsModal() {
  const m = byId('device-timings-modal')
  if (m) m.hidden = true
}
function showDeviceTimingsModal(id) {
  const model = (modelsCache || []).find((m) => m.id === id)
  if (!model || !model.deviceBenchmark) return
  const view = window.Models.deviceBenchmarkView(model.deviceBenchmark)
  let modal = byId('device-timings-modal')
  if (!modal) {
    modal = document.createElement('div')
    modal.id = 'device-timings-modal'
    modal.className = 'chart-modal'
    document.body.appendChild(modal)
    modal.addEventListener('click', (event) => {
      if (event.target === modal || event.target.closest('[data-timings-close]'))
        return closeDeviceTimingsModal()
      const rerun = event.target.closest('[data-timings-rerun]')
      if (rerun) {
        closeDeviceTimingsModal()
        onBenchmarkModelDevice(rerun.dataset.timingsRerun)
      }
    })
  }
  const rows = view.perDevice
    .map((pd) => {
      const us = pd.usPerStep != null ? `${Math.round(pd.usPerStep)} µs` : '—'
      const secs = pd.seconds != null ? `${pd.seconds.toFixed(2)} s` : '—'
      const status = pd.isBest
        ? `<span class="run-badge ${pd.chipClass}">fastest</span>`
        : pd.error
          ? `<span class="card-sub">${escapeHtml(pd.error)}</span>`
          : pd.usPerStep != null
            ? ''
            : '<span class="card-sub">not measured</span>'
      return `<tr><th><span class="run-badge ${pd.chipClass}">${escapeHtml(pd.device.toUpperCase())}</span></th><td class="num">${us}</td><td class="num">${secs}</td><td>${status}</td></tr>`
    })
    .join('')
  const meta = [
    view.budget ? `${view.budget} steps` : '',
    view.benchmarkedAt ? `benchmarked ${escapeHtml(formatWhen(view.benchmarkedAt))}` : '',
  ]
    .filter(Boolean)
    .join(' · ')
  modal.innerHTML =
    `<div class="chart-modal__backdrop" data-timings-close></div>` +
    `<div class="chart-modal__panel device-timings-panel" role="dialog" aria-label="Device timings">` +
    `<div class="chart-modal__head"><strong>${escapeHtml(model.name || model.id)} <span class="card-sub">— device timings</span></strong>` +
    `<button type="button" class="icon-btn" data-timings-close title="Close (Esc)" aria-label="Close">✕</button></div>` +
    `<div class="chart-modal__scroll">` +
    `<table class="kv-table"><thead><tr><th>device</th><th class="num">µs/step</th><th class="num">seconds</th><th>status</th></tr></thead><tbody>${rows}</tbody></table>` +
    (meta ? `<p class="card-sub device-timings-meta">${meta} · µs/step lower = faster.</p>` : '') +
    `<div class="device-timings-actions"><button type="button" class="ghost-btn" data-timings-rerun="${escapeHtml(model.id)}">Re-benchmark</button></div>` +
    `</div></div>`
  modal.hidden = false
}
// Benchmark ONE model on CPU vs MPS — enqueue the activity and show the spinner immediately. The activity
// lifecycle (applyQuickDispatchState / refreshAfterQuickDispatch) owns the rest: it spins the chip while
// RUNNING and re-reads the model record (the new preferredDevice chip) on completion. queueCache covers
// the QUEUED-but-not-yet-running window. So the chip spins from click → through queue → run → result.
async function onBenchmarkModelDevice(id) {
  if (!embedded()) {
    setStatusLine('models-status', 'Open inside the Overseer to benchmark devices.', true)
    return
  }
  if (modelBenchmarkPending(id)) return
  const epoch = projectEpoch
  try {
    const result = await startOrEnqueue(
      'benchmark-model-device',
      trainerActivityParams({ modelId: id }),
      'Benchmark device',
    )
    if (epoch !== projectEpoch) return
    setStatusLine(
      'models-status',
      result.queued ? queuedStatusText(result.ahead) : 'Benchmarking CPU vs MPS…',
    )
    await refreshQueue() // so queueCache reflects a queued benchmark and the chip spins right away
    await renderModels()
  } catch {
    if (epoch === projectEpoch) {
      setStatusLine('models-status', 'Could not benchmark — please try again.', true)
    }
  }
}
async function onDeleteModel(id) {
  try {
    await deleteModelRecord(id)
  } catch {
    setStatusLine('models-status', 'Could not delete — please try again.', true)
    return
  }
  modelsCache = modelsCache.filter((m) => m.id !== id)
  await renderModels()
}
async function setModelStatus(id, value) {
  const model = modelsCache.find((m) => m.id === id)
  if (!model) return
  const next =
    value === 'auto'
      ? { ...model, statusSource: 'auto' }
      : { ...model, statusSource: 'manual', status: value }
  try {
    await putModel(next)
  } catch {
    setStatusLine('models-status', 'Could not update status — please try again.', true)
    return
  }
  await renderModels()
}
// Add a paper-proposed (missing) model to the catalog as a `proposed` record, linked to the paper.
async function addProposedModelToCatalog(paperId, slug) {
  const missing = paperMissingModels.get(paperId) || []
  const proposed = missing.find((p) => p.slug === slug)
  if (!proposed) return
  const rec = window.Models.buildProposedModelRecord(proposed, paperId, nowIso())
  try {
    await putModel(rec)
    const p = papersCache.find((x) => x.id === paperId)
    if (p) {
      const ids = Array.from(new Set([...(p.modelIds || []), rec.id]))
      await putPaper({ ...p, modelIds: ids })
      p.modelIds = ids
    }
  } catch {
    setStatusLine('papers-status', 'Could not add the model — please try again.', true)
    return
  }
  paperMissingModels.set(
    paperId,
    missing.filter((x) => x.slug !== slug),
  )
  modelsCache = await readModels()
  updatePaperCard(paperId)
  showToast(`Added “${proposed.name}” to the Models catalog — view`, () => {
    showTab('models')
  })
}
// The linked + proposed-missing models shown on a Papers card (the "add missing model" affordance).
function paperModelsHtml(paper) {
  if (!window.Models) return ''
  const linked = window.Models.modelsForPaper(paper, modelsCache)
  const missing = paperMissingModels.get(paper.id) || []
  if (!linked.length && !missing.length) return ''
  const linkedRows = linked
    .map(
      (m) =>
        `<li><span class="paper-hyp-title" data-action="goto-model" data-id="${escapeHtml(m.id)}">${escapeHtml(m.name || m.id)}</span> ${modelStatusBadgeHtml(m)}</li>`,
    )
    .join('')
  const missingRows = missing
    .map(
      (p) =>
        `<li><span class="paper-hyp-title">${escapeHtml(p.name)}</span> <span class="run-badge is-queued">not in catalog</span> <button type="button" class="icon-btn" data-action="add-model" data-paper="${escapeHtml(paper.id)}" data-id="${escapeHtml(p.slug)}" title="Add to the Models catalog as a proposed model" aria-label="Add model to catalog">+</button></li>`,
    )
    .join('')
  return `<div class="paper-models"><span class="card-sub">Models</span><ul class="paper-hyp-list">${linkedRows}${missingRows}</ul></div>`
}
function setupModels() {
  const scanBtn = byId('models-scan-btn')
  if (scanBtn) scanBtn.addEventListener('click', onScanModels)
  const consolidateBtn = byId('models-consolidate-btn')
  if (consolidateBtn) consolidateBtn.addEventListener('click', onConsolidateModels)
  const hypConsolidateBtn = byId('hypotheses-consolidate-btn')
  if (hypConsolidateBtn) hypConsolidateBtn.addEventListener('click', onConsolidateHypotheses)
  const body = byId('models-body')
  if (!body) return
  // Track expand/collapse so it survives re-render (the `toggle` event doesn't bubble — capture it).
  body.addEventListener(
    'toggle',
    (event) => {
      const d = event.target
      if (!d || !d.classList || !d.classList.contains('model-card') || !d.dataset.id) return
      if (d.open) modelExpanded.add(d.dataset.id)
      else modelExpanded.delete(d.dataset.id)
    },
    true,
  )
  body.addEventListener('change', (event) => {
    const sel = event.target.closest('select[data-action="set-status"]')
    if (sel) {
      setModelStatus(sel.dataset.id, sel.value)
      return
    }
    const sort = event.target.closest('#model-sort-select')
    if (sort) {
      modelSortKey = sort.value
      renderModels()
    }
  })
  body.addEventListener('click', (event) => {
    if (event.target.closest('summary') && event.target.closest('[data-action]')) {
      event.preventDefault()
    }
    const filterBtn = event.target.closest('button[data-category]')
    if (filterBtn) {
      modelCategoryFilter = filterBtn.dataset.category
      renderModels()
      return
    }
    const gotoPaper = event.target.closest('[data-action="goto-paper"]')
    if (gotoPaper) {
      showTab('papers')
      return
    }
    const gotoHyp = event.target.closest('[data-action="goto-hyp"]')
    if (gotoHyp) {
      showTab('hypotheses')
      return
    }
    const btn = event.target.closest('button[data-action]')
    if (!btn) return
    const { action, id } = btn.dataset
    if (action === 'import-seeds') importSeedModels()
    else if (action === 'open-model') focusModel(id)
    else if (action === 'discuss-model') chatAboutModel(id)
    else if (action === 'benchmark-device') {
      event.preventDefault() // the chip lives in <summary>; don't toggle the card open/closed on click
      onBenchmarkModelDevice(id)
    } else if (action === 'device-timings') {
      event.preventDefault()
      showDeviceTimingsModal(id)
    } else if (action === 'delete-model') onDeleteModel(id)
  })
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
  const tfSpec = (manifest && manifest.levers && manifest.levers.timeframe) || {}
  const settings = {}
  for (const [key, spec] of datasetLeverEntries()) {
    if (spec.default === undefined) continue
    let value = spec.default
    // The synthetic 'manifest defaults' bundle must be CONCRETE — never carry an input synonym like
    // fidelity_set 'auto'. Resolve fidelity_set via the auto rule on the manifest's default timeframe;
    // any other synonym falls back to the first concrete choice.
    if (INPUT_SYNONYMS.includes(String(value))) {
      value =
        key === 'fidelity_set'
          ? autoFidelity(tfSpec.default)
          : (spec.choices || []).map(String).find((c) => !INPUT_SYNONYMS.includes(c))
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
  return window.Datasets.findDuplicateDataset(manifest, datasetsCache, name, settings, exceptId)
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
// Dataset identity lives in viewer/datasets.js (pure + unit-tested); these bind the module-global manifest
// + dataset list. Identity is keyed on the manifest's scope:'dataset' levers — so the manifest must be
// fresh (see refreshProjectManifest) or a newly-scoped lever like walk_forward_window stays invisible.
function runDatasetSignature(run) {
  return window.Datasets.runDatasetSignature(manifest, run)
}
function datasetSettingsSignature(settings) {
  return window.Datasets.datasetSettingsSignature(manifest, settings)
}
function runDatasetName(run) {
  return window.Datasets.runDatasetName(manifest, allDatasets(), run)
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
    if (
      v === undefined ||
      !Number.isFinite(obj) ||
      r.summary.status === 'failed' ||
      r.summary.status === 'invalid'
    )
      continue
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
// Human-readable "applies only when reward_model = combo_unified" note for a conditional lever.
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
// The compute DEVICE the whole campaign trains on. "auto" (default) leaves each run unpinned, so the
// per-model benchmarked `preferredDevice` applies; picking cpu/mps/cuda forces it for every run (the engine
// never overrides an explicit `device`). The list is fixed (not the local machine's) since the campaign can
// run on a remote runner with a different device.
const LAUNCH_DEVICES = ['auto', 'cpu', 'mps', 'cuda']
function savedLaunchDevice() {
  try {
    const v = sessionStorage.getItem(LAUNCH_DEVICE_SS)
    return v && LAUNCH_DEVICES.indexOf(v) >= 0 ? v : 'auto'
  } catch {
    return 'auto'
  }
}
function saveLaunchDevice(value) {
  try {
    sessionStorage.setItem(LAUNCH_DEVICE_SS, LAUNCH_DEVICES.indexOf(value) >= 0 ? value : 'auto')
  } catch {
    /* sessionStorage unavailable — non-fatal */
  }
}
function launchDeviceFieldHtml(selected) {
  const sel = LAUNCH_DEVICES.indexOf(String(selected)) >= 0 ? String(selected) : 'auto'
  const opts = LAUNCH_DEVICES.map((d) => {
    const label = d === 'auto' ? 'Auto (per-model best)' : d.toUpperCase()
    return `<option value="${d}"${d === sel ? ' selected' : ''}>${label}</option>`
  }).join('')
  return `<span>Device</span>
    <select name="device">${opts}</select>
    <em class="field-hint">Auto uses each model&rsquo;s benchmarked fastest device; pick one to force it for the whole campaign.</em>`
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
        <label class="field launch-device" id="launch-device-field">${launchDeviceFieldHtml(savedLaunchDevice())}</label>
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
  // Campaign-wide compute device: pin every run's config when a specific device is chosen; "auto" leaves it
  // unset so each model's benchmarked preferredDevice applies. Held FIXED (a single choice, never swept).
  const deviceEl = form.elements.device
  const device = deviceEl && deviceEl.value
  if (device && device !== 'auto') fixed.device = device
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
    if (event.target && event.target.name === 'device') saveLaunchDevice(event.target.value)
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
// Queue-reorder glyphs: move up / down one slot, and move to the FRONT (an up arrow to a bar).
const QUEUE_ICON_UP =
  '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 12.5V4"/><path d="M4.5 7.5 8 4l3.5 3.5"/></svg>'
const QUEUE_ICON_DOWN =
  '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 3.5V12"/><path d="M4.5 8.5 8 12l3.5-3.5"/></svg>'
const QUEUE_ICON_TOP =
  '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 3.2h8"/><path d="M8 13V6.4"/><path d="M5 9 8 6l3 3"/></svg>'

// A persisted queue for one lane: each waiting entry with its type chip, reorder controls and a
// remove button. `items` is the lane's dispatch order, so index 0 is next up.
function queueSectionHtml(items, title) {
  if (!items.length) return ''
  const rows = items
    .map((item, idx) => {
      const id = escapeHtml(item.id)
      const first = idx === 0 ? ' disabled' : ''
      const last = idx === items.length - 1 ? ' disabled' : ''
      return `<li class="queue-item">
      <span class="badge queue-chip">${escapeHtml(item.activityType)}</span>
      <span class="queue-label">${escapeHtml(item.label || item.activityType)}</span>
      <button type="button" class="queue-move" data-queue-up="${id}" aria-label="Move up"${first}>${QUEUE_ICON_UP}</button>
      <button type="button" class="queue-move" data-queue-down="${id}" aria-label="Move down"${last}>${QUEUE_ICON_DOWN}</button>
      <button type="button" class="queue-move" data-queue-top="${id}" aria-label="Move to top"${first}>${QUEUE_ICON_TOP}</button>
      <button type="button" class="queue-remove" data-queue-remove="${id}" aria-label="Remove from queue">✕</button>
    </li>`
    })
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
let _activityRenderRaf = 0
// Coalesce renders: each live campaign runs its OWN 3s observe loop (plus the 1s ticker), and every one
// calls renderActivity() — which rebuilds the entire experiment+task columns. Collapse all the calls in a
// frame into ONE render of the latest state (and none while the tab is hidden — rAF pauses; onViewerVisible
// re-renders on return). Without this, K concurrent campaigns = K full HTML builds every poll.
function renderActivity() {
  if (_activityRenderRaf) return
  _activityRenderRaf = requestAnimationFrame(() => {
    _activityRenderRaf = 0
    renderActivityNow()
  })
}
function renderActivityNow() {
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
    const upBtn = event.target.closest('button[data-queue-up]')
    if (upBtn) {
      moveQueueItem(upBtn.dataset.queueUp, 'up')
      return
    }
    const downBtn = event.target.closest('button[data-queue-down]')
    if (downBtn) {
      moveQueueItem(downBtn.dataset.queueDown, 'down')
      return
    }
    const topBtn = event.target.closest('button[data-queue-top]')
    if (topBtn) {
      moveQueueItem(topBtn.dataset.queueTop, 'top')
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
  if (main) main.classList.toggle('is-fullwidth', target === 'runs' || target === 'xai')
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
  if (target === 'models') renderModels()
  if (target === 'speed') renderSpeed()
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
    btn.innerHTML = `${tab.icon ? `<span class="tab-icon" aria-hidden="true">${tab.icon(14)}</span>` : ''}<span class="tab-label">${escapeHtml(tab.label)}</span><span class="tab-live" aria-hidden="true"></span>`
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
  setupEnvironments()
  setupDatasets()
  setupHypotheses()
  setupPapers()
  setupModels()
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
