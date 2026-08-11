// A2 — full AI action parity: the single-source declaration of what a chat agent may do to a trainer
// project. Every mutating/launching user action in the viewer maps to a chat capability here, so parity
// cannot silently regress (the guard is `trainerCapabilities.parity.test.ts`, which scrapes the viewer's
// real launch + write sites and asserts each is launchable, editable, or explicitly exempt).
//
// Two surfaces:
//   • LAUNCHES go through the bespoke, approval-gated `startTrainerActivity` tool (backend), which injects
//     the mandatory project context (recordType/dir/manifestRelPath) and validates train/explore specs.
//     The generic `startProjectActivity` path can't do either — it advertises bare activity names with no
//     param hints and never injects recordType — so trainer activities are NOT declared in the manifest's
//     generic `activities` list (kept empty on purpose).
//   • RECORD EDITS ride the generic `createProjectRecord`/`updateProjectRecord` tools against the record
//     types declared below (each with an editable/creatable field allow-list).
//
// The viewer keeps a byte-behaviour twin at `viewer/trainerCapabilities.js` (pinned equal by
// `trainerCapabilitiesViewer.test.ts`), since the no-build viewer can't import this TS module.

import type {
  TrainerCapabilityRecordType,
  TrainerDataCapabilityManifest,
  TrainerExemptRecord,
  TrainerLaunchableActivity,
} from './modelTrainerTypes.js'

/**
 * Every trainer background activity a chat may launch. `requiresApi:true` ⇒ a CLI-only model is rejected at
 * admission (the activity calls the in-process LLM). Params listed here drive the tool schema + the chat
 * addendum; the launcher validates the required ones and (for train/explore) the spec before admitting.
 */
export const TRAINER_LAUNCHABLE_ACTIVITIES: TrainerLaunchableActivity[] = [
  {
    activityType: 'train',
    label: 'Train a campaign',
    description:
      'Run a training campaign from an experiment spec ({fixed?, sweep?, seeds?, environments?, datasets?} over the manifest lever names). The spec is validated against the manifest before launch (unknown levers / empty sweeps / over-cap matrices are rejected). Alternatively pass suggestionId to launch a parked AI suggestion.',
    requiredParams: ['spec'],
    optionalParams: ['refresh', 'concurrency', 'keepCheckpoints', 'crossTest', 'skipExplored'],
  },
  {
    activityType: 'explore',
    label: 'Explore (autopilot)',
    description:
      'Start the closed-loop exploration autopilot that searches the lever space for maxima, spawning train rounds until it converges or hits its budget.',
    optionalParams: ['maxRuns', 'maxConcurrent', 'targetObjective'],
  },
  {
    activityType: 'train-champion',
    label: 'Train champion (autopilot)',
    description:
      'Start the champion-training autopilot for a LEARNED core (e.g. alphazero): keep training WARM-STARTED generations against the league and PROMOTING the stronger net until strength plateaus / a target win-rate vs the strong-mcts yardstick is hit / the generation budget is spent. Each generation runs as a standard `train` experiment (so the RL training shows under Experiments). This is the compounding "find the best model" loop — distinct from explore (config-space search). `hyperparams` carries the per-generation alphazero levers (e.g. {az_sims, az_iterations, az_selfplay_games, az_epochs}).',
    optionalParams: ['maxGenerations', 'targetStrength', 'patience', 'opponent', 'evalGames', 'hyperparams'],
  },
  {
    activityType: 'rate-models',
    label: 'Rate models (leaderboard)',
    description:
      'Rank every completed model on ONE COMPARABLE strength scale by playing each against the manifest’s fixed gauntlet spine (+ champions) — no retraining. This replaces "win-rate vs whatever opponent the lever picked": a run that only beat `random` can no longer outrank one that beat a strong `mcts`. Writes a `{recordType}-leaderboard` ranked by the conservative lower bound. Use to answer "which model is actually strongest?" and "is the champion getting better?".',
    optionalParams: ['runKeys', 'gamesPerRung'],
  },
  {
    activityType: 'side-experiment',
    label: 'Run a side-experiment',
    description:
      'Run a DIAGNOSTIC matrix through the project’s training CLI that produces NO model (a baseline scan, breadth analysis, ablation or correctness probe). Its cells persist in ONE `<recordType>-experiment` record — never in the run store, which stays apples-to-apples pure — and count as evidence for a thesis alongside RL runs. Pass `thesis` (the claim it tests) and `matrix` (an experiment spec {fixed?, sweep?, seeds?, environments?, datasets?, compare?} over the manifest lever names, validated before launch), plus `hypothesisId` to link it to a hypothesis so its cells count as that thesis’s evidence.',
    requiredParams: ['thesis', 'matrix'],
    optionalParams: ['hypothesisId', 'thesisTarget', 'refresh', 'concurrency'],
  },
  {
    activityType: 'judge',
    label: 'Judge runs',
    description:
      'LLM-judge runs against the scorecard, writing a verdict per run. Judges the given runKeys, else all un-judged completed runs.',
    optionalParams: ['runKeys', 'instructions'],
    requiresApi: true,
  },
  {
    activityType: 'evaluate',
    label: 'Evaluate checkpoints',
    description: 'Re-test saved checkpoints (no training) for the given runs, writing fresh eval metrics.',
    requiredParams: ['runKeys'],
    optionalParams: ['concurrency'],
  },
  {
    activityType: 'cross-test',
    label: 'Cross-test a run',
    description:
      'Replay a run’s checkpoint on other values of a lever (default asset) with no retrain, to test generalisation.',
    requiredParams: ['runKey'],
    optionalParams: ['lever', 'values'],
  },
  {
    activityType: 'continue-training',
    label: 'Continue training',
    description: 'Extend a run from its checkpoint (optionally on new datasets/seeds).',
    requiredParams: ['runKey'],
    optionalParams: ['datasetOverrides', 'seeds', 'refresh', 'concurrency'],
  },
  {
    activityType: 'config-space-analyze',
    label: 'Analyse config space',
    description:
      'Deterministic lever-importance / config-space analysis over the run cohort for a fitness criterion.',
    optionalParams: ['criterionKey', 'criterionDir', 'environment'],
  },
  {
    activityType: 'diagnose',
    label: 'Diagnose the search',
    description:
      'Compute the deterministic search diagnosis + the composite champion "declare steady" verdict over the run cohort and cache it as the `-diagnosis` record the Diagnosis tab reads. Read-only analysis (no training) — the single source of the champion verdict the viewer renders.',
    optionalParams: ['project'],
  },
  {
    activityType: 'run-xai-analyze',
    label: 'Analyse a run (xAI)',
    description: 'Compute + cache the deterministic xAI analysis of one run (attribution, latent probe, rank).',
    requiredParams: ['runKey'],
    optionalParams: ['siblingKey', 'criterionKey', 'criterionDir'],
  },
  {
    activityType: 'xai-narrate',
    label: 'Narrate a run (xAI)',
    description: 'LLM-narrate one run’s xAI analysis into a plain-language reliability read.',
    requiredParams: ['runKey'],
    optionalParams: ['siblingKey', 'criterionKey', 'criterionDir', 'criterionLabel'],
    requiresApi: true,
  },
  {
    activityType: 'propose',
    label: 'Propose hypotheses',
    description: 'LLM-propose new hypothesis records from the current run evidence.',
    optionalParams: ['count', 'instructions'],
    requiresApi: true,
  },
  {
    activityType: 'propose-experiments',
    label: 'Propose experiments',
    description:
      'LLM-propose runnable experiment suggestions (parked in the xAI Suggested view). Prefer recommendTrainingExperiments when YOU already know the specs to try.',
    optionalParams: ['count', 'instructions'],
    requiresApi: true,
  },
  {
    activityType: 'analyze-paper',
    label: 'Analyse a paper URL',
    description: 'Read a paper/source URL and draft a paper record + its testable hypotheses.',
    requiredParams: ['url'],
    optionalParams: ['notes'],
    requiresApi: true,
  },
  {
    activityType: 'research-training-papers',
    label: 'Research papers',
    description: 'Open-ended web research for relevant papers, drafting paper records for the ones that survive.',
    optionalParams: ['count', 'notes', 'researchBudget'],
  },
  {
    activityType: 'suggest-paper-hypotheses',
    label: 'Suggest paper hypotheses',
    description: 'LLM-extract testable hypotheses from one paper record.',
    requiredParams: ['paperId'],
    requiresApi: true,
  },
  {
    activityType: 'weigh-paper-hypotheses',
    label: 'Re-weigh paper hypotheses',
    description: 'Re-score a paper’s hypotheses against the current evidence.',
    requiredParams: ['paperId'],
    requiresApi: true,
  },
  {
    activityType: 'analyze-paper-models',
    label: 'Find paper models',
    description: 'Identify the model architectures a paper relies on and link them into the model catalog.',
    requiredParams: ['paperId'],
    requiresApi: true,
  },
  {
    activityType: 'scan-models',
    label: 'Scan models',
    description: 'LLM-scan the project’s code + runs to catalog the model architectures it uses.',
    requiresApi: true,
  },
  {
    activityType: 'consolidate-models',
    label: 'Consolidate models',
    description: 'Propose merges of duplicate/near-duplicate model catalog entries (proposal only).',
    requiresApi: true,
  },
  {
    activityType: 'consolidate-hypotheses',
    label: 'Consolidate hypotheses',
    description: 'Deterministically fold same-core hypotheses together.',
  },
  {
    activityType: 'benchmark-model-device',
    label: 'Benchmark a model device',
    description: 'Benchmark a model architecture CPU vs MPS and persist its preferred device.',
    requiredParams: ['modelId'],
    optionalParams: ['modelName', 'computeTarget'],
  },
  {
    activityType: 'data-catalog',
    label: 'Scan data catalog',
    description: 'Read-only scan of the project’s on-disk data into the data-catalog record.',
  },
  {
    activityType: 'mine-data',
    label: 'Mine data',
    description:
      'Idempotently download/derive data for symbols or a whole asset class, then rescan the catalog.',
    optionalParams: ['symbols', 'class', 'intervals', 'through'],
  },
  {
    activityType: 'discover-data',
    label: 'Discover data sources',
    description: 'Research candidate data sources for a problem statement, drafting datasource proposals.',
    requiredParams: ['problemStatement'],
    optionalParams: ['goal', 'limit', 'researchBudget'],
  },
  {
    activityType: 'approve-data-source',
    label: 'Approve a data source',
    description: 'Approve a proposed data source (optionally kicking off a mine of its symbols/class).',
    requiredParams: ['sourceId'],
    optionalParams: ['symbols', 'class'],
  },
]

/**
 * Activities the chat deliberately CANNOT launch. `inspect-trainer` is the hub bootstrap — it re-reads the
 * on-disk manifest into the `trainer-project-manifest` record and needs hub params (projectKey/dir), not a
 * within-project operation; it runs when a project is added/inspected, before any chat context exists.
 */
export const TRAINER_EXEMPT_ACTIVITY_TYPES: readonly string[] = ['inspect-trainer']

/**
 * The DataStorage record types a chat may query — and, where flagged, create/edit — via the generic
 * project-data tools. `suffix` is appended to the manifest `recordType` (`''` = the run record); `fixedType`
 * is an absolute type. The first five reproduce the pre-A2 declaration verbatim; the rest are A2 additions.
 */
export const TRAINER_CAPABILITY_RECORD_TYPES: TrainerCapabilityRecordType[] = [
  {
    suffix: '',
    label: 'Training run',
    description: 'A completed run: its config levers, metrics, dataset, objective and status.',
    view: { view: 'runs', keyParam: 'run' },
  },
  {
    suffix: '-hypothesis',
    label: 'Hypothesis',
    description:
      'A tested claim about which lever settings help. To override a verdict, set BOTH status (proven/disproved/untested) and verdictSource="manual". Create a new one with a `spec` ({fixed?, sweep?, seeds?, environments?, datasets?, compare?}) using the manifest’s declared lever names.',
    editable: true,
    editableFields: ['title', 'claim', 'rationale', 'verdictNote', 'dismissed', 'status', 'verdictSource'],
    creatable: true,
    creatableFields: ['title', 'claim', 'rationale', 'spec', 'comparison'],
    createDefaults: { status: 'untested', verdictSource: 'auto', source: 'llm' },
    view: { view: 'hypotheses', keyParam: 'focus' },
  },
  {
    suffix: '-experiment',
    label: 'Side experiment',
    description:
      'A diagnostic side-experiment: a matrix of cells run through the SAME CLI as training but producing NO model (baseline scans, breadth, ablations), persisted as thesis evidence WITHOUT polluting the run store. Link it to a hypothesis with `hypothesisId` so its cells count as that thesis’s evidence alongside RL runs. Create with `thesis`, a `matrix` ({fixed?, sweep?, seeds?, environments?, datasets?, compare?} over declared lever names) and optional `hypothesisId`/`thesisTarget`. Key is the matrix hash.',
    editable: true,
    editableFields: ['thesis', 'thesisTarget', 'hypothesisId', 'status'],
    creatable: true,
    creatableFields: ['thesis', 'thesisTarget', 'hypothesisId', 'matrix'],
    createDefaults: { status: 'queued', source: 'llm' },
    // No keyParam: an experiment's key is its MATRIX hash, not a hypothesis id — passing it as `focus`
    // would deep-link to a hypothesis that doesn't exist. The link opens the registry the evidence feeds.
    view: { view: 'hypotheses' },
  },
  {
    suffix: '-paper',
    label: 'Paper',
    description: 'A research paper and its testable hypotheses.',
    editable: true,
    editableFields: ['title', 'claim', 'approach', 'verdictNote', 'url', 'authors', 'dismissed', 'year', 'tags'],
    creatable: true,
    creatableFields: ['title', 'claim', 'approach', 'url', 'authors', 'year', 'tags'],
    createDefaults: { status: 'untested', source: 'research' },
    view: { view: 'papers', keyParam: 'paper' },
  },
  {
    suffix: '-scorecard',
    label: 'Scorecard',
    description:
      'A named definition of "good": `gates` (accept/reject predicates over run-summary metrics, each {metric, op (one of >,>=,<,<=,==,!=), value (a number or {metric} to compare against another metric), label?}) + `fitness` (ranking objectives, each {metric, direction (max|min)}), separate from the training reward. The "Default" is seeded from the manifest; create alternates to score the same runs by a different bar. Which card is ACTIVE (drives the Runs table) is set via the Scorecards tab.',
    editable: true,
    editableFields: ['name', 'description', 'gates', 'fitness'],
    creatable: true,
    creatableFields: ['name', 'description', 'gates', 'fitness'],
    createDefaults: { source: 'llm' },
    view: { view: 'scorecards' },
  },
  {
    suffix: '-xai-suggestion',
    label: 'AI experiment suggestion',
    description:
      'A runnable experiment the AI recommended (parked in the xAI Suggested view). Launch one with startTrainerActivity {activityType:"train", suggestionId:<id>}.',
    view: { view: 'xai', params: { scope: 'all' } },
  },
  // --- A2 additions: the project config + per-run flags the chat can now edit/create ------------------
  {
    suffix: '-environment',
    label: 'Environment',
    description:
      'A named set of environment-lever values a model trains under. `settings` maps each environment-lever name to its value; `default` flags the one used when none is chosen. Key is the environment id.',
    editable: true,
    editableFields: ['name', 'settings', 'default'],
    creatable: true,
    creatableFields: ['id', 'name', 'settings', 'default'],
  },
  {
    suffix: '-dataset',
    label: 'Dataset',
    description:
      'A named set of dataset-lever values (which data a model runs against — asset, time window, fidelity). `settings` maps each dataset-lever name to its value; `default` flags the one launched by default. Key is the dataset id.',
    editable: true,
    editableFields: ['name', 'settings', 'default'],
    creatable: true,
    creatableFields: ['id', 'name', 'settings', 'default'],
  },
  {
    suffix: '-model',
    label: 'Model',
    description:
      'A model-architecture catalog entry. Set `status` (and it becomes a manual status) to mark it tried / rejected / shortlisted.',
    editable: true,
    editableFields: ['status'],
    view: { view: 'models' },
  },
  {
    suffix: '-reliability',
    label: 'Run reliability override',
    description:
      'Overturn a run’s reliability verdict as an explicit decision. `level` is the project’s verdict level (e.g. ok / warn / bad); the record key is the run id.',
    editable: true,
    editableFields: ['level'],
    creatable: true,
    creatableFields: ['runKey', 'level'],
    createDefaults: { source: 'user', reasons: [] },
  },
  {
    suffix: '-unrunnable',
    label: 'Unrunnable setup',
    description:
      'Mark a setup (a config signature) unrunnable for the current pipeline version so the planner skips it. The record key is the setupKey.',
    editable: true,
    editableFields: ['unrunnable'],
    creatable: true,
    creatableFields: ['setupKey', 'unrunnable', 'pipelineVersion'],
    createDefaults: { unrunnable: true },
  },
  {
    suffix: '-filter-rule',
    label: 'Runs filter rule',
    description:
      'A saved numeric filter for the Runs table: {field, op (one of >,>=,<,<=,==,!=), value}. One record per rule (key = its id).',
    editable: true,
    editableFields: ['field', 'op', 'value', 'values'],
    creatable: true,
    creatableFields: ['id', 'field', 'op', 'value', 'values'],
    view: { view: 'runs' },
  },
  {
    suffix: '-hypothesis-config',
    label: 'Hypothesis judging config',
    description:
      'Project setting: `minRuns` = how many matching runs a hypothesis needs before it is judged. Singleton, key "latest".',
    editable: true,
    editableFields: ['minRuns'],
    creatable: true,
    creatableFields: ['minRuns'],
  },
  {
    suffix: '-scorecard-active',
    label: 'Active scorecard pointer',
    description:
      'Singleton {activeId} naming which scorecard drives the Runs verdict/sort/filter. Set activeId to switch. Key "active".',
    editable: true,
    editableFields: ['activeId'],
    creatable: true,
    creatableFields: ['activeId'],
  },
  {
    suffix: '-favorites',
    label: 'Favorite runs',
    description: 'Singleton {keys:[runId,…]} of favorited runs (quick-pick in xAI). Key "favorites".',
    editable: true,
    editableFields: ['keys'],
  },
  {
    suffix: '-strategy',
    label: 'Campaign strategy',
    description:
      'The AI companion’s working memory for driving this project — the current plan and reasoning, surfaced in the app so the human sees the AI’s thinking. `summary` = the current strategy in prose; `decided` = settled conclusions (string[]); `open` = open questions (string[]); `nextSteps` = the experiments to run next and why (string[]). Singleton, key "latest" — READ it to resume the plan across the long launch→wait→read→decide loop, and UPDATE it as decisions land. The Diagnosis plan + hypotheses are the evidence it reasons over.',
    editable: true,
    editableFields: ['summary', 'decided', 'open', 'nextSteps'],
    creatable: true,
    creatableFields: ['summary', 'decided', 'open', 'nextSteps'],
    createDefaults: { source: 'llm' },
    view: { view: 'diagnosis' },
  },
  {
    fixedType: 'trainer-activity-limits',
    label: 'Activity concurrency limits',
    description:
      'Per-lane server concurrency budgets {experiment, task} — how many training vs task activities run at once. Singleton, key "latest".',
    editable: true,
    editableFields: ['experiment', 'task'],
    creatable: true,
    creatableFields: ['experiment', 'task'],
  },
]

/**
 * Record types the chat deliberately CANNOT create/edit, each with the reason. UI state and per-user
 * tracking are transient; derived records are recomputed by their producing activity (and cascade-deleted
 * with the run via the bespoke deleteRuns tool); hub registration is a home-shell operation, not a
 * within-project chat action.
 */
export const TRAINER_EXEMPT_RECORDS: TrainerExemptRecord[] = [
  { suffix: '-dismissed-failure', reason: 'UI state — a run dismissed from the Activity list.' },
  { suffix: '-model-stats', reason: 'Backend-derived aggregate — recomputed from the runs.' },
  { suffix: '-exploration', reason: 'Autopilot-managed search state — owned by the explore activity.' },
  { suffix: '-evaluation', reason: 'Derived by the evaluate activity; cascade-deleted with the run.' },
  { suffix: '-settest', reason: 'Derived by the cross-test activity; cascade-deleted with the run.' },
  { suffix: '-verdict', reason: 'Derived by the judge activity; cascade-deleted with the run.' },
  { suffix: '-xai-narrative', reason: 'Derived by the xai-narrate activity; cascade-deleted with the run.' },
  { fixedType: 'trainer-seen', reason: 'Per-user UI state — which runs have been seen (unseen badge).' },
  { fixedType: 'trainer-queue', reason: 'Infra — host activity-queue markers.' },
  { fixedType: 'trainer-queue-order', reason: 'UI state — the queued-activity drain order.' },
  { fixedType: 'trainer-project', reason: 'Hub registration — a home-shell action, not a within-project edit.' },
]

/**
 * The RUN-KEYED derived children removed when a run is deleted (each keyed by the run's own key). This is the
 * SINGLE SOURCE consumed by the engine's cascade (`ModelTrainerTools` imports it) and mirrored by the viewer
 * twin (`deleteRelatedRunRecords` iterates it) — so the bespoke deleteRuns tool, the migration delete path,
 * and the viewer's own delete all cascade identically.
 */
export const TRAINER_RUN_KEYED_CHILD_SUFFIXES: readonly string[] = [
  '-evaluation',
  '-settest',
  '-verdict',
  '-xai-narrative',
  '-reliability',
]

/** The SETUP-keyed marker removed with a run (keyed by the run's setupKey, not its run key). */
export const TRAINER_RUN_SETUP_KEYED_SUFFIX = '-unrunnable'

/** Every derived-record suffix a run delete touches (run-keyed children + the setup-keyed marker). */
export const TRAINER_RUN_DELETE_CASCADE_SUFFIXES: readonly string[] = [
  ...TRAINER_RUN_KEYED_CHILD_SUFFIXES,
  TRAINER_RUN_SETUP_KEYED_SUFFIX,
]

/** The absolute launchable activity types, derived so the tool schema + audit never drift from the catalog. */
export const TRAINER_LAUNCHABLE_ACTIVITY_TYPES: readonly string[] = TRAINER_LAUNCHABLE_ACTIVITIES.map(
  (a) => a.activityType,
)

/**
 * Build the data-capability manifest a trainer project reports for `recordType`, resolving every suffix to
 * an absolute type. `activities` is empty on purpose — trainer launches use the bespoke startTrainerActivity
 * tool, not the generic startProjectActivity path (see the module header).
 */
export function buildTrainerDataCapabilityManifest(recordType: string): TrainerDataCapabilityManifest {
  return {
    types: TRAINER_CAPABILITY_RECORD_TYPES.map((t) => {
      const type = t.fixedType !== undefined ? t.fixedType : `${recordType}${t.suffix ?? ''}`
      const out: TrainerDataCapabilityManifest['types'][number] = { type, label: t.label, description: t.description }
      if (t.editable) out.editable = true
      if (t.editableFields) out.editableFields = t.editableFields
      if (t.creatable) out.creatable = true
      if (t.creatableFields) out.creatableFields = t.creatableFields
      if (t.createDefaults) out.createDefaults = t.createDefaults
      if (t.view) out.view = t.view
      return out
    }),
    activities: [],
  }
}
