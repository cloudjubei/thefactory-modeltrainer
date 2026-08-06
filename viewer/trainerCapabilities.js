// The no-build browser twin of the engine's src/trainerCapabilities.ts — ONLY the record-type declarations
// the viewer reports (the launchable-activity catalog is backend-only). `trainerDataCapabilityManifest` in
// app.js calls buildTrainerDataCapabilityManifest to declare which record types the Overseer chat's generic
// project-data tools may query/create/edit. Kept byte-behaviour equal to the engine (pinned by
// src/trainerCapabilitiesViewer.test.ts) so the viewer, the backend tool, and the parity audit never drift.
// Pure + dual-loaded (browser `window.TrainerCapabilities` + node `module.exports`).
;(function (root) {
  'use strict'

  // The DataStorage record types a chat may query — and, where flagged, create/edit — via the generic tools.
  // `suffix` is appended to the manifest recordType (`''` = the run record); `fixedType` is an absolute type.
  var TRAINER_CAPABILITY_RECORD_TYPES = [
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

  // Build the reported manifest for `recordType`, resolving every suffix to an absolute type. `activities` is
  // empty on purpose — trainer launches use the bespoke startTrainerActivity backend tool, not the generic
  // startProjectActivity path (which can't inject the mandatory recordType or describe an activity's params).
  function buildTrainerDataCapabilityManifest(recordType) {
    return {
      types: TRAINER_CAPABILITY_RECORD_TYPES.map(function (t) {
        var type = t.fixedType !== undefined ? t.fixedType : recordType + (t.suffix || '')
        var out = { type: type, label: t.label, description: t.description }
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

  // The RUN-KEYED derived children a run delete removes (mirror of the engine's TRAINER_RUN_KEYED_CHILD_SUFFIXES,
  // pinned equal by trainerCapabilitiesViewer.test.ts). deleteRelatedRunRecords iterates this so the viewer's
  // cascade can't drift from the engine's deleteRuns cascade. (The setup-keyed '-unrunnable' is handled separately.)
  var TRAINER_RUN_KEYED_CHILD_SUFFIXES = [
    '-evaluation',
    '-settest',
    '-verdict',
    '-xai-narrative',
    '-reliability',
  ]

  var TrainerCapabilities = {
    TRAINER_CAPABILITY_RECORD_TYPES: TRAINER_CAPABILITY_RECORD_TYPES,
    TRAINER_RUN_KEYED_CHILD_SUFFIXES: TRAINER_RUN_KEYED_CHILD_SUFFIXES,
    buildTrainerDataCapabilityManifest: buildTrainerDataCapabilityManifest,
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = TrainerCapabilities
  if (root) root.TrainerCapabilities = TrainerCapabilities
})(typeof window !== 'undefined' ? window : null)
