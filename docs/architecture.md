# Architecture

How a training campaign flows through the system, and which repo owns which piece.

**The hub model:** ONE Overseer project (the thefactory-modeltrainer checkout itself, with
`hasApp` on and `appDir: "viewer"` in its project settings) hosts the Model Trainer app.
Training projects
(cartpole, BlackSwan, …) are **not Overseer projects** — they are directories registered
_inside_ the app (`trainer-project` records; relative to the host checkout or absolute).
The `inspect-trainer` activity reads each one's manifest server-side into a
`trainer-project-manifest` record (so the app can render a launch form for any directory),
and every `train`/`judge`/`propose` activity carries the target's `dir`, resolved against
the host checkout. All records live in the host project's scope, namespaced per training
project by its manifest's `recordType`.

```
┌─ Overseer client (web/desktop/mobile) ──────────────────────────────────────┐
│  App tab (host project) → sandboxed iframe → the hub app (viewer/, appDir)  │
│  home: registered training projects → per-project dashboard                 │
│  window.OverseerBridge: queryData / putData / startActivity / abort         │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │ postMessage → host → REST
┌─ thefactory-backend ─────────────▼───────────────────────────────────────────┐
│  POST /projects/:id/activities/run { activityType, params: { dir, … } }      │
│  activityDefinitions.ts: inspect-trainer / train / judge / propose           │
│    projectRoot = resolve(host checkout, params.dir)  (absolute passes thru)  │
│    run(ctx) → ctx.modelTrainerTools.*  → records + updateStep → WS events    │
│  server.ts composes createModelTrainerTools({ computeRunner, storage,        │
│    inferenceExecutor }) once                                                  │
│  GET /projects/:id/view/* serves the checkout root, or the project-config    │
│    `appDir` subdir when set (editable in project settings)                   │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
┌─ thefactory-modeltrainer (this repo, src/) ──▼───────────────────────────────┐
│  createModelTrainerTools: manifest → plan matrix → skip-if-fresh →           │
│  per-item ComputeRunner job → validate RunSummary → upsert {recordType}      │
│  record → progress callbacks → best-run selection                            │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │ ComputeRunner seam (thefactory-tools)
┌─ thefactory-tools ───────────────▼───────────────────────────────────────────┐
│  src/computeRunner: ComputeJob/Handle/Result (+dataFiles), LocalComputeRunner│
│  (streaming spawn, temp config/summary, materialises declared data first)    │
│  src/dataCache: ContentAddressedDataCache (sha256 objects + index,           │
│  hardlink materialise, fetch-only-misses)                                    │
│  src/activity: runActivityWorkItems (plan→fresh→run→progress loop)           │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │ spawn `{run template}` in the checkout
┌─ a trainer-conformant project (e.g. examples/cartpole, BlackSwan later) ─────┐
│  .factory/trainer.json (TrainerManifest)                                     │
│  python -m trainer.run --config-json X --summary-out Y [--calibrate]         │
│  writes RunSummary JSON; checkpoints stay in the checkout                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Ownership

| Piece                                                          | Repo                                          | Why there                               |
| -------------------------------------------------------------- | --------------------------------------------- | --------------------------------------- |
| TrainerManifest/RunSummary contract + docs                     | this repo (`docs/model-training-standard.md`) | the domain standard                     |
| Matrix planner, campaign loop, judge/propose (Phase 5)         | this repo `src/`                              | the training domain                     |
| The hub app (home + per-project dashboards)                    | this repo `viewer/`                           | one app manages every training project  |
| ComputeRunner seam + LocalComputeRunner + work-item engine     | thefactory-tools                              | generic infra, reusable beyond training |
| `train` activity registration + composition + app-view serving | thefactory-backend                            | host wiring (mirrors `recommendTools`)  |
| Model specifics (levers, objective, training code)             | each conformant project                       | the engine stays domain-oblivious       |

## Key decisions

- **Records over files**: every result is a DataStorage record (`scope = the HOST projectId`,
  type from each training project's `manifest.recordType`), so the viewer, badge, and resume
  all work from the same substrate and survive restarts. Run identity = 12-hex hash of the
  resolved config — skip-if-fresh and re-launch idempotency fall out of that.
- **Training projects are registered, not forked**: a `trainer-project` record ({name, dir})
  plus an `inspect-trainer` pass is all it takes to bring a directory under management —
  no Overseer project per model, no clone.
- **The engine never reads model code**: it knows `.factory/trainer.json`, two command
  templates, and the RunSummary shape. Conformance, not integration.
- **Calibrate-first ETA**: campaigns optionally start with the manifest's tiny calibrate run;
  `unitsPerSecond` × the plan's total units gives the upfront ETA streamed into progress.
- **Abort/resume**: abort flows an AbortSignal from the activity into the spawned process;
  resume re-launches the same campaign and skip-if-fresh makes completed items free.
- **Failures are per-item**: a failed run records a `failed` count and the campaign continues;
  a completed-but-malformed summary is a failure (never silently ingested).
- **Explainability is an opt-in artifact, not a code path**: a project may attach a domain-oblivious
  `artifacts.decisionTrace` (`DecisionTrace`) the hub's Explain view renders (action distribution,
  per-action value over time, confidence, input attribution — run-aggregate AND per-step by input GROUP,
  the temporal `saliencyByGroup` companion summarised by `summarizeStepAttribution`). The engine never
  computes it — each project emits its own (BlackSwan replays its deterministic test once more to capture
  per-step confidence/Q-values + saliency by a pluggable method: gradient-saliency, integrated-gradients,
  occlusion, or permutation `tabular-shap`, each Adebayo model-randomization sanity-checked); the engine
  only soft-validates it (`validateDecisionTrace`,
  dropping an unusable trace) so a run without one ingests normally. Two runs that share a
  dataset/window are diffable step-by-step (`diffDecisionTraces`) — the viewer's Compare pane reads how
  a lever tweak changed the model's DECISIONS, with a "heuristic, not causal" decision-quality verdict
  (reward delta at the divergent steps, controlled against unchanged steps) kept separate from the
  objective — all off the already-emitted trace, no extra engine or trainer infra.
- **xAI is deterministic analysis over records, not a model call**: `src/xaiUtils.ts` (pure, domain-
  oblivious) computes config-effect analysis — one-factor-at-a-time contrasts (no confounding), IQM +
  seeded-bootstrap CIs, difference-CI significance + Benjamini-Hochberg FDR, lever importance — and a
  non-LLM experiment recommender, over the stored run records (config + metrics + trace). The sandboxed
  viewer can't import the build, so `viewer/xai.js` is a parity-mirrored copy (a `scripts/` harness
  asserts byte-identical results); the xAI tab renders it and fires the recommender's specs as batched
  `train` campaigns, closing an analyse→run→re-analyse loop with no LLM in it. A seeded random-forest
  surrogate (`fitConfigSurrogate`) over (config → criterion) adds the global view — fANOVA importance, a
  greedy ablation tree, and a 2-lever interaction grid — predicting unobserved configs (the determinism is
  load-bearing: the forest is seeded from the data, so analysis never drifts between runs).
- **Current-run across-axis views pool over an axis, hold everything else fixed** (`viewer/comparison.js`,
  pure, tested in `comparisonViewer.test.ts`): a run is in the pool iff `sameSetupExceptAxis` — it equals the
  focus config on EVERY locked lever (exhaustive + strict, `unset ≡ null ≡ 'n/a'`, so a lever the focus leaves
  unset can't leak in a run that sets it), differing only in the AXIS — the dataset levers ("By dataset"), the
  environment levers ("By environment"), or one chosen tunable lever ("By value", one-factor-at-a-time). So a
  row is exactly one config's seeds. Each row shows the metric [min·avg·max] over seeds plus its normalised
  STANDING (robust-z within that axis value); `robustnessVerdict` classifies robust/mixed/weak. `seed` is
  never a locked or axis lever (a nuisance param pooled over, matching `setupKeyOfRun`). Regime slice toggles
  narrow the pool (dataset by `timeframe`, environment by `allow_shorting`/`no_sell_action`); levers the
  manifest marks `active: false` (declared but not wired into the sim — `position_sizing`/`transaction_fee`/
  `vol_target`) are hidden from the value line via `isLeverActive` and never swept. Columns sort by any metric
  (a numeric axis — By value lever values — sorts numerically, not lexically); rows select to **Add runs**
  with fresh seeds, or **Sweep** the axis for first-time-only cells (By value via a recommended-values popup —
  numeric CHOICE levers like `batch_size`/`lookback_window` get a full grid, not just their presets).
- **"Sweep all datasets/environments" fires ONE campaign of pruned, deduped bundles**, not a blown-up
  cartesian (`axisSweepBundleSpec` → `spec.environments`/`spec.datasets`, each bundle a complete set of the
  active axis levers). Candidate values come from `xaiAxisSweepValues` — a boolean's two states, a choice's
  options, else the values already OBSERVED across runs (no invented numeric grid, so the environment's
  numeric exit levers stay at their handful of real values). `axisSweepCombos` then collapses any lever whose
  manifest `dependsOn` control makes it inert (`trailing_take_profit` off unless `take_profit` is on;
  `no_sell_action` off while `allow_shorting`, since the sim ignores it there) to its off value and dedupes —
  so an environment sweep is the ~dozens of runs that differ in behaviour, not 100k+. Launches go through
  `xaiLaunchBatch`, which stays on the xAI tab.
- **Every "open these runs" gesture lands in the Selection view** (`openRunsSelection`): a By value / By
  dataset / By environment row's "runs ↗" (the runs BEHIND that row), a comparison drill, a fANOVA/interaction
  cell, or a hypothesis's runs all route to one `runsViewMode: 'selection'` tab that holds the LAST selection,
  with a ← Back that restores where it was opened from (an `{ label, back() }` origin). Favorites + Selection
  are curated key-list views: resolved from the full-runs snapshot (`findRunAnywhere`, fetched by key) so a
  member never vanishes off-page, and the exploratory filters never drop one (text search only). The xAI
  favorite picker jumps to a run in Runs via **View in Runs**.
- **LLM only synthesises the xAI, never computes it**: a thin layer sits on top of the deterministic engine.
  `xaiNarrate` (tool) digests ONE run's own deterministic xAI server-side (its action mix, input attribution
  - the Adebayo sanity verdict, reward breakdown, latent probe, the decision-diff vs a passed-in sibling,
    plus its rank + cross-run lever importances for context), then makes ONE inference to write a short
    **per-run narrative** — a `{recordType}-xai-narrative` record KEYED BY RUN, via the `xai-narrate` activity.
    The viewer shows the focused run's narrative at the top of the tab with an "N new runs since" refresh (the
    cross-run context drifts as the campaign grows); selecting a different run shows that run's own narrative.
    A **"Discuss xAI"** button seeds an interactive chat with the same per-run analysis. A **"Propose with AI"**
    button feeds the lever-importance + gap signal as instructions into the existing `proposeTrainingHypotheses`.
    The deterministic records stay the source of truth; the model interprets them and is told to hedge on the
    confounded/surrogate signals (e.g. an attribution that FAILED its sanity check). A chat agent can also pull
    ANY run mid-conversation (not just the seeded one) via two AGENT READ TOOLS — `getRunData` (a run's
    config/metrics + a compact decision-trace digest) and `getRunXAI` (the same deterministic `RunXaiDigest`)
    — advertised on the project chat through the backend's `extraToolSchemas` seam (`trainerReadTools.ts`,
    the knowledge-read-tools precedent; no ToolSchemas regen). They resolve a run id → its training project by
    searching the host's `trainer-project-manifest` records, and the `getRunXAI`/narrative facts share ONE
    `buildRunXaiDigest`. Read-only.
- **Judging blends, never replaces, the objective**: `judgeTrainingRuns` min–max-normalises
  the objective (direction-aware) and blends it 50/50 with the LLM's 0–100 verdict
  (`{recordType}-verdict` records, key = run key) — a money-losing run can't be ranked best
  by prose. Health-flagged runs are auto-rejected without spending an LLM call. Runs the LLM
  skips keep an objective-only verdict.
- **Hypotheses are the one registry — claims runs prove or disprove**: a `{recordType}-hypothesis`
  record's `spec` both launches its runs AND identifies them (a run is evidence iff its config is
  consistent with the spec); its verdict (untested/proven/disproved) auto-derives from those runs
  (beats-buy-and-hold OOS), re-checks on settle/tab-open, and records which runs flipped it
  (`transitions[]`). Identity = the spec hash, so identical specs from any source (LLM
  `proposeTrainingHypotheses`, manual add, paper Extract, migrated model architecture) dedupe.
  Pure decision logic lives in node-tested `viewer/hypothesis.js` (the `migrate.js`/`xai.js`
  dual-loaded precedent). **Papers** are containers of `hypothesisIds[]` (Extract / add / link),
  verdict rolled up. A model architecture can still BE a hypothesis (its `spec.fixed` pins
  `model_name`), but cataloguing — "what models do we have, what do they need" — is its own concern,
  the **Models catalog** below.
- **Models are the catalog — an aggregating layer over runs, papers and hypotheses**: a
  `<recordType>-model` record (key = `slug`) names a model architecture/algorithm the project can
  train. It OWNS its runs by binding one or more `model_name` lever values (`modelNames`): a run
  trains it iff its `config.model_name` is one of them, so its status (`proposed` / `implemented` /
  `failing`) AUTO-derives from those runs (`deriveModelStatus` — proposed when unimplemented +
  unrun, implemented when lever-bound or `implPath`-backed, failing when every matching run is
  health-flagged) just like a hypothesis verdict, with `needs-improvement` / `deprecated` as manual
  pins. It LINKS the papers that introduce/improve it (`paperIds`) and the hypotheses that test it
  (`hypothesisIds`); the viewer unions both link directions. Pure decision logic lives in node-tested
  `viewer/models.js`; the LLM-facing build logic (slug/category/humanize heuristics, scan/analyse
  prompt-builders + coercers) lives in `modelTrainerUtils.ts`. Three population paths: manifest
  `models[]` SEEDS (imported once by slug, parallel to `papers[]`/`hypotheses[]`); **Scan Project**
  (`scanProjectModels` → `scan-models` activity: discovers `model_name` choices the catalog doesn't
  cover, heuristic-first, LLM-enriched — the engine reads the manifest only, never model code); and
  the Papers tab's **Find models** (`analyzePaperModels` → `analyze-paper-models` activity: an LLM
  links a paper to the catalog models it is about AND names the ones it proposes with no entry yet —
  returned, not persisted, so the card offers a one-click "Add to catalog" that writes a `proposed`
  record). "Discuss" seeds a project chat (the `discussTopic` seam, like a run) whose seed adapts to
  status — implement (proposed), fix (failing), or improve — so an agent can be put to work on any
  model. A model's `category` (`rl` / `supervised` / `baseline` / `component`) groups the tab;
  `component` entries catalog the reusable BUILDING BLOCKS (feature extractors, policies/Q-nets, replay
  buffers, optimizers, NN blocks) and each model flavor declares the `components` it is composed of,
  rendered as linked chips with a reverse "used by" on the component card (`flavorComponents` /
  `modelsUsingComponent`; `components` is descriptive only — it never affects run matching). The engine
  stays domain-oblivious: a model is data + the `model_name` binding.
- **Datasets/environments are user-managed bundles with a settable default**: levers tagged
  `scope: "dataset"`/`"environment"` aren't model knobs — they're managed in their own tabs as
  named `{recordType}-dataset`/`-environment` records (`{id, name, settings, default}`). One record
  holds the `default` flag (the launch picker pre-checks it; removing it promotes the next record);
  the first one created becomes the default, and a save is refused if a record with the same name or
  settings already exists. The manifest-defaults card stays only as a read-only clone-to-start seed.
  A project that declares such levers but has none defined cannot launch (a valid just-started state).
- **A named dataset always pins a CONCRETE identity**: runs group by the VALUE signature of their
  dataset/env levers, so the input-only `fidelity_set: "auto"` synonym would fragment a run away from its
  explicit-dataset siblings. The dataset form therefore requires a concrete pick (no "— default —"
  escape), the runs filter hides input synonyms, and the synthetic manifest-defaults seed is shown
  resolved via the auto rule (`Migrate.autoFidelity` / `Migrate.INPUT_SYNONYMS` in `viewer/migrate.js`,
  mirroring `fidelity.py`) — so `auto` never enters a newly-created dataset or surfaces in the UI.

## Remote compute

A campaign (or evaluation) names a `computeTarget` — the engine resolves it through the
backend's `resolveComputeRunner` to either the local runner or a **paired remote runner**.
Pairing is a one-time PIN exchange (Overseer Settings → Compute Runners shows the PIN; the
runner claims it on the unauthenticated `pair` endpoint and receives its bearer token once —
only the hash is stored, as a `compute-runner` record). The runner agent (`runner/agent.mjs`,
also Dockerised) then **long-polls** `POST /runners/channel/poll` with its own token,
executes jobs with the same primitives as local (streaming spawn + the content-addressed
data cache at `~/.thefactory-runner/cache`), streams log batches, and posts the RunSummary
back — the backend's `RunnerChannel` resolves the awaiting ComputeJobHandle so the engine
can't tell remote from local. Run records carry the target as `ranBy`. Jobs reference
projects by local path in v1 (same machine / shared mount); git repoRefs clone but assume a
self-bootstrapping checkout.

Phases 7–9 (BlackSwan migration, autopilot, notebooks) extend this skeleton — see
`implementation-plan.md`.
