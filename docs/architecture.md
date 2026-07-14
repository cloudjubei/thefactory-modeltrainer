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
  never a locked or axis lever (a nuisance param pooled over, matching `setupKeyOfRun`). When the axis is a
  CONTROL lever (one that gates others via `appliesWhen`, e.g. `model_name`), `lockedLeverKeys` drops the
  levers it gates (`axisGatedLevers`) from the lock AND the sweep pins — each model brings its own
  `prob_threshold`/`momentum_lookback`, which the store normalises to `n/a` where they don't apply, so locking
  them would hide every sibling that isn't the focus's own model. Regime slice toggles
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
- **Per-model lever relevance + run reliability** guard against wasteful runs and lucky edges. The By value
  lever picker screens each lever's empirical effect for the focus RUN's model (`xaiModelLeverEffects` =
  `leverImportances` over that model's runs) and marks the near-zero ones "· no effect" (a sweep there won't
  move the score). Every run carries a RELIABILITY verdict (`assessRunReliability`, pure/tested): when a
  model's whole score spread concentrates (≥½) in a manifest `probabilistic: true` lever (a decision cutoff
  like `prob_threshold`) and the config is weak/mixed across datasets, the run is **dubious** (threshold-tuned
  luck, not a learned edge); unverified-across-datasets softens it to **threshold-driven**; else **ok**.
- **The verdict is a persisted, overturnable, filterable field** (only for projects with a `probabilistic`
  lever). Layering (`resolveReliability`): a persisted **user override** wins over a persisted **LLM** verdict
  wins over the **heuristic** baseline. Only the authoritative verdicts are stored (`<recordType>-reliability`
  overlay record per run, `source: 'user'|'llm'`, like eval verdicts); the heuristic is recomputed in memory
  into `reliabilityHeuristicCache` on each global Refresh (only the flagged runs kept, and the probabilistic-
  edge screen skips non-candidate models so it stays O(pool)). Surfaced as a headline badge + a **Reliability**
  section (Mark dubious / Mark reliable / Reset to auto) in run detail, a **Reliability** column, and a Runs
  filter (`runsReliabilityFilter`: any / flagged / dubious / threshold-driven / ok). The filter lives in an
  overlay record, not a run-record field, so it can't push into the server `where` — it forces the unpaged
  client-filter path (like text search) and scores the full set in `refreshRuns`. The **LLM** verdict is
  produced by the existing xAI narrate pass (`xaiNarrate` runs a second structured call — `buildReliability
  SystemPrompt` + `coerceReliabilityVerdict` over the same run digest — and persists a `source: 'llm'` overlay),
  so "Narrate" refines the verdict with the run's full context; it never fails the narrative if the verdict
  call errors.
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
    — advertised on the project chat through the backend's `extraToolSchemas` seam (`trainerTools.ts`,
    the knowledge-read-tools precedent; no ToolSchemas regen). They resolve a run id → its training project by
    searching the host's `trainer-project-manifest` records, and the `getRunXAI`/narrative facts share ONE
    `buildRunXaiDigest`.
- **The conversational hub — discuss ANYTHING, act from chat.** One viewer seam, `discussBundle({title,
  seed, intro, bundle})`, JSON-serialises exactly what the user is seeing into the topic chat's persistent
  system prompt (capped at 60k chars) — every surface carries a 💬 Discuss built on it: hypothesis cards
  (record + hygiene diagnosis + judging rule), the hypothesis registry-health banner (the full census +
  every blocked spec), paper cards (record, verdict roll-up, coverage gaps — also per-gap —, proposed
  improvements, linked hypotheses + verdicts), version cards (changelog entry, per-version run stats,
  matching invalidations/migrations), and the Datasets/Environments capability sections (levers + usage +
  presets). The ACT half: `recommendTrainingExperiments` — the trainer chat's one WRITE tool (schema +
  dispatch in the backend's `trainerTools.ts`; handler in `ModelTrainerTools`) — validates the chat LLM's
  suggested specs against the manifest (migrated first; unknown levers/empty sweeps/over-cap matrices are
  rejected with the reason) and persists deduped `-xai-suggestion` records the xAI Suggested view runs as
  one-click ✦ AI batches; it never launches compute. Papers additionally offer per-coverage-gap and
  per-improvement **→ story** buttons (`OverseerBridge.createStory`, host `story.create` handler) so "work
  on this" lands in the project's Stories for agents to pick up.
- **Hypothesis hygiene — why an undecided hypothesis is undecided.** Pure `hypothesisHygiene` /
  `hypothesisHygieneCensus` (`viewer/hypothesis.js`, tested): a per-fixed-pin census against ALL runs with
  the dead-pin CAUSE — structural (`migrated` = a manifest migration rewrites the pin so stored configs
  never match; `na-pinned` = the pin violates its own `appliesWhen` so stored configs hold `'n/a'`;
  `off-manifest` = outside choices/range) vs launchable (`missing-key`, `never-run`) — plus per-sweep-option
  and per-compare-cell run counts, planned-item math (an EXPLICIT seed list smaller than minRuns can never
  be judged → `underplanned`), `no-metric` (the family never reports `return_vs_hold_pct`), `single-cell`,
  and `baseline-out-of-range`. Status = `judged` | `blocked` (structural — fix the spec) | `starved` (just
  needs runs). Surfaced as a census banner + per-card diagnosis line, computed in one cached sweep per
  all-runs snapshot. `compareContexts` + `coerceComparison` now bound `baselineIndex` (an out-of-range
  index stays untested / clamps to 0 instead of silently mis-judging).
- **The single-context judging rule is manifest-declared** — `hypothesisBenchmark: {metric, threshold?,
  direction?}` (CartPole: `eval_return_mean > 475`; omitted ⇒ the trading default `return_vs_hold_pct >
  0`). `resolveBenchmark`/`measuredFromRuns` in `hypothesis.js` thread it through every verdict path,
  the hygiene `no-metric` diagnosis names the ACTUAL benchmark metric (and says to declare one when
  missing), and all evidence/criteria copy renders the project's own rule instead of "beats
  buy-and-hold". The stored `measured.beatsHold` field name is historical — it means "meets the
  benchmark". Cross-context (compare) hypotheses judge by the objective and ignore this.
- **Chat write tools are approval-gated.** Beyond the auto-callable reads + `recommendTraining
  Experiments`, the chat has `updateTrainingHypothesis` / `updateTrainingPaper` — allow-listed in-place
  record updates (a chat `verdict` becomes a MANUAL override; a replacement spec is migrated + validated
  like a launch). They are advertised but NEVER auto-called (`TRAINER_AUTO_TOOL_NAMES` excludes them),
  so the chat surfaces every mutation for the user to approve; code/feature work is steered to
  `addStory`/`addFeature` instead.
- **One standard "work on this" affordance** — `iconStorySvg` + an always-on callout — everywhere a
  view creates a story/feature: paper coverage gaps + proposed improvements (stories via
  `OverseerBridge.createStory`) and proposed models / components (a FEATURE under the shared
  "Implement missing model components" story via `OverseerBridge.createStoryFeature` → the host's
  find-or-create `story.feature.create` handler, all three clients).
- **Datasets/Environments are capability sections + named presets.** Each dataset/environment lever renders
  as a FEATURE card (description, choices/range with the ★ default, per-value RUN USAGE from the loaded
  pool, `active:false` = "not wired" badge, `dependsOn` dependency line), with the named records below as
  pure PRESETS (the same bundle table + CRUD). The Datasets section carries the comparability warning
  (metrics compare apples-to-apples only WITHIN one dataset). The environment form greys a dependent lever
  while its control's current value makes it inert (`Comparison.dependencyMet`) and badges unwired levers;
  values and signatures are unchanged (identity still spans every env lever).
- **The Models catalog requires a `model_name` choice lever** — discovery, flavor binding, and run roll-up
  all key on it. A manifest without one gets an explicit hint (Models empty state + a guarded Scan button +
  `noIdentityLever` on the scan result) instead of a silent zero; `examples/cartpole` renames its `algo`
  lever accordingly, with manifest `migrations` rolling old run configs forward.
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
