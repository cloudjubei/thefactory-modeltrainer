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
  per-action value over time, confidence, input attribution). The engine never computes it — each
  project emits its own (BlackSwan replays its deterministic test once more to capture per-step
  confidence/Q-values + saliency); the engine only soft-validates it (`validateDecisionTrace`,
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
- **LLM only synthesises the xAI, never computes it**: a thin layer sits on top of the deterministic engine.
  `xaiNarrate` (tool) runs the SAME deterministic analysis server-side, then makes ONE inference to write a
  short campaign **narrative** (`{recordType}-xai-narrative` 'latest' record, via the `xai-narrate`
  activity) the viewer shows at the top of the tab with an "N new runs since" refresh. A **"Discuss xAI"**
  button seeds a chat with one run's FULL analysis (attribution + its Adebayo sanity check, latent probe,
  the decision diff vs its nearest sibling). A **"Propose with AI"** button feeds the lever-importance +
  gap signal as instructions into the existing `proposeTrainingHypotheses`. The deterministic records stay
  the source of truth; the model interprets them and is told to hedge on the confounded/surrogate signals.
- **Judging blends, never replaces, the objective**: `judgeTrainingRuns` min–max-normalises
  the objective (direction-aware) and blends it 50/50 with the LLM's 0–100 verdict
  (`{recordType}-verdict` records, key = run key) — a money-losing run can't be ranked best
  by prose. Health-flagged runs are auto-rejected without spending an LLM call. Runs the LLM
  skips keep an objective-only verdict.
- **Proposals are validated + deduped data**: `proposeTrainingHypotheses` coerces the LLM's
  ideas against the manifest's levers (unknown lever ⇒ dropped), keys each
  `{recordType}-hypothesis` record by the spec's hash (identical proposals dedupe, existing
  statuses survive re-proposing), and the viewer's accept → "Run campaign" turns a hypothesis
  into a `train` activity with that spec. Backlog statuses are plain record edits.
- **Datasets/environments are user-managed bundles with a settable default**: levers tagged
  `scope: "dataset"`/`"environment"` aren't model knobs — they're managed in their own tabs as
  named `{recordType}-dataset`/`-environment` records (`{id, name, settings, default}`). One record
  holds the `default` flag (the launch picker pre-checks it; removing it promotes the next record);
  the first one created becomes the default, and a save is refused if a record with the same name or
  settings already exists. The manifest-defaults card stays only as a read-only clone-to-start seed.
  A project that declares such levers but has none defined cannot launch (a valid just-started state).
- **Run dataset identity is normalized on open, not at the call site**: runs group by the VALUE
  signature of their dataset/env levers, so the `fidelity_set: "auto"` synonym (and pre-hub ledger
  imports carrying no fidelity_set) would fragment a run away from its explicit-dataset siblings.
  `viewer/migrate.js` (a pure, parity-mirror-style module, unit-tested via `src/migrateViewer.test.ts`)
  derives each run's CONCRETE identity from data it already carries — its own `dataset.layers`, or the
  `historical_data` tag for imports (sub-hourly/retired stacks get a truthful `legacy:` label, never
  force-merged into a runnable set). The viewer runs it idempotently on project open (the whole backlog)
  and on every runs refresh (in-flight `auto` runs), recomputing `setupKey` so by-setup regroups too.

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
