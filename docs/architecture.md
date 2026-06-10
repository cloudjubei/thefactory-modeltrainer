# Architecture

How a training campaign flows through the system, and which repo owns which piece.

**The hub model:** ONE Overseer project (the thefactory-modeltrainer checkout itself —
the hub app lives at the repo root like the template apps, so the App tab needs only the
`hasApp` toggle) hosts the Model Trainer app. Training projects
(cartpole, BlackSwan, …) are **not Overseer projects** — they are directories registered
*inside* the app (`trainer-project` records; relative to the host checkout or absolute).
The `inspect-trainer` activity reads each one's manifest server-side into a
`trainer-project-manifest` record (so the app can render a launch form for any directory),
and every `train`/`judge`/`propose` activity carries the target's `dir`, resolved against
the host checkout. All records live in the host project's scope, namespaced per training
project by its manifest's `recordType`.

```
┌─ Overseer client (web/desktop/mobile) ──────────────────────────────────────┐
│  App tab (host project) → sandboxed iframe → the hub app (repo root)        │
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
│  src/computeRunner: ComputeJob/Handle/Result types, LocalComputeRunner       │
│  (streaming spawn, temp config/summary files), calibrate → fps → ETA         │
│  src/activity: runActivityWorkItems (plan→fresh→run→progress loop)           │
│  RemoteComputeRunner + runner agent + data cache: Phase 6                    │
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
| The hub app (home + per-project dashboards)                    | this repo's root app files                    | one app manages every training project  |
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

Phases 6–9 (remote runner + PIN pairing + data cache, BlackSwan migration, autopilot,
notebooks) extend this skeleton — see `implementation-plan.md`.
