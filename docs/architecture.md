# Architecture

How a training campaign flows through the system, and which repo owns which piece.

```
┌─ Overseer client (web/desktop/mobile) ──────────────────────────────────────┐
│  App tab → sandboxed iframe → viewer/ (this repo, static, no-build)         │
│  window.OverseerBridge: queryData / startActivity / listActivities / abort  │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │ postMessage → host → REST
┌─ thefactory-backend ─────────────▼───────────────────────────────────────────┐
│  POST /projects/:id/activities/run { activityType:'train', params }          │
│  activityDefinitions.ts: trainModelActivity                                  │
│    run(ctx) → ctx.modelTrainerTools.runTrainingCampaign(...)                 │
│    progress → {recordType}-progress record + updateStep → activity:updated   │
│  server.ts composes createModelTrainerTools({ computeRunner, storage }) once │
│  GET /projects/:id/view/* serves viewer/ via metadata.appSource              │
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
| Viewer dashboard                                               | this repo `viewer/`                           | one viewer for every training project   |
| ComputeRunner seam + LocalComputeRunner + work-item engine     | thefactory-tools                              | generic infra, reusable beyond training |
| `train` activity registration + composition + app-view serving | thefactory-backend                            | host wiring (mirrors `recommendTools`)  |
| Model specifics (levers, objective, training code)             | each conformant project                       | the engine stays domain-oblivious       |

## Key decisions

- **Records over files**: every result is a DataStorage record (`scope = projectId`,
  type from `manifest.recordType`), so the viewer, badge, and resume all work from the same
  substrate and survive restarts. Run identity = 12-hex hash of the resolved config —
  skip-if-fresh and re-launch idempotency fall out of that.
- **The engine never reads model code**: it knows `.factory/trainer.json`, two command
  templates, and the RunSummary shape. Conformance, not integration.
- **Calibrate-first ETA**: campaigns optionally start with the manifest's tiny calibrate run;
  `unitsPerSecond` × the plan's total units gives the upfront ETA streamed into progress.
- **Abort/resume**: abort flows an AbortSignal from the activity into the spawned process;
  resume re-launches the same campaign and skip-if-fresh makes completed items free.
- **Failures are per-item**: a failed run records a `failed` count and the campaign continues;
  a completed-but-malformed summary is a failure (never silently ingested).

Phases 5–9 (judge/propose, remote runner + PIN pairing + data cache, BlackSwan migration,
autopilot, notebooks) extend this skeleton — see `implementation-plan.md`.
