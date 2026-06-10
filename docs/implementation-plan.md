# thefactory-modeltrainer — implementation plan

A standalone repo that turns "train a model, run experiment campaigns, judge results,
propose the next batch" into a generic, Overseer-driven capability. It is **orthogonal** to
`thefactory-tools` and consumes it the same way `thefactory-knowledge` does (npm `file:` link,
narrow imports, own build).

**BlackSwanExperiments** (Python RL crypto-trading) is the **first consumer**: get it running
end-to-end with the least special-casing, then keep everything domain-oblivious so a second
model (any custom model for a particular problem) is *data + a thin CLI contract*, not new
engine code.

---

## Repo split (ownership is the whole point)

| Repo | Owns |
| --- | --- |
| **thefactory-modeltrainer** (this repo) | `ModelTrainerTools` facade (+ deps-injected `createModelTrainerTools` for the backend); the generic experiment-matrix planner; judge/propose orchestration; the embeddable **viewer** (dashboard SPA with charts); types/constants. Reads each model's `TrainerManifest`. |
| **thefactory-tools** | Generic infra only: `ComputeRunner` seam + `LocalComputeRunner`; `RemoteComputeRunner` client; `ContentAddressedDataCache`; the `RunnerCredentialEntry` credential kind + pairing crypto; the work-item **step engine** extracted from `RecommendTools.buildProductCatalog`; a `--gpus` field on `SandboxTools`. |
| **thefactory-backend** | The `train` Activity definition (mirrors `recommend-catalog`); composing `ModelTrainerTools` at boot + extending `ActivityRunContext`; PIN-pairing endpoints + runner-token auth; the authenticated connect-out **runner WS channel**; serving a package's viewer bundle as the App view. |
| **clients** (overseer-web/ui/native) | Global **Compute Runners** settings + pairing screen (native, cross-project). Everything else (activity badge/spinner, model chip, App-view iframe) is reused as-is. |
| **the runner agent** | A thin generic program packaged as a **Docker image**; dials the backend, runs jobs via `SandboxTools` primitives, owns the `ContentAddressedDataCache`. Home TBD (own `thefactory-runner` package vs in this repo). |
| **BlackSwanExperiments** (Python) | A `TrainerManifest` (`.factory/trainer.json`) + the thin CLI contract (`--config-json` / `--summary-out` / `--calibrate`). No Overseer code. |

### How this repo consumes thefactory-tools (mirror thefactory-knowledge)
- Standalone npm package: `"type":"module"`, `"main":"dist/index.js"`, `"types":"dist/index.d.ts"`,
  `"files":["dist"]`, single `"exports": { ".": "./dist/index.js" }`, own `tsc` build + vitest.
- `"thefactory-tools": "file:../thefactory-tools"` in `dependencies` (symlinked). Import only the
  narrow surface from `thefactory-tools` / `thefactory-tools/types` / `thefactory-tools/utils`.
  Never duplicate git/file/LLM helpers — add missing ones to thefactory-tools and import them.
- Placement rule: low-level/generic primitives belong in thefactory-tools; the whole training
  domain stays here.

---

## Genericity spine: the model-training standard

The engine never names BlackSwan. Every training project conforms to **the standard**
(`docs/model-training-standard.md`): a `TrainerManifest` (`.factory/trainer.json` — run/calibrate
command templates, lever schema, objective, data + resource declarations), a **CLI contract**
(`--config-json` / `--summary-out` / `--calibrate`), and a structured `RunSummary`.

`ModelTrainerTools` reads the manifest, plans the work-item matrix from an `ExperimentSpec`
(which levers to sweep), dispatches each item through a `ComputeRunner`, ingests each `RunSummary`
as a record. **Adding a model = ship a conformant manifest + entrypoint — no engine code.**

### Reference projects ARE the standard (de-risk before BlackSwan)
A schema drifts; a conformant project pins it. Two small projects under `examples/` are the
executable spec of the standard, the engine's CI fixtures, and the onboarding templates — and
they let the whole orchestrator be built at seconds-per-run instead of BlackSwan's hours-per-run:

- **`examples/cartpole` (RL, NO data)** — SB3 + gymnasium, CPU, trains in seconds. Mirrors
  BlackSwan's stack (env + training loop + eval + discrete actions) so learnings transfer almost
  directly, with zero dataset. Establishes config-as-data, seed/repro, `RunSummary`, MLflow
  logging, checkpoint-best, the `--calibrate` smoke gate.
- **`examples/tabular` (small data + DVC)** — a few-MB dataset. Establishes the **data path**
  (dataSpec → content-addressed cache → fetch-only-misses → materialize) cheaply — the dress
  rehearsal for BlackSwan's 7.6 GB and the remote runner.

**BlackSwanExperiments then migrates to the proven standard** — conformance, not invention on an
expensive workload. This is where "make BlackSwan better + easier to orchestrate" lands.

---

## ComputeRunner seam (local + remote) — lives in thefactory-tools

Internal interface (not a `*Tools` surface; like `InferenceExecutor`):

```ts
interface ComputeRunner {
  calibrate(probe: CalibrationProbe): Promise<CalibrationResult>   // { fps, secondsObserved }
  runJob(job: ComputeJob): ComputeJobHandle                        // { runId, onLog, done, abort }
}
```

`ComputeJob` is self-describing so the same shape works local and remote:
`{ repoRef: {gitUrl,commit} | {localPath}, command, args, env?, dataSpec, artifactGlobs[], abortSignal? }`.
`dataSpec` names the datasets needed (→ cache + proxy allowlist) with a `credentialRef`, never
inline secrets. `ComputeJobResult = { exitCode, status, summary?, artifacts: ArtifactRef[] }` —
`summary` is the captured `RunSummary` (opaque to the engine); heavy artifacts are referenced,
not shipped.

- **`LocalComputeRunner`** — runs on the host the backend runs on, via `SandboxTools` (Docker
  isolation, proxied egress) or direct spawn; reuses `src/process/runCommand` for short ops.
  This is the "estimate how long tests take on whatever machine the BE is deployed on" path:
  `calibrate()` → `fps` → campaign ETA = Σ items `(timesteps×episodes×iterations)/fps`, shown
  before launch.
- **`RemoteComputeRunner`** — the client half; drives the runner agent over the WS channel.

### The runner agent — trivial to set up (Docker + PIN)
- **Setup = one `docker run`.** The agent is a Docker image baked with `uid 10001`, the training
  stack, and a job shim. `docker run thefactory/runner --pair` prompts for a PIN.
- **2-way PIN pairing.** Overseer (Settings → Compute Runners) shows a PIN. The operator enters
  it on the runner; the runner calls the unauthed `POST /runners/pair {pin, name}`; the backend
  validates PIN+TTL, mints a **hashed** runner token (new `RunnerCredentialEntry`), returns it
  once. The "2-way" confirm = the Overseer shows the pending runner's self-reported name before
  the operator grants. Thereafter the runner dials the authenticated WS and holds it.
- **Connect-out** (runner dials backend): works behind NAT/firewall, no inbound ports on the
  runner — the only model compatible with "trivial setup." Reuses the existing `wsBroadcast` +
  `makeCliRunWsProgress` log-streaming pattern back to UI clients.
- **On a job:** clone `repoRef@commit`, materialize data via the cache (below), run training in a
  `SandboxTools` container (allowlisted egress to only the data host + Overseer), stream logs,
  report `RunSummary` + artifact refs. Checkpoints stay on the runner, referenced.

### Content-addressed data cache (no re-download on a config toggle)
`ContentAddressedDataCache` (plain class in thefactory-tools, clones the `DataStorageSourceCache`
read-through pattern; uses `hashKey`/`snapshotDirByHash`/`atomicWriteFile`):
- On-disk **object store** on the runner (a Docker volume, survives `--rm`):
  `objects/<sha256-of-bytes>` — each data file stored once; identical bytes dedupe.
- **Manifest** per dataset in `DataStorage` (`type='training-data-manifest'`): logical path →
  `{ contentHash, size, sourceUrl, fetchedAt }`. Bytes never go in DataStorage.
- Read-through: resolve each required file's content hash (DVC pointers already record it) →
  present ⇒ hit (zero download) → fetch **only misses** in one batched pass over the
  proxy-allowlisted egress → temp-then-rename so a crashed download never looks present.
- Materialize the run's `/data` by **hardlinking** objects into their logical paths, bind-mount
  read-only into the training container.
- **Config-toggle-only re-run ⇒ identical file set ⇒ 100% hits, zero download.** Content
  addressing *is* DVC's model, so the object store seeds straight from `dvc pull`.

---

## Backend integration (mirror the shipped `recommendTools` path)

- Compose `createModelTrainerTools({ completionTools, computeRunner, storage, db, logger })`
  (imported from this repo) once in `server.ts`; `fastify.decorate('modelTrainerTools', …)`.
- Extend `ActivityRunContext` with one field `modelTrainerTools` (next to `recommendTools`);
  thread `fastify.modelTrainerTools` through `launch()` in `routes/activities.ts`.
- Register a `train` `ActivityDefinition` in `activityDefinitions.ts`
  (`steps: [{key:'plan'},{key:'run'},{key:'judge'},{key:'propose'}]`); `run(ctx)` calls
  `ctx.modelTrainerTools.*`, streams via `ctx.updateStep`, writes records via
  `ctx.storage.upsertRecord` + `ctx.broadcast('data:updated', …)`. No new route — generic
  `activities/{run,list,abort,resume}` already serve it; resumable via `resumeToken`,
  model-selectable via the active agent LLM config.

### App view: serve this repo's viewer bundle
- The flag is done: `metadata.hasApp` is a working toggle in `ProjectEditorForm` /
  `ProjectManagerModal`, gated in `Sidebar`. BlackSwan = an existing repo registered as a project
  (`language: 'python'`) with `hasApp` on; heavy artifacts stay on disk.
- One generalization: the App-view route serves from the **checkout root** today (carfinder).
  Add: a project may declare its app source is a **package viewer bundle**
  (`appSource: { package: 'thefactory-modeltrainer', dir: 'dist/viewer' }`) → serve from there.
  This keeps the dashboard in this repo (one viewer, every training project) and closes the same
  gap thefactory-knowledge has. The viewer bundles its own charts; talks home via `bridge.js`
  (`OverseerBridge`) — `startActivity('train', params)`, polls records.

---

## Records (DataStorage, scope = projectId; `recordType` from the manifest)

- `activity-run` / activityId — the resumable campaign (steps, status, modelRef,
  computeTargetRef, ETA). Drives the nav badge. (Existing flat type.)
- `{recordType}` items — one `RunSummary` per (config × seed-aggregate): objective, metrics,
  health flags, hyperparameters, artifact ref, `ranBy` target, freshness `{status, specHash, completedAt}`.
- `{recordType}-progress` / latest — phase/counts + ETA.
- `{recordType}-hypothesis` items — the durable backlog (proposed lever changes, status
  pending|accepted|rejected, rationale, source human|llm). Nothing lost between sessions.
- `{recordType}-verdict` items — judge score + reasoning per run/batch, `judgedBy` model.
- `training-data-manifest` — per-dataset cache index (above).
- `RunnerCredentialEntry` — paired runners (in the encrypted credential store, token hashed).

---

## UI (reuse-heavy)

**Reused as-is:** activity plumbing (`useProjectActivities`, `activitiesStore`,
`dispatchActivitiesBridge` + `runActivity`/`listActivities`/`abort`/`resume`), nav badge/spinner
(`useBadgeCounts`, `NavRow`), the model chip (`ModelChipConnected`), the App-view iframe host
(`ProjectAppView`/`ProjectAppTab`/`useProjectAppView`) + bridge dispatchers, the `hasApp` toggle.

**Built in this repo (the viewer SPA):** runs list, run detail with live metric/equity curves
(bundles its own charting), launch form (lever inputs from the manifest + model chip + ETA),
hypothesis backlog. Embedded as the App view.

**Built native in the Overseer (cross-project, global):** a **Compute Runners** category in
`SettingsView` + a **PIN-pairing form** — closest template is `GitCredentialsForm` (it already
models a device-style pairing flow). Three-client parity: web first-class, desktop/mobile mirror.

---

## Staged build (standard-first; everything proven on cheap fixtures before BlackSwan)

### Phase 0 — The standard + Reference Project A + stand up this repo
Write `docs/model-training-standard.md` (done). Scaffold this package (mirror thefactory-knowledge:
`file:` link to thefactory-tools, own build/vitest, `src/index.ts`). Build `examples/cartpole`
(SB3 + gymnasium, no data, CPU) conforming to the standard — the executable spec + CI fixture.
Acceptance: `cartpole` runs `--config-json` / `--summary-out` / `--calibrate` and emits a valid
`RunSummary`; the standard doc is the contract.

### Phase 1 — `ModelTrainerTools` planner (local, in-process) against cartpole
`ModelTrainerTools` class + `modelTrainerTypes.ts` + `TrainerManifest` types + the experiment-matrix
planner. Acceptance: a small sweep over cartpole's levers produces one `RunSummary` record per item
+ seed-aggregation — all in-process, seconds per item.

### Phase 2 — ComputeRunner seam + LocalComputeRunner + calibration (thefactory-tools)
`src/computeRunner/` (`computeRunnerTypes.ts`, `LocalComputeRunner.ts`, helpers/utils/constants/
type-validations; `*.test.ts` per file, spawn mocked). `--gpus` on `SandboxTools` (unused by
cartpole, present for BlackSwan). Acceptance, validated on cartpole: `runJob` streams logs + returns
a `RunSummary`; `calibrate` returns `fps`; ETA math unit-tested.

### Phase 3 — Generic step engine + capability-injected ActivityRunContext (thefactory-tools + backend)
Extract the plan → skip-if-fresh → per-item-run → checkpoint → stream loop out of
`RecommendTools.buildProductCatalog` (keep recommend green); generalize `ActivityRunContext` to
inject a capability by `activityType`. Acceptance: recommend still passes; a trivial activity runs
N work-items via an injected executor.

### Phase 4 — `train` Activity + viewer (backend + this repo + the App-view generalization)
Register the `train` activity; compose `modelTrainerTools`; serve this repo's viewer bundle as the
App view; ship the viewer (runs/detail/launch/backlog). Acceptance: launching a cartpole campaign
from the viewer runs detached, streams progress + per-run metrics, survives restart — the entire
loop demonstrably working in seconds, no GPU, no babysitting.

### Phase 5 — LLM judging + proposing (this repo via InferenceExecutor) — problems #2 & #3
`judge`: rank template (`runInference` + `coerceRankedItems` + `blendScores`) blending the
deterministic objective with the LLM verdict; auto-reject degenerate (health-flagged) runs;
`{recordType}-verdict`. `propose`: research/inference over run history → `{recordType}-hypothesis`
backlog; human accept/edit/reject in the viewer → accepted become work-items. Validated on cartpole.

### Phase 6 — Reference Project B (data path) + RemoteComputeRunner + runner agent + PIN + cache
Build `examples/tabular` (small DVC-tracked dataset) — establishes + tests the data path cheaply.
thefactory-tools (`RemoteComputeRunner`, `ContentAddressedDataCache`, `RunnerCredentialEntry`) +
backend (`routes/runners.ts` PIN issue/pair, runner-token auth in `authPlugin`, authenticated runner
WS channel + registry + job dispatch) + the runner Docker image + the native Compute Runners
settings/pairing UI. **Decide the runner-repo home here** (see Open questions). Acceptance:
`docker run … --pair` + a PIN registers a runner; a `tabular` campaign targeting it clones
repo@commit, fetches only missing data (config-toggle re-run = zero download), runs, streams logs,
returns identical records to local.

### Phase 7 — Migrate BlackSwanExperiments to the standard (the payoff; via Stories/CLI-agents)
Now BlackSwan conforms to a fully-proven contract rather than inventing it. Conform the entrypoint
to `--config-json` / `--summary-out` / `--calibrate`; implement a single risk-adjusted objective
(the stubbed Sharpe/CAGR/max-drawdown), seed-aggregation, hodl-relative baseline, degenerate-policy
health flag; re-enable MLflow params+metrics; checkpoint-best + run→metrics traceability; write
`.factory/trainer.json` + a Dockerfile (declared GPU/resources); prune dead paths (dead env types,
commented MLflow, broken default config). Then run real campaigns — local first, then remote.
BlackSwan is now "the third conformant project," not a bespoke integration.

### Phase 8 (optional) — Autopilot + live handoff
Scheduled meta-activity (propose → run → judge → promote, human-approved); on a winner, tag the
checkpoint for the live trading handoff (`run_server_model.py`) with full run→metrics traceability.

### Phase 9 — Jupyter notebooks: run/view/edit `.ipynb` — **UNDERSCOPED**
Standard ML-workflow companion: view, edit, and execute `.ipynb` notebooks belonging to a training
project from the Overseer (exploration, result analysis, ad-hoc plots against run records).
No spec yet — deliberately parked until the core loop ships. To scope later: rendering (read-only
render vs full editor), execution model (kernel on the backend host? in the sandbox/runner
container? remote runner?), how notebooks read campaign records/artifacts, and security
(notebook code execution = arbitrary code; likely the sandbox profile). Do not build preemptively.

> Note: BlackSwan's instrumentation (Phase 7) is independent enough to pilot earlier in parallel
> if early signal is wanted — but doing it as conformance to the settled standard avoids rework.

---

## Standards (this repo + thefactory-tools)

- Public types in `*Types.ts` (canonical `modelTrainerTypes.ts`); TSDoc on exported members.
  `ComputeRunner` impls, `ContentAddressedDataCache`, the runner agent are **internal classes**,
  not `*Tools`. `createModelTrainerTools` is the deps-injected agent-facing variant (still `*Tools`,
  like `createRecommendTools`).
- `*Utils` pure / `*Helpers` impure; constants in `*Constants.ts`; extend `src/file/fileHelpers.ts`
  (in thefactory-tools) for FS primitives; never duplicate git/file/LLM helpers — add to
  thefactory-tools and import.
- `sendCompletion`/`runInference` for raw JSON must pass `structuredOutput: false`.
- TDD, near-100% coverage (mock spawn/network/dvc/docker). No inline convention comments; no
  back-compat shims. Update thefactory-tools `CODE_STANDARD.md` / `FILE_ORGANISATION.md` when its
  modules land; keep this repo's `docs/architecture.md` current as work lands.

---

## Open questions (resolve before the dependent phase)

- **Runner agent home + image name** — own `thefactory-runner` package vs inside this repo;
  published image name. **Deliberately deferred to Phase 6** — decide once the LOCAL runner path
  (Phases 2–5) is working and the runner's real surface is concrete; do not pre-commit.
- **Judge/proposer model transport** — `ModelSelection` is API-only today (CLI is the unbuilt
  activity-engine Stage 5). API now; CLI when it lands. (Phase 5.)
- **Remote artifact/checkpoint storage** — keep-on-runner + reference vs upload; and how a winning
  remote checkpoint reaches the live server. (Phase 6 / 7.)
- **GPU + the aggressive sandbox profile** — training images write to caches; confirm the
  `--read-only` rootfs + writable `/workspace`/cache-volume combo, and `--gpus` wiring. (Phase 6.)
- **Connect channel** — authenticated WS connect-out (live logs) vs REST long-poll fallback
  (simpler, higher latency). Recommend WS; ship poll as fallback. (Phase 6.)
- **App-view-from-package generalization shape** — confirm `appSource` declaration vs a per-project
  committed bundle; this also unblocks thefactory-knowledge's viewer. (Phase 4.)
- **Add-existing-local-repo-as-project** — confirm it's first-class or add it (studied path was
  template-fork only). (Phase 0 / 4.)
