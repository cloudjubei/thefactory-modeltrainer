# thefactory-modeltrainer — implementation plan

Remaining work only. What's built and how it fits: `docs/architecture.md`. The contract:
`docs/model-training-standard.md`. **BlackSwanExperiments** (Python RL crypto-trading) is the
first consumer — migrated in Phase 7 below; the engine stays domain-oblivious so any further
model is _data + the thin CLI contract_, not engine code.

## Repo split (governs all phases)

| Repo                                    | Owns                                                                                                                                                           |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **thefactory-modeltrainer** (this repo) | `ModelTrainerTools`; matrix planner; campaign loop; judge/propose orchestration; the viewer; the standard + `examples/`.                                       |
| **thefactory-tools**                    | Generic infra only: `ComputeRunner` seam (+ future `RemoteComputeRunner`, `ContentAddressedDataCache`, `RunnerCredentialEntry` pairing); the work-item engine. |
| **thefactory-backend**                  | Activity registration + composition; app-view serving; future PIN-pairing endpoints + runner WS channel.                                                       |
| **clients**                             | Future Compute Runners settings/pairing screen (native, cross-project); everything else reused.                                                                |
| **the runner agent**                    | Future Docker-packaged connect-out program. Home decided in Phase 6.                                                                                           |
| **BlackSwanExperiments**                | Its `TrainerManifest` + CLI conformance (Phase 7). No Overseer code.                                                                                           |

---

## Phase 6 — Remote compute: runner agent + PIN pairing + data cache

- `examples/tabular` — small DVC-tracked dataset; establishes + tests the data path cheaply
  before BlackSwan's 7.6 GB.
- thefactory-tools: `RemoteComputeRunner` (client half over the runner WS channel);
  `ContentAddressedDataCache` (clone the `DataStorageSourceCache` read-through pattern:
  sha256 object store on a docker volume, manifest in DataStorage, hardlink materialise,
  fetch-only-misses); `RunnerCredentialEntry` credential kind (token stored hashed).
- thefactory-backend: `routes/runners.ts` (PIN issue — Overseer-authed; `pair` — unauthed,
  validates PIN+TTL, mints the runner token once); runner-token acceptance in `authPlugin`;
  authenticated runner WS channel + registry + job dispatch (reuse the
  `wsBroadcast`/`makeCliRunWsProgress` log-streaming pattern back to UI clients).
- Runner agent: Docker image (`uid 10001`, training stack, job shim), `docker run … --pair`
  prompts for the PIN, then dials out and holds the WS. Jobs are self-describing
  (repo@commit + data manifest + `credentialRef`); checkpoints stay on the runner, referenced.
- Clients: Compute Runners settings category + PIN-pairing form (template:
  `GitCredentialsForm` device flow). Three-client parity.
- **Decide here:** runner agent home (own `thefactory-runner` package vs this repo) + image name.

## Phase 7 — Migrate BlackSwanExperiments to the standard (the payoff)

Via Overseer Stories / CLI-agents against the Python repo:

- Conform the entrypoint: `--config-json` / `--summary-out` / `--calibrate`; write
  `.factory/trainer.json` (+ a Dockerfile declaring GPU/resources).
- Make the science judgeable: implement the stubbed risk-adjusted objective
  (Sharpe/CAGR/max-drawdown), seed-aggregate the `iterations_to_pick_best` runs,
  hodl-relative baseline, degenerate-policy health flag.
- Re-enable MLflow params+metrics logging; checkpoint-best with run→metrics traceability.
- Prune dead paths (dead env types, commented MLflow, broken `config_simple` default).
- Register BlackSwan in the Model Trainer hub by its absolute directory (it stays a plain
  repo on disk — no Overseer project needed for running; Stories/CLI-agent editing of the
  Python code is a separate concern). Then run real campaigns — local first, then remote.

## Phase 8 (optional) — Autopilot + live handoff

Scheduled meta-activity (propose → run → judge → promote, human-approved); on a winner, tag
the checkpoint for the live trading handoff (`run_server_model.py`).

## Phase 9 — Jupyter notebooks: run/view/edit `.ipynb` — **UNDERSCOPED**

View/edit/execute notebooks belonging to a training project from the Overseer. No spec yet —
parked until the core loop ships. To scope: render-vs-edit depth, where the kernel executes
(backend host / sandbox / remote runner), how notebooks read campaign records/artifacts,
security (notebook execution = arbitrary code; likely the sandbox profile). Do not build
preemptively.

---

## Carried-over cleanups (small, no phase dependency)

- Migrate `RecommendTools.buildProductCatalog`'s internal per-item loop onto
  `runActivityWorkItems` (the generic engine was built fresh; recommend still owns a private
  copy of the pattern). Behaviour-preserving; keep recommend tests green.
- Mobile parity: the mobile `ProjectEditorForm` lacks the "Has App surface" toggle and the new
  "App directory" field (web/desktop have both); mirror them per the three-client rule.
- Viewer: surface `failures[]` from the `{recordType}-campaign` record (data already flows).
- Viewer: re-attach to a live judge/propose activity after a page reload (today only `train`
  re-attaches; a reload mid-judge just shows results on the next refresh).

## Open questions

- **Remote artifact/checkpoint storage** — keep-on-runner + reference vs upload; how a winning
  remote checkpoint reaches the live trading server. (Phase 6/8.)
- **GPU + sandbox profile for training images** — `--read-only` rootfs vs ML caches; `--gpus`
  is wired in SandboxTools but unexercised. (Phase 6/7.)
- **Judge/proposer model transport** — `ModelSelection` is API-only until the activity engine's
  CLI stage lands; start with API. (Phase 5.)
