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

- BlackSwan dip/regression line: the repo's currently-active research
  (`regression_predict` env + `model_regression_dip`, f1-style objective) is NOT covered by
  the trading-line manifest (one objective per project). Give it its own manifest/entrypoint
  when wanted — needs either a second manifest path convention or a sibling registration.
- BlackSwan seed plumbing depth: `trainer/run.py` seeds python/numpy/torch/SB3 globally,
  but most algo constructors in `model_factory.py` don't take a `seed=` arg — per-algo
  determinism is best-effort until seeds are forwarded into the constructors.

- Migrate `RecommendTools.buildProductCatalog`'s internal per-item loop onto
  `runActivityWorkItems` (the generic engine was built fresh; recommend still owns a private
  copy of the pattern). Behaviour-preserving; keep recommend tests green.
- Mobile parity: the mobile `ProjectEditorForm` lacks the "Has App surface" toggle and the new
  "App directory" field; mobile/desktop also need a native Compute Runners settings mirror
  (`useComputeRunners` is headless-shared already). Per the three-client rule.
- Runner channel WS upgrade: the long-poll protocol works and is verified; a held WS would cut
  log latency. Upgrade `/runners/channel` when it matters.
- Viewer "Run on": replace the free-text runner-id input with a proper dropdown once a bridge
  op (or record mirror) exposes the runner list to embedded apps.
- Remote git repoRefs: the agent clones but assumes a self-bootstrapping checkout; the engine
  still emits local paths only. Wire git refs + project bootstrap when a real second machine
  needs it (BlackSwan local path covers Phase 7).
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
