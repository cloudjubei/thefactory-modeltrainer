# thefactory-modeltrainer — implementation plan

Remaining work only. What's built and how it fits: `docs/architecture.md`. The contract:
`docs/model-training-standard.md`. The core loop (Phases 0–7) is **built + verified**: the
engine, backend activities, viewer, remote runner, and three conformant consumers —
`examples/cartpole`, `examples/tabular`, and **BlackSwan** (`/Users/cloud/Documents/Work/BlackSwan`,
the trading line, Phase 7 done). The engine stays domain-oblivious so any further model is
_data + the thin CLI contract_, not engine code. What's below is optional phases, small
cleanups, and deferred new work.

## Repo split (governs all phases)

| Repo                                    | Owns                                                                                                                                                           |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **thefactory-modeltrainer** (this repo) | `ModelTrainerTools`; matrix planner; campaign loop; judge/propose orchestration; the viewer; the standard + `examples/`.                                       |
| **thefactory-tools**                    | Generic infra only: `ComputeRunner` seam (+ future `RemoteComputeRunner`, `ContentAddressedDataCache`, `RunnerCredentialEntry` pairing); the work-item engine. |
| **thefactory-backend**                  | Activity registration + composition; app-view serving; future PIN-pairing endpoints + runner WS channel.                                                       |
| **clients**                             | Future Compute Runners settings/pairing screen (native, cross-project); everything else reused.                                                                |
| **the runner agent**                    | Future Docker-packaged connect-out program. Home decided in Phase 6.                                                                                           |
| **BlackSwan** (the trading repo)        | Its `TrainerManifest` + additive `trainer/` CLI conformance (Phase 7, done). No Overseer code.                                                                 |

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

- Migrate `RecommendTools.buildProductCatalog`'s internal per-item loop onto
  `runActivityWorkItems` (the generic engine was built fresh; recommend still owns a private
  copy of the pattern). Behaviour-preserving; keep recommend tests green.
- Mobile parity: the mobile `ProjectEditorForm` lacks the "Has App surface" toggle and the new
  "App directory" field; mobile/desktop also need a native Compute Runners settings mirror
  (`useComputeRunners` is headless-shared already). Per the three-client rule.
- Runner channel WS upgrade: the long-poll protocol works and is verified; a held WS would cut
  log latency. Upgrade `/runners/channel` when it matters.
- Viewer "Run on": replace the free-text runner-id input with a proper dropdown — now
  **unblocked** (the `runners.list` bridge op exists; the app can populate a select).
- Remote git repoRefs: the agent clones but assumes a self-bootstrapping checkout; the engine
  still emits local paths only. Wire git refs + project bootstrap when a real second machine
  needs it (BlackSwan local path covers Phase 7).
- Viewer: re-attach to a live judge/propose activity after a page reload (today only `train`
  re-attaches; a reload mid-judge just shows results on the next refresh).
- App-nav unseen badge: the Overseer App tab already spins on a live activity, but an
  unseen-results badge while idle needs app→host plumbing (the embedded app reports its unseen
  count). The in-app unseen badges cover the immediate need.

---

## BlackSwan improvements

Make the first consumer genuinely better; each one also improves ModelTrainer by extension.
Pick these up after the core plan is finished.

- **Dip/regression line as its own project.** The repo's currently-active research
  (`regression_predict` env + `model_regression_dip`, an f1-style objective, different
  `get_run_state` shapes) is NOT in the trading-line manifest — one objective per project.
  Give it its own manifest/entrypoint (a second conformant project pointing at the same repo,
  or a manifest-selectable line) so dip/regression runs are judged on f1, not Sharpe.
- **Seed plumbing depth.** `trainer/run.py` seeds python/numpy/torch/SB3 globally, but most
  algo constructors in `model_factory.py` don't take a `seed=` arg — per-algo determinism is
  best-effort until seeds are forwarded into the constructors. Wire `seed` through so a
  campaign's seed sweep is genuinely reproducible.
- **Run health & continuation (resume mid-run).** A run that fails partway (crash, OOM,
  interrupted) is wholly lost today — skip-if-fresh only knows completed vs not. Add a notion
  of run health + checkpoint-aware continuation: persist partial state (the training is already
  checkpointed per episode), detect a half-finished run, and resume from its last checkpoint
  rather than restarting. Spans the engine (a `resumable`/`partial` item state) + the trainer
  CLI (`--resume-from <checkpoint>`, which BlackSwan's `checkpoint_to_load` already supports).
- **Parallel runs (concurrency setting in Activity).** Today a campaign runs items
  sequentially. Add an Activity-tab setting for max concurrent runs (default 1), and run that
  many work-items at once — with a guard that they don't conflict (each run already uses a
  unique checkpoint id + its own temp config/summary; the constraint is host CPU/GPU/RAM, so
  the cap is the safety valve). Spans the work-item engine (`runActivityWorkItems` gains a
  concurrency arg) + the campaign params + the Activity UI control.
- **Lighter default first-run.** The tuned default (`reppo-custom`, `[8192,512]`) is heavy on
  CPU — a great "best known" but a slow first experience. Consider a manifest hint or a
  recommended "quick start" sweep so a new user's first campaign returns fast.

## The data mine — a shared dataset project for every model trainer

A standalone project (its own repo, e.g. `thefactory-datamine`) that is the **source of truth
for training data** across all ModelTrainer consumers: gather raw data (exchange klines for
BlackSwan, datasets for the tabular/code lines, …), clean + validate + normalise it, and
publish it in the best shape the models can consume — versioned and reproducible. Each trainer
project then declares which prepared dataset(s) it needs (the manifest's `data[]` already names
them) and the data mine + the content-addressed cache deliver them; no trainer fetches or
cleans raw data itself. This is where the user's separate dataset repo for BlackSwan folds in
(the current klines are likely stale). Deferred until the core loop + BlackSwan improvements
are in — but it's the natural home for "prepare the data correctly so the models have it in the
best form," and it makes the remote-runner data path (Phase 6 cache + `data[]` + `credentialRef`)
pull from one curated origin.

## Code-change risk model — the second workspace ML tool (deferred)

A trainer-conformant project that scores an agent's diff/PR by how likely it is to introduce a
bug (later reverted/fixed, or CI-failing) — a calibrated risk signal the platform uses to gate
review effort, trigger an expert-panel/verifier pass, and flag risky agent output before merge.
A genuinely different third consumer shape (calibrated binary classification, AUC/precision
objective) that hardens the generic engine, and it makes the agents measurably safer.

**Deferred — gathering the labelled data is the hard part, so the FIRST step is research, not
code:**

1. **Research (do this first).** Survey existing public datasets before mining our own — this is
   likely language-specific (academic just-in-time defect-prediction corpora skew Java; JS/TS is
   thinner). Look at e.g. ApacheJIT, Defectors, JIT-Defect4J, ManySStuBs4J, CVEfixes, Big-Vul,
   Devign, CodeXGLUE defect-detection. Decide: bootstrap the model on an existing dataset
   (faster, transferable) vs mine our own from the workspace git histories (in-domain, but needs
   labeling). Use the deep-research harness; output a short cited report + a go/no-go on a
   workspace-mined dataset.
2. **Data (via the data mine).** If mining our own: an SZZ-style labeling pass over the
   `thefactory-*` git histories — commits later reverted or bug-fixed = positive; features from
   codeIntel (churn, files/complexity touched, test coverage of touched code, diff size). This is
   the data mine's first real job. Versioned, reproducible `(features, label)` records.
3. **Train.** A `risk-classifier` trainer-conformant project (its own `.factory/trainer.json`,
   sklearn or torch, objective = AUC or precision-at-k, health flags for class collapse), the
   `data[]` naming the prepared dataset. Register in the hub exactly like BlackSwan.
4. **Consume.** Wire the score into the review / expert-panel / verifier path (a risk badge on a
   diff; gate the expensive panel on high risk).

Depends on the data mine; pick up after the BlackSwan improvements.

## Open questions

- **Remote artifact/checkpoint storage** — keep-on-runner + reference vs upload; how a winning
  remote checkpoint reaches the live trading server. (Phase 6/8.)
- **GPU + sandbox profile for training images** — `--read-only` rootfs vs ML caches; `--gpus`
  is wired in SandboxTools but unexercised. (Phase 6/7.)
- **Judge/proposer model transport** — `ModelSelection` is API-only until the activity engine's
  CLI stage lands; start with API. (Phase 5.)
