# thefactory-modeltrainer — implementation plan

Remaining work only (shipped history lives in git + the session memory). What's built and how it
fits: `docs/architecture.md`. The contract: `docs/model-training-standard.md`. The core loop
(Phases 0–7) is **built**: engine, backend activities, viewer, remote runner, and three conformant
consumers — `examples/cartpole`, `examples/tabular`, and **BlackSwan** (the trading line). The
engine stays domain-oblivious: any further model is _data + the thin CLI contract_, not engine code.

## North star — two co-equal outcomes

1. **Best generic pipeline/app for creating ANY model**, end to end (propose → run → judge → explore),
   with self-explanatory results/comparison UI and a data layer that stores the minimum and derives
   the rest at runtime — and that **guides a user from "here's my problem" to "here's what data to mine
   and how"** (see "the data mine").
2. **Use it to make BlackSwan the best trading model**, in STRICT ORDER: **(A) correctness** →
   **(B) find ONE setup that trades well** → **(C) huge space exploration**. Measurable progress is
   the point. BlackSwan is the forcing function that hardens the generic pipeline — don't trade one
   outcome for the other.

## Repo split (governs all phases)

| Repo                                    | Owns                                                                                                                            |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **thefactory-modeltrainer** (this repo) | `ModelTrainerTools`; matrix planner; campaign loop; judge/propose orchestration; the viewer; the standard + `examples/`.       |
| **thefactory-tools**                    | Generic infra only: `ComputeRunner` seam (+ future `RemoteComputeRunner`, `ContentAddressedDataCache`, pairing); work-item engine. |
| **thefactory-backend**                  | Activity registration + composition; app-view serving; future PIN-pairing endpoints + runner WS channel.                       |
| **clients**                             | Future Compute Runners settings/pairing screen (native, cross-project).                                                        |
| **the runner agent**                    | Future Docker-packaged connect-out program (home decided in Phase 6).                                                          |
| **BlackSwan** (the trading repo)        | Its `TrainerManifest` + additive `trainer/` CLI conformance. No Overseer code.                                                 |

---

## NEXT — the active work

### 1. Live-verify the recent fixes (one reload + one multi-run campaign)

The parallelism/progress/self-stop fixes + the UI batch were built without a live render. After a
`dev:force` backend restart + viewer reload, confirm on a real campaign (concurrency > 1):

- N in-flight rows show (not 1); the progress bar is determinate and the ETA shows "~Xm (est.)".
- A brief backend blip or backgrounding the app no longer self-pauses a live campaign; queued
  follow-ups resume on return.
- Bulk delete works (two-click arm); the Runs master-detail scrolls independently (controls + table
  header pinned, only rows scroll); the compare pane (combined config-diff+metrics, colour-coded ids
  + curves, h-scroll); hover callouts stay in view; degenerate runs hide Verdict; Eval is gone for
  BlackSwan.

### 2. BlackSwan Phase B — find ONE setup that trades well

Phase-A correctness gate is GREEN. **Success = a config that trades OFTEN and PROFITABLY, stable
across seeds, beating buy-and-hold (as trade-gated `traded_return`).** Run it:

- Start with `use_indicators` on/off on 1h, multi-seed (the 1d prelim was seed-dependent).
- Read By-experiment + By-setup (median/IQR/stability); lock in the first setup that clears the bar.

Ground rules: profit is the objective (hold is a display control, never the target); trade often
**and** well (a 1-trade run ≈ hold); the reward family is intentional — add variants, never collapse;
BTC-only until the data mine backfills altcoins; `combo_noaction=-1` is a latent synthetic-short bias
— sweep as a variant, don't edit in place.

### 3. BlackSwan Phase C — huge space exploration (after a Phase-B baseline)

Sweep toggles broadly from the baseline; `skipExplored` + by-setup aggregation + the ledger keep it
self-pruning. Fold in feature bets ONLY if an experiment says they help: **RB1b/2** (more indicators
in `src/data/indicators.py` + `_add_curated_indicators`), **RB5** (causal `dip_score` into the env).

Two Phase-A items were consciously deferred here (baseline-changing experiments, not blind fixes):
- **Calibrate `MIN_TRADES_FOR_FULL_CREDIT`** (now 20) to the real test-window length — needs a couple
  of multi-window runs.
- **Out-of-[-1,1] features on the 1h winning path.** 8 of 25 base-layer features exceed the declared
  `Box(-1,1)` obs bound; `process_df_simple` doesn't squash them. SB3 doesn't clip → the NN sees raw
  values. Historical "winning path" results used exactly this representation, so squashing changes the
  baseline — run it as a normalization EXPERIMENT (tanh-squash the 8 / clip the obs — does it help?),
  mirroring the `np.tanh(x/scale)` pattern in `abstract_dataprovider.py`.

---

## Deferred — bigger work, picked up after the active work

### Multi-dataset / cross-asset robustness testing (replaces Eval for RL)

RL models are validated on a live environment/market, so the held-out "Eval" was removed (gated on a
manifest `evaluate` command; BlackSwan declares none). The replacement: test a trained checkpoint
against MULTIPLE named **regime slices** (long uptrend / long downtrend / choppy-high-swing) and
**cross-asset** (any asset in the same data format) to catch regime-overfit. Sketch:

- The trainer already replays a checkpoint deterministically; the missing piece is selecting the data
  WINDOW/asset (a `testSet` param: asset + time-range, or a named curated slice).
- Likely a manifest `testSets` list (`{id, asset, range/description}`) + a generic "test on set"
  activity writing a `<recordType>-regimetest` record per (run, set).
- Surface a per-regime matrix in run-detail + a compare overlay; flag setups that win on one regime
  but collapse on others. Keep it generic; BlackSwan is the first consumer.

### Environments — SHIPPED (needs a backend restart for the planner change)

Environment levers (market mechanics — fees, TP/SL — distinct from model hyperparameters) are now a
first-class concept: a lever spec tagged `scope: 'environment'` is managed as a named **environment** a
model runs AGAINST, not a model knob. Shipped: `TrainerLeverSpec.scope` + `ExperimentSpec.environments`
(a BUNDLE axis — `expandExperimentMatrix` crosses the model matrix with each env bundle, applied
together, not cartesian; TDD); BlackSwan tags fee/TP/SL/trailing as environment; an **Environments tab**
(CRUD named environments, persisted as `<recordType>-environment` records, with an implicit Default from
manifest defaults); the **Launch form shows only model levers** + an environment multi-select (pick
several → one campaign of configs × environments × seeds); a **By-environment** runs grouping + the
environment shown in run detail (matched from a run's env-lever values). Follow-ups (deferred): a unified
**test matrix** surface (model × environment × dataset, folding in the regime testing); presets/clone
currently ignore env-lever values (env values come from the picker now).

### Activity concurrency — viewer-only budget (SHIPPED, needs live verify) + a server-side pass (deferred)

**SHIPPED — viewer-only activity-count budget.** Replaced the singleton observe + one-at-a-time pump
with a `liveActivities` map + a configurable "Max concurrent activities" budget (localStorage, default 3):
the pump dispatches up to the budget non-blocking, each activity self-observes (`observeActivity` →
`observeTrainActivity`/quick branch), and the Activity tab renders one block per live activity (per-block
Abort/Resume). Resolves the headline pains — a judge no longer blocks campaigns; multiple campaigns run
at once. Adversarially reviewed; 3 bugs fixed (launchActivity slot-leak on a bookkeeping throw; pumpQueue
dropping items on a transient launch failure; concurrent-campaign auto-eval drop → `enqueueMissingEvaluations`
now falls back to all-completed-missing). KNOWN LIMITATION (drives the deferred work below): the backend
keys the `-progress` and `-campaign` records as `'latest'` per recordType, so two concurrent campaigns of
the SAME project share that record — their per-block LIVE progress is best-effort (run records + results
are unaffected). Resource trade-off accepted: N campaigns each keep their own "Max parallel runs", so total
processes = the sum (no host-enforced cap yet). Needs LIVE verification (start 2 campaigns + a judge).

The rest still want a host-aware, browsable, concurrency-capped center (deferred):

- **Per-activityId progress/campaign records.** Key the `-progress`/`-campaign` records by activityId (not
  `'latest'`) so concurrent same-project campaigns each show their own live progress — the clean fix for
  the limitation above.
- **Host-enforced global RUN cap.** A backend semaphore on `LocalComputeRunner.runJob` (the shared
  `trainerLocalRunner`) so total concurrent training runs are bounded regardless of how many campaigns
  are live — the durable resource guard the viewer-only budget doesn't provide.
- **Server-side queue drain.** Chain the next queued activity on the backend when one settles, so
  follow-ups advance while the app/viewer is closed.
- **Stalled-run resume — SHIPPED (client-driven).** A backend restart leaves a `running` ActivityRun in
  the store with no live controller; the viewer now surfaces each such stalled activity as a PAUSED
  block with **Resume** (re-launches via `resumeActivity` → the trainer's completed-record skip re-runs
  only the PENDING runs) and **Discard** (backend `abortActivity` now marks an orphaned `running` record
  aborted even with no controller, TDD). DEFERRED: a server-side BOOT scan that auto-reclaims/relaunches
  resumable runs with no client present (the durable version) — the client-driven path covers it while
  the app is open.
- **App-nav unseen-results badge.** HOST-DERIVED (count trainer activities `finishedAt` since the user
  last opened the app tab → Sidebar app-tab badge, web + mobile); the "app reports unseenCount" design
  is broken because the viewer unmounts when you leave.
- **Run→Activity link.** Needs a browsable per-activity history first; then `activityId`-tag
  eval/verdict records and link into it.

### The data mine — a shared dataset project for every model trainer

A standalone repo (`thefactory-datamine`) that is the **source of truth for training data**: gather raw
data, clean + validate + normalise, publish versioned/reproducible. Trainers declare which prepared
dataset(s) they need (the manifest's `data[]`); the data mine + content-addressed cache deliver them.
Architecture decision: store **MINIMAL raw OHLCV only** — indicators + higher fidelities are derived AT
RUNTIME in the consumer, so storage stays small and an indicator fix is a one-line code change. The
basis exists (`BlackSwanPriceEmitter`: Binance miner + indicator engine, ~80% the right shape; the
indicator engine is now a REFERENCE for the runtime formulas, not a storage artifact). Remaining job:
(a) gather + clean raw OHLCV (gap/dedup/continuity checks, NaN sanitisation, mine missing intervals);
(b) generalise the QW6 `derive_cache` (1m canonical → derive+cache fidelities centrally); (c) the
content-addressed cache + remote-runner data path from one curated origin.

**Guided data discovery (north-star 1).** Take a problem statement + model goal → run deep research on
what data exists (sources/APIs/coverage/cost/licence/granularity, reuse the deep-research harness) →
propose candidate datasets + a mining plan + trade-offs → hand off to gather→clean→cache. Makes "I have
a problem" → "I have training data" a guided flow, output a cited report + an approved mining plan.

### Code-change risk model — a third ML consumer (research first)

A trainer-conformant project scoring an agent's diff/PR by bug-likelihood (later reverted/fixed or
CI-failing) — a calibrated signal to gate review effort. Gathering labelled data is the hard part, so:

1. **Research first** (deep-research harness): survey public JIT-defect datasets (ApacheJIT, Defectors,
   JIT-Defect4J, ManySStuBs4J, CVEfixes, Big-Vul, Devign, CodeXGLUE) vs mining our own from the
   `thefactory-*` git histories. Output a cited report + go/no-go on a workspace-mined dataset.
2. **Data** (via the data mine): SZZ-style labeling over git histories; features from codeIntel
   (churn, complexity, coverage of touched code, diff size).
3. **Train**: a `risk-classifier` trainer project (sklearn/torch, objective = AUC or precision-at-k).
4. **Consume**: wire the score into the review / expert-panel / verifier path.

Depends on the data mine.

### Optional phases

- **Phase 8 — Autopilot + live handoff.** Scheduled meta-activity (propose → run → judge → promote,
  human-approved); on a winner, tag the checkpoint for live trading (`run_server_model.py`).
- **Phase 9 — Jupyter notebooks (UNDERSCOPED).** View/edit/execute a project's `.ipynb` from the
  Overseer. To scope: render-vs-edit depth, where the kernel runs (host/sandbox/remote runner), how
  notebooks read campaign records/artifacts, security (arbitrary code → likely the sandbox profile).

### Small deferred items

- **Full RL resume** — per-episode RL checkpointing + `set_env` continuation for true mid-training
  resume (the regression line already resumes; RL restarts from zero). Revisit if worth the loop surgery.
- **True multi-asset** — the `asset` lever is BTC-only until the data mine backfills altcoin 1d/1h
  klines; the lever + per-symbol globs + `data_inventory` gating are already correct, so it expands with
  no code change.
- **Runs column-condense on select** — today the list pane narrows + scrolls horizontally; a true
  fewer-columns-when-selected mode is a follow-up.
- **Per-version leaderboard** — today: the Versions tab + per-run version + by-setup/by-experiment
  groupings; a dedicated per-version leaderboard is a follow-up.
- **Runner-channel WebSocket upgrade** — job dispatch is already ~instant (long-poll `wake()`); a WS only
  shaves ~1.5s log-batch latency, invisible until a live-log UI consumes it.
- **Remote git repoRefs** — the engine emits local paths only; wire git refs + project bootstrap when a
  real remote machine needs it (BlackSwan local path covers today).

## Open questions (decide when the dependency lands)

- **Remote artifact/checkpoint storage** — keep-on-runner + reference vs upload; how a winning remote
  checkpoint reaches the live trading server. Has meaning only once remote runs **and** the live-trading
  handoff (Phase 8) both exist. (Phase 6/8.)
- **GPU + sandbox profile for training images** — `--read-only` rootfs vs ML caches; `--gpus` is wired
  but unexercised because the runner agent runs jobs directly (not through Docker-sandboxed
  `SandboxTools`). Revisit if/when training runs inside the sandbox image. (Phase 6/7.)
- **Judge/proposer model transport** — `ModelSelection` (API vs CLI), being overtaken by the in-flight
  `ModelSelection` refactor (ctx carries `model: ModelSelection` with a `cli` member). Revisit once the
  CLI inference stage lands — until then judge/propose run on API. (Phase 5.)
