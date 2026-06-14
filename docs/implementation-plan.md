# thefactory-modeltrainer — implementation plan

Remaining work only. What's built and how it fits: `docs/architecture.md`. The contract:
`docs/model-training-standard.md`. The core loop (Phases 0–7) is **built + verified**: the
engine, backend activities, viewer, remote runner, and three conformant consumers —
`examples/cartpole`, `examples/tabular`, and **BlackSwan** (`/Users/cloud/Documents/Work/BlackSwan`,
the trading line, Phase 7 done). The engine stays domain-oblivious so any further model is
_data + the thin CLI contract_, not engine code. What's below is optional phases, small
cleanups, and deferred new work.

## North star — two co-equal outcomes

1. **Make the pipeline/app/project the best it can be for creating models** — for ANY model, end to end
   (propose → run → judge → explore), with excellent, self-explanatory results + comparison UI, and a data
   layer that stores the minimum and derives the rest at runtime. **The data layer must also GUIDE a user
   who has a problem to solve but doesn't know what data to mine or how** — walk them from "here's my
   problem" to "here's what data exists, what to mine, and how", via deep research + exploration of
   available sources (see "the data mine" below).
2. **Use it to make BlackSwan the best trading model.** Simple, in STRICT ORDER: **(A) correctness first**
   — all the data, all the rewards, all our processes must be CORRECT before any result is trusted;
   **(B) find ONE setup that trades well** — a config that trades often + profitably, stable across seeds,
   vs buy-and-hold; **(C) then a huge space exploration** of the toggles to find the best. **Measurable
   progress is the whole point** — that's what the model trainer is for.

Co-equal — don't trade one for the other; the BlackSwan work is the forcing function that hardens the
generic pipeline. The BlackSwan section below is now structured around the A → B → C order.

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

- App-nav unseen badge: the Overseer App tab already spins on a live activity, but an
  unseen-results badge while idle needs app→host plumbing (the embedded app reports its unseen
  count). The in-app unseen badges cover the immediate need.

### Considered and declined (so they aren't re-raised)

- `RecommendTools.buildProductCatalog` → `runActivityWorkItems` migration: **not a real
  duplicate.** Recommend's loop is the richer of the two — it streams mid-supplier progress
  (`gathering`→`done`), carries a `supplierStates` map, and absorbs a gather failure as
  "done-but-failed" instead of throwing; the generic engine only emits
  `{done,total,skipped,failed}` per completed item. Forcing the migration would be a
  behaviour-changing downgrade for ~10 lines saved.

### Deferred to after Phase 9

- Runner-channel WebSocket upgrade: deferred, not dropped. Job dispatch is already effectively
  instant (the channel `wake()`s a waiting long-poll the moment a job is enqueued); a WS would
  only shave the agent's ~1.5s log-batch latency, and raw runner logs aren't surfaced in the
  viewer today — so the win is currently invisible. Revisit when a live-log UI exists to consume
  it.
- Remote git repoRefs: the runner agent clones but assumes a self-bootstrapping checkout; the
  engine still emits local paths only. Wire git refs + project bootstrap when a real remote
  machine needs it (BlackSwan local path covers Phase 7).

---

## Live verification owed

Not yet run **embedded in Overseer** (build/test passing ≠ live-on-device): the BlackSwan-improvements
UI (price-action `runChart` + dataset badge, "Max concurrent runs", "Quick start", the dip line as a
second project via `.factory/trainer-dip.json`) AND this round's additions (By-experiment view, the
LLM-verdict + your-conclusion ledger columns, the seed-stability column, the trade-aware `traded_return`
objective + `few_trades` flag). Reload the viewer; BlackSwan changes apply on the next training run.

### Deferred (consciously)

- **Full RL resume** — per-episode RL checkpointing + `set_env` continuation for true mid-training
  resume. Lean run-health shipped (a failed run is recorded + re-dispatched, and the regression
  line resumes from its per-episode checkpoint), but the RL trading line still restarts from zero.
  Revisit if mid-run RL continuation is worth the training-loop surgery.
- **True multi-asset** — only BTCUSDT has 1d/1h klines, so the `asset` lever is BTC-only today.
  The lever + per-symbol globs + `data_inventory` capability gating are root-cause-correct; altcoin
  1d/1h backfill is the data mine's job, after which the lever expands with no code change.

## Hub UI — open follow-ups

(The A1–A3 / B1–B2 / C1–C4 roadmap shipped — see git.) Not blocking; each needs a trigger:

- **Global concurrency budget (issues #1+#3)** — the client queue pump runs ONE activity at a time
  (`findLiveTrainerActivity` guard, app.js) and "Max parallel runs" is a PER-CAMPAIGN pool, so evaluate
  runs serially and a campaign under the cap leaves spare capacity idle. The fix is a GLOBAL job budget:
  a backend semaphore on `ComputeRunner.runJob` (cap total concurrent subprocesses across all activities)
  + let the pump dispatch multiple activities. Architectural (backend + client); has a "global budget
  shared across activities/projects" UX to communicate. ~M.
- **Model chip recents + reactivity (issue #5)** — the activity model chip shows only the active agent's
  single model, not a recents dropdown, and doesn't react when the active agent changes. Fix in the shared
  headless: surface a CLI/model recents list to `useActivityChipCli` + subscribe to active-CLI/agent state
  changes (CliConfigsProvider). Cross-client (web/desktop/mobile). ~M.
- **Run→Activity link (issue #4 secondary)** — Runs now spins only for run-specific work (judge/eval); add
  a "View activity" link from a run's evaluation/verdict to its Activity item (store `activityId` on the
  eval/verdict records in ModelTrainerTools, render a link in run detail).
- **One-click "AI help" on a failed run** — failed runs now show the error + `logTail` in run detail. The
  next step is a button that opens a chat seeded with the failure (error + logTail + config) to diagnose/
  fix. Needs a bridge `discussTopic`/`requestChatSidebar` + host support (copy the knowledge-viewer pattern).
(Closed: **B1 conditional-best** — choice options now annotate the best value CONDITIONAL on the other
selected choice levers, refreshed in place on every change (no form re-render, so sweep/seed/thesis
selections survive). **evaluate command** — `trainer/run.py --evaluate` + manifest `evaluate`; re-tests a
checkpoint without retraining. **Ledger-note affordance** — decided: leave drill-in editing.)

## BlackSwan — the path to a trading model (A → B → C)

Strictly ordered: **(A) make everything correct → (B) find ONE setup that trades well → (C) explore the
toggle space for the best.** "Tried" is data-driven — every run records `config` + `setupKey` + `metrics`;
`skipExplored` prevents re-running a setup; the by-setup / by-experiment ledger is the memory. Full
pipeline map + holes: `docs/blackswan-pipeline-map.md`. (Context from the 2026-06-13 scrutiny: the model/
reward/feature space is already heavily explored — ≈15 RL algos, the combo family, many TP/SL variants;
indicators "tried and didn't help"; minute data ≈ noise — so the leverage is correctness + a disciplined
search, NOT more feature-richness.)

### Phase A — Correctness (the gate: no result is trusted until this passes)

Shipped this round (data + rewards + processes): fidelity-scaled rolling windows (QW1) + taker order-flow
(QW2); 1m-canonical derive+cache for fidelities (QW6, derived==native to 0.000000) + runtime-computed
indicators (verified == the emitter to ~1e-8); trade-aware `traded_return` objective + `few_trades`/
degenerate health; `combo_all_fee` reward variant; wider test window + per-window robustness (RB7); the
experiment ledger (by-setup median/IQR/stability, by-experiment, conclusions); + the historical
`results_hour.ods` import. (QW5: `model_config.py` default restored to the winning `[512,64]`.)

**NEXT — the scrutinous correctness audit (do TOGETHER, before trusting any result or picking the Phase-B
setup).** Look hard at the actual run data + code and, per pillar, decide it's right:
- **Data** — do the derived bars + runtime indicators actually feed the model correctly (no constant/NaN/
  zero columns, right per-layer alignment, no leakage)? Is the train/held-out split clean?
- **Rewards** — is the reward FAMILY itself sound (combo_all & friends): does it genuinely reward "trade
  often + well", or can it be gamed? Is the `combo_noaction=-1` synthetic-short bias intended?
- **Processes** — is `traded_return` the right scalar? do the metrics + health flags measure what we think?
  is the eval honest (single held-out window, multi-seed, window-robustness)? does the hub surface + flag
  everything a newcomer needs to read a result correctly?
Output: a short list of any correctness defects to fix → green light for Phase B. Calibrate
`MIN_TRADES_FOR_FULL_CREDIT` (now 20, `trainer/summary.py`) to the real window length here.

### Phase B — Find ONE setup that trades well

Tools shipped: the Launch-tab experiment presets (Exp 1–5), the `use_indicators` on/off experiment,
multi-seed + the seed-stability column, the trade-aware objective + window-robustness. **Success = a config
that trades OFTEN and PROFITABLY, stable across seeds, beating buy-and-hold.** Start: `use_indicators`
on/off on 1h, multi-seed (the 1d prelim was seed-dependent, so seeds matter); read By-experiment +
By-setup; lock in the first setup that clears the bar.

### Phase C — Huge space exploration (after a Phase-B baseline)

Sweep the toggles broadly from the baseline; `skipExplored` + by-setup aggregation + the ledger keep it
self-pruning. Fold in feature bets ONLY if an experiment says they help: **RB1b/2** (compute more
indicators in `src/data/indicators.py` + `_add_curated_indicators`), **RB5** (causal `dip_score` into the
trading env, `[L]`).

**Ground rules (do not violate):** profit is the objective, never beat-hold (hold is a display control),
expressed as the trade-gated `traded_return`; trade **often and well** (a 1-trade run ≈ hold); the reward
family is intentional — add variants, never collapse; BTC-only until the data mine backfills altcoins.
`combo_noaction=-1` is a latent synthetic-short bias — sweep as a variant, don't edit in place.

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

**The basis already exists — `BlackSwanPriceEmitter`** (`/Users/cloud/Documents/Work/BlackSwanPriceEmitter`):
a NestJS Binance miner (`mines/price-mine-binance.service.ts` → `raw/prices/<COIN>/<interval>/<year>-<month>.json`,
full OHLCV **+ taker order-flow**) plus an indicator engine (`indicators/core/inidicators-core.service.ts`)
that produces exactly the 112-key `indicators` object in BlackSwan's klines. It is ~80% the right shape
(mine → persist → facade). **Architecture decision (2026-06-14): the data mine stores MINIMAL raw OHLCV
only — it does NOT bake in or version indicators.** Indicators (and higher fidelities) are DERIVED AT
RUNTIME in the consumer (`src/data/indicators.py` + `trainer/derive_cache.py`), so the stored/transferred
data stays small and an indicator fix is a one-line code change, not a re-derivation of gigabytes. So the
data mine's remaining job is: (a) **gather + clean raw OHLCV** — gap/dedup/timestamp-continuity checks,
inf/NaN sanitisation, and mining the missing intervals (the emitter's indicator engine becomes a REFERENCE
for the runtime formulas — already ported + verified — not a storage artifact); (b) **generalise the
QW6 `derive_cache`** — 1m canonical, derive+cache higher fidelities centrally so consumers never
re-aggregate; (c) the content-addressed cache + remote-runner data path pulling from one curated origin.

**Guided data discovery (north-star outcome 1).** A user often has a PROBLEM to solve but doesn't know
what data to mine or how. The data mine must guide them from "here's my problem" to a concrete dataset:
(1) take the problem statement + the model goal; (2) run **deep research / exploration of what data
actually exists** (sources, APIs, public datasets, coverage, cost, licence, granularity) — reuse the
deep-research harness; (3) propose candidate datasets + what to mine + how (the miner config) + the
trade-offs; (4) hand off to the gather→clean→cache pipeline above. This makes "I have a problem" → "I have
training data" a guided flow, not tribal knowledge — the data-side analogue of the hub's propose→run→judge
loop. Output a short cited report + a recommended mining plan the user approves.

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

## Open questions (deferred — decisions to make when their dependency lands)

- **Remote artifact/checkpoint storage** — keep-on-runner + reference vs upload; how a winning
  remote checkpoint reaches the live trading server. _Deferred because_ it only has meaning once
  remote runs **and** the live-trading handoff (Phase 8) both exist; deciding now would be
  deciding in a vacuum. Revisit when a remote campaign produces a checkpoint someone needs
  elsewhere. (Phase 6/8.)
- **GPU + sandbox profile for training images** — `--read-only` rootfs vs ML caches; `--gpus`
  is wired in SandboxTools but unexercised. _Deferred because_ it isn't on the active path: the
  runner agent runs jobs directly (`spawnStreamingCommand`), **not** through the Docker-sandboxed
  `SandboxTools`, so there's no rootfs/GPU profile to exercise until a Docker-sandboxed runner is
  real. Revisit if/when training runs inside the sandbox image. (Phase 6/7.)
- **Judge/proposer model transport** — `ModelSelection` (API vs CLI). _Deferred because_ it's
  being overtaken by the in-flight `ModelSelection` refactor (the activity ctx now carries
  `model: ModelSelection`, and `ModelSelection` gained a `cli` member); resolving it means
  finishing that refactor's CLI inference path. Revisit once the CLI inference stage lands — until
  then judge/propose run on API. (Phase 5.)
