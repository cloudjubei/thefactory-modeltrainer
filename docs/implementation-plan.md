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

## Future path & UX scrutiny (after the BlackSwan improvements above)

**Massive scrutinisation task and critical look to decide on the future path for the project, as
well as UI/UX usability improvements.** Once the improvements above land and there are real
multi-asset / multi-timeframe results to read, step back and look hard at the actual run data:
judge whether the current experiment design, objective, and levers are the right ones, and
propose the next campaigns to run — expect this may force big changes to how experiments are run.
Pair it with a usability pass over the hub app and the per-run result UI.

## Hub UI — open follow-ups

(The A1–A3 / B1–B2 / C1–C4 roadmap shipped — see git.) Not blocking; each needs a trigger:

- **A1 live concurrency resize** — concurrency is fixed at launch; can't change it on a running campaign
  (label now says so). Real fix needs a mid-run control signal: UI → running activity → a resizable
  `runActivityWorkItems` pool (drop the fixed `Promise.all(N)` for a top-up scheduler reading a live target).
- **One-click "AI help" on a failed run** — failed runs now show the error + `logTail` in run detail. The
  next step is a button that opens a chat seeded with the failure (error + logTail + config) to diagnose/
  fix. Needs a bridge `discussTopic`/`requestChatSidebar` + host support (copy the knowledge-viewer pattern).
(Closed: **B1 conditional-best** — choice options now annotate the best value CONDITIONAL on the other
selected choice levers, refreshed in place on every change (no form re-render, so sweep/seed/thesis
selections survive). **evaluate command** — `trainer/run.py --evaluate` + manifest `evaluate`; re-tests a
checkpoint without retraining. **Ledger-note affordance** — decided: leave drill-in editing.)

## Ongoing Research — what to try next (BlackSwan)

**"Tried" is data-driven, not a manual list** — what's been run lives in the run records
(`config` + `setupKey` + `metrics`); `skipExplored` prevents re-running a setup. Prune an item once
its result is recorded; add ideas as they surface.

> **Why this backlog (2026-06-13 scrutiny):** the model / reward / feature space is already heavily
> explored (≈15 RL algos, the combo reward family, many TP/SL variants; indicators tried and didn't
> help; minute data ≈ noise). So feature-richness (RB1) is partly re-treading; the under-explored holes
> are evaluation rigor, problem formulation, and the experiment ledger. Map: `docs/blackswan-pipeline-map.md`.

### Open items (carry-over from this round)

- **Calibrate `MIN_TRADES_FOR_FULL_CREDIT`** (currently 20, in `trainer/summary.py`) — now actionable:
  RB7 shipped, so run a few multi-window campaigns and set the "trade often" bar to the window length.

(Closed: **ledger live-write** — per-project storage roots shipped (`dataRoot` on the project registry;
`thefactory-modeltrainer` → `BlackSwan/.factory/data`), 258 historical records written there + the hub
reads them in file-storage mode. **Objective-scale migration** — moot, no pre-change records. **Pi-Cycle
windows** — deliberately left unscaled.)

### Remaining quick wins

(All cleared. **QW6 done — the right way (1m stays canonical).** `trainer/derive_cache.py`
`ensure_derived` derives 1h from the 1m source (verified derived == native 1h to 0.000000) + caches to
`binance/derived/` (rebuilt when 1m changes); the 1h path reads the cache instead of re-reading ~1.5GB of
1m every run. NO separately-mined native files (that would be a 2nd source). Derived bars carry no
indicators, so `use_indicators` is a no-op on 1h — RB1-on-1h waits for the data mine to compute indicators
on the canonical bars. (RB1-on-1d still works via native 1d files; those are an eventual derive+cache
candidate too.) First 1h run builds the cache (one-time ~1.5GB read); fast after. **QW5 done** — `model_config.py`
default restored to the winning `[512,64]`.)

### Research bets (higher potential; bigger or uncertain)

- **RB1b/2 — compute more indicators** `[MED]` — the curated set is small (4 Tier-1 + 3 squashed Tier-2).
  More can be added to `src/data/indicators.py` + `_add_curated_indicators` (Tier-1b bollinger/donchian
  ratios, kallman/disparityIndex; more Tier-2 cci/sortino with RB6 scales) once an experiment shows the
  current set helps. Decide the set from the `use_indicators` experiment, don't bulk-add blind.
- **RB5 — feed the working dip score into the trading env** `[HIGH/L]` — add a causal `dip_score`
  observation, or a cheaper inference-only buy-veto.

(Closed: **RB6** — parameter-free `tanh(x/scale)` squash (constant scales → identical train/test),
bounding clamp-hostile indicators to [-1,1]. **RB1 — indicators are now COMPUTED at runtime from OHLCV**
(`src/data/indicators.py`, formulas verified == the emitter to ~1e-8) via a shared `_add_curated_indicators`
called from `process_df` (1d) AND `process_df_simple` (1h), gated by a `use_indicators` lever (default off);
**works on EVERY path incl. the 1m-derived 1h winning path — no data mine, no precomputed dict** (this is
the user's "store minimal data, derive everything at runtime" architecture; was [[RB8]] data-mine work,
now resolved at runtime). **RB4** — `combo_all_fee` reward variant. **RB7** — wider test window + per-window
robustness metrics. **RB8** — moot: indicators derived at runtime, so no need to sanitise/version a stored
indicator artifact.)

**Ground rules (do not violate):** profit is the objective, never beat-hold (hold is a display control),
now expressed as the trade-gated `traded_return`; trade **often and well** (a 1-trade run ≈ hold);
the reward family is intentional — add variants, never collapse; BTC-only until the data mine backfills
altcoin 1d/1h. `combo_noaction=-1` is a latent synthetic-short bias — sweep as a variant, don't edit in place.

**Recommended next:** **run the `use_indicators` experiment on the 1h winning path** (indicators are now
computed at runtime there too) — sweep on/off, multi-seed, read `traded_return` / win% / the window-
robustness metrics in the hub. A quick 1d prelim showed a mixed, seed-dependent effect (helped one seed,
hurt another), so multi-seed is essential. That result decides whether to expand the indicator set (RB1b/2).
Remaining: **RB1b/2** (more runtime indicators, after the experiment), **RB5** dip-into-env `[L]`, **A1**
live-resize `[L]`, the host-blocked **AI-help** button, and the **data mine** (now just gathering + cleaning
minimal raw OHLCV + generalising the derive-cache — no longer responsible for indicators). Deep web research deferred.

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
