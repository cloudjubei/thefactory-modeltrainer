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

- **B1 conditional-best** — choice options show the marginal best-so-far value (★) but not the value
  best _conditional on the other selected levers_. Needs a live form re-render per lever change
  (recompute from history filtered to the current selection). Do when marginal-best gets misleading.
- **A1 live concurrency resize** — concurrency is fixed at launch; can't change it on a running campaign
  (label now says so). Real fix needs a mid-run control signal: UI → running activity → a resizable
  `runActivityWorkItems` pool (drop the fixed `Promise.all(N)` for a top-up scheduler reading a live target).
- **One-click "AI help" on a failed run** — failed runs now show the error + `logTail` in run detail. The
  next step is a button that opens a chat seeded with the failure (error + logTail + config) to diagnose/
  fix. Needs a bridge `discussTopic`/`requestChatSidebar` + host support (copy the knowledge-viewer pattern).
(Closed: **evaluate command** — `trainer/run.py --evaluate` + manifest `evaluate` shipped; re-tests a
saved checkpoint without retraining (skips the train phase), Evaluate button re-enabled. **Ledger-note
affordance** — decided: leave drill-in editing, it's context-rich.)

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

- **QW6 — use native 1h klines** `[MED→HIGH/M]` — 85 `BTCUSDT-1h-*.json` exist but the 1h path loads
  ~1.5GB of 1m JSON to rebuild bars that already exist. (Bonus: at native fidelity the QW1 window-scaling
  is identity, and it removes the `process_fidelity` aggregation cost.)

(QW5 done — `model_config.py` default restored to the winning `[512,64]`; `[8192,512]` was the −40% loser.)

### Research bets (higher potential; bigger or uncertain)

- **RB1 — curated indicators into the winning path** `[HIGH/M]` — the 112 precomputed indicators are
  unreachable on `process_df_simple`. Tier by clamp-safety: **Tier 1 = drop-in free** (rsi/williams/
  stochastic/choppiness — natively `[-1,1]`); **Tier 1b = light rescale** (bollinger/donchian ~[0.85,1.17];
  kallman/disparityIndex ±0.03 near-inert); **Tier 2 = clamp-hostile, needs RB6 first** (meanReversion ±4,
  cci, turbulence 0–15, obv −2.7, sortino→inf). Start Tier 1 → RB6 → 1b/2.
- **RB5 — feed the working dip score into the trading env** `[HIGH/L]` — add a causal `dip_score`
  observation, or a cheaper inference-only buy-veto.
- **RB6 — rescale features instead of hard-clamping obs to [-1,1]** `[HIGH/S→M]` — **prerequisite for
  RB1's rich features.** Per-feature tanh/robust-quantile squash applied identically at train+test.
- **RB8 — sanitise indicator data quality at source + version the indicator spec** `[MED/M]` — data-mine job.

(Closed: **RB4** — `combo_all_fee` reward variant shipped (combo_all base − per-trade turnover penalty,
tunable `combo_fee_penalty`; in the manifest choices). **RB7** — test window widened to all on-disk 2024
(auto-clipped) + per-window robustness in `summary.py` (`worst_window_return_pct`, `windows_profitable_pct`,
`window_returns_pct` series).)

**Ground rules (do not violate):** profit is the objective, never beat-hold (hold is a display control),
now expressed as the trade-gated `traded_return`; trade **often and well** (a 1-trade run ≈ hold);
the reward family is intentional — add variants, never collapse; BTC-only until the data mine backfills
altcoin 1d/1h. `combo_noaction=-1` is a latent synthetic-short bias — sweep as a variant, don't edit in place.

**Recommended next:** the **features track — RB6 → RB1 Tier 1 → RB1 1b/2** (RB6 first: per-feature
rescaling gates the clamp-hostile indicators; then surface the precomputed indicators into
`process_df_simple`). These are feature-engineering bets whose payoff is proven by a campaign, not just
the code change — build + then run an experiment to confirm. Also tractable: **B1** conditional-best
(viewer) and **QW6** (native 1h, housekeeping). Larger/blocked: **A1** live-resize and **RB5** dip-into-env
are `[L]`; the **AI-help** button is host-blocked (needs bridge `discussTopic`). Deep web research deferred.

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
(mine → persist → indicator-compute → facade) but needs three upgrades to be the data mine: (a) **persist
+ version indicators as a first-class artifact** — today they're recomputed-on-read / live-only (the
`raw/indicators/` backfill is commented out) and `config.json` has drifted from the compute engine, so
stamp an indicator-spec version; (b) **data-engineering BlackSwan can't do at source** — gap/dedup/
timestamp-continuity checks, inf/NaN sanitisation (sortino/volatilityVolume divide-by-zero), server-side
resampling (which would retire BlackSwan's `process_fidelity` + the QW1 window bug), and mining the
missing 1s/15m intervals the config promises; (c) **store raw indicator values + a declared normalisation
spec** so the consumer applies clamp-aware scaling (RB6) instead of fighting a baked-in `÷price`/`×2−1`/`/100`
transform. This is the durable home for the Ongoing-Research feature work above.

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
