# thefactory-modeltrainer — implementation plan

Remaining work only — shipped history lives in git + the session memory. What's built and how it
fits: `docs/architecture.md`. The contract: `docs/model-training-standard.md`. The core loop (engine,
backend activities, viewer, remote runner) is built, with three conformant consumers —
`examples/cartpole`, `examples/tabular`, and **BlackSwan** (the trading line). The engine stays
domain-oblivious: any further model is _data + the thin CLI contract_, not engine code.

## North star — two co-equal outcomes

1. **Best generic pipeline/app for creating ANY model**, end to end (propose → run → judge → explore),
   with self-explanatory results/comparison UI and a data layer that stores the minimum and derives
   the rest at runtime — and that **guides a user from "here's my problem" to "here's what data to mine
   and how"** (see "the data mine").
2. **Use it to make BlackSwan the best trading model**, in STRICT ORDER: **(A) correctness** →
   **(B) find ONE setup that trades well** → **(C) huge space exploration**. BlackSwan is the forcing
   function that hardens the generic pipeline — don't trade one outcome for the other.

## Repo split (governs all phases)

| Repo                                    | Owns                                                                                                                               |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **thefactory-modeltrainer** (this repo) | `ModelTrainerTools`; matrix planner; campaign loop; judge/propose orchestration; the viewer; the standard + `examples/`.           |
| **thefactory-tools**                    | Generic infra only: `ComputeRunner` seam (+ future `RemoteComputeRunner`, `ContentAddressedDataCache`, pairing); work-item engine. |
| **thefactory-backend**                  | Activity registration + composition; app-view serving; future PIN-pairing endpoints + runner WS channel.                           |
| **clients**                             | Future Compute Runners settings/pairing screen (native, cross-project).                                                            |
| **the runner agent**                    | Future Docker-packaged connect-out program.                                                                                        |
| **BlackSwan** (the trading repo)        | Its `TrainerManifest` + additive `trainer/` CLI conformance. No Overseer code.                                                     |

---

## NEXT — the active work

### 1. BlackSwan Phase B — find ONE setup that trades well

The OOS-honest lever matrix is in place: walk-forward windows (2022/2023/2024), shorting, vol-targeted
sizing, multi-fidelity (`fidelity_set`), the unified `combo_unified` reward weights, regime/vol features,
and the `obs_squash` normalization experiment. A setup counts only when it **beats
buy-and-hold out-of-sample net of 0.1% fees**, with profit that is NOT concentrated in up-regimes
(genuine timing, not beta), stable across seeds AND windows.

**2026-07-20 corpus diagnosis (20,888 leak-free runs; `project_blackswan_search_diagnosis`).** No setup
trades AND beats buy-and-hold across all four walk-forward windows (≥2 windows: 46 setups, ≥3: 4, ≥4: **0**);
**93% of strong wins are the single 2022 bear window** and invert in up-years — a GENERALIZATION failure, not a
bad objective (beta-neutral: corr with hold 0.008) nor low Sharpe (annualized winners ≈1.3). The v5 cohort is
mostly VALID (only the `tf=1d/fs=1h` look-ahead config leaked). Next, in EV order — driven from the new
**Diagnosis tab**:
1. **Replicate** the ~top-20 setups across ALL windows × ≥5 seeds; select on worst-window `return_vs_hold`
   (Diagnosis → Replicate — one campaign per window).
2. **Reseed** the promising single-seed setups to ≥5 seeds (Diagnosis → Reseed) so luck separates from edge.
3. **`max_drawdown` experiment (lever + metric SHIPPED).** `trainer/summary.py` now emits `max_drawdown_pct`
   (signed peak-to-trough of the OOS equity curve) and a new sweepable reward lever `combo_drawdown_penalty`
   (default 0 = off; `penalty = weight × |open-position drawdown|`, combo_unified only) is wired end-to-end
   (manifest → config_builder → model_config → abstract_model → `base_crypto_env._drawdown_penalty`), all
   green (1290 BlackSwan tests). REMAINING (user): run the sweep `combo_drawdown_penalty ∈ {0, …}` on a fixed
   base and add the hypothesis "drawdown penalty makes zero-trade setups trade (n_trades 0→>0)" via the
   Hypotheses tab (custom benchmark `n_trades`), watching that it doesn't just silence already-trading setups.
   NB prior attempts suggest it may silence rather than help — a falsifiable test.
4. If nothing survives replication, the edge likely isn't on single-asset BTC — see the less-noisy asset-class
   direction (Deferred → Multi-asset).

- **Wave 3 — multi-asset portfolio / cross-sectional long-short. PROJECT-SPLIT** (see Deferred).

### 2. Model-trainer app

The **Hypotheses registry**, **Papers library**, and **Models catalog** are built — how they fit lives in
`docs/architecture.md` (Hypotheses = falsifiable claims runs prove/disprove; Papers = containers of
hypotheses; Models = the catalog of implemented/proposed models).

- **Reusable components — surfaced + composed.** BlackSwan's manifest seeds the building blocks (feature
  extractors, custom policies/Q-nets, replay buffers, attention + NN blocks, the `DGWO` optimizer) as 12
  `component` catalog entries; each model flavor declares its `components`, rendered in the Models tab as
  linked chips with reverse "used by" on component cards (`flavorComponents` / `modelsUsingComponent` in
  `viewer/models.js`). Optional follow-on: the
  literal `custom_net_arch` block recipe per `-custom` flavor (which attention/NN-block layers a recipe
  uses) is not yet surfaced — derive it from the source rather than hand-declaring (today
  `attention-blocks`/`nn-building-blocks` show no "used by" because recipe wiring isn't asserted).

### 3. Conversational hub — remaining work

The core SHIPPED (see `docs/architecture.md`): the universal `discussBundle` Discuss seam, hypothesis
hygiene census, `recommendTrainingExperiments` + the approval-gated update tools, generation-time spec
validation (`validateSpecAgainstManifest` in `coerceHypothesisItems`), hypothesis-spec migration in
`migrateTrainingRuns` (`planHypothesisSpecMigration`), the `getTrainerState` orientation read tool, and
the 2026-07-16 UX pass (hypothesis-health banner naming + humanized census, per-card status tooltips,
capability-card copy, mobile wrap/grid fixes, tool-view copy). Remaining:

1. **Touch-reachable help layer.** `setupHelpTooltips` fires only on hover/focus; icon buttons and
   badges are the viewer's whole explanatory layer and are unreachable on touch (the hygiene badges
   now carry `tabindex` as a stopgap). Needs a deliberate design: tap-to-reveal for coarse pointers
   or inline glosses at small widths.
2. **Blocked-card action.** A starved card has "Launch the missing runs"; a blocked card's remedy
   (fix the spec) lives only in the tooltip + Discuss seed — add an on-card affordance (e.g. open
   the editor pre-seeded, or a scoped Discuss button).
3. **On-device mobile parity check** of the new hub surfaces (the ≤767px fixes are code-reviewed,
   not device-verified).

### 4. Continued training + cross-dataset evaluation

Train a model FURTHER on additional datasets from a saved checkpoint, then evaluate any checkpoint
against named evaluation datasets — the apples-to-apples yardstick BlackSwan lacks (walk-forward
windows have different test years, so today's metrics are never directly comparable).

1. **Cross-asset + extended-OOS robustness testing (Phase E) — ENGINE SHIPPED, UI remaining.** Re-test a
   TRAINED model on data after its cutoff + on OTHER assets it wasn't trained on, WITHOUT retraining —
   checkpoint-replay via `trainer/run.py --evaluate` (skips training, rebuilds data purely from the config's
   `asset`/`walk_forward_window`). Decisions: checkpoints are **per-run opt-in** (a Launch toggle, only
   checkpointed runs are re-testable — existing runs have none); the display is the **full** set; a **Launch
   setting** declares which assets to cross-test on (all/specified) with a settable **campaign default**.
   - **SHIPPED (all TDD):** open-ended `oos-*` walk-forward windows (test cutoff→latest on disk, via the
     existence-filter — `walk_forward.py`) so a run's own test spans all post-cutoff data; the `evaluate`
     command in BlackSwan's manifest; `missingCrossTestValues` (requested − trained − already-tested, over
     the lever's choices); `ModelTrainerTools.crossTestRun({runKey, lever='asset', values})` — replays the
     checkpoint with the lever overridden, fills only MISSING cells, writes a `{recordType}-settest` matrix
     record keyed by run (a `CrossTestResult` per value, compact — no series/ledger); the `cross-test`
     backend activity (experiment lane). Obs-shaping levers (fidelity/lookback/indicators) must stay fixed;
     only `asset`/`walk_forward_window` vary.
   - **UI SHIPPED too (2026-07-23):** Launch "Keep checkpoints" toggle + "Cross-test after training"
     control (off / all other assets / specific assets, persisted campaign default; selecting it implies
     kept checkpoints) → `keepCheckpoints`/`crossTest` train params, with the train activity spawning ONE
     auto `cross-test` batch over the campaign's runs; run-detail **Cross-test section** (per-asset matrix
     + "Test N missing assets" + "Extend test window (oos-*)"); the runs-table **Robust column**
     (`n/m beat hold` chip from the `-settest` cache); the **Robustness lens** (run × asset vs-hold
     matrix view-mode); and the Diagnosis **cross-asset robustness check** (`checkCrossAssetRobustness`
     — incumbent robust / partial / asset-bound from settest cells, no re-training). Pure logic in
     `viewer/crossTest.js` (tested). REMAINING: exercise live once a checkpointed campaign exists (no
     existing run has a checkpoint); consider auto-refreshing the settest cache on `data:updated`.
2. **Continued training (extra-train).** A launch mode that seeds from an existing run's checkpoint
   (`checkpoint_to_load` exists on the regression line; RL needs SB3 `.load()` + `set_env` +
   `learn(reset_num_timesteps=False)`) and trains on a DIFFERENT dataset bundle; provenance links
   parent run → continued run (`continuedFrom` on the summary) so lineage renders in run detail.
   Judged on the standardised test sets from (1), never on the shifted train window.
3. **Viewer.** "Continue training on…" action on a run (dataset picker), lineage chips, and the
   per-set evaluation matrix. Hypothesis/objective plumbing unchanged — continued runs are normal
   runs with provenance.

### 5. Data refresh + stocks — remaining work

The backfill SHIPPED (see `BlackSwan/data_refresh_report.md`): binance/ through 2026-06 (BTC 1m/1h/1d
from 2017-08; 8 altcoins 1m from 2022-01 + derived 1h/1d), a new stocks/ dir (10 US tickers, 1d,
2018-01..2026-06, yfinance — stooq is bot-walled), idempotent `scripts/backfill_klines.py` +
`backfill_stocks.py`, `scan_stocks_inventory`, walk-forward windows `2025` + `alt-2024`/`alt-2025`
(altcoins MUST use alt-* — plain windows train from 2020), and the manifest's asset lever now lists
all 9 coins. Remaining:

1. **Stocks as a first-class asset class.** Decide how stock tickers enter the trading manifest
   (same `asset` lever vs a separate manifest à la the dip line — sessions/gaps differ from crypto),
   wire the provider path end-to-end for one ticker, and add stock-suitable walk-forward windows.
2. **data_inventory-driven capabilities.** The Datasets capability sections read what is ACTUALLY
   on disk (a small inventory record the trainer publishes) instead of hardcoded choices, so new
   assets appear as data lands; per-asset window capability (which windows an asset's history
   supports) rendered on the asset card.
3. **Refresh cadence.** Re-run the backfill scripts on a schedule (they are idempotent) so data
   never goes stale again.

### 6. xAI — explain WHY the model acted (parallel track)

The xAI track is shipped (git + `docs/architecture.md`); the model-trainer side stays domain-oblivious.
Shipped: the decision-trace spine + the full xAI tab (Phases 1–5); config-space exploration (surrogate +
EI `acquisitionRecommendations` + fANOVA/Sobol + lever-coupling + PCA-projection); and reward/metric
NORMALISATION (`normalizeByEnvironment` re-expressing each run as a robust z-score within its OWN
environment, consumed by the Runs "By dataset/By environment" pooled view and the xAI current-run
"By dataset"/"By environment" standing + `robustnessVerdict` tabs — pure logic in `comparison.js`,
tested in `comparisonViewer.test.ts`); per-step group-saliency (**C1**) + permutation-SHAP attribution
(**B1**, `decision_trace_method="tabular-shap"`, Adebayo-checked, no new dep). Also shipped: the current-run
across-axis UX — favorites resolved against the full snapshot (`findRunAnywhere` + fetch-by-key, exploratory
filters never drop a pin, **View in Runs** jump), `seed` excluded from locked/axis levers in `comparison.js`
(pools over seeds), regime slice toggles (dataset by `timeframe`, environment by `allow_shorting`/`no_sell_action`;
unused env levers hidden), the **By value** one-factor sub-tab (any tunable lever, sortable, Add-runs + Sweep-
with-recommended-values popup), and xAI launches that toast-and-stay (no auto-switch to Activity). Remaining,
in order:

- **B2 — attention-weight viz (attn/custom-net recipes only).** Attention modules compute the weight
  matrices but discard them (`src/model/custom/attention.py`). Surface them into the trace on the BlackSwan
  side, then add a 2-D matrix attribution type + heatmap renderer (the current renderer is 1-D only).
- **C2 — mid-training-checkpoint trace.** Only one final checkpoint is saved today (`src/model/rl_model.py`;
  the supervised path overwrites one best file). Add periodic retained checkpoints, emit a trace per
  snapshot, then diff them via the shipped `DecisionTraceDiff` spine.

Parked (real blocker / low value):

- **Generative counterfactual states** — needs a net-new GAN/VAE over the observation space; none exists.
- **Step-by-step ANIMATION replay** + scrubber. No trace-artifact change needed.
- **`seed` still counts as a model lever in the engine's fANOVA importance** (a separate path from the viewer
  axis logic; needs a manifest scope change to `ignore` + a re-analysis to take effect) — flag, don't
  silently change.

### 7. Exploration autopilot — remaining

The closed-loop strategist, durable `explore` controller, and viewer tab are built and all three consumers are
enabled — including BlackSwan (capped categorical basin axes with `model_name` on top; scalar `traded_return`
north-star). How it fits: `docs/architecture.md`; history: `project_exploration_autopilot` memory. Remaining:

1. **Live acceptance (user-run — needs Python/SB3; restart the backend to load the server-side reducer/controller).**
   CartPole rediscovers ≈500 and enumerates the basins; Wine + BlackSwan cover the space across their models and
   declare the best consistently. Optional refinements once it's exercised: emit a `baseline` metric in BlackSwan's
   `trainer/summary.py` (cleaner basin threshold than the worst-region fallback); let the escalation ladder unfreeze
   non-axis categoricals; Pareto basins.

### 8. Research Diagnostician — diagnose WHY a search hasn't found a winner

A read-only battery over any project's run corpus that answers "is there a strong candidate, and if not,
why" and turns the answer into launchable campaigns. Complements the exploration autopilot (§7): that
SEARCHES, this DIAGNOSES. Domain-oblivious — every check reads the manifest. Motivated by the 20,888-run
BlackSwan diagnosis (see §1); history in `project_blackswan_search_diagnosis` memory.

SHIPPED (viewer, all green — 1202 tests): `viewer/diagnostics.js` (dual-export, `src/diagnosticsViewer.test.ts`,
16 tests) + a **Diagnosis tab**. Seven checks → severity-ranked findings + a verdict + a "do next":
cohort-integrity, discriminability/seeds, null-ceiling (beats the declared baseline / at ceiling?),
**split-consistency** (does the incumbent hold across walk-forward/CV/dataset splits, or is it single-split
luck?), incumbent-separation, budget/coverage, objective-confound. Verdicts derive from EXISTING manifest
fields (`hypothesisBenchmark.metric === objective` ⇒ ceiling target; else ⇒ per-run null baseline). Two
campaign generators wired to one-click launch (a SERIES via `startOrEnqueue`): **reseed** (lift the top-N
promising setups to ≥5 seeds — only the MISSING seeds, never re-run everything) and **replicate** (the
shortlist across every split value, ONE campaign per split/data-bundle for data-load efficiency). Validated
on real data: CartPole → `converged` (hits the 475 target, multiple optima), Wine → `winner-emerging`.

Remaining:
1. **Manifest `diagnostics` blocks** for the projects so the optional checks light up — the defaults already
   work off `hypothesisBenchmark`, but declaring `splitAxis`/`confoundMetrics`/`riskMetrics`/`degenerateWhen`
   sharpens them. For BlackSwan the single line `splitAxis: walk_forward_window` is what turns single-window
   overfit from invisible into a blocker; for Wine, `nullBaseline` = mean-predictor + `splitAxis` = CV fold.
2. **Server-side engine + chat + gate** (optional follow-on): `src/diagnosticsUtils.ts` mirroring the viewer
   logic (Phase-0 exports of `bootstrapDiff`/`benjaminiHochberg`/`baselineOf`) → a `diagnoseSearch` chat read
   tool + an `xaiNarrate`-style narration + feeding the diagnosis into `nextExplorationStep` as a convergence
   GATE (block `converged` until the incumbent holds across ≥K splits — the deterministic fix for §7's
   single-window false-convergence).

---

## Deferred — bigger work, picked up after the active work

### Cross-asset robustness testing

**Pulled forward into NEXT §4 (standardised `testSets` + evaluate-on-set activity).** What stays
deferred here is only the broad cross-asset sweep matrix once many assets are on disk (NEXT §5).


### Activity concurrency — remaining view work

The server-side pass SHIPPED (per-activityId records, host RUN cap, server queue drain, boot-scan
resume, unseen badge, Run→Activity link data). Remaining:

- **Browsable per-activity history VIEW** (Phase 5c) — then surface the Run→Activity link end-to-end.

### Position-blind signal model — a SEPARATE objective (always-on correct signal)

The trading line trains a position MANAGER: actions are position-gated, so an out-of-position or
redundant action (sell while flat, buy while already long) earns ZERO differentiating reward and is
never shaped — at deterministic eval the policy emits uncontrolled junk there (the `blocked_signal_ratio`
metric quantifies it; ~0.95 even on great-return runs). Exp 15's penalties can only SILENCE that output
(push it to hold); they cannot make a flat-state signal MEANINGFUL. A model whose raw per-step output is
directly consumable as a long/flat(/short) signal — independent of any current position, since we trade
in percentages so position size is irrelevant — is a **different objective**, planned separately.

**The clean manager is NOT this (don't conflate them).** Turning on `combo_noop_penalty` (e.g. `duel-dqn-custom`
at 0.1) drops `blocked_signal_ratio` ~0.95 → ~0.02: the model learns to go SILENT (hold) when an action would be
a no-op, using the `in_position` feature. That makes its EXECUTED entry/exit stream clean + directly usable — but
it is still a position MANAGER, read WITH position; it gives no counterfactual per-step opinion ("flat right now —
should I be long?"). So before building the forecaster: confirm the clean manager (explorable TODAY via the
`blocked %` filter / Exp 15) is not already enough. The forecaster needs FRESH training — you cannot convert a
manager run into one — and adds another quality axis to validate, so it is explicitly later, lower priority.

- **Mechanism (reuse, don't fork the env).** The env already carries a `buy_sell_signal` reward family
  (`base_crypto_env.py`) that scores each step's action against the NEXT-step price move regardless of
  position, and a supervised direction line (`forward_horizon`/`prob_threshold`, logreg/gbm) that predicts
  next-return direction. The signal model builds on one of these: a per-step forecast reward (every step
  scored as a prediction, no gate) or a first-class supervised "direction head" whose output IS the signal.
  Reuses the data provider, feature engineering, SB3 algos, and walk-forward harness.
- **It needs its own objective + metrics.** Not `traded_return` (gated portfolio return) but SIGNAL
  QUALITY — precision/recall/coverage of the signal vs realized forward returns, or a signal-following
  backtest — with its own objective name + direction. Likely its OWN manifest (`trainer-signal.json`),
  mirroring how the dip/trend prediction line already splits from the trading manifest (model-training-
  standard §3), rather than overloading the trading manifest with a mode flag.
- **UI implications (must design up front).** The hub assumes one trading objective everywhere: the
  single-objective compare, the hypothesis verdict (`beats-buy-and-hold OOS`), the run chart (trade
  markers), the judge, and the lever picker all bake in `traded_return`. A signal model needs: its own
  "good" definition + verdict rule, signal-quality run metrics, and a signal-overlay chart (predicted
  long/flat vs forward return) instead of trade markers. Decide manifest split vs mode early — the verdict
  - objective plumbing is the bulk of the work, not the model.
- **Migration (derive, don't rewrite).** Runs with `combo_noop_penalty>0` are CLEAN-MANAGER runs, NOT forecaster
  runs — they do not migrate into this approach (different objective + training). Represent the split as a DERIVED
  `approach` facet (from `reward_model`/objective), grouped + filtered at runtime per "store the minimum, derive the
  rest"; only stamp an explicit stored tag via the idempotent `migrations` engine if a persisted field is needed.
- **Cross-approach comparison.** Compare manager vs forecaster on a SHARED yardstick: push the manager's executed
  trades AND the forecaster's per-step signal through the SAME signal-following backtest (same 0.1% fees), then rank
  both on `return_vs_hold_pct` with each approach's native objective shown alongside — a best-of-each-approach
  leaderboard answering "which actually produces the better tradeable signal".

### Multi-asset portfolio / cross-sectional long-short — a SEPARATE project (Phase-B Wave 3)

The one genuine project split. BlackSwan's single-asset env hardcodes one asset everywhere — 1D
state, single-position discrete action, single-symbol data paths, single equity-curve objective — so
multi-asset needs a fundamentally different env: a **3D observation** (`asset × lookback × features`),
a **portfolio action space** (per-asset long/short/weight), a **timestamp-aligning N-symbol data
provider** (misalignment = silent P&L corruption — unit-test against a 3-coin × 100-bar fixture), and
a **rebalance-count or direct Sharpe/Calmar objective with a correlation penalty**. It REUSES
BlackSwan's reward components, feature engineering, SB3 algos, and the walk-forward harness, but ships
as its own ~3–4 week project and blocks none of the in-place wins. Hard dependency: only BTCUSDT
klines are on disk → altcoin backfill (via the data mine) is a prerequisite. Research calls
cross-sectional long-short the strongest-edge config, so it is promising — deliberately sequenced
last. (Distinct from "Other single assets" below, which is just the `asset` lever + a backfill.)

**Strategic direction — less-noisy asset classes with fundamental data (post-diagnosis).** The 20,888-run
BlackSwan diagnosis (§1) points to single-asset crypto PRICE being too noisy to carry a learnable edge. The
sequenced answer after multi-asset: move to **commodities, stocks, and FX** — lower-noise series that,
crucially, are tied to geopolitical/macro-economic drivers, so each brings NEW fundamental data crypto lacks
(rates, CPI/inflation, earnings, inventories/COT, trade balances, an event calendar) that could be the actual
source of edge. Requires the data mine to gather + timestamp-align these series with their fundamentals,
per-asset-class walk-forward windows, and features that FUSE price with the fundamental stream. Test the SAME
model families across classes and judge with the Diagnosis tab's split-consistency + a cross-class leaderboard
(is any edge asset-class-specific or general?). This is the "look at less noisy things" hypothesis made falsifiable.

### The data mine — view + acquire training data on demand

The **source of truth for training data**: gather raw data, clean + validate, and let a user SEE what
data exists (on disk + available to acquire) and DOWNLOAD more on demand from inside the app, then run
tests on it. Storage stays **MINIMAL raw OHLCV only** — indicators + higher fidelities are derived at
RUNTIME in the consumer (`derive_cache`), so an indicator fix is a one-line code change, never a
re-mine. Beyond price, the mine grows to **augment** an asset with the fundamentals that move it (US
macro releases, company events) and to **link** an asset to the related series that drive it (Nvidia↔
semiconductors, Tesla↔lithium) — each new stream point-in-time-correct so a model only ever sees what
was public at that instant.

**Build-in-place decision (supersedes the standalone-repo sketch).** BlackSwan already OWNS the data
(`binance/`, `stocks/`), the idempotent miners (`scripts/backfill_*.py`), the inventory
(`trainer/data_inventory.py`), and the runtime derive (`trainer/derive_cache.py`). So the mine is built
**in-place first**: BlackSwan owns the catalog + miners + a small **data CLI**; the model-trainer
surfaces a **Data tab** + on-demand mining through the app; extract to a standalone `thefactory-datamine`
repo only once a SECOND trainer needs the same data (YAGNI until then). The generic
`ContentAddressedDataCache` + `manifest.data[]` URL path stays for URL-delivered datasets; BlackSwan's
data is mined by script, not fetched by URL, so it uses the catalog+CLI path instead.

**Architecture / seam (domain-oblivious engine, project-declared data commands).** Mirrors the existing
`evaluate` / `benchmarkDevice` optional-command pattern end to end:

- **Catalog registry** (`trainer/data_catalog.py`) — the single static "menu": asset class → instruments
  `{symbol, label, source, sourceSymbol, intervals, directory, tier}`. Consumed by BOTH the miner (what
  to fetch from where) and the catalog emitter (what to show as available).
- **Coverage inventory** (`trainer/data_inventory.scan_coverage`, ✅ built) — the on-disk read-model:
  per symbol/timeframe `{start, end, months, gaps}`. The catalog = menu; coverage = what's in the fridge.
- **Data CLI** (`trainer/data_cli.py`) — two subcommands emitting JSON to `{summaryOut}`: `catalog`
  (registry ⋈ coverage → the full catalog record) and `mine` (read a request `{configPath}`, run the
  right miner for the requested class/symbols/intervals/range, emit a result). Idempotent (skips months
  already on disk); after mining, re-derives fidelities and re-scans coverage.
- **Manifest commands** — two new optional templates on `TrainerManifest`: `dataCatalog` (only
  `{summaryOut}`) and `mineData` (`{configPath}` + `{summaryOut}`), declared in BlackSwan's
  `.factory/trainer.json`.
- **Tool methods** (`ModelTrainerTools`) — `scanProjectDataCatalog(...)` and `mineProjectData(request)`
  invoke those commands via the ComputeRunner (same contract as `benchmarkDevice`) and parse the summary.
- **Backend activities** — `data-catalog` (runs the scan, writes a `{recordType}-datacatalog` record)
  and `mine-data` (runs the mine, streams per-symbol progress, re-writes the catalog record). New `data`
  queue lane so a long download never blocks the training lane.
- **Data tab** (`viewer/data.js` + `app.js`) — reads the `-datacatalog` record via `queryData`, renders
  per-asset-class cards (each instrument: tier, on-disk range + gaps vs available, size), a **Download**
  button → `startActivity('mine-data', …)` with live progress, and a **Refresh** → `data-catalog`. This
  is a THIRD "data" surface, distinct from the *Datasets* tab (run-grouping identity) and
  `manifest.data[]` (URL requirements) — do not conflate them.

#### Phase D1 — Catalog + inventory + on-demand price download — SHIPPED

The full vertical is built + tested: **BlackSwan** `trainer/data_inventory.scan_coverage` (per symbol/tf
range+gaps), `trainer/data_catalog.py` (4 classes, verified tickers, per-instrument `barCloseTz`),
`scripts/backfill_market.py` (generalized yfinance daily miner for commodities+FX, real yfinance smoke),
`trainer/data_cli.py` (`catalog` + `mine` subcommands, idempotent, real catalog emit + mine dry-run
proven); **trainer** manifest `dataCatalog`/`mineData` commands + `scanProjectDataCatalog`/
`mineProjectData` tool methods + `parseDataCatalog`/`parseMineResult` coercers; **backend** `data-catalog`
+ `mine-data` activities (mine on the `research` lane, writes the singleton `{recordType}-datacatalog` +
per-run `-mine` records); **viewer** Data tab (`viewer/data.js` pure helpers + app.js/index.html) showing
per-class cards (on-disk range/gaps vs available, tier, source, `barCloseTz`) with Download / Download-all
→ `mine-data`, and Refresh → `data-catalog`. BlackSwan's `.factory/trainer.json` declares both commands.
Remaining polish: on-device in-app verification of the tab; ETF proxy tickers (`SOXX`/`SMH`/`LIT`) added
to the catalog for D3 linkage; the config-drift cleanups below.

The headline picks below shipped as the catalog entries (research pass D0 verified the tickers, history
depth, and roll/adjustment/`barCloseTz` gotchas — folded into the correctness rules):

- **Crypto (top 5, source `binance`)** — BTC + the 4 largest of the 8 altcoins already on disk (ETH, SOL,
  XRP, DOGE …), mined 1m + derived 1h/1d. Existing `backfill_klines.py` path, now catalog-driven.
- **Stocks (top 10, source `yfinance`)** — the 10 US tickers already on disk (NVDA, MSFT, AAPL, GOOGL,
  AMZN, META, AVGO, TSLA, JPM, WMT), 1d. Existing `backfill_stocks.py` path.
- **Commodities (top ~6–8, source `yfinance` futures, NEW)** — Gold `GC=F`, Silver `SI=F`, WTI `CL=F`,
  Brent `BZ=F`, NatGas `NG=F`, Copper `HG=F` (+ Corn `ZC=F`/Wheat `ZW=F`), 1d, local symbols `GOLD`/
  `SILVER`/`WTI`/… into a new `commodities/` dir.
- **FX (top 5 majors, source `yfinance` `=X`, NEW)** — EURUSD, USDJPY, GBPUSD, AUDUSD, USDCAD, 1d, local
  symbols `EURUSD`/… into a new `fx/` dir (yfinance FX carries no real volume → neutral fill like stocks).

Build: generalize the yfinance daily miner to `(symbol, sourceSymbol, outDir)` so commodities + FX reuse
it; catalog registry; data CLI; the two tool methods + two activities; the Data tab. Also fold the
config-drift cleanups the refresh flagged: `walk_forward.py` windows still cap at test-year 2025 though
2026 is on disk; `data_config.py:48`'s "not on disk yet" comment is stale — a catalog-driven system
should derive available windows from actual coverage.

#### Phase D2 — Augmentation: US macro + company fundamentals (point-in-time) — ACQUISITION SHIPPED; CONSUMING (fusion → training) in progress

New non-price release series, acquired point-in-time and viewable/downloadable through the SAME D1
vertical (the catalog now carries `macro` (13 FRED series) + `fundamentals` (10 EDGAR tickers) classes;
the mine activity → CLI dispatches FRED/EDGAR; the FRED key rides the backend env passthrough — no
plumbing). **The whole value is point-in-time correctness** (the user's "unemployment rates WHEN they are
announced"): a model may only see a value AS KNOWN then. Built + tested (real EDGAR: 1120 point-in-time
observations for AAPL):
- `trainer/pit_fusion.py` — the **leakage guard** (`asof_join` + `release_datetime_ms` DST-aware + a
  per-series publish-time table): a bar sees a value only from its release instant, forward-filled. The
  jobs-report fixture proves January's number is invisible until its February release.
- `scripts/backfill_macro.py` — FRED `output_type=4` (initial-release vintages, `realtime_start` = release
  date); `scripts/backfill_fundamentals.py` — SEC EDGAR companyfacts stamped at `filingDate`, restatements
  kept as distinct rows; `data_inventory.scan_series_coverage` for the release-series dirs.

CONSUMING STEP — the "data projection" design (RAW LEVELS; the model derives its own features). A dataset
is N aligned series; each asset carries a PER-ASSET projection = how much of its own data is exposed.
Levels: **0** = per-asset projection primitive; **A** = one tradeable + global context channels; **B** =
mix N assets, trade one; **C** = mix N, trade many (portfolio — the big env change, parked). Context splits
two ways: GLOBAL (macro — asset-agnostic, a run-level set) vs ASSET-SPECIFIC (folded into that asset's
projection as `with_extra_data`). Representation rule: a series that is already a rate / percentage /
bounded ratio is shown as its LEVEL; a value / count / price / index is shown as CHANGE over time. Both
causal (no fitted/full-sample scaler — the pipeline has none, so a naive standardizer would leak).

- **Level 0 — per-asset projection primitive — SHIPPED (2026-07-25, TDD).** `trainer/projection.py`
  (`resolve_projection`; a per-asset `projections` map overrides the scalar `projection`; coarsest→richest
  `minimal` / `standard` / `with_indicators`). `config_builder` derives the low-level `type` + `use_indicators`
  from it. The manifest `use_indicators` boolean was UNIFIED into a `projection` choice lever (scope
  `dataset`): 19 papers/presets migrated, historical runs migrated (`use_indicators:true` → `with_indicators`,
  else `standard`) via combined set+unset rules that converge. History-preserving: `standard` == the old
  default, `with_indicators` == the old `use_indicators:true`. Runnable NOW: the minimal/standard/with_indicators
  own-data ablation on a single asset.
- **Step 2 — global context channels — SHIPPED (2026-07-25, TDD).** `trainer/context.py` (registry
  `CONTEXT_SERIES`/`CONTEXT_PANELS`, `resolve_context`, pure `fuse_context_series` = pit_fusion + the
  level/change representation, `context_columns`, fail-fast `load_context_observations`). Fused as raw-level
  per-bar COLUMNS in BOTH `process_df` and `process_df_simple` via `AbstractDataProvider._add_context_columns`
  — a NO-OP when `context='none'` (default, byte-identical to pre-context runs); the context set joins all
  three feature-cache keys only when set (no mass cache invalidation). `DataConfig.context` + a `context_set`
  choice lever (`none`/`rates`/`macro_core`, scope dataset). Because the panel is GLOBAL (asset-agnostic
  macro), a context-trained checkpoint cross-tests on other assets with NO extra obs-signature work — the
  same panel fused onto any asset yields the same obs width. Verified: trainer 501 + src/data 235 green.
  Follow-up: pre-first-release bars currently `fillna(0)` (a window-restriction to post-first-release is a
  `config_builder` follow-up needing coverage data; a non-issue for the deep-history FRED panels).
  The FRED macro mine is now LIVE (2026-07-26): `scripts/backfill_macro.py` was fixed category-aware (the
  request had never worked against live FRED — `output_type=4` needs an explicit full-history realtime
  window, and daily rates exceed FRED's vintage-date cap so they fetch standard observations stamped as-of).
  All 13 series mined to `macro/`; `rates`/`macro_core` proven end-to-end on real BTC klines (values
  economically correct incl. the 2023 curve inversion). Key lives in `BlackSwan/.env`; app-triggered mines
  still need it in the backend env.
- **Step 2b — price-source context — SHIPPED + LIVE-PROVEN (2026-07-25, TDD).** The keyless path that works
  on klines already on disk: a linked/peer asset's return fused as a context column (`load_price_observations`
  + a `load_series_observations` source dispatcher; `majors` = BTC+ETH returns, for use on an alt). Proven
  end-to-end on real ADAUSDT klines through the real `process_df_simple`. This is the concrete
  "tie assets to related assets", and the same price-source mechanism a yfinance `gold`/`dxy`/`spx` panel
  (also keyless) will reuse. The live e2e caught two real bugs clean fixtures missed (nanosecond
  `pd.to_datetime` unless via `read_json`; string-typed kline prices).
- **Step 3 — `with_extra_data`.** The 4th projection rung: this asset's own fused context series, same seam.
  This is where the explicit obs-signature gate IS needed (asset-specific panels differ across assets).
- **Step 4 — the 4-arm ablation** (own-price minimal|full × context off|on), pre-registered OOS net-of-fees +
  Sharpe, once a checkpointed panel run exists.

- **US macro (FRED/ALFRED, free API key).** Starter set, exact series ids: unemployment `UNRATE`,
  nonfarm payrolls `PAYEMS`, CPI `CPIAUCNS` (NSA — never revised) / `CPIAUCSL` (SA), core PCE `PCEPILFE`,
  Fed funds effective `DFF` + target `DFEDTARU`, initial claims `ICSA`, retail sales `RSAFS`, real GDP
  `GDPC1`, 10y `DGS10`, curve `T10Y2Y`. **ISM/PMI is intentionally dropped** (removed from FRED ~2016 for
  licensing — do NOT substitute a differently-timed proxy under a "PMI" id). Fetch **ALFRED vintages**,
  not the default series (default FRED returns only latest-revised values for every historical date — a
  silent leak): `fred/series/observations` with `output_type=4` (initial release) or explicit
  `vintage_dates`; store the full vintage matrix, never one collapsed column.
- **Company fundamentals (SEC EDGAR, free, no key).** `companyfacts`/`companyconcept` JSON — earnings/
  report EPS/revenue + slow fundamentals, stamped at the actual **`filingDate`/`acceptanceDateTime`**
  (EDGAR is point-in-time by construction), NEVER the fiscal-period end (yfinance dates FY figures to
  period-end with no filing date → a 45–90-day accounting look-ahead). Restatements are new dated rows,
  not overwrites. Survivorship: EDGAR/yfinance current-universe screens drop delisted names — use a
  point-in-time constituent set or document the bias as unmitigated.
- **Fusion rule (the leakage guard).** Stamp each observation with its actual **release datetime** from a
  per-release publish-time table (08:30 ET employment/CPI/claims/retail/GDP; 14:00 ET FOMC; ~16:15 ET
  H.15 rates — NOT a blanket 08:30), join to price bars on that timestamp, then **forward-fill** to the
  next release. A macro feature at bar *t* = the latest value whose release ≤ *t*. Derived changes
  (MoM/YoY) = **diff of the SAME vintage**, never latest-revised. Unit-test the alignment against a known
  jobs-report calendar (release ≈ 1st Friday for the prior month) — a fixture that fails if reference-
  period alignment ever sneaks back in.

#### Phase D3 — Asset-linkage graph — SHIPPED

Tie a traded asset to the related series that drive it, each a mineable proxy that REUSES the D1 miner.
Built + tested: `trainer/data_linkage.py` (8 verified seed edges between catalogued instruments, each
decorated with its proxy's source + `barCloseTz`), folded into the `data-catalog` command emit →
`scanProjectDataCatalog` returns `linkage` → the `-datacatalog` record carries it → the Data tab renders
per-asset **"related data" chips** (each a one-click mine of the driver, rationale on hover). REMAINING:
add ETF proxy tickers (`SOXX`/`SMH`/`LIT`) to the catalog so semiconductor/lithium edges become mineable.

- **Model.** Nodes = mineable series; typed directed edges `{from, to, type, proxySymbol, rationale}`
  where `type ∈ {input-cost, supplier, competitor, sector-peer, macro-driver}`. Stored as a
  `{recordType}-linkage` record; rendered on the Data tab as per-asset "related data" chips.
- **Seed set (verified edges, proxies already mineable via D1/D2)** — banks (JPM) → curve slope
  `T10Y2Y` (net-interest-margin); USD/JPY → 10y `DGS10` (rate-differential carry); AUD/USD → copper
  `HG=F` (terms-of-trade commodity currency); USD/CAD → WTI `CL=F` inverse (oil→CAD); energy equities →
  WTI `CL=F`; airlines/transport → WTI `CL=F` inverse (fuel cost); gold miners → gold `GC=F`; gold →
  real yield `DFII10`; rate-sensitive equities (utilities/REITs/homebuilders) → `DGS10`; broad equities →
  Fed funds target `DFEDTARU` (steps at FOMC only); cyclicals → initial claims `ICSA` (growth nowcast);
  ag equities → corn/wheat `ZC=F`/`ZW=F`. Semiconductors (NVDA/AVGO/AAPL) → a `SOXX`/`SMH` proxy once
  ETF tickers are added to the catalog. Edges are **versioned** (`validFrom`/`validTo`) and the loader
  joins by TIMESTAMP (honouring each series' `barCloseTz`/release stamp), never by date string.
- **Growth.** Deep-research harness proposes new edges from an asset (supply-chain, ETF-holdings reverse
  lookup) → human-approve gate → the proxy series is added to the catalog and mined.

#### Phase D4 — Guided data discovery (north-star 1) — SHIPPED (research + proposals)

Problem statement + model goal → deep-research harness surveys what data exists → proposes candidate
data sources. Built + tested: `discoverData(problemStatement, goal)` (`ModelTrainerTools`, reuses the
injected `DeepResearchEngine`: `gather` → `extract` breadth → `coerceDataSourceCandidates`), persisting
each as a `{recordType}-datasource` draft (source/coverage/cost/licence/how-to-acquire + cited sources);
the `discover-data` activity (research lane, runs on `ModelSelection` — API or CLI); a Data-tab **Discover
form** (problem + goal → proposals rendered as cards). REMAINING: an approve gate that adds an approved
NEW source to the catalog registry + auto-launches its mine (today the proposal is a reviewable draft;
mining a source already IN the catalog works via D1).

#### Phase D5 — Extraction + cache + remote data path (deferred)

Only when a SECOND trainer needs the same data: extract catalog+miners+CLI to `thefactory-datamine`;
generalize `derive_cache` to a central 1m→fidelities service; wire the `ContentAddressedDataCache` +
remote-runner data path from one curated origin. `BlackSwanPriceEmitter`'s indicator engine stays a
runtime-formula REFERENCE, not a storage artifact.

**Providers (access / auth / cost / licence / point-in-time).**

| Provider | Access | Auth | Cost | Licence + PIT |
| --- | --- | --- | --- | --- |
| yfinance (Yahoo) | Python lib (existing miner) | none | free | personal/research, no redistribution; **not PIT** (latest marks, opaque continuous stitch) |
| Stooq | CSV per symbol | none | free | personal; longer history — cross-check source; not PIT |
| Frankfurter/ECB | REST no-key, 1 fix/biz day, 1999→ | none | free | **open licence** (only redistribution-safe series); EUR-base, one 16:00 CET fix |
| FRED / ALFRED | REST + `vintage_dates`/`realtime_*` + `releases/dates` | free key | free | non-commercial + attribution; **PIT via vintages** — the macro backbone |
| SEC EDGAR | REST `companyfacts` JSON | none (set UA) | free | public domain; **PIT via `filingDate`/`acceptanceDateTime`** |
| Dukascopy / Polygon (optional) | tick/agg FX | none / free key | free tier | personal; the only genuine FX **volume proxy** (tick counts) — high fetch cost / rate-capped |

**Correctness rules (leakage-first — enforce in the loader, not just docs).**

- **Store minimal raw only**, derive fidelities+indicators at runtime.
- **Join by TIMESTAMP, never by date string.** Each price row carries `barCloseTz`; a same-calendar-date
  cross-class join imports later-session info (FX 17:00 NY, commodity Globex, macro release times) — a
  1–7h look-ahead.
- **Macro is point-in-time.** ALFRED vintages (`output_type=4`), stamp at the per-release datetime
  (not a blanket 08:30), forward-fill to next release; MoM/YoY = diff of the SAME vintage.
- **Post-close series only next session.** `DGS10`/`T10Y2Y` (H.15 ~16:15 ET) publish after the equity
  close — forward-fill to next open, don't align to the session they price.
- **Fundamentals stamp at filing/acceptance**, not period-end; restatements are new dated rows.
- **Commodity continuous roll is a look-ahead machine** — back-adjust with recorded roll dates (or raw
  contracts); stamp at exchange settle, not 16:00 ET.
- **FX `Volume ≡ 0` is a constant, not data** — neutral-fill and audit every feature step so no FX row
  feeds a volume feature. Always use the 6-char `=X` form (short forms mix base/quote → inverted series).
- **Never forward-fill one leg of a ratio** (ECB triangulation compounds different holiday calendars →
  spurious holiday-correlated signal); reconcile calendars on the intersection before merging.
- **Idempotent + validated** mining (monotonic unique timestamps, positive prices, cross-month
  continuity, sane trading-day counts); split/dividend-adjust equities/ETFs.
- **Licence-gate any shareable output** — only Frankfurter/ECB is redistribution-safe; the rest are
  personal/research only.

### Code-change risk model — a third ML consumer (research first)

A trainer-conformant project scoring an agent's diff/PR by bug-likelihood (later reverted/fixed or
CI-failing) — a calibrated signal to gate review effort. Gathering labelled data is the hard part:

1. **Research first** (deep-research harness): survey public JIT-defect datasets (ApacheJIT, Defectors,
   JIT-Defect4J, ManySStuBs4J, CVEfixes, Big-Vul, Devign, CodeXGLUE) vs mining our own from the
   `thefactory-*` git histories. Output a cited report + go/no-go on a workspace-mined dataset.
2. **Data** (via the data mine): SZZ-style labeling over git histories; features from codeIntel
   (churn, complexity, coverage of touched code, diff size).
3. **Train**: a `risk-classifier` trainer project (sklearn/torch, objective = AUC or precision-at-k).
4. **Consume**: wire the score into the review / expert-panel / verifier path.

Depends on the data mine.

_Further out — **FastContext-style repository explorer** (a candidate fourth trainer-conformant consumer): a trained 4B–30B repo-exploration subagent, objective = file-recovery accuracy, trained via an SFT-bootstrap→task-grounded-RL ladder (reward broad-first-turn search / multi-turn evidence / precise citations). The reward-sweep + xAI config-effect engine is the right tool to tune the shaping. Gated by the same deep-research go/no-go and by LLM-training compute the current ComputeRunner has never exercised. Ref: microsoft/fastcontext (hf.co/papers/2606.14066), captured in thefactory-references._

### Optional phases

- **Live handoff.** On a winning run (the exploration autopilot's declared global max — see NEXT §7), tag the
  checkpoint for live trading (`run_server_model.py`). The autopilot supersedes the old "Phase 8 propose→run→judge"
  sketch; only the checkpoint→live-server tagging remains here.
- **Phase 9 — Jupyter notebooks (UNDERSCOPED).** View/edit/execute a project's `.ipynb` from the
  Overseer. To scope: render-vs-edit depth, where the kernel runs (host/sandbox/remote runner), how
  notebooks read campaign records/artifacts, security (arbitrary code → likely the sandbox profile).

### Small deferred items

- **Full RL resume** — per-episode RL checkpointing + `set_env` continuation for true mid-training
  resume (the regression line already resumes; RL restarts from zero). Revisit if worth the loop surgery.
- **Other single assets** — superseded by NEXT §5 (backfill latest + derive altcoin 1h/1d from 1m +
  per-asset walk-forward windows). NOT multi-asset portfolio — that is the project split above.
- **Runner-channel WebSocket upgrade** — job dispatch is already ~instant (long-poll `wake()`); a WS only
  shaves ~1.5s log-batch latency, invisible until a live-log UI consumes it.
- **Remote git repoRefs** — the engine emits local paths only; wire git refs + project bootstrap when a
  real remote machine needs it (BlackSwan local path covers today).

## Open questions (decide when the dependency lands)

- **Remote artifact/checkpoint storage** — keep-on-runner + reference vs upload; how a winning remote
  checkpoint reaches the live trading server. Has meaning only once remote runs **and** the live-trading
  handoff (Phase 8) both exist.
- **GPU + sandbox profile for training images** — `--read-only` rootfs vs ML caches; `--gpus` is wired
  but unexercised because the runner agent runs jobs directly (not through Docker-sandboxed
  `SandboxTools`). Revisit if/when training runs inside the sandbox image.
- **Judge/proposer model transport** — `ModelSelection` (API vs CLI), being overtaken by the in-flight
  `ModelSelection` refactor (ctx carries `model: ModelSelection` with a `cli` member). Revisit once the
  CLI inference stage lands — until then judge/propose run on API.
