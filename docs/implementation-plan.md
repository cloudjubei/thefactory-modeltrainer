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

1. **Standardised scoring test sets.** Manifest `testSets` list (`{id, asset, timeframe, range}`) —
   pulled forward from the deferred cross-asset section. A generic "evaluate on test set" activity
   replays a checkpoint on each named set and writes a `<recordType>-settest` record per (run, set);
   surfaced as a per-set matrix in run detail + a compare overlay. Same test window ⇒ comparable.
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

### The data mine — a shared dataset project for every model trainer

A standalone repo (`thefactory-datamine`) that is the **source of truth for training data**: gather raw
data, clean + validate + normalise, publish versioned/reproducible. Trainers declare which prepared
dataset(s) they need (the manifest's `data[]`); the data mine + content-addressed cache deliver them.
Architecture decision: store **MINIMAL raw OHLCV only** — indicators + higher fidelities are derived AT
RUNTIME in the consumer, so storage stays small and an indicator fix is a one-line code change. The
basis exists (`BlackSwanPriceEmitter`: Binance miner + indicator engine, ~80% the right shape; the
indicator engine is now a REFERENCE for the runtime formulas, not a storage artifact). Remaining job:
(a) gather + clean raw OHLCV (gap/dedup/continuity checks, NaN sanitisation, mine missing intervals);
(b) generalise the `derive_cache` (1m canonical → derive+cache fidelities centrally); (c) the
content-addressed cache + remote-runner data path from one curated origin.

**Guided data discovery (north-star 1).** Take a problem statement + model goal → run deep research on
what data exists (sources/APIs/coverage/cost/licence/granularity, reuse the deep-research harness) →
propose candidate datasets + a mining plan + trade-offs → hand off to gather→clean→cache. Output a
cited report + an approved mining plan.

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
