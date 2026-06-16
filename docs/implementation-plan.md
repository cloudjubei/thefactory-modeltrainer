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

### 2. BlackSwan Phase B — find ONE setup that trades well (research-driven)

Phase-A correctness is GREEN. Deep research + many failed campaigns say: **no result is trustworthy
until out-of-sample evaluation is honest**, and end-to-end value-based deep RL on a single split is
the cautioned-against path. So Phase B runs as waves — **keep and broaden single-asset RL while
running other approaches in parallel** (never stop RL: an unpublished edge is plausible). A setup
counts only when it **beats buy-and-hold out-of-sample on risk-adjusted terms, net of 0.1% fees**
(`sharpe_alpha > 0`, not just `traded_return`). Effort tags: **EASY / HARD / PROJECT-SPLIT**.

**Wave 0 — the measuring stick (do first; blocks trust in every later result). MEDIUM.** The
hardcoded 2020–23 / 2024 single split is the overfit trap that makes the +60% sweep results
untrustworthy. Walk-forward / purged-CV evaluation + cost-honest reporting:
- `walk_forward_preset` lever (none/monthly/quarterly/biannual) → rolling `(train,test)` windows +
  embargo in `config_builder.py`; `build_data_config` takes a `window_id` instead of the hardcoded
  `_TRAIN_PAIRS`/`_TEST_PAIRS`; window identity threaded through `run.py` `_run_one` into `env_test`.
- `summary.py`: per-window Sharpe/Cagr/maxDD nested under a `windows` key + top-level `sharpe_alpha`
  (strategy − hold Sharpe; benchmark prices already computed) + `realized_cost_bps`
  (`(initial−final)/initial·1e4`; fee already in `net_worth`, no double-count).
- `trainer.json` pipelineVersion → 3 (changelog: rolling-window runs are NOT comparable to the old
  single split) + a "quarterly walk-forward (4 seeds)" preset. Embargo may defer to 3.1.
- TDD the window/embargo date logic (`test_start > train_end + embargo`, gap-year edges); a hodl
  baseline must reproduce matching per-window Sharpe. Model-trainer matrix needs no change (a
  choice lever auto-expands the campaign); the RESULTS surface does — see §3a.

**Wave 1 — cheap high-payoff RL wins (parallel, all swept under Wave 0).**
- **Shorting — HARD (single-asset env tweak, NOT a split).** `allow_shorting` + `max_short_size` in
  `env_config.py` (mirrors the existing `no_sell_action` lever); `Discrete(5)`
  (hold/buy/close-long/short/cover); signed `positions`; fix the ~6 long-only `position > 0` branches
  (`base_crypto_env.py` lines 204/310/321/431/455/515) to use SIGN; mirror `resolve_tpsl` +
  `update_position_prices` for shorts (track the low while short); generalise trade-tracking to
  entry/exit-with-direction. Long-only literally cannot express the dominant 2024 down-signal. Risk =
  silent P&L sign bug → comprehensive grep + per-reward-model tests.
- **Vol-targeted position sizing — MEDIUM, high payoff.** Scale the notional INSIDE `take_action` by
  `min(1, kelly_fraction / realized_vol)` floored at a min (keep the Discrete head — a Box action
  space forces a from-scratch SB3 retrain). Lookahead-free rolling vol (≤ the 1h lookback of 32).
  `leverage_target` lever [0.5,1.0,1.5]. The structural fix for "60 trades for +1%".
- **Drop/soften the trade-count gate — EASY.** `trade_gate_mode` lever (none/quadratic/linear/
  threshold) routed in `summary.py` (post-run only — zero training-loop impact). Realistic fees
  already regulate churn; keep sweepable so both regimes stay comparable. (Subsumes the old
  "calibrate `MIN_TRADES_FOR_FULL_CREDIT`" item.)
- **Broaden the space — EASY.** `lookback_window` [1,8,16,32,64]; regime/vol features (rolling-vol
  bin, trend slope) in `_add_curated_indicators`; a fixed-fraction `position_size` lever; two new
  reward models (`profit_percentage_direct` raw-return, an inline differential-Sharpe approximation);
  expose `exploration_fraction`/`exploration_final_eps`/`episodes` (already in `ModelRLConfigSearch`,
  just unwired); Exp6–11 presets on the 1h winners. Also fold in the deferred **out-of-[-1,1]
  normalization experiment** (tanh-squash the 8 over-bound base features / clip the obs — does it
  help?) as one such lever. Choice levers auto-expand the matrix — no model-trainer change.
- **Multi-fidelity stacking, made VISIBLE — MEDIUM.** The observation already stacks timeframes
  (`MultiTimelineDataProvider` over a `layers` list — this IS the user's 1h vs 1h+1d vs 1h+1d+1w
  experiments, and more layers empirically helped), but the only lever is `timeframe` (1h hardcodes
  `layers=[1h,1d]` in `config_builder.py`) and `summary.py` never records which layers ran — so it is
  invisible. Add a `fidelity_set` CHOICE lever (`[1h]`/`[1h+1d]`/`[1h+1d+1w]`/`[1d]`) mapped in
  `build_data_config`, and emit `fidelity_set` + `layers` in the summary as a generic metadata field
  (domain-oblivious pass-through) so a run is self-documenting and the app can show a "fidelity"
  column/badge in Runs + run-detail. Gotcha: `derive_cache` only buckets 1h/1d from the 1m canonical
  — add WEEKLY bucketing before `1w` is a real choice. Wider sets ≈ quadratic obs growth → pair with
  a `net_arch`/`batch_size` note in the preset.

**Wave 1-parallel — the decisive non-RL baseline. EASY.** Supervised direction/return predictor +
rules-based execution. The trainer contract is model-agnostic (hodl/regression/technical already
coexist), so a new `supervised_rules` model_type is ~200 lines, additive, touches nothing in the RL
track. Predictor fits on TRAIN-ONLY backward-looking labels (audit `get_signal_buy_profitable` for
lookahead — the #1 failure mode); a deterministic `RulesExecutor` applies vol-targeted sizing + a
cost-aware threshold (trade only if `|edge| > fee`) → env actions 0/1/2. Answers "is supervision
beating RL on this env?"

**Wave 2 — replicate + falsify published methods under real costs. MEDIUM.** Pre-register the thesis
that most papers omit fees and won't replicate — the value is rigorous falsification plus the one or
two that align with the research-favoured "direct/recurrent RL on a risk-adjusted utility" direction.
Each becomes a **Papers/Library card** (§3b) with claimed-vs-measured.
- **Moody & Saffell direct-recurrent — EASY (~2–3d).** RecurrentPPO (already imported) + reward =
  log-return or differential Sharpe; `lstm_hidden_size` lever. Tests the favoured kind of RL vs the
  current value-based DQN.
- **Zhang/Zohren/Roberts vol-scaling — EASY (reuses Wave-1 vol).**
- **Trend-following + mean-reversion exit overlay — EASY (~1–2d).** `trend_filter` + `ma_period` +
  `reversion_threshold` levers; forced exit in `resolve_tpsl`. Pure logic, no model change.
- **Supervised-LSTM direction filter — MEDIUM (~5–6d).** BCE next-bar-direction on a STRICT separate
  train split; gates RL entries.
- Every replication runs under Wave-0 walk-forward + 0.1% fees; keep only OOS-beat-hold survivors.

**Wave 3 — multi-asset portfolio / cross-sectional long-short. PROJECT-SPLIT.** The one genuine split
(see Deferred). Reuses BlackSwan rewards/features/algos + the walk-forward harness, ships separately,
blocks nothing.

Phase-C exploration (huge sweep from the first OOS-validated baseline; `skipExplored` + by-setup
aggregation self-prune; fold in **RB1b/2** more indicators + **RB5** causal `dip_score` only if an
experiment says they help) continues to run continuously once a Wave-0-validated baseline exists.

### 3. Model-trainer app — surface the new results + a Papers/Library tab

**(3a) Walk-forward / multi-window results surfacing — MEDIUM.** Today `TrainingRunSummary` carries
one flat metric set and the viewer shows one number per run; walk-forward emits a DISTRIBUTION. Stay
domain-oblivious (no trading vocabulary in core types — `windows`/`sharpe_alpha`/`realized_cost_bps`
are opaque fields the viewer interprets):
- Types: optional `windows: Array<{ name?: string; metrics: Record<string, number> }>` on
  `TrainingRunSummary` + pass-through top-level scalars. Single-window runs unchanged (optional).
- Viewer: synthetic aggregate columns (mean/min/max/std across windows) with a "W" badge; a
  run-detail "Windows" sub-table + per-metric sparkline/whisker; compare overlays per-window curves;
  a **banner + filter when single- and multi-window runs are mixed** ("scored under different eval
  strategies — not directly comparable, even at the same pipelineVersion"); aggregate only
  same-eval-strategy runs per setup. The user's UX bar: show ALL windows + flag the
  not-comparable warning, newcomer-legible. (Subsumes the deferred regime-slice testing — windows
  ARE named slices.)

**(3b) Papers / Library tab — HARD (mostly surface area; reuses Environments CRUD + Hypotheses
linking + clone-to-launch).** A roster of approach cards turning "try every positive paper, prove it
good or fluff" into a durable, explorable, evidence-backed registry. Generic ("an approach with a
source + a claim"); BlackSwan's first consumers = the Wave-2 papers.
- Data model: a `<recordType>-paper` data record →
  `{ id, title, url, authors, year, claim, claimedMetrics?, assumptions (the fine print — fees?
  gross/net? retrain cadence? frictionless? multi-asset?), approach, replicateConfig? (a partial
  launch config / lever preset), status: 'untested'|'replicating'|'holds-up'|'fluff',
  linkedRunKeys?/campaignActivityId?, measuredSummary?, verdictNote, source: 'manual'|'research',
  tags?, updatedAt }`. Type `TrainingPaperRecord` in `modelTrainerTypes.ts`.
- CRUD mirrors Environments: `readPapers`/`putPaper`/`deletePaperRecord` + `setupPapers`/
  `renderPapers`/`paperFormHtml`/`togglePaperForm`/`onSavePaper`/`onDeletePaper`.
- UI: cards with title→url, authors/year, a colour-coded verdict badge, the claim, assumption chips,
  **claimed-vs-measured side by side** (measured read from `linkedRunKeys`), a verdict filter;
  per-card actions: open link, **Replicate** (prefill the Launch form from `replicateConfig` via the
  existing clone-to-launch path → switch to Launch), **Link runs/campaign**, **Record verdict**
  (status + note; auto-suggest holds-up/fluff from whether measured beats hold OOS, user confirms),
  Edit/Delete. index.html tab + section; style.css cards/badges.
- **Research-seeded**: extend the existing propose/research flow so a result can
  `putPaper(… source:'research')` — closes research → experiment → verdict. A paper only earns
  ✅ holds-up if it survives §3a walk-forward + real costs (honest by construction).

### 4. xAI — explain WHY the model acted (parallel track, like Papers)

A decision drill-down: understand what the model did and why. Lands additively across BlackSwan +
model-trainer (NOT a new project); the model-trainer side stays domain-oblivious (generic "decision
trace" / "step log", arbitrary action strings, no trading vocabulary). Effort overall **HARD** (sum
of cheap parts), with the animation capstone parked.

**The anomaly, explained (document it — 0 code).** "Many buys, few sells" is BY DESIGN: the profitable
presets set `no_sell_action=True`, shrinking the action space to `Discrete(2)`=[hold,buy]
(`trade_all_crypto_env.py:10-12`) — the agent has no sell action, so it only times ENTRIES and ALL
exits are TP/SL-forced (`base_crypto_env.py:283-297`). The env already records exit mode (`tpsls`,
`trades_won/lost`). The Explain view should make this obvious instead of looking like a glitch.

- **Decision-trace emission (BlackSwan) — EASY.** The env already retains per-step arrays (`actions`,
  `actions_made`, `forced_actions`, `tpsls`, `positions`, `balances`, `net_worths`,
  `rewards_history`). A `trainer/decision_trace.py` emits them as `{summary}.traces.jsonl`, hooked
  after `model.test()` in `run.py`; capture model confidence from `.predict()`'s second return (DQN
  Q-values; PPO/TRPO logits via `policy.get_distribution`). Store the per-step obs (1 line) to enable
  attribution.
- **Generic trace artifact + Explain view (model-trainer) — MEDIUM.** Types in `modelTrainerTypes.ts`:
  `DecisionStep {step, action, confidence?, features?, state?, alternativeAction?}` +
  `DecisionTrace {steps, featureAttribution?, actionCounts?}` on `artifacts.decisionTrace`, with a
  soft `validateDecisionTrace` (missing trace ≠ error). An "Explain" sub-view: a decision TIMELINE
  aligned to the existing Price&actions / Equity-vs-hold charts (shared step axis); an
  ACTION-DISTRIBUTION diagnostic that flags anomalies generically ("buy ≫ sell", "TP ≫ SL"). Reuses
  the existing chart + record plumbing; renders arbitrary action strings (domain-oblivious).
- **Feature attribution — MEDIUM.** Gradient saliency on the torch policy (backprop the chosen
  action's Q-value/logit w.r.t. the observation), selective (only when `actions_made`; skip
  TP/SL-forced steps), ~2–3× test time; aggregate saliency by fidelity layer to keep the heatmap
  legible. Permutation/SHAP deferred (expensive).
- **Discuss-with-agent — EASY.** Enrich the existing `chatAboutRun`/`discussTopic` seed with a trace
  SUMMARY (action counts, top features) so "why so few sells?" has context.
- **PARKED — step-by-step ANIMATION replay** + scrubber, discussed live with an agent. Added last; no
  trace-artifact change needed.

---

## Deferred — bigger work, picked up after the active work

### Multi-dataset / cross-asset robustness testing (replaces Eval for RL)

RL models are validated on a live environment/market, so the held-out "Eval" was removed (gated on a
manifest `evaluate` command; BlackSwan declares none). The **windowing mechanism is now Phase-B Wave 0
+ §3a** (walk-forward windows ARE named slices, surfaced as a per-run distribution); what remains here
is the **cross-asset** dimension — test a trained checkpoint against named **regime slices** and any
asset in the same data format to catch regime-overfit. Sketch:

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

### Multi-asset portfolio / cross-sectional long-short — a SEPARATE project (Phase-B Wave 3)

The one genuine project split. BlackSwan's single-asset env hardcodes one asset everywhere — 1D
state, single-position discrete action, single-symbol data paths, single equity-curve objective — so
multi-asset needs a fundamentally different env, not an in-place tweak: a **3D observation**
(`asset × lookback × features`), a **portfolio action space** (per-asset long/short/weight), a
**timestamp-aligning N-symbol data provider** (misalignment = silent P&L corruption — unit-test
against a 3-coin × 100-bar fixture), and a **rebalance-count or direct Sharpe/Calmar objective with a
correlation penalty**. It REUSES BlackSwan's reward components, feature engineering, SB3 algos, and
the walk-forward harness, but ships as its own ~3–4 week project and blocks none of the in-place
wins. Hard dependency: only BTCUSDT klines are on disk → altcoin backfill (via the data mine) is a
prerequisite. Research calls cross-sectional long-short the strongest-edge config, so it is promising
— deliberately sequenced last. (Distinct from "Other single assets" above, which is just the `asset`
lever + a backfill.)

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
- **Other single assets** — running the SAME single-asset model on ETH/SOL/etc. is just the `asset`
  lever + an altcoin 1d/1h backfill (lever + per-symbol globs + `data_inventory` gating already
  correct, no code change). This is NOT multi-asset portfolio — that is the project split below.
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
