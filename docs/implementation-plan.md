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

| Repo                                    | Owns                                                                                                                            |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **thefactory-modeltrainer** (this repo) | `ModelTrainerTools`; matrix planner; campaign loop; judge/propose orchestration; the viewer; the standard + `examples/`.       |
| **thefactory-tools**                    | Generic infra only: `ComputeRunner` seam (+ future `RemoteComputeRunner`, `ContentAddressedDataCache`, pairing); work-item engine. |
| **thefactory-backend**                  | Activity registration + composition; app-view serving; future PIN-pairing endpoints + runner WS channel.                       |
| **clients**                             | Future Compute Runners settings/pairing screen (native, cross-project).                                                        |
| **the runner agent**                    | Future Docker-packaged connect-out program.                                                                                    |
| **BlackSwan** (the trading repo)        | Its `TrainerManifest` + additive `trainer/` CLI conformance. No Overseer code.                                                 |

---

## NEXT — the active work

### 1. BlackSwan Phase B — find ONE setup that trades well

The OOS-honest lever matrix is in place: walk-forward windows (2022/2023/2024), shorting, vol-targeted
sizing, the `trade_gate_mode`, multi-fidelity (`fidelity_set`), direct + differential-Sharpe rewards,
regime/vol features, and the `obs_squash` normalization experiment. A setup counts only when it **beats
buy-and-hold out-of-sample net of 0.1% fees**, with profit that is NOT concentrated in up-regimes
(genuine timing, not beta), stable across seeds AND windows.

- **Run the Exp campaigns.** Exp 6–11 (walk-forward robustness, shorting, vol-target, equity-path
  rewards, fidelity stack, obs-squash) on the 1h winner; read the cross-window OOS distributions, then
  lock in the first config that clears the bar. Keep RL alive while running the other approaches in
  parallel (below) — an unpublished edge is plausible.
- **Wave 1-parallel — the decisive non-RL baseline. EASY.** A supervised direction/return predictor +
  rules-based execution. The trainer contract is model-agnostic (hodl/regression/technical coexist),
  so a new `supervised_rules` model_type is ~200 lines, additive, touches nothing in the RL track:
  predictor fits on TRAIN-ONLY backward-looking labels (audit `get_signal_buy_profitable` for
  lookahead — the #1 failure mode); a deterministic `RulesExecutor` applies vol-targeted sizing + a
  cost-aware threshold (trade only if `|edge| > fee`) → env actions 0/1/2. Answers "is supervision
  beating RL on this env?"
- **Wave 2 — replicate + falsify published methods under real costs. MEDIUM.** Pre-register the
  thesis that most papers omit fees and won't replicate — value is rigorous falsification plus the one
  or two that align with "direct/recurrent RL on a risk-adjusted utility". Each becomes a
  **Papers/Library card** (§2b) with claimed-vs-measured. Candidates: Moody & Saffell direct-recurrent
  (RecurrentPPO already imported + log-return/diff-Sharpe reward + `lstm_hidden_size` lever);
  Zhang/Zohren/Roberts vol-scaling (reuses the Wave-1 vol); a trend-following + mean-reversion exit
  overlay (`trend_filter`/`ma_period`/`reversion_threshold` levers, forced exit in `resolve_tpsl`); a
  supervised-LSTM direction filter gating RL entries. Every replication runs under walk-forward + 0.1%
  fees; keep only OOS-beat-hold survivors.
- **Wave 3 — multi-asset portfolio / cross-sectional long-short. PROJECT-SPLIT** (see Deferred).
- **Phase C exploration.** Huge sweep from the first OOS-validated baseline; `skipExplored` + by-setup
  aggregation self-prune; fold in **RB1b/2** (more indicators) + **RB5** (causal `dip_score` into the
  env) ONLY if an experiment says they help.

### 2. Model-trainer app

**(2a) Walk-forward by-window view — MOSTLY DONE; minor polish left.** The behavioural surfacing
shipped: run-detail renders the exit-reason breakdown, per-regime (windowed + trend) skill-vs-luck
attribution, the trade ledger, and `return_vs_hold_pct` + `realized_cost_bps` in the metrics table;
the By-dataset view groups runs by `walk_forward_window` (the by-window distribution) and run-detail
shows `fidelity_set`. LEFT (optional polish): an explicit cross-window aggregate (mean / worst-window
`return_vs_hold_pct` per config) rather than the generic objective avg/min/max the By-dataset table
shows; and a clearer mixed single/multi-window not-comparable banner.

**(2b) Papers / Library tab — HARD (mostly surface area; reuses Environments CRUD + Hypotheses
linking + clone-to-launch).** A roster of approach cards turning "try every positive paper, prove it
good or fluff" into a durable, explorable, evidence-backed registry. Generic ("an approach with a
source + a claim"); BlackSwan's first consumers = the Wave-2 papers.
- Data model: a `<recordType>-paper` data record →
  `{ id, title, url, authors, year, claim, claimedMetrics?, assumptions (fees? gross/net? retrain
  cadence? frictionless? multi-asset?), approach, replicateConfig? (a partial launch config / lever
  preset), status: 'untested'|'replicating'|'holds-up'|'fluff', linkedRunKeys?/campaignActivityId?,
  measuredSummary?, verdictNote, source: 'manual'|'research', tags?, updatedAt }`. Type
  `TrainingPaperRecord` in `modelTrainerTypes.ts`.
- CRUD mirrors Environments: `readPapers`/`putPaper`/`deletePaperRecord` + `setupPapers`/
  `renderPapers`/`paperFormHtml`/`togglePaperForm`/`onSavePaper`/`onDeletePaper`.
- UI: cards with title→url, authors/year, a colour-coded verdict badge, the claim, assumption chips,
  **claimed-vs-measured side by side** (measured read from `linkedRunKeys`), a verdict filter;
  per-card actions: open link, **Replicate** (prefill the Launch form from `replicateConfig` via the
  existing clone-to-launch path → switch to Launch), **Link runs/campaign**, **Record verdict**
  (status + note; auto-suggest holds-up/fluff from whether measured beats hold OOS, user confirms),
  Edit/Delete. index.html tab + section; style.css cards/badges.
- **Research-seeded**: extend the propose/research flow so a result can `putPaper(… source:'research')`
  — closes research → experiment → verdict. A paper earns ✅ holds-up only if it survives §2a
  walk-forward + real costs (honest by construction).

### 3c. Models / Architectures library (like Papers, for model build-ups) — HARD

We've never looked under BlackSwan's hood — what `reppo-custom` / `trpo-custom` / the dueling /
Munchausen / LSTM variants actually compose, and how `net_arch` / `activation` / `optimizer` shape
them. A registry parallel to §3b Papers, generic (BlackSwan first consumer):
- **The registry/tab** (mirror Papers): each card = a model architecture — its build-up (algo + net
  shape + policy internals + any custom head), provenance/rationale, and a verdict
  proven/disproved by linked runs (claimed-vs-measured). `<recordType>-model` records;
  Replicate→Launch prefills `model_name`/`net_arch`/…; by-model grouping reuses the model-lever
  signature; research/propose can seed cards. Mirrors the Papers plumbing.
- **The research + build direction** (the LLM-advances angle): survey modern sequence architectures
  (attention/transformers, longer-context recurrent) and add the promising ones as new `model_name`s
  in BlackSwan's `model_factory`, each documented as a card and proven/disproved by sweeps. Real ML
  work; sequenced after the registry exists.

### 3d. Presets: clean up + SL/TP/trailing sweep + top-3 auto-include — EASY/MEDIUM

The preset list has grown noisy (each run many times; unclear which are good now).
- **Clean up** the `trainer.json` presets — drop stale/duplicate, keep a curated set.
- **Add a strong SL+TP+trailing-stop SWEEP preset** — the exit-mechanic combinations are
  under-explored and make a good sweep starting point.
- **Auto-include the top-3 results as presets**: the viewer reads the best runs (by objective, OOS)
  and offers them as one-click "clone-to-sweep" seeds alongside the manifest's presets, so homing in
  on a winner is point-and-click. (clean-up + SL/TP/trailing = quick `trainer.json`; top-3 = a viewer
  feature synthesizing presets from the best runs.)

### 3. xAI — explain WHY the model acted (parallel track, like Papers)

A decision drill-down: understand what the model did and why. Lands additively across BlackSwan +
model-trainer (NOT a new project); the model-trainer side stays domain-oblivious (generic "decision
trace" / "step log", arbitrary action strings, no trading vocabulary). Effort overall **HARD** (sum of
cheap parts), with the animation capstone parked.

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
  the existing chart + record plumbing; renders arbitrary action strings. (The "many buys, few sells"
  case is already explained in BlackSwan's manifest description — the view should make it self-evident
  from the trace.)
- **The sparse-sell deep-dive (the live symptom).** Even WITH a sell action, several profitable runs
  fire many BUYs but rarely SELL, so they trade infrequently. The Explain view must show WHY the sell
  action stays dormant — per-step sell-Q vs hold-Q (or sell-logit vs hold), and the action
  distribution over time: is hold's value persistently higher, is the sell signal never learned, or
  is it crowded out by TP/SL exits? A concrete diagnostic, not just a count — this is the first
  question the decision trace + confidence capture should answer.
- **Feature attribution — MEDIUM.** Gradient saliency on the torch policy (backprop the chosen
  action's Q-value/logit w.r.t. the observation), selective (only when `actions_made`; skip
  TP/SL-forced steps), ~2–3× test time; aggregate saliency by fidelity layer. Permutation/SHAP
  deferred (expensive).
- **Data-influence on decisions/weights — HARD, research.** See how a NEW piece of information changes
  the model — its weights AND its subsequent decisions — and whether that change is GOOD. The goal: judge
  a tweak as positive from the DECISION deltas even when the final return isn't there yet (so we can
  steer by "the decisions improved" not just "the score rose"). Approaches to evaluate: counterfactual
  decision-trace diffing (re-run the test with/without a feature → per-step decision + P&L delta),
  attribution of a decision to the new input, or a before/after decision-diff around a fine-tune. Builds
  on the base decision trace + attribution; parked behind them.
- **Discuss-with-agent — EASY.** Enrich the `chatAboutRun` system prompt with a trace SUMMARY (action
  counts, top features) so "why so few sells?" has context.
- **PARKED — step-by-step ANIMATION replay** + scrubber, discussed live with an agent. Added last; no
  trace-artifact change needed.

---

## Deferred — bigger work, picked up after the active work

### Cross-asset robustness testing

The windowing dimension is now Phase-B Wave 0 + §2a (walk-forward windows are named slices). What
remains is the **cross-asset** dimension — test a trained checkpoint against any asset in the same
data format to catch regime/asset overfit:
- The trainer already replays a checkpoint deterministically; the missing piece is selecting the data
  WINDOW/asset (a `testSet` param: asset + time-range, or a named curated slice).
- Likely a manifest `testSets` list (`{id, asset, range/description}`) + a generic "test on set"
  activity writing a `<recordType>-regimetest` record per (run, set), surfaced as a per-set matrix in
  run-detail + a compare overlay. Keep it generic; BlackSwan is the first consumer.

### Daily-step multi-fidelity (provider enhancement)

`timeframe` (the agent's STEP) and `fidelity_set` (the LAYERS observed) are now decoupled + validated:
an hourly step can observe coarser layers (the new `1d+1w` set), and incompatible combos fail fast.
STILL UNSUPPORTED (fails fast for now): a DAILY step observing FINER layers (e.g. a 1d-step agent
seeing 1h+1d) — `MultiTimelineDataProvider` has no `divider_run` mapping for `input=1h → run=1d`.
Enable by adding that mapping + verifying the step/aggregation against a fixture + a live run. Then
the full step × layers matrix is open.

### Environments — follow-ups

The Environments tab + `scope: 'environment'` lever bundles ship. Remaining: a unified **test matrix**
surface (model × environment × dataset, folding in cross-asset testing); presets/clone currently
ignore env-lever values (env values come from the picker).

### Activity concurrency — server-side pass

The viewer-only activity-count budget + client-driven stalled-run resume ship. The rest wants a
host-aware, browsable, concurrency-capped center:
- **Per-activityId progress/campaign records.** Key the `-progress`/`-campaign` records by activityId
  (not `'latest'`) so concurrent same-project campaigns each show their own live progress.
- **Host-enforced global RUN cap.** A backend semaphore on `LocalComputeRunner.runJob` (the shared
  `trainerLocalRunner`) so total concurrent training runs are bounded regardless of live-campaign
  count — the durable resource guard the viewer-only budget doesn't provide.
- **Server-side queue drain.** Chain the next queued activity on the backend when one settles, so
  follow-ups advance while the app/viewer is closed.
- **Boot-scan resume.** A server-side boot scan that auto-reclaims/relaunches resumable runs with no
  client present (the durable version of the client-driven resume).
- **App-nav unseen-results badge.** HOST-DERIVED (count trainer activities `finishedAt` since the user
  last opened the app tab → Sidebar app-tab badge, web + mobile).
- **Run→Activity link.** Needs a browsable per-activity history first; then `activityId`-tag
  eval/verdict records and link into it.

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
  correct, no code change). NOT multi-asset portfolio — that is the project split above.
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
  handoff (Phase 8) both exist.
- **GPU + sandbox profile for training images** — `--read-only` rootfs vs ML caches; `--gpus` is wired
  but unexercised because the runner agent runs jobs directly (not through Docker-sandboxed
  `SandboxTools`). Revisit if/when training runs inside the sandbox image.
- **Judge/proposer model transport** — `ModelSelection` (API vs CLI), being overtaken by the in-flight
  `ModelSelection` refactor (ctx carries `model: ModelSelection` with a `cli` member). Revisit once the
  CLI inference stage lands — until then judge/propose run on API.
