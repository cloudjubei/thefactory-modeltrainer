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

- **Run the Exp campaigns.** Exp 6–15 on the 1h winner — walk-forward robustness, shorting, vol-target,
  equity-path rewards, fidelity stack, obs-squash, **Exp 12 exit-mechanic sweep (SL × TP × trailing)**,
  **Exp 13 the supervised baseline (logreg/gbm vs RL vs hold)**, **Exp 14 architecture A/B (LSTM vs
  attn-ppo vs tcn-ppo)**, **Exp 15 no-op/wrong-action × reward-scale sweep**. Read the cross-window OOS
  distributions (the By-dataset robustness table), then lock in the first config that clears the bar. (The
  supervised+rules baseline trades through the unchanged env, so its RunSummary is directly comparable.)
  Keep RL alive while the other approaches run in parallel.
  - **Exp 15 reads the new `blocked_signal_ratio` metric** (summary.py: share of the agent's buy/sell
    output that was a no-op it couldn't act on; ~0.95 on the current great-return runs). The sweep crosses
    `combo_noop_penalty` × `combo_sell` × `combo_wrongaction` to test that a penalty drives
    `blocked_signal_ratio` → 0 WITHOUT collapsing `traded_return`, and that it only bites as `combo_sell`
    shrinks (a ≤0.5 penalty is washed out at the baked `combo_sell=1000`). If even the favourable cells
    (low `combo_sell`, high penalty) show no suppression, raise `episodes` next — `episodes=1` may be too
    few to learn the penalty. NOTE: this only ever SILENCES out-of-position output (pushes it to hold); it
    cannot make a flat-state signal meaningful — that needs the position-blind objective (Deferred).
- **Wave 2 — replicate + falsify published methods under real costs. MEDIUM.** Pre-register the
  thesis that most papers omit fees and won't replicate — value is rigorous falsification plus the one
  or two that align with "direct/recurrent RL on a risk-adjusted utility". Each becomes a
  **Papers/Library card** (§2b) with claimed-vs-measured. Candidates: Moody & Saffell direct-recurrent
  (RecurrentPPO already imported + log-return reward; diff-Sharpe risk-adjustment and an `lstm_hidden_size`
  lever still need adding — LSTM hidden size is hardcoded per-variant today);
  Zhang/Zohren/Roberts vol-scaling (reuses the Wave-1 vol); a trend-following + mean-reversion exit
  overlay (would add `trend_filter`/`ma_period`/`reversion_threshold` levers + a forced exit); a
  supervised-LSTM direction filter gating RL entries (the supervised logreg/gbm line +
  `forward_horizon`/`prob_threshold` already exist). Every replication runs under walk-forward + 0.1%
  fees; keep only OOS-beat-hold survivors.
- **Wave 3 — multi-asset portfolio / cross-sectional long-short. PROJECT-SPLIT** (see Deferred).
- **Phase C exploration.** Huge sweep from the first OOS-validated baseline; `skipExplored` + by-setup
  aggregation self-prune; fold in **RB1b/2** (more indicators) + **RB5** (causal `dip_score` into the
  env) ONLY if an experiment says they help.

### 2. Model-trainer app

The **Hypotheses registry**, **Papers library**, and **Models catalog** are built — how they fit lives in
`docs/architecture.md` (Hypotheses = falsifiable claims runs prove/disprove; Papers = containers of
hypotheses; Models = the catalog aggregating runs + papers + hypotheses, with `scanProjectModels` /
`analyzePaperModels` and a Discuss-to-work chat). Pending:

- Open-ended `researchTrainingPapers` (discover N papers) + the heavy auto-seed/verify pipeline
  (find → web-verify → synthesize). Deferred.
- **Models catalog — live VISUAL pass in the Overseer.** The Models tab + the Papers "Find models" /
  add-missing flow are reference-clean and logic-tested but have NOT been eyeballed in the running app
  (parity is automatic — one embedded viewer across web/desktop/mobile — but the visual/device check is
  still open).
- **Per-model launch (optional).** A "Launch" affordance on a model card that prefills the Launch tab with
  `model_name` fixed — today runs launch from Launch/Hypotheses; the catalog focuses on cataloguing +
  agent work + chat.

### 2b. Models — remaining backlog

Still PENDING (the rest of the code/paper model catalog is already exposed):

- **EIIE (Jiang 2017) — deferred, blocked.** A multi-asset PORTFOLIO architecture (one weight-shared
  evaluator per asset → softmax allocation); it cannot be faithful in the single-asset, discrete-action env
  (it degenerates to a position sizer). Build the **Wave-3 multi-asset / portfolio env first**, then EIIE
  (and cross-sectional long-short) become possible. Catalogued `deferred`.
- **GRU recurrent core + SSM (S4D).** A custom `RecurrentActorCriticPolicy` subclass (recurrent state in the
  policy, not a features-extractor; expected ~wash vs LSTM) and an SSM falsification arm. Both low value.
- **Non-RL `model_type` paths still unexposed:** `technical` (`TechnicalStrategyModel`) and regression `mlp`
  (`regression` is missing from the documented `model_type` choices despite `create_regression_model`
  existing). A `model_type`-lever change (the `time`/`momentum`/`hodl`/`supervised` non-RL paths are wired).

**Reusable components (surface as `component` catalog entries / building blocks) — low priority:**

- Features-extractors: `SequenceFeaturesExtractor` (attn/tcn, `custom/sequence_extractor.py`), `LSTMFCE`
  (`dqn_lstm_policy.py`), `CustomMlpExtractor` (`custom/custommlpextractor.py`).
- Custom policies / Q-nets: `custom/policies.py` (ActorCritic / RecurrentActorCritic / DQN / Dueling /
  Rainbow / QRDQN), `custom/policy_iqn.py` (`CustomIQNPolicy`), `custom/customqnetwork.py`
  (`create_mlp_custom`), `dueling_dqn/policies.py`.
- Replay buffers: `rainbow_dqn/prioritized_replay_buffer.py`, `rainbow/replay_buffer.py` (+ `segment_tree.py`),
  `custom/agent57/agent57.py` (`CustomAgent57ReplayBuffer` / `EpisodicMemory`).
- Optimizer: `custom/dgwo.py` (`DGWO`, grey-wolf). NN blocks: `custom/attention.py`, `custom/noisylinear.py`,
  `custom/dropconnect.py`, `custom/residualblock.py`, `custom/denseblock.py`, `custom/memorymodels.py`.

### 3. xAI — explain WHY the model acted (parallel track)

The xAI track — decision-trace spine + the full xAI tab (Phases 1–5) — is shipped (git +
`docs/architecture.md`); the model-trainer side stays domain-oblivious. Pending:

- **Live VISUAL pass in the Overseer.** The whole xAI viewer is engine-parity-tested + syntax/reference
  clean but has NOT been eyeballed in the running app — the one open verification.
- **Config-space exploration — reward/metric NORMALISATION (the one pending follow-on).** The
  surrogate + EI acquisition (`acquisitionRecommendations`) + fANOVA/Sobol importance + lever-coupling +
  PCA-projection stack is shipped (git + `docs/architecture.md`); its visual check folds into the VISUAL
  pass above. Remaining: reward/metric NORMALISATION so cross-run/cross-setup comparison isn't distorted
  by raw scale (which quantities, against what baseline) — fold into the surrogate/criterion layer,
  deterministic + parity-mirrored like the rest of the engine. (Cited deep-research report pending.)
- **Deeper attribution (parked — lower value / heavier).** TabularSHAP/DeepSHAP (likely fails the
  sanity-check like IG — input-magnitude-dominated — + needs a tree-surrogate dep); attention-weight viz
  (attn-ppo only); generative counterfactual states (needs a GAN).
- **Data-influence follow-ons (parked).** Per-step group-saliency + a mid-training-checkpoint trace —
  both reuse the shipped `DecisionTraceDiff` spine but need a NEW BlackSwan emission first.
- **PARKED — step-by-step ANIMATION replay** + scrubber. No trace-artifact change needed.

---

## Deferred — bigger work, picked up after the active work

### Cross-asset robustness testing

The windowing dimension is covered by the walk-forward windows (named slices) + the By-dataset
robustness view. What remains is the **cross-asset** dimension — test a trained checkpoint against any
asset in the same data format to catch regime/asset overfit:

- The trainer already replays a checkpoint deterministically; the missing piece is selecting the data
  WINDOW/asset (a `testSet` param: asset + time-range, or a named curated slice).
- Likely a manifest `testSets` list (`{id, asset, range/description}`) + a generic "test on set"
  activity writing a `<recordType>-regimetest` record per (run, set), surfaced as a per-set matrix in
  run-detail + a compare overlay. Keep it generic; BlackSwan is the first consumer.

### Environments — follow-ups

The Environments tab + `scope: 'environment'` lever bundles ship; presets can now carry `environments`
bundles (the SL/TP/trailing sweep uses them, rendered read-only in the launch picker so the swept
profiles are visible) and the top-3 best-run presets reproduce a run's env values. Remaining: a unified
**test matrix** surface (model × environment × dataset, folding in cross-asset testing); the fixed-only
clone-to-launch path (`applyPresetFixed`) still pins model levers only; an optional "save this preset's
profile as a named environment" promote affordance (deferred — these are git-authored experiment
constants, not user-tuned regimes).

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
