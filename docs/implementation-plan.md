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
sizing, the `trade_gate_mode`, multi-fidelity (`fidelity_set`), direct + differential-Sharpe rewards,
regime/vol features, and the `obs_squash` normalization experiment. A setup counts only when it **beats
buy-and-hold out-of-sample net of 0.1% fees**, with profit that is NOT concentrated in up-regimes
(genuine timing, not beta), stable across seeds AND windows.

- **Run the Exp campaigns.** Exp 6–15 on the 1h winner — walk-forward robustness, shorting, vol-target,
  equity-path rewards, fidelity stack, obs-squash, **Exp 12 exit-mechanic sweep (SL × TP × trailing)**,
  **Exp 13 the supervised baseline (logreg/gbm vs RL vs hold)**, **Exp 14 architecture A/B (LSTM vs
  attn-ppo vs tcn-ppo)**, **Exp 15 no-op-penalty sweep**. Read the cross-window OOS distributions (the
  By-dataset robustness table), then lock in the first config that clears the bar. (The supervised+rules
  baseline trades through the unchanged env, so its RunSummary is directly comparable.) Keep RL alive
  while the other approaches run in parallel.
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

**Unified Hypotheses registry** — shipped. Models + Papers + Hypotheses collapsed into ONE primitive: a
hypothesis is a claim runs prove or disprove. Its `spec` both launches the runs AND identifies them (a run
is evidence iff its config is consistent with the spec); the verdict (untested/proven/disproved) auto-derives
from those runs (beats-buy-and-hold OOS), re-checks on settle/tab-open, records which runs flipped it
(`transitions[]`), is filterable + manually overridable + dismissible. Identity = `hashTrainingConfig(spec)`
so identical specs dedup across human/llm/paper/migrated sources. The **Models tab is removed** (a model
architecture = a hypothesis whose `spec.fixed` pins the levers; the 12 seeds migrated to `manifest.hypotheses[]`,
old `-model` records auto-migrate on open). **Papers are containers** of N hypotheses created three ways —
**Extract** (the reworked Automatic-Fill: the LLM drafts the paper AND extracts its testable hypotheses,
linked back), manual **Add hypothesis**, and **Link existing**; the paper's verdict rolls up from them. Pure
decision logic lives in node-tested `viewer/hypothesis.js`. Pending:

- Open-ended `researchTrainingPapers` (discover N papers) + the heavy auto-seed/verify pipeline
  (find → web-verify → synthesize). Deferred.
- Optional: a card-level "Extract" that re-analyses an existing paper's link and merges new hypotheses INTO
  that paper (needs a `paperId` param on `analyzePaperFromUrl`). Today Extract is the add-chooser path only.

### 3c. Model architectures (now hypotheses)

The `attn-ppo` / `tcn-ppo` encoders are shipped; architectures are now hypotheses (their `spec.fixed` pins
`model_name`). Pending:

- **LEFT (optional):** a **GRU** recurrent core — must be a custom `RecurrentActorCriticPolicy` subclass
  (state lives in the policy, not a features-extractor), expected to be a wash vs LSTM; and an SSM
  (S4D) falsification arm. Both deferred (lower value / higher effort); the registry + sweep pattern
  make them drop-in when wanted.

### 3. xAI — explain WHY the model acted (parallel track)

The xAI track — decision-trace spine + the full xAI tab (Phases 1–5) — is shipped (git +
`docs/architecture.md`); the model-trainer side stays domain-oblivious. Pending:

- **Live VISUAL pass in the Overseer.** The whole xAI viewer is engine-parity-tested + syntax/reference
  clean but has NOT been eyeballed in the running app — the one open verification.
- **Configuration-space map (t-SNE / projection) + reward normalization.** Next major xAI push (AFTER the
  current batch). Two parts the user will spec in detail: (1) a t-SNE/UMAP-style 2-D map of the
  CONFIGURATION space (runs positioned by config similarity, coloured by a criterion) so neighbourhoods +
  gradients are visible the way the latent map shows the policy's state space; (2) NORMALISING some
  signals — especially the model REWARDS — so cross-run/cross-setup comparison isn't distorted by raw scale.
  Open questions to resolve when specced: which projection (deterministic vs t-SNE seeded), what distance
  over mixed numeric/categorical levers, and exactly which quantities get normalised + against what
  baseline. Deterministic + parity-mirrored like the rest of the engine.
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

### Daily-step multi-fidelity (provider enhancement)

`timeframe` (the agent's STEP) and `fidelity_set` (the LAYERS observed) are now decoupled + validated:
an hourly step can observe coarser layers (the new `1d+1w` set), and incompatible combos fail fast.
STILL UNSUPPORTED (fails fast for now): a DAILY step observing FINER layers (e.g. a 1d-step agent
seeing 1h+1d) — `MultiTimelineDataProvider` has no `divider_run` mapping for `input=1h → run=1d`.
Enable by adding that mapping + verifying the step/aggregation against a fixture + a live run. Then
the full step × layers matrix is open.

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
