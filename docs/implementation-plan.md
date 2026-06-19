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
  By-dataset robustness table), then lock in the first config that clears the bar. The supervised+rules
  baseline is BUILT and trades through the unchanged env → comparable RunSummary. Keep RL alive while the
  other approaches run in parallel.
- **`combo_all_noop` reward + tunable penalty levers (DONE).** A reward model = `combo_all` plus an
  explicit penalty for NO-OP trades (buy while already holding, sell while holding cash — `take_action`
  returns False), via a `_noop_penalty()` mirroring `_turnover_penalty()`. The penalty magnitude is the
  `combo_noop_penalty` lever (0 = off), and the long-existing turnover penalty got its `combo_fee_penalty`
  lever exposed too — both wired through `reward_multiplier_*` → `get_reward_multipliers` → env. NEW
  generic capability: a lever can declare `appliesWhen` (e.g. `{reward_model: ['combo_all_noop']}`) and
  the Launch form greys out / drops from the spec any conditional lever whose controlling lever isn't set
  to a matching value — so a setting only some reward models use isn't swept where it does nothing.
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

**(2b) Papers / Library tab — SHIPPED (manual registry); research-seeding deferred.** The roster of
approach cards is built: `TrainingPaperRecord` type, a `<recordType>-paper` data record, full CRUD
(`readPapers`/`putPaper`/`deletePaperRecord` + `setupPapers`/`renderPapers`/`paperCardHtml`/
`paperFormHtml`/`togglePaperForm`/`onSavePaper`/`onDeletePaper`), a Papers tab (TABS + index.html
section + showTab/setup wiring), cards with title→url, authors/year, a colour-coded verdict badge, the
claim, assumption chips (fees / gross / frictionless / multi-asset / retrain), **claimed-vs-measured**
(measured = best objective + beats-hold from `linkedRunKeys`), a verdict filter, and per-card actions
**Replicate** (prefill Launch from `replicateConfig` — full preset or flat fixed map), **Link selected
runs** (links the Runs-tab compare selection), Edit (records verdict + note, with a measured-suggested
verdict hint on the card), Delete. New entries open as a two-step CHOOSER (link + **Manual Entry** /
**Automatic Fill**); Manual Entry reveals the full form, Automatic Fill shows a spinner then a
coming-soon toast (the LLM fill is the deferred backend below). Campaign↔paper linking: **Replicate
auto-connects** the launched campaign (`launchFromPaperId` → `extra.paperId` → `stampPaperCampaign` at
launch, `stampPaperCampaignResults` fills `linkedRunKeys` from the campaign record on settle), and a
**"Link a running campaign to a paper"** area at the bottom of the tab links a live campaign manually;
cards show a "campaign linked · N runs linked" chip.

- **Starter-paper seeding (DONE):** the manifest ships `papers?: TrainingPaperSeed[]` and the Papers
  tab shows an "Import N starter approaches" banner that upserts them once (by id, skipping any the
  user already has so edits/verdicts aren't clobbered). BlackSwan's `trainer.json` carries the curated
  top-10 trading papers.
- **"Automatic Fill" backend — SHIPPED.** Paste a link in the add-paper chooser → "Automatic Fill" →
  an LLM drafts the entry. Built across all three layers: (1) `analyzePaperFromUrl` tool method in
  `ModelTrainerTools` (the TOOL fetches + text-extracts the page via `fetchPaperText` — arXiv pdf→abs
  normalised, PDFs rejected with a clear message; `extractPaperText` is pure HTML→text; then ONE
  structured-output inference via the existing `inferenceExecutor`, `parseFirstValidJson` →
  `coercePaperDraft` → a `TrainingPaperRecord` draft incl. a suggested `replicateConfig` against the
  manifest levers; upserts a `<recordType>-paper` record `status:'untested' source:'research'` keyed by
  `uuidv4`, fires `onRecordWritten`). (2) Backend `analyze-paper` activity (`requiresApi:true`, mirrors
  `propose`; also writes a `-paper-analysis` 'latest' summary). (3) Viewer `onPaperAutoFill` triggers it,
  holds the spinner, observes, then closes the form + re-renders Papers so the **draft appears as a
  reviewable card** (Design X — the bridge has no generic request→response, so we auto-create + review
  rather than prefill the add-form). TDD'd (utils + tool with mock executor/storage/fetch).
  - Still DEFERRED: open-ended `researchTrainingPapers` (discover N papers) + the heavy
    auto-seed/verify pipeline. Optional UX upgrade — prefill the add-form from the draft instead of
    auto-creating the card — would need a request→response bridge verb.

### 3c. Models / Architectures library (like Papers, for model build-ups)

- **The registry/tab — SHIPPED.** A Models tab mirroring Papers: `TrainingModelRecord`/`TrainingModelSeed`
  types + `<recordType>-model` records + CRUD, cards showing the build-up (model_name / algo / net /
  policy internals) + rationale + claimed-vs-measured + a proven/disproved verdict badge + a verdict
  filter, Replicate→Launch (prefills from `replicateConfig` or `{fixed: match}`), Edit/Delete, and the
  manifest seed-import banner. KEY DIFFERENCE from Papers: evidence is AUTO-DERIVED — a card's `match`
  names the model-lever values, and `modelMatchingRuns` finds the runs using that architecture, so
  measured + the suggested verdict come from them automatically (no manual run-linking / campaign
  machinery). "View N runs" filters the Runs tab to the architecture's runs. BlackSwan's `trainer.json`
  seeds 10 architectures (reppo-custom, trpo-custom, ppo-custom, duel-dqn-custom(+lstm),
  munchausen-dqn-custom, qrdqn-custom, vanilla dqn, supervised+rules, hodl) with honest pre-v3 caveats.
- **The research + build direction — STARTED (attention + TCN shipped).** A web-grounded survey
  (5 angles → ranked) picked two cheap, defensible, genuinely-new encoders. Built
  `src/model/custom/sequence_extractor.py` — a `SequenceFeaturesExtractor(BaseFeaturesExtractor)` that
  reshapes the flat (time-major) obs back to its `[lookback, per_bar]` bar grid and runs a TRUE
  temporal encoder over the bars (the existing `custom_net_arch` attention tokens run on the flat
  vector — degenerate), then pools → policy MLP. Two encoders: `attn` (1-block self-attention +
  learned positional) and `tcn` (dilated causal residual conv). Wired as PPO model_names
  **`attn-ppo` / `tcn-ppo`** in `model_factory` (plain `MlpPolicy` + `features_extractor_class`;
  lookback from `env.data_provider`), added to the manifest model_name choices, seeded as Models cards
  - an **Exp 14 architecture-A/B sweep** (reppo LSTM vs attn vs tcn). TDD'd (9 tests incl. the
    reshape-ordering/leakage watchpoint + a `create_model` build smoke). HONEST FRAMING (in the cards):
    architecture is almost certainly NOT the high-leverage lever on one noisy asset at 0.1% fees — these
    are variance-reduction + a diagnostic; the acceptance gate is beating BOTH buy-and-hold AND the tuned
    LSTM on walk-forward `traded_return` across ≥3 seeds + windows.
- **LEFT (optional):** a **GRU** recurrent core — must be a custom `RecurrentActorCriticPolicy` subclass
  (state lives in the policy, not a features-extractor), expected to be a wash vs LSTM; and an SSM
  (S4D) falsification arm. Both deferred (lower value / higher effort); the registry + sweep pattern
  make them drop-in when wanted.

### 3. xAI — explain WHY the model acted (parallel track, like Papers)

A decision drill-down across BlackSwan + model-trainer (NOT a new project); the model-trainer side
stays domain-oblivious (generic "decision trace", arbitrary action strings, no trading vocabulary).

**The spine — SHIPPED.** `trainer/decision_trace.py` (BlackSwan) reconstructs the per-step action trace
from the env's retained arrays, then DETERMINISTICALLY replays `model.test()` once more to capture
per-step confidence + per-action values (DQN Q-values, PPO/TRPO action probs) and gradient saliency,
attaching a compact `DecisionTrace` to `summary.artifacts.decisionTrace` (full per-step + obs sidecar
opt-in via `decision_trace_full`); hooked after `build_summary` in `run.py`, best-effort (a missing
trace is never an error). Generic types + a soft `validateDecisionTrace` live in the engine
(`modelTrainerTypes.ts`/`modelTrainerUtils.ts`; ingestion strips an unusable trace). The viewer's
**Explain** section renders the action-distribution diagnostic (generic anomaly + dormant-action
flags), the **sparse-sell deep-dive** (per-action value-over-time chart on the shared step axis — makes
"is hold persistently worth more than sell?" self-evident), confidence-over-time, and **input
attribution by fidelity layer + engineered signal**; `chatAboutRun` is enriched with a trace summary
(counts, dormant-action value gaps, top attributed inputs).

Feature attribution (gradient saliency) ships `perFeature` AND a precise `byGroup` — `layer:1h` /
`layer:1d` (each fidelity layer's columns) and `engineered:drawdown` / `in_position` / … — derived in
`decision_trace.py` from the env's observation layout (`config.layers` + the active engineered extras),
reconciled against `obs_dim/lookback` so any drift degrades to per-feature-only rather than mislabel.
Verified live on 1d and 1h+1d runs.

**Data-influence on decisions — counterfactual diff SHIPPED.** "Did this new information change the
model's DECISIONS, and for the better — even when the score hasn't moved?" The engine's pure
`diffDecisionTraces(baseline, tweak)` (`modelTrainerUtils.ts`; types `DecisionTraceDiff` /
`DecisionStepDelta` / `DecisionQualitySignal`) aligns two runs that share a dataset/window step-by-step
(by `datasetAlignmentSignature` + equal `totalSteps`) and reports divergence rate, per-action count
deltas, confidence shift, and — the honest core — a decision-quality verdict from the realized reward
delta AT the divergent steps, CONTROLLED against unchanged steps, with an insufficiency guard and
"heuristic, not causal" labeling; the objective delta is context, not the verdict, and a disagreement
between them is surfaced as the interesting case. ZERO new BlackSwan infra (runs off the already-emitted
trace). The viewer's 2-run Compare pane grows a **Decision diff** section + a **Discuss these two runs**
chat seeded with the diff. Verified live on a real 1d run pair (`use_indicators` off→on: 53% of decisions
changed, verdict `worse`, agreeing with the objective drop).

**The xAI TAB — Phase 1+2 SHIPPED.** A dedicated viewer tab collecting both explainability LEVELS, plus
the analyse→run loop, all deterministic + non-LLM. Two deep-research passes
(`config-level`: fANOVA/ablation/DoE/variance; `feature-level`: RL-policy interpretability) ground the
design.

- **Config-effect engine (TypeScript, pure, parity-mirrored to the viewer).** `src/xaiUtils.ts` +
  types/constants: a SELECTABLE-criterion (objective / any metric / runtime, max-or-min) analysis over
  stored run records. `ofatContrasts` gives the exact one-factor-at-a-time read (runs identical on every
  OTHER lever + dataset → no confounding) with seed-variance rigor — **IQM** aggregate, deterministic
  seeded **bootstrap CIs**, the **difference-CI** significance route (not the CI-overlap fallacy), and
  **Benjamini-Hochberg FDR** across comparisons; `leverImportances` is the cheap surrogate-free screening
  view. `viewer/xai.js` mirrors it exactly (a `scripts/xaiParityCheck.mjs` harness asserts byte-identical
  results). 100% function coverage; the viewer "Config effects" panel renders importance + the controlled
  contrast.
- **Model internals (feature-level).** The xAI tab re-homes Explain + the Decision diff and adds cheap
  trace-only reads: **decisiveness** (top-2 action-value gap), **policy entropy over time**, and
  **confidence calibration** (mean realised reward by confidence bin). "Analyze in xAI" on a run opens it.
- **The analyse→run loop.** `recommendExperiments` deterministically surfaces gaps — **missing factorial
  cells** + **variance-thin top setups** (need ≥5 seeds) — each as a launchable `ExperimentSpec`; the
  "Suggested experiments" panel fires them as **batched campaigns** (concurrency selector → existing
  `train` activity, NOT the LLM `propose` path), and the tab auto-recomputes on `data:updated`.

Remaining (the expansion):

- **Phase 3 — the ablation TREE + global importance.** `fANOVA` (main + interaction effects) + ablation
  paths via a seeded TypeScript random-forest surrogate (the validated methods); slice/contour interaction
  views ("does A help universally or only at some B?"). The OFAT engine is the no-surrogate floor; this is
  the across-the-whole-space refinement.
- **Phase 4 — deeper feature attribution (needs BlackSwan emission).** Integrated Gradients + occlusion /
  permutation importance + the **Adebayo sanity-check badge** (recompute with randomised weights — visual
  plausibility ≠ faithfulness) + per-step **reward-component decomposition** (BlackSwan's reward is already
  a named additive sum). Subsumes the old "permutation/SHAP" bullet.
- **Phase 5 — parked.** TabularSHAP/DeepSHAP, latent t-SNE/UMAP + probing, attention-weight viz (attn-ppo),
  generative counterfactual states (needs a GAN), and the per-step group-saliency + mid-training-checkpoint
  data-influence follow-ons.
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
