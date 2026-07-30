# thefactory-modeltrainer — implementation plan

**Remaining work only** — shipped history lives in git + session memory. Architecture: `docs/architecture.md`.
Contract: `docs/model-training-standard.md`. The core loop (engine, backend activities, viewer, remote
runner) and three conformant consumers (`examples/cartpole`, `examples/tabular`, **BlackSwan**) are built;
the engine stays domain-oblivious — any further model is _data + the thin CLI contract_, not engine code.

**Structure:** §A = things to **BUILD** (the running backlog of platform improvements — keep adding to it).
§B = **experiments to RUN + interpret** (validation, not code — all lumped here). §C = bigger deferred
projects. §D = open questions.

## North star (frames prioritization)

1. **Best generic pipeline for creating ANY model** end to end (propose → run → judge → explore), with a
   self-explanatory results UI, a minimal-storage/derive-at-runtime data layer, and guidance from "here's my
   problem" to "here's what data to mine."
2. **Use it to make BlackSwan the best trading model**, in STRICT ORDER: **(A) correctness → (B) find ONE
   setup that trades well → (C) huge-space exploration.** BlackSwan is the forcing function that hardens the
   generic pipeline. "Trades well" is defined by the **scorecard** (gates + fitness), NOT the reward — the
   reward is a training proxy that for BlackSwan does not equal success (A2).
3. **Fully AI-operable, shipped as a template.** Anything a user can do in the app, the API/CLI AI can do via
   tools through the approval gate (A3) — so a user drives the whole loop as a CONVERSATION where the AI
   researches/proposes/runs and the human mostly approves (A4); and the app is a SINGLE-PURPOSE template a
   project copies + a library it consumes for engine + base UI (A5), with BlackSwan the first consumer.

## Repo split (governs where work lands)

| Repo | Owns |
| --- | --- |
| **thefactory-modeltrainer** (this repo) | `ModelTrainerTools`; matrix planner; campaign loop; judge/propose orchestration; the viewer; the standard + `examples/`. |
| **thefactory-tools** | Generic infra: `ComputeRunner` seam (+ future `RemoteComputeRunner`, `ContentAddressedDataCache`, pairing); work-item engine. |
| **thefactory-backend** | Activity registration + composition; app-view serving; future PIN-pairing endpoints + runner WS channel. |
| **clients** | Future Compute Runners settings/pairing screen (native, cross-project). |
| **the runner agent** | Future Docker-packaged connect-out program. |
| **BlackSwan** (the trading repo) | Its `TrainerManifest` + additive `trainer/` CLI conformance. No Overseer code. |

---

## A. Build — implementation tasks

Ordered by value. Each is something to **implement**. (Experiments to run live in §B.)

### A1. Context / data-fusion — finish the consuming feature

The projection ladder + `context_set` panels (`rates`/`macro_core`/`majors`/`market`) + presets + advanced
launch are shipped. Remaining builds:

1. **Step 3 — `with_extra_data`** (4th projection rung: an asset's OWN fused series + the obs-signature gate).
   Trigger-blocked: only useful once an asset has its own series — premature for crypto (empty per-asset edges).
2. **Post-first-release window restriction** in `config_builder` (pre-release context bars `fillna(0)`).
   A no-op for the deep-history FRED panels — do when a short-history context series is added.
3. **App-triggered macro mines** — `FRED_API_KEY` in the BACKEND env so the Data tab can mine `rates`/`macro_core`
   (the local `BlackSwan/.env` covers CLI-run mines only).
4. **Auto-refresh the cross-test (`-settest`) cache** on `data:updated`.

### A2. Define success — reward vs scorecard (gates + fitness), and measure their alignment

**Why this comes first (blocks A3–A5 + B2).** The **reward** is a training PROXY, not the definition of
success. For CartPole/Wine reward = acceptance = ranking all coincide (which is *why* they feel solved and the
Diagnosis correctly calls them exhausted). For BlackSwan they DIVERGE: a high `traded_return` does NOT mean a
usable trader — a good model must beat buy-and-hold, trade often enough *over time*, and hold drawdown in
bounds, and whether `no_op`s matter depends on the deployment case. So the thing a human uses to filter good
runs from bad lives only in their head; it is not declared, not computed, and — critically — not what
selection/ranking/exploration optimise. This is the missing first-class layer. It must land before the AI
companion (A4) and full parity (A3): the AI can only drive and rank the loop if "good" is machine-readable.
**Generic principle:** every project declares THREE layers, not one — `objective` (reward, steers learning),
`gates` (accept/reject predicates), `fitness` (ranking metric(s), possibly Pareto). Simple projects collapse
all three; BlackSwan is the forcing function that separates them.

1. **Scorecard schema (generic, the core build).** Alongside the manifest `objective`, declare two new blocks:
   `gates` = pass/fail predicates over summary metrics (a run is ACCEPTED only if all pass) and `fitness` =
   the ranking metric(s) (single scalar or a Pareto set). Compute a per-run **scorecard** post-hoc from
   existing metrics. Switch incumbent selection / convergence / exploration ranking and the viewer's
   filter+sort to read the **scorecard**, not the reward (today `criterion`/incumbent read the objective). For
   CartPole/Wine `gates`/`fitness` default to the objective — a deliberate no-op that proves the collapse.
2. **BlackSwan gates + fitness — Case 2 (position manager, what exists).** Gates: `oos_return_pct > 0`,
   `oos_return_pct > hold_return_pct` (beat hold), `trades_per_day >= r_min` (time-normalised liveness — a raw
   `n_trades >= 10` is confounded by window length; normalise per day/month), `max_drawdown_pct < dd_max`
   (`combo_drawdown_penalty` lever already emits the metric). Fitness: a Pareto/weighted set over `oos_sharpe`
   (or Sortino/Calmar — see C1), excess-over-hold, drawdown, and a trade-rate band. Emit `trades_per_day` +
   the derived scorecard fields from `trainer/summary.py`.
3. **Reward rework to match Case 2 honestly.** Replace `traded_return = total_return × min(1,(n_trades/20)²)`
   (a multiplicative hack with a magic 20) with a proper portfolio reward = per-step mark-to-market equity
   log-return minus fees, and move trade-frequency from a multiplier to a **constraint/penalty band**
   (`trade_rate ∈ [lo, hi]`) — behaviourally enforced, no arbitrary constant. **Governing rule:** in-sample
   behavioural SHAPE (trade frequency, no-op fraction) → the reward (as a constraint); out-of-sample OUTCOME
   (beat-hold, OOS Sharpe) → the scorecard ONLY, never the reward — rewarding on the metric you then select on
   re-creates the single-window overfit already diagnosed.
4. **Case-1 forward-horizon expectancy lens (cheap, on existing checkpoints).** The two deployment cases are
   two different problem FORMULATIONS, each with its own reward AND score: **Case 2** = single-asset position
   manager (net-of-fees equity reward; `no_op`/hold is a legitimate decision so it's not a quality signal;
   strong guarantee; does not generalise) — that's items 2–3. **Case 1** = asset-agnostic signal emitter (C2)
   where EVERY buy/sell must stand on its own. The automated "verify every signal" IS forward-return labelling
   (the `forward_horizon` machinery already exists): score each signal by realised forward return over H bars →
   per-signal expectancy / hit-rate / coverage, independent of position. Run it as a SECOND lens on existing
   Case-2 checkpoints (no retrain) → answers "does anything generalise?" cheaply; a Case-2 winner that ALSO
   scores as a Case-1 emitter is the strongest signal. Feeds C2 (own objective + `trainer-signal.json`).
5. **Reward–success alignment diagnostic (generic — the answer to "how do we KNOW a high reward isn't a good
   model?").** A new Diagnosis check on the shipped diagnostics engine (the `incumbentSplitHoldout` /
   `diagnoseSearch` layer): across the cohort, correlate the training reward with each gate/fitness metric.
   Strongly correlated → the reward is a good proxy (CartPole). Reward high but DECORRELATED from fitness → the
   reward is a misaligned proxy (BlackSwan, precisely) → the check reports it and points at the scorecard.
   Turns the qualitative worry into a number on ANY project.
6. **Update the standard.** Codify `objective` / `gates` / `fitness` as the three declared layers in
   `docs/model-training-standard.md`: reward stays the training signal; gate acceptance reads `gates`;
   ranking/incumbent/convergence read `fitness`.

### A3. Full AI action parity — anything the user can do, the AI can do

**Standing rule:** every mutating/launching user action has an equivalent chat capability, routed through the
existing **approval gate** — `startProjectActivity` / `createProjectRecord` / `updateProjectRecord`
(thefactory-tools `projectData`, approval-gated) + the trainer tools. The mechanism already exists; the gap is
a DECLARATION gap. Today the chat can launch only `manifest.activities = [inspect-trainer, scan-models,
propose-experiments]` and edit only hypothesis/paper records (`trainerDataCapabilityManifest`,
`viewer/app.js:2180-2240`); every other user action (train, explore, judge, evaluate, cross-test, mine/discover
data, research papers, config/xai analysis, consolidate, delete/invalidate) is unreachable via chat.

1. **Declare the full surface** in `trainerDataCapabilityManifest`:
   - Launchable activities (each approval-gated): add `train`, `explore`, `judge`, `evaluate`, `cross-test`,
     `continue-training`, `data-catalog`, `mine-data`, `discover-data`, `approve-data-source`,
     `research-training-papers`, `analyze-paper`, `suggest-paper-hypotheses`, `weigh-paper-hypotheses`,
     `analyze-paper-models`, `config-space-analyze`, `run-xai-analyze`, `xai-narrate`, `consolidate-models`,
     `consolidate-hypotheses`, `benchmark-model-device`.
   - Editable/creatable record types: environments, datasets, presets, run flags (reliability / favorite /
     unrunnable), filter + bad-run rules, hypothesis min-runs, activity queue/budget. (Manifest + levers stay
     code → the AI files a story/feature, never edits them in place.)
2. **Spec-validated launch.** `train`/`explore` need matrix validation — reuse `recommendTrainingExperiments`'
   `expandExperimentMatrix` + lever validation. Add a `launchTrainingCampaign` tool (or validate server-side in
   the `train` activity) that rejects malformed matrices with the same `rejected[]` diagnostics, and fold in
   "launch an approved `-xai-suggestion`" (today only a human click launches a suggestion — see B2/xAI Suggested).
3. **Missing verbs** (no activity/record path today): `deleteRuns` (destructive — ALWAYS explicit approval),
   `invalidateRuns`, `migrateTrainingRuns` — add as approval-gated activities/tools.
4. **Parity audit (prevent regression).** A test that enumerates the viewer's `startOrEnqueue(<type>)` +
   `putData`/`deleteData(<recordType>)` sites and asserts each is declared in the data-capability manifest (or
   explicitly exempted as code). Codify the rule in `docs/model-training-standard.md`.

### A4. AI companion — drive the whole loop as a conversation with approvals

**Goal:** a long conversation where the AI researches/thinks/runs and the human mostly approves. Built on A3
(full action parity) + the approval gate.

1. **Orient tools.** Promote the shipped `diagnoseSearch` to a chat tool (the AI reads the same Diagnosis plan
   the user sees) and let it rank candidates on the A2 **scorecard** (not the reward); add campaign/activity-status
   reads (via `queryProjectData` on activity-run records) so the AI knows what's running / pending / done.
2. **Approval surface (new VIEW — the "just approve" seam).** A proposals inbox: the AI's pending actions
   (launch / edit / delete) with rationale + expected cost (runs × ETA) + one-click approve / reject, batched.
   Reuse the agent change-review design (trust×cadence dials, ChangeSet, approve/reject/tweak). Per-project
   **trust level**: reads + cheap analyses auto; launches + writes need approval; destructive always explicit.
3. **Wait / resume across long runs.** The loop is launch → wait (minutes–hours) → read → decide. Add a
   **campaign-complete → wake-the-chat-topic** hook so the AI re-engages when its launched runs land and proposes
   the next step (v1 fallback: the human says "done" and the AI reads results). The AI polls status + resumes the
   strategy across waits.
4. **Campaign-strategy memory.** A chat-owned `strategy` record the AI reads+writes (which experiments, why,
   what's decided / open) surfaced in the app so the human sees the AI's thinking — the Diagnosis plan +
   hypotheses registry are the substrate.
5. **Decide:** the wake-the-chat hook (how the backend notifies a chat topic on run completion); the approval
   granularity (per-action vs batched vs trust-tiered — lean on the change-review design).

### A5. Single-purpose template + BlackSwan-as-library

**Strategy change.** Today's docs prescribe a multi-project HUB (`architecture.md`; projects "registered, not
forked"; repo-split table). New model: modeltrainer is ALSO a SINGLE-PURPOSE template + LIBRARY a project copies
/ consumes; BlackSwan gets its own app built on it, pulling updates. Both coexist (hub for dev/multi-project;
template for a shipped single-purpose app).

1. **Single-purpose boot (viewer).** Add `bootProject(manifest, dir)` that boots straight into the dashboard
   from ONE manifest (load the project's own `.factory/trainer.json` via `inspect-trainer dir:'.'` or bundled),
   bypassing the home shell; gate the multi-project shell (`projectsCache` / `renderHome` / add-remove-inspect,
   `app.js:1355-1745`) behind a hub-mode flag. Low risk — `currentProject` has ~10 refs and the dashboard is
   already single-project (reads `manifest` + `dir`).
2. **Project override layer.** A thin per-project `project.js` / config the base viewer loads for project chrome
   (labels, presets, extra panels) WITHOUT forking the monolith — so BlackSwan customizes and modeltrainer
   updates still flow.
3. **Package + publish.** Engine (`src/`→`dist`) is already a clean npm lib (backend consumes it); the tarball
   already ships `viewer/` (`files:[dist,viewer]`). Publish a versioned build so consumers pin it.
4. **BlackSwan consumes modeltrainer.** Constraint: the overseer serves app files ONLY from inside the project's
   checkout (`files.ts` escape guard), so the viewer bytes must live in BlackSwan. **Option A (recommended):**
   BlackSwan adds `package.json` depending on `thefactory-modeltrainer`; a postinstall/build copies
   `node_modules/thefactory-modeltrainer/viewer` → `BlackSwan/app/` (its `appDir`); update via `npm update` +
   re-copy. **Option B (fallback):** git submodule/subtree of `viewer/` at `BlackSwan/app` (no Node toolchain,
   commit-pinned). Reject symlink (escape guard). BlackSwan settings: `hasApp:true`, `appDir:"app"`. No Python
   change — the trainer CLI already conforms; engine stays a backend library (BlackSwan consumes the viewer only).
5. **`app.js` de-monolith (enabler, not blocker).** The 900KB+ `app.js` is the main reuse/override obstacle;
   optionally split into loadable chunks like the existing IIFE modules so a template override never edits the
   monolith. Options A/B work without it.
6. **Update docs** (`architecture.md`, repo-split table, `model-training-standard.md`) to describe the
   template+library model alongside the hub.

### A6. Follow-ons on the shipped features (small + opportunistic — none blocking)

Remaining slivers on features that are otherwise done; pick each up alongside its related work.

- **xAI traces:** per-STEP attention matrices (v1 is a run-aggregate — needs a sidecar route, never inline); a
  viewer surface for the `snapshotTraces` index (diff consecutive mid-training snapshots in the Explain view);
  a snapshot-count / disk cap (ring-buffer).
- **Exploration:** **Pareto basins** in the reducer — qualify/rank basins on the Pareto front, once A2's
  multi-objective `fitness` lands (today the reducer ranks on a single scalar).
- **Activity:** the run↔activity JUMP end-to-end — the `activityId` is already stamped on every record an
  activity wrote, so link a run detail ↔ its producing/judging activity (the Activity History popup is shipped).
- **Datasets:** derive the `asset` lever CHOICES from disk (today a manifest list) + dim data-less values in
  the Datasets cards; deeper stock walk-forward windows (train from 2018).
- **On-device verification (test, not build):** mobile parity pass of the conversational-hub surfaces; in-app
  Data-tab check. Both need a device.
- **Parked (need a net-new dependency):** generative counterfactual states (a GAN/VAE); step-by-step decision
  animation replay + scrubber; `seed` still counts as a model lever in the engine's fANOVA (needs a manifest
  lever `scope:'ignore'` + re-analysis — flag, don't silently change).

---

## B. Testing & experiments — RUN + interpret

Validation, not code. BlackSwan campaigns launch one-click from the Launch presets / Diagnosis tab. RL variance
is large, so read via the Diagnosis **split-consistency** verdict, never raw per-arm averages.

### B1. Context experiments — does external context help time an asset?

Run the **Context ablation** preset (own-price `minimal`|`with_indicators` × context `none`|`macro_core`) AND
the `market` (keyless gold/SPY/dollar) + `majors` presets, at **≥5 seeds × 2–3 windows**. Reads: B(bare+ctx)
vs A(full,noctx) = substitution; C(full+ctx) vs A = complement. _First runs were inconclusive — needs the
power (seeds×windows) + the pre-registered read to separate a real effect from RL noise._ Context: motivated
by the B2 diagnosis that single-asset price may be too noisy alone.

### B2. BlackSwan Phase B — find ONE setup that generalizes

Context (`project_blackswan_search_diagnosis`): across 20,888 leak-free runs, NO setup beats buy-and-hold OOS
across all four walk-forward windows (≥2: 46, ≥3: 4, ≥4: **0**); **93% of strong wins are the single 2022 bear
window** and invert in up-years — a generalization failure (beta-neutral, Sharpe ≈1.3). In EV order:

1. **Replicate** the ~top-20 setups across ALL windows × ≥5 seeds; select on worst-window `return_vs_hold`.
2. **Reseed** promising single-seed setups to ≥5 seeds so luck separates from edge.
3. **Drawdown-penalty sweep** — `combo_drawdown_penalty ∈ {0, …}`; hypothesis "makes zero-trade setups trade
   (`n_trades` 0→>0)" (may silence already-trading setups instead — falsifiable). Lever + metric already wired.

If nothing survives replication, the edge likely isn't single-asset BTC price → move to **lower-noise asset
classes** (stocks, already tradeable; then commodities/FX via C3's data coverage) and read a **cross-class
leaderboard** (is any edge asset-class-specific, or general?), and/or the Deferred multi-asset split (C1).

### B3. Cross-asset robustness — exercise live

Launch any campaign with **Keep-checkpoints on**; the auto cross-test replays each model on other assets +
extended (`oos-*`) windows with no retraining → read the **Robust column + Robustness lens + Diagnosis
cross-asset check**. The cheapest generalization test — directly attacks B2's overfit. (No existing run has a
checkpoint, so this needs one checkpointed campaign first.)

### B4. Exploration autopilot — live acceptance

Restart the backend (loads the server-side reducer/controller), then run on CartPole (should rediscover ≈500 +
enumerate basins), Wine, and BlackSwan (covers the space, declares the best). Validates the loop; on BlackSwan
expect it to confirm B2 (no edge in the current space) until A1/B1 (or a lower-noise class) add a space that
might contain one.

---

## C. Deferred — bigger implementation projects

### C1. Multi-asset portfolio / cross-sectional long-short — a SEPARATE project

The one genuine project split. BlackSwan's single-asset env hardcodes one asset everywhere, so multi-asset
needs a fundamentally different env: a **3-D observation** (`asset × lookback × features`), a **portfolio
action space** (per-asset long/short/weight), a **timestamp-aligning N-symbol data provider** (misalignment =
silent P&L corruption — unit-test against a 3-coin × 100-bar fixture), and an **A2 scorecard for a portfolio**
(reward = portfolio return minus fees + a correlation-penalty CONSTRAINT; fitness = Sharpe/Calmar — not a
"Sharpe objective" that steers directly). REUSES BlackSwan's reward components, feature engineering, SB3 algos,
walk-forward harness; ~3–4 week project, blocks none of the in-place wins. Research calls cross-sectional
long-short the strongest-edge config — promising, deliberately sequenced last.

### C2. Position-blind signal model — a SEPARATE objective

The trading line trains a position MANAGER (actions are position-gated; out-of-position output is never shaped
— `blocked_signal_ratio` ~0.95). A model whose raw per-step output IS a long/flat(/short) signal, independent
of position, is a **different objective** (the "Case 1" of A2). Probe it cheaply FIRST via A2's
forward-horizon expectancy lens on existing manager checkpoints — build this full project only if that lens
shows a generalisable per-signal edge. First confirm the clean manager (via `combo_noop_penalty`, which
drops `blocked_signal_ratio` ~0.95→~0.02 by going silent on no-ops) isn't already enough. Then: reuse the env's
`buy_sell_signal` reward family or the supervised direction head; its OWN objective (signal precision/recall/
coverage vs realized forward returns, or a signal-following backtest); likely its OWN manifest
(`trainer-signal.json`); its own verdict rule + signal-overlay chart (the verdict+objective plumbing is the
bulk, not the model). Runs with `combo_noop_penalty>0` are CLEAN-MANAGER runs — don't migrate them; represent
the split as a DERIVED `approach` facet. Compare manager vs forecaster on a SHARED signal-following backtest.

### C3. Data platform — remaining source coverage + productionization

**Remaining source coverage** — the lower-noise asset-class line (this supersedes the old "strategic
direction": the mine that enables it is largely built). Macro `rates`/`macro_core`, `majors`, `market`, and
stocks are already mined + tradeable (A1 + the shipped stocks class). Left: make **commodities + FX** tradeable
classes (catalog + `asset` lever + backfill, per the shipped stocks pattern — absorbs the old "other single
assets" item) and mine the **fundamental panels crypto lacks** — earnings, COT/inventories, trade balances, an
event calendar — each under the correctness rules below. The edge-search across these classes is B2's
cross-class leaderboard, not a build here.

**Productionization (D5) — only when a SECOND trainer needs the same data:** extract catalog+miners+CLI to a
standalone `thefactory-datamine`; generalize `derive_cache` to a central 1m→fidelities service; wire the
`ContentAddressedDataCache` + remote-runner data path from one curated origin.

**Data correctness rules (enforce in the loader, for all future mining):** store minimal raw only (derive at
runtime); **join by TIMESTAMP, never date string** (each row carries `barCloseTz`); macro is point-in-time
(ALFRED vintages, stamp at the real per-release datetime, forward-fill, MoM/YoY = diff of the SAME vintage;
daily rates that exceed the vintage cap fetch standard obs stamped as-of since they're non-revised); post-close
series (H.15 ~16:15 ET) publish next session; fundamentals stamp at filing/acceptance not period-end
(restatements = new rows); commodity continuous roll is a look-ahead machine (back-adjust with roll dates); FX
`Volume≡0` is a constant not data (neutral-fill; 6-char `=X`); never forward-fill one leg of a ratio;
idempotent + validated mining (monotonic timestamps, positive prices, split/div-adjust); licence-gate shareable
output (only Frankfurter/ECB is redistribution-safe).

### C4. Code-change risk model — a third ML consumer (research first)

A trainer-conformant project scoring an agent's diff/PR by bug-likelihood. (1) Research: survey public
JIT-defect datasets vs mining our own from the `thefactory-*` git histories → cited report + go/no-go.
(2) Data (via the mine): SZZ-style labeling + codeIntel features (churn, complexity, coverage, diff size).
(3) Train: a `risk-classifier` project (objective = AUC / precision-at-k). (4) Consume: wire the score into
the review / expert-panel / verifier path. _Further out: a FastContext-style repo-explorer subagent — gated by
the same go/no-go + LLM-training compute the ComputeRunner has never exercised._

### C5. Optional + small deferred

- **Live handoff** — tag the exploration autopilot's global-max checkpoint for live trading (`run_server_model.py`).
- **Jupyter notebooks (UNDERSCOPED)** — view/edit/execute a project's `.ipynb`; scope kernel location + security.
- **Runner-channel WebSocket upgrade** — dispatch is already ~instant; a WS only shaves ~1.5s log latency.
- **Remote git repoRefs** — wire git refs + project bootstrap when a real remote machine needs it.

## D. Open questions (decide when the dependency lands)

- **Remote artifact/checkpoint storage** — keep-on-runner + reference vs upload; how a winning remote
  checkpoint reaches the live trading server. Meaningful once remote runs AND live handoff both exist.
- **GPU + sandbox profile for training images** — `--read-only` rootfs vs ML caches; `--gpus` is wired but
  unexercised (the runner runs jobs directly, not through Docker-sandboxed `SandboxTools`).
- **Judge/proposer model transport** — `ModelSelection` (API vs CLI), overtaken by the in-flight refactor.
  Revisit once the CLI inference stage lands.
