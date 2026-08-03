# thefactory-modeltrainer — implementation plan

**Remaining work only** — shipped history lives in git + session memory. Architecture: `docs/architecture.md`.
Contract: `docs/model-training-standard.md`. The core loop (engine, backend activities, viewer, remote
runner) and three conformant consumers (`examples/cartpole`, `examples/tabular`, **BlackSwan**) are built;
the engine stays domain-oblivious — any further model is _data + the thin CLI contract_, not engine code.

**Structure:** §A = things to **BUILD** now (the ready backlog of platform improvements — keep adding to it).
§B = bigger deferred projects. §C = open questions + trigger-blocked builds (act when the dependency lands).
(Experiments are no longer a plan section — the exploration autopilot / Diagnosis plan drives them; findings
live in git + memory.)

## North star (frames prioritization)

1. **Best generic pipeline for creating ANY model** end to end (propose → run → judge → explore), with a
   self-explanatory results UI, a minimal-storage/derive-at-runtime data layer, and guidance from "here's my
   problem" to "here's what data to mine."
2. **Use it to make BlackSwan the best trading model**, in STRICT ORDER: **(A) correctness → (B) find ONE
   setup that trades well → (C) huge-space exploration.** BlackSwan is the forcing function that hardens the
   generic pipeline. "Trades well" is defined by the **scorecard** (gates + fitness), NOT the reward — the
   reward is a training proxy that for BlackSwan does not equal success (A1).
3. **Fully AI-operable, shipped as a template.** Anything a user can do in the app, the API/CLI AI can do via
   tools through the approval gate (A2) — so a user drives the whole loop as a CONVERSATION where the AI
   researches/proposes/runs and the human mostly approves (A3); and the app is a SINGLE-PURPOSE template a
   project copies + a library it consumes for engine + base UI (A4), with BlackSwan the first consumer.

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

Ordered by value. Each is something to **implement**.

### A1. Define success — reward vs scorecard (gates + fitness) — SHIPPED (only a viewer visual pass remains)

**Shipped** (see memory `project_modeltrainer_scorecard_shipped`): every project declares THREE layers —
`objective` (reward, steers) / `gates` (accept-reject) / `fitness` (ranking, maybe Pareto); omit gates/fitness ⇒
collapse to the objective (CartPole/Wine). `computeScorecard` + the verdict layer (`incumbentSplitHoldout` /
`diagnoseSearch` prefer gate-accepted runs, rank by fitness), the reward–success **alignment** diagnostic
(Pearson reward-vs-metric), the standard doc, and the viewer's accepted-first sort (`viewer/scorecard.js`) are
all built + tested. BlackSwan declares its Case-2 scorecard (4 gates + 3-metric fitness) and emits
`trades_per_day` + the **Case-1 signal lens** (`signal_expectancy`/`_hit_rate`/`_coverage`, position-blind
forward-return edge per buy/sell — the cheap probe that gates whether to build B2). Architecture call: the
exploration reducer keeps CLIMBING on the objective (its baseline/noiseFloor are objective-scoped); the
scorecard drives the VERDICT + convergence gate — steer on reward, decide on scorecard.

Also shipped: the viewer's **acceptance badge** ("Success" column: green accepted / red rejected + failed-gate
hover) and the **success: accepted/rejected filter** (client-only, forces the unpaged path; gated on
`hasScorecard` so no-scorecard projects are unchanged) — built + adversarially reviewed, pending a visual pass
in the running Overseer.

**Also shipped — honest BlackSwan objective (NO history break).** The objective is now `total_return_pct` (the
raw post-fee return), NOT the magic-20 `traded_return` (`total_return × min(1,(n/20)²)`); trade frequency is the
`trades_per_day` GATE, not baked into the objective. This did NOT need invalidation: `total_return_pct` is
already emitted on every run, so a generic backfill in `migrateTrainingRuns` enforces the invariant
`objective == metrics[objective.name]` — the boot-time `sweepTrainerMigrations` (BlackSwan has `migrations`, so
it runs; idempotent) rewrites each stale run's objective from its stored metric on the next Overseer boot.
`traded_return`/`trade_gate` are kept as diagnostic metrics.

**Remaining:** only a visual pass of the viewer Success badge + filter in the running Overseer.

### A1b. Multiple named, in-app-tunable scorecards — SHIPPED (pending a viewer visual pass)

The shipped scorecard is a SINGLE manifest-declared gates+fitness per project. The intended (and originally
verbally-discussed, never-written) design is a SET of named scorecards, tunable IN-APP, with one **active**
driving the Runs table and the others shown in run detail for comparison. **Reconciliation of a prior decision:**
"manifest + levers stay code" governs the RUN SPACE (levers/command/data — change them and a run's meaning
changes). A scorecard is a **post-hoc evaluation lens** — tuning it only re-scores existing runs — so scorecards
are deliberately EXEMPT: they become user-editable DataStorage records, while the manifest's gates/fitness merely
SEED a "Default" card. The three-layer model + steer-on-reward/decide-on-scorecard split carry forward intact
(a scorecard is still one objective-agnostic gates+fitness triple; multi just lets several coexist + selectable).

- **Phase A — data model + read side.** A `<recordType>-scorecard` record `{ id, name, gates[], fitness[] }`;
  seed a "Default" from `manifest.gates/fitness` on first load; a per-project **active-scorecard** selection.
  Rewire every READ to the ACTIVE card, not `manifest.gates` directly: the viewer badge/filter/sort + the
  engine verdict layer (`recordsToAnalysisRuns.accepted` / `diagnoseSearch` / convergence gate) resolve the
  active card's gates/fitness. Run detail shows the run scored against ALL cards (active highlighted).
- **Phase B — the Scorecards tab + editor.** A dedicated tab: list cards; create/clone/edit (gate editor =
  metric picker from run metrics + operator + threshold; fitness editor = metric + max/min); delete; set-active.
  Editing is CRUD on the scorecard records (declare `<recordType>-scorecard` editable/creatable in the
  data-capability manifest so the chat AI can drive it too). `computeScorecard(gatesFitness, run)` already takes
  gates/fitness — so the only engine change is RESOLVING the active card, not the evaluation.

### A2. Full AI action parity — anything the user can do, the AI can do

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
   "launch an approved `-xai-suggestion`" (today only a human click launches a suggestion — see the xAI Suggested surface).
3. **Missing verbs** (no activity/record path today): `deleteRuns` (destructive — ALWAYS explicit approval),
   `invalidateRuns`, `migrateTrainingRuns` — add as approval-gated activities/tools.
4. **Parity audit (prevent regression).** A test that enumerates the viewer's `startOrEnqueue(<type>)` +
   `putData`/`deleteData(<recordType>)` sites and asserts each is declared in the data-capability manifest (or
   explicitly exempted as code). Codify the rule in `docs/model-training-standard.md`.

### A3. AI companion — drive the whole loop as a conversation with approvals

**Goal:** a long conversation where the AI researches/thinks/runs and the human mostly approves. Built on A2
(full action parity) + the approval gate.

1. **Orient tools.** Promote the shipped `diagnoseSearch` to a chat tool (the AI reads the same Diagnosis plan
   the user sees) and let it rank candidates on the A1 **scorecard** (not the reward); add campaign/activity-status
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

### A4. Single-purpose template + BlackSwan-as-library

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

### A5. Follow-ons on the shipped xAI / snapshot / activity features (small + opportunistic — none blocking)

- **xAI traces:** a VIEWER surface for the per-step attention sidecar (fetch the `.attn.jsonl`, animate/scrub
  the attention over the rollout — the producer is shipped, the viewer only renders the inline aggregate +
  the snapshot index today); and a **diff-consecutive** arm on the snapshot index (lazily fetch two snapshots'
  traces and feed them to the shipped `DecisionTraceDiff` — today the index lists snapshots, doesn't diff).
- **Exploration:** **Pareto basins** in the reducer — qualify/rank basins on the Pareto front, once A1's
  multi-objective `fitness` lands (today the reducer ranks on a single scalar).
- **Datasets:** derive the `asset` lever CHOICES themselves from disk (today a manifest list — needs a manifest
  generator; the on-disk DIMMING of listed-but-absent values is shipped).
- **On-device verification (test, not build):** mobile parity pass of the conversational-hub surfaces; in-app
  Data-tab check. Both need a device.
- **Parked (need a net-new dependency):** generative counterfactual states (a GAN/VAE); step-by-step decision
  animation replay + scrubber; `seed` still counts as a model lever in the engine's fANOVA (needs a manifest
  lever `scope:'ignore'` + re-analysis — flag, don't silently change).

---

## B. Deferred — bigger implementation projects

### B1. Multi-asset portfolio / cross-sectional long-short — a SEPARATE project

The one genuine project split. BlackSwan's single-asset env hardcodes one asset everywhere, so multi-asset
needs a fundamentally different env: a **3-D observation** (`asset × lookback × features`), a **portfolio
action space** (per-asset long/short/weight), a **timestamp-aligning N-symbol data provider** (misalignment =
silent P&L corruption — unit-test against a 3-coin × 100-bar fixture), and an **A1 scorecard for a portfolio**
(reward = portfolio return minus fees + a correlation-penalty CONSTRAINT; fitness = Sharpe/Calmar — not a
"Sharpe objective" that steers directly). REUSES BlackSwan's reward components, feature engineering, SB3 algos,
walk-forward harness; ~3–4 week project, blocks none of the in-place wins. Research calls cross-sectional
long-short the strongest-edge config — promising, deliberately sequenced last.

### B2. Position-blind signal model — a SEPARATE objective

The trading line trains a position MANAGER (actions are position-gated; out-of-position output is never shaped
— `blocked_signal_ratio` ~0.95). A model whose raw per-step output IS a long/flat(/short) signal, independent
of position, is a **different objective** (the "Case 1" of A1). Probe it cheaply FIRST via A1's
forward-horizon expectancy lens on existing manager checkpoints — build this full project only if that lens
shows a generalisable per-signal edge. First confirm the clean manager (via `combo_noop_penalty`, which
drops `blocked_signal_ratio` ~0.95→~0.02 by going silent on no-ops) isn't already enough. Then: reuse the env's
`buy_sell_signal` reward family or the supervised direction head; its OWN objective (signal precision/recall/
coverage vs realized forward returns, or a signal-following backtest); likely its OWN manifest
(`trainer-signal.json`); its own verdict rule + signal-overlay chart (the verdict+objective plumbing is the
bulk, not the model). Runs with `combo_noop_penalty>0` are CLEAN-MANAGER runs — don't migrate them; represent
the split as a DERIVED `approach` facet. Compare manager vs forecaster on a SHARED signal-following backtest.

### B3. Data platform — remaining source coverage + productionization

**Remaining source coverage** — the lower-noise asset-class line (this supersedes the old "strategic
direction": the mine that enables it is largely built). Macro `rates`/`macro_core`, `majors`, `market`, and
stocks are already mined + tradeable (context/data-fusion + the shipped stocks class). Left: make **commodities
+ FX** tradeable classes (catalog + `asset` lever + backfill, per the shipped stocks pattern — absorbs the old
"other single assets" item) and mine the **fundamental panels crypto lacks** — earnings, COT/inventories, trade
balances, an event calendar — each under the correctness rules below. The edge-search across these classes is
the exploration autopilot's job + the Diagnosis cross-class read, not a build here.

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

### B4. Code-change risk model — a third ML consumer (research first)

A trainer-conformant project scoring an agent's diff/PR by bug-likelihood. (1) Research: survey public
JIT-defect datasets vs mining our own from the `thefactory-*` git histories → cited report + go/no-go.
(2) Data (via the mine): SZZ-style labeling + codeIntel features (churn, complexity, coverage, diff size).
(3) Train: a `risk-classifier` project (objective = AUC / precision-at-k). (4) Consume: wire the score into
the review / expert-panel / verifier path. _Further out: a FastContext-style repo-explorer subagent — gated by
the same go/no-go + LLM-training compute the ComputeRunner has never exercised._

### B5. Optional + small deferred

- **Live handoff** — tag the exploration autopilot's global-max checkpoint for live trading (`run_server_model.py`).
- **Jupyter notebooks (UNDERSCOPED)** — view/edit/execute a project's `.ipynb`; scope kernel location + security.
- **Runner-channel WebSocket upgrade** — dispatch is already ~instant; a WS only shaves ~1.5s log latency.
- **Remote git repoRefs** — wire git refs + project bootstrap when a real remote machine needs it.

## C. Open questions + trigger-blocked builds (act when the dependency lands)

- **Context — `with_extra_data` projection rung** (blocked on data). The 4th projection rung: an asset's OWN
  fused series + the obs-signature gate as a REAL replay guard. Every context panel today is GLOBAL
  (macro/majors/market); no per-asset series is mined (`fundamentals/`, EDGAR are empty). The fusion substrate
  + loader already exist, so the rung is small — build it once ≥1 asset-specific series is mined.
- **Context — post-release window restriction** in `config_builder` (blocked + currently a no-op). Bar
  pre-release context bars via `fillna(0)`. Every context series on disk is deep-history (FRED to 1776;
  GOLD/SPY from 2018), so `fillna(0)` never zeroes a real signal — build it when a LATE-STARTING context series
  is mined.
- **Host→iframe `data:updated` push channel** — the viewer is poll-only; the bridge forwards only `nav.open`.
  A push channel (thefactory-ui `appBridge` + `ProjectAppView` + `bridge.js`) would let the viewer live-refresh
  on a data change instead of polling. Unblocks the cross-test (`-settest`) auto-refresh (the viewer half is a
  trivial `onDataUpdated` handler debouncing a `readCrossTests()`) + trims poll latency generally.
- **Remote artifact/checkpoint storage** — keep-on-runner + reference vs upload; how a winning remote
  checkpoint reaches the live trading server. Meaningful once remote runs AND live handoff both exist.
- **GPU + sandbox profile for training images** — `--read-only` rootfs vs ML caches; `--gpus` is wired but
  unexercised (the runner runs jobs directly, not through Docker-sandboxed `SandboxTools`).
- **Judge/proposer model transport** — `ModelSelection` (API vs CLI), overtaken by the in-flight refactor.
  Revisit once the CLI inference stage lands.
