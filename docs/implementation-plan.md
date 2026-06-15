# thefactory-modeltrainer — implementation plan

Remaining work only. What's built and how it fits: `docs/architecture.md`. The contract:
`docs/model-training-standard.md`. The core loop (Phases 0–7) is **built + verified**: the
engine, backend activities, viewer, remote runner, and three conformant consumers —
`examples/cartpole`, `examples/tabular`, and **BlackSwan** (`/Users/cloud/Documents/Work/BlackSwan`,
the trading line, Phase 7 done). The engine stays domain-oblivious so any further model is
_data + the thin CLI contract_, not engine code. What's below is optional phases, small
cleanups, and deferred new work.

## North star — two co-equal outcomes

1. **Make the pipeline/app/project the best it can be for creating models** — for ANY model, end to end
   (propose → run → judge → explore), with excellent, self-explanatory results + comparison UI, and a data
   layer that stores the minimum and derives the rest at runtime. **The data layer must also GUIDE a user
   who has a problem to solve but doesn't know what data to mine or how** — walk them from "here's my
   problem" to "here's what data exists, what to mine, and how", via deep research + exploration of
   available sources (see "the data mine" below).
2. **Use it to make BlackSwan the best trading model.** Simple, in STRICT ORDER: **(A) correctness first**
   — all the data, all the rewards, all our processes must be CORRECT before any result is trusted;
   **(B) find ONE setup that trades well** — a config that trades often + profitably, stable across seeds,
   vs buy-and-hold; **(C) then a huge space exploration** of the toggles to find the best. **Measurable
   progress is the whole point** — that's what the model trainer is for.

Co-equal — don't trade one for the other; the BlackSwan work is the forcing function that hardens the
generic pipeline. The BlackSwan section below is now structured around the A → B → C order.

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

## Hub UI — open follow-ups

(The A1–A3 / B1–B2 / C1–C4 roadmap shipped — see git.)

### "Activity & concurrency center" — one dedicated pass (with LIVE verification)

Deep investigation (2026-06-15) found these three "follow-ups" share one root and can't be built well in
isolation/blind — they need the Activity surface to become a host-aware, browsable, concurrency-capped center:

- **Global concurrency budget (#3).** Needs BOTH a pump rework (today strictly serial: dispatch → await full
  settlement → next) AND an Activity-rendering rework (assumes exactly one live campaign —
  `currentActivityId`/`lastProgress`/`lastCampaign`). Pair with a backend semaphore on
  `LocalComputeRunner.runJob` (the shared `trainerLocalRunner`). Reworks the just-stabilised pump — needs live
  test. (#1, evals one-at-a-time, is already fixed by the batch parallel-evaluate.)
- **App-nav unseen-results badge.** The mapped "app reports unseenCount" design is BROKEN: the embedded viewer
  unmounts when you leave the app tab, so it can't report results landing while you're away. Correct design is
  HOST-DERIVED: count trainer activities finished (server-side `finishedAt`) since the user last opened the app
  tab → Sidebar app-tab badge (web + mobile). Generic (uses activity timestamps, not trainer schema).
- **Run→Activity link (#4).** Blocked: the Activity tab isn't a browsable per-activity history, so a link from a
  finished eval/verdict has no real target (would just show the current campaign — misleading). Build the
  history/addressable-activity view first, then `activityId`-tag eval/verdict records + link into it.

Why one pass: all three want the same substrate (browsable activity history + per-activity identity + a global
budget + completion-derived state) + live verification against a running backend. Doing them piecemeal/blind
ships flawed UI.

(Closed: **Model chip recents + reactivity (#5)** — shipped: `recentActivityCliModels` in LLMConfigsContext +
follow-active-agent in `useActivityChipCli` + recents quick-pick in the web Picker & native BottomSheet
(web/desktop/mobile; device-verify owed). **One-click "AI help" on a failed run** — shipped: `discussTopic`/
`requestChatSidebar` added to the viewer bridge + an "Ask AI for help" button in run-detail that seeds the
host chat with error + logTail + config (host handles `chat.discuss` on web + mobile). **B1 conditional-best**,
**evaluate command**, **Ledger-note affordance** — see git.)

## Runs + failure recovery + pipeline versioning (PRE-PHASE-B — surfaced from live use 2026-06-15)

Real pain hit while running a campaign with many failures.

SHIPPED:
- **Failed-run/activity recovery.** Activity failures are now per-entry **See error** (→ run detail with the
  full `error` + `logTail` + Ask-AI-for-help), **Re-run** (clone the exact config to Launch — also from the
  failed-run detail), and **Dismiss** (persisted `*-dismissed-failure` records, filtered from the list).
- **Mark a setup UNRUNNABLE.** Run-detail "Mark unrunnable/runnable" writes a version-scoped
  `<recordType>-unrunnable` marker (keyed by setupKey); `runTrainingCampaign.isFresh` skips marked setups
  unless `refresh`, and a pipeline-version bump ignores older-version marks. 6 TDD tests.
- **Pipeline versioning + changelog.** `TrainerManifest.pipelineVersion` + `pipelineChangelog[]`; every run is
  tagged with its version; `skipExplored`/`unrunnable` are version-scoped (a breaking bump re-opens everything,
  since scores aren't comparable). BlackSwan declared **v2** (this session's data/scoring overhaul) with a
  changelog vs v1 (the historical ledger).
- **Runs columns + filter.** **%return + #trades lead** the metric columns, both colour-coded like `vs hold`
  (return by sign; #trades green when it actually traded, red when ≤2 ≈ hold). A **Hide bad runs** toggle drops
  failed/errored + degenerate (≤2-trade / health-flagged) runs.

- **Runs master-detail LAYOUT.** SHIPPED: the Runs tab is now full-width (`.tab-main.is-fullwidth` drops the
  centred max-width) with a `.runs-md` grid — a left LIST pane + right DETAIL pane that each scroll in their
  own view; the panes split (`has-detail`) only when a run/compare is open, else the list spans full width;
  the list narrows on select (table scrolls horizontally) and stacks on narrow screens. NEEDS LIVE VISUAL
  CHECK (built without a render). Follow-up: a true column-condense (fewer columns when selected) — today it's
  narrow-pane + horizontal scroll.
- **Pipeline versions VIEW.** SHIPPED: a collapsible "Pipeline v{N} — versions & changelog" panel above the
  runs (current version + breaking-flagged entries), each run-detail shows the version it ran under, and
  Compare warns when selected runs span multiple versions. Follow-up: a version filter/dropdown + a dedicated
  per-version leaderboard (today: per-run version + the by-setup/by-experiment groupings).

## BlackSwan — the path to a trading model (A → B → C)

Strictly ordered: **(A) make everything correct → (B) find ONE setup that trades well → (C) explore the
toggle space for the best.** "Tried" is data-driven — every run records `config` + `setupKey` + `metrics`;
`skipExplored` prevents re-running a setup; the by-setup / by-experiment ledger is the memory. Full
pipeline map + holes: `docs/blackswan-pipeline-map.md`. (Context from the 2026-06-13 scrutiny: the model/
reward/feature space is already heavily explored — ≈15 RL algos, the combo family, many TP/SL variants;
indicators "tried and didn't help"; minute data ≈ noise — so the leverage is correctness + a disciplined
search, NOT more feature-richness.)

### Phase A — Correctness (the gate: no result is trusted until this passes)

Shipped this round (data + rewards + processes): fidelity-scaled rolling windows (QW1) + taker order-flow
(QW2); 1m-canonical derive+cache for fidelities (QW6, derived==native to 0.000000) + runtime-computed
indicators (verified == the emitter to ~1e-8); trade-aware `traded_return` objective + `few_trades`/
degenerate health; `combo_all_fee` reward variant; wider test window + per-window robustness (RB7); the
experiment ledger (by-setup median/IQR/stability, by-experiment, conclusions); + the historical
`results_hour.ods` import. (QW5: `model_config.py` default restored to the winning `[512,64]`.)

**The scrutinous correctness audit RAN (2026-06-15, 3 adversarial readers — data/rewards/processes).**
FIXED this round:

- **`n_trades` miscount (HIGH, corrupted the objective).** `base_crypto_env.py:583` returned
  `state[17] = (len(buys)+len(sells))/2` (fractional, e.g. 3.5) instead of the completed-trade count.
  `state[17]` feeds `trade_gate`, `traded_return`, the displayed `#trades`, AND the `few_trades`/`zero_trades`
  health flags — every gated objective + health verdict was off. Now returns `total_trades`.
- **Benchmark / chart lookback misalignment (HIGH, display-only).** `_run_prices` sliced `provider.prices[:n]`
  (offset-blind), but `SingleDataProvider.get_price(step) = prices[step + get_start_index()]` — so buy-and-hold
  + the chart were computed over a window shifted by `get_start_index()` bars on the single-provider path
  (empirically confirmed: old window started in the lookback region; MultiTimeline pre-strips its prices so it
  was already aligned). Fixed by indexing through the provider's own `get_price(i)` — the exact mapping the env
  trades on — correct for both providers.
- **`combo_noaction<0` synthetic-short incentive.** Default flipped to `0` (direction-neutral for a flat
  long-only agent) in `model_rl`; kept sweepable via a new `combo_noaction` trainer lever (choices `[0, -1]`)
  so the bias can still be probed deliberately.
- **Linear trade gate.** `trade_gate` is now QUADRATIC — `min(1, (n_trades/MIN)²)` — so under-trading is
  punished steeply (half-threshold keeps a quarter of its return); ≥MIN still = full credit.
- **By-setup degenerate blindness (clarity).** The hub's `aggregateBySetup` now excludes health-flagged
  (degenerate) runs from a setup's averages / best-run / surfaced verdict, annotates `(k degenerate)`, and
  marks all-degenerate setups — so a lucky 1-trade fluke can't masquerade as a setup's result
  ([[feedback_ui_ux_scrutiny]]).

**DEFERRED to after Phase B (need experiments / change the baseline — your rule: "needs more work → defer"):**
- **Calibrate `MIN_TRADES_FOR_FULL_CREDIT`** (now 20) to the real test-window length — needs a couple of
  multi-window runs.
- **Out-of-[-1,1] features on the 1h winning path.** Empirically (2024 Jan–Apr 1h, `type='only_price_percent'`),
  8 of 25 base-layer features exceed the declared `spaces.Box(-1,1)` obs bound: `volume_percent`,
  `total_volume_percent`, `volume_quote_percent`, `trades_number_percent` (pct_change spikes to +10.9),
  `price_z_score_1d/1m` (±4), `price_to_avg_1d/1m` (>1). `process_df_simple` (the 1h multi-layer path) does NOT
  filter by `type` and does not squash these (only the curated indicators + `taker_buy_ratio` are bounded).
  The audit's `_1y` / `Pi_Cycle_*_Ratio` columns are ≈0/dead with <1yr of data. SB3 does NOT clip — the NN
  gets the raw values (not a silent-clip bug). **Deferred because the historical "winning path" results were
  produced with exactly this representation; squashing changes the baseline + breaks comparability, so it's a
  normalization EXPERIMENT (tanh-squash the 8, or clip the observation — does it help?), best run in Phase B/C,
  not a blind pre-B fix.** The fix would mirror the existing `np.tanh(x/scale)` curated-indicator pattern in
  `abstract_dataprovider.py`.

Confirmed CORRECT by the audit: chronological train(2020-23)/test(2024) split; reward signals not leaked into
observations; Tier-1/Tier-2 indicators properly normalised; `taker_buy_ratio` clipped; derived-bar OHLCV
aggregation; deterministic single test per seed; median/IQR/stability math; degenerate-policy/few-trades
detection. **Phase-A gate is GREEN** — every correctness defect fixed; the two items above are consciously
deferred to after Phase B (experiments / baseline-changing). Phase B is unblocked.

### Phase B — Find ONE setup that trades well

Tools shipped: the Launch-tab experiment presets (Exp 1–5), the `use_indicators` on/off experiment,
multi-seed + the seed-stability column, the trade-aware objective + window-robustness. **Success = a config
that trades OFTEN and PROFITABLY, stable across seeds, beating buy-and-hold.** Start: `use_indicators`
on/off on 1h, multi-seed (the 1d prelim was seed-dependent, so seeds matter); read By-experiment +
By-setup; lock in the first setup that clears the bar.

### Phase C — Huge space exploration (after a Phase-B baseline)

Sweep the toggles broadly from the baseline; `skipExplored` + by-setup aggregation + the ledger keep it
self-pruning. Fold in feature bets ONLY if an experiment says they help: **RB1b/2** (compute more
indicators in `src/data/indicators.py` + `_add_curated_indicators`), **RB5** (causal `dip_score` into the
trading env, `[L]`).

**Ground rules (do not violate):** profit is the objective, never beat-hold (hold is a display control),
expressed as the trade-gated `traded_return`; trade **often and well** (a 1-trade run ≈ hold); the reward
family is intentional — add variants, never collapse; BTC-only until the data mine backfills altcoins.
`combo_noaction=-1` is a latent synthetic-short bias — sweep as a variant, don't edit in place.

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
(mine → persist → facade). **Architecture decision (2026-06-14): the data mine stores MINIMAL raw OHLCV
only — it does NOT bake in or version indicators.** Indicators (and higher fidelities) are DERIVED AT
RUNTIME in the consumer (`src/data/indicators.py` + `trainer/derive_cache.py`), so the stored/transferred
data stays small and an indicator fix is a one-line code change, not a re-derivation of gigabytes. So the
data mine's remaining job is: (a) **gather + clean raw OHLCV** — gap/dedup/timestamp-continuity checks,
inf/NaN sanitisation, and mining the missing intervals (the emitter's indicator engine becomes a REFERENCE
for the runtime formulas — already ported + verified — not a storage artifact); (b) **generalise the
QW6 `derive_cache`** — 1m canonical, derive+cache higher fidelities centrally so consumers never
re-aggregate; (c) the content-addressed cache + remote-runner data path pulling from one curated origin.

**Guided data discovery (north-star outcome 1).** A user often has a PROBLEM to solve but doesn't know
what data to mine or how. The data mine must guide them from "here's my problem" to a concrete dataset:
(1) take the problem statement + the model goal; (2) run **deep research / exploration of what data
actually exists** (sources, APIs, public datasets, coverage, cost, licence, granularity) — reuse the
deep-research harness; (3) propose candidate datasets + what to mine + how (the miner config) + the
trade-offs; (4) hand off to the gather→clean→cache pipeline above. This makes "I have a problem" → "I have
training data" a guided flow, not tribal knowledge — the data-side analogue of the hub's propose→run→judge
loop. Output a short cited report + a recommended mining plan the user approves.

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
