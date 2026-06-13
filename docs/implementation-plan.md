# thefactory-modeltrainer — implementation plan

Remaining work only. What's built and how it fits: `docs/architecture.md`. The contract:
`docs/model-training-standard.md`. The core loop (Phases 0–7) is **built + verified**: the
engine, backend activities, viewer, remote runner, and three conformant consumers —
`examples/cartpole`, `examples/tabular`, and **BlackSwan** (`/Users/cloud/Documents/Work/BlackSwan`,
the trading line, Phase 7 done). The engine stays domain-oblivious so any further model is
_data + the thin CLI contract_, not engine code. What's below is optional phases, small
cleanups, and deferred new work.

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

## Open verification (BlackSwan-improvements round)

The round shipped — all seven improvements (seed plumbing, parallel-runs concurrency,
data-visibility `asset` lever + `dataset` descriptor, the price-action `runChart` + custom per-run
UI, lean run-health, the dip/regression line as a second hub project, the `quickStart` preset).
The Python paths and the engine/backend changes are verified by real runs + unit tests. One pass
remains: run the hub **embedded in Overseer** and confirm the viewer additions render live — the
custom price-action chart + dataset badge in run detail, the "Max concurrent runs" control, the
"Quick start" button, and registering the dip line as a second project (same repo, manifest file
`.factory/trainer-dip.json`). Build + typecheck + tests passing is not the same as a live embed.

### Deferred (consciously, from this round)

- **Full RL resume** — per-episode RL checkpointing + `set_env` continuation for true mid-training
  resume. Lean run-health shipped (a failed run is recorded + re-dispatched, and the regression
  line resumes from its per-episode checkpoint), but the RL trading line still restarts from zero.
  Revisit if mid-run RL continuation is worth the training-loop surgery.
- **True multi-asset** — only BTCUSDT has 1d/1h klines, so the `asset` lever is BTC-only today.
  The lever + per-symbol globs + `data_inventory` capability gating are root-cause-correct; altcoin
  1d/1h backfill is the data mine's job, after which the lever expands with no code change.

## Future path & UX scrutiny (after the BlackSwan improvements above)

**Massive scrutinisation task and critical look to decide on the future path for the project, as
well as UI/UX usability improvements.** Once the improvements above land and there are real
multi-asset / multi-timeframe results to read, step back and look hard at the actual run data:
judge whether the current experiment design, objective, and levers are the right ones, and
propose the next campaigns to run — expect this may force big changes to how experiments are run.
Pair it with a usability pass over the hub app and the per-run result UI.

## Hub UI — SHIPPED (2026-06-13), with open follow-ups

The full Hub UI roadmap (A1–A3, B1–B2, C1–C4) and the Results-workbench UX-clarity gaps all shipped
(see git). The Results view now has three modes — **Runs** (sortable/filterable, dynamic metric
columns + `vs hold` + per-header help), **By setup** (config-minus-seed aggregation: seed count,
min/max/avg/median, vs-hold, plus the LLM-verdict + your-conclusion ledger columns), **By experiment**
(group by launch thesis) — plus multi-select compare, clone-to-Launch, per-run progress + ETA, tab
spinners, lever descriptions + best-so-far annotations, skip-explored default-on, and per-setup
conclusion notes (`{recordType}-note` records, edited when drilled into a setup).

**Open follow-ups (not blocking; each needs a clear next step or trigger):**

- **B1 conditional-best** — choice options are annotated with the marginal best-so-far value (★), but
  not the value that is best _conditional on the other currently-selected levers_. Needs a live
  re-render of the form on each lever change (recompute from history filtered to the current
  selection). Moderate; do when the lever count makes marginal-best misleading.
- **A1 live concurrency resize** — concurrency is set per-campaign on the Activity tab, but cannot be
  changed on an _already-running_ campaign (the bounded pool is fixed at launch). Engine work: the
  worker pool would need to accept a resize signal mid-run.
- **Ledger note affordance** — the by-setup "Your conclusion" cell shows `add note ✎`, but editing
  happens after clicking the row to drill in (the editor sits above that setup's runs). This gives
  full context but the cell looks inline-editable. Decide: leave (context-rich) vs inline edit.

### New-user UX review of C3/C4 (2026-06-13) — fixed inline

Applied the `feedback_ui_ux_scrutiny` lens to the just-built thesis/ledger work; the two real
newcomer traps were fixed in the same pass: (1) **By-experiment looked broken** for a project whose
history predates thesis-tagging — every old run fell into one `(untagged)` row; now that case shows a
"set a Thesis at launch" empty-state instead. (2) **The ledger columns were unexplained** — the
by-setup legend now says the LLM verdict needs the judge to have run, `—` = not yet scored/noted, and
that clicking a row is how you write a conclusion. The launch Thesis / "Testing which setting?" fields
already carry help callouts + "(optional)".

## Ongoing Research — what to try next (BlackSwan)

The live backlog of experiment-design improvements, from the 2026-06-12 scrutiny (real results in
`BlackSwan/results_hour.ods` + a grounded code analysis). **"Tried" is data-driven, not a manual
list** — what's been run lives in the run records (`config` + `setupKey` + `metrics`), and the
`skipExplored` campaign option already prevents re-running a setup under a new seed. Prune an item
here once its result is recorded; add ideas as they surface.

> **Reframing (2026-06-13):** the commented-out experiment log shows the model / feature / action /
> reward space is already **heavily explored** (≈15 RL algos incl. distributional + recurrent; the
> combo reward family; many TP/SL/trailing variants; indicators tried and **didn't help**; minute
> data behaves like **noise**; `lookback1` worse; 15m layer did nothing). So feature-richness items
> (esp. RB1) are largely **re-treading**. The genuinely under-explored, highest-value holes are
> **(1) evaluation rigor** (selection-on-test leakage, single regime, single seed → results not yet
> trustworthy), **(2) problem formulation** (RL-on-noisy-price is known-hard), and **(3) a systematic
> thesis-testing UI + experiment ledger**. Full map + holes: `docs/blackswan-pipeline-map.md`. The
> quick wins below stand as correctness/measurement fixes (QW3/QW4 done); the deep research targets (1)+(2).

**The one structural truth:** the path that beats hold (1h multi-layer) runs the _impoverished_
feature pipeline (`abstract_dataprovider.process_df_simple`) — it ignores `config.type`/`indicator`,
drops the 112 precomputed indicators + taker order-flow, and applies minute-indexed rolling windows
to hour-aggregated frames (so `price_z_score_1m/1y` are constant-zero). Big nets lose / small nets win
because the bottleneck is **feature content, not model capacity** — most items below converge here.

### Experiment-recording system (so a run is never repeated) — to build

Every run already records `{config, setupKey, metrics, health, dataset}` and `skipExplored` dedupes by
setup. Missing for a _complete_ ledger: (a) a **"tried setups" view** in the hub listing every config
run + its outcome (queryable, sortable by objective, so "what's been tried" is visible); (b) a one-off
**import of the historical `results_hour.ods` runs** into that ledger so pre-hub experiments count as
tried; (c) surface `setupKey` collisions ("this setup ran on N seeds → median/IQR"). This is the
data-mine-adjacent infrastructure that makes the backlog below self-pruning.

### Quick wins (correctness + cheap information; low risk)

- **QW1 — fidelity-aware rolling windows** `[HIGH/S]` — `process_df_simple` windows are minute counts
  applied to hour bars, so `price_z_score_1m/1y` are NaN→0 constant columns. Scale by minutes-per-bar.
- **QW2 — taker (aggressor) order-flow features** `[HIGH/S]` — `asset_volume_taker_base/quote` are in
  every kline but dropped (`# for now lets ignore these`). Add `taker_buy_ratio` (bounded [0,1]).
- **QW3 — fix `reset_num_timesteps`** `[HIGH/S]` — `rl_model.py` re-warms `learning_starts=20000` each
  episode and never finishes annealing epsilon; pass `reset_num_timesteps=(i==0)`. Latent bug; re-baseline.
- **QW4 — in-process buy-and-hold benchmark in every summary** `[HIGH/S]` — free (reuses
  `provider.prices`); display-only control, NOT a reward target.
- **QW5 — restore the winning `net_arch`** `[MED/S]` — `model_config.py` has the loser `[8192,512]`
  active with winners commented; the manifest default now overrides for hub runs, but the legacy
  `main.py` path still reproduces the −40% loser.
- **QW6 — use native 1h klines** `[MED→HIGH/M]` — 85 `BTCUSDT-1h-*.json` exist but the 1h path loads
  ~1.5GB of 1m JSON to rebuild bars that already exist.

### Research bets (higher potential; bigger or uncertain)

- **RB1 — curated indicators into the winning path** `[HIGH/M]` — the 112 precomputed indicators (the
  emitter's real output, present in every kline) are unreachable on `process_df_simple`. They tier by
  clamp-safety (measured ranges, from the `BlackSwanPriceEmitter` analysis): **Tier 1 = drop-in free**
  (rsi/williams/stochastic/choppiness — natively `[-1,1]`, 0-centered); **Tier 1b = light rescale**
  (bollinger/donchian price-ratios ~[0.85,1.17]; kallman/disparityIndex live in ±0.03 → near-inert
  under the clamp); **Tier 2 = clamp-hostile, needs RB6 first** (meanReversion ±4, cci, turbulence
  0–15, obv −2.7, sortino→inf). Start with Tier 1, then RB6, then 1b/2.
- **RB2 — selection + held-out validation window** `[HIGH/M]` — `iterations_to_pick_best` runs 10× with
  NO selection; a human cherry-picks the luckiest-of-10 _on the test set_ (leakage). Keep the argmax on
  a third `val` split; score = risk-adjusted profit, never delta-vs-hold.
- **RB3 — multi-seed aggregation** `[HIGH/M]` — small-net RL swings +50→+60% on seed alone; report
  median/IQR/fraction-positive + an `unstable` sign-flip flag for shortlisted configs.
- **RB4 — fee/turnover penalty reward _variant_** `[HIGH/S]` — no reward branch has an explicit fee
  term; the losing big-net over-traded. Add as a NEW variant (serves profit-net-of-cost).
- **RB5 — feed the working dip score into the trading env** `[HIGH/L]` — the f1-0.67 classifier is a
  dead-end signal; add a causal `dip_score` observation, or a cheaper inference-only buy-veto.
- **RB6 — rescale features instead of hard-clamping obs to [-1,1]** `[HIGH/S→M]` — **now a PREREQUISITE
  for RB1's rich features, not a parallel nice-to-have.** The emitter analysis measured the best
  indicators as either clamp-hostile (meanReversion ±4, turbulence 0–15) or compressed-to-inert
  (kallman/timeseriesMomentum ±0.03), so feeding them under the current `[-1,1]` clamp feeds dead or
  clipped columns. Apply a per-feature tanh/robust-quantile squash identically at train+test.
- **RB7 — expand the test window beyond 2024-Q1** `[HIGH/M]` — a single bull window can't show profit
  robustness; add post-train 2024 Q2–Q4 windows, report per-window worst/mean/fraction-profitable.
- **RB8 — sanitise indicator data quality at the source + version the indicator spec** `[MED/M]` — the
  emitter bakes normalisation into stored values and emits `inf`/`NaN` (sortino/volatilityVolume
  divide-by-zero) that BlackSwan only silently zeroes downstream; and `config.json` has already drifted
  from the compute engine (declares atr/sar/ema/macd/pump/dump that are commented out). The data mine
  must sanitise inf/NaN, prefer storing raw values + a declared normalisation spec, and stamp an
  **indicator-spec version** so drift can't silently corrupt training data.

**Ground rules (do not violate):** profit is the objective, never beat-hold (hold is a display control);
the reward family is intentional — add variants, never collapse; BTC-only until the data mine backfills
altcoin 1d/1h. `combo_noaction=-1` is a latent synthetic-short bias — sweep as a variant, don't edit in place.

**Recommended next (revised after the `BlackSwanPriceEmitter` analysis):** QW1 → QW2 → (QW5 + QW3,
re-baseline) → QW4 → **RB1 Tier 1** (free clamp-safe indicators) → **RB6** (rescaling framework) →
**RB1 Tier 1b/2** (the rich indicators that need it) → RB2. The features sequence changed: the
high-value indicators are mostly clamp-hostile, so RB6 moves ahead of the bulk of RB1.

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
(mine → persist → indicator-compute → facade) but needs three upgrades to be the data mine: (a) **persist
+ version indicators as a first-class artifact** — today they're recomputed-on-read / live-only (the
`raw/indicators/` backfill is commented out) and `config.json` has drifted from the compute engine, so
stamp an indicator-spec version; (b) **data-engineering BlackSwan can't do at source** — gap/dedup/
timestamp-continuity checks, inf/NaN sanitisation (sortino/volatilityVolume divide-by-zero), server-side
resampling (which would retire BlackSwan's `process_fidelity` + the QW1 window bug), and mining the
missing 1s/15m intervals the config promises; (c) **store raw indicator values + a declared normalisation
spec** so the consumer applies clamp-aware scaling (RB6) instead of fighting a baked-in `÷price`/`×2−1`/`/100`
transform. This is the durable home for the Ongoing-Research feature work above.

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
