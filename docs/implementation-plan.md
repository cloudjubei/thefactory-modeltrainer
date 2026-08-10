# thefactory-modeltrainer — implementation plan

**Remaining work only** — shipped history lives in git + memory. Architecture: `docs/architecture.md`.
Contract: `docs/model-training-standard.md`. The engine stays domain-oblivious — any further model is
_data + the thin CLI contract_, not engine code.

## North star (frames prioritization)

1. **Best generic pipeline for creating ANY model** end to end (propose → run → judge → explore), a
   self-explanatory results UI, a minimal-storage data layer, and guidance from "here's my problem" to
   "here's what data to mine." **SHIPPED and hardened** — incl. the hypothesis/verdict trail with declared
   gates + Deflated-Sharpe multiple-testing correction, the side-experiment framework, and cross-project tooling.
2. **Use it to make BlackSwan trade well — now a COMPREHENSIVELY MEASURED result: no cost-surviving edge exists
   in the free data we hold.** **21 gate-backed, adversarially-verified nulls** across price (RL / supervised /
   deterministic / intraday), positioning (funding), microstructure (order-flow), scheduled events (macro
   releases), mechanical (liquidation cascades), macro state (regime timing), and attention (Wikipedia). The
   unifying finding: everything **public, price-derived, scheduled, mechanical, or macro is priced in or noise**
   — order-flow proved it structurally (imbalance is *contemporaneous*, not predictive; a same-bar peek returns
   +558% vs the honest t+1's +61%). The only genuinely-untested channel left is **LLM-read directional crowd
   sentiment** (§B3, Reddit via Arctic Shift — data is feasible but the attention null makes it low-EV). **The
   trading question is essentially answered.** The durable asset is the *evaluation engine* that produced these
   nulls — leakage control, Deflated-Sharpe multiplicity correction, walk-forward OOS, pre-registered gates, and
   adversarial verification — and hardening THAT into a first-class, generic ML-experiment-evaluation capability
   is the go-forward mission (**§C**), a direct deepening of #1. Its strived-for OUTPUT is a scientific
   publication — the nulls, plus a reproduce-and-refute of the papers that claim an edge exists (**§D**).
3. **Fully AI-operable, shipped as a template. SHIPPED** — A5 complete: modeltrainer is the base + a one-time
   seed; BlackSwan runs as its own single-purpose app.

## Repo split (governs where work lands)

| Repo | Owns |
| --- | --- |
| **thefactory-modeltrainer** (this repo) | `ModelTrainerTools`; matrix planner; campaign loop; judge/propose orchestration; the viewer; the standard + `examples/`. |
| **thefactory-tools** | Generic infra: `ComputeRunner` seam (+ `RemoteComputeRunner`, `ContentAddressedDataCache`, pairing); work-item engine. |
| **thefactory-backend** | Activity registration + composition; app-view serving; PIN-pairing + runner WS channel. |
| **BlackSwan** (the trading repo) | Its `TrainerManifest` + additive `trainer/` CLI conformance. No Overseer code. |

---

## A. Platform — COMPLETE

The core loop (engine, backend activities, viewer, remote runner), three conformant consumers
(`examples/cartpole`, `examples/tabular`, **BlackSwan**), the hypothesis/verdict trail (gates:
`beats-hold` / `majority-beats-hold` / `deflated-sharpe`), the side-experiment framework, cross-project tooling,
and **A5** (base + one-time seed; BlackSwan as its own single-purpose app) are all shipped — detail in git +
memory. **One runtime step outside the repo:** set BlackSwan's overseer project `metadata.hasApp=true`,
`metadata.appDir="app"`. New platform work gets added here as it arises; there is no pending platform backlog.

---

## B. The trading frontier — the information layer (price, at every frequency, is exhausted)

Everything in §B is gated the SAME way as the price work: a **cheap pre-registered probe (hypothesis + declared
gate, DSR-corrected) BEFORE any build**; a null is a result, recorded in the trail, never re-litigated.

### Where PRICE stands — CLOSED by measurement (do not redo)

Recorded as gate-backed nulls in the trail, so we never repeat them:
- **Single-asset directional timing** (RL PPO/DQN/recurrent/transformer, supervised GBM, deterministic
  momentum/MA/breakout) — null, DSR-corrected. NB: trained models only ever ran on **BTC**.
- **Cross-sectional long/short + reversal** (the old "multi-asset env / B1" premise) — measured null; the
  3–4-week multi-asset-env build it would have justified is therefore **not authorised**.
- **Position-blind signal model** (old "B2") — the per-signal edge is real but **below the cost floor**
  (proven-null `e4ed1bb153b6`); building a forecaster changes nothing.
- **Volatility-as-sizing** and **cross-class risk-parity** — both disproved; the diversification premium is
  already captured by simply holding the basket.
- **Intraday decision-frequency** (breakout-momentum + high-vol-regime, 8h/15m/60m bars — the frequency
  BlackSwan actually targets) — measured null: per-trade expectancy is NEGATIVE *before* cost and cost is
  catastrophic at that cadence (`probe-breakout-momentum`, `probe-regime-conditional`). Intraday is worse than
  daily, not better; the cost floor is the enemy, not the missing frequency.

### The forward program — build the INFORMATION LAYER around price (price alone is exhausted)

The one thing never tested with real effort: does **non-price information** carry an edge, fed to the trading
model at the short-term frequency? Four directions, each probe-first:

**B1. Continuously-gathered data streams — "is there data we can ingest continuously that gives an edge?"**
A CONTINUOUS ingestion + feature pipeline (not a one-shot backtest) over accessible non-price streams that
plausibly carry signal price lacks: crypto **on-chain** (exchange net-flows, active addresses, whale transfers),
**perp funding rates + basis**, **order-flow / taker imbalance**, cross-exchange spreads. Probe each first: does
the stream predict a move that clears cost, DSR-corrected? **Leakage trap to pin:** many vendor feeds are
RETROACTIVELY revised (address labels, reorg-adjusted flows) = look-ahead baked in — only "what was knowable at
the bar." Build the pipeline only for a stream whose probe survives.
- **Perp funding rate — PROBED, measured null** (`probe-funding-contrarian`, `probe-funding-momentum`). Real
  Binance 8h funding (BTC/ETH/SOL, 2022→, mined to `binance-funding/`), `trainer/funding.py` (reuses the
  intraday mechanics; three mutation-proven leakage guards: tolerance-join dropping off-cycle 4h fundings,
  past-only funding quantile, apply-to-t+1). Neither fading nor riding a funding extreme beats hold net of
  cost; the one positive cell is a one-window coinflip that DSR deflates to 0. Adversarially verified (3
  independent skeptics + synthesis: null-survives; holds even at a realistic 0.04%/side perp fee). **Do not redo.**
- **Order-flow / taker imbalance — PROBED, measured null** (`probe-flow-momentum`, `probe-flow-contrarian`).
  Zero-fetch: the 1m files already carry `asset_volume_taker_base`, so imbalance = `(2*taker_buy-volume)/volume`.
  `trainer/orderflow.py` + the shared `trainer/signal_extremes.py` past-only core (mutation-proven; funding
  refactored onto the same core). Neither riding nor fading an imbalance extreme beats hold net of cost;
  momentum's one positive cell is a one-window coinflip, contrarian is 0/48. Adversarially verified
  (null-survives). **Key structural finding — why this is null and likely why ALL pure-flow microstructure
  will be:** taker imbalance is **contemporaneous, not predictive** — it correlates +0.42..+0.52 with the
  SAME-bar return (a same-bar peek returns +558% vs the honest t+1's +61%), i.e. it tells you where price
  *went*, not where it's *going*. It adds nothing a price signal doesn't already carry. **Do not redo.**
- **REMAINING B1 streams (only if a cheap probe motivates the mine): on-chain + funding-basis.** These are the
  streams that are genuinely EXOGENOUS to price (exchange net-flows, whale transfers, active addresses;
  perp-vs-spot basis term structure) rather than contemporaneous with it — the one class not yet ruled out.
  They need real mining (Glassnode/CryptoQuant-class feeds or a node; basis needs spot+perp mark), so probe
  the cheapest proxy first. NB the two on-disk/near-free streams (funding, taker-flow) are now both null.

**B2. Events / triggers to watch for — "are there events that open an exploitable window?"**
Recognizable events — SCHEDULED (FOMC/CPI, earnings, perp funding resets every 8h, token unlocks, listings) and
DETECTED (large liquidations, whale moves, exchange outages) — that create a window where the expected move ≫
cost. Build an event calendar + detector; test event-CONDITIONED trading (trade only in the window). Probe: does
the event window carry a directionally-persistent move net of cost?
- **Scheduled MACRO releases (CPI/jobs/retail/PCE/GDP) — PROBED, measured null** (`probe-event-drift`,
  `probe-event-fade`). Zero-fetch: `trainer/events.py` reuses the on-disk FRED point-in-time releases +
  `pit_fusion` (release datetime, DST-aware) + the intraday mechanics; the reaction to the first post-release
  bar is ridden/faded. Neither drifts nor fades net of cost. **Trading only in event windows finally tames
  cost (~1000 bps/yr, not 14000), so cost is NOT the blocker here — the per-trade EDGE is, and it's absent.**
  Adversarially verified (null-survives): the exposure gate is confounded (a ~99%-flat book loses to a bull
  hold, "beats" a bear hold by absence), so the decider is per-trade expectancy — mean NEGATIVE, and the
  handful of positive-both-window configs are thin-sample (CPI ≈11 events/yr) multiplicity flukes (best |t|
  1.874 < the 3.17 a 54-cell noise sweep expects). **Gate-bias lesson recorded:** `majority-beats-hold` is
  structurally unfair to SPARSE/low-exposure strategies in trending years — for any sparse §B line, read
  per-trade `signal_expectancy` as the exposure-neutral decider (a future `majority-positive-expectancy` gate
  kind would formalize this; not built yet). **Do not redo the macro-release arm.**
- **Detected LIQUIDATION cascades — PROBED, measured null** (`probe-cascade-reversion`, `probe-cascade-momentum`).
  Zero-fetch: real liquidation feeds aren't free, so `trainer/liquidation.py` DETECTS a cascade proxy from
  on-disk data — a bar with (1) move > k·trailing-vol, (2) a past-only volume spike, and (3) a same-side taker-
  imbalance extreme (all three required; the compound AND + past-only gates + t+1 causality are mutation-proven,
  and the detector demonstrably catches the real Aug-5-2024 crash). The mechanical-overshoot **reversion thesis
  is not just null — it is significantly net-NEGATIVE where it has power** (fading cascades loses, BTC t=−3.95);
  if anything cascades mildly CONTINUE. Adversarially verified (null-survives): no config is significant in both
  windows; the positive-both survivors are thin-sample/single-asset/best-of-24 multiplicity flukes (the lone
  |t|>2 is the momentum control on one asset/window, a ~45% coin-flip under multiplicity). **Do not redo.** NB
  this used a PROXY; a real liquidation feed (paid/recorded WS) could sharpen it, but the mechanical-reversion
  logic itself failed, so that is low priority.
- **REMAINING B2 (needs a calendar/detector mine): FOMC decision days** (a real meeting calendar, not the daily
  DFEDTARU rate print), **token unlocks / listings, whale moves, exchange outages.** Each is probe-first once its
  calendar/detector exists; the macro-release AND cascade nulls suggest the bar is high (crypto prices public/
  mechanical events near-instantly), so favour events with a genuine, slower information-asymmetry window.

**B3. Sentiment / social / crowd psychology — "can we read the crowd (Reddit-for-GameStop) and trade it?"**
Ingest social/forum streams (Reddit, X, Discord, news), turn conversations / mood / attitude toward SPECIFIC
assets into a signal, and trade ahead of or with a crowd surge (the GameStop archetype). Pipeline: source
ingestion → per-asset sentiment/attention extraction (**a natural fit for the engine's LLM path**) → a
sentiment→price probe. **Traps to pin:** social sentiment is heavily arbitraged and mostly predicts
VOLATILITY/attention not direction; bot/manipulation noise; strict "known-when" to avoid look-ahead. Probe
first, build second.
- **DATA FEASIBILITY (scoped by live-fetching 7 sources):** free historical daily-alignable data DOES exist —
  **Wikipedia pageviews** (per-asset attention, no key, 2015+, reliable), **Arctic Shift** (r/CryptoCurrency +
  r/Bitcoin post/comment archive with text+scores, free, the PullPush successor — the actual conversations),
  and **Crypto Fear & Greed** (market-wide, daily, 2018+, but partly price-derived). Heavier/flakier: GDELT
  news-tone (free via bulk files), Google Trends (pytrends, rate-limited). Dead ends: CryptoCompare/CoinGecko
  social (key/deprecated), X/Twitter (paid).
- **ATTENTION (Wikipedia pageviews) — PROBED, measured null** (`probe-attention-momentum`,
  `probe-attention-reversion`). `trainer/attention.py` (reuses `signal_extremes` + intraday; publication-lagged
  join mutation-proven, `lag_days>=1` enforced). The SIGN FLIPS by regime (momentum wins the bull, reversion
  the bear) — the textbook "attention tracks volatility, not direction." The one tempting cluster (fade ETH
  surges, +2.1%/trade) was **adversarially demolished as a fat-tail/single-asset/best-of-96 fluke** (t≈1.0/1.9,
  median trade negative, top-3 trades = 114% of profit, BTC-negative), driven by a **non-stationarity confound**:
  ETH's Wikipedia traffic fell ~2-3x from 2022, so the past-only EXPANDING quantile mislabels normal days —
  a methodological note for any future level-based surge detector (use a TRAILING window or detrend). **Do not redo.**
- **REMAINING B3 — the real swing (Reddit → LLM sentiment):** Arctic Shift gives the actual crowd conversations;
  the untested bet is LLM-extracted per-asset MOOD/attitude (not mere attention volume) → sentiment→price probe.
  This is the last genuinely-different, non-price-derived signal in the whole program. Same probe-first
  discipline; the attention null raises the bar (attention/volume is priced-in — only true sentiment DIRECTION,
  if it leads price, could survive).

**B4. Macro / class-level world-model — "things affecting a whole class → a world model that FEEDS the model."**
Model the MARKET STATE (macro regime, risk-on/off, rates, liquidity, sector/class rotation) as a "world model"
layer that feeds CONTEXT to the trading model rather than trading directly — turning the macro/context we
already mine into a coherent regime/state estimate that CONDITIONS the trading decision (which asset, on/off,
size). This is the macro-scale version of B0's "recognize the condition" idea. Probe: does conditioning on the
regime estimate improve a cost-surviving decision anywhere?
- **Macro-regime market-TIMING overlay — PROBED, measured null** (`probe-regime-overlay`, `probe-regime-inverse`).
  Zero-fetch: `trainer/regime.py` times crypto EXPOSURE (hold in risk-on / cash in risk-off) from on-disk
  point-in-time macro (rates easing, curve steepening, claims falling, composite; joined as-of via `pit_fusion`,
  the mutation-proven leakage guard). Judged on the RIGHT bar — `sharpe_vs_hold` (risk-adjusted), since a
  de-risking overlay gives up upside in a bull so raw return can't be the gate. Null on both the gate (no
  majority improves Sharpe in either window) and the decisive **overlay-vs-inverse symmetry test**: the overlay
  is statistically indistinguishable from its mirror control (mean Sharpe-diff t=1.13, `corr=−0.735` — two
  complementary NOISE partitions), its own mean `sharpe_vs_hold` is negative, and the 3/24 positive-both cells
  are exactly the chance overlap (Fisher p=1.00). The overlay DOES cut drawdown (−38% vs −43%) but that is
  mechanical de-risking (cash ~36% of the time) that costs return and nets out under Sharpe. Adversarially
  verified (null-survives). **Do not redo.** A learned/multivariate regime model over the same macro would face
  the same near-zero signal + few-regime-cycles power wall; not worth building.

**Data correctness rules (enforce in the loader for ALL of the above).** Store minimal raw only (derive at
runtime); **join by TIMESTAMP, never date string** (each row carries `barCloseTz`); macro is point-in-time
(ALFRED vintages, stamp at the real per-release datetime, forward-fill, MoM/YoY = diff of the SAME vintage;
post-close series publish next session); fundamentals stamp at filing/acceptance not period-end (restatements =
new rows); commodity continuous roll is a look-ahead machine (back-adjust with roll dates); FX `Volume≡0` is a
constant not data; never forward-fill one leg of a ratio; idempotent + validated mining (monotonic timestamps,
positive prices, split/div-adjust); licence-gate shareable output (only Frankfurter/ECB is redistribution-safe).
The existing tradeable classes (crypto, stocks, macro/rates context) + the 1m→fidelities derive-cache already
exist; commodities + FX as tradeable classes and the `thefactory-datamine` extraction remain, built only when a
surviving probe needs them.

### B5. Non-trading engine reuse — the fallback if the information layer also nulls

The hardened engine (leakage control, DSR, time-split, hypothesis trail) is domain-agnostic. If the non-price
trading frontier also nulls, the best non-trading reuse is a **code-change / defect-risk model** — but the
research graded it a documented-null-REPRODUCER, not an edge: worth at most ONE pre-registered decision-probe
("does any JIT config beat a size/ManualUp baseline out-of-time, DSR-corrected?" — expected no → the deliverable
is "use deterministic SAST/linters/mandatory review of AI diffs, not a learned gate"). Not a build until the
trading frontier is exhausted AND "safer AI-written code" is the stated mission.

### B6. Optional + small deferred

- **Live handoff** — tag the exploration autopilot's global-max checkpoint for live trading (`run_server_model.py`).
- **Jupyter notebooks (UNDERSCOPED)** — view/edit/execute a project's `.ipynb`; scope kernel location + security.
- **Runner-channel WebSocket upgrade** — dispatch is already ~instant; a WS only shaves ~1.5s log latency.
- **Remote git repoRefs** — wire git refs + project bootstrap when a real remote machine needs it.

## C. Harden ML-experiment EVALUATION — AUDIT MANDATE (the go-forward mission; a deepening of North-star #1)

**Why now.** The trading program (§B) is comprehensively null — 21 gate-backed, adversarially-verified nulls.
That result is a product of the EVALUATION DISCIPLINE, not the trading ideas: leakage-controlled point-in-time
joins, past-only + t+1 causality with mutation-proven guards, pre-registered hypotheses with declared gates,
Deflated-Sharpe multiplicity correction, walk-forward OOS, and independent adversarial verification each killed
illusions a naive backtest would have shipped (a fat-tail outlier that was 114% of profit; a non-stationarity
confound; a dozen "best-of-N" flukes). **That discipline is the durable asset, and it is domain-agnostic.**
North-star #1 (best generic pipeline for creating ANY model) is only as strong as its evidence layer — so
modeltrainer should absorb these guarantees as FIRST-CLASS, GENERIC ML-experiment-evaluation capabilities, so
every training campaign gets the same rigor for free.

**The asymmetry that makes this high-value.** Trading is adversarial + efficient, so a real edge is competed to
zero → the gates correctly returned nulls. General ML is NOT adversarial → a genuine improvement is NOT competed
away, so the SAME gates that killed every trading signal will PASS a real ML edge. The ML failure mode is the
mirror image: **false positives** — shipping a "win" that is a lucky seed, an overfit-to-validation
hyperparameter sweep, or a proxy-metric artifact. This machinery is exactly the false-positive filter that ML
experimentation lacks by default, and that RL (high variance, unstable) needs most.

### C.1 — The mandate (for a scrutinous audit process)

Do a rigorous capability audit of modeltrainer against the checklist below (§C.2), grounded end-to-end in the
driving case (§C.3). This is a **research + gap-finding** task, NOT a licence to build blindly. For EACH
capability: (a) does modeltrainer HAVE it, PARTIALLY, or NOT AT ALL — cite the concrete code/seam
(`viewer/hypothesis.js` gates, the side-experiment framework, `ModelTrainerTools`, the run/summary contract);
(b) where it is missing or TRADING-SPECIFIC, what should be RESEARCHED and how to make it GENERIC + first-class;
(c) a probe-first plan — the cheapest demonstration on the driving case that proves the capability bites BEFORE
any large build (mirror §B's discipline: a cheap pre-registered demonstration, then build only what survives).
Deliver: gap findings ranked by value, each with a proposed capability + a driving-case demonstration + the
open questions it raises. Retire nothing already shipped; extend it.

### C.2 — Capability checklist: what rigorous ML-experiment evaluation requires

1. **Pre-registered hypotheses + declared success gates, GENERIC.** The hypothesis/verdict trail + gates exist
   (`beats-hold` / `majority-beats-hold` / `deflated-sharpe` / `beats-baseline` / `invariant` / `differs`) but
   are trading-flavoured. Audit: are there generic ML gates declarable BEFORE a run — beats-baseline-by-EFFECT-
   SIZE, improves-metric on a held-out TEST, `majority-positive-across-seeds`? (The last is noted-but-unbuilt
   even for trading — the exposure-neutral decider §B kept needing.) What is the minimal generic gate set?

2. **Multiplicity / "best-of-N is inflated" correction — likely the biggest gap.** Sweeping N configs and
   reporting the best validation score is the #1 way ML fools itself: the max of N noisy draws is upward-biased
   (this is precisely how every tempting trading cell died — "best of 96" needs ~t>3, not t>2). The trading
   engine corrects this with Deflated-Sharpe (expected-max-under-null, `deflatedSharpeFromStats`). Audit: is
   there a GENERIC multiplicity correction — deflated best-metric / expected-max-under-null / Bonferroni-Holm
   across the swept trials — so "config X is best of 50" is discounted for the search size? DSR is Sharpe-
   specific; generalise the same math to any metric (accuracy, win-rate, loss) with its own trial-variance.

3. **Held-out TEST discipline (not just validation).** Model selection tunes on validation; a LOCKED test set
   the selection never touched is the only honest final number. Audit: does modeltrainer enforce/track a
   train/val/TEST split where TEST is consumed ONCE, post-selection, and flags reuse? Or is it consumer-
   responsibility with no guardrail? (Trading analog: walk-forward windows accounted strictly OOS.)

4. **Seed-variance robustness — CRITICAL for RL.** RL is notoriously unstable across seeds; a config that
   "wins" on one seed is usually noise — the exact "one-window fluke" the adversarial passes killed again and
   again, reincarnated as "one lucky seed." Audit: does modeltrainer run multi-seed and gate on "improvement
   EXCEEDS seed-variance" (a delta-vs-seed-std significance test, e.g. paired across seeds), not a single lucky
   run? Report deltas with seed-CIs, never a bare point estimate. This is the highest-leverage gap for the case.

5. **Leakage / data-snooping guardrails, generalised.** Point-in-time joins were trading-specific; the general
   analog is no test contamination, no feature-engineering-on-test, no selecting-on-test, no train/test overlap
   (e.g. board positions from the same game across splits). Audit: what leakage guardrails or at least detectors
   / warnings does modeltrainer offer a generic consumer, versus leaving it entirely to the CLI?

6. **Adversarial verification as a CAPABILITY.** The 2–3-skeptic "try to refute this verdict" pass was run by
   hand (a workflow) and repeatedly caught real artifacts. Audit: should modeltrainer offer a built-in
   "adversarially verify this claimed improvement" pass — independent re-analysis + robustness perturbations
   (perturb seed / split / nuisance hyperparameters; does the win survive?) — before a verdict is trusted?

7. **Effect size + significance reporting.** "X beat Y by 0.3%" is meaningless without n + variance. Audit:
   does the trail report deltas with confidence intervals / significance, or only point estimates?

8. **Proxy-vs-true-objective discipline.** modeltrainer already frames "reward is a PROXY vs the scorecard" —
   does an improvement on the TRAINING proxy actually move the TRUE objective? Audit: is there machinery to test
   proxy→true robustly? The driving case makes this crisp (does reward-shaping that raises episode-reward
   actually raise WIN-RATE, or just game the proxy?).

9. **Reproducibility / provenance / search-space capture.** Provenance fingerprints + byte-exact reproduction
   exist. Audit: is the trail sufficient for a third party to re-derive every verdict, AND does it record the
   full SEARCH SPACE searched (the count the multiplicity correction in §2 needs)?

### C.3 — Driving case: an RL agent for board games (a COMPLETELY RESOLVABLE task)

Unlike trading, a board game has GROUND TRUTH: the agent either learns to play well or it does not, measured by
win-rate against defined opponents — non-adversarial, no efficient market erasing the edge, a real learnable
objective. This makes it the ideal vehicle to (a) accomplish a genuine ML task the owner wants AND (b) STRESS
every capability in §C.2 until the gaps surface concretely. Set it up as a new conformant modeltrainer consumer
(the fourth, alongside `examples/cartpole`, `examples/tabular`, and BlackSwan): an RL-board-game trainer that
implements only the thin CLI/`TrainerManifest` contract — no engine changes to run it; the engine changes are
whatever §C.2 gaps it exposes.

The case exercises the gaps by construction:
- **Seed variance (§C.2.4):** board-game RL swings wildly across seeds → forces multi-seed + variance-aware gates.
- **Opponent generalisation (§C.2.3):** win-rate vs WHICH opponents (random / fixed heuristic / self-play / prior
  checkpoints)? "Beats the random opponent but loses to the heuristic" is the RL twin of the trading one-window
  fluke — the held-out OPPONENT is the test set, and beating the training opponent while failing a held-out one
  is exactly the overfit the §C.2 gates must catch.
- **Proxy vs true (§C.2.8):** does reward-shaping / curriculum that raises episode-reward raise win-rate?
- **Multiplicity (§C.2.2):** sweeping algorithm / network / reward configs and picking the best inflates — the
  first thing to correct for.

**Open questions for the case (decide with the owner, who supplies the specific game + constraints):** which
game(s) — start with ONE small, fully-observable, two-player perfect-information game (Connect-Four / small
board scale) so training is cheap and ground truth crisp, then a harder one; which RL algorithm(s) — self-play
PPO / AlphaZero-style MCTS / tabular for the trivial baseline; the OPPONENT LADDER that defines the test set;
the compute budget + whether remote runners are needed; how the win-rate scorecard + the proxy (episode reward)
relate (§C.2.8). The owner will name the specific game(s) and constraints; scope the first probe to the cheapest
game that makes the seed-variance + multiplicity gaps (§C.2.2/4) bite.

### C.4 — Engine gates SHIPPED; what remains

The §C.2 capabilities are now GENERIC + first-class in the engine, each pinned by a divergence probe in
[src/evalRigorProbes.test.ts](src/evalRigorProbes.test.ts) (S1–S9) plus direct unit tests, and all OPT-IN so
CartPole/tabular/BlackSwan collapse to their existing behaviour untouched. Built: seed-significance gate
(`evalSeedSignificanceGate` + paired bootstrap), metric-agnostic best-of-N (`evalBestOfNGate`, beside DSR,
one shared `diagnostics.searchSpace` trial floor), `validateRunProvenance` soft-flag, fail-closed
`hypothesisBenchmark` + declarable seed-quorum, effect+CI on the verdict (`ChampionGate.effect` + CI-based
split held-test via `splitAxis.alpha`), locked held-out TEST role (`splitAxis.testValues`), proxy
selection-regret (`proxyAlignment`), first-class adversarial verify (`verifyImprovement`), and the degeneracy
gate (`degenerateWhen` now bites the champion verdict).

Remaining (forward):
- **Game suite** ([examples/boardgames/](../examples/boardgames/)) — the 4th conformant consumer is stood up:
  a game-agnostic self-play harness + `connect4` (dependency-light random/heuristic/mcts cores + a fixed-rung
  opponent ladder), emitting the full §C metric battery + a `cost` block + a sampled replay, with a manifest
  declaring every gate. Verified: a real 36-run sweep runs through `assembleChampionVerdict` and the gates fire
  (incumbent selected off the held-out `mcts` test; not-steady on 3 seeds). Remaining games (simplest→hardest):
  skull · flip7 · skull_king · for_sale, then the frontier (terra_mystica · poker[CFR] · catan · altered[two-
  model]). Plus: a **neural self-play core** (`ppo_selfplay`/`alphazero` as `model_name` levers); **personas +
  league play** for the luck games (champion pool onto the `opponent` axis via `choicesFrom`); the
  **specialist-vs-generalist-finetuned** §C hypothesis (needs a HuggingFace survey); and a **BoardGameArena**
  live-play bridge (drives `load_policy`) as the final real-world test.
- **Expose `verifyImprovement` as a chat tool/activity** — the engine function is done; the thin remaining
  piece is the thefactory-backend `trainerTools` schema + dispatch wrapper.
- **S9 leakage tail** (lowest value, do when it bites): a per-split membership signature
  (`RunSummary.splitSignature`) + a train/eval-overlap disjointness detector; and relocate the trading fidelity
  predicate (`isRunAffectedByFidelity*`) out of shared `modelTrainerUtils` into BlackSwan (repoint-and-DELETE).
- **Owner ratifications** (conservative defaults are live; revisit if desired): fail-closed-*with-reason*
  benchmark (not hard-required — preserves BlackSwan's `return_vs_hold_pct`); best-of-N BESIDE DSR sharing one
  `nTrials` floor; reuse / `unverifiable` flags advisory, not hard-block.

## D. Publish the evidence — a scientific paper (a mission-level OUTPUT of the engine)

**The ambition.** The §B trail is not just a private result — it is a publishable scientific dataset. The
strived-for outcome: a rigorous, peer-review-grade paper that (1) demonstrates, across a broad PRE-REGISTERED
battery, that no cost-surviving out-of-sample edge exists in the accessible data over the tested markets/period,
and (2) takes the specific published claims that argue otherwise, reproduces them under the SAME discipline, and
shows they do not survive. More generally, modeltrainer should treat "**accumulate enough rigorous, reproducible
evidence to produce a publication once a big enough thesis arises**" as a first-class OUTPUT — for BlackSwan
first, then for ANY domain the engine is pointed at (the ML-evaluation + RL-board-game work in §C is itself a
methods-paper candidate). This is a deepening of North-star #1 and the natural terminus of §C: the evaluation
engine's evidence deserves publication, not just a private trail.

**Honest scope (so the claim is provable, not an over-claim).** A paper cannot prove the universal non-existence
of edge, and the write-up must never pretend it does. What IS defensible — and strong:
- **The battery claim.** A pre-registered set of N signal families, each tested with leakage-controlled
  point-in-time joins, walk-forward OOS, per-trade cost, and multiplicity correction, ALL fail their declared
  gate — with the search space FULLY DISCLOSED so the nulls cannot be waved away as "you didn't try X" (state
  exactly what was and was not tested, and over which markets/period).
- **The refutation claim (the provable core).** For K specific published "edge" papers, a faithful
  re-implementation under the same discipline fails where the original passed, and the paper names the EXACT
  methodological hole that flipped it — in-sample selection, no transaction cost, no multiplicity correction,
  look-ahead in a retroactively-revised feed, or a single-window/single-asset fluke. The claim is not "no edge
  exists" but "**these published edges do not survive honest evaluation, and here is precisely why**."

### D.1 — What the engine must accumulate/produce (the generic capability, small; build alongside §C)

- **Publication-grade evidence export.** A one-command export of the hypothesis/verdict trail into a
  reproducible scientific artifact: every pre-registered hypothesis + declared gate + result + verdict, the full
  search space (§C.2.9), the leakage guards proven to bite, the DSR/multiplicity math, the
  adversarial-verification record (§C.2.6), and a data/code provenance manifest a third party can re-run to the
  same verdicts. This is §C.2.9 taken one step further — from "reproducible internally" to "**a referee can
  re-derive every number**."
- **Reproduce-and-refute workflow.** A repeatable pipeline that takes a published claim, re-implements it as a
  conformant probe (thin CLI contract, no engine changes), runs it under the discipline, and records whether it
  survives + WHY it flips if not. Each reproduced paper becomes another entry in the trail and another row in the
  refutation table — this is the generative engine behind the "papers arguing otherwise are wrong" claim, and it
  reuses the §B probe-first machinery wholesale.
  - **First execution + the UNIT-OF-ANALYSIS lesson (Mou 2011, "Front-Running the Goldman Roll").** Reproduced as
    the WTI M1-M2 calendar-spread probe (BlackSwan `trainer/roll.py` + `.factory/trainer-roll.json`; EIA RCLC1..4
    settlements; four mutation-proven leakage guards; adversarially verified, 3 skeptics + synthesis).
    **DISPROVED** as a cost-surviving WTI edge — "tail-contaminated pseudo-reproduction, sign-only, never
    DSR-significant": only the SIGN reproduced (fade is an algebraic mirror with zero independent power); the
    tempting +3%/yr was a max-Sharpe-**selected-cell illusion** (unbiased cross-config mean ~+0.5%/yr, ~6x below
    Mou) carried by 2009 GFC / 2020 COVID super-contango tails, insignificant even against a **zero** bar
    (t~1.2), and flatly null OOS. **The capability lesson:** reproduce at the PAPER'S UNIT OF ANALYSIS. Single-
    asset WTI discarded ~√24 ≈ 4.9x of Mou's **pooled ~24-commodity** t-stat, so this null is scope-limited and
    does NOT refute the basket claim — the workflow must match the original's cross-sectional scope/power or
    record the result as underpowered, never as a refutation. Corollaries baked into the capability: report the
    **unbiased cross-config mean, not the selected best cell**; a mechanically-mirrored control arm carries no
    independent evidence.
  - **Pooled follow-up EXECUTED — the energy-basket roll (`trainer/roll_basket.py` + `.factory/trainer-roll-basket.json`;
    EIA crude+heating-oil+RBOB+natgas M1/M2; equal-weight monthly portfolio; adversarially verified, 2 skeptics +
    synthesis).** **DISPROVED, scoped to the energy front-run only.** Pooling did NOT recover a positive roll
    premium — the portfolio front-run is negative in both windows (t~−1.2, insignificant but genuine sign,
    verified robust to inverse-vol weighting: heating oil, the LOWEST-vol leg, carries it at t−3.62, so
    risk-weighting *strengthens* to t−2.58). **Named reason:** opposite-sign heterogeneity from energy demand/
    storage SEASONALITY — the seasonal legs (heatoil, natgas) LOSE the front-run direction; only crude showed the
    weak crash-tail positive. This ANSWERS the WTI open question: **the WTI positive was crude-specific and
    crash-tail, not a diversifiable premium** ("underpowered" is refuted — the other legs are significant
    wrong-sign losers, not noisy positives). **Scope held:** refutes the ENERGY roll only; Mou's uncorrelated
    metals/ags/livestock legs (paid data) remain **untested** — "no evidence for," not "evidence against," the
    broader basket. That non-energy pool is the only remaining way to fully adjudicate Mou, and is **data-gated**
    (paid CME/metals/ags term structure), so not built.
- **Cited-papers → Library linkage.** The papers being reproduced/refuted are tracked in the Library, so the
  corpus of "claims tested" is itself a maintained, auditable artifact (not an ad-hoc list in the write-up).

### D.2 — The two concrete papers

1. **The BlackSwan no-edge paper (first target — the corpus already largely exists).** The 21 gate-backed nulls
   (§B) are the spine; the reproduce-and-refute table is the punch. Remaining to make it publishable, none of it
   a new engine build: (a) run the reproduce-and-refute workflow against the specific papers claiming
   funding / order-flow / sentiment / regime edges (the exact families we nulled); (b) export the reproducibility
   package; (c) write the honest-scope framing above. Once the export + reproduce-and-refute capability exists,
   this is a WRITE-UP + refutation-run task, not a build.
2. **The methods paper (the durable, domain-agnostic contribution).** The evaluation engine itself —
   pre-registered generic gates, generalized multiplicity correction, seed-variance-aware gating, adversarial
   verification as a built-in pass (the §C capabilities) — demonstrated across MULTIPLE domains: the trading
   nulls (must return null — efficient market) AND the RL board-game true positive (must return a real win —
   learnable game) AND an `examples/` baseline. The strongest scientific contribution is not "crypto has no edge"
   but "**here is a generic, reproducible false-positive filter for ML experimentation, validated both where it
   must return null and where it must return a true positive**."

### D.3 — Measurable success criteria (set BEFORE the write-up, per the measurable-criteria discipline)

- **Battery:** ≥ [N] pre-registered signal families, each with a leakage-controlled OOS + multiplicity-corrected
  verdict and a disclosed search space; ALL null. (N ≈ 21 already; decide the publishable threshold + which
  families are in-scope, and over how many markets/regimes.)
- **Refutation:** ≥ [K] specific published claims re-implemented and shown not to survive, each with the named
  methodological hole. (Decide K + the target papers with the owner.)
- **Reproducibility:** a third party, given the exported package, re-derives ≥ [X]% of the verdicts byte-exact
  (target 100% for deterministic cells).
- **Trigger for the write-up ("a big enough thesis"):** a battery + refutation table that survives an independent
  adversarial-verification pass, across ≥ [M] markets and ≥ [P] regimes/windows. Define these thresholds with the
  owner; the generic capabilities (§D.1) are worth building now, the write-up triggers when the corpus clears the
  bar (for BlackSwan that is close — the battery exists; only the refutation runs + package export remain).

## E. Open questions + trigger-blocked builds (act when the dependency lands)

- **Context — `with_extra_data` projection rung** (blocked on data). An asset's OWN fused series + the
  obs-signature gate as a REAL replay guard. Every context panel today is GLOBAL (macro/majors/market); no
  per-asset series is mined yet. The fusion substrate + loader already exist, so the rung is small — build once
  ≥1 asset-specific series (B1/B3 output) is mined.
- **Host→iframe `data:updated` push channel** — the viewer is poll-only; the bridge forwards only `nav.open`.
  A push channel would let the viewer live-refresh on a data change instead of polling.
- **Remote artifact/checkpoint storage** — keep-on-runner + reference vs upload; how a winning remote
  checkpoint reaches a live trading server. Meaningful once remote runs AND live handoff both exist.
- **GPU + sandbox profile for training images** — `--read-only` rootfs vs ML caches; `--gpus` is wired but
  unexercised.
- **Judge/proposer model transport** — `ModelSelection` (API vs CLI), revisit once the CLI inference stage lands.
