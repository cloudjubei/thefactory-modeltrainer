# thefactory-modeltrainer — implementation plan

**Remaining work only** — shipped history lives in git + memory. Architecture: `docs/architecture.md`.
Contract: `docs/model-training-standard.md`. The engine stays domain-oblivious — any further model is
_data + the thin CLI contract_, not engine code.

## North star (frames prioritization)

> **★ OVERARCHING RULE — no matter what we work on:** build EVERYTHING as **reusable, chat-reachable modeltrainer
> tools**. Any capability we need becomes a first-class harness capability + a `.factory/trainer.json` activity + a
> chat/agentic tool (`ModelTrainerTools` / backend `trainerTools`) — never a one-off script — and is designed generic
> so it serves OTHER trainings/model work too (the LLM-tool-parity rule: every UI/engine capability is chat-invocable).
> Prototyping in a script is fine; it is not DONE until promoted to a reusable tool and the script is deleted.

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
- **GOLD macro world model — PROBED, INCONCLUSIVE/underpowered null** (`probe-gold-worldmodel`,
  `probe-gold-worldmodel-inverse`). The world model generalised beyond crypto: `trainer/worldmodel.py` conditions
  a long/short/flat GOLD position on its three canonical macro drivers (real rate DFII10↓, broad dollar
  DTWEXBGS↓, breakeven DGS10−DFII10↑) joined point-in-time (pit_fusion); thesis + inverse control; 4 mutation-
  proven leakage guards; run + adversarially verified (2 skeptics + synthesis) via the committed BlackSwan
  `experiments/` tooling. **EDGE = NULL** (no DSR-deflated, hold-beating, majority-passing cell; best |t|~0.93).
  The thesis FOUGHT the 2023-24 gold bull (−12%, −9% vs hold +12%, +27%) — gold **decoupled** from its textbook
  drivers (rallied into high real rates + strong USD on central-bank/de-dollarization demand the macro series
  can't see). Recorded **INCONCLUSIVE, not disproved** (that overclaims): the verification caught **two real
  defects** — (a) the composite is **collinear** (DFII10 double-counted across the real-rate + breakeven legs,
  → a real-rate trend follower with USD as tiebreaker), and (b) a **DTWEXBGS publication-lag look-ahead** (fixed:
  `_ASOF_LAG_DAYS` + re-mine; verified immaterial). The inverse's tempting 2024 cell (+38.5%, t~2.27) is
  **beta + multiplicity + mechanically-coupled**, ~zero alpha over hold — NOT an edge.
- **GOLD world model FAIR TEST + real-rate-alone — RAN, now DISPROVED (composite-scoped)** (`probe-gold-worldmodel-fair`,
  `probe-gold-realrate`). The two INCONCLUSIVE caveats were resolved: **de-collinearised** (breakeven from the
  standalone T10YIE, DFII10 counted once) and the **2020-21** favorable regime **added** (5 windows 2020-24). The
  composite STILL nulls (best thesis t~0.48; underperforms hold in every bull window), and the skeptic's own
  falsifiable prediction — "it goes long and wins in the 2020-21 real-rate collapse" — is **directionally REFUTED**
  (2020: gold +24% but the fair thesis made +8%, negative alpha, t~0.48). The **real-rate-ALONE** arm was run to
  close the "equal-weight vote masked a predictive driver" escape hatch — it **also nulls** (best t~0.95; 2020
  only +9.8% vs +24%). So the null generalises from the composite to the isolated real-rate channel. **Scope/power
  caveat (recorded):** this disproves the macro-composite + real-rate channel as gold timers, NOT "no macro model
  can time gold" — the ~252-obs DSR gate (critical t~2.7) can't reject a modest Sharpe-0.3-0.8 edge, so the verdict
  rests on the powered directional refutation, not a zero-edge CI.
- **SILVER + COPPER world models — RAN, INCONCLUSIVE (partial models)** (`probe-silver-worldmodel`,
  `probe-copper-worldmodel`; adversarially verified). Same monetary drivers on **silver** null (even the
  favorable-regime high-beta best case underperformed hold) — but silver_macro3 omits silver's **industrial**
  channel, so "no evidence for," not "against". **Copper** = a financial-conditions-only model (USD/rates/curve);
  its 2020 cell (t~1.61) is COVID-reflation **beta**, and its **real drivers (China demand, LME inventories) are
  absent** — so the *full* copper model is untested. Third latent leak fixed en route: **T10YIE** publish-time
  stamped 16:15 (was 08:30) to match its H.15 siblings (immaterial to daily/next-bar; consistency).

**B7. Positioning / flow — CFTC COT (the last free, previously-untested signal class).** `trainer/cot.py` +
`scripts/fetch_cot.py` (free CFTC Socrata API, managed-money net/OI, 1052 weeks 2006-2026, GOLD/SILVER/COPPER;
release-lagged join + contrarian-invert both mutation-proven). Structural story: crowded specs are FORCED
unwinders. Ran + adversarially verified (3 workflows) via the `experiments/` tooling.
- **COT level-extreme (expanding quantile), contrarian + momentum — INCONCLUSIVE** (`probe-cot-{gold,silver,copper}-*`).
  No majority-passing DSR edge; every tempting cell is single-window beta + multiplicity; the winning *arm*
  regime-tracks price (2020 momentum year / 2023 contrarian year) → it's price-timing, COT redundant. Flaw the
  verification caught (my error): the expanding quantile anchored to the 2010-11 mania made gold's extreme
  UNREACHABLE (gold fired 0 trades in 2020/21) — the same non-stationarity trap the attention probe recorded.
- **COT INDEX (trailing 3yr Williams min-max) — the fix, INCONCLUSIVE** (`probe-cotidx-*`). Fixed the gold hole
  (gold now fires in 2020); the LEVEL-extreme sub-class now tested under **two independent normalizations** (both
  arms, 3 metals, 5 windows) and **zero cells clear the gate**. The program's two highest cells died on mechanism:
  copper-momentum 2020 (t~2.63) = reflation **beta** (mirror copper-contrarian t~−2.64; non-persistent; fails
  best-of-20 DSR); silver-contrarian 2023 (t~2.12) = the **expected best-of-N outlier** (E[max]~2.0-2.1),
  single-window, and its cross-spec appearance is one event via two correlated lenses, not replication.
  **INCONCLUSIVE not disproved:** power (best-of-10/20 at ~252 obs can't reject a modest Sharpe-0.4-0.8 persistent
  edge) + one untested variant.
- **COT FLOW/CHANGE (weekly delta) — RAN, INCONCLUSIVE; positioning class CLOSED** (`probe-cotflow-*`, verified).
  The last untested variant of the managed-money extreme→side machinery. Even WEAKER than the level (max cell
  silver-flow-momentum 2024 t~1.74, below even naive 1.96 and below the level's expected best-of-N outlier ~2.0);
  the wins are regime-tracking **beta** exactly as pre-registered (momentum arms carry up/reflation years,
  contrarian the reverting year) — managed-money delta is collinear with price momentum (already null).
- **CROSS-SECTIONAL COT (market-neutral relative value) — RAN, INCONCLUSIVE; commodity positioning class fully
  CLOSED** (`probe-cross-cot-*`, verified). The structurally-different thread: a dollar-neutral long-least-crowded
  / short-most-crowded book across a 6-commodity basket (GOLD/SILVER/COPPER/WTI/CORN/WHEAT — metals+energy+ags),
  so the common commodity **beta** that made every single-asset tempting cell a beta artifact CANCELS. It still
  nulls — mirror arms, regime-tracking (cross-momentum wins trends, contrarian wins chop), no cell clears
  best-of-20 DSR or the majority gate. **The beta-neutral construction did not manufacture an edge; it replaced
  single-asset beta with cross-sectional momentum-factor redundancy**, confirmed EMPIRICALLY: the per-bar Spearman
  ρ(COT-index rank vs 126d trailing-return rank) = **+0.45 mean / +0.54 median, 87% of bars positive** — managed
  money is a CTA/trend cohort, so its ranking ≈ the price-momentum ranking. (Data note: the skeptics' assumed WTI
  negative-price artifact was VERIFIED FALSE — yfinance CL=F lacks April-2020; `px>0` filter + forward-fill.)
  **The COMMODITY managed-money positioning class is now CLOSED across all constructions** (single-asset level /
  Williams index / flow / cross-sectional), all defeated by collinearity-with-price. Commercials/hedgers is NOT
  an independent residual (COT adding-up identity → commercial net mirrors managed-money net).
- **CROSS-ASSET-CLASS COT (financial futures) — RAN, DISPROVED; the cross-asset hypothesis is REFUTED**
  (`probe-cotfin-{spy,ief,tlt}-*`). The one genuinely-different free universe: CFTC **TFF leveraged-funds**
  positioning on equity (E-mini S&P), 10y notes (IEF), long bonds (TLT), where specs include real hedgers /
  risk-parity so the CTA-collinearity is weaker. It STILL nulls — pooled over **12 windows 2010-2021**, contrarian
  arms mean Sharpe ~0 / coinflip, momentum arms negative (IEF momentum t~−2.2 = bond positioning mean-reverts, but
  the contrarian mirror ~0, no exploitable edge). **Positioning class now closed across BOTH commodity AND
  financial universes.** (TFF COT coverage 2006-2022.)

**"Try harder to disprove properly" — the EXTENDED-HISTORY pass (INCONCLUSIVE → DISPROVED).** The 26 inconclusive
verdicts rested only on the power caveat: the 5-window (2020-24) DSR gate couldn't reject a *modest* persistent
edge. Fix: extend the OOS. Added `--start` to `backfill_market`, mined all commodity + financial **price back to
2006** (matching COT + macro), and added walk-forward windows **2008-2019** — giving **17 independent yearly
windows (2008-2024)** across the GFC / 2011 / 2015-16 / 2018 / 2020. Re-tested every inconclusive family
in-process (single config, no lever-selection): **COT** (level/index/flow × 3 metals) per-window Sharpe mean ~0,
|t_win|<1.5, 29-59% positive — a coinflip; **cross-sectional COT** redundancy measured (ρ=0.45 with price-momentum);
**world models** (gold/silver/copper) mean Sharpe ~0-to-negative with **no alpha over hold** (the earlier weak
positive was best-of-lookback selection + long-biased asset beta that UNDERPERFORMS buy-and-hold). A modest edge
(Sharpe 0.3-0.8) would show t_win>2 and >65% positive — none does. **All 26 upgraded INCONCLUSIVE → DISPROVED**
(residual: only a tiny sub-0.15-Sharpe edge is inherently non-excludable — a limit, not a live thread).

**Infrastructure shipped by these probes.** (1) **Commodities are now a tradeable class** — `_load_bars` resolves
each instrument's native interval (crypto `-1m-`, commodities/stocks/fx/etfs `-1d-`), so GOLD/WTI/SILVER/COPPER/
CORN/WHEAT load (backward-compatible; 30 loader tests green). (2) **BlackSwan `experiments/` tooling** — a
committed, reproducible registry + runner (`list` / `preregister` / `run` / `analyze`, idempotent-safe so a
materialised verdict is never clobbered), so every probe is recorded and re-runnable rather than living in
scratchpad. This is the paper-trail substrate: the DB hypothesis trail (`blackswan-experiments`) holds all 21
probe verdicts.

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

   **SHIPPED (powered-null primitives, `src/deflatedSharpe.ts`, golden-pinned to `BlackSwan/trainer/sharpe.py`):**
   `sharpeStandardError` (Lo/Mertens non-normality SE, reuses the PSR denominator), `sharpeConfidenceInterval`,
   `minimumDetectableSharpe`, `sharpePower`, `benjaminiHochberg` (FDR), and `poweredNullVerdict` (survivor /
   powered-null / inconclusive via one-sided CI bounds). The point: a null is only informative once POWERED — a
   swept-search that clears nothing is indistinguishable from one with no power until you report the minimum
   detectable effect + a CI on the true metric. `poweredNullVerdict` labels a cell **inconclusive** (not
   "disproved") when the sample cannot rule out the economically meaningful effect. BlackSwan's cross-asset
   factor battery used exactly this to retire an over-claimed "zero survivors" into "5 powered-null / 11
   inconclusive / 0 FDR-survivor" — the general lesson the engine's gates should encode for any consumer.

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

**RESOLVED with the owner:** the first game is **Connect 4** (small, fully-observable, two-player,
perfect-information — cheap training + crisp ground truth); cores are `random`/`heuristic`/`mcts`/`alphazero`
(AlphaZero-style, self-play + warm-start league); the opponent ladder is the fixed rating spine. The crisp
target is now **"solved"** — the `alphazero` core reaches the known-optimal winning strategy, measured against a
perfect-play oracle (see §C.4 "Connect-4 SOLVED"). Harder games follow (README game roadmap).

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
- **Connect-4 SOLVED — the crisp end-state of §C.3 → its own program in §C.5 (ACTIVE).** The oracle, near-perfect
  ladder rungs, distillation, and the measurable SOLVED criterion (`wins_as_p1_vs_EXACT_oracle == 1.0`) are all
  built. The remaining work is the COMPUTE grind (M1 winning-strategy proof → M2/M3 verify+distil) and a net
  training pass — tracked in **§C.5 "The pending work"**. This is still the trigger that unblocks the
  model-comparison view (§E).
- **Unified single "find the best model" process + Runs→Models view (ACTIVE).** Collapse the two autopilots
  (config-space Exploration + champion Improve) into ONE reducer with stages `screen-new → search → improve →
  converged`. On start/resume a `leverSetHash` + `screenedChoices` in `ExplorationState` re-screens any
  newly-added `model_name` choice FIRST (closes the "new models are never checked" gap — `stepScreen` today only
  screens while the archive is under a sample floor). A **learned/compound core** (declared via a manifest
  `compoundCores: [{modelName, warmStartLever}]`) climbs a warm-start ladder in a new `improve` stage
  (`nextChampionStep` folded in, `concurrency:1`) instead of independent grid points; the comparable leaderboard
  refreshes each round. UI: **one surface at a time** via a `unifiedProcessState()` selector (idle-never-run /
  improving / searching / converged / has-champion / needs-attention — a LIVE activity always wins, a
  `processMode` flag disambiguates only the idle case), replacing the rejected stacked-panels + collapse layout.
  **The primary view becomes MODELS ranked by strength (the leaderboard promoted to the main surface); a "Run"
  becomes one training STEP in a model's history (its ladder), not a top-level flat list** — model identity
  spans runs via the champion lineage. Delete `runChampionTraining` / `trainChampionActivity` (bodies fold into
  the one `explore` controller; repoint-and-DELETE, no shim). Open decision: the compound-core signal is an
  explicit manifest `compoundCores` field (preferred, one line in `trainer.json`) vs derived from a
  `*_warm_start` lever gated by `appliesWhen` (zero-config but fragile).
- **S9 leakage tail** (lowest value, do when it bites): a per-split membership signature
  (`RunSummary.splitSignature`) + a train/eval-overlap disjointness detector; and relocate the trading fidelity
  predicate (`isRunAffectedByFidelity*`) out of shared `modelTrainerUtils` into BlackSwan (repoint-and-DELETE).
- **Owner ratifications** (conservative defaults are live; revisit if desired): fail-closed-*with-reason*
  benchmark (not hard-required — preserves BlackSwan's `return_vs_hold_pct`); best-of-N BESIDE DSR sharing one
  `nTrials` floor; reuse / `unverifiable` flags advisory, not hard-block.

### C.5 — Optimal-play trainer (boardgames) — the efficient GENERIC near-optimal learning process (Connect 4 as the calibration harness)

**Goal (reframed 2026-08-24, user-directed):** the most COMPUTE-EFFICIENT GENERIC learning process that produces a
NEAR-OPTIMAL trained MODEL, validated on Connect 4 (where the exact solver lets us measure distance-to-optimal) and
designed to run UNCHANGED on a harder unknown game (chess/Go) where nothing can be solved. NOT brute-force solving,
NOT a compiled solver — those teach nothing generic. Connect 4 is the calibration harness: its exact solver +
proven book let us PROVE that our generic (solver-free) near-optimal proxies actually track optimality, before we
point them at a game we can't solve. The Connect-4 AUDIT ceiling stays `wins_as_p1_vs_EXACT_oracle == 1.0` (a model
that beats the exact solver as P1), but the go-forward THRUST is the efficient generic loop below, not the solve.

**Definition of SOLVED (measurable, no proxies):** M is optimal iff, as first player, it converts the proven
first-player win against the EXACT solver 100%. Corroborated by: M's entire main line is proven-optimal
(`optimality_verified_plies` = full game, `first_blunder_ply` = none) and, as second player, M never loses a
drawn/won position. Only a win vs the EXACT solver counts — a depth-limited oracle is a beatable PROXY (evidence,
not proof); the viewer gates every ✓ / "solved" on the exact label.

#### Shipped foundation (context — the measurement + reference apparatus)

The measurement/reference apparatus below exists and is green (TDD) — it is the CALIBRATION HARNESS (exact solver,
proven book, optimality ladder, verify_solved). The go-forward WORK is the efficient generic LEARNING loop (Gumbel
+ Reanalyze + targets, next subsection), NOT more of this — this stays as the audit oracle + optional teacher.

- **Store + solver.** `harness/tablebase.py` (PROVEN/ESTIMATE columnar store; `proven_value` so exact consumers
  ignore estimates; priority eviction; symmetry-canonical keys). `harness/solver.py` (Pons bitboard negamax +
  persistent symmetry-canonical TT; `OracleAgent` exact / `NearPerfectOracle` depth-limited). `SolvableGame`
  hooks on `connect4` + fully-solved `tictactoe`.
- **Book builder** `harness/book.py` — bounded + resumable + accumulating, modes: SEED (midgame subtrees), GRADED
  (book-aware bounded-search ESTIMATEs for the deep opening), PARALLEL/deadline-safe exact accumulator (`workers`,
  `max_position_seconds` → DEFERRED, no hang), and WINNING-STRATEGY (`prove_winning_strategy` — the directed M1
  tree). Plus `book_optimal_actions` / `principal_variation` / `play_until_decided` / `book_coverage` /
  `winning_strategy_coverage`.
- **Agents.** `BookAgent` (opening book + exact endgame + fallback); MCTS-Solver in BOTH cores (`MctsAgent`,
  `AlphaZeroAgent`) — proof leaves + SELECTION/propagation (a node is a proven win the instant one child is a
  proven loss); `ExactOptimalAgent` (generic exact oracle). Exact-endgame cutoff ON by default for deployed nets
  (`DEFAULT_AZ_SOLVE_ENDGAME=22`).
- **SOLVE-IT M0–M3 machinery (2026-08-21, TDD — `tests/test_solve_it.py`).** M0: `benchmark.p1_conversion` /
  `optimality_ladder` (depth-6→8→10→12→EXACT frontier) / `verify_solved` (the ONE gate that may say "perfect");
  tournament `oracle_exact` + opt-in `ladder` (verdict `optimal`/`solved` unreachable via the proxy →
  `converts-proxy`). M1: `prove_winning_strategy` proves the strategist's directed tree (our one optimal move +
  every opponent reply), bounded/resumable/endgame-back; `winning_strategy_coverage` is the honest tracker. M2/M3:
  BookAgent converts vs the exact solver; `book_distill_examples` targets = exactly the proven optimal set + exact
  value. Demonstrated end-to-end: tic-tac-toe fully solved (both seats never lose) + a real C4 forced-win subtree
  converts.
- **Distillation + net.** `oracle_distill_games` (value-relabel from the proven book), `book_distill_examples`
  (soft targets, proofs oversampled), `_mix_training_set` anti-drift anchor (`distill_fraction=0.34`),
  `train_alphazero(book=…)`.
- **In-app + honesty.** `Exploration → Start → autopilot → build-book / improve / rate / play-off`, all
  manifest-driven (`bookBuild` numbers+booleans, `improve`, `rate`); Start cycles in minutes; the viewer never
  overclaims (book coverage split proven/estimate; play-off ranks by Converts-P1; ✓ gated on the exact label;
  winning-strategy `provenFraction` + ladder frontier surfaced).
- **Solver speed (settled).** The 158s opening wall is the pure-Python execution tax (~30–100× vs C++), already
  at Pons's fastest; amortization via the bottom-up book delivers on-demand speed with ZERO dependency (a booked
  frontier collapses a 3008-node solve to 0). If cold-arbitrary speed is ever needed: an iterative explicit-stack
  Numba rewrite (Numba could not compile the recursive negamax) or bind a compiled C solver (bitbully) behind the
  connect4 hook — both optional, off the critical path.

#### VALIDATED FOUNDATION — the generic near-optimal recipe (2026-08-24, superseded as the FORWARD thrust by §C.6)

This §C.5 investigation is COMPLETE; it produced the first portfolio entry (the large-perfect-info arm). Blow-by-blow
detail lives in memory `project_modeltrainer_solve_it_shipped`; the forward action is now §C.6 (the process portfolio).

**VALIDATED GENERIC RECIPE** — MEASURED on the C4 calibration harness, distillation OFF, all levers game-agnostic
(→ chess): **Gumbel/Sequential-Halving completed-Q SEARCH (deploy + self-play) + calibrated completed-Q policy target
(fixed-range σ `_norm_q`, c_scale=0.1) + n-step value target off a lagged target net (n≈8, k=2).** All in
`harness/neural.py`, TDD (`tests/test_neural.py`, 51-pass suite). oracle-match 0.887–0.893, +0.067/+0.087 over the
visit-count baseline (2-seed replicated), dominant at EVERY sim budget under both searches. **Leverage ranked by
measurement (the key finding, via the disentanglement grid net×deploy-search×sims): value-target ≫ deployment-search
(Gumbel) > policy-target (completed-Q, neutral alone); Reanalyze not-a-win at this small scale (data-limited/large-scale
lever — primitives shipped).** The completed-Q min-max-normalisation bug (near-one-hot target from search noise) was
caught by a 5-agent adversarial workflow vs Danihelka 2022 and fixed (fixed-range σ) — the methodology, not just the
result, is the asset.

**Solver-optional MEASUREMENT (the generic part):** sim-scaling curve; search-consistency KL(prior‖post-search)→0;
best-response robustness. Oracle AUDIT (C4 only): oracle-move-match + depth-6→8→10 conversion ladder + `verify_solved`.
Calibrate the oracle-free proxies against the oracle ONCE, then rely on them for chess.

**SCALE-UP FINDING (24×48, distill OFF) — the motivating gap for §C.6:** strong MID-GAME (oracle 0.887) but
**frontier=none on the opening-conversion ladder** (does not convert the P1 forced win). Pure self-play generalises
tactics fast but is slow on precise OPENING theory (the last mile to *solved*); a teacher (distill/book) fixes it but
is game-specific. "Best process" bifurcates on **has-teacher?** and **need-solved-vs-grandmaster?** — exactly what the
portfolio must select on. (Reference: the brute-force winning-strategy book grind is built/correct/resumable and kept
as an audit oracle + optional teacher, NOT the generic thrust.)

#### Design decisions (resolved — reference)

- **(a) Estimator = bounded SEARCH (MCTS-Solver self-play), never the raw net value.** The book must be an
  INDEPENDENT reference that CORRECTS the net (net-sourced estimates are circular: book ≈ net → distilling
  book→net teaches nothing). The net may LATER serve as the search PRIOR, but the stored estimate is always the
  search result.
- **(b) RECONSTRUCT PVs from stored optimal moves; never persist explicit paths** (`principal_variation` walks the
  optimal set to a terminal; a proven line reconstructs in full, a thin region yields an honest partial line).
- **(c) HYBRID upgrade cadence:** EAGER for the free upgrade (all children booked → minimax lookup, keeps
  proven-coverage monotone every build); ON-DEMAND for the expensive one (children not all booked → new search
  only when queried / region-focused, never speculatively every pass).

#### Scaling doctrine — Connect 4 → Checkers → Chess → Go (what "solve" means, and what we'd need)

Stress-testing the design against chess (state ~10^46, tree ~10^123, UNSOLVED) confirms the architecture and
names the gaps. Every strong engine is the SAME four organs — a **proof store at the edges**, a **cached book**,
a **learned evaluator**, and a **search that backs proofs up** — differing only in which organ carries the
weight. Our components already map 1:1: Tablebase PROVEN layer = endgame tablebase; `build_book` = opening book;
the AlphaZero net = the learned eval; the `evaluate` ladder / MCTS = the search. **The PROVEN/ESTIMATE split IS
how real engines actually work** — Stockfish's Syzygy WDL/DTZ = our PROVEN; its NNUE eval = our ESTIMATE; a value
graduates to PROVEN only on a tablebase hit or a resolved terminal, exactly our ladder. So for a chess-class game
"solve" HONESTLY becomes **"play near-optimally; proofs exist only at the edges (≤7-man tablebases + forced
mates)"** — the proven fraction is ~10⁻³¹ of the state space. Publish `book_coverage` as the headline; never call
it solved.

**Tiers (what changes as complexity grows):**
- **Tier 0 — Connect 4 class (≤~10²¹), SOLVABLE.** Current design is correct and complete: BFS-enumerate, exact
  solve bottom-up. "Solve" = literally weakly solve. No change.
- **Tier 1 — Checkers class (~10²¹–10³¹), SOLVABLE via retrograde (Chinook).** Add a generic RETROGRADE endgame-DB
  builder (backward from terminals) + a forward best-first PROOF-TREE driver that stops each line on a DB hit.
  Full-frontier BFS is replaced by retrograde + best-first forward proof.
- **Tier 2 — Chess class (~10⁴⁶), UNSOLVED.** Abandon enumeration and rollouts. Must have, in order: (a) a game
  plug-in with bitboards + full legal move-gen + **incremental Zobrist** behind the hooks (`symmetries()` =
  identity — no board symmetry to exploit); (b) a learned eval as the PRIMARY strength (NNUE — a tiny int8
  incrementally-updatable net under alpha-beta — vs a large policy/value net under MCTS); (c) a real search
  (alpha-beta+TT or PUCT-MCTS) using the net as the ESTIMATE and backing proofs up; (d) endgame-tablebase IMPORT
  (Syzygy) + probe-in-search as PROVEN entries; (e) a SAMPLED / best-first book with drop-out tolerance
  (reach-probability priority), never enumeration; (f) forced-mate detection propagated as exact proofs.
- **Tier 3 — Go class (~10¹⁷¹), UNSOLVED.** Tablebases vanish; proofs shrink to forced sequences; strength is
  entirely net + MCTS (KataGo). "Solve" = "play superhuman"; PROVEN fraction → 0.

**The 5 concrete gaps in our current design** (each with the generic abstraction it needs):
1. **Rollout estimator** (`estimate_position`/`_rollout_outcome`) is tactically blind past ~10¹² states → the
   `evaluate` seam already takes a pluggable `estimator`; supply a **search+net Evaluator** (alpha-beta+NNUE or
   PUCT+net) at the leaf and retire the full-game-rollout path for hard games.
2. **Full-frontier enumeration** in `build_book` explodes at branching ~35 → a **sampled / best-first
   `FrontierSource`** yielding `(position, reach_probability)`, stored by priority (the Tablebase already evicts
   by priority). Enumeration stays only as the Tier-0/1 path.
3. **Connect4-specific solver fallback** — `book.py` falls back to the Pons bitboard when a game omits hooks;
   that's nonsense for any other game. Route ALL exact solving through the game hooks and **DELETE the connect4
   fallback from the generic layer** (matches "thefactory stays generic"). Add a generic retrograde routine + an
   external-tablebase import hook.
4. **Board-shape-specific net** → derive the net input from `observation()` with a **swappable architecture**
   (NNUE for alpha-beta, or a residual net for MCTS) + a training loop that scales.
5. **Symmetry assumptions** → plain incremental Zobrist with `symmetries()` = identity where none exists (the
   ~50% mirror saving simply vanishes — set the expectation, it's not a regression).

**Cross-cutting invariants to bank now:** keep the PROVEN/ESTIMATE gating (`proven_value`); make the estimator a
pluggable search+net Evaluator; swap enumeration for a sampled FrontierSource once bᵈ exceeds budget; put all
exact solving behind hooks; require per-game incremental Zobrist; live search must prefer proofs over estimates;
report an honest coverage number as the headline.

#### Deferred phase — "Lean-Model Frontier" (make finding the best model the best it can be)

The step AFTER the general system is proven — runs once the Connect-4 SOLVED bar is reached by SOME architecture
(`oracle_optimality_rate ≥ 0.99` AND `wins_as_p1_vs_oracle == 1.0`). GOAL: the **leanest + fastest** net that
still holds the SOLVED bar — the winning point on the **strength × cost** Pareto frontier, not the biggest net.

**Why lean, and why "more ResNet depth won't help HERE" (turned from assertion into a measured number):** on
SIMPLE/near-solved games capacity SATURATES (the AlphaZero-Zipf study shows Checkers/Oware Elo scaling *negatively*
past a size threshold), so on fully-solved Connect 4 adding blocks is predicted inert — matching our measurement
(a 32-ch 2-conv net already ~0.983 late-game; the residual gap is OPENING coverage + SEARCH, localised by
`optimality_verified_plies` / `first_blunder_ply`, not value-head capacity). NNUE is the doctrine's exemplar: a
tiny cheap net + huge search beats a big net + shallow search, because the cheap net BUYS more search — so spend
the capacity budget on search + coverage, not parameters. Capacity only re-enters as a lever at Tier 2+.

**Method — reuse the exploration autopilot we already have** (it's lever-agnostic + multi-objective Pareto is
already wired); the additions are small and mechanical:
1. Add architecture LEVERS: `az_channels`, `az_blocks` (+ optional `az_residual`/`az_quant`) on the config,
   threaded into the net (today hardcodes 32ch/2-conv), declared model-scoped in the manifest.
2. Add a **net-cost metric** to the cost block (`paramCount`, `checkpointBytes`, `msPerMove`) — the one real gap;
   compute-cost is captured, net size/latency isn't.
3. Point the autopilot at those levers with `fitness = [oracle_optimality_rate max, params-or-ms/move min]` →
   `qualifyParetoBasins`/`paretoFrontier` emit the strength×cost frontier for free.
4. Multi-fidelity: `az_iterations` as the Hyperband/ASHA budget ladder; **fast proxy fitness** = fixed-corpus
   `oracle_optimality_rate` + `optimality_verified_plies` (gated above `archiveNoiseFloor` so we home in without
   chasing noise), with zero-cost NAS proxies (NASWOT/SynFlow) to pre-prune obviously-bad shapes and
   distillation-top-1-vs-oracle for cheap ranking. (Supernets/OFA/DARTS are OVERKILL for a ~dozens-of-configs
   space — noted.)
5. **Reason about which lever matters, measured:** fANOVA TOTAL-effect (already in the types) — a lever whose
   total-effect on optimality sits at/below the noise floor is INERT; that is how "depth is inert HERE" becomes
   data (`az_channels`/`az_blocks` below floor while `az_distill_games`/`az_sims` carry the variance), gated by
   `LeverImportance.confident`. Plus AblationPath + controlled single-lever sweeps + sample-efficiency curves.
6. Then DISTILL the champion into the leanest arch that holds the bar (book/oracle teacher), and prune + int8
   (the NNUE recipe) for deploy latency — reported on the cost axis, never regressing the bar.

**Measurable success:** the autopilot emits a Pareto frontier with ≥1 point at `oracle_optimality_rate ≥ 0.99`
AND `wins_as_p1_vs_oracle == 1.0` whose params/ms-move ≤ the 32ch/2-conv baseline; the fANOVA total-effect of the
capacity levers is below `archiveNoiseFloor` with `confident == true` (the recorded "capacity wasn't the
bottleneck — coverage + search were"); a distilled student within 0.01 optimality of the teacher at ≤ its param
count; an int8+pruned deploy net that holds the bar with a measured ms/move reduction. HONESTY: this is a
compression / frontier phase, not strength-discovery — it can only trim a net that ALREADY solves.

#### Honesty rails

- The book is only as sound as its solver; a value-only entry is a proof of outcome, not of the line — playing
  optimally from it still needs the one-ply lookahead (cheap) or the endgame solver.
- For unsolvable games the "book" holds *beliefs* (agent-consensus), not proofs — label it as such in the UI.
- Report book coverage honestly (how much of the reachable opening is exact) so "optimal" is never overclaimed.

### C.6 — The process PORTFOLIO & meta-selection framework — THE MAIN COURSE (2026-08-25)

**The mission.** One recipe won't fit all games. Build a PORTFOLIO of learning processes with CHARACTERISED trade-offs
+ a META-SELECTOR so a NEW unseen game is auto-classified and routed to the best process to reach solvability (where
feasible) and ≥ grandmaster play, FAST — up to real-time games (StarCraft/Dota). Then, over many games, learn the best
way to build such models. Literature-grounded (8-agent survey, folded here from the retired `game-learning-portfolio.md`).

**THREE GOVERNING DIRECTIVES (user, 2026-08-25) — apply to every task under §C.6:**
1. **MEASURE, don't argue.** The transferable asset is the METHOD (disentanglement grid → leverage ranking → learn
   features→process online), not any single recipe. Every lever/process claim is a measured A/B on the calibration
   harness with a pre-registered metric·corpus·threshold.
2. **Build as reusable, CHAT-REACHABLE modeltrainer TOOLS — never one-off scripts.** Any capability we need becomes a
   first-class harness capability + a chat/agentic tool (the LLM-tool-parity rule), especially anything reusable across
   OTHER trainings/model work. The current `scripts/gumbel_ab.py` / `gumbel_disentangle.py` / `scale_up.py` are
   PROTOTYPES to be promoted into capabilities + activities + chat tools (see "Tools to build" below); delete the
   scripts once promoted (repoint-and-delete).
3. **PROVE it's good, then rinse-and-repeat improve ALL areas.** Producing the process is not enough — we must
   rigorously PROVE a produced model/process is good (the evaluation spine below), publish that evidence, and loop:
   survey → hypothesise → test → prove → improve every area → add a game → repeat.

**GOAL MET — Exploration produces a PROVEN near-optimal Connect-4 model (2026-08-25).** Trigger `Exploration → Start`;
the autopilot now runs `improve` (the validated recipe) → `play-off` → **`score`** (new finalize step → the
`process-eval` scorecard on the champion). The scorecard's HONEST verdict on the champion: **"near-optimal
(exact-proven)" — converts 8/8 (and 16/16 at larger K) PROVEN forced wins vs the EXACT solver** (exact, not a proxy,
via few-empty forced-win roots that sidestep the opening wall), net-alone optimal on late/mid positions (1.0 on ≤16
empties, ~0.94 at ~26 empties), converts depth-12 near-perfect 6/0/0. HONEST framing: the DEPLOYED model (net +
exact-endgame cutoff ≤22 empties) is the near-optimal product — perfect endgame (solver) + strong net; the NET ALONE is
~0.92–1.0 depending on depth (opening is the ~gap). **100% requires the full opening book** (`build-book`) — the user
accepts ~99% + proof, which is met. `net_oracle_match` (solver OFF) is reported alongside `oracle_match` so the solver
is never credited for the net's optimality. All encoded as REUSABLE trainer capabilities (per directive): `benchmark`
`sample_forced_win_roots`/`verify_forced_win_conversion`, `process_eval` forced-win + net/deployed split, the autopilot
`score` action.

**VERIFIABLE MILESTONE (2026-08-25): the trained model beats NEAR-PERFECT play as first player.** Model (recipe +
teacher + diverse-openings, via the `harness.run` tool) as P1 converts the forced win **6/0/0 vs depth-10 AND 6/0/0
vs depth-12** near-perfect oracles, mid-game oracle-match **1.0** — i.e. on the CANONICAL line it plays essentially
perfectly (depth-12 blunders only in deep endgames the model's exact cutoff handles). Verify via `process-eval` /
`p1_conversion`. THE EXACT gate (vs the true solver) is a MULTI-HOUR grind (the opening wall: the exact oracle
re-solves the opening for each of its ~20 moves/game) → NOT interactively runnable. `books/connect4.tt` is NOT
present in this checkout (the earlier 61k book wasn't committed), so warming the solver needs a `build-book` grind
FIRST; until then depth-10/12 near-perfect is the strong runnable proxy. NB the milestone uses solve_endgame=22 → the
ENDGAME (≤22 empties) is solver-perfect; the OPENING+midgame (>22 empties) is the NET — a legitimate deployed model.
Remaining gap = OFF-LINE ROBUSTNESS (from diverse openings the net still loses ~46%, partly the opening_plies
confound of random-opening lost positions); improving via diverse self-play + broad optimal distillation (trend:
off-line not-lost 33%→54%).

#### §C.7 — PURE self-play is not enough (measured), and the #1/#2/#3 process (2026-08-25)

**PURE self-play negative result (net-vs-net, NO crutch — the honest generic test).** 100 iters × 120 games ≈ **12,000
games**, Gumbel + 8-step value, 48 sims, 32-ch net, `az_pool_frac=0` (verified `0 vs-pool` every iter — the earlier
"pure" run was CONTAMINATED by a `NearPerfectOracle(depth=6)` league opponent at `pool_frac=0.35`; fixed by the new
`az_pool_frac` knob). Champion `ab21b4bbc345`. Measured (net-only, no solver): **oracle-match 0.79** (~21% of moves
diverge); **opening value 0.10** (true = +1 forced win — the net never learned P1 is winning); two champions from
diverse openings **P1 W/D/L 0.46/0.11/0.43** (a proven P1-win game a near-optimal pair ~never loses as P1); ladder
frontier **none** (doesn't clear a depth-6 oracle off-line); P1 vs depth-8 with diverse openings **8W/3D/13L (loses
54%)**; converts **14/16** proven forced wins with search (loses 2 outright). `process-eval` verdict: **"strong but not
near-optimal"** — main-line strong (beats depth-8 100% at every sim budget on its canonical line, late-game match
0.925), OFF-LINE brittle. So the earlier "near-optimal" champion's strength came from the SOLVER crutch (oracle league +
distillation + endgame cutoff), NOT the generic process. This directly answers the user's intuition: 12k pure games at
this compute do NOT robustly solve even C4.

**The target process (user, 2026-08-25) = #1 → #2 → #3, confirmed against the code:**
- **#1 AlphaZero self-play → optimal: BUILT.** The pure run IS #1; at this compute it's strong-but-brittle. In-#1 levers
  to close the gap: more sims, bigger net, more iters, forced diverse-opening coverage.
- **#2 record endgame while playing → drastic speedup: BUILT (2026-08-25, TDD, 8 steps).** **HONESTY CORRECTION (forced
  by the design review): pure #1 training solves ZERO endgames** (the learner has no book/solve during self-play), so #2
  is NOT "avoid re-solving" — nothing to amortise there. The real mechanism is **exact endgame VALUES as training
  targets → the value head converges to a fixed quality target in fewer iters/wall-clock**, plus memoising the solves #2
  itself introduces. The loop (all in `harness/`): one run-owned **value-only** `Tablebase`; `AlphaZeroAgent._proven_value`
  is a **write-through memo** (solve each endgame ONCE via `position_value(...,book=self.book)`, `put_proven` value-only,
  `endgame_solves`/`endgame_hits` gauge); `self_play_game` gains `exact_value_targets` (mover-relative direct-assign
  override); `extend_endgame_frontier` = end-of-iteration **budgeted retrograde climb** via `book._prove` only (free
  minimax + winning-child + ≤`max_empty` cheap solve — inherently bounded, never `_prove_bounded`'s unbounded solve that
  hangs on worker threads); `train_alphazero` wires it behind `_endgame_enabled(game, tb)` (needs `canonical_key` +
  `exact_optimal_actions` → **generic degradation**: byte-identical to #1 when absent); `run.py` builds/persists the
  run tablebase (200k cap given the OOM history), summary block `alphazero.endgame` {booked, frontier_empties, solves,
  hits} + `endgame_hit_rate`. **Value-only avoids the `best_actions` mirror column-flip bug** (policy-distill deferred).
  8 `az_endgame_*` levers in `.factory/trainer.json` (default OFF) → chat-reachable via train/side-experiment. Tests:
  `test_endgame_selfplay.py` + additions to `test_neural.py`/`test_config.py` (99 pass, no regressions). Smoke run: iter
  2 = 44 solves / 104 memo-hits, store 57, frontier climbed to 36 empties. **PROVING speedup:** compute-to-target A/B
  (τ = net-only `oracle_optimality_rate` ≥ 0.85, per-iter `opening_value` proxy now in history), matched seed, ≥2 seeds
  — RUNNING. `endgame_hit_rate` is reported as a boundedness gauge, NEVER as the vs-#1 number.
- **#3 robustness / rule learning: partially built.** The self-play distribution + ladder ARE the exploitability signal
  (the 43% off-line P1-loss is the "exploit the shortcomings" number). Next: exploitability-descent training (freeze
  champion, train adversary, fold refutations into the buffer) + forced diverse-opening coverage; rule/invariant
  learning later.

**#2 PROOF (A/B, 2 seeds, 2026-08-25):** endgame-on vs pure-#1, matched seed, net-only τ=oracle_optimality_rate.
**Both seeds: B(#2) 0.9667 vs A(#1) 0.9500, Δτ=+0.0167** (B booked ~48-54K endgame positions). Small but perfectly
consistent — on the late-game corpus the 20K-param net is near-saturated (both clear τ=0.85 at 0.95), so exact targets
have little room; #2's payoff should grow with the scaled net + at the opening/off-line frontier. This REINFORCES that
capacity is the dominant lever.

**WHY PURE SELF-PLAY UNDERPERFORMED (4-lens research, 2026-08-25) — no wall of principle, we under-resourced it.**
Pure self-play DOES near-solve C4 in the literature (agent-built AZ beat Pons 7/8; AlphaZero.jl near-solver). Root
causes ranked: (DOMINANT) **under-capacity net** — NOT the 5x5 receptive field (a red herring: linear heads flatten
the whole board, sight is global) but the **bare single-Linear heads** that cannot represent the nonlinear AND-of-threats
(forks) or odd/even threat-parity; ~20.6K params vs the ~1.6M (5-19 blocks x128) reproduction floor, 50-1000x under.
(DOMINANT, coupled) **value-target chicken-and-egg** — equal-strength self-play from a won-but-unconverted opening → ~50/50
outcomes → MSE-optimal value ≈ 0 (= our opening_value 0.10); +1 only emerges once the policy can convert, which the
capacity ceiling prevents. (MAJOR) **compute under-resourcing** (48 sims vs 200-800; buffer 8-24K vs 400K-1M → forgets
off-line lines; N=1 seed) — real but secondary. Plus two METHOD BUGS scaling won't fix: (a) **n-step SUPPRESSES opening
value** at this net size (measured pure-MC +0.38 vs n8 +0.26) — contradicts the code's "binding lever" claim; (b) champions
crowned under plain PUCT though trained with Gumbel.

**SCALED-NET BUILD SHIPPED (2026-08-25, TDD, full suite 249 pass).** Net capacity is now CONFIG-DRIVEN (§C.7 levers):
`Connect4Net(channels, blocks, residual, batchnorm, head_hidden)` — DEFAULT reproduces the legacy 20K net (306 old
checkpoints + tests unaffected), `residual=True` builds a ResNet tower + MLP head towers; save_net/load_net persist the
arch (old blobs → legacy). Levers `az_channels/az_blocks/az_residual/az_batchnorm/az_head_hidden` in config + manifest
(chat-reachable). Method bugs fixed: n-step honest default (manifest 8→0; comment corrected), deploy-operator match
(`_az_deploy` passes gumbel to eval + promotion gate), net_value docstring honesty. Warm-start arch-guard (won't pin a
legacy shape onto a scaled config). **Batched/resumable driver `harness/scaled_run.py`** (the user's requirement — train
in BATCHES, checkpoint each, resume): pure self-play, per-batch metrics = opening_value on the TRUE empty board +
net-only oracle_optimality_rate + **off-line P1-loss vs depth-8 oracle from 50 fixed openings** (the pre-registered
robustness metric). **FALSIFIABLE TEST RUNNING:** 1.79M-param net (128x6+BN+head64), 96 sims, pure-MC, 320 games/batch,
24 batches, seed 0 (seeds 1-2 to follow). Verdict CONFIRMS under-resourcing if off-line P1-loss ≤15% AND opening_value
≥+0.7 (95% CI excl +0.3); REFUTES (residual problem) if loss >40% OR opening_value ≤+0.3. See
[[project_modeltrainer_endgame_from_play]].

**Immediate driving case (user-chosen): push Connect-4 to SOLVED — PROGRESSING, with an HONEST correction.**
**(2026-08-25) recipe-into-real-training-path + teacher → main-line opening SOUND but the net is MAIN-LINE-BRITTLE.**
A run via the wired `harness.run` tool (recipe: gumbel + n=8/k=2 + distillation teacher, 10×24) → mid-game oracle-match
**1.00**, `first_blunder_ply` **−1** (no blunder through 19 verified plies), `opening_value` **+0.45**. The FIRST
`process-eval` read (single deterministic line) looked great — "converts depth-6 16/0/0" — but that was an OVERCLAIM
(the ladder even inverted: depth-4=0.0/depth-6=1.0). **PROVE-IT-GOOD RIGOR FIX (SHIPPED, TDD): `opening_plies` (diverse
random openings) added to `p1_conversion`/`optimality_ladder`/`process_eval` (default 2).** The robust re-measure
CORRECTED the record: **vs depth-6 with diverse openings the net goes 8W / 0D / 16L (24 games) — it LOSES 67% of
off-line games, never draws.** So the net is teacher-line OVERFIT: perfect on the canonical line + mid-game corpus, but
it LOSES away from the main line (a near-optimal model never loses a drawable position). The rigor fix EXPOSED this —
the deterministic ladder had hidden it. **NEXT — the real gap is ROBUSTNESS, not opening soundness:** the net needs
OFF-LINE coverage — diverse self-play exploration (the generic loop's job) + the teacher, so it doesn't collapse away
from the main line. Add a robust-DEFENCE audit too (oracle rng tie-break, keeping the model's win intact) alongside
`opening_plies` (which forfeits some wins). Then climb depth-8 → depth-10 → EXACT (`exact:true` verify_solved gate).
The generic (distill-off) arm stays the transfer story.

**(historical) pure-generic 24×48 net** — tactically STRONG mid-game
(distance-to-optimal 0.88) but **LOST the opening as P1 — 0 wins / 0 draws / 10 LOSSES vs even a depth-4 oracle**
(frontier=none). Not just failing to convert — actively playing into losing lines a SHALLOW defender exploits (never
discovered the centre-first forced win; distillation-off self-play didn't explore the opening enough, and the balanced
self-play outcomes leave the opening VALUE ~0 so there's no gradient toward the win). A red flag to explain: sim-scaling
INVERTS (8 sims beats the ref, 32 loses it) — more deterministic search converging to a WORSE opening move (or the
~1-deterministic-line small-sample at games=10; re-measure with more games/opening-plies). **ATTACK (generic first,
teacher last):** (a) OPENING EXPLORATION — the missing lever: KataGo playout-cap randomization + forced-playouts, and/or
stronger early-move exploration, so self-play actually visits centre-first; (b) VALUE accuracy at the opening; (c)
SCALE; (d) TEACHER (book/oracle opening distillation) only as the last resort / ceiling reference since it doesn't
transfer. Every piece is built GENERIC + chat-reachable so it serves the whole portfolio.

#### Taxonomy — the axes that DECIDE which process wins (each cheaply probed off the `Game` interface)

| Axis | Values | Why it FLIPS the process (+ detection) |
|---|---|---|
| **Information** | perfect vs imperfect/hidden | THE switch. Hidden info ⇒ solution is a stochastic **Nash**; minimax/PUCT over `state_key` is UNSOUND (exploitable; self-play cycles). Route → CFR/ReBeL/R-NaD; metric → **exploitability/NashConv**. Detect: does `observation` reconstruct `state_key`? |
| **Determinism / chance** | deterministic \| small-known \| large/unknown | `step` random ⇒ targets become EXPECTATIONS ⇒ value-target VARIANCE dominates (attacks our #1 lever). Small-known → afterstate + expectimax; unknown → Stochastic-MuZero. Detect: fixed action × rng seeds → distinct outcomes. |
| **Solvability tier vs budget** | strong \| weak \| ultra-weak \| unsolvable | Certificate vs Elo, and whether an exact oracle exists to calibrate. Storage/verification is the *strong*-solve wall, not search. |
| **Action structure** | small-discrete \| combinatorial \| continuous \| simultaneous | Child enumeration needs small-discrete. Else Sampled(+Gumbel) + autoregressive/factored heads + masking; simultaneous breaks `current_player`. |
| **Players & sum** | 1 (MDP) \| 2p0s \| n-player/general-sum | 2p0s well-behaved (Nash exists; negamax duality). Beyond it → league/population over single-track self-play. |
| **Reward density & horizon** | dense \| sparse-short \| sparse-long | Long-sparse ⇒ must BOOTSTRAP (our n-step lever) + often shaping/imitation warm-start. |
| **Tree non-uniformity / transpositions** | balanced \| forced-threat \| DAG-heavy | Sub-router in the solvable class: alpha-beta vs **df-pn/threat-space**; high transposition ⇒ GHI handling. |
| **Simulator cost** | cheap known \| unknown/expensive | Cheap ⇒ real model (AlphaZero). Else learned latent model (MuZero) pays its approximation error (our grid: latent/Reanalyze net-negative on a cheap-sim game). |
| **Real-time / latency** | untimed \| real-time tick | Latency ⇒ reactive forward-pass policy, search → train-time ⇒ model-free actor-critic + league. Our search+completed-Q recipe is the WRONG default here. |
| **Symmetry richness** | rich \| weak/none | Storage multiplier for strong solve + free sample multiplier (augmentation). `SolvableGame.symmetries()`. |

#### The process portfolio (trade-offs + repo maturity)

| Process | Best for | Trade-offs | Repo |
|---|---|---|---|
| Forward alpha-beta/negamax + TT + ordering | solvable-PI small branching; also the calibration ORACLE | lowest wall-clock/mem to a weak result, zero training; proves one root only | **shipped** (C4-specific) |
| Backward retrograde / tablebases | strong solves that fit storage | total coverage + O(1) perfect play; enormous compute+**storage** (the wall), needs invertible moves | **partial** (store; no predecessor hook) |
| Best-first proof search (PNS/df-pn + GHI, threat-space) | high-branching forced-threat (Gomoku/Qubic) | best node-efficiency on skewed trees; memory-hungry, GHI risk | **absent** (top solver gap) |
| **Validated AlphaZero recipe** (Gumbel completed-Q + calibrated policy + n-step value/lagged net) | large unsolvable PI, cheap sim, enumerable actions | strong generic default, guaranteed low-sim improvement; high self-play compute, UNSOUND under hidden info | **shipped** |
| KataGo efficiency stack (playout-cap, forced-playouts+prune, global pool, aux heads) | max Elo-per-compute on any board game | ~50× aggregate savings, mostly game-agnostic; new hyperparameters | **absent** (highest-value transfer) |
| MuZero family (learned model+Reanalyze; Sampled; Stochastic; EfficientZero) | unknown/expensive dynamics, large/continuous chance/actions | max generality; model-learning compute + latent bias; only MATCHES AZ on cheap-sim PI | **partial** (reanalyze; measured not-a-win small-scale) |
| Afterstate TD(λ) + expectimax/\*-minimax | stochastic small-known-chance + clean afterstate (2048/backgammon/Pig) | best sample-eff + wall-clock; caps without search, needs hand afterstate | **absent** |
| CFR family → Deep-CFR/NFSP → ReBeL/SoG → R-NaD | hidden-info 2p0s across scale | ONLY sound family under hidden info; tabular exact but O(\|infosets\|). **Our lagged target-net ≡ R-NaD's lagged reference** | **absent** |
| Model-free actor-critic + league (PPO/IMPALA/SAC; PFSP+exploiters+PBT) | real-time, partial-obs, huge/continuous/multi-agent | only viable where planning is impossible; sample-hungry, no certificate, shaping can distort objective | **absent** (Elo ladder + Agent registry are substrate) |
| **Meta-selector** (SATzilla/Rice hardness-selection + AutoRL PBT/SH) | the algorithm-selection problem itself | low decision-time compute for large savings; cold-start extrapolates badly OOD | **absent** (disentanglement grid = first labelled points) |

#### Meta-selection framework — the decision procedure for an unseen game (build as a chat tool)

1. **Probe features** (no training, short rollouts + flags): chance branching, information (observation-partition vs
   state_key), players/sum, action metrics, horizon, reward density, symmetry, solvability tier.
2. **Rule-based router (presolver)** → a CANDIDATE SET (never one guess): imperfect → {CFR+/Deep-CFR/ReBeL/R-NaD by
   scale}; chance → {afterstate-TD+expectimax / Stochastic-MuZero}; real-time/simultaneous/continuous → {model-free +
   league}; else PI → {retrograde / alpha-beta / df-pn / validated-AZ+KataGo} by tier + tree shape.
3. **Score candidates on the generalised disentanglement grid** (class-correct metric: distance-to-optimal vs oracle
   where one exists, else Elo±CI ladder w/ common-random-numbers, imperfect-info gated on exploitability). AutoRL inner
   loop. Pick least regret-to-target per compute.
4. **Learn online.** Every run appends a (features → per-process performance) point; upgrade the router to a learned
   predictor. Meta-success = low regret vs oracle-best on a HELD-OUT game suite at fixed budget.

#### The PROVE-IT-GOOD evaluation spine (directive #3 — a first-class, chat-reachable capability)

A produced model/process is not "good" until PROVEN so, honestly. The spine (extend the shipped `benchmark`/tournament
capabilities, expose each as a chat tool): (a) **distance-to-optimal** vs the exact oracle where one exists
(`optimality_rate`, `verify_solved`, the ladder); (b) **certificate check** (H9: tablebase self-consistency — a net at
100% corpus-optimal is NOT solved until off-corpus positions verify); (c) **strength-per-compute** (sim-scaling curve,
AUC); (d) **oracle-free proxies** for the chess regime (search-consistency KL→0; best-response/exploitability
robustness) calibrated ONCE against the oracle; (e) **statistical rigor** (mean±CI, paired/common-random-number seeds,
≥2-seed replication — already our discipline). The evaluation IS a mission output (feeds §D publication).

#### VERIFYING NEAR-OPTIMALITY WITHOUT AN EXACT ORACLE (the chess-realistic crux — literature-grounded, 2026-08-25)

**Thesis (the answer to "how do you know it's near-optimal when you can't solve the game?"): you CANNOT certify
ε=0 solver-free — stop trying. Produce instead ONE rigorous one-sided BOUND + a set of CALIBRATED convergence GATES,
and NEVER confuse them.** There is exactly one solver-free method that genuinely BOUNDS distance-to-optimal at chess
scale: **approximate exploitability** (freeze the champion, TRAIN an adversary; Timbers 2020 ran it on chess/Go/HUNL).
In a 2p0s perfect-info game an optimal player concedes ≤ the game value v\* to ANY opponent, so a concrete adversary's
worst-seat excess over v\* is a valid LOWER BOUND on the champion's true exploitability — a POSITIVE reading PROVES
suboptimality by that margin; a ZERO reading is necessary-not-sufficient and budget-relative ("survives a 10M-game
adversary", never "unexploitable"). Everything else is either EXACT-but-only-on-a-solved-slice (tablebase WDL/DTZ
agreement — a real bound ≤7-man, silent above), or a CONVERGENCE PROXY (search-consistency KL→0, vanishing Gumbel
Δ→0, policy-value alignment, sim-scaling flatness, Elo saturation) that detects a FIXED POINT of the net's OWN operator
— which approximate policy iteration (Munos 2003; Bertsekas-Tsitsiklis) proves can be arbitrarily suboptimal when prior
and value err in the same direction (**the suboptimal-fixed-point trap**). Selling a convergence proxy as a bound is
the central over-claim risk.

**THE UNIFICATION (why this = our robustness work):** our "off-line loss rate vs a depth-k oracle from diverse
openings" IS a **Local-Best-Response (LBR) exploitability lower bound** (Lisý-Bowling 2016) — a fixed cheap responder
whose excess over v\* refutes near-optimality solver-free. So our net LOSING off-line = it is EXPLOITABLE = provably
NOT near-optimal. (a) verify-without-solver and (b) robustness are the SAME axis: exploitability. Driving losses→0 =
driving exploitability→0 = **exploitability descent** (Lockhart 2019; ApproxED 2025) — and our `vs_opponent_game`
(learn-to-beat a frozen opponent) is the substrate for both the ADVERSARY (measure) and adversarial TRAINING (reduce).

**Method ranks (BOUND vs proxy; all transfer to chess unless noted):** BOUND — approx-exploitability (learned
adversary; the claim-gate), tablebase WDL/DTZ agreement (exact on the solved shell; disable the model's solve_endgame
to grade the LEARNED policy), LBR (cheap always-on refuter), iterated adversarial-robustness gauntlet (R≥3 fresh
adversaries; necessary-not-sufficient safety), Williams-Baird value-residual bound (certified ONLY where the reachable
set is enumerable, else a correlate). CONVERGENCE-PROXY (cheap gates, refute-only) — KL(prior‖post-search)→0 (Grill
2020), Gumbel Δ→0 (Danihelka 2022), policy-value alignment, sim-scaling flatness (a STEEP curve REFUTES). TRACTABLE-ONLY
(C4 calibration, NOT chess) — weak optimal-SET membership (the warm-TT exact gate), winning-strategy proof from a
forced-win root, retrograde partial coverage.

**CALIBRATION (do ONCE on C4, on the shipped disentanglement grid):** for a ladder of C4 nets (bad→near-perfect)
compute the EXACT `oracle_optimality_rate` (x-axis = true distance-to-optimal), read every convergence proxy on the
same corpus WITHOUT the oracle, regress proxy→optimality, PIN the proxy value at the checkpoint where the exact rate
first ≥0.99 as the PASS threshold (expected anchors: KL≈0.05 nats, Δ≈0.0–0.02, misalignment≤2%), and PUBLISH each
proxy's **false-PASS rate** (its honesty tax). Validate the one transferable bound: on known-suboptimal C4 nets, confirm
the learned adversary recovers ≥~80% of the solver's EXACT best-response gap before trusting it solver-free. Recalibrate
opportunistically on any chess-scale exact anchor (Syzygy ≤7-man endgame subtrees).

**CHESS RECIPE (claim gated ONLY by the bounds; proxies are cheap pre-filters):** S0 calibrate on C4 · S1 cheap
convergence gates (any steep sim-scaling / fat-tailed KL REFUTES → stop) · S2 exact endgame floor (Syzygy WDL/DTZ =
1.000, solve_endgame OFF) · S3 LBR fast exploitability screen · S4 THE BOUND: learned approx-exploitability to L ≫
deployment compute (assume draw-under-perfect-play value floor) · S5 iterated R≥3 fresh-adversary safety. VERDICT: emit
"**near-optimal at adversary budget L**" only if S1–S3 pass AND S4 worst-seat ≤ v\*+ε (n.s. at p<0.01) AND S5 survives —
always reporting L, the tablebase-coverage boundary, and that a stronger adversary can always tighten the bound.

**TO BUILD (reuse our tools; each a `process-eval`/`disentangle` extension or a new capability):** the learned
approx-exploitability loop (freeze champion via `champions.py` + adversary via `vs_opponent_game`/`head_to_head` + the
C4 BR-gap recovery calibration) — the ONE bound; the LBR cheap-screen (wrap `NearPerfectOracle(depth=k)` as a
restricted best-RESPONDER emitting an LBR-value); convergence-proxy cards in `process_eval` (KL from
`completed_q_policy`/`_policy_value`, Gumbel Δ, policy-value misalignment — each reading its C4-calibrated threshold on
reached-state corpora); the calibration harness extending `gumbel_disentangle.py` (join exact rate vs each proxy, fit
the map, publish false-PASS rates); a single **near-optimality-verdict aggregator** that emits "near-optimal" ONLY when
the best-response BOUND passes AND tablebase agreement is 1.0 AND all calibrated proxies pass — structurally preventing
a convergence proxy from being sold alone as a bound. Chess adapters (Syzygy WDL/DTZ; STS/ERET EPD suites) behind the
game-agnostic seams. **This is the honest verification spine that transfers to chess — and it doubles as the robustness
engine (exploitability descent).** Full method table + papers in the workflow output (`verify-without-oracle`).

#### TOOLS to build (promote prototypes → reusable chat-reachable capabilities; directive #2)

Turn the throwaway scripts into first-class modeltrainer capabilities (`harness/` capability + `.factory/trainer.json`
activity + `src/ModelTrainerTools.ts`/backend `trainerTools.ts` chat tool), each generic and reusable:
- **`disentangle` capability** — the net×lever×deploy-search×sims grid + leverage ranking, on ANY `SolvableGame`
  (generalise `gumbel_disentangle.py`; the meta-selector's scoring engine).
- **`process-eval` capability — SHIPPED chat-reachable (2026-08-25).** `harness/process_eval.py` +
  `.factory/trainer.json processEval` + `ModelTrainerTools.runProcessEval` + backend `processEvalActivity` +
  `trainerCapabilities` `process-eval` entry + viewer "✓ Score" button; writes `{recordType}-process-eval`. Verified
  green across both repos (parity/twin + 202 backend + py). The reusable ADD-A-CAPABILITY pattern is now proven —
  mirror it for `disentangle`.
- **`train-generic` — SHIPPED into the REAL training path (2026-08-25).** The validated recipe knobs
  (`az_gumbel`/`az_c_scale`/`az_value_n_step`/`az_target_refresh`) are now `TrainerConfig` fields, threaded through
  `run.py::_run_alphazero_training → train_alphazero`, carried on the checkpoint spec, and DEPLOYED via
  `build_alphazero_agent` + `gauntlet._az_factory` (so every train/eval/play-off uses the trained-for Gumbel search,
  not plain PUCT). Enabled in the boardgames manifest `improve.hyperparams` (gumbel + n=8/k=2). So the actual
  `train`/`improve`/autopilot activities now run the recipe — no ad-hoc script. (`scale_up.py`/`gumbel_ab.py` remain
  only as measurement prototypes; delete once `disentangle` lands.) End-to-end `harness.run` verified (checkpoint
  carries `az_gumbel:true`).
- **`game feature-probe` tool** — the meta-selector stage-1 feature extractor off the `Game` interface.
- **`process registry`** — a uniform PROCESS API so router/portfolio members are pluggable (today only AZ+solver+book
  are wired with no common abstraction).
- Interface extensions as the sequencing demands them: `chance_outcomes`/`afterstate`, predecessor enumeration,
  `information_set_key`/`sample_world`/`public_state`, simultaneous-move, factored/continuous actions, game-derived net.

#### Hypothesis register (15; each has metric·corpus·threshold — the rinse-and-repeat backlog)

**Testable NOW on the shipped Connect-4/TTT harness (STEP 0):**
- **H1** value≫search>policy REPLICATES on TTT (class property, not a C4 artifact). *[high]*
- **H2** value-target is the VARIANCE-limited lever — value-target noise degrades > policy-target noise; lag/multi-target
  recover it (chance-variance proxy before a stochastic plug-in). *[high]*
- **H3** net move-ordering SPEEDS the exact solver, reduction grows with hardness (net-proposes/exact-disposes). *[high]*
- **H4** Gumbel beats PUCT only at LOW sims; crossover sim rises with m (action-width). *[high]*
- **H5** lagged-net value ≡ R-NaD lagged reference — a KL-to-lagged-policy term speeds/stabilises self-play (the
  anti-cycling bridge to imperfect info). *[high]*
- **H6** symmetry-aug gain scales with group order (TTT D4 > C4 mirror). *[med]*
- **H7** Reanalyze is SCALE-conditional (Δ turns positive as capacity/iters grow, else record the ceiling). *[med]*
- **H8** optimal n grows with horizon (argmax-n(C4) > argmax-n(TTT)). *[med]*
- **H9** 100% corpus-optimal ≠ solved — certificate self-consistency surfaces off-corpus errors. *[med]*
- **H10** distillation ~null once the value recipe is in place (keeps the C4-specific crutch out of the default). *[low]*

**Plugin-gated (drive the sequencing):**
- **H11** afterstate value is the DOMINANT stochastic lever when a clean afterstate exists (Pig/2048). *[med]*
- **H12** win-rate MIS-RANKS under hidden info; exploitability/NashConv required (Kuhn→Leduc). *[med]*
- **H13** CFR+ beats neural-BR below an infoset-count crossover K (routing feature = infoset-count). *[med]*
- **H14** proof-cost head beats win-prob value for guiding proof search, advantage grows with hardness (Gomoku+df-pn). *[med]*
- **H15** cheap interface features predict the winning arm → rule-based router beats the single-fixed-recipe on
  held-out regret. *[high — the meta-goal]*

**First experiments (all NOW, all cheap; run as the new `disentangle`/`process-eval` tools, not scripts):** E1 (H1)
disentangle on TTT vs exact solver; E2 (H2) value- vs policy-target noise-injection on C4; E3 (H3) net move-ordering in
the alpha-beta solver on fixed C4 endgames; E4 (H4) add an m-axis to the sim-sweep; E5 (H5) KL-to-lagged-policy in C4
self-play. **Sequencing (one axis per plug-in, keep an exact oracle as long as feasible):** STEP 0 exhaust H1–H10 on
C4+TTT + build the feature-probe (H15) + the Connect-4-to-SOLVED driving case; STEP 1 stochastic (Pig→2048: afterstate/
chance, H11); STEP 2 imperfect (Kuhn→Leduc: NashConv H12 / CFR-route H13); STEP 3 high-branching (Gomoku/Qubic + df-pn:
proof-cost H14); STEP 4 real-time/simultaneous (model-free + league — route AWAY from AZ). H15 graduates the router
from rule-based to learned once each arm has ≥3 labelled game-instances.

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
  re-derive every number**." **SHIPPED.** The generic render capability is in modeltrainer — pure, domain-oblivious
  `batteryReportUtils.ts` (`buildBattery(trail, {familyOf, familyOrder})` → structured `ExperimentBattery`;
  `renderBatteryHtml(battery, {title, sections, …})` → a **single self-contained, shareable static HTML page**:
  inline CSS, no scripts, no external assets, theme-aware, XSS-escaped; types in `modelTrainerTypes.ts`, direct
  tests). BlackSwan's `experiments/export.mjs` supplies the family labels + narrative and writes
  `experiments/battery.json` (diffable machine evidence) + `experiments/BATTERY.html` (the paper's spine):
  **71 pre-registered probes, 21 families, 2,004 backtested cells — 69 disproved, 2 inconclusive** (after the
  extended-history disproof pass, the cross-asset COT extension, the full published-anomaly family group below, the
  Baltussen 2021 "Global Factor Premiums" foil, the pre-FOMC drift (Lucca-Moench, SPY+BTC) and overnight-vs-intraday
  (Lou-Polk-Skouras) probes, and the energy carry/basis test — the two inconclusives are cross-sectional reversal +
  pairs mean-reversion, the same weak relative-value effect and the honestly-recorded open threads),
  with the honest-scope claim, the reproduce-and-refute (Mou 2011) entry, the recurring failure-signature section,
  and the power caveat. **This IS the publishable no-edge battery.** (Next enrichment: richer per-probe verdict
  extraction — currently gate + status + cells + title.)
- **PUBLISHED-ANOMALY BATTERY (the academic canon the paper must confront) — a new family group.** The §B trail
  covered crypto + commodities + positioning; the paper is only credible if it also confronts the classic
  academic anomalies the literature claims DO survive. Built as DSR-gated probes on the free, **survivorship-free**
  daily panel (7 commodities + SPY/TLT/IEF/UUP, 2006-→), 17 deep-history OOS windows 2008-2024, realistic 5bps/side
  cost, mutation-proven leakage guards, adversarial verification. Each new module reuses the cross-sectional
  plumbing (`align_prices`/`backtest`/`tradeable_mask`) + `summary.py` metrics verbatim.
  - **Time-series momentum / trend-following (the flagship — Moskowitz-Ooi-Pedersen 2012; the CTA industry).**
    `trainer/tsmom.py` — long/short by sign of trailing 3-12m return, inverse-vol sized, monthly rebalance.
    **DISPROVED**, adversarially verified (4-agent panel: all 3 skeptics refuted=false, synthesis DISPROVED).
    Textbook **McLean-Pontiff post-publication DECAY**: pre-2012 annualized Sharpe **+0.51** (the edge was real
    in-sample; canonical 2008/2010 trend years faithfully reproduced — a positive control), post-2012 **−0.18**;
    full 17-window t_win −0.10, annualized −0.02. Gross ≈ net (a gross null, not a cost kill); a live one-bar
    leakage mutation barely moved the Sharpe. Scoped to this construction; 17 windows exclude any annualized
    Sharpe > ~0.36, residual sub-0.2 admitted.
  - **Cross-sectional momentum / reversal (Jegadeesh-Titman 1993).** `trainer/xsection.py` on the survivorship-free
    basket (supersedes the B1-era survivorship-biased-megacap null). **Momentum DISPROVED — and it INVERTS**: the
    published long-winners/short-losers book is significantly NEGATIVE (t_win −2.26, −12%/yr) on the commodity-heavy
    free universe. Its mirror, **cross-sectional REVERSAL, is recorded INCONCLUSIVE — the battery's ONE open
    thread** (its only positive finding): cost-surviving (+0.43 annualized, survives 40bps) BUT time-unstable
    (concentrated 2016-2024, a coinflip 2008-2015), lookback-fragile (strong 63/252, weak 126), and the
    guaranteed-positive side of a pre-registered mirror pair so it sits at the best-of-6 multiplicity max (t~2 <
    DSR-critical ~2.5). Flagged for the "try harder" follow-up (out-of-basket replication + commodities-vs-financials
    decomposition + leave-one-out + regime test — the same power-and-replication treatment that upgraded the COT
    inconclusives).
  - **Low-volatility / Betting-Against-Beta (Frazzini-Pedersen 2014).** `trainer/lowvol.py` — long low-beta /
    short high-beta AND trailing-vol ranked (both persisted). **DISPROVED**, adversarially verified (skeptic +
    jackknife): a coinflip (t≈0, negative net) across all formation windows; the one edge-like cell is entirely
    the 2008 GFC long-Treasuries/short-oil trade. Caveat recorded: equal-weight book (realized β≈−2.1) is not a
    faithful beta-neutralized FP-BAB, and equity cross-sections are data-gated/untested.
  - **Calendar / seasonal (turn-of-month — Lakonishok-Smidt; sell-in-May — Bouman-Jacobsen; Monday — French).**
    `trainer/seasonal.py` — exposure-balanced (market-drift-neutral) calendar spread on SPY. **DISPROVED**,
    adversarially verified: turn-of-month null-to-negative, the Monday effect fully DECAYED (gross t≈0, net
    cost-bled); sell-in-May a weak **disproved-marginal** tilt (pooled daily t 0.85, carried by ~2 of 17 years).
  - **Pairs / statistical arbitrage (Gatev-Goetzmann-Rouwenhorst 2006).** `trainer/pairs.py` — distance pairs,
    fade divergence / close on convergence, formation-only selection + one-bar lag (mutation-proven). **INCONCLUSIVE
    — the SECOND open thread**: a weak, PERSISTENT (both sub-periods), cost-surviving (to 40bps) mean-reversion
    tilt (≈+0.35 annualized, 12/17 windows) that does not clear multiplicity (per-config t~1.5). Its inverse
    (chase divergence) is DISPROVED-negative.
  - **The multi-asset FOIL — Baltussen (2021) "Global Factor Premiums" — the paper's spine.** `trainer/globalfactors.py`
    (reuses trend/momentum/low-beta build_weights; adds VALUE = 5yr cross-sectional reversal + the diversified equal-risk
    combination; `cf-*` deep-train windows so value's 5yr formation always exists). **DISPROVED but SCOPE-LIMITED**
    (3-skeptic adversarial panel, all refuted=false, synthesis SCOPE-LIMITED). Baltussen's diversified premium FAILS
    TO REPLICATE on the free tradeable slice (t=−1.20, 4/13 windows; no construction — unit-gross/equal-vol/drop-momentum
    — rescues it). **Honesty corrections the panel forced (mandatory for the paper):** the collapse is PANEL COMPOSITION,
    not cost (momentum inverts even at zero fee; the financials momentum cross-section is a *degenerate empty book* —
    4 ETFs cannot form a k=3 book); it is a 4-of-6-factor, commodity-heavy, equal-weight PROXY, not his vol-scaled
    cross-asset engine; and value (the sole positive, +1.43 = commodity 5yr reversal = a third expression of the
    residual) is sub-significant, so the LTA reading is directional. **We do NOT claim to refute his full multi-asset
    result** (needs equity single-stock/FX/term-structure data). Lead with the LTA horse-race; Baltussen is a scoped
    supporting piece. **The panel catching this overclaim is itself a demonstration of the adversarial-verification
    methods contribution.** Source papers stored in the Library (66 `blackswan-run-paper` entities; see
    `experiments/SOURCES_TO_REPLICATE.md` + `CONCLUSIONS.md`).
  - **The two inconclusive threads are ONE effect.** Cross-sectional reversal and pairs mean-reversion are both
    weak, cost-surviving, sub-significant RELATIVE-VALUE / mean-reversion tilts on the commodity-heavy panel — the
    free data's one residual signal cluster. This is the paper's honest "what we couldn't kill" section, and the
    motivated **"try harder" follow-up**: out-of-basket replication (does mean-reversion persist on a larger/
    different universe?), commodities-vs-financials decomposition + leave-one-out (is it one asset/pair?), and a
    joint test of whether the two are the same underlying factor — the same power-and-replication treatment that
    upgraded the COT inconclusives to disproved.
  - **Headline:** the battery is no longer "0 inconclusive" — TWO honestly-recorded open threads (both mean-reversion).
    That STRENGTHENS the paper: a battery that finds a residual effect, stress-tests it, and shows it is not bankable
    is far more credible than an all-null sweep. `node experiments/export.mjs` regenerates the exact counts.
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
- **Model comparison view (DEFERRED — act ONLY once Connect-4 is SOLVED).** A side-by-side surface comparing
  champions / architectures head-to-head on the ONE gauntlet scale, with the §C gate battery per model.
  Deferred deliberately: comparison is only meaningful once ≥1 model provably reaches ground truth on
  Connect-4 (the §C.4 "Connect-4 SOLVED" milestone with a *steady* verdict) — before that there is nothing
  certified to compare against, and a comparison UI would invite the §C.2.2 multiplicity error the gates exist
  to catch. Trigger: Connect-4-solved lands steady. Scope then: a READ surface over the existing
  `{recordType}-leaderboard` records (the Models view), not a new activity.
