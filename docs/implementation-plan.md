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

**GOAL MET (BOOKAGENT ONLY — see §C.41 correction) — Exploration produces a PROVEN near-optimal Connect-4 model (2026-08-25).** Trigger `Exploration → Start`;
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
robustness metric). **FALSIFIABLE TEST — DONE (3 seeds, 1.79M net 128x6+BN+head64, 96 sims, pure-MC, ~16k games/seed).** RESULT (last-3-batch
avg, mean over seeds): **off-line P1-loss 0.278** (seeds .233/.317/.283) vs tiny-net 0.43-0.54 @12k — a ~40% relative drop;
**opening_value -0.05** (seeds -0.198/+0.024/+0.029) — FLAT/negative, no climb; oracle 0.96. VERDICT: pre-registered
thresholds (loss ≤0.15 AND openval ≥+0.7) NOT met. HONEST SPLIT — the two dominant causes separated cleanly: (1) CAPACITY
is a real under-resourcing (bigger net → substantially better off-line robustness at matched compute) but NOT sufficient
alone at 16k games; (2) VALUE-COLLAPSE is NOT fixed by capacity/compute — opening_value stayed ~0 across all seeds, tripping
the pre-registered "keep hunting, don't just add compute" condition. So "we under-resourced it" was HALF-right: scale earns
robustness, but near-optimal pure self-play needs to BREAK the value-collapse via the exact-target lever (#2 endgame targets)
/ distillation — NOT just a bigger net. NEXT: rerun scaled net with az_endgame_tablebase=1 (or distillation) → does
opening_value climb + off-line loss reach ≤0.15? See [[project_modeltrainer_endgame_from_play]].

**#1 FOLLOW-UP — DISTILLATION SOLVES THE VALUE-COLLAPSE (2026-08-27).** Two arms on the scaled net (128x6+BN+head64,
96 sims, seed 0): **DISTILL** (az_distill_games=80, oracle-labelled opening→endgame corpus, wired into scaled_run.py as
distill_corpus) vs **ENDGAME** (az_endgame_tablebase=1). RESULT: **DISTILL opening_value 0→+0.938→+1.000→+1.000 in ONE-TWO
batches (400-800 games)** — the C4 opening is now correctly read as a P1 WIN, which 16k games of pure self-play never
learned. ENDGAME opening_value stayed FLAT (+0.16→+0.09→-0.00) — the endgame frontier can't climb 30+ plies to the opening,
so it's the wrong depth for the OPENING belief (right for endgame accuracy). CONFIRMS the dual-cause thesis: capacity→
robustness, exact-OPENING-targets→value-collapse. CAVEAT: off-line loss still high early (~0.7 @1200 games, undertrained) —
watching whether distill's robustness now converges FASTER/LOWER than pure self-play with the correct value signal.
**#3 SHARPENED:** distillation works but is a SOLVER CRUTCH (oracle labels); the generic solver-free #3 must reproduce
this WITHOUT the oracle — endgame-lever and pure-exploration both leave opening_value flat, so #3 likely needs a
CURRICULUM/LEAGUE (stronger opponent forces conversion-learning → outcomes label the opening +1). Design #3 after the
distill robustness trajectory lands.

**#1 DISTILL COMPLETE (8k games, scaled net) — value-collapse SOLVED + robustness ~4x faster.** FINAL (last-3 avg):
**opening_value +0.998** (locked +1.0 from batch 1, held all 20 batches) — the C4 opening now correctly read as a P1 WIN
that 16k pure games never learned; **off-line P1-loss 0.208** (0.75→0.21 trajectory) — reached pure's 16k-game robustness
(0.28) by ~4k games, finishing BELOW it at half the compute. Strict near-optimal ≤0.15 NOT hit in 8k games (plateaued
~0.20-0.25) — partly the random-opening metric FLOOR (some diverse openings are theoretically P1-lost regardless of play;
a cleaner check = forced-win conversion from proven-won roots). NET: near-optimal C4 via a generic process = capacity
(scaled net) + exact OPENING targets (distillation) — both levers, two causes, confirmed end-to-end. CAVEAT: distillation
LABELS openings with an ORACLE (solver crutch, absent for chess/Go) → #3 must reproduce this solver-free (curriculum/league,
in design). Run: checkpoints/scaled_runs/s1_distill.

**#3 SOLVER-FREE LEAGUE — BUILT (2026-08-28, TDD, 25 affected tests green).** Reproduce distillation's opening-value fix
WITHOUT an oracle so it transfers to chess/Go. Mechanism (design workflow wf_60de0c54-470): the learner plays weak→strong
SOLVER-FREE opponents from the TRUE opening (vs_opponent_game, no opening_plies skip) so game OUTCOMES label the empty
board +1. `harness/league.py` (NEW): build_solver_free_pool = RandomAgent + HeuristicAgent(gated on heuristic_action) +
MctsAgent[30,120] (solve_endgame=0, book=None) + AlphaZeroAgent over the RUN'S OWN arch-matched ckpt snapshots
(select_snapshot_spread weak→recent) — NO NearPerfectOracle/book/legacy-champions. train_alphazero: `league_p1_frac`
(seat-priority ~0.7 onto the won P1 seat), `opening_anchor_cap`+`league_anchor_frac` (self-generated opening anchor:
accumulate seat-0 empty-board examples with TRUE outcomes, pin via _mix_training_set so the ~200 opening examples aren't
diluted in 400k buffer), `league_frozen_self` (batch-start deepcopy = equal-strength honesty rung). Wired into scaled_run
(league_* request knobs) + config (az_league asserts solver-free) + manifest + run.py (drops the oracle append).
**DECISIVE metric = off-line P1-loss vs eval-only depth-8 oracle** (NEVER in training) — can't be gamed by beating weak
play; opening_value is DEMOTED to a sanity signal (near-tautological, the league trains the empty board). EMPIRICAL ARM
RUNNING: scaled net seed 0, 20 batches, league_frac=0.4/p1_frac=0.7/snapshots=4/frozen_self=1/anchor_frac=0.25, NO
oracle/distill/endgame. Open crux: can solver-free opponents (max = mcts@120 + frozen-self, both < depth-8 oracle) close
off-line loss to distillation's ~0.21, or plateau above? Run: checkpoints/scaled_runs/s3_league.

**#3 LEAGUE COMPLETE (8k games, NO oracle) — solver-free value-collapse fix VALIDATED (partial).** FINAL (last-4 avg):
**opening_value +0.60** (0→+0.6 trajectory; noisy but robustly positive — pure never left ~0), **off-line P1-loss 0.375**
(0.75→~0.28-0.375; dipped to 0.275 @b18). THREE-WAY on the scaled net: pure ~0/0.28@16k · DISTILL(oracle) +1.0/0.21@8k ·
LEAGUE(NO oracle) +0.60/0.375@8k. VERDICT: the solver-free league GENUINELY breaks the value-collapse from play outcomes
alone (the chess/Go-transferable mechanism) and improves robustness, but doesn't fully match oracle distillation — its
best teacher (mcts@120 + frozen-self) is sub-depth-8-oracle, the expected ceiling. Trajectory still improving/noisy →
stronger solver-free opponents (higher-sim MCTS, more snapshots) + more games would likely narrow the gap. **COMPLETE
ANSWER to "why did self-play fail":** near-optimal generic C4 = CAPACITY (scaled net) + BREAK THE VALUE-COLLAPSE — cleanly
via oracle distillation (+1.0/0.21) OR solver-free via league (+0.60/0.375, the transferable one). Both end-to-end confirmed.

**"MAKE IT BETTER" RUN (2026-08-30/31) — combined recipe, solver-free, IMPROVED.** Scaled net + STRONGER league
(mcts 60/200, 6 snapshots@128) + grow-as-you-go ENDGAME table ON (#2, filled ~199k positions) + 24 batches/9.6k games.
FINAL (last-4 avg): **opening_value +0.878** (peaked +0.955 @b18) — up from first league's +0.60, NEAR the oracle's +1.0;
**off-line P1-loss 0.356** (best 0.30) — modest gain over 0.375. So the combined recipe closed most of the OPENING-belief
gap to the oracle solver-free, but the ROBUSTNESS gap (0.36 vs oracle 0.21) only narrowed slightly = the sub-oracle
TEACHER CEILING (best solver-free teacher mcts@200+frozen-self < depth-8 oracle). Next lever to close it: even stronger
solver-free opponents (higher-sim MCTS / deeper self-play) + more games — measurable, not a wall. Config relaxed:
az_league now COEXISTS with az_endgame_tablebase (complementary: league=opening, endgame table=endgame precision); only
az_distill_games (oracle opening crutch) is mutually exclusive with league. Run: checkpoints/scaled_runs/better1.

**⭐ CLEAN METRIC REFRAMES THE GAP (2026-08-31): the solver-free LEAGUE is the BEST at real near-optimality.** The
off-line-P1-loss proxy (league 0.36 vs distill 0.21) has a big IRREDUCIBLE FLOOR — it counts losses from RANDOM diverse
openings, many of which are theoretically P1-LOST regardless of play. On the CLEAN metric (verify_forced_win_conversion:
does the net convert positions that ARE proven-won, net-only, vs the exact solver, empties=24, 16 roots): **better1-LEAGUE
15/16 = 0.938 > distill-ORACLE 14/16 = 0.875 > pure 13/16 = 0.812**. So the generic, SOLVER-FREE league process reaches
near-optimal (0.94) and EDGES OUT the oracle-based distillation — the "robustness gap" was largely a metric artifact,
not real weakness. LESSON (again): pick the metric that measures near-optimality (proven-win conversion), NOT a proxy
with a floor. Measurement: scratch/measure_champs.py.

**⭐ "PUSH THE LAST ROBUSTNESS GAP" = PLATEAU, not breakthrough (2026-08-31, COMPLETE).** Resumed better1 batch 24→40 (16
more batches) with the STRONGEST solver-free teachers yet: league mcts 100/300 sims, 8 snapshots @200 sims, league_frac
0.45, opening_plies 6, endgame table ON. FINAL b40: opening_value +0.755 (last-4 avg), off-line P1-loss 0.317 (best recent
0.267) — NO gain over the pre-push b23 (+0.90/0.30). DECISIVE clean-metric head-to-head (verify_forced_win_conversion,
net-only, n_roots=**32**, empties=24, seed=7): **pre-push b23 = 31/32 = 0.969 > final b39 = 29/32 = 0.906.** ⚠️ **SUPERSEDED (§C.9, 2026-09-03): both numbers are selection+easy-subset artifacts. Paired re-measurement on held-out seed 99 (n=128) puts ckpt_23 at 0.852; the 'regression' does not survive — at larger n the push run scored HIGHER. Do not cite 0.969 or the push regression.** b23 leads at
BOTH n=16 (15/16 vs 14/16) and n=32 — a consistent ~2-root edge that never once favors the extra training (within noise,
but the push clearly did not help). VERDICT: **ckpt_23 measures 0.852 (n=128, held-out seed 99) — NOT the 0.969 originally reported (§C.9).** The residual off-line-loss gap to the oracle (0.30 vs 0.21) is NOT teacher-strength-limited — it is
the METHOD/ARCHITECTURE CEILING for this 1.79M net; more of the same solver-free teacher does not break it. Closing it
further needs a DIFFERENT lever (bigger net, a different value target, or a genuinely stronger-than-depth-8 teacher), not
more league games. This closes the #1/#2/#3 arc: generic solver-free self-play reaches ~0.97 near-optimality on C4, and we
now know precisely where its ceiling is and why.

### §C.8 — NEXT-LEVERS ROADMAP (2026-08-31): what actually moves a MEASURED ceiling

> ⚠️ **READ §C.9 + §C.10 BEFORE ACTING ON THIS SECTION (added 2026-09-04).** §C.8 is framed throughout around
> "beating 0.969" — **that number was never real** (selection + easy-root artifact; the same checkpoint measures
> **0.852** paired at n=128 on a held-out seed). The "ceiling" this roadmap sets out to break is therefore not
> established, and neither is the plateau that motivated it. The LEVERS below remain reasonable engineering
> hypotheses, but their PRIORITISATION is invalid: §C.10 BUILD #15 moves the architectural A/Bs to LAST, behind
> measurement (n=256 paired), cost accounting, the loss statistic, and the carry-to-40-batches run — because a
> +0.02–0.05 effect cannot be detected by the instrument this section assumes. Treat §C.8 as a lever CATALOGUE,
> not as a plan of record.

Source: an 11-agent literature+adversarial-critique workflow over 5 questions (efficiency / net-arch / generic solver /
constraint-programming / leftover items). **Honest filter kept throughout: separate levers that can raise the 0.969 forced-win
ceiling from levers that only make training cheaper or more robust — we MEASURED that more league games / stronger teachers do
NOT move the ceiling, so ordering leads with the ceiling-movers.** Tags: `[verified]` = citation confirmed against source this
session; `[unverified]` = flagged by the critic as unconfirmed, treat the specific number as directional; `[reasoning]` =
inference from our harness. Priority order below is the recommended execution order.

**A. CEILING-MOVERS — the only levers that can beat 0.969 (do these to break the plateau).**

1. **Net-architecture SCREEN → pick the best arch** [Q2] — RUNNING (see ACTION). Isolates the architecture variable; names
   the arch to commit to. The residual gap is a MEASURED architecture/method ceiling, so this is the front of the queue.
2. **Global-pooling branch — BUILT 2026-09-01** (`az_global_pool` lever, config+manifest; `_ResBlock.gpool_fc`: whole-board
   mean+max → FC → per-channel bias) [Q2]. Best-evidenced ceiling lever: fits C4's **odd/even threat-parity / Allis zugzwang** —
   a whole-board count stacked local convs represent poorly `[reasoning]`; `[verified]` KataGo +176 Elo / 1.6×. Does NOT
   contradict our "receptive-field is a red herring" note (that was head sight; this is mid-tower global conditioning).
   BUILD GOTCHA (mutation-test caught it): a per-channel-constant bias added BEFORE BatchNorm is exactly cancelled by its mean
   subtraction (provably dead at batch 1) — the bias is applied AFTER bn2, before the skip-add, where nothing normalizes it away.
   *Test to run:* A/B the ckpt_23 recipe ± branch, `verify_forced_win_conversion` n=32.
3. **Classification / distributional value head — BUILT 2026-09-01** (`az_value_bins` lever, config+manifest; K-bin categorical
   head + two-hot cross-entropy via `_two_hot`; `forward` still returns the SCALAR expectation over the support so every
   consumer — MCTS, probes, eval — is untouched; `forward_train` exposes the bin logits) [Q2]. Attacks value-collapse at the
   root where MSE-optimal ≈ 0. `[verified]` Farebrother "Stop Regressing" 2024 (incl. searchless chess). *Test to run:*
   opening-value climb-rate + conversion vs ckpt_23; stacks with `az_endgame_exact_targets`.
4. **Auxiliary heads — BUILT 2026-09-02** (`az_aux_heads` lever, requires residual; `own_head` 1×1-conv ownership map +
   `reply_head` opponent-reply logits; `forward_aux`; self-play AUTO-records mover-relative aux targets as 5-tuples when the
   net carries the heads — no extra knob; `augment_examples` mirrors ownership by column and reply by index; `train_net`
   masks aux losses (0.15 own-MSE + 0.15 reply-CE) to zero for 3-tuple league/distill examples; aux+reanalyze refused —
   the state buffer drops aux fields) [Q2] — `[verified]` KataGo ~190 Elo / 1.65×; targets value-collapse, STACKS with #3.
   *Test to run:* games-to-threshold + conversion vs ckpt_23.
5. **Exploitability-descent — CHEAP FORM (refutation-replay) BUILT 2026-09-02** (`harness/refutation.py` RefutationStore:
   add/sample/resolve, retire only after N CONSECUTIVE survivals, JSON persistence; P1-seat league LOSSES store the opening
   prefix as a nogood via `vs_opponent_game(record=)`; self-play force-replays a sampled nogood with prob `refutation_frac`
   via `self_play_game(forced_opening=)` — scripted UNRECORDED prefix, no fake policy targets; `_resolve_refutation` checks
   whether P1 lost it AGAIN (parity sign-flip); scaled_run knobs `refutation_frac`/`refutation_prefix_plies` +
   `refutations.json` per-batch persistence for resume) [Q5-T1, Q4 nogood-learning]. The STRONG form (a LEARNED adversary =
   the Timbers-2020 solver-free bound `[verified]`) remains the escalation build. *Escalation rule:* if #2-#4 + the best
   swept arch still can't beat 0.969, the ceiling is genuinely representational-plus-value-target and the learned-adversary
   loop is the next move — not more search or more league games.

**B. EFFICIENCY — makes training cheaper/monotone; FUNDS section A, does NOT raise the ceiling.**

6. **Promotion GATE on champion crowning — BUILT 2026-09-01** (`gate_roots` request knob in scaled_run: per-batch
   `gate_probe` = net-only forced-win conversion on a fixed seed; `update_gate` crowns champion.pt/champion.json ONLY on a
   STRICTLY better rate — ties keep the incumbent) [Q1]. *Sharpest cheap win:* our own push regressed b23→b39; this gate would
   have KEPT b23 and it kills the lucky-seed-checkpoint risk `[reasoning]`. First consumer: the auto-queued carry-forward run.
   (Weight-EMA/SWA is a code-add companion, NOT an existing knob — do not present it as a free flag.)
7. **Cut training sims 96 → ~32** (`az_sims`, `config.py:76`; keep eval/deploy sims high — already decoupled) [Q1]. Gumbel
   completed-Q + `gumbel_m=16` ≥ all 7 columns ⇒ 96 is likely 2-3× above the efficient frontier `[reasoning]`. *Test:* an
   equal-wall-clock sweep over `az_sims ∈ {16,24,48,96}`. NOTE the "MiniZero 9-16×" figure the research cited was `[unverified]`
   / misattributed — settle it empirically, do not cite the number.
8. **Reanalyze — EXPOSED 2026-09-02** (`reanalyze_frac` request knob in scaled_run; machinery pre-existed in
   train_alphazero) [Q1] — reuses buffered positions; cheap once sims are low. Speeds convergence to the SAME ceiling.
9. **KataGo playout-cap randomization** [Q5-T1] — good fit for value-collapse (more games → more value signal). SKIP the
   forced-playout-pruning half — redundant with our Gumbel completed-Q path `[reasoning]`.

**C. SOLVER / CORRECTNESS — cheap, bounded, complementary (not ceiling-movers on their own).**

10. **Net → frontier PRIORITIZATION — BUILT 2026-09-02, honestly RESCOPED** (`endgame_net_priority` knob →
    `_frontier_order`: ply-DESC stays PRIMARY — the retrograde children-before-parents invariant — with the net's |value|
    ASC as the WITHIN-TIER tiebreak, so a tight proof budget lands on the positions the net is most UNCERTAIN about; one
    batched net call per frontier via `_batched_net_values`) [Q3]. RESCOPE RATIONALE: per-NODE net move-ordering (the
    "Neural MoveMap" idea `[unverified]`) would be a large SLOWDOWN in our µs-node pure-python prover (net call ≈ ms) —
    selection, not per-node ordering, is where a net pays here.
11. **Proof-as-exact-refutation into the buffer** [Q3] — fold proven-LOSS lines into training; doubles as a rigorous solver-free
    C4 correctness check (proof-agreement card in `process_eval`).

**D. MEASUREMENT / ENABLERS.**

12. **`disentangle` capability** (promote `gumbel_disentangle.py`) [Q5-T1] — the measurement engine that unlocks the H1-H10
    hypothesis program on any `SolvableGame`; prerequisite for most hypothesis work.
13. **Mixed-openings self-play — BUILT 2026-09-02** (`opening_plies_zero_frac` knob: that share of self-play games starts
    CANONICAL, the rest from diverse openings — sharpness AND coverage from one buffer; NO rng draw when 0.0, so existing
    runs stay byte-identical) [Q5-T1] — the cheapest concrete robustness experiment left; the experiment itself queues
    behind the pipeline.
14. **LBR cheap-screen — BUILT 2026-09-02** (`lbr_screen` in benchmark.py: the model plays BOTH seats vs
    `NearPerfectOracle(depth=k)` best-responders over a fixed diverse-opening corpus → per-depth exploit_rate profile;
    deterministic instrument) [Q5-T1] — always-on cheap exploitability refuter; first profile run on ckpt_23 in flight.

**E. EXPLICITLY DECLINED / GATED — recorded so we do not revisit prematurely.**

- **EfficientZero** (learned-model + reward-prefix tricks are moot on a perfect simulator with terminal reward) and
  **prioritized replay** (de-prioritizes the opening positions that most need signal) [Q1] `[verified/reasoning]`.
- **Attention/transformer, GNN, nested-bottleneck on C4** [Q2] — right levers, wrong board: transformers win on 19×19 Go where
  conv can't span; GNNs on relational topologies like chess piece-graphs (`[verified]` AlphaGateau NeurIPS 2024, a chess-transfer
  item); nested-bottleneck only pays ≥256ch.
- **CP / SAT / QBF as the engine** [Q4] — CATEGORY ERROR: game-depth = QBF-alternation-depth is the complexity wall; generic
  solvers stall at 4×4 tic-tac-toe where specialized minimax wins `[verified]`. The two big CP ideas (canonical-key transposition
  + left-right symmetry breaking) are ALREADY ours (`solver.py:83-105`, `tablebase.py`/`book.py`, `augment_examples`
  `neural.py:718`); the only new borrowable idea (nogood learning) is item #5.
- **Full generic df-pn + GHI engine / EWS** [Q3] `[unverified]` and the **per-game-class portfolio** (df-pn for high-branching
  threat games; afterstate-TD/expectimax for stochastic; CFR→Deep-CFR/NFSP→R-NaD for hidden-info; MuZero) [Q5-T2/3] — real
  backlog but GATED on actually moving to a harder game class. Do NOT pull forward.

**ADVERSARIAL REVIEW OF THE §C.8 BUILDS (2026-09-02, 17-agent workflow, every finding EXECUTION-verified; all fixed
same-day, regression tests added):** ⭐ POSITIVE: an empirical HEAD-vs-worktree A/B confirmed all new flags at defaults are
BYTE-IDENTICAL (buffers, weights, rng streams) — the queued runs pick up this code safely; ab_bins/ab_gpool train-path
consistency also verified. FIXED (high): (1) the refutation resolver judged replays by the first example's VALUE TARGET —
n-step-bootstrapped/tablebase-overridden — so retirement tracked the net's OPINION, not the outcome; now `self_play_game`
fills a `record` with the game's ACTUAL per-seat returns and `_resolve_refutation` reads only that (works even for
zero-example replays). (2) gate probe shared seed=7 with the final scorecard — its 12 roots were literally the first 12 of
the final 32, max-selecting the champion on 37.5% of the measurement; gate now uses held-out `GATE_PROBE_SEED=1013`.
(3) the carry-forward launcher lived in a SESSION task while the detached A/B chain waited on its output (split-brain,
dies with VS Code); replaced by ONE self-contained detached pipeline script (run_pipeline_chain.sh). FIXED (med/low):
terminal-prefix nogoods could never retire (`_nogood_prefix` always excludes the final move); store eviction was
oldest-ADDED not oldest-REFUTED (resolve(lost) now freshens); refutations.json/champion.json writes were non-atomic with
crash-prone readers (temp+rename `_write_json_atomic` + tolerant reads — the FileStorage incident class); refutation
without league silently inert (now refused); endgame_extend_positions/seconds were silently dropped by scaled_run (now
forwarded). ACCEPTED, not fixed: one-way checkpoint compat (OLD code can't read NEW 9-key-arch checkpoints — no old-code
readers exist); refutations.json lags a mid-batch abort (consistent with ckpt/buffer semantics); JSON string "0" for a
bool knob coerces truthy (configs use real booleans).

**PIPELINE IN FLIGHT (2026-09-02, all detached/resumable, one training process at a time):** screen (below) → auto-scored,
best NON-champion arch auto-carried into the full b23 recipe + gate (`carry_<arch>`) → then `ab_chain` (scratchpad
run_ab_chain.sh, detached) runs the two ceiling A/B arms SEQUENTIALLY on the champion arch — `ab_gpool` (global_pool=1,
1.98M params) and `ab_bins` (value_bins=21) — each the exact b23 recipe (league 60/200 + snap6@128 + endgame, 24 batches,
seed 0) with `gate_roots=12`, ending with an n=32 scorecard of ALL gated champions vs baseline better1/ckpt_23 (0.969),
logged to `checkpoints/scaled_runs/ab_chain.log`, completion marker `PIPELINE_DONE`. Reading: an arm strictly above 31/32
beats the ceiling; at/below ⇒ that lever alone does not (next: stack levers, or exploitability-descent per A5).

**ACTION IN FLIGHT — net-arch SCREEN** (`checkpoints/scaled_runs/archscreen/`): 8 archs (20.6K→4.74M params, verified),
recipe = **distillation + endgame exact-targets** so each net reaches ITS OWN representational ceiling — isolates the arch
variable (the oracle is a MEASUREMENT crutch here, not part of the deployed generic process). Ranking is RELATIVE across archs
(NOT vs 0.969 — this is a cheap screen); the winner is carried into the full solver-free league recipe for the real vs-0.969
test, and only crowned after a second-seed re-run (standing don't-trust-one-sweep rule). Bounded: 2,560 games/arch, sims 48,
one arch at a time, niced + OMP-capped (BLAS-thrash guard), resumable, ~12-15h; smoke-tested green. PROCESS NOTE: a workflow
subagent auto-launched a FIRST sweep as pure-#1; the critic caught that pure-#1 collapses for every arch (0.812 even at 16k
games ⇒ no discrimination) — killed and relaunched with this distill+endgame screen.

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

### §C.9 — MEASUREMENT INTEGRITY PROTOCOL (2026-09-03) — MANDATORY for every reported strength number

**WHY THIS EXISTS.** A phantom result — "0.969 forced-win conversion, best ever" — survived for days, was written
into this plan and into memory as settled, and drove real decisions (it justified the whole "push the ceiling"
arc and the arch screen). It was never real. Independent paired re-measurement on a fresh seed put the same
checkpoint at **0.852** (n=128, seed 99). FOUR compounding errors, all ours, all now guarded IN CODE:

| # | Error | What it did | Guard (code + test) |
|---|---|---|---|
| **E1** | **Selection on the test set** | the promotion gate probed the SAME roots the scorecard reported (its 12 roots were literally the first 12 of the final 32) | `harness/measurement.py` `SELECTION_SEEDS` / `MEASUREMENT_SEEDS` are disjoint frozensets; `assert_seed_roles_disjoint`; `gate_probe` REFUSES a measurement seed. `test_measurement.py::test_selection_and_measurement_seeds_are_disjoint`, `test_scaled_run.py::test_gate_rejects_a_measurement_seed` |
| **E2** | **Winner's curse** | the headline was max-over-40-checkpoints on 32 roots. At true rate 0.85, E[max] = **0.962** — the "breakthrough" was the EXPECTED value of the selection procedure | `expected_selected_max()`; `report_rate()` refuses to emit a selected number without the inflation estimate + caveat; the promotion gate now needs the gain to clear `promotion_margin()` (one SE of the incumbent probe). `test_expected_selected_max_quantifies_the_winners_curse`, `test_report_rate_refuses_a_naked_selected_number`, `test_promotion_margin_blocks_noise_but_admits_decisive_gains`, `test_gate_requires_the_gain_to_clear_measurement_noise` |
| **E3** | **Mixed-provenance comparison** | one net's SELECTED-best vs another's FINAL — manufactured a "significant" arch difference (p=0.041) that VANISHED (p=0.754) once both were compared final-vs-final on identical roots | `mcnemar_exact()` requires equal-length PAIRED vectors and raises otherwise. `test_mcnemar_exact_matches_hand_computed_cases` |
| **E4** | **Underpowered ranking** | 8 architectures ranked on 16 roots where ±1 root = 0.06 and the true spread was ~0.08; re-seeding the roots REVERSES the headline | `min_roots_for(delta, base)` — detecting the ~0.02 differences we ranked on needs >1000 roots. `test_min_roots_for_powers_the_claim` |

**THE RULES (non-negotiable, enforced by the tests above):**
1. **Seeds have ROLES.** A seed used to CHOOSE (gate, sweep, early stop, arch pick) may NEVER be used to REPORT.
2. **Every reported rate carries n, seed and a Wilson CI.** `report_rate()` is the only sanctioned emitter. A bare
   point estimate is not a result. `0.969` and `0.85` are the same measurement at n=32.
3. **A selected maximum is not an estimate.** Best-of-k must be labelled and must carry `expected_selected_max`.
   Re-measure the chosen candidate on a held-out measurement seed before quoting it.
4. **Comparisons are PAIRED on identical roots, final-vs-final** (or selected-vs-selected — never mixed), reported
   with exact McNemar.
5. **Power the claim before making it.** `min_roots_for()` before ranking anything.
6. **One training run is one sample.** Architecture-level claims need replication across TRAINING seeds; a paired
   root test only establishes that two WEIGHT VECTORS differ.

**TWO REAL BUGS THIS EXPOSED, BOTH FIXED:** (a) `train_net` constructed a fresh `torch.optim.Adam` on EVERY call,
so Adam's moments reset every training iteration (~200× in a long run) — an `opt_state` dict now carries one
optimizer for the life of the run (default path unchanged, `test_train_net_can_persist_optimizer_state_across_calls`).
(b) `lr=1e-3` / `batch=64` were HARD-CODED across a 20K→4.7M parameter span, so every "big nets are worse" reading
was confounded by an lr tuned for neither — now `az_lr` / `az_batch_size` levers (config + manifest + scaled_run).

**CORRECTED RESULTS TABLE (paired, n=128, held-out seed 99, final-vs-final):**
better1 ckpt_23 (1.79M) **0.852** · carry final ckpt_23 (302K) **0.836** · McNemar **p=0.754 — INDISTINGUISHABLE**.
carry's GATE-PROMOTED champion 0.773 — i.e. the gate promoted a net **0.063 WORSE** than the run's own final
checkpoint (p=0.077). **The 302K net matches the 1.79M net at equal games with 1/6 the parameters and ~2.5×
the training speed. There is no measured capacity crossover, and there never was a 0.969.**

### §C.10 — WHAT WE LEARNED ABOUT NETS, GAMES AND SEARCH PROCESS (2026-09-04)

Compiled from a 6-agent analysis + adversarial critique + **my own independent re-measurements**. Provenance is
marked: `[measured-here]` = re-run this session; `[agent]` = analysis-agent measurement not independently re-run;
`[verified]` / `[unverified]` = citation confirmed (or NOT confirmed — treat as directional); `[reasoning]`.
**Read §C.9 first — it is why most of the older numbers in this plan are wrong.**

#### Q1 — Why the models perform as they do (and why "small beat big" was mostly an artifact)

**The headline collapsed under re-measurement.** Paired, FINAL-vs-FINAL, identical roots, held-out seed 99:
1.79M = **0.852**, 302K = **0.836**, McNemar **p=0.754 — indistinguishable** `[measured-here, n=128]`. The
"crossover" (small wins cheap, loses at scale) is **not supported by any measurement we own**. Beyond §C.9's
E1–E4, one more error was mine: I compared the big net's *selected* checkpoint against the small net's
*gate-damaged* one, manufacturing p=0.041; like-for-like gives p=0.754.

**What survives:**
1. **A representational FLOOR, not a peak.** The only significant architectural result we own: the 20K legacy net
   (`blocks=0, head_hidden=0` — two convs then a **bare Linear**) is worse than everything ≥302K (p=0.0025
   `[agent]`). Forks are AND-of-threats, odd/even parity is XOR-like; an affine readout represents neither.
   **Above ~3×10⁵ params every pairwise comparison is null** — extra capacity is neither help nor harm at these
   budgets, only cost.
2. **Cost is the real axis.** 302K trains ~2.5× faster per batch, ~2.1× cheaper per forward `[agent]`. Equal skill
   at 1/6 the params is an EFFICIENCY win, not a strength win. ⚠️ The critic found the agent's wall-clock model
   self-contradictory (53× vs 15.8× for the same span; cost non-monotone in params because per-batch EVAL is
   bundled into the timing) — **re-measure cost cleanly before budgeting on it** (BUILD #4).
3. **TARGET QUALITY was the binding constraint, not capacity.** Pure self-play at 1.79M for 16k games left
   `opening_value ≈ 0`: equal-strength self-play from a won-but-unconverted position scores ~50/50, so the
   MSE-optimal value target is literally 0. Distillation fixed it in 1–2 batches. **The data was wrong, not
   scarce.** (The stronger "recipe×capacity INTERACTION" claim is a difference-of-significance fallacy, p≈0.08,
   **n.s.** — do not cite it as established.)
4. **Two confounds invalidated every capacity reading anyway**, both fixed in §C.9: Adam's moments reset every
   iteration; one hard-coded lr spanned a 230× parameter range.
5. **`final_loss` explained nothing** — it was never a train loss, just one mini-batch (verified in code).
   Loss-vs-conversion r ≈ −0.09; ~80% of its variance tracks policy-target ENTROPY `[agent]`. Any KataGo-style
   "switch when the loss catches up" rule is unusable until BUILD #3.

**⚠️ THE LIMIT ON ALL OF IT: one training run per architecture.** A paired root test shows two WEIGHT VECTORS
differ — never that two ARCHITECTURES do. Training-seed variance has never been measured here. Every architectural
statement above, **including the corrected null result**, is provisional until replicated.

#### Q2 — Predicting a good net for a NEW game

**Tier 1 — what our data licenses (only these three):**
- **R1 FLOOR.** Trunk receptive field must cover one win-line; the head must be NONLINEAR. Below that nothing
  helps; above it this rule says nothing.
- **R2 KNEE, not peak.** Sweep params at a fixed cheap budget, then take the **smallest arch whose CI overlaps the
  top** — never the point-estimate winner.
- **R3 RECIPE BEFORE ARCHITECTURE.** The recipe moves skill more than any architectural contrast we measured.

**Tier 2 — property → knob (literature checklist; a HYPOTHESIS generator, not our measurement):** whole-board
aggregate (parity/komi/score/connection) → **global-pool branch** `[unverified 1.60×]`; long-range relations
(ladders/connections) → **attention blocks** `[verified: ResTNet, Hex 19×19 50.4→58.0%]`; score-margin value or
common draws → **categorical value + draw bin** `[verified: Stop Regressing 2403.03950]`; per-cell final label →
**aux heads** `[unverified 1.65×]`; non-lattice topology → **drop the conv trunk** (MLP/set encoder) `[verified:
TD-Gammon]`; large action space → plane-stacked policy + raise `gumbel_m`; hidden info → **change the ALGORITHM,
not the net** (PBS/CFR/ReBeL; subgame decomposition is theorem-forbidden) `[verified: 2007.13544]`; large symmetry
group + scarce data → equivariant layers, else augmentation.

**⚠️ TRANSFER WARNING:** "48–96 sims suffices" rests on `gumbel_m=16 ≥ A=7`. For Go (A=362), Hex 11×11 (121),
chess (4,672), m ≪ A. **Do not carry this sims budget to any larger-action game.**

**Worked starting points (hypotheses, not decisions):** tic-tac-toe → *tabulate, no net*; **Connect-4 → c64×b4
hh32 (302K)** as the deployed net — reversing the old 1.79M recommendation on COST grounds, not strength; Go 9×9 →
c96×b8 + global-pool + ownership aux; Go 19×19 → c256×b20 `[verified: AlphaGo Zero 20×256]`; Hex 11×11 → c128×b12
+ transformer blocks (**no oracle exists — needs Elo-vs-MoHex, not a conversion score**); backgammon → **not a
ResNet**: MLP + expectimax `[verified: TD-Gammon]`; poker/hidden-info → algorithm change first.

**DELETED — do not carry forward:** the `P* ≈ 115·G` params-per-game formula (both fit points noise-selected; its
"confirmation" was a 12-root gate artifact; α≈115 for us vs ≈0.5 for AlphaZero chess — 230× apart, so not a
constant of anything). The quantitative depth floor `b_RF = ceil((2D−1)/2)` — our measured depth ordering is flat
and non-monotone (b2 .691, b4 .738, b6 .723/.730, b8 .707, b10 .715 `[agent]`); keep only as a prior for unrun
games. "Big nets forget openings" — non-monotone in size, and Connect-4 is explicitly exempted from inverse
scaling `[verified: Neumann & Gros]`.

#### Q3 — The streamlined process: **Bracket-and-Fit**

An 8-arch full-budget grid is ≈3 CPU-months at measured throughput, so the procedure must EXTRAPOLATE, not rank.

- **Step 0 — fix the MEASUREMENT before spending a training hour.** n=256 paired roots, common random numbers,
  McNemar + Wilson. Ship a **NULL ARM** in every sweep (a ≤1%-param duplicate of the centre arch — 05/06 differ by
  8,704 params, the natural pair): its spread IS the sweep's noise floor. **Refuse to emit a ranking** when
  homogeneity p > 0.20 or the top-2 gap < the null-arm spread. Our screen would have auto-flagged: **χ²=4.96,
  df=7, p=0.665** `[agent]`.
- **Step 1 — 4 arms**, three bracketing the knee at ±½ decade plus the null arm. Not 8.
- **Step 2 — rungs budgeted in WALL-CLOCK, not games.** Equal-games gave a large wall-clock spread; the small nets
  were NEVER run at the budget they can afford — the 20K net's true ceiling is simply unmeasured.
- **Step 3 — THE STRUCTURAL FIX: never eliminate on RANK at a rung.** Kill an arm only when its fitted curve's
  upper 90% bound AT THE TARGET BUDGET is below the incumbent's lower 90% bound. That makes "we did not discard the
  eventual winner" a property, not a hope.

#### Q4 — The most EFFICIENT vs the BEST net: three objectives, not one

| | Definition | Answer shape | Our current answer |
|---|---|---|---|
| **Q_eff(B)** | best skill at training budget B | a FRONTIER, one winner per budget | 302K at ~9.6h `[agent, sub-noise]` |
| **Q_best** | best fitted asymptote (budget → patience) | one net + tolerance + stated recipe | **UNDETERMINED — neither net has been run to plateau** |
| **Q_deploy** | skill per unit INFERENCE cost | a second frontier | 302K by ~2.1× (ms/forward at batch=1 — MCTS forwards are NOT batched, so batch-1 latency is the currency) |

**The single-budget trap, dissected** (all three bit us): the point is noise (n=16 ⇒ CI half-width ±0.22); the
budget is not equal in the currency you pay; and rank at B does not determine rank at B′ `[verified: Hoffmann]`.
A fourth trap sat on top: best-of-N reporting (§C.9 E2).

**Protocol:** curves not points (one experiment answers Q_eff AND Q_best); eliminate only on CI-at-target; null arm
in every sweep; DEV/SEL/TEST seed partition so the reported number was never optimised against; report the frontier
WITH undetermined regions marked; report Q_deploy separately.

**Honest limit:** no procedure guarantees a global optimum over an unbounded architecture space. This one
guarantees: (a) no arm discarded except on evidence of inferiority AT THE TARGET BUDGET; (b) no reported difference
smaller than the sweep's own measured noise floor; (c) efficiency and asymptote separated, each with a CI; (d) the
reported number was never optimised against.

#### BUILD BACKLOG (information per engineering-hour; **1–4 BLOCKING** — nothing above them is interpretable)

1. **n=256 paired conversion as the harness default** + widen the root sampler beyond the easy first-32 (roots
   0–31 measured ~+0.077 optimistic for EVERY net `[agent]`). *Partly done ad hoc — promote into `benchmark.py`.*
2. **Null arm + refuse-to-rank guard** (homogeneity χ², null-arm spread). Turns "the screen found nothing" from a
   post-mortem into an automatic gate. *(`harness/measurement.py` has the primitives; the guard is NOT wired.)*
3. **Fix the loss statistic**: return an EPOCH MEAN from `train_net`; log anchor loss separately; add a held-out
   solver-labelled probe set never trained on.
4. **Cost accounting**: `wall_s, cpu_s, games, forwards, cum_wall_s` per batch (`scaled_run` records **no timing
   at all**) + an `ArchCostProbe`. Q_eff is currently unanswerable from existing logs.
5. ✅ **DONE (§C.9)**: `az_lr`/`az_batch_size` levers + persistent optimizer state. The lr sweep on the widest arch
   is now possible — it decides "mistuned vs capacity", i.e. whether the screen must be redone per-arch.
6. **Run `carry_302K` to 40 batches (16k games)** — **the single most informative experiment available**, ~6.4h
   compute, zero engineering. ≥0.879 ⇒ no capacity ceiling below 16k games; flat ~0.85 ⇒ our first defensible
   ceiling claim.
7. **Graded conversion + `games_per_root>1`** with randomised defender tie-breaks — more bits per root at no extra
   sampling cost (the binary metric needs ~870 roots to resolve 0.848 vs 0.879).
8. ✅ **PARTLY DONE (§C.9)**: gate now needs the gain to clear one SE. Still to add: promote-then-CONFIRM on a
   larger root set, and explicit DEV/SEL/TEST seed families.
9. **LBR at ≥200 openings/cell, paired, floor-subtracted** — n=20/cell is non-monotone, therefore uninterpretable;
   our champion's "shallow-tactical weakness" reading rests on it and is **not yet safe**.
10. **Playout Cap Randomization + Forced Playouts** `[unverified 1.37×/1.25×]` — a ~1.7× throughput gain would be
    worth more than any architecture difference this program has measured.
11. **`harness/curvefit.py`** — fit/predict/crossover/pareto + on-disk pre-registration of the next rung's prediction.
12. **`harness/bracket.py`** — the Bracket-and-Fit driver, resumable, emits `frontier.json`.
13. **Per-depth (empties-bucketed) loss breakdown.**
14. **Reset / shrink-and-perturb at plateau** — ranked LOW: sampling noise now explains the plateau without it.
15. **`az_global_pool` / `az_aux_heads` / categorical-value A/Bs — DELIBERATELY LAST.** A +0.02–0.05 effect needs
    ~870–1,500 paired roots or the graded metric FIRST. ⚠️ The arms in flight (ab_gpool, ab_bins) were launched
    before this was understood — single-run, n=32-scorecard experiments on a +0.02 question. **They will most
    likely produce a null dressed as a finding; score them under §C.9 and expect "indistinguishable".**

**DO FIRST:** #1 → #3 → #4 → **#6** → the lr sweep. Under a day of compute; either restores or finishes off the
capacity thesis.

#### PROGRESS 2026-09-04 — blocking items shipped, compute REPRIORITISED

**Built (TDD, all green):**
- ✅ **#2 refuse-to-rank guard** — `rank_or_refuse()` (chi-square homogeneity + null-arm noise floor + required-n).
  Live-fire regression test replays OUR OWN 8-arch/16-root screen and **REFUSES to rank it** (χ²=4.96, df=7,
  p=0.665, "~1,400 roots per arm needed"). The guard also refuses a *statistically significant* gap that is
  smaller than the sweep's null-arm spread — significant ≠ real.
- ✅ **#3 honest loss** — `train_net` returns an **epoch mean**, not the last mini-batch. Test pins stability
  (spread < 0.15 across identical runs) where the old single-batch statistic had SD ≈ 0.23.
- ✅ **#4 cost accounting** — every batch record now carries `wall_s, cpu_s, games, cum_wall_s, eval_wall_s`.
  **Training is timed SEPARATELY from eval** — bundling them is what made the old cost model non-monotone in
  params and produced the contradictory 53×-vs-15.8× ratios. Q_eff is answerable from logs for the first time.
- ✅ (§C.9) `az_lr`/`az_batch_size`, persistent optimizer, gate margin, measurement primitives.

**Compute reprioritised** (the arch A/Bs were ~40h for a *predicted* null; both paused, fully resumable at
`ab_gpool` ckpt_4):
1. **`carry_302K` 24 → 40 batches** — the capacity question at a budget MATCHED to better1's 16k games (~6.4h).
2. **`seedrep_302K_s101`** — same arch, same recipe, same budget, **different training seed**. This measures the
   TRAINING-SEED NOISE FLOOR, which is the gap underneath every architectural claim in this document. Until it
   exists, "arch A ≈ arch B" is a statement about two weight vectors.

**Still open from the backlog:** #1 (n=256 paired as harness default), #7 (graded conversion), #9 (LBR at
n≥200), #11/#12 (curvefit + bracket drivers), #10 (playout-cap randomisation — likely worth more throughput than
any architecture difference measured so far).

### §C.11 — ANALYSIS LEDGER (2026-09-04): guards belong ON the path, not beside it

**The lesson that produced this section.** §C.9 shipped correct measurement primitives. Within HOURS I made three
NEW errors anyway — not statistics failures, BOOKKEEPING failures — because the primitives were available but
nothing forced measurements through them:

| | The error I actually made | Ledger guard |
|---|---|---|
| **L1** | put a GATE-SELECTED checkpoint in a grid beside FINAL ones and compared them | `compare()` raises on mixed provenance (`provenance` is a mandatory field on every record) |
| **L2** | labelled a comparison "matched budget" when the arms had 9.6k vs 16k games | every record carries `games`; `compare()` returns `budget_matched` + an explicit note |
| **L3** | ran ~6 paired tests on overlapping roots, then read p=0.039 as significant | the ledger counts comparisons per ROOT FAMILY and returns `alpha_corrected` + `significant` |

`harness/ledger.py` + `tests/test_ledger.py` (6 tests, one per error class). **Replayed against the real
2026-09-04 grid it blocks L1 outright, flags L2, and downgrades my own headline: the 1.79M-vs-302K@16k result
(raw p=0.0386) is NOT significant once multiplicity is counted.**

**THE GENERALISABLE RULE: a guard that is optional will eventually be bypassed by whoever is in a hurry —
including the author.** Put the guard on the path (a ledger you must record through) rather than beside it (a
helper you may call). This is why §C.10 BUILD #11/#12 (curvefit + bracket drivers) matter more than any individual
lever: they make single-budget ranking structurally impossible rather than merely discouraged.

**Standing practice (user, 2026-09-04):** the PROCESS is the deliverable. Every step must harden the system —
each mistake becomes a code guard with a regression test naming the incident; prefer the experiment that makes
FUTURE experiments interpretable over the one that answers today's question.

### §C.12 — THE SEED-NOISE FLOOR, MEASURED (2026-09-04) — and the design it prescribes

**The number every architecture claim in this document was implicitly assuming, finally measured.** Two runs of the
IDENTICAL config (302K arch, same recipe, same 9,600-game budget), differing ONLY by training seed, scored:

| run | conversion (paired, n=128, held-out seed 99, final ckpt) |
|---|---|
| 302K seed 0   | 107/128 = **0.836** |
| 302K seed 101 | 111/128 = **0.867** |

**Seed-pair gap = 0.031** (McNemar p=0.424, budget-matched, both `provenance=final`, drawn through the ledger).
Estimated per-run seed SD ≈ **0.022** (`seed_sd_from_pair`; ONE pair is a weak variance estimate — treat as an
order of magnitude).

**What this settles about the capacity question:** the 1.79M-vs-302K gap at 16k games was **0.062** — roughly 2×
the seed gap, so NOT obviously an artifact — but at n=128 the metric's own SE is ~0.032, so the effect is the same
size as the combined noise. **The capacity signal is neither confirmed nor refuted.** That is precisely why it
flip-flopped across three consecutive reports: we were reading noise each time, in whichever direction the latest
comparison happened to fall.

**The prescribed design** (`required_seeds()`, new in `harness/measurement.py`, tested):

| n_roots | training seeds per arch to resolve Δ=0.062 |
|---|---|
| 128 | 7 |
| 256 | 4 |
| **512** | **3** |
| 1024 | 3 |

**⭐ THE PROCESS LESSON: ROOTS ARE CHEAP, RUNS ARE EXPENSIVE.** A root costs seconds; a training run costs ~10
hours. Quadrupling roots (128→512) drops the requirement from 7 seeds to 3 — **trading ~1 hour of measurement for
~40 hours of saved training.** Any future architecture experiment should max out the measurement budget FIRST and
only then buy training seeds. This inverts how we have been spending: every sweep so far bought more runs and
skimped on roots, which is the most expensive possible way to be uncertain.

**Consequence for the backlog:** §C.10 BUILD #1 (n=256+ paired as the harness default) and #7 (graded conversion,
more bits per root) are now clearly ahead of ANY further architecture work — they are what make architecture work
affordable. The arch A/Bs still paused at `ab_gpool` ckpt_4 are targeting a predicted +0.02–0.05 effect, which at
n=128 would need **>20 training seeds per arm**; at n=512 with a graded metric it becomes plausible. Do not resume
them until the measurement side is built.

### §C.13 — GRADED CONVERSION: BUILT, MEASURED, AND PARTLY REFUTED (2026-09-05)

**Built:** `RandomizedOracleAgent` (perfect defence that samples UNIFORMLY among equally-optimal replies),
`graded_conversion` (per-root FRACTION of perfect defences beaten), `paired_conversion` (all arms on identical
roots — pairing is now structural), shuffled root sampling (prefixes representative), and a runtime
opening-wall guard in the solver (`MAX_SOLVE_EMPTIES`, set by `tests/conftest.py`).

**⛔ THE HEADLINE IS A REFUTATION OF MY OWN BUILD #7.** Graded scoring reduces per-root variance by a real but
MODEST amount — effective-n gain **×1.18–1.41 (mean ×1.32)** — while costing **4× the games** (`games_per_root=4`).
Normalised for compute it is a **LOSS**:

| at an equal 384-game budget | SE of the mean |
|---|---|
| graded, 96 roots × 4 games | 0.029–0.038 |
| **binary, 384 roots × 1 game** | **0.018–0.021** |

**Binary-with-4×-roots is 1.75× more precise than graded-with-4-games at identical cost.** The mechanism:
BETWEEN-root difficulty variance dominates (positions genuinely differ in hardness); extra games per root only
shrink the small WITHIN-root component. **RULE: buy ROOTS, not games per root.** Keep the randomized defence
(it is a validity fix — the metric now means "beats perfect play" rather than "beats one canonical line") but run
it at `games_per_root=1`, where each root samples a different optimal defence for free.

**Measured on existing checkpoints (n=96 roots, graded, randomized defence, paired, held-out seed 99):**

| comparison | delta | p (paired permutation) |
|---|---|---|
| ARCH gap, 1.79M vs 302K @16k | **+0.060** | 0.021 |
| SEED gap, identical config s0 vs s101 | −0.036 | 0.329 |
| BUDGET gap, 9.6k → 16k games (same seed) | +0.018 | 0.656 |

Three comparisons on this root family ⇒ corrected α = 0.0167, so **the arch gap at p=0.021 STILL does not clear
correction**. It is now consistent across two independent instruments (binary/deterministic: +0.062; graded/
randomized: +0.060) — suggestive, reproducible, and still not established. The randomized defence did NOT reorder
the nets, so the validity concern, while real in principle, did not change any conclusion here.

**⭐ THE STRATEGIC FINDING: YOU CANNOT MEASURE YOUR WAY OUT OF TRAINING-SEED VARIANCE.** The seed floor (0.036)
is ~60% of the arch gap (0.060). Measurement improvements have hit diminishing returns — better instruments make
the seed floor *more precisely visible*, not smaller. The remaining paths are only:
1. **Buy seeds** — 4–5 training runs per arm at n≥256 roots (~40–50h) settles it definitively.
2. **Reduce training variance itself** — the untried lever, and the interesting one: EMA/SWA, longer runs, a more
   stable recipe. If training variance halved, the SAME arch question becomes answerable with ~2 seeds instead of 5.
3. **Accept "indistinguishable at ~0.06 resolution"** and spend the compute on the recipe, which has produced every
   large measured effect in this project.

**Backlog correction:** BUILD #7 (graded conversion) is DONE but should NOT be used at `games_per_root>1`.
BUILD #1 (more paired roots) is upgraded — it is now the ONLY measurement lever with a favourable exchange rate.

### §C.15 — THE CAPACITY QUESTION CLOSED, AND GLOBAL-POOL IS A NULL (2026-09-07)

> **⛔ CORRECTED 2026-09-08 — READ §C.17 AND §C.18 FIRST. Both experiments below reused controls trained by DIFFERENT
> TRAINING CODE than their arms, so each varied more than the one flag under test. The paragraph immediately
> below, claiming this was "the first experiment designed correctly from the start", is exactly wrong: the design
> was careful about everything the ledger checked and blind to the thing it did not. §C.17 has the audit, what
> survives, and what is being re-run.**

**Two questions settled with properly-designed experiments — the first in this project designed correctly from
the start (pre-registered read, 2 seeds, pre-existing controls, paired at n=384, ledger-enforced provenance).**

#### 1. Architecture (capacity): NO DIFFERENCE — closed

| root family | n | 1.79M vs 302K gap |
|---|---|---|
| s99 | 96-128 | +0.060 / +0.062 (what I reported, repeatedly) |
| s99 | **384** | **+0.023, p=0.23** |
| s257 | **384** | **-0.005, p=0.87** |

The effect DECAYED toward zero as roots increased ON THE SAME FAMILY, and flips sign across families; the two
n=384 estimates average **+0.009**. **There is no measurable architecture difference. The 302K net matches the
1.79M net at 1/6 the parameters and ~2.5x the training speed — use the small one.** (Note the two families differ
in absolute difficulty by ~4 points, 0.87 vs 0.90 — cross-family ABSOLUTE rates are meaningless; only paired
within-family comparisons are valid.)

#### 2. Global-pool branch: NO DETECTABLE EFFECT

302K arch, identical recipe/budget/seed, one flag changed; controls already existed at zero cost.

| seed | control | +global_pool | effect |
|---|---|---|---|
| 0 | 0.8880 | 0.8854 | -0.003 (p=1.00) |
| 101 | 0.8828 | 0.8750 | -0.008 (p=0.76) |
| **mean** | | | **-0.005** |

**Why this is informative rather than merely disappointing:** global-pool was the BEST-motivated architectural
lever we had — C4's win condition is odd/even threat parity, a whole-board COUNT that stacked 3x3 convs represent
poorly, and KataGo measures real gains from exactly this branch on Go. The honest reading of a null here is a
BOUNDARY CONDITION: **6x7 is small enough that the plain conv tower's receptive field already spans the board, so
there is no non-local information left for pooling to add.** Expect the lever to pay on 19x19, not on 6x7. That is
a transferable prediction, not a dead end.

#### The consistent picture across everything measured

**Architecture is NOT the binding constraint on this game** — not width, not depth, not parameter count, and now
not whole-board aggregation. Every large measured effect came from TARGET QUALITY (distillation breaking
value-collapse; exact endgame targets) or from MEASUREMENT DISCIPLINE. Both nets sit at ~0.88-0.90 conversion,
i.e. ~1 in 9 proven-won positions is still thrown away, and no architectural change we have tried moves it.

**Remaining candidates for that last ~10%:** the value-target FORM (categorical head — built, still untested), the
exploitability thread (LBR measured a depth-2 refuter beating the champion from 15% of diverse openings as P1),
and the TRANSFER TEST — porting the process unchanged to a game with no solver, which is the actual north star and
the one question Connect-4 cannot answer.

**Process note:** this experiment cost ~20h and its controls were FREE, because the 302K net had already been run
at matched budget on two seeds. Designing the A/B on the CHEAP arch (justified by finding #1) made it 2.5x cheaper
with no loss of validity. The abandoned 1.79M `ab_gpool` run would have cost ~36h and still needed its own control.

### §C.16 — THE MEASUREMENT STEP IS NOW A SCRIPT (2026-09-08)

**The gap:** §C.9 gave us correct primitives and §C.11 gave us a ledger, but the step that USES them — score the
arms, record them, draw the comparison — was still a hand-written heredoc, rewritten from scratch for the capacity
A/B and again for the global-pool A/B. Nothing carried root count, `games_per_root`, seed role, simulation budget
or provenance from one to the next. A protocol re-decided by hand each time is the same failure mode as a guard
that is available but optional: it holds until someone is in a hurry.

`scripts/measure_ab.py` is that step. It refuses a non-MEASUREMENT seed, folds `sims` into `roots_id` so arms
measured under different search cannot be paired, records every arm through the ledger, and prints the
pre-registered read (`NULL` / `EFFECT` / `INCONCLUSIVE`) beside the numbers so a null cannot quietly become
"promising" while it is being written up.

**`harness.ledger.run_budget` — provenance is DERIVED, not asserted.** Batch index comes from the filename, games
from the run's own `metrics.jsonl` (or, for runs predating per-batch cost accounting, `iterations_done` x the
config's games-per-iteration), and any checkpoint that is not the run's last batch is `budget_matched`. The point
is not convenience: pick an earlier checkpoint because it looked good and its budget shrinks with it, so `compare`
refuses the pairing. A cherry-picked checkpoint cannot wear an honest label.

**The L1 guard was checking the wrong thing.** It compared provenance LABELS for equality. The bins A/B ran 24
batches against a control configured for 40, so the like-for-like arm is the control's ckpt_23 — selection-free,
but not `final`. Label-equality refused that legitimate comparison, and the only way through was
`allow_mixed_provenance=True`, which also switches off the guard against the actual L1 error. What a comparison
must agree on is **selection status**, not the label: `SCORE_SELECTED = {gate_selected, best_of_n, arbitrary}`
(`arbitrary` gets the pessimistic reading — an unexplained checkpoint cannot be shown not to have been picked by
looking). `compare` now raises only when the arms straddle that line, and separately refuses a `budget_matched`
arm whose budgets do not in fact match — a label that would otherwise let the L2 error through wearing a
reassuring name.

**Generalised lesson, third instance:** a guard that fires on the wrong cases teaches people to disable it. The
over-broad solve-depth guard (§C.9) and this one failed the same way — both checked a PROXY (source text; a label)
instead of the thing itself (actual solve depth; whether a score was consulted). When a guard produces a false
positive, the fix is to make it check the real property, never to add an opt-out beside it.

### §C.17 — THE CONTROLS WERE NEVER MATCHED: A TRAINING-CODE AUDIT (2026-09-08)

**Every A/B this project has run compared arms trained by DIFFERENT CODE.** Not different flags — different
harness source. Controls are files on disk, files do not record what wrote them, and reusing an old control was
celebrated in §C.15 as making the experiment cheap. Provenance, budget, root family, seed roles and multiplicity
all passed. None of them could see it.

#### The audit

Fingerprints are `training_fingerprint(game, revision=<last commit before the run started>)`.

| run | role | training code | |
|---|---|---|---|
| `better1` (1.79M) | capacity arm | `35e07af6e079` | |
| `carry_03` (302K) | capacity arm, **and control for gpool + bins** | `bceb94d254eb` | |
| `seedrep_302K_s101` | control for gpool + bins (seed 101) | `bceb94d254eb` | |
| `ab302_gpool_s0` / `_s101` | gpool arms | `488601381041` | |
| `ab302_bins_s0` / `_s101` | categorical-head arms | `5a55087160db` | |
| `ctrl302_postfix_s0` / `_s101` | **NEW, launched 10:04** | `5a55087160db` | ✅ matches the bins arms |

Four distinct states; every comparison drawn so far crossed at least one boundary. The largest single difference
is the optimizer: Adam was rebuilt every ITERATION (~200x/run) before `619e252`, every BATCH between `619e252`
and `eacf154`, and once per run after. So each A/B varied its flag *and* the optimizer regime.

#### What survives, and what does not

**Global-pool (null) and categorical head (null): the readings survive, conditionally.** The confound is
directional — in both, the TREATMENT held the newer optimizer and still lost (−0.005, −0.010). If persistent Adam
is neutral-or-better, "not a win" holds. But §C.14 shipped persistent Adam as a *fix* and never measured it, and
deliberately resetting optimizer state across a shifting self-play distribution is a real RL technique. The sign
is unknown, so this is a conditional reading, not a result.

**Capacity ("no architecture difference"): NOT ESTABLISHED.** Worse than a code-era gap. The span between the two
arms includes `c47d562`, a 298-line neural.py change that introduced `_frontier_order` — called unconditionally
in the endgame loop that BOTH runs used, so their endgame training data was selected differently. And `better1`
has no config and no summary: **its recipe was never recorded at all**, so recipe parity cannot be checked even
in principle. The practical decision is unaffected — the 302K net is 2.5x faster and measured no worse, so it
remains the right net to build on — but the scientific claim is withdrawn.

#### One run bounds all three

`ctrl302_postfix_s0` is the era-`5a55087160db` scalar baseline. Against `carry_03` (era `bceb94d254eb`, same
arch, same seed, same budget) it MEASURES the optimizer-era effect directly. If that effect is small, the bound
propagates to every confounded comparison above and retro-validates them; if it is large, the nulls were reading
the optimizer. Either way it also de-confounds the categorical-head A/B, which is what it was launched for.

#### The guard

`harness/fingerprint.py` hashes the normalized syntax trees (docstrings and comments stripped) of the nine
modules that can change weights. Runs write `provenance.json` at start — fingerprint plus the full request, since
`better1` proved a run that does not record its own recipe is unauditable forever. `Ledger.compare` refuses
mismatched fingerprints and warns when either is unknown, so every legacy entry now says so out loud.

**Why not the commit hash.** It changes when the docs change. A guard that fires on valid comparisons gets
switched off, and this is the THIRD time that failure mode has bitten: the over-broad solve-depth guard scanned
source text instead of measuring solve depth (§C.9), the L1 provenance check compared labels instead of asking
whether a score was consulted (§C.16), and a commit hash would name a revision instead of the code that computes
weights. **A guard must check the property itself. Checking a proxy produces false positives, false positives
produce opt-outs, and an opt-out is how the guard was going to fail anyway.**

Measurement, benchmark and ledger code is deliberately excluded from the fingerprint — it reads checkpoints and
never feeds back into training, so including it would make the guard fire on measurement work. Verified: today's
ledger and measurement changes leave the fingerprint at `5a55087160db`, unchanged from `eacf154`.

#### §C.17a — the guard's first act was to catch the same mistake being made again (2026-09-08, same morning)

Adding the `provenance.json` write to `scaled_run.py` moved the fingerprint `5a55087160db` -> `4bb7ae0dc3ce` —
because `scaled_run.py` is itself one of the nine fingerprinted modules. `ctrl302_postfix_s0` had already started
under `5a55087160db`; `ctrl302_postfix_s101` starts hours later and would have loaded the edited code. **The two
controls built to fix a cross-era comparison would have been in different eras from each other and from the arms**
— the identical failure, two hours after building the guard against it, committed by the person who built it.

The instrumentation is therefore REVERTED until both controls finish. Two rules follow, and they are the real
output of this episode:

1. **Never edit a training-path module while runs whose comparison is not yet drawn are outstanding.** A control
   is not "done" when its checkpoints exist; it is done when its comparison has been recorded.
2. **Instrumentation must live OUTSIDE the fingerprinted set.** Writing run metadata cannot change weights, but
   putting the call in `scaled_run.py` makes it change the fingerprint. When the controls land, the recording
   moves to a non-fingerprinted `harness/runlog.py` so that future changes to WHAT is recorded cost nothing. The
   one-time call-site edit is unavoidable; the recurring churn is not.

**Pending when the controls finish:** (a) re-apply run-start provenance recording via `harness/runlog.py`;
(b) a resume guard — `scaled_run` resumes from checkpoints, so resuming a run after a training-path edit splices
two code eras inside a SINGLE run, which no comparison-time check can detect.

### §C.18 — CANDIDATE #1 VERDICT + THE ADAM "FIX" WAS A NULL (2026-09-09)

**The first within-era A/Bs this project has produced.** All arms fingerprinted `5a55087160db`; the ledger drew
the comparisons instead of refusing them. Two seeds each, paired at n=384 on measurement seed 257, pre-registered
read (|mean| < 0.03 with neither seed significant ⇒ null).

#### Categorical value head (value_bins=21) — NULL, refuted as a lever

| | seed 0 | seed 101 | pooled (n=768) |
|---|---|---|---|
| bins vs scalar (same code) | −0.0078 | −0.0078 | **−0.0078**, 95% CI [−0.031, +0.015], p=0.58 |

Identical across seeds — a small, consistent, non-significant NEGATIVE, not noise-around-zero. The categorical
head is if anything marginally worse than a scalar value head at 6x7/302K. Rules out any benefit beyond ~1.5
points. **The last value-FORM lever is closed.** Target-QUALITY levers (distillation, exact endgame targets)
remain the only thing that has ever moved this metric; the target's parameterisation does not.

#### §C.14 optimizer change — NULL, and it was mis-shipped as a fix

| | seed 0 | seed 101 | pooled (n=768) |
|---|---|---|---|
| Adam per-run (new) vs per-iteration (old), same config | −0.0026 | −0.0208 | **−0.0117**, 95% CI [−0.035, +0.012], p=0.39 |

`ctrl302_postfix` (era `5a55087160db`) vs `carry_03`/`seedrep` (era `bceb94d254eb`), config held fixed, code the
treatment. §C.14 shipped "one Adam per run" as a CORRECTNESS fix on the theory that ~200 per-iteration resets
were destroying optimizer state. **They were not: the change is not measurably better, and leans mildly negative.**
Per-batch training is long enough that Adam's moment warm-up is negligible, so resetting it costs nothing. §C.14's
"BUG FIX" framing is withdrawn — it is a behaviour-neutral refactor.

#### Correction to §C.17's "the nulls survive"

§C.17 (and what I told the user on 2026-09-08) leaned on the seed-0 optimizer number (−0.003) to argue the
cross-era confound was "an order of magnitude below" the gpool/capacity effects. **The second seed pulls the
pooled optimizer effect to −0.012 — the SAME order as the gpool −0.005, not below it.** So the optimizer axis
BOUNDS the confound (at ~0.01–0.02) but does not eliminate it; the gpool and capacity comparisons remain
confounded at a magnitude comparable to their own effect sizes, and their precise point estimates are not
trustworthy. This is the 2-seed rule working exactly as intended: a one-seed number that happened to be small got
generalised, and the second seed caught it.

**What is robust across all of it:** no architectural or value-form lever tried — width, depth, parameter count,
global-pool, categorical value head — moves conversion beyond ~0.03 at n=384. Every large measured effect in this
project came from target quality or measurement discipline. That conclusion does not depend on any of the
confounded point estimates; it is the consistent sign of every well-powered null.

#### Candidate #1 CLOSED. Next: #2 exploitability (needs no training), then #3 transfer test.

### §C.19 — EXPLOITABILITY IS CONFOUNDED BY OPENING VALUE; THE PAIRED FORM IS THE TRANSFER METRIC (2026-09-09)

**Candidate #2. The raw LBR exploit_rate is not a model-quality number** — it is dominated by the game-theoretic
value of the opening, and reporting it as exploitability is the same class of error as value-collapse.

#### The screen, taken naively, looks alarming

302K final net (`ctrl302_postfix_s0`), `lbr_screen`, 64 diverse 4-ply openings x2 seats, measurement seed 131:

| refuter depth | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| exploit_rate | 0.148 | 0.234 | 0.250 | 0.242 |

A depth-2 refuter "beats" the 0.88-conversion model 23% of the time — apparently contradicting conversion. The
tell that it is not real: the rate rises 1→2 then PLATEAUS. Genuine exploitation deepens with the refuter; a fixed
pool of already-lost openings does not.

#### Paired against a strong reference, the signal vanishes

Connect-4 is a P1 win with perfect play, so a random 4-ply opening is often theoretically LOST for the model's
seat (especially as P2). When an oracle beats the model from such an opening it is realising its own win, not
exploiting the model. Subtract what a strong reference (depth-10 + endgame solve) also loses on the SAME openings
against the SAME depth-2 refuter:

| | loss rate |
|---|---|
| model | 0.2344 (30/128) |
| strong reference | 0.2500 (32/128) — openings simply lost |
| **CLEAN = model − reference** | **−0.0156** (17 vs 19 discordant, McNemar NS) |

**The model is statistically indistinguishable from a depth-10 reference at avoiding losses from these openings.**
The exploitability lens AGREES with the 0.88 conversion once the opening-value confound is removed; the two never
conflicted. The earlier "depth-2 refuter beats the champion 15% as P1" (§C.8 #14) was this confound reported as a
model finding — **withdrawn.**

#### What this means for the metric

- **Raw single-agent exploit_rate must never be reported as model quality.** It measures the opening corpus, not
  the policy. `lbr_screen` is a diagnostic profile, not a scorecard.
- **Exploitability must be PAIRED** — model minus a reference on identical openings — so the opening-value
  confound cancels, exactly as root-difficulty cancels in paired conversion (§C.13).
- **Absolute vs relative is a real fork.** An ABSOLUTE number (vs a truly-perfect reference) at diverse shallow
  openings needs the solver, which hits the opening wall from 4-ply; at deep openings the solver is cheap but the
  measurement overlaps conversion. So exploitability adds nothing as a second ABSOLUTE Connect-4 number.
- **Its unique value is the RELATIVE, solver-free form: model A vs model B on shared openings.** The confound
  cancels between arms with no solver at all — which makes it the near-optimality yardstick that TRANSFERS to
  chess/Go (candidate #3), where conversion cannot go because it needs the solver to label won roots.

**Deliverable:** `paired_exploitability` in benchmark.py — the transfer-ready metric, built and tested here on the
calibration game so #3 inherits a measured instrument rather than a fresh heredoc.

### §C.20 — CROSS-GAME REGRESSION MATRIX (user direction, 2026-09-09) — the payoff of the fingerprint

**User:** "once we have enough types of games in ... introduce a new model or variation or some change — see how all
the games' training scores behave."

The deliverable: a **change x game matrix**. Rows = a change (a new net, a lever, any variation), columns = the
games, cells = the paired score delta with CI. It turns a single-game A/B — the thing we spent §C.15-§C.19
fighting confounds on — into a GENERALISATION test: does a lever move EVERY game (a real transferable lever) or
just one (game-specific, or noise)? Candidate #1 is the motivating case: value-form levers are null ON CONNECT-4;
only the matrix says whether that null generalises.

**Most of the machinery already exists — this is why the fingerprint was worth building:**
- A change IS a training-code **fingerprint delta** (§C.17). A matrix row is "the same code change applied to
  every game", verified by construction, not asserted.
- `Ledger.compare` already refuses cross-fingerprint comparisons unless `treatment="code"` (§C.18) — so a row is
  structurally a clean code-A/B per game, with `config` held fixed.
- Each column contributes a PAIRED number: `paired_conversion` where a solver exists (checkers, NMM, pentago all
  have known values), `paired_exploitability` where it does not (othello, and the north-star games) — so every
  cell is confound-controlled, never a raw rate.
- The pre-registered read (`measure_ab`) and multiplicity correction extend to the matrix: N games = N
  comparisons on the family; a lever "wins" only if it clears the corrected bar, and "generalises" only if the
  sign is consistent across columns.

**The one genuinely new piece:** a multi-game runner + matrix view — run the same config across all encoded
games at a fixed fingerprint, collect the per-game paired deltas into one ledger-backed table, and render it
(chat-reachable capability, per the north star). This is also the first labelled data for the §C.6 META-SELECTOR
(features -> which process suits which game class).

**Gated on:** (a) the unified rule-encoding (§C.19-followup, in flight) so games are cheap to add and measured
identically; (b) >=3-4 games actually encoded (the current thread: checkers -> NMM -> pentago -> othello). Not
built now; recorded so the encoding and the games are built with this matrix as their downstream consumer.

### §C.21 — UNIFIED RULE ENCODING: ATFP-v2, ADOPTED WITH THE TRANSFER CLAIM DEMOTED (2026-09-10)

Designed by a 14-agent workflow (survey GDL/GGP/RBG/OpenSpiel/MuZero → decompose the 4 games → synthesize →
3 adversarial critics → revise). Full artifact: the workflow output (tasks/wczn6dtrc.output). The critics forced
the headline claim down from "seamless transfer by construction" to an honest, measured position. What we adopt:

**THE BANKABLE WIN IS THE RULE-MODULE LIBRARY, NOT A SHARED TRUNK.** A game is ~8 pure composable hooks over the
already-generic Game Protocol + already-generic search: board() (cells, coord, delta, valid_mask, normalized
adjacency A, symmetries as (cell_perm, channel_perm, bank_perm) triples), piece_types(), plane_spec(),
atoms(state) (from can_place/can_step/can_lift/can_drop/can_remove + a mandatory() filter), effect() (via reusable
primitives ray_walk/flip_flank/jump-over/place/remove/rotate_quadrant/promote), is_switch() (the SINGLE compound
mechanism), terminal() (eliminated/immobilized/line_of_n/count_majority/board_full/repetition), returns(). This
delivers genuine "one file per game" and is independent of any weight-sharing.

**ONE ACTION MECHANISM.** num_actions = C_a·H·W + G: a spatial (channel,cell) atom tensor PLUS a small global/
region bank (PASS, claim-draw, Pentago's 8 rotate-quadrant-direction slots). Every multi-decision move is a
SEQUENCE of atoms with current_player unchanged until is_switch fires (checkers multi-jump, NMM lift→drop and
mill→remove, Pentago place→rotate). The first draft's two flaws are fixed: Pentago's rotate is a bank action (not
a faked cell-anchor entangling place+rotate in one channel), and there is exactly one compound mechanism (no
per-game joint-vs-sequence fork). Global-bank slots are permutation-EXEMPT under board symmetry.

**TRANSFER IS A HYPOTHESIS WITH A MANDATORY CONTROL, NOT A CONSTRUCTION.** The critics invoked our own memory
(phantom 0.969; every A/B crossed a code era) to refuse an unfalsifiable "by construction" claim. Before ANY
trunk sharing ships: a compute-matched cold-start vs warm-start control, per channel-group and per component
(trunk vs head), and no sharing ships without a positive matched delta. Named risks: (a) trunk transfer can be
NEGATIVE — "more of my pieces = good" is value-relevant with OPPOSITE sign in Othello vs checkers/NMM, so the
value readout stays per-game; (b) PLACE-channel semantics diverge (gravity/flank/graph-drop), so cross-game
policy warm-start is partial-to-negative; (c) Connect-4 is NOT a transfer donor — it keeps its 7-wide head and
306 checkpoints as a LEGACY EVAL BASELINE only. The one genuine size-invariance mechanism kept: MASKED global-
pool lets a trunk evaluate a different board size.

**NMM IS AN HONEST TOPOLOGY MISFIT.** The 24-node graph does NOT embed on grid-8 adjacency (ring edges span 3
grid cells, invisible to a 3×3 conv; grid-diagonals are false neighbours; mill-lines span up to 7 cells). It needs
a gated adjacency-propagation layer (h' = A_norm·h, a GCN layer that recovers the conv exactly when handed the
grid graph) + its 16 mill-lines as fixed membership planes. This relaxes "CNN unchanged" for graph games — stated
plainly, gated so grid games and the 306 C4 checkpoints are untouched.

#### THREE LIVE HARNESS DEFECTS — verified in current code, blocking prerequisites for game #2

The critics claimed three; all three CONFIRMED against source (they do not bite Connect-4, so the suite is green,
but each silently corrupts the moment a second game is added — the project's recurring latent-defect class):
1. `neural.py:384` `masked = torch.full((COLS,), -1e9)` — inference policy mask hardcoded to 7 wide; truncates/
   misaligns any game with num_actions != 7. FIX: `torch.full((game.num_actions,), ...)` + a test that FAILS for
   num_actions != COLS.
2. `neural.py:60` global-pool `h.mean(dim=(2,3))`/`h.amax` over ALL 64 cells, no valid-mask — dilutes/pollutes
   value aggregates for padded boards (NMM 24/64, checkers 32/64). FIX: valid-mask-weighted mean + masked_fill max.
3. `neural.py:27` `ROWS,COLS=6,7` module constants + `Linear(...,COLS)` heads — board dims + action count
   hardwired. FIX: read game.board_shape()/num_actions; conv-policy head behind an opt-in arch flag so the 306
   checkpoints load unchanged.
Two DESIGN-level bugs the first draft would have shipped, also caught: checkers "column mirror" maps dark→light
squares (parity flip, corrupts augmented data) — real symmetry is the valid_mask-preserving perms (180° + main
diagonals) with side-swap + STEP-DIR relabel; and Othello PASS is not D4-invariant (global slots must be
permutation-exempt). augment_examples() (neural.py:845, last-axis-only `x[...,perm]`) cannot express either and
must accept full (cell_perm, channel_perm, bank_perm) triples, each validated by asserting legal→legal on samples.

#### RECOMMENDED FIRST STEP (from the design; note the tension with the stated game order)

Build the rule-module library + game-agnostic plumbing + per-game right-sized heads, fix the three live defects,
and land ONE game as the second game and transfer probe — the design recommends **Othello** (cleanest fit: pure
PLACE + flip + PASS, no sub-turns, closest to Connect-4 for a cold-start transfer measurement). This is in TENSION
with the user's stated order (checkers → NMM → pentago → othello): checkers is actually the HARDEST first port
(multi-jump sub-turns + the dark-square symmetry bug + diagonal-only waste). Decision to surface to the user:
honor the stated order (checkers first) vs. do the engineering-rational order (othello first as the cleanest
library validation + transfer probe, then checkers). Do NOT build the shared trunk on faith either way — it ships
only behind the cold-start control (§C.20's matrix is the natural home for that measurement).

#### §C.21 Increment 1 — LANDED (2026-09-10): the net is board-shape/action-count aware; Connect-4 byte-identical

All three live defects fixed at the ARCH level (persisted with the weights, defaults = Connect-4, so the 306
checkpoints build and load strictly unchanged): `board_shape`, `num_actions`, `valid_mask` on `Connect4Net`;
heads sized from them; `_policy_value` mask is `num_actions` wide and REFUSES a net built for another game
(`game.num_actions != net.num_actions` raises BEFORE the forward pass); global-pool aggregates go through a pure
`_masked_pool(h, mask)` that ignores pad cells; `encode`/self-play/augment/train read the shape from the game or
net, never from module constants. Protocol gains optional `board_shape` (Connect-4 and TicTacToe declare it).
Tests: `tests/test_net_shape.py` (11) — goldens computed with the pre-refactor code pin the legacy and tower
forward outputs to 1e-6, and a real checkpoint's initial-state value (0.50067037).

**A test caught being vacuous, by its own mutation guard.** The first D2 test asserted "net value output is
identical with pad noise under a mask" — and passed. Its guard ("...and DIFFERENT without the mask") failed:
the probe showed `v_raw` = 0.1431 in all four cases. The randomly-initialised value head was dead-ReLU on that
input, so nothing reached it either way. Replaced by tests of the property itself (`_masked_pool` ignores pad
cells; unmasked leaks; masked == plain pool over only the valid cells) plus wiring tests (block and net route the
mask). Third instance this week of "check the property, not a proxy" — this time the proxy was a downstream
output that happened to be constant.

**Training-code era moved, legitimately: `5a55087160db` → `d31c86a58185`.** The net class is training-path code. Every
comparison on the 5a55 era was drawn before this edit (candidates #1/#2 closed, controls measured), so §C.17a
rule 1 is satisfied. Any FUTURE arm compared against a 5a55 checkpoint is a code-A/B and must be declared as such.

**Not in this increment (next):** the rule-module library (hooks + effect primitives), the factored
`C_a·H·W + G` action head behind an opt-in flag, full 2-D symmetry triples in `augment_examples`, the sub-turn
macro-transition value target, the NMM adjacency layer. Othello needs only the first two.

#### §C.21 Increment 2 — LANDED (2026-09-10): rule-module library + Othello; THE PROCESS RUNS WITH NO SOLVER

**First north-star data point.** `scaled_run` — the real, chat-reachable path, UNCHANGED — trained an Othello net
end to end (`othello_smoke`: 2 batches, 27 s, loss 5.35→5.16, checkpoints + buffer + metrics + summary). Every
solver-backed probe reported `null` instead of crashing (the "chess/Go degrade cleanly" path, exercised for the
first time). The checkpoint carries the derived arch (`board_shape [8,8], num_actions 65`), loads strictly, and a
net-guided Gumbel-MCTS plays legal Othello from the opening. Nothing in the trainer, search, league, replay or
gating knows the game changed.

**What landed.** `harness/rules.py` — the library's first primitives (`DIRS8`, `grid_rays`, `flank`, `majority`),
pure and tested (8). `games/othello.py` — composed ON TOP of them: move generation is `flank` over rays, terminal
is `majority`, PASS = action 64 at the end of the space (the design's global-bank slot), `done` DERIVED from the
position so the state stays Markov without a pass counter (12 tests, incl. a hand-built pass→move→terminal→
majority scenario). `arch_for_game` — a config never hand-types 65: `board_shape`/`num_actions` are FILLED from
the game and REFUSED when stated differently, so the D1 defect is caught at construction. Registry entry.
Suite: 388 passed / 2 skipped after Increment 1; +Increment 2's 33 targeted tests green; definitive run pending.

**Deliberate omissions, with reasons.** No `symmetries()` on Othello: `augment_examples` permutes the LAST tensor
axis (Connect-4 columns), so exposing 64-cell dihedral perms would corrupt data, not multiply it (TicTacToe's
cell-perm `symmetries()` has the same latent hazard — no caller trains it augmented). The augmenter rewrite to
(cell_perm, channel_perm, bank_perm) triples with permutation-exempt global slots is its own increment. `summary.json`
echoes the REQUEST's net_arch rather than the built net's (the checkpoint is authoritative; cosmetic, deferred so
the training era is not moved right before a long run). Value target stays ±1/0 (disc margin via value_bins is a
later experiment, not a port requirement).

**Training-code era:** connect4 `21a7debbf2d7`, othello `880d339bd65e` (the game module is part of its fingerprint).
All Connect-4 comparisons on `5a55087160db` were drawn before any of this; a future arm vs a 5a55 checkpoint is a
declared code-A/B.

**Next (the transfer test proper):** a real Othello run at the Connect-4 recipe's scale (solver-free league on,
endgame/distill off, probes zeroed), then a SOLVER-FREE learning measurement — `paired_exploitability` between
checkpoints with the game-agnostic `MctsAgent` (pure UCT) as the refuter — the yardstick §C.19 built for exactly
this. The cold-start transfer control (§C.21) follows once a trained Othello net exists to warm-start from.

#### §C.21 Increment 2a — the transfer test's first CATCH: a Protocol leak in the "domain-oblivious" search (2026-09-10)

Smoke-testing the solver-free measurement tool (`scripts/measure_exploit.py`: paired exploitability with the
game-agnostic pure-UCT `MctsAgent` as refuter, through the ledger) crashed on Othello: `MctsAgent.act` read
`game.step(state, a).winner` — the STATE DATACLASS's field, not the Protocol's `game.winner(state)`. Connect-4 and
TicTacToe states happen to carry `winner`, so it had always worked. Two sites (`agents.py:105`, `:220`), plus
`benchmark.sample_solvable_positions` reading `s.done` (harmless today, same class). All three now speak the
Protocol; a regression test plays every generic agent — and a full UCT-vs-UCT game — on Othello, the first
state without a `winner` field. Grep confirms no state-field reads remain in generic harness code.

Why it matters beyond the fix: the smoke run passed because it ran `league=False`; the real run turns the league
on, and the league's opponents ARE `MctsAgent` — a multi-day run would have died mid-batch. This is what
"run the process unchanged on a new game" is FOR: every hidden Connect-4 assumption surfaces as a crash on the
second game rather than as a silent bias.

**Fingerprint list grown.** `harness/agents.py` was not in TRAINING_MODULES, yet league opponents shape the
self-play data — training-path code. Added. The ledger's recorded eras (`5a55087160db`, `bceb94d254eb`, ...) were
computed under the nine-module list; `TRAINING_MODULES_V1` keeps it re-derivable (tested: `eacf154` under V1 still
hashes to `5a55087160db`). **Launch era for the Othello run: othello `ac11124cf17a`, connect4 `07db4fa43df0`.**

#### §C.21 THE TRANSFER RUN LAUNCHED — `othello_302K_s0` (2026-09-10 00:41)

Gate: 414 passed / 2 skipped on the exact tree that trains. Config = the Connect-4 302K control recipe
(`ctrl302_postfix_s0.json`) with ONLY the solver-backed parts removed: `endgame` off, `benchmark_positions`/
`offline_openings`/`gate_roots` 0; solver-free league ON; net shape derived (`[8,8]`, 65 actions). 24 batches x
(5 iters x 80 games), seed 0. Provenance written by the launcher from outside the fingerprinted modules:
training era **`4a9e255a90c6`** (V3 list; launched as `ac11124cf17a` under V2 before `harness/rules.py` was
recognised as training-path code — re-stamped after verifying no fingerprinted file had changed; the V2 value is
kept in provenance.json), config **`147a3de64336`** — the first run that records its own code.

What it answers: does the UNCHANGED process learn on a game with no solver, read by a SOLVER-FREE yardstick —
`scripts/measure_exploit.py` (paired exploitability vs the game-agnostic UCT refuter) between late and early
checkpoints. It is also the COLD-START arm of the §C.21 transfer control; the warm-start arm (Connect-4 trunk
into Othello) needs this trained net first. Per-batch pace and ETA recorded once batch 0 lands (Othello games run
~3x Connect-4's length; expect slow).

#### PRE-REGISTERED READ for the Othello transfer run (written 2026-09-10 00:45, before any checkpoint exists)

The question: does the unchanged process LEARN on a game with no solver? The measurement, fixed now so the numbers
cannot steer it (the candidate-#1 lesson): `scripts/measure_exploit.py --game othello --refuter uct --depths 200
--n-openings 128 --sims 96 --seed 131`, arms = `ckpt_23` (late), `ckpt_11` (mid), `ckpt_3` (early), pairs drawn
with `treatment="budget"` (same run, same code, budget the variable). Outcome = HELD rate (1 − loss) over 256
paired cells (128 openings × 2 seats).
- **Learning:** late − early > 0 AND significant at the family-corrected alpha (2 comparisons on the family ⇒
  0.025). Sanity: mid between early and late in sign (not required to be significant).
- **Null / not learning:** |late − early| < 0.03 and not significant ⇒ the process does NOT measurably learn Othello
  at this budget — reported as such, not as "needs more batches".
- **Inconclusive:** |diff| ≥ 0.03 but not significant ⇒ a second seed (`othello_302K_s101`) before any claim.
Detection limit at n=256 paired cells is roughly ±0.04 on the held-rate difference; a real learning effect from a
random-init net over 9,600 games should be far larger than that, so a null here is informative, not underpowered.
The Ledger's `treatment="budget"` (learning curve: code + config held fixed, games must differ) was added for
exactly this comparison so it never has to be drawn outside the ledger.

**Measured cost of the unchanged recipe on Othello (2026-09-10 01:10, niced probe beside the run):** 17.1 s/game
vs Connect-4's 4.8 s at identical settings (sims 96, 302K net) — **3.6×**, from 60 plies/game at 283 ms/ply vs
26 at 187 ms/ply. A 400-game batch ≈ 1.9 h; 24 batches ≈ 2 days uncontended, 2–3 on the shared machine. Kept the
recipe unchanged (that is the experiment); the lever if the cost is unacceptable is self-play sims 96→32 (Gumbel
makes low sims nearly free on low-branching games — MiniZero's Othello result), which would be a declared recipe
change for BOTH transfer-control arms, not a quiet edit. **Early read scheduled:** mid (`ckpt_11`) vs early
(`ckpt_3`) fires automatically when batch 11 lands (~1 day) as comparison #1 on the pre-registered family; late vs
early and late vs mid follow at batch 23 (the ledger corrects alpha for all three).

#### §C.22 — RESUME is a measurement surface, not a convenience (2026-09-10)

The decision to let the 2–3 day run stand promoted **resume** to the load-bearing path: on this machine the run
will be interrupted more often than it is launched. `scaled_run` was written resumable (checkpoint + replay
buffer + nogood store per batch, `start = last ckpt + 1`) and the mechanics hold — verified end-to-end on
Othello, including the shape-aware net's 8×8/65-action arch surviving `save_net`/`load_net` and the buffer
surviving `torch.load` under torch 2.13's `weights_only=True` default. But reading the resume path found three
ways a resume corrupts the RECORD while looking like it worked. All three were **confirmed empirically**, not
merely argued:

- **R1 CODE DRIFT (the likely one).** `provenance.json` is stamped ONCE at launch; resume trains against
  whatever source is on disk. Edit a training module during the run, let it be interrupted, and later batches
  carry a different era than the run's own label claims — §C.17's control-reuse confound reintroduced
  *invisibly, through the mechanism built to detect it*. Days of a live run is exactly when code gets edited,
  so this is a near-certainty rather than a tail risk. Worse, the old launcher re-stamped `provenance.json`
  unconditionally, so resuming through it would have **overwritten the launch era with today's** and destroyed
  the evidence.
- **R2 ORPHAN CHECKPOINT.** `ckpt_N.pt` is written before batch N's metrics row (the window spans the ~50 MB
  buffer write and the metric pass). Killed in between, that row is lost for good — resume restarts at N+1 and
  never revisits N — and `run_budget`, which establishes a budget by COUNTING rows, then refuses `ckpt_N` *and
  every checkpoint above it*. Confirmed: `run_budget(ckpt_1)` → `ValueError: claims batch 1 but its run records
  1 batches up to it`. Training continues perfectly while the run silently becomes unreadable by the ledger —
  i.e. the pre-registered comparison the run exists to draw could never be drawn.
- **R3 TRUNCATED BUFFER.** `torch.save(buf, buffer_path)` is not atomic — unlike the `_write_json_atomic`
  written for this exact failure class a few lines above it. Confirmed: a half-written `buffer.pt` raises
  `OSError` on **every** subsequent resume; the run is wedged until a human deletes the file.

**Guard (`harness/resume.py`, 20 tests, all six mutations caught).** `preflight(run_dir, repair=True)` refuses
R1 and repairs R2/R3. Repair is deliberately conservative: it deletes only artefacts the ledger has been shown
unable to read (that batch is retrained, costing what the interruption cost anyway), it truncates the stale
metrics rows above the gap (otherwise retraining appends a SECOND row for that batch and breaks `run_budget`
all over again), and it touches nothing while the era check is failing — code drift is a judgement for a human,
not something to tidy away. The property test is the real one: *after repair, `run_budget` succeeds on every
surviving checkpoint*. The module sits outside `TRAINING_MODULES` and is imported by nothing in the training
path, so consulting it cannot move the era it checks (§C.17a rule 2).

**On the path, not beside it** (the §C.11 ledger lesson). `scripts/run_scaled.sh` is now the single command for
both launch and resume: it stamps provenance only when absent, runs the preflight, and **refuses to repair while
a trainer for that run is alive** — a live trainer looks exactly like an orphan checkpoint for the seconds
between `save_net` and the metrics append, so an unguarded repair tool would have become the data-loss cause it
exists to prevent. Both refusals demonstrated against the live run.

**Declared, not fixed: resume is not trajectory-equivalent.** §C.14 keeps ONE Adam per *process* and no optimizer
state is persisted, so every resume restarts the moment estimates — a real difference (per-run vs per-batch Adam
measured pooled −0.012 in §C.18, small but not zero). Preventing it means editing `scaled_run.py`, which is
FROZEN while this run's comparison is undrawn (§C.17a rule 1). So the preflight instead **counts resumes into
`provenance.json`**, putting the number of such discontinuities on the record where the ledger can see it. Fixing
R2/R3 at the root (metrics row and checkpoint written atomically together; `buffer.pt` via temp+rename) is
queued for when the freeze lifts.

#### §C.23 — the unchanged process LEARNS a game with no solver (2026-09-10, batch 11 read)

The pre-registered comparison #1 fired automatically when batch 11 landed. Paired exploitability, 128 openings ×
both seats, both arms refuted by the same game-agnostic UCT-200 searcher, budget declared as the treatment:

| arm | checkpoint | games | held (1 − loss) | 95% CI |
|---|---|---|---|---|
| mid | `ckpt_11` | 4800 | **0.5820** | [0.5208, 0.6408] |
| early | `ckpt_3` | 1600 | 0.1250 | [0.0900, 0.1711] |

**+0.4570, McNemar p = 5.7e-27**, discordant pairs 127/10, one comparison on the family. Same code
(`4a9e255a90c6`), same config (`147a3de64336`), same run — a clean learning curve. This is the north-star
question's first real evidence: the Connect-4 recipe, transplanted unchanged to a game with **no solver, no
endgame tablebase, no opening book and no oracle-labelled targets**, learns — and the yardstick that measured it
needed none of those either (§C.19).

**What it does NOT say.** `ckpt_3` (1600 games) is barely trained, so this is a low bar cleared decisively, not
near-optimality: `mid` still loses 41.8% of paired openings to a plain 200-sim UCT searcher. The informative
read is late-vs-mid at batch 23 — does it keep climbing or saturate? `opening_value` continuing to drift
positive (+0.55 by batch 13) on a game believed near-drawn is the number to watch, and precisely why the
adjudicating metric is paired exploitability rather than the net's own value.

**The defect the read exposed — L1 through a DERIVED label.** The result printed `provenance=final` for
`ckpt_11`. It is not final; it was merely the newest row when the measurement looked. `run_budget` derived
"final" from *is this the last row in metrics.jsonl*, which on a live run is a fact about **when you looked**,
not a property of the checkpoint. The dangerous form is not the stale label but what it would launder: stop a
run early at a nice-looking batch — and we are watching `opening_value` and `final_loss` on a live run right now
— and its last checkpoint reads "final", the most trusted label in the system, when it was in fact chosen by
looking at a score. That is exactly L1, re-entering through a label nobody asserts by hand.

**Guard.** "final" now requires the run to have reached the end it DECLARED (`request.batches` in
provenance.json); any other index is `budget_matched`, which is what it already meant — an index pinned in
advance and never scored. `run_budget` also returns `run_complete`, recorded on the ledger entry and surfaced by
both measurement tools as a `completeness_warning`, so a mid-run reading announces itself as a progress report.
Re-derived on the live run, `ckpt_11` and `ckpt_3` both now read `budget_matched, run_complete=False` —
correct. 4 mutations, all caught; suite 445 green. The stale `final` in the ledger entry is harmless (neither
label is score-selected, so the comparison stands unchanged) and is overwritten when the batch-23 read
re-records that arm.

#### §C.24 — the run never saturated, and the proxies said it had (2026-09-11, batch 23 read)

Both pre-registered comparisons, same family, alpha corrected for all three:

| arm | checkpoint | games | held (1 − loss) | 95% CI |
|---|---|---|---|---|
| **late** | `ckpt_23` | 9600 | **0.8398** | [0.7899, 0.8797] |
| mid | `ckpt_11` | 4800 | 0.5820 | [0.5208, 0.6408] |
| early | `ckpt_3` | 1600 | 0.1250 | [0.0900, 0.1711] |

- **late vs early +0.7148**, p = 2.1e-44, discordant 194/11, alpha 0.025 — EFFECT.
- **late vs mid +0.2578**, p = 1.1e-09, discordant 93/27, alpha 0.0167 — EFFECT.

**The correction.** From the loss curve flattening (1.38 → 1.28 over batches 15–23) and `opening_value`
plateauing (~+0.55 from batch 12), I read the run as having "saturated in its second half". That was wrong, and
the measurement refutes it decisively: the second half bought **+0.2578 held** — loss to a 200-sim UCT refuter
fell from 41.8% to 16.0%, a 62% relative reduction, in exactly the stretch the proxies called flat.

**The lesson, which is the transferable part.** Training loss and the net's own `opening_value` are NOT proxies
for strength, and here they were actively misleading — both flattened while real strength climbed steeply. Loss
is computed against the net's own moving self-play targets, so it measures how well the net fits its current
data, not how good that data has become; `opening_value` is the net's opinion of one position, which is the very
quantity training is free to drift. Had we early-stopped on either — the obvious compute-efficiency move — we
would have thrown away the most productive half of the run. This is the §C.19 argument arriving as data: the
only number that tracked reality is the one measured by PLAY against an external opponent.

**What it means for the north star.** The run was stopped at 24 batches because 24 was pre-registered, not
because it stopped improving — it was still gaining fast at the end. So Othello's compute-efficiency frontier is
NOT yet located; 9600 games is a lower bound on what the recipe can use, not a plateau. A longer run (or the
sims 96→32 lever, now measurable against this curve as the control) is the next question, and `late` at 16.0%
loss is still far from near-optimal.

**Tooling defect fixed.** This read ran ~9 h emitting nothing until it finished, and for the third time in two
days a starved process and a wedged one were indistinguishable from outside — the actual state (CPU starvation:
12 minutes of CPU in 4h49m, load 42 from two vitest watchers, a pegged VS Code plugin host, Docker and a VM) was
only diagnosable by sampling `ps` counters. `paired_exploitability` now takes `on_progress`, and
`measure_exploit.py` prints games-done/total, s/game, elapsed and ETA every 16 games. 2 tests, suite 447 green.
The exit code 1 on that run was the broken `tee` path in the launch pipeline, not the measurement.

#### §C.22a — the three resume defects fixed at the root (2026-09-11, freeze lifted)

With the pre-registered comparisons drawn, §C.17a rule 1 released the training path and the §C.22 defects were
fixed where they live rather than only detected from outside. New era **0fb1adf0bce9** (12 modules).

- **R3 buffer atomicity** — `_save_atomic` (temp + `os.replace`, scratch removed on failure) replaces the bare
  `torch.save` onto `buffer.pt`. The same guard `_write_json_atomic` already gave JSON, which the ~50 MB replay
  buffer never had.
- **R2 orphan checkpoint** — the resume point is now `ledger_readable_prefix` (the batches for which a
  checkpoint AND its metrics row both exist) instead of "newest ckpt + 1", so an orphan is retrained rather
  than stranding the run; and `_append_metrics` REPLACES any existing row for a batch, because a duplicate row
  breaks `run_budget`'s counting exactly as a missing one does.
- **Optimizer persistence** — the run's single Adam (§C.14) is saved to `opt.pt` each batch and reloaded into
  the fresh optimizer on resume. The property test is equivalence, not existence: *a run split by an
  interruption reaches the same Adam step count as one that was never interrupted*. Resume is now the same
  experiment; the `resumes` counter stays as the record of how often it happened.

**A mutation survived, and it was the same lesson for the fourth time.** Reverting the call site to
`torch.save(buf, buffer_path)` left all six tests green: they proved `_save_atomic` *works*, never that the
batch loop *uses* it. Helper-correctness is not wiring. Fixed with a monkeypatch spy asserting `buffer.pt` and
`opt.pt` both go through the atomic path; the mutation now fails. Five mutations, all caught.

**Fingerprint coverage gap closed in the same breath.** `scaled_run` now imports `harness.resume` to choose
which batch training restarts from — that is training-path code by the identical argument that pulled in
agents.py and rules.py, and it was not in `TRAINING_MODULES`. An edit silently changing a run's resume point
would not have moved the era. Added, with `TRAINING_MODULES_V3` preserving the list the completed Othello run
was stamped under. The guard immediately proved itself: the finished run's preflight now REFUSES it
("launched as 4a9e255a90c6, on disk now 0fb1adf0bce9"), which is correct — that run cannot be extended under
different code.

Suite 458 passed, 2 skipped.

#### §C.25 — checkers: the game that actually tests the encoding design (2026-09-11)

Othello proved the rule-module library *composes*; checkers is what stresses it, because it breaks three
assumptions every earlier game satisfied. All three were absorbed by declaration rather than by special-casing
the harness — which is the ATFP-v2 claim, now under load:

1. **A turn is not a move.** A multi-jump is one turn made of several decisions, so `to_move` does not alternate
   on every `step`. The state carries `jumping` (the cell the sequence must continue from) and
   `current_player` simply keeps returning the same player. The design's sub-turn `is_switch` hook cost one
   field and NO harness change — self-play, MCTS and the league never learned that sub-turns exist.
2. **Pieces have types.** Men and kings cannot be expressed in the own/opponent 2-plane encoding, which was
   hardcoded in `encode`. A game may now declare `input_planes` and emit its own planes; checkers declares 4
   (own men, own kings, opponent men, opponent kings). The 2-plane path is untouched and Connect-4/Othello
   encode byte-identically, verified by test.
3. **Half the board is dead.** Play is on dark squares only, so 32 of 64 cells are permanently empty. Checkers
   declares `valid_mask` and `arch_for_game` routes it into the net, where the masked pooling built in §C.21
   Increment 1 — written months before any game needed it — excludes those cells from the global statistics
   instead of averaging 32 structural zeros into every feature. First real consumer of that work.

Actions are FACTORED as `direction * 64 + from_cell` (4 diagonals x 64 = 256), the design's C_a·H·W action-plane
stack, so a conv policy head can later predict them positionally with no change here. New rule primitives are
`diag_steps` and `diag_jumps` — pure geometry, present only when the target is on the board, so a lookup miss IS
the edge test and no caller repeats the arithmetic where wrap-around bugs live.

Rules are standard English draughts: capture mandatory, men forward-only, kings one square in four diagonals,
promotion ends the turn mid-sequence, no legal move loses, 40-move idle draw. `state_key` includes the idle
clock, since two identical positions with different clocks have different values near the draw limit.

**Verified:** 22 direct tests, 6/6 mutations caught (capture-not-mandatory, multi-jump-ends-turn,
promotion-keeps-jumping, men-move-backwards, idle-never-counts, winner-flipped). Generic harness agents play it
through the Protocol (the §C.21 leak regression). A 2-batch smoke trained end-to-end through the UNCHANGED
`scaled_run` with the league on in 11 s, and the checkpoint round-trips with `input_planes=4`, `num_actions=256`,
`board_shape=[8,8]` and a 32-cell mask buffer. One real bug found and fixed: an out-of-range action raised
`IndexError` from the label formatter instead of a clean `ValueError`. Suite 483 passed.

**Not yet done:** no training run (the frontier question from §C.24 is still open and a real run is a multi-day
machine commitment), and no `symmetries()` — checkers has a left-right mirror, but the augmenter is
last-axis-only and feeding it a 256-action factored permutation would corrupt data rather than multiply it.

#### §C.26 — PRE-REGISTRATION: the frontier and the sims-efficiency arm (2026-09-11, written BEFORE launch)

§C.24 left two questions open: how far does the recipe keep improving (it was still climbing when its budget
ran out), and does a smaller search budget buy more strength per unit compute. One pair of runs answers both,
because the frontier arm IS the efficiency control.

| arm | sims | batches | games | total simulations |
|---|---|---|---|---|
| `othello_frontier_s96` (A) | 96 | 48 | 19,200 | 1.84M |
| `othello_frontier_s32` (B) | 32 | 144 | 57,600 | 1.84M |

**Compute-matching is on SIMULATIONS, not games or wall-clock.** 96·N ≡ 32·3N, so B gets 3x the games. A
measured probe puts 32-sim play at **3.61x** cheaper per game than 96-sim (9.39 vs 2.60 s/game — the saving is
super-linear in sims, since a smaller tree also means fewer net evaluations per move). Matching on simulations
therefore CREDITS B WITH LESS COMPUTE THAN IT ACTUALLY SAVES: if B wins on this design it wins by more in
wall-clock. Simulations are also reproducible where contended wall-clock is not (yesterday's measurement lost
~3x to machine load), so they are the honest experimental unit.

**B keeps A's batch, not a bigger one.** The first sketch gave B 3x-larger batches; that would have tripled its
replay-buffer turnover and made the buffer horizon a second variable. B instead runs 3x as MANY identical
batches (400 games, `buffer_cap` 50000), so per-batch dynamics match and **only `sims` differs**. Everything
else is held fixed deliberately, including the league opponents (`league_mcts_sims` [60,200],
`league_snapshot_sims` 128): the league is the curriculum, not the treatment, so it must not move with the arm.

**Pre-registered reads** (paired exploitability, UCT-200 refuter, MEASUREMENT seed 131, 128 openings):
- FRONTIER, `treatment="budget"` within an arm: A@11 vs A@23, A@23 vs A@47; B@35 vs B@71, B@71 vs B@143.
- EFFICIENCY, `treatment="config"` at compute-matched points: A@11 vs B@35, A@23 vs B@71, A@47 vs B@143.

Seven comparisons on one root family, so the ledger corrects alpha to 0.05/7 = 0.0071. Reads are FIXED here in
advance; per §C.24 neither training loss nor `opening_value` may be used to judge progress or to stop either
run early, and per §C.23 "final" is earned only by reaching the declared batch count.

**Pre-registered caveat.** The ledger matches budgets on `games`, so a compute-matched comparison will trip its
"BUDGETS DIFFER" note. That note is correct about games and misleading about compute; the ledger needs a
compute budget alongside the games budget, which is the tooling task to build while these train.

**Schedule:** sequential (user's call), A first — the machine is shared with Docker, a VM and vitest watchers at
load ~31/10 cores, and parallel arms would thrash. A ~53 h, B ~44 h at the measured rates.

**Deployment budget at measurement — both arms at sims 96.** The arms differ in TRAINING search budget; they
are measured at the SAME deployment budget (96, as in every earlier read) so the comparison isolates *which
recipe produced the better model*. Measuring B at 32 would conflate the training question with a separate
deployment question, and the north star is a compute-efficient process producing a near-optimal MODEL, not a
cheap-to-deploy one. All seven reads share `--tag frontier`, so they form ONE root family and the ledger's
alpha correction applies across all of them.

**Consequence of launching: the training path is FROZEN again** (§C.17a rule 1) until BOTH arms are measured —
arm B must be trained by the same era as arm A (`3f2bacd402c7`) or the efficiency comparison is a code
comparison and the ledger will refuse it. That blocks `harness/rules.py` and `harness/neural.py`, and therefore
blocks nine men's morris (new geometry primitives) and the augmenter rewrite for ~4 days. Non-fingerprinted
work is unaffected: the ledger's compute-budget gap, measurement tooling, and the §C.20 cross-game matrix
runner all live outside `TRAINING_MODULES`.

#### §C.27 — the frontier arm reproduced the old run BIT-FOR-BIT, and that is two findings (2026-09-12)

Pre-registered frontier read #1 (A@11 vs A@23) fired on schedule: held **0.8398 vs 0.5820, +0.2578**, discordant
93/27, p<0.0001 — significant. It is also, to the digit, the completed run's late-vs-mid result. Checked rather
than assumed: `ckpt_11` and `ckpt_23` are **byte-identical** across the two runs (sha256 `97fd3042348e3ed4`,
`e99b4ebf28ca9eb7`), and all 24 batches match exactly on `final_loss`, `opening_value` and `league_vs_pool`.

**Finding 1 — the pipeline is bit-reproducible.** Same seed, ~26 h of stochastic self-play, 9600 games, a league
rebuilt per batch, and the run re-derived itself exactly under a DIFFERENT code era (`4a9e255a90c6` →
`3f2bacd402c7`). The checkers work touched `neural.py` and `rules.py` but was behaviour-preserving for Othello
(the multi-plane `encode` branch is taken only when a game declares `input_planes != 2`; `diag_steps`/
`diag_jumps` are new functions Othello never calls). This is a strong property most RL codebases do not have,
and it is now demonstrated rather than hoped for.

**Finding 2 — the era guard is conservative, and conservatism has a price: ~26 h here.** The guard refuses to
resume a run whose training source changed *textually*, which is the right default — behaviour-preservation is
undecidable in general and a wrong guess reintroduces the §C.17 confound silently. But in this case the old
run's checkpoints were provably reusable, and re-deriving them bought nothing. Frontier read #1 is therefore a
REPLICATION, not new information; the genuinely new territory is batches 24-47, and the frontier answer arrives
at A@47.

**Queued tool (outside TRAINING_MODULES, so the freeze does not block it): equivalence-based run seeding.** Given
a config and an existing run whose `config_fingerprint` matches, re-derive ONE batch and compare the checkpoint
hash to the existing one. Bit-identical ⇒ the eras are behaviourally equivalent *for this game* and the existing
checkpoints may be adopted, skipping the recompute; different ⇒ the code genuinely moved and the full run is
required. This CHECKS the property (a re-derived checkpoint matches) instead of trusting a judgement about
which edits "should" be behaviour-preserving — the §C.9 lesson applied to compute rather than to statistics.
It would have saved 26 h here and saves more as runs accumulate. Arm B gains nothing from it (144 batches at 32
sims are all new), so it is queued, not urgent.

#### §C.28 — the frontier is FOUND, and it is much closer than §C.24 implied (2026-09-14)

Arm A completed 48/48 batches in 53.7 h. Both pre-registered frontier reads are drawn:

| budget | games | held (1 − loss) | marginal gain |
|---|---|---|---|
| A@11 | 4,800 | 0.5820 | — |
| A@23 | 9,600 | 0.8398 | **+0.2578**, p<0.0001 — EFFECT |
| A@47 | 19,200 | 0.8945 | **+0.0547**, p=0.098 — INCONCLUSIVE |

**Returns collapse by ~5x across one doubling.** The first doubling of budget bought +0.258 held; the second
bought +0.055 and does not reach the corrected threshold (alpha 0.025). For this recipe on Othello, the useful
frontier sits around **9,600 games** — roughly where the original 24-batch run stopped.

**This corrects the extrapolation §C.24 invited, though not its literal claim.** §C.24 said the run "was still
gaining fast at the end" and that 9,600 games was "a lower bound, not a plateau". Both were true of the 11→23
interval. But the natural reading — double again and gain comparably — is refuted: 23→47 gained less than a
fifth as much. The honest summary is that §C.24 correctly refused to call a plateau from the loss curve, and
this read locates the plateau properly, by play.

**INCONCLUSIVE is not NULL, and the difference matters here.** The minimum effect n=256 can resolve at this
discordant count (~62 pairs) is **|b−c| ≥ 20, i.e. ~0.078 held**. The observed +0.0547 is BELOW that limit, so
this measurement could not have detected it even if it is entirely real. We may say "the second doubling's
return is small enough to be invisible at n=256"; we may NOT say it is zero. (The same detection-limit trap as
the knowledge project's "<0.044 invisible at N=5".)

**Consequence for the efficiency reads, handled WITHOUT moving the goalposts.** If arm B's compute-matched
differences are also in the 0.05 range, they will land inconclusive for the same reason. The fix is NOT to
raise `--n-openings` now: `n` is part of `roots_id`, so changing it mid-experiment creates a different root
family, breaks the pre-registered multiplicity accounting, and is post-hoc flexibility of exactly the kind that
invalidates inference. Instead, PRE-REGISTERED HERE IN ADVANCE: if any efficiency read returns INCONCLUSIVE
with |diff| < 0.078, a single higher-powered confirmation is run on a NEW family at `--n-openings 512`
(detection limit ~0.039), comparing only the arms that were inconclusive, and reported as its own family with
its own alpha.

`a47` reaches 0.8945 held — 10.6% loss to a 200-sim UCT refuter, down from 16.0% at A@23. Better, still not
near-optimal.

**CORRECTION to §C.26's cost claim (2026-09-14, measured on arm B's first batches).** §C.26 asserted that
matching on simulations "CREDITS B WITH LESS COMPUTE THAN IT ACTUALLY SAVES: if B wins on this design it wins
by more in wall-clock." **That is refuted.** Arm B runs at 40.6 min/batch against arm A's 67.1 — a ratio of
0.61, not the 0.28 the 3.61x per-game probe implied — so arm B's full 144 batches cost **97 h against arm A's
53.7 h: 1.82x MORE wall-clock for the identical 1.84M simulations.**

The cause is the league, and it is arithmetic, not noise. `league_frac` 0.4 sends a large share of self-play
against opponents whose search is pinned by `league_mcts_sims` [60,200] and `league_snapshot_sims` 128 — held
FIXED across arms deliberately, because the league is the curriculum and not the treatment. Those opponent
moves cost the same whatever the arm's own sims are, so only roughly half the work gets cheaper: predicted
blend 0.5/3.61 + 0.5 = 0.64 against 0.61 observed.

**The error was measuring a proxy instead of the property** — again. The 3.61x ratio came from a net-vs-net
probe at matched sims, which is not the workload a training batch actually runs. The property was "what does a
BATCH cost", and one batch of each arm would have answered it in 45 minutes before launch.

**What does NOT change:** the pre-registered comparison stays simulation-matched at 1.84M each. Cost information
is not outcome information, but redesigning mid-flight is how pre-registration dies, and no efficiency read has
been drawn. **What DOES change is the interpretation:** a B win on this design can no longer be read as a
wall-clock win — B would have to beat A while having spent 1.82x the wall-clock. The wall-clock accounting is
reported alongside the result as a cost fact, NOT smuggled in as a second ledger comparison. A genuinely
wall-clock-matched efficiency test would have to scale the league with the arm's sims, which is a different
experiment with a different curriculum.

#### §C.29 — the efficiency verdict INVERTS with the accounting unit (2026-09-17, efficiency read #1)

> **⛔ HEADLINE REFUTED BY ITS OWN SUCCESSOR READS (see §C.33). Read this section as a snapshot, not the conclusion.** The +0.1523 low-search win below was real at 460,800 simulations and DECAYED to +0.0156 (921,600) then reversed to -0.0625 (1,843,200) as both arms converged: a11's 4,800-game control was simply undertrained. At full convergence the DEEP-search arm wins, and a three-budget deployment sweep (§C.33) puts the pooled effect at **+0.0312, p=0.0007**. The wall-clock figures here are also wrong — arm B cost 127 h CPU / 142 h wall, not the ‘24.3 h’-era extrapolation; ratio 1.88x not 1.70x (§C.33). h3/h5 are marked SUPERSEDED in the register.

Pre-registered efficiency read #1 is drawn, and the ledger's compute-matching (built 2026-09-14, hours before it
was needed) carried it: "COMPUTE-MATCHED: b35 played 14400 games and a11 4800, which differ BY DESIGN — both
spent 460800 simulations". Without it the pair would have been refused as a mislabelled budget.

| arm | sims | games | simulations | wall | held |
|---|---|---|---|---|---|
| a11 | 96 | 4,800 | 460,800 | 14.3 h | 0.5820 |
| **b35** | **32** | **14,400** | **460,800** | **24.3 h** | **0.7344** |
| a23 | 96 | 9,600 | 921,600 | 26.9 h | 0.8398 |

**AT EQUAL SIMULATIONS THE LOW-SEARCH ARM WINS, decisively: +0.1523, p=0.0006**, significant at the corrected
alpha 0.0167. Three times the games at a third of the search beats the deep-search arm on the same simulation
budget. That is the pre-registered result and it stands.

**But the same rows show the verdict inverting under wall-clock.** b35 needed **24.3 h** to reach 0.7344; a23
reached **0.8398 in 26.9 h** — comparable time, clearly stronger. Arm A converts simulations to wall-clock at
32.2k sims/h, arm B at only 19.0k: **B's simulations are 1.70x more expensive in real time.**

The mechanism is exact and is the §C.26 correction compounding. At fixed simulations, arm B buys its advantage
by playing 3x as many GAMES, and a large part of a game's cost does not scale with the arm's own search — the
league opponents are pinned at 60/128/200 sims. So arm B pays roughly 3x the league bill for the same nominal
compute. Playing more games is not free, and "simulations" is blind to the part of the bill that games incur.

**Status of each claim, kept separate on purpose.** The sims-matched comparison is pre-registered, paired,
ledger-drawn and causal: at equal simulations, lower search trains better. The wall-clock observation is
DESCRIPTIVE only — a23 vs b35 differ in sims AND games AND simulations at once, so the ledger rightly refuses
to draw it and no cause may be attributed. It is reported as a cost fact from the runs' own metrics, not
promoted to a finding, and it is emphatically not a result I went looking for after seeing the outcome: §C.26's
correction predicted this shape three days before this read landed.

**What it means for the north star.** "Compute-efficient" needs its unit named. Under simulations — the unit
that is reproducible and machine-independent — fewer sims and more games is the better recipe. Under wall-clock
on this machine it is not, and the gap is caused by a recipe detail (fixed-cost league opponents), not by
anything intrinsic to search depth. The experiment that would make both units agree is **32 sims with the
league scaled proportionally** (60/128/200 → 20/43/67), which is pre-registered here as the natural successor
and NOT run now: the training path is frozen until arm B's reads complete, and changing the curriculum
mid-experiment would forfeit the comparison in flight.

#### §C.30 — HYPOTHESIS REGISTER: claims become falsifiable objects the runs judge (2026-09-17)

Every finding in this track so far lived as PROSE — a plan section, a memory file. Prose cannot be wrong in a
way the system notices: nothing linked a claim to the run that would test it, nothing stopped a claim being
written up after the result was in, and nothing forced a claim to say in advance which outcome would kill it.
`harness/hypotheses.py` closes that, enforcing three rules rather than relying on me to remember them:

- **H1 STATUS IS DERIVED, NEVER ASSERTED.** A hypothesis does not get to declare itself true. `status` is
  computed from linked ledger comparisons (supported / refuted / inconclusive / contested / untested) and
  `register()` has no `status` parameter at all — passing one is a TypeError.
- **H2 PRE-REGISTRATION IS CHECKED, NOT CLAIMED.** `pre_registered` is true only when EVERY piece of evidence
  was drawn after the claim was registered, verified against the ledger's own `drawn_at`. The difference
  between a prediction and a rationalisation is a timestamp. Status is unaffected — late evidence still counts;
  only the claim to foresight is withdrawn.
- **H3 COMPARISON AND UNIT DECLARED UP FRONT.** A hypothesis names both arms, the DIRECTION that would support
  it, and the UNIT. Without direction, any significant result can be spun as confirmation; without a unit the
  claim is not well-formed, because §C.29 measured the same recipe winning in simulations and losing in
  wall-clock. Linking a comparison between arms other than the declared pair is refused, so a claim cannot
  harvest whichever result happened to be significant.

Evidence accumulates rather than overwrites: the declared comparison drawn twice attaches twice, so a
replication is visible and a contradiction becomes `contested` instead of averaging away.

**Two real defects fell out of building it.** (1) The dedupe key was `(roots_id, drawn_at)`, which collapsed two
genuinely different comparisons drawn in the same second — fixed with microsecond stamps and a full-record key.
(2) Comparisons drawn before the direction fields existed crashed `link`. They are now handled by DERIVING
`diff`/`significant` on read from the entries' stored rates (nothing invented, and no re-drawing, which would
add a row to the family and silently tighten alpha for every other comparison on it), while `drawn_at` stays
None because it is genuinely unrecoverable — and a record of unknown age forfeits the pre-registration claim.
A comparison whose direction cannot be recovered is refused as evidence outright.

**The board, backfilled with this session's claims** (`scripts/hypotheses.py list`):

| status | id | claim | |
|---|---|---|---|
| SUPPORTED | h1 | the unchanged process learns a game with no solver | POST-HOC |
| SUPPORTED | h3 | at equal simulations, 32 sims beats 96 | POST-HOC |
| INCONCL. | h2 | budget past ~9600 games keeps buying strength | POST-HOC |
| untested | h4 | h3 at the next compute-matched point | **pre-registered, run in flight** |
| untested | h5 | h3 at full compute match (1.84M sims) | **pre-registered, run in flight** |
| untested | h6 | arm B keeps improving on its own curve | **pre-registered, run in flight** |

The three findings already in hand are marked POST-HOC by the system, correctly: they were discovered and
written up before any register existed. h4/h5/h6 are the first claims this project has made BEFORE seeing their
evidence, and arm B is now running to judge them. That distinction is the whole point, and from here it is
recorded automatically rather than by my remembering to be honest about it.

14 tests, 6/6 mutations caught (pre-registration always true, direction ignored, contested collapsed to
supported, links any comparison, unit not required, non-significant counts as support). Suite 507 passed.

#### §C.30a — the flow is now FORCED, and test-backed claims have a home (2026-09-17)

Standing instruction: *whenever something is found it MUST be recorded with tests/proof backing it; if a
capability is missing to support that, add it.* Auditing §C.30 against that found two gaps, both now closed.

**Gap 1 — the register sat BESIDE the path.** `measure_exploit.py` drew comparisons without asking whether any
hypothesis had predicted them, so a result could still land as prose. It now links every drawn comparison to
the hypotheses that DECLARED it and prints the verdict inline (`HYPOTHESIS h4 -> SUPPORTED (pre-registered)`);
a comparison no hypothesis predicted prints **UNREGISTERED COMPARISON** and says to register the claim it
answers. Same lesson as the §C.11 ledger: a guard that is optional gets bypassed by whoever is in a hurry.

**Gap 2 — most findings are not A/B comparisons.** "A resumed run keeps its optimizer state", "encode was
hardcoded to two planes", "an orphan checkpoint strands the ledger" are proved by REGRESSION TESTS, and the
register had nowhere to put them. A claim may now be TEST-BACKED: `proof` is a pytest node id, `verify()` runs
it, status derives from the result. The load-bearing guard is vacuity: **a proof that collects no tests is
REFUSED, not recorded** — a typo'd node id makes pytest exit 0 having run nothing, and green-by-vacuum is worse
than no proof because it reads as evidence. Re-verifying accumulates, so a later regression turns a supported
claim `contested` rather than silently flipping it.

**The board now holds 14 claims** (`scripts/hypotheses.py list`): 10 supported, 1 inconclusive, 3 untested. The
eight test-backed ones (t1-t8) were this session's prose findings, each now pinned to a named regression test
that was actually executed and passed. h1/h2/h3 remain marked POST-HOC — they were discovered before any
register existed, and the system says so rather than letting me imply otherwise. h4/h5/h6 are pre-registered
with arm B running to judge them.

25 tests on the register, 11/11 mutations caught across both rounds. Suite 516 passed.

#### §C.31 — the sleep guard was checking the process, not the property (2026-09-19)

Arm B lost roughly **20 hours of wall-clock to 169 sleep episodes (23.7 h asleep)** between batch 112 and 113,
while `caffeinate -is` was alive for the whole five days. `caffeinate -s` is documented as valid ONLY on AC
power; the machine was on battery, so the assertion was held and the system slept through it anyway.

The failure is not the flag, it is what I verified. I checked that the guard process was RUNNING — a proxy —
instead of checking that sleep was PREVENTED, which is the property. That is the fourth instance of this exact
shape (solve-depth guard, L1 label check, vacuous net-level pool test, `_save_atomic` wiring), and the first
where the cost was wall-clock rather than a wrong conclusion.

Diagnosis was evidence, not inference: the trainer's CPU time advanced 42 s in a 25 s sample (so it was working,
not wedged), while `pmset -g log` showed 169 "Entering Sleep state ... Using Batt" episodes since the last
checkpoint was written. A stalled run and a slept-through run look identical from the batch timestamps alone.

**`harness/machine.py`** (outside `TRAINING_MODULES`, so it cannot move an era): `power_source()`,
`sleep_guard_effective()` — which requires an assertion **and** AC power, and names the battery case explicitly
— and `slept_seconds_since()`, which reads the system's own sleep log so a paused run can be told from a stuck
one by evidence. 6 tests, 3/3 mutations caught, including the one that IS the bug ("battery ignored").
`scripts/run_scaled.sh` now prints the verdict before launching, so the next long run says up front whether it
is actually protected. Registered as **t9** in the hypothesis register, backed by the battery test.

**Nothing was lost** — the process survived every sleep, checkpoints are intact, and no resume was even needed.
Arm B sits at 113/144 with 76.2 h of real compute done. The machine is now on AC and the guard reports
effective.


#### §C.32 — MUTATION TESTING BECOMES A TOOL, because run by hand it lied four ways (2026-09-20)

The standing rule "mutation-test the guards" was carried out by hand — rewrite the module with sed, run pytest,
read the count. In ten minutes it lied twice on a real guard. **M-A STALE BYTECODE:** a `.pyc` validates on
`(mtime, size)`, so restoring `harness/ledger.py` to bytes that had just scored **46 passed** re-scored **2
failed**; same bytes, cache cleared, 46 passed. Either direction is fatal. **M-D KILLED IS A PROXY:** the
mutation meant to move a guard *after* the write instead went inert (it re-read the row it had just written), so
both tests died on "DID NOT RAISE" and the property under test was never exercised. Two more lie in wait: an
UNAPPLIED mutation (typo'd `old` text) reads as SURVIVED, and a RED baseline makes every mutation look killed.

`harness/mutation.py` (+ `scripts/mutate.py`, chat-reachable) refuses all four: it clears bytecode, requires each
edit's `old` to occur exactly once, refuses a red or empty baseline, and reports `killed_by` so kills are
checkable one-to-one. **Multi-edit mutations** express the highest-value class — relocating a guard so it still
runs but too late — which a single substitution cannot. Validated by running the runner against itself: 9
mutations, all killed, each by exactly the one test asserting its property. Registered t10–t13, t15, t17; L4
(name-collision that would silently overwrite a family's outcomes) fixed the same day. This is the recurring
"guard must check the PROPERTY, not a proxy" lesson turned on mutation testing itself.

#### §C.33 — THE EFFICIENCY VERDICT REVERSED, AND AN ADVERSARIAL AUDIT SCOPED IT (2026-09-20 → 09-22)

The pre-registered headline read h5 (b143 vs a47, compute-matched at 1,843,200 sims) came back **-0.0625,
INCONCLUSIVE** — the opposite sign to §C.29. h6 (arm B's own budget curve) was NULL. The whole §C.29 story was
an undertrained control: a11 sat below the ~9,600-game frontier, so h3 measured "more games beats an unconverged
net", not search depth. h8 replicated the reversal on a fresh root family: **+0.0342, p=0.0152**.

Rather than bank it, the h8 conclusion was handed to a 4-lens adversarial audit (measurement asymmetry, budget
confound, statistics, implementation), each objection then verified against the actual files. **8 of 22 confirmed.**
The sharpest was found by an agent that RAN a counter-experiment: every one of the 29 ledger entries was drawn at
**96 sims — exactly arm A's training budget and 3x arm B's**. The ledger refused mixed provenance, unmatched
budgets and drifted code, but never recorded the budget the arms were MEASURED at, so it was blind to this. Fixed
(L5/L6): entries carry `train_sims`/`deploy_sims`; `compare` warns "HOME BUDGET", refuses arms measured at
different budgets, and PERSISTS every caveat onto the stored record so a verdict never outlives its reasons to
doubt (t14, t17). A `_status` precedence bug — significance tested before the null band, so a significant effect
smaller than its own declared band read `supported` — was fixed with a NEGLIGIBLE status (t15). The "95.8 h" I
quoted for arm B three times was also wrong: it was **142.69 h wall / 127.16 h CPU** across two segments, the gap
being 5.79 h of sleep; `harness.machine.run_cost` now sums segments, quotes CPU, and flags any run whose CPU/wall
ratio falls under its thread count (t16). Corrected cost ratio: **1.88x**, matching the audit's independent number.

#### §C.34 — THE DEPLOYMENT SWEEP, POOLED: real, small, and no home-ground effect (2026-09-22)

The pair was read at all three deployment budgets on the SAME 1024 roots:

| deployment | a47 (96-trained) | b143 (32-trained) | diff | p |
|---|---|---|---|---|
| 32 sims (B home) | 0.8311 | 0.8154 | +0.0156 | 0.38 — NULL |
| 64 sims (neutral) | 0.8779 | 0.8340 | +0.0439 | 0.0042 |
| 96 sims (A home) | 0.9111 | 0.8770 | +0.0342 | 0.0152 |

The home-ground objection is **refuted**: if it were home advantage, 96 would be the maximum — the NEUTRAL 64 is.
And "significant at 64/96, NULL at 32" must NOT be read as budget-dependence — that is the significant/non-
significant fallacy (I was one sentence from it). All three pairwise diff-in-diffs on shared roots are null,
**max |z| = 1.30**, so the honest summary is ONE pooled effect: **+0.0312, SE 0.0093, z=3.37, p=0.0007, 95% CI
[+0.0131, +0.0494]** (`harness.measurement.pooled_paired_effect`, t18). Two caveats travel with it: the CI's lower
bound is *below* the 0.03 band the claims themselves declared as absent, and the SE prices ROOT noise only — with
n=1 training run per arm the dominant term is unpriced. **VERDICT: deeper training search wins by ~0.031 across
every deployment budget, small and within confounds.** Two confounds remain, both needing fresh training runs the
user has judged not worth the compute: the league pinned in absolute sims, and n=1 seed against a requirement of 7.
The measurement question is closed; the training-path freeze can lift.


#### §C.35 — FREEZE LIFTED, and the augmenter generalized to 2D symmetries (2026-09-22)

With the measurement question closed (§C.34), the training-path freeze — held only to protect the comparison in
flight — is lifted, and the code-forward queue resumes. First item: the symmetry augmenter, which turned out to
hide a **latent bug**, not just a missing feature.

`augment_examples` did `x[..., perm]`, reindexing the board's WIDTH axis with the action permutation. That is
correct only when `num_actions == board width` — the Connect-4 coincidence (actions ARE columns). tictactoe's
`symmetries()` returned 8 valid 9-cell dihedral perms; the augmenter indexed a size-3 axis with cell indices up
to 8 and threw **IndexError**. othello and checkers omitted `symmetries()` entirely for exactly this reason
(a comment in `test_othello` recorded the limitation). So "checkers `symmetries()`" and "augmenter rewrite" were
one task with a failing anchor case already in the tree.

The contract is now a **`(cell_perm, action_perm)` pair** per symmetry: `cell_perm` (rows*cols) permutes the
FLATTENED board, `action_perm` (num_actions) the policy; aux ownership transforms by `cell_perm`, the aux reply
by `action_perm`. A 2D isometry is now expressible. **Connect-4 is byte-identical** (guarded by a reproducibility
test), so its runs' training data does not move. Registered t19; 6/6 mutations killed, including the one that IS
the original bug (board indexed by the action perm). This UNBLOCKS othello's 8-fold dihedral and checkers'
symmetries as deferred units — it does not add them yet, because a checkers symmetry needs its own geometry work
(directional men + the dark-square parity make the valid group non-obvious, so it will be MEASURED by a verified
finder, not asserted). `neural.py` is a training module, so this **moves the era 3f2bacd402c7 → 1bab977cec75**;
acceptable because no comparison is outstanding and Connect-4/Othello behaviour is unchanged.

**Queue after this:** checkers `symmetries()` (verified finder) → nine men's morris (adjacency primitives) →
cross-game matrix runner. All no-compute capability work; a third-game TRAINING run is deferred by user decision.


#### §C.36 — VERIFIED SYMMETRY FINDER, and it corrected my own hand-analysis (2026-09-22)

The augmenter (§C.35) unblocked 2D symmetries; this fills them in — by PROOF, not assertion. A false symmetry
teaches the net that two positions are equivalent when they are not, so `harness/symmetry.py` enumerates the
board's shape-preserving isometries and keeps only those that COMMUTE with the game's own dynamics:
`transform(step(s, a)) == step(transform(s), action_perm[a])` for every legal `a` over 200 random positions,
with the side to move preserved (commutation + fixed side-to-move IS value-invariance). Games supply two small
encoding-aware hooks (`transform_state`, `transform_action`); the enumeration and verification are generic.

Results: connect4 `{id, flip_h}` (byte-identical to the old hand-written mirror), tictactoe the full dihedral
`D4 = 8`, and **checkers `{id, flip_h}`**. That last one CORRECTS a hand-analysis I had written into this very
session: I argued checkers was probably identity-only because a left-right mirror sends dark squares to light.
The finder disagreed and it is right — `flip_h` flips only the COLUMN component, so the men's forward direction
survives, and because the encoding uses all 64 cells the mirror is an isomorphic copy on the light sublattice;
`rot180`/`flip_v` reverse the forward direction and the diagonal maps rotate it sideways, all refused by the
verifier. This is the single cleanest instance of "measure, don't assert" on the whole track: the tool caught a
wrong assertion the moment it was made. Registered t20; 6/6 mutations killed, each verifier check isolated by a
fake game that lies one way (bad bijection, flipped side-to-move, wrong-successor bijection). `games/` and
`symmetry.py` are NOT training modules, so the era stays 1bab977cec75.

**Queue after this:** nine men's morris (adjacency primitives — its own encoding test) → cross-game matrix
runner. Othello's D4 is now a two-line addition (its transform hooks) whenever wanted. No-compute throughout.


#### §C.37 — NINE MEN'S MORRIS: the process transfers to a NON-GRID, three-phase game (2026-09-22)

The north-star claim is that the UNCHANGED process learns a new game. Every game so far was a grid with one move
type. Nine Men's Morris is the real stress test: a 24-point GRAPH board (three concentric squares joined by
spokes, no diagonals), THREE phases (placing / moving / flying-when-down-to-3), a mill->remove SUB-TURN, and a
symmetry group (D4 x inner/outer ring-swap = 16) that is not a grid dihedral. It was designed by a 4-agent panel
+ synthesis before a line was written.

It went in as ONE new file (`games/ninemensmorris.py`) plus a 2-line registry wiring and a small GENERIC
extension to the symmetry finder — **zero edits to any fingerprinted training module** (verified: othello's era
stays 1bab977cec75). Every harness assumption it breaks is absorbed by DECLARATION:
- **Non-grid board** -> embed the 24 points on a 7x7 grid at their natural coordinates and declare the 25
  non-points dead via `valid_mask` (checkers' masked-pooling path). PROVEN in-system: the 7x7 grid's D4 permutes
  the 24-point set onto itself, so the verified symmetry finder recovers the board's 8 dihedral symmetries with
  no change; the game supplies the 8 ring-swap candidates too, and `find_symmetries` verifies all **16** against
  the game's own dynamics (a construction bug would yield fewer, never a false symmetry).
- **>2 non-derivable features** (phase, men-in-hand, pending-removal, flying) -> `input_planes=8`, folded into
  observation PLANES because `encode` discards anything past `planes*h*w` (the one silent-corruption trap).
- **Sub-turn** -> a `pending_removal` bool with `to_move` frozen through the removal (checkers' `jumping`).
- **Three terminal kinds + draw** -> `_settle` + a bounded `idle` counter (IDLE_LIMIT=100) in `state_key`.
- **No solver** -> implements Game (not SolvableGame), like othello; the paired-exploitability metric runs with
  the game-agnostic MctsAgent refuter (integration-tested to completion).

Geometry is GENERATED from `point = ring*8 + pos`, guarded by a transcription test (32 edges, degree histogram
{2:12,3:8,4:4}, 16 mills, every point in exactly two, mill points collinear on the grid). 38 direct tests cover
all three phases, the sub-turn, capture priority (both branches), attrition/blockade/draw, encoding, the action
round-trip, the 16 symmetries, and the wrong-candidate rejection. Rules guards mutation-tested. The generic win:
`harness/symmetry.py` gained `iso_from_cell_map` + an `isometries=` override so a game can offer board
automorphisms the grid enumerator cannot — still proven, never trusted.

**Queue after this:** cross-game matrix runner (run the same process across all five games and tabulate);
othello's D4 (a two-line hook addition). No-compute throughout; a third-game TRAINING run remains the user's call.


#### §C.38 — CROSS-GAME REGRESSION MATRIX: the payoff §C.20 was recorded for (2026-09-22)

With five games encoded and the unified encoding done, the matrix §C.20 was gated on is built: rows = a change,
columns = games, cells = a paired delta, and a DERIVED verdict — generalises / local / contested / null /
inconclusive / single_game — that says whether a lever transfers. `harness/matrix.py` (the tested capability) +
`scripts/cross_game_matrix.py` (a chat-reachable runner: `sweep` trains and measures, `assemble` reads a ledger).

It was built, then handed to a 4-dimension adversarial review that CONFIRMED 16 findings; the material ones were
real and are fixed:
- **The generalisation bar was over-corrected.** "Every game moved" is a CONJUNCTION — an intersection-union
  test at alpha (Berger's min-test; the reason TOST uses alpha not alpha/2), NOT Bonferroni alpha/N. The original
  applied alpha/N to both, so a genuinely transferable lever read "inconclusive" — the opposite of the truth.
  Now the union question (did ANY game move -> local/contested) keeps alpha/N; the conjunction uses alpha (t22).
- **"null" was read from the point estimate**, the absence-of-evidence fallacy the module rails against. Now a
  row is `null` only when every cell's CI lies inside its null band (a TOST equivalence); a small diff with a
  wide CI is `inconclusive` (t22).
- **The FATAL one: a single training run per arm** measured opening variance for one net vs one net — the exact
  single-net confound §C.20 exists to kill (§C.19). Now the unit of replication is the TRAINING RUN:
  `cell_from_seed_deltas` tests the per-seed paired deltas between seeds with a t-test (new, table-validated
  `student_t_two_sided_p`/`t_critical` in measurement.py — the codebase had chi-square but no t). One seed can
  never be significant and carries a standing caveat (t23).
- Plus: N==1 reads `single_game` not `generalises`; `assemble` is a pure read (`Ledger.compare(record=False)`, so
  viewing the matrix never tightens the family alpha); exploitability `roots_id` now carries the game name (so
  the cross-family guard and L4's name-collision guard work across games); per-arm eval operator; a saturation
  caveat; fail-fast game resolution and per-game error isolation. Caveats propagate into every cell (L6).

A smoke sweep (augment lever, 2 seeds, connect4+tictactoe, tiny nets) runs end-to-end and returns INCONCLUSIVE
with every cell caveated SMOKE SCALE — the honest read at that power. Nothing here touches a training module, so
the era stays 1bab977cec75. This is also the first labelled data shape for the §C.6 meta-selector. The matrix is
ready for a real sweep whenever the compute is spent.


#### §C.39 — OTHELLO's DIHEDRAL GROUP, and all five games now carry verified symmetries (2026-09-22)

The last deferred symmetry hook: othello's D4. `transform_state` relabels the 64-cell board, `transform_action`
maps a cell to its image and PASS->PASS; `find_symmetries` verifies all 8 dihedral against othello's own
dynamics (the flank-capture rule is direction-agnostic, so every rotation/reflection of a position preserves
value). Registered t24. The full table now:

| game | verified symmetries |
|---|---|
| connect4 | {id, flip_h} (2) |
| tictactoe | full D4 (8) |
| checkers | {id, flip_h} (2) |
| othello | full D4 (8) |
| nine men's morris | D4 x ring-swap (16) |

This MOVES othello's own era (1bab977cec75 -> f180acfa235f) because games/othello.py folds into its fingerprint
AND othello training now augments 8x where it was identity-only — a real, intended improvement. The shared-module
canary `training_fingerprint(None) = 5b2e23684c22` is unchanged, so no training-path drift; only othello's own era
moved, exactly as the per-game fingerprint is designed to isolate. The queue §C.20 recorded is now fully cleared:
augmenter (§C.35), verified finder (§C.36), the fifth game (§C.37), the matrix (§C.38), and every game's symmetries.


#### §C.40 — THE FIRST REAL CROSS-GAME SWEEP: augment is consistent but underpowered (2026-09-22)

The matrix's first real run (the lever = symmetry AUGMENTATION on/off, 5 training seeds, all five games, paired
exploitability vs a UCT-48 refuter):

| game | verified symmetry | augment Δ | p | 95% CI |
|---|---|---|---|---|
| othello | 8x | +0.069 | 0.18 | [-0.049, +0.186] |
| tictactoe | 8x | +0.050 | **0.016** | [+0.015, +0.085] |
| connect4 | 2x | +0.044 | 0.094 | [-0.012, +0.099] |
| checkers | 2x | +0.013 | 0.51 | [-0.036, +0.061] |
| ninemensmorris | 16x | +0.009 | 0.30 | [-0.013, +0.031] |

**Verdict: INCONCLUSIVE** (0/5 clear the union bar alpha/5=0.01; only tictactoe clears the per-cell IUT bar
alpha=0.05, so not `generalises`). The DIRECTION is unanimous — augment helped all five games — and among the
games the metric can measure (excluding morris), the effect tracks symmetry richness: the 8x games move most,
the 2x games least. But at 5 seeds the +0.01-0.07 effects are underpowered (the review's `required_seeds ~= 7`
was right), so the matrix correctly refuses to call it generalisation.

This is the instrument working as the review forced it to: the naive "5/5 positive -> generalises!" overclaim is
exactly what the IUT-at-alpha + between-seed-t-test rigor prevents. The honest read is a consistent positive
direction, not yet statistically resolved.

**Dogfooding find:** morris was NOT flagged saturated though it is clearly ceiling-bound (all rates 0.97-1.0 —
both arms almost never lose to the bounded refuter in a draw-heavy game). The saturation threshold (`lo>0.98`)
missed it by 0.011; `is_saturated` moved into the tested capability with a 0.95 threshold and a mutation-tested
guard. Morris's number was honest anyway (tiny near-null effect), just unlabelled.

**Pre-registered follow-up** (when compute is spent): rerun at >=7-10 training seeds to resolve the ~0.05 effect
on the 8x games; for morris, use a STRONGER refuter (or a win-rate metric, not not-losing) so the cell can
discriminate. No training-module was touched; othello's era is f180acfa235f, the shared canary 5b2e23684c22.


#### §C.41 — OPTIMALITY MEASURED: the SOLVED gate, and coverage as the north-star yardstick (2026-09-23)

Prompted by a scrutinous audit (the track had drifted to RELATIVE transfer/efficiency on Othello and never
certified ABSOLUTE optimality on the calibration game), two things were done.

**(1) The SOLVED gate was actually run on the best trained net** (ab302_gpool_s0 ckpt_16, opening_value 0.979).
The pure from-empty-board gate (net P1 vs the EXACT solver defending from move 1) is IMPRACTICAL — the solver
hits the opening wall (minutes per defensive move), which is exactly why the plan marked it opt-in/SLOW and it
had never been run. The feasible, more diagnostic read (forced-win conversion vs EXACT defence by distance from
the opening, sims=64):

| position | proven-win conversion |
|---|---|
| 34 stones in (deep endgame) | 0.917 |
| 30 stones in | 0.958 |
| 26 stones in | 0.958 |
| 22 stones in (midgame) | 0.917 |

plus state-space coverage 1.000 on sampled >=30-stone states. **Honest verdict: the net is near-optimal in the
ENDGAME (coverage 1.0 deep), but demonstrably NOT optimal at full-game play — it throws away ~4-8% of PROVEN
wins whenever a longer correct sequence is required — and the OPENING (the real gap) stays unmeasured because
solving there is the wall.** This is LESS damning than the "loses ~30% as P1" proxy read (that was confounded by
random lost openings and a depth-8 proxy), but the plan's own SOLVED bar (100% from the opening vs the exact
solver) is still not met, and line 691's "GOAL MET — PROVEN near-optimal Connect-4 model" refers to the BOOKAGENT
(the solver crutch the north star excludes), NOT a trained net.

**(2) OPTIMALITY-AS-COVERAGE (harness/coverage.py, t25) — a measurable definition of "how close to perfect
play", per user direction.** Distance-to-optimal = the SYMMETRY-REDUCED fraction of reachable states where the
model's move is in the solver's optimal set (`state_coverage`), with a per-ply breakdown that localizes the
failure, plus the `decided_frontier` (the share of states from which a non-loss is PROVEN — the subtree beyond a
proven win needs no search, the pruning the user asked to measure). EXACT for enumerable games (tictactoe,
collapsed by the verified §C.36 symmetry group: optimal agent 1.0, random <1.0), SAMPLED for large ones.

**NORTH-STAR ADDENDUM (research path).** For solver-free games (othello, and the chess-scale target) there is no
optimal set to grade against. The path to make "how close to perfect play" MEASURABLE there: replace the solver
with a STRONG REFERENCE (a much deeper search, or a learned approx-exploitability adversary), and CALIBRATE that
proxy against the exact solver on connect4/tictactoe so its false-optimal rate is known and published, before
trusting it off-solver. Optimality then becomes a coverage number on any game — exact where a solver exists,
bounded where it does not. This is added to the north star as an explicit path to build, not a solved problem.
harness/coverage.py + measurement.py are not training modules; era unchanged.

**(3) END-TO-END DEMONSTRATION + THE CEILING, MEASURED (2026-09-23).** The scorecard was run through the whole
loop — generic process → trained model → certified distance-to-optimal — on the one game small enough to
enumerate EXACTLY (tictactoe, 627 canonical states, `complete=True`):

| agent | coverage | decided-kept | weakest ply |
|---|---|---|---|
| optimal (solver) | 1.0000 | 1.000 | — |
| random | 0.6029 | 0.487 | ply 1 (0.33) |
| trained ×2 | 0.9904 | 0.998 | ply 2 (0.917) |
| trained ×6 | 0.9904 | 1.000 | ply 2 (0.917) |
| trained ×14 | 0.9841 | 0.993 | ply 2 (0.917) |

**Even on tic-tac-toe the process plateaus at ~0.99 and never reaches certified-perfect — and MORE training does
not close it, it slightly REGRESSES (the model drifts; opening_value +0.49→+0.01). The miss is LOCALIZED and
PERSISTENT: ply 2 stuck at 0.917 regardless of budget** — a specific ~8% of midgame positions misplayed no matter
how long it trains. This is exactly why the proxy metrics misled: `opening_value` looked fine, coverage says
"99%, not 100%, stuck at ply 2".

The matching Connect-4 test (§C.41 step 3): a PURE solver-free run (14 iters, opening-diverse, NO endgame-solver
crutch) converts proven midgame wins at 0.71–0.88 — at/BELOW the endgame-TAUGHT ab302 baseline (~0.92). So (a) the
exact-endgame teaching the strong runs used was doing real work — the pure solver-free process is materially
further from optimal; (b) more training + opening diversity did NOT close the gap. Both games show the SAME thing:
this AlphaZero recipe gets close to optimal and stops, with a localized midgame blind spot. That is now a MEASURED
claim, not a proxy. (Caveats: n=1 seed each, tictactoe drift and connect4 solver-free-vs-taught want matched
≥7-seed runs to firm up — but "more training closes it" is refuted, and the ceiling is real.)


#### §C.42 — FORWARD: the north star meets a measured ceiling — the plan from here (2026-09-23)

The instrument work is done; the honest picture it produced reframes the whole effort. **The generic solver-free
process does not reach certified-optimal play on ANY game — not even tic-tac-toe — and the failure is a specific,
persistent MIDGAME blind spot that more training does not fix (§C.41).** Capacity is not the lever (302K ≈ 1.79M,
refuted §C.15), compute-to-convergence is not the lever (both games plateau then drift), and the strong Connect-4
numbers leaned on a solver crutch the north star excludes. So the forward question is no longer "train more" — it
is **why does this recipe stop ~1–8% short in the midgame, and is that fixable or intrinsic?** — and for the first
time it is directly measurable via per-ply coverage.

**The central open question (now measurable):** is the midgame ceiling (a) a fixable property of the recipe — the
usual suspects are train-time search depth (the net is asked to imitate a shallow search), value/policy target
quality on midgame states, or a value↔policy conflict there — or (b) an intrinsic ceiling of solver-free
AlphaZero at this scale? The per-ply coverage breakdown + `decided_frontier` localize the failing states exactly,
so this is an experiment now, not a guess.

**Prioritized forward work** (measurement is cheap and unblocks the rest; every A/B is coverage-scored, seed-
replicated, and read through the cross-game matrix + the register):
1. ~~**Localize the ceiling.**~~ **DONE 2026-09-23 → §C.43.** Systematic (h11), data-starved (h13), NOT
   off-manifold (h16); the net-vs-budget read is confounded on tic-tac-toe (h15). Outcome re-points item 3.
2. **Matched multi-seed coverage A/Bs** to firm up the two n=1 results: pure solver-free vs endgame-taught
   (Connect-4), and tictactoe drift, at ≥7 seeds under identical conditions — resolves the ~1–8% against the
   seed floor (~0.036).
3. ~~**Ablate the suspected causes — self-play diversity first.**~~ **DONE 2026-09-23 → §C.44. The first fix
   FAILED — and informatively.** The pre-registered A/B (mixed-openings self-play vs baseline, fresh seeds 11-20)
   REPLICATED the blind spot (h18) and did not fix it at its bar. **CORRECTED by §C.45:** at n=10 a §C.44-sized
   effect had ~8% power, so "not the cause" (h22) was an overstatement; the bounded reading (h23, supersedes h22) is
   that exposure raised visits 4.5× yet repairs at most ~35% of the blind spot (95% upper bound) and does not
   measurably repair the prior. Next cause: **label quality** → §C.45.
4. ~~**§C.45 — does LABEL QUALITY cause the blind spot?**~~ **DONE 2026-09-23 → §C.45.** Partially: near-exact
   labels at matched exposure repair ~half the target shortfall through the prior (h26, h28); deeper self-play as
   the delivery route FAILS (h27, h29, h30). No seed reaches NET-level coverage 1.0 yet.
4b. ~~**§C.46**~~ **DONE 2026-09-24 → §C.46.** One-ply siblings cut raw-policy failures ~10× (E1/E2) but mostly
   in-sample on a 627-state game (h38); architecture inconclusive (A1); exact labels are not a ceiling (h40); the
   coverage metric was canonical-image only — strict all-orientation perfection 0/20 (h39). Floor test not met.
4c. ~~**§C.47**~~ **DONE 2026-09-25 → §C.47.**
   - Floor MET under the pre-registered symmetry-averaged policy (h41: 17/20, rate ≥ 0.66). It holds only at
     averaging order ≥ 4 (order 2 gives 10/20) and with a near-oracle labeller.
   - Connect-4 Stage 0: h43 NOT_RUN (thin cells), then h44 GO on fresh nets. There is an exposure gap E = 0.043 on a
     0.39 base and a label headroom L = 0.084, but the labeller is weak (0.65) and siblings reach under 10% of the
     first errors that decide conversion.
   - Evidence left git; the register pins its judges.
4d. **Stage 1 (Connect-4 siblings), next.** Redesign per the §C.47 requirements (randomised-holdout primary
   endpoint, compute-matched control, label-quality competitor, pre-solved evaluation bank), with design review
   before any compute. Estimated 60-70 h of measurement unless the evaluation bank lands first.
5. **Build the calibrated solver-free reference** (§C.41 addendum) so the ceiling can be tracked on Othello and
   the chess-scale target — the only way "how close to perfect play" becomes a number where no solver exists.
6. **Promote the coverage scorecard to a first-class per-checkpoint metric** (as a post-hoc tool over checkpoints,
   OUTSIDE the fingerprinted training path per §C.17a rule 2), retiring `opening_value`/late-corpus proxies the
   §C.24 audit and §C.41 both showed mislead.

**Measurable success criteria (the bar, restated in coverage terms):** (i) certified coverage → 1.0 on tic-tac-toe
— the floor test: a process that cannot solve the smallest game cannot be called near-optimal-generic — **stated
at the NET's level** (raw-policy coverage, and coverage at a fixed small budget, each read against the search-alone
control): §C.43 showed an UNTRAINED net at 800 sims already covers the failing states, so a search-budget 1.0 on
tic-tac-toe is brute force, not learning; (ii)
Connect-4 solver-free midgame proven-win conversion → ≥0.99; (iii) a calibrated reference whose false-optimal rate
on solvable games is published, so Othello-and-beyond optimality is a bounded coverage number. **Explicitly OFF
the table:** more parameters, more raw compute without a localized cause, and any proxy (loss, opening_value,
late-corpus move-match) standing in for coverage.

**North-star restatement.** The literal "certified near-optimal, generic, solver-free" is partly self-contradictory
— certification needs a solver, which only 2 of 5 games have — so the working north star is: *drive certified
coverage → 1.0 where a solver exists (tic-tac-toe, then Connect-4), and elsewhere drive a CALIBRATED coverage
bound as high as it will go, on the unchanged process.* Perfect play on the solvable games is the falsifiable
milestone the whole program now aims at, because the coverage instrument finally makes "how close" a fact.


#### §C.43 — THE CEILING LOCALIZED: a systematic, data-starved blind spot (2026-09-23)

§C.42 item 1, run as a pre-registered experiment. 10 independently trained tic-tac-toe seeds (6 iters × 48
games, train sims 32, the §C.41 demo recipe), every misplayed state dumped (`coverage.coverage_failures`), four
claims registered BEFORE the run and judged by `harness/ceiling.py` from `examples/boardgames/evidence/
tictactoe_ceiling.json` (stored in the repo tree but NOT yet committed — corrected 2026-09-23; the register times each
claim against the file's own `started`). Family α=0.05,
split /4. Training fingerprint c4b351282e91 (unchanged: no training module was touched).

**A measurement defect found first (t26).** The §C.41 scorecard reused one search tree across all 627 states, so
later states were searched with the budget of every state before them — coverage depended on ORDER (same net:
0.9904 forward, 0.9968 reversed). Fixed at the root: `state_coverage` now REFUSES an act_fn that answers a state
differently when re-asked, and `per_state_act` builds a fresh agent + rng per state. The honest number for the
§C.41 demo net is 0.9872, not 0.9904.

| claim (pre-registered unless noted) | verdict | numbers |
|---|---|---|
| h11 the blind spot is SYSTEMATIC across seeds | **SUPPORTED** | 179 shared-failure pairs vs 4.9 expected (36×), p<1e-4; 69 failures over only 17 distinct states; one state missed by 10/10 seeds |
| h12 more deploy search fixes < half (net, not budget) | **REFUTED** → superseded by h15 | 400 sims fixes 67/69 |
| h13 failing states were STARVED of training data | **SUPPORTED** | 1.9 vs 10.4 mean final-buffer visits, same seed and ply, stratified permutation p<1e-4 |
| h14 failing states are enriched OFF the optimal-play manifold | **REFUTED** → superseded by h16 | 13/17 off vs an 80% base rate, p=0.76 |
| h15 (post-hoc) the deep-search check is CONFOUNDED | SUPPORTED | an UNTRAINED net at 400 sims fixes 72% of the same failures; at 800 sims, all 17 |
| h16 (post-hoc) no off-manifold enrichment | SUPPORTED | as h14 |
| h17 (post-hoc) tic-tac-toe coverage is mostly SEARCH | SUPPORTED | untrained net + 48 sims = 0.935; trained 0.982–0.994; raw policy 0.54 → 0.94–0.96 |

**What the blind spot is.** 67/69 failures: the net's PRIOR ranks the played move above every optimal one (the
value head does in 53/69). The recurring states are odd openings (a corner and an edge) where the win needs a
quiet forking move and the net plays the centre, its default. At 16 sims the trained prior does WORSE than an
untrained one on these states (6/17 vs 10/17 correct, one seed): the learned prior is not merely thin there, it is
confidently wrong. The states are not positions reached only after a blunder (h16). They are positions self-play
stopped visiting (h13): the state missed 10/10 had zero final-buffer visits in every seed. **Verdict on §C.42's
central question: the ceiling is a property of the RECIPE, not seed noise (a), and specifically of its data
distribution:** self-play concentrates on its own lines, the prior generalises "centre" to states it never sees,
and a small search cannot overturn a confident prior.

**Caveats.** Visits are counted in the FINAL replay buffer (cap 8000), not over all of training. The deep-search
check is uninformative on tic-tac-toe because the tree is small; it becomes informative only where search is far
from exhaustive (Connect-4), with the same control. n=1 recipe and 1 game: that this generalises is untested.

**Guards added** (all mutation-tested, every mutant killed): stateful act_fn refused (t26);
`blind_spot_concentration` judged over the FAILABLE universe (states where a wrong move exists), refuses a
single seed; `ceiling.localize_report` marks the net-vs-budget reading `attributable` only when a search-alone
control fails the same states; Wilson bounds, not point estimates, for every rate verdict; stratified (seed×ply)
visit comparison; data-timed pre-registration in the register (t27: h15–h17 had read "pre-registered" because a
test-backed claim was timed against its verify call). Suite 727 passed.

**Forward** (§C.42 item 3 re-pointed): the next experiment tests one localized cause. Self-play state diversity
vs baseline at ≥10 seeds, scored at the NET's level against the search-alone control, with these 17 states as
the pre-declared target set.



#### §C.44 — THE FIRST FIX FAILED, INFORMATIVELY: starvation is a correlate, not the cause (2026-09-23)

§C.42 item 3, run as a pre-registered A/B. §C.43 found the tic-tac-toe blind spot correlated with data-starvation
(h13: failing states had ~5× fewer self-play visits). The obvious fix is to make self-play visit them, via the
existing mixed-openings knob (`selfplay_opening_plies=2` + `opening_plies_zero_frac=0.5`). Baseline vs treatment,
FRESH seeds 11-20 (the target was localized from seeds 1-10, so evaluating on 1-10 would be regression to the
mean — `ab_report` refuses it), 17 recurring states as the pre-declared target, four claims at α/4, paired by
seed with an exact sign-flip test. h18-h21 registered before either arm trained; fingerprint c4b351282e91.

| claim | verdict | numbers |
|---|---|---|
| h18 the blind spot REPLICATES on fresh seeds | **SUPPORTED** | 58/60 baseline failures land in the 17-state target (its universe share is 3.9%), hypergeometric p≈0 |
| h19 mixed openings raise TARGET correctness | **REFUTED** | +0.053, sign-flip p=0.14 |
| h20 mixed openings raise RAW-POLICY coverage | **REFUTED** | +0.009, p=0.044 (> α/4=0.0125) |
| h21 mixed openings raise eval-budget coverage | **REFUTED** | +0.001, p=0.19 |
| h22 (post-hoc) starvation is NOT the cause | **SUPPORTED** | see below |

**The manipulation check is the finding (h22).** A refuted fix is only informative if the treatment did what it
claimed. It did: mixed openings raised mean target-state self-play visits from 36.8 to **166.4 per seed** (4.5×,
paired sign-flip p=0.002), cutting zero-visit (seed,state) pairs from 148/170 to 113/170. Correctness stayed flat
anyway. Pooled across both arms, a target state that WAS visited in self-play is misplayed at 0.30, one that was
NOT at 0.32 — indistinguishable. ~~So the net SEES these states and still plays the wrong move; §C.43's
data-starvation was a correlate, not the operative cause.~~ **CORRECTED (§C.45, h23 supersedes h22):** a
non-significant +0.053 at n=10 had ~8% power, so it cannot say "not the cause". Inverting the sign-flip test gives
what the data DO bound: target gain < +0.118 (95% one-sided), i.e. exposure repairs at most ~35% of the blind spot,
and raw-prior gain at the target < +0.044 (under 1 of 17 states). Exposure is not ruled out as a partial cause.

**Where this points.** The cause is upstream of exposure. The recurring misses are odd openings where a quiet
forking move wins and the net plays the centre; the failure is in the PRIOR (§C.43: 67/69). Raising exposure at a
32-sim self-play budget did not help — which implicates the TARGET the net imitates: a 32-sim search from these
fork openings may itself not find the fork, so self-play labels the state with the wrong move and more visits just
teach the wrong label harder. That is the next pre-registered A/B: **train-time self-play sims** (a deeper label
search), same 17-state target, same search-alone control. Value-target source / reanalyze only if that fails.

**Method note — a pre-registered fix failed and the system caught it honestly.** The register shows h19-h21 REFUTED
with timestamps proving the claims preceded the run; the manipulation check (baked into `ab_report`, mutation-
tested) is what turned "the fix didn't work" into "starvation isn't the cause" rather than "maybe the knob didn't
reach the states". `ab_report` refuses unpaired seeds, moved training code, a target chosen from the evaluated
seeds, and arms differing in more than the declared treatment; every rate verdict is an exact paired test at α/4.
Every new guard was mutation-tested (each mutant killed); the full suite passes green. No training module touched.


#### §C.45 — Does LABEL QUALITY cause the blind spot? (pre-registered; design rebuilt by adversarial review, 2026-09-23)

**Why this experiment.** §C.44 ruled out exposure as the whole cause. The net's PRIOR is what is wrong at the 17
target states (§C.43), and a prior is learned from self-play policy LABELS (the Gumbel completed-Q target
`softmax(log prior + (c_visit+maxN)·c_scale·q̂)`, c_scale 0.1). If the 32-sim label search misplays these fork
openings, the net imitates a wrong target however often it visits them.

**Pre-work and what it did NOT show.** `harness/targets.py` splits one search into prior / completed-Q / label
(`label_ok`, `prior_anchor` = search found it but the target kept the prior, `search_miss` = search never found
it). On the final nets of seeds 11-20: labels right 0.998 on other states but 0.597 on the target states; deeper
label search (200 sims) 0.949; a more Q-trusting target (c_scale 1.0) only 0.629. **The design review showed this
contrast is circular** — the target set IS where those same final nets fail, so re-searching them measures the
failure, not its cause — and it is recorded only as the hypothesis's motivation. A ladder of relabel budgets
(`evidence/tictactoe_labels_ladder.json`) found NO "transferable" budget S* (trained labels ≥ 0.85 while an
untrained net's search stays ≤ 0.65): at 96 sims trained 0.847 / search-alone 0.702; at 128, 0.898 / 0.765. **On
tic-tac-toe, good labels and brute-force search cannot be told apart by budget** — so the planned S* arm (A3) was
dropped by its own pre-stated rule, and any positive result here says "labels matter", not "a small budget fixes
them".

**Two adversarial reviews before any compute.** (1) Design review (4 critics + synthesis): the original 2×2 would
have (a) passed its manipulation check on the INSTRUMENT alone (an untrained net at 200 sims labels 0.81 right),
making "causal" a copy of the outcome; (b) tested deep-vs-base, which delivers almost no target labels (base
buffers hold 0-14 target positions per seed); (c) had ~8-48% power for plausible effects at n=10 and alpha/6;
(d) changed play, outcomes, game length and label sharpness together (train_sims is not a label-only treatment).
Rebuilt: a label-ONLY treatment (new `reanalyze_sims`: relabel the buffer at 200 sims while self-play stays at
32), exposure held high (mixed openings in every treatment arm), n=20, a DELIVERED-dose manipulation (the labels
the net actually trained on, recorded by wrapping `train_net`), gates instead of alpha-spending certain claims,
fixed-sequence alpha. (2) Implementation review (4 reviewers, every finding independently re-derived): 14
confirmed defects fixed before launch — most importantly the confidence bound landed a hair past exact ties
(0.5 = 3/6 …) so a clean fix would have read "moved" in ~4-11% of runs; plus relabeler noise parity without
Gumbel, target-set mismatch between arms and report, measurement-fingerprint stamping, a silent no-op flag, and
a dropped claim still emitted.

**Also corrected in passing (h23 supersedes h22):** §C.44's "starvation is not the cause" was an underpowered
null read as a refutation; the bounded claim is "exposure repairs at most ~35% of the blind spot".

**The pre-registered design** (h24-h27, registered before any registered arm trained; disclosures in the notes):
seeds 21-40; arms base / mixed / mixed_deep (train_sims 200) / mixed_R32 / mixed_R200 (reanalyze_frac 1.0 at 32 /
200 sims); `harness.ceiling.c45_report` emits only: **G1** gate replication (h24); **G2** delivery gate per
contrast (correct target labels trained on must rise; every seed trained on ≥1 target example; R200 delivered
share ≥ 0.85); **G3** migration gate (upper bound on outside-target failure rise ≤ 0.5); **chain A** fixed
sequence at α 0.04 — **A1** (h25) R200 vs R32 target correctness, **A2** (h26) R200 vs R32 raw prior at the
target; **B1** (h27) mixed_deep vs mixed target correctness at α 0.01. A null reads REFUTED only when the 95%
upper bound excludes half the control's shortfall; otherwise INCONCLUSIVE. Known, matched caveat: the reanalyze
path caps its buffer in raw states (×8 augmented) where the plain path caps augmented examples, so the R arms keep
more history than `mixed` — only R32 vs R200 is a claim, and they are matched.

**Infrastructure this added** (all mutation-tested, every mutant killed): register — `inconclusive_proof` (a
failed test-backed claim reads INCONCLUSIVE when its undecidability test passes; before, every failed proof read
REFUTED) and data-timed pre-registration from §C.43; `harness/ceiling.py` — exact meet-in-the-middle sign-flip
(n ≤ 30) inverted into one-sided confidence bounds, tie-robust bound comparisons, `ab_report` with explicit
`alpha_each`, `dose`/`labels` manipulations that refuse missing fields, migration guard, target-set and
measurement-fingerprint refusals, `c45_report`; `harness/targets.py` — `search_decomposition`, `target_error`,
`record_training_labels`; `harness/coverage.py` — `raw_encoding_lookup`, `label_dose`; `harness/neural.py` —
`reanalyze_sims` (byte-identical when unset or equal to sims, with or without Gumbel; refused without
reanalyze). **Training-fingerprint era change:** tic-tac-toe c4b351282e91 → e92d405fae26, shared canary
5b2e23684c22 → 845458df8455 (every §C.45 arm trains under the new era; no run was in flight). Suite 805 passed.

**RESULTS (2026-09-23; verified before recording).** All five arms × 20 seeds completed under one training era
(e92d405fae26) and one measurement stamp (71586a13e928); every run started after registration. A verification
pass (4 independent verifiers + a completeness critic) re-derived every number with its own code (no harness
import; exact 2^20 enumeration), retrained seeds of four arms bit-for-bit, and ran two exploratory control arms.

| pre-registered claim | verdict | numbers |
|---|---|---|
| h24 G1 replication (3rd time) | **SUPPORTED** | 119/125 base failures in the 17-state target (3.9% of states), p=4e-168 |
| h25 A1 200-sim relabelling raises target correctness | **SUPPORTED** → reframed by h28 | 0.706 → 0.859, +0.153, 19/20 seeds up, 0 down, p=1.9e-6, 95% lower bound +0.125; G2 delivered (correct share 0.387 → 0.958 on relabelled passes); G3 held (outside failures 0.25 → 0.30) |
| h26 A2 … through the PRIOR | **SUPPORTED** | raw prior correct at the target 0.438 → 0.712, +0.274, p=9.5e-7, lower bound +0.235 |
| h27 B1 deeper SELF-PLAY raises target correctness | **REFUTED** | 0.721 → 0.647, −0.074, 95% upper bound −0.024 (excludes any gain); migration gate FAILED (outside failures 0.20 → 1.35/seed) |

| arm | 48-sim coverage | raw-policy coverage | target correct | target prior | seeds at coverage 1.0 |
|---|---|---|---|---|---|
| base | 0.9900 | 0.9486 | 0.650 | 0.309 | 0 |
| mixed | 0.9921 | 0.9691 | 0.721 | 0.391 | 0 |
| mixed_R32 | 0.9916 | 0.9730 | 0.706 | 0.438 | 0 |
| **mixed_R200** | **0.9957** | **0.9827** | **0.859** | **0.712** | **2 / 20** |
| mixed_deep | 0.9883 | 0.9486 | 0.647 | 0.435 | 0 |

**What is established (and the bounded wording the register now carries).**
- **Label quality is a PARTIAL cause (h28, supersedes h25's headline).** With exposure matched, near-exact policy
  labels repair ~52% of the target shortfall (lower bound 42.5% — below the pre-registered SESOI of half) and ~49%
  of the prior shortfall, through the PRIOR (A2; the value head is unchanged, R200 vs R32 +0.009). The blind spot is
  halved, not removed: 48 of R200's 54 failures are still in the target (key 601 fails 13/20 seeds, 1608 15/20; at
  601 even the 200-sim relabeller is only 0.65 correct). The attribution is clean: pass-1 training is byte-identical
  across mixed/R32/R200 in 20/20 seeds, R32 reproduces plain Reanalyze byte-for-byte, exposure slightly FAVOURS R32.
- **Deeper self-play is the wrong way to buy better labels (h27, h29, h30).** B1 refutes the RECIPE, not labels: its
  delivery gate passed only through key 90 (without it the correct-label dose does not rise), it trained on 32%
  fewer target examples, and 200-sim self-play after two random plies NEVER produces off-manifold positions where O
  (to move) wins — zero in 20/20 seeds — and its decline sits on exactly those O-to-move states (p=3e-4). Hypothesis
  (not shown): its value head learns "O to move in an odd position is lost".
- **Milestone, stated at the level it holds:** 2 of 20 R200 seeds reach 48-sim coverage 1.0 — the first perfect
  coverage on this track — but ZERO seeds in any arm reach raw-policy (net-level) coverage 1.0 (R200 best 0.9952),
  so the §C.42 floor test is NOT met. At R200 the net-level gap now lies mostly OUTSIDE the 17-state target (5.95
  vs 4.90 raw-policy failures per seed), because the target was chosen from 48-sim eval failures.

**What is NOT established.** Transfer: 200 sims nearly exhausts 16 of the 17 target subtrees, an untrained net at
200 sims already labels 0.795 of them right, and no "transferable" budget S* existed — so this says "near-exact
labels work", not "a search budget Connect-4 can afford works"; there a relabel search leans on the value head,
which no arm fixed. "Label quality is THE cause" — correct labels × enough exposure is the surviving post-hoc
reading (neither alone sufficed: §C.44 exposure with wrong labels, B1 good labels at low exposure).

**Exploratory, pending replication (not register-grade: patched one-off runs, no fingerprints).** A verifier ran
two control arms on seeds 21-40: 200-sim relabels ONLY at the 17 target states (32 elsewhere) reproduced the entire
A1 effect (0.862 vs R200 0.859); 200-sim relabels everywhere EXCEPT the target recovered only +0.038. Correct labels
at the blind-spot states are sufficient for the repair — the obvious first claim to pre-register next.

**Process lessons turned into guards this section.** (1) A design review BEFORE compute caught a manipulation check
that passes on its instrument alone (t-level: `dose` reads the labels actually trained on). (2) An implementation
review caught a confidence bound landing past exact ties (t29). (3) The register could not say "inconclusive" for a
test-backed claim (t28). (4) The critic caught verification about to run without data timing — the §C.43 guard only
works if `attach-data` precedes `verify` — now GUARDED (t30): every new test-backed claim must declare its data
file (even one not produced yet, timed when verify first finds it) or `--reads-no-data`, so it cannot be forgotten.
Suite 816 passed. (5) G2's pooled-count
delivery gate could not see per-state delivery (h29) — the next design's gate must be per state.

**CORRECTION — the network every §C.41-§C.45 tic-tac-toe run actually trained (found by the §C.46 design review,
2026-09-23).** The scripts passed `net_arch={"channels": 32, "blocks": 3, "head_hidden": 32}` with no `residual`;
the legacy branch of `Connect4Net` ignores `blocks`/`head_hidden`, so every seed trained the 12,746-parameter
2-conv legacy net, not the 57,453-parameter residual net the config described. All results stand as measured, but
they are results about the LEGACY net. The §C.46 G0 capacity gate then showed why it matters: trained directly on
the EXACT policy for all 627 states at the recipe's step budget, the residual net reaches 0 raw-policy failures on
4/5 seeds while the legacy net reaches 0 on 0/5 (1/5 even at 3× the steps; key 601 fails in 8/15 legacy runs) — part
of the "ceiling" may be capacity/optimisation of the legacy net, not the recipe. Now GUARDED at the root:
`arch_for_game` and `validate_config` refuse residual-only settings without the residual tower; `localize_ceiling`
takes an explicit `--arch legacy|residual` and records `params_expected`, and the §C.46 report refuses a seed whose
built parameter count differs from it.


#### §C.46 — Labels × exposure × architecture at the NET level (pre-registered 2026-09-24)

**Why.** §C.45 left two routes to the §C.42 floor test (raw-policy coverage → 1.0): the residual gap sits half in
cells that never received a training label, and positions self-play never generates (h30). Siblings — every
one-move deviation from recorded self-play states — are the generic exposure mechanism. Then the design review
found something bigger: **every §C.41-§C.45 tic-tac-toe run trained the 12,746-parameter LEGACY net** (the recorded
`{32, 3, 32}` had no `residual`), and the offline capacity gate G0 showed that net cannot fit the exact policy at
the recipe's step budget (0/5 seeds at zero failures; the residual net 4/5). So §C.46 asks three questions at once:
does the right ARCHITECTURE move the ceiling (A1), does sibling EXPOSURE (E1/E2), and do the §C.45 LABEL results
replicate on fresh seeds and on the right net (chain R).

**Design history.** Sketch (6 arms incl. target-only relabel and exact-value arms) → design review (4 critics +
synthesis; `scratchpad/c46/spec.md`) → rebuilt. The review killed: a 17/17 per-state delivery gate that would fail
~98% of the time (key 4983 is reachable by siblings only at depth 2); a migration gate on raw-policy failures that
fails by noise alone; a SESOI that read real gains as refutations; sibling injection at iteration 0 (it destroyed
the pass-1 pairing that gives the test its power); the target-only and exact-value arms (solver-derived, no generic
decision follows). Implementation review (4 reviewers, each finding re-derived; 15 real, all fixed before launch):
D3's generalisation ratio compared arms over different key sets; D9 dropped children of non-failable visited
states; G0 reached the report as a bare boolean; the delivery gate could pass with siblings that never changed
training (now GE(vi), t32); a parameter check that compared a number with itself; a holdout without a salt; NaN
oracle labels; aux-head NaN values; old-evidence re-analysis broken by the new arch guard.

**Arms** (seeds 41-60; all: mixed openings 2/0.5, 32-sim self-play, reanalyze_frac 1.0, 6 iters × 48 games, eval
48-sim fresh tree): leg_R32, leg_R200 (legacy net); R32, R200 (residual); R32S, R200S (residual + siblings,
policy-only, value masked, steps-matched); Rx, RxS, RxS_H (DIAGNOSTIC: exact-solver labels; RxS_H withholds half
the sibling keys to measure generalisation).

**Claims** (`harness.ceiling.c46_report`; h31-h37 registered against data not yet produced): **E1** (primary,
α 0.03) R200 → R200S raw-policy failures over 627 states, gate GE (siblings on every relabelled pass, exposure
rises, exact one-ply closure re-derived from the rules, step ratio 0.75-1.33, sibling-label share ≥ 0.90, siblings
changed training); **E2** R32 → R32S (only if E1 supported); **A1** (α 0.01, gated by G0) leg_R200 → R200;
**chain R** (α 0.01, fixed sequence) L1/L2 legacy replication of §C.45 A1/A2, R1/R2 on the residual net.
Descriptives: D1 (does RxS reach raw-policy 1.0), D3 (generalisation to withheld keys), D6 (does an exact value
make a 32-sim label right — the transfer probe), D9 (where the remaining failures sit), M (milestone vs
search-alone, with exact intervals). Floor test registered: met iff a generic residual arm reaches raw-policy 1.0
in ≥ 15/20 seeds.

**Training-code changes** (era c4b351282e91 → … → **2600dc4f574a**; every step reproduced §C.45 seed 21
bit-for-bit with the new knobs off): masked value loss (`_value_loss`, bit-identical when nothing is masked —
equivalent mutant documented), `train_net(epoch_examples)` steps matching, `reanalyze_examples` refuses terminals,
`one_ply_siblings`, `train_alphazero(reanalyze_siblings, steps_matched, sibling_holdout, policy_target_fn)` with
refusals and per-iteration history, `arch_for_game`/`validate_config` refuse residual-only settings without the
tower (t31), `legacy_arch_as_built` for re-analysing old evidence. Measurement: `record_training_passes` (per-pass
dose over all 627 keys, weights hash, raw-policy failures), `exact_policy_target`, `scripts/capacity_gate.py` (G0).
Suite 982 passed; every new guard mutation-tested.

**RESULTS (2026-09-24; verified before recording).** Nine arms × 20 seeds under one era (2600dc4f574a), every run
started 5.5 min after h31-h37 were registered. Verification (4 verifiers + critic) re-derived every number with its
own code (exact 2^20 sign-flip), retrained 6 seeds bit-for-bit (the critic later all 20 R200S seeds).

| pre-registered claim | verdict | numbers (raw-policy failures per seed, 627 states, ONE canonical image each) |
|---|---|---|
| h31 E1 siblings, 200-sim labels (R200 → R200S) | **SUPPORTED** | 10.0 → 0.9, gain +9.1, 20/20 seeds, p=9.5e-7, 95% lower bound 6.9; GE held (labels 99.87% right, steps 0.975-1.027, siblings changed training in 20/20) |
| h32 E2 siblings, 32-sim labels (R32 → R32S) | **SUPPORTED** | 12.35 → 4.75, gain +7.6, p=1.9e-6, lower bound 6.17 |
| h33 A1 residual vs legacy net (leg_R200 → R200) | **INCONCLUSIVE** | 12.8 → 10.0, gain +2.8, p=0.031 (> α 0.01), bounds [0.42, 5.2] straddle SESOI 3.2 — the register's new inconclusive outcome, not a refutation |
| h34/h35 L1/L2 legacy relabelling (replication of §C.45) | **SUPPORTED** | target 0.726 → 0.835, prior 0.462 → 0.674 — direction and significance replicate; magnitudes smaller than §C.45 |
| h36/h37 R1/R2 residual relabelling | **SUPPORTED** | target 0.856 → 0.959, prior 0.697 → 0.868 |

**What E1 does and does NOT show (h38, post-hoc).** On a 627-state game, self-play plus one-ply siblings trains on
~89% of the failable positions. ~89% of the E1 gain is the net learning labels on positions R200S trained and R200
never did; the spillover to positions NEITHER arm trained is ~1.0 failure/seed — real (p≈0.001) but small. Only the
spillover part could transfer to Connect-4, where one-ply siblings of visited positions are a vanishing fraction of
the space. So E1 is a strong EXPOSURE result on tic-tac-toe, not yet a transferable mechanism; D3's withheld-sibling
arm puts generalisation to never-trained withheld positions at 0.34 of the direct effect (bootstrap CI [0.21, 0.45]).

**Exact labels are not a ceiling (h40, post-hoc).** Rx (solver labels uniform over the optimal set, no siblings) is
the WORST residual arm — 18.3 failures/seed vs R200's 10.0 and R32's 12.35 — while training MORE positions than R200:
it generalises to untrained neighbours ~2× worse (8.3% vs 4.1%). Label argmax accuracy is not what makes a label teach;
the soft search target carries a ranking among moves that the uniform oracle label does not (hypothesis). Once
siblings cover the ring, 200-sim labels do as well as solver labels (D4 R200S − RxS = +0.15 [−0.6, +0.9]).

**Where the residual sits (R200S).** 14 of its 18 remaining failures are late tactical positions exactly two plies
from self-play that received no training row on any pass; the other 4 are trained positions (mostly multi-optimal,
small logit margins). R200S is flat from pass 5 to 6 (0.9/seed) — more iterations will not close it. A scratch pilot
(verifier, not register-grade, on the evaluation seeds) with TWO-ply siblings reached 15/20 canonical-perfect seeds.

**THE METRIC WAS WRONG IN A WAY THAT MATTERS (h39, post-hoc).** Every coverage number since §C.41 scored the raw
policy on ONE canonical image per state. A net trained with augmentation is not exactly symmetric. Rebuilt
bit-for-bit (`scripts/orientation_evidence.py` refuses any net whose weights hash differs from the evidence), the
same nets score:

| perfect seeds (raw policy) | one canonical image | every raw position (4,520) | averaged over the 8 verified images |
|---|---|---|---|
| R200S | 9/20 | **0/20** | 16/20 |
| RxS (exact labels) | 12/20 | **0/20** | 17/20 |
| R32S | 2/20 | 1/20 | 2/20 |

E1/E2 stay valid (both arms of each comparison were scored the same way), but every "perfect" count and every
"distance to optimal" since §C.41 is canonical-image. The raw net misplays some rotation of a position it plays
right canonically (R200S: 3.85 failing positions per seed over all images vs 0.9 canonically). Averaging the policy
over the verified symmetry group removes almost all of it — and an averaged policy is EXACTLY equivariant, so for it
the canonical and all-orientation readings coincide. **Guarded now:** `coverage.orientation_failures` scores any
policy all three ways and refuses an isometry the game does not obey; `localize_ceiling` records it for every future
arm; `symmetry.verified_isometries` supplies the proven isometries.

**Floor test after §C.46.** NOT met under any reading as registered (the bar is ≥ 15/20 seeds under an operator
fixed in advance; §C.42 never fixed one). The symmetry-averaged reading reaching 16/20 was found AFTER the data — it
can motivate a pre-registration on fresh seeds, not satisfy one.

**Also corrected / guarded this section.** The report overwrote each E/A claim's arm name with a per-seed list
(fixed; the saved scratch report predates the fix); the §C.43 printout crashed on a perfect arm (fixed); the
register can now time a claim against SEVERAL data files, the earliest start binding (t33); `harness/symmetry.py`
(augmentation) and `harness/game.py` are training-path code missing from the training fingerprint — to be added at
the next era change (adding them now would move the era under §C.46's evidence).


#### §C.47 — The floor on fresh seeds, and the Connect-4 transfer go/no-go (2026-09-24)

**Leg 0 — housekeeping, done before any §C.47 data.**
- **Evidence leaves git.** 36 MB of pretty-printed JSON in `evidence/` (the §C.46 arms were 3.3 MB each, 76% of
  it the per-pass training record) is now gzip-compressed minified JSON (`*.json.gz`, 2.1 MB in total). The
  directory is gitignored except `evidence/manifest.json`. That manifest restores what leaving git lost, a record of
  what each file contains: `harness/evidence.py` refuses to read a file whose content hash differs from the one
  pinned there, and writes each file once, with any replacement naming the version it replaces.
  - A restored backup is therefore proved to be the data the verdicts were drawn from.
  - The register moved from the gitignored `checkpoints/scaled_runs/` to the tracked `hypotheses.json`.
    `Register.relocate_data` re-pointed h11-h40 at the compressed files, and refused unless each held exactly the
    old content.
  - A SKIPPED proof now refuses to verify. Pytest exits 0 on a skip, and the evidence proofs skip on a checkout
    without the data, so a missing file would otherwise have verified as a pass.
  - Every guard is mutation-tested.
- **Fingerprint.** `harness/game.py`, `harness/symmetry.py` and `harness/registry.py` joined `TRAINING_MODULES`,
  moving the tic-tac-toe era from 2600dc4f574a to 27933b3a3bba. No training code changed; §C.46's era re-derives
  under `TRAINING_MODULES_V4` at 4cfcfbe.
  - The list has now missed training-path code three times, so a test walks the import graph from every
    fingerprinted module, through `games/`. Everything it reaches must be fingerprinted or appear in
    `NOT_TRAINING_PATH`, each entry with its reason: benchmark, bookagent, measurement, fingerprint — all reached
    only under solver or league knobs, or used only to stamp.

**Design review (3 critics: statistics, scientific validity, implementation; findings re-derived by each).** Rebuilt
the draft on 20+ findings. The ones that changed the design:
- Seeds 61-65 had already been used by the G0 capacity gate, with identical starting weights. Leg F therefore runs on
  seeds 81-100.
- "Refuted at ≤ 14" was not a refutation: 14/20 has a one-sided 95% upper bound of 0.86. F1 now has three outcomes.
- F2 as drafted (strict raw ≤ 5/20) could hardly fail. It is now an exact sign test over the seeds the two readings
  disagree on.
- The report took the era, the isometries and the reference from its caller, so it could pass vacuously. They are now
  constants in `harness/floor.py`, and the register PINS that file and the proof test (new: `Register(pins=…)`,
  guard t34). Verify refuses once their behaviour changes, because the register stores a proof's node id, not its
  content.
- The driver's knobs sit outside the training era. So R200S seed 41 must first rebuild bit-for-bit, under the same
  stamps, before the arm counts (`c47_F_repro.json.gz`).
- A crash or schema gap must read NOT_RUN, never a refutation.
- The Connect-4 half of the draft was rejected outright; see leg C below.

**Amendment to §C.42(i), dated 2026-09-24, before any §C.47 data.** "Net level" means the policy averaged over the
game's verified symmetry group:
- π̄(a|s) = mean over g of softmax_legal(net(g·s))[g·a], move = argmax, the first legal move on an exact tie.
- It is an evaluation-time ensemble of |G| forward passes: no search, no solver, derived from the rules.
- The strict raw reading (every raw orientation) and the canonical reading §C.46 registered are reported alongside it,
  with equal weight.
- Why the change: h39 showed the raw net is not equivariant, and the operator was chosen AFTER §C.46's data. Its rescue
  power scales with the group: 8 on tic-tac-toe, 2 on Connect-4, about 1 beyond. Leg F records π̄ over subgroups of
  order 1, 2 and 4 so that dependence is measured, not assumed.

**Leg F — the floor, pre-registered.**
- **Arm:** R200S's recipe unchanged, run on seeds 81-100 with `--threads 1`. Era 27933b3a3bba; measurement code
  e5a9cb7b706d.
  - The integrity gate checks every recipe field against `C47_REFERENCE_CONFIG`, which is tested against the pinned
    R200S evidence.
  - It also checks the 8 isometries by name; 6 passes with siblings on iterations 2-6; the parameter count; 4,520
    positions; and the subgroup readings.
- **F1:** π̄-perfect seeds.
  - SUPPORTED at ≥ 15/20: an exact one-sided test that the per-seed rate exceeds 0.5 (size 0.021). The lower bound is
    0.544 at 15/20.
  - REFUTED at ≤ 11: the rate is then below 0.75 at 95%.
  - INCONCLUSIVE between.
  - P(pass) by true rate: 0.6 → 0.13, 0.7 → 0.42, 0.75 → 0.62, 0.8 → 0.80, 0.85 → 0.93. §C.46's 16/20 was the best
    of three readings chosen post hoc, so its predictive pass rate is nearer 0.7 than 0.8.
- **F2:** an exact one-sided sign test on the discordant seeds (π̄-perfect but not strictly perfect, against the
  reverse).
  - SUPPORTED at p < 0.01.
  - REFUTED if the reverse direction reaches p < 0.01.
  - INCONCLUSIVE otherwise.
- **NOT_RUN:** any failure of the automated integrity list, which is the only route to it. One rerun of the same
  seeds is allowed, under a new claim id.
- **Wording:** fixed in `floor.py`. The SUPPORTED text names the operator and the fact that it was adopted after
  §C.46. It also gives the one-sided lower bound, the canonical and strict counts, "labels from a search
  near-exhaustive at this game size", and the never-trained share re-measured on seeds 81-100 (not §C.46's 89%).
- **Descriptives:**
  - failures by reading and by subgroup, with seeded bootstrap intervals (the counts are mostly zero, so no
    t-intervals);
  - failures at trained vs never-trained positions;
  - any seed whose π̄ reading over all images differs from its canonical one;
  - the operator sanity check: π̄ of 5 untrained nets, and the labelling search from an untrained net at 200 sims on
    the 431 failable states.

**Leg C — the Connect-4 transfer, Stage 0: redesigned before any data.**
- **The draft was rejected.** It classified first raw-policy errors on uniformly random proven-win roots.
  - The validity and implementation critics measured the pre-fixed roots: 46-56 of 64 per depth were win-in-one.
  - None of 192 roots lay within one move of a proxy self-play set, so STOP was decided by the root sampler, not
    by the net.
  - That design also could not see the generalisation route, which h38 says is the only part that could transfer.
  - The same win-in-one caveat applies to §C.41's 0.71-0.88 conversion numbers.
- **Why a fresh run.** The §C.41 solver-free Connect-4 net was a scratchpad file wiped with /private/tmp. Its data
  was never saved, and no weights hash exists to prove a rebuild. Stage 0 therefore trains 5 nets (seeds 201-205) on
  the Stage-1 control recipe C_R: `harness.stage0.C0_RECIPE`.
  - The net: residual {32,3,32}, 60,555 params, scalar value head.
  - Training: 14 iterations × 48 games, 64-sim Gumbel self-play; openings of 4 random plies with a zero-ply share of
    0.3; reanalyze 1.0 at 64 sims.
  - One `train_alphazero` call per net.
  - Guards: the new self-play recorder is on (bit-identity tested); the exact solver is FORBIDDEN during training (it
    raises if called); raw-policy accuracy on a fixed probe is recorded per pass, as a plateau check.
- **Implementation review (before launch).** The rule changed before h43 was registered; the plan text here had
  predated it.
  - All three drafted plies were even, so only the FIRST player was ever to move. Siblings exist above all for the
    second player's positions after a deviation (h30).
  - Taking min(E, L) per net was biased low by about one full SESOI.
  - Plies pooled with unequal counts could manufacture a gap.
  - At 60 positions per cell, the tic-tac-toe signature read GO only about 45% of the time.
  - Separately, I found that mirrored positions shared a value cache in the column frame of whichever was solved
    first. The cache is now keyed by the exact position.
- **h43, as registered** (`harness/stage0.py`, pinned):
  - Plies 16/17/19/20/22/23, three for each mover.
  - Decision cells: a census of the final relabel buffer, and exactly `neural.one_ply_siblings` of that buffer.
  - E = the raw error rate at the siblings minus the rate at the buffer. L = the 64-sim label accuracy minus the
    raw accuracy at the siblings. Both are computed per ply and averaged with equal weight.
  - The rule is an intersection-union test with the net as the unit: GO iff both means ≥ 0.02 and both one-sided
    95% lower bounds > 0; STOP iff either upper bound < 0.02; UNDECIDED otherwise.
  - Each decision cell needs ≥ 80 positions, else NOT_RUN. STOP is never worded as "siblings are not the lever".
- **Calibration**, from §C.46 R200 before any Connect-4 data. The tic-tac-toe exposure gap was 0.042 (0.0006 on
  trained keys, 0.042 one move off; per-seed sd ≈ 0.019): the signature of the channel that cut failures ~10×.
  - A 0.05 bar would have called it STOP.
  - Simulated at the registered sizes, the rule reads that signature GO 0.93 of the time. A true zero gap reads GO
    0.02 and STOP 0.47-0.69.

**h43 → NOT_RUN (INCONCLUSIVE in the register, pre-registered). The integrity gate refused the run.**
- After the non-trivial filter, the final buffer held 44-78 positions in 16 of 30 cells, under the registered 80.
  I had assumed about 150.
- The report stops before E or L is computed, so no outcome on seeds 201-205 has been seen. The nets and their
  recorded self-play are kept under `checkpoints/c47_C0/`.
- The measurement took about 7.5 h against 45-52 min of training, almost all of it exact solves.

**h44 — Stage 0 v2** (`harness/stage0_v2.py`, pinned with `stage0.py`), registered BEFORE its data on fresh seeds
206-210. It was designed from v1's cell COUNTS only.
- The trained reference cell is every position the net trained on at any iteration, as the tic-tac-toe calibration
  used, with final-buffer membership recorded on each position.
- Each decision cell needs ≥ 60 positions. v1's counts, re-read this way, give 70-234.
- Sampling is smaller where it is only descriptive: siblings 250 per ply, rings and random positions 30, conversion
  roots 6.
- The rule and SESOI are unchanged.
- Simulated at v1's realised cell sizes (raw error ~0.45): the tic-tac-toe signature reads GO 0.81; a true zero
  reads GO 0.03-0.04 and STOP 0.35-0.46.
- **Descriptives** (cannot change the verdict):
  - raw, mirror-averaged and label accuracy per class × ply, and by mover;
  - how often random-play positions fall within two moves of the visited set;
  - proven-win conversion from non-trivial random roots and from the net's own buffer, each first error classed by
    distance;
  - the per-pass plateau probe.
- **What follows.** Stage 1, if GO, uses fresh seeds: neither the 201-205 nor the 206-210 nets are ever its controls.

**LEG F RESULTS (2026-09-24, verified).** The reproduction came first: R200S seed 41 rebuilt bit-for-bit under
today's code (weights 78052b2a…, all six passes). The arm, seeds 81-100, followed at era 27933b3a3bba.
- **Verification.** An independent verifier re-derived every number with its own code and reloaded all 20 saved
  nets. Every net's weights hash equals its final pass, and its readings match the evidence field-for-field
  (180/180).
- **Timing.** h41/h42 were registered at 11:49:08Z. The data started at 11:49:40Z (reproduction) and 11:55:33Z (arm).
  The pins recompute.

| reading, 20 fresh seeds | perfect seeds |
|---|---|
| **π̄, the policy averaged over the 8 verified isometries — h41 F1 SUPPORTED** | **17/20**; per-seed rate ≥ 0.66 (one-sided 95%), upper bound 0.96 |
| canonical image (the reading §C.46 registered) | 10/20 |
| strict: every one of the 4,520 raw positions | 3/20 |
| π̄ over subgroups of order 1 / 2 / 4 / 8 | 3 / 10 / 17 / 17 |
| 48-sim search (canonical) | 19/20 |

- **h42 F2 SUPPORTED.** 14 seeds are π̄-perfect but not strictly perfect, against 0 the reverse (p = 6×10⁻⁵).
  - This largely restates h39 (raw nets are not equivariant).
  - The fairer comparison, π̄ against the canonical reading, is 7 vs 0 discordant seeds (p = 0.008).
- **Replication.** §C.46 R200S gave 16 / 9 / 0 (π̄ / canonical / strict); here it is 17 / 10 / 3.

**What the result does and does NOT show.** The wording fixed in `floor.py` stands, with these qualifications made
explicit:
1. **The floor depends on the operator's group order.**
   - Averaging over order 2, the size of Connect-4's group, gives 10/20. That is at the refutation bar, as are the
     canonical (10/20) and strict (3/20) readings.
   - SUPPORTED needs averaging of order ≥ 4; the gain saturates at 4.
   - The rescue that meets the floor on tic-tac-toe is not available at that strength on Connect-4, so "not evidence
     the mechanism transfers" is, if anything, an understatement.
2. **The labeller is near-oracle at this size.**
   - An UNTRAINED net plus the 200-sim relabel search picks an optimal move at 98.4% of the 431 failable states.
   - The trained runs' final-pass labels are right on 187,080/187,080 self-play rows and 31,088/31,096 sibling rows.
   - The floor shows that a net can distil near-exact labels over a nearly enumerated space: 89.6% of failable
     positions received a training row (range 84-95% per seed). It does not show the loop discovering play.
3. **"Met" is a rate, not a guarantee.** The bound allows about one run in three to miss π̄-perfection.
4. **The residual failures are not a coverage failure.**
   - 3 of the 5 π̄ failures are key 601 (ply 3, on the optimal line, a §C.43 target key), trained in every seed:
     - Seed 86 saw it only as a sibling, with 8 of 8 rows mislabelled. It is the only wrong final-pass label in the
       whole failable universe across 20 seeds.
     - Seeds 84 and 99 had right labels at the minimum dose and never fitted them.
     - The value head's one-ply choice is wrong at 601 in 19/20 seeds.
   - The other 2 are never-trained keys (4983, 5306), one seed each. Generalisation to unseen positions mostly works
     (4983 went unseen in 9 seeds and failed in 1), but not always.
   - §C.46's residual keys do not recur, so there is no fixed residual state across studies.

**LEG C STAGE 0 RESULTS (2026-09-25, verified).**
- **h44 SUPPORTED, pre-registered.** Registered 21:02:49Z; the data started 21:02:50Z; every training part started
  after registration; the pins recompute.
- **Verification.** Two verifiers and a critic re-derived every number with their own code. They rebuilt 4 of 5 nets
  (weights hashes match) and re-graded 440 positions exactly: exact optimal set, raw, mirror-averaged and the 64-sim
  label.
- **Irregularity.** One position (seed 210) sits in both decision cells; it moves that net's E by 0.0003.

| per net, plies 16-23 averaged | 206 | 207 | 208 | 209 | 210 | mean [one-sided 95%] |
|---|---|---|---|---|---|---|
| E: raw error, siblings minus trained positions | .036 | .096 | .056 | .019 | .007 | **.043 [.009, .076]** |
| L: label minus raw accuracy at siblings | .084 | .094 | .066 | .079 | .096 | **.084 [.072, .095]** |

**What GO licenses, and what it does not.**
1. **Same number as tic-tac-toe, different meaning.**
   - Tic-tac-toe's 0.042 sat on a base error of 0.0006, a gap of about 70×. Connect-4's 0.043 sits on 0.393 raw
     error at trained positions against 0.436 at siblings, about 1.1×.
   - Closing E entirely would cut sibling errors by ~10%, not ~10×.
   - An absolute SESOI calibrated on a near-zero base does not scale; E must always be quoted with its base rate.
2. **The GO is fragile; its direction replicates.**
   - The lower bound is 0.009. Dropping net 206 or 208 crosses zero, and nets 209 and 210 are individually below
     the SESOI.
   - The first player's E alone has a lower bound of −0.002; the gap rests on the second player (0.046).
   - The between-net sd of E is 0.035, 1.8× the tic-tac-toe sd assumed in the power simulation, so the quoted
     0.81 power was optimistic.
   - The v2 minimum of 60 per cell was binding: cells of 63, 66 and 79 would have failed v1's bar of 80.
   - Post hoc and descriptive: the h43 nets 201-205 give E ≈ 0.064-0.070 and L ≈ 0.090, so E > 0 holds across all
     10 nets.
   - There is no v3 path.
3. **Where E sits.**
   - Difficulty does not explain it away. Siblings are, if anything, easier by chance (more optimal moves), and
     standardising them to the visited mix raises E to about 0.05.
   - But E sits almost entirely in FORCED-DEFENCE positions, where exactly one move avoids an immediate loss. There
     the raw net is right 0.72 at visited positions and 0.44 at siblings; elsewhere siblings do as well or better.
   - So the gap is mostly failing to parry threats that arise off the net's own lines, not a general drop in move
     quality.
4. **Exposure evidence.**
   - For: raw accuracy falls from visited 0.607 to siblings 0.564, two moves off 0.546 and random 0.504, while
     label accuracy is flat at 0.64-0.66; evicted positions score 0.584 against 0.618 still in the buffer.
   - Against: one move off vs two moves off differ by only 0.012, so there is little neighbourhood
     generalisation.
   - The selection confound (visited = the net's own moves) is removable only by a randomised holdout.
   - The plateau probe was still rising at pass 14.
5. **L is nearly guaranteed, and the ceiling is the informative part.**
   - L > 0 in every class, including visited (+0.049).
   - The 64-sim label is right only 0.65 of the time at siblings, and 4.8% of sibling positions are raw-right but
     label-wrong. Direct distillation is capped near 0.65.
   - The Leg F near-oracle-labeller caveat does NOT apply to Connect-4: the labeller is weak.
6. **The dominant failure is on trained positions.**
   - Raw accuracy at visited positions (0.607) is below their own labels (0.656).
   - First errors in proven-win conversion from the net's own buffer are mostly at buffer positions: 77 of 150, 67
     of them at the root. Only 8% are one move off. From random roots, 100% are "further".
   - Raw conversion is 14-17%.
   - GO says there is headroom where siblings train; siblings reach under 10% of the errors that decide conversion.

**Stage 1 design requirements that follow (not yet designed or run).**
- Primary endpoint: per-position accuracy on a hash-withheld half of C_RS_H's own siblings, with a randomised holdout.
  Proven-win conversion is secondary; at n = 10 its minimum detectable difference is ~0.08 against a plausible gain
  of +1-2 points.
- Strata registered in advance: forced-defence vs other, the (value, number of optimal moves) mix, and ply parity.
  Accuracy above chance is reported.
- A compute- and dose-matched control: siblings number 3.5-4.3× the buffer. There is also a harm endpoint on trained
  positions.
- A label-quality competitor: more label sims, or a threat-aware search, is a live alternative lever at a 0.65
  labeller.
- Training to a pre-registered plateau, or stating the under-training.
- Power sized from the realised between-net sd (0.035) with a SESOI relative to the base rate.
- A shared pre-solved evaluation bank. Stage 0 v2 spent ~10 h per 5 nets on exact solves; 3 arms × 10 nets would be
  60-70 h.
- Fresh seeds.


#### §C.48 — The smallest setup, and why the process misses 100% on tic-tac-toe (2026-09-25)

**Prompt (user).** Tic-tac-toe at 100% should be trivial: find the smallest net that holds perfect play without any
symmetry. Then find the smallest SETUP (architecture, encoding, training process) and the process that reaches it.

**T1: the smallest perfect net** (`scripts/smallest_net.py`, `evidence/c48_T1_smallest.json.gz`). Supervised on
exact labels over all 4,520 raw positions, with no symmetry and training until perfect or stuck (5 seeds each):

| net | params | perfect at every raw position |
|---|---|---|
| residual, width 16 | 5,629 | 5/5 |
| legacy conv, width 16 | 4,074 | 3/5 |
| legacy conv, width 32 | 12,746 | 5/5 |
| all nets ≤ 1,866 params | — | 0/5 |

- The smallest perfect net in the harness's family is about 4-6K parameters.
- §C.46's G0 finding that the 12,746-param net cannot fit tic-tac-toe was an artefact of the step budget.
- MLPs larger than 1.9K params were not swept.

**T2: the generic process run to convergence** (`harness/floor_converge.py`, pinned; h45/h46, pre-registered).
- Setup: the R200S recipe for 30 iterations instead of 6, seeds 301-310. The RAW net is scored at all 4,520 raw
  positions after every pass.
- **h45 (with augmentation): REFUTED, 0/10 perfect at the final pass.**
  - Failures fall to about 4 by pass 8, then hover at 1-9 without settling. 4 seeds touched 0 and relapsed.
  - From iteration 24 failures rise to about 20. That is exactly when the 8,000-state buffer fills and first-in-first-out
    eviction starts, in every seed.
- **h46 (no symmetry anywhere): REFUTED, 0/10.** Failures plateau at about 190. Self-play visits only about
  1,100-1,300 of the 4,520 raw positions in 30 iterations.

**Reading (post hoc; the eviction timing is observed, not yet tested).**
1. The obstacle is the PROCESS, not capacity.
2. Coverage: the buffer holds 8,000 states that are mostly repeats of about 1,150 distinct positions. Eviction discards
   rare positions first and the net forgets them.
3. There is no convergence mechanism. The learning rate is constant, each pass re-fits the whole relabelled buffer,
   and boundary positions flip back and forth.
4. Without augmentation, self-play alone never reaches three-quarters of the positions.

These feed the smallest-setup design (see the user-requested analysis after T3).

**Bank and T3: the Connect-4 ceiling by net size** (`scripts/build_bank.py`, `scripts/c4_ceiling.py`;
`evidence/c48_bank.json.gz`, `evidence/c48_T3_ceiling.json.gz`; h48, descriptive, registered after the data).
- **Bank.** 48,000 exactly solved positions at plies 14-38: half from the ten §C.47 nets' own self-play, half one
  move off it. Split train/test by a hash of the canonical key, so no test position has a mirror in training. It is
  reusable for any later probe (exact solves cost ~10 h per 5 nets).
- **T3.** Supervised on exact labels, 2 seeds per cell, early-stopped on held-out training positions:

| net | params | 4k labels | 12k | 35k | training-set fit at 35k |
|---|---|---|---|---|---|
| legacy32 | 20,616 | 0.54 | 0.61 | 0.66 | 0.89 |
| residual32 (the §C.47 net) | 60,555 | 0.53 | 0.60 | 0.71 | 0.92 |
| residual64 | 302,353 | 0.55 | 0.66 | 0.77 | 0.98 |
| residual128 | 1,785,873 | 0.59 | 0.73 | **0.82** | 0.98 |

  (Accuracy on unseen NON-TRIVIAL positions.)
- **Reading.**
  - Connect-4 is DATA-limited first: every net gains ~0.1 per tripling of exact labels, and none has flattened.
  - It is capacity-limited second: the 20K and 60K nets cannot even fit 35k positions.
  - Self-play positions and siblings, and early and late plies, all score within ~0.02 of each other.
  - For comparison, the §C.47 solver-free nets (60K, trained on 64-sim labels of about 65% quality) score ~0.56-0.61
    raw on comparable positions. The 60K net trained on EXACT labels reaches 0.71. That is the label-quality gap, at
    the same size.

**The smallest-setup programme (decided 2026-09-26 with the user).**
- **Objective (primary).** The fewest PARAMETERS that play perfectly, at ONE FORWARD PASS at play time (the raw net,
  no search). A setup is architecture + input encoding + symmetry handling + training process. More elaborate setups
  count if they need fewer parameters. Parameter-free preprocessing is disclosed with its cost: canonicalising the
  input costs rule calls.
- **Secondary, reported alongside.** The same frontier at small play-time search budgets, so the parameters-vs-search
  trade-off is visible.
- **Deferred alternative (recorded, not yet pursued).** Minimise TOTAL play-time compute, parameters and search
  together. Revisit once the primary frontier has good results across the games.
- **The process, per game.**
  1. ORACLE FRONTIER. Train each candidate setup on exact answers, to convergence, sweeping size: the smallest
     representable setup. On games that cannot be enumerated, data volume is a second axis (T3).
  2. PROCESS GAP. Run the generic solver-free process with the frontier setup and measure its distance from the
     frontier: coverage, label accuracy, the per-pass probe.
  3. CLOSE THE GAP, one pre-registered A/B per fix.
  4. The smallest setup is the smallest frontier point the process actually reaches.
- **First steps (tic-tac-toe).**
  - Step 1: T1 extended to setup variants — canonicalised input, larger and deeper MLPs, and rule-derived input
    features.
  - Step 3, in parallel: the T2 process fixes.
    - A de-duplicated buffer: one entry per position with its latest label, replacing FIFO eviction.
    - A settling phase.
    - A solver-free stopping rule: stop when the net agrees with its own search labels across the buffer.


**Step 1 result: T1b, the oracle frontier over setups** (`scripts/smallest_net.py`, `evidence/c48_T1b_setups.json.gz`;
h51, descriptive, registered after the data, VERIFIED).
- Same supervised fit as T1, 5 seeds each, on 29 new setups. Every setup is scored on all 4,520 raw positions; a
  canonicalising setup trains on the 627 canonical images and is scored through its own map back.

| setup | params | perfect on every seed |
|---|---|---|
| raw, residual16 (T1) | 5,629 | 5/5 |
| raw, MLP 64×64 | 6,026 | 5/5 |
| raw, MLP 128 | 3,722 | 2/5 |
| **canonicalised input, MLP 32** | **938** | **5/5** |
| canonicalised input, MLP 16×16 | 746 | 2/5 |
| canonicalised input, legacy conv 4 | 594 | 0/5 (12-29 failures) |
| rule features (raw or canonical), any size up to 1,514 | — | never 5/5 |

- **Canonicalising the input cuts the frontier about 6-fold, 5,629 → 938.** This is the more elaborate setup with
  fewer parameters. Its cost is 8 board transforms per decision, to be counted under the deferred total-compute
  objective.
- Rule features (wins now / hands a win) do not lower the frontier. On raw input they are worse than the canonical
  map alone at equal width.
- Next on the frontier: the canonical frontier sits between 594 and 938. Step 2 (process gap) should use the
  canonical setup once the fixed process (T4) is known.

**Step 3: the process fixes, and T4** (`harness/neural.py`, `harness/floor_settle.py` pinned, `scripts/settle_floor.py`;
h49/h50, PRE-REGISTERED 2026-09-26, VERIFIED the same day).
- `train_alphazero` gains three knobs, all off by default (the default path trains bit-for-bit as before, pinned by
  weight hashes recorded before the change). All three are refused off the reanalyze path.
  - `buffer_unique`: one row per position (`state_key`), refreshed to the newest end when seen again. Eviction drops
    the least recently seen position, never a repeat. Records `merged`.
  - `settle_epochs` / `settle_lr_final`: after the last iteration, extra epochs on the last training set with the
    learning rate decaying linearly to the floor (`train_net(..., lr_end=)`). Recorded as a history entry
    `iteration: "settle"`.
  - `record_self_agreement`: after each pass, the share of buffer positions where the raw argmax is one of its own
    label's best moves. Observation only; it is the candidate solver-free stopping signal.
- **T4** = T2's R200S recipe plus the three knobs (30 settle epochs, floor 1e-5), seeds 311-320, scored raw at all
  4,520 positions after all 31 passes. Bars as T2: supported at ≥8/10 perfect after the settle, refuted at ≤5.
  - h49 (augment): expected supported.
  - h50 (no_augment): expected refuted. Neither fix adds coverage.
  - Descriptives: seeds perfect BEFORE settling (the buffer fix alone), relapses, and FALSE STOPS (passes where
    self-agreement is 1.0 but the net fails somewhere). These calibrate the stopping rule.

**T4 result (verified 2026-09-26).**
- **h49 (augment): INCONCLUSIVE, 6/10 perfect after the settle** (bar 8; T2 was 0/10 on the same recipe without the
  fixes). Final failures by seed: 6, 0, 0, 0, 0, 0, 1, 4, 0, 2.
  - The unique buffer removed the eviction relapse. The buffer holds the 1,000-1,180 positions self-play visits, and
    nothing is ever evicted.
  - Before settling, 4/10 were perfect. The settle took 2 seeds to zero, improved 2 and worsened 2 (5→6 and 2→4). It
    helps but does not stop the flicker.
  - 7 seeds touched zero and relapsed. The flicker at 0-5 failures persists without eviction, so it comes from
    re-fitting relabelled targets each pass, not from forgetting.
  - The residual failures are scattered: 6 distinct canonical positions, only one shared by two seeds. Not one hard
    position.
- **h50 (no_augment): REFUTED, 0/10**, as predicted. 157-224 failures; self-play reaches 1,090-1,280 of the 4,520 raw
  positions.
- **Self-agreement is NOT a usable stopping rule as defined. It fails in both directions.**
  - no_augment: 6 FALSE STOPS. Full agreement at the final pass with ~200 failures, because it cannot see positions
    self-play never reached.
  - augment: it never reaches 1.0 (final 0.93-0.96) even on perfect nets. The likely cause, not yet checked, is that
    it counts the net's move as a disagreement when it picks a different optimal move from the label's argmax.
  - A stopping rule needs both a coverage term and a tie-tolerant agreement.
- **Where the gap stands.** Representation is solved: 938 params with canonical input, about 5.6K raw. The generic
  process now gets 6/10 raw-perfect with augmentation. What remains is (a) the last few flickering positions and
  (b) coverage without symmetry.
- **Post-hoc diagnostic: the residual failures are not label errors.** At all 13 positions the final augment nets
  still get wrong, a 200-sim search with that net returns an optimal move as its top choice. 9 of the 13 are forced
  single-answer positions with a one-hot label. The search knows the answer, and the raw net does not hold it.
  Most are late-game positions self-play rarely reaches. Likely cause: COVERAGE, the same obstacle as no_augment at a
  smaller scale. Not yet tested: the per-seed visited sets were not stored, so "never in the buffer" is not proven.
- **Next A/B candidate: coverage.** Solver-free ways to put unvisited positions in the buffer: deeper siblings,
  more random opening plies, or relabelling a rule-enumerated frontier.

**T5: coverage levers** (`harness/neural.py`, `harness/transfer.py`, `harness/floor_coverage.py` pinned,
`scripts/coverage_floor.py`; h52-h57, PRE-REGISTERED 2026-09-26, VERIFIED the same day).
- **Bug found and fixed at the root.** Without augmentation, siblings were de-duplicated by the CANONICAL key. A
  sibling whose mirror image was recorded never entered training, although a no-augment net must learn every
  orientation separately. The sibling key now follows augmentation: canonical with it, the raw state key without.
  The augment path is unchanged (pinned weight hashes). This affected T2/T4's no-augment arms (h46, h50).
- **New knob `sibling_depth`** (default 1, which is exactly the old siblings). Ring k is the new non-terminal children
  of ring k-1, not recorded and not in an earlier ring; the holdout is respected. It is checked against an independent
  breadth-first reference to depth 4, and history records `sibling_rings`.
- **The recorder** now logs every sibling set (observation only). The driver therefore knows each net's TRAINED set:
  all self-play positions (the unique buffer never evicts here; the driver refuses if it does) plus the last sibling
  set, all images of each with augmentation.
- **Arms** (T4 recipe + one lever; seeds 311-320, paired with T4):

| arm | lever | control | claim |
|---|---|---|---|
| augment_sib2 | siblings 2 moves deep | T4 augment (6/10) | h52 |
| augment_open6 | up to 6 random opening moves | T4 augment | h53 |
| no_augment_rawkey | the key fix alone | T4 no_augment (0/10) | h54 |
| no_augment_sib2 | key fix + 2-deep siblings | no_augment_rawkey | h55 |
| no_augment_open6 | key fix + 6 opening moves | no_augment_rawkey | h56 |

  - Bars as T4: ≥8/10 raw-perfect after the settle supports; ≤5 refutes.
  - **Reproduction gate:** T4's augment seed 311 must rebuild bit-for-bit under the T5 code, or every verdict is
    NOT_RUN.
- **h57, the mechanism** (tests T4's post-hoc reading). Pooled over all 50 final nets, with the net as the unit:
  untrained positions fail at a higher rate than trained ones. Supported if a one-sided sign test gives p < 0.01 over
  ≥10 failing nets AND the pooled rate ratio is ≥5. Refuted if the ratio is ≤2.

**T5 result (verified 2026-09-26; reproduction gate passed: T4 augment seed 311 rebuilt bit-for-bit).**

| arm | raw-perfect after the settle | final failures by seed | positions trained (of 4,520) | claim |
|---|---|---|---|---|
| augment_sib2 | **10/10** | all 0 (also 10/10 BEFORE the settle) | ~4,500 | **h52 SUPPORTED** |
| augment_open6 | 3/10 | 3, 0, 5, 0, 1, 0, 1, 12, 4, 6 | ~4,190 | h53 refuted |
| no_augment_rawkey | 0/10 | 20-48 (T4: 157-224) | ~3,060 | h54 refuted |
| no_augment_sib2 | 5/10 | 0, 0, 1, 0, 1, 2, 1, 0, 0, 1 | ~4,360 | h55 refuted, exactly at the bar |
| no_augment_open6 | 0/10 | 26-84 | ~2,960 | h56 refuted |

- **h57 (mechanism): SUPPORTED, decisively.** In all 32 of 32 failing nets, untrained positions fail at the higher
  rate (sign test p = 2e-10). The pooled rate ratio is ~249: 2.7% of untrained positions fail against 0.01% of
  trained ones. 98.6-100% of every arm's failures are on positions the net never trained on.
- **The generic process now reaches perfect tic-tac-toe.** With augmentation and 2-deep siblings, 10/10 seeds are
  raw-perfect at all 4,520 positions: no averaging, no canonical images, solver-free labels (200-sim search), and the
  same cost as T4 (~14 min per seed). Paired with T4, the 4 seeds T4 left failing (6, 1, 4, 2) are all 0. The settle
  is not needed here: all 10 were already perfect before it.
- **The sibling-key bug mattered.** Fixing it alone cut no-augment failures ~7-fold (≈200 → 20-48).
- **More random openings do not help.** They move coverage to later positions but lose early ones: augment_open6 is
  worse than T4 (3/10 against 6/10).
- **Without symmetry** 2-deep siblings get 5/10, with 0-2 failures, all on the ~160 positions still untrained.
  Coverage is still the whole story.
- **Caveat for transfer.** At this size the rings nearly enumerate the game (2-deep siblings plus augmentation train
  ~4,500 of 4,520). Connect-4 cannot be enumerated, so the lesson that transfers is "the failures are the untrained
  positions". The lever, blind rings, does not transfer. Connect-4 needs coverage aimed where the net is likely wrong.
- **Suite fix found while verifying.** tests/conftest.py matched register proofs by function name without the
  parameter id, so parametrized proofs (one per arm) never followed their own claim. It now matches the full node
  name, with a test.

#### §C.49 — A repeatable smallest-net process, then Connect-4 perfect play (2026-09-27)

**Prompt (user).** Make the smallest-net method a real, repeatable process, then apply all of it to Connect-4 to get
perfect play.

**What "perfect play" means for Connect-4 (fixed before any work).** "Optimal at every position" (~4.5 trillion)
cannot be measured. Two measurable levels:
- **P-START (the goal).** The RAW net (one forward pass per move, no search) as first player wins against EVERY
  defence from the empty board. It is certified by walking the tree of the net's single move at each of its turns
  and every legal reply at the opponent's, checking with the exact solver that each net move keeps a proven win. A
  strategy that keeps a proven win at every one of its moves must win, because a finished draw is not a win.
  Success: zero non-winning net moves over the whole tree.
- **P-SAMPLED (reported alongside).** On a broad held-out sample of positions from every stage of the game, the raw
  move never gives away value (win → draw/loss, draw → loss). Reported as an error rate with an upper confidence
  bound.

The PROCESS stays solver-free, as in §C.47-48: training labels come from search, never the solver. The solver is
used only to MEASURE: certify P-START, grade P-SAMPLED, and map the oracle frontier.

**Tasks.**
1. **The net can take a standardised orientation.** `Connect4Net` gets `canonical_input`: the input is mapped to one
   fixed image under the game's verified symmetries, and the policy is mapped back. Every consumer (search, training,
   scoring, play) works unchanged. The net also gets an MLP body (`mlp_hidden`), so the §C.48 frontier setups are
   ordinary nets.
   **Success:** the output is exactly equivariant under every verified symmetry (tests); existing arches build
   bit-identically; `canonical_input` + `mlp_hidden=[32]` on tic-tac-toe has exactly 938 params.
2. **A reusable smallest-net tool** (`harness/frontier.py`, `scripts/frontier.py`). It takes any game, a family of
   setups with a width parameter, and a pluggable target: every enumerable position, or a pre-solved bank. It finds
   the smallest width that succeeds on every seed, by doubling then bisection, under several training recipes. It
   records every probed point and flags non-monotone widths.
   **Success:** unit tests, plus it re-derives §C.48 T1b on tic-tac-toe: a canonical-MLP frontier ≤ 938 params with
   every seed perfect and the width below failing.
3. **Step 4 on tic-tac-toe** (pre-registered): the T5 process (unique buffer, 2-deep siblings, settle) trained at
   the frontier setups instead of the 57K net.
   **Success:** the smallest setup the process makes raw-perfect on ≥ 8/10 seeds is named.
4. **Connect-4 instruments first.** The P-START certifier (policy-restricted tree walk plus exact solver, with its
   size and cost measured) and the P-SAMPLED grader. Baseline: the best existing Connect-4 nets.
   **Success:** the certifier verifies the solver's own move choices as P-START-perfect and catches a deliberately
   planted losing move.
5. **Connect-4 oracle frontier and process.** First, what can a net represent near the net's own proof tree. Then the
   solver-free process with coverage aimed at that tree: all opponent replies to the net's own line, the §C.48
   lesson that failures are untrained positions. Each lever is a pre-registered A/B.
   **Success:** P-START certified for a raw net.

**§C.49 progress (2026-09-27).**
- **Task 1 DONE.** `Connect4Net` gets `canonical_input` (the input is mapped to one image under the verified
  symmetries, and the policy is mapped back, averaged in sorted order over the symmetries that produce the image) and
  `mlp_hidden`.
  - Exactly equivariant on tic-tac-toe and Connect-4, for all three body types including fully symmetric positions,
    and checked against a hand computation. An MLP body reproduces 938/746/1,866/3,722 params. Old arches build
    bit-identically.
  - In the loop: a canonical net refuses augmentation, and its siblings and unique buffer key by symmetry class.
- **Task 2 DONE (tool), criterion met under 2 of 3 recipes.** `harness/frontier.py` + `scripts/frontier.py`,
  evidence `c49_frontier_tictactoe.json.gz` (8 families × 3 recipes × 5 seeds):

| family | default | slow | fast |
|---|---|---|---|
| raw mlp | 4,534 | 4,650 | 4,389 |
| raw mlp2 | 4,826 | 4,409 | 4,969 |
| raw conv | 4,906 | 5,349 | 4,906 |
| raw residual | 5,008 | 5,629 | 5,008 |
| **canon mlp** | 1,054 | 938 | **851** |
| canon mlp2 | 1,154 | 1,010 | 1,010 |
| **canon conv** | **994** | **994** | **994** |
| canon residual | 1,669 | 2,038 | 2,443 |

  - Standardised orientation cuts the frontier about 5× in EVERY body type. Canon conv-6 (994) is the only point
    that held under all three recipes.
  - **Finding: the frontier is a BAND, not a line.** Near it, success depends on the seed: canon MLP widths 24-37
    (706-1,083 params) succeed on some seeds, while failing seeds stall almost always at exactly 8 failures, likely
    one hard position in all 8 orientations. An "every seed" single-width frontier moves with that luck (1,054 /
    938 / 851 by recipe).
  - The tool now runs every seed at every probe and reports the success rate per width, plus three frontiers:
    STRICT (the searched width), ROBUST (every wider probe also succeeded) and FIRST SUCCESS (the optimistic
    representability bound).
- **Task 3 (T6) PRE-REGISTERED (h58-h62), running.** T5's process at 938 (canon MLP-32, plain and 1,500-epoch
  settle), 994 (canon conv-6, long settle) and 5,008 (raw residual-15, plain and long settle). The long settle tests
  whether a shortfall is the training budget: the process trains ~210 epochs, and the oracle fits took ~1,000+.
- **Task 4 instruments.**
  - `harness/certify.py` (tested, mutation-checked): certifies that a player keeps the exact value at every
    reachable position against every reply. It catches and names a single planted losing move and certifies the
    solver's own play.
  - **Scale measured:** the tree under the best existing net's first-player strategy (ab302, 335K params) has 162K
    distinct positions at ply 16 and grows ~3.5× per 2 plies, on the order of 10^9 by game end.
  - **The Python solver cannot certify the opening:** 8 s to ~2 h per ply-8 position.
  - **Native solver** (`harness/c4solver.c` + `harness/native_solver.py`). The user chose "write our own" over the
    AGPL-3.0 BitBully. It is a line-for-line C port of `harness/solver.py`, compiled on first use into `build/`
    (gitignored), and agrees with the Python solver on every tested position, weak and strong, even with a 4-entry
    table. Ply-8 positions: 0.1 s and 37 s, against 8 s and 6,894 s in Python.
  - NOTE for the user: `harness/solver.py` describes itself as a port of Pascal Pons' solver. Its provenance (his
    tutorial blog or his AGPL repository) is unverified.
- **T6 result (verified 2026-09-27).**

| arm | params | raw-perfect | final failures | claim |
|---|---|---|---|---|
| **residual15_long** | **5,008** | **9/10** | 0×9, 1 | **h62 SUPPORTED** |
| residual15 | 5,008 | 1/10 | 0-9 | h61 refuted |
| canon_mlp32 | 938 | 0/10 | 314-450 | h58 refuted |
| canon_mlp32_long | 938 | 0/10 | 22-114 | h59 refuted |
| canon_conv6_long | 994 | 0/10 | 21-96 | h60 refuted |

  - **The smallest setup the process reaches is 5,008 params** (raw residual-15, augmentation, 2-deep siblings, a
    1,500-epoch settle), 11× smaller than T5's net and at the raw oracle frontier.
  - **The ~1K standardised setups are representable but the process does not fit them, and it is not coverage.**
    Every failure is on a TRAINED position (~4,515 of 4,520 covered).
  - Likeliest cause, not yet tested: the optimisation budget. A standardised net trains on ~625 rows (one per
    symmetry class) against ~11,500 for the augmented raw net, so under the same epochs it gets ~20× fewer gradient
    steps. Its settle also starts at lr 1e-3, where the oracle fits used 2e-3-5e-3.
  - Self-agreement is ~0.94 even for perfect nets. This confirms it counts a different optimal move as a
    disagreement, so it cannot be a stopping signal as defined.
  - **Next A/B (proposed):** the standardised setups with the settle matched to the oracle fit's budget (steps and
    learning rate).
- **Task 4 native solver, faster (still exact).** A two-sided transposition table (lower bounds from cutoffs, plus
  the upper bounds the reference keeps) and a direct one-solve `solve_position` (weak only, because strong distance
  scores differ by one between the two routes in the Python reference).
  - Ply-8 position: 37 s → 3 s. Ply 1 after the centre opening (the hardest node of the certificate): 49 min, once.
  - Our own solver confirms the empty board is a first-player win.
  - Certification of ab302 through depth 12 is running.
- **T7 result (verified 2026-09-27; h63-h65 all REFUTED, but the budget reading holds).** The same standardised
  setups with the settle matched to the oracle fit's budget (3,000 epochs from lr 5e-3):
  - canon MLP-32 (938): final failures 8-16 (T6 long settle: 22-114; before the settle: 350-498).
  - canon MLP-48 (1,402): 8-16.
  - canon conv-6 (994): 0-36, one seed perfect.
  - Every failure is on a TRAINED position, and every count is a multiple of 8: each net is wrong on ONE or TWO
    symmetry classes (8 raw orientations each).
  - **The residual is one position.** Class 608 fails in 28 of 30 nets:
    `X O . / O . X / . . .`, X to move, whose only winning move is the corner fork (bottom-right).
    - The 200-sim search labels it RIGHT 19/20 times even from an untrained net, so this is representation, not
      labels. It is the same position the oracle fits stall on ("stuck at 8"), so it is what makes the ~1K
      frontier a band.
    - Class 601 (10 nets) is different: labels there are right only 3/20 times from an untrained net (the §C.47
      mislabel position).
  - **Tic-tac-toe answer for now.** The smallest setup the process reaches is 5,008 params (h62). At ~1K it gets
    within one or two positions, stopped by one fork that sits at the edge of what a ~1K net can represent.
- **P-START baseline (h66, descriptive, VERIFIED).** The strongest existing net (ab302, 335K params), raw, as first
  player through 12 plies: NOT certified. 50 value-losing moves: depth 2: 2 of 7 positions, depth 6: 4, depth 8: 9,
  depth 10: 35. These are lower bounds, since subtrees below a failure are not walked.
  - The first failures come at its SECOND move. After the centre opening and a neighbouring-column reply, it
    answers in that same column, which only keeps a draw.
  - The walk took 2.0 h, 1.6 h of it on the first two plies of solves (ply 1: 52 min; ply 3: 45 min).
  - The distance to P-START is large and starts in the opening.
- **Task 5 tooling.** `harness/strategy_fit.py` grows a net on its own strategy tree: walk, solver-label, retrain,
  repeat until certified through a horizon. It is tested on tic-tac-toe (it grows a certified strategy; a width-1 net
  does not) and mutation-checked. `scripts/strategy_frontier.py` is the Connect-4 driver, with a persistent
  exact-label cache (`books/c4_labels.json.gz`).
  - The first run (depth 6, canon conv-8 and residual-16) is under way.
- **Strategy frontier, depth 6** (`evidence/c49_strategy_d6.json.gz`). Canon conv-8 (3,432 params) and residual-16
  (7,179) are both CERTIFIED through 6 plies after 6 rounds (failures per round: 1,4,6,4,4,0 and 1,4,8,1,5,0), on 92
  labelled positions.
  - Labelling cost 5.6 h, all in the opening: a full label is 7 child solves. The second setup took 9 s from the
    cache.
- **Check-first labelling** (`expand_round(check_many=...)`, tested, mutation-checked). The net's move is checked
  with ONE solve. If it keeps a win, that move alone is a valid label, since nothing beats a win. Only a
  non-winning move costs the full label.
- **Depths 8, 10 and 12 running**: canon conv 4/8, residual 8/16, 12 rounds each, shared label cache.
- **Strategy frontier, depths 8 and 10** (`c49_strategy_d8/_d10.json.gz`). Depth 12 was cancelled with the user's
  agreement: no setup certified at depth 10.

| setup | params | depth 8 | depth 10 |
|---|---|---|---|
| canon conv-4 | 1,576 | **certified** (12 rounds, 1,086 positions) | root collapse |
| canon conv-8 | 3,432 | **certified** (6 rounds, 710 positions) | root collapse |
| residual-8 | 2,443 | root collapse (capacity) | root collapse |
| residual-16 | 7,179 | not certified: out of ROUNDS (fits all held; failures 3-8 at the end) | root collapse |

  - **Two failure modes.** (1) CAPACITY: a refit cannot hold all its data, the opening move is lost, and the walk
    stops at the root. (2) ROUND BUDGET: every fit holds, and failures are still falling when the rounds run out.
  - **Tool fix:** growth stops as STALLED when a round labels nothing new after a refit that could not hold its
    data. Such rounds only repeated the same fit (residual-8 ran 10 of them).
  - **Running:** depth 10 with canon conv-16/32 and residual-32, 20 rounds.
- **Restart (2026-09-29, user's choice: stop and restart with fixes).** The first wide run was stopped after 13.8 h
  on its first setup: refitting a fresh net each round on growing data dominated. Two fixes:
  - (1) **Warm-start refits** (`fit(init_net=)`, `fit_strategy(warm_start=True)`; tested, mutation-checked): each
    round continues from the previous net.
  - (2) **The label cache is saved after every labelling and check batch**, atomically (temp file + rename). The
    stopped run's new labels were lost because the cache was saved only per setup; 13,130 earlier labels were kept.
  - Relaunched: depth 10, canon conv-16/32 and residual-32, 20 rounds, warm start.
