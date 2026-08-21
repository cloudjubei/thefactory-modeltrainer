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
- **Connect-4 SOLVED — the crisp end-state of §C.3 (ACTIVE).** Connect 4 has ground truth (first player wins
  with perfect play). Today the `alphazero` core plateaus WEAK (on disk: champion frozen at gen12, gens 13–15
  trained but never promoted; the owner can still beat it) — so "reach the known winning strategy" is the crux,
  not a nicety. Build, staged: (1) a perfect-play **oracle** — a torch-free `harness/solver.py` (bitboard
  negamax + alpha-beta + transposition + a one-time opening book) exposed as an `OracleAgent`, registered as a
  `connect4` persona AND the top rung of the rating spine, so every model is measured against perfect play;
  known-answer TDD (centre is the unique optimal opening, empty board is a P1 win, takes wins / blocks losses).
  (2) **near-perfect ladder rungs** — MCTS with a tactical (heuristic) rollout + immediate win/loss checks at
  ~2–5k sims, and a depth-limited oracle rung (the heuristic stays an honest weak floor; the full Allis rule
  engine is not worth chasing). (3) an alphazero setup that actually reaches it — a small **ResNet**, a real
  budget (~40–60 iters × 150–250 self-play × 200–800 sims × 8–10 epochs, ~100k buffer), the oracle in the
  league, and **oracle distillation** (pretrain the net on solver-labelled optimal-move + value targets) as the
  biggest lever. (4) a **measurable SOLVED criterion** — `oracle_optimality_rate` (model's greedy move ∈ the
  solver's optimal-move SET over a fixed benchmark corpus, `harness/benchmark.py`) ≥ 0.99 AND
  `wins_as_p1_vs_oracle == 1.0`; NOT win-rate-vs-oracle (degenerate 0.5 under seat alternation, below the 0.55
  spine threshold). Surface a `health: solved` badge + the `oracle_*` metrics. This is the trigger that unblocks
  the model-comparison view (§E).
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
- **Expose `verifyImprovement` as a chat tool/activity** — the engine function is done; the thin remaining
  piece is the thefactory-backend `trainerTools` schema + dispatch wrapper.
- **S9 leakage tail** (lowest value, do when it bites): a per-split membership signature
  (`RunSummary.splitSignature`) + a train/eval-overlap disjointness detector; and relocate the trading fidelity
  predicate (`isRunAffectedByFidelity*`) out of shared `modelTrainerUtils` into BlackSwan (repoint-and-DELETE).
- **Owner ratifications** (conservative defaults are live; revisit if desired): fail-closed-*with-reason*
  benchmark (not hard-required — preserves BlackSwan's `return_vs_hold_pct`); best-of-N BESIDE DSR sharing one
  `nTrials` floor; reuse / `unverifiable` flags advisory, not hard-block.

### C.5 — Optimal-play trainer (boardgames) — detailed log

The full engineering record for the §C.4 "Connect-4 SOLVED" milestone: **making a solved game
*computably* optimal, and generalizing it.**

#### The wall we hit (measured)

Reaching **provably optimal** play requires playing the **opening** perfectly. Exact solving costs
`~16ms` at move 20, `~1.4s` at move 12, but **~158 seconds per position at move 10** — the in-memory
transposition table (capped at 2M) thrashes on the opening subtree. So:

- exact **opening** labels for the net are infeasible one-at-a-time, and
- no fast/learned agent can be made provably optimal without help in the opening.

What already works and stays: **oracle-opening distillation** (fixes the net's broken edge-first opening →
centre-first; SHIPPED), the **exact endgame cutoff** (`solve_endgame` — provably-perfect endgame; SHIPPED),
the **Play-off** (objective who-wins + `wins_as_p1_vs_oracle` optimality gauge; SHIPPED).

#### The idea (user's three levers) — this is exactly how Connect 4 was actually solved

A persistent **opening book + endgame tablebase**, **symmetry-reduced**, with **early game termination**.
Composed correctly these break the wall and make an *exactly optimal, fast* agent, and they generalize.

##### Lever 1 — a persistent solved-position store (tablebase / book)

Store the game-theoretic value (win / loss / draw under optimal play — optionally the signed distance-to-end)
of *some* positions, keyed by a canonical position key, persisted to disk and **accumulated across runs**.

- **Value-only + one-ply lookahead is enough to PLAY optimally**: at a position, look up each child's stored
  value and pick the negamax-best move. No need to store best-moves separately (the "rainbow" walk-to-the-win
  is then implicit); storing the strong (signed-distance) value additionally gives fastest-win / slowest-loss.
- **Priority to keep**: *hub* positions (high transposition in-degree — reached from many move orders, so one
  solve saves many) and *hard* positions (deep solves). Bounded size with priority eviction.
- **Why it breaks the wall**: build it **bottom-up** (store the deep frontier first). Once moves 12–18 are in
  the book, solving a move-8 position is *shallow* — its children are instant lookups. The 158s solve becomes
  a handful of table hits. Each training/exploration run extends the frontier upward ("over time we store the
  difficult ones") until the opening is covered.

##### Lever 2 — symmetry (mirror) canonicalization

Connect 4 is symmetric under left↔right reflection about the centre column. Canonicalize every key to
`min(key, mirror(key))`:

- **~50% fewer positions** to solve and store (a position and its mirror share one entry).
- game-theoretic value is **mirror-invariant**, so value-only lookups need **no** move-remapping.
- **the net gets it too**: canonical (or symmetry-augmented) encoding → the net learns a symmetry-invariant
  policy for free → ~2× data efficiency + consistency. (Generalizes to richer symmetry groups; see below.)

##### Lever 3 — early game termination

During self-play, evaluation, the Play-off, distillation labelling, *and* inside the solver: the moment a
position's value is known (book hit), **end the game / cut the search** with that outcome instead of playing
it out.

- speeds up **everything** (fewer plies per game; solved subtrees never re-expanded);
- lets even the **net** agent "know" the result — a game that reaches a stored won/lost/drawn position ends
  immediately with the true result, so we neither waste time nor let the net misplay a decided position.

#### How they combine → the deliverables

1. **Exact opening labels become feasible** (book-accelerated solving) → distillation trains the net on *true*
   optimal moves in the opening, not depth-limited approximations → the net approaches optimal.
2. **A deployable, provably-optimal, FAST agent**: opening = book (one-ply table lookup, instant) + endgame =
   solver cutoff (exact) + net as the fallback where the book is thin. No 158s solves at play time.
3. **Everything stays visibly testable** in the Play-off: `wins_as_p1_vs_oracle → 1.0`, champion self-play
   first-player-wins → 100%, and the book-agent shows as optimal.

#### Generalization (Connect 4 is just the first game)

Keep the engine game-agnostic; put only the hard parts behind per-game hooks. As games get more complex and
harder to encode, the reusable engine (below) is what carries over — **especially the net trainer**.

- **Tablebase (Lever 1)** — fully game-agnostic: `bytes canonical_key → value`. Knows nothing about any game.
- **`SolvableGame` protocol** (extends `Game`) — the per-game hooks:
  - `canonical_key(state) -> bytes` — symmetry-reduced position key.
  - `symmetries() -> [Symmetry]` — each `Symmetry` maps an encoded input tensor **and** a policy vector (for
    net augmentation) and the position key (for canonicalization). Connect 4 = {identity, mirror}. Square-board
    games (tic-tac-toe/gomoku) = the 8 dihedral maps. A game with no exploitable symmetry returns `{identity}`
    and simply gets no space saving — the rest still works.
  - `solve(state, book) -> value` — the exact solver (Connect 4 = the bitboard negamax; a game with none just
    has no book → no early-termination, but the same framework).
- **Early termination (Lever 3)** — a generic `play_until_decided(game, agents, book)` used by every game path.
- **Net trainer** — the encoding + architecture is the per-game frontier. The engine helps it three ways that
  matter more as games get complex: (a) **symmetry augmentation** (declared once per game) multiplies data and
  bakes in invariance; (b) **book-accelerated exact distillation** gives the net *ground-truth* targets wherever
  the game is solvable, so the net doesn't have to discover them; (c) **early termination** on book hits keeps
  self-play cheap. For games with no solver, the book is seeded from strong-agent agreement / self-play consensus
  instead of exact solves — same store, weaker guarantee.

#### Status — ALL PHASES SHIPPED (TDD; 137 boardgames tests green)

Phases 1–5 are complete. `harness/book.py` (builder + `book_optimal_actions` + `play_until_decided` +
`build-book` CLI + seed mode), `harness/bookagent.py` (the deployable `book` agent), `harness/tablebase.py`,
the solver's symmetry-canonical persistent TT, net symmetry augmentation, and the `SolvableGame` hooks on
both `connect4` and the new fully-solved `tictactoe` all landed with direct tests. Wired to the app: a
`build-book` capability/activity + a "Build book" button in the Play-off panel, and the `book` agent enters
the Play-off + gauntlet as an optimal-play competitor (a `book` rung in the manifest ratingSpine).

**The honest coverage picture (measured).** Cold opening solves cost `~2ms` at ply 16, `~120ms` at ply 14,
`~4s` at ply 12, `~9s` at ply 10, and MINUTES near the empty board; enumerating the deep frontier from the root
blows up breadth-first. So full opening coverage is a genuine long-running ACCUMULATOR (each `build-book` run
extends it, persisted + symmetry-reduced), not a one-shot — exactly the design. What ships working today:
a committed **60k-position** connect4 midgame/endgame book (seed mode; opening coverage honestly ~0% and
reported as such), the exact endgame solver (always perfect), and the depth-limited oracle for the unbooked
middle. **Tic-tac-toe is the crisp complete proof**: its whole tree solves instantly, the book completes to
100% coverage, and the `book` agent is provably optimal — it tops the Play-off and its self-play is a 100%
DRAW (tic-tac-toe's true value), demonstrating the identical engine yields optimal play end-to-end.

#### Phased plan (each phase TDD, each ends with a measurement)

- **Phase 1 — Foundation + wall-break proof.** `harness/tablebase.py` (persistent store: get/put/contains/
  load/save, compact value codec, size cap + priority eviction, game-agnostic). Connect 4 `_mirror` + canonical
  key. Solver reads the book (exact cutoff on hit) and `solve_and_store` writes it. **Measurement**: batch-solve
  opening positions with a warmed/symmetry book vs cold — show the batch (and a re-solve, and a mirror) go from
  seconds to ~instant. *(this is the proof it's computable)*
- **Phase 2 — Incremental opening-book builder + a `build-book` trainer activity.** A bounded, resumable pass
  that solves+stores the reachable frontier bottom-up, priority-ordered (hub/hard first), persisting to a
  project-committed book file. Runs incrementally (each Start extends it). Exposed as a chat-invocable capability.
- **Phase 3 — Book-accelerated exact distillation + the deployable optimal agent.** Rewire `oracle_distill_games`
  / `build_distill_corpus` to pull *exact* opening labels from the book (fall back to the depth-oracle only where
  the book is thin). Add a `book` agent (opening book + endgame solver + net fallback) — provably optimal + fast;
  register it as a ratingSpine rung + opponent so the Play-off can crown it and measure everyone vs it.
- **Phase 4 — Symmetry in the net + early-termination everywhere.** Canonical/augmented encoding via
  `Game.symmetries()`; `play_until_decided` in self-play, evaluation, gauntlet, tournament. **Measurement**:
  training + play-off wall-clock down; `wins_as_p1_vs_oracle` of the distilled champion up toward 1.0.
- **Phase 5 — Generalize.** Land the `SolvableGame` protocol; make tablebase/early-term/symmetry consume it;
  document how a new game plugs in (key, symmetries, solver-or-seed). Prove on a 2nd game (tic-tac-toe: trivially
  fully-solvable, exercises the 8-fold dihedral symmetry) that the same engine yields an optimal book-agent.

#### Next enhancements (queued — documented now, to tackle soon)

These extend the shipped engine; not yet built.

##### E1 — Player-colour (p1↔p2) collapse — INVESTIGATED, NOT A WIN (2026-08-19)

The earlier idea (and two research passes) claimed a mover-relative key would collapse colour-swap pairs and
~halve the generic book. **Measured on tic-tac-toe: it saves 0 entries** — enumerating all 4,520 reachable
non-terminal positions gives **627 distinct current keys and 627 distinct mover-relative keys**. Reason: in a
strictly-ALTERNATING game the side-to-move is determined by the piece counts, so a position's colour-swapped /
turn-swapped twin has the wrong parity and is **UNREACHABLE** — there are never two reachable positions to merge.
The `* 2 + to_move` turn bit in `tictactoe.canonical_key` is redundant (turn is a function of the board) but
harmless, and it splits nothing.

What the intuition ("colour doesn't matter once a value is known") really wants is **mover-relative VALUE
storage**, and that is ALREADY how everything works: `position_value` / the solver / the book store the value
from the side-to-move's perspective (win +1 / draw 0 / loss −1) and every lookahead negates child values
(`book.py:91`, `solver.py:289-292`), so a stored value applies regardless of which physical colour is on the
move. The real geometric saving is the board symmetry — Connect 4's left↔right mirror and tic-tac-toe's full
8-fold dihedral — which is already exploited. **No code change; keep the mirror/dihedral canonicalisation as-is.**
(A game that is NOT strictly alternating — passes, variable move counts — could in principle benefit; revisit
per-game only if such a game appears.)

##### E2 — Related follow-ups (candidates)

- **Grow real opening coverage.** Connect 4 opening coverage is honestly ~0% (the committed book is
  midgame/endgame only). Longer / offline `build-book` accumulation (bottom-up, symmetry-reduced) is what lifts
  `wins_as_p1_vs_oracle` toward a genuine optimality proof — the accumulator design is already in place, it just
  needs the compute budget to climb the frontier upward.
- **A stronger play-off yardstick.** The play-off oracle is depth-6 (endgame-exact, but beatable in the
  opening), so "optimality vs oracle" is a yardstick, not a proof. A stronger yardstick depends on the
  opening-coverage growth above (a deeper *live* oracle stays minutes/move in the opening — the wall).

#### Phase 6 — Generic SELF-PRODUCED approximate book (current focus)

Reframes Tier 1 of the deep-research proposal. We do **NOT** import external databases (Tromp / bitbully) — the
whole point is a system that PRODUCES its own opening knowledge for ANY game, even when that knowledge is
incomplete. Connect 4 is only the honing example. The book graduates from an exact-only tablebase into a
generic store that mixes PROVEN and APPROXIMATE knowledge and upgrades one into the other over successive runs.

##### The richer entry (supersedes the scalar Tablebase value)
Per canonical key, store:
- `status`: `PROVEN_WIN | PROVEN_LOSS | PROVEN_DRAW | ESTIMATE` — a single int8 column.
- `value`: exact {−1, 0, +1} when proven; else an estimate in [−1, +1] (mover-relative tendency).
- `best_actions`: a **bitmask** (`num_actions ≤ 64` → one uint64) of the optimal set (proven) or top moves
  (estimate). This is the one-ply-lookahead move set, the model's policy target, AND the IMPLICIT principal
  variation — walking `best_actions` from a position reproduces the winning/drawing LINE, so "raw paths" are
  reconstructible on demand and need not be stored per entry (a `principal_variation(book, game, state)` walk).
- `wdl` (optional): win/draw/loss counts behind an ESTIMATE (uint16×3) — the win/loss RATIO indicator a deep
  model reads to grade moves where nothing is proven yet.
- `n` / confidence: sample size behind an estimate (so estimates are comparable + upgradable).
- `depth_to_end` (optional): signed distance-to-result for fastest-win / slowest-loss.

Persisted COLUMNAR (parallel numpy arrays like today's `.npz`) so winner/loser/draw filtering is a vectorised
mask and lookups stay O(1). Values stay MOVER-RELATIVE (colour-agnostic — see E1).

##### The evaluator ladder (exact → bounded-proof → estimate)
`evaluate(game, state, book, budget) -> Entry`, tried in order, each reusing already-booked children:
1. terminal → PROVEN from the winner.
2. book hit → return the stored entry.
3. cheap EXACT: `position_value` / `exact_optimal_actions` resolves within budget (endgame / small tree) → PROVEN.
4. bounded PROOF: an MCTS-Solver / depth-limited αβ that treats booked children as PROVEN leaves; if it resolves
   the position within a node/time budget → PROVEN (+ `best_actions`). This is the wall-break — a shallow proof
   collapses to child lookups.
5. ESTIMATE: N bounded games / rollouts (a supplied agent factory, or a learned value head) → a win/draw/loss
   ratio → ESTIMATE (+ `best_actions` by estimated value + `n`).

Bottom-up minimax over booked children UPGRADES estimates → proofs automatically: a parent is `PROVEN_WIN` if any
child is a proven loss for the child's mover; `PROVEN_LOSS` if every child is a proven win for the opponent;
`PROVEN_DRAW` if the best child is a proven draw and none is a proven win. Each pass proves more and sharpens the
rest. **"Opening solved" = a root/opening position reaches `PROVEN_WIN` with a stored winning line.**

##### The builder
Extends today's `build_book`: enumerate a bounded, symmetry-reduced frontier; order deepest + hub-first;
`evaluate(...)` each; store the richer entry; RESUMABLE + ACCUMULATING (re-runs deepen coverage AND upgrade
ESTIMATE→PROVEN). Priority eviction keeps proofs over estimates and hubs over leaves.

##### Storage / operation optimisations (the user's third requirement)
- `best_actions` as a uint64 bitmask → O(1) store, fast set ops, cheap PV reconstruction.
- `status` as int8 → vectorised "all proven wins / losses / draws" and "100%-blocked = `PROVEN_DRAW`" filters.
- in-memory dict for build/play; columnar `.npz` on disk; a sorted-key + bisect read path if the book outgrows RAM.

##### Generic via the SolvableGame hooks
Reuses `canonical_key` / `ply` / `legal_actions` / `step` / `is_terminal` / `winner`, the exact `position_value`
/ `exact_optimal_actions` (proof rungs), and a NEW pluggable `estimator(game, state) -> (value, best_actions,
wdl)` (estimate rung; default = N bounded self-play games with a supplied agent, or a learned value head). A new
game plugs in exactly as tic-tac-toe / connect4 do.

##### How the deep model consumes it
The richer entry IS the distillation target: policy = `best_actions`, value = proven value or estimate, with the
`wdl` ratio + confidence as auxiliary signals. Proven entries give EXACT supervision; estimates give a GRADED
signal on the frontier the proofs haven't reached — so the model learns from the book everywhere, not only where
it is solved.

##### Measurable success criteria
- `optimality_verified_plies` (shipped in Tier 0) climbs toward full game length as PROVEN opening coverage grows.
- proven-opening count (ply ≤ K) and `book_coverage` (proven fraction of the reachable opening) climb per build.
- `wins_as_p1_vs_oracle` / self-play first-player-win climb toward 1.0 as PROVEN coverage reaches the root.
- tic-tac-toe stays the reference: the generic builder reaches 100% PROVEN coverage and a provably-optimal book
  agent (regression on the existing tests).

##### Honesty rails (Phase 6)
- An ESTIMATE is a BELIEF, not a proof — label it; never report an estimated opening as "solved".
- The approximate win/loss ratio is only as good as the estimator (bounded search / rollouts); it sharpens as
  proofs replace it.
- The wall is unchanged for PROOFS (a cold exact opening solve stays expensive); estimates exist to give the
  model useful gradient NOW while proofs accumulate bottom-up.

##### First buildable milestone — SHIPPED (2026-08-19, TDD; 168 boardgames + 1846 TS green)
- **Richer `Tablebase` entry** (`harness/tablebase.py`): `PROVEN`/`ESTIMATE` status + value + `best_actions`
  bitmask + confidence `n`, persisted columnar. **Backward-compatible**: `get()` unchanged; a new
  `proven_value()` returns a value only for PROOFS, so every exact consumer (`book_optimal_actions`,
  `book_value`, `play_until_decided`, the solver's `book=` short-circuit) was repointed to it and now ignores
  estimates by construction; legacy value-only `.npz` (committed books + the solver `.tt`) load as all-PROVEN.
- **Generic bounded-search estimator** `estimate_position` + the **evaluator ladder** `evaluate` (`harness/
  book.py`): terminal → PROVEN; PROVEN book hit → keep; FREE minimax over booked-PROVEN children → PROVEN; cheap
  exact (`exact_optimal_actions` ≤ `max_exact_empty`) → PROVEN; else a net-independent bounded-self-play ESTIMATE
  (+ `best_actions` + `n`).
- **Builder wiring**: `build_book(..., estimator=, max_exact_empty=)` — default path byte-identical (exact,
  value-only); estimator mode stores rich entries and **skips only PROOFS, re-evaluating ESTIMATEs** (the eager
  free upgrade). Proven end-to-end: ttt proves the WHOLE tree bottom-up from terminals with the estimator NEVER
  called (max_exact_empty=0), and the connect4 opening band yields ESTIMATE entries carrying `best_actions` + `n`.
- **`principal_variation`** reconstructs the raw line from stored optimal moves (Q2).

- **Distillation value-relabel** (SHIPPED): `oracle_distill_games` now takes the VALUE target from the book's
  PROVEN value where available, not the noisy self-play outcome — the fix for the opening value-label
  contamination that forfeits the first-player win (test: a proven opening value overrides the game outcome).
- **Proof leaves in MCTS** (SHIPPED): `MctsAgent(book=…)` backs up the EXACT proven/solvable value at a
  descended leaf instead of a rollout (`_proven_returns`; the root is always expanded so `act` can still rank
  moves). Reference rungs stay pure (opt-in). So book coverage pays off in PLAY, and a book-aware agent makes the
  estimator's rollouts sharper. (The safe "oracle-leaves" half of MCTS-Solver — proven-win/loss SELECTION +
  propagation, and the AlphaZero-core port, are the follow-up.)

- **Real-game coverage-loop PROVEN (SHIPPED, 2 durable connect4 tests + a live demo).** Correctness: a
  bottom-up midgame book's `book_optimal_actions` equals an INDEPENDENT from-scratch solve on every position of
  its principal variation (the book plays exactly what a fresh solver would). Loop: booking a subtree lifts a
  real line's `optimality_verified_plies`. Live demo on the oracle's 34-ply game: **14/34 verified with no book
  → 21/34 after booking the ply-13→24 midgame band** (200k positions, 187s), first-blunder None, and 9/9 of the
  line's booked midgame positions matched an independent solve. The opening (ply 0-13) stays honestly unverified
  — the 158s wall — so deeper coverage is the accumulator grind, exactly as the plan predicted.
- **Accumulator RUN (2026-08-20).** Bottom-up bands seeded progressively shallower on a real 23-ply near-perfect
  line, into one growing book — `optimality_verified_plies` climbed **monotonically 3 → 7 → 9 → 11 → 13/23**
  (booking from ply 16→14→12→10), with cost rising steeply toward the opening (**6s → 14s → 78s → 336s/band**;
  book 0 → 182k), the wall. Uses the existing `build_book` (banded + resumable) — no new code. Two honest
  findings, BOTH SINCE CLOSED (see next bullet): (a) reaching the deep opening (ply 0-9 here) is a compute-bound
  grind — multiprocessing the many-position bands would speed it, but a single hard opening solve stays serial;
  (b) `build_book`'s deadline is checked BETWEEN positions, not during a solve, so one cold opening solve
  (minutes) overruns the band budget (the ply-10 band ran 336s against an 80s deadline).
- **Parallel band solver + per-position cap + within-band deadline — SHIPPED (2026-08-20, 4 TDD tests).**
  `build_book(workers=N, max_position_seconds=S)` solves each ply-band across a spawn `ProcessPoolExecutor`
  (`_solve_frontier_banded`): positions in a band are independent given the deeper booked band, so each worker
  gets the current book snapshot (`_band_init`) for child short-circuits and solves are booked AS THEY COMPLETE
  (`as_completed`). Each solve is wall-clock-capped (`_run_bounded`, SIGALRM; a worker is the main thread of its
  own process so the cap holds there) — a hard position is DEFERRED (`None`), booked later once its children are
  cheap. Same-run RE-MEASURED parallel (8 workers, 3s cap) vs the sequential baseline above: **ply 12 78s→66s,
  ply 10 336s→186s (1.8×)** — the speedup GROWS with band depth (deep bands are compute-bound; shallow bands the
  pool spawn barely helps: ply 16 6s→10s), 0 deferred through ply 10 (no single ply-≥10 solve exceeds 3s; the
  336s was the *aggregate*). Closes (a). For (b): the per-position cap bounds any single solve, AND the
  deadline/`max_positions` budget is now checked WITHIN a band (not just between bands) — booking as solves
  complete and cancelling the not-yet-started ones — so a time-bounded run stops within ~`max_seconds` of its
  deadline. Proven by `test_build_book_respects_the_deadline_within_a_band` (a ply-14 band that grinds ~30s
  unguarded returns in <4s under a 0.4s deadline). Parity (`test_build_book_parallel_matches_sequential`):
  `workers=4` yields the byte-identical book to the sequential build. GOTCHA (bit us once): spawn re-imports
  `__main__`, so an ad-hoc driver script calling `build_book(workers>1)` at module top-level recursively spawns —
  the driver MUST sit under `if __name__ == "__main__":` (the real `run_build_book` tool path is inside a
  function, so it is unaffected; the parity test passes under pytest for the same reason).

- **Anti-drift ANCHOR — SHIPPED** (`neural._mix_training_set`, TDD): the exact distilled anchor is held at a
  FIXED fraction (`distill_fraction=0.34`, DQfD-style) of every training pass instead of concatenated into the
  8000-buffer where it diluted to ~5% — the fix for the net drifting off the optimal opening it was distilled on.
- **AlphaZero-core PROOF LEAVES — SHIPPED** (`AlphaZeroAgent(book=…)` + `_proven_value`, TDD): the net's search
  backs up the EXACT value at a booked/endgame-solvable leaf instead of the value head's estimate — truth
  propagates through the PUCT tree (completes the MctsAgent proof-leaves). Opt-in (no book + solve_endgame 0 →
  pure net, self-play unchanged); the deployed champion (solve_endgame 22) gets exact endgames in search.
- **MCTS-Solver SELECTION / PROPAGATION half — SHIPPED (2026-08-20, both cores, 9 TDD tests).** The leaf half
  only backed up exact values; this makes a proof PROPAGATE. Shared pure algebra in `agents.py`
  (`prove_node`/`child_move_value`/`mover_returns`, negamax with a win short-circuit) maintains a `_proven`
  overlay (position key → +1/0/-1, mover-relative): a node is a proven WIN the instant one child is a proven loss
  for the opponent, a proven LOSS/DRAW only when EVERY child is proven. In `MctsAgent`, each simulation seeds the
  overlay at its leaf and propagates deepest-first up the visited path; a proven node is then treated as a leaf
  (selection pruning), the root's proof is played outright, and the sim loop STOPS the moment the root is proven.
  Verified: the root becomes a *derived* proof (not a high average) and the move is optimal with a tiny budget
  (`sims=60`); the search terminates early (`sims_used < 100` of 500); a proof bubbles up through TWO plies from
  solved grandchildren the leaves alone can't reach; a drawn root proves via the all-children branch; and a
  20-position differential sweep vs the exact solver plays optimally everywhere (a sign bug would misplay). In
  `AlphaZeroAgent` the same overlay is populated as a search byproduct (writes only — descent/backup untouched, so
  the self-play visit-count policy π is provably intact: `run_search` spends its full budget over all legal moves)
  and CONSUMED only in greedy deployment (`temperature<=0`): an untrained net still plays a propagated proven win.
  Reference purity guarded both cores: no book + `solve_endgame=0` → the overlay is inert (`_proven == {}`, full
  sim budget), so the fixed-strength rungs and self-play dynamics are byte-identical.

- **Book-aware DEFAULT estimator — SHIPPED (2026-08-20, `book.make_book_estimator`, 2 TDD tests + wired into
  `run_build_book`).** Realises design decision (a): the estimator that grades an unprovable position is now a
  factory returning book-aware **MCTS-Solver self-play** (`MctsAgent(book=…, solve_endgame=…)`, a fresh agent per
  seat) — not the hand-rolled `HeuristicAgent` the tests used, and never a trained net. Its bounded games back up
  EXACT values wherever the book/solver reaches beneath the position, so the estimate is grounded in proofs and
  sharpens as coverage grows. Proven where its search reaches ground truth: on a solvable endgame the estimate
  COLLAPSES onto `position_value` exactly and its best set == `optimal_columns` (both weak-outcome semantics).
  `run_build_book` gained opt-in GRADED mode (`estimate_games>0` → build the estimator + pass `max_exact_empty`),
  so the graded opening book is producible from the tool/CLI with no minutes-long solve; proofs still win the
  evaluator ladder, estimates stay invisible to exact consumers (`proven_value` None).
- **Book → net SOFT distillation targets — SHIPPED (2026-08-20, `neural.book_distill_examples` + wired into
  `train_alphazero`, 3 TDD tests).** Closes the book→net learning loop: the net now learns from the WHOLE book, not
  only the exact late-solve corpus. For each covered position the policy target is uniform over the entry's stored
  `best_actions` and the value target is the entry's value — EXACT for a proof, the bounded-search belief (kept
  SOFT) for an estimate — so the graded opening the exact labeller can't reach becomes trainable signal. Proofs
  outweigh beliefs by whole-copy REPLICATION (`proof_copies` vs `estimate_copies`, the same oversampling the
  distill anchor already uses) — no per-example loss weights, so `augment_examples`/`_mix_training_set` are
  untouched. `train_alphazero(book=…, book_distill_positions=…)` folds these into the persistent distill anchor
  beside the oracle distillation. Uncovered / `best_actions`-less positions are skipped. NOTE: the payoff scales
  with opening coverage — until the accumulator fills the graded opening, the book supplies mostly late proofs
  (which the oracle corpus already had); the value lands once coverage climbs toward the root.
- **In-app grind launch path COMPLETE + robust (2026-08-20, TS + backend, 6 TDD tests).** The opening grind runs
  IN-APP via `Exploration → Start → autopilot → build-book → buildBook → run_build_book`. The new Python knobs
  (`workers`, `max_position_seconds`, graded `estimate_*`, `max_enumerate`) are now wired the whole way:
  `BuildBookParams` + `buildBook()` map them and ALWAYS set a per-position cap (`max_position_seconds=5`) by
  default so no pass can hang on a single opening solve (proven byte-identical to the plain build); the backend
  `buildBookActivity` forwards them; and the autopilot's build-book child takes its config from a new manifest
  `bookBuild` object (numeric knobs, validated). **Decision (user, 2026-08-20): each Start runs the GRADED opening
  grind** — the boardgames manifest `bookBuild` = `{seedGames:0, maxPlies:10, estimateGames:3, estimateSims:24,
  estimateSolveEndgame:14, maxExactEmpty:22, maxEnumerate:30000}`, so every Start extends the graded opening book
  (bounded ~120s, resumable), feeding the soft-distillation targets. Smoke-verified: the exact config adds ESTIMATE
  entries for the deep opening under a short deadline with no hang. (Requires the backend restart to load the new
  dist, per the usual convention.)
- **Autopilot INTERACTIVITY — Start now cycles end-to-end in minutes (2026-08-21, TS + backend + gauntlet.py, TDD).**
  A 12h+ Start exposed three stalls, all fixed so `Start` reliably reaches its results: (1) **endless improve** —
  `deriveAutopilotSignals` treated only `plateau`/`reached-target` as improve-done, so a champion that never
  plateaus (weak-promotion churn stops on `'budget'`) made the autopilot re-select improve every round forever
  (reached champion gen 60 without finalizing); now `'budget'` (the per-launch generation allotment spent) also ends
  the round → finalize. (2) **40-min improve generations** — the champion used the heavy manifest defaults; a new
  manifest `improve` field (`{maxGenerations, patience?, targetStrength?, hyperparams}`) gives the autopilot a
  bounded LIGHT budget (boardgames: 1 gen, az_iterations 2 / selfplay 8 / sims 80 / distill_games 16 / eval 20 →
  **~2 min/gen measured** vs ~40). (3) **9.5h rate step** — `rateModels` rated ALL **369** accumulated checkpoints ×
  7 rungs × 40 games incl. mcts@1000 / oracle@12; new manifest `rate` field + `RateModelsParams` bound it to the
  most-recent `maxModels` (16) with fewer `gamesPerRung` (12) and `maxReferenceSims` capping the mcts RUNGS
  (gauntlet.py `_rung_factory(max_sims)`; the model stays full-strength). Plus a viewer **Stop** button that aborts
  every live activity the process is running (parent autopilot + wedged children) — the panel had no way to stop
  itself. All three configs are manifest-driven (validated) and the autopilot's improve/rate children read them.
- **Results HONESTY — play-off numbers were noise + crowned the wrong player (2026-08-21, verified).** The play-off
  ran `gamesPerPair=4`, so BOTH the head-to-head win% AND the optimality verdict (`m = min(n,12)` = 4 games as P1
  vs a DEPTH-6 oracle) were statistical noise — a champion goes 4/4 by opening luck and gets a "✓ optimal" badge.
  VERIFIED at 30 games/pair (m=12): a real champion scores **0/12 as P1 vs the depth-6 oracle → suboptimal**, so
  the user's `alphazero·d1f978` "✓ optimal" was a 4-game fluke. And the viewer crowned "🏆 True winner (most games
  won)" = the highest round-robin win% (a non-optimal mcts), which for a SOLVED game (seat-noisy random openings) is
  the wrong arbiter. FIXES: (1) `gamesPerPair` default 4→**12** (reliable win% + m=12 optimality); (2) viewer
  `renderPlayoffResults` ranks by OPTIMALITY first (converts the first-player win) then head-to-head, crowns the
  optimal player (not the win% leader), shows sample sizes, and reconciles the three views — play-off *Optimal?*
  (the arbiter for a solved game), head-to-head *win%* (seat-noisy, secondary), Models *Strength* (a separate
  continuous gauntlet scale); (3) never overclaim — "✓ optimal vs oracle" → "converts the first-player win (beats
  the depth-N reference as P1, n games)"; the `book (optimal)` competitor → `book (exact where solved)` (its opening
  coverage is ~0%, so it is endgame-exact, not optimal). Self-play first-player-win is framed as CORROBORATING
  optimality, not proving it (a strong-imperfect P1 can also reach 100%).

#### SOLVE-IT PLAN — get a MODEL that plays Connect 4 perfectly (2026-08-21)

The game is solved (proven first-player win; we have an exact solver). What is NOT done: a trained MODEL that plays
perfectly. Honest current state — NO model has been tested against, let alone beaten, the EXACT solver; optimality
was only ever measured vs a depth-6 PROXY (weak, beatable in the opening); the opening book is 0% covered.

**Definition of SOLVED (measurable, no proxies):** a model M is optimal iff, as FIRST player from the empty board,
it beats the EXACT solver (perfect defender) in 100% of games — it converts the proven first-player win against
perfect play. Corroborated by: M's entire main line is proven-optimal (`optimality_verified_plies` = full game,
`first_blunder_ply` = none) and, as second player, M never loses a position that is a draw/win under perfect play.
Headline metric: `wins_as_p1_vs_EXACT_oracle == 1.0`.

**Phase 1 — HONEST MEASUREMENT (the optimality ladder).** (1) Make the play-off/optimality oracle configurable up
to the EXACT solver (`OracleAgent`), label it "oracle (exact)"; the viewer already gates the ✓/"solved" verdict on
an exact label (never on a proxy). (2) The LADDER: test M vs depth 6→8→10→12→exact, report the deepest rung it
clears (its "optimality frontier"). (3) `optimality_verified_plies` on M's ACTUAL line vs the EXACT reference — the
honest distance-to-solved. Cost: exact-oracle games are minutes/move from the opening → keep the round-robin on the
fast proxy, run the EXACT P1-conversion check on few games only.

**Phase 2 — MAKE THE MODEL PERFECT (the winning-strategy grind, not the whole space).** The first-player win lives
in the opening (ply 0-~14). KEY INSIGHT: don't prove ALL ≤14-ply positions — prove the WINNING-STRATEGY TREE (our
optimal move at each of our nodes + EVERY opponent reply we must answer), a bounded subtree the exact solver walks
from the root down. (a) Grind that PV+refutations tree into the book (the parallel, deadline-safe accumulator, from
the endgame back). Tracked by opening `provenFraction` climbing 0%→100% of the reachable winning line. (b) Teach
the model: a book-aware agent plays the proven strategy + exact endgames immediately (a perfect but lookup-bound
player); DISTILL the proven `best_actions` into the NET so it plays the perfect opening FAST without lookup (the
soft-target wiring). (c) VERIFY vs the exact solver as P1 → 100% = solved.

**Phase 3 — the real win + generalisation.** A distilled NET (fast, no lookup) that beats the exact solver as P1 =
a fast perfect model — the ML result the slow solver can't give. The processes (exact reference, verified-plies,
proof accumulation, distillation, the optimality ladder, "not-yet" honesty) generalise to unsolved games (chess:
no exact oracle, but a strong reference + verified-plies + self-improvement + honest reporting are the same
machinery). Getting it RIGHT on Connect 4, where we hold ground truth, validates the process before scaling.

**Milestones (measurable, NONE done yet):** M0 exact-oracle ladder + verified-plies-vs-exact (the next step) · M1
winning-strategy book `provenFraction` 0%→100% (the grind) · M2 book-aware agent converts P1 vs EXACT solver
50%→100% · M3 distilled NET converts P1 vs EXACT solver = 100% → SOLVED. Bottleneck is M1; the rest follows.

NEXT: run it — press Start in Exploration and watch opening coverage + the graded book grow across Starts; the
exact-proof accumulator (option 2, parallel band solver) remains available via `bookBuild` for a proofs-first pass.
**DESIGN DECISIONS (resolved 2026-08-19):**
- **(a) Estimator = bounded SEARCH (MCTS-Solver self-play), never the raw net value — SHIPPED (see above).** The
  book must be an INDEPENDENT reference that CORRECTS the net's opening errors; sourcing estimates from the net is
  circular (book ≈ net → distilling book→net teaches nothing). Search also works on day one for a new game with no
  trained net, shares the proof rung's substrate (degrades gracefully, sharpens as booked children accumulate).
  The net may LATER serve as the search PRIOR to strengthen it per-sim — but the stored estimate is always the
  search result, never the net's value.
- **(b) RECONSTRUCT PVs from the stored optimal moves; never persist explicit paths.** Shipped: `book.principal_
  variation(book, game, state)` walks the optimal set to a terminal (terminal / unbooked / cycle / max-len
  guards). A proven line reconstructs in full (its winning continuation was booked when it was proven); a thin
  region yields an honest partial line.
- **(c) HYBRID upgrade cadence: eager for the FREE upgrade, on-demand for the EXPENSIVE one.** An estimate whose
  children are now ALL booked is upgraded to a proof by a pure MINIMAX LOOKUP over those children — nearly free,
  and already part of the bottom-up pass, so do it EAGERLY (keeps proven-coverage monotone every build). An
  estimate whose children are NOT all booked needs NEW search to prove — that is real compute, so do it
  ON-DEMAND (a play-time query, a priority/regret-guided frontier expansion, or a focused build on a region), not
  speculatively every pass.

#### Tier 0 (research proposal) — SHIPPED (2026-08-19)
The three "free wins" that unblock measuring everything above: (1) the exact-endgame cutoff is now ON by default
for DEPLOYED/eval nets (`DEFAULT_AZ_SOLVE_ENDGAME=22`; the `AlphaZeroAgent` class default stays 0 so self-play
exploration is unaffected); (2) a generic, opening-inclusive `optimality_trace` (`harness/benchmark.py`) reports
`first_blunder_ply` + `optimality_verified_plies` (how deep the agent's ACTUAL line is provably optimal) — the
yardstick Phase 6 will move; (3) an `opening_value` metric (the net's value on the standard opening) exposes the
value-label contamination behind the forfeited first-player win. All TDD; tic-tac-toe proves the trace verifies a
full game once coverage exists.

#### Solver speed — the 158s opening wall (attacked 2026-08-19)

Research verdict (Pons tutorial + Numba, grounded in profiling): **the 158s is the pure-Python execution tax
(~30–100× vs C++; ours ~145K pos/s vs C++ ~12M), not algorithm — our solver is already at Pons's fastest**
(dynamic threat-count move ordering, non-losing pruning, tight weak `[-1,1]` window, and a symmetry-canonical TT
that is *ahead* of the tutorial). No move-ordering/TT tweak closes an 80× gap; PNS/df-pn is the wrong lever (it
proves one boolean, not the per-move values a labeller needs).

- **Pure-Python free wins — SHIPPED, 2.4×** (2.9s→1.2s on a ply-12 solve; guarded by a `_mirror`-vs-reference
  test + the brute-force cross-check): unrolled `_mirror` (was a 7-iter loop on every TT probe = 19%), a
  `_COL_MASKS` table, and native `int.bit_count()` popcount. Extrapolates the ply-10 wall ~158s → ~65s.
- **THE on-demand answer — AMORTIZATION via the bottom-up book (no dependency, generic).** Solve deep offline
  once; a shallower opening solve then reads its booked children and **collapses to lookups** — exactly why Pons
  ships an opening book. PROVEN deterministically (`test_connect4_solve_collapses_to_lookups_when_the_frontier_
  below_is_booked`): once the frontier one ply below is booked, a solve that searched 3008 nodes searches **0**,
  same answer. So "fast on-demand" = a good solver + a self-produced book beneath it, which we already have.
- **Numba cold-solve accelerator — ATTEMPTED, ABANDONED (2026-08-19).** Wrote a full `@njit` transliteration
  (`solver_numba.py`: bitboard negamax + array open-addressing TT, faithful to the pure algorithm). Numba's njit
  **could not compile the recursive `_negamax` in bounded time** — the compile stalled for minutes across six
  fixes (explicit signatures on every function, `cache=False`, removing the in-recursion `np.empty`, removing
  runtime-indexed global arrays), even though a *trivial* recursive njit compiles in 0.0s. Could not isolate the
  exact trigger without unbounded debugging, so the backend was **deleted and numba uninstalled** to keep the
  pure-Python baseline clean. If cold-arbitrary speed is ever needed: (a) a from-scratch ITERATIVE explicit-stack
  njit rewrite (uncertain, given Numba's resistance here), or (b) bind a compiled C solver (bitbully) behind the
  connect4 hook — both bigger, both optional. **The amortization book already delivers on-demand speed with zero
  dependency, so this is not on the critical path.**

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
