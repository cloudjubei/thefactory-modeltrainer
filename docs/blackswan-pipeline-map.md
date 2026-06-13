# BlackSwan — pipeline map + holes

A condensed map of the BlackSwan trading pipeline, what's already been tried (from the
commented-out experiment log in `src/conf/*` + `results_hour.ods` + the owner's notes), and where
the real, under-explored holes are. Purpose: a single map of where we can still try things — so the
deep research and new experiments target genuinely new ground, not what's already been covered.

**Owner-validated findings (do not re-tread as if new):** more indicators did **not** help (and
dropping them is faster — `process_df_simple` is the deliberate result); **minute data behaves like
noise** and the RL models cope badly — daily/hourly is better; `lookback1` made it **much worse**;
adding a 15m layer **did nothing**; small nets win, big nets (8192,512) overfit and lose. 1m is the
source-of-truth; 1h/1d/1w should be **derived** from it (and derive-and-cached, not re-derived per run).

## The pipeline, stage by stage

| Stage | What it does | Tried (≈covered) | Holes (under-explored / weak) |
|---|---|---|---|
| **1. Data** | Binance klines; 1m → aggregate to 1h/1d via `process_fidelity` | minute vs hour vs day (hour/day win); single asset (BTC); ~2017–2024 | derive-and-**cache** from 1m (today re-derives ~1.5GB/run); no gap/dedup/continuity/NaN-inf sanitisation; one asset; one span |
| **2. Features / obs** | `process_df_simple`: price/volume %-changes, z-scores, price-to-max/avg, Pi-Cycle | 112 indicators (didn't help); 15m layer (nothing); lookback (1 worse, 32 used) | the **used** regime features are broken (`z_score_1m/1y` = constant 0 — QW1); taker order-flow unused (QW2); obs hard-clamped to [-1,1] (RB6) — but all moot if richer features genuinely don't help → the real question is **feature *selection*** (which few features actually carry signal?), never done systematically |
| **3. Action space** | discrete buy/sell/hold; TP/SL/trailing/no-sell variants | **extensively** (many TP/SL/trailing combos; no-sell good) | position **sizing** (only all-in/all-out); `combo_noaction=-1` synthetic-short bias |
| **4. Reward** | `combo_*` family (combo_all main), 16 multipliers | the family (intentional, multiple approaches) | **no explicit fee/turnover term** (RB4); reward-component **attribution** never measured (which multiplier drives behaviour?) |
| **5. Model** | ~15 RL algos | **heavily** — PPO/TRPO/REPPO/A2C/ARS, DQN family, distributional (Rainbow/IQN/QRDQN), recurrent LSTM, Munchausen | **NOT the bottleneck** — well-explored; small-net-wins/big-overfits is the signature of a low-information signal, i.e. the limit is upstream (data/formulation), not the algorithm |
| **6. Training** | episode loop, `learning_starts`, `iterations_to_pick_best=10` | episode counts, learning_starts 20000, customs | `reset_num_timesteps` reset every episode (QW3 — **fixed**); `iterations_to_pick_best` does **no selection** (RB2); single seed (RB3) |
| **7. Eval / selection** | single test window (2024 Q1), human picks best-of-10 on test, raw Sharpe | — | **THE BIGGEST HOLE.** selection-on-test leakage; single bull regime; single seed; no walk-forward; no multiple-testing correction → the historical "+50–60%" may be **over-fit / cherry-picked**, so we can't yet trust any ranking |
| **8. Process** | run via hub; records carry config+metrics+setupKey; `skipExplored` dedupes | the new hub loop | no **thesis-testing UI** (run a model/arch set → compare/sort/filter); no **ledger view** of what's been tried; historical `results_hour.ods` not imported |

## Where the real ground is (verdict)

The **model / feature / action / reward search space is well-explored** — more of it is re-treading.
The three genuinely under-explored, highest-value directions are:

1. **Evaluation rigor** — until selection-on-test leakage, single-regime, and single-seed are fixed,
   no result is trustworthy. This is the prerequisite for *everything*: you can't tell signal from
   luck today. (Walk-forward, held-out validation, multi-seed, deflated/probabilistic Sharpe.)
2. **Problem formulation** — RL on raw noisy price is a known-hard framing (minute=noise is a symptom).
   Worth researching systematic alternatives the project hasn't tried: supervised forecasting +
   meta-labeling / triple-barrier (López de Prado), regime/state conditioning, separating
   entry-signal from position-sizing, ensemble/stacking — vs. continuing to permute RL algos.
3. **Systematic process** — the thesis-testing UI + experiment ledger the owner wants: pick a model/
   architecture set, run the series, compare/sort/filter, and never repeat a setup. This is what turns
   the above into a repeatable engine instead of one-off sweeps.

The deep research (web, cited) should target #1 and #2 — *what does the evidence say works for
financial RL/ML and for trustworthy backtest evaluation* — explicitly excluding what's already been
tried here. #3 is a build task (the hub's comparison/ledger UI).
