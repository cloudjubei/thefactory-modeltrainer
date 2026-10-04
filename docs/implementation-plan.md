# thefactory-modeltrainer — implementation plan

**Remaining work only.** Architecture: `docs/architecture.md`. Contract: `docs/model-training-standard.md`.
- **What was tried and how it came out:**
  - the claims register `examples/boardgames/hypotheses.json`;
  - the generated `docs/process-knowledge.md`;
  - the history in `docs/experiment-journal.md`.
- **Raw evidence:** `examples/boardgames/evidence/`, backed up to the private repo `thefactory-modeltrainer-evidence`.

## North star

> **★ Overarching rule:** build everything as **reusable, chat-reachable modeltrainer tools**: a harness capability +
> a `.factory/trainer.json` activity + a chat tool (`ModelTrainerTools` / backend `trainerTools`). Each is generic, so
> it serves other trainings too. A script is a prototype; work is done only when it is promoted and the script is
> deleted.

1. **The best generic pipeline for creating any model** (propose → run → judge → explore), with an evaluation
   engine that filters false positives: leakage control, pre-registered gates, multiplicity correction, adversarial
   verification.
2. **One fool-proof HYBRID process for any finite game.** It produces an agent that plays perfectly, plus **the most
   compact written description of a perfect strategy**: an ordered list of logical rules, a small net for what the
   rules do not cover, and a small exception table. All of it is certified, with evidence that the process is the
   best of those tried. Two tracks run in parallel and feed each other through one verifier:
   - **L** — the self-play learner;
   - **R** — written rules: discovered, extracted from nets, or known.
3. **Publish the evidence** (§D) once the corpus clears its pre-defined bar.

## Repo split

| Repo | Owns |
| --- | --- |
| **thefactory-modeltrainer** (this repo) | `ModelTrainerTools`; matrix planner; campaign loop; judge/propose orchestration; the viewer; the standard + `examples/`. |
| **thefactory-tools** | Generic infra: `ComputeRunner` seam (+ `RemoteComputeRunner`, `ContentAddressedDataCache`, pairing); work-item engine. |
| **thefactory-backend** | Activity registration + composition; app-view serving; PIN-pairing + runner WS channel. |
| **BlackSwan** (the trading repo) | Its `TrainerManifest` + additive `trainer/` CLI conformance. No Overseer code. |

## Rules for every experiment

- **Pre-register before the data.** A claim in the register, with:
  - a proof test (and an undecidable proof where a gate can fail);
  - the data file it will read;
  - pinned judge files;
  - measurable SUPPORTED / REFUTED / INCONCLUSIVE bars.

  A claim written after the data says so ("descriptive, registered after the data").
- **Measurement integrity:**
  - seeds have roles: a choosing seed never reports;
  - every rate carries n, seed and a CI;
  - a selected maximum is re-measured on a held-out seed;
  - comparisons are paired, final-vs-final;
  - power the claim before making it;
  - one training run is one sample.
- **Fingerprinted code is frozen while a run is active.** Runs refuse to write evidence if their training or
  measurement fingerprint changes. Edit in a scratch copy and copy back after the run exits, merging
  `hypotheses.json` and `evidence/manifest.json` rather than overwriting them.
  - Never mutation-test a module that a running job's workers import.
- **Every guard is mutation-tested through a tracked spec** (`tests/mutations/*.json`, run with
  `scripts/mutate.py --spec`), never by hand.
- **Capture everything:**
  - pilots and smoke runs that inform a decision → `evidence/` + a registered claim;
  - bugs and wrong turns → a dead-end claim proven by the regression test;
  - abandoned trials without a proof → `knowledge_map.json` `unproven_trials`;
  - every new claim → filed in `knowledge_map.json`, then `scripts/knowledge_digest.py`;
  - after new evidence → `scripts/backup_evidence.py --push`.
- **Licences:** no AGPL code or dependencies.

---

## 1. Board games — in flight: why does self-play stall on Connect-4?

**Where it stands.**
- At our budget the self-play process plateaus wrong in the opening, and its own search agrees with the wrong moves
  (h91, h92).
- More labelling search (h96), written certificates (h99, h100) and a value-aware stop (h102, h105) did not fix it.
- The net can represent the answer (h72), so the **training method** is the suspect.
- Out of bounds (user, 2026-10-02): bigger nets or days of compute as the answer to a failure, and rules supplied
  from outside the process.

**1.1 What D3 and D4 found: in the opening the training target itself is wrong.**
- The value head backs 42% of the nets' wrong moves, 57% at plies 0–4. At the empty board every value head rates the
  winning centre below at least two other moves (h106, h107).
- Where the value head ranks the right move higher one move ahead, the net's own 200-sim search still labels a
  wrong move: 85% of the time at plies 2–4. Where the value head backs the error, the search follows it 82% of the
  time (h108, h109).
- So the policy target is wrong at most opening errors. Fixing how the policy fits its labels cannot help. The lever
  is **the value signal the search reads**: its leaf evaluations come from the value head, which is worst in the
  opening.

**1.2 Value targets the net computes itself do not help (T14).**
- Tree-position targets from the search's root value and n-step targets from the lagged net both lowered the share
  of optimal opening moves: 86% → 82% and 75% (h111, h112 refuted; 4 paired seeds, 20 iterations, 6-ply tree).
- They echo: with search-value targets the value head backs 73% of the opening errors, against 36% without (h113).
- Lesson for the process: **new information has to come from real game outcomes**, not from the net's own estimates.

**1.3 The backward curriculum did not show a large gain either (T16), and the pilot is too noisy to see less.**
- `train_alphazero(backplay=...)` is built and does no harm on tic-tac-toe (h114). On Connect-4 it scored 72% optimal
  opening moves against base's 73% (1 win of 4; h115 refuted).
- The pilot measure is noisy: the unchanged process scores 86% and 73% on two sets of four seeds, and single nets
  range from 62% to 94% (h116). h111, h112 and h115 rule out large gains, not moderate ones.

**1.4 Calibration done: a fixed position set is ~16× less noisy per comparison (h117).**
- `harness/fixed_set.py` + `scripts/c4_fixed_set_readout.py`: every net plays its raw move at the same 5,142 exactly
  valued positions (the label cache's White positions, plies 0–8). Seconds per run, no solver.
- Nets of one arm spread 3.5 points; seed-matched differences 3.1. A paired pilot needs 7 pairs for a 3-point gain,
  16 for 2 points, 3 for 5 points.
- Post hoc, the three refuted treatments read +1.7 to +2.1 points over base on this set — a hypothesis, not evidence.
- The trade-off: the fixed set measures general play (only 114 positions at plies 0–4); P-START needs the first
  player's own opening tree, which stays the secondary reading with certification.

**1.5 T17 result: only n-step is a lead, and it is not yet proven (h118–h121).**
- Fixed-set share against base, 7 seed pairs, Holm across three: n-step +3.4 points (5 of 7, p 0.031: inconclusive,
  short of Holm's 0.0167); tree values −0.4 and the curriculum −2.3 (refuted). 0 of 28 nets certified through 6 plies.
- Two of the three post-hoc readings from the calibration vanished on fresh seeds (h121).
- Caveat: n-step's gain is on general play (mostly plies 6–8). On the own-tree opening it was worse in T14 (h112).

**1.6 The punishing opponent changed nothing (T19).**
- No harm on tic-tac-toe (h124, after the recorder fix, h123). On Connect-4: +0.06 points on the fixed set (p 0.48;
  h125 inconclusive), 71.1% vs 71.2% optimal at plies 0–4 (h126). 0 of 14 nets certified.
- Five fixes tried on this pilot footing (tree values, n-step, curriculum, exploiter, plus the stop variants): only
  n-step shows a lead (+3.4 points, p 0.031, uncorrected), and none moves the opening.

**1.7 T20 result: general play keeps learning, the opening does not.**
- Base recipe for 60 iterations, seeds 471–473, fixed-set reading after every pass. From passes 16–20 to 56–60:
  +5.5 points overall (h128 supported), +0.5 points at plies 0–4 (h127 refuted; one seed fell 75% → 68%).
- So longer runs would not rescue the opening fixes, and the real-event reward and n-step replication were **not**
  launched (the user's order was "if it makes sense"): both would be judged where nothing moves.

**1.8 H1 running: the hybrid's first measurement (h129, h130).** `harness/opening_table.py`, `harness/floor_hybrid.py`,
`scripts/c4_hybrid.py`.
- The process computes an exception table with an exact solve step: at the first player's positions in the hybrid's
  own tree before a horizon, an optimal move wherever the net's move is not optimal. The net plays the rest.
- On the ten base-recipe nets, horizons 3/5/7, the hybrid is certified one White ply past the table, so the net
  carries that ply. Judged at 3 and 5 (≥ 8/10); 7 and every table's size reported. Predicted refuted.
- Its output is the trade-off curve the hybrid needs: table entries against certified depth.
- Next, depending on it: put the table in the training loop (self-play starts from the table's opening, the
  strategy tree trains the net where the table stops), and size the table that certifies through 10.

**1.9 Stored knowledge (now H1) the process computes itself** (a solved opening/exception table counted in the description
length) remains an option for the hybrid, not the fix for 1.1.

## 2. Board games — next

**2.1 Depth-12 oracle frontier** (`scripts/strategy_frontier.py`; nets saved; independent certification).
- conv-64 (~53K params) first. ~3–5 days on CPU.
- Optional first: an MPS device option for refits, with a test that MPS and CPU fits agree.
- **Success:** certified through 12 plies, giving the depth 8 → 10 → 12 curve. Then report plainly whether a
  whole-game first-player certificate (~10^9 positions) is feasible on this machine.
- Optional: sharpen the depth-10 bracket (8,008, 20,616] with conv-20/24.

**2.2 Self-play Connect-4 at depth 12**, once the process certifies at depth 10.

## 3. Written rules (track R) — runs beside self-play

`harness/playbook.py` holds:
- the rule language — `IF ∃m: L1(m) ∧ … THEN play m`, ordered decision lists;
- the generic tactical predicates and tic-tac-toe geometry;
- `evaluate` (correct only when every recommended move is optimal) and `own_play_positions`.

Every rule is graded **position-sound** (optimal at any legal position) or **strategy-sound** (optimal on its own
play's positions only). Only position-sound rules may label arbitrary positions or override a net. Measure each
rule standalone and in decision-list order, by ply.

- **3.1 E2b — a richer predicate language for tic-tac-toe.**
  - Why: with the 11 current predicates no position-sound playbook exists up to 60 rules (h94 refuted, h95). The
    obstruction is across positions, so add predicates, not rules.
  - Candidates: layered MIGO-style predicates `wins_in_k` / `loses_in_k` (k = 1, 2, 3, bounded lookahead),
    `blocks_fork_by_threat`, line counts. Each predicate's lookahead cost is declared, so a rule cannot hide a solver.
  - Method: `harness/rule_search.py` (SAT; python-sat installed) and `scripts/minimal_playbook.py`.
  - **Success (pre-register):** a proven-minimal position-sound playbook with the richest predicate costing at most
    k = 3 plies of lookahead. Report the size against Newell–Simon's 9 rules.
  - Then the strategy-sound list per side.
- **3.2 E4 — extraction from a net.** Fit the rule language to the 938-param net's moves (VIPER-style,
  mistake-cost weighting), then verify with the solver. **Success:** as precise as E2's list.
- **3.3 Connect-4 rules.**
  - Move rules: win, block, never play under their threat, double threats via threat-space search.
  - **Track C — value certificates:** the pairing (claimeven + baseinverse) is built and sound (h98–h100); next are
    Allis's other rules with their compatibility check (see 1.1 (a2)).
  - Measure on three sets: a ply-matched sample, positions our agents visit, and the rules' own play.
- **3.4 E3 — hybrid on Connect-4.** The playbook in front of a smaller certified net at depth 8. **Success:** at
  most half of 1,576 params, with the rules' compute declared and bounded and reported beside the params.
- **3.5 Feeding self-play (R → L).** One pre-registered experiment each for labels, features and search pruning.
  A rule counts as feeding only if a fresh self-play agent measurably improves with it.

## 4. The hybrid process — definition of done

- **"Best", measurably:**
  - a certificate for every claim;
  - the smallest total description in bits (rules + net params × bits + table), each part reported; WeakC4's ~25 KB
    first-player Connect-4 strategy is the outside benchmark;
  - bounded compute per move;
  - minimality proven relative to the declared language or net family;
  - the best process wins on every ladder game with **zero game-specific process edits**.
- **Fool-proof requirements (each a test):**
  - claims pre-registered with pinned judges;
  - no game names in process code;
  - every game proven to terminate (a progress measure, checked on random play);
  - the verifier independent of what it checks;
  - fingerprinted, bit-reproducible runs;
  - a process fix re-runs every earlier game.
- **Game ladder (finite, with an exact oracle):**
  1. tic-tac-toe;
  2. Connect-4;
  3. **Kalah**, small variants first: game module, solver, termination test (seeds in pits or their distance to the
     owner's store falls every move);
  4. **Othello 6×6**: add a board-size parameter to `games/othello.py` (fixed at 8);
  5. **small Hex or Dots and Boxes**.
- **Order:**
  - finish Connect-4 (sections 1–3);
  - the hybrid on tic-tac-toe and Connect-4 (smallest certified total description vs pure net and pure rules);
  - Kalah → Othello 6×6 → Hex / Dots and Boxes;
  - freeze the process and write the "any finite game" runbook (inputs, certificates issued, stated limits);
  - promote it to chat-reachable tools (★ rule).

## 5. Other tracks (carried over unchanged — re-check the current state before starting)

- **Trading (BlackSwan)** — probe first, build second.
  - **Remaining probes:**
    - B1: on-chain flows and the funding-basis term structure (only if a cheap proxy motivates the mine);
    - B2: FOMC decision days, token unlocks / listings, whale moves, exchange outages;
    - B3: LLM-read Reddit crowd sentiment (Arctic Shift);
    - B4: silver/copper world models were inconclusive as partial models (industrial drivers missing).
  - **B5 (trigger: trading exhausted AND "safer AI-written code" is the stated mission):** one pre-registered
    defect-risk decision probe.
  - **B6, optional:** live handoff of the autopilot champion; Jupyter notebooks (scope kernel + security); a
    runner-channel WebSocket; remote git repoRefs.
  - Runtime step: set BlackSwan's overseer project `metadata.hasApp=true`, `metadata.appDir="app"`.
- **Engine / evaluation.**
  - Game suite: skull · flip7 · skull_king · for_sale, then terra_mystica · poker (CFR) · catan · altered.
  - A neural self-play core as a `model_name` lever; personas + league play for the luck games.
  - The specialist-vs-generalist-finetuned hypothesis; a BoardGameArena live-play bridge.
  - A unified "find the best model" process with a Models view. Open decision: an explicit manifest
    `compoundCores` vs deriving it from a `*_warm_start` lever.
  - The S9 leakage tail: per-split signature + overlap detector; move the trading fidelity predicate into
    BlackSwan.
  - Owner ratifications of the conservative gate defaults.
- **Tools to build:**
  - the `disentangle` capability, after which `scale_up.py` / `gumbel_ab.py` are deleted;
  - a game feature-probe tool;
  - a process registry;
  - Game-interface extensions as sequencing demands them.
- **Publication (§D).**
  - Evidence export as a reproducible package, and a reproduce-and-refute workflow.
  - Paper 1: BlackSwan no-edge (battery + refutation table).
  - Paper 2: the methods paper (null where it must be null, true positive where it must be positive).
  - Define with the owner: N families, K refuted papers, X% byte-exact reproduction, M markets, P regimes.

## 6. Open questions and trigger-blocked items

- **The provenance of `harness/solver.py`** (possible AGPL derivation) — the user's call. The C solver is ported
  from it.
- **`with_extra_data` projection rung** — build once at least one asset-specific series is mined.
- **Host → iframe `data:updated` push channel** — the viewer is poll-only.
- **Remote artifact / checkpoint storage** — once remote runs and live handoff both exist.
- **GPU + sandbox profile for training images** — `--gpus` is wired but unexercised.
- **Judge/proposer model transport** (`ModelSelection` API vs CLI) — revisit once the CLI inference stage lands.
- **Model comparison view** — trigger: a Connect-4 net certified for the whole game. A read surface over the
  existing leaderboard records.
