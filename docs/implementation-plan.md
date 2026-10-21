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
- **Launch from the registered recipe, never by hand** (h138): `scripts/c4_solver_free.py --recipe MODULE:ARM`
  trains a judge's SPEC arm with its seeds and refuses hand-set training flags. New judges compare configs with
  `harness.recipe.recipe_matches`, which ignores only settings proven not to change training (`certify_depth`,
  `relabel_workers`); judges already pinned by claims stay as registered.
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

**1.8 H1 result: the net carries no ply on its own past an exact opening table (h129–h131).**
- Ten base-recipe nets, tables before ply 3/5/7: no hybrid certifies one ply further at any horizon (0/10 each).
- The table overrides 50% / 26% / 15% of its positions; the net is wrong at 23% / 12% / 12% of the ply it carries.
  It improves with depth, but its tree grows ~6× per White ply.
- A hybrid certified through 10 needs ~190 table entries per net (~12% of its first-player positions at plies 0–8).

**1.9 H2 result: training beside the table works (h132, h133).**
- Self-play from the table's frontier and the strategy tree walked from there: 81.6% optimal at the 284 positions one
  ply past the table, against 75.0% for the base process (+6.5 points, 6/7 pairs, p 0.016). Fewer failures at the
  never-trained ply 8 too (681 vs 865). Still no hybrid certified through 9 plies (0/14).

**1.10 H3 result: no judged gain, and the size accounting says the table wins through ply 9 (h134–h141).**
- Not judged: launched with 6 relabel workers against the registered 4, so the pinned judge refused (h138; h134,
  h135 inconclusive by protocol). With the run's own worker count both tests are still inconclusive: +1.2 points at
  ply 8 (5/7, p 0.14), and the ply-6 curve flattened (+0.6 points over the last ten passes; h139).
- H3 needs fewer table exceptions for a certified-through-9 hybrid on 6/7 seeds (253 vs 278; h141).
- **Every certified-through-9 hybrid is over 25× a table alone over the same tree** (~663K vs ~25K bits): the
  20,616-parameter net at 32 bits dominates; break-even is ~1 bit per weight (h140).
- So for the north star the open question is no longer "can the net carry ply 6/8" but **where a net can ever be
  cheaper than the table**: only where it carries far deeper than a table can afford (the tree grows ~6× per
  first-player ply), or at very low precision. **Next (user, 2026-10-05): in this order** — (1) net cost against
  table cost as the certified depth grows (`harness/description_length.py`), before any further training fix;
  (2) the two launch/judge fixes below; (3) the alternation fix and Kalah's registration (§4).
- Process fixes proposed by h138: build launch commands from the registered SPEC; exclude knobs proven not to change
  training (relabel workers) from the judges' recipe check, as `certify_depth` already is.

**1.11 Net vs table by depth: the table wins through 13 unless the net is ~4 bits a weight (h142–h146).**
- Accounting corrected (`harness/depth_cost.py`): a complete table is 3 bits per first-player position in walk order;
  h140's index charge superseded (h144).
- Break-even quadruples every two plies: ~0.2 bits per weight at depth 9, ~1 at 11, 3.7–4.35 at 13 (h144; judged bar
  of 4 split, h143 inconclusive). The net stays wrong at ~9–11% of its own tree, also on untrained plies (h145; h142
  inconclusive).
- The canonical table is not the smallest: the nets' own certified trees are up to 57% smaller, and against such a
  table break-even is ~1.3 bits (h146). **The fair baseline is the smallest certified table** — optimal moves chosen
  to shrink the tree, as WeakC4's opening does — and the cheapest leaves are discovered rules (§3.6), not a net.
- Next: §3.6 S1 (steady states at the frontier) with a tree-minimising table as the baseline.

**1.12 Stored knowledge (now H1/H2) the process computes itself** (a solved opening/exception table counted in the description
length) remains an option for the hybrid, not the fix for 1.1.

## 2. Board games — next

**2.1 Depth-12 oracle frontier** (`scripts/strategy_frontier.py`; nets saved; independent certification).
- conv-64 (~53K params) first. ~3–5 days on CPU.
- Optional first: an MPS device option for refits, with a test that MPS and CPU fits agree.
- **Success:** certified through 12 plies, giving the depth 8 → 10 → 12 curve. Then report plainly whether a
  whole-game first-player certificate (~10^9 positions) is feasible on this machine.
- Optional: sharpen the depth-10 bracket (8,008, 20,616] with conv-20/24.

**2.2 Self-play Connect-4 at depth 12**, once the process certifies at depth 10.

**2.3 Game-state compression (user, 2026-10-05) — research running.**
- Idea: encode a position by the moves that reached it — Connect-4 one column per move (3 bits; 0 spare as a
  marker), Kalah the hole sown (the rest follows from the rules); the player is implied by the rules. Build
  "rainbow tables" keyed by these sequences, filling in between as the process explores, so stored knowledge maps
  more and more of the game.
- Questions: how many bits per position each encoding needs (move sequence, board bitboard, a perfect index over
  reachable positions); how many move sequences collapse to one position (transpositions) and so how big a
  sequence-keyed table is against a position-keyed one; worst-case size to cover Connect-4 and Kalah whole, and
  whether that is ever worth computing; how it scales to Chess and Go; whether a compact encoding helps the net or
  the table in the description length (§4).
- Study done (`docs/research/game-state-compression.md`, 2026-10-05). Verdict: a sequence-keyed table is never
  smaller than a position-keyed one — measured on Connect-4, sequences per distinct position are 31× at ply 8 and
  404× at ply 11, growing ~2.4× a ply (counts match Tromp's OEIS A212693); Kalah(4,3) 31× by move 17. Storing every
  Connect-4 value takes ~1.1 TB perfectly indexed (Böck 2025 does it in 89.6 GB with BDDs); checkers, chess and Go are
  out of reach by 10^20+ bytes. A compact code is a storage format, not a useful net input.
- What carries over: the canonical-walk table needs no keys at all (3 bits an entry — `harness/depth_cost.py`), and an
  entry can be coded more cheaply still by its rank under a parameter-free move ordering (~1.1 bits an entry at
  ply 9, measured by the agent). Caveat: the agent's ordering used win/block/centre — win and block are generic
  (one-ply search), "centre" is Connect-4 knowledge and is out unless learned. E1 overlaps the depth study (§1.11).
- **E2 done (h149, h150 supported, 2026-10-06):** on every Kalah shape with a graph >= 1,000 positions a certified
  first-player strategy decides at <= 0.45% of them (Kalah(3,6): 1.57M positions, 202 decisions, 404 bits); choosing
  among value-keeping moves to minimise the tree halves it on 7 of 9 shapes (Kalah(3,6): 2,699 -> 202).
  `harness/strategy_size.py` is generic (any game with an exact `position_value`).

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

- **3.6 S1 — discovered steady states at the table's frontier (user, 2026-10-05; design after the research below).**
  - Why: WeakC4 plays all of Connect-4 for the first player in ~12 KB — an opening tree of <2,600 nodes whose ~1,700
    leaves are "steady states" (win; block; else a per-square priority map), found by search (GA, later SAT) and
    verified by brute force. So compact certified play exists past the opening, and its content was discovered, not
    written by hand. Our own pairing certificates found nothing through ply 8 (h99, h100): setups live deeper.
  - Question: at the frontier of the exact opening table (first-player positions at plies ~9–13), what fraction
    admit a small verified steady state in a generic language, at how many bits each — and does table + discovered
    rules beat table + net in `harness/depth_cost.py`'s accounting?
  - Research done (`docs/research/discovering-winning-setups.md`, 2026-10-05). WeakC4 at HEAD: 132 opening decisions
    + 522 leaves using 481 distinct priority-map diagrams, all found by search (GA, hill-climbing, SAT with
    counterexamples) and exhaustively verified; ~35–45K bits entropy-coded, 15–20× under our net. A median diagram
    (~70 bits) replaces a median 4,038-position first-player subtree. Leaves are 76% of stored first-player nodes at
    ply 8, 86–95% at plies 10–14. The language is our decision lists plus a cell-identity predicate and an
    "exactly one" firing rule; what we lack is **closed-loop certification** (a diagram must hold on every position
    it steers into, against all replies — `rule_search.py` only fits fixed labelled positions).
  - Agent pilot (not evidence; it reused WeakC4's own diagrams, which is outside knowledge for us): on uncurated
    winning frontier positions, win/block alone and plain claimeven/claimodd covered 0/120; with library reuse plus
    exact SAT, ≥37% at ply 10, ≥33% at 12, ≥43% at 14, 97% at 16.
  - Design (to pre-register): frontier positions at plies 10/12/14 (first player to move, winning); a library built
    only by our own search; exact SAT existence per position (found / impossible-in-language / timeout = not covered);
    entropy-coded diagram bits; the opening tree chosen by DP (cost = min(diagram, move + children)); table-only vs
    table + rules vs table + net in `harness/depth_cost.py` terms. The checker: exactly one move at every reached
    position, every reply covered, no draw/opponent four, sound memo across move orders and mirrors, never consults
    the solver; a second independent checker plus mutation tests. Caps: subtree ≤ 10^6 positions, SAT time/state
    budget. Vacuity guards: exclude positions win/block alone wins; "impossible in the language" is not a game value.
  - Before any external code (dsat, CaDiCaL) enters the repo: check its licence (no AGPL).
  - **Built and running (2026-10-06):** `harness/steady_state.py` (language + solver-free verifier), `harness/
    steady_search.py` (SAT with counterexamples, exact "impossible"; agrees with brute force on every 1-3 piece
    tic-tac-toe position), `harness/floor_s1.py`, `scripts/c4_steady_states.py`, frontier positions in
    `evidence/c49_frontier_positions.json.gz` (ply 10: 6,983; ply 12: 27,586; ply 14 derived: 97,072). Pilots (h152):
    nothing found at plies 10-12 within budget, 1/5 at ply 14 (a probe: 5/7) — so the registered run is judged at
    ply 14: h153 (coverage >= 30%), h154 (median compression >= 10x).
  - **Result (2026-10-06):** h153 supported — 25/50 non-trivial ply-14 positions (50%) get a verified steady state
    from our own search; 11/58 (19%) at ply 12 (h156). h154 inconclusive (median 5.7x) because value is bimodal: 11 of
    25 states replace 12-432x their bits (85 bits play 12,229 first-player positions), the rest govern a handful of
    positions (h155). Time budget was soft (h157), now a hard deadline (h158).
  - **S2 running (2026-10-06): complete certified strategies for subtrees.** `harness/strategy_builder.py` (13 tests,
    17/17 mutants): depth first, a position becomes a leaf (empty map, then the 200 most recent maps — a sibling's map
    covers a third of non-trivial siblings — then a 30 s search) or a table move chosen to make its children ready;
    `check` re-walks without any oracle. Pilot (h159): a ply-10 root got a complete certified strategy — 2,791 bits for
    31,701 first-player positions, 34x under its table — in 11 minutes. Registered on 8 ply-10 roots: h160 (>= 6/8
    complete in 2 h), h161 (median >= 10x).
  - **S2 result: h160, h161 supported.** 8/8 roots complete (slowest 28 min); median 36x (24-69x); in all 18,219 bits
    play 250,840 first-player positions to the end of the game, every strategy checked three ways.
  - **Gap to the whole game:** leaves start at ply 12 (searches find nothing earlier within budget), so a whole-game
    strategy would need a table over every ply-10 position its opening reaches — ~2.3K bits per ply-10 subtree times
    thousands of subtrees, far above WeakC4 (~35-45K bits), whose leaves start around ply 8. The lever is earlier
    leaves: a stronger search at plies 8-10 and opening moves chosen to reach steady positions sooner.
  - **Diagnostic (h162 inconclusive, h163 refuted):** at ply 10 a 6x budget finds 1/8 (21 min, 135K constraints) and
    14 levels find 0/8 — the limit is the search's efficiency. Options: (a) a faster exact search — Waffle3z's dsat
    (C++, CaDiCaL; licence to check) behind our own encoding and verifier; (b) a different search — WeakC4's genetic
    proposals checked by our verifier; (c) steer the opening into easier positions — build from ply-8 roots and let
    move choice look one step further for ready children.
  - **Local search (h166 refuted; h167 superseded by h170):** hill-climbing over maps (`harness/steady_local.py`)
    found 0/8 complete at ply 10. Its 'one line short' was the empty map, undefined at the root: the failure count
    stopped at the first failure, so any map giving a root move scored worse (h170). Patching those maps with <= 100
    exceptions gave no leaf (h168 refuted; h169 undecidable, h171 — its registered test read 'no leaf' as refuted).
  - **Fixed score (h172, h173 supported; h174 after the data):** a map is charged for the complete leaf it makes —
    map bits plus the exceptions it needs over every line (`steady_exceptions.needed`, one walk; exceptions cost the
    cheaper of a sparse index or a mask). On the same 8 ply-10 positions (`scripts/c4_steady_leaf_probe.py`, 30 min):
    4 pure steady states (80-98 bits for 1,300-5,208 positions; #705 in 23 s, SAT 21 min), the other 4 verified
    map-plus-exceptions leaves (8-1,850 exceptions); median 39x under the table (2.5-159x), 260x under the empty map's
    leaf. The score, not the language, was the ply-10 limit. Cost: the oracle on every newly reached position (up to
    2.4M cached, ~4 GB a worker).
  - **S3 result (h164 inconclusive, h165 supported):** from 4 ply-8 roots (leaves from depth 4, SAT leaf search,
    4 h each) 2 strategies completed — 647 positions in 205 bits; 318,473 in 34,115 bits — median 18.7x under their
    tables. The other two timed out with ~400 of ~760 leaf searches failing at up to 30 s each: failed searches, not
    the tree, use the time. A size-scored leaf never fails (it falls back to exceptions).
  - **SAT hand-off (h175, h176 refuted; h177 after the data):** on the 4 leaves left with exceptions, SAT started at
    local search's best map with its exception positions constrained first found nothing in 30 min / 200K constraints,
    and did the same work as SAT started cold (within 6%). Local search is the leaf engine at ply 10; option 2 (a
    better SAT encoding) loses priority accordingly.
  - **S4 result (h180, h181 supported; h182 refuted):** S3's study with leaves by size-scored local search (30 s,
    shared oracle cache, `harness/winning_cache.py`), a leaf kept with exceptions when >= 10x under its table. 3 of 4
    roots complete in 4 h (S3: 2) — #14026, which S3 could not finish: 657,615 positions in 99,451 bits — median
    18.3x (4.7x, 18.3x, 19.8x). Not smaller where both complete: #3591 64,047 bits vs S3's 34,115, 78% exceptions.
    Pilots: 4x completes fast at 6.2x with exceptions > 90% of bits (h178); 10x is slower (h179).
  - **Exception coding (h183, h184 supported):** exceptions coded in walk order (`harness/exception_coding.py`:
    flags only at contested positions, enumerative; undefined positions implicit; moves among the safe moves left)
    cut S4's exception bits by a median 59% — #3591 64,047 -> 33,782 bits (S3: 34,115), #14026 99,451 -> 62,188;
    compression 34.7x and 31.7x. Only ~37% of a leaf's positions are contested, ~1.5% of those flagged.
  - **Done 2026-10-07:** `Facts` bounded (default 2M positions, emptied at the limit; ~1 KB a position — the cause
    of S4's 9.6 GB worker); the builder charges exceptions in walk order in its accept test and size.
  - **S5 result (h186 inconclusive, h187 supported, h188 refuted; h185, h189 after the data):** leaves from ply 10,
    kept with exceptions at >= 30x, walk-order accounting. 2 of 4 complete within 4 h on a heavily shared machine;
    median 20.3x. #3591: 24,908 bits for 295,930 positions (35.6x) — the smallest for that root, > 20% under S3
    (34,115) and S4 in walk order (33,782). #2071 10 bits over S4's. The pilot at 10x settled at 10.5x (h185).
  - **Memory (done 2026-10-07):** all three per-worker caches are bounded — `Facts` 2M positions, the winning-moves
    cache 1.5M, `_SOLVED` (exact values, `scripts/c4_steady_states.py`) 3M — each emptied at its limit; ~5 GB a worker
    at most instead of 8.5+ GB.
  - **Open:** #11342 has timed out in S3, S4 and S5 alike; completion under a shared machine is not comparable
    across runs (S5's workers got a fraction of a core for much of its 4 h).
  - **Whole game, step 1 — W1 projection (h190 supported, h191 refuted; h192 after the data):** `harness/opening.py`
    walks an opening from the empty board to ply 8 with the label cache as oracle (the winning move whose replies the
    empty map most often wins; centre first): 204 moves, 259 positions already won, 671 frontier positions. 8 sampled,
    built as S5 with a 2 h cap: 7 complete, each 33-49x under its own table, but the whole strategy projects to at
    least 7.97M bits (12x the net, ~200x WeakC4) and ~415 h of builds; three of the eight carry 86% of the bits.
  - **Cheap size stand-ins fail (h193):** unwon positions two plies down and the solver's win distance do not
    order W1's sizes; size tracks the positions a strategy plays (~40x compression throughout).
  - **Next — the opening is the lever, but blocked:** choosing opening moves for small frontier strategies needs a
    size estimate, and the cheap ones fail (h193).
  - **P1 — bigger pure leaves (h194, h195 inconclusive; h196 after the data):** W1's three largest rebuilt with
    leaves allowed at the frontier root and long pure-map searches (20 min at ply 8, 5 min at ply 10), 3 h cap. #323
    completes at 4,304 bits (W1: 25,324) playing 52K positions (W1: 389K) — a map near the root changes which part of
    the game the strategy reaches. #316 and #230 run out of time after 54-56 searches (W1: 238-311). Fixed after the
    run: a search's budget is capped at the build's remaining time (it could overrun by up to its budget).
  - **P2 (h197, h198 inconclusive; h199 after the data):** only the root's 20-minute search, 30 s below. #323
    completes at 2,415 bits (W1 25,324; P1 4,304) — one table move and one excepted leaf covering 48K positions. #316
    and #230 make only 16 searches each in 3 h and stay unfinished (partials 2,443 and 1,665 bits).
  - **Profile (h200, supersedes h199):** leaf searches take > 95% of a build; near the root one search step walks
    up to 450K positions (65-209 s) and overruns its budget 2-3.5x. Fixed: the leaf walk honours the search deadline
    (`needed(deadline=...)`), so budgets are real.
  - **Map library (built 2026-10-08):** `Builder(library=...)` tries maps from earlier builds; a build pays only its
    references, library bits are paid once; `found_maps()` feeds the next build. The strategy script builds roots in
    waves (`wave_size`), each wave starting from every map the earlier waves found.
  - **W2 result (h201 supported; h202, h203 inconclusive; h205 after the data, correcting h204):** 6 of 8 complete;
    every frontier strategy smaller than W1's, the three largest more than tenfold (#323 1,264 vs 25,324); the
    projection's lower bound falls from 7.97M to 533K bits — under the net's ~660K, undecided while #316 and #230 are
    unfinished. The library is barely used (2 of 61 leaves): the root searches carry the gain.
  - **W3 result (h206 inconclusive, h207 refuted; h209 after the data):** from W2's library, #316 finishes at 9,057
    bits — nearly tenfold its 2-hour partial — and #230 passes 5,446 unfinished in 6 h. Partials undercount badly;
    the whole first-player strategy projects to at least 1.62M bits: 2.5x the net (~660K), ~40x WeakC4, ~5x under W1.
  - **Build speed, step 1 (h208):** `harness/native_leaf.py` + `harness/c4leafwalk.c` — the leaf walk in C for
    Connect-4, identical results to the Python walk (5 tests, 15/15 C mutants) but only 2-7x faster warm and 1-3x
    cold: the exact solves of newly reached positions dominate. Wired in (2026-10-08): `local_search(walker=...)`,
    and a search spec's `"walker": "native"`; runs record the C sources' hashes.
  - **Fewer solves (h210 supported, user's choice 2026-10-08):** the C walk solves only the moves that decide — the
    map's move where it gives one, otherwise columns up to the lowest winning one — caching single move results:
    cold walks 3.4-8.5x faster again (median ~6x), results unchanged (17/17 C mutants). Against the Python walk a
    cold walk is now 5-17x faster.
  - **W4 result (h211, h212 refuted; h213 after the data):** the C walk leaves a time-bounded build's time unchanged
    (#316 15,851 s vs 15,637; 8,804 bits vs 9,057) and #230 unfinished (13,083 bits); both builds make fewer
    searches than W3. Searches run to their budgets, so walk speed becomes evaluations, not wall time; the build's
    time goes outside the searches — the builder's own Python walks (covering checks, simplify, walk-order coding)
    are the suspects. Profiling a W4 build (2026-10-09) to place it.
  - **CORRECTION (h216, supersedes h213):** a budget entry covers its depth and every depth below. P2-W4's
    [[0, 1200]] gave every search 20 minutes (P1's [[0, 1200], [2, 300]]: 5 minutes at every depth from ply 10);
    the "root-only long search" was never run, and W4's build time was its searches' own budgets. Specs must close
    the root budget explicitly: [[0, 1200], [2, 30]].
  - **C walk wrapper (h214, h215):** exception keys built only when read and buffers reused — warm walks ~16x
    faster again (40-135x the Python walk).
  - **W5 stopped (h220; h217-h219 superseded unjudged):** #362 became one pure steady state at the ply-8 root —
    90 bits for 13,934 positions in 39 s — but three first-wave builds overran the 2 h cap by over an hour: the
    builder's own Python walks (covering checks, simplify, walk-order coding) ignored its deadline, exposed by the C
    walk's fast, large leaves. Fixed: all of them stop at the build's deadline, and maps are checked by a C verify
    (`native_leaf.verify`, one C walk shared with the leaf walk; identical to the Python verify, counts included).
  - **W5b result (h221, h223 supported; h222 refuted; h224 after the data):** all 8 complete and certified within
    2 h — the longest, #230, in 56 minutes — in 30% of W2's total build time (9,562 s vs 32,232). Six of the eight are
    one map at the frontier root (90-470 bits); #316 takes 6,675 bits in 35 minutes (W3: 9,057 in 4 h 21 min). Where
    W2 or W3 also finished, W5b is 44% smaller in total (7,766 bits vs 13,842; 6 of 7 smaller). But the projection is
    2.42M bits, over W3's 1.62M lower bound: #230, unfinished in W1-W4, completes for the first time at 21,054 bits —
    73% of the sample's bits, 72% of its own bits exceptions — where W3 counted a 5,446-bit partial. One position sets
    the 8-root mean. The library offered 39 maps to the second wave and none was used (W2: 2 of 61 leaves). h221-h223
    read post-hoc only because they were registered in the second the run started; the register now waits out that
    second (t34).
  - **W6 result (h225, h226, h227 supported, pre-registered; h228, h229 after the data):** the next 32 of W1's
    seeded frontier positions, built as W5b without the library: all 32 complete and certified within 2 h (longest 51
    minutes). The whole game projects to 4.31M bits, 95% bootstrap interval 2.10M-6.85M — all above the net's ~660K;
    W5b's 8 had read low. The 4 largest carry 57% of the bits. The frontier splits in two: 16 of 32 take under 200
    bits (one map at the root, near enough) and hold under 1% of the bits; the other 16 carry the size, 84% of all bits
    exceptions. The first 40 frontier positions project to 3.93M bits; building all 671 would take ~210 h.
  - **The opening lever is shelved at ply 6 (h229):** of the 13 opening decisions above the 12 sampled positions of
    >= 5,000 bits, 6 have one winning move, and every alternative at the other 7 leaves 7 unwon replies — never fewer
    than the move chosen (`scripts/c4_opening_alternatives.py`). A swap trades one heavy position for seven unbuilt
    ones, half of them heavy on W6's rate.
  - **The lever is now the exceptions.** The builder keeps the FIRST leaf with exceptions that is `accept` times under
    its table (30x) without asking whether a move and smaller leaves below would cost less; five of W6's heavy
    positions are exactly that — one root map carrying 133-1,671 exceptions.
  - **A1 result (h230, h231 supported, pre-registered; h232 after the data):** the four heavy positions rebuilt with
    `accept` 100x all complete within 2 h and together take 42,464 bits — 58% of W6's 72,883. Exceptions fall from
    84-97% of the bits to 7-31%; maps take their place (45-173 per position, ~70 bits each); builds 38-82 minutes. But
    the threshold cuts both ways: #594 32,544 -> 6,207 bits, #535 30,009 -> 12,857, #95 6,453 -> 5,307, while #565
    grows 3,877 -> 18,093 — its one root map with exceptions covered 43,562 positions, the split reaches 321,970 (its
    table moves are chosen by readiness, not size). The smaller build per position: 28,248 bits, 39% of W6's.
  - **A2 result (h233, h234 inconclusive; h235 after the data):** `Builder(choose_by_size=True)` — at every leaf with
    exceptions that clears 30x, build the split below too (recursively) and keep the smaller. Where it finished it beat
    both thresholds: #95 3,481 bits (best threshold build 5,307), #535 6,637 (12,857); the three finished total 13,995
    against 22,041. But it does not fit 2 hours: #565's split never finished (the leaf was kept — W6's 3,877) and #594,
    with no leaf at its root to fall back on, did not finish (6,376-bit partial).
  - **A3 result (h236, h237 supported, pre-registered; h239 after the data, correcting h238):** `Builder.improve()`
    — W6's build first, then the costliest leaves with exceptions split while time remains, each split kept only if
    the whole strategy shrinks. All four complete in 16,740 bits: 23% of W6's 72,883, 59% of the best fixed threshold
    per position (28,248). Each queue ran dry within 15 minutes of completion; the longest build took 45 minutes.
    Completed -> improved: #95 6,453 -> 2,919, #535 18,081 -> 6,559, #565 3,877 -> 1,253, #594 34,210 -> 6,009.
  - **W7 result (h240, h241, h242 supported, pre-registered; h243 after the data):** W6's 32 positions rebuilt as W6
    plus `improve`: all 32 complete (longest 69 minutes). The whole game projects to 1.47M bits — 34% of W6's 4.31M —
    its 95% interval 0.78M-2.30M still entirely above the net's ~660K. Exceptions fall from 84% to 36% of the bits.
    Two parts: the same settings' builds completed at 154,777 bits against W6's 205,322 (run-to-run variance, a tenth
    to 2.4x per position), and improving cut them to 69,847 — alone, 3.25M -> 1.47M. Four positions (#95, #271, #374,
    #512) kept no split: one root leaf each, whose first split was larger one level down, so nothing below was tried.
    The whole frontier would take ~229 CPU-hours.
  - **A4 result (h244, h245 supported, pre-registered; h246 after the data):** `improve(lookahead=True)` — a split not
    smaller at once is judged after its own new leaves are improved, kept or undone with them. Measured inside each
    build after plain improvement ran dry: a further 21% (37,848 -> 29,989 bits over eight positions); both positions
    plain improvement could not change were cut (#374 949 -> 514, #271 1,324 -> 720). Build, plain and lookahead
    together cut the completed builds by 63%; every build was done within 72 minutes of its 2-hour cap.
  - **A5 result (h247 refuted, pre-registered; h249 after the data; h248 a dead end):** re-searching the costliest
    leaves with exceptions for 120 s each in the idle time, then improving with lookahead once more, cut a further
    0.56% (28,892 -> 28,729 bits): only 29 leaves with exceptions were left to try across the eight (3 kept).
    Exceptions are now 14% of these strategies' bits; maps are 64%. No map is paid twice and 148 of 233 maps already
    serve several leaves; coding maps as edits of the previous map costs 18% more (h248). `research` stays in the
    builder, unused by default.
  - **W8 result (h250, h252, h253 supported; h251 refuted — all pre-registered; h254 after the data):** W6's 32
    built, improved and improved with lookahead: all complete; lookahead cut 16% of what plain improvement left inside
    the builds; the whole game projects to 1.57M bits (95% interval 0.83M-2.41M), above the net — and above W7's
    1.47M, because this run's builds completed 20% larger (186,359 bits against 154,777): run-to-run variance is
    bigger than the pipeline's last gains. The smaller of W7's and W8's certified strategy per position projects 1.26M.
    Exceptions are 22% of the bits, maps 58%. At W8's pace the whole frontier is ~300 CPU-hours.
  - **Decided (user, 2026-10-11): the leaf language first.**
  - **L1 result (h255, h256 refuted, pre-registered; h257 after the data — a dead end):** maps of 4 levels at 2 bits
    instead of 8 at 3: only 23 of 32 complete in 2 hours and the 32 take at least 276,519 bits, 3.7x the 8-level runs
    — 2,132 maps against W8's 586, larger on 25 of 32. A map's expressiveness, not its cell cost, binds.
  - **L2 result (h258, h259 supported, pre-registered; h260 after the data):** `Facts(ties="lowest")` — the first
    level holding any safe move decides, with its lowest move. All 32 complete (longest 47 minutes) in 14,666 bits: a
    fifth of the exact rule's two runs (69,847 and 74,714), with 90 maps instead of 586. The whole game projects to
    308,603 bits, its 95% bootstrap interval 177K-469K entirely under the net's ~660K — the first projection below the
    net, but on positions built five times before. Builds 7.7 CPU-hours (whole frontier ~161 CPU-hours).
  - **W9 result (h261, h262 supported, pre-registered):** the next 32 positions of W1's seeded order, never built
    before, with L2's rule and pipeline: all complete (longest 61 minutes) in 12,908 bits with 104 maps; the whole
    first-player strategy projects to 271,740 bits, its 95% bootstrap interval 137K-476K entirely below the trained
    net's ~660K. Whole frontier ~153 CPU-hours (~1.6 days on 4 workers). WeakC4's ~35-45K is ~7x below.
  - **Decided (user, 2026-10-11): more language first.**
  - **L3 result (h263 supported, h264 refuted, pre-registered; h265 after the data):** two-ply safety
    (`Facts(safe_depth=2)`) — all 32 complete, but 15,343 bits against L2's 14,666 (+4.6%): exceptions fall from 34%
    to 16% of the bits while maps rise from 90 to 118; builds a fifth slower. Kept in the language, off by default.
  - **Maps carry across the frontier (h266, after the data):** under L2's rule 35% of W9's fresh leaves are won from
    by one of L2's 90 maps found on other positions (none of the leaves with exceptions; 3 of 32 roots outright).
  - **W10 result (h267, h268 supported; h269 inconclusive — pre-registered; h270 after the data):** W9's 32 rebuilt
    from a library of L2's 90 maps (8,618 bits paid once): all complete; the whole game projects to 158,844 bits —
    58% of W9's 271,740, about a quarter of the net, ~4x WeakC4; 13% of leaves use the library; maps 104 -> 55. But
    most of the cut is one position (#638 4,322 -> 356, no library leaf) and the rest is 21% smaller: whether the
    library helps as a search start, by covering, or not at all (variance) is not separated.
  - **Decided (user, 2026-10-12): build the whole frontier.** F1 (h271-h273): all 671 positions with L2's rule, in
    waves of 32 from a growing library seeded with L2's 90 maps; ~38 hours on 4 workers, resumable (`--resume`,
    progress, after the run, in checkpoints/runs/ with its log — evidence/ holds managed evidence only).
  - **F1 result (h271, h273 supported; h272 refuted — pre-registered; h274-h276 after the data):** all 671 frontier
    positions certified complete: a certified first-player strategy from the empty board in **548,424 bits** (opening
    1,075 + seed library 8,618 + positions 538,731), below the net's ~660K, ~13x WeakC4's ~40K — and twice W9's
    projection, above its 95% high. Sampling in a heavy tail, not drift: the heaviest 10% of positions hold 58% of the
    bits, the median takes 154; the library carried 9.9% of leaves; a 32-position bootstrap high missed this
    frontier's mean 11% of the time, so projections from samples are guesses, not bounds.
  - **Strategic note:** the whole game is now built and below the net; the gap to WeakC4 (~13x) sits in the heavy
    tail (67 positions, 311,823 bits) and in maps (292,054 of the positions' bits). Next fork is the user's: the
    heavy tail (rebuild the heaviest with the 3,196-map library / best-of-runs), the leaf language (maps), or the
    opening (which frontier the heavy positions come from).
  - **Decided (user, 2026-10-15): the leaf language next.** Maps are 292,054 of F1's 538,731 position bits. Judge
    candidates on the heaviest positions plus a stratified sample, never a random 32 alone (h276).
  - **L4 result (h279, h280 refuted — pre-registered; h281, h282 after the data): a dead end.** Preferring forcing
    moves (`Facts(forcing=True)`, zero bits) completes 4 of 32 within 2 hours and takes at least 160,659 bits against
    the control's 130,061 (all complete). It starves the leaf search (maps in 12% of searches against 44%), so builds
    run out of time splitting. The control rebuilds F1's 32 a tenth smaller without any library (run-to-run variance).
    Kept in the language, off by default.
  - **M1 result (h283 refuted — pre-registered; h284 after the data): map coding is a dead end.** Exact cheaper codes
    for F1's 3,196 maps (`harness/map_coding.py`: chosen cells, levels by rank or by count) save 2.5% with a per-map
    selector, 3.2% with one code for all. Map bits are set by how many maps and levelled cells there are — the next
    gain must come from fewer or simpler maps (rule or search), not from writing them better.
  - **L5 result (h285 supported, h286 inconclusive — pre-registered; h287 after the data): a real gain.**
    `Facts(unlevelled="last")` — unlevelled safe moves a last level — completes all 32 in 109,017 bits against
    L4's control's 130,061 (0.84; the registered bar was 0.80). Smaller on 24 of 32 (sign test p = 0.003); 28% fewer
    levelled cells, 22% fewer map bits; the heaviest 16 at 0.80. Next: the user's choice (e.g. carry L5 into the whole
    frontier, or stack another rule on it).
  - **F2 result (h288, h289 supported — pre-registered; h290 after the data): the whole game in 455,959 bits.**
    F1 with L5's rule: every position complete; 0.83 of F1's 548,424 (maps −24%, exceptions −21%; F1's heaviest 67 at
    0.72). The heavy tail still holds 57% of the bits. ~11x WeakC4's ~40K.
  - **B1 result (h291 supported, h292 refuted — pre-registered; h293 after the data): longer searches are a dead
    end.** 120-s deeper searches complete all 16 heaviest positions but take 1.30 times a 30-s control's bits: they
    settle near the root for leaves that clear the 30x acceptance threshold with thousands of exceptions (9,094
    against 1,972). The lever is the acceptance rule (what a leaf may cost against a split), not search time.
  - **A6 result (h294 supported, h295 refuted — pre-registered; h296 after the data): a stricter bar is a dead end.**
    Leaves kept only at 100x: 15 of 16 complete, but 1.34 times the control's bits (140,404 against 104,860; smaller
    on 3). The first complete build is under half the control's (156,348 against 348,245 where both complete), but
    improvement — which compares a leaf with its split — then cuts the control by 73% and A6 by 23%: the 30x bar
    hands it leaves to split selectively, 100x splits blindly at build time. Exceptions saved 3,436 bits; maps and
    splits cost 38,980 more. Search time and the acceptance bar are both settled; the heavy tail needs something else.
  - **D1 result (h297 supported — registered before its measured run): which zero-bit default.** On 4,523 own
    positions with several safe moves sampled from L5's certified leaves and solved, the lowest move keeps the win at
    93.2% (random 92.8%); plain reorderings do no better, but "tactics" — first a move that wins by force in two, then
    one that does not lose by force in two, else the lowest — misses 200 against 308 (0.65). It becomes L6's rule.
  - **L6 result (h298 supported, h300 inconclusive — pre-registered; h299 superseded; h301 after the data): a
    better default is no lever on the heavy tail.** Ties broken by tactics: 32/32 complete, 102,289 bits against L5's
    109,017 (0.94, smaller on 19, larger on 11, p = 0.2 — within run-to-run swing). Maps −13%, exceptions +13%; the
    light 16 at 0.66, the heavy 16 at 0.97. D1's 35% fewer default misses did not become a smaller heavy tail.
  - **K0 result (h302 inconclusive — pre-registered; h303 after the data): two-phase maps (user's choice: richer
    leaves, paid in bits).** Built: `SWITCH`/`phase_cell` maps (Python + C walk), phase moves in the local search
    (judged without the phase's fixed cost), `phases` in the builder (flag bit a map). Piloted on 45 of L6's heavy split
    positions, same 30-s search with/without: 20 smaller, 18 larger; gains only on leaves too large to keep (total
    0.53, a few 17K-613K-bit leaves); under 5,000 bits larger on 17 of 26.
  - **K1 (h304, h305, h306; running since 2026-10-21 14:55Z, user's choice: run it anyway):** L6 with two-phase maps,
    L4's 32 positions, against L6 (102,289 bits; heavy 16 94,250): ≥28/32 complete; ≤0.9 of L6 in total; ≤0.9 on the
    heavy 16. ~7-8 h. Run files in checkpoints/runs/c49_k1*.

## 4. The hybrid process — definition of done

- **"Best", measurably:**
  - a certificate for every claim;
  - the smallest total description in bits (rules + net params × bits + table), each part reported; WeakC4's ~12 KB
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
  3. **Kalah**, small variants first. Built: `games/kalah.py` (rules of Irving, Donkers & Uiterwijk 2000; exact
     solver memoised on the counters in play; `progress` rises every move, checked on random play in six shapes) and
     `harness/kalah_paper.py` (the paper's tables and the judge, h136, h137). Solver memory: Kalah(5, 2) needs 2.4M
     memo entries (16 s); Kalah(6, 2) passed 10 GB unfinished. Before it enters the process:
     - done 2026-10-06: the certifier, the search's proof step and the n-step target compare movers (h148);
       alternating-game training is bit-for-bit unchanged (h147), so Connect-4's new era 6eedd699d12d and the old
       33939d5e2d76 train identically. Registered as `kalah` (6×4), `kalah4x3`, `kalah3x3` (each shape its own name;
       the fingerprint maps a name to its module). Smoke: a Kalah(3,3) net trains with self-play and relabel workers.
       `harness/exact_values.py` still builds Connect-4 states — generalise it before the hybrid runs on Kalah;
     - next on Kalah: the hybrid on `kalah4x3` (exact solver in seconds), then larger shapes — a solver for the
       6-hole game needs a compact memo (the paper's databases rank positions by counters in play, 4 bits each);
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
