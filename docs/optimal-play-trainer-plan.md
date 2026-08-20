# Optimal-Play Trainer — making a solved game *computably* optimal, and generalizing it

## The wall we hit (measured)

Reaching **provably optimal** play requires playing the **opening** perfectly. Exact solving costs
`~16ms` at move 20, `~1.4s` at move 12, but **~158 seconds per position at move 10** — the in-memory
transposition table (capped at 2M) thrashes on the opening subtree. So:

- exact **opening** labels for the net are infeasible one-at-a-time, and
- no fast/learned agent can be made provably optimal without help in the opening.

What already works and stays: **oracle-opening distillation** (fixes the net's broken edge-first opening →
centre-first; SHIPPED), the **exact endgame cutoff** (`solve_endgame` — provably-perfect endgame; SHIPPED),
the **Play-off** (objective who-wins + `wins_as_p1_vs_oracle` optimality gauge; SHIPPED).

## The idea (user's three levers) — this is exactly how Connect 4 was actually solved

A persistent **opening book + endgame tablebase**, **symmetry-reduced**, with **early game termination**.
Composed correctly these break the wall and make an *exactly optimal, fast* agent, and they generalize.

### Lever 1 — a persistent solved-position store (tablebase / book)

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

### Lever 2 — symmetry (mirror) canonicalization

Connect 4 is symmetric under left↔right reflection about the centre column. Canonicalize every key to
`min(key, mirror(key))`:

- **~50% fewer positions** to solve and store (a position and its mirror share one entry).
- game-theoretic value is **mirror-invariant**, so value-only lookups need **no** move-remapping.
- **the net gets it too**: canonical (or symmetry-augmented) encoding → the net learns a symmetry-invariant
  policy for free → ~2× data efficiency + consistency. (Generalizes to richer symmetry groups; see below.)

### Lever 3 — early game termination

During self-play, evaluation, the Play-off, distillation labelling, *and* inside the solver: the moment a
position's value is known (book hit), **end the game / cut the search** with that outcome instead of playing
it out.

- speeds up **everything** (fewer plies per game; solved subtrees never re-expanded);
- lets even the **net** agent "know" the result — a game that reaches a stored won/lost/drawn position ends
  immediately with the true result, so we neither waste time nor let the net misplay a decided position.

## How they combine → the deliverables

1. **Exact opening labels become feasible** (book-accelerated solving) → distillation trains the net on *true*
   optimal moves in the opening, not depth-limited approximations → the net approaches optimal.
2. **A deployable, provably-optimal, FAST agent**: opening = book (one-ply table lookup, instant) + endgame =
   solver cutoff (exact) + net as the fallback where the book is thin. No 158s solves at play time.
3. **Everything stays visibly testable** in the Play-off: `wins_as_p1_vs_oracle → 1.0`, champion self-play
   first-player-wins → 100%, and the book-agent shows as optimal.

## Generalization (Connect 4 is just the first game)

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

## Status — ALL PHASES SHIPPED (TDD; 137 boardgames tests green)

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

## Phased plan (each phase TDD, each ends with a measurement)

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

## Next enhancements (queued — documented now, to tackle soon)

These extend the shipped engine; not yet built.

### E1 — Player-colour (p1↔p2) collapse — INVESTIGATED, NOT A WIN (2026-08-19)

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

### E2 — Related follow-ups (candidates)

- **Grow real opening coverage.** Connect 4 opening coverage is honestly ~0% (the committed book is
  midgame/endgame only). Longer / offline `build-book` accumulation (bottom-up, symmetry-reduced) is what lifts
  `wins_as_p1_vs_oracle` toward a genuine optimality proof — the accumulator design is already in place, it just
  needs the compute budget to climb the frontier upward.
- **A stronger play-off yardstick.** The play-off oracle is depth-6 (endgame-exact, but beatable in the
  opening), so "optimality vs oracle" is a yardstick, not a proof. A stronger yardstick depends on the
  opening-coverage growth above (a deeper *live* oracle stays minutes/move in the opening — the wall).

## Phase 6 — Generic SELF-PRODUCED approximate book (current focus)

Reframes Tier 1 of the deep-research proposal. We do **NOT** import external databases (Tromp / bitbully) — the
whole point is a system that PRODUCES its own opening knowledge for ANY game, even when that knowledge is
incomplete. Connect 4 is only the honing example. The book graduates from an exact-only tablebase into a
generic store that mixes PROVEN and APPROXIMATE knowledge and upgrades one into the other over successive runs.

### The richer entry (supersedes the scalar Tablebase value)
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

### The evaluator ladder (exact → bounded-proof → estimate)
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

### The builder
Extends today's `build_book`: enumerate a bounded, symmetry-reduced frontier; order deepest + hub-first;
`evaluate(...)` each; store the richer entry; RESUMABLE + ACCUMULATING (re-runs deepen coverage AND upgrade
ESTIMATE→PROVEN). Priority eviction keeps proofs over estimates and hubs over leaves.

### Storage / operation optimisations (the user's third requirement)
- `best_actions` as a uint64 bitmask → O(1) store, fast set ops, cheap PV reconstruction.
- `status` as int8 → vectorised "all proven wins / losses / draws" and "100%-blocked = `PROVEN_DRAW`" filters.
- in-memory dict for build/play; columnar `.npz` on disk; a sorted-key + bisect read path if the book outgrows RAM.

### Generic via the SolvableGame hooks
Reuses `canonical_key` / `ply` / `legal_actions` / `step` / `is_terminal` / `winner`, the exact `position_value`
/ `exact_optimal_actions` (proof rungs), and a NEW pluggable `estimator(game, state) -> (value, best_actions,
wdl)` (estimate rung; default = N bounded self-play games with a supplied agent, or a learned value head). A new
game plugs in exactly as tic-tac-toe / connect4 do.

### How the deep model consumes it
The richer entry IS the distillation target: policy = `best_actions`, value = proven value or estimate, with the
`wdl` ratio + confidence as auxiliary signals. Proven entries give EXACT supervision; estimates give a GRADED
signal on the frontier the proofs haven't reached — so the model learns from the book everywhere, not only where
it is solved.

### Measurable success criteria
- `optimality_verified_plies` (shipped in Tier 0) climbs toward full game length as PROVEN opening coverage grows.
- proven-opening count (ply ≤ K) and `book_coverage` (proven fraction of the reachable opening) climb per build.
- `wins_as_p1_vs_oracle` / self-play first-player-win climb toward 1.0 as PROVEN coverage reaches the root.
- tic-tac-toe stays the reference: the generic builder reaches 100% PROVEN coverage and a provably-optimal book
  agent (regression on the existing tests).

### Honesty rails (Phase 6)
- An ESTIMATE is a BELIEF, not a proof — label it; never report an estimated opening as "solved".
- The approximate win/loss ratio is only as good as the estimator (bounded search / rollouts); it sharpens as
  proofs replace it.
- The wall is unchanged for PROOFS (a cold exact opening solve stays expensive); estimates exist to give the
  model useful gradient NOW while proofs accumulate bottom-up.

### First buildable milestone — SHIPPED (2026-08-19, TDD; 168 boardgames + 1846 TS green)
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

## Tier 0 (research proposal) — SHIPPED (2026-08-19)
The three "free wins" that unblock measuring everything above: (1) the exact-endgame cutoff is now ON by default
for DEPLOYED/eval nets (`DEFAULT_AZ_SOLVE_ENDGAME=22`; the `AlphaZeroAgent` class default stays 0 so self-play
exploration is unaffected); (2) a generic, opening-inclusive `optimality_trace` (`harness/benchmark.py`) reports
`first_blunder_ply` + `optimality_verified_plies` (how deep the agent's ACTUAL line is provably optimal) — the
yardstick Phase 6 will move; (3) an `opening_value` metric (the net's value on the standard opening) exposes the
value-label contamination behind the forfeited first-player win. All TDD; tic-tac-toe proves the trace verifies a
full game once coverage exists.

## Solver speed — the 158s opening wall (attacked 2026-08-19)

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

## Scaling doctrine — Connect 4 → Checkers → Chess → Go (what "solve" means, and what we'd need)

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

## Deferred phase — "Lean-Model Frontier" (make finding the best model the best it can be)

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

## Honesty rails

- The book is only as sound as its solver; a value-only entry is a proof of outcome, not of the line — playing
  optimally from it still needs the one-ply lookahead (cheap) or the endgame solver.
- For unsolvable games the "book" holds *beliefs* (agent-consensus), not proofs — label it as such in the UI.
- Report book coverage honestly (how much of the reachable opening is exact) so "optimal" is never overclaimed.
