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

## Honesty rails

- The book is only as sound as its solver; a value-only entry is a proof of outcome, not of the line — playing
  optimally from it still needs the one-ply lookahead (cheap) or the endgame solver.
- For unsolvable games the "book" holds *beliefs* (agent-consensus), not proofs — label it as such in the UI.
- Report book coverage honestly (how much of the reachable opening is exact) so "optimal" is never overclaimed.
