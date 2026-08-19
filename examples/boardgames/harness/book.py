"""Incremental, resumable OPENING BOOK — Lever 1 (persistent solved-position store), Lever 2 (symmetry-canonical
keys), and the bottom-up wall-break (harness/solver.move_values reads the book for solved children).

The book maps a position's SYMMETRY-CANONICAL key -> its exact WEAK game-theoretic value (win +1 / draw 0 /
loss -1) from the player-to-move's perspective. Value-only + one-ply lookahead is enough to PLAY optimally
(`book_optimal_actions`); the same store lets the solver end a search early on a hit, and lets a game stop the
moment it reaches a decided position. Built BOTTOM-UP (deepest frontier first) so every shallower solve reads
its already-booked children — the measured 158s opening solve collapses to table lookups. Bounded + resumable:
each run extends the reachable frontier and persists it (a project-committed .npz), so coverage accumulates.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path
from typing import Callable

from harness.game import Game, State
from harness.solver import canonical_key, move_values, to_bitboard
from harness.tablebase import ESTIMATE, PROVEN, Entry, Tablebase

BOOK_DIR = Path(__file__).resolve().parent.parent / "books"


def book_path(game_name: str, book_dir: str | None = None) -> str:
    return str(Path(book_dir or BOOK_DIR) / f"{game_name}")


def tt_cache_path(game_name: str, book_dir: str | None = None) -> str:
    """The solver's TRANSPOSITION accelerator (search bounds), persisted beside the book so successive builds
    start warm — distinct from the exact play book, and a disposable cache (not the committed artifact)."""
    return str(Path(book_dir or BOOK_DIR) / f"{game_name}.tt")


def load_book(game_name: str, cap: int = 20_000_000, book_dir: str | None = None) -> Tablebase:
    """The project-committed book for a game (empty Tablebase if none exists yet)."""
    return Tablebase.load(book_path(game_name, book_dir), cap=cap)


def _key(game: Game, state: State) -> int:
    """A position's SYMMETRY-CANONICAL key via the game's own hook (SolvableGame.canonical_key); falls back to
    the Connect 4 bitboard canon so book.py stays game-agnostic — a new solvable game only supplies the hooks."""
    fn = getattr(game, "canonical_key", None)
    if fn is not None:
        return fn(state)
    position, mask, _ = to_bitboard(state)
    return canonical_key(position, mask)


def _ply(game: Game, state: State) -> int:
    fn = getattr(game, "ply", None)
    return fn(state) if fn is not None else to_bitboard(state)[2]


def position_value(game: Game, state: State, book: Tablebase | None = None, weak: bool = True) -> int:
    """Exact game-theoretic value to the player to move (win +1 / draw 0 / loss -1) via the game's own solver
    (SolvableGame.position_value), which consults `book` for already-solved children; falls back to the Connect 4
    bitboard solver. Once the frontier below is booked this collapses to a handful of lookups."""
    fn = getattr(game, "position_value", None)
    if fn is not None:
        return fn(state, book)
    vals = move_values(state, weak=weak, book=book)
    return max(vals.values()) if vals else 0


def book_value(book: Tablebase, game: Game, state: State) -> int | None:
    """The PROVEN value for a position (player-to-move perspective), or None if it isn't proven in the book — a
    mere ESTIMATE is not a value you can play optimally from, so exact consumers see only proofs."""
    return book.proven_value(_key(game, state))


def book_optimal_actions(book: Tablebase, game: Game, state: State) -> list[int] | None:
    """The optimal move SET derived PURELY from the book — one-ply lookahead over stored child values, no
    solving at play time. Returns None if any non-terminal child is missing (the book is too thin here to
    guarantee optimal play)."""
    legal = game.legal_actions(state)
    if not legal:
        return None
    me = game.current_player(state)
    vals: dict[int, int] = {}
    for a in legal:
        child = game.step(state, a)
        if game.is_terminal(child):
            vals[a] = 1 if game.winner(child) == me else 0
            continue
        bv = book.proven_value(_key(game, child))  # PROVEN children only — an estimate can't guarantee optimal play
        if bv is None:
            return None
        vals[a] = -bv  # stored from the child-mover's view → negate for our side
    best = max(vals.values())
    return sorted(a for a, v in vals.items() if v == best)


def principal_variation(book: Tablebase, game: Game, state: State, max_len: int = 1000) -> list[int]:
    """Reconstruct the optimal LINE from `state` by walking the book's optimal moves (one-ply lookahead over
    stored values) to a terminal — the 'raw path' to the winning/drawing outcome, materialised ON DEMAND so it
    never needs storing per entry. A PROVEN line reconstructs in full (the winning continuation was booked when
    the position was proven); where the book is too thin the walk stops early (an honest PARTIAL line). Stops at
    a terminal, an unbooked position, a repeated position (cycle guard for non-progressing games), or `max_len`."""
    line: list[int] = []
    seen: set[int] = set()
    s = state
    for _ in range(max_len):
        if game.is_terminal(s):
            break
        key = _key(game, s)
        if key in seen:
            break
        seen.add(key)
        acts = book_optimal_actions(book, game, s)
        if not acts:
            break
        move = min(acts, key=lambda a: (abs(a - game.num_actions // 2), a))  # centre-first, deterministic
        line.append(move)
        s = game.step(s, move)
    return line


# --- the generic APPROXIMATE evaluator (Phase 6): exact where cheap, a bounded-search belief otherwise ------
def _bitmask(actions) -> int:
    m = 0
    for a in actions:
        m |= 1 << int(a)
    return m


def _terminal_value(game: Game, state: State, me: int) -> float:
    w = game.winner(state)
    return 1.0 if w == me else (0.0 if w is None else -1.0)


def _rollout_outcome(game: Game, state: State, agent_factory: Callable, me: int, rng: random.Random) -> float:
    """Play one bounded self-play game from `state` (both seats from `agent_factory` — a FRESH agent per seat so
    no search state bleeds) and return the outcome for player `me` (+1 win / 0 draw / −1 loss)."""
    seats = [agent_factory(), agent_factory()]
    s = state
    while not game.is_terminal(s):
        s = game.step(s, seats[game.current_player(s)].act(game, s, rng))
    return _terminal_value(game, s, me)


def estimate_position(
    game: Game, state: State, agent_factory: Callable, games: int = 10, rng: random.Random | None = None
) -> tuple[float, int, int]:
    """A bounded-SEARCH value estimate for a position we cannot (yet) prove — one-ply lookahead where each child
    is scored by `games` bounded self-play games from `agent_factory`. Returns (value_to_mover, best_actions
    bitmask, sample_size). Game-agnostic and independent of any trained net (so the book CORRECTS a net, not
    mirrors it); it sharpens automatically as booked children make the rollouts sharper."""
    rng = rng or random.Random(0)
    me = game.current_player(state)
    child_vals: dict[int, float] = {}
    total = 0
    for a in game.legal_actions(state):
        child = game.step(state, a)
        if game.is_terminal(child):
            child_vals[a] = _terminal_value(game, child, me)
            continue
        score = 0.0
        for _ in range(games):
            score += _rollout_outcome(game, child, agent_factory, me, rng)
            total += 1
        child_vals[a] = score / games
    if not child_vals:
        return 0.0, 0, 0
    best = max(child_vals.values())
    return best, _bitmask(a for a, v in child_vals.items() if v >= best - 1e-9), total


def _prove(game: Game, state: State, book: Tablebase, max_exact_empty: int) -> tuple[float, int] | None:
    """Try to PROVE a position's value → (value_to_mover, best_actions) or None. (1) FREE: minimax over children
    that are already terminal or PROVEN in the book (no search — the eager bottom-up upgrade). (2) cheap EXACT:
    a game-solvable position (`exact_optimal_actions` within `max_exact_empty`) → solve it outright."""
    legal = game.legal_actions(state)
    if not legal:
        return None
    me = game.current_player(state)
    vals: dict[int, int] = {}
    all_resolved = True
    for a in legal:
        child = game.step(state, a)
        if game.is_terminal(child):
            vals[a] = int(_terminal_value(game, child, me))
            continue
        bv = book.proven_value(_key(game, child))  # PROVEN children only — an estimate can't prove a parent
        if bv is None:
            all_resolved = False
            break
        vals[a] = -bv
    if all_resolved:
        best = max(vals.values())
        return float(best), _bitmask(a for a, v in vals.items() if v == best)
    solve = getattr(game, "exact_optimal_actions", None)
    acts = solve(state, max_exact_empty) if solve is not None else None
    if acts:
        return float(position_value(game, state, book=book)), _bitmask(acts)
    return None


def evaluate(
    game: Game,
    state: State,
    book: Tablebase,
    estimator: Callable[[Game, State], tuple[float, int, int]],
    max_exact_empty: int = 0,
) -> Entry:
    """The evaluator LADDER → an `Entry`: terminal → PROVEN; an already-PROVEN book hit → keep it; a proof from
    booked children or a cheap exact solve → PROVEN (+ best_actions); else an ESTIMATE from bounded search. This
    is how a position graduates from a belief to a proof as the book fills beneath it."""
    if game.is_terminal(state):
        return Entry(status=PROVEN, value=_terminal_value(game, state, game.current_player(state)), best_actions=0, n=0)
    existing = book.entry(_key(game, state))
    if existing is not None and existing.status == PROVEN:
        return existing
    proven = _prove(game, state, book, max_exact_empty)
    if proven is not None:
        return Entry(status=PROVEN, value=proven[0], best_actions=proven[1], n=0)
    value, best_actions, n = estimator(game, state)
    return Entry(status=ESTIMATE, value=value, best_actions=best_actions, n=n)


def sample_seeds(game: Game, n: int, plies: int, seed: int = 0) -> list[State]:
    """`n` non-terminal positions reached by random play to `plies` stones — cheap MIDGAME roots whose small
    subtrees solve in milliseconds. Seeding the book from these grows real coverage fast (endgame/midgame first),
    which is exactly what early-termination consumes; the expensive shallow opening is the from-root mode's job."""
    rng = random.Random(seed)
    out: list[State] = []
    tries = 0
    while len(out) < n and tries < n * 20 + 50:
        tries += 1
        s = game.initial_state(rng)
        ok = True
        for _ in range(plies):
            legal = game.legal_actions(s)
            if not legal:
                ok = False
                break
            s = game.step(s, rng.choice(legal), rng)
            if game.is_terminal(s):
                ok = False
                break
        if ok and not game.is_terminal(s):
            out.append(s)
    return out


def play_until_decided(
    game: Game,
    seats: list,
    rng: random.Random,
    book: Tablebase | None = None,
    key_fn: Callable[[State], int] | None = None,
    start: State | None = None,
    opening_plies: int = 0,
) -> tuple[int | None, int, bool]:
    """Play a game to the end, but if `book` is given STOP the instant a reached position is booked — its result
    under optimal play is already known, so a settled position needn't be played out (Lever 3, early
    termination). Returns (winner_seat_or_None, plies_played, ended_early). SOUND when the seats play optimally
    from the decided position (the book/oracle references, or a solved-game agent): it then reports the true
    result faster; for a possibly-suboptimal agent it reports the OPTIMAL result, not necessarily the realised
    one — so use it to speed EXACT self-play/eval, not to score a weak model."""
    key_of = key_fn or (lambda s: _key(game, s))
    state = game.initial_state(rng) if start is None else start
    ply = 0
    while not game.is_terminal(state):
        if book is not None and ply >= opening_plies:
            bv = book.proven_value(key_of(state))  # end early only on a PROOF, never a mere estimate
            if bv is not None:
                mover = game.current_player(state)
                if bv > 0:
                    return (mover, ply, True)
                if bv < 0:
                    return ((1 - mover) if game.num_players == 2 else None, ply, True)
                return (None, ply, True)
        if ply < opening_plies:
            action = rng.choice(game.legal_actions(state))
        else:
            action = seats[game.current_player(state)].act(game, state, rng)
        state = game.step(state, action)
        ply += 1
    return (game.winner(state), ply, False)


def build_book(
    game: Game,
    book: Tablebase,
    roots: list[State] | None = None,
    max_plies: int = 12,
    min_plies: int = 0,
    max_positions: int = 2000,
    max_enumerate: int = 200_000,
    weak: bool = True,
    deadline: float | None = None,
    log: Callable[[str], None] | None = None,
    estimator: Callable[[Game, State], tuple[float, int, int]] | None = None,
    max_exact_empty: int = 0,
) -> dict:
    """Solve+store the reachable frontier BOTTOM-UP (deepest ply first, hub positions prioritised), bounded and
    resumable. Enumerate reachable non-terminal positions (from `roots`, default the initial position) with
    ≤ `max_plies` stones down, dedup by canonical key, count transposition in-degree (hub priority), then solve
    deepest-first so each shallower solve reads its already-booked children. Only positions with ≥ `min_plies`
    stones are stored (band the work: cover the cheap deep band first, then lower the floor on the next run).
    PROVEN positions are skipped (resumable), but ESTIMATE positions are RE-EVALUATED so a later run upgrades a
    belief to a proof the moment its children are booked — so re-invoking EXTENDS coverage rather than redoing it.

    Default (no `estimator`): every position is solved EXACTLY (`position_value`) and stored value-only — the
    unchanged exact builder. With an `estimator` (Phase 6): each position runs the `evaluate` ladder — PROVEN
    (from booked children or a cheap ≤ `max_exact_empty` exact solve) with its `best_actions`, else a bounded-
    search ESTIMATE — so a game whose opening is too deep to solve still gets a graded, model-usable book."""
    if roots is None:
        roots = [game.initial_state(random.Random(0))]
    seen: dict[int, tuple[State, int]] = {}
    indeg: dict[int, int] = {}
    dq: deque[State] = deque()
    for r in roots:
        if game.is_terminal(r):
            continue
        ck = _key(game, r)
        if ck not in seen:
            seen[ck] = (r, _ply(game, r))
            dq.append(r)
    while dq and len(seen) < max_enumerate:
        s = dq.popleft()
        if _ply(game, s) >= max_plies:
            continue
        for a in game.legal_actions(s):
            c = game.step(s, a)
            if game.is_terminal(c):
                continue
            ck = _key(game, c)
            indeg[ck] = indeg.get(ck, 0) + 1
            if ck not in seen:
                seen[ck] = (c, _ply(game, c))
                if len(seen) < max_enumerate:
                    dq.append(c)

    order = sorted(seen.items(), key=lambda kv: (-kv[1][1], -indeg.get(kv[0], 0)))
    solved = skipped = 0
    for ck, (state, ply) in order:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if solved >= max_positions:
            break
        if ply < min_plies:
            continue
        if book.is_proven(ck):  # a PROOF is final; an ESTIMATE is re-evaluated below (eager upgrade)
            skipped += 1
            continue
        priority = indeg.get(ck, 0) + ply
        if estimator is not None:
            e = evaluate(game, state, book, estimator, max_exact_empty=max_exact_empty)
            if e.status == PROVEN:
                book.put_proven(ck, int(e.value), best_actions=e.best_actions, priority=priority)
            else:
                book.put_estimate(ck, e.value, best_actions=e.best_actions, n=e.n, priority=priority)
        else:
            book.put(ck, position_value(game, state, book=book, weak=weak), priority=priority)
        solved += 1
        if log and solved % 25 == 0:
            log(f"book: solved {solved}, booked {len(book)}, ply≈{ply}")

    stats = {
        "solved": solved,
        "skipped": skipped,
        "enumerated": len(seen),
        "booked_total": len(book),
        "max_plies": max_plies,
        "min_plies": min_plies,
    }
    if log:
        log(f"book build: +{solved} solved, {skipped} already booked, {len(book)} total "
            f"(plies {min_plies}..{max_plies})")
    return stats


def book_coverage(game: Game, book: Tablebase, plies: int = 8, max_enumerate: int = 100_000) -> dict:
    """Honesty gauge: of the reachable non-terminal positions with ≤ `plies` stones down, how many are booked.
    `fraction` is the honest 'how much of the opening is exact' number the UI should never overclaim past."""
    root = game.initial_state(random.Random(0))
    seen: set[int] = set()
    dq: deque[State] = deque()
    ck0 = _key(game, root)
    seen.add(ck0)
    dq.append(root)
    booked = 1 if ck0 in book else 0
    while dq and len(seen) < max_enumerate:
        s = dq.popleft()
        if _ply(game, s) >= plies:
            continue
        for a in game.legal_actions(s):
            c = game.step(s, a)
            if game.is_terminal(c):
                continue
            ck = _key(game, c)
            if ck not in seen:
                seen.add(ck)
                if ck in book:
                    booked += 1
                if len(seen) < max_enumerate:
                    dq.append(c)
    reachable = len(seen)
    return {"plies": plies, "reachable": reachable, "booked": booked,
            "fraction": (booked / reachable) if reachable else 0.0}


def run_build_book(request: dict, log: Callable[[str], None] | None = print, book_dir: str | None = None) -> dict:
    """Extend the project-committed book for a game: warm the solver's TT accelerator from prior runs, solve+
    store a bounded frontier band (deepest-first), persist BOTH the exact book and the accelerator, and report
    coverage. Resumable — each call extends coverage. Request: { game, max_plies, min_plies, max_positions,
    deadline_seconds }."""
    from harness import solver
    from harness.registry import resolve_game

    game = resolve_game(request.get("game", "connect4"))
    if game.name == "connect4":
        solver.load_book(tt_cache_path(game.name, book_dir))  # warm the connect4 transposition accelerator
    book = load_book(game.name, book_dir=book_dir)
    before = len(book)
    max_plies = int(request.get("max_plies", 10))
    min_plies = int(request.get("min_plies", 0))
    max_positions = int(request.get("max_positions", 400))
    max_enumerate = int(request.get("max_enumerate", 200_000))
    ds = request.get("deadline_seconds")
    deadline = (time.perf_counter() + float(ds)) if ds is not None else None

    # Two modes: from-root (the long-running OPENING accumulator, default) or seed-sampled MIDGAME roots (fast,
    # grows real coverage from the endgame back — what early-termination feeds on).
    roots = None
    seed_games = int(request.get("seed_games", 0))
    if seed_games > 0:
        roots = sample_seeds(game, seed_games, int(request.get("seed_plies", 26)), int(request.get("seed", 0)))

    build = build_book(
        game, book, roots=roots, max_plies=max_plies, min_plies=min_plies, max_positions=max_positions,
        max_enumerate=max_enumerate, deadline=deadline, log=log,
    )
    book.save(book_path(game.name, book_dir))
    if game.name == "connect4":
        solver.save_book(tt_cache_path(game.name, book_dir))
    coverage = book_coverage(game, book, plies=min(max_plies, 10))
    result = {
        "game": game.name,
        "added": len(book) - before,
        "total": len(book),
        "coverage": coverage,
        "build": build,
    }
    if log:
        log(f"book: +{result['added']} (total {result['total']}); "
            f"opening coverage ≤{coverage['plies']} plies = {coverage['booked']}/{coverage['reachable']} "
            f"({coverage['fraction'] * 100:.1f}%)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(prog="harness.book", description="Extend a game's opening book (resumable).")
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()
    result = run_build_book(json.loads(Path(args.config_json).read_text()))
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
