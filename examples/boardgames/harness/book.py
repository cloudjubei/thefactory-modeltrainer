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
from harness.tablebase import Tablebase

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
    """The stored value for a position (player-to-move perspective), or None if it isn't booked."""
    return book.get(_key(game, state))


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
        bv = book.get(_key(game, child))
        if bv is None:
            return None
        vals[a] = -bv  # stored from the child-mover's view → negate for our side
    best = max(vals.values())
    return sorted(a for a, v in vals.items() if v == best)


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
            bv = book.get(key_of(state))
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
) -> dict:
    """Solve+store the reachable frontier BOTTOM-UP (deepest ply first, hub positions prioritised), bounded and
    resumable. Enumerate reachable non-terminal positions (from `roots`, default the initial position) with
    ≤ `max_plies` stones down, dedup by canonical key, count transposition in-degree (hub priority), then solve
    deepest-first so each shallower solve reads its already-booked children. Only positions with ≥ `min_plies`
    stones are stored (band the work: cover the cheap deep band first, then lower the floor on the next run).
    Stops at `max_positions` / `deadline`; already-booked positions are skipped, so re-invoking EXTENDS coverage
    rather than redoing it."""
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
        if ck in book:
            skipped += 1
            continue
        val = position_value(game, state, book=book, weak=weak)
        book.put(ck, val, priority=indeg.get(ck, 0) + ply)
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
