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
import signal
import threading
import time
from collections import deque
from itertools import groupby
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
    vals: dict[int, int | None] = {}
    for a in legal:
        child = game.step(state, a)
        if game.is_terminal(child):
            vals[a] = 1 if game.winner(child) == me else 0
            continue
        bv = book.proven_value(_key(game, child))  # PROVEN children only — an estimate can't guarantee optimal play
        vals[a] = None if bv is None else -bv  # stored from the child-mover's view → negate for our side
    v = book.proven_value(_key(game, state))  # the position's OWN proven value, if proven
    if v is not None:
        # PROVEN position: a move whose known child achieves the proven value V is optimal, and unproven siblings
        # can't beat V (it is the max), so they are irrelevant. This is what makes the winning-strategy grind —
        # which books only the SINGLE optimal line at a strategist node — playable: one booked achiever is enough.
        achievers = sorted(a for a, cv in vals.items() if cv is not None and cv == v)
        if achievers:
            return achievers
    if any(cv is None for cv in vals.values()):
        return None  # position not proven AND a child missing → too thin to guarantee the optimal set
    best = max(cv for cv in vals.values() if cv is not None)
    return sorted(a for a, cv in vals.items() if cv == best)


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


def make_book_estimator(
    book: Tablebase, sims: int = 64, solve_endgame: int = 14, games: int = 8, seed: int = 0
) -> Callable[[Game, State], tuple[float, int, int]]:
    """The DEFAULT bounded-SEARCH estimator (Phase 6 design decision (a)): score an unprovable position by one-ply
    lookahead where each child is played out by BOOK-AWARE MCTS-Solver self-play. The agents read PROVEN booked
    children and solve cheap endgames, so their games back up EXACT values wherever the book/solver reaches — the
    estimate is grounded in proofs beneath it (never a trained net, so the book CORRECTS a net rather than mirrors
    it) and sharpens toward a proof as coverage grows under it. A FRESH agent per seat (no search-state bleed).
    Returns an `estimator(game, state) -> (value_to_mover, best_actions, n)` closure over the growing `book`."""
    from harness.agents import MctsAgent

    rng = random.Random(seed)

    def _estimator(game: Game, state: State) -> tuple[float, int, int]:
        factory = lambda: MctsAgent(sims=sims, solve_endgame=solve_endgame, book=book)
        return estimate_position(game, state, factory, games=games, rng=rng)

    return _estimator


def _prove(game: Game, state: State, book: Tablebase, max_exact_empty: int) -> tuple[float, int] | None:
    """Try to PROVE a position's value → (value_to_mover, best_actions) or None. (1) a proven/terminal WINNING
    child proves the node a win outright (siblings irrelevant — one win is enough, which is what lets the pruned
    winning-strategy grind prove a node from its single booked winning line); (2) FREE minimax once EVERY child is
    terminal/PROVEN (the eager bottom-up upgrade); (3) cheap EXACT: a ≤ `max_exact_empty` solvable position."""
    legal = game.legal_actions(state)
    if not legal:
        return None
    me = game.current_player(state)
    vals: dict[int, int] = {}
    wins: list[int] = []
    all_resolved = True
    for a in legal:
        child = game.step(state, a)
        if game.is_terminal(child):
            cv = int(_terminal_value(game, child, me))
        else:
            bv = book.proven_value(_key(game, child))  # PROVEN children only — an estimate can't prove a parent
            if bv is None:
                all_resolved = False
                continue
            cv = -bv
        vals[a] = cv
        if cv == 1:
            wins.append(a)
    if wins:
        return 1.0, _bitmask(wins)
    if all_resolved and vals:
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


class _SolveTimeout(Exception):
    pass


def _run_bounded(fn: Callable, seconds: float):
    """Run `fn()` with a wall-clock cap (SIGALRM). Raises `_SolveTimeout` on overrun; no cap when `seconds<=0` or
    off the main thread (a parallel worker IS the main thread of its own process, so per-position caps still hold
    there; a threaded server context simply gets no cap rather than a crash). Politely restores the prior handler."""
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        return fn()

    def _handler(signum, frame):
        raise _SolveTimeout()

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


def _solve_exact_bounded(game: Game, state: State, book: Tablebase, weak: bool, max_seconds: float):
    """The exact value of `state`, or None if the solve overran `max_seconds` — DEFERRED, not failed: a later
    band/run books its children, after which the same solve is a cheap one-ply lookup (the bottom-up point). A
    partially-populated solver transposition table is left behind as valid bounds, so the deferral costs nothing."""
    try:
        return _run_bounded(lambda: position_value(game, state, book=book, weak=weak), max_seconds)
    except _SolveTimeout:
        return None


# --- parallel band solving: positions at the SAME ply are independent given the DEEPER (already-booked) band ---
_BAND: dict = {}


def _band_init(game_name: str, book_values: dict, weak: bool, max_seconds: float) -> None:
    from harness.registry import resolve_game

    _BAND["game"] = resolve_game(game_name)
    tb = Tablebase(cap=max(1, len(book_values) + 1))
    tb._v = dict(book_values)  # the exact accumulator stores PROVEN values, so proven_value works from _v alone
    _BAND["book"] = tb
    _BAND["weak"] = bool(weak)
    _BAND["secs"] = float(max_seconds)


def _band_task(item):
    ck, state = item
    return ck, _solve_exact_bounded(_BAND["game"], state, _BAND["book"], _BAND["weak"], _BAND["secs"])


def _deadline_hit(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def _solve_frontier_banded(
    game: Game, book: Tablebase, order: list, indeg: dict, min_plies: int, max_positions: int, weak: bool,
    deadline: float | None, workers: int, max_seconds: float, log: Callable[[str], None] | None,
) -> tuple[int, int, int]:
    """Solve the ordered frontier ply-band by ply-band (deepest first). Within a band every position is
    independent given the deeper booked band, so a band is solved across a PROCESS POOL (each worker gets the
    current book snapshot for child short-circuits); each solve is capped at `max_seconds` (a hard position is
    DEFERRED). Results are booked AS THEY COMPLETE and the `deadline`/`max_positions` budget is honoured WITHIN a
    band, not merely between bands — so a time-bounded run stops within ~`max_seconds` of its deadline instead of
    grinding a whole band first (cancelling the not-yet-started solves). Falls back to in-process sequential if a
    pool can't be created. Returns (solved, skipped, deferred); the final book equals the sequential build when the
    budget is not hit (exact solves are deterministic)."""
    solved = skipped = deferred = 0

    def _book(ck: int, p: int, val) -> bool:  # book one result; True once the deadline/max_positions budget is spent
        nonlocal solved, deferred
        if val is None:
            deferred += 1
        else:
            book.put(ck, int(val), priority=indeg.get(ck, 0) + p)
            solved += 1
        return solved >= max_positions or _deadline_hit(deadline)

    for ply, grp in groupby(order, key=lambda kv: kv[1][1]):  # `order` is sorted -ply → deepest band first
        if _deadline_hit(deadline) or solved >= max_positions:
            break
        band = []
        for ck, (state, p) in grp:
            if p < min_plies:
                continue
            if book.is_proven(ck):
                skipped += 1
                continue
            band.append((ck, state, p))
        if not band:
            continue
        stop = False
        ex = None
        if workers > 1 and len(band) >= 2:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed

            try:
                ex = ProcessPoolExecutor(
                    max_workers=min(workers, len(band)),
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_band_init,
                    initargs=(game.name, dict(book._v), weak, max_seconds),
                )
            except Exception:
                ex = None
        if ex is not None:
            pri = {ck: p for ck, _st, p in band}
            futures = {ex.submit(_band_task, (ck, st)): ck for ck, st, _p in band}
            try:
                for fut in as_completed(futures):
                    ck = futures[fut]
                    _, val = fut.result()
                    if _book(ck, pri[ck], val):
                        stop = True
                        break
            finally:
                ex.shutdown(wait=True, cancel_futures=True)
        else:
            for ck, st, p in band:
                if _book(ck, p, _solve_exact_bounded(game, st, book, weak, max_seconds)):
                    stop = True
                    break
        if log:
            log(f"book: ply {ply} band → {solved} solved, {deferred} deferred, {len(book)} total")
        if stop:
            break
    return solved, skipped, deferred


def _prove_bounded(
    game: Game, state: State, book: Tablebase, max_exact_empty: int, max_seconds: float
) -> tuple[float, int] | None:
    """PROVE (value_to_mover, best_actions_bitmask) for `state`, bounded — or None if the exact solve overran
    `max_seconds` (DEFERRED, retried once its children are booked). (1) the FREE/cheap path (`_prove`): minimax
    over already-booked/terminal children, or a ≤ `max_exact_empty` cheap solve; else (2) a bounded FULL solve
    (a strategist node in the deep opening whose children aren't booked yet) via the game's exact hooks."""
    proven = _prove(game, state, book, max_exact_empty)
    if proven is not None:
        return proven
    solve = getattr(game, "exact_optimal_actions", None)
    if solve is None:
        return None

    def _full() -> tuple[float, int] | None:
        acts = solve(state, 10**9)  # huge empty budget forces a full solve → the exact optimal SET
        if not acts:
            return None
        return float(position_value(game, state, book=book)), _bitmask(acts)

    try:
        return _run_bounded(_full, max_seconds)
    except _SolveTimeout:
        return None


def prove_winning_strategy(
    game: Game,
    book: Tablebase,
    root: State | None = None,
    strategist: int = 0,
    max_plies: int = 14,
    max_exact_empty: int = 22,
    deadline: float | None = None,
    max_positions: int = 10**9,
    max_seconds: float = 0.0,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Prove the STRATEGIST's WINNING-STRATEGY TREE into the book — the directed M1 grind that is far smaller than
    the whole opening (SOLVE-IT §C.5 Phase 2). The tree from `root`: at a STRATEGIST node we need only ONE optimal
    move (a single winning line answers our obligation), so we follow just that child; at an OPPONENT node we must
    answer EVERY legal reply, so we recurse into all. It bottoms out where the solver finishes cheaply (≤
    `max_exact_empty` empties) or at `max_plies`, and PROVES bottom-up (post-order): a node is proved for free once
    its children are booked (`_prove`), else by a bounded full solve. BOUNDED (`deadline` / `max_positions`) and
    RESUMABLE (proven nodes are skipped; a solve that overruns `max_seconds` is DEFERRED and retried next pass once
    the frontier beneath it is booked and the solve collapses to lookups — the endgame-back grind). `max_seconds`
    MUST be > 0 for a from-opening grind or a single hard solve runs unbounded. Returns proof/deferral counts."""
    if root is None:
        root = game.initial_state(random.Random(0))
    stats = {"proven": 0, "deferred": 0, "visited": 0, "leaves": 0}
    solve_hook = getattr(game, "exact_optimal_actions", None)
    mid = game.num_actions // 2

    def _cheap(state: State):
        return solve_hook(state, max_exact_empty) if solve_hook is not None else None

    def _spent() -> bool:
        return stats["proven"] >= max_positions or _deadline_hit(deadline)

    def _child_value(state: State, action: int) -> float | None:
        """The mover-relative value of `action` from booked/terminal children only (None if the child is unproven)."""
        child = game.step(state, action)
        if game.is_terminal(child):
            return _terminal_value(game, child, game.current_player(state))
        bv = book.proven_value(_key(game, child))
        return None if bv is None else -bv

    def rec(state: State, in_path: frozenset) -> bool | None:
        if game.is_terminal(state):
            return True
        ck = _key(game, state)
        if book.is_proven(ck):
            return True
        if _spent() or ck in in_path:
            return None
        stats["visited"] += 1
        ply = _ply(game, state)
        acts = _cheap(state)  # LEAF A: the solver finishes cheaply → a proven leaf (value + optimal set)
        if acts is not None:
            book.put_proven(ck, int(position_value(game, state, book=book)), best_actions=_bitmask(acts), priority=ply)
            stats["proven"] += 1
            stats["leaves"] += 1
            return True
        if ply >= max_plies:  # LEAF B: the ply cap with no cheap solve → a bounded full solve
            pv = _prove_bounded(game, state, book, max_exact_empty, max_seconds)
            if pv is None:
                stats["deferred"] += 1
                return None
            book.put_proven(ck, int(pv[0]), best_actions=pv[1], priority=ply)
            stats["proven"] += 1
            stats["leaves"] += 1
            return True
        in_path = in_path | {ck}
        if game.current_player(state) == strategist:
            # DISCOVER this node's value + optimal set (a bounded solve; NOT booked yet), descend ONE optimal line,
            # then book the node from that now-booked achiever — so a proven strategist node ALWAYS has its winning
            # line booked beneath it (resumable: a proven node is safe to skip; playable: the book knows its move).
            disc = _prove_bounded(game, state, book, max_exact_empty, max_seconds)
            if disc is None:
                stats["deferred"] += 1
                return None
            value = disc[0]
            opt = [c for c in range(game.num_actions) if (disc[1] >> c) & 1] or game.legal_actions(state)
            if rec(game.step(state, min(opt, key=lambda a: (abs(a - mid), a))), in_path) is not True:
                return None  # the chosen line is not fully booked (budget) → leave unproven for a resume
            achievers = [a for a in game.legal_actions(state) if _child_value(state, a) == value]
            if not achievers:
                stats["deferred"] += 1
                return None
            book.put_proven(ck, int(value), best_actions=_bitmask(achievers), priority=ply)
            stats["proven"] += 1
            return True
        for a in game.legal_actions(state):  # OPPONENT node: answer EVERY reply, book only once all are proven
            child = game.step(state, a)
            if not game.is_terminal(child) and rec(child, in_path) is not True:
                return None
        pv = _prove(game, state, book, max_exact_empty)  # all replies booked → free minimax (never a fresh solve)
        if pv is None:
            stats["deferred"] += 1
            return None
        book.put_proven(ck, int(pv[0]), best_actions=pv[1], priority=ply)
        stats["proven"] += 1
        return True

    rec(root, frozenset())
    if log:
        log(f"winning-strategy: +{stats['proven']} proven ({stats['leaves']} leaves), {stats['deferred']} deferred")
    return stats


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
    workers: int = 1,
    max_position_seconds: float = 0.0,
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
    deferred = 0
    if estimator is None and (workers > 1 or max_position_seconds > 0):
        # PARALLEL / time-capped exact accumulator (the deep-opening grind): solve bands across cores, cap each
        # solve so one hard position can't blow the budget (it's deferred, then cheap once its children are booked).
        solved, skipped, deferred = _solve_frontier_banded(
            game, book, order, indeg, min_plies, max_positions, weak, deadline, workers, max_position_seconds, log,
        )
    else:
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
        "deferred": deferred,
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
    proven = 1 if book.is_proven(ck0) else 0  # a booked entry may be a graded ESTIMATE, not a proof — count them apart
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
                    if book.is_proven(ck):
                        proven += 1
                if len(seen) < max_enumerate:
                    dq.append(c)
    reachable = len(seen)
    return {"plies": plies, "reachable": reachable, "booked": booked, "proven": proven,
            "fraction": (booked / reachable) if reachable else 0.0,
            "provenFraction": (proven / reachable) if reachable else 0.0}


def winning_strategy_coverage(
    game: Game, book: Tablebase, root: State | None = None, strategist: int = 0, max_plies: int = 14,
    max_exact_empty: int = 0, max_enumerate: int = 200_000,
) -> dict:
    """The honest M1 tracker: how much of the STRATEGIST's winning-strategy tree from `root` is PROVEN in the book
    (SOLVE-IT §C.5 Phase 2). Enumerate that tree — at a strategist node follow the book's believed-optimal move
    (the entry's `best_actions` where present, else all legal, since we can't yet prune), at an opponent node every
    reply — to `max_plies` or a terminal / ≤ `max_exact_empty` solvable leaf. `provenFraction` climbs 0→1 as the
    grind books the tree from the endgame back; `root_proven`/`root_value` expose whether the whole strategy exists
    (a proven WIN from the opening = a lookup-perfect first player); `complete` = every enumerated node is proven."""
    if root is None:
        root = game.initial_state(random.Random(0))
    solve_hook = getattr(game, "exact_optimal_actions", None)
    mid = game.num_actions // 2
    root_key = _key(game, root)
    seen: set[int] = {root_key}
    dq: deque[tuple[State, int]] = deque([(root, root_key)])
    nodes = proven = 0
    while dq and len(seen) < max_enumerate:
        state, ck = dq.popleft()
        if game.is_terminal(state):
            continue
        nodes += 1
        if book.is_proven(ck):
            proven += 1
        if solve_hook is not None and max_exact_empty > 0 and solve_hook(state, max_exact_empty) is not None:
            continue  # a solvable leaf — the solver finishes below here; counted, not expanded
        if _ply(game, state) >= max_plies:
            continue
        acts = book_optimal_actions(book, game, state) if game.current_player(state) == strategist else None
        if acts:
            follow = [min(acts, key=lambda a: (abs(a - mid), a))]  # one proven optimal line (orientation-safe)
        else:
            follow = game.legal_actions(state)  # opponent replies, or an unproven strategist node (can't prune yet)
        for a in follow:
            child = game.step(state, a)
            cck = _key(game, child)
            if cck not in seen:
                seen.add(cck)
                dq.append((child, cck))
    root_val = book.proven_value(root_key)
    return {
        "nodes": nodes, "proven": proven,
        "provenFraction": (proven / nodes) if nodes else 0.0,
        "root_proven": root_val is not None, "root_value": root_val,
        "complete": nodes > 0 and proven == nodes and root_val is not None,
        "strategist": strategist, "max_plies": max_plies,
    }


def run_build_book(request: dict, log: Callable[[str], None] | None = print, book_dir: str | None = None) -> dict:
    """Extend the project-committed book for a game: warm the solver's TT accelerator from prior runs, solve+
    store a bounded frontier band (deepest-first), persist BOTH the exact book and the accelerator, and report
    coverage. Resumable — each call extends coverage. Request keys: `game`, `max_plies`, `min_plies`,
    `max_positions`, `max_enumerate`, `deadline_seconds`; SEED mode `seed_games`/`seed_plies`; ROBUSTNESS
    `workers` (parallel band solving) + `max_position_seconds` (per-solve cap → DEFERRED, no hang); GRADED mode
    `estimate_games`/`estimate_sims`/`estimate_solve_endgame`/`max_exact_empty` (book-aware search beliefs for the
    deep opening the solver can't reach); WINNING-STRATEGY mode `winning_strategy`/`strategist` (SOLVE-IT M1 — prove
    the directed winning-strategy tree, reporting a `winningStrategy` coverage that climbs toward a proven strategy)."""
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

    # Opt-in GRADED mode (Phase 6): where a position is too deep to solve, store a book-aware bounded-SEARCH
    # ESTIMATE (value + best_actions + sample size a deep model can use) instead of nothing — the graded opening
    # book that needs no minutes-long solve. `estimate_games>0` enables it; proofs still win the evaluator ladder.
    estimator = None
    max_exact_empty = int(request.get("max_exact_empty", 0))
    estimate_games = int(request.get("estimate_games", 0))
    if estimate_games > 0:
        estimator = make_book_estimator(
            book, sims=int(request.get("estimate_sims", 64)),
            solve_endgame=int(request.get("estimate_solve_endgame", 14)), games=estimate_games,
            seed=int(request.get("seed", 0)),
        )

    # Robustness knobs (the parallel/deadline-safe accumulator): `workers` solves each ply-band across a process
    # pool; `max_position_seconds` caps EACH solve so one hard opening position is DEFERRED (booked later) instead
    # of blowing the pass budget — a from-root opening grind can't hang the activity. Both are no-ops in exact mode
    # when left at their defaults (unchanged sequential build).
    workers = int(request.get("workers", 1))
    max_position_seconds = float(request.get("max_position_seconds", 0.0))

    # Opt-in WINNING-STRATEGY mode (SOLVE-IT §C.5 M1): instead of the breadth-first opening accumulator, prove the
    # STRATEGIST's directed winning-strategy tree (our one optimal move + every opponent reply) — far smaller than
    # the whole opening, and the actual path to a lookup-perfect player. Bounded + resumable, from the endgame back.
    winning_strategy = bool(request.get("winning_strategy", False))
    strategist = int(request.get("strategist", 0))
    ws_empty = int(request.get("max_exact_empty", 22)) if winning_strategy else max_exact_empty
    if winning_strategy:
        build = prove_winning_strategy(
            game, book, root=(roots[0] if roots else None), strategist=strategist, max_plies=max_plies,
            max_exact_empty=ws_empty, deadline=deadline, max_positions=max_positions,
            max_seconds=max_position_seconds, log=log,
        )
    else:
        build = build_book(
            game, book, roots=roots, max_plies=max_plies, min_plies=min_plies, max_positions=max_positions,
            max_enumerate=max_enumerate, deadline=deadline, log=log, estimator=estimator,
            max_exact_empty=max_exact_empty, workers=workers, max_position_seconds=max_position_seconds,
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
    if winning_strategy:
        result["winningStrategy"] = winning_strategy_coverage(
            game, book, root=(roots[0] if roots else None), strategist=strategist,
            max_plies=min(max_plies, 12), max_exact_empty=ws_empty,
        )
    if log:
        ws = result.get("winningStrategy")
        extra = (f"; winning-strategy proven {ws['proven']}/{ws['nodes']} "
                 f"({ws['provenFraction'] * 100:.1f}%){' — COMPLETE' if ws['complete'] else ''}") if ws else ""
        log(f"book: +{result['added']} (total {result['total']}); "
            f"opening coverage ≤{coverage['plies']} plies = {coverage['booked']}/{coverage['reachable']} "
            f"({coverage['fraction'] * 100:.1f}%){extra}")
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
