"""Perfect-play Connect 4 oracle — a bitboard negamax solver exposed as an Agent.

Connect 4 is SOLVED: the first player wins with perfect play by opening in the centre column. This module is
the GROUND TRUTH the trained models are measured against — the top rung of the rating spine and the labeller
for the `oracle_optimality_rate` benchmark (harness/benchmark.py).

Torch-free, like the other baseline cores. It uses the standard 7-bit-per-column board encoding (6 playable
rows + 1 sentinel row per column, so shifts can't wrap a line across columns), alpha-beta negamax, and the two
things that make an exact solve tractable in pure Python: a transposition table and non-losing-move
anticipation (never explore a move that hands the opponent an immediate win). The table persists at module
level, so repeated play (a gauntlet) and benchmark scoring amortise into one shared solve. The algorithm is
Pascal Pons' well-known solver, ported to Python; `_half` restores C's truncate-toward-zero division, which the
iterative-deepening window relies on.
"""
from __future__ import annotations

import random

from games.connect4 import COLS, ROWS, C4State

WIDTH = COLS  # 7
HEIGHT = ROWS  # 6
_H1 = HEIGHT + 1  # 7 bits per column (6 rows + 1 sentinel)
_TOTAL = WIDTH * HEIGHT  # 42
MIN_SCORE = -(_TOTAL // 2) + 3  # -18
MAX_SCORE = (_TOTAL + 1) // 2 - 3  # 18
_CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)  # centre-first move ordering (the known winning bias)

_BOTTOM_MASK = 0
for _c in range(WIDTH):
    _BOTTOM_MASK |= 1 << (_c * _H1)
_BOARD_MASK = _BOTTOM_MASK * ((1 << HEIGHT) - 1)  # the 42 playable cells

# Persistent transposition table (position+mask key -> encoded upper bound). Capped for memory safety: past the
# cap we stop inserting (correctness is unaffected — only the shared cache stops growing).
# The transposition table is a PERSISTENT, symmetry-canonical opening/endgame book: it survives across solves
# (load_book/save_book) and accumulates the hard mid-depth frontier, so the opening — which thrashed a small
# in-memory table (the measured 158s/position at move 10) — becomes tractable as the frontier fills in. Each
# entry is a proven upper bound on the position's intrinsic score, valid in any later search (see _negamax).
_TT: dict[int, int] = {}
_TT_CAP = 6_000_000


def load_book(path: str) -> int:
    """Load a persisted transposition/book file into the shared table (accumulating). Returns entries loaded."""
    from harness.tablebase import Tablebase

    tb = Tablebase.load(path)
    _TT.update(tb._v)
    return len(tb._v)


def save_book(path: str) -> int:
    """Persist the shared transposition/book table so the next run starts warm. Returns entries written."""
    from harness.tablebase import Tablebase

    tb = Tablebase(cap=_TT_CAP)
    tb._v = dict(_TT)
    tb.save(path)
    return len(_TT)


def _bottom_mask_col(c: int) -> int:
    return 1 << (c * _H1)


def _top_mask_col(c: int) -> int:
    return 1 << (c * _H1 + HEIGHT - 1)


def _column_mask(c: int) -> int:
    return ((1 << HEIGHT) - 1) << (c * _H1)


_COL_BITS = (1 << _H1) - 1  # the 7 bits of one column (6 rows + sentinel)


def _mirror(x: int) -> int:
    """Reflect a bitboard left↔right about the centre column (column c ↔ column WIDTH-1-c). The game-theoretic
    value is mirror-invariant, so canonicalising every transposition key to this reflection halves the table."""
    m = 0
    for c in range(WIDTH):
        m |= ((x >> (c * _H1)) & _COL_BITS) << ((WIDTH - 1 - c) * _H1)
    return m


def canonical_key(position: int, mask: int) -> int:
    """The SYMMETRY-REDUCED transposition key: the smaller of the position's Pons key (`position + mask`) and
    its left-right mirror's. A position and its reflection therefore share ONE table entry — ~50% fewer to
    solve/store, and any two lines that are mirror images reuse each other's search."""
    k = position + mask
    km = _mirror(position) + _mirror(mask)
    return k if k <= km else km


def _half(x: int) -> int:
    """Divide by 2 truncating TOWARD ZERO (C semantics), unlike Python's floor `//` for negatives."""
    return -((-x) // 2) if x < 0 else x // 2


def _alignment(pos: int) -> bool:
    """Does `pos` (one player's stones) contain four in a row?"""
    m = pos & (pos >> _H1)  # horizontal
    if m & (m >> (2 * _H1)):
        return True
    m = pos & (pos >> HEIGHT)  # diagonal "\"
    if m & (m >> (2 * HEIGHT)):
        return True
    m = pos & (pos >> (HEIGHT + 2))  # diagonal "/"
    if m & (m >> (2 * (HEIGHT + 2))):
        return True
    m = pos & (pos >> 1)  # vertical
    if m & (m >> 2):
        return True
    return False


def _compute_winning_position(position: int, mask: int) -> int:
    """All empty cells where `position`'s owner would complete a line (their immediate winning spots)."""
    # vertical
    r = (position << 1) & (position << 2) & (position << 3)
    # horizontal
    p = (position << _H1) & (position << (2 * _H1))
    r |= p & (position << (3 * _H1))
    r |= p & (position >> _H1)
    p = (position >> _H1) & (position >> (2 * _H1))
    r |= p & (position << _H1)
    r |= p & (position >> (3 * _H1))
    # diagonal "\"
    p = (position << HEIGHT) & (position << (2 * HEIGHT))
    r |= p & (position << (3 * HEIGHT))
    r |= p & (position >> HEIGHT)
    p = (position >> HEIGHT) & (position >> (2 * HEIGHT))
    r |= p & (position << HEIGHT)
    r |= p & (position >> (3 * HEIGHT))
    # diagonal "/"
    d = HEIGHT + 2
    p = (position << d) & (position << (2 * d))
    r |= p & (position << (3 * d))
    r |= p & (position >> d)
    p = (position >> d) & (position >> (2 * d))
    r |= p & (position << d)
    r |= p & (position >> (3 * d))
    return r & (_BOARD_MASK ^ mask)


def _can_play(mask: int, c: int) -> bool:
    return (mask & _top_mask_col(c)) == 0


def _possible(mask: int) -> int:
    return (mask + _BOTTOM_MASK) & _BOARD_MASK


def _is_winning_move(position: int, mask: int, c: int) -> bool:
    pos2 = position | ((mask + _bottom_mask_col(c)) & _column_mask(c))
    return _alignment(pos2)


def _can_win_next(position: int, mask: int) -> bool:
    return bool(_compute_winning_position(position, mask) & _possible(mask))


def _possible_non_losing_moves(position: int, mask: int) -> int:
    """Playable cells that don't immediately lose: if the opponent threatens a win we're forced to block it
    (two forced blocks = lost), and we never play directly under an opponent's winning square."""
    possible_mask = _possible(mask)
    opponent_win = _compute_winning_position(position ^ mask, mask)
    forced = possible_mask & opponent_win
    if forced:
        if forced & (forced - 1):  # more than one forced block → the opponent wins next move
            return 0
        possible_mask = forced
    return possible_mask & ~(opponent_win >> 1)


def _move_score(position: int, mask: int, move: int) -> int:
    winning = _compute_winning_position(position | move, mask)
    return bin(winning).count("1")


def _negamax(position: int, mask: int, moves: int, alpha: int, beta: int, tt: dict[int, int]) -> int:
    """Score the position (to-move perspective; +N = win in N-ish plies) assuming the mover has no immediate
    win (the `_solve` driver strips that case, and every non-losing child preserves the invariant)."""
    possible = _possible_non_losing_moves(position, mask)
    if possible == 0:  # cannot avoid letting the opponent win next move
        return -(_TOTAL - moves) // 2
    if moves >= _TOTAL - 2:  # board fills with no winner
        return 0
    min_score = -(_TOTAL - 2 - moves) // 2
    if alpha < min_score:
        alpha = min_score
        if alpha >= beta:
            return alpha
    max_score = (_TOTAL - 1 - moves) // 2
    if beta > max_score:
        beta = max_score
        if alpha >= beta:
            return beta
    key = canonical_key(position, mask)  # symmetry-reduced → a position and its mirror share this entry
    entry = tt.get(key)
    if entry is not None:  # the table stores an upper bound (encoded +1 off MIN_SCORE)
        tt_max = entry + MIN_SCORE - 1
        if beta > tt_max:
            beta = tt_max
            if alpha >= beta:
                return beta

    ordered = []
    for c in _CENTER_ORDER:
        move = possible & _column_mask(c)
        if move:
            ordered.append((_move_score(position, mask, move), move))
    ordered.sort(key=lambda t: t[0], reverse=True)

    for _, move in ordered:
        score = -_negamax(position ^ mask, mask | move, moves + 1, -beta, -alpha, tt)
        if score >= beta:
            return score
        if score > alpha:
            alpha = score
    if len(tt) < _TT_CAP:
        tt[key] = alpha - MIN_SCORE + 1
    return alpha


def _solve(position: int, mask: int, moves: int, weak: bool, tt: dict[int, int]) -> int:
    """Exact game-theoretic value of a position (mover perspective) via null-window iterative deepening.
    `weak=True` returns only the sign (win +1 / draw 0 / loss -1), which is faster and enough for the optimal
    SET; `weak=False` returns the signed distance-to-end so a win is scored higher the sooner it comes."""
    if _can_win_next(position, mask):
        return (_TOTAL + 1 - moves) // 2
    min_s = -(_TOTAL - moves) // 2
    max_s = (_TOTAL + 1 - moves) // 2
    if weak:
        min_s, max_s = -1, 1
    while min_s < max_s:
        med = min_s + (max_s - min_s) // 2
        if med <= 0 and _half(min_s) < med:
            med = _half(min_s)
        elif med >= 0 and _half(max_s) > med:
            med = _half(max_s)
        r = _negamax(position, mask, moves, med, med + 1, tt)
        if r <= med:
            max_s = r
        else:
            min_s = r
    return min_s


def to_bitboard(state: C4State) -> tuple[int, int, int]:
    """Convert a `C4State` into (current-player stones, all stones, moves-played)."""
    position = 0
    mask = 0
    cur = state.to_move + 1
    moves = 0
    for c in range(WIDTH):
        for row in range(HEIGHT):
            v = state.board[row * WIDTH + c]
            if v != 0:
                bit = 1 << (c * _H1 + row)
                mask |= bit
                moves += 1
                if v == cur:
                    position |= bit
    return position, mask, moves


def move_values(state: C4State, weak: bool = True, tt: dict[int, int] | None = None, book=None) -> dict[int, int]:
    """Game-theoretic value of every legal column from the mover's perspective (higher = better). The optimal
    SET is the argmax; `weak` scores by outcome (win/draw/loss), `weak=False` by outcome-then-speed.

    `book` (any `.get(canonical_key) -> value|None`, the exact WEAK opening/endgame store) short-circuits a
    child that is already solved: instead of re-searching it we read its stored value. This is the bottom-up
    wall-break — once the deeper frontier is booked, a shallower position's children are all instant lookups,
    so its own solve collapses to a handful of table hits. Only consulted when `weak` (the store is weak-valued)."""
    tt = _TT if tt is None else tt
    if state.done:
        return {}
    position, mask, moves = to_bitboard(state)
    values: dict[int, int] = {}
    for c in range(WIDTH):
        if not _can_play(mask, c):
            continue
        if _is_winning_move(position, mask, c):
            values[c] = 1 if weak else (_TOTAL + 1 - moves) // 2
            continue
        pos2, mask2 = position ^ mask, mask | ((mask + _bottom_mask_col(c)) & _column_mask(c))
        if book is not None and weak:
            bv = book.proven_value(canonical_key(pos2, mask2))  # PROOFS only — an estimate isn't an exact bound
            if bv is not None:  # child stored from the CHILD-mover's view → negate for our side
                values[c] = max(-1, min(1, -bv))
                continue
        s = -_solve(pos2, mask2, moves + 1, weak, tt)
        values[c] = max(-1, min(1, s)) if weak else s
    return values


def optimal_columns(state: C4State, tt: dict[int, int] | None = None, book=None) -> list[int]:
    """The set of outcome-optimal columns — every move that preserves the best achievable result. A model's
    move is 'optimal' iff it lies in this set (multiple moves can be equally optimal)."""
    values = move_values(state, weak=True, tt=tt, book=book)
    if not values:
        return []
    best = max(values.values())
    return sorted(c for c, v in values.items() if v == best)


_WIN = 100_000  # terminal magnitude — far above any heuristic leaf so a real win/loss always dominates
_INF = 10**9


def _heuristic_eval(position: int, mask: int) -> int:
    """A small positional score (to-move perspective) for depth-limited leaves: threats created minus
    threats conceded, plus centre control. Deliberately tiny so it never outweighs a real win/loss."""
    my = bin(_compute_winning_position(position, mask)).count("1")
    opp = bin(_compute_winning_position(position ^ mask, mask)).count("1")
    center = _column_mask(WIDTH // 2)
    my_c = bin(position & center).count("1")
    opp_c = bin((position ^ mask) & center).count("1")
    return 3 * (my - opp) + (my_c - opp_c)


def _dl_negamax(position: int, mask: int, moves: int, depth: int, alpha: int, beta: int) -> int:
    """Depth-limited negamax with the same tactical layer as the exact solver (immediate win, forced-loss
    anticipation, never play under an opponent's winning square) but a heuristic leaf once `depth` runs out —
    so it stays fast enough to play from the opening while remaining tactically sharp. Exact whenever `depth`
    covers the plies remaining (the leaf is then never reached)."""
    if _can_win_next(position, mask):
        return _WIN - moves
    if moves >= _TOTAL:
        return 0
    possible = _possible_non_losing_moves(position, mask)
    if possible == 0:  # opponent has an unstoppable win next move
        return -(_WIN - moves)
    if depth <= 0:
        return _heuristic_eval(position, mask)
    best = -_INF
    for c in _CENTER_ORDER:
        move = possible & _column_mask(c)
        if not move:
            continue
        score = -_dl_negamax(position ^ mask, mask | move, moves + 1, depth - 1, -beta, -alpha)
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


class OracleAgent:
    """Plays game-theoretically optimal Connect 4 — the perfect-play reference. Among equally-optimal moves it
    prefers the fastest win / slowest loss (the strong score), tie-broken centre-first for determinism."""

    kind = "oracle"

    def act(self, game, state, rng: random.Random) -> int:
        position, mask, moves = to_bitboard(state)
        legal = [c for c in range(WIDTH) if _can_play(mask, c)]
        if not legal:
            raise ValueError("oracle asked to move in a terminal position")
        # Fast path: an immediate win dominates everything, so take it without solving the other branches.
        wins = [c for c in legal if _is_winning_move(position, mask, c)]
        if wins:
            wins.sort(key=_CENTER_ORDER.index)
            return wins[0]
        values = move_values(state, weak=False)
        best = max(values.values())
        best_cols = [c for c, v in values.items() if v == best]
        best_cols.sort(key=_CENTER_ORDER.index)
        return best_cols[0]


class NearPerfectOracle:
    """A fast, tactically-perfect Connect 4 opponent — depth-limited negamax (default 10 plies) with the exact
    solver's tactical layer. Unlike the exact `OracleAgent` it plays the OPENING in milliseconds, so it's the
    strong reference the models train against and the top rung of the rating spine. It is EXACT once the search
    depth covers the plies left (endgames); earlier it is near-perfect, blundering only beyond its horizon."""

    kind = "oracle_depth"

    def __init__(self, depth: int = 10, solve_endgame: int = 0):
        self.depth = int(depth)
        # Opt-in EXACT-ENDGAME cutoff (empty-cell threshold): once a position is cheap to solve exactly, play the
        # provably-optimal move via the bitboard solver instead of the depth-limited search — this is both EXACT
        # (perfect endgame, no horizon blunder) and much FASTER than depth-searching the whole endgame, which is
        # what makes the oracle affordable as a play-off opponent over many games. 0 = pure depth-limited.
        self.solve_endgame = int(solve_endgame)

    def act(self, game, state, rng: random.Random) -> int:
        position, mask, _ = to_bitboard(state)
        legal = [c for c in range(WIDTH) if _can_play(mask, c)]
        if not legal:
            raise ValueError("oracle asked to move in a terminal position")
        wins = [c for c in legal if _is_winning_move(position, mask, c)]
        if wins:
            wins.sort(key=_CENTER_ORDER.index)
            return wins[0]
        if self.solve_endgame > 0 and (_TOTAL - bin(mask).count("1")) <= self.solve_endgame:
            optimal = optimal_columns(state)  # exact + fast once few cells remain
            if optimal:
                optimal.sort(key=_CENTER_ORDER.index)
                return optimal[0]
        possible = _possible_non_losing_moves(position, mask)
        candidates = legal if possible == 0 else [c for c in legal if possible & _column_mask(c)]
        moves = bin(mask).count("1")
        best_col, best_val = candidates[0], -_INF
        for c in sorted(candidates, key=_CENTER_ORDER.index):
            move = (mask + _bottom_mask_col(c)) & _column_mask(c)
            val = -_dl_negamax(position ^ mask, mask | move, moves + 1, self.depth - 1, -_INF, _INF)
            if val > best_val:
                best_val, best_col = val, c
        return best_col
