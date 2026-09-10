"""§C.21 RULE-MODULE LIBRARY — the reusable effect primitives a game is COMPOSED from.

The unified-encoding design's bankable win was never a shared trunk; it was this: small pure primitives (rays,
flank capture, majority) that a game file assembles rather than re-implements, so a new game is one file and a
rule change is a one-clause change to a predicate. Everything here is board-geometry and effect arithmetic with
no knowledge of any game. Boards are flat tuples of ints — 0 empty, 1 player-0, 2 player-1 — the convention the
whole harness (Connect-4, TicTacToe, the encoder) already uses.

Deliberately minimal: only what the first consumer (Othello) exercises. Primitives for jumps, rotation, mills and
promotion arrive with the games that need them; unused primitives are untested primitives.
"""
from __future__ import annotations

DIRS8: tuple[tuple[int, int], ...] = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))

Rays = dict[tuple[int, tuple[int, int]], tuple[int, ...]]


def grid_rays(h: int, w: int) -> Rays:
    """For every cell and every one of the 8 directions, the cells walked OUTWARD to the board edge, nearest
    first. Precomputed once per board so move generation is a table walk, not arithmetic."""
    rays: Rays = {}
    for r in range(h):
        for c in range(w):
            for dr, dc in DIRS8:
                out = []
                rr, cc = r + dr, c + dc
                while 0 <= rr < h and 0 <= cc < w:
                    out.append(rr * w + cc)
                    rr += dr
                    cc += dc
                rays[r * w + c, (dr, dc)] = tuple(out)
    return rays


def flank(board: tuple, cell: int, rays: Rays, me: int, opp: int) -> list[int]:
    """The cells captured if `me` plays on the EMPTY `cell`: along each direction, a contiguous run of `opp`
    that is terminated by one of `me`. An unterminated run (edge, empty, or nothing there) captures nothing."""
    if board[cell] != 0:
        return []
    captured: list[int] = []
    for d in DIRS8:
        run: list[int] = []
        for nxt in rays[cell, d]:
            v = board[nxt]
            if v == opp:
                run.append(nxt)
            else:
                if v == me and run:
                    captured.extend(run)
                break
    return captured


def majority(board: tuple) -> int | None:
    """The player (0 or 1) holding more discs, or None on a tie."""
    ones = sum(1 for v in board if v == 1)
    twos = sum(1 for v in board if v == 2)
    if ones == twos:
        return None
    return 0 if ones > twos else 1
