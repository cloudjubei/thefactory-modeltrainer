"""Othello 8x8 — the second game, and the first with NO solver: the transfer probe of the north star.

Composed from the rule-module library (harness/rules.py): move generation is `flank` over `grid_rays`, the
terminal test is `majority`; nothing here knows how to walk a board. Board indexed `row * 8 + col`, cells hold
0 (empty), 1 (player 0, black, moves first), 2 (player 1, white). Actions 0..63 place a disc on that cell; action
64 is PASS, legal ONLY when the mover has no flipping move and the opponent does — PASS lives at the end of the
action space as the design's "global bank" slot, so a future 2-D symmetry augmentation can leave it fixed.

`done` is DERIVED from the position (no empties, or neither player can move), so the state stays Markov without
a pass counter. No `symmetries()` yet: the augmenter is column-permutation-only (§C.21 follow-up), and exposing
64-cell dihedral permutations to it would corrupt training data rather than multiply it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from harness.rules import flank, grid_rays, majority

N = 8
CELLS = N * N
PASS = CELLS
_RAYS = grid_rays(N, N)
_CORNERS = (0, N - 1, (N - 1) * N, CELLS - 1)


@dataclass(frozen=True)
class OthelloState:
    board: tuple  # length 64, values in {0,1,2}
    to_move: int  # 0 or 1
    done: bool


def _moves(board: tuple, player: int) -> dict[int, list[int]]:
    me, opp = player + 1, 2 - player
    out: dict[int, list[int]] = {}
    for cell in range(CELLS):
        if board[cell] == 0:
            got = flank(board, cell, _RAYS, me, opp)
            if got:
                out[cell] = got
    return out


def _finished(board: tuple) -> bool:
    if all(v != 0 for v in board):
        return True
    return not _moves(board, 0) and not _moves(board, 1)


class Othello:
    name = "othello"
    num_players = 2
    num_actions = CELLS + 1
    board_shape = (N, N)

    def initial_state(self, rng: random.Random | None = None) -> OthelloState:
        board = [0] * CELLS
        board[3 * N + 3] = 2
        board[4 * N + 4] = 2
        board[3 * N + 4] = 1
        board[4 * N + 3] = 1
        return OthelloState(board=tuple(board), to_move=0, done=False)

    def current_player(self, state: OthelloState) -> int:
        return state.to_move

    def legal_actions(self, state: OthelloState) -> list[int]:
        if state.done:
            return []
        moves = _moves(state.board, state.to_move)
        return sorted(moves) if moves else [PASS]

    def step(self, state: OthelloState, action: int, rng: random.Random | None = None) -> OthelloState:
        if state.done:
            raise ValueError("game is over")
        if action == PASS:
            if _moves(state.board, state.to_move):
                raise ValueError("illegal pass: a flipping move is available")
            return OthelloState(board=state.board, to_move=1 - state.to_move, done=_finished(state.board))
        if not (0 <= action < CELLS):
            raise ValueError(f"action {action} out of range")
        me, opp = state.to_move + 1, 2 - state.to_move
        captured = flank(state.board, action, _RAYS, me, opp)
        if not captured:
            raise ValueError(f"illegal move: cell {action} flips nothing")
        board = list(state.board)
        board[action] = me
        for c in captured:
            board[c] = me
        board_t = tuple(board)
        return OthelloState(board=board_t, to_move=1 - state.to_move, done=_finished(board_t))

    def is_terminal(self, state: OthelloState) -> bool:
        return state.done

    def winner(self, state: OthelloState) -> int | None:
        return majority(state.board) if state.done else None

    def returns(self, state: OthelloState) -> list[float]:
        w = self.winner(state)
        if w is None:
            return [0.0, 0.0]
        payoff = [-1.0, -1.0]
        payoff[w] = 1.0
        return payoff

    def observation(self, state: OthelloState, player: int) -> list[float]:
        own = player + 1
        obs = [1.0 if v == own else -1.0 if v != 0 else 0.0 for v in state.board]
        obs.append(1.0 if state.to_move == player else 0.0)
        return obs

    def heuristic_action(self, state: OthelloState, rng: random.Random) -> int:
        """A corner if one is available, else the move that flips the most discs, else pass."""
        moves = _moves(state.board, state.to_move)
        if not moves:
            return PASS
        for c in _CORNERS:
            if c in moves:
                return c
        best = max(len(v) for v in moves.values())
        return rng.choice(sorted(c for c, v in moves.items() if len(v) == best))

    def render(self, state: OthelloState) -> str:
        glyph = {0: ".", 1: "X", 2: "O"}
        rows = [" ".join(glyph[state.board[r * N + c]] for c in range(N)) for r in range(N)]
        x = sum(1 for v in state.board if v == 1)
        o = sum(1 for v in state.board if v == 2)
        if state.done:
            w = self.winner(state)
            status = "draw" if w is None else f"{glyph[w + 1]} wins"
        else:
            status = f"{glyph[state.to_move + 1]} to move"
        return "\n".join(rows) + f"   [X {x} - O {o}; {status}]"

    def action_label(self, state: OthelloState, action: int) -> str:
        if action == PASS:
            return "pass"
        return f"{'abcdefgh'[action % N]}{action // N + 1}"

    def state_key(self, state: OthelloState) -> tuple:
        return (state.board, state.to_move)
