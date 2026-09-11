"""Checkers (English draughts, 8x8) — the third game, and the first with SUB-TURNS and a second piece type.

Othello proved the rule-module library composes; checkers is what actually stresses the §C.21 design, because it
breaks three assumptions every earlier game satisfied:

  1. A TURN IS NOT A MOVE. A multi-jump is one turn made of several decisions, so `to_move` does not alternate
     on every `step`. The state carries `jumping` — the cell a piece must continue capturing from — and the
     harness needs no new concept: `current_player` simply keeps returning the same player until the sequence
     ends. This is the design's sub-turn `is_switch` hook, paid for by one field.
  2. PIECES HAVE TYPES. Men and kings move differently, so the position no longer fits the 2-plane
     (own/opponent) encoding every earlier game used. The game declares `input_planes = 4` and emits its own
     planes; `encode` reshapes them directly (see harness/neural.py).
  3. HALF THE BOARD IS DEAD. Play happens only on dark squares, so 32 of 64 cells are permanently empty. The
     game declares `valid_mask`, and the net's masked pooling (§C.21 Increment 1) excludes those cells from its
     global statistics instead of averaging 32 structural zeros into every feature.

Board indexed `row * 8 + col`; cells hold 0 empty, 1/2 a man of player 0/1, 3/4 a king of player 0/1. Player 0
moves toward row 0, player 1 toward row 7. Actions are FACTORED as `direction * 64 + from_cell` (4 diagonal
directions x 64 cells = 256), the design's C_a·H·W action plane stack — a conv policy head can later predict
them positionally without re-encoding anything here.

Rules implemented are standard English draughts: capture is MANDATORY, men capture forward only, kings move and
capture one square in all four diagonals (no flying kings), a man promoting mid-sequence ENDS the turn, and a
player with no legal move LOSES. Draw by the 40-move idle rule (no capture and no man move).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from harness.rules import DIRS4, diag_jumps, diag_steps

N = 8
CELLS = N * N
NUM_ACTIONS = len(DIRS4) * CELLS
IDLE_LIMIT = 80  # half-moves without a capture or a man move

_STEPS = diag_steps(N, N)
_JUMPS = diag_jumps(N, N)
_DARK = tuple(1.0 if (c // N + c % N) % 2 == 1 else 0.0 for c in range(CELLS))

_OWNER = {1: 0, 3: 0, 2: 1, 4: 1}
_FORWARD = (-1, 1)  # player 0 moves toward row 0, player 1 toward row 7
_PROMOTION_ROW = (0, N - 1)


def _is_king(v: int) -> bool:
    return v >= 3


def _dirs_for(v: int) -> tuple[tuple[int, int], ...]:
    if _is_king(v):
        return DIRS4
    fwd = _FORWARD[_OWNER[v]]
    return tuple(d for d in DIRS4 if d[0] == fwd)


def _jumps_from(board: tuple, cell: int) -> list[int]:
    """Action ids for every legal capture by the piece on `cell`."""
    v = board[cell]
    if v == 0:
        return []
    me = _OWNER[v]
    out = []
    for d in _dirs_for(v):
        hop = _JUMPS.get((cell, d))
        if hop is None:
            continue
        over, land = hop
        victim = board[over]
        if victim != 0 and _OWNER[victim] != me and board[land] == 0:
            out.append(DIRS4.index(d) * CELLS + cell)
    return out


def _steps_from(board: tuple, cell: int) -> list[int]:
    v = board[cell]
    if v == 0:
        return []
    out = []
    for d in _dirs_for(v):
        dest = _STEPS.get((cell, d))
        if dest is not None and board[dest] == 0:
            out.append(DIRS4.index(d) * CELLS + cell)
    return out


def _legal(board: tuple, to_move: int, jumping: int | None) -> list[int]:
    """Mandatory capture: if ANY capture exists the quiet moves are not legal. Mid-sequence, only the jumping
    piece may act, which is what makes a multi-jump one turn rather than several."""
    if jumping is not None:
        return sorted(_jumps_from(board, jumping))
    mine = [c for c in range(CELLS) if board[c] != 0 and _OWNER[board[c]] == to_move]
    jumps = [a for c in mine for a in _jumps_from(board, c)]
    if jumps:
        return sorted(jumps)
    return sorted(a for c in mine for a in _steps_from(board, c))


@dataclass(frozen=True)
class CheckersState:
    board: tuple  # length 64, values in {0,1,2,3,4}
    to_move: int
    jumping: int | None  # the cell a multi-jump must continue from, else None
    idle: int
    done: bool


def _settle(board: tuple, to_move: int, jumping: int | None, idle: int) -> CheckersState:
    done = idle >= IDLE_LIMIT or not _legal(board, to_move, jumping)
    return CheckersState(board=board, to_move=to_move, jumping=jumping, idle=idle, done=done)


class Checkers:
    name = "checkers"
    num_players = 2
    num_actions = NUM_ACTIONS
    board_shape = (N, N)
    input_planes = 4  # own men, own kings, opponent men, opponent kings
    valid_mask = _DARK

    def initial_state(self, rng: random.Random | None = None) -> CheckersState:
        board = [0] * CELLS
        for c in range(CELLS):
            if not _DARK[c]:
                continue
            if c // N <= 2:
                board[c] = 2
            elif c // N >= N - 3:
                board[c] = 1
        return _settle(tuple(board), 0, None, 0)

    def current_player(self, state: CheckersState) -> int:
        return state.to_move

    def legal_actions(self, state: CheckersState) -> list[int]:
        if state.done:
            return []
        return _legal(state.board, state.to_move, state.jumping)

    def step(self, state: CheckersState, action: int, rng: random.Random | None = None) -> CheckersState:
        if state.done:
            raise ValueError("game is over")
        if not (0 <= action < NUM_ACTIONS):
            raise ValueError(f"action {action} out of range")
        if action not in self.legal_actions(state):
            raise ValueError(f"illegal action {action} ({self.action_label(state, action)})")
        d = DIRS4[action // CELLS]
        cell = action % CELLS
        board = list(state.board)
        v = board[cell]
        hop = _JUMPS.get((cell, d))
        captured = hop is not None and hop[1] < CELLS and board[hop[0]] != 0 \
            and _OWNER[board[hop[0]]] != state.to_move and board[hop[1]] == 0
        if captured:
            over, dest = hop
            board[over] = 0
        else:
            dest = _STEPS[cell, d]
        board[cell] = 0
        board[dest] = v

        promoted = not _is_king(v) and dest // N == _PROMOTION_ROW[state.to_move]
        if promoted:
            board[dest] = v + 2
        board_t = tuple(board)

        # A man promoting ENDS the turn even mid-sequence (English draughts), so the new king cannot immediately
        # keep capturing with its new powers on the same turn.
        if captured and not promoted and _jumps_from(board_t, dest):
            return _settle(board_t, state.to_move, dest, state.idle)
        idle = 0 if (captured or not _is_king(v)) else state.idle + 1
        return _settle(board_t, 1 - state.to_move, None, idle)

    def is_terminal(self, state: CheckersState) -> bool:
        return state.done

    def winner(self, state: CheckersState) -> int | None:
        if not state.done or state.idle >= IDLE_LIMIT:
            return None
        return 1 - state.to_move  # the player with no legal move loses

    def returns(self, state: CheckersState) -> list[float]:
        w = self.winner(state)
        if w is None:
            return [0.0, 0.0]
        payoff = [-1.0, -1.0]
        payoff[w] = 1.0
        return payoff

    def observation(self, state: CheckersState, player: int) -> list[float]:
        """The 4 planes the net consumes (own men, own kings, opponent men, opponent kings), then the
        to-move flag and a mid-multi-jump flag."""
        own_man, own_king = (1, 3) if player == 0 else (2, 4)
        opp_man, opp_king = (2, 4) if player == 0 else (1, 3)
        obs: list[float] = []
        for piece in (own_man, own_king, opp_man, opp_king):
            obs.extend(1.0 if v == piece else 0.0 for v in state.board)
        obs.append(1.0 if state.to_move == player else 0.0)
        obs.append(1.0 if state.jumping is not None else 0.0)
        return obs

    def heuristic_action(self, state: CheckersState, rng: random.Random) -> int:
        """Captures are forced anyway, so the judgement is among quiet moves: promote when possible, else
        advance, preferring the back rank to stay home (a cheap but real draughts principle)."""
        actions = self.legal_actions(state)
        promoting = [a for a in actions if self._promotes(state, a)]
        if promoting:
            return rng.choice(promoting)
        back = _PROMOTION_ROW[1 - state.to_move]
        forward = [a for a in actions if (a % CELLS) // N != back]
        return rng.choice(forward or actions)

    def _promotes(self, state: CheckersState, action: int) -> bool:
        cell = action % CELLS
        v = state.board[cell]
        if _is_king(v):
            return False
        d = DIRS4[action // CELLS]
        hop = _JUMPS.get((cell, d))
        dest = hop[1] if (hop and state.board[hop[1]] == 0 and state.board[hop[0]] != 0) else _STEPS.get((cell, d))
        return dest is not None and dest // N == _PROMOTION_ROW[state.to_move]

    def render(self, state: CheckersState) -> str:
        glyph = {0: ".", 1: "x", 2: "o", 3: "X", 4: "O"}
        rows = [" ".join(glyph[state.board[r * N + c]] for c in range(N)) for r in range(N)]
        counts = {p: sum(1 for v in state.board if v == p) for p in (1, 2, 3, 4)}
        if state.done:
            w = self.winner(state)
            status = "draw" if w is None else f"{'xo'[w]} wins"
        else:
            status = f"{'xo'[state.to_move]} to move" + (" (jumping)" if state.jumping is not None else "")
        return "\n".join(rows) + (f"   [x {counts[1]}+{counts[3]}K - o {counts[2]}+{counts[4]}K; {status}]")

    def action_label(self, state: CheckersState, action: int) -> str:
        if not (0 <= action < NUM_ACTIONS):
            return f"?{action}"
        cell = action % CELLS
        d = DIRS4[action // CELLS]
        hop = _JUMPS.get((cell, d))
        dest = _STEPS.get((cell, d))
        src = f"{'abcdefgh'[cell % N]}{N - cell // N}"
        if hop is not None and state.board[hop[1]] == 0 and state.board[hop[0]] != 0:
            land = hop[1]
            return f"{src}x{'abcdefgh'[land % N]}{N - land // N}"
        if dest is None:
            return f"{src}?"
        return f"{src}-{'abcdefgh'[dest % N]}{N - dest // N}"

    def state_key(self, state: CheckersState) -> tuple:
        # `idle` belongs in the key: two identical positions with different idle counts have different values
        # near the draw limit, so collapsing them in the transposition table would be wrong.
        return (state.board, state.to_move, state.jumping, state.idle)
