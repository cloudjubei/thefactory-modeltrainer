"""§C.50 — WRITTEN RULES: play knowledge as logical statements, verified against exact move values.

A PREDICATE is a yes/no fact about a position and one legal move (`wins`, `blocks`, `corner`, ...). A RULE is a
conjunction of predicates and their negations, read as `IF ∃m: L1(m) ∧ ... ∧ Lk(m) THEN play such an m`; it
RECOMMENDS every legal move meeting all its literals and FIRES when there is one. A PLAYBOOK is an ordered decision
list: the first rule that fires decides.

The tactical predicates are generic: they use only the game's `step`, `is_terminal`, `winner` and "their view" —
the same board with the other side to move, which is how an opponent's threat is read (the square they would win
on if it were their turn). The geometry predicates name tic-tac-toe's cell classes.

`evaluate` scores a playbook against exact move values: a recommendation is CORRECT only when every move it
recommends is value-optimal, so a rule cannot pass by offering one good move among bad ones."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache


def their_view(state):
    """The same board with the other side to move."""
    return replace(state, to_move=1 - state.to_move)


def _wins(game, state, move) -> bool:
    child = game.step(state, move)
    return game.is_terminal(child) and game.winner(child) == game.current_player(state)


def _winning_moves(game, state) -> list:
    return [b for b in game.legal_actions(state) if _wins(game, state, b)]


def _child(game, state, move):
    child = game.step(state, move)
    return None if game.is_terminal(child) else child


def wins(game, state, move) -> bool:
    return _wins(game, state, move)


def blocks(game, state, move) -> bool:
    """The opponent would win by playing `move` if it were their turn."""
    return _wins(game, their_view(state), move)


def gives_win(game, state, move) -> bool:
    """After `move` the opponent can win at once."""
    child = _child(game, state, move)
    return child is not None and bool(_winning_moves(game, child))


def makes_threat(game, state, move) -> bool:
    """After `move` we would have a winning move if it were our turn again."""
    child = _child(game, state, move)
    return child is not None and bool(_winning_moves(game, their_view(child)))


def forks(game, state, move) -> bool:
    """After `move` we have two or more winning moves and the opponent has none: one block cannot stop both."""
    child = _child(game, state, move)
    return child is not None and not _winning_moves(game, child) \
        and len(_winning_moves(game, their_view(child))) >= 2


def opp_fork_at(game, state, move) -> bool:
    """The opponent would fork by playing `move` if it were their turn."""
    return forks(game, their_view(state), move)


def gives_fork(game, state, move) -> bool:
    """After `move` the opponent has a forking reply."""
    child = _child(game, state, move)
    return child is not None and any(forks(game, child, b) for b in game.legal_actions(child))


@lru_cache(maxsize=None)
def forced_win(game, state, move, plies: int) -> bool:
    """Playing `move` wins by force within `plies` plies counting the move itself: every reply leaves a move that
    keeps the forced win within what remains (MIGO's layered win_k, as a bounded AND-OR search)."""
    mover = game.current_player(state)
    child = game.step(state, move)
    if game.is_terminal(child):
        return game.winner(child) == mover
    if plies < 3:
        return False
    for reply in game.legal_actions(child):
        after = game.step(child, reply)
        if game.is_terminal(after):
            return False
        if not any(forced_win(game, after, m, plies - 2) for m in game.legal_actions(after)):
            return False
    return True


def wins_in_5(game, state, move) -> bool:
    return forced_win(game, state, move, 5)


def _gives_loss(game, state, move, plies: int) -> bool:
    child = _child(game, state, move)
    return child is not None and any(forced_win(game, child, b, plies - 1) for b in game.legal_actions(child))


def gives_loss_in_4(game, state, move) -> bool:
    """After `move` the opponent can force a win within their next 3 plies."""
    return _gives_loss(game, state, move, 4)


def gives_loss_in_6(game, state, move) -> bool:
    """After `move` the opponent can force a win within their next 5 plies."""
    return _gives_loss(game, state, move, 6)


def _ttt_only(game) -> None:
    if game.name != "tictactoe":
        raise ValueError(f"this geometry predicate is defined for tic-tac-toe only, not {game.name}")


def centre(game, state, move) -> bool:
    _ttt_only(game)
    return move == 4


def corner(game, state, move) -> bool:
    _ttt_only(game)
    return move in (0, 2, 6, 8)


def side(game, state, move) -> bool:
    _ttt_only(game)
    return move in (1, 3, 5, 7)


def opposite_corner(game, state, move) -> bool:
    """A corner whose opposite corner holds the opponent's piece."""
    _ttt_only(game)
    return move in (0, 2, 6, 8) and state.board[8 - move] == (1 - state.to_move) + 1


PREDICATES = {f.__name__: f for f in (wins, blocks, gives_win, makes_threat, forks, opp_fork_at, gives_fork,
                                      centre, corner, side, opposite_corner, wins_in_5, gives_loss_in_4,
                                      gives_loss_in_6)}
LOOKAHEAD = {"wins": 1, "blocks": 1, "gives_win": 2, "makes_threat": 2, "forks": 2, "opp_fork_at": 2, "gives_fork": 3,
             "centre": 0, "corner": 0, "side": 0, "opposite_corner": 0, "wins_in_5": 5, "gives_loss_in_4": 4,
             "gives_loss_in_6": 6}


@dataclass(frozen=True)
class Rule:
    name: str
    literals: tuple

    def __post_init__(self):
        unknown = [p for p, _positive in self.literals if p not in PREDICATES]
        if unknown or not self.literals:
            raise ValueError(f"rule {self.name!r}: unknown predicate(s) {unknown} or no literals")

    def moves(self, game, state) -> list:
        """Every legal move meeting all the literals."""
        return [m for m in game.legal_actions(state)
                if all(PREDICATES[p](game, state, m) == positive for p, positive in self.literals)]

    def text(self) -> str:
        body = " ∧ ".join(f"{'' if positive else '¬'}{p}(m)" for p, positive in self.literals)
        return f"{self.name}: IF ∃m: {body} THEN play m"


def decide(book: list, game, state) -> tuple:
    """(index of the first rule that fires, the moves it recommends), or (None, []) when none fires."""
    for i, rule in enumerate(book):
        moves = rule.moves(game, state)
        if moves:
            return i, moves
    return None, []


def own_play_positions(book: list, game, root, player: int) -> list:
    """Every non-terminal position of `player` reached from `root` when `player` plays EVERY move the book
    recommends (every legal move where no rule fires) and the other side plays every legal move — each once. The
    set on which a playbook that is right "from the start" must be right."""
    level = {game.state_key(root): root}
    seen: set = set()
    out = []
    while level:
        nxt: dict = {}
        for k, s in level.items():
            if k in seen or game.is_terminal(s):
                continue
            seen.add(k)
            if game.current_player(s) == player:
                out.append(s)
                moves = decide(book, game, s)[1] or game.legal_actions(s)
            else:
                moves = game.legal_actions(s)
            for a in moves:
                child = game.step(s, a)
                nxt.setdefault(game.state_key(child), child)
        level = nxt
    return out


def evaluate(book: list, game, positions: list, values: list) -> dict:
    """Coverage and precision of `book` against exact move values (one {move: value to the mover} per position).
    Per rule: `fires`/`correct` on its own, and `first_fires`/`first_correct` where it is the rule that decides.
    For the book: `covered` positions where some rule fires, `correct` of those where the decision is optimal."""
    if len(positions) != len(values):
        raise ValueError(f"positions and values must be the same length, got {len(positions)} and {len(values)}")
    rows = [{"text": r.text(), "fires": 0, "correct": 0, "first_fires": 0, "first_correct": 0} for r in book]
    covered = correct = 0
    for state, vals in zip(positions, values):
        best = max(vals.values())
        optimal = {int(a) for a, v in vals.items() if v == best}
        decided = False
        for row, rule in zip(rows, book):
            moves = rule.moves(game, state)
            if not moves:
                continue
            ok = set(moves) <= optimal
            row["fires"] += 1
            row["correct"] += ok
            if not decided:
                decided = True
                row["first_fires"] += 1
                row["first_correct"] += ok
                covered += 1
                correct += ok
    return {"positions": len(positions), "covered": covered, "correct": correct, "rules": rows}


def tactics() -> list:
    """The generic tactical playbook: win now, else block their win, else fork."""
    return [Rule("win", (("wins", True),)), Rule("block", (("blocks", True),)), Rule("fork", (("forks", True),))]


def newell_simon() -> list:
    """Newell & Simon's tic-tac-toe rules in their classic order, written in this language."""
    return tactics() + [
        Rule("block fork", (("opp_fork_at", True), ("gives_fork", False))),
        Rule("force without a fork", (("makes_threat", True), ("gives_fork", False))),
        Rule("centre", (("centre", True),)),
        Rule("opposite corner", (("opposite_corner", True),)),
        Rule("empty corner", (("corner", True),)),
        Rule("empty side", (("side", True),)),
    ]
