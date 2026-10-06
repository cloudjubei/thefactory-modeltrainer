"""Exact Connect-4 values to the side to move — the process's "solve this position" step, as the hybrid uses it.
Recorded label-cache values answer first (a fully recorded position gives every move's value, and each move's child
inherits minus that value); everything else is solved in one batch per call through a worker pool
(harness.c4_oracle.solve) and remembered. `counts` says how many answers were recorded and how many solved.

Solves are submitted in slices of at most SUBMIT_LIMIT, each drained before the next: one ProcessPoolExecutor.map over
~190,000 jobs deadlocked on Python 3.10 (2026-10-06) — every submit writes a wake-up byte to a pipe the executor's
manager thread reads, and once that pipe fills while the manager is busy, both sides wait forever (CPython gh-105829,
fixed in 3.11.5 and 3.12)."""
from __future__ import annotations

SUBMIT_LIMIT = 2048


class ExactValues:
    def __init__(self, game, pool, rows: list):
        from games.connect4 import C4State

        self.game, self.pool = game, pool
        self.moves: dict = {}
        self.position: dict = {}
        self.counts = {"recorded": 0, "solved": 0}
        for row in rows:
            s = C4State(tuple(row["board"]), row["to_move"], None, False)
            values = {int(a): int(v) for a, v in row["values"].items()}
            for a, v in values.items():
                child = game.step(s, a)
                if not game.is_terminal(child):
                    self.position[game.state_key(child)] = -v
            if set(values) == set(game.legal_actions(s)):
                self.moves[game.state_key(s)] = values
                self.position[game.state_key(s)] = max(values.values())

    def positions(self, states: list) -> list:
        """Each position's exact value to the side to move."""
        from harness.c4_oracle import solve

        key = self.game.state_key
        todo = list({key(s): s for s in states if key(s) not in self.position}.values())
        for start in range(0, len(todo), SUBMIT_LIMIT):
            part = todo[start:start + SUBMIT_LIMIT]
            for s, v in zip(part, list(self.pool.map(solve, [(s.board, s.to_move) for s in part], chunksize=2))):
                self.position[key(s)] = int(v)
        self.counts["solved"] += len(todo)
        self.counts["recorded"] += len(states) - len(todo)
        return [self.position[key(s)] for s in states]

    def move_values(self, states: list) -> list:
        """Each position's {move: exact value to the mover}."""
        key = self.game.state_key
        todo = [s for s in states if key(s) not in self.moves]
        children = [(s, a, self.game.step(s, a)) for s in todo for a in self.game.legal_actions(s)]
        values = iter(self.positions([c for _s, _a, c in children if not self.game.is_terminal(c)]))
        for s, a, c in children:
            mover = self.game.current_player(s)
            v = round(self.game.returns(c)[mover]) if self.game.is_terminal(c) else -next(values)
            self.moves.setdefault(key(s), {})[a] = v
        return [self.moves[key(s)] for s in states]
