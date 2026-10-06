"""Finding a steady state by SAT with counterexamples (§3.6 S1), after Waffle3z's reduction of WeakC4's steady states
to satisfiability (https://github.com/Waffle3z/weakc4lab). The unknowns are the priority map's levels on the cells
empty at the root; the constraints come from positions the rule reaches:

  - each constrained position reached must get a move from the rule (no undefined position), and that move must be
    one of its winning moves (`winning`, the search's oracle — the result is verified without it);
  - reaching is guarded: a constrained position counts only when the rule actually steers into it from the root
    (r_child <- r_parent AND the rule picks the move leading there), so a constraint never binds a position the
    map avoids.

The loop: solve, walk the candidate map from the root, and on the first failing line add every one of the rule's
positions on it as constraints, then solve again. A failing line always breaks one of its own constraints, so the
same candidate never comes back. UNSAT is therefore exact — no map in the language wins — and a map that walks clean
is returned for independent verification (harness.steady_state.verify). Requires python-sat (MIT)."""
from __future__ import annotations

import threading
import time

from harness.steady_state import Facts, choose


def _failures(facts: Facts, root, levels: dict, n_levels: int, cap: int, limit: int, deadline: float | None = None):
    """("clean", []) when every line is won; ("failed", lines) with up to `limit` failing lines, each the rule's
    positions from the root down to the failure (a failure's subtree is not walked further); ("cap", []) when the walk
    outgrows `cap` positions; ("timeout", []) when time.monotonic() passes `deadline` mid-walk."""
    game = facts.game
    player = game.current_player(root)
    owner: dict = {}
    seen: set = set()
    lines: list = []
    stack = [(root, None)]
    while stack and len(lines) < limit:
        s, last_own = stack.pop()
        key = game.state_key(s)
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > cap:
            return "cap", []
        if deadline is not None and len(seen) % 1024 == 1 and time.monotonic() > deadline:
            return "timeout", []
        if game.is_terminal(s):
            if game.winner(s) != player:
                lines.append(_line(owner, last_own))
            continue
        if game.current_player(s) == player:
            owner[key] = (s, last_own)
            move = choose(facts, s, levels, n_levels)
            if move is None:
                lines.append(_line(owner, key))
                continue
            stack.append((game.step(s, move), key))
        else:
            stack.extend((game.step(s, b), last_own) for b in game.legal_actions(s))
    return ("failed", lines) if lines else ("clean", [])


def _line(owner: dict, key) -> list:
    line = []
    while key is not None:
        s, key = owner[key]
        line.append(s)
    return line[::-1]


class _Encoding:
    def __init__(self, facts: Facts, root, n_levels: int, solver_name: str):
        from pysat.formula import IDPool
        from pysat.solvers import Solver

        self.facts, self.game, self.n_levels = facts, facts.game, n_levels
        self.player = self.game.current_player(root)
        self.cells = facts.empty_cells(root)
        self.pool = IDPool()
        self.solver = Solver(name=solver_name)
        self.constrained: set = set()
        for c in self.cells:
            xs = [self.x(c, k) for k in range(n_levels)]
            for i in range(len(xs)):
                for j in range(i + 1, len(xs)):
                    self.solver.add_clause([-xs[i], -xs[j]])
        self.solver.add_clause([self.r(root)])

    def x(self, cell, level) -> int:
        return self.pool.id(("x", cell, level))

    def r(self, state) -> int:
        return self.pool.id(("r", self.game.state_key(state)))

    def _children(self, state, move) -> list:
        game = self.game
        child = game.step(state, move)
        if game.is_terminal(child):
            return []
        if game.current_player(child) == self.player:
            return [child]
        return [c for b in game.legal_actions(child) for c in [game.step(child, b)] if not game.is_terminal(c)]

    def constrain(self, state, winning: set) -> None:
        key = self.game.state_key(state)
        if key in self.constrained:
            return
        self.constrained.add(key)
        add, r = self.solver.add_clause, self.r(state)
        wins, safe, cell = self.facts.of(state)
        forced = min(wins) if wins else (safe[0] if len(safe) == 1 else None)
        if forced is not None or not safe:
            if forced is None or forced not in winning:
                add([-r])
                return
            for child in self._children(state, forced):
                add([-r, self.r(child)])
            return
        e = [self.pool.id(("e", key, k)) for k in range(self.n_levels)]
        f = [self.pool.id(("f", key, k)) for k in range(self.n_levels)]
        for k in range(self.n_levels):
            xs = [self.x(cell[a], k) for a in safe]
            add([-e[k]] + xs)
            for i in range(len(xs)):
                add([e[k], -xs[i]] + [xs[j] for j in range(len(xs)) if j != i])
                for j in range(i + 1, len(xs)):
                    add([-e[k], -xs[i], -xs[j]])
            add([-f[k], e[k]])
            for j in range(k):
                add([-f[k], -e[j]])
        add([-r] + f)
        for a in safe:
            children = self._children(state, a) if a in winning else None
            for k in range(self.n_levels):
                guard = [-r, -self.x(cell[a], k), -f[k]]
                if children is None:
                    add(guard)
                else:
                    for child in children:
                        add(guard + [self.r(child)])

    def hint(self, levels: dict) -> None:
        """Start the solver at `levels`: each cell's variables are given the phase the map assigns them."""
        bad = {c: k for c, k in levels.items() if c not in self.cells or not 0 <= k < self.n_levels}
        if bad:
            raise ValueError(f"hint {bad} is outside the language: cells empty at the root, levels below "
                             f"{self.n_levels}")
        self.solver.set_phases([self.x(c, k) if levels.get(c) == k else -self.x(c, k)
                                for c in self.cells for k in range(self.n_levels)])

    def model_levels(self) -> dict:
        model = set(lit for lit in self.solver.get_model() if lit > 0)
        return {c: k for c in self.cells for k in range(self.n_levels) if self.x(c, k) in model}


def find_steady_state(facts: Facts, root, winning, n_levels: int, max_constraints: int, conflicts: int,
                      seconds: float, cap: int = 1_000_000, lines: int = 1, solver_name: str = "g4",
                      hint: dict | None = None, seeds: list | tuple = ()) -> dict:
    """Search for a priority map that wins every line from `root`. `winning(states)` gives each position's set of
    winning moves; each round constrains every position on up to `lines` failing lines. A `hint` (a map, e.g. local
    search's best) is where the solver starts, and `seeds` (positions, e.g. where that map needed exceptions) are
    constrained before the first solve; neither changes the answer, only the route to it. Returns {"status": "found" |
    "impossible" | "budget", "levels" (the map when found), "constraints"
    (positions constrained), "iterations"}; "budget" when the constraint count, a solve's conflict budget, the wall
    time (a hard deadline: a solve still running is interrupted, a walk stops) or the walk cap runs out."""
    enc = _Encoding(facts, root, n_levels, solver_name)
    if hint:
        enc.hint(hint)
    deadline, iterations = time.monotonic() + seconds, 0

    def out(status, levels=None):
        enc.solver.delete()
        return {"status": status, "levels": levels, "constraints": len(enc.constrained), "iterations": iterations}

    enc.constrain(root, winning([root])[0])
    seeds = list(seeds)
    for s, wins in zip(seeds, winning(seeds) if seeds else []):
        enc.constrain(s, wins)
    while True:
        remaining = deadline - time.monotonic()
        if len(enc.constrained) > max_constraints or remaining <= 0:
            return out("budget")
        enc.solver.conf_budget(conflicts)
        timer = threading.Timer(remaining, enc.solver.interrupt)
        timer.start()
        try:
            ok = enc.solver.solve_limited(expect_interrupt=True)
        finally:
            timer.cancel()
        enc.solver.clear_interrupt()
        iterations += 1
        if ok is None:
            return out("budget")
        if not ok:
            return out("impossible")
        levels = enc.model_levels()
        status, failing = _failures(facts, root, levels, n_levels, cap, lines, deadline)
        if status == "clean":
            return out("found", levels)
        if status in ("cap", "timeout"):
            return out("budget")
        fresh = list({enc.game.state_key(s): s for line in failing for s in line
                      if enc.game.state_key(s) not in enc.constrained}.values())
        if not fresh:
            raise RuntimeError("a failing line broke no constraint — the oracle's winning moves or the encoding is wrong")
        for s, wins in zip(fresh, winning(fresh) if fresh else []):
            enc.constrain(s, wins)
