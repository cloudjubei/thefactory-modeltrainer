"""§3.7 — best-first proving with no solver: proof-number search over the game tree, with maps as proofs of whole
subtrees. harness.self_proof's `Prover` works depth first and sinks its whole budget into the first move it tries
(65 of 66 unproved positions at 4,000 calls). Here each of the side's positions is an OR node, proved when one of its
safe moves is; each move is an AND node over the side's next positions, proved when all are. Every node carries a
proof number (how many unknown positions must still be proved, at least) and a disproof number; each step expands the
unknown position on the path that keeps choosing the child with the smallest proof number at an OR node and the
smallest disproof number at an AND node, so effort goes where a proof is closest. A new position is first given to
the `Prover`'s leaf step: a map that wins every line from it (where its walk is small, `search_below`; the empty map
everywhere) proves it outright. Positions proved by a move are stored with the `Prover`'s, so with `compose` the maps'
walks stop at them. The result is the `Prover`'s strategy, checked by harness.composed or replayed. A solver may
grade it afterwards; it is never a means."""
from __future__ import annotations

import time

from harness.self_proof import Prover, _replies

INF = float("inf")


class _Node:
    __slots__ = ("state", "key", "kind", "move", "pn", "dn", "children", "parents", "expanded")

    def __init__(self, state, key, kind, move=None):
        self.state, self.key, self.kind, self.move = state, key, kind, move
        self.pn, self.dn = 1, 1
        self.children: list = []
        self.parents: list = []
        self.expanded = False


class BestFirst:
    def __init__(self, prover: Prover, max_evaluations: int | None = None, deadline: float | None = None,
                 maps: bool = True):
        """`max_evaluations` bounds the positions evaluated, the unit of the `Prover`'s calls. With `maps` off no map
        is tried: plain proof-number search, every position proved by a move, the measure of what the maps save."""
        self.prover, self.facts, self.game, self.maps = prover, prover.facts, prover.game, maps
        self.max_evaluations, self.deadline = max_evaluations, deadline
        self.nodes: dict = {}
        self.expansions = 0
        self.evaluations = 0

    def prove(self, root):
        """True when `root` is proved won for the side to move there (its proof in `prover.proved`), False when it is
        disproved, None when the budget ran out first."""
        top = self._own(root)
        while top.pn != 0 and top.dn != 0 and not self._spent():
            node = top
            while node.expanded:
                node = min(node.children, key=(lambda c: c.pn) if node.kind == "or" else (lambda c: c.dn))
            self._expand(node)
            self._update(node)
        return True if top.pn == 0 else False if top.dn == 0 else None

    def _spent(self) -> bool:
        return ((self.max_evaluations is not None and self.evaluations >= self.max_evaluations)
                or (self.deadline is not None and time.monotonic() > self.deadline))

    def _own(self, state) -> _Node:
        key = self.game.state_key(state)
        node = self.nodes.get(key)
        if node is None:
            node = self.nodes[key] = _Node(state, key, "or")
            self._evaluate(node)
        return node

    def _evaluate(self, node: _Node) -> None:
        self.evaluations += 1
        prover, state, key = self.prover, node.state, node.key
        if key in prover.proved:
            node.pn, node.dn = 0, INF
        elif not self.facts.safe_moves(state):
            node.pn, node.dn = INF, 0
        elif not self.maps:
            return
        elif prover._small(state) and prover._leaf(state, key) is True:
            node.pn, node.dn = 0, INF
        elif not prover._small(state) and prover._wins(state, {}):
            prover.proved[key] = {"leaf": 0}
            node.pn, node.dn = 0, INF

    def _expand(self, node: _Node) -> None:
        self.expansions += 1
        node.expanded = True
        player = self.game.current_player(node.state)
        for action in self.facts.safe_moves(node.state):
            move = _Node(node.state, (node.key, action), "and", action)
            move.parents.append(node)
            node.children.append(move)
            replies = _replies(self.game, player, self.game.step(node.state, action))
            move.expanded = True
            if replies is None:
                move.pn, move.dn = INF, 0
                continue
            for reply in replies:
                child = self._own(reply)
                child.parents.append(move)
                move.children.append(child)
            self._settle(move)
        self._settle(node)

    def _settle(self, node: _Node) -> None:
        if not node.children:
            if node.kind == "and" and node.dn != 0:
                node.pn, node.dn = 0, INF
            elif node.kind == "or" and node.expanded:
                node.pn, node.dn = INF, 0
            return
        if node.kind == "or":
            node.pn, node.dn = min(c.pn for c in node.children), sum(c.dn for c in node.children)
        else:
            node.pn, node.dn = sum(c.pn for c in node.children), min(c.dn for c in node.children)
        if node.kind == "or" and node.pn == 0 and node.key not in self.prover.proved:
            winner = next(c for c in node.children if c.pn == 0)
            self.prover.proved[node.key] = {"move": winner.move}

    def _update(self, node: _Node) -> None:
        stack = list(node.parents)
        while stack:
            n = stack.pop()
            before = (n.pn, n.dn)
            self._settle(n)
            if (n.pn, n.dn) != before:
                stack.extend(n.parents)
