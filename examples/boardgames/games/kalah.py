"""Kalah(m, n) — m holes a side, n counters a hole, under the rules of Irving, Donkers & Uiterwijk (2000), "Solving
Kalah", ICGA Journal 23(3), whose solved values and perfect games are this module's outside check.

`pits` holds South's holes 0..m-1 then North's; each side numbers its holes from 0 in sowing order, so hole m-1 is
next to its owner's store and hole i faces the opponent's hole m-1-i. A move lifts every counter from one of the
mover's holes and sows them anticlockwise, into the mover's store but never the opponent's. A last counter in the
mover's store moves again; a last counter in an empty hole of the mover takes it and the opposite hole into the store,
even when that hole is empty. A move that leaves either side empty ends the game, and each player stores what is left
on their own side. Nothing ever leaves a store, which bounds the game: every move either stores a counter or moves
counters nearer their owner's store (`progress`). The future of a position depends only on the counters still in
play, so the exact solver memoises the margin still to come on the holes alone, read from the side to move.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class KalahState:
    pits: tuple
    stores: tuple
    to_move: int
    done: bool


def _sow(own: tuple, opp: tuple, hole: int) -> tuple[tuple, tuple, int, bool, int]:
    """One move from the mover's side: (own holes, opponent holes, counters stored, moves again, counters captured)."""
    m = len(own)
    ring = list(own) + [0] + list(opp)
    seeds, ring[hole], pos = ring[hole], 0, hole
    for _ in range(seeds):
        pos = (pos + 1) % len(ring)
        ring[pos] += 1
    captured = 0
    if pos < m and ring[pos] == 1:
        captured = 1 + ring[2 * m - pos]
        ring[pos] = ring[2 * m - pos] = 0
        ring[m] += captured
    return tuple(ring[:m]), tuple(ring[m + 1:]), ring[m], pos == m, captured


class Kalah:
    name = "kalah"
    num_players = 2
    input_planes = 1

    def __init__(self, holes: int = 6, counters: int = 4):
        if holes < 1:
            raise ValueError(f"Kalah needs at least one hole a side, got {holes} holes")
        if counters < 1:
            raise ValueError(f"Kalah needs at least one counter a hole, got {counters} counters")
        self.holes, self.counters = holes, counters
        self.num_actions = holes
        self.board_shape = (2, holes + 1)
        self._future: dict = {}

    def initial_state(self, rng: random.Random | None = None) -> KalahState:
        return KalahState(pits=(self.counters,) * (2 * self.holes), stores=(0, 0), to_move=0, done=False)

    def current_player(self, state: KalahState) -> int:
        return state.to_move

    def _sides(self, state: KalahState, player: int) -> tuple[tuple, tuple]:
        m = self.holes
        south, north = state.pits[:m], state.pits[m:]
        return (south, north) if player == 0 else (north, south)

    def legal_actions(self, state: KalahState) -> list[int]:
        own, _ = self._sides(state, state.to_move)
        return [h for h in range(self.holes) if own[h]]

    def step(self, state: KalahState, action: int, rng: random.Random | None = None) -> KalahState:
        if action not in self.legal_actions(state):
            raise ValueError(f"illegal move: hole {action} is empty or not one of the mover's {self.holes} holes")
        me = state.to_move
        own, opp = self._sides(state, me)
        own, opp, stored, again, _ = _sow(own, opp, action)
        stores = [0, 0]
        stores[me] = state.stores[me] + stored
        stores[1 - me] = state.stores[1 - me]
        done = not any(own) or not any(opp)
        if done:
            stores[me] += sum(own)
            stores[1 - me] += sum(opp)
            own, opp = (0,) * self.holes, (0,) * self.holes
        pits = own + opp if me == 0 else opp + own
        return KalahState(pits=pits, stores=tuple(stores), to_move=me if again else 1 - me, done=done)

    def is_terminal(self, state: KalahState) -> bool:
        return state.done

    def winner(self, state: KalahState) -> int | None:
        if not state.done or state.stores[0] == state.stores[1]:
            return None
        return 0 if state.stores[0] > state.stores[1] else 1

    def returns(self, state: KalahState) -> list[float]:
        w = self.winner(state)
        if w is None:
            return [0.0, 0.0]
        payoff = [-1.0, -1.0]
        payoff[w] = 1.0
        return payoff

    def observation(self, state: KalahState, player: int) -> list[float]:
        """One (2, m+1) plane of counter shares: the viewer's holes then store, under them the opponent's holes
        reversed (so each hole sits over the one it captures) then store; a final flag marks the viewer to move."""
        own, opp = self._sides(state, player)
        total = 2 * self.holes * self.counters
        row = list(own) + [state.stores[player]] + list(reversed(opp)) + [state.stores[1 - player]]
        return [v / total for v in row] + [1.0 if state.to_move == player else 0.0]

    def heuristic_action(self, state: KalahState, rng: random.Random) -> int:
        own, opp = self._sides(state, state.to_move)

        def rank(hole: int) -> tuple:
            _, _, _, again, captured = _sow(own, opp, hole)
            return (2, 0, hole) if again else (1, captured, hole) if captured else (0, 0, hole)

        return max(self.legal_actions(state), key=rank)

    def render(self, state: KalahState) -> str:
        m = self.holes
        south, north = state.pits[:m], state.pits[m:]
        if state.done:
            w = self.winner(state)
            status = "draw" if w is None else f"{'SN'[w]} wins"
        else:
            status = f"{'SN'[state.to_move]} to move"
        return "\n".join([f"N   {' '.join(str(v) for v in reversed(north))}",
                          f"{state.stores[1]}  {'  ' * m}{state.stores[0]}",
                          f"S   {' '.join(str(v) for v in south)}   [{status}]"])

    def action_label(self, state: KalahState, action: int) -> str:
        return f"hole {action}"

    def state_key(self, state: KalahState) -> tuple:
        return (state.pits, state.stores, state.to_move)

    def ply(self, state: KalahState) -> int:
        return sum(state.stores)

    def progress(self, state: KalahState) -> tuple[int, int]:
        """Rises with every move: the counters stored, then the negated total distance of the counters in play to
        their owner's store."""
        m = self.holes
        distance = sum(v * (m - i) for side in (state.pits[:m], state.pits[m:]) for i, v in enumerate(side))
        return sum(state.stores), -distance

    def symmetries(self) -> list[tuple[list[int], list[int]]]:
        return [(list(range(2 * (self.holes + 1))), list(range(self.holes)))]

    def canonical_key(self, state: KalahState) -> int:
        """The position read from the side to move, so a position and its side-swapped image share a key."""
        own, opp = self._sides(state, state.to_move)
        me = state.to_move
        key = 0
        for v in own + opp + (state.stores[me], state.stores[1 - me]):
            key = key * (2 * self.holes * self.counters + 1) + v
        return key

    def position_value(self, state: KalahState, book=None) -> int:
        """Exact value to the side to move: +1 win, 0 draw, -1 loss. `book` is accepted for interface parity."""
        m = self.margin(state)
        return (m > 0) - (m < 0)

    def margin(self, state: KalahState) -> int:
        """The side to move's final store minus the opponent's under perfect play by both."""
        me = state.to_move
        lead = state.stores[me] - state.stores[1 - me]
        return lead if state.done else lead + self._best(*self._sides(state, me))

    def move_margin(self, state: KalahState, action: int) -> int:
        """The side to move's final margin when it plays `action` and both play perfectly after."""
        me = state.to_move
        own, opp = self._sides(state, me)
        return state.stores[me] - state.stores[1 - me] + self._after(own, opp, action)

    def optimal_actions(self, state: KalahState) -> list[int]:
        values = {a: self.move_margin(state, a) for a in self.legal_actions(state)}
        return sorted(a for a, v in values.items() if v == max(values.values())) if values else []

    def _after(self, own: tuple, opp: tuple, hole: int) -> int:
        own, opp, stored, again, _ = _sow(own, opp, hole)
        if not any(own) or not any(opp):
            return stored + sum(own) - sum(opp)
        return stored + (self._best(own, opp) if again else -self._best(opp, own))

    def _best(self, own: tuple, opp: tuple) -> int:
        key = (own, opp)
        if key not in self._future:
            self._future[key] = max(self._after(own, opp, h) for h in range(self.holes) if own[h])
        return self._future[key]
