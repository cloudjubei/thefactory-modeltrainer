"""Direct tests for harness/exact_values.py — exact Connect-4 values to the side to move: recorded label-cache values
first, then batched solves through a worker pool (the process's "solve this position" step)."""
from __future__ import annotations

import random

from games.connect4 import Connect4
from harness.exact_values import ExactValues

C4 = Connect4()


class _Pool:
    def __init__(self):
        self.batches = []

    def map(self, fn, jobs, chunksize=1):
        jobs = list(jobs)
        self.batches.append(len(jobs))
        return [7 for _ in jobs]


def _state(moves):
    s = C4.initial_state(random.Random(0))
    for m in moves:
        s = C4.step(s, m)
    return s


def _row(moves, values):
    s = _state(moves)
    return {"board": list(s.board), "to_move": s.to_move, "values": {str(a): v for a, v in values.items()}}


def test_a_fully_recorded_position_is_answered_without_solving_and_its_children_too():
    pool = _Pool()
    v = ExactValues(C4, pool, [_row([], {a: (1 if a == 3 else 0) for a in range(7)})])
    assert v.move_values([_state([])]) == [{a: (1 if a == 3 else 0) for a in range(7)}]
    assert v.positions([_state([]), _state([3]), _state([0])]) == [1, -1, 0]
    assert pool.batches == [] and v.counts == {"recorded": 3, "solved": 0}


def test_unknown_positions_are_solved_once_in_one_batch_and_remembered():
    pool = _Pool()
    v = ExactValues(C4, pool, [])
    assert v.positions([_state([1]), _state([2]), _state([1])]) == [7, 7, 7]
    assert pool.batches == [2] and v.counts == {"recorded": 1, "solved": 2}
    assert v.positions([_state([2])]) == [7] and pool.batches == [2]


def test_move_values_read_finished_games_from_the_result_and_the_rest_as_minus_the_child_s_value():
    pool = _Pool()
    v = ExactValues(C4, pool, [])
    s = _state([0, 1, 0, 1, 0, 1])
    values = v.move_values([s])[0]
    assert values[0] == 1 and all(values[a] == -7 for a in range(1, 7))
    assert pool.batches == [6]


def test_a_partially_recorded_position_is_completed_by_solving():
    pool = _Pool()
    v = ExactValues(C4, pool, [_row([], {3: 1})])
    values = v.move_values([_state([])])[0]
    assert values[3] == -(-1) and all(values[a] == -7 for a in range(7) if a != 3)
    assert pool.batches == [6]
