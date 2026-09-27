"""Direct tests for harness/native_solver.py (and harness/c4solver.c) — the C port of harness/solver.py's exact
Connect-4 solve. The Python solver is the reference: the port must give identical values, weak and strong, on
positions of every shape, whatever its transposition table size."""
from __future__ import annotations

import random

import pytest

from harness import native_solver as ns
from harness import solver


def _positions(n, lo, hi, seed):
    from games.connect4 import Connect4

    g = Connect4()
    r = random.Random(seed)
    out = []
    while len(out) < n:
        s = g.initial_state(r)
        for _ in range(r.randrange(lo, hi)):
            if g.is_terminal(s):
                break
            s = g.step(s, r.choice(g.legal_actions(s)), r)
        if not g.is_terminal(s):
            out.append(s)
    return out


@pytest.fixture(scope="module")
def positions():
    return _positions(160, 16, 38, 7)


def _shapes(states):
    from games.connect4 import Connect4

    g = Connect4()
    immediate = sum(1 for s in states if any(g.is_terminal(g.step(s, a, random.Random(0))) for a in g.legal_actions(s)))
    values = {max(solver.move_values(s).values()) for s in states}
    full_column = sum(1 for s in states if len(g.legal_actions(s)) < 7)
    return immediate, values, full_column


@pytest.mark.parametrize("weak", [True, False])
def test_the_port_agrees_with_the_python_solver_on_every_position(positions, weak):
    immediate, values, full_column = _shapes(positions)
    assert immediate > 5 and values == {-1, 0, 1} and full_column > 5
    for s in positions:
        assert ns.move_values(s, weak=weak) == solver.move_values(s, weak=weak)


def test_values_are_exact_whatever_the_table_size_even_when_every_probe_collides(positions):
    try:
        ns.set_table_size(2)
        for s in positions[:60]:
            assert ns.move_values(s, weak=False) == solver.move_values(s, weak=False)
    finally:
        ns.set_table_size(ns.DEFAULT_LOG2_TABLE)


def test_a_reset_table_changes_no_value(positions):
    before = [ns.move_values(s) for s in positions[:40]]
    ns.reset_table()
    assert [ns.move_values(s) for s in positions[:40]] == before


def test_position_value_is_the_best_move_value_and_a_finished_game_has_none(positions):
    from games.connect4 import Connect4

    for s in positions[:30]:
        assert ns.position_value(s) == max(solver.move_values(s).values())
    g = Connect4()
    r = random.Random(0)
    s = g.initial_state(r)
    while not g.is_terminal(s):
        s = g.step(s, r.choice(g.legal_actions(s)), r)
    assert ns.move_values(s) == {} and ns.position_value(s) == 0


def test_the_book_short_circuits_the_children_it_proves(positions):
    class Book:
        def __init__(self):
            self.asked = 0

        def proven_value(self, key):
            self.asked += 1
            return -1

    book = Book()
    s = next(p for p in positions if any(v < 1 for v in solver.move_values(p).values()))
    vals = ns.move_values(s, book=book)
    assert book.asked > 0 and all(v == 1 for v in vals.values())


@pytest.mark.parametrize("log2", [0, 31])
def test_a_table_size_out_of_range_is_refused(log2):
    with pytest.raises(ValueError, match="table size"):
        ns.set_table_size(log2)


def test_a_source_that_does_not_compile_is_refused_with_the_compiler_s_words(tmp_path, monkeypatch):
    bad = tmp_path / "c4solver.c"
    bad.write_text("int c4_init(int x) { return x +; }\n")
    monkeypatch.setattr(ns, "SOURCE", bad)
    monkeypatch.setattr(ns, "BUILD_DIR", tmp_path / "build")
    monkeypatch.setattr(ns, "_LIB", None)
    with pytest.raises(RuntimeError, match="compiling c4solver.c failed"):
        ns.move_values(_positions(1, 20, 21, 0)[0])


def test_the_library_is_named_by_its_source_so_an_edited_source_is_rebuilt(tmp_path, monkeypatch):
    src = tmp_path / "c4solver.c"
    src.write_text(ns.SOURCE.read_text())
    monkeypatch.setattr(ns, "SOURCE", src)
    monkeypatch.setattr(ns, "BUILD_DIR", tmp_path / "build")
    monkeypatch.setattr(ns, "_LIB", None)
    ns.move_values(_positions(1, 20, 21, 0)[0])
    first = sorted(p.name for p in (tmp_path / "build").iterdir())
    src.write_text(src.read_text() + "\n/* edited */\n")
    monkeypatch.setattr(ns, "_LIB", None)
    ns.move_values(_positions(1, 20, 21, 0)[0])
    second = sorted(p.name for p in (tmp_path / "build").iterdir())
    assert len(first) == 1 and len(second) == 2 and first[0] in second


def test_one_solve_of_the_position_equals_the_best_of_its_children(positions):
    for s in positions:
        assert ns.solve_position(s) == max(solver.move_values(s).values())


def test_a_book_answers_a_position_it_proves_without_solving(positions):
    class Book:
        def proven_value(self, key):
            return -1
    s = next(p for p in positions if max(solver.move_values(p).values()) == 1)
    assert ns.solve_position(s, book=Book()) == -1 and ns.solve_position(s) == 1


def test_a_finished_game_has_no_value_to_solve():
    from games.connect4 import Connect4

    g = Connect4()
    r = random.Random(0)
    s = g.initial_state(r)
    while not g.is_terminal(s):
        s = g.step(s, r.choice(g.legal_actions(s)), r)
    with pytest.raises(ValueError, match="finished"):
        ns.solve_position(s)
