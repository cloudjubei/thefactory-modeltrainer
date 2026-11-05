"""Direct tests for harness/floor_j1.py — J1, the whole game assembled: the ply-6 opening, and at each of its frontier
positions either the base game's piece (the deeper opening's move, the trivial positions below it and the base
strategies there, each deeper position counted once however many shallow positions reach it) or a strategy built from
the shallow position itself, every library map some piece plays paid once."""
from __future__ import annotations

from harness.floor_j1 import j1_bits, j1_choose, j1_piece, j1_valid

OPENING = {"moves": 2, "trivial": 3}
BELOW = {"p": ["a", "b", "t"], "q": ["b", "c"], "r": ["u"]}
BASE = {"a": (10, frozenset({1})), "b": (20, frozenset({2})), "c": (30, frozenset({1, 3}))}
COST = {1: 100, 2: 200, 3: 300, 4: 400}


def _bits(built=None, base=BASE, below=BELOW):
    return j1_bits(OPENING, below, base, built or {}, COST)


def test_the_base_game_is_its_opening_moves_trivial_positions_strategies_and_maps_once():
    r = _bits()
    assert r["bits"] == 2 * 4 + 3 + 3 * 4 + 2 + (10 + 20 + 30) + (100 + 200 + 300)
    assert (r["kept"], r["trivial"], r["strategies"], r["library_maps"]) == (3, 2, 3, 3)


def test_a_built_piece_replaces_the_move_the_trivial_positions_and_strategies_only_it_reached():
    r = _bits({"p": (7, frozenset({4}))})
    assert r["bits"] == 2 * 4 + 3 + 2 * 4 + 1 + (20 + 30) + 7 + (100 + 200 + 300 + 400)
    r = _bits({"q": (7, frozenset())})
    assert r["bits"] == 2 * 4 + 3 + 2 * 4 + 2 + (10 + 20) + 7 + (100 + 200)


def test_a_library_map_no_kept_piece_plays_any_more_is_no_longer_paid():
    r = _bits({"p": (7, frozenset()), "q": (5, frozenset())})
    assert r["bits"] == 2 * 4 + 3 + 4 + 1 + 0 + 12 and r["library_maps"] == 0


def test_the_chooser_takes_a_build_only_where_the_whole_game_shrinks():
    candidates = {"p": [("p2", (500, frozenset()))], "q": [("q1", (7, frozenset())), ("q5", (6, frozenset({3})))]}
    chosen, bits = j1_choose(OPENING, BELOW, BASE, candidates, COST)
    assert chosen == {"q": "q1"} and bits == _bits({"q": (7, frozenset())})["bits"]


def test_the_chooser_counts_what_two_builds_save_together():
    base = {"a": (10, frozenset()), "b": (2000, frozenset())}
    below = {"p": ["a", "b"], "q": ["b"]}
    candidates = {"p": [("p", (1015, frozenset()))], "q": [("q", (1000, frozenset()))]}
    chosen, bits = j1_choose(OPENING, below, base, candidates, COST)
    assert chosen == {"p": "p", "q": "q"} and bits == _bits({"p": (1015, frozenset()), "q": (1000, frozenset())},
                                                            base, below)["bits"]


def test_the_chooser_keeps_swapping_while_the_game_shrinks():
    candidates = {"q": [("q1", (7, frozenset()))], "r": [("r1", (1, frozenset()))]}
    chosen, bits = j1_choose(OPENING, BELOW, BASE, candidates, COST)
    assert chosen == {"q": "q1", "r": "r1"}
    assert bits == _bits({"q": (7, frozenset()), "r": (1, frozenset())})["bits"]


def test_with_nothing_better_the_chooser_keeps_the_base_game():
    chosen, bits = j1_choose(OPENING, BELOW, BASE, {"p": [("p", (10_000, frozenset()))]}, COST)
    assert chosen == {} and bits == _bits()["bits"]


def _row(complete=True, checked=True, leaves=(1, 5), library=({"3": 0}, {"5": 1})):
    board = [0] * 42
    nodes = [[board, 0, {"move": 3}]] + [[[i + 1] + [0] * 41, 0, {"leaf": leaf}] for i, leaf in enumerate(leaves)]
    return {"complete": complete, "checked": checked, "moves_win": True, "leaves_certified": True, "board": board,
            "to_move": 0, "nodes": nodes, "maps": [{}, *library, {"9": 0}], "map_empty": [0, 42, 42, 40],
            "shared": len(library), "bits": {"nodes": 50, "leaves": len(leaves)}}


LIBRARY = [({"3": 0}, 42), ({"5": 1}, 42)]


def test_a_piece_is_its_strategy_bits_and_the_library_maps_its_leaves_play(monkeypatch):
    import harness.floor_j1 as floor_j1
    monkeypatch.setattr(floor_j1, "_coded", lambda r: r["bits"]["nodes"] + 1)
    assert j1_piece(_row(leaves=(1, 3, 2, 0)), LIBRARY) == (51, frozenset({1, 2}))


def test_only_finished_checked_pieces_built_from_the_library_are_valid():
    assert j1_valid(_row(), LIBRARY)
    assert not j1_valid(_row(complete=False), LIBRARY)
    assert not j1_valid(_row(checked=False), LIBRARY)
    assert not j1_valid({**_row(), "moves_win": False}, LIBRARY)
    assert not j1_valid({**_row(), "leaves_certified": None}, LIBRARY)
    assert not j1_valid(_row(library=({"5": 1}, {"3": 0})), LIBRARY)
    assert not j1_valid({**_row(), "map_empty": [0, 41, 42, 40]}, LIBRARY)
    assert not j1_valid({**_row(), "shared": 1}, LIBRARY)
