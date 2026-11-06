"""Direct tests for harness/floor_t1.py — T1, the whole game as one file: the opening's moves and trivial positions and
the pieces the game takes, read as one strategy over positions, each position taking the node of the piece that reaches
it first; then judged by the file's own length and by what a decoder reads back and plays."""
from __future__ import annotations

import pytest

from games.tictactoe import TicTacToe, TTTState
from harness.floor_t1 import (SPEC, mirror_cells, t1_differences, t1_pieces, t1_report, t1_rule, t1_source)
from harness.steady_state import Facts

GAME = TicTacToe()
ROOT = GAME.initial_state()


def _ttt(x, o):
    board = [0] * 9
    for i in x:
        board[i] = 1
    for i in o:
        board[i] = 2
    return TTTState(board=tuple(board), to_move=0 if len(x) == len(o) else 1, winner=None, done=False)


def _row(s, nodes, maps=({},), shared=0):
    return {"board": list(s.board), "to_move": s.to_move, "shared": shared,
            "maps": [{str(c): k for c, k in m.items()} for m in maps],
            "nodes": [[list(k[0]), k[1], n] for k, n in nodes.items()]}


def _key(s):
    return GAME.state_key(s)


def _after(s, move):
    a = GAME.step(s, move)
    return [GAME.step(a, b) for b in GAME.legal_actions(a)]


A = _ttt((4,), (0,))
B = _ttt((4,), (1,))


def _opening(pieces_at=()):
    """4 at the root, then every reply a piece's root or trivial."""
    return {"moves": {_key(ROOT): 4}, "trivial": {_key(c) for c in _after(ROOT, 4) if c not in pieces_at}}


def test_the_opening_alone_is_moves_and_empty_leaves():
    source = t1_source(GAME, ROOT, _opening(), {})
    assert source["maps"] == [{}] and source["nodes"][_key(ROOT)] == {"move": 4}
    assert all(source["nodes"][_key(c)] == {"leaf": 0} for c in _after(ROOT, 4))


def test_a_piece_takes_over_at_its_root():
    row = _row(A, {_key(A): {"leaf": 1}}, [{}, {8: 2}], shared=1)
    source = t1_source(GAME, ROOT, _opening([A]), {_key(A): (row, False)})
    assert source["nodes"][_key(A)] == {"leaf": 1} and source["maps"] == [{}, {8: 2}]


def test_a_piece_s_moves_are_followed_inside_the_piece():
    after = _after(A, 8)
    nodes = {_key(A): {"move": 8}, **{_key(c): {"leaf": 0} for c in after}}
    source = t1_source(GAME, ROOT, _opening([A]), {_key(A): (_row(A, nodes), False)})
    assert all(_key(c) in source["nodes"] for c in after)


def test_a_position_the_piece_leaves_without_a_node_raises():
    nodes = {_key(A): {"move": 8}}
    with pytest.raises(KeyError):
        t1_source(GAME, ROOT, _opening([A]), {_key(A): (_row(A, nodes), False)})


def test_a_position_neither_opening_nor_piece_covers_raises():
    with pytest.raises(KeyError):
        t1_source(GAME, ROOT, {"moves": {_key(ROOT): 4}, "trivial": set()}, {})


def test_a_position_two_pieces_reach_takes_the_first_piece_s_node():
    shared = _ttt((4, 8), (0, 1))
    from_a = [c for c in _after(A, 8)]
    from_b = [c for c in _after(B, 8)]
    assert shared in from_a and shared in from_b
    a = {_key(A): {"move": 8}, **{_key(c): {"leaf": 1} for c in from_a}}
    b = {_key(B): {"move": 8}, **{_key(c): {"leaf": 2} for c in from_b}}
    source = t1_source(GAME, ROOT, _opening([A, B]), {
        _key(A): (_row(A, a, [{}, {2: 1}], shared=1), False),
        _key(B): (_row(B, b, [{}, {3: 1}, {5: 1}], shared=1), False)})
    first = source["maps"][source["nodes"][_key(shared)]["leaf"]]
    assert first in ({2: 1}, {5: 1}) and source["nodes"][_key(shared)]["leaf"] != 0
    assert len(source["maps"]) == 3


def test_equal_maps_from_different_pieces_are_one_map():
    source = t1_source(GAME, ROOT, _opening([A, B]), {
        _key(A): (_row(A, {_key(A): {"leaf": 1}}, [{}, {8: 2, 2: 1}], shared=1), False),
        _key(B): (_row(B, {_key(B): {"leaf": 2}}, [{}, {5: 1}, {2: 1, 8: 2}], shared=2), False)})
    assert source["nodes"][_key(A)]["leaf"] == source["nodes"][_key(B)]["leaf"] and len(source["maps"]) == 2


def test_only_a_mirror_run_s_first_half_of_the_library_reads_mirrored():
    maps = [{}, {8: 1}, {8: 1}, {5: 1}]
    for leaf, mirrored in ((1, True), (2, False), (3, False)):
        row = _row(A, {_key(A): {"leaf": leaf}}, maps, shared=2)
        node = t1_source(GAME, ROOT, _opening([A]), {_key(A): (row, True)})["nodes"][_key(A)]
        assert bool(node.get("mirrored")) is mirrored
    row = _row(A, {_key(A): {"leaf": 1}}, maps, shared=2)
    assert "mirrored" not in t1_source(GAME, ROOT, _opening([A]), {_key(A): (row, False)})["nodes"][_key(A)]


def test_exceptions_are_read_by_position():
    board = list(_ttt((4, 8), (0, 1)).board)
    row = _row(A, {_key(A): {"leaf": 0, "exceptions": [[board, 0, 2]], "own": 9, "coded": 7}})
    node = t1_source(GAME, ROOT, _opening([A]), {_key(A): (row, False)})["nodes"][_key(A)]
    assert node == {"leaf": 0, "exceptions": {(tuple(board), 0): 2}}


def test_the_pieces_are_the_ones_the_game_plays():
    rows = {"g3": {0: {"board": list(A.board), "to_move": 0}, 1: {"board": list(B.board), "to_move": 0}}}
    pieces = t1_pieces(lambda: [("g3", 0, rows["g3"][0])], {"g3": False})
    assert pieces == {_key(A): (rows["g3"][0], False)}


def test_each_piece_reads_by_its_run_s_mirror():
    row = {"board": list(A.board), "to_move": 0}
    assert t1_pieces(lambda: [("x3", 7, row)], {"x3": True})[_key(A)][1] is True


def test_two_pieces_at_one_position_are_refused():
    row = {"board": list(A.board), "to_move": 0}
    with pytest.raises(ValueError):
        t1_pieces(lambda: [("g3", 0, row), ("g4", 0, row)], {"g3": False, "g4": False})


def test_the_rule_must_be_one_rule():
    b = {"ties": "lowest", "unlevelled": "last", "n_levels": 8}
    assert t1_rule([b, {**b, "mirror": True, "seconds": 3}]) == {
        "ties": "lowest", "unlevelled": "last", "safe_depth": 1, "forcing": False, "order": "lowest", "n_levels": 8}
    for other in ({**b, "ties": "exact"}, {**b, "safe_depth": 2}, {**b, "forcing": True}, {**b, "n_levels": 4},
                  {**b, "unlevelled": "none"}, {**b, "order": "tactics"}):
        with pytest.raises(ValueError):
            t1_rule([b, other])


def test_the_mirror_flips_columns():
    assert mirror_cells(3, 3) == [2, 1, 0, 5, 4, 3, 8, 7, 6]
    assert mirror_cells(6, 7)[0] == 6 and mirror_cells(6, 7)[3] == 3 and mirror_cells(6, 7)[41] == 35


def test_differences_compare_content_not_map_numbers():
    a = {"nodes": {"p": {"leaf": 1}, "q": {"move": 2}}, "maps": [{}, {4: 1}]}
    b = {"nodes": {"p": {"leaf": 2}, "q": {"move": 2}}, "maps": [{}, {9: 9}, {4: 1}]}
    assert t1_differences(a, b) == 0


def test_differences_count_every_kind_of_change():
    a = {"nodes": {"p": {"leaf": 1}, "q": {"move": 2}}, "maps": [{}, {4: 1}]}
    assert t1_differences(a, {"nodes": {"p": {"leaf": 1}, "q": {"move": 3}}, "maps": [{}, {4: 1}]}) == 1
    assert t1_differences(a, {"nodes": {"p": {"leaf": 1}, "q": {"move": 2}}, "maps": [{}, {4: 2}]}) == 1
    assert t1_differences(a, {"nodes": {"p": {"leaf": 1, "mirrored": True}, "q": {"move": 2}},
                              "maps": [{}, {4: 1}]}) == 1
    assert t1_differences(a, {"nodes": {"p": {"leaf": 1, "exceptions": {"r": 1}}, "q": {"move": 2}},
                              "maps": [{}, {4: 1}]}) == 1
    assert t1_differences(a, {"nodes": {"p": {"leaf": 1}}, "maps": [{}, {4: 1}]}) == 1
    assert t1_differences({"nodes": {"p": {"leaf": 1}}, "maps": [{}, {4: 1}]}, a) == 1


CONTROL = 1_000
JUDGED = {**SPEC, "control_bits": CONTROL, "support": (99, 100), "refute": (101, 100)}


def _evidence(bits=900, won=True, leaves=7):
    return {"spec": "harness.floor_t1", "bits": bits, "file": "x", "sha256": "y", "differences": 0,
            "replay": {"won": won, "leaves": leaves, "reason": None}}


def _report(e=None, read=None, spec=JUDGED):
    e = e or _evidence()
    return t1_report(e, read or {"bits": e["bits"], "differences": 0, "leaves": 7}, spec)


def test_a_file_that_reads_back_and_wins_every_line_is_decoded():
    assert _report()["decoded"] == "supported"


def test_a_file_that_reads_back_differently_refutes():
    assert _report(read={"bits": 900, "differences": 1, "leaves": 7})["decoded"] == "refuted"


def test_a_lost_replay_refutes():
    assert _report(_evidence(won=False))["decoded"] == "refuted"


def test_a_replay_of_other_leaves_refutes():
    assert _report(_evidence(leaves=6))["decoded"] == "refuted"


def test_a_file_of_another_length_refutes():
    assert _report(read={"bits": 901, "differences": 0, "leaves": 7})["decoded"] == "refuted"


def test_the_size_bars_are_inclusive():
    assert _report(_evidence(bits=990))["size"] == "supported"
    assert _report(_evidence(bits=991))["size"] == "inconclusive"
    assert _report(_evidence(bits=1_009))["size"] == "inconclusive"
    assert _report(_evidence(bits=1_010))["size"] == "refuted"


def test_a_file_that_fails_to_decode_has_no_size():
    assert _report(_evidence(won=False, bits=10))["size"] == "not_run"


def test_a_run_off_its_spec_is_not_run():
    assert _report({**_evidence(), "spec": "other"}) == {"decoded": "not_run", "size": "not_run"}


def test_the_rule_runs_with_the_facts_it_names():
    assert SPEC["rule"] == {"ties": "lowest", "unlevelled": "last", "safe_depth": 1, "forcing": False,
                            "order": "lowest", "n_levels": 8}
    Facts(GAME, **{k: v for k, v in SPEC["rule"].items() if k != "n_levels"})
