"""Direct tests for harness/floor_e1.py — E1, earlier leaves: the 8 ply-6 positions whose subtrees G3 paid most for,
rebuilt from ply 6 with leaves allowed there, judged on completion out of 8 and on their bits against G3's for the same
subtrees (the opening move, trivial positions, and the strategies below)."""
from __future__ import annotations

from games.connect4 import C4State, Connect4
from harness.floor_e1 import SPEC, e1_control, e1_report, e1_size
from harness.floor_g1 import _coded
from harness.floor_g3 import SPEC as G3
from harness.floor_w6 import w6_report
from harness.reference_coding import code_bits
from harness.steady_state import map_bits

GAME = Connect4()
LIB = [{"3": 0}, {"3": 0, "10": 1}, {"5": 1}]
P = GAME.step(GAME.step(GAME.initial_state(), 3), 3)
Q = GAME.step(GAME.step(GAME.initial_state(), 0), 1)
MOVES = {GAME.state_key(P): 1, GAME.state_key(Q): 3}
LIBRARY = [(m, 42) for m in LIB]


def _children(s, move):
    child = GAME.step(s, move)
    return [c for c in (GAME.step(child, a) for a in GAME.legal_actions(child)) if not GAME.is_terminal(c)]


def _root(state, nodes, leaf, maps, shared, complete=True, index=0, empty=42):
    return {"index": index, "complete": complete, "bits": {"nodes": nodes, "leaves": 1}, "board": list(state.board),
            "to_move": state.to_move, "nodes": [[list(state.board), state.to_move, {"leaf": leaf}]],
            "maps": [{}, *maps], "map_empty": [0] + [empty] * len(maps), "shared": shared,
            "found_maps": [[m, 42] for m in maps[shared:]], "build_seconds": 600.0,
            "checked": True if complete else None, "moves_win": True if complete else None,
            "leaves_certified": True if complete else None}


def _g3(heavy=3, nodes=50):
    """G3's frontier under P and Q: the first `heavy` children of each a strategy of 50 bits on library map 1, the
    rest trivial — and one other frontier position, under neither."""
    roots = [_root(c, nodes, 1, LIB, 3, index=i) for s in (P, Q)
             for i, c in enumerate(_children(s, MOVES[GAME.state_key(s)])[:heavy])]
    other = GAME.step(GAME.step(GAME.initial_state(), 6), 6)
    roots.append(_root(other, 70, 3, LIB, 3, index=99))
    return {"roots": roots}


def _per_subtree(heavy=3, nodes=50):
    return [4 + len(_children(s, MOVES[GAME.state_key(s)])) - heavy + heavy * _coded(_root(P, nodes, 1, LIB, 3))
            for s in (P, Q)]


def _e(roots, **over):
    return {"measurement_fingerprint": SPEC["measurement_fp"], "positions_fingerprint": SPEC["positions_fp"],
            "builder": SPEC["builder"], "opening": {"moves": 56, "trivial": 110, "frontier": 148}, "roots": roots,
            **over}


def _arm(sizes=(30, 30), leaves=(1, 1), incomplete=(), library=LIB, empty=42):
    return _e([_root(s, n, leaf, library, len(library), complete=k not in incomplete, index=k, empty=empty)
               for k, (s, n, leaf) in enumerate(zip((P, Q), sizes, leaves))])


JUDGED = {**SPEC, "roots": 2, "g3_subtree_bits": sum(_per_subtree())}


def test_e1_rebuilds_g3_s_heaviest_ply_6_subtrees_from_ply_6():
    assert SPEC["indices"] == [18, 39, 69, 79, 115, 130, 145, 146] and SPEC["roots"] == 8
    assert (SPEC["ply"], SPEC["frontier"], SPEC["source"]) == (6, 148, "opening")
    assert (SPEC["complete_support"], SPEC["complete_refute"], SPEC["support"]) == (7, 5, (4, 5))
    assert SPEC["g3_subtree_bits"] == 47_038 and SPEC["library_from"] == G3["library_from"]
    assert SPEC["builder"] == {**G3["builder"], "seconds": 21_600.0, "budgets": [[0, 1200.0], [2, 1200.0], [4, 30.0]]}
    assert SPEC["search"] == G3["search"] and SPEC["measurement_fp"] == "77fc4515e75e" and SPEC["seed"] == G3["seed"]
    assert SPEC["positions_fp"] == G3["positions_fp"] and e1_report is w6_report


def test_g3_s_bits_for_a_subtree_are_its_opening_move_its_trivial_positions_and_its_strategies():
    assert e1_control([P, Q], _g3(), MOVES) == _per_subtree()
    assert e1_control([P], _g3(heavy=1), MOVES) == _per_subtree(heavy=1)[:1]


def test_a_reply_that_ends_the_game_is_no_position_below():
    s = GAME.initial_state()
    for a in (6, 0, 6, 1, 5, 2):
        s = GAME.step(s, a)
    assert e1_control([s], {"roots": []}, {GAME.state_key(s): 5}) == [4 + 6]


def test_the_arm_is_its_strategies_by_first_use_and_the_library_maps_g3_never_played():
    r = e1_size(_arm(leaves=(1, 3)), _g3(), LIBRARY, MOVES, JUDGED)
    bits = _coded(_root(P, 30, 1, LIB, 3)) + _coded(_root(Q, 30, 3, LIB, 3))
    assert r == {"verdict": "supported", "bits": bits, "control_bits": sum(_per_subtree()),
                 "ratio": bits / sum(_per_subtree()), "lower_bound": False}
    g3 = _g3()
    g3["roots"][-1]["nodes"][0][2]["leaf"] = 1
    r = e1_size(_arm(leaves=(1, 3)), g3, LIBRARY, MOVES, JUDGED)
    assert r["bits"] == bits + map_bits({5: 1}, 42, 3)


def test_the_verdict_is_judged_against_g3_s_bits_for_the_same_subtrees():
    nodes = 50 + next(k for k in range(5) if sum(_per_subtree(nodes=50 + k)) % 5 == 0)
    control = sum(_per_subtree(nodes=nodes))
    leaf = _coded(_root(P, 0, 1, LIB, 3))
    judged = {**JUDGED, "g3_subtree_bits": control}
    for total, verdict in ((control * 4 // 5, "supported"), (control * 4 // 5 + 1, "inconclusive"),
                           (control - 1, "inconclusive"), (control, "refuted")):
        first = (total - 2 * leaf) // 2
        r = e1_size(_arm(sizes=(first, total - 2 * leaf - first)), _g3(nodes=nodes), LIBRARY, MOVES, judged)
        assert (r["bits"], r["verdict"]) == (total, verdict)


def test_an_unfinished_root_counts_without_its_references_and_can_only_refute():
    r = e1_size(_arm(incomplete=(0,)), _g3(), LIBRARY, MOVES, JUDGED)
    assert r["bits"] == 30 - code_bits("fixed", [1], 4, 3) + _coded(_root(Q, 30, 1, LIB, 3))
    assert r["lower_bound"] and r["verdict"] == "inconclusive"
    assert e1_size(_arm(sizes=(500, 500), incomplete=(0,)), _g3(), LIBRARY, MOVES, JUDGED)["verdict"] == "refuted"


def test_a_build_that_did_not_start_from_g3_s_library_or_its_registration_is_not_judged():
    assert e1_size(_arm(library=LIB[::-1]), _g3(), LIBRARY, MOVES, JUDGED)["verdict"] == "not_run"
    assert e1_size(_arm(empty=41), _g3(), LIBRARY, MOVES, JUDGED)["verdict"] == "not_run"
    arm = _arm()
    arm["roots"][1]["shared"] = 2
    assert e1_size(arm, _g3(), LIBRARY, MOVES, JUDGED)["verdict"] == "not_run"
    arm = _arm()
    assert e1_size({**arm, "builder": G3["builder"]}, _g3(), LIBRARY, MOVES, JUDGED)["verdict"] == "not_run"
    assert e1_size({**arm, "measurement_fingerprint": "x"}, _g3(), LIBRARY, MOVES, JUDGED)["verdict"] == "not_run"
    assert e1_size(arm, _g3(), LIBRARY, MOVES, {**JUDGED, "g3_subtree_bits": sum(_per_subtree()) + 1})[
        "verdict"] == "not_run"


def test_g3_must_have_finished_every_position_below_the_rebuilt_subtrees():
    g3 = _g3()
    g3["roots"][0]["complete"] = False
    states = [P, Q]
    assert e1_size(_arm(), g3, LIBRARY, MOVES, {**JUDGED, "g3_subtree_bits": sum(e1_control(states, g3, MOVES))})[
        "verdict"] == "not_run"
