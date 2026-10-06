"""Direct tests for harness/floor_s1.py — S1, discovered steady states at the exact table's frontier. Coverage is the
share of sampled non-trivial positions (where win/block/forced moves alone do not already win) that get a verified
steady state within budget; compression is 3 bits per first-player position the state replaces over the map's bits.
Every found map must pass both independent checks (the solver-free walk and harness.certify)."""
from __future__ import annotations

import pytest

from harness.floor_s1 import SPEC, next_frontier, pilot_indices, s1_report, sample_indices

TEST_SPEC = {"measurement_fp": "m" * 12, "positions_fp": "p" * 12, "plies": [10, 12], "per_ply": 10, "seed": 1,
             "n_levels": 8, "level_bits": 3, "budget": {"seconds": 60}, "judged_ply": 12, "coverage_support": 0.30,
             "coverage_refute": 0.10, "compression_support": 10, "compression_refute": 1, "min_found": 3}


def _row(ply, status, own=900, bits=60, trivial=False, verified=True, certified=True):
    found = status == "found"
    return {"ply": ply, "trivial": trivial, "status": "trivial" if trivial else status,
            "bits": bits if found else None, "own_positions": own if found else None,
            "verified": verified if found else None, "certified": certified if found else None, "seconds": 5.0}


def _readout(rows12=None, rows10=None):
    rows12 = rows12 if rows12 is not None else ([_row(12, "found")] * 4 + [_row(12, "impossible")] * 3
                                                + [_row(12, "budget")] * 2 + [_row(12, "found", trivial=True)])
    rows10 = rows10 if rows10 is not None else [_row(10, "found")] * 2 + [_row(10, "budget")] * 8
    return {"measurement_fingerprint": "m" * 12, "positions_fingerprint": "p" * 12,
            "config": {k: TEST_SPEC[k] for k in ("n_levels", "level_bits", "budget")}, "results": rows10 + rows12}


def test_coverage_counts_found_over_non_trivial_positions():
    r = s1_report(_readout(), TEST_SPEC)
    assert r["integrity"] == []
    assert r["plies"]["12"] == {"sampled": 10, "trivial": 1, "found": 4, "impossible": 3, "budget": 2,
                                "coverage": pytest.approx(4 / 9), "median_bits": 60, "median_own_positions": 900}
    assert r["plies"]["10"]["coverage"] == pytest.approx(0.2)
    assert r["coverage"]["verdict"] == "supported"


@pytest.mark.parametrize("found,verdict", [(3, "supported"), (2, "inconclusive"), (1, "inconclusive"),
                                           (0, "refuted")])
def test_the_coverage_bars_are_registered(found, verdict):
    rows = [_row(12, "found")] * found + [_row(12, "impossible")] * (10 - found)
    assert s1_report(_readout(rows12=rows), TEST_SPEC)["coverage"]["verdict"] == verdict


def test_compression_is_three_bits_per_replaced_position_over_the_map_bits():
    rows = [_row(12, "found", own=o, bits=b) for o, b in ((900, 60), (100, 60), (2000, 100))] \
        + [_row(12, "impossible")] * 7
    r = s1_report(_readout(rows12=rows), TEST_SPEC)["compression"]
    assert r["values"] == [pytest.approx(45.0), pytest.approx(5.0), pytest.approx(60.0)]
    assert r["median"] == pytest.approx(45.0) and r["verdict"] == "supported"


@pytest.mark.parametrize("own,verdict", [(150, "inconclusive"), (10, "refuted"), (200, "supported"),
                                         (20, "inconclusive")])
def test_the_compression_bars_are_registered(own, verdict):
    rows = [_row(12, "found", own=own, bits=60)] * 3 + [_row(12, "impossible")] * 7
    assert s1_report(_readout(rows12=rows), TEST_SPEC)["compression"]["verdict"] == verdict


def test_too_few_found_states_leave_compression_inconclusive():
    rows = [_row(12, "found", own=5000)] * 2 + [_row(12, "impossible")] * 8
    assert s1_report(_readout(rows12=rows), TEST_SPEC)["compression"]["verdict"] == "inconclusive"


@pytest.mark.parametrize("breakage", ["fingerprint", "positions", "config", "short sample", "unverified",
                                      "uncertified"])
def test_a_readout_that_is_not_the_registered_measurement_is_not_run(breakage):
    e = _readout()
    if breakage == "fingerprint":
        e["measurement_fingerprint"] = "0" * 12
    elif breakage == "positions":
        e["positions_fingerprint"] = "0" * 12
    elif breakage == "config":
        e["config"]["n_levels"] = 4
    elif breakage == "short sample":
        e["results"] = e["results"][:-1]
    elif breakage == "unverified":
        e["results"][-2] = _row(12, "found", verified=False)
    else:
        e["results"][-2] = _row(12, "found", certified=False)
    r = s1_report(e, TEST_SPEC)
    assert r["integrity"] and r["coverage"] == {"verdict": "not_run"} and r["compression"] == {"verdict": "not_run"}


def test_the_sample_is_fixed_by_the_seed_and_the_pilot_never_touches_it():
    sample = sample_indices(1000, 100, 1)
    assert sample == sample_indices(1000, 100, 1) and len(set(sample)) == 100 and sample != sample_indices(1000, 100, 2)
    pilot = pilot_indices(1000, 100, 1, 10)
    assert len(set(pilot)) == 10 and not set(pilot) & set(sample)


def test_the_registered_study_samples_100_positions_at_plies_12_and_14():
    assert (SPEC["plies"], SPEC["per_ply"], SPEC["seed"], SPEC["judged_ply"]) == ([12, 14], 100, 1, 14)
    assert (SPEC["n_levels"], SPEC["level_bits"]) == (8, 3)
    assert SPEC["budget"] == {"max_constraints": 60_000, "conflicts": 1_000_000, "seconds": 300.0, "cap": 1_000_000,
                              "lines": 64}
    assert (SPEC["coverage_support"], SPEC["coverage_refute"]) == (0.30, 0.10)
    assert (SPEC["compression_support"], SPEC["compression_refute"], SPEC["min_found"]) == (10, 1, 5)


def test_the_next_frontier_is_the_table_s_move_then_every_reply_each_position_once():
    from games.connect4 import C4State, Connect4

    game = Connect4()
    empty = game.initial_state()
    rows = [{"board": list(empty.board), "to_move": 0, "values": {"0": 0, "3": 1, "4": 1, "6": -1}},
            {"board": list(game.step(game.step(empty, 3), 3).board), "to_move": 0,
             "values": {"2": 1, "3": 1}}]
    nxt = next_frontier(game, rows)
    first = game.step(empty, 3)
    second = game.step(game.step(game.step(empty, 3), 3), 2)
    expected = [game.step(first, b) for b in game.legal_actions(first)]
    expected += [c for c in (game.step(second, b) for b in game.legal_actions(second))
                 if game.state_key(c) not in {game.state_key(e) for e in expected}]
    assert [(tuple(r["board"]), r["to_move"]) for r in nxt] == [(s.board, s.to_move) for s in expected]
    assert all(isinstance(r["board"], list) for r in nxt) and C4State
    assert next_frontier(game, rows + [rows[0]]) == nxt
