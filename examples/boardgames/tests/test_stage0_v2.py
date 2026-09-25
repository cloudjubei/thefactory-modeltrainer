"""Direct tests for harness/stage0_v2.py — the Stage 0 re-run after v1 read NOT_RUN. Same rule as v1; the trained
reference is every position the net trained on (with a final-buffer flag), the decision cells need 60 positions,
and the noisy operating-characteristic test uses v1's REALISED cell sizes, the ones the calibration claim is about."""
from __future__ import annotations

import random

import pytest

from harness.stage0 import C0_ERA, C0_PLIES, C0_RECIPE
from harness.stage0_v2 import (C0V2_MEASUREMENT_FP, C0V2_SAMPLING, C0V2_SEEDS, net_gaps, stage0_v2_report)

N = 500


def _cell(cls, ply, raw_ok_rate, label_ok_rate, n=N, in_final=None):
    k_raw, k_lab = round(raw_ok_rate * n), round(label_ok_rate * n)
    return [{"cls": cls, "ply": ply, "key": i, "raw_ok": i < k_raw, "label_ok": i < k_lab, "avg_ok": i < k_raw,
             **({} if in_final is None else {"in_final": in_final(i)})} for i in range(n)]


def _net(seed, err_trained=0.1, err_sib=0.3, label_sib=0.9, per_ply=None, **over):
    positions = []
    for ply in C0_PLIES:
        et, es, ls = (per_ply or {}).get(ply, (err_trained, err_sib, label_sib))
        positions += _cell("visited", ply, 1 - et, 0.95, in_final=lambda i: i % 2 == 0)
        positions += _cell("sibling", ply, 1 - es, ls)
        positions += _cell("two_moves_off", ply, 0.6, 0.8, n=30)
    history = [{"iteration": i + 1, "selfplay_states": 800 + i, "state_buffer": 8000} for i in range(14)]
    row = {"seed": seed, "params": 60555, "history": history, "recorded_states": sum(h["selfplay_states"] for h in history),
           "games_per_pass": [48] * 14, "solver_forbidden": True, "positions": positions, "probe": [0.4, 0.5],
           "sizes": {"final_buffer": 8000}, "containment": {}, "conversion": []}
    row.update(over)
    return row


def _config(**over):
    return {**C0_RECIPE, "sampling": dict(C0V2_SAMPLING), "plies": list(C0_PLIES), "design": "v2", **over}


def _evidence(nets=None, **top):
    nets = nets if nets is not None else [_net(s) for s in C0V2_SEEDS]
    return {"training_fingerprint": C0_ERA, "measurement_fingerprint": C0V2_MEASUREMENT_FP, "config": _config(),
            "seeds": nets, **top}


def _spread(gaps_e, gaps_l=None, base_err=0.1):
    gaps_l = gaps_l or [0.2] * 5
    return _evidence([_net(s, err_trained=base_err, err_sib=base_err + e, label_sib=(1 - base_err - e) + lab)
                      for s, e, lab in zip(C0V2_SEEDS, gaps_e, gaps_l)])


def test_a_consistent_gap_and_headroom_is_GO_and_says_what_it_is_not():
    r = stage0_v2_report(_spread([0.2, 0.18, 0.22, 0.2, 0.2], [0.2, 0.16, 0.24, 0.2, 0.18]))
    assert r["integrity"] == [] and r["C0"]["verdict"] == "go"
    assert "not evidence that siblings raise conversion" in r["C0"]["reading"]


@pytest.mark.parametrize("gaps_e,gaps_l,base,binding", [
    ([0.0, 0.01, -0.01, 0.0, 0.0], None, 0.1, "exposure gap"),
    ([0.3] * 5, [0.0, 0.01, -0.01, 0.0, 0.0], 0.3, "label headroom"),
])
def test_either_gap_without_headroom_is_STOP_naming_the_binding_one(gaps_e, gaps_l, base, binding):
    r = stage0_v2_report(_spread(gaps_e, gaps_l, base_err=base))
    assert r["C0"]["verdict"] == "stop" and f"binding: the {binding}" in r["C0"]["reading"]
    assert "untested" in r["C0"]["reading"]


@pytest.mark.parametrize("gaps_e,gaps_l,base", [
    ([0.4, -0.1, 0.34, -0.06, 0.1], None, 0.1),
    ([0.0, 0.03, -0.01, 0.04, 0.01], None, 0.1),
    ([0.3] * 5, [0.06, -0.02, 0.05, -0.01, 0.04], 0.3),
])
def test_bounds_that_straddle_are_UNDECIDED(gaps_e, gaps_l, base):
    assert stage0_v2_report(_spread(gaps_e, gaps_l, base_err=base))["C0"]["verdict"] == "undecided"


def test_a_real_but_tiny_gap_below_the_smallest_effect_worth_a_stage_1_is_STOP_not_GO():
    r = stage0_v2_report(_spread([0.012, 0.014, 0.012, 0.014, 0.012]))
    assert r["C0"]["E"]["lower"] > 0 and r["C0"]["verdict"] == "stop"


def test_the_tic_tac_toe_signature_reads_GO():
    r = stage0_v2_report(_spread([0.06, 0.02, 0.04, 0.04, 0.04], [0.3] * 5, base_err=0.0))
    assert r["C0"]["E"]["mean"] == pytest.approx(0.04) and r["C0"]["verdict"] == "go"


def test_the_trained_reference_is_the_VISITED_cell_read_through_the_raw_policy():
    row = _net(206, err_trained=0.1, err_sib=0.3, label_sib=0.9)
    for p in row["positions"]:
        p["avg_ok"] = not p["raw_ok"]
    g = net_gaps(row)
    assert g["E"] == pytest.approx(0.2) and g["L"] == pytest.approx(0.2)
    assert g["per_ply"][str(C0_PLIES[0])]["err_visited"] == pytest.approx(0.1)


def test_plies_are_averaged_with_equal_weight():
    rates = {ply: (0.10 + 0.02 * i, 0.10 + 0.02 * i, 0.9) for i, ply in enumerate(C0_PLIES)}
    row = _net(206, per_ply=rates)
    row["positions"] = [p for p in row["positions"]
                        if not (p["cls"] == "visited" and p["ply"] in C0_PLIES[-2:] and p["key"] % 5)]
    assert net_gaps(row)["E"] == pytest.approx(0.0, abs=1e-9)


def test_each_mover_is_reported():
    rates = {p: ((0.1, 0.3, 0.9) if p % 2 == 0 else (0.1, 0.1, 0.9)) for p in C0_PLIES}
    by = stage0_v2_report(_evidence([_net(s, per_ply=rates) for s in C0V2_SEEDS]))["descriptives"]["by_mover"]
    assert by["first"]["E"]["mean"] == pytest.approx(0.2) and by["second"]["E"]["mean"] == pytest.approx(0.0)


def test_final_buffer_membership_is_a_descriptive_split_of_the_visited_cell():
    cells = stage0_v2_report(_evidence())["descriptives"]["cells"]
    assert cells["visited_in_final_buffer"]["n"] == cells["visited_evicted"]["n"] == 5 * len(C0_PLIES) * N // 2


V1_VISITED = [185, 154, 118, 136, 104, 89, 133, 140, 100, 77, 70, 76, 125, 124, 108, 91, 70, 74,
              171, 156, 128, 103, 87, 84, 234, 230, 194, 146, 131, 150]
V1_SIBLING = [300, 300, 300, 300, 255, 234, 300, 300, 300, 300, 204, 187, 300, 300, 300, 300, 217, 169,
              300, 300, 300, 300, 300, 290, 300, 300, 300, 300, 300, 300]


def _noisy(rng, e_mean, e_sd=0.019, l_gap=0.2, base=0.45):
    nets = []
    for n, seed in enumerate(C0V2_SEEDS):
        raw_err = min(max(base + rng.gauss(e_mean, e_sd), 0.0), 1.0)
        positions = []
        for j, ply in enumerate(C0_PLIES):
            nv, ns = V1_VISITED[n * 6 + j], min(C0V2_SAMPLING["sibling"], V1_SIBLING[n * 6 + j])
            positions += [{"cls": "visited", "ply": ply, "raw_ok": rng.random() >= base, "label_ok": True,
                           "avg_ok": True} for _ in range(nv)]
            positions += [{"cls": "sibling", "ply": ply, "raw_ok": rng.random() >= raw_err,
                           "label_ok": rng.random() < min(1.0, 1 - raw_err + l_gap), "avg_ok": True} for _ in range(ns)]
        nets.append(_net(seed, positions=positions))
    return _evidence(nets)


def test_the_operating_characteristic_on_noisy_evidence_at_v1_s_realised_cell_sizes():
    rng = random.Random(1)
    signature = [stage0_v2_report(_noisy(rng, 0.042))["C0"]["verdict"] for _ in range(60)]
    zero = [stage0_v2_report(_noisy(rng, 0.0))["C0"]["verdict"] for _ in range(60)]
    assert signature.count("go") >= 40
    assert zero.count("go") <= 6 and zero.count("stop") >= 10


def _broken_net(**over):
    return lambda: _evidence([_net(s, **(over if s == 208 else {})) for s in C0V2_SEEDS])


BREAKAGES = {
    "era": lambda: _evidence(training_fingerprint="3c8faf6a7505"),
    "measurement": lambda: _evidence(measurement_fingerprint="0" * 12),
    "v1 design": lambda: {**_evidence(), "config": _config(design="v1")},
    "v1 sampling": lambda: {**_evidence(), "config": _config(sampling={**C0V2_SAMPLING, "sibling": 300})},
    "labels": lambda: {**_evidence(), "config": _config(reanalyze_sims=200)},
    "one mover": lambda: {**_evidence(), "config": _config(plies=[p for p in C0_PLIES if p % 2 == 0])},
    "v1 seeds": lambda: _evidence([_net(s) for s in (201, 202, 203, 204, 205)]),
    "four nets": lambda: _evidence([_net(s) for s in C0V2_SEEDS[:4]]),
    "history": _broken_net(history=[{"selfplay_states": 800, "state_buffer": 8000}] * 13, recorded_states=800 * 13),
    "recorder missed states": _broken_net(recorded_states=5),
    "recorder missed a game": _broken_net(games_per_pass=[48] * 13 + [47]),
    "solver allowed": _broken_net(solver_forbidden=False),
    "params": _broken_net(params=20616),
    "final buffer rebuilt wrong": _broken_net(sizes={"final_buffer": 7999}),
    "no positions": _broken_net(positions=None),
}


@pytest.mark.parametrize("name", sorted(BREAKAGES))
def test_a_broken_integrity_condition_is_NOT_RUN(name):
    r = stage0_v2_report(BREAKAGES[name]())
    assert r["C0"]["verdict"] == "not_run" and r["integrity"]
    assert not any("could not be read" in p for p in r["integrity"])


@pytest.mark.parametrize("cls,ply", [("visited", C0_PLIES[0]), ("sibling", C0_PLIES[-1])])
def test_a_decision_cell_under_60_is_NOT_RUN_and_60_is_enough(cls, ply):
    for keep, verdict in ((59, "not_run"), (60, "go")):
        ev = _spread([0.2] * 5)
        row = ev["seeds"][2]
        kept = [p for p in row["positions"] if (p["cls"], p["ply"]) == (cls, ply)][:keep]
        row["positions"] = [p for p in row["positions"] if (p["cls"], p["ply"]) != (cls, ply)] + kept
        assert stage0_v2_report(ev)["C0"]["verdict"] == verdict


@pytest.mark.parametrize("junk", [None, {}, {"seeds": "x", "config": {}}, {"seeds": [None], "config": {}}])
def test_evidence_that_cannot_be_read_is_NOT_RUN(junk):
    assert stage0_v2_report(junk)["C0"]["verdict"] == "not_run"


def test_a_malformed_position_that_passes_integrity_is_NOT_RUN_with_the_reason():
    ev = _evidence()
    del ev["seeds"][1]["positions"][0]["raw_ok"]
    r = stage0_v2_report(ev)
    assert r["C0"]["verdict"] == "not_run" and "could not be read" in r["integrity"][0]
