"""Direct tests for harness/stage0.py — the pre-registered Connect-4 Stage 0 go/no-go. Synthetic evidence in the
full schema sets each net's error and label rates per position class and ply, so the verdict is known by
construction; the net is the unit, so the fixtures vary the spread BETWEEN nets, and one test runs the rule on
binomially NOISY evidence at the registered sample sizes, where the calibration claim has to hold."""
from __future__ import annotations

import random

import pytest

from harness.measurement import t_critical
from harness.stage0 import (C0_ERA, C0_MEASUREMENT_FP, C0_PLIES, C0_RECIPE, C0_SAMPLING, C0_SEEDS, _across,
                            net_gaps, stage0_report)

N = 500


def _cell(cls, ply, raw_ok_rate, label_ok_rate, n=N):
    k_raw, k_lab = round(raw_ok_rate * n), round(label_ok_rate * n)
    return [{"cls": cls, "ply": ply, "key": i, "raw_ok": i < k_raw, "label_ok": i < k_lab, "avg_ok": i < k_raw}
            for i in range(n)]


def _net(seed, err_final=0.1, err_sib=0.3, label_sib=0.9, per_ply=None, **over):
    positions = []
    for ply in C0_PLIES:
        ef, es, ls = (per_ply or {}).get(ply, (err_final, err_sib, label_sib))
        positions += _cell("final_buffer", ply, 1 - ef, 0.95)
        positions += _cell("sibling", ply, 1 - es, ls)
        positions += _cell("two_moves_off", ply, 0.6, 0.8, n=60)
        positions += _cell("random", ply, 0.5, 0.7, n=60)
    history = [{"iteration": i + 1, "selfplay_states": 800 + i, "state_buffer": 8000} for i in range(14)]
    row = {"seed": seed, "params": 60555, "history": history, "recorded_states": sum(h["selfplay_states"] for h in history),
           "games_per_pass": [48] * 14, "solver_forbidden": True, "positions": positions, "probe": [0.5, 0.6],
           "sizes": {"final_buffer": 8000}, "containment": {}, "conversion": []}
    row.update(over)
    return row


def _config(**over):
    return {**C0_RECIPE, "sampling": dict(C0_SAMPLING), "plies": list(C0_PLIES), **over}


def _evidence(nets=None, **top):
    nets = nets if nets is not None else [_net(s) for s in C0_SEEDS]
    return {"training_fingerprint": C0_ERA, "measurement_fingerprint": C0_MEASUREMENT_FP, "config": _config(),
            "seeds": nets, **top}


def _spread(gaps_e, gaps_l=None, base_err=0.1):
    gaps_l = gaps_l or [0.2] * 5
    nets = []
    for seed, e, lab in zip(C0_SEEDS, gaps_e, gaps_l):
        err_sib = base_err + e
        nets.append(_net(seed, err_final=base_err, err_sib=err_sib, label_sib=(1 - err_sib) + lab))
    return _evidence(nets)


def test_a_consistent_exposure_gap_and_label_headroom_is_GO():
    r = stage0_report(_spread([0.2, 0.18, 0.22, 0.2, 0.2], [0.2, 0.16, 0.24, 0.2, 0.18]))
    assert r["integrity"] == [] and r["C0"]["verdict"] == "go"
    assert r["C0"]["E"]["mean"] == pytest.approx(0.2) and r["C0"]["E"]["lower"] > 0 and r["C0"]["L"]["lower"] > 0
    assert "not evidence that siblings raise conversion" in r["C0"]["reading"]


def test_no_exposure_gap_is_STOP_and_says_generalisation_is_untested():
    r = stage0_report(_spread([0.0, 0.01, -0.01, 0.0, 0.0]))
    assert r["C0"]["verdict"] == "stop" and "binding: the exposure gap" in r["C0"]["reading"]
    assert "untested" in r["C0"]["reading"]


def test_no_label_headroom_is_STOP_even_with_a_large_exposure_gap():
    r = stage0_report(_spread([0.3] * 5, [0.0, 0.01, -0.01, 0.0, 0.0], base_err=0.3))
    assert r["C0"]["verdict"] == "stop" and "binding: the label headroom" in r["C0"]["reading"]


def test_the_TIC_TAC_TOE_signature_where_siblings_worked_reads_GO():
    r = stage0_report(_spread([0.06, 0.02, 0.04, 0.04, 0.04], [0.3] * 5, base_err=0.0))
    assert r["C0"]["E"]["mean"] == pytest.approx(0.04) and r["C0"]["verdict"] == "go"


def test_both_gaps_must_clear_their_bounds_a_large_E_cannot_carry_a_marginal_L():
    r = stage0_report(_spread([0.3] * 5, [0.06, -0.02, 0.05, -0.01, 0.04], base_err=0.3))
    assert r["C0"]["E"]["lower"] > 0 and r["C0"]["L"]["lower"] <= 0 and r["C0"]["verdict"] == "undecided"


def test_a_gap_that_varies_across_nets_is_UNDECIDED_however_large_its_mean():
    r = stage0_report(_spread([0.4, -0.1, 0.34, -0.06, 0.1]))
    assert r["C0"]["E"]["mean"] == pytest.approx(0.136) and r["C0"]["verdict"] == "undecided"


def test_a_real_but_tiny_gap_below_the_smallest_effect_worth_a_stage_1_is_STOP():
    r = stage0_report(_spread([0.012, 0.014, 0.012, 0.014, 0.012]))
    assert r["C0"]["E"]["lower"] > 0 and r["C0"]["verdict"] == "stop"


def test_a_small_mean_whose_upper_bound_still_reaches_the_smallest_effect_is_UNDECIDED_not_STOP():
    r = stage0_report(_spread([0.0, 0.03, -0.01, 0.04, 0.01]))
    e = r["C0"]["E"]
    assert e["mean"] < 0.02 <= e["upper"] and r["C0"]["verdict"] == "undecided"


def test_the_bounds_are_one_sided_95_with_the_net_as_the_unit():
    m = _across([0.1, 0.2, 0.3, 0.2, 0.2])
    se = (sum((v - 0.2) ** 2 for v in [0.1, 0.2, 0.3, 0.2, 0.2]) / 4 / 5) ** 0.5
    assert m["mean"] == pytest.approx(0.2) and m["se"] == pytest.approx(se)
    assert m["lower"] == pytest.approx(0.2 - t_critical(4, 0.10) * se)
    assert m["upper"] == pytest.approx(0.2 + t_critical(4, 0.10) * se)


def test_the_gaps_read_the_RAW_policy():
    row = _net(201, err_final=0.1, err_sib=0.3, label_sib=0.9)
    for p in row["positions"]:
        p["avg_ok"] = not p["raw_ok"]
    g = net_gaps(row)
    assert g["E"] == pytest.approx(0.2) and g["L"] == pytest.approx(0.2)


def test_plies_are_averaged_with_EQUAL_weight_so_an_uneven_sample_cannot_make_a_gap():
    """Review finding: pooling a thin final-buffer cell at a late, error-prone ply with full sibling cells made a gap
    out of nothing. Per-ply gaps averaged with equal weight do not."""
    rates = {ply: (0.10 + 0.02 * i, 0.10 + 0.02 * i, 0.9) for i, ply in enumerate(C0_PLIES)}
    row = _net(201, per_ply=rates)
    row["positions"] = [p for p in row["positions"]
                        if not (p["cls"] == "final_buffer" and p["ply"] in C0_PLIES[-2:] and p["key"] % 5)]
    assert net_gaps(row)["E"] == pytest.approx(0.0, abs=1e-9)


def test_each_mover_s_gaps_are_reported_separately():
    rates = {p: ((0.1, 0.3, 0.9) if p % 2 == 0 else (0.1, 0.1, 0.9)) for p in C0_PLIES}
    g = net_gaps(_net(201, per_ply=rates))
    assert g["by_mover"]["first"]["E"] == pytest.approx(0.2) and g["by_mover"]["second"]["E"] == pytest.approx(0.0)
    assert g["E"] == pytest.approx(0.1)
    by = stage0_report(_evidence([_net(s, per_ply=rates) for s in C0_SEEDS]))["descriptives"]["by_mover"]
    assert by["first"]["E"]["mean"] == pytest.approx(0.2) and by["second"]["E"]["mean"] == pytest.approx(0.0)


def _noisy_evidence(rng, e_mean, e_sd, l_gap, base=0.2):
    nets = []
    for seed in C0_SEEDS:
        e_true = rng.gauss(e_mean, e_sd)
        raw_err = min(max(base + e_true, 0.0), 1.0)
        lab_ok = min(1.0, 1 - raw_err + l_gap)
        positions = []
        for ply in C0_PLIES:
            positions += [{"cls": "final_buffer", "ply": ply, "raw_ok": rng.random() >= base, "label_ok": True,
                           "avg_ok": True} for _ in range(150)]
            positions += [{"cls": "sibling", "ply": ply, "raw_ok": rng.random() >= raw_err,
                           "label_ok": rng.random() < lab_ok, "avg_ok": True} for _ in range(C0_SAMPLING["sibling"])]
        nets.append(_net(seed, positions=positions))
    return _evidence(nets)


def test_the_rule_s_operating_characteristic_on_NOISY_evidence_at_the_registered_sizes():
    """The calibration claim, checked where it must hold: with binomial sampling at 150 final-buffer and 300
    sibling positions per ply and the tic-tac-toe spread between nets, the tic-tac-toe signature reads GO most of
    the time, and a true zero gap almost never does."""
    rng = random.Random(0)
    signature = [stage0_report(_noisy_evidence(rng, 0.042, 0.019, 0.2))["C0"]["verdict"] for _ in range(60)]
    zero = [stage0_report(_noisy_evidence(rng, 0.0, 0.019, 0.2))["C0"]["verdict"] for _ in range(60)]
    assert signature.count("go") >= 48
    assert zero.count("go") <= 5 and zero.count("stop") >= 15


def _broken_net(**over):
    return lambda: _evidence([_net(s, **(over if s == 203 else {})) for s in C0_SEEDS])


def _config_break(**over):
    return lambda: {**_evidence(), "config": _config(**over)}


BREAKAGES = {
    "era": lambda: _evidence(training_fingerprint="3c8faf6a7505"),
    "measurement": lambda: _evidence(measurement_fingerprint="0" * 12),
    "sims": _config_break(sims=32),
    "labels": _config_break(reanalyze_sims=200),
    "net": _config_break(arch={"channels": 64}),
    "iterations": _config_break(iterations=13),
    "fewer siblings sampled": _config_break(sampling={**C0_SAMPLING, "sibling": 30}),
    "one mover only": _config_break(plies=[p for p in C0_PLIES if p % 2 == 0]),
    "seeds": lambda: _evidence([_net(s) for s in (201, 202, 203, 204, 206)]),
    "four nets": lambda: _evidence([_net(s) for s in C0_SEEDS[:4]]),
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
    r = stage0_report(BREAKAGES[name]())
    assert r["C0"]["verdict"] == "not_run" and r["integrity"]
    assert not any("could not be read" in p for p in r["integrity"])


@pytest.mark.parametrize("cls,ply", [("final_buffer", C0_PLIES[0]), ("sibling", C0_PLIES[1]), ("sibling", C0_PLIES[-1])])
def test_a_decision_cell_with_too_few_positions_is_NOT_RUN(cls, ply):
    ev = _evidence()
    row = ev["seeds"][2]
    kept = [p for p in row["positions"] if (p["cls"], p["ply"]) == (cls, ply)][:79]
    row["positions"] = [p for p in row["positions"] if (p["cls"], p["ply"]) != (cls, ply)] + kept
    r = stage0_report(ev)
    assert r["C0"]["verdict"] == "not_run" and any(f"at ply {ply}" in p for p in r["integrity"])


def test_descriptive_classes_may_be_thin_without_breaking_integrity():
    ev = _evidence()
    for row in ev["seeds"]:
        row["positions"] = [p for p in row["positions"] if p["cls"] in ("final_buffer", "sibling")]
    assert stage0_report(ev)["integrity"] == []


@pytest.mark.parametrize("junk", [None, {}, {"seeds": "x", "config": {}}, {"seeds": [None], "config": {}}])
def test_evidence_that_cannot_be_read_is_NOT_RUN_never_an_exception(junk):
    assert stage0_report(junk)["C0"]["verdict"] == "not_run"


def test_a_malformed_position_that_passes_integrity_is_NOT_RUN_with_the_reason():
    ev = _evidence()
    del ev["seeds"][1]["positions"][0]["raw_ok"]
    r = stage0_report(ev)
    assert r["C0"]["verdict"] == "not_run" and "could not be read" in r["integrity"][0]


def test_the_cells_describe_every_class_and_ply_for_every_policy():
    cells = stage0_report(_evidence())["descriptives"]["cells"]
    assert set(cells) == {f"{c}@{p}" for c in ("final_buffer", "sibling", "two_moves_off", "random") for p in C0_PLIES}
    assert cells[f"sibling@{C0_PLIES[1]}"] == {"n": 5 * N, "raw_ok": pytest.approx(0.7), "avg_ok": pytest.approx(0.7),
                                   "label_ok": pytest.approx(0.9)}
