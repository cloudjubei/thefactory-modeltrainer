"""Direct tests for harness/floor.py — the pre-registered §C.47 floor verdicts. Synthetic arms in the full evidence
schema vary how many seeds are perfect under each reading and which integrity condition breaks, so no verdict can
pass by one layout; a malformed record must read NOT_RUN, never a refutation."""
from __future__ import annotations

import pytest

from harness.floor import (C47_ERA, C47_ISOMETRIES, C47_MEASUREMENT_FP, C47_REFERENCE_CONFIG, C47_REPRO, C47_SEEDS,
                           C47_SUBGROUPS, _bootstrap_mean, _sign_p, floor_report)

UNIVERSE = list(range(100, 140))


def _row(seed, sym=0, strict=0, canon=0, search=0, untrained=(), history_siblings=270, sibling_only=()):
    trained = [k for k in UNIVERSE if k not in untrained]
    dose = [[k, 3 if k in trained and k not in sibling_only else 0, 3, 1 if k in trained else 0, 1] for k in UNIVERSE]
    return {"seed": seed, "params": 57453, "policy_fail_keys": list(range(canon)),
            "failures": [{"key": 900 + i} for i in range(search)],
            "orientation": {"symmetrized_fail_keys": UNIVERSE[:sym], "image_fail_keys": UNIVERSE[:strict],
                            "canonical_fail_keys": UNIVERSE[:canon], "failing_images": 2 * strict,
                            "positions": 4520, "isometries": list(C47_ISOMETRIES)},
            "subgroups": {o: {"isometries": list(g), "fail_keys": UNIVERSE[:sym if o == "8" else strict],
                              "failing_positions": sym if o == "8" else 2 * strict, "positions": 4520}
                          for o, g in C47_SUBGROUPS.items()},
            "passes": [{"pass": i + 1, "weights_sha": f"{seed}-{i}", "dose": dose} for i in range(6)],
            "history": [{"iteration": i + 1, "siblings": 0 if i == 0 else history_siblings} for i in range(6)]}


def _arm(perfect_sym=16, perfect_strict=1, perfect_canon=10, seeds=C47_SEEDS, **config):
    rows = [_row(s, sym=0 if i < perfect_sym else 1 + i % 3, strict=0 if i < perfect_strict else 2 + i % 4,
                 canon=0 if i < perfect_canon else 1, search=i % 2) for i, s in enumerate(seeds)]
    return {"game": "tictactoe", "training_fingerprint": C47_ERA, "measurement_fingerprint": C47_MEASUREMENT_FP,
            "config": {**C47_REFERENCE_CONFIG, **config, "seeds": list(seeds)}, "universe": UNIVERSE, "seeds": rows,
            "operator_sanity": {"untrained_symmetrized_failures": [208, 211, 199, 214, 205]}}


def _repro(**over):
    row = {"seed": C47_REPRO["seed"], "policy_fail_keys": list(C47_REPRO["policy_fail_keys"]),
           "passes": [{"weights_sha": "x"}, {"weights_sha": C47_REPRO["weights_sha"]}]}
    return {"training_fingerprint": C47_ERA, "measurement_fingerprint": C47_MEASUREMENT_FP,
            "config": {**C47_REFERENCE_CONFIG, "seeds": [41]}, "seeds": [row], **over}


@pytest.mark.parametrize("perfect,verdict", [(20, "supported"), (15, "supported"), (14, "inconclusive"),
                                             (12, "inconclusive"), (11, "refuted"), (0, "refuted")])
def test_F1_is_supported_at_15_refuted_at_11_and_inconclusive_between(perfect, verdict):
    r = floor_report(_arm(perfect_sym=perfect, perfect_canon=min(perfect, 9)), _repro())
    assert r["integrity"] == [] and r["F1"]["verdict"] == verdict and r["F1"]["perfect"] == perfect
    assert f"{perfect}/20" in r["F1"]["reading"]


def test_F1_supported_wording_names_the_operator_the_bound_and_the_registered_raw_readings():
    r = floor_report(_arm(perfect_sym=16, perfect_strict=1, perfect_canon=9), _repro())
    text = r["F1"]["reading"]
    assert "symmetry-averaged" in text and "adopted after §C.46" in text and "9/20" in text and "1/20" in text
    assert f"{r['F1']['lower']:.2f}" in text and "near-exhaustive" in text and "not evidence the mechanism transfers" in text


def test_F1_reads_the_averaged_operator_not_the_canonical_or_the_strict_one():
    assert floor_report(_arm(perfect_sym=15, perfect_strict=20, perfect_canon=20), _repro())["F1"]["perfect"] == 15
    assert floor_report(_arm(perfect_sym=11, perfect_strict=20, perfect_canon=20), _repro())["F1"]["verdict"] == "refuted"


def _discordant(only_avg, only_raw, both=2):
    arm = _arm(perfect_sym=0, perfect_strict=0)
    rows = arm["seeds"]
    for i in range(only_avg):
        rows[i]["orientation"]["symmetrized_fail_keys"], rows[i]["orientation"]["image_fail_keys"] = [], [1]
    for i in range(only_avg, only_avg + only_raw):
        rows[i]["orientation"]["symmetrized_fail_keys"], rows[i]["orientation"]["image_fail_keys"] = [1], []
    for i in range(only_avg + only_raw, only_avg + only_raw + both):
        rows[i]["orientation"]["symmetrized_fail_keys"], rows[i]["orientation"]["image_fail_keys"] = [], []
    return arm


@pytest.mark.parametrize("only_avg,only_raw,verdict", [(16, 0, "supported"), (10, 0, "supported"),
                                                      (6, 0, "inconclusive"), (9, 3, "inconclusive"),
                                                      (0, 8, "refuted"), (0, 0, "inconclusive")])
def test_F2_is_an_exact_sign_test_over_the_seeds_the_two_readings_disagree_on(only_avg, only_raw, verdict):
    r = floor_report(_discordant(only_avg, only_raw), _repro())
    f2 = r["F2"]
    assert (f2["only_averaged_perfect"], f2["only_strict_perfect"], f2["verdict"]) == (only_avg, only_raw, verdict)
    assert f2["p"] == pytest.approx(_sign_p(only_avg, only_avg + only_raw))


def test_the_sign_test_is_the_exact_binomial_tail():
    assert _sign_p(10, 10) == pytest.approx(1 / 1024)
    assert _sign_p(6, 6) == pytest.approx(1 / 64)
    assert _sign_p(0, 0) == 1.0 and _sign_p(0, 7) == 1.0
    assert _sign_p(9, 12) == pytest.approx(sum(__import__("math").comb(12, i) for i in range(9, 13)) / 4096)


def _break(path, value):
    def apply(arm, repro):
        target = {"arm": arm, "repro": repro}[path[0]]
        for k in path[1:-1]:
            target = target[k]
        if value is KeyError:
            del target[path[-1]]
        else:
            target[path[-1]] = value
    return apply


def _both(field, value):
    def apply(arm, repro):
        arm[field] = repro[field] = value
    return apply


BREAKAGES = {
    "era": _break(("arm", "training_fingerprint"), "2600dc4f574a"),
    "era, reproduced under it too": _both("training_fingerprint", "2600dc4f574a"),
    "measurement": _break(("arm", "measurement_fingerprint"), "a9a9f94a0ee7"),
    "measurement, reproduced under it too": _both("measurement_fingerprint", "a9a9f94a0ee7"),
    "labels": _break(("arm", "config", "reanalyze_sims"), 32),
    "siblings off": _break(("arm", "config", "reanalyze_siblings"), False),
    "legacy net": _break(("arm", "config", "arch"), {"channels": 32}),
    "threads": _break(("arm", "config", "threads"), 2),
    "openings": _break(("arm", "config", "opening_zero_frac"), 0.0),
    "other game": _break(("arm", "config", "game"), "connect4"),
    "config seeds": _break(("arm", "config", "seeds"), list(range(61, 81))),
    "params": _break(("arm", "seeds", 3, "params"), 12746),
    "positions": _break(("arm", "seeds", 7, "orientation", "positions"), 4519),
    "isometries": _break(("arm", "seeds", 7, "orientation", "isometries"), list(C47_ISOMETRIES[::-1])),
    "no orientation": _break(("arm", "seeds", 7, "orientation"), KeyError),
    "subgroup": _break(("arm", "seeds", 2, "subgroups", "2", "isometries"), ["identity", "flip_v"]),
    "five passes": _break(("arm", "seeds", 5, "passes"), [{"dose": []}] * 5),
    "siblings missing": _break(("arm", "seeds", 5, "history", 3, "siblings"), 0),
    "no search failures": _break(("arm", "seeds", 5, "failures"), KeyError),
    "repro weights": _break(("repro", "seeds", 0, "passes", 1, "weights_sha"), "0" * 64),
    "repro misplays": _break(("repro", "seeds", 0, "policy_fail_keys"), [601]),
    "repro era": _break(("repro", "training_fingerprint"), "2600dc4f574a"),
    "repro config": _break(("repro", "config", "reanalyze_sims"), 32),
    "repro seed": _break(("repro", "seeds", 0, "seed"), 42),
    "no repro": _break(("repro", "seeds"), []),
}


@pytest.mark.parametrize("name", sorted(BREAKAGES))
def test_any_broken_integrity_condition_makes_both_claims_NOT_RUN(name):
    arm, repro = _arm(), _repro()
    BREAKAGES[name](arm, repro)
    r = floor_report(arm, repro)
    assert r["F1"]["verdict"] == "not_run" and r["F2"]["verdict"] == "not_run" and r["integrity"]
    assert not any("could not be read" in p for p in r["integrity"]), "an explicit check must name the breakage"


@pytest.mark.parametrize("seeds", [range(81, 100), range(61, 81), list(range(81, 100)) + [81]])
def test_other_seeds_are_NOT_RUN(seeds):
    assert floor_report(_arm(seeds=seeds), _repro())["F1"]["verdict"] == "not_run"


@pytest.mark.parametrize("junk", [None, {}, {"seeds": "x", "config": {}}, {"seeds": [None], "config": {}}])
def test_evidence_that_cannot_be_read_is_NOT_RUN_never_an_exception(junk):
    assert floor_report(junk, _repro())["F1"]["verdict"] == "not_run"
    assert floor_report(_arm(), junk)["F1"]["verdict"] == "not_run"


def test_a_record_that_passes_integrity_but_cannot_be_judged_is_NOT_RUN_with_the_reason():
    arm = _arm()
    arm["seeds"][4]["passes"][2]["dose"] = [[100]]
    r = floor_report(arm, _repro())
    assert r["F1"]["verdict"] == "not_run" and "could not be read" in r["integrity"][0]


def test_measurement_only_extras_do_not_break_integrity():
    arm = _arm(target_evidence="evidence/tictactoe_ceiling.json.gz", versions={"torch": "2.13.0"})
    assert floor_report(arm, _repro())["integrity"] == []


def test_exposure_splits_each_reading_s_failures_into_trained_and_never_trained_positions():
    arm = _arm(perfect_sym=18, perfect_canon=18)
    for row in arm["seeds"]:
        row.update(_row(row["seed"], untrained=UNIVERSE[:4], sibling_only=UNIVERSE[4:8]))
    arm["seeds"][0]["orientation"]["canonical_fail_keys"] = UNIVERSE[2:6]
    arm["seeds"][1]["orientation"]["symmetrized_fail_keys"] = UNIVERSE[3:5]
    e = floor_report(arm, _repro())["descriptives"]["exposure"]
    assert e["mean_never_trained_share"] == pytest.approx(4 / 40)
    assert e["failures"]["canonical"] == {"trained": 2, "never_trained": 2}
    assert e["failures"]["symmetrized"] == {"trained": 1, "never_trained": 1}


def test_descriptives_count_every_reading_and_every_subgroup():
    d = floor_report(_arm(perfect_sym=16, perfect_strict=1, perfect_canon=10), _repro())["descriptives"]
    assert d["perfect"]["symmetrized"] == 16 and d["perfect"]["strict"] == 1 and d["perfect"]["canonical"] == 10
    assert d["perfect"]["search"] == 10 and d["perfect"]["subgroup_8"] == 16 and d["perfect"]["subgroup_1"] == 1
    assert d["averaged_reading_disagrees_across_images"] == []
    assert d["operator_sanity"]["untrained_symmetrized_failures"][0] == 208


def test_a_seed_whose_all_image_averaged_reading_differs_from_its_canonical_one_is_named():
    arm = _arm()
    arm["seeds"][2]["subgroups"]["8"]["fail_keys"] = [123]
    assert floor_report(arm, _repro())["descriptives"]["averaged_reading_disagrees_across_images"] == [83]


@pytest.mark.parametrize("values", [[0] * 16 + [1] * 4, [0] * 20, [3, 1, 4, 1, 5, 9, 2, 6]])
def test_the_bootstrap_interval_brackets_the_mean(values):
    a = _bootstrap_mean(values)
    assert a["ci"][0] <= a["mean"] <= a["ci"][1] and a["ci"][0] >= min(values)


def test_the_bootstrap_interval_is_reproducible_from_its_seed():
    values = [3, 1, 4, 1, 5, 9, 2, 6]
    runs = [_bootstrap_mean(values, draws=40, seed=s)["ci"] for s in (0, 0, 1, 2, 3)]
    assert runs[0] == runs[1] and len({tuple(r) for r in runs[1:]}) > 1


def test_the_registered_constants_are_the_game_s_verified_isometries_and_groups():
    from games.tictactoe import TicTacToe
    from harness.symmetry import verified_isometries
    assert list(C47_ISOMETRIES) == [iso.name for iso, _f in verified_isometries(TicTacToe())]
    assert all(g[0] == "identity" and set(g) <= set(C47_ISOMETRIES) for g in C47_SUBGROUPS.values())
    assert [len(g) for g in C47_SUBGROUPS.values()] == [1, 2, 4, 8]


def test_the_reference_recipe_is_R200S_as_its_pinned_evidence_recorded():
    from pathlib import Path

    from harness.evidence import load_evidence
    path = Path(__file__).resolve().parent.parent / "evidence" / "c46_R200S.json.gz"
    if not path.exists():
        pytest.skip("evidence/ is gitignored and c46_R200S is not on this machine")
    ev = load_evidence(path)
    assert {k: ev["config"].get(k) for k in C47_REFERENCE_CONFIG} == C47_REFERENCE_CONFIG
    seed = next(s for s in ev["seeds"] if s["seed"] == C47_REPRO["seed"])
    assert seed["passes"][-1]["weights_sha"] == C47_REPRO["weights_sha"]
    assert sorted(seed["policy_fail_keys"]) == C47_REPRO["policy_fail_keys"]
    assert not set(C47_SEEDS) & {s for f in ("c46_G0_capacity.json.gz",) for r in load_evidence(path.parent / f)["runs"]
                                 for s in [r["seed"]]}
