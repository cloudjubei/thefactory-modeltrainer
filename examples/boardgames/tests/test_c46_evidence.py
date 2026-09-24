"""§C.46 — the PRE-REGISTERED labels x exposure x architecture claims, judged by harness.ceiling.c46_report from the
nine arm evidence files (seeds 41-60), the G0 capacity gate (evidence/c46_G0_capacity.json) and the §C.45 R200 arm
as the reference for the recurrent-failure set P. Every claim has a proof (passes only on `supported`) and an
undecidable proof (passes when the report says the claim could not be judged); a claim is refuted only when both
fail — its one-sided 95% upper bound excluded the smallest effect of interest.

The register verified h31-h37 against these nodes as registered (2026-09-24): E1, E2, L1, L2, R1, R2 SUPPORTED;
A1 INCONCLUSIVE. The undecidable nodes of the decided claims are gone (they are consulted only when a proof fails),
and A1's proof node now pins its recorded inconclusive reading — the register keeps the frozen verdicts."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from harness.ceiling import C46_ARMS, c46_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
UNDECIDABLE = {"inconclusive", "not_delivered", "not_reached", "not_run", "moved", "stopped"}


@lru_cache(maxsize=1)
def _report():
    from games.tictactoe import TicTacToe
    arms = {a: json.loads((EVIDENCE / f"c46_{a}.json").read_text()) for a in C46_ARMS}
    assert all(len(e["seeds"]) == 20 for e in arms.values())
    return c46_report(arms, json.loads((EVIDENCE / "tictactoe_ceiling.json").read_text()), TicTacToe(),
                      json.loads((EVIDENCE / "c46_G0_capacity.json").read_text()),
                      reference=json.loads((EVIDENCE / "c45_mixed_R200.json").read_text()),
                      notes={"depth2_gap": [4983], "label_limited": [601]})


def _verdict(claim):
    return _report()["claims"][claim]["verdict"]


def test_c46_E1_siblings_raise_net_level_coverage_with_200_sim_labels():
    assert _verdict("E1") == "supported"



def test_c46_E2_siblings_raise_net_level_coverage_with_32_sim_labels():
    assert _verdict("E2") == "supported"



def test_c46_A1_INCONCLUSIVE_the_residual_gain_is_real_but_its_bounds_straddle_the_smallest_effect():
    a1 = _report()["claims"]["A1"]
    assert a1["verdict"] == "inconclusive"
    assert 0 < a1["lower"] < a1["sesoi"] < a1["upper"]


def test_c46_A1_is_undecidable():
    assert _verdict("A1") in UNDECIDABLE


def test_c46_L1_legacy_200_sim_relabelling_replicates_the_target_gain():
    assert _verdict("L1") == "supported"



def test_c46_L2_legacy_200_sim_relabelling_replicates_the_prior_gain():
    assert _verdict("L2") == "supported"



def test_c46_R1_residual_200_sim_relabelling_raises_target_correctness():
    assert _verdict("R1") == "supported"



def test_c46_R2_residual_200_sim_relabelling_raises_the_target_prior():
    assert _verdict("R2") == "supported"



def _arm(name):
    return json.loads((EVIDENCE / f"c46_{name}.json").read_text())


def _trained(row):
    return {k for p in row["passes"] for k, sp_n, _a, sib_n, _b in p["dose"] if sp_n + sib_n > 0}


def _by_seed(arm):
    return {s["seed"]: s for s in arm["seeds"]}


def test_h38_E1_is_MOSTLY_IN_SAMPLE_with_a_small_real_spillover():
    from harness.ceiling import _paired_gain
    c, t = _by_seed(_arm("R200")), _by_seed(_arm("R200S"))
    universe = set(_arm("R200")["universe"])
    order = sorted(c)
    share = sum(len(_trained(t[s]) & universe) for s in order) / (len(order) * len(universe))
    assert share > 0.80
    direct, spill = [], []
    for s in order:
        only_t = _trained(t[s]) - _trained(c[s])
        neither = universe - _trained(t[s]) - _trained(c[s])
        fc, ft = set(c[s]["policy_fail_keys"]), set(t[s]["policy_fail_keys"])
        direct.append(len(fc & only_t) - len(ft & only_t))
        spill.append(len(fc & neither) - len(ft & neither))
    total = sum(len(c[s]["policy_fail_keys"]) - len(t[s]["policy_fail_keys"]) for s in order)
    assert sum(direct) / total > 0.8
    assert 0 < sum(spill) / total < 0.2 and _paired_gain(spill, 0.05)["p"] < 0.01


def test_h40_exact_solver_labels_WITHOUT_siblings_are_the_worst_residual_arm():
    from harness.ceiling import _paired_gain
    rx, r200, r32 = (_by_seed(_arm(a)) for a in ("Rx", "R200", "R32"))
    order = sorted(rx)
    worse_than_r200 = [len(rx[s]["policy_fail_keys"]) - len(r200[s]["policy_fail_keys"]) for s in order]
    worse_than_r32 = [len(rx[s]["policy_fail_keys"]) - len(r32[s]["policy_fail_keys"]) for s in order]
    assert _paired_gain(worse_than_r200, 0.05)["p"] < 0.01 and _paired_gain(worse_than_r32, 0.05)["p"] < 0.01
    universe = set(_arm("Rx")["universe"])

    def untrained_rate(arm):
        fails = sum(len(set(arm[s]["policy_fail_keys"]) & (universe - _trained(arm[s]))) for s in order)
        cells = sum(len(universe - _trained(arm[s])) for s in order)
        return fails / cells
    assert sum(len(_trained(rx[s]) & universe) for s in order) > sum(len(_trained(r200[s]) & universe) for s in order)
    assert untrained_rate(rx) > 1.5 * untrained_rate(r200)


def _orientation(name):
    return json.loads((EVIDENCE / f"c46_{name}_orientation.json").read_text())["seeds"]


def test_h39_the_canonical_image_metric_OVERSTATED_perfection_no_R200S_net_is_perfect_in_every_orientation():
    rows = _orientation("R200S")
    assert len(rows) == 20 and all(r["reproduced"] for r in rows)
    recorded = {s["seed"]: max(s["passes"], key=lambda p: p["pass"])["weights_sha"] for s in _arm("R200S")["seeds"]}
    assert all(r["weights_sha"] == recorded[r["seed"]] for r in rows)
    canonical = sum(1 for r in rows if not r["canonical_fail_keys"])
    strict = sum(1 for r in rows if not r["image_fail_keys"])
    averaged = sum(1 for r in rows if not r["symmetrized_fail_keys"])
    assert canonical == 9 and strict == 0 and averaged >= 15
    assert all(r["positions"] == 4520 for r in rows)
