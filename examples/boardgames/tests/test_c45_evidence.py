"""§C.45 — the PRE-REGISTERED label-quality claims, judged by harness.ceiling.c45_report from the five arm evidence
files (seeds 21-40), and the bounded post-hoc readings the verification pass added (h28-h30).

The register verified h24-h27 against the proof nodes as registered; B1 (h27) was REFUTED, so its proof node now
pins the refutation, and the inconclusive-proof nodes (consulted only when a proof fails) are gone — the register
keeps the frozen verdicts, and these tests keep the suite green and pin each verdict against later code changes."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from harness.ceiling import c45_report

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
ARMS = ("base", "mixed", "mixed_deep", "mixed_R32", "mixed_R200")
UNDECIDABLE = {"inconclusive", "not_delivered", "not_reached", "not_run", "moved", "stopped"}


@lru_cache(maxsize=1)
def _report():
    arms = {a: json.loads((EVIDENCE / f"c45_{a}.json").read_text()) for a in ARMS}
    assert all(len(e["seeds"]) == 20 for e in arms.values())
    return c45_report(arms, json.loads((EVIDENCE / "tictactoe_ceiling.json").read_text()))


def _verdict(claim):
    return _report()["claims"][claim]["verdict"]


def test_c45_G1_the_blind_spot_replicates_on_seeds_21_to_40():
    assert _report()["G1"]["passed"] is True


def test_c45_A1_deeper_RELABELLING_fixes_the_target_states():
    assert _verdict("A1") == "supported"


def test_c45_A2_deeper_relabelling_fixes_the_target_PRIOR():
    assert _verdict("A2") == "supported"


def test_c45_B1_REFUTED_deeper_self_play_does_not_raise_target_correctness():
    b1 = _report()["claims"]["B1"]
    assert b1["verdict"] == "refuted" and b1["upper"] < 0


def _arm(name):
    return json.loads((EVIDENCE / f"c45_{name}.json").read_text())


def _target():
    return {f["key"] for s in json.loads((EVIDENCE / "tictactoe_ceiling.json").read_text())["seeds"]
            for f in s["failures"]}


def test_h28_label_quality_is_a_PARTIAL_cause_about_half_the_target_shortfall_is_repaired():
    a1, a2 = _report()["claims"]["A1"], _report()["claims"]["A2"]
    r32 = [1 - len({f["key"] for f in s["failures"]} & _target()) / 17 for s in _arm("mixed_R32")["seeds"]]
    shortfall = 1 - sum(r32) / len(r32)
    assert 0.45 < a1["mean_delta"] / shortfall < 0.60
    assert a1["lower"] < a1["sesoi"] and a2["lower"] < a2["sesoi"]
    r200 = _arm("mixed_R200")["seeds"]
    inside = sum(len({f["key"] for f in s["failures"]} & _target()) for s in r200)
    total = sum(len(s["failures"]) for s in r200)
    assert inside / total > 0.85


def _dose_without(arm, drop):
    return [sum(row["argmax_ok"] for p in s["written_labels"] for k, row in p["per_key"] if k not in drop)
            for s in sorted(_arm(arm)["seeds"], key=lambda s: s["seed"])]


def test_h29_B1_refutes_the_RECIPE_not_labels_its_delivery_rested_on_one_state():
    from harness.ceiling import _paired_gain
    mixed, deep = _dose_without("mixed", {90}), _dose_without("mixed_deep", {90})
    assert _paired_gain([d - m for m, d in zip(mixed, deep)], 0.05)["p"] > 0.5
    exposure = {a: sum(p["n_target"] for s in _arm(a)["seeds"] for p in s["written_labels"])
                for a in ("mixed", "mixed_deep")}
    assert exposure["mixed_deep"] < 0.75 * exposure["mixed"]


def test_h30_deep_self_play_never_trains_on_O_to_move_winning_positions_and_fails_there():
    from games.tictactoe import TicTacToe
    from harness.ceiling import _paired_gain
    from harness.coverage import optimal_play_keys, reachable_states
    g = TicTacToe()
    states, _ = reachable_states(g, exact=True, symmetry=True)
    on_path = optimal_play_keys(g)
    o_wins = {g.canonical_key(s) for s in states if s.to_move == 1 and g.canonical_key(s) not in on_path
              and g.position_value(s) == 1}
    deep = _arm("mixed_deep")["seeds"]
    assert all(sum(c for k, c in s["visits"] if k in o_wins) == 0 for s in deep)
    assert sum(1 for s in _arm("mixed")["seeds"] if sum(c for k, c in s["visits"] if k in o_wins) > 0) >= 15
    o_target = {g.canonical_key(s) for s in states if s.to_move == 1} & _target()

    def o_fails(arm):
        return [len({f["key"] for f in s["failures"]} & o_target)
                for s in sorted(_arm(arm)["seeds"], key=lambda s: s["seed"])]
    assert _paired_gain([d - m for m, d in zip(o_fails("mixed"), o_fails("mixed_deep"))], 0.05)["p"] < 1e-3
