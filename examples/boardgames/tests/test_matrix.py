"""Direct tests for harness.matrix — the cross-game regression matrix (§C.20/§C.37b).

A single-game A/B is confound-prone and easy to overfit a conclusion to (§C.15-§C.19). The matrix turns it into a
GENERALISATION test: rows = a change, columns = games, cells = a paired delta. The verdict a row earns is the
whole point — a lever that moves EVERY game in ONE direction (generalises) is a different thing from one that
moves a single game (local), from one that helps some and hurts others (contested), from one that is absent
everywhere (null). Multiplicity is counted ACROSS the row: N games is N comparisons, so a cell clears the bar
only at alpha/N — the same discipline measure_ab applies within a game, applied across games."""
from __future__ import annotations

import pytest

from harness.matrix import build_matrix, corrected_alpha, render_matrix, row_verdict


def _cell(game, diff, p, n=256, null_below=0.03, caveats=None, half=None):
    # CI half-width defaults to a narrow, well-powered interval; pass `half` to exercise the underpowered regime
    h = 0.02 if half is None else half
    return {"game": game, "diff": diff, "p": p, "n": n, "ci": (diff - h, diff + h),
            "null_below": null_below, "caveats": caveats or {}}


def test_corrected_alpha_divides_by_the_game_count():
    assert corrected_alpha(1) == 0.05
    assert corrected_alpha(5) == 0.05 / 5
    assert corrected_alpha(0) == 0.05        # never divides by zero


def test_a_change_that_moves_every_game_the_same_way_GENERALISES():
    cells = [_cell("connect4", +0.20, 0.0001), _cell("othello", +0.15, 0.0002),
             _cell("checkers", +0.18, 0.0003)]
    v = row_verdict(cells)
    assert v["status"] == "generalises" and v["direction"] == "+"
    assert v["n_significant"] == 3 and v["n_games"] == 3


def test_generalises_uses_the_per_cell_IUT_bar_not_alpha_over_N():
    # every game moves the same way at per-game p=0.02: that is a CONJUNCTION (intersection-union test), whose
    # level is alpha=0.05 with NO Bonferroni — so five games at p=0.02 GENERALISES, though 0.02 > 0.05/5.
    cells = [_cell(f"g{i}", +0.10, 0.02) for i in range(5)]
    v = row_verdict(cells)
    assert v["status"] == "generalises" and v["direction"] == "+"
    assert all(c["iut_significant"] and not c["significant"] for c in v["cells"])   # IUT-sig, not union-sig


def test_the_UNION_questions_use_the_corrected_bar():
    # local/contested are union questions (did ANY game move) and DO take alpha/N: one game at p=0.02 among 5
    # does not clear alpha/5=0.01, so it cannot alone make the row "local"; with the others null it is null.
    cells = [_cell("g0", +0.006, 0.02)] + [_cell(f"g{i}", +0.001, 0.9) for i in range(1, 5)]
    v = row_verdict(cells)
    assert not v["cells"][0]["significant"]          # 0.02 > alpha/5 = 0.01
    assert v["status"] == "null"


def test_a_change_significant_in_OPPOSITE_directions_is_CONTESTED():
    cells = [_cell("connect4", +0.20, 0.0001), _cell("othello", -0.18, 0.0002)]
    v = row_verdict(cells)
    assert v["status"] == "contested"        # helps one game, hurts another -> not a transferable lever


def test_a_change_that_moves_only_SOME_games_is_LOCAL():
    cells = [_cell("connect4", +0.20, 0.0001), _cell("othello", +0.01, 0.6)]  # 2nd not significant, tiny
    v = row_verdict(cells)
    assert v["status"] == "local" and v["direction"] == "+"
    assert v["n_significant"] == 1 and v["n_games"] == 2


def test_a_change_absent_everywhere_is_NULL():
    # NULL requires the whole CI inside the null band (provably negligible), not just a small point estimate
    cells = [_cell("connect4", +0.005, 0.7, half=0.015), _cell("othello", -0.004, 0.9, half=0.015)]
    assert row_verdict(cells)["status"] == "null"


def test_a_small_point_estimate_with_a_WIDE_ci_is_INCONCLUSIVE_not_null():
    # the absence-of-evidence fallacy: diff ~ 0 but the interval is wide (underpowered) is NOT proven absence
    cells = [_cell("connect4", +0.01, 0.9, half=0.35), _cell("othello", 0.0, 1.0, half=0.35)]
    assert row_verdict(cells)["status"] == "inconclusive"


def test_a_big_but_underpowered_change_is_INCONCLUSIVE_not_null():
    cells = [_cell("connect4", +0.09, 0.2), _cell("othello", +0.08, 0.3)]   # big diffs, no power
    assert row_verdict(cells)["status"] == "inconclusive"


def test_sign_is_read_only_from_SIGNIFICANT_cells():
    # a non-significant cell with an opposite sign must NOT make the row contested
    cells = [_cell("connect4", +0.20, 0.0001), _cell("othello", -0.20, 0.5)]  # 2nd not significant
    v = row_verdict(cells)
    assert v["status"] == "local" and v["direction"] == "+"


def test_an_empty_row_is_untested():
    assert row_verdict([])["status"] == "untested"


def test_a_single_game_row_is_NOT_generalises():
    # one game cannot evidence generalisation — that is a relabelled single-game A/B, the thing the matrix abolishes
    v = row_verdict([_cell("connect4", +0.20, 0.0001)])
    assert v["status"] == "single_game" and v["sub_status"] == "effect" and v["direction"] == "+"


def test_caveated_cells_are_flagged_in_the_row():
    cells = [_cell("connect4", +0.2, 0.0001, caveats={"deployment": "HOME BUDGET ..."}),
             _cell("othello", +0.15, 0.0002)]
    v = row_verdict(cells)
    assert v["caveated_games"] == ["connect4"]


def test_build_matrix_groups_rows_and_columns_and_renders():
    rows = {
        "wider_net": [_cell("connect4", +0.20, 0.0001), _cell("othello", +0.15, 0.0002)],
        "value_form": [_cell("connect4", +0.005, 0.8, half=0.01), _cell("othello", -0.01, 0.7, half=0.01)],
    }
    m = build_matrix(rows)
    assert m["games"] == ["connect4", "othello"]          # sorted, stable column order
    assert m["rows"]["wider_net"]["status"] == "generalises"
    assert m["rows"]["value_form"]["status"] == "null"
    txt = render_matrix(m)
    assert "wider_net" in txt and "connect4" in txt and "generalises" in txt.lower()


def test_render_marks_significant_cells_distinctly_from_null_ones():
    rows = {"lever": [_cell("connect4", +0.20, 0.0001), _cell("othello", +0.004, 0.9)]}
    txt = render_matrix(build_matrix(rows))
    # the significant +0.20 and the null +0.004 must not render identically
    assert "+0.20" in txt and "+0.00" in txt


def test_a_non_significant_cell_abstains_with_sign_zero():
    # the per-cell `sign` is a directional VOTE; a cell that does not clear the bar must abstain, not vote its
    # noisy diff sign — otherwise a downstream reader miscounts the direction.
    v = row_verdict([_cell("a", +0.20, 0.0001), _cell("b", -0.20, 0.5)])   # b not significant
    by = {c["game"]: c for c in v["cells"]}
    assert by["a"]["sign"] == "+" and by["a"]["significant"]
    assert by["b"]["sign"] == "0" and not by["b"]["significant"]


def test_p_exactly_at_the_corrected_bar_is_significant():
    # 2 games -> corrected alpha 0.025; a cell at exactly 0.025 must count (<=), the boundary a `<` would drop
    v = row_verdict([_cell("a", +0.20, 0.025), _cell("b", +0.20, 0.001)])
    assert all(c["significant"] for c in v["cells"])
    assert v["status"] == "generalises"


def test_cell_from_compare_reads_diff_p_and_caveats_from_a_ledger_result():
    from harness.matrix import cell_from_compare
    res = {"rate_a": 0.90, "rate_b": 0.78, "p": 0.0012, "only_a": 40, "only_b": 10,
           "deployment_warning": "HOME BUDGET ...", "provenance_warning": "", "code_warning": "",
           "completeness_warning": "", "budget_note": ""}
    cell = cell_from_compare(res, "connect4", n=256)
    assert cell["game"] == "connect4"
    assert abs(cell["diff"] - 0.12) < 1e-9 and cell["p"] == 0.0012
    assert cell["ci"][0] < cell["diff"] < cell["ci"][1]      # a real interval around the diff
    assert cell["caveats"] == {"deployment": "HOME BUDGET ..."}


def test_cell_from_compare_carries_a_clean_result_with_no_caveats():
    from harness.matrix import cell_from_compare
    res = {"rate_a": 0.6, "rate_b": 0.6, "p": 1.0, "only_a": 5, "only_b": 5,
           "deployment_warning": "", "provenance_warning": "", "code_warning": "",
           "completeness_warning": "", "budget_note": ""}
    assert cell_from_compare(res, "othello", n=128)["caveats"] == {}


def test_opposite_direction_cells_are_NOT_generalises_even_when_all_clear_the_IUT_bar():
    # two games move in OPPOSITE directions, each p=0.03 (IUT-significant at 0.05 but not union-significant at
    # 0.025): "generalises" requires ONE direction, so this must NOT read generalises. With neither at the strict
    # bar it is not contested either — the honest verdict is inconclusive (weak conflicting evidence).
    cells = [_cell("connect4", +0.10, 0.03), _cell("othello", -0.10, 0.03)]
    v = row_verdict(cells)
    assert all(c["iut_significant"] for c in v["cells"]) and not any(c["significant"] for c in v["cells"])
    assert v["status"] == "inconclusive"     # NOT generalises (opposite signs), NOT contested (not union-sig)


# ---- seed-replication cells (the training-run is the unit, not the opening) ----

def test_a_single_seed_cell_is_never_significant_and_carries_a_standing_caveat():
    from harness.matrix import cell_from_seed_deltas
    c = cell_from_seed_deltas("connect4", [+0.25])       # one training draw
    assert c["p"] == 1.0 and c["n_seeds"] == 1
    assert "single training draw" in c["caveats"]["provenance"]


def test_consistent_seed_deltas_across_runs_are_significant():
    from harness.matrix import cell_from_seed_deltas
    c = cell_from_seed_deltas("connect4", [+0.20, +0.18, +0.22, +0.19])   # tight, all positive
    assert c["diff"] > 0 and c["p"] < 0.01 and c["ci"][0] > 0            # CI excludes zero
    assert c["caveats"] == {}


def test_noisy_seed_deltas_that_straddle_zero_are_not_significant():
    from harness.matrix import cell_from_seed_deltas
    c = cell_from_seed_deltas("othello", [+0.20, -0.15, +0.05, -0.10])   # big spread, mean near zero
    assert c["p"] > 0.05 and c["ci"][0] < 0 < c["ci"][1]


def test_seed_cells_feed_the_row_verdict_correctly():
    from harness.matrix import cell_from_seed_deltas, row_verdict
    # a lever consistent across seeds in BOTH games -> generalises; the seed test is what earns it
    cells = [cell_from_seed_deltas("connect4", [+0.20, +0.18, +0.22]),
             cell_from_seed_deltas("othello", [+0.15, +0.17, +0.14])]
    assert row_verdict(cells)["status"] == "generalises"
    # single-seed cells can never generalise (each p=1.0)
    singles = [cell_from_seed_deltas("connect4", [+0.20]), cell_from_seed_deltas("othello", [+0.15])]
    assert row_verdict(singles)["status"] in ("inconclusive", "null")


def test_seed_cell_matches_the_exact_sample_t_formula():
    # pins sample sd (ddof=1), SE = sd/sqrt(r), and df = r-1 to their exact values, so a switch to population sd,
    # df=r, or an unshrunk SE moves p/CI and fails here.
    import math
    import statistics

    from harness.matrix import cell_from_seed_deltas
    from harness.measurement import student_t_two_sided_p, t_critical

    deltas = [0.10, 0.20, 0.30]
    r = len(deltas)
    mean = statistics.fmean(deltas)
    se = statistics.stdev(deltas) / math.sqrt(r)          # sample sd, shrunk by sqrt(r)
    exp_p = student_t_two_sided_p(mean / se, r - 1)        # df = r - 1
    exp_half = t_critical(r - 1, 0.05) * se
    c = cell_from_seed_deltas("g", deltas)
    assert c["diff"] == pytest.approx(mean, abs=1e-12)
    assert c["p"] == pytest.approx(exp_p, abs=1e-9)
    assert c["ci"][0] == pytest.approx(mean - exp_half, abs=1e-9)
    assert c["ci"][1] == pytest.approx(mean + exp_half, abs=1e-9)
    # sanity: the wrong formulae give a materially DIFFERENT p, so this test actually discriminates
    p_pop = student_t_two_sided_p(mean / (statistics.pstdev(deltas) / math.sqrt(r)), r - 1)   # population sd
    p_dfr = student_t_two_sided_p(mean / se, r)                                               # df = r
    p_nose = student_t_two_sided_p(mean / statistics.stdev(deltas), r - 1)                    # SE unshrunk
    assert abs(p_pop - exp_p) > 1e-4 and abs(p_dfr - exp_p) > 1e-4 and abs(p_nose - exp_p) > 1e-4


def test_is_saturated_flags_a_ceiling_bound_cell():
    from harness.matrix import is_saturated
    # morris on the first real sweep: all rates 0.969..1.0 — both arms never lose, so the cell cannot discriminate
    assert is_saturated([0.984, 1.0, 1.0, 0.984, 0.969, 0.984]) is True     # ceiling
    assert is_saturated([0.0, 0.016, 0.0, 0.031]) is True                    # floor
    assert is_saturated([0.14, 0.25, 0.11, 0.19]) is False                   # mid-range -> can discriminate
    assert is_saturated([]) is False
