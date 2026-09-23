"""Direct tests for the Student's-t primitives in harness.measurement — the between-run test a cross-game
seed-replication cell needs (§C.20). Checked against published t-table values."""
from __future__ import annotations

import pytest

from harness.measurement import student_t_two_sided_p, t_critical


def test_two_sided_p_matches_the_t_table():
    # classic table points: t=2.776 at df=4 is the 5% two-sided critical value
    assert student_t_two_sided_p(2.776, 4) == pytest.approx(0.05, abs=1e-3)
    assert student_t_two_sided_p(3.182, 3) == pytest.approx(0.05, abs=1e-3)
    assert student_t_two_sided_p(2.228, 10) == pytest.approx(0.05, abs=1e-3)
    assert student_t_two_sided_p(1.0, 1) == pytest.approx(0.5, abs=1e-3)     # Cauchy: |T|>1 has prob 1/2


def test_large_df_approaches_the_normal():
    assert student_t_two_sided_p(1.96, 100000) == pytest.approx(0.05, abs=2e-3)


def test_tail_is_monotone_and_bounded():
    assert student_t_two_sided_p(0.0, 5) == 1.0
    assert student_t_two_sided_p(1000.0, 5) < 1e-6
    assert student_t_two_sided_p(1.0, 5) > student_t_two_sided_p(2.0, 5)     # bigger t -> smaller p


def test_t_critical_inverts_the_tail():
    for df in (2, 4, 8, 30):
        tc = t_critical(df, 0.05)
        assert student_t_two_sided_p(tc, df) == pytest.approx(0.05, abs=1e-4)
    assert t_critical(4, 0.05) == pytest.approx(2.776, abs=1e-2)
