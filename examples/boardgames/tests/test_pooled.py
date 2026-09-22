"""Direct tests for harness.measurement.pooled_paired_effect — pooling one comparison read at several settings.

Written 2026-09-22. The 96-vs-32 pair was read at three deployment budgets (32/64/96 sims) on the SAME 1024
roots. Read one at a time they say "significant, significant, NULL", which invites the significant/non-
significant fallacy — concluding the effect depends on the setting when the INTERACTION was never tested. The
interaction is null (z<=1.3), so the honest summary is one pooled effect, and pooling is only legitimate
BECAUSE the roots are shared, which is the precondition this function refuses to proceed without."""
from __future__ import annotations

import pytest

from harness.measurement import pooled_paired_effect


def _arms(diffs):
    """Build per-setting (a, b) outcome pairs with an exact per-root difference pattern."""
    out = []
    for d in diffs:
        n = 100
        a = [1] * n
        b = [1] * (n - d) + [0] * d
        out.append((a, b))
    return out


def test_it_pools_settings_that_share_roots(tmp_path):
    r = pooled_paired_effect(_arms([4, 8, 6]))
    assert r["n_roots"] == 100 and r["settings"] == 3
    assert r["effect"] == pytest.approx((0.04 + 0.08 + 0.06) / 3, abs=1e-9)


def test_it_reports_a_confidence_interval_and_flags_a_lower_bound_inside_the_null_band(tmp_path):
    r = pooled_paired_effect(_arms([4, 8, 6]), null_below=0.10)
    assert r["ci"][0] < r["effect"] < r["ci"][1]
    assert r["ci_reaches_null_band"] is True


def test_a_tight_effect_well_clear_of_the_band_is_not_flagged(tmp_path):
    r = pooled_paired_effect(_arms([50, 50, 50]), null_below=0.03)
    assert r["ci"][0] > 0.03 and r["ci_reaches_null_band"] is False


def test_settings_with_DIFFERENT_root_counts_are_refused(tmp_path):
    """Pooling is only valid because the roots are the same; unequal lengths mean they are not."""
    bad = _arms([4, 8])
    bad[1] = (bad[1][0][:50], bad[1][1][:50])
    with pytest.raises(ValueError) as exc:
        pooled_paired_effect(bad)
    assert "same roots" in str(exc.value).lower()


def test_a_single_setting_is_refused_as_nothing_to_pool(tmp_path):
    with pytest.raises(ValueError) as exc:
        pooled_paired_effect(_arms([4]))
    assert "at least two" in str(exc.value).lower()


def test_the_interaction_across_settings_is_reported_not_assumed(tmp_path):
    """Pooling a real interaction would hide it, so the function must hand back the evidence against itself."""
    wide = pooled_paired_effect(_arms([0, 0, 60]))
    flat = pooled_paired_effect(_arms([5, 6, 5]))
    assert wide["max_interaction_z"] > flat["max_interaction_z"]
    assert flat["poolable"] and not wide["poolable"]
