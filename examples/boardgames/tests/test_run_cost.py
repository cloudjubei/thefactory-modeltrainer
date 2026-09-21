"""Direct tests for harness.machine.run_cost — what a run actually COST, read off its own log.

Written 2026-09-21 after I reported arm B's cost as "95.8 h" three times. The real figures were 142.69 h wall
and 127.16 h CPU, and the gap between them was SLEEP: arm B's run slept mid-flight, so its wall-clock
over-stated its cost while its CPU time did not. `sleep_guard_effective()` only ever checked the moment of
launch, and `slept_seconds_since` was never called again, so nothing connected the two.

The property each test defends is that a cost figure is never quietly wrong: every segment counted, CPU kept
separate from wall, and a run whose CPU/wall ratio falls below its thread count reported as contaminated
rather than returned as a clean number."""
from __future__ import annotations

import pytest

from harness.machine import run_cost

ONE = """=== START run Mon Sep 14 10:01:41 CEST 2026 ===
batch 0 done
real 513684.73
user 372503.21
sys 85269.93
"""

RESUMED = ONE + """=== START run Sun Sep 20 08:43:06 CEST 2026 ===
nothing to do
real 8.09
user 7.69
sys 0.20
=== END run Sun Sep 20 08:43:14 CEST 2026 ===
"""

HEALTHY = """=== START run Fri Sep 11 17:20:02 CEST 2026 ===
real 193318.15
user 200716.75
sys 43022.90
=== END run Sun Sep 13 23:02:00 CEST 2026 ===
"""


def _log(tmp_path, text, name="run.log"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_it_sums_EVERY_segment_not_just_the_last(tmp_path):
    """The exact shape of the error: a resumed run's last segment was 8 seconds of doing nothing."""
    c = run_cost(_log(tmp_path, RESUMED))
    assert c["segments"] == 2
    assert c["wall_h"] == pytest.approx((513684.73 + 8.09) / 3600, abs=1e-6)


def test_cpu_is_reported_SEPARATELY_from_wall(tmp_path):
    c = run_cost(_log(tmp_path, RESUMED))
    assert c["cpu_h"] == pytest.approx((372503.21 + 85269.93 + 7.69 + 0.20) / 3600, abs=1e-6)
    assert c["cpu_h"] < c["wall_h"]


def test_a_run_that_slept_is_flagged_rather_than_returned_as_a_clean_number(tmp_path):
    """cpu/wall below 1 on a multithreaded run means wall-clock ran while the CPU did not."""
    c = run_cost(_log(tmp_path, RESUMED), threads=4)
    assert c["cpu_wall_ratio"] < 1
    assert c["contaminated"] and "sleep" in c["warning"].lower()


def test_a_healthy_multithreaded_run_is_NOT_flagged(tmp_path):
    """Arm A: cpu 67.71 h against wall 53.70 h — ratio 1.26, exactly what 4 threads should look like."""
    c = run_cost(_log(tmp_path, HEALTHY), threads=4)
    assert c["cpu_wall_ratio"] == pytest.approx(1.26, abs=0.01)
    assert not c["contaminated"] and c["warning"] == ""


def test_the_cost_to_QUOTE_is_cpu_not_wall(tmp_path):
    """A caller that wants one number must get the sleep-immune one, or the guard is decorative."""
    assert run_cost(_log(tmp_path, RESUMED))["cost_h"] == pytest.approx(run_cost(_log(tmp_path, RESUMED))["cpu_h"])


def test_a_log_with_no_timing_block_does_not_report_a_cost_of_zero(tmp_path):
    """Silence must read as UNKNOWN. A zero here would be quoted as a real figure."""
    with pytest.raises(ValueError) as exc:
        run_cost(_log(tmp_path, "=== START run ===\nbatch 0 done\n"))
    assert "no timing" in str(exc.value).lower()


def test_a_segment_missing_part_of_its_triple_is_refused(tmp_path):
    """A half-parsed segment would silently under-count CPU and inflate the apparent sleep loss."""
    with pytest.raises(ValueError) as exc:
        run_cost(_log(tmp_path, "real 100.0\nuser 90.0\nreal 50.0\nuser 40.0\nsys 5.0\n"))
    assert "incomplete" in str(exc.value).lower()
