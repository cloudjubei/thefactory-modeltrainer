"""Suite-wide safety rails.

The exact Connect-4 solver is milliseconds on late positions and MINUTES-TO-HOURS near the opening. A test that
accidentally triggers the latter hangs the whole suite, and a hung suite looks exactly like a slow one — I hit
this three separate times (the LBR screen test, then the randomized-oracle tests). Rather than rely on reviewer
attention, the solver is given a depth limit for the duration of the test session so the wall fails FAST and
names the fix. A test that genuinely needs a deep solve opts in with @pytest.mark.allow_deep_solve.
"""
import pytest

from harness import solver

TEST_MAX_SOLVE_EMPTIES = 30  # mid-game solves are fine; the measured wall in this suite is 34+ empties


def pytest_configure(config):
    config.addinivalue_line("markers", "allow_deep_solve: permit an exact solve near the opening (slow)")


@pytest.fixture(autouse=True)
def _guard_opening_wall(request):
    previous = solver.MAX_SOLVE_EMPTIES
    solver.MAX_SOLVE_EMPTIES = 0 if request.node.get_closest_marker("allow_deep_solve") else TEST_MAX_SOLVE_EMPTIES
    yield
    solver.MAX_SOLVE_EMPTIES = previous
