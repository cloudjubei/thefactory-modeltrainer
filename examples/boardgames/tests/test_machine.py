"""Direct tests for harness/machine.py — whether a long run is actually protected from sleep.

2026-09-19: arm B lost ~20 h of wall-clock to 169 sleep episodes while `caffeinate -is` was alive the whole
time. `caffeinate -s` is documented as valid ONLY on AC power, so on battery the guard was inert. I had
verified the guard was RUNNING (a proxy) instead of verifying that sleep was PREVENTED (the property)."""
from __future__ import annotations

from harness.machine import power_source, sleep_guard_effective, slept_seconds_since

AC = "Now drawing from 'AC Power'\n -InternalBattery-0	93%; charging"
BATT = "Now drawing from 'Battery Power'\n -InternalBattery-0	95%; discharging"
HELD = "   PreventUserIdleSystemSleep     1\n   PreventSystemSleep             1\n"
NONE = "   PreventUserIdleSystemSleep     0\n   PreventSystemSleep             0\n"


def _fake(batt, assertions, log=""):
    def run(cmd):
        if "batt" in cmd:
            return batt
        if "assertions" in cmd:
            return assertions
        return log
    return run


def test_power_source_reads_ac_and_battery():
    assert power_source(run=_fake(AC, HELD)) == "ac"
    assert power_source(run=_fake(BATT, HELD)) == "battery"


def test_the_guard_is_NOT_effective_on_battery_even_with_the_assertion_held():
    """The exact failure: the assertion was held for five days and the machine slept anyway."""
    ok, why = sleep_guard_effective(run=_fake(BATT, HELD))
    assert ok is False
    assert "battery" in why.lower() and "ac" in why.lower()


def test_the_guard_is_effective_on_ac_with_the_assertion_held():
    ok, why = sleep_guard_effective(run=_fake(AC, HELD))
    assert ok is True and why == ""


def test_the_guard_is_not_effective_without_the_assertion_even_on_ac():
    ok, why = sleep_guard_effective(run=_fake(AC, NONE))
    assert ok is False and "assertion" in why.lower()


def test_slept_seconds_since_totals_only_episodes_after_the_timestamp():
    log = (
        "2026-09-17 10:00:00 +0200 Sleep   Entering Sleep state due to 'Maintenance Sleep': 100 secs\n"
        "2026-09-18 20:00:00 +0200 Sleep   Entering Sleep state due to 'Maintenance Sleep': 600 secs\n"
        "2026-09-18 22:00:00 +0200 Sleep   Entering Sleep state due to 'Maintenance Sleep': 300 secs\n"
        "2026-09-18 23:00:00 +0200 Wake    Wake from Deep Idle\n"
    )
    assert slept_seconds_since("2026-09-18", run=_fake(AC, HELD, log)) == 900
    assert slept_seconds_since("2026-09-17", run=_fake(AC, HELD, log)) == 1000


def test_slept_seconds_is_zero_when_the_log_shows_no_sleep():
    assert slept_seconds_since("2026-09-18", run=_fake(AC, HELD, "nothing here\n")) == 0
