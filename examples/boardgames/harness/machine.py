"""§C.31 IS THE MACHINE ACTUALLY AWAKE — the guard that checks the property, not the process.

WHY (2026-09-19): the Othello efficiency arm lost ~20 h of wall-clock to 169 sleep episodes while
`caffeinate -is` was alive for the whole five days. `caffeinate -s` is valid ONLY on AC power, so on battery
the guard was inert: the assertion was held and the machine slept anyway. I had been checking that the guard
was RUNNING — a proxy — rather than that sleep was PREVENTED, which is the property. The same shape as the
solve-depth guard, the L1 label check and the vacuous pool test before it.

Nothing here is in the training path (`fingerprint.TRAINING_MODULES`), so consulting it cannot move an era."""
from __future__ import annotations

import re
import subprocess


def _run(cmd: str) -> str:
    r = subprocess.run(cmd.split(), capture_output=True, text=True)
    return (r.stdout or "") + (r.stderr or "")


def power_source(run=_run) -> str:
    """'ac' or 'battery' — the fact that decides whether a sleep assertion means anything."""
    return "ac" if "AC Power" in run("pmset -g batt") else "battery"


def sleep_guard_effective(run=_run) -> tuple[bool, str]:
    """Is this machine genuinely prevented from sleeping right now?

    Both conditions are required, and the battery case is the one that actually bit: an assertion alone reads
    as protection while the machine sleeps through a multi-day run."""
    assertions = run("pmset -g assertions")
    held = any(re.search(rf"{name}\s+1", assertions)
               for name in ("PreventUserIdleSystemSleep", "PreventSystemSleep"))
    if not held:
        return False, "no sleep assertion is held — nothing is stopping this machine from sleeping"
    if power_source(run=run) != "ac":
        return False, ("a sleep assertion is held but this machine is on BATTERY, where `caffeinate -s` is "
                       "inert — plug into AC or the run will sleep with the guard apparently in place")
    return True, ""


def slept_seconds_since(since: str, run=_run) -> float:
    """Total seconds spent asleep since `since` (an ISO date prefix), read from the system's own sleep log —
    so a stalled run can be told apart from a slept-through one by evidence rather than by guessing."""
    total = 0.0
    for line in run("pmset -g log").splitlines():
        if "Entering Sleep state" not in line or line[:len(since)] < since:
            continue
        m = re.search(r"(\d+)\s+secs", line)
        if m:
            total += float(m.group(1))
    return total
