"""§C.31 IS THE MACHINE ACTUALLY AWAKE — the guard that checks the property, not the process.

WHY (2026-09-19): the Othello efficiency arm lost ~20 h of wall-clock to 169 sleep episodes while
`caffeinate -is` was alive for the whole five days. `caffeinate -s` is valid ONLY on AC power, so on battery
the guard was inert: the assertion was held and the machine slept anyway. I had been checking that the guard
was RUNNING — a proxy — rather than that sleep was PREVENTED, which is the property. The same shape as the
solve-depth guard, the L1 label check and the vacuous pool test before it.

Nothing here is in the training path (`fingerprint.TRAINING_MODULES`), so consulting it cannot move an era."""
from __future__ import annotations

from pathlib import Path

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

def run_cost(log_path, threads: int = 4) -> dict:
    """What a run actually COST, read off the `/usr/bin/time -p` blocks its own log already contains.

    Written because "arm B cost 95.8 h" was quoted three times and was wrong: the run had TWO segments and
    142.69 h of wall against 127.16 h of CPU. The gap was sleep. `sleep_guard_effective()` checks the moment
    of launch only, so a run that begins protected and sleeps six hours later looks identical to one that
    never slept — unless something compares CPU against wall afterwards, which is what this does.

    `cost_h` is the CPU figure deliberately: wall-clock is not a cost when the machine was asleep for part
    of it, and a caller asking for one number must be handed the sleep-immune one."""
    import re

    text = Path(log_path).read_text()
    reals = [float(x) for x in re.findall(r"^real (\d+(?:\.\d+)?)", text, re.M)]
    users = [float(x) for x in re.findall(r"^user (\d+(?:\.\d+)?)", text, re.M)]
    syss = [float(x) for x in re.findall(r"^sys (\d+(?:\.\d+)?)", text, re.M)]
    if not reals:
        raise ValueError(f"{log_path}: no timing block found — a run with no `real`/`user`/`sys` lines has an "
                         f"UNKNOWN cost, which must never be reported as zero")
    if not len(reals) == len(users) == len(syss):
        raise ValueError(f"{log_path}: incomplete timing segments (real={len(reals)}, user={len(users)}, "
                         f"sys={len(syss)}) — a half-parsed segment under-counts CPU and inflates the "
                         f"apparent sleep loss")
    wall, cpu = sum(reals), sum(users) + sum(syss)
    ratio = cpu / wall if wall else 0.0
    contaminated = ratio < 1.0 and threads > 1
    warning = ""
    if contaminated:
        warning = (f"CPU/wall is {ratio:.2f} on a {threads}-thread run: {wall / 3600:.2f} h of wall-clock "
                   f"passed for {cpu / 3600:.2f} h of CPU, so the machine was asleep or idle for part of the "
                   f"run. Quote the CPU figure — wall-clock here is not a cost.")
    return {"segments": len(reals), "wall_h": wall / 3600, "cpu_h": cpu / 3600, "cost_h": cpu / 3600,
            "cpu_wall_ratio": ratio, "contaminated": contaminated, "warning": warning}
