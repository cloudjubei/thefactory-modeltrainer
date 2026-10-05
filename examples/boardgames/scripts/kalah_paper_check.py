"""Check games/kalah.py against Irving, Donkers & Uiterwijk (2000): solve every registered Kalah(m, n) from the start
and replay the paper's 17 perfect games under its stated rules (through games/kalah.py itself) and under the three
other readings of them — the start hole skipped on a lap, a capture only when the opposite hole holds counters, and
both — then try every one-character edit of any line that breaks. Shapes are solved one at a time, each memo dropped
before the next: Kalah(6, 2) and up need more than 10 GB, so only shapes whose memo was measured are registered.
Judged by harness.kalah_paper.paper_report.

    PYTHONPATH=. .venv/bin/python scripts/kalah_paper_check.py --out evidence/kalah_paper_check.json.gz
"""
from __future__ import annotations

import argparse
import platform
import resource
import sys
import time
from datetime import datetime, timezone

MEASUREMENT_MODULES = ("games/kalah.py", "scripts/kalah_paper_check.py")


def _sow(own, opp, hole, skip_start, empty_capture):
    m = len(own)
    ring = list(own) + [0] + list(opp)
    seeds, ring[hole], pos = ring[hole], 0, hole
    while seeds:
        pos = (pos + 1) % len(ring)
        if skip_start and pos == hole:
            continue
        ring[pos] += 1
        seeds -= 1
    if pos < m and ring[pos] == 1 and (empty_capture or ring[2 * m - pos]):
        captured = 1 + ring[2 * m - pos]
        ring[pos] = ring[2 * m - pos] = 0
        ring[m] += captured
    return ring[:m], ring[m + 1:], ring[m], pos == m


def _outcome(result, margin=None, turn=None, move=None):
    return {"result": result, "margin": margin, "turn": turn, "move": move}


def _replay_reading(m, n, line, skip_start, empty_capture):
    sides, stores, me = [[n] * m, [n] * m], [0, 0], 0
    turns = line.split("-")
    for k, turn in enumerate(turns):
        for i, h in enumerate(turn):
            hole = int(h)
            if hole >= m or not sides[me][hole]:
                return _outcome("empty hole", turn=k, move=i)
            own, opp, stored, again = _sow(sides[me], sides[1 - me], hole, skip_start, empty_capture)
            sides[me], sides[1 - me] = own, opp
            stores[me] += stored
            last = k == len(turns) - 1 and i == len(turn) - 1
            if not any(own) or not any(opp):
                stores[me] += sum(own)
                stores[1 - me] += sum(opp)
                return _outcome("end", stores[0] - stores[1]) if last else _outcome("ended early", turn=k, move=i)
            if again == (i == len(turn) - 1):
                return _outcome("extra-move mismatch", turn=k, move=i)
        me = 1 - me
    return _outcome("never ended")


def _replay_module(game, line):
    s = game.initial_state()
    turns = line.split("-")
    for k, turn in enumerate(turns):
        mover = game.current_player(s)
        for i, h in enumerate(turn):
            if int(h) not in game.legal_actions(s):
                return _outcome("empty hole", turn=k, move=i)
            s = game.step(s, int(h))
            last = k == len(turns) - 1 and i == len(turn) - 1
            if game.is_terminal(s):
                return _outcome("end", s.stores[0] - s.stores[1]) if last else _outcome("ended early", turn=k, move=i)
            if (game.current_player(s) == mover) == (i == len(turn) - 1):
                return _outcome("extra-move mismatch", turn=k, move=i)
    return _outcome("never ended")


def _edits(line, digits):
    out = set()
    for i in range(len(line) + 1):
        for c in digits + "-":
            out.add(line[:i] + c + line[i:])
            if i < len(line):
                out.add(line[:i] + c + line[i + 1:])
        if i < len(line):
            out.add(line[:i] + line[i + 1:])
    out.discard(line)
    return sorted(out)


def main() -> None:
    from games.kalah import Kalah
    from harness.evidence import save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.kalah_paper import PAPER, PERFECT_GAMES, SPEC

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.setrecursionlimit(10000)
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    readings = {"paper": (False, True), "skip_start": (True, True), "nonempty_capture": (False, False),
                "both": (True, False)}
    replays, repairs = {v: [] for v in SPEC["variants"]}, []
    for (m, n), (margin, line) in sorted(PERFECT_GAMES.items()):
        game = Kalah(m, n)
        ours = _replay_module(game, line)
        for variant in SPEC["variants"]:
            row = ours if variant == "paper" else _replay_reading(m, n, line, *readings[variant])
            replays[variant].append({"holes": m, "counters": n, **row})
        if _replay_reading(m, n, line, *readings["paper"]) != ours:
            raise SystemExit(f"Kalah({m}, {n}): the script's reading of the paper's rules is not games/kalah.py's")
        if not (ours["result"] == "end" and ours["margin"] == margin):
            fixed = [e for e in _edits(line, "".join(str(h) for h in range(m)))
                     if _replay_module(game, e) == _outcome("end", margin)]
            repairs.extend(fixed)
            print(f"Kalah({m}, {n}) line breaks: {ours}; one-character repairs {fixed}", flush=True)
    solved = []
    for m, n in SPEC["solve"]:
        game = Kalah(m, n)
        t = time.perf_counter()
        margin = game.margin(game.initial_state())
        seconds = round(time.perf_counter() - t, 2)
        positions = len(game._future)
        del game
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        solved.append({"holes": m, "counters": n, "margin": margin, "positions": positions, "seconds": seconds})
        print(f"Kalah({m}, {n}): margin {margin}, {positions} positions, {seconds} s, peak rss {rss}", flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the check ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "paper": PAPER,
                             "versions": {"python": platform.python_version()},
                             "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                             "solved": solved, "replays": replays, "repairs": repairs})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
