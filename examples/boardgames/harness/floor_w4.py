"""§3.6 W4 — W3 again with one change: the leaves walked in C (harness.native_leaf, solving only the moves that decide,
h210) instead of Python. Same two frontier positions (#316, #230), the same library (every map W2 found) and the same
6-hour cap. Judged against W3 on the same positions: #316 finishing in at most half W3's time, and #230 — unfinished
in W3 — finishing. Strategy sizes are reported, not claimed: a faster walk lets the time-bounded searches try more
maps, so sizes may differ."""
from __future__ import annotations

from harness.floor_w3 import SPEC as W3, w3_report

SPEC = {**W3, "measurement_fp": "c7705789fd85", "search": {**W3["search"], "walker": "native"}, "time_support": 0.5}

w4_projection = w3_report


def _integrity(e: dict, spec: dict, w3: dict) -> list[str]:
    problems = []
    if e.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement {e.get('measurement_fingerprint')} is not the registered {spec['measurement_fp']}")
    if e.get("builder") != spec["builder"] or e.get("search") != spec["search"]:
        problems.append("the strategies were built with other settings than registered")
    before = {r["index"]: r for r in w3["roots"]}
    if sorted(r["index"] for r in e["roots"]) != sorted(spec["indices"]):
        problems.append(f"roots {[r['index'] for r in e['roots']]} are not the registered {spec['indices']}")
    if any(r["index"] not in before or r["board"] != before[r["index"]]["board"]
           or r["shared"] != before[r["index"]]["shared"] for r in e["roots"]):
        problems.append("a root is not W3's position, or did not start from W3's library")
    if not all(r["checked"] and r["moves_win"] and r["leaves_certified"] for r in e["roots"] if r["complete"]):
        problems.append("a completed strategy failed an independent check")
    return problems


def w4_report(e: dict, spec: dict, w3: dict) -> dict:
    integrity = _integrity(e, spec, w3)
    if integrity:
        return {"integrity": integrity, "time": {"verdict": "not_run"}, "unfinished": {"verdict": "not_run"}}
    now = {r["index"]: r for r in e["roots"]}
    before = {r["index"]: r for r in w3["roots"]}
    done_before = [i for i in now if before[i]["complete"]]
    not_before = [i for i in now if not before[i]["complete"]]
    ratios = {i: now[i]["build_seconds"] / before[i]["build_seconds"] for i in done_before if now[i]["complete"]}
    if not done_before:
        time = {"verdict": "not_run"}
    elif len(ratios) < len(done_before) or any(r > 1 for r in ratios.values()):
        time = {"verdict": "refuted", "ratios": ratios}
    else:
        time = {"verdict": "supported" if all(r <= spec["time_support"] for r in ratios.values()) else "inconclusive",
                "ratios": ratios}
    finished = sum(1 for i in not_before if now[i]["complete"])
    unfinished = ({"verdict": "not_run"} if not not_before else
                  {"verdict": "supported" if finished == len(not_before) else "refuted" if finished == 0
                   else "inconclusive", "finished": finished})
    sizes = {i: {"w4": now[i]["bits"]["nodes"], "w3": before[i]["bits"]["nodes"]} for i in now}
    return {"integrity": [], "time": time, "unfinished": unfinished, "sizes": sizes}
