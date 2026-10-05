"""The published Kalah results games/kalah.py is checked against — Irving, Donkers & Uiterwijk (2000), "Solving
Kalah", ICGA Journal 23(3):139-147 — and the judge that reads a check (scripts/kalah_paper_check.py).

Table 9 gives, for Kalah(4..6, n), the first player's margin under perfect play and one perfect game: per turn the
holes played (numbered from 0 on each side in sowing order), a dash where the turn passes. Table 10 gives win/draw/loss
for every m, n <= 6 but (6, 6). Two claims are read: the solver's values agree with both tables on every registered
shape; and the perfect games replay to their stated margins under the paper's stated rules, better than under any
other reading of them, with the one line that does not (4(6)) breaking where registered and no one-character edit
repairing it."""
from __future__ import annotations

PAPER = "Irving, Donkers & Uiterwijk (2000), Solving Kalah, ICGA Journal 23(3):139-147"

PERFECT_GAMES = {
    (4, 1): (2, "32-32-1"),
    (4, 2): (6, "3-3-230-231-1323"),
    (4, 3): (8, "13-1-032-2-3-0-1320-3-0-2"),
    (4, 4): (2, "02-2-3-0-2-21-303231-0-32-1-0"),
    (4, 5): (2, "0-2-1-3-0-2-1-0-32-3-021-2-0-02-2-1-3-3-313032-13031-1-2-2-3"),
    (4, 6): (0, "1-0-2-3-1330-01-2-2-21-201-2-23-3-23-230-32-32-12-1-0-32"),
    (5, 1): (0, "43-43-2-2"),
    (5, 2): (0, "31-32-24-0-0-31-3"),
    (5, 3): (8, "24-3-3-03-41-1-04342-3-3-2-2-4-414342-3"),
    (5, 4): (12, "12-04-03-2-20-41-1-42-32-3-2-40-0-1"),
    (5, 5): (2, "03-2-2-1-1-23-24-2-0-34414340-4240-2-14342"),
    (5, 6): (2, "2-0-0-3-3-0-2-1-440-4-42-2-121-12-3-1-4-20-0-34-24-324-1-3-3-1-41-43-2-2-3"),
    (6, 1): (2, "54-54-3-3-2"),
    (6, 2): (10, "42-42-30-0-1-1-4-5"),
    (6, 3): (2, "4-5-35-250-2-154-451535452-53-3-54-2"),
    (6, 4): (10, "25-10-3-3-5153-1-4-5-045-4-535452-53-4-1-3-2-0-54-1-3"),
    (6, 5): (12, "12-02-05-2-4-51-53-3-45-20-3-2-2-345-5-4-351-0-54-1-52-354-4-254-3-3"),
}

TABLE_10 = {1: "DLWLWD", 2: "WLLLWW", 3: "DWWWWL", 4: "WWWWWD", 5: "DDWWWW", 6: "WWWWW"}

SPEC = {
    "measurement_fp": "5b474aa32b1c",
    "solve": [[m, n] for m in (1, 2, 3) for n in range(1, 7)] + [[4, 1], [4, 2], [4, 3], [5, 1], [5, 2], [6, 1]],
    "variants": ["paper", "skip_start", "nonempty_capture", "both"],
    "inconsistent": {"4,6": [9, 1]},
}


def _integrity(e: dict, spec: dict) -> list[str]:
    problems = []
    if e.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement {e.get('measurement_fingerprint')} is not the registered {spec['measurement_fp']}")
    solved = sorted([r["holes"], r["counters"]] for r in e["solved"])
    if solved != sorted(spec["solve"]):
        problems.append(f"solved shapes {solved} are not the registered {sorted(spec['solve'])}")
    for variant in spec["variants"]:
        rows = e["replays"].get(variant)
        if rows is None or sorted((r["holes"], r["counters"]) for r in rows) != sorted(PERFECT_GAMES):
            problems.append(f"rule reading {variant} did not replay every perfect game")
    return problems


def _consistent(row: dict) -> bool:
    return row["result"] == "end" and row["margin"] == PERFECT_GAMES[(row["holes"], row["counters"])][0]


def paper_report(e: dict, spec: dict) -> dict:
    integrity = _integrity(e, spec)
    if integrity:
        return {"integrity": integrity, "values": {"verdict": "not_run"}, "replays": {"verdict": "not_run"}}
    disagreements = []
    for r in sorted(e["solved"], key=lambda r: (r["holes"], r["counters"])):
        shape, margin = (r["holes"], r["counters"]), r["margin"]
        wdl = TABLE_10[shape[0]][shape[1] - 1]
        if "LDW"[(margin > 0) - (margin < 0) + 1] != wdl:
            disagreements.append({"shape": list(shape), "ours": margin, "paper": wdl})
        elif shape in PERFECT_GAMES and margin != PERFECT_GAMES[shape][0]:
            disagreements.append({"shape": list(shape), "ours": margin, "paper": PERFECT_GAMES[shape][0]})
    consistent = {v: sum(1 for r in e["replays"][v] if _consistent(r)) for v in spec["variants"]}
    broken = {f"{r['holes']},{r['counters']}": [r["turn"], r["move"]] for r in e["replays"]["paper"]
              if not _consistent(r)}
    best_alone = all(consistent[v] < consistent["paper"] for v in spec["variants"] if v != "paper")
    replays_hold = broken == spec["inconsistent"] and best_alone and not e["repairs"]
    return {"integrity": [],
            "values": {"verdict": "refuted" if disagreements else "supported", "disagreements": disagreements},
            "replays": {"verdict": "supported" if replays_hold else "refuted", "consistent": consistent,
                        "broken": broken, "repairs": e["repairs"]}}
