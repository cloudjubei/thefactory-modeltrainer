"""§3.6 M1 — F1's maps under exact cheaper codes (harness.map_coding): every map a build found (its own maps, not the
library's), coded with each code and with the cheapest plus its selector. The builder's own code must reproduce the
evidence's recorded map bits root by root, or nothing is written. Saved: the totals per code and for the best, how
often each code is the best, and per root the builder's bits and the best.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_map_coding.py --spec harness.floor_m1 --out evidence/c49_m1.json.gz
"""
from __future__ import annotations

import argparse
import importlib
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENT_MODULES = ("scripts/c4_map_coding.py", "harness/map_coding.py", "harness/exception_coding.py",
                       "harness/steady_state.py")


def main() -> None:
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.map_coding import CODES, best_bits, code_bits

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    spec = importlib.import_module(args.spec).SPEC
    n_levels = spec["n_levels"]
    totals = Counter()
    chosen = Counter()
    rows = []
    maps = 0
    for r in load_evidence(ROOT / "evidence" / spec["source"])["roots"]:
        own = [({int(c): k for c, k in r["maps"][i].items()}, r["map_empty"][i])
               for i in range(r["shared"] + 1, len(r["maps"]))]
        builder = sum(code_bits("bitmap", m, e, n_levels) for m, e in own)
        if builder != r["bits"]["map_bits"]:
            raise SystemExit(f"root #{r['index']}: the builder's code gives {builder} map bits, the evidence "
                             f"{r['bits']['map_bits']} — not the evidence's accounting; nothing written")
        best = 0
        for m, e in own:
            for c in CODES:
                totals[c] += code_bits(c, m, e, n_levels)
            code, bits = best_bits(m, e, n_levels)
            chosen[code] += 1
            best += bits
        totals["best"] += best
        maps += len(own)
        rows.append({"index": r["index"], "maps": len(own), "bitmap": builder, "best": best})
    print({**totals, "maps": maps, "chosen": dict(chosen)}, flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the run ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "spec": args.spec, "source": spec["source"], "maps": maps, "totals": dict(totals),
                             "chosen": dict(chosen), "rows": rows})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
