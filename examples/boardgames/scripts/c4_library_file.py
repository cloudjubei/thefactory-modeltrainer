"""§3.6 — a run's starting library, written as evidence for its SPEC's `library_from`. scripts/c4_strategy_s2.py
starts every build from the found_maps of that file's roots, so a SPEC that builds from maps no one run found
(harness.floor_g3: every map G2's whole game holds) has them written here first: one root holding, in order, what
the SPEC module's `--library` function makes of its LIBRARY_SOURCES. The judge recomputes the library from the
sources and refuses a run whose builds did not start from it, so this file is a convenience, never a proof.

    PYTHONPATH=. .venv/bin/python scripts/c4_library_file.py --spec harness.floor_g3 --library g3_library"""
from __future__ import annotations

import argparse
import importlib
from datetime import datetime, timezone
from pathlib import Path

from harness.evidence import load_evidence, save_evidence

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="the judge whose SPEC names the library file and its sources")
    ap.add_argument("--library", required=True, help="the judge's function making the library from the sources")
    args = ap.parse_args()
    module = importlib.import_module(args.spec)
    out = ROOT / "evidence" / module.SPEC["library_from"]
    if out.exists():
        raise SystemExit(f"{out.name} exists — a run may already have started from it")
    sources = [load_evidence(ROOT / "evidence" / name) for name in module.LIBRARY_SOURCES]
    library = getattr(module, args.library)(*sources)
    save_evidence(out, {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "spec": args.spec,
                        "library": args.library, "sources": list(module.LIBRARY_SOURCES),
                        "roots": [{"found_maps": [[levels, empty] for levels, empty in library]}]})
    print(f"{out.name}: {len(library)} maps from {', '.join(module.LIBRARY_SOURCES)}", flush=True)


if __name__ == "__main__":
    main()
