"""Regenerate docs/process-knowledge.md from the claims register and knowledge_map.json (harness/knowledge.py).
Refuses to write when the map leaves a registered claim unfiled or cites one it does not list.

    PYTHONPATH=. .venv/bin/python scripts/knowledge_digest.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    from harness.hypotheses import Register
    from harness.knowledge import check_map, render

    views = Register(ROOT / "hypotheses.json").report()
    kmap = json.loads((ROOT / "knowledge_map.json").read_text())
    problems = check_map(kmap, views)
    if problems:
        raise SystemExit("knowledge_map.json is incomplete:\n  " + "\n  ".join(problems))
    out = ROOT.parent.parent / "docs" / "process-knowledge.md"
    out.write_text(render(kmap, views))
    print(f"wrote {out} ({len(views)} claims, {len(kmap['ingredients'])} ingredients)")


if __name__ == "__main__":
    main()
