"""§C.32 MUTATION-TEST A GUARD FROM THE COMMAND LINE — prove a guard is load-bearing, don't assert it.

A guard with a green test beside it proves nothing: the test may pass with the guard deleted. The rule on this
track has been "mutation-test the guards", carried out by hand — and on 2026-09-20 the hand method lied twice
in ten minutes (stale bytecode inverted a 46-pass suite; a mutation went inert and killed the wrong test).
harness.mutation makes the correct procedure the default; this is its front door.

    cat > /tmp/muts.json <<'EOF'
    [{"name": "guard removed", "old": "if prior is not None", "new": "if False"},
     {"name": "guard runs too late", "edits": [{"old": "    check()\n", "new": ""},
                                               {"old": "    write()\n", "new": "    write()\n    check()\n"}]}]
    EOF
    PYTHONPATH=. .venv/bin/python scripts/mutate.py --file harness/ledger.py \
        --tests tests/test_ledger.py --mutations /tmp/muts.json

A tracked spec under tests/mutations/ ({"file", "tests", "mutations"}) runs with --spec alone, and the outcome is
written back into it under "last_run", so every guard proof stays re-runnable and its last verdict is on record
(tests/test_mutation_specs.py fails when a spec's code has moved on without it):

    PYTHONPATH=. .venv/bin/python scripts/mutate.py --spec tests/mutations/playbook.json

Exits non-zero if ANY mutation survives, so it can gate a change rather than merely inform one. Read the
`by:` column, not just the verdict — a mutation killed by a test unrelated to the property it breaks means the
guard is still unproven (M-D)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from harness.mutation import mutate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", help="a tracked spec: {file, tests, mutations}; the outcome is recorded into it")
    ap.add_argument("--file", help="the module whose guard is under test")
    ap.add_argument("--tests", nargs="+", help="pytest targets that should catch the mutations")
    ap.add_argument("--mutations", help="JSON list of {name, old, new}")
    ap.add_argument("--cwd", default=".")
    ap.add_argument("--timeout", type=float, help="seconds before a hung suite is stopped and read as killing the "
                                                  "mutation; a spec may carry its own `timeout`")
    args = ap.parse_args()

    if args.spec:
        spec = json.loads(Path(args.spec).read_text())
        args.file, args.tests, muts = spec["file"], spec["tests"], spec["mutations"]
        args.timeout = spec.get("timeout", args.timeout)
    elif not (args.file and args.tests and args.mutations):
        raise SystemExit("give --spec, or all of --file, --tests and --mutations")
    else:
        muts = json.loads(Path(args.mutations).read_text())
    if not isinstance(muts, list) or not muts:
        raise SystemExit("--mutations must be a non-empty JSON list of {name, old, new}")
    for m in muts:
        if "name" not in m:
            raise SystemExit(f"mutation {m!r} is missing 'name'")
        if "edits" not in m and {"old", "new"} - set(m):
            raise SystemExit(f"mutation {m['name']!r} needs either 'old' + 'new', or 'edits': a list of "
                             f"{{old, new}} — use 'edits' to RELOCATE a guard (delete here, re-insert there), "
                             f"the mutation class that catches a check which runs but too late")

    res = mutate(args.file, muts, args.tests, cwd=Path(args.cwd), timeout=args.timeout)
    width = max(len(m["name"]) for m in muts)
    for r in res["results"]:
        by = ", ".join(t.split("::")[-1] for t in r["killed_by"]) or "-"
        print(f"{'KILLED  ' if r['killed'] else 'SURVIVED'} {r['name']:{width}s}  by: {by}")
    if args.spec:
        spec["last_run"] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "killed": sum(1 for r in res["results"] if r["killed"]), "survived": res["survived"],
                            "killed_by": {r["name"]: sorted(t.split("::")[-1] for t in r["killed_by"])
                                          for r in res["results"]}}
        Path(args.spec).write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n")
    if res["survived"]:
        print(f"\n{len(res['survived'])} mutation(s) SURVIVED: {res['survived']} — the guard is not proven; "
              f"the suite passes with it broken.")
        return 1
    print(f"\nall {len(res['results'])} mutation(s) killed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
