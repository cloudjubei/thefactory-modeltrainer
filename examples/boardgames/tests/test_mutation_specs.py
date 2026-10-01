"""Every tracked mutation spec under tests/mutations/ (run by scripts/mutate.py --spec) is a guard proof that can be
re-run: its module and test targets exist, every mutation still applies to exactly one place in the module (a spec
whose code has moved on is stale and must be refreshed, not silently skipped), and its last recorded run killed
every mutation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPECS = sorted((ROOT / "tests" / "mutations").glob("*.json"))


def test_there_are_tracked_mutation_specs():
    assert SPECS


@pytest.mark.parametrize("spec", SPECS, ids=[p.stem for p in SPECS])
def test_a_mutation_spec_still_applies_and_its_last_run_killed_everything(spec):
    s = json.loads(spec.read_text())
    source = (ROOT / s["file"]).read_text()
    for m in s["mutations"]:
        for edit in m.get("edits") or [m]:
            assert source.count(edit["old"]) == 1, f"{m['name']}: no longer applies to exactly one place"
    for target in s["tests"]:
        path, _sep, node = target.partition("::")
        assert (ROOT / path).exists(), target
        assert not node or f"def {node.split('[')[0]}(" in (ROOT / path).read_text(), target
    run = s.get("last_run")
    assert run is not None, "never run: python scripts/mutate.py --spec " + str(spec.relative_to(ROOT))
    assert run["survived"] == [] and run["killed"] == len(s["mutations"])
