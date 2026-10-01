"""Direct tests for harness/knowledge.py — the process-knowledge digest: every registered claim filed under the
process ingredients it is about, each ingredient's conclusion citing the claims it rests on, and the page generated
from the register so it can never drift from the evidence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.knowledge import check_map, render

VIEWS = [{"id": "h1", "claim": "Siblings help.", "status": "supported"},
         {"id": "h2", "claim": "Openings help.", "status": "refuted"},
         {"id": "t1", "claim": "Resume keeps state.", "status": "supported"},
         {"id": "h3", "claim": "Pending run.", "status": "untested"}]
MAP = {"ingredients": [
    {"key": "siblings", "name": "Siblings", "what": "Relabel one-move deviations.", "claims": ["h1", "h3"],
     "conclusion": "They work (h1).", "based_on": ["h1"]},
    {"key": "openings", "name": "Random openings", "what": "Widen coverage.", "claims": ["h2"],
     "conclusion": "They do not (h2).", "based_on": ["h2"]},
    {"key": "method", "name": "Methodology", "what": "Guards.", "claims": ["t1"], "conclusion": "Kept (t1).",
     "based_on": ["t1"]}],
    "unproven_trials": [{"what": "A run stopped early.", "lesson": "Save labels per batch.", "plan": "§C.49"}]}


def test_a_complete_map_has_no_problems():
    assert check_map(MAP, VIEWS) == []


@pytest.mark.parametrize("breakage,needle", [
    (lambda m: m["ingredients"][1]["claims"].remove("h2"), "h2 is not filed"),
    (lambda m: m["ingredients"][0]["claims"].append("h9"), "h9 is not registered"),
    (lambda m: m["ingredients"][0]["based_on"].append("h2"), "cites h2"),
    (lambda m: m["ingredients"][0].update(conclusion=" "), "no conclusion"),
    (lambda m: m["ingredients"][2].update(key="siblings"), "twice"),
    (lambda m: m["unproven_trials"][0].pop("lesson"), "trial"),
])
def test_an_incomplete_or_inconsistent_map_is_reported(breakage, needle):
    m = json.loads(json.dumps(MAP))
    breakage(m)
    assert any(needle in p for p in check_map(m, VIEWS))


def test_the_page_lists_each_ingredient_with_its_conclusion_and_every_claim_with_its_current_verdict():
    page = render(MAP, VIEWS)
    sib = page[page.index("## Siblings"):page.index("## Random openings")]
    assert "They work (h1)." in sib and "**h1** SUPPORTED — Siblings help." in sib
    assert "**h3** UNTESTED — Pending run." in sib and "rests on: h1 (supported)" in sib
    assert "**h2** REFUTED — Openings help." in page
    assert "A run stopped early." in page and "Save labels per batch." in page
    assert "4 claims" in page and "1 supported" not in page.split("\n")[0]


def test_a_conclusion_resting_on_an_undecided_claim_says_so():
    m = json.loads(json.dumps(MAP))
    m["ingredients"][0]["based_on"] = ["h1", "h3"]
    page = render(m, VIEWS)
    assert "rests on: h1 (supported), h3 (untested — PENDING)" in page


def test_the_repository_map_files_every_registered_claim_and_the_page_is_current():
    from harness.hypotheses import Register

    root = Path(__file__).resolve().parent.parent
    views = Register(root / "hypotheses.json").report()
    kmap = json.loads((root / "knowledge_map.json").read_text())
    assert check_map(kmap, views) == []
    page = (root.parent.parent / "docs" / "process-knowledge.md").read_text()
    assert page == render(kmap, views), "stale: PYTHONPATH=. .venv/bin/python scripts/knowledge_digest.py"
