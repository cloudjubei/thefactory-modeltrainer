"""Direct tests for harness/hypotheses.py — the register that turns a discovered claim into a falsifiable
object the runs then judge.

The discipline this encodes is the one the whole §C series keeps re-learning by hand: a claim is worth nothing
until it names, IN ADVANCE, which comparison would support it and which would kill it. So status is DERIVED
from ledger evidence and never asserted, and "pre-registered" is CHECKED against when the evidence was drawn —
the difference between a prediction and a rationalisation is a timestamp, not a intention."""
from __future__ import annotations

import json

import pytest

from harness.hypotheses import Register
from harness.ledger import Ledger


def _ledger(tmp_path, diff=+0.15, p=0.0006):
    led = Ledger(tmp_path / "ledger.json")
    n = 200
    k_a = int(n * (0.5 + diff / 2))
    k_b = int(n * (0.5 - diff / 2))
    led.record("b35", outcomes=[1] * k_a + [0] * (n - k_a), params=1, games=14400, provenance="budget_matched",
               seed=131, roots_id="R", code="C", config="G32", compute=460800)
    led.record("a11", outcomes=[1] * k_b + [0] * (n - k_b), params=1, games=4800, provenance="budget_matched",
               seed=131, roots_id="R", code="C", config="G96", compute=460800)
    return led


def _discordant(tmp_path, only_a, only_b, both=100, name="disc"):
    """A ledger whose two arms differ by a CONTROLLED number of discordant pairs, so a test can ask for a
    given effect size AND a given p — the nested helper above always yields one-sided discordance, which is
    always significant and cannot express 'a real-looking difference that misses significance'."""
    led = Ledger(tmp_path / f"{name}.json")
    n = both + only_a + only_b
    a = [1] * both + [1] * only_a + [0] * only_b
    b = [1] * both + [0] * only_a + [1] * only_b
    led.record("b35", outcomes=a, params=1, games=14400, provenance="budget_matched", seed=131,
               roots_id="R", code="C", config="G32", compute=460800)
    led.record("a11", outcomes=b, params=1, games=4800, provenance="budget_matched", seed=131,
               roots_id="R", code="C", config="G96", compute=460800)
    assert len(a) == len(b) == n
    return led


def _reg(tmp_path, now="2026-09-01T00:00:00"):
    return Register(tmp_path / "hypotheses.json", now=lambda: now)


def test_a_registered_hypothesis_starts_untested(tmp_path):
    r = _reg(tmp_path)
    h = r.register("h1", claim="lower search trains better per simulation",
                   a="b35", b="a11", direction="a>b", unit="simulations")
    assert h["status"] == "untested" and h["evidence"] == []
    assert h["unit"] == "simulations" and h["direction"] == "a>b"


def test_status_cannot_be_asserted_by_hand(tmp_path):
    """The whole point: a claim does not get to declare itself true."""
    r = _reg(tmp_path)
    with pytest.raises(TypeError):
        r.register("h1", claim="c", a="x", b="y", direction="a>b", unit="simulations", status="supported")


def test_a_significant_comparison_in_the_predicted_direction_supports_it(tmp_path):
    led = _ledger(tmp_path)
    led.compare("b35", "a11", treatment="config")
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations")
    h = r.link("h1", led)
    assert h["status"] == "supported" and len(h["evidence"]) == 1
    assert h["evidence"][0]["significant"] and h["evidence"][0]["diff"] > 0


def test_a_significant_comparison_the_WRONG_way_refutes_it(tmp_path):
    led = _ledger(tmp_path, diff=-0.15)
    led.compare("b35", "a11", treatment="config")
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations")
    assert r.link("h1", led)["status"] == "refuted"


def test_a_non_significant_comparison_leaves_it_inconclusive(tmp_path):
    led = _discordant(tmp_path, only_a=25, only_b=15)   # +0.077, p high -> a real-looking miss
    led.compare("b35", "a11", treatment="config")
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations")
    h = r.link("h1", led)
    assert h["status"] == "inconclusive" and not h["evidence"][0]["significant"]


def test_evidence_must_be_the_comparison_the_hypothesis_DECLARED(tmp_path):
    """Linking any old comparison would let a claim harvest whichever result happened to be significant."""
    led = _ledger(tmp_path)
    led.compare("b35", "a11", treatment="config")
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="b71", b="a23", direction="a>b", unit="simulations")
    with pytest.raises(ValueError, match="(?i)declared"):
        r.link("h1", led)


def test_pre_registration_is_checked_against_when_the_evidence_was_drawn(tmp_path):
    led = _ledger(tmp_path)
    led.compare("b35", "a11", treatment="config")          # drawn NOW
    late = Register(tmp_path / "h.json", now=lambda: "2099-01-01T00:00:00")
    late.register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations")
    h = late.link("h1", led)
    assert h["pre_registered"] is False, "registered AFTER its evidence — that is a rationalisation"
    assert h["status"] == "supported", "status is unaffected; only the CLAIM to foresight is"


def test_a_hypothesis_registered_before_its_evidence_is_pre_registered(tmp_path):
    early = _reg(tmp_path, now="2000-01-01T00:00:00")
    early.register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations")
    led = _ledger(tmp_path)
    led.compare("b35", "a11", treatment="config")
    assert early.link("h1", led)["pre_registered"] is True


def test_conflicting_evidence_is_contested_not_quietly_supported(tmp_path):
    led = _ledger(tmp_path)
    led.compare("b35", "a11", treatment="config")
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations")
    r.link("h1", led)
    led2 = _ledger(tmp_path / "second", diff=-0.15)
    led2.compare("b35", "a11", treatment="config")
    h = r.link("h1", led2)
    assert h["status"] == "contested" and len(h["evidence"]) == 2


def test_the_unit_is_mandatory_because_a_verdict_can_invert_with_it(tmp_path):
    """§C.29: at equal simulations low-search won; under wall-clock it lost. A claim with no unit is not one."""
    r = _reg(tmp_path)
    with pytest.raises(ValueError, match="(?i)unit"):
        r.register("h1", claim="c", a="x", b="y", direction="a>b", unit="")


def test_direction_must_be_declared_and_legible(tmp_path):
    r = _reg(tmp_path)
    with pytest.raises(ValueError, match="(?i)direction"):
        r.register("h1", claim="c", a="x", b="y", direction="probably better", unit="simulations")


def test_the_register_persists_and_reloads(tmp_path):
    r = _reg(tmp_path)
    r.register("h1", claim="lower search wins", a="b35", b="a11", direction="a>b", unit="simulations")
    again = Register(tmp_path / "hypotheses.json")
    assert again.get("h1")["claim"] == "lower search wins"
    assert json.loads((tmp_path / "hypotheses.json").read_text())["hypotheses"]["h1"]["id"] == "h1"


def test_registering_the_same_id_twice_is_refused(tmp_path):
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="x", b="y", direction="a>b", unit="simulations")
    with pytest.raises(ValueError, match="(?i)exists"):
        r.register("h1", claim="different", a="x", b="y", direction="a>b", unit="simulations")


def test_report_lists_every_hypothesis_with_its_derived_status(tmp_path):
    led = _ledger(tmp_path)
    led.compare("b35", "a11", treatment="config")
    r = _reg(tmp_path)
    r.register("h1", claim="supported one", a="b35", b="a11", direction="a>b", unit="simulations")
    r.register("h2", claim="untested one", a="b143", b="a47", direction="a>b", unit="simulations")
    r.link("h1", led)
    rows = {h["id"]: h["status"] for h in r.report()}
    assert rows == {"h1": "supported", "h2": "untested"}


def test_evidence_with_an_unknown_direction_is_not_counted(tmp_path):
    """A comparison whose direction cannot be established says nothing about a directional claim."""
    led = _ledger(tmp_path)
    led._comparisons.append({"a": "b35", "b": "a11", "roots_id": "R", "p": 0.001,
                             "diff": None, "significant": None, "drawn_at": None})
    led._save()
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations")
    with pytest.raises(ValueError, match="(?i)direction"):
        r.link("h1", led)


def test_evidence_of_unknown_age_forfeits_the_pre_registration_claim(tmp_path):
    led = _ledger(tmp_path)
    led.compare("b35", "a11", treatment="config")
    led._comparisons[-1]["drawn_at"] = None
    led._save()
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations")
    h = r.link("h1", led)
    assert h["status"] == "supported" and h["pre_registered"] is False


# --- test-backed claims: most findings are not A/B comparisons -------------------------------------------

def _pass(nodeid):
    return {"ok": True, "collected": 1, "detail": "1 passed"}


def _fail(nodeid):
    return {"ok": False, "collected": 1, "detail": "1 failed"}


def _nothing(nodeid):
    return {"ok": True, "collected": 0, "detail": "no tests ran"}


def test_a_claim_can_be_backed_by_a_test_instead_of_a_comparison(tmp_path):
    """Most of what this project learns is not an A/B: 'resume loses the Adam state', 'encode was hardcoded to
    two planes'. Those are proved by a regression test, and the register has to hold them too."""
    r = _reg(tmp_path)
    h = r.register("t1", claim="a resumed run keeps its optimizer state",
                   proof="tests/test_resume_integrity.py::test_the_optimizer_survives_a_resume")
    assert h["status"] == "untested" and h["mode"] == "test"
    h = r.verify("t1", run_test=_pass)
    assert h["status"] == "supported" and h["evidence"][0]["ok"]


def test_a_failing_test_refutes_the_claim(tmp_path):
    r = _reg(tmp_path)
    r.register("t1", claim="c", proof="tests/x.py::test_y")
    assert r.verify("t1", run_test=_fail)["status"] == "refuted"


def test_a_proof_that_COLLECTS_NOTHING_is_refused_not_counted_as_passing(tmp_path):
    """The vacuity trap in its purest form: a typo'd node id makes pytest exit 0 having run nothing, and a
    green-by-vacuum proof is worse than no proof because it looks like evidence."""
    r = _reg(tmp_path)
    r.register("t1", claim="c", proof="tests/typo.py::test_does_not_exist")
    with pytest.raises(ValueError, match="(?i)collected no tests"):
        r.verify("t1", run_test=_nothing)
    assert r.get("t1")["status"] == "untested", "a vacuous proof must leave the claim unproven"


def test_a_claim_needs_either_a_comparison_or_a_proof(tmp_path):
    r = _reg(tmp_path)
    with pytest.raises(ValueError, match="(?i)either"):
        r.register("t1", claim="c")


def test_a_claim_cannot_be_both(tmp_path):
    r = _reg(tmp_path)
    with pytest.raises(ValueError, match="(?i)exactly one"):
        r.register("t1", claim="c", a="x", b="y", direction="a>b", unit="simulations",
                   proof="tests/x.py::test_y")


def test_a_comparison_claim_still_demands_its_unit(tmp_path):
    r = _reg(tmp_path)
    with pytest.raises(ValueError, match="(?i)unit"):
        r.register("t1", claim="c", a="x", b="y", direction="a>b")


def test_verify_refuses_a_comparison_backed_claim(tmp_path):
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="x", b="y", direction="a>b", unit="simulations")
    with pytest.raises(ValueError, match="(?i)test-backed"):
        r.verify("h1", run_test=_pass)


def test_link_refuses_a_test_backed_claim(tmp_path):
    led = _ledger(tmp_path)
    led.compare("b35", "a11", treatment="config")
    r = _reg(tmp_path)
    r.register("t1", claim="c", proof="tests/x.py::test_y")
    with pytest.raises(ValueError, match="(?i)comparison-backed"):
        r.link("t1", led)


def test_reverifying_accumulates_so_a_regression_becomes_contested(tmp_path):
    r = _reg(tmp_path)
    r.register("t1", claim="c", proof="tests/x.py::test_y")
    r.verify("t1", run_test=_pass)
    h = r.verify("t1", run_test=_fail)
    assert h["status"] == "contested" and len(h["evidence"]) == 2


def test_a_record_written_before_mode_existed_is_still_read_correctly(tmp_path):
    """Records registered before the test-backed mode was added carry no `mode` field. Defaulting them the
    wrong way silently sent every one of them down the test-backed path."""
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations")
    blob = json.loads((tmp_path / "hypotheses.json").read_text())
    del blob["hypotheses"]["h1"]["mode"]
    (tmp_path / "hypotheses.json").write_text(json.dumps(blob))

    reloaded = Register(tmp_path / "hypotheses.json")
    assert reloaded.get("h1")["mode"] == "comparison"
    led = _ledger(tmp_path)
    led.compare("b35", "a11", treatment="config")
    assert reloaded.link("h1", led)["status"] == "supported"


def test_a_difference_below_the_declared_null_threshold_is_NULL_not_merely_inconclusive(tmp_path):
    """NULL and INCONCLUSIVE are different claims: NULL says the effect is absent, INCONCLUSIVE says the
    measurement could not see it. §C.28 (+0.0547, underpowered) and §C.29's follow-up (+0.0156, genuinely flat)
    must not collapse into the same word."""
    led = _ledger(tmp_path, diff=0.01)
    led.compare("b35", "a11", treatment="config")
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations", null_below=0.03)
    assert r.link("h1", led)["status"] == "null"


def test_a_difference_above_the_threshold_that_misses_significance_stays_inconclusive(tmp_path):
    led = _discordant(tmp_path, only_a=25, only_b=15)
    led.compare("b35", "a11", treatment="config")
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations", null_below=0.03)
    assert r.link("h1", led)["status"] == "inconclusive"


def test_the_null_threshold_is_recorded_with_the_claim_not_chosen_at_reading_time(tmp_path):
    r = _reg(tmp_path)
    h = r.register("h1", claim="c", a="x", b="y", direction="a>b", unit="simulations", null_below=0.05)
    assert h["null_below"] == 0.05
