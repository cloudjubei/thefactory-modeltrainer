"""Direct tests for harness/hypotheses.py — the register that turns a discovered claim into a falsifiable
object the runs then judge.

The discipline this encodes is the one the whole §C series keeps re-learning by hand: a claim is worth nothing
until it names, IN ADVANCE, which comparison would support it and which would kill it. So status is DERIVED
from ledger evidence and never asserted, and "pre-registered" is CHECKED against when the evidence was drawn —
the difference between a prediction and a rationalisation is a timestamp, not a intention."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.evidence import save_evidence
from harness.hypotheses import Register, _pytest_outcome
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
                   proof="tests/test_resume_integrity.py::test_the_optimizer_survives_a_resume", reads_no_data=True)
    assert h["status"] == "untested" and h["mode"] == "test"
    h = r.verify("t1", run_test=_pass)
    assert h["status"] == "supported" and h["evidence"][0]["ok"]


def test_a_failing_test_refutes_the_claim(tmp_path):
    r = _reg(tmp_path)
    r.register("t1", claim="c", proof="tests/x.py::test_y", reads_no_data=True)
    assert r.verify("t1", run_test=_fail)["status"] == "refuted"


def test_a_proof_that_COLLECTS_NOTHING_is_refused_not_counted_as_passing(tmp_path):
    """The vacuity trap in its purest form: a typo'd node id makes pytest exit 0 having run nothing, and a
    green-by-vacuum proof is worse than no proof because it looks like evidence."""
    r = _reg(tmp_path)
    r.register("t1", claim="c", proof="tests/typo.py::test_does_not_exist", reads_no_data=True)
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
    r.register("t1", claim="c", proof="tests/x.py::test_y", reads_no_data=True)
    with pytest.raises(ValueError, match="(?i)comparison-backed"):
        r.link("t1", led)


def test_reverifying_accumulates_so_a_regression_becomes_contested(tmp_path):
    r = _reg(tmp_path)
    r.register("t1", claim="c", proof="tests/x.py::test_y", reads_no_data=True)
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


# SUPERSESSION — found 2026-09-21 when h8 landed. h3 ("at equal simulations LOWER search wins") and h8 (the
# exact opposite) both read SUPPORTED on the board, each an honest report of its own evidence: h3 was true at
# 460,800 simulations and false at 1,843,200. Nothing in the register could say "do not build on h3", which is
# precisely how a stale finding gets re-used. Supersession is a POINTER, never a status rewrite — rewriting h3
# to refuted would be a lie about what its own comparison found.

def _supported(tmp_path, reg, hid, name="ev"):
    led = _discordant(tmp_path, only_a=40, only_b=5, name=name)
    led.compare("b35", "a11", treatment="config")
    reg.register(hid, claim=f"claim {hid}", a="b35", b="a11", direction="a>b", unit="simulations")
    return reg.link(hid, led)


def test_a_claim_can_be_superseded_by_a_later_one(tmp_path):
    r = _reg(tmp_path)
    _supported(tmp_path, r, "h3", name="e1")
    _supported(tmp_path, r, "h8", name="e2")
    h = r.supersede("h3", by="h8", reason="true at 460,800 sims, false at 1,843,200 — budget-scoped")
    assert h["superseded_by"] == "h8"
    assert "budget-scoped" in h["supersession"]["reason"]


def test_supersession_does_NOT_rewrite_the_superseded_claim_s_status(tmp_path):
    """h3's own comparison really was significant in its predicted direction. The register must keep saying so."""
    r = _reg(tmp_path)
    _supported(tmp_path, r, "h3", name="e1")
    _supported(tmp_path, r, "h8", name="e2")
    assert r.get("h3")["status"] == "supported"
    r.supersede("h3", by="h8", reason="later evidence at higher budget reverses it")
    again = r.get("h3")
    assert again["status"] == "supported" and again["superseded_by"] == "h8"


def test_a_claim_with_NO_evidence_cannot_retire_another(tmp_path):
    """The whole point is that a finding is retired by EVIDENCE, not by someone changing their mind."""
    r = _reg(tmp_path)
    _supported(tmp_path, r, "h3", name="e1")
    r.register("h9", claim="speculation", a="x", b="y", direction="a>b", unit="simulations")
    with pytest.raises(ValueError) as exc:
        r.supersede("h3", by="h9", reason="I think this is better")
    assert "no evidence" in str(exc.value)


def test_a_reason_is_required(tmp_path):
    r = _reg(tmp_path)
    _supported(tmp_path, r, "h3", name="e1")
    _supported(tmp_path, r, "h8", name="e2")
    with pytest.raises(ValueError):
        r.supersede("h3", by="h8", reason="   ")


def test_self_supersession_is_refused_as_the_degenerate_CYCLE(tmp_path):
    """There is deliberately no separate self-check: the cycle walk is seeded with `id`, so it already covers
    this. A second guard for the same property would be dead code that reads as protection."""
    r = _reg(tmp_path)
    _supported(tmp_path, r, "h3", name="e1")
    with pytest.raises(ValueError) as exc:
        r.supersede("h3", by="h3", reason="circular")
    assert "cycle" in str(exc.value).lower()


def test_a_supersession_CYCLE_is_refused(tmp_path):
    """A mutual supersession makes both claims unreadable — neither can be built on, and nothing says why."""
    r = _reg(tmp_path)
    _supported(tmp_path, r, "h3", name="e1")
    _supported(tmp_path, r, "h8", name="e2")
    r.supersede("h3", by="h8", reason="higher budget reverses it")
    with pytest.raises(ValueError) as exc:
        r.supersede("h8", by="h3", reason="going back")
    assert "cycle" in str(exc.value).lower()


def test_superseding_an_unknown_claim_is_refused(tmp_path):
    r = _reg(tmp_path)
    _supported(tmp_path, r, "h8", name="e2")
    with pytest.raises(KeyError):
        r.supersede("nope", by="h8", reason="r")
    with pytest.raises(KeyError):
        r.supersede("h8", by="nope", reason="r")


# NEGLIGIBLE — found 2026-09-21 by an adversarial audit. `_status` filtered on significance FIRST and returned
# `supported` before `null_below` was ever read, so a claim could clear alpha on an effect smaller than the
# threshold its own registrant declared as "absent" and still read as supported. Power, not effect size, was
# deciding the verdict — which is exactly backwards for a claim that ASSERTS an effect exists.

def test_a_significant_effect_INSIDE_the_claim_s_own_null_band_is_not_supported(tmp_path):
    led = _discordant(tmp_path, only_a=60, only_b=25, both=2000, name="tiny")
    led.compare("b35", "a11", treatment="config")
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations", null_below=0.03)
    h = r.link("h1", led)
    assert h["evidence"][0]["significant"] and 0 < h["evidence"][0]["diff"] < 0.03
    assert h["status"] == "negligible"


def test_a_significant_effect_ABOVE_the_null_band_is_still_supported(tmp_path):
    """The guard must not simply demote everything — h8 at +0.0342 against a 0.03 band stays supported."""
    led = _discordant(tmp_path, only_a=116, only_b=81, both=827, name="h8like")
    led.compare("b35", "a11", treatment="config")
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations", null_below=0.03)
    h = r.link("h1", led)
    assert h["evidence"][0]["diff"] > 0.03 and h["status"] == "supported"


def test_the_band_that_applies_is_the_one_the_CLAIM_declared(tmp_path):
    """Same evidence, stricter declared band — the verdict must follow the registrant's own bar, not a default."""
    led = _discordant(tmp_path, only_a=116, only_b=81, both=827, name="h8like2")
    led.compare("b35", "a11", treatment="config")
    r = _reg(tmp_path)
    r.register("strict", claim="c", a="b35", b="a11", direction="a>b", unit="simulations", null_below=0.10)
    assert r.link("strict", led)["status"] == "negligible"


def test_a_significant_effect_in_the_WRONG_direction_still_refutes_however_small(tmp_path):
    """Asymmetric on purpose: magnitude gates a claim being ASSERTED, never a claim being contradicted."""
    led = _discordant(tmp_path, only_a=25, only_b=60, both=2000, name="wrongdir")
    led.compare("b35", "a11", treatment="config")
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations", null_below=0.03)
    h = r.link("h1", led)
    assert h["evidence"][0]["diff"] < 0 and h["status"] == "refuted"


def test_one_replication_above_the_band_is_enough_to_support_it(tmp_path):
    """Accumulation is the point: a claim with a meaningful effect in one draw and a tiny one in another is
    SUPPORTED, not demoted. Only a claim whose every supporting draw is under its own bar is negligible."""
    led = _discordant(tmp_path, only_a=116, only_b=81, both=827, name="mixed")
    led.compare("b35", "a11", treatment="config")
    led.record("b35", outcomes=[1] * 1030 + [0] * 994, params=1, games=14400, provenance="budget_matched",
               seed=131, roots_id="R", code="C", config="G32", compute=460800)
    led.record("a11", outcomes=[1] * 1010 + [0] * 1014, params=1, games=4800, provenance="budget_matched",
               seed=131, roots_id="R", code="C", config="G96", compute=460800)
    led.compare("b35", "a11", treatment="config")
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations", null_below=0.03)
    h = r.link("h1", led)
    sigs = [e for e in h["evidence"] if e["significant"] and e["diff"] > 0]
    assert len(h["evidence"]) == 2 and any(e["diff"] > 0.03 for e in sigs)
    assert h["status"] == "supported"


def test_evidence_carries_the_CAVEATS_the_comparison_was_drawn_with(tmp_path):
    """The register exists so findings are not prose. A verdict stored without its caveats is prose with a
    p-value: h8's row said +0.0342/significant and nothing about being drawn at one arm's home budget."""
    led = _discordant(tmp_path, only_a=40, only_b=5, name="cav")
    led._entries["b35"].update(train_sims=32, deploy_sims=32)
    led._entries["a11"].update(train_sims=96, deploy_sims=32)
    led.compare("b35", "a11", treatment="config")
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations")
    h = r.link("h1", led)
    assert "HOME BUDGET" in h["evidence"][0]["caveats"]["deployment"]


def test_a_claim_whose_every_draw_is_caveated_says_so_on_the_board(tmp_path):
    led = _discordant(tmp_path, only_a=40, only_b=5, name="cav2")
    led._entries["b35"].update(train_sims=32, deploy_sims=32)
    led._entries["a11"].update(train_sims=96, deploy_sims=32)
    led.compare("b35", "a11", treatment="config")
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations")
    assert r.link("h1", led)["caveats"] == ["deployment"]


def test_an_uncaveated_claim_reports_no_caveats(tmp_path):
    led = _discordant(tmp_path, only_a=40, only_b=5, name="cav3")
    led._entries["b35"].update(train_sims=32, deploy_sims=64)
    led._entries["a11"].update(train_sims=96, deploy_sims=64)
    led.compare("b35", "a11", treatment="config")
    r = _reg(tmp_path)
    r.register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations")
    assert r.link("h1", led)["caveats"] == []


def _data(tmp_path, started, name="ev.json.gz"):
    p = tmp_path / name
    save_evidence(p, {"started": started, "rows": []})
    return str(p)


def test_a_claim_about_STORED_evidence_is_timed_against_the_DATA_not_the_verify_call(tmp_path):
    data = _data(tmp_path, "2026-09-10T00:00:00")
    _reg(tmp_path, now="2026-09-20T00:00:00").register("h1", claim="c", proof="tests/x.py::t", data=data)
    h = _reg(tmp_path, now="2026-09-21T00:00:00").verify("h1", run_test=_pass)
    assert h["pre_registered"] is False, "written after the data existed, however soon it was then verified"
    assert h["status"] == "supported"


def test_a_claim_registered_before_its_data_was_produced_is_pre_registered(tmp_path):
    data = _data(tmp_path, "2026-09-10T00:00:00")
    _reg(tmp_path, now="2026-09-01T00:00:00").register("h1", claim="c", proof="tests/x.py::t", data=data)
    assert _reg(tmp_path, now="2026-09-15T00:00:00").verify("h1", run_test=_pass)["pre_registered"] is True


def test_data_of_unknown_age_is_refused_at_declaration(tmp_path):
    p = tmp_path / "ev.json.gz"
    save_evidence(p, {"rows": []})
    with pytest.raises(ValueError, match="started"):
        _reg(tmp_path).register("h1", claim="c", proof="tests/x.py::t", data=str(p))


def test_declared_data_cannot_be_swapped_or_regenerated_under_the_claim(tmp_path):
    data = _data(tmp_path, "2026-09-10T00:00:00")
    r = _reg(tmp_path, now="2026-09-01T00:00:00")
    r.register("h1", claim="c", proof="tests/x.py::t", reads_no_data=True)
    r.attach_data("h1", data)
    with pytest.raises(ValueError, match="already"):
        r.attach_data("h1", _data(tmp_path, "2026-09-11T00:00:00", name="other.json.gz"))
    _data(tmp_path, "2026-09-12T00:00:00")
    with pytest.raises(ValueError, match="changed"):
        r.verify("h1", run_test=_pass)


def test_attaching_data_late_can_only_WITHDRAW_foresight_it_is_judged_by_the_registration_time(tmp_path):
    data = _data(tmp_path, "2026-09-10T00:00:00")
    Register(tmp_path / "h.json", now=lambda: "2026-09-20T00:00:00").register("h1", claim="c", proof="tests/x.py::t", reads_no_data=True)
    r = Register(tmp_path / "h.json", now=lambda: "2026-09-21T00:00:00")
    r.verify("h1", run_test=_pass)
    assert r.get("h1")["pre_registered"] is True
    assert r.attach_data("h1", data)["pre_registered"] is False


def test_only_a_test_backed_claim_takes_a_data_file(tmp_path):
    data = _data(tmp_path, "2026-09-10T00:00:00")
    with pytest.raises(ValueError, match="test-backed"):
        _reg(tmp_path).register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations", data=data)


def _runner(outcomes):
    """A proof runner whose result depends on WHICH node id it is asked to run."""
    def run(nodeid):
        return {"ok": outcomes[nodeid], "collected": 1, "detail": ""}
    return run


def test_a_failed_proof_whose_INCONCLUSIVE_proof_passes_reads_inconclusive_not_refuted(tmp_path):
    r = _reg(tmp_path)
    r.register("h1", claim="c", proof="t::claim", inconclusive_proof="t::undecidable", reads_no_data=True)
    h = r.verify("h1", run_test=_runner({"t::claim": False, "t::undecidable": True}))
    assert h["status"] == "inconclusive"


def test_a_failed_proof_whose_inconclusive_proof_ALSO_fails_is_refuted(tmp_path):
    r = _reg(tmp_path)
    r.register("h1", claim="c", proof="t::claim", inconclusive_proof="t::undecidable", reads_no_data=True)
    h = r.verify("h1", run_test=_runner({"t::claim": False, "t::undecidable": False}))
    assert h["status"] == "refuted"


def test_a_passing_proof_is_supported_without_consulting_the_inconclusive_proof(tmp_path):
    asked = []

    def run(nodeid):
        asked.append(nodeid)
        return {"ok": True, "collected": 1, "detail": ""}
    r = _reg(tmp_path)
    r.register("h1", claim="c", proof="t::claim", inconclusive_proof="t::undecidable", reads_no_data=True)
    assert r.verify("h1", run_test=run)["status"] == "supported"
    assert asked == ["t::claim"]


def test_inconclusive_evidence_never_contradicts_a_pass_but_a_fail_does(tmp_path):
    r = _reg(tmp_path)
    r.register("h1", claim="c", proof="t::claim", inconclusive_proof="t::undecidable", reads_no_data=True)
    r.verify("h1", run_test=_runner({"t::claim": False, "t::undecidable": True}))
    assert r.verify("h1", run_test=_runner({"t::claim": True, "t::undecidable": True}))["status"] == "supported"
    assert r.verify("h1", run_test=_runner({"t::claim": False, "t::undecidable": False}))["status"] == "contested"


def test_the_inconclusive_proof_must_differ_from_the_proof_and_must_collect(tmp_path):
    r = _reg(tmp_path)
    with pytest.raises(ValueError, match="differ"):
        r.register("h1", claim="c", proof="t::claim", inconclusive_proof="t::claim", reads_no_data=True)
    with pytest.raises(ValueError, match="test-backed"):
        r.register("h2", claim="c", a="b35", b="a11", direction="a>b", unit="simulations",
                   inconclusive_proof="t::undecidable")
    r.register("h3", claim="c", proof="t::claim", inconclusive_proof="t::gone", reads_no_data=True)

    def run(nodeid):
        return {"ok": False, "collected": 0 if nodeid == "t::gone" else 1, "detail": ""}
    with pytest.raises(ValueError, match="collected no tests"):
        r.verify("h3", run_test=run)


def test_a_test_backed_claim_must_DECLARE_whether_its_proof_reads_stored_data(tmp_path):
    r = _reg(tmp_path)
    with pytest.raises(ValueError, match="declare"):
        r.register("h1", claim="c", proof="t::p")
    with pytest.raises(ValueError, match="both"):
        r.register("h2", claim="c", proof="t::p", data=_data(tmp_path, "2026-09-10T00:00:00"), reads_no_data=True)
    assert r.register("h3", claim="c", proof="t::p", reads_no_data=True)["reads_no_data"] is True


def test_a_PRE_registered_claim_names_data_that_does_not_exist_yet_and_is_timed_when_it_does(tmp_path):
    future = tmp_path / "later.json.gz"
    _reg(tmp_path, now="2026-09-01T00:00:00").register("h1", claim="c", proof="t::p", data=str(future))
    later = _reg(tmp_path, now="2026-09-20T00:00:00")
    with pytest.raises(ValueError, match="not been produced"):
        later.verify("h1", run_test=_pass)
    save_evidence(future, {"started": "2026-09-10T00:00:00"})
    h = later.verify("h1", run_test=_pass)
    assert h["pre_registered"] is True and h["data"]["started"] == "2026-09-10T00:00:00"


def test_data_named_at_registration_that_ALREADY_exists_is_timed_at_registration(tmp_path):
    data = _data(tmp_path, "2026-09-10T00:00:00")
    _reg(tmp_path, now="2026-09-20T00:00:00").register("h1", claim="c", proof="t::p", data=data)
    assert _reg(tmp_path, now="2026-09-21T00:00:00").verify("h1", run_test=_pass)["pre_registered"] is False


def test_a_stored_claim_from_before_the_declaration_rule_still_verifies(tmp_path):
    path = tmp_path / "hypotheses.json"
    path.write_text(json.dumps({"hypotheses": {"t1": {"id": "t1", "claim": "c", "mode": "test", "proof": "t::p",
                                                      "note": "", "registered_at": "2026-09-01T00:00:00",
                                                      "evidence": []}}}))
    assert Register(path, now=lambda: "2026-09-02T00:00:00").verify("t1", run_test=_pass)["status"] == "supported"


def test_a_claim_reading_SEVERAL_data_files_is_timed_against_the_EARLIEST_of_them(tmp_path):
    early = _data(tmp_path, "2026-09-10T00:00:00", name="a.json.gz")
    late = _data(tmp_path, "2026-09-12T00:00:00", name="b.json.gz")
    _reg(tmp_path, now="2026-09-11T00:00:00").register("h1", claim="c", proof="t::p", data=[early, late])
    h = _reg(tmp_path, now="2026-09-13T00:00:00").verify("h1", run_test=_pass)
    assert h["pre_registered"] is False
    _reg(tmp_path, now="2026-09-01T00:00:00").register("h2", claim="c", proof="t::p", data=[early, late])
    assert _reg(tmp_path, now="2026-09-13T00:00:00").verify("h2", run_test=_pass)["pre_registered"] is True


def test_several_data_files_are_ALL_required_before_verify_and_none_may_be_regenerated(tmp_path):
    first = _data(tmp_path, "2026-09-10T00:00:00", name="a.json.gz")
    pending = tmp_path / "b.json.gz"
    _reg(tmp_path, now="2026-09-01T00:00:00").register("h1", claim="c", proof="t::p", data=[first, str(pending)])
    later = _reg(tmp_path, now="2026-09-20T00:00:00")
    with pytest.raises(ValueError, match="not been produced"):
        later.verify("h1", run_test=_pass)
    save_evidence(pending, {"started": "2026-09-11T00:00:00"})
    assert later.verify("h1", run_test=_pass)["pre_registered"] is True
    _data(tmp_path, "2026-09-15T00:00:00", name="a.json.gz")
    with pytest.raises(ValueError, match="changed"):
        later.verify("h1", run_test=_pass)


def test_declared_data_must_be_stored_evidence_even_before_it_exists(tmp_path):
    r = _reg(tmp_path)
    for bad in ("evidence/arm.json", "evidence/arm.gz"):
        with pytest.raises(ValueError, match="gzip JSON"):
            r.register("h1", claim="c", proof="t::p", data=bad)
        with pytest.raises(ValueError, match="gzip JSON"):
            r.register("h2", claim="c", proof="t::p", data=["evidence/ok.json.gz", bad])
    assert r.report() == []


def _legacy(tmp_path, obj, name):
    p = tmp_path / name
    p.write_text(json.dumps(obj, indent=1))
    return str(p)


def _stored_claim(tmp_path, data, id="h1"):
    path = tmp_path / "hypotheses.json"
    blob = json.loads(path.read_text()) if path.exists() else {"hypotheses": {}}
    files = [{"path": d, "started": "2026-09-10T00:00:00"} for d in data]
    blob["hypotheses"][id] = {"id": id, "claim": "c", "mode": "test", "proof": "t::p", "note": "",
                              "registered_at": "2026-09-01T00:00:00", "evidence": [],
                              "data": files[0] if len(files) == 1 else files}
    path.write_text(json.dumps(blob))


def test_relocating_data_moves_every_claim_that_declares_it_and_keeps_its_timing(tmp_path):
    obj = {"started": "2026-09-10T00:00:00", "seeds": [{"seed": 41, "fails": [3, 1]}]}
    old = _legacy(tmp_path, obj, "arm.json")
    other = _legacy(tmp_path, {"started": "2026-09-10T00:00:00"}, "ref.json")
    _stored_claim(tmp_path, [old])
    _stored_claim(tmp_path, [other, old], id="h2")
    _stored_claim(tmp_path, [other], id="h3")
    new = str(tmp_path / "arm.json.gz")
    save_evidence(new, obj)
    r = _reg(tmp_path, now="2026-09-20T00:00:00")
    assert r.relocate_data(old, new) == ["h1", "h2"]
    reread = _reg(tmp_path, now="2026-09-20T00:00:00")
    assert reread.get("h1")["data"] == {"path": new, "started": "2026-09-10T00:00:00", "moved_from": old}
    assert reread.get("h2")["data"] == [{"path": other, "started": "2026-09-10T00:00:00"},
                                        {"path": new, "started": "2026-09-10T00:00:00", "moved_from": old}]
    assert reread.get("h3")["data"] == {"path": other, "started": "2026-09-10T00:00:00"}
    h = reread.verify("h1", run_test=_pass)
    assert h["status"] == "supported" and h["pre_registered"] is True


def test_relocation_is_refused_unless_the_new_file_holds_EXACTLY_the_old_content(tmp_path):
    old = _legacy(tmp_path, {"started": "2026-09-10T00:00:00", "rows": [1, 2]}, "arm.json")
    _stored_claim(tmp_path, [old])
    r = _reg(tmp_path)
    for i, changed in enumerate(({"started": "2026-09-10T00:00:00", "rows": [1, 3]},
                                 {"started": "2026-09-10T00:00:00", "rows": [1, 2], "extra": None},
                                 {"started": "2026-09-11T00:00:00", "rows": [1, 2]})):
        new = str(tmp_path / f"arm{i}.json.gz")
        save_evidence(new, changed)
        with pytest.raises(ValueError, match="differs"):
            r.relocate_data(old, new)
    assert _reg(tmp_path).get("h1")["data"]["path"] == old


def test_relocation_needs_both_files_and_a_claim_that_declares_the_old_one(tmp_path):
    obj = {"started": "2026-09-10T00:00:00"}
    old = _legacy(tmp_path, obj, "arm.json")
    new = str(tmp_path / "arm.json.gz")
    _stored_claim(tmp_path, [old])
    r = _reg(tmp_path)
    with pytest.raises(ValueError, match="not exist"):
        r.relocate_data(old, new)
    save_evidence(new, obj)
    with pytest.raises(ValueError, match="no claim"):
        r.relocate_data(str(tmp_path / "unknown.json"), new)
    with pytest.raises(ValueError, match="gzip JSON"):
        r.relocate_data(old, str(tmp_path / "moved.json"))
    Path(old).unlink()
    with pytest.raises(ValueError, match="not exist"):
        r.relocate_data(old, new)


@pytest.mark.parametrize("code,out,ok,collected,skipped", [
    (0, "....\n1 passed in 0.05s", True, 1, 0),
    (0, "1 passed, 2 warnings in 0.05s", True, 1, 0),
    (1, "F\n1 failed in 0.05s", False, 1, 0),
    (1, "1 failed, 3 passed in 0.05s", False, 1, 0),
    (1, "1 error in 0.05s", False, 1, 0),
    (5, "no tests ran in 0.01s", False, 0, 0),
    (0, "s\n1 skipped in 0.01s", False, 1, 1),
    (0, "3 passed, 1 skipped in 0.01s", False, 1, 1),
    (0, "captured: the fixture reported 2 skipped\n1 passed in 0.05s", True, 1, 0),
])
def test_the_proof_runner_reads_pytest_s_outcome_and_a_skip_is_never_a_pass(code, out, ok, collected, skipped):
    res = _pytest_outcome(code, out)
    assert (res["ok"], res["collected"], res["skipped"]) == (ok, collected, skipped)


def test_a_SKIPPED_proof_is_refused_not_recorded_because_it_ran_nothing(tmp_path):
    """The evidence directory is not in git: on a checkout without it the evidence proofs skip, and pytest exits 0
    on a skip — without this refusal a missing data file would verify as a pass."""
    def skipped(nodeid):
        return {"ok": False, "collected": 1, "skipped": 1, "detail": "1 skipped"}

    def by_node(nodeid):
        return {"t::p": {"ok": False, "collected": 1, "detail": "1 failed"},
                "t::u": {"ok": False, "collected": 1, "skipped": 1, "detail": "1 skipped"}}[nodeid]
    r = _reg(tmp_path)
    r.register("t1", claim="c", proof="t::p", reads_no_data=True)
    with pytest.raises(ValueError, match="skipped"):
        r.verify("t1", run_test=skipped)
    r.register("t2", claim="c", proof="t::p", inconclusive_proof="t::u", reads_no_data=True)
    with pytest.raises(ValueError, match="skipped"):
        r.verify("t2", run_test=by_node)
    assert r.get("t1")["status"] == "untested" and r.get("t2")["status"] == "untested"


def _judge_file(tmp_path, body="def judge(r):\n    return r >= 15\n", name="judge.py"):
    p = tmp_path / name
    p.write_text(body)
    return str(p)


def test_a_claim_PINS_the_code_that_judges_it_and_verify_refuses_once_that_code_changed(tmp_path):
    """The register stores a proof's node id, not its content: without a pin, the bar or the test could be edited
    after the data is in and the claim would still verify as pre-registered (§C.47 review)."""
    judge = _judge_file(tmp_path)
    test = _judge_file(tmp_path, "def test_it():\n    assert True\n", name="test_judge.py")
    r = _reg(tmp_path)
    h = r.register("h1", claim="c", proof="t::p", reads_no_data=True, pins=[judge, test])
    assert sorted(h["pins"]) == sorted([judge, test])
    assert r.verify("h1", run_test=_pass)["status"] == "supported"
    Path(judge).write_text("def judge(r):\n    return r >= 14\n")
    with pytest.raises(ValueError, match="changed since"):
        r.verify("h1", run_test=_pass)
    assert len(r.get("h1")["evidence"]) == 1


def test_a_pinned_file_may_change_its_prose_but_not_its_behaviour(tmp_path):
    judge = _judge_file(tmp_path, 'def judge(r):\n    """old words"""\n    return r >= 15  # a note\n')
    r = _reg(tmp_path)
    r.register("h1", claim="c", proof="t::p", reads_no_data=True, pins=[judge])
    Path(judge).write_text('def judge(r):\n    """new words"""\n    return r >= 15\n')
    assert r.verify("h1", run_test=_pass)["status"] == "supported"


def test_a_pinned_file_that_is_gone_or_never_existed_is_refused(tmp_path):
    judge = _judge_file(tmp_path)
    r = _reg(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        r.register("h0", claim="c", proof="t::p", reads_no_data=True, pins=[str(tmp_path / "nope.py")])
    r.register("h1", claim="c", proof="t::p", reads_no_data=True, pins=[judge])
    Path(judge).unlink()
    with pytest.raises(ValueError, match="changed since"):
        r.verify("h1", run_test=_pass)


def test_only_a_test_backed_claim_takes_pins(tmp_path):
    judge = _judge_file(tmp_path)
    with pytest.raises(ValueError, match="test-backed"):
        _reg(tmp_path).register("h1", claim="c", a="b35", b="a11", direction="a>b", unit="simulations", pins=[judge])


def _suite_on(tmp_path, claims, bodies, register_proof=False):
    """Run pytest (with the suite's conftest) over a probe test file, against a register holding `claims` — each
    (proof, inconclusive_proof, recorded ok, recorded inconclusive) over the probe's test names."""
    import os
    import subprocess
    import sys

    test = tmp_path / "test_probe_claims.py"
    test.write_text("\n\n".join(f"def {name}():\n    assert {'True' if ok else 'False'}" for name, ok in bodies.items()))
    hyps = {}
    for i, (proof, inc, ok, inconclusive, *earlier) in enumerate(claims):
        entries = []
        for run_ok, run_inc in [*earlier, (ok, inconclusive)]:
            entry = {"proof": f"{test}::{proof}", "ok": run_ok, "detail": "", "drawn_at": "2026-09-24T00:00:00"}
            if run_inc is not None:
                entry["inconclusive"] = run_inc
            entries.append(entry)
        hyps[f"p{i}"] = {"id": f"p{i}", "claim": "c", "mode": "test", "proof": f"{test}::{proof}", "note": "",
                         "registered_at": "2026-09-01T00:00:00", "evidence": entries,
                         **({"inconclusive_proof": f"{test}::{inc}"} if inc else {})}
    register = tmp_path / "register.json"
    register.write_text(json.dumps({"hypotheses": hyps}))
    env = {k: v for k, v in os.environ.items() if k != "REGISTER_PROOF"}
    env["HYPOTHESES_REGISTER"] = str(register)
    if register_proof:
        env["REGISTER_PROOF"] = "1"
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "tests.conftest", "-p", "no:randomly", "-rsxX",
                        str(test)], capture_output=True, text=True, env=env, cwd=Path(__file__).resolve().parent.parent)
    return r.returncode, r.stdout.strip().splitlines()[-1]


def test_the_suite_follows_the_REGISTER_a_supported_claim_s_undecidable_proof_is_not_run(tmp_path):
    code, summary = _suite_on(tmp_path, [("test_holds", "test_undecidable", True, None)],
                              {"test_holds": True, "test_undecidable": False})
    assert code == 0 and summary.startswith("1 passed, 1 skipped")


def test_an_INCONCLUSIVE_claim_s_failing_proof_is_expected_to_fail_and_its_undecidable_proof_runs(tmp_path):
    code, summary = _suite_on(tmp_path, [("test_holds", "test_undecidable", False, True)],
                              {"test_holds": False, "test_undecidable": True})
    assert code == 0 and summary.startswith("1 passed, 1 xfailed")


def test_a_REFUTED_claim_s_proofs_are_both_expected_to_fail(tmp_path):
    code, summary = _suite_on(tmp_path, [("test_holds", "test_undecidable", False, False)],
                              {"test_holds": False, "test_undecidable": False})
    assert code == 0 and summary.startswith("2 xfailed")


def test_a_proof_recorded_as_failing_that_now_PASSES_fails_the_suite(tmp_path):
    """Strict: a registered failure that turns into a pass means the evidence or the judge changed under the
    claim — it must be seen, not absorbed."""
    code, summary = _suite_on(tmp_path, [("test_holds", "test_undecidable", False, True)],
                              {"test_holds": True, "test_undecidable": True})
    assert code != 0 and ("xpass" in summary.lower() or "failed" in summary)


def test_under_REGISTER_PROOF_every_proof_runs_as_written(tmp_path):
    code, summary = _suite_on(tmp_path, [("test_holds", "test_undecidable", True, None)],
                              {"test_holds": True, "test_undecidable": False}, register_proof=True)
    assert code != 0 and summary.startswith("1 failed, 1 passed")


def test_the_LAST_recorded_verification_decides_what_the_suite_expects(tmp_path):
    code, summary = _suite_on(tmp_path, [("test_holds", "test_undecidable", True, None, (False, True))],
                              {"test_holds": True, "test_undecidable": False})
    assert code == 0 and summary.startswith("1 passed, 1 skipped")


def test_tests_the_register_does_not_name_are_untouched(tmp_path):
    code, summary = _suite_on(tmp_path, [], {"test_plain": True, "test_plain_is_undecidable": True})
    assert code == 0 and summary.startswith("2 passed")


def test_the_register_runs_its_proofs_with_REGISTER_PROOF_set(tmp_path, monkeypatch):
    import subprocess

    from harness.hypotheses import _run_pytest
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout="1 passed in 0.01s", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _run_pytest("tests/x.py::test_y")["ok"]
    assert seen["env"]["REGISTER_PROOF"] == "1"
