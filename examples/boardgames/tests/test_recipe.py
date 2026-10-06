"""Direct tests for harness/recipe.py — a run is launched from its registered recipe, never by hand (h138: H3 was
launched with 6 relabel workers against a registered 4, and its judge rightly refused it), and a judge compares a
run's recorded config with the registered one, ignoring only the settings proven not to change what is trained."""
from __future__ import annotations

import sys
import types

import pytest

from harness.recipe import NOT_TRAINING, load_recipe, recipe_matches

ARM = {"iterations": 20, "relabel_workers": 4, "strategy_tree": {"player": 0, "depth": 4}}


def test_a_recorded_config_matches_its_recipe_whatever_the_relabel_workers_and_certify_depth():
    recorded = {**ARM, "relabel_workers": 6, "seeds": [1, 2], "certify_depth": 6}
    assert recipe_matches(recorded, ARM, [1, 2])


@pytest.mark.parametrize("change", [{"iterations": 21}, {"strategy_tree": {"player": 0, "depth": 2}},
                                    {"extra_knob": True}, {"seeds": [1, 3]}])
def test_any_other_difference_is_not_the_recipe(change):
    recorded = {**ARM, "seeds": [1, 2], **change}
    assert not recipe_matches(recorded, ARM, [1, 2])


def test_a_missing_setting_is_not_the_recipe():
    recorded = {k: v for k, v in ARM.items() if k != "iterations"}
    assert not recipe_matches({**recorded, "seeds": [1, 2]}, ARM, [1, 2])


def test_only_settings_proven_not_to_change_training_are_ignored():
    assert NOT_TRAINING == ("certify_depth", "relabel_workers")


def _spec_module(monkeypatch, spec):
    mod = types.ModuleType("fake_floor")
    mod.SPEC = spec
    monkeypatch.setitem(sys.modules, "fake_floor", mod)


def test_a_recipe_is_loaded_from_a_registered_spec_arm_with_its_seeds(monkeypatch):
    _spec_module(monkeypatch, {"arms": {"h3": ARM}, "seeds": (5, 6)})
    cfg, seeds = load_recipe("fake_floor:h3")
    assert cfg == ARM and seeds == [5, 6] and cfg is not ARM


@pytest.mark.parametrize("ref,match", [("fake_floor", "MODULE:ARM"), ("fake_floor:nope", "nope"),
                                       ("fake_floor:h3:x", "MODULE:ARM")])
def test_a_reference_that_names_no_registered_arm_is_refused(monkeypatch, ref, match):
    _spec_module(monkeypatch, {"arms": {"h3": ARM}, "seeds": (5, 6)})
    with pytest.raises(ValueError, match=match):
        load_recipe(ref)


def test_the_h3_recipe_loads_from_its_judge():
    cfg, seeds = load_recipe("harness.floor_h3:h3")
    assert cfg["strategy_tree"] == {"player": 0, "depth": 4} and seeds == [481, 482, 483, 484, 485, 486, 487]
    assert cfg["relabel_workers"] == 4
