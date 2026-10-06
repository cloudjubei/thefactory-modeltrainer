"""A run's registered recipe: where a launch takes its training config, and how a judge compares a run with it.

A recipe is one arm of a judge's SPEC (`SPEC["arms"][arm]`) with the SPEC's seeds, named `module:arm` (e.g.
`harness.floor_h3:h3`). Launching from it, not from hand-typed flags, makes the recorded config the registered one by
construction (h138: a hand-typed `--workers 6` against a registered 4 voided H3's judgement). A judge compares every
setting except those proven not to change what is trained: `certify_depth` is measured after training, and relabelling
trains bit-identically across worker counts
(tests/test_neural.py::test_c49_relabelling_across_worker_processes_trains_bit_identically)."""
from __future__ import annotations

import copy
import importlib

NOT_TRAINING = ("certify_depth", "relabel_workers")


def recipe_matches(recorded: dict, registered: dict, seeds) -> bool:
    def trained(cfg: dict) -> dict:
        return {k: v for k, v in cfg.items() if k not in NOT_TRAINING}

    return trained(recorded) == trained({**registered, "seeds": list(seeds)})


def load_recipe(ref: str) -> tuple[dict, list]:
    parts = ref.split(":")
    if len(parts) != 2:
        raise ValueError(f"a recipe is named MODULE:ARM (e.g. harness.floor_h3:h3), got {ref!r}")
    module, arm = parts
    spec = importlib.import_module(module).SPEC
    if arm not in spec["arms"]:
        raise ValueError(f"{module} registers no arm {arm!r}; its arms are {sorted(spec['arms'])}")
    return copy.deepcopy(spec["arms"][arm]), list(spec["seeds"])
