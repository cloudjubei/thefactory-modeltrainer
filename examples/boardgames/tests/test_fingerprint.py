"""§C.17 TRAINING-CODE FINGERPRINT.

Replays the 2026-09-08 incident: the gpool and categorical-head A/Bs both reused controls trained under an older
Adam regime, so each varied two things at once and every existing guard passed. The fingerprint has to fire on a
behaviour change in the training path and stay silent on everything else — a guard that fires on prose edits or on
measurement work gets bypassed, which is how we got here.
"""
import subprocess

import pytest

from harness.fingerprint import TRAINING_MODULES, normalize, training_fingerprint


def test_fingerprint_is_stable_across_calls():
    assert training_fingerprint("connect4") == training_fingerprint("connect4")


def test_the_game_module_is_part_of_the_fingerprint():
    assert training_fingerprint("connect4") != training_fingerprint("tictactoe")
    assert training_fingerprint("connect4") != training_fingerprint(None)


def test_a_behaviour_change_in_the_training_path_changes_the_hash():
    # The real incident, in miniature: an optimizer that keeps its state vs one that does not.
    before = normalize("def train(net, opt_state=None):\n    opt = Adam(net)\n    return opt\n")
    after = normalize("def train(net, opt_state=None):\n    opt = opt_state.get('opt') or Adam(net)\n    return opt\n")
    assert before != after


def test_prose_edits_do_not_change_the_hash():
    # A guard that fires on documentation edits is a guard people switch off (§C.16).
    plain = normalize("def f(x):\n    return x + 1\n")
    documented = normalize('def f(x):\n    """Adds one to x, at length."""\n    # and a comment\n    return x + 1\n')
    assert plain == documented


def test_reformatting_does_not_change_the_hash():
    assert normalize("def f(x):\n    return x+1\n") == normalize("def f(\n    x,\n):\n    return x + 1\n")


def test_docstring_only_function_survives_stripping():
    assert normalize('def f():\n    """only a docstring"""\n')  # must not raise on an emptied body


def test_measurement_code_is_not_in_the_fingerprint():
    # benchmark/measurement/ledger read checkpoints and never feed back into training; including them would make
    # the guard fire on measurement work, which is the false positive that trains people to bypass it.
    for excluded in ("harness/benchmark.py", "harness/measurement.py", "harness/ledger.py"):
        assert excluded not in TRAINING_MODULES


def test_fingerprint_can_be_computed_at_a_past_revision():
    # This is what lets us say WHICH era an old run belongs to instead of guessing from file dates.
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert training_fingerprint("connect4", revision=rev)


def test_the_adam_eras_have_distinct_fingerprints():
    # eacf154 hoisted Adam to run scope; 619e252 moved it from per-iteration to per-batch. Three regimes, three
    # fingerprints — and every control we owned sat in an earlier one than the arm it was compared against.
    eras = {name: training_fingerprint("connect4", revision=rev)
            for name, rev in (("per_iteration", "619e252~1"), ("per_batch", "619e252"), ("per_run", "eacf154"))}
    assert len(set(eras.values())) == 3, eras


def test_a_module_that_did_not_exist_yet_is_absent_not_an_error():
    # refutation.py post-dates the earliest runs we still compare against. Raising here would make the guard
    # useless for exactly the historical question it exists to answer, and a crashing guard gets routed around.
    early = training_fingerprint("connect4", revision="9c84431~1")
    assert early and early != training_fingerprint("connect4")


@pytest.mark.parametrize("path", TRAINING_MODULES)
def test_every_declared_training_module_exists(path):
    from harness.fingerprint import HARNESS_ROOT

    assert (HARNESS_ROOT / path).exists(), f"{path} is fingerprinted but missing — the guard would crash, not warn"


def test_config_fingerprint_ignores_naming_and_run_length():
    # ctrl302_postfix_s0 (24 batches) vs carry_03 (40): at the budget-matched index these are the SAME recipe,
    # and the ledger's budget check is what enforces equal games.
    from harness.fingerprint import config_fingerprint

    a = {"run_dir": "x", "batches": 24, "seed": 0, "sims": 96}
    b = {"run_dir": "y", "batches": 40, "seed": 0, "sims": 96}
    assert config_fingerprint(a) == config_fingerprint(b)


def test_config_fingerprint_separates_the_flag_under_test():
    from harness.fingerprint import config_fingerprint

    base = {"run_dir": "x", "batches": 24, "net_arch": {"channels": 64, "value_bins": 0}}
    bins = {"run_dir": "x", "batches": 24, "net_arch": {"channels": 64, "value_bins": 21}}
    assert config_fingerprint(base) != config_fingerprint(bins)


def test_config_fingerprint_is_order_insensitive():
    from harness.fingerprint import config_fingerprint

    assert config_fingerprint({"a": 1, "b": 2}) == config_fingerprint({"b": 2, "a": 1})
