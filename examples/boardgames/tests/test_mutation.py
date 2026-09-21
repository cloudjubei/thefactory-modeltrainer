"""Direct tests for harness/mutation.py — the runner that makes a mutation test mean what it claims.

Every test here is one of the four ways a hand-run mutation lied on 2026-09-20 (see the module docstring).
The one that matters most is test_a_stale_bytecode_cache_cannot_hide_a_mutation: it does not assert that
caches are deleted (a mechanism), it REPRODUCES the stale-import condition and asserts the mutation is still
seen (the property), with a control proving the fixture really is poisoned."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from harness.mutation import clear_bytecode, mutate, run_suite

MOD = "VALUE = 1\n"
TEST = "from mod import VALUE\n\n\ndef test_value():\n    assert VALUE == 1\n"


def _writes_pyc():
    """These tests need bytecode WRITTEN, so they must not inherit the caller's PYTHONDONTWRITEBYTECODE —
    harness.mutation sets it, which would quietly un-poison the very fixture the test depends on."""
    env = {**os.environ}
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    return env


def _tree(tmp_path, mod=MOD):
    (tmp_path / "mod.py").write_text(mod)
    (tmp_path / "test_mod.py").write_text(TEST)
    return tmp_path


def _fake(outcomes):
    """A runner scripted turn by turn, so the orchestration is tested without paying for pytest."""
    seq = list(outcomes)

    def run(tests):
        return seq.pop(0)
    return run


def _ok(failed=()):
    return {"green": not failed, "collected": True, "failed": sorted(failed), "detail": ""}


def test_a_stale_bytecode_cache_cannot_hide_a_mutation(tmp_path):
    """M-A. The pyc is validated on (mtime, size), so a same-size edit restamped to the old mtime is imported
    from cache. This is not hypothetical — it inverted a real 46-pass suite into 2 failures and back."""
    d = _tree(tmp_path)
    mod = d / "mod.py"
    first = run_suite(["test_mod.py"], cwd=d)
    assert first["green"], first["detail"]
    subprocess.run([sys.executable, "-c", "import mod"], cwd=str(d), check=True, env=_writes_pyc())
    before = mod.stat()
    mod.write_text("VALUE = 2\n")                                                  # same SIZE as VALUE = 1
    os.utime(mod, (before.st_atime, before.st_mtime))                              # and now the same mtime
    assert mod.stat().st_size == before.st_size

    control = subprocess.run([sys.executable, "-m", "pytest", "test_mod.py", "-q", "-p", "no:randomly"],
                             cwd=str(d), capture_output=True, text=True)
    assert control.returncode == 0, "fixture is not actually poisoned — the cache was not used, so this test " \
                                    "would pass for the wrong reason"

    seen = run_suite(["test_mod.py"], cwd=d)
    assert not seen["green"] and seen["failed"] == ["test_mod.py::test_value"]


def test_clear_bytecode_reports_what_it_removed(tmp_path):
    d = _tree(tmp_path)
    (d / "pkg").mkdir()
    subprocess.run([sys.executable, "-c", "import mod"], cwd=str(d), check=True, env=_writes_pyc())
    assert (d / "__pycache__").exists()
    assert clear_bytecode(d) == 1 and not (d / "__pycache__").exists()
    assert clear_bytecode(d) == 0


def test_run_suite_refuses_to_call_an_empty_collection_a_pass(tmp_path):
    d = _tree(tmp_path)
    res = run_suite(["test_nothing_at_all.py"], cwd=d)
    assert not res["collected"]


def test_a_mutation_whose_text_is_absent_is_an_error_not_a_survivor(tmp_path):
    """M-B: the most dangerous failure mode — a typo reports the loudest possible verdict."""
    d = _tree(tmp_path)
    with pytest.raises(ValueError) as exc:
        mutate(d / "mod.py", [{"name": "typo", "old": "VALUE = 9", "new": "VALUE = 3"}],
               ["test_mod.py"], cwd=d, runner=_fake([_ok()]))
    assert "occurs 0 times" in str(exc.value) and "SURVIVED" in str(exc.value)


def test_an_ambiguous_mutation_is_refused(tmp_path):
    d = _tree(tmp_path, mod="X = 1\nY = 1\n")
    with pytest.raises(ValueError) as exc:
        mutate(d / "mod.py", [{"name": "amb", "old": "= 1", "new": "= 2"}],
               ["test_mod.py"], cwd=d, runner=_fake([_ok()]))
    assert "occurs 2 times" in str(exc.value)


def test_a_red_baseline_is_refused_because_every_mutation_would_look_killed(tmp_path):
    """M-C."""
    d = _tree(tmp_path)
    with pytest.raises(ValueError) as exc:
        mutate(d / "mod.py", [{"name": "m", "old": "VALUE = 1", "new": "VALUE = 2"}],
               ["test_mod.py"], cwd=d, runner=_fake([_ok(["test_mod.py::test_value"])]))
    assert "RED" in str(exc.value)


def test_an_empty_baseline_collection_is_refused(tmp_path):
    d = _tree(tmp_path)
    with pytest.raises(ValueError) as exc:
        mutate(d / "mod.py", [{"name": "m", "old": "VALUE = 1", "new": "VALUE = 2"}],
               ["test_mod.py"], cwd=d, runner=_fake([{"green": True, "collected": False, "failed": [],
                                                      "detail": ""}]))
    assert "collected no tests" in str(exc.value)


def test_it_names_WHICH_tests_each_mutation_killed(tmp_path):
    """M-D: 'killed' is a proxy. A mutation killed only by an unrelated test has not exercised the guard."""
    d = _tree(tmp_path)
    res = mutate(d / "mod.py", [{"name": "flip", "old": "VALUE = 1", "new": "VALUE = 2"}], ["test_mod.py"],
                 cwd=d, runner=_fake([_ok(), _ok(["test_mod.py::test_value"])]))
    assert res["all_killed"] and res["survived"] == []
    assert res["results"][0]["killed_by"] == ["test_mod.py::test_value"]


def test_a_surviving_mutation_is_reported_as_surviving(tmp_path):
    d = _tree(tmp_path)
    res = mutate(d / "mod.py", [{"name": "silent", "old": "VALUE = 1", "new": "VALUE = 2"}], ["test_mod.py"],
                 cwd=d, runner=_fake([_ok(), _ok()]))
    assert res["survived"] == ["silent"] and not res["all_killed"]


def test_each_mutation_is_applied_to_the_PRISTINE_file_not_stacked(tmp_path):
    d = _tree(tmp_path, mod="A = 1\nB = 1\n")
    seen = []

    def run(tests):
        seen.append((d / "mod.py").read_text())
        return _ok() if len(seen) == 1 else _ok(["test_mod.py::test_value"])
    mutate(d / "mod.py", [{"name": "a", "old": "A = 1", "new": "A = 9"},
                          {"name": "b", "old": "B = 1", "new": "B = 9"}], ["test_mod.py"], cwd=d, runner=run)
    assert seen[1] == "A = 9\nB = 1\n" and seen[2] == "A = 1\nB = 9\n"


def test_the_file_is_restored_even_when_the_suite_explodes(tmp_path):
    d = _tree(tmp_path)

    def boom(tests):
        if (d / "mod.py").read_text() != MOD:
            raise RuntimeError("pytest died mid-mutation")
        return _ok()
    with pytest.raises(RuntimeError):
        mutate(d / "mod.py", [{"name": "m", "old": "VALUE = 1", "new": "VALUE = 2"}],
               ["test_mod.py"], cwd=d, runner=boom)
    assert (d / "mod.py").read_text() == MOD


def test_a_mutation_can_carry_SEVERAL_edits_so_a_guard_can_be_RELOCATED(tmp_path):
    """The most valuable mutation class is 'the guard still runs, just too late' — it is what separates a
    guard from a guard-shaped object, and it cannot be written as one substitution because the check has to
    be deleted in one place and re-inserted in another. A tool that cannot express it cannot test for it."""
    d = _tree(tmp_path, mod="def f(x):\n    check(x)\n    write(x)\n    return x\n")
    seen = []

    def run(tests):
        seen.append((d / "mod.py").read_text())
        return _ok() if len(seen) == 1 else _ok(["test_mod.py::test_value"])
    res = mutate(d / "mod.py", [{"name": "check after write",
                                 "edits": [{"old": "    check(x)\n", "new": ""},
                                           {"old": "    write(x)\n", "new": "    write(x)\n    check(x)\n"}]}],
                 ["test_mod.py"], cwd=d, runner=run)
    assert seen[1] == "def f(x):\n    write(x)\n    check(x)\n    return x\n"
    assert res["all_killed"] and res["results"][0]["killed_by"] == ["test_mod.py::test_value"]


def test_every_edit_in_a_multi_edit_mutation_must_apply_exactly_once(tmp_path):
    """M-B again: a relocation whose second edit silently misses leaves the guard DELETED rather than moved,
    which is a different mutation than the one named in the report."""
    d = _tree(tmp_path, mod="def f(x):\n    check(x)\n    write(x)\n")
    with pytest.raises(ValueError) as exc:
        mutate(d / "mod.py", [{"name": "half a relocation",
                               "edits": [{"old": "    check(x)\n", "new": ""},
                                         {"old": "    flush(x)\n", "new": "    flush(x)\n    check(x)\n"}]}],
               ["test_mod.py"], cwd=d, runner=_fake([_ok()]))
    assert "occurs 0 times" in str(exc.value) and "half a relocation" in str(exc.value)


def test_edits_are_applied_against_the_ORIGINAL_offsets_not_cascaded(tmp_path):
    """Each edit must match the pristine file; otherwise edit 2 could match text that edit 1 just created and
    the mutation silently becomes something else."""
    d = _tree(tmp_path, mod="A\nB\n")
    seen = []

    def run(tests):
        seen.append((d / "mod.py").read_text())
        return _ok() if len(seen) == 1 else _ok(["test_mod.py::test_value"])
    mutate(d / "mod.py", [{"name": "two", "edits": [{"old": "A\n", "new": "B\n"}, {"old": "B\n", "new": "C\n"}]}],
           ["test_mod.py"], cwd=d, runner=run)
    assert seen[1] == "B\nC\n"


def test_overlapping_edits_are_refused_rather_than_silently_order_dependent(tmp_path):
    d = _tree(tmp_path, mod="abcdef\n")
    with pytest.raises(ValueError) as exc:
        mutate(d / "mod.py", [{"name": "clash", "edits": [{"old": "abcd", "new": "X"},
                                                          {"old": "cdef", "new": "Y"}]}],
               ["test_mod.py"], cwd=d, runner=_fake([_ok()]))
    assert "overlap" in str(exc.value)


def test_an_empty_edit_list_is_refused(tmp_path):
    d = _tree(tmp_path)
    with pytest.raises(ValueError) as exc:
        mutate(d / "mod.py", [{"name": "nothing", "edits": []}], ["test_mod.py"], cwd=d, runner=_fake([_ok()]))
    assert "SURVIVED" in str(exc.value)
