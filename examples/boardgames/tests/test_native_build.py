"""Direct tests for harness/native_build.py — compiling a C library on first use so that processes starting together
on a fresh build directory (a strategy run's workers) never take each other's half-written file."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness import native_build as nb


def _fake_cc(calls, rival=None):
    def run(cmd, **kwargs):
        calls.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"lib" + str(len(calls)).encode())
        if rival is not None and len(calls) == 1:
            rival()

        class R:
            returncode, stderr = 0, ""
        return R()
    return run


def test_a_library_is_compiled_once_into_place(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(nb.subprocess, "run", _fake_cc(calls))
    out = tmp_path / "build" / "x-abc.so"
    nb.compile_library(out, ["-O3"], tmp_path / "x.c")
    nb.compile_library(out, ["-O3"], tmp_path / "x.c")
    assert out.read_bytes() == b"lib1" and len(calls) == 1
    assert calls[0][:2] == ["cc", "-O3"] and calls[0][-1] == str(tmp_path / "x.c")
    assert sorted(p.name for p in out.parent.iterdir()) == ["x-abc.so"]


def test_a_rival_finishing_between_our_compile_and_its_move_does_not_break_the_build(tmp_path, monkeypatch):
    calls = []
    out = tmp_path / "x-abc.so"
    monkeypatch.setattr(nb.subprocess, "run",
                        _fake_cc(calls, rival=lambda: nb.compile_library(out, [], tmp_path / "x.c")))
    nb.compile_library(out, [], tmp_path / "x.c")
    assert len(calls) == 2 and out.read_bytes() in (b"lib1", b"lib2")
    assert [p.name for p in tmp_path.iterdir()] == ["x-abc.so"]


def test_a_failed_compile_raises_and_leaves_nothing_behind(tmp_path, monkeypatch):
    class Failed:
        returncode, stderr = 1, "boom"
    monkeypatch.setattr(nb.subprocess, "run", lambda cmd, **kw: Failed())
    with pytest.raises(RuntimeError, match="x.c failed:\nboom"):
        nb.compile_library(tmp_path / "x-abc.so", [], tmp_path / "x.c")
    assert list(tmp_path.iterdir()) == []
