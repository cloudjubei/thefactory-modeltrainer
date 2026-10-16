"""Compiling a C library on first use (harness.native_solver, harness.native_leaf) — safely when processes start together
on a fresh build directory, as a strategy run's workers do: each compiles into a temporary file of its own beside the
library and moves it into place atomically, so none takes another's half-written file and the last to finish wins."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def compile_library(out: Path, flags: list[str], source: Path) -> None:
    """Compile `source` with the system compiler and `flags` into the shared library `out`, unless it is there."""
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=f".{out.stem}-", suffix=".tmp.so")
    os.close(fd)
    try:
        r = subprocess.run(["cc", *flags, "-o", tmp, str(source)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"compiling {Path(source).name} failed:\n{r.stderr}")
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
