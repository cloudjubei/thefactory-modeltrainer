"""§3.6 — harness.steady_exceptions.needed for Connect-4 in C (harness/c4leafwalk.c), same contract: {"status": "ok" |
"cap" | "timeout", "exceptions": {position key: column}, "own_positions"}. The C walk takes winning moves from the
exact solve (harness/c4solver.c, included in it) instead of a `winning` callable, so it serves Connect-4 only; the
Python walk stays the reference (tests/test_native_leaf.py). Compiled on first use like harness.native_solver."""
from __future__ import annotations

import ctypes
import hashlib
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

from harness.solver import to_bitboard

HERE = Path(__file__).resolve().parent
SOURCES = (HERE / "c4leafwalk.c", HERE / "c4solver.c")
BUILD_DIR = HERE.parent / "build"
LOG2_SOLVER_TABLE = 24
LOG2_MOVE_CACHE = 24
MAX_EXCEPTIONS = 1_000_000
STATUS = {0: "ok", 1: "cap", 2: "timeout"}
ERRORS = {3: "a position the oracle's winning moves reach has no winning move — the oracle is wrong",
          4: "a line the oracle's winning moves lead to is not won — the oracle is wrong",
          6: f"more than {MAX_EXCEPTIONS} exceptions — raise MAX_EXCEPTIONS",
          7: "the C walk ran out of memory"}
_LIB = None
_BUFFERS = None


def _library():
    global _LIB
    if _LIB is None:
        digest = hashlib.sha256(b"".join(s.read_bytes() for s in SOURCES)).hexdigest()[:12]
        out = BUILD_DIR / f"c4leafwalk-{digest}.so"
        if not out.exists():
            BUILD_DIR.mkdir(exist_ok=True)
            tmp = out.with_suffix(".tmp.so")
            r = subprocess.run(["cc", "-O3", "-shared", "-fPIC", "-I", str(HERE), "-o", str(tmp), str(SOURCES[0])],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"compiling {SOURCES[0].name} failed:\n{r.stderr}")
            tmp.rename(out)
        lib = ctypes.CDLL(str(out))
        lib.c4_leaf_init.argtypes = [ctypes.c_int, ctypes.c_int]
        u64p, i8p, i64p = (ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int8),
                           ctypes.POINTER(ctypes.c_int64))
        lib.c4_leaf_walk.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, i8p, ctypes.c_int,
                                     ctypes.c_int64, ctypes.c_double, u64p, u64p, i8p, ctypes.c_int64, i64p, i64p]
        lib.c4_leaf_walk.restype = ctypes.c_int
        lib.c4_verify_walk.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, i8p, ctypes.c_int,
                                       ctypes.c_int64, ctypes.c_double, i64p, i64p]
        lib.c4_verify_walk.restype = ctypes.c_int
        if not lib.c4_leaf_init(LOG2_SOLVER_TABLE, LOG2_MOVE_CACHE):
            raise MemoryError("could not allocate the C walk's tables")
        _LIB = lib
    return _LIB


def _board(position: int, mask: int, to_move: int) -> tuple:
    """The 42-cell board (row * 7 + col, row 0 the bottom) of a bitboard whose `position` holds `to_move`'s stones."""
    mover, other = to_move + 1, 2 - to_move
    return tuple(0 if not mask >> (col * 7 + row) & 1 else mover if position >> (col * 7 + row) & 1 else other
                 for row in range(6) for col in range(7))


class Exceptions(Mapping):
    """A walk's exceptions, {position key: column}, holding the walk's raw bitboards and building the keys (board
    tuples) only when read — local search needs only how many there are (h214: building them was ~45% of a walk)."""

    def __init__(self, raw: list, to_move: int):
        self._raw, self._to_move, self._built = raw, to_move, None

    def _keys(self) -> dict:
        if self._built is None:
            self._built = {(_board(p, m, self._to_move), self._to_move): c for p, m, c in self._raw}
        return self._built

    def __len__(self) -> int:
        return len(self._raw)

    def __iter__(self):
        return iter(self._keys())

    def __getitem__(self, key):
        return self._keys()[key]


def _buffers():
    global _BUFFERS
    if _BUFFERS is None:
        _BUFFERS = ((ctypes.c_uint64 * MAX_EXCEPTIONS)(), (ctypes.c_uint64 * MAX_EXCEPTIONS)(),
                    (ctypes.c_int8 * MAX_EXCEPTIONS)())
    return _BUFFERS


VERIFY_REASONS = {0: None, 1: "cap", 2: "timeout", 8: "undefined", 9: "draw", 10: "loss"}


def verify(facts, root, levels: dict, n_levels: int, cap: int = 1_000_000, deadline: float | None = None) -> dict:
    """harness.steady_state.verify for a Connect-4 `root`, in C: the same result, counts included."""
    lib = _library()
    position, mask, moves = to_bitboard(root)
    cells = (ctypes.c_int8 * 42)(*[levels.get(i, -1) for i in range(42)])
    own, walked = ctypes.c_int64(0), ctypes.c_int64(0)
    seconds = -1.0 if deadline is None else max(0.0, deadline - time.monotonic())
    status = lib.c4_verify_walk(position, mask, moves, cells, n_levels, cap, seconds, ctypes.byref(own),
                                ctypes.byref(walked))
    if status not in VERIFY_REASONS:
        raise RuntimeError(ERRORS[status])
    reason = VERIFY_REASONS[status]
    return {"won": reason is None, "reason": reason, "own_positions": own.value, "positions": walked.value}


def needed(facts, root, levels: dict, n_levels: int, winning, cap: int, deadline: float | None = None) -> dict:
    """harness.steady_exceptions.needed for a Connect-4 `root`; `winning` is unused — the exact solve decides. The
    exceptions are an `Exceptions` mapping, equal to the reference's dict."""
    lib = _library()
    position, mask, moves = to_bitboard(root)
    cells = (ctypes.c_int8 * 42)(*[levels.get(i, -1) for i in range(42)])
    exc_position, exc_mask, exc_move = _buffers()
    count, own = ctypes.c_int64(0), ctypes.c_int64(0)
    seconds = -1.0 if deadline is None else max(0.0, deadline - time.monotonic())
    status = lib.c4_leaf_walk(position, mask, moves, cells, n_levels, cap, seconds, exc_position, exc_mask, exc_move,
                              MAX_EXCEPTIONS, ctypes.byref(count), ctypes.byref(own))
    if status == 5:
        raise ValueError("the root is a position its side cannot win — no map or exception can")
    if status in ERRORS:
        raise RuntimeError(ERRORS[status])
    if status != 0:
        return {"status": STATUS[status], "exceptions": {}, "own_positions": own.value}
    n = count.value
    raw = list(zip(exc_position[:n], exc_mask[:n], exc_move[:n]))
    return {"status": "ok", "exceptions": Exceptions(raw, root.to_move), "own_positions": own.value}
