"""§C.49 — the exact Connect-4 solve in C (harness/c4solver.c), with the same interface as harness/solver.py's
`move_values`/`position_value`. The C source is compiled on first use with the system compiler into build/, named
by the source's hash so an edited source is never served by a stale library. harness/solver.py stays the reference
the port is tested against; this module only exists to make solves near the opening fast enough to certify a
net's play (§C.49 P-START). Measurement-only: training never imports it."""
from __future__ import annotations

import ctypes
import hashlib
import subprocess
from pathlib import Path

from harness.solver import _can_play, _is_winning_move, _bottom_mask_col, _column_mask, _TOTAL, WIDTH, to_bitboard

SOURCE = Path(__file__).resolve().parent / "c4solver.c"
BUILD_DIR = Path(__file__).resolve().parent.parent / "build"
DEFAULT_LOG2_TABLE = 24
_LIB = None


def _library(log2_table: int = DEFAULT_LOG2_TABLE):
    global _LIB
    if _LIB is None:
        digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:12]
        out = BUILD_DIR / f"c4solver-{digest}.so"
        if not out.exists():
            BUILD_DIR.mkdir(exist_ok=True)
            tmp = out.with_suffix(".tmp.so")
            r = subprocess.run(["cc", "-O3", "-shared", "-fPIC", "-o", str(tmp), str(SOURCE)], capture_output=True,
                               text=True)
            if r.returncode != 0:
                raise RuntimeError(f"compiling {SOURCE.name} failed:\n{r.stderr}")
            tmp.rename(out)
        lib = ctypes.CDLL(str(out))
        lib.c4_init.argtypes = [ctypes.c_int]
        lib.c4_solve.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
        lib.c4_solve.restype = ctypes.c_int
        lib.c4_canonical_key.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        lib.c4_canonical_key.restype = ctypes.c_uint64
        if not lib.c4_init(log2_table):
            raise MemoryError(f"could not allocate a 2^{log2_table}-entry transposition table")
        _LIB = lib
    return _LIB


def set_table_size(log2_table: int) -> None:
    """Reallocate the transposition table (emptying it). Values are exact at any size; only speed changes."""
    if not 1 <= log2_table <= 30:
        raise ValueError(f"log2 table size {log2_table} outside 1..30")
    _library().c4_init(log2_table)


def reset_table() -> None:
    _library().c4_reset()


def move_values(state, weak: bool = True, book=None) -> dict[int, int]:
    """Game-theoretic value of every legal column, mover's perspective — harness.solver.move_values exactly, with
    the solve in C. `book` (proven values keyed by the canonical key) short-circuits children it holds."""
    lib = _library()
    if state.done:
        return {}
    position, mask, moves = to_bitboard(state)
    values: dict[int, int] = {}
    for c in range(WIDTH):
        if not _can_play(mask, c):
            continue
        if _is_winning_move(position, mask, c):
            values[c] = 1 if weak else (_TOTAL + 1 - moves) // 2
            continue
        pos2, mask2 = position ^ mask, mask | ((mask + _bottom_mask_col(c)) & _column_mask(c))
        if book is not None and weak:
            bv = book.proven_value(int(lib.c4_canonical_key(pos2, mask2)))
            if bv is not None:
                values[c] = max(-1, min(1, -bv))
                continue
        s = -lib.c4_solve(pos2, mask2, moves + 1, 1 if weak else 0)
        values[c] = max(-1, min(1, s)) if weak else s
    return values


def position_value(state, weak: bool = True, book=None) -> int:
    """The exact value to the side to move (the best of its move values)."""
    vals = move_values(state, weak=weak, book=book)
    return max(vals.values()) if vals else 0


def solve_position(state, book=None) -> int:
    """The exact WEAK value (win +1 / draw 0 / loss -1) to the side to move by ONE solve of the position itself —
    equal to `position_value`, which solves every child instead, and several times cheaper. Weak only: the strong
    distance scores of a lost position differ by one between the two routes (the Python reference floors a
    "loses next move" score in one place and truncates it in the other), and certifying needs only win/draw/loss.
    `book` answers a position it proves outright."""
    if state.done:
        raise ValueError("a finished game has no value to the side to move — read its result instead")
    lib = _library()
    position, mask, moves = to_bitboard(state)
    if book is not None:
        bv = book.proven_value(int(lib.c4_canonical_key(position, mask)))
        if bv is not None:
            return max(-1, min(1, bv))
    return max(-1, min(1, lib.c4_solve(position, mask, moves, 1)))
