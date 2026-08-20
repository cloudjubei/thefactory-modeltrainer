"""A persistent, game-AGNOSTIC solved-position store — the opening-book / endgame-tablebase backing that makes
a solved game *computably* optimal (see docs/implementation-plan.md (§C.5)).

It knows nothing about any game: a game supplies canonical integer keys (symmetry-reduced) and the values'
meaning (e.g. a game-theoretic score, or a search bound). Bounded with PRIORITY eviction — when full, the
lowest-priority entries go first, so the expensive/hub positions (which cost the most to recompute) are kept.
Persisted as a compact columnar file so it accumulates across runs ("over time we store the difficult ones").

Entries are either PROVEN (an exact game-theoretic value) or an ESTIMATE (a bounded-search belief, for the
opening the exact solve can't yet reach). `get()` / `proven_value()` return a value only for PROOFS — so every
EXACT consumer (one-ply lookahead, the solver's book short-circuit) automatically ignores estimates and stays
correct — while `entry()` exposes the richer record (status + best-move bitmask + confidence) a deep model reads.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    import numpy as _np
except Exception:  # numpy is present in the trainer venv; degrade to JSON if ever absent
    _np = None

PROVEN = 0
ESTIMATE = 1
_EST_SCALE = 1000  # estimate values live in [-1, 1]; persisted as a scaled int16 column


@dataclass(frozen=True)
class Entry:
    """A book record. `value` is the exact game-theoretic value to the side-to-move (win +1 / draw 0 / loss −1)
    when `status == PROVEN`, else a bounded-search estimate in [−1, 1]. `best_actions` is a bitmask of the
    optimal (proven) / top (estimate) moves — the one-ply-lookahead set, the model's policy target, and the
    implicit principal variation. `n` is the sample size behind an estimate (0 for a proof)."""

    status: int
    value: float
    best_actions: int
    n: int


class Tablebase:
    def __init__(self, cap: int = 20_000_000):
        self.cap = int(cap)
        self._v: dict[int, int] = {}  # canonical key -> value (exact int, TT bound, or scaled estimate)
        self._p: dict[int, int] = {}  # canonical key -> priority (higher = keep)
        self._s: dict[int, int] = {}  # canonical key -> status; ABSENT ⇒ PROVEN (so raw _v writes stay proven)
        self._a: dict[int, int] = {}  # canonical key -> best-actions bitmask (absent ⇒ 0)
        self._n: dict[int, int] = {}  # canonical key -> estimate sample size (absent ⇒ 0)
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._v)

    def __contains__(self, key: int) -> bool:
        return key in self._v

    def keys(self):
        return self._v.keys()

    def get(self, key: int) -> Optional[int]:
        """The raw stored value (exact value / TT bound / scaled estimate), or None if absent. Unchanged from the
        original store — exact opening-book / TT consumers that require a PROOF must use `proven_value`."""
        v = self._v.get(key)
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def is_proven(self, key: int) -> bool:
        return key in self._v and self._s.get(key, PROVEN) == PROVEN

    def proven_value(self, key: int) -> Optional[int]:
        """The exact value if this key holds a PROOF, else None — the getter every EXACT consumer uses, so a
        mere ESTIMATE is never mistaken for game-theoretic truth (one-ply lookahead / the solver short-circuit)."""
        if self._s.get(key, PROVEN) != PROVEN:
            return None
        return self._v.get(key)

    def entry(self, key: int) -> Optional[Entry]:
        if key not in self._v:
            return None
        status = self._s.get(key, PROVEN)
        raw = self._v[key]
        value = float(raw) if status == PROVEN else raw / _EST_SCALE
        return Entry(status=status, value=value, best_actions=self._a.get(key, 0), n=self._n.get(key, 0))

    def put(self, key: int, value: int, priority: int = 0) -> None:
        """Store a PROVEN value (backward-compatible: exact opening-book entries + the solver TT accelerator)."""
        self._store(key, int(value), PROVEN, best_actions=0, n=0, priority=priority)

    def put_proven(self, key: int, value: int, best_actions: int = 0, priority: int = 0) -> None:
        self._store(key, int(value), PROVEN, best_actions=int(best_actions), n=0, priority=priority)

    def put_estimate(self, key: int, value: float, best_actions: int = 0, n: int = 0, priority: int = 0) -> None:
        scaled = max(-_EST_SCALE, min(_EST_SCALE, round(float(value) * _EST_SCALE)))
        self._store(key, scaled, ESTIMATE, best_actions=int(best_actions), n=int(n), priority=priority)

    def _store(self, key: int, raw: int, status: int, best_actions: int, n: int, priority: int) -> None:
        if key not in self._v and len(self._v) >= self.cap:
            self._evict()
        self._v[key] = raw
        if status == PROVEN:
            self._s.pop(key, None)  # ABSENT ⇒ PROVEN, keeps the common case column-free
        else:
            self._s[key] = status
        if best_actions:
            self._a[key] = best_actions
        else:
            self._a.pop(key, None)
        if n:
            self._n[key] = n
        else:
            self._n.pop(key, None)
        # keep the HIGHEST priority ever seen for a key (a hub position reached many ways stays)
        prev = self._p.get(key)
        self._p[key] = int(priority) if prev is None else max(prev, int(priority))

    def _evict(self) -> None:
        # drop ~1% of the lowest-priority entries in one pass (cheap amortised eviction)
        n_drop = max(1, len(self._v) // 100)
        for key in sorted(self._p, key=self._p.__getitem__)[:n_drop]:
            self._v.pop(key, None)
            self._p.pop(key, None)
            self._s.pop(key, None)
            self._a.pop(key, None)
            self._n.pop(key, None)

    # --- persistence -------------------------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        keys = list(self._v.keys())
        if _np is not None:
            _np.savez_compressed(
                path if path.endswith(".npz") else path + ".npz",
                keys=_np.array(keys, dtype=_np.uint64),
                vals=_np.array([self._v[k] for k in keys], dtype=_np.int32),
                pris=_np.array([self._p.get(k, 0) for k in keys], dtype=_np.int32),
                stats=_np.array([self._s.get(k, PROVEN) for k in keys], dtype=_np.int8),
                acts=_np.array([self._a.get(k, 0) for k in keys], dtype=_np.uint64),
                ns=_np.array([self._n.get(k, 0) for k in keys], dtype=_np.int32),
            )
        else:  # pragma: no cover - numpy is always present in the venv
            import json

            with open(path, "w") as f:
                json.dump(
                    {
                        "v": {str(k): self._v[k] for k in keys},
                        "p": {str(k): self._p.get(k, 0) for k in keys},
                        "s": {str(k): self._s.get(k, PROVEN) for k in keys},
                        "a": {str(k): self._a.get(k, 0) for k in keys},
                        "n": {str(k): self._n.get(k, 0) for k in keys},
                    },
                    f,
                )

    @classmethod
    def load(cls, path: str, cap: int = 20_000_000) -> "Tablebase":
        tb = cls(cap=cap)
        npz = path if path.endswith(".npz") else path + ".npz"
        if _np is not None and os.path.isfile(npz):
            data = _np.load(npz)
            keys = data["keys"].tolist()
            vals = data["vals"].tolist()
            pris = data["pris"].tolist()
            # Legacy (value-only) books + the solver .tt accelerator have no status/best-action columns → PROVEN.
            stats = data["stats"].tolist() if "stats" in data.files else [PROVEN] * len(keys)
            acts = data["acts"].tolist() if "acts" in data.files else [0] * len(keys)
            ns = data["ns"].tolist() if "ns" in data.files else [0] * len(keys)
            for k, v, p, s, a, n in zip(keys, vals, pris, stats, acts, ns):
                tb._v[int(k)] = int(v)
                tb._p[int(k)] = int(p)
                if int(s) != PROVEN:
                    tb._s[int(k)] = int(s)
                if int(a):
                    tb._a[int(k)] = int(a)
                if int(n):
                    tb._n[int(k)] = int(n)
        elif os.path.isfile(path):  # pragma: no cover
            import json

            with open(path) as f:
                d = json.load(f)
            tb._v = {int(k): int(v) for k, v in d.get("v", {}).items()}
            tb._p = {int(k): int(v) for k, v in d.get("p", {}).items()}
            tb._s = {int(k): int(v) for k, v in d.get("s", {}).items() if int(v) != PROVEN}
            tb._a = {int(k): int(v) for k, v in d.get("a", {}).items() if int(v)}
            tb._n = {int(k): int(v) for k, v in d.get("n", {}).items() if int(v)}
        return tb
