"""A persistent, game-AGNOSTIC solved-position store — the opening-book / endgame-tablebase backing that makes
a solved game *computably* optimal (see docs/optimal-play-trainer-plan.md).

It knows nothing about any game: a game supplies canonical integer keys (symmetry-reduced) and the values'
meaning (e.g. a game-theoretic score, or a search bound). Bounded with PRIORITY eviction — when full, the
lowest-priority entries go first, so the expensive/hub positions (which cost the most to recompute) are kept.
Persisted as a compact two-array file so it accumulates across runs ("over time we store the difficult ones").
"""
from __future__ import annotations

import os
from typing import Optional

try:
    import numpy as _np
except Exception:  # numpy is present in the trainer venv; degrade to JSON if ever absent
    _np = None


class Tablebase:
    def __init__(self, cap: int = 20_000_000):
        self.cap = int(cap)
        self._v: dict[int, int] = {}  # canonical key -> value
        self._p: dict[int, int] = {}  # canonical key -> priority (higher = keep)
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._v)

    def __contains__(self, key: int) -> bool:
        return key in self._v

    def get(self, key: int) -> Optional[int]:
        v = self._v.get(key)
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, key: int, value: int, priority: int = 0) -> None:
        if key not in self._v and len(self._v) >= self.cap:
            self._evict()
        self._v[key] = int(value)
        # keep the HIGHEST priority ever seen for a key (a hub position reached many ways stays)
        prev = self._p.get(key)
        self._p[key] = int(priority) if prev is None else max(prev, int(priority))

    def _evict(self) -> None:
        # drop ~1% of the lowest-priority entries in one pass (cheap amortised eviction)
        n_drop = max(1, len(self._v) // 100)
        for key in sorted(self._p, key=self._p.__getitem__)[:n_drop]:
            self._v.pop(key, None)
            self._p.pop(key, None)

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
            )
        else:  # pragma: no cover - numpy is always present in the venv
            import json

            with open(path, "w") as f:
                json.dump({"v": {str(k): self._v[k] for k in keys}, "p": {str(k): self._p.get(k, 0) for k in keys}}, f)

    @classmethod
    def load(cls, path: str, cap: int = 20_000_000) -> "Tablebase":
        tb = cls(cap=cap)
        npz = path if path.endswith(".npz") else path + ".npz"
        if _np is not None and os.path.isfile(npz):
            data = _np.load(npz)
            for k, v, p in zip(data["keys"].tolist(), data["vals"].tolist(), data["pris"].tolist()):
                tb._v[int(k)] = int(v)
                tb._p[int(k)] = int(p)
        elif os.path.isfile(path):  # pragma: no cover
            import json

            with open(path) as f:
                d = json.load(f)
            tb._v = {int(k): int(v) for k, v in d.get("v", {}).items()}
            tb._p = {int(k): int(v) for k, v in d.get("p", {}).items()}
        return tb
