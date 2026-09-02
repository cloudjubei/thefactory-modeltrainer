"""§C.8 #5 (cheap form) — refutation-replay, CP's nogood/conflict-learning mapped onto self-play: when an
opponent REFUTES an opening the learner entered (learner loses from it), the opening's action prefix is stored
as a nogood and FORCE-replayed in a fraction of subsequent self-play games — exact-line pressure exactly where
the learner was just proven exploitable — until the learner stops losing it. The strong form of this lever is
the learned-adversary exploitability-descent loop; this store is its bookkeeping and works with any opponent."""
from __future__ import annotations

import random


class RefutationStore:
    """Bounded store of refuted opening lines. `add` on a loss (deduped, capped FIFO), `sample` for replay,
    `resolve` after each replay — a line retires only after `retire_after` CONSECUTIVE non-losses, so one lucky
    game does not count as fixed."""

    def __init__(self, cap: int = 64, retire_after: int = 3):
        self.cap = int(cap)
        self.retire_after = int(retire_after)
        self._lines: dict[tuple[int, ...], dict] = {}

    def __len__(self) -> int:
        return len(self._lines)

    def add(self, prefix: tuple[int, ...]) -> None:
        prefix = tuple(int(a) for a in prefix)
        if not prefix:
            return
        entry = self._lines.pop(prefix, {"refuted": 0, "streak": 0})
        entry["refuted"] += 1
        entry["streak"] = 0  # freshly refuted → any survival streak restarts
        self._lines[prefix] = entry  # (re)insert at the end: newest-refuted is last to be evicted
        while len(self._lines) > self.cap:
            self._lines.pop(next(iter(self._lines)))

    def sample(self, rng: random.Random) -> tuple[int, ...] | None:
        if not self._lines:
            return None
        return rng.choice(list(self._lines.keys()))

    def resolve(self, prefix: tuple[int, ...], lost: bool) -> None:
        entry = self._lines.get(tuple(prefix))
        if entry is None:
            return
        if lost:
            entry["refuted"] += 1
            entry["streak"] = 0
            # freshen eviction order: a line STILL being lost is the newest refutation, last to be evicted
            # (review-confirmed: without this, eviction order was oldest-ADDED, not oldest-REFUTED)
            del self._lines[tuple(prefix)]
            self._lines[tuple(prefix)] = entry
        else:
            entry["streak"] += 1
            if entry["streak"] >= self.retire_after:
                del self._lines[tuple(prefix)]

    def to_json(self) -> dict:
        """Serializable snapshot (batched runs persist the store so a RESUME keeps its nogoods)."""
        return {"cap": self.cap, "retire_after": self.retire_after,
                "lines": [{"prefix": list(p), **e} for p, e in self._lines.items()]}

    @classmethod
    def from_json(cls, blob: dict) -> "RefutationStore":
        st = cls(cap=int(blob.get("cap", 64)), retire_after=int(blob.get("retire_after", 3)))
        for line in blob.get("lines", []):
            st._lines[tuple(int(a) for a in line["prefix"])] = {
                "refuted": int(line.get("refuted", 1)), "streak": int(line.get("streak", 0))}
        return st
