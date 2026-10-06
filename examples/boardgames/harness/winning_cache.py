"""The oracle's winning moves remembered across the searches of one build (§3.6): local search scores every map by a
walk that asks the oracle at each newly reached position, and siblings reach the same positions, so a shared cache
turns most asks into lookups. Bounded: when a batch would take it past `limit` entries it is emptied first (an
oracle cache reached ~4 GB a worker at ~2M positions, h172)."""
from __future__ import annotations


class CachedWinning:
    def __init__(self, winning, key, limit: int):
        self.winning, self.key, self.limit = winning, key, limit
        self.cache: dict = {}
        self.cleared = 0

    def __call__(self, states) -> list:
        keys = [self.key(s) for s in states]
        known = {k: self.cache[k] for k in keys if k in self.cache}
        missing = {k: s for k, s in zip(keys, states) if k not in known}
        if missing:
            if len(self.cache) + len(missing) > self.limit:
                self.cache.clear()
                self.cleared += 1
            fresh = dict(zip(missing, self.winning(list(missing.values()))))
            self.cache.update(fresh)
            known.update(fresh)
        return [known[k] for k in keys]
