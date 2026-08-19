import numpy as np

from harness.tablebase import ESTIMATE, PROVEN, Entry, Tablebase


def test_estimates_are_invisible_to_exact_consumers_but_readable_as_entries():
    tb = Tablebase(cap=100)
    tb.put_proven(1, 1, best_actions=0b0101, priority=5)  # a PROVEN win, optimal moves {0, 2}
    tb.put_estimate(2, 0.4, best_actions=0b0010, n=20, priority=1)  # a BELIEF, not a proof
    # exact consumers use proven_value → they see ONLY the proof; an estimate is never trusted as exact
    assert tb.proven_value(1) == 1 and tb.is_proven(1)
    assert tb.proven_value(2) is None and not tb.is_proven(2)
    # the richer entry exposes both, with best_actions + confidence a deep model can read
    e1 = tb.entry(1)
    assert e1 == Entry(status=PROVEN, value=1.0, best_actions=0b0101, n=0)
    e2 = tb.entry(2)
    assert e2.status == ESTIMATE and abs(e2.value - 0.4) < 1e-3 and e2.best_actions == 0b0010 and e2.n == 20
    assert 1 in tb and 2 in tb


def test_richer_entries_survive_persistence(tmp_path):
    tb = Tablebase(cap=100)
    tb.put_proven(1, -1, best_actions=0b1, priority=2)
    tb.put_estimate(2, -0.3, best_actions=0b100, n=8, priority=1)
    p = str(tmp_path / "book")
    tb.save(p)
    tb2 = Tablebase.load(p)
    assert tb2.entry(1) == Entry(status=PROVEN, value=-1.0, best_actions=0b1, n=0)
    e = tb2.entry(2)
    assert e.status == ESTIMATE and abs(e.value + 0.3) < 1e-3 and e.n == 8


def test_load_tolerates_legacy_value_only_npz_as_all_proven(tmp_path):
    # the committed books + the solver .tt accelerator were saved value-only; they must load as PROVEN.
    p = str(tmp_path / "legacy")
    np.savez_compressed(
        p + ".npz",
        keys=np.array([9], dtype=np.uint64),
        vals=np.array([42], dtype=np.int32),
        pris=np.array([0], dtype=np.int32),
    )
    tb = Tablebase.load(p)
    assert tb.is_proven(9) and tb.get(9) == 42 and tb.proven_value(9) == 42 and tb.entry(9).best_actions == 0


def test_put_get_contains_and_miss():
    tb = Tablebase(cap=100)
    tb.put(7, 42, priority=3)
    assert tb.get(7) == 42 and 7 in tb
    assert tb.get(999) is None and 999 not in tb
    assert tb.hits == 1 and tb.misses == 1


def test_persistence_roundtrip(tmp_path):
    tb = Tablebase(cap=100)
    tb.put(1, -18, priority=1)
    tb.put(2, 18, priority=9)
    p = str(tmp_path / "book")
    tb.save(p)
    tb2 = Tablebase.load(p)
    assert len(tb2) == 2 and tb2.get(1) == -18 and tb2.get(2) == 18


def test_priority_eviction_keeps_the_hubs():
    tb = Tablebase(cap=5)
    tb.put(2, 20, priority=100)  # a "hub" — expensive/high in-degree, must survive
    for k in range(10, 40):  # flood with low-priority entries
        tb.put(k, k, priority=0)
    assert len(tb) <= 5
    assert tb.get(2) == 20  # the high-priority hub was kept while low-priority entries were evicted
