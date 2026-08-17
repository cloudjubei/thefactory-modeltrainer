from harness.tablebase import Tablebase


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
