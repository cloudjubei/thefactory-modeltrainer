"""The transcription guard for Nine Men's Morris geometry (§C.37, step 1): the constants are GENERATED, and this
regenerates their invariants independently so a wrong index formula cannot ship. Everything the rules do indexes
into these tables, so they are the foundation gate."""
from __future__ import annotations

from games import ninemensmorris as M


def test_twenty_four_points_on_distinct_grid_cells():
    assert len(M.POINT_CELL) == 24
    assert len(set(M.POINT_CELL)) == 24                      # injective embedding
    assert all(0 <= c < 49 for c in M.POINT_CELL)
    assert M.CELL_POINT == {c: p for p, c in enumerate(M.POINT_CELL)}


def test_valid_mask_marks_exactly_the_24_point_cells():
    assert len(M.VALID_MASK) == 49 and sum(M.VALID_MASK) == 24
    assert all((M.VALID_MASK[c] == 1.0) == (c in M.CELL_POINT) for c in range(49))


def test_adjacency_is_symmetric_with_the_right_degree_histogram():
    # 32 undirected edges; the four middle-ring midpoints (9,11,13,15) have degree 4, the eight other spoke
    # endpoints degree 3, the twelve corners degree 2 — a distinctive fingerprint a mis-indexed graph fails.
    edges = {frozenset((p, q)) for p in range(24) for q in M.ADJ[p]}
    assert all(p in M.ADJ[q] for p in range(24) for q in M.ADJ[p])   # symmetric
    assert all(p not in M.ADJ[p] for p in range(24))                 # no self-loops
    assert len(edges) == 32
    from collections import Counter
    hist = Counter(len(M.ADJ[p]) for p in range(24))
    assert dict(hist) == {2: 12, 3: 8, 4: 4}
    assert {p for p in range(24) if len(M.ADJ[p]) == 4} == {9, 11, 13, 15}


def test_sixteen_mills_each_a_valid_line_and_every_point_in_exactly_two():
    assert len(M.MILLS) == 16
    assert len(set(map(frozenset, M.MILLS))) == 16                   # distinct
    for m in M.MILLS:
        assert len(set(m)) == 3 and all(0 <= p < 24 for p in m)
    from collections import Counter
    membership = Counter(p for m in M.MILLS for p in m)
    assert all(membership[p] == 2 for p in range(24))
    assert all(len(M.MILLS_THROUGH[p]) == 2 for p in range(24))


def test_mill_points_are_actually_collinear_on_the_grid():
    # a real mill is three grid cells sharing a row or a column (morris lines are straight); this catches a
    # triple that is a valid index set but not a geometric line.
    for m in M.MILLS:
        rc = [divmod(M.POINT_CELL[p], 7) for p in m]
        rows = {r for r, _ in rc}
        cols = {c for _, c in rc}
        assert len(rows) == 1 or len(cols) == 1


def test_ring_and_spoke_mill_shapes_are_both_present():
    # vary the FIXTURE SHAPE: the 12 ring-side triples live within one ring; the 4 spoke triples cross all three.
    ring_mills = [m for m in M.MILLS if len({p // 8 for p in m}) == 1]
    spoke_mills = [m for m in M.MILLS if len({p // 8 for p in m}) == 3]
    assert len(ring_mills) == 12 and len(spoke_mills) == 4
    assert (0, 1, 2) in [tuple(m) for m in M.MILLS]                  # an outer-ring corner
    assert any(set(m) == {1, 9, 17} for m in M.MILLS)               # the N spoke across rings
