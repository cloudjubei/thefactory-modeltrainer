"""§C.8 #5 (cheap form) — refutation-replay: openings an opponent REFUTES (the learner entered and lost) are
stored as nogoods and force-replayed in self-play until the learner stops losing them. CP's conflict-learning
mapped onto self-play: 'learn a clause so you never lose that line again'."""
import random

from harness.refutation import RefutationStore


def test_store_records_dedupes_and_samples():
    st = RefutationStore(cap=4)
    st.add((3, 3, 2))
    st.add((3, 3, 2))  # same line refuted again → one entry, counted
    st.add((0, 1))
    assert len(st) == 2
    rng = random.Random(0)
    seen = {st.sample(rng) for _ in range(20)}
    assert seen == {(3, 3, 2), (0, 1)}  # sampling covers the live nogoods
    assert RefutationStore(cap=4).sample(rng) is None  # empty store → nothing to replay


def test_resolve_retires_a_line_after_consecutive_survivals():
    # A nogood retires only after `retire_after` CONSECUTIVE non-losses — one lucky game is not "fixed".
    st = RefutationStore(cap=4, retire_after=2)
    st.add((3, 3))
    st.resolve((3, 3), lost=False)
    assert len(st) == 1  # one survival is not enough
    st.resolve((3, 3), lost=True)  # lost again → streak resets
    st.resolve((3, 3), lost=False)
    assert len(st) == 1
    st.resolve((3, 3), lost=False)
    assert len(st) == 0  # two consecutive survivals → retired


def test_cap_evicts_the_oldest_line():
    st = RefutationStore(cap=2)
    st.add((0,))
    st.add((1,))
    st.add((2,))  # over cap → the oldest refutation is evicted
    assert len(st) == 2
    rng = random.Random(1)
    assert {st.sample(rng) for _ in range(20)} == {(1,), (2,)}


def test_resolve_lost_freshens_eviction_order():
    # Review-confirmed: eviction order must be oldest-REFUTED, not oldest-ADDED — a line still being lost on
    # every replay is the freshest refutation and must be the LAST to be evicted.
    st = RefutationStore(cap=2)
    st.add((0,))
    st.add((1,))
    st.resolve((0,), lost=True)  # (0,) is now the most recently refuted
    st.add((2,))  # over cap → evict the oldest-refuted, which is (1,), NOT (0,)
    rng = random.Random(2)
    assert {st.sample(rng) for _ in range(20)} == {(0,), (2,)}


def test_from_json_roundtrips_lines_and_stats():
    st = RefutationStore(cap=8, retire_after=2)
    st.add((3, 3))
    st.resolve((3, 3), lost=False)
    back = RefutationStore.from_json(st.to_json())
    assert len(back) == 1 and back.retire_after == 2
    back.resolve((3, 3), lost=False)  # streak carried over the roundtrip → one more survival retires it
    assert len(back) == 0
