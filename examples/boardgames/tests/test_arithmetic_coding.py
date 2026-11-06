"""Direct tests for harness/arithmetic_coding.py — an exact arithmetic coder over adaptive symbols: what goes in comes
out, a decoder needs nothing but the bytes and the same calls, and the file is as long as the symbols' probabilities
say, to within the coder's few closing bits."""
from __future__ import annotations

import math
import random

import pytest

from harness.arithmetic_coding import ALPHA_DEN, ALPHA_NUM, Decoder, Encoder


def _round_trip(calls):
    e = Encoder()
    for family, context, n, value in calls:
        (e.uniform(family, n, value) if context is None else e.symbol(family, context, n, value))
    data, bits = e.finish()
    d = Decoder(data)
    out = [d.uniform(family, n) if context is None else d.symbol(family, context, n) for family, context, n, _ in calls]
    return [c[3] for c in calls], out, bits, e


def test_symbols_and_uniform_choices_come_back_in_order():
    r = random.Random(1)
    calls = [("a", r.randrange(3), 5, r.randrange(5)) if r.random() < 0.7 else ("b", None, 11, r.randrange(11))
             for _ in range(3_000)]
    sent, got, _, _ = _round_trip(calls)
    assert got == sent


def test_a_skewed_source_takes_about_its_entropy():
    r = random.Random(2)
    calls = [("a", 0, 2, int(r.random() < 0.05)) for _ in range(20_000)]
    sent, got, bits, _ = _round_trip(calls)
    assert got == sent
    entropy = -(0.05 * math.log2(0.05) + 0.95 * math.log2(0.95)) * len(calls)
    assert entropy * 0.9 < bits < entropy * 1.1


def test_contexts_learn_apart():
    calls = [("a", i % 2, 2, i % 2) for i in range(4_000)]
    assert _round_trip(calls)[2] < 60


def test_a_uniform_choice_costs_its_logarithm():
    calls = [("b", None, 8, i % 8) for i in range(1_000)]
    assert 3_000 <= _round_trip(calls)[2] <= 3_004


def test_one_choice_costs_nothing_and_comes_back_as_zero():
    e = Encoder()
    assert e.symbol("a", 0, 1, 0) == 0 and e.uniform("b", 1, 0) == 0
    assert e.finish()[1] <= 2
    d = Decoder(b"")
    assert d.symbol("a", 0, 1) == 0 and d.uniform("b", 1) == 0


def test_the_first_symbol_of_a_context_is_priced_by_the_prior():
    e = Encoder()
    e.symbol("a", 0, 4, 2)
    assert e.parts == {"a": pytest.approx(2.0)}
    e.symbol("a", 0, 4, 2)
    seed, step = ALPHA_NUM, ALPHA_DEN
    assert e.parts["a"] == pytest.approx(2.0 - math.log2((seed + step) / (4 * seed + step)))


def test_the_parts_add_up_to_the_file():
    r = random.Random(3)
    calls = [("a", 0, 7, r.randrange(7)) for _ in range(500)] + [("b", None, 5, r.randrange(5)) for _ in range(500)]
    *_, bits, e = _round_trip(calls)
    assert set(e.parts) == {"a", "b"} and abs(sum(e.parts.values()) - bits) < 4


def test_values_outside_the_alphabet_are_refused():
    e = Encoder()
    for bad in (-1, 3):
        with pytest.raises(ValueError):
            e.symbol("a", 0, 3, bad)
        with pytest.raises(ValueError):
            e.uniform("a", 3, bad)


def test_a_decoder_reads_zeros_past_the_end_of_the_file():
    calls = [("a", 0, 2, 0) for _ in range(2_000)]
    sent, got, bits, _ = _round_trip(calls)
    assert got == sent and bits < 40


def test_a_file_never_ends_in_a_zero_bit():
    r = random.Random(4)
    for _ in range(200):
        e = Encoder()
        for _ in range(r.randrange(1, 40)):
            e.uniform("a", 5, r.randrange(5))
        data, bits = e.finish()
        assert bits == 0 or (data[(bits - 1) // 8] >> (7 - (bits - 1) % 8)) & 1


def test_short_files_read_back():
    r = random.Random(5)
    for _ in range(500):
        calls = [("a", 0, 3, r.randrange(3)) for _ in range(r.randrange(1, 12))]
        sent, got, _, _ = _round_trip(calls)
        assert got == sent


def test_the_bit_count_is_within_a_byte_of_the_file():
    e = Encoder()
    for i in range(1_000):
        e.uniform("a", 3, i % 3)
    data, bits = e.finish()
    assert (bits + 7) // 8 == len(data)
