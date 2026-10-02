"""Direct tests for harness/rule_search.py — §C.50 E2: the smallest ordered list of conjunctive rules that recommends
only optimal moves and covers every position, found and proven minimal by SAT."""
from __future__ import annotations

import pytest

from harness.rule_search import Case, decode_rules, minimal_playbook, solve_k

NAMES = ("a", "b", "c")


def _case(profiles, optimal):
    """`profiles`: one tuple of predicate truth values per move; `optimal`: the indices of the optimal moves."""
    return Case(tuple(tuple(p) for p in profiles), frozenset(optimal))


def _plays(rules, case):
    for lits in rules:
        recommended = {i for i, p in enumerate(case.profiles) if all(p[k] == v for k, v in lits)}
        if recommended:
            return recommended
    return set()


def _perfect(rules, cases):
    return all(_plays(rules, c) and _plays(rules, c) <= c.optimal for c in cases)


ONE_RULE = [_case([(1, 0, 0), (0, 0, 0)], {0}), _case([(1, 1, 0), (0, 1, 1)], {0})]
TWO_RULES = [_case([(1, 0, 0), (0, 0, 0)], {0}), _case([(0, 1, 0), (0, 0, 0)], {0})]


def test_one_literal_can_be_enough():
    r = minimal_playbook(ONE_RULE, len(NAMES), max_rules=4)
    assert r["k"] == 1 and r["proven_minimal"] and _perfect(r["rules"], ONE_RULE)
    assert r["literals"] == 1 and r["literals_proven_minimal"]


def test_the_smallest_list_is_found_and_one_rule_fewer_is_proven_impossible():
    r = minimal_playbook(TWO_RULES, len(NAMES), max_rules=4)
    assert r["k"] == 2 and r["proven_minimal"] and _perfect(r["rules"], TWO_RULES)
    assert solve_k(TWO_RULES, len(NAMES), 1) is None
    assert r["literals"] == 2 and solve_k(TWO_RULES, len(NAMES), 2, max_literals=1) is None


def test_order_matters_a_later_rule_may_recommend_a_bad_move_only_where_an_earlier_one_fires():
    cases = [_case([(1, 1, 0), (0, 1, 0)], {0}), _case([(0, 1, 0), (0, 0, 0)], {0})]
    rules = solve_k(cases, len(NAMES), 2)
    assert rules is not None and _perfect(rules, cases)
    assert solve_k(cases, len(NAMES), 1) is None


def test_one_separable_optimal_move_is_enough_to_be_playable():
    case = _case([(1, 0, 0), (1, 0, 0), (0, 1, 0)], {0, 2})
    r = minimal_playbook([case], len(NAMES), max_rules=2)
    assert r["inseparable"] == [] and r["k"] == 1 and _perfect(r["rules"], [case])


def test_positions_where_no_list_can_separate_the_moves_have_no_playbook():
    cases = [_case([(1, 0, 0), (1, 0, 0)], {0})]
    r = minimal_playbook(cases, len(NAMES), max_rules=3)
    assert r["k"] is None and r["impossible_up_to"] == 3
    assert r["inseparable"] == [0]


def test_rules_print_as_named_literals():
    assert decode_rules([[(0, True), (2, False)]], NAMES) == [(("a", True), ("c", False))]


def test_a_case_with_no_optimal_move_is_refused():
    with pytest.raises(ValueError, match="optimal"):
        _case([(1, 0, 0)], set()).check()
    with pytest.raises(ValueError, match="optimal"):
        minimal_playbook([_case([(1, 0, 0)], set())], len(NAMES), max_rules=2)


def test_every_rule_names_at_least_one_literal_as_the_language_requires():
    indifferent = _case([(0, 0, 0), (0, 0, 0)], {0, 1})
    r = minimal_playbook([indifferent], len(NAMES), max_rules=2)
    assert r["k"] == 1 and r["literals"] == 1 and all(r["rules"])
