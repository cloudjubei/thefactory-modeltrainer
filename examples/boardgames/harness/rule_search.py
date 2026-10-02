"""§C.50 E2 — the SMALLEST written playbook, found and proven minimal by SAT (the encoding idea of Yu, Ignatiev,
Stuckey & Le Bodic 2021, for ordered rule lists).

The language is harness.playbook's: a rule is a conjunction of at least one predicate or negation, and it recommends
every legal move meeting all its literals; a playbook is an ordered list, and the first rule that recommends anything
decides. A CASE is one position reduced to what the language can see: each legal move's predicate profile (a tuple of
truth values) and which moves are optimal. A playbook is PERFECT on a set of cases when, at every case, some rule
fires and the first one that does recommends only optimal moves.

`solve_k` asks a SAT solver for a perfect playbook of exactly `k` rules (optionally with at most `max_literals`
literals in all); `minimal_playbook` tries k = 1, 2, … and then shrinks the literal count, so the answer it returns
is proven minimal relative to the language — UNSAT at k − 1 rules, and at one literal fewer.

Requires python-sat (MIT)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    profiles: tuple
    optimal: frozenset

    def check(self) -> None:
        if not self.optimal or not self.optimal <= set(range(len(self.profiles))):
            raise ValueError("a case needs at least one optimal move among its moves")


def _reduce(cases: list) -> list:
    """Each case as (its profiles, the profiles that recommend a non-optimal move), duplicates dropped."""
    out = set()
    for c in cases:
        bad = frozenset(p for i, p in enumerate(c.profiles) if i not in c.optimal)
        out.add((frozenset(c.profiles), bad))
    return sorted(out, key=lambda t: (sorted(t[0]), sorted(t[1])))


def inseparable(cases: list) -> list:
    """Indices of cases no playbook can play: every profile carrying an optimal move also carries a bad one."""
    out = []
    for i, c in enumerate(cases):
        bad = {p for j, p in enumerate(c.profiles) if j not in c.optimal}
        if all(c.profiles[j] in bad for j in c.optimal):
            out.append(i)
    return out


def solve_k(cases: list, n_predicates: int, k: int, max_literals: int | None = None) -> list | None:
    """A perfect playbook of `k` rules as [[(predicate index, required value), …], …], or None when none exists."""
    from pysat.card import CardEnc, EncType
    from pysat.formula import IDPool
    from pysat.solvers import Cadical153

    reduced = _reduce(cases)
    profiles = sorted({p for allp, _bad in reduced for p in allp})
    pool = IDPool()
    x = {(j, p, v): pool.id(("x", j, p, v)) for j in range(k) for p in range(n_predicates) for v in (True, False)}
    y = {(j, c): pool.id(("y", j, c)) for j in range(k) for c in profiles}
    clauses = []
    for j in range(k):
        for p in range(n_predicates):
            clauses.append([-x[j, p, True], -x[j, p, False]])
        clauses.append([x[j, p, v] for p in range(n_predicates) for v in (True, False)])
        for c in profiles:
            failing = [x[j, p, v] for p in range(n_predicates) for v in (True, False) if bool(c[p]) != v]
            clauses += [[-y[j, c], -lit] for lit in failing]
            clauses.append([y[j, c]] + failing)
    for allp, bad in reduced:
        clauses.append([y[j, c] for j in range(k) for c in allp])
        for j in range(k):
            earlier = [y[i, c] for i in range(j) for c in allp]
            clauses += [[-y[j, c]] + earlier for c in bad]
    if max_literals is not None:
        card = CardEnc.atmost(lits=list(x.values()), bound=max_literals, vpool=pool, encoding=EncType.seqcounter)
        clauses += card.clauses
    with Cadical153(bootstrap_with=clauses) as solver:
        if not solver.solve():
            return None
        model = set(lit for lit in solver.get_model() if lit > 0)
    return [[(p, v) for p in range(n_predicates) for v in (True, False) if x[j, p, v] in model] for j in range(k)]


def minimal_playbook(cases: list, n_predicates: int, max_rules: int) -> dict:
    """The fewest rules (then the fewest literals) of a perfect playbook, each proven minimal, or the bound up to
    which none exists."""
    for c in cases:
        c.check()
    blocked = inseparable(cases)
    if blocked:
        return {"k": None, "rules": None, "proven_minimal": False, "impossible_up_to": max_rules,
                "inseparable": blocked}
    for k in range(1, max_rules + 1):
        rules = solve_k(cases, n_predicates, k)
        if rules is None:
            continue
        literals = sum(len(r) for r in rules)
        while literals > 0:
            fewer = solve_k(cases, n_predicates, k, max_literals=literals - 1)
            if fewer is None or sum(len(r) for r in fewer) >= literals:
                break
            rules, literals = fewer, sum(len(r) for r in fewer)
        return {"k": k, "rules": rules, "proven_minimal": True, "impossible_up_to": k - 1, "literals": literals,
                "literals_proven_minimal": True, "inseparable": []}
    return {"k": None, "rules": None, "proven_minimal": False, "impossible_up_to": max_rules, "inseparable": []}


def decode_rules(rules: list, names: tuple) -> list:
    """Rules with predicate indices replaced by names."""
    return [tuple((names[p], v) for p, v in r) for r in rules]
