"""Direct tests for harness/self_proof.py — proving won positions with no solver: a map is scored by the lines it
loses (`failing`), and `Prover` proves a position by a map that wins every line or by a move whose every reply it has
proved, refutes one when every safe move fails, and says neither when its budget runs out. The grader here is a plain
minimax over tic-tac-toe, which the prover never sees. Fixtures: FORK, X{0,1} O{2,3}, X to move, which the empty map
wins by the centre and the map WRONG loses; LONE, X{5} O{6}, where only 8 wins and the empty map plays 0."""
from __future__ import annotations

import random
from functools import lru_cache

import pytest

from games.tictactoe import TicTacToe, TTTState
from harness.self_proof import Prover, Replies, aimed_search, aimed_trial, bounded_search, failing, map_search
from harness.steady_state import Facts, choose, verify
from harness.strategy_file import replay

GAME = TicTacToe()
CELLS = [2, 1, 0, 5, 4, 3, 8, 7, 6]
LEVELS = 8
CAP = 100_000


def _facts():
    return Facts(GAME, ties="lowest", unlevelled="last")


def _ttt(x, o):
    board = [0] * 9
    for i in x:
        board[i] = 1
    for i in o:
        board[i] = 2
    return TTTState(board=tuple(board), to_move=0 if len(x) == len(o) else 1, winner=None, done=False)


EMPTY, FORK, LONE = GAME.initial_state(), _ttt((0, 1), (2, 3)), _ttt((5,), (6,))
KEY, LONE_KEY = GAME.state_key(FORK), GAME.state_key(LONE)
WRONG = {5: 0}


@lru_cache(maxsize=None)
def _value(s) -> int:
    """1 when the side to move at `s` wins with best play, -1 when it loses, 0 for a draw."""
    if GAME.is_terminal(s):
        w = GAME.winner(s)
        return 0 if w is None else 1 if w == GAME.current_player(s) else -1
    return max(-_value(GAME.step(s, a)) for a in GAME.legal_actions(s))


def _reachable():
    seen, stack = {}, [EMPTY]
    while stack:
        s = stack.pop()
        if s in seen or GAME.is_terminal(s):
            continue
        seen[s] = True
        stack.extend(GAME.step(s, a) for a in GAME.legal_actions(s))
    return list(seen)


def _never(*_args, **_kwargs):
    raise AssertionError("the solver was asked")


def _prover(search=None, **kwargs):
    return Prover(_facts(), LEVELS, CAP, search=search, **kwargs)


def test_a_map_that_wins_every_line_has_no_failing_position():
    r = failing(_facts(), FORK, {4: 0}, LEVELS, _never, CAP)
    assert r == {"status": "ok", "exceptions": {}, "own_positions": r["own_positions"]} and r["own_positions"] >= 1
    assert verify(_facts(), FORK, {4: 0}, LEVELS)["won"]


def test_a_map_is_charged_at_each_position_whose_move_ends_a_line_unwon():
    facts = _facts()
    r = failing(facts, FORK, WRONG, LEVELS, _never, CAP)
    assert choose(facts, FORK, WRONG, LEVELS) == 5 and not verify(facts, FORK, WRONG, LEVELS)["won"]
    assert r["status"] == "ok" and r["own_positions"] == 10
    assert r["exceptions"] == {GAME.state_key(_ttt(x, o)): None for x, o in [
        ((0, 1, 4, 5), (2, 3, 7, 8)), ((0, 1, 5, 6), (2, 3, 4, 8)), ((0, 1, 5, 6), (2, 3, 4, 7))]}


def test_a_position_the_map_gives_no_move_is_charged():
    facts = Facts(GAME, ties="exact", unlevelled="none")
    assert choose(facts, FORK, {}, LEVELS) is None
    assert failing(facts, FORK, {}, LEVELS, _never, CAP) == {"status": "ok", "exceptions": {KEY: None},
                                                             "own_positions": 1}


def test_a_walk_past_its_cap_or_deadline_has_no_count():
    assert failing(_facts(), EMPTY, {}, LEVELS, _never, 3)["status"] == "cap"
    assert failing(_facts(), EMPTY, {}, LEVELS, _never, CAP, deadline=0.0)["status"] == "timeout"


def test_a_position_a_map_wins_is_proved_by_a_leaf():
    after = GAME.step(GAME.step(FORK, 4), 5)
    p = _prover()
    assert p.prove(after) is True and p.proved[GAME.state_key(after)] == {"leaf": 0}


def test_a_position_is_proved_by_a_move_whose_every_reply_is_proved():
    p = _prover()
    assert not verify(_facts(), LONE, {}, LEVELS)["won"]
    assert p.prove(LONE) is True and p.proved[LONE_KEY] == {"move": 8}
    calls = p.calls
    assert p.prove(LONE) is True and p.calls == calls
    strategy = p.strategy(LONE)
    assert strategy["nodes"][LONE_KEY] == {"move": 8} and len(strategy["nodes"]) > 1
    assert replay(_facts(), LONE, strategy, LEVELS, CAP, CELLS)["won"]


def test_a_position_with_no_winning_move_is_refuted_and_remembered():
    p = _prover()
    assert p.prove(EMPTY) is False and GAME.state_key(EMPTY) in p.refuted and GAME.state_key(EMPTY) not in p.proved
    calls = p.calls
    assert p.prove(EMPTY) is False and p.calls == calls


def test_every_position_is_proved_exactly_when_its_side_wins():
    p = _prover()
    positions = _reachable()
    assert len(positions) == 4_520
    assert all(p.prove(s) is (_value(s) == 1) for s in positions)
    assert (len(p.proved), len(p.refuted)) == (2_836, 1_684)


def test_a_spent_budget_says_neither_and_remembers_nothing():
    p = _prover(max_calls=1)
    assert p.prove(LONE) is None and not p.proved and not p.refuted and p.calls == 1
    assert _prover(max_calls=1).prove(EMPTY) is None
    assert _prover(deadline=0.0).prove(FORK) is None and _prover(max_calls=0).prove(FORK) is None


def test_an_unknown_reply_leaves_the_position_unknown_not_refuted():
    p = _prover(max_calls=3)
    assert p.prove(EMPTY) is None and GAME.state_key(EMPTY) not in p.refuted and p.calls == 3
    p.max_calls = None
    assert p.prove(EMPTY) is False


def test_a_searched_map_with_no_failing_position_becomes_a_leaf_and_is_simplified():
    p = _prover(search=lambda state, start: {"levels": {8: 0, 2: 1}, "exceptions": {}})
    assert p.prove(LONE) is True and p.proved[LONE_KEY] == {"leaf": 1} and p.maps == [{}, {8: 0}]
    assert replay(_facts(), LONE, p.strategy(LONE), LEVELS, CAP, CELLS)["won"]


def test_a_searched_map_is_checked_before_it_is_believed():
    p = _prover(search=lambda state, start: {"levels": {7: 0}, "exceptions": {}})
    assert p.prove(LONE) is True and p.proved[LONE_KEY] == {"move": 8} and p.maps == [{}]


def test_a_searched_map_with_failing_positions_is_no_leaf_but_its_move_is_tried_first():
    tried = []

    class Watching(Prover):
        def _move(self, state, action):
            if state == LONE:
                tried.append(action)
            return super()._move(state, action)

    p = Watching(_facts(), LEVELS, CAP, search=lambda state, start: (
        {"levels": {7: 0}, "exceptions": {LONE_KEY: None}} if state == LONE else None))
    assert p.prove(LONE) is True and p.proved[LONE_KEY] == {"move": 8} and p.maps == [{}]
    assert tried == [7, 0, 1, 2, 3, 4, 8]


def test_a_known_map_is_read_again_before_any_search():
    after = GAME.step(GAME.step(FORK, 4), 5)
    p = _prover(search=_never)
    p.maps.append({8: 0})
    assert p.prove(after) is True and p.proved[GAME.state_key(after)] == {"leaf": 0}
    other = _ttt((5, 7), (3, 6))
    p = _prover(search=_never, reuse_window=1)
    p.maps.append({8: 6})
    assert not verify(_facts(), other, {}, LEVELS)["won"]
    assert p.prove(other) is True and p.proved[GAME.state_key(other)] == {"leaf": 1}
    p = _prover(reuse_window=1)
    p.maps.extend([{8: 6}, {0: 0}])
    assert p.prove(other) is True and "move" in p.proved[GAME.state_key(other)]


def test_the_search_starts_from_the_last_map_on_the_position_s_empty_cells():
    starts = []
    p = _prover(search=lambda state, start: starts.append((state, start)))
    p.maps.append({0: 1, 5: 2, 7: 3})
    p.prove(LONE)
    assert starts[0] == (LONE, {0: 1, 7: 3})


def test_the_search_scores_maps_by_their_failing_positions_and_never_asks_a_solver():
    search = map_search(_facts(), LEVELS, 3, random.Random(0), seconds=5.0, cap=CAP)
    found = search(LONE, {})
    assert found["exceptions"] == {} and verify(_facts(), LONE, found["levels"], LEVELS)["won"]
    p = _prover(search=search)
    assert p.prove(LONE) is True and p.proved[LONE_KEY] == {"leaf": 1}


def test_the_strategy_holds_only_what_the_root_reaches():
    p = _prover()
    p.prove(FORK)
    p.prove(LONE)
    strategy = p.strategy(LONE)
    assert len(strategy["nodes"]) < len(p.proved) and strategy["maps"] is p.maps
    with pytest.raises(KeyError):
        p.strategy(EMPTY)


def test_a_walk_stops_once_it_is_over_its_limit():
    facts = _facts()
    assert len(failing(facts, FORK, WRONG, LEVELS, _never, CAP)["exceptions"]) == 3
    assert failing(facts, FORK, WRONG, LEVELS, _never, CAP, limit=3)["status"] == "ok"
    over = failing(facts, FORK, WRONG, LEVELS, _never, CAP, limit=2)
    assert over["status"] == "over" and over["exceptions"] == {}
    first = failing(facts, FORK, WRONG, LEVELS, _never, CAP, limit=0)
    assert first["status"] == "over" and first["own_positions"] < 10
    assert failing(facts, FORK, {4: 0}, LEVELS, _never, CAP, limit=0)["status"] == "ok"


def test_the_bounded_search_climbs_down_to_a_map_with_no_failing_position():
    search = bounded_search(_facts(), LEVELS, random.Random(1), CAP, evaluations=200)
    found = search(LONE, {})
    assert found["status"] == "found" and found["exceptions"] == {} and 1 < found["evaluations"] <= 200
    assert verify(_facts(), LONE, found["levels"], LEVELS)["won"]
    p = _prover(search=search)
    assert p.prove(LONE) is True and p.proved[LONE_KEY] == {"leaf": 1}


def test_the_bounded_search_never_keeps_a_map_that_fails_more():
    kept = []
    facts = _facts()
    start = len(failing(facts, LONE, {}, LEVELS, _never, CAP)["exceptions"])
    for budget in range(1, 30):
        r = bounded_search(facts, LEVELS, random.Random(3), CAP, evaluations=budget)(LONE, {})
        kept.append(len(r["exceptions"]))
        assert r["evaluations"] <= budget and len(failing(facts, LONE, r["levels"], LEVELS, _never, CAP)[
            "exceptions"]) == len(r["exceptions"])
    assert kept[0] == start > 0 and kept == sorted(kept, reverse=True) and kept[-1] < start


def test_the_bounded_search_starts_from_the_map_it_is_given_and_stops_at_once_when_it_wins():
    search = bounded_search(_facts(), LEVELS, random.Random(0), CAP, evaluations=50)
    assert search(LONE, {8: 0}) == {"status": "found", "levels": {8: 0}, "exceptions": {}, "evaluations": 1,
                                    "own_positions": search(LONE, {8: 0})["own_positions"]}
    assert search(LONE, {})["levels"] != {}


def test_the_bounded_search_gives_nothing_when_its_first_walk_cannot_finish():
    assert bounded_search(_facts(), LEVELS, random.Random(0), 3)(EMPTY, {}) is None
    assert bounded_search(_facts(), LEVELS, random.Random(0), CAP, seconds=-1.0)(EMPTY, {}) is None
    out = bounded_search(_facts(), LEVELS, random.Random(0), CAP, evaluations=1)(LONE, {})
    assert out["status"] == "budget" and out["levels"] == {} and out["evaluations"] == 1


class _Scripted:
    def __init__(self, picks):
        self.picks = list(picks)

    def choice(self, cells):
        return self.picks[0][0]

    def randrange(self, n):
        return self.picks.pop(0)[1]


def test_the_bounded_search_clears_a_cell_and_does_not_count_a_change_that_changes_nothing():
    facts = _facts()
    start = {0: 0, 8: 1}
    assert failing(facts, LONE, start, LEVELS, _never, CAP)["exceptions"]
    found = bounded_search(facts, LEVELS, _Scripted([(0, 0), (0, LEVELS)]), CAP)(LONE, start)
    assert (found["status"], found["levels"], found["evaluations"]) == ("found", {8: 1}, 2)


def test_a_map_is_searched_for_only_where_the_empty_map_s_walk_is_within_the_bound():
    asked = []

    def search(state, start):
        asked.append(state)
        return {"levels": {8: 0}, "exceptions": {}}

    assert failing(_facts(), LONE, {}, LEVELS, _never, CAP)["own_positions"] == 39
    p = _prover(search=search, search_below=38)
    assert p.prove(LONE) is True and p.proved[LONE_KEY] == {"move": 8} and LONE not in asked
    p = _prover(search=search, search_below=39)
    assert p.prove(LONE) is True and p.proved[LONE_KEY] == {"leaf": 1} and asked[-1] == LONE


def test_the_bound_never_keeps_a_known_map_from_proving_a_position():
    after = GAME.step(GAME.step(FORK, 4), 5)
    p = _prover(search=_never, search_below=0)
    assert p.prove(after) is True and p.proved[GAME.state_key(after)] == {"leaf": 0}


class _Counting(TicTacToe):
    def __init__(self):
        super().__init__()
        self.steps = 0

    def step(self, state, action):
        self.steps += 1
        return super().step(state, action)


@pytest.mark.parametrize("root, levels", [(FORK, WRONG), (LONE, {}), (EMPTY, {}), (FORK, {4: 0}),
                                          (EMPTY, {4: 0, 0: 1})])
def test_a_walk_with_remembered_replies_finds_what_a_plain_walk_finds(root, levels):
    facts = _facts()
    plain = failing(facts, root, levels, LEVELS, _never, CAP)
    replies = Replies(GAME)
    assert failing(facts, root, levels, LEVELS, _never, CAP, replies=replies) == plain
    assert failing(facts, root, levels, LEVELS, _never, CAP, replies=replies) == plain
    for limit in (0, 2, 3):
        assert failing(facts, root, levels, LEVELS, _never, CAP, limit=limit) == failing(
            facts, root, levels, LEVELS, _never, CAP, limit=limit, replies=replies)
    assert failing(facts, root, levels, LEVELS, _never, 3, replies=replies) == failing(facts, root, levels, LEVELS,
                                                                                      _never, 3)


def _steps(levels_seq, remember):
    """The game's steps in the last of the walks of `levels_seq` from the empty board, after the rule's facts and any
    remembered replies are warm from the walks before."""
    game = _Counting()
    facts = Facts(game, ties="lowest", unlevelled="last")
    replies = Replies(game) if remember else None
    for levels in levels_seq:
        before = game.steps
        failing(facts, EMPTY, levels, LEVELS, _never, CAP, replies=replies)
    return game.steps - before


def test_remembered_replies_are_stepped_once_however_many_maps_walk_them():
    assert _steps([{}, {}], True) == 0 and _steps([{}, {}], False) > 0
    assert 0 < _steps([{}, {4: 0}], True) < _steps([{}, {4: 0}], False)


def test_remembered_replies_are_forgotten_when_full():
    replies = Replies(GAME, limit=3)
    failing(_facts(), EMPTY, {}, LEVELS, _never, CAP, replies=replies)
    assert 0 < len(replies.memo) <= 3 and replies.cleared > 0
    assert failing(_facts(), EMPTY, {}, LEVELS, _never, CAP, replies=replies) == failing(
        _facts(), EMPTY, {}, LEVELS, _never, CAP)


def test_the_bounded_search_shares_its_replies_and_finds_what_it_found_without():
    shared = Replies(GAME)
    with_memo = bounded_search(_facts(), LEVELS, random.Random(5), CAP, evaluations=60, replies=shared)(LONE, {})
    without = bounded_search(_facts(), LEVELS, random.Random(5), CAP, evaluations=60, replies=None)(LONE, {})
    assert with_memo == without and shared.memo
    own = bounded_search(_facts(), LEVELS, random.Random(5), CAP, evaluations=60)
    assert own(LONE, {}) == without


def test_remembered_replies_never_hold_more_than_their_limit():
    replies = Replies(GAME, limit=3)
    moves = [(LONE, a) for a in _facts().safe_moves(LONE)][:5]
    sizes = []
    for state, move in moves:
        replies(state, GAME.state_key(state), move)
        sizes.append(len(replies.memo))
    assert sizes == [1, 2, 3, 1, 2] and replies.cleared == 1


def test_the_bounded_search_keeps_a_memory_of_its_own_from_its_first_walk_on():
    shared = Replies(GAME)
    bounded_search(_facts(), LEVELS, random.Random(5), CAP, evaluations=1, replies=shared)(LONE, {})
    assert shared.memo

    def steps(**kwargs):
        game = _Counting()
        facts = Facts(game, ties="lowest", unlevelled="last")
        search = bounded_search(facts, LEVELS, random.Random(5), CAP, evaluations=40, **kwargs)
        search(LONE, {})
        before = game.steps
        search(LONE, {})
        return game.steps - before

    assert steps() < steps(replies=None)


def _line(walk, key):
    out = []
    while key is not None:
        state, key = walk["came"][key]
        out.append(state)
    return out[::-1]


@pytest.mark.parametrize("root, levels", [(LONE, {}), (FORK, WRONG), (EMPTY, {})])
def test_a_traced_walk_gives_the_line_to_each_charged_position(root, levels):
    plain = failing(_facts(), root, levels, LEVELS, _never, CAP)
    traced = failing(_facts(), root, levels, LEVELS, _never, CAP, trace=True)
    assert {k: v for k, v in traced.items() if k != "came"} == plain and "came" not in plain
    assert traced["came"][GAME.state_key(root)] == (root, None)
    for key in plain["exceptions"]:
        line = _line(traced, key)
        assert line[0] == root and GAME.state_key(line[-1]) == key
        for a, b in zip(line, line[1:]):
            nexts = [GAME.state_key(c) for c in _replies_of(a, levels)]
            assert GAME.state_key(b) in nexts


def _replies_of(state, levels):
    from harness.self_proof import _replies
    move = choose(_facts(), state, levels, LEVELS)
    return _replies(GAME, GAME.current_player(state), GAME.step(state, move))


@pytest.mark.parametrize("root, levels", [(LONE, {}), (FORK, WRONG), (EMPTY, {}), (EMPTY, {4: 0, 0: 1})])
def test_an_aimed_trial_plays_another_move_on_a_line_that_failed(root, levels):
    facts = _facts()
    walk = failing(facts, root, levels, LEVELS, _never, CAP, trace=True)
    lines = {GAME.state_key(s) for key in walk["exceptions"] for s in _line(walk, key)}
    for seed in range(25):
        trial = aimed_trial(facts, LEVELS, random.Random(seed), levels, walk)
        if trial is None:
            continue
        changed = {c for c in set(trial) | set(levels) if trial.get(c) != levels.get(c)}
        assert 1 <= len(changed) <= 2
        assert any(choose(facts, s, trial, LEVELS) != choose(facts, s, levels, LEVELS)
                   and facts.of(s)[2][choose(facts, s, trial, LEVELS)] in changed
                   for s in (walk["came"][k][0] for k in lines))


def test_an_aimed_trial_has_nothing_to_aim_at_when_no_position_on_a_failed_line_has_a_choice():
    walk = {"exceptions": {"x": None}, "came": {"x": (_ttt((0, 2, 5, 7), (1, 4, 6, 8)), None)}}
    assert aimed_trial(_facts(), LEVELS, random.Random(0), {}, walk) is None


def test_the_aimed_search_finds_a_winning_map():
    for seed in range(4):
        found = aimed_search(_facts(), LEVELS, random.Random(seed), CAP, evaluations=300)(LONE, {})
        assert found["status"] == "found" and verify(_facts(), LONE, found["levels"], LEVELS)["won"]
    assert aimed_search(_facts(), LEVELS, random.Random(0), 3)(EMPTY, {}) is None


@pytest.mark.parametrize("share, aimed_only, random_only",
                         [(1.0, True, False), (0.0, False, True), (0.5, False, False)])
def test_the_aimed_search_aims_its_share_of_the_trials(monkeypatch, share, aimed_only, random_only):
    import harness.self_proof as self_proof
    made = {"aimed": 0, "random": 0}
    real_aimed, real_random = self_proof.aimed_trial, self_proof._random_trial

    def aimed(*args):
        made["aimed"] += 1
        return real_aimed(*args)

    def random_(*args):
        made["random"] += 1
        return real_random(*args)

    monkeypatch.setattr(self_proof, "aimed_trial", aimed)
    monkeypatch.setattr(self_proof, "_random_trial", random_)
    aimed_search(_facts(), LEVELS, random.Random(2), CAP, evaluations=60, share=share)(EMPTY, {})
    assert (made["random"] == 0) == aimed_only and (made["aimed"] == 0) == random_only
    if not aimed_only and not random_only:
        assert made["aimed"] > 5 and made["random"] > 5


OPEN = _ttt((8,), (7,))
OPEN_KEY = GAME.state_key(OPEN)


def _grandchildren(state, move):
    after = GAME.step(state, move)
    return [g for g in (GAME.step(after, b) for b in GAME.legal_actions(after)) if not GAME.is_terminal(g)]


def test_a_walk_does_not_go_past_a_position_it_is_told_to_stop_at():
    facts = _facts()
    plain = failing(facts, LONE, {}, LEVELS, _never, CAP)
    stopped = failing(facts, LONE, {}, LEVELS, _never, CAP, stop=set(plain["exceptions"]))
    assert plain["exceptions"] and stopped["exceptions"] == {} and stopped["own_positions"] < plain["own_positions"]
    assert failing(facts, LONE, {}, LEVELS, _never, CAP, stop={LONE_KEY}) == plain
    assert failing(facts, LONE, {}, LEVELS, _never, CAP, stop=set()) == plain


def test_a_composing_prover_proves_exactly_the_won_positions_and_each_proof_checks():
    from harness.composed import check
    p = Prover(_facts(), LEVELS, CAP, compose=True)
    positions = _reachable()
    assert all(p.prove(s) is (_value(s) == 1) for s in positions)
    assert (len(p.proved), len(p.refuted)) == (2_836, 1_684)
    assert all(check(_facts(), s, p.strategy(s), LEVELS, CAP)["won"] for s in positions if _value(s) == 1)


def test_a_map_need_only_win_the_lines_no_proof_below_already_wins():
    from harness.composed import check
    facts = _facts()
    search = lambda state, start, stop=None: {"levels": {4: 0}, "exceptions": {}} if state == OPEN else None
    assert not verify(facts, OPEN, {4: 0}, LEVELS)["won"]
    plain = Prover(facts, LEVELS, CAP, search=lambda state, start: search(state, start))
    composing = Prover(facts, LEVELS, CAP, search=search, compose=True)
    for p in (plain, composing):
        for g in _grandchildren(OPEN, 4):
            assert p.prove(g) is True
        assert p.prove(OPEN) is True
    assert "move" in plain.proved[OPEN_KEY] and composing.proved[OPEN_KEY] == {"leaf": 1}
    strategy = composing.strategy(OPEN)
    assert strategy["nodes"][OPEN_KEY] == {"leaf": 1} and len(strategy["nodes"]) > 1
    assert check(facts, OPEN, strategy, LEVELS, CAP)["won"]
    assert not check(facts, OPEN, {"nodes": {OPEN_KEY: {"leaf": 1}}, "maps": strategy["maps"]}, LEVELS, CAP)["won"]


def test_a_composing_search_is_told_what_is_proved():
    told = []
    p = Prover(_facts(), LEVELS, CAP, search=lambda state, start, stop=None: told.append(stop), compose=True)
    p.prove(LONE)
    assert told and all(stop is p.proved for stop in told)
    found = bounded_search(_facts(), LEVELS, random.Random(0), CAP, evaluations=5)
    walls = {GAME.state_key(g) for g in _grandchildren(OPEN, 4)}
    assert found(OPEN, {4: 0}, stop=walls)["status"] == "found"
    assert found(OPEN, {4: 0})["status"] == "budget"


def test_a_composing_prover_simplifies_by_the_composed_walk():
    def search(state, start, stop=None):
        return {"levels": {4: 0, 0: 3}, "exceptions": {}} if state == OPEN else None

    p = Prover(_facts(), LEVELS, CAP, compose=True, search=search)
    for g in _grandchildren(OPEN, 4):
        p.prove(g)
    assert p.prove(OPEN) is True and p.proved[OPEN_KEY] == {"leaf": 1} and p.maps[1] == {4: 0}


@pytest.mark.parametrize("compose", [True, False])
def test_every_walk_of_a_composing_prover_stops_at_its_proofs(monkeypatch, compose):
    import harness.self_proof as self_proof
    stops = []
    real = self_proof.failing

    def recorded(*args, **kwargs):
        stops.append(kwargs.get("stop"))
        return real(*args, **kwargs)

    monkeypatch.setattr(self_proof, "failing", recorded)
    p = Prover(_facts(), LEVELS, CAP, compose=compose, search_below=CAP,
               search=bounded_search(_facts(), LEVELS, random.Random(0), CAP, evaluations=30))
    p.prove(LONE)
    assert stops and all((stop is p.proved) if compose else stop is None for stop in stops)


def test_an_aimed_trial_does_not_aim_at_a_position_with_a_win_now():
    win_now = _ttt((0, 1), (3, 4))
    assert _facts().of(win_now)[0] and len(_facts().of(win_now)[1]) > 1
    walk = {"exceptions": {"x": None}, "came": {"x": (win_now, None)}}
    assert aimed_trial(_facts(), LEVELS, random.Random(0), {}, walk) is None


def test_the_composed_export_follows_a_map_through_positions_it_has_not_proved():
    facts = _facts()
    through = _grandchildren(OPEN, 4)[0]
    deeper = _grandchildren(through, choose(facts, through, {4: 0}, LEVELS))[0]
    p = Prover(facts, LEVELS, CAP, compose=True)
    p.maps.append({4: 0})
    p.proved.update({OPEN_KEY: {"leaf": 1}, GAME.state_key(deeper): {"leaf": 0}})
    nodes = p.strategy(OPEN)["nodes"]
    assert GAME.state_key(through) not in nodes and GAME.state_key(deeper) in nodes

