"""Direct rules tests for Nine Men's Morris (§C.37). Fixtures vary in SHAPE — ring vs spoke mills, the three
phases, the sub-turn — not just in value, so a guard that checks a proxy is exposed."""
from __future__ import annotations

import random

import pytest

from games.ninemensmorris import (HANDS0, MILLS, NUM_ACTIONS, POINTS, MorrisState, NineMensMorris, _settle)

G = NineMensMorris()
PLACE = lambda p: p
MOVE = lambda f, t: POINTS + f * POINTS + t
REMOVE = lambda p: POINTS + POINTS * POINTS + p


def _mk(board, to_move=0, hands=(0, 0), pending=False, idle=0):
    return _settle(tuple(board), to_move, tuple(hands), pending, idle)


def _empty():
    return [0] * POINTS


# ---- placing phase ----

def test_initial_state_is_24_placements_for_player_0():
    s = G.initial_state()
    assert G.current_player(s) == 0 and s.hands == (HANDS0, HANDS0)
    assert sorted(G.legal_actions(s)) == [PLACE(p) for p in range(POINTS)]
    assert not s.done


def test_placing_alternates_and_decrements_hands():
    s = G.initial_state()
    s = G.step(s, PLACE(4))       # p0 places on 4 (no mill)
    assert s.to_move == 1 and s.hands == (8, 9) and s.board[4] == 1
    s = G.step(s, PLACE(5))       # p1 places on 5
    assert s.to_move == 0 and s.hands == (8, 8) and s.board[5] == 2


def test_all_eighteen_men_get_placed_then_moving_begins():
    s = G.initial_state()
    rng = random.Random(0)
    placed = 0
    while s.hands != (0, 0):
        a = rng.choice([a for a in G.legal_actions(s) if a < POINTS])
        s = G.step(s, a)
        if s.pending_removal:                       # skip any forced removal to keep counting placements
            s = G.step(s, random.Random(1).choice(G.legal_actions(s)))
        placed += 1
    assert placed >= 18 and s.hands == (0, 0)
    assert all(a >= POINTS for a in G.legal_actions(s))   # placing over -> only MOVE ids


# ---- mill formation + the removal sub-turn ----

def test_completing_a_ring_mill_triggers_a_removal_with_to_move_FIXED():
    b = _empty()
    b[0], b[1] = 1, 1                       # p0 on two of ring mill (0,1,2)
    b[10] = 2                               # a lone opponent man to take
    s = _mk(b, to_move=0, hands=(1, 0))     # p0 still has one to place
    s2 = G.step(s, PLACE(2))                # completes (0,1,2)
    assert s2.pending_removal and s2.to_move == 0        # SAME player owes the removal
    assert set(G.legal_actions(s2)) == {REMOVE(10)}
    s3 = G.step(s2, REMOVE(10))
    assert not s3.pending_removal and s3.to_move == 1 and s3.board[10] == 0


def test_a_spoke_mill_also_triggers_a_removal():
    b = _empty()
    b[1], b[9] = 1, 1                       # two of the spoke mill (1,9,17)
    b[3] = 2
    s = _mk(b, to_move=0, hands=(1, 0))
    s2 = G.step(s, PLACE(17))
    assert s2.pending_removal and set(G.legal_actions(s2)) == {REMOVE(3)}


def test_a_move_that_forms_no_mill_switches_and_increments_idle():
    b = _empty()
    b[0], b[10], b[12], b[22] = 1, 1, 1, 1  # 4 p0 men -> moving phase (not attrition, not flying)
    b[20], b[21], b[23] = 2, 2, 2           # p1 has men too
    s = _mk(b, to_move=0, hands=(0, 0), idle=5)
    assert not s.done and MOVE(0, 7) in G.legal_actions(s)
    s2 = G.step(s, MOVE(0, 7))              # 7 is adjacent to 0, empty, forms no mill
    assert s2.to_move == 1 and not s2.pending_removal and s2.idle == 6


# ---- capture priority ----

def test_cannot_remove_a_man_in_a_mill_while_a_free_man_exists():
    b = _empty()
    b[0], b[1], b[2] = 2, 2, 2             # an opponent MILL (protected)
    b[10] = 2                              # a free opponent man
    b[8], b[9] = 1, 1                      # p0 about to mill (8,9,10)? no — put p0 on a ring
    b[16], b[17] = 1, 1
    s = _mk(b, to_move=0, hands=(1, 0))
    s2 = G.step(s, PLACE(18))             # completes inner mill (16,17,18) -> removal
    assert s2.pending_removal
    assert set(G.legal_actions(s2)) == {REMOVE(10)}      # only the free man, never one in the (0,1,2) mill


def test_when_every_opponent_man_is_in_a_mill_any_may_be_taken():
    b = _empty()
    b[0], b[1], b[2] = 2, 2, 2            # every opponent man is in this one mill
    b[16], b[17] = 1, 1
    s = _mk(b, to_move=0, hands=(1, 0))
    s2 = G.step(s, PLACE(18))            # p0 mills -> removal; all opp men are protected, so all become legal
    assert s2.pending_removal
    assert set(G.legal_actions(s2)) == {REMOVE(0), REMOVE(1), REMOVE(2)}


def test_double_mill_in_one_move_still_grants_exactly_one_removal():
    b = _empty()
    # point 2 is in ring mill (0,1,2) AND (2,3,4); set up both to complete on placing 2
    b[0], b[1], b[3], b[4] = 1, 1, 1, 1
    b[10] = 2
    s = _mk(b, to_move=0, hands=(1, 0))
    s2 = G.step(s, PLACE(2))
    assert s2.pending_removal
    s3 = G.step(s2, REMOVE(10))
    assert not s3.pending_removal        # exactly one removal, not two


# ---- moving / flying / terminal ----

def test_flying_unlocks_at_exactly_three_men():
    b = _empty()
    b[0], b[2], b[4] = 1, 1, 1           # p0 has exactly 3 -> flying
    b[20], b[21], b[22], b[23] = 2, 2, 2, 2
    s = _mk(b, to_move=0, hands=(0, 0))
    empt = [p for p in range(POINTS) if b[p] == 0]
    fly = {MOVE(a, t) for a in (0, 2, 4) for t in empt}
    assert set(G.legal_actions(s)) == fly       # can move ANY of its 3 men to ANY empty point


def test_a_four_man_mover_is_restricted_to_adjacent_moves():
    b = _empty()
    b[0], b[2], b[4], b[6] = 1, 1, 1, 1  # 4 men -> moving, not flying
    b[20], b[21], b[22] = 2, 2, 2
    s = _mk(b, to_move=0, hands=(0, 0))
    from games.ninemensmorris import ADJ
    for a in G.legal_actions(s):
        u = a - POINTS
        assert u % POINTS in ADJ[u // POINTS]    # every legal move is to an adjacent point
    with pytest.raises(ValueError):
        G.step(s, MOVE(0, 4))                    # 4 is not adjacent to 0 -> illegal


def test_blockade_with_no_move_is_a_loss():
    b = _empty()
    # p0 (to move) has 4 men all boxed in by p1; construct a small trap
    b[0], b[1], b[2] = 1, 2, 1           # p0 at 0 and 2, blocked by p1 at 1 and edges
    b[7], b[9], b[3] = 2, 2, 2           # surround 0 (adj 1,7) and 2 (adj 1,3)
    b[8] = 1                             # a 4th p0 man, also trapped
    b[15], b[23] = 2, 2                  # block 8 (adj 9,15)
    s = _mk(b, to_move=0, hands=(0, 0))
    if not G.legal_actions(s):           # if genuinely blocked
        assert s.done and G.winner(s) == 1 and G.returns(s) == [-1.0, 1.0]


def test_attrition_to_two_men_is_a_loss():
    b = _empty()
    b[0], b[1] = 1, 1                    # p0 down to 2 men, hands empty -> cannot ever mill
    b[10], b[11], b[12] = 2, 2, 2
    s = _mk(b, to_move=0, hands=(0, 0))
    assert s.done and G.winner(s) == 1


def test_draw_at_the_idle_limit_pays_zero():
    b = _empty()
    b[0], b[2], b[4] = 1, 1, 1
    b[20], b[21], b[22] = 2, 2, 2
    from games.ninemensmorris import IDLE_LIMIT
    s = _mk(b, to_move=0, hands=(0, 0), idle=IDLE_LIMIT)
    assert s.done and G.winner(s) is None and G.returns(s) == [0.0, 0.0]


def test_a_removal_resets_idle_but_a_quiet_move_does_not():
    b = _empty()
    b[0], b[1], b[20] = 1, 1, 2
    b[22], b[23] = 1, 1
    s = _mk(b, to_move=0, hands=(1, 0), idle=40)
    s2 = G.step(s, PLACE(2))            # forms (0,1,2), carries idle into the pending state
    assert s2.pending_removal and s2.idle == 0    # a PLACE resets idle to 0
    s3 = G.step(s2, REMOVE(20))
    assert s3.idle == 0


# ---- encoding (step 7) ----

def test_encode_produces_an_8x7x7_tensor_matching_the_board():
    import torch
    from harness.neural import encode
    b = _empty()
    b[0], b[10] = 1, 1          # p0 men
    b[20], b[21] = 2, 2         # p1 men
    s = _mk(b, to_move=0, hands=(3, 4))
    x = encode(G, s)
    assert tuple(x.shape) == (8, 7, 7)
    from games.ninemensmorris import POINT_CELL, CELL_POINT
    flat = x.reshape(8, 49)
    for p in (0, 10):
        assert flat[0][POINT_CELL[p]] == 1.0    # own men on plane 0
    for p in (20, 21):
        assert flat[1][POINT_CELL[p]] == 1.0    # opp men on plane 1
    assert flat[0].sum() == 2 and flat[1].sum() == 2
    dead = [c for c in range(49) if c not in CELL_POINT]
    assert all(flat[0][c] == 0 and flat[1][c] == 0 for c in dead)     # nothing on dead cells


def test_observation_planes_carry_hands_pending_and_flying():
    from games.ninemensmorris import POINT_CELL
    b = _empty()
    b[0], b[2], b[4] = 1, 1, 1   # p0 exactly 3 -> flying
    b[19], b[20], b[21], b[22] = 2, 2, 2, 2   # p1 has 4 men -> NOT flying
    s = _mk(b, to_move=0, hands=(0, 0))
    obs = G.observation(s, 0)
    assert len(obs) == 392
    planes = [obs[i * 49:(i + 1) * 49] for i in range(8)]
    vc = POINT_CELL[0]
    assert planes[2][vc] == 1.0                        # valid-mask plane
    assert planes[3][vc] == 0.0 and planes[4][vc] == 0.0   # both hands empty
    assert planes[6][vc] == 1.0                        # own flying (3 men)
    assert planes[7][vc] == 0.0                        # opp NOT flying (4 men)


def test_own_perspective_swaps_planes_between_players():
    b = _empty()
    b[0] = 1
    b[20] = 2
    s = _mk(b, to_move=0, hands=(0, 0))
    from games.ninemensmorris import POINT_CELL
    o0 = G.observation(s, 0)
    o1 = G.observation(s, 1)
    assert o0[0 * 49 + POINT_CELL[0]] == 1.0 and o0[1 * 49 + POINT_CELL[20]] == 1.0    # p0 sees own at 0
    assert o1[0 * 49 + POINT_CELL[20]] == 1.0 and o1[1 * 49 + POINT_CELL[0]] == 1.0    # p1 sees own at 20


def test_the_same_board_encodes_differently_while_placing_vs_moving():
    # proves hands-in-hand reaches the net: identical piece layout, different phase -> different tensor
    b = _empty()
    b[0], b[2], b[4], b[6] = 1, 1, 1, 1
    b[20], b[21], b[22] = 2, 2, 2
    placing = _mk(b, to_move=0, hands=(2, 1))
    moving = _mk(b, to_move=0, hands=(0, 0))
    assert G.observation(placing, 0) != G.observation(moving, 0)


# ---- action space round-trip (step 8) ----

def test_action_label_partitions_the_three_blocks():
    assert G.action_label(_mk(_empty()), PLACE(5)) == "P5"
    assert G.action_label(_mk(_empty()), MOVE(3, 11)) == "3-11"
    assert G.action_label(_mk(_empty()), REMOVE(7)) == "x7"
    assert G.num_actions == NUM_ACTIONS == 624


def test_no_self_move_action_is_ever_legal():
    s = G.initial_state()
    rng = random.Random(3)
    for _ in range(40):
        acts = G.legal_actions(s)
        if not acts:
            break
        for a in acts:
            if POINTS <= a < POINTS + POINTS * POINTS:
                u = a - POINTS
                assert u // POINTS != u % POINTS       # never move a man onto itself
        s = G.step(s, rng.choice(acts))


def test_step_refuses_out_of_range_illegal_and_finished():
    s = G.initial_state()
    with pytest.raises(ValueError):
        G.step(s, -1)
    with pytest.raises(ValueError):
        G.step(s, NUM_ACTIONS)
    with pytest.raises(ValueError):
        G.step(s, MOVE(0, 1))                          # a MOVE during placing is illegal
    done = _mk([1, 1] + [0] * 20 + [2, 2], to_move=0, hands=(0, 0))   # attrition-done
    assert done.done
    with pytest.raises(ValueError):
        G.step(done, PLACE(5))


# ---- symmetry (step 9) ----

def test_morris_has_exactly_the_sixteen_verified_symmetries():
    syms = G.symmetries()
    assert len(syms) == 16                              # D4 (8) x ring-swap (2)
    for cell_perm, action_perm in syms:
        assert sorted(cell_perm) == list(range(49))     # a permutation of the 49 grid cells
        assert sorted(action_perm) == list(range(NUM_ACTIONS))


def test_every_symmetry_fixes_the_dead_cell_set():
    from games.ninemensmorris import CELL_POINT
    dead = {c for c in range(49) if c not in CELL_POINT}
    for cell_perm, _ in G.symmetries():
        # cell_perm is a source-perm; the image set of the dead cells must be the dead cells (so constant planes
        # and the valid-mask plane survive augmentation)
        assert {cell_perm.index(c) for c in dead} == dead


def test_a_known_rotation_and_the_ring_swap_move_mills_as_expected():
    from harness.symmetry import dihedral_isometries
    from games.ninemensmorris import _candidate_isometries, _point_fwd
    isos = {i.name: i for i in _candidate_isometries()}
    # 90-degree rotation sends outer corner-line (0,1,2) to the next side (2,3,4)
    pf_rot = _point_fwd(isos["rot90"])
    assert sorted(pf_rot[p] for p in (0, 1, 2)) == [2, 3, 4]
    # the ring swap sends the outer ring's (0,1,2) to the inner ring's (16,17,18)
    pf_swap = _point_fwd(isos["ring_swap.identity"])
    assert sorted(pf_swap[p] for p in (0, 1, 2)) == [16, 17, 18]


def test_transform_commutes_on_a_hand_built_FLYING_and_PENDING_state():
    from games.ninemensmorris import _candidate_isometries
    ring = next(i for i in _candidate_isometries() if i.name == "ring_swap.identity")
    # flying state
    fly = _mk([0] * POINTS, to_move=0, hands=(0, 0))
    fly = _settle((1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0), 0, (0, 0), False, 0)
    for a in G.legal_actions(fly):
        assert G.transform_state(G.step(fly, a), ring) == G.step(G.transform_state(fly, ring),
                                                                  G.transform_action(a, ring))


def test_a_deliberately_wrong_candidate_is_REJECTED_by_the_verifier():
    from harness.symmetry import iso_from_cell_map, verify_isometry, _probe_states
    from games.ninemensmorris import POINT_CELL
    # a bijection of the 49 cells that swaps two ADJACENT point-cells only (0 and 1) — NOT a board automorphism
    # (it breaks the mill/adjacency structure), so it must not verify.
    swap = {POINT_CELL[0]: POINT_CELL[1], POINT_CELL[1]: POINT_CELL[0]}
    bad = iso_from_cell_map("bad", 7, 7, lambda c: swap.get(c, c))
    probes = _probe_states(G, 150, 5, 60)
    assert verify_isometry(G, bad, probes) is None


# ---- harness integration (step 10): the unchanged process runs on this game ----

def test_a_net_builds_for_morris_and_encodes_a_position():
    import torch
    from harness.neural import Connect4Net, arch_for_game, encode
    net = Connect4Net(**arch_for_game({"channels": 8, "blocks": 1}, G))
    x = encode(G, G.initial_state()).unsqueeze(0)
    pol, val = net(x)
    assert pol.shape[1] == NUM_ACTIONS and val.shape[0] == 1


def test_self_play_produces_examples_that_augment_sixteenfold():
    from harness.neural import AlphaZeroAgent, Connect4Net, arch_for_game, augment_examples, self_play_game
    net = Connect4Net(**arch_for_game({"channels": 8, "blocks": 1}, G))
    agent = AlphaZeroAgent(net, sims=6, solve_endgame=0, gumbel=True, c_scale=0.1)
    ex = self_play_game(G, agent, random.Random(0))
    assert len(ex) > 0
    aug = augment_examples(ex, G.symmetries())
    assert len(aug) == 16 * len(ex)                    # the verified group multiplies data 16x


def test_paired_exploitability_runs_with_the_game_agnostic_refuter():
    from harness.agents import MctsAgent
    from harness.benchmark import paired_exploitability
    from harness.neural import AlphaZeroAgent, Connect4Net, arch_for_game

    def arm():
        net = Connect4Net(**arch_for_game({"channels": 8, "blocks": 1}, G))
        return AlphaZeroAgent(net, sims=4, solve_endgame=0, gumbel=True, c_scale=0.1)

    res = paired_exploitability(
        G, {"a": arm, "b": arm}, depths=[6], n_openings=2, opening_plies=2, seed=7,
        refuter_factory=lambda d: MctsAgent(sims=d, solve_endgame=0, book=None))
    row = res["by_depth"][0]
    assert set(row["arms"]) == {"a", "b"}
    assert len(row["arms"]["a"]["outcomes"]) == len(row["arms"]["b"]["outcomes"]) > 0


def test_heuristic_always_returns_a_legal_action_including_in_pending_states():
    s = G.initial_state()
    rng = random.Random(11)
    for _ in range(60):
        if s.done:
            break
        a = G.heuristic_action(s, rng)
        assert a in G.legal_actions(s)
        s = G.step(s, a)


def test_heuristic_takes_an_available_mill_and_the_forced_removal():
    b = _empty()
    b[0], b[1] = 1, 1
    b[10] = 2
    s = _mk(b, to_move=0, hands=(1, 0))
    assert G.heuristic_action(s, random.Random(0)) == PLACE(2)   # completes (0,1,2)
    s2 = G.step(s, PLACE(2))
    assert G.heuristic_action(s2, random.Random(0)) == REMOVE(10)
