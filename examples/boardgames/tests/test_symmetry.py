"""Direct tests for harness.symmetry — the VERIFIED symmetry finder (§C.36, 2026-09-22).

A game's symmetries were hand-written per game and, for anything but Connect-4, either absent or wrong (§C.35).
The finder replaces assertion with proof: it enumerates the board's candidate isometries and keeps only those
that provably COMMUTE with the game's own dynamics (transform then step == step then transform) with the side to
move preserved. So a claimed symmetry that would corrupt training (a directional game's mirror, say) is refused
by the game itself, not trusted. The enumerator is tested here against groups known by hand; the per-game verdict
is tested in each game's own file."""
from __future__ import annotations

from harness.symmetry import dihedral_isometries


def _is_perm(p, n):
    return sorted(p) == list(range(n))


def test_a_square_board_has_the_eight_dihedral_isometries():
    isos = dihedral_isometries(3, 3)
    assert len(isos) == 8
    assert all(_is_perm(i.cell_perm, 9) for i in isos)
    assert len({tuple(i.cell_perm) for i in isos}) == 8          # all distinct
    assert any(i.name == "identity" for i in isos)


def test_a_rectangle_only_keeps_the_shape_PRESERVING_isometries():
    # A 6x7 board: rot90/transpose would change the shape, so only id, rot180, flip_h, flip_v survive.
    isos = dihedral_isometries(6, 7)
    assert len(isos) == 4
    assert {i.name for i in isos} == {"identity", "rot180", "flip_h", "flip_v"}
    assert all(_is_perm(i.cell_perm, 42) for i in isos)


def test_the_isometries_match_tictactoes_hand_written_dihedral_group():
    from games.tictactoe import _D4                              # a trusted independent source
    mine = {tuple(i.cell_perm) for i in dihedral_isometries(3, 3)}
    # _D4 is a group closed under inverse, so its forward perms and their inverses are the same SET; the finder's
    # source-permutations are those inverses, so the two sets must coincide exactly.
    assert mine == {tuple(p) for p in _D4}


def test_identity_is_always_first_and_is_the_identity_permutation():
    for h, w in ((3, 3), (6, 7), (8, 8)):
        isos = dihedral_isometries(h, w)
        assert isos[0].name == "identity"
        assert isos[0].cell_perm == list(range(h * w))


def test_each_isometry_maps_cells_consistently_with_its_forward_map():
    for i in dihedral_isometries(4, 4):
        for src in range(16):
            r, c = divmod(src, 4)
            nr, nc = i.map(r, c)
            assert i.cell_perm[nr * 4 + nc] == src              # cell_perm is the source-perm dest<-src


def test_the_verifier_REJECTS_a_plausible_but_false_candidate():
    """The whole point: a candidate that does not commute with dynamics must be refused, or the finder is a
    rubber stamp. Checkers' men are directional, so rot180 (which reverses forward motion) is NOT a symmetry
    however symmetric the board looks — while flip_h (column mirror, forward motion preserved) IS."""
    from harness.symmetry import dihedral_isometries, verify_isometry, _probe_states
    from harness.registry import resolve_game

    g = resolve_game("checkers")
    probes = _probe_states(g, 120, 1, 24)
    by_name = {i.name: i for i in dihedral_isometries(8, 8)}
    assert verify_isometry(g, by_name["rot180"], probes) is None       # reverses the men → refused
    assert verify_isometry(g, by_name["flip_v"], probes) is None       # reverses the men → refused
    assert verify_isometry(g, by_name["transpose"], probes) is None    # rotates forward sideways → refused
    assert verify_isometry(g, by_name["flip_h"], probes) is not None   # column mirror → kept


def test_a_game_without_transform_hooks_gets_only_the_identity():
    from harness.symmetry import find_symmetries
    from harness.registry import resolve_game

    oth = resolve_game("othello")  # has no transform_state/transform_action (deferred)
    syms = find_symmetries(oth)
    assert len(syms) == 1
    assert syms[0] == (list(range(64)), list(range(oth.num_actions)))


def test_a_returned_pair_augments_board_and_policy_CONSISTENTLY():
    """End-to-end: a verified pair fed to augment_examples must move the board and the policy by the SAME
    symmetry, or the augmentation teaches a lie. Checked on tictactoe, where an action is a cell so the mark and
    its policy mass must land on the same transformed cell."""
    import torch
    from harness.neural import augment_examples
    from harness.registry import resolve_game

    g = resolve_game("tictactoe")
    x = torch.zeros(2, 3, 3)
    x[0, 0, 1] = 1.0                      # own mark at cell 1 (row 0, col 1)
    pi = [0.0] * 9
    pi[1] = 1.0                            # all policy on cell 1
    for bx, bpi, _ in augment_examples([(x, pi, 0.0)], g.symmetries()):
        mark = int(torch.argmax(bx[0].reshape(-1)))
        assert bpi[mark] == 1.0            # policy mass sits on the SAME cell the mark moved to


# The verifier's individual checks are load-bearing for a game that supplies a BAD transform hook (the reason the
# finder verifies rather than trusts). No real game exercises them one at a time, so a minimal fake game that
# lies in exactly one way each isolates them — the "vary the fixture SHAPE" discipline applied to a guard.
from dataclasses import dataclass


@dataclass(frozen=True)
class _PS:
    board: tuple
    to_move: int


class _Place:
    """A trivially D4-symmetric 2x2 place-a-mark game; every dihedral map is a true symmetry."""
    board_shape = (2, 2)
    num_actions = 4

    def initial_state(self, rng=None):
        return _PS((0, 0, 0, 0), 0)

    def current_player(self, s):
        return s.to_move

    def legal_actions(self, s):
        return [i for i, v in enumerate(s.board) if v == 0]

    def step(self, s, a, rng=None):
        nb = list(s.board)
        nb[a] = s.to_move + 1
        return _PS(tuple(nb), 1 - s.to_move)

    def is_terminal(self, s):
        return all(s.board)

    def state_key(self, s):
        return (s.board, s.to_move)

    def transform_state(self, s, iso):
        return _PS(tuple(s.board[iso.cell_perm[d]] for d in range(4)), s.to_move)

    def transform_action(self, a, iso):
        return iso.cell_image(a)


def _flip_h():
    return next(i for i in dihedral_isometries(2, 2) if i.name == "flip_h")


def test_verifier_ACCEPTS_a_genuine_symmetry_of_the_fake_game():
    from harness.symmetry import verify_isometry
    g = _Place()
    assert verify_isometry(g, _flip_h(), [g.initial_state()]) is not None


def test_verifier_rejects_a_transform_action_that_is_not_a_bijection():
    from harness.symmetry import verify_isometry
    g = _Place()
    # Probe with cell 0 already filled, so action 0 is ILLEGAL and never exercised by the dynamics checks. The
    # bad map is correct on the legal actions {1,2,3} but collides action 0 onto flip_h(1)=0 — a non-bijection
    # that the legal-set and commutation checks cannot see, so ONLY the bijection check can reject it.
    good = _Place().transform_action
    g.transform_action = lambda a, iso: 0 if a == 0 else good(a, iso)
    probe = _PS((1, 0, 0, 0), 0)
    assert sorted(g.transform_action(a, _flip_h()) for a in range(4)) != [0, 1, 2, 3]  # indeed not a bijection
    assert verify_isometry(g, _flip_h(), [probe]) is None


class _Neutral(_Place):
    """Marks are uncolored (always 1), so the BOARD cannot reveal whose turn it is — only `current_player`
    can. This isolates the side-to-move check: a to_move flip is invisible to the board-commutation comparison."""

    def step(self, s, a, rng=None):
        nb = list(s.board)
        nb[a] = 1
        return _PS(tuple(nb), 1 - s.to_move)


def test_verifier_rejects_a_transform_that_flips_the_side_to_move():
    from harness.symmetry import verify_isometry
    g = _Neutral()
    good = g.transform_state
    g.transform_state = lambda s, iso: _PS(good(s, iso).board, 1 - s.to_move)   # value is NOT invariant
    # board dynamics still commute (marks are uncolored), so ONLY the current_player check can reject this
    assert verify_isometry(g, _flip_h(), [g.initial_state()]) is None


def test_verifier_rejects_a_bijection_that_sends_moves_to_the_WRONG_successor():
    """The action perm is a bijection AND permutes legal actions to legal actions (all 4 are legal on the empty
    board), so only the step-image comparison can catch that it is the wrong bijection."""
    from harness.symmetry import verify_isometry
    g = _Place()
    g.transform_action = lambda a, iso: {0: 2, 1: 3, 2: 0, 3: 1}[a]   # a bijection, but not flip_h's
    assert verify_isometry(g, _flip_h(), [g.initial_state()]) is None
