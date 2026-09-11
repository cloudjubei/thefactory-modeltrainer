"""§C.21 rule-module library — the reusable effect primitives a game is COMPOSED from (Increment 2).

The design's bankable win was never a shared trunk; it was this: pure, tested primitives (rays, flank capture,
majority) that a game file assembles rather than re-implements. Othello is the first consumer and the proof —
its move generation is `flank` over `grid_rays`, its terminal is `majority`, and nothing in the game file knows
how to walk a board. Boards are flat tuples of ints: 0 empty, 1 player-0, 2 player-1 (the harness convention).
"""
import pytest

from harness.rules import DIRS8, flank, grid_rays, majority


def test_eight_directions_are_the_king_moves():
    assert len(DIRS8) == 8 and (0, 0) not in DIRS8
    assert set(DIRS8) == {(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)} - {(0, 0)}


def test_rays_walk_outward_to_the_edge():
    rays = grid_rays(8, 8)
    assert len(rays) == 64 * 8
    corner = 0  # (0,0)
    lengths = {d: len(rays[corner, d]) for d in DIRS8}
    assert lengths[(0, 1)] == 7 and lengths[(1, 0)] == 7 and lengths[(1, 1)] == 7  # E, S, SE
    assert lengths[(-1, 0)] == 0 and lengths[(0, -1)] == 0 and lengths[(-1, -1)] == 0
    assert rays[corner, (0, 1)] == (1, 2, 3, 4, 5, 6, 7)
    centre = 3 * 8 + 3
    assert rays[centre, (-1, -1)] == (18, 9, 0)


def test_rays_respect_non_square_boards():
    rays = grid_rays(3, 4)
    assert len(rays) == 12 * 8
    assert rays[0, (1, 1)] == (5, 10)  # (0,0) -> (1,1) -> (2,2); no (3,3) on 3 rows


def _othello_start():
    b = [0] * 64
    b[3 * 8 + 3] = 2; b[4 * 8 + 4] = 2  # d4, e5 white (player 1)
    b[3 * 8 + 4] = 1; b[4 * 8 + 3] = 1  # e4, d5 black (player 0)
    return tuple(b)


def test_flank_captures_a_bracketed_run():
    rays = grid_rays(8, 8)
    board = _othello_start()
    d3 = 2 * 8 + 3
    assert flank(board, d3, rays, me=1, opp=2) == [3 * 8 + 3]  # black at d3 flips d4 via d5


def test_flank_needs_a_terminator_of_my_own():
    rays = grid_rays(8, 8)
    board = list(_othello_start())
    board[3 * 8 + 3] = 2; board[3 * 8 + 4] = 2; board[4 * 8 + 3] = 2; board[4 * 8 + 4] = 2  # all white
    assert flank(tuple(board), 2 * 8 + 3, rays, me=1, opp=2) == []  # a run to nothing flips nothing


def test_flank_ignores_empty_gaps_and_occupied_cells():
    rays = grid_rays(8, 8)
    board = _othello_start()
    assert flank(board, 0, rays, me=1, opp=2) == []  # corner: no adjacent enemy at all
    assert flank(board, 3 * 8 + 3, rays, me=1, opp=2) == []  # occupied cell: never a capture site


def test_flank_can_capture_along_several_rays_at_once():
    rays = grid_rays(8, 8)
    b = [0] * 64
    # me at (2,2); enemies at (2,3),(3,2),(3,3); my terminators at (2,4),(4,2),(4,4). Playing (2,2)... use (2,2)
    # as the empty site: enemies E/S/SE of it, terminators beyond each.
    site = 2 * 8 + 2
    for r, c in ((2, 3), (3, 2), (3, 3)):
        b[r * 8 + c] = 2
    for r, c in ((2, 4), (4, 2), (4, 4)):
        b[r * 8 + c] = 1
    flipped = flank(tuple(b), site, rays, me=1, opp=2)
    assert sorted(flipped) == sorted([2 * 8 + 3, 3 * 8 + 2, 3 * 8 + 3])


def test_majority_names_the_winner_or_a_draw():
    assert majority((1, 1, 2, 0)) == 0
    assert majority((2, 2, 1, 0)) == 1
    assert majority((1, 2, 0, 0)) is None
    assert majority((0, 0, 0)) is None


def test_diag_steps_are_the_four_diagonal_neighbours_inside_the_board():
    from harness.rules import DIRS4, diag_steps

    s = diag_steps(8, 8)
    assert s[27, (-1, -1)] == 18 and s[27, (1, 1)] == 36
    assert (0, (-1, -1)) not in s, "a corner has no off-board neighbour"
    assert (0, (1, 1)) in s
    assert len(DIRS4) == 4 and all(abs(dr) == 1 and abs(dc) == 1 for dr, dc in DIRS4)


def test_diag_steps_never_wrap_around_a_row_edge():
    from harness.rules import diag_steps

    s = diag_steps(8, 8)
    assert (8, (-1, -1)) not in s and (8, (1, -1)) not in s, "col 0 has no left diagonal"
    assert (15, (-1, 1)) not in s and (15, (1, 1)) not in s, "col 7 has no right diagonal"


def test_diag_jumps_give_the_jumped_cell_and_the_landing_cell():
    from harness.rules import diag_jumps

    j = diag_jumps(8, 8)
    assert j[27, (-1, -1)] == (18, 9) and j[27, (1, 1)] == (36, 45)
    assert (9, (-1, -1)) not in j, "landing would be off the board"


def test_diag_jumps_land_two_diagonal_steps_away_on_the_same_colour():
    from harness.rules import diag_jumps

    j = diag_jumps(8, 8)
    for (cell, _d), (over, land) in j.items():
        assert (cell // 8 + land // 8) % 2 == 0 and (cell % 8 + land % 8) % 2 == 0
        assert over != cell and land != over
