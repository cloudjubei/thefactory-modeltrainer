"""§C.36 VERIFIED SYMMETRY FINDER — a game's exploitable symmetries, proved against its own dynamics.

WHY (2026-09-22): symmetries were hand-written per game and, outside Connect-4, wrong or absent (§C.35). A false
symmetry is not a harmless omission — it teaches the net that two positions are equivalent when they are not, so
it must be PROVEN, not asserted. This is the project's standing rule (measure, don't assert) applied to the one
place a plausible geometric argument is easy to get wrong: checkers' men are directional, and a left-right mirror
of an even board maps dark squares to light, so the valid group is not the board's full symmetry group.

The finder enumerates the board's candidate isometries (the dihedral group of the grid, restricted to the maps
that preserve the board SHAPE) and keeps only those that COMMUTE with the game: for random legal positions,
`transform(step(s, a)) == step(transform(s), action_perm[a])` for every legal `a`, with the side to move
preserved. Commutation with side-to-move fixed is exactly value-invariance — the two game trees are isomorphic —
so a kept symmetry is safe to augment with. The board relabel and the action relabel are the two small
encoding-aware hooks each game supplies (`transform_state`, `transform_action`); the enumeration and the
verification are generic here."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Iso:
    """One board isometry: its name, its forward geometric map (r,c)->(r,c), and the resulting cell
    source-permutation `cell_perm` (`dest <- src`, length rows*cols) that relabels a flattened board."""
    name: str
    rows: int
    cols: int
    map: Callable[[int, int], tuple]
    cell_perm: list[int]

    def cell_image(self, cell: int) -> int:
        """The flat cell index this isometry sends `cell` to (its FORWARD image)."""
        nr, nc = self.map(*divmod(cell, self.cols))
        return nr * self.cols + nc

    def dir_image(self, dr: int, dc: int) -> tuple:
        """The isometry's LINEAR part applied to a direction vector (translation cancels), so a game can carry a
        move's direction through the transform."""
        b, q = self.map(0, 0), self.map(dr, dc)
        return (q[0] - b[0], q[1] - b[1])


# The eight dihedral maps of a grid, as forward (r,c)->(r',c'). `needs_square` marks the four that only preserve
# a board's shape when rows == cols (they transpose the axes).
_MAPS = [
    ("identity", False, lambda r, c, h, w: (r, c)),
    ("rot180", False, lambda r, c, h, w: (h - 1 - r, w - 1 - c)),
    ("flip_h", False, lambda r, c, h, w: (r, w - 1 - c)),
    ("flip_v", False, lambda r, c, h, w: (h - 1 - r, c)),
    ("rot90", True, lambda r, c, h, w: (c, h - 1 - r)),
    ("rot270", True, lambda r, c, h, w: (w - 1 - c, r)),
    ("transpose", True, lambda r, c, h, w: (c, r)),
    ("anti_transpose", True, lambda r, c, h, w: (w - 1 - c, h - 1 - r)),
]


def dihedral_isometries(rows: int, cols: int) -> list[Iso]:
    """The board's shape-preserving isometries, identity first. Eight for a square, four for a rectangle (the
    transposing maps are dropped because they would change rows×cols)."""
    out = []
    for name, needs_square, fmap in _MAPS:
        if needs_square and rows != cols:
            continue
        f = lambda r, c, fmap=fmap: fmap(r, c, rows, cols)
        cell_perm = [0] * (rows * cols)
        for src in range(rows * cols):
            r, c = divmod(src, cols)
            nr, nc = f(r, c)
            cell_perm[nr * cols + nc] = src   # source-perm: the cell now AT dest came FROM src
        out.append(Iso(name=name, rows=rows, cols=cols, map=f, cell_perm=cell_perm))
    return out


def iso_from_cell_map(name: str, rows: int, cols: int, forward_cell) -> Iso:
    """Build an Iso from a FORWARD cell map `src -> dest` (any board automorphism, not only the grid dihedral).
    Lets a game offer candidate isometries the grid enumerator cannot express — e.g. Nine Men's Morris's
    inner<->outer ring swap — for `find_symmetries` to verify like any other."""
    cell_perm = [0] * (rows * cols)
    for src in range(rows * cols):
        cell_perm[forward_cell(src)] = src
    return Iso(name=name, rows=rows, cols=cols,
               map=lambda r, c: divmod(forward_cell(r * cols + c), cols), cell_perm=cell_perm)


def _probe_states(game, n: int, seed: int, max_plies: int) -> list:
    """Random legal non-terminal positions reached by short uniform playouts from the initial state."""
    rng = random.Random(seed)
    seen, out = set(), []
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        s = game.initial_state(random.Random(rng.random()))
        for _ in range(rng.randint(0, max_plies)):
            if game.is_terminal(s):
                break
            s = game.step(s, rng.choice(game.legal_actions(s)))
        if game.is_terminal(s):
            continue
        key = game.state_key(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def verify_isometry(game, iso: Iso, probes: list) -> list[int] | None:
    """Return the ACTION permutation if `iso` is a genuine same-player symmetry of `game`, else None.

    Requires the game to expose `transform_state(state, iso)` and `transform_action(action, iso)`. The action
    permutation is the game's claim; this checks it against dynamics over every probe and refuses it on the first
    disagreement, so a wrong hook cannot smuggle a false symmetry through."""
    n = game.num_actions
    try:
        action_perm = [game.transform_action(a, iso) for a in range(n)]
    except (ValueError, IndexError):
        return None
    if sorted(action_perm) != list(range(n)):
        return None                                   # not even a bijection of the action space
    for s in probes:
        gs = game.transform_state(s, iso)
        if gs is None or game.current_player(gs) != game.current_player(s):
            return None
        if sorted(game.legal_actions(gs)) != sorted(action_perm[a] for a in game.legal_actions(s)):
            return None
        for a in game.legal_actions(s):
            if game.transform_state(game.step(s, a), iso) != game.step(gs, action_perm[a]):
                return None
    return action_perm


def verified_isometries(game, n_probes: int = 200, seed: int = 20260922, max_plies: int = 30,
                        isometries: list | None = None) -> list[tuple[Iso, list[int]]]:
    """The game's VERIFIED isometries as `(iso, forward_action_image)` pairs, identity first — the objects a caller
    needs to TRANSFORM positions (e.g. scoring a policy in every orientation), where `find_symmetries` gives only the
    permutations augmentation needs. Every candidate is proven against dynamics; one that is not a real symmetry is
    left out, never trusted. A game without the transform hooks has only the identity."""
    rows, cols = game.board_shape
    candidates = isometries if isometries is not None else dihedral_isometries(rows, cols)
    identity = next((i for i in candidates if i.name == "identity"), None) \
        or next(i for i in dihedral_isometries(rows, cols) if i.name == "identity")
    out = [(identity, list(range(game.num_actions)))]
    if not (hasattr(game, "transform_state") and hasattr(game, "transform_action")):
        return out
    probes = _probe_states(game, n_probes, seed, max_plies)
    for iso in candidates:
        if iso.name == "identity":
            continue
        forward = verify_isometry(game, iso, probes)
        if forward is not None:
            out.append((iso, forward))
    return out


def find_symmetries(game, n_probes: int = 200, seed: int = 20260922, max_plies: int = 30,
                    isometries: list | None = None) -> list[tuple[list[int], list[int]]]:
    """The game's VERIFIED symmetries as `(cell_perm, action_perm)` pairs (identity always included). A game that
    does not expose the transform hooks has only the identity — safe, just no augmentation space saving.

    `isometries` overrides the candidate set (default: the grid's dihedral group) so a game can add board
    automorphisms the grid enumerator cannot propose — each is still PROVEN against dynamics, so an extra
    candidate that is not a real symmetry is refused, never trusted."""
    rows, cols = game.board_shape
    out = []
    for iso, forward in verified_isometries(game, n_probes, seed, max_plies, isometries):
        if iso.name == "identity":
            out.append((list(range(rows * cols)), list(range(game.num_actions))))
            continue
        # the verifier works with FORWARD action images (action_perm[a] = image of a); augment_examples
        # consumes SOURCE-permutations (new[dest] = old[perm[dest]]) to match cell_perm, so invert.
        source = [0] * len(forward)
        for a, img in enumerate(forward):
            source[img] = a
        out.append((iso.cell_perm, source))
    return out
