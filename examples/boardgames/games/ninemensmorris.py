"""§C.37 NINE MEN'S MORRIS — the harness's first NON-GRID game, added to test that the unchanged process
transfers to a graph board with three phases and a sub-turn (§C.6 transfer target; no solver, like Othello).

The board is 24 points on three concentric squares joined by spokes at the edge midpoints (no diagonals). It is
embedded on a 7x7 grid at the points' natural coordinates so the harness's grid machinery works unchanged: the
25 non-point cells are declared dead via `valid_mask` (checkers' masked-pooling path), and — proven, not
asserted — the 7x7 grid's dihedral group permutes the 24-point set onto itself, so the verified symmetry finder
recovers the board's D4 with no change, and the game adds the non-grid inner<->outer ring swap as an extra
candidate for a full order-16 group.

Geometry is GENERATED, never hand-transcribed: a point is `ring*8 + pos` (ring 0=outer/1=middle/2=inner, pos
0..7 clockwise from NW), and adjacency / mills / grid-cell all fall out of that indexing. A transcription guard
(test) regenerates the invariants (32 edges, degree histogram, 16 mills, every point in exactly two mills)."""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

N = 7                       # the embedding grid is 7x7
CELLS = N * N               # 49
POINTS = 24
NUM_ACTIONS = POINTS + POINTS * POINTS + POINTS   # PLACE(24) + MOVE/FLY(24x24) + REMOVE(24) = 624
IDLE_LIMIT = 100            # half-moves with no mill and no capture -> draw (the 50-move tournament rule)
HANDS0 = 9                  # men each player places

_RADIUS = {0: 3, 1: 2, 2: 1}
_POS_OFF = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))  # NW,N,NE,E,SE,S,SW,W


def _gen_point_cell() -> tuple[int, ...]:
    out = []
    for ring in range(3):
        k = _RADIUS[ring]
        for pos in range(8):
            dr, dc = _POS_OFF[pos]
            out.append((3 + dr * k) * N + (3 + dc * k))
    return tuple(out)


def _gen_adj() -> tuple[tuple[int, ...], ...]:
    adj: list[set] = [set() for _ in range(POINTS)]
    for ring in range(3):                                   # the 8-cycle around each ring
        for pos in range(8):
            p = ring * 8 + pos
            adj[p].add(ring * 8 + (pos + 1) % 8)
            adj[p].add(ring * 8 + (pos - 1) % 8)
    for pos in (1, 3, 5, 7):                                # spokes at the edge midpoints join adjacent rings
        adj[0 * 8 + pos].add(1 * 8 + pos)
        adj[1 * 8 + pos].add(0 * 8 + pos)
        adj[1 * 8 + pos].add(2 * 8 + pos)
        adj[2 * 8 + pos].add(1 * 8 + pos)
    return tuple(tuple(sorted(a)) for a in adj)


def _gen_mills() -> tuple[tuple[int, int, int], ...]:
    mills = []
    for ring in range(3):                                   # three sides per ring corner: (0,1,2),(2,3,4),(4,5,6),(6,7,0)
        for start in (0, 2, 4, 6):
            mills.append(tuple(ring * 8 + (start + i) % 8 for i in range(3)))
    for pos in (1, 3, 5, 7):                                # the spoke triples across the three rings
        mills.append((0 * 8 + pos, 1 * 8 + pos, 2 * 8 + pos))
    return tuple(mills)


POINT_CELL = _gen_point_cell()
CELL_POINT = {c: p for p, c in enumerate(POINT_CELL)}
ADJ = _gen_adj()
MILLS = _gen_mills()
MILLS_THROUGH = tuple(tuple(m for m in MILLS if p in m) for p in range(POINTS))
VALID_MASK = tuple(1.0 if c in CELL_POINT else 0.0 for c in range(CELLS))


@dataclass(frozen=True)
class MorrisState:
    board: tuple           # length 24, POINT-indexed: 0 empty, 1 = player-0 man, 2 = player-1 man
    to_move: int           # 0/1; STAYS FIXED through the mill->remove sub-turn (checkers `jumping` analogue)
    hands: tuple           # (men_in_hand_p0, men_in_hand_p1); both 0 => placing over
    pending_removal: bool  # True == mover owes exactly one removal; legal_actions are removals only
    idle: int              # half-moves since the last mill/removal/placement (the draw clock)
    done: bool


def _on_board(board: tuple, player: int) -> int:
    return board.count(player + 1)


def _forms_mill(board: tuple, d: int, val: int) -> bool:
    """A completed line of `val` through point `d` (which was just occupied)."""
    return any(all(board[q] == val for q in m) for m in MILLS_THROUGH[d])


def _in_mill(board: tuple, q: int) -> bool:
    """`q` (occupied) is part of a completed line of its own colour."""
    v = board[q]
    return v != 0 and any(all(board[r] == v for r in m) for m in MILLS_THROUGH[q])


def _removable(board: tuple, opp: int) -> list[int]:
    """Opponent men that may be taken: those NOT in a mill, or ALL of them if every one is in a mill."""
    opp_pts = [q for q in range(POINTS) if board[q] == opp + 1]
    free = [q for q in opp_pts if not _in_mill(board, q)]
    return free if free else opp_pts


def _legal(board: tuple, to_move: int, hands: tuple, pending: bool) -> list[int]:
    if pending:
        return [POINTS + POINTS * POINTS + q for q in _removable(board, 1 - to_move)]
    if hands[to_move] > 0:
        return [p for p in range(POINTS) if board[p] == 0]              # PLACE ids are the point index
    own = to_move + 1
    mine = [p for p in range(POINTS) if board[p] == own]
    empt = [p for p in range(POINTS) if board[p] == 0]
    if len(mine) == 3:                                                  # FLYING: any empty point
        return [POINTS + a * POINTS + b for a in mine for b in empt]
    return [POINTS + a * POINTS + b for a in mine for b in ADJ[a] if board[b] == 0]  # MOVING: adjacent empties


def _settle(board: tuple, to_move: int, hands: tuple, pending: bool, idle: int) -> MorrisState:
    if pending:
        done = False
    else:
        attrition = hands[to_move] == 0 and _on_board(board, to_move) <= 2
        blocked = not _legal(board, to_move, hands, False)
        done = idle >= IDLE_LIMIT or attrition or blocked
    return MorrisState(board, to_move, hands, pending, idle, done)


class NineMensMorris:
    name = "ninemensmorris"
    num_players = 2
    num_actions = NUM_ACTIONS
    board_shape = (N, N)
    input_planes = 8
    valid_mask = VALID_MASK

    def initial_state(self, rng: random.Random | None = None) -> MorrisState:
        return _settle((0,) * POINTS, 0, (HANDS0, HANDS0), False, 0)

    def current_player(self, state: MorrisState) -> int:
        return state.to_move

    def legal_actions(self, state: MorrisState) -> list[int]:
        if state.done:
            return []
        return _legal(state.board, state.to_move, state.hands, state.pending_removal)

    def step(self, state: MorrisState, action: int, rng: random.Random | None = None) -> MorrisState:
        if state.done:
            raise ValueError("game is over")
        if not (0 <= action < NUM_ACTIONS):
            raise ValueError(f"action {action} out of range [0,{NUM_ACTIONS})")
        if action not in self.legal_actions(state):
            raise ValueError(f"illegal action {action}: {self.action_label(state, action)}")
        to = state.to_move
        board = list(state.board)
        if action >= POINTS + POINTS * POINTS:                         # REMOVE
            p = action - (POINTS + POINTS * POINTS)
            board[p] = 0
            return _settle(tuple(board), 1 - to, state.hands, False, 0)
        if action < POINTS:                                            # PLACE
            board[action] = to + 1
            hands = list(state.hands)
            hands[to] -= 1
            if _forms_mill(tuple(board), action, to + 1):
                return _settle(tuple(board), to, tuple(hands), True, 0)   # sub-turn: to_move FIXED
            return _settle(tuple(board), 1 - to, tuple(hands), False, 0)
        u = action - POINTS                                           # MOVE / FLY
        f, t = u // POINTS, u % POINTS
        board[f] = 0
        board[t] = to + 1
        if _forms_mill(tuple(board), t, to + 1):
            return _settle(tuple(board), to, state.hands, True, state.idle)   # carry idle; removal resets it
        return _settle(tuple(board), 1 - to, state.hands, False, state.idle + 1)

    def is_terminal(self, state: MorrisState) -> bool:
        return state.done

    def winner(self, state: MorrisState) -> int | None:
        if not state.done or state.idle >= IDLE_LIMIT:
            return None
        return 1 - state.to_move                                       # the player to move has no future

    def returns(self, state: MorrisState) -> list[float]:
        w = self.winner(state)
        if w is None:
            return [0.0, 0.0]
        payoff = [-1.0, -1.0]
        payoff[w] = 1.0
        return payoff

    def state_key(self, state: MorrisState):
        return (state.board, state.to_move, state.hands, state.pending_removal, state.idle)

    def observation(self, state: MorrisState, player: int) -> list[float]:
        own, opp = player + 1, 2 - player
        planes = [[0.0] * CELLS for _ in range(8)]
        for p in range(POINTS):
            c = POINT_CELL[p]
            planes[2][c] = 1.0                                         # static valid-point mask plane
            if state.board[p] == own:
                planes[0][c] = 1.0
            elif state.board[p] == opp:
                planes[1][c] = 1.0
        own_hand = state.hands[player] / HANDS0
        opp_hand = state.hands[1 - player] / HANDS0
        pending = 1.0 if state.pending_removal else 0.0
        own_fly = 1.0 if state.hands[player] == 0 and _on_board(state.board, player) == 3 else 0.0
        opp_fly = 1.0 if state.hands[1 - player] == 0 and _on_board(state.board, 1 - player) == 3 else 0.0
        for c in range(POINTS):                                        # constant planes ride only on valid cells
            cell = POINT_CELL[c]
            planes[3][cell] = own_hand
            planes[4][cell] = opp_hand
            planes[5][cell] = pending
            planes[6][cell] = own_fly
            planes[7][cell] = opp_fly
        return [v for plane in planes for v in plane]

    def action_label(self, state: MorrisState, action: int) -> str:
        if not (0 <= action < NUM_ACTIONS):
            return f"?{action}"
        if action < POINTS:
            return f"P{action}"
        if action >= POINTS + POINTS * POINTS:
            return f"x{action - (POINTS + POINTS * POINTS)}"
        u = action - POINTS
        return f"{u // POINTS}-{u % POINTS}"

    def render(self, state: MorrisState) -> str:
        glyph = {0: ".", 1: "x", 2: "o"}
        rows = []
        for r in range(N):
            cells = []
            for c in range(N):
                cell = r * N + c
                cells.append(glyph[state.board[CELL_POINT[cell]]] if cell in CELL_POINT else " ")
            rows.append(" ".join(cells))
        head = f"hands {state.hands}  to_move {state.to_move}" + ("  [remove]" if state.pending_removal else "")
        if state.done:
            w = self.winner(state)
            head += "  DRAW" if w is None else f"  winner {w}"
        return head + "\n" + "\n".join(rows)

    _symmetries_cache = None

    def symmetries(self):
        """The VERIFIED symmetry group (order 16: D4 x inner/outer ring-swap), proven against the game's own
        dynamics via harness.symmetry.find_symmetries — the game only SUPPLIES the candidate isometries and the
        two transform hooks; it asserts nothing."""
        cls = type(self)
        if cls._symmetries_cache is None:
            from harness.symmetry import find_symmetries
            cls._symmetries_cache = find_symmetries(self, n_probes=200, max_plies=60,
                                                    isometries=_candidate_isometries())
        return cls._symmetries_cache

    def transform_state(self, state, iso):
        pf = _point_fwd(iso)
        nb = [0] * POINTS
        for p in range(POINTS):
            nb[pf[p]] = state.board[p]                    # board is 24-long -> map through the point perm
        return replace(state, board=tuple(nb))

    def transform_action(self, action, iso):
        pf = _point_fwd(iso)
        if action < POINTS:                              # PLACE
            return pf[action]
        if action >= POINTS + POINTS * POINTS:           # REMOVE
            return POINTS + POINTS * POINTS + pf[action - (POINTS + POINTS * POINTS)]
        u = action - POINTS                              # MOVE / FLY: relabel both endpoints
        return POINTS + pf[u // POINTS] * POINTS + pf[u % POINTS]

    def heuristic_action(self, state: MorrisState, rng: random.Random) -> int:
        actions = self.legal_actions(state)
        if state.pending_removal:
            breaking = [a for a in actions
                        if self._removal_breaks_threat(state.board, a - (POINTS + POINTS * POINTS))]
            return rng.choice(breaking or actions)
        milling = [a for a in actions if self._action_forms_mill(state, a)]
        if milling:
            return rng.choice(milling)
        blocking = [a for a in actions if self._action_blocks(state, a)]
        if blocking:
            return rng.choice(blocking)
        return rng.choice(actions)

    def _action_forms_mill(self, state: MorrisState, action: int) -> bool:
        board = list(state.board)
        val = state.to_move + 1
        if action < POINTS:
            d = action
            board[d] = val
        else:
            u = action - POINTS
            f, t = u // POINTS, u % POINTS
            board[f] = 0
            board[t] = val
            d = t
        return _forms_mill(tuple(board), d, val)

    def _action_blocks(self, state: MorrisState, action: int) -> bool:
        # occupying a point that would complete an OPPONENT mill next
        opp = 2 - state.to_move
        if action < POINTS:
            d = action
        else:
            d = (action - POINTS) % POINTS
        for m in MILLS_THROUGH[d]:
            others = [q for q in m if q != d]
            if all(state.board[q] == opp for q in others):
                return True
        return False

    def _removal_breaks_threat(self, board: tuple, p: int) -> bool:
        for m in MILLS_THROUGH[p]:
            if sum(1 for q in m if board[q] == board[p]) == 2 and any(board[q] == 0 for q in m):
                return True
        return False


def _point_fwd(iso) -> list[int]:
    """The FORWARD 24-point permutation an isometry induces (point p -> point pf[p]). Raises ValueError — the
    error verify_isometry catches — if the isometry sends a point off the 24-point set, so a bad candidate is
    refused rather than crashing the finder."""
    pf = [0] * POINTS
    for p in range(POINTS):
        q = CELL_POINT.get(iso.cell_image(POINT_CELL[p]))
        if q is None:
            raise ValueError("isometry maps a point off the 24-point set")
        pf[p] = q
    return pf


def _candidate_isometries() -> list:
    """The 16 candidates for the order-16 group D4 x ring-swap: the 8 grid-dihedral maps (which the finder would
    propose on its own) plus the 8 ring-swap composites (which it cannot, the inner<->outer swap being no grid
    isometry). Every one is still PROVEN against the game's dynamics by find_symmetries; a construction bug yields
    fewer than 16, never a false symmetry."""
    from harness.symmetry import dihedral_isometries, iso_from_cell_map

    base = dihedral_isometries(N, N)                     # 8, identity first
    swap = {}
    for pos in range(8):
        a, b = POINT_CELL[pos], POINT_CELL[16 + pos]     # outer pos <-> inner pos
        swap[a], swap[b] = b, a
    rs = lambda c: swap.get(c, c)                        # fixes middle-ring point-cells and all dead cells
    out = list(base)
    for d in base:
        out.append(iso_from_cell_map(f"ring_swap.{d.name}", N, N, lambda c, d=d: rs(d.cell_image(c))))
    return out
