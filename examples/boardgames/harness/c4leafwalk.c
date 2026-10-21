/* §3.6 — harness.steady_exceptions.needed for Connect-4, in C (h200: the Python walk is > 95% of a strategy build,
 * ~15K positions a second). One walk over every line from a root: at each of the root side's positions the
 * steady-state rule of harness.steady_state.choose plays (win at once; else the only safe move; else the first level
 * of the priority map holding exactly one safe move — or, with `ties_lowest`, any safe move, the lowest column
 * deciding; else no move — a move safe when the opponent cannot win at once after it, and with `safe_depth` 2 nor
 * force a win in two; with `forcing`, of several safe moves those leaving the opponent at most one reply that stops an
 * immediate win, when any do; with `unlevelled_last`, the safe moves on unlevelled cells read as a last level; with
 * `switch_at` >= 0, the map's second 42 levels from that many stones on), and where its move is not a winning one an
 * exception plays the lowest winning column instead; every reply is followed. Moves are judged by the exact solve of
 * harness/c4solver.c (included below), and only the moves that decide: the map's move where it gives one, otherwise
 * columns from the left up to the first winning one (h210) — each result cached by the position it leads to across
 * walks. The Python walk stays the REFERENCE:
 * tests/test_native_leaf.py holds the two to identical exceptions, counts and statuses.
 *
 * Bitboards as c4solver.c: `position` holds the stones of the side to move, `mask` all stones, bit = col * 7 + row.
 * A cell of the priority map is row * 7 + col, row 0 the bottom (games/connect4.py). */
#include <time.h>

#include "c4solver.c"

#define WALK_OK 0
#define WALK_CAP 1
#define WALK_TIMEOUT 2
#define WALK_NO_WINNING_MOVE 3
#define WALK_LINE_NOT_WON 4
#define WALK_ROOT_UNWINNABLE 5
#define WALK_EXCEPTIONS_FULL 6
#define WALK_OUT_OF_MEMORY 7
#define VERIFY_UNDEFINED 8
#define VERIFY_DRAW 9
#define VERIFY_LOSS 10
#define CHECK_EVERY 1024

static uint64_t *move_keys = NULL;
static uint8_t *move_wins_cache = NULL;
static uint64_t move_size = 0;

int c4_leaf_init(int log2_solver_table, int log2_move_cache) {
    if (!c4_init(log2_solver_table)) return 0;
    free(move_keys);
    free(move_wins_cache);
    move_size = (uint64_t)1 << log2_move_cache;
    move_keys = calloc(move_size, sizeof(uint64_t));
    move_wins_cache = calloc(move_size, sizeof(uint8_t));
    return move_keys != NULL && move_wins_cache != NULL;
}

static uint64_t top_bit(int col) { return (uint64_t)1 << (col * H1 + HEIGHT - 1); }
static uint64_t bottom_bit(int col) { return (uint64_t)1 << (col * H1); }
static int playable(uint64_t mask, int col) { return (mask & top_bit(col)) == 0; }
static uint64_t move_bit(uint64_t mask, int col) { return (mask + bottom_bit(col)) & COL_MASKS[col]; }
static int full_after(uint64_t mask) { return (mask & BOARD_MASK) == BOARD_MASK; }

static int landing_row(uint64_t mask, int col) {
    return __builtin_popcountll(mask & COL_MASKS[col]);
}

static uint64_t mix(uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    return key;
}

/* Whether column c wins for the side to move: at once, or because the exact solve of the position after it is a loss
 * for the opponent; cached by that position. */
static int move_wins(uint64_t position, uint64_t mask, int moves, int c) {
    uint64_t mb = move_bit(mask, c), mine = position | mb, all = mask | mb;
    if (alignment(mine)) return 1;
    if (full_after(all)) return 0;
    uint64_t key = (all ^ mine) + all, slot = mix(key) & (move_size - 1);
    if (move_keys[slot] == key) return move_wins_cache[slot];
    int wins = c4_solve(all ^ mine, all, moves + 1, 1) < 0;
    move_keys[slot] = key;
    move_wins_cache[slot] = (uint8_t)wins;
    return wins;
}

/* The lowest winning column for the side to move, or -1 where none wins. */
static int lowest_winning(uint64_t position, uint64_t mask, int moves) {
    for (int c = 0; c < WIDTH; c++)
        if (playable(mask, c) && move_wins(position, mask, moves, c)) return c;
    return -1;
}

/* Whether the opponent can answer column c with a move after which every reply of the side to move loses at once.
 * Mirrors harness.steady_state.Facts._loses_in_two. */
static int loses_in_two(uint64_t position, uint64_t mask, int c) {
    uint64_t mb = move_bit(mask, c), mine = position | mb, all = mask | mb;
    if (alignment(mine) || full_after(all)) return 0;
    uint64_t theirs = all ^ mine;
    for (int b = 0; b < WIDTH; b++) {
        if (!playable(all, b)) continue;
        uint64_t bb = move_bit(all, b), theirs2 = theirs | bb, all2 = all | bb;
        if (alignment(theirs2) || full_after(all2)) continue;
        int held = 0;
        for (int r = 0; r < WIDTH && !held; r++) {
            if (!playable(all2, r)) continue;
            uint64_t rb = move_bit(all2, r), all3 = all2 | rb;
            if (alignment(mine | rb) || full_after(all3)) {
                held = 1;
                continue;
            }
            int lost = 0;
            for (int d = 0; d < WIDTH && !lost; d++)
                if (playable(all3, d) && alignment(theirs2 | move_bit(all3, d))) lost = 1;
            held = !lost;
        }
        if (!held) return 1;
    }
    return 0;
}

/* Whether every answer of the opponent's to column c leaves the side to move a column that wins at once. Called only
 * for one of several safe moves where none wins at once — so c ends nothing (a move ending the game wins nothing in two
 * in harness.steady_state.Facts._wins_in_two, which this mirrors), no answer wins for the opponent, and an answer
 * filling the board leaves no column to win with. */
static int wins_in_two(uint64_t position, uint64_t mask, int c) {
    uint64_t mb = move_bit(mask, c), mine = position | mb, all = mask | mb;
    for (int b = 0; b < WIDTH; b++) {
        if (!playable(all, b)) continue;
        uint64_t all2 = all | move_bit(all, b);
        int threat = 0;
        for (int d = 0; d < WIDTH && !threat; d++)
            if (playable(all2, d) && alignment(mine | move_bit(all2, d))) threat = 1;
        if (!threat) return 0;
    }
    return 1;
}

/* The column a deciding level holding the n columns `at` (ascending) plays: the lowest, or under `tactics` the lowest
 * of those that win in two, else of those that do not lose in two, else the lowest. Mirrors Facts.pick. */
static int pick(uint64_t position, uint64_t mask, const int *at, int n, int tactics) {
    if (!tactics || n == 1) return at[0];
    int best = at[0], best_rank = 4;
    for (int i = 0; i < n; i++) {
        int rank = (!wins_in_two(position, mask, at[i])) * 2 + loses_in_two(position, mask, at[i]);
        if (rank < best_rank) {
            best = at[i];
            best_rank = rank;
        }
    }
    return best;
}

/* Whether column c leaves the opponent at most one reply after which the side to move cannot win at once. Called only
 * for one of several safe moves where none wins at once — so c ends nothing (a move ending the game forces nothing in
 * harness.steady_state.Facts._forces, which this mirrors), no reply wins for the opponent, and a reply filling the
 * board leaves no column to win with. */
static int forces(uint64_t position, uint64_t mask, int c) {
    uint64_t mb = move_bit(mask, c), mine = position | mb, all = mask | mb;
    int safe_replies = 0;
    for (int b = 0; b < WIDTH; b++) {
        if (!playable(all, b)) continue;
        uint64_t all2 = all | move_bit(all, b);
        int threat = 0;
        for (int d = 0; d < WIDTH && !threat; d++)
            if (playable(all2, d) && alignment(mine | move_bit(all2, d))) threat = 1;
        if (!threat) safe_replies++;
    }
    return safe_replies <= 1;
}

/* The rule's column at a position of the side to move, or -1 where it gives none; *immediate set when it wins at
 * once. Mirrors harness.steady_state.Facts.of and choose (`ties_lowest`: Facts.ties == "lowest"; `safe_depth`:
 * Facts.safe_depth; `forcing`: Facts.forcing; `unlevelled_last`: Facts.unlevelled == "last"; `tactics`:
 * Facts.order == "tactics"). */
static int rule_move(uint64_t position, uint64_t mask, const int8_t *levels, int n_levels, int ties_lowest,
                     int safe_depth, int forcing, int unlevelled_last, int tactics, int *immediate) {
    int safe[WIDTH], cells[WIDTH], n_safe = 0;
    *immediate = 0;
    for (int c = 0; c < WIDTH; c++) {
        if (playable(mask, c) && alignment(position | move_bit(mask, c))) {
            *immediate = 1;
            return c;
        }
    }
    for (int c = 0; c < WIDTH; c++) {
        if (!playable(mask, c)) continue;
        uint64_t mb = move_bit(mask, c), mine = position | mb, all = mask | mb;
        int is_safe = full_after(all);
        if (!is_safe) {
            uint64_t theirs = all ^ mine;
            is_safe = 1;
            for (int b = 0; b < WIDTH && is_safe; b++)
                if (playable(all, b) && alignment(theirs | move_bit(all, b))) is_safe = 0;
        }
        if (is_safe) {
            safe[n_safe] = c;
            cells[n_safe] = landing_row(mask, c) * WIDTH + c;
            n_safe++;
        }
    }
    if (safe_depth == 2 && n_safe > 1) {
        int kept = 0;
        for (int i = 0; i < n_safe; i++)
            if (!loses_in_two(position, mask, safe[i])) {
                safe[kept] = safe[i];
                cells[kept] = cells[i];
                kept++;
            }
        if (kept) n_safe = kept;
    }
    if (forcing && n_safe > 1) {
        int kept = 0;
        for (int i = 0; i < n_safe; i++)
            if (forces(position, mask, safe[i])) {
                safe[kept] = safe[i];
                cells[kept] = cells[i];
                kept++;
            }
        if (kept) n_safe = kept;
    }
    if (n_safe == 1) return safe[0];
    int at[WIDTH];
    for (int k = 0; k < n_levels; k++) {
        int count = 0;
        for (int i = 0; i < n_safe; i++)
            if (levels[cells[i]] == k) at[count++] = safe[i];
        if (count == 1 || (count > 0 && ties_lowest)) return pick(position, mask, at, count, tactics);
    }
    if (unlevelled_last) {
        int count = 0;
        for (int i = 0; i < n_safe; i++)
            if (levels[cells[i]] < 0) at[count++] = safe[i];
        if (count == 1 || (count > 0 && ties_lowest)) return pick(position, mask, at, count, tactics);
    }
    return -1;
}

typedef struct {
    uint64_t *keys;
    uint64_t size, used;
} KeySet;

static int keyset_add(KeySet *s, uint64_t key) {
    uint64_t i = mix(key) & (s->size - 1);
    while (s->keys[i]) {
        if (s->keys[i] == key) return 0;
        i = (i + 1) & (s->size - 1);
    }
    s->keys[i] = key;
    s->used++;
    return 1;
}

typedef struct {
    uint64_t position, mask;
    int moves, done, winner_is_player;
} Node;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* One walk over every line from the root (the side to move there is "the player"). With `verify_only` it is
 * harness.steady_state.verify: the rule alone, stopping at the first line it does not win (VERIFY_UNDEFINED,
 * VERIFY_DRAW, VERIFY_LOSS). Otherwise it is harness.steady_exceptions.needed: where the rule's move does not win, an
 * exception plays the lowest winning column (written as (position, mask, column) triples). Positions are visited in
 * the Python walks' order, so counts match them even at a failure. */
static int walk(uint64_t root_position, uint64_t root_mask, int root_moves, const int8_t *levels, int n_levels,
                int switch_at, int ties_lowest, int safe_depth, int forcing, int unlevelled_last, int tactics, int64_t cap,
                double seconds, int verify_only, uint64_t *exc_position, uint64_t *exc_mask, int8_t *exc_move,
                int64_t max_exceptions, int64_t *n_exceptions, int64_t *own_positions, int64_t *positions) {
    *n_exceptions = 0;
    *own_positions = 0;
    *positions = 0;
    if (!verify_only && lowest_winning(root_position, root_mask, root_moves) < 0) return WALK_ROOT_UNWINNABLE;
    double deadline = seconds >= 0 ? now() + seconds : -1;
    KeySet seen = {NULL, 1, 0};
    while (seen.size < (uint64_t)(cap + 1) * 2) seen.size <<= 1;
    seen.keys = calloc(seen.size, sizeof(uint64_t));
    uint64_t stack_size = 1024, top = 0;
    Node *stack = malloc(stack_size * sizeof(Node));
    if (!seen.keys || !stack) {
        free(seen.keys);
        free(stack);
        return WALK_OUT_OF_MEMORY;
    }
    int status = WALK_OK, player_parity = root_moves & 1;
    stack[top++] = (Node){root_position, root_mask, root_moves, 0, 0};
    while (top) {
        Node s = stack[--top];
        if (!keyset_add(&seen, s.position + s.mask + 1)) continue;
        if ((int64_t)seen.used > cap) {
            status = WALK_CAP;
            break;
        }
        if (deadline >= 0 && seen.used % CHECK_EVERY == 1 % CHECK_EVERY && now() > deadline) {
            status = WALK_TIMEOUT;
            break;
        }
        if (s.done) {
            if (s.winner_is_player != 1) {
                status = !verify_only ? WALK_LINE_NOT_WON : s.winner_is_player == 2 ? VERIFY_LOSS : VERIFY_DRAW;
                break;
            }
            continue;
        }
        if (top + WIDTH + 1 > stack_size) {
            stack_size *= 2;
            Node *grown = realloc(stack, stack_size * sizeof(Node));
            if (!grown) {
                status = WALK_OUT_OF_MEMORY;
                break;
            }
            stack = grown;
        }
        if ((s.moves & 1) == player_parity) {
            (*own_positions)++;
            const int8_t *phase = switch_at >= 0 && s.moves >= switch_at ? levels + TOTAL : levels;
            int immediate, move = rule_move(s.position, s.mask, phase, n_levels, ties_lowest, safe_depth, forcing,
                                            unlevelled_last, tactics, &immediate);
            if (verify_only && move < 0) {
                status = VERIFY_UNDEFINED;
                break;
            }
            if (!verify_only && !immediate && (move < 0 || !move_wins(s.position, s.mask, s.moves, move))) {
                int fix = lowest_winning(s.position, s.mask, s.moves);
                if (fix < 0) {
                    status = WALK_NO_WINNING_MOVE;
                    break;
                }
                if (*n_exceptions >= max_exceptions) {
                    status = WALK_EXCEPTIONS_FULL;
                    break;
                }
                move = fix;
                exc_position[*n_exceptions] = s.position;
                exc_mask[*n_exceptions] = s.mask;
                exc_move[*n_exceptions] = (int8_t)move;
                (*n_exceptions)++;
            }
            uint64_t mb = move_bit(s.mask, move), mine = s.position | mb, all = s.mask | mb;
            int won = alignment(mine);
            stack[top++] = (Node){all ^ mine, all, s.moves + 1, won || full_after(all), won ? 1 : 0};
        } else {
            for (int b = 0; b < WIDTH; b++) {
                if (!playable(s.mask, b)) continue;
                uint64_t mb = move_bit(s.mask, b), mine = s.position | mb, all = s.mask | mb;
                int won = alignment(mine);
                stack[top++] = (Node){all ^ mine, all, s.moves + 1, won || full_after(all), won ? 2 : 0};
            }
        }
    }
    *positions = (int64_t)seen.used;
    free(seen.keys);
    free(stack);
    return status;
}

/* harness.steady_exceptions.needed: a WALK_* status; on WALK_OK the exceptions and *own_positions are set. */
int c4_leaf_walk(uint64_t root_position, uint64_t root_mask, int root_moves, const int8_t *levels, int n_levels,
                 int switch_at, int ties_lowest, int safe_depth, int forcing, int unlevelled_last, int tactics, int64_t cap,
                 double seconds, uint64_t *exc_position, uint64_t *exc_mask, int8_t *exc_move, int64_t max_exceptions,
                 int64_t *n_exceptions, int64_t *own_positions) {
    int64_t positions;
    return walk(root_position, root_mask, root_moves, levels, n_levels, switch_at, ties_lowest, safe_depth, forcing,
                unlevelled_last, tactics, cap, seconds, 0, exc_position, exc_mask, exc_move, max_exceptions, n_exceptions,
                own_positions, &positions);
}

/* harness.steady_state.verify: 0 when every line is won, else WALK_CAP, WALK_TIMEOUT, VERIFY_UNDEFINED, VERIFY_DRAW,
 * VERIFY_LOSS or WALK_OUT_OF_MEMORY; *own_positions and *positions as walked. */
int c4_verify_walk(uint64_t root_position, uint64_t root_mask, int root_moves, const int8_t *levels, int n_levels,
                   int switch_at, int ties_lowest, int safe_depth, int forcing, int unlevelled_last, int tactics, int64_t cap,
                   double seconds, int64_t *own_positions, int64_t *positions) {
    int64_t n_exceptions;
    return walk(root_position, root_mask, root_moves, levels, n_levels, switch_at, ties_lowest, safe_depth, forcing,
                unlevelled_last, tactics, cap, seconds, 1, NULL, NULL, NULL, 0, &n_exceptions, own_positions, positions);
}
