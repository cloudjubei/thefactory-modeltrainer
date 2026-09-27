/* §C.49 — the exact Connect-4 solve of harness/solver.py, ported line for line to C so certifying a net's play
 * from the empty board is feasible (the Python solve takes minutes to hours at ply 8). The Python module stays the
 * REFERENCE: every function here mirrors the one of the same name there, and tests/test_native_solver.py checks
 * the two agree position by position. Python's floor division on negative scores is reproduced exactly
 * (fdiv2), as is its stable descending sort of the candidate moves.
 *
 * The transposition table is direct-mapped with the full key stored, so a collision only replaces an entry and
 * can never return another position's bound: values are exact whatever the table size. Unlike the Python reference,
 * which keeps only the upper bound a fail-low proves, each entry also keeps the LOWER bound a cutoff proves — the
 * standard two-sided alpha-beta table. Bounds only narrow the search window, so every value stays exact; the tests
 * hold the two solvers to identical answers. */
#include <stdint.h>
#include <stdlib.h>

#define WIDTH 7
#define HEIGHT 6
#define H1 (HEIGHT + 1)
#define TOTAL (WIDTH * HEIGHT)
#define MIN_SCORE (-(TOTAL / 2) + 3)

static const int CENTER_ORDER[WIDTH] = {3, 2, 4, 1, 5, 0, 6};
static uint64_t BOTTOM_MASK, BOARD_MASK, COL_MASKS[WIDTH], MC[WIDTH];
static uint64_t *tt_keys = NULL;
static uint8_t *tt_upper = NULL;
static uint8_t *tt_lower = NULL;
static uint64_t tt_size = 0;

static int fdiv2(int x) { return x >= 0 ? x / 2 : -((-x + 1) / 2); }
static int half(int x) { return x / 2; }

int c4_init(int log2_size) {
    BOTTOM_MASK = 0;
    for (int c = 0; c < WIDTH; c++) {
        BOTTOM_MASK |= (uint64_t)1 << (c * H1);
        COL_MASKS[c] = (((uint64_t)1 << HEIGHT) - 1) << (c * H1);
        MC[c] = (((uint64_t)1 << H1) - 1) << (c * H1);
    }
    BOARD_MASK = BOTTOM_MASK * (((uint64_t)1 << HEIGHT) - 1);
    free(tt_keys);
    free(tt_upper);
    free(tt_lower);
    tt_size = (uint64_t)1 << log2_size;
    tt_keys = calloc(tt_size, sizeof(uint64_t));
    tt_upper = calloc(tt_size, sizeof(uint8_t));
    tt_lower = calloc(tt_size, sizeof(uint8_t));
    return tt_keys != NULL && tt_upper != NULL && tt_lower != NULL;
}

void c4_reset(void) {
    for (uint64_t i = 0; i < tt_size; i++) {
        tt_keys[i] = 0;
        tt_upper[i] = 0;
        tt_lower[i] = 0;
    }
}

static uint64_t mirror(uint64_t x) {
    return ((x & MC[0]) << (6 * H1)) | ((x & MC[1]) << (4 * H1)) | ((x & MC[2]) << (2 * H1)) | (x & MC[3]) |
           ((x & MC[4]) >> (2 * H1)) | ((x & MC[5]) >> (4 * H1)) | ((x & MC[6]) >> (6 * H1));
}

uint64_t c4_canonical_key(uint64_t position, uint64_t mask) {
    uint64_t k = position + mask, km = mirror(position) + mirror(mask);
    return k <= km ? k : km;
}

static int alignment(uint64_t pos) {
    uint64_t m = pos & (pos >> H1);
    if (m & (m >> (2 * H1))) return 1;
    m = pos & (pos >> HEIGHT);
    if (m & (m >> (2 * HEIGHT))) return 1;
    m = pos & (pos >> (HEIGHT + 2));
    if (m & (m >> (2 * (HEIGHT + 2)))) return 1;
    m = pos & (pos >> 1);
    if (m & (m >> 2)) return 1;
    return 0;
}

static uint64_t compute_winning_position(uint64_t position, uint64_t mask) {
    uint64_t r = (position << 1) & (position << 2) & (position << 3);
    uint64_t p = (position << H1) & (position << (2 * H1));
    r |= p & (position << (3 * H1));
    r |= p & (position >> H1);
    p = (position >> H1) & (position >> (2 * H1));
    r |= p & (position << H1);
    r |= p & (position >> (3 * H1));
    p = (position << HEIGHT) & (position << (2 * HEIGHT));
    r |= p & (position << (3 * HEIGHT));
    r |= p & (position >> HEIGHT);
    p = (position >> HEIGHT) & (position >> (2 * HEIGHT));
    r |= p & (position << HEIGHT);
    r |= p & (position >> (3 * HEIGHT));
    const int d = HEIGHT + 2;
    p = (position << d) & (position << (2 * d));
    r |= p & (position << (3 * d));
    r |= p & (position >> d);
    p = (position >> d) & (position >> (2 * d));
    r |= p & (position << d);
    r |= p & (position >> (3 * d));
    return r & (BOARD_MASK ^ mask);
}

static uint64_t possible(uint64_t mask) { return (mask + BOTTOM_MASK) & BOARD_MASK; }

static int can_win_next(uint64_t position, uint64_t mask) {
    return (compute_winning_position(position, mask) & possible(mask)) != 0;
}

static uint64_t possible_non_losing_moves(uint64_t position, uint64_t mask) {
    uint64_t possible_mask = possible(mask);
    uint64_t opponent_win = compute_winning_position(position ^ mask, mask);
    uint64_t forced = possible_mask & opponent_win;
    if (forced) {
        if (forced & (forced - 1)) return 0;
        possible_mask = forced;
    }
    return possible_mask & ~(opponent_win >> 1);
}

static int move_score(uint64_t position, uint64_t mask, uint64_t move) {
    return __builtin_popcountll(compute_winning_position(position | move, mask));
}

static void store(uint64_t slot, uint64_t key, int is_lower, int value) {
    if (tt_keys[slot] != key) {
        tt_keys[slot] = key;
        tt_upper[slot] = 0;
        tt_lower[slot] = 0;
    }
    uint8_t encoded = (uint8_t)(value - MIN_SCORE + 1);
    if (is_lower)
        tt_lower[slot] = encoded;
    else
        tt_upper[slot] = encoded;
}

static int negamax(uint64_t position, uint64_t mask, int moves, int alpha, int beta) {
    uint64_t poss = possible_non_losing_moves(position, mask);
    if (poss == 0) return fdiv2(-(TOTAL - moves));
    if (moves >= TOTAL - 2) return 0;
    int min_score = fdiv2(-(TOTAL - 2 - moves));
    if (alpha < min_score) {
        alpha = min_score;
        if (alpha >= beta) return alpha;
    }
    int max_score = (TOTAL - 1 - moves) / 2;
    if (beta > max_score) {
        beta = max_score;
        if (alpha >= beta) return beta;
    }
    uint64_t key = c4_canonical_key(position, mask);
    uint64_t slot = key & (tt_size - 1);
    if (tt_keys[slot] == key) {
        if (tt_upper[slot]) {
            int tt_max = (int)tt_upper[slot] + MIN_SCORE - 1;
            if (beta > tt_max) {
                beta = tt_max;
                if (alpha >= beta) return beta;
            }
        }
        if (tt_lower[slot]) {
            int tt_min = (int)tt_lower[slot] + MIN_SCORE - 1;
            if (alpha < tt_min) {
                alpha = tt_min;
                if (alpha >= beta) return alpha;
            }
        }
    }
    uint64_t moves_list[WIDTH];
    int scores[WIDTH], n = 0;
    for (int i = 0; i < WIDTH; i++) {
        uint64_t move = poss & COL_MASKS[CENTER_ORDER[i]];
        if (!move) continue;
        int s = move_score(position, mask, move), j = n++;
        while (j > 0 && scores[j - 1] < s) {
            scores[j] = scores[j - 1];
            moves_list[j] = moves_list[j - 1];
            j--;
        }
        scores[j] = s;
        moves_list[j] = move;
    }
    for (int i = 0; i < n; i++) {
        int score = -negamax(position ^ mask, mask | moves_list[i], moves + 1, -beta, -alpha);
        if (score >= beta) {
            store(slot, key, 1, score);
            return score;
        }
        if (score > alpha) alpha = score;
    }
    store(slot, key, 0, alpha);
    return alpha;
}

int c4_solve(uint64_t position, uint64_t mask, int moves, int weak) {
    if (can_win_next(position, mask)) return (TOTAL + 1 - moves) / 2;
    int min_s = fdiv2(-(TOTAL - moves)), max_s = (TOTAL + 1 - moves) / 2;
    if (weak) {
        min_s = -1;
        max_s = 1;
    }
    while (min_s < max_s) {
        int med = min_s + (max_s - min_s) / 2;
        if (med <= 0 && half(min_s) < med)
            med = half(min_s);
        else if (med >= 0 && half(max_s) > med)
            med = half(max_s);
        int r = negamax(position, mask, moves, med, med + 1);
        if (r <= med)
            max_s = r;
        else
            min_s = r;
    }
    return min_s;
}
