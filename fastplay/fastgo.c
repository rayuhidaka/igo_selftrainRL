/*
 * fastgo: the rules engine and PUCT search of engine/position.py, engine/scoring.py and
 * mcts/mcts.py in C, for batched self-play (fastplay/README.md). Loaded with ctypes by
 * fastplay/fastgo.py, which builds it with the system C compiler on first use.
 *
 * It must decide exactly what the Python versions decide: tests/test_fastgo.py checks legal
 * moves, captures, ko and scores against engine/ on random games, and visit counts against
 * mcts/mcts.py with the same net. If you change the rules or the search in one place, change
 * the other too.
 *
 * Boards are row-major, stones 0 empty / 1 black / 2 white. A move is a point index, or
 * size*size for a pass.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define FG_MAX_POINTS 361
#define FG_EMPTY 0
#define FG_BLACK 1
#define FG_WHITE 2
#define FG_NO_KO (-1)
#define FG_MAX_PATH 4100 /* longest root-to-leaf path; a search runs at most 4096 simulations */

typedef struct {
    int32_t size;
    int32_t to_play;
    int32_t ko;          /* point index or FG_NO_KO */
    int32_t pass_count;
    int8_t stones[FG_MAX_POINTS];
} fg_board;

static int opponent(int color) { return color == FG_BLACK ? FG_WHITE : FG_BLACK; }

/* The neighbors of point p, in engine/position.py's order: up, down, left, right. */
static int neighbors(int size, int p, int out[4]) {
    int row = p / size, col = p % size, n = 0;
    if (row > 0) out[n++] = p - size;
    if (row < size - 1) out[n++] = p + size;
    if (col > 0) out[n++] = p - 1;
    if (col < size - 1) out[n++] = p + 1;
    return n;
}

/* Flood-fills the chain at start on stones; marks it in chain[] (1) and returns whether it
 * has a liberty. Stores its members in members[] when given, returning their count via
 * *count. */
static int chain_has_liberty(int size, const int8_t *stones, int start, uint8_t *mark, int *members, int *count) {
    int stack[FG_MAX_POINTS], top = 0, n = 0, has_liberty = 0, nb[4];
    int color = stones[start];
    mark[start] = 1;
    stack[top++] = start;
    while (top > 0) {
        int p = stack[--top];
        if (members) members[n] = p;
        n++;
        int k = neighbors(size, p, nb);
        for (int i = 0; i < k; i++) {
            int q = nb[i];
            if (stones[q] == FG_EMPTY) {
                has_liberty = 1;
            } else if (stones[q] == color && !mark[q]) {
                mark[q] = 1;
                stack[top++] = q;
            }
        }
    }
    if (count) *count = n;
    return has_liberty;
}

/* Whether a stone of the player to move at p would be suicide (position.py's _would_be_suicide). */
static int would_be_suicide(const fg_board *b, int p) {
    int size = b->size, nb[4];
    int8_t trial[FG_MAX_POINTS];
    uint8_t mark[FG_MAX_POINTS];
    memcpy(trial, b->stones, size * size);
    trial[p] = (int8_t)b->to_play;
    int opp = opponent(b->to_play);
    int k = neighbors(size, p, nb);
    for (int i = 0; i < k; i++) {
        if (trial[nb[i]] == opp) {
            memset(mark, 0, size * size);
            if (!chain_has_liberty(size, trial, nb[i], mark, NULL, NULL)) return 0; /* captures */
        }
    }
    memset(mark, 0, size * size);
    return !chain_has_liberty(size, trial, p, mark, NULL, NULL);
}

/** Whether move (a point, or size*size for a pass) is legal for b's player to move. */
int fg_is_legal(const fg_board *b, int move) {
    int points = b->size * b->size;
    if (move == points) return 1;
    if (move < 0 || move > points) return 0;
    if (b->stones[move] != FG_EMPTY) return 0;
    if (move == b->ko) return 0;
    return !would_be_suicide(b, move);
}

/** Writes 1/0 per move (size*size points, then the pass) into out; returns how many are legal. */
int fg_legal_mask(const fg_board *b, uint8_t *out) {
    int points = b->size * b->size, n = 0;
    for (int p = 0; p <= points; p++) {
        out[p] = (uint8_t)fg_is_legal(b, p);
        n += out[p];
    }
    return n;
}

/** Plays move in place (position.py's play). Returns the number of stones captured, or -1
 * (board unchanged) if the move is illegal. */
int fg_play(fg_board *b, int move) {
    int size = b->size, points = size * size, nb[4];
    if (!fg_is_legal(b, move)) return -1;
    if (move == points) {
        b->to_play = opponent(b->to_play);
        b->ko = FG_NO_KO;
        b->pass_count += 1;
        return 0;
    }
    int opp = opponent(b->to_play);
    uint8_t captured[FG_MAX_POINTS], mark[FG_MAX_POINTS];
    int members[FG_MAX_POINTS], count, total = 0, last = -1;
    memset(captured, 0, points);
    b->stones[move] = (int8_t)b->to_play;
    int k = neighbors(size, move, nb);
    for (int i = 0; i < k; i++) {
        int q = nb[i];
        if (b->stones[q] == opp && !captured[q]) {
            memset(mark, 0, points);
            if (!chain_has_liberty(size, b->stones, q, mark, members, &count)) {
                for (int j = 0; j < count; j++) captured[members[j]] = 1;
            }
        }
    }
    for (int p = 0; p < points; p++) {
        if (captured[p]) {
            b->stones[p] = FG_EMPTY;
            total++;
            last = p;
        }
    }
    int new_ko = FG_NO_KO;
    if (total == 1) {
        /* Ko: the stone just played is alone and its only liberty is the captured point. */
        memset(mark, 0, points);
        chain_has_liberty(size, b->stones, move, mark, members, &count);
        if (count == 1) {
            int liberties = 0, only = -1;
            k = neighbors(size, move, nb);
            for (int i = 0; i < k; i++) {
                if (b->stones[nb[i]] == FG_EMPTY) {
                    liberties++;
                    only = nb[i];
                }
            }
            if (liberties == 1 && only == last) new_ko = last;
        }
    }
    b->to_play = opp;
    b->ko = new_ko;
    b->pass_count = 0;
    return total;
}

/** Writes each point's owner under area scoring (scoring.py's territory_ownership) into out:
 * the stone's color, the only color bordering its empty region, or 0. */
void fg_ownership(const fg_board *b, int8_t *out) {
    int size = b->size, points = size * size, nb[4];
    uint8_t visited[FG_MAX_POINTS];
    int region[FG_MAX_POINTS], stack[FG_MAX_POINTS];
    memset(visited, 0, points);
    for (int p = 0; p < points; p++) {
        if (visited[p]) continue;
        if (b->stones[p] != FG_EMPTY) {
            visited[p] = 1;
            out[p] = b->stones[p];
            continue;
        }
        int n = 0, top = 0, borders = 0; /* bit 1 black, bit 2 white */
        visited[p] = 1;
        stack[top++] = p;
        while (top > 0) {
            int q = stack[--top];
            region[n++] = q;
            int k = neighbors(size, q, nb);
            for (int i = 0; i < k; i++) {
                int r = nb[i];
                if (b->stones[r] == FG_EMPTY) {
                    if (!visited[r]) {
                        visited[r] = 1;
                        stack[top++] = r;
                    }
                } else {
                    borders |= b->stones[r];
                }
            }
        }
        int owner = borders == FG_BLACK ? FG_BLACK : borders == FG_WHITE ? FG_WHITE : FG_EMPTY;
        for (int i = 0; i < n; i++) out[region[i]] = (int8_t)owner;
    }
}

/** Area score (stones plus surrounded empty points) per color into *black / *white. */
void fg_area_score(const fg_board *b, int *black, int *white) {
    int8_t owner[FG_MAX_POINTS];
    fg_ownership(b, owner);
    *black = *white = 0;
    for (int p = 0; p < b->size * b->size; p++) {
        if (owner[p] == FG_BLACK) (*black)++;
        else if (owner[p] == FG_WHITE) (*white)++;
    }
}

/** Area-score winner with komi for white: FG_BLACK, FG_WHITE, or FG_EMPTY for a tie. */
int fg_winner(const fg_board *b, double komi) {
    int black, white;
    fg_area_score(b, &black, &white);
    double white_total = white + komi;
    if (black > white_total) return FG_BLACK;
    if (white_total > black) return FG_WHITE;
    return FG_EMPTY;
}

/* ---------------------------------------------------------------------------------------- */
/* Cleanup rule (mcts/dead_stone_estimator.py's is_cleanup_move and is_settled), used to      */
/* play a finished game on until its dead stones are off the board, for ownership labels.     */
/* ---------------------------------------------------------------------------------------- */

/* The number of distinct liberties of the chain at start; *only gets the liberty when there is
 * exactly one. */
static int chain_liberties(int size, const int8_t *stones, int start, int *only) {
    int stack[FG_MAX_POINTS], top = 0, count = 0, nb[4];
    uint8_t mark[FG_MAX_POINTS], liberty[FG_MAX_POINTS];
    int color = stones[start];
    memset(mark, 0, size * size);
    memset(liberty, 0, size * size);
    mark[start] = 1;
    stack[top++] = start;
    while (top > 0) {
        int p = stack[--top];
        int k = neighbors(size, p, nb);
        for (int i = 0; i < k; i++) {
            int q = nb[i];
            if (stones[q] == FG_EMPTY) {
                if (!liberty[q]) {
                    liberty[q] = 1;
                    count++;
                    *only = q;
                }
            } else if (stones[q] == color && !mark[q]) {
                mark[q] = 1;
                stack[top++] = q;
            }
        }
    }
    return count;
}

/* scoring.py's saves_chain_in_atari: playing p gives a liberty to a color chain whose only
 * liberty is p. */
static int saves_chain_in_atari(const fg_board *b, int p, int color) {
    int nb[4], k = neighbors(b->size, p, nb), only = -1;
    for (int i = 0; i < k; i++) {
        if (b->stones[nb[i]] != color) continue;
        if (chain_liberties(b->size, b->stones, nb[i], &only) == 1 && only == p) return 1;
    }
    return 0;
}

/* dead_stone_estimator.py's is_self_atari, for a legal point p: playing it captures nothing
 * and leaves the mover's chain with at most one liberty. */
static int is_self_atari(const fg_board *b, int p) {
    fg_board after = *b;
    int only = -1;
    if (fg_play(&after, p) != 0) return 0; /* captured something (or illegal) */
    return chain_liberties(b->size, after.stones, p, &only) <= 1;
}

/** Writes 1/0 per move into out (size*size points, then the pass, always 0): the legal moves
 * dead_stone_estimator.py's is_cleanup_move allows -- not filling the mover's own area
 * (unless it saves a chain in atari), not inside the other side's area, not a self-atari.
 * Returns how many there are. */
int fg_cleanup_mask(const fg_board *b, uint8_t *out) {
    int points = b->size * b->size, n = 0, mover = b->to_play, opp = opponent(b->to_play);
    int8_t owner[FG_MAX_POINTS];
    fg_ownership(b, owner);
    out[points] = 0;
    for (int p = 0; p < points; p++) {
        int allowed = fg_is_legal(b, p)
            && !(owner[p] == mover && !saves_chain_in_atari(b, p, mover))
            && owner[p] != opp
            && !is_self_atari(b, p);
        out[p] = (uint8_t)allowed;
        n += allowed;
    }
    return n;
}

/** dead_stone_estimator.py's is_settled: every empty point counts for exactly one color. */
int fg_is_settled(const fg_board *b) {
    int8_t owner[FG_MAX_POINTS];
    fg_ownership(b, owner);
    for (int p = 0; p < b->size * b->size; p++) {
        if (b->stones[p] == FG_EMPTY && owner[p] == FG_EMPTY) return 0;
    }
    return 1;
}

/** The net's input (bootstrap/inference.py's encode_planes): [3][size][size] own, opponent,
 * empty, relative to the player to move. */
void fg_encode(const fg_board *b, float *out) {
    int points = b->size * b->size, opp = opponent(b->to_play);
    for (int p = 0; p < points; p++) {
        int s = b->stones[p];
        out[p] = s == b->to_play ? 1.0f : 0.0f;
        out[points + p] = s == opp ? 1.0f : 0.0f;
        out[2 * points + p] = s == FG_EMPTY ? 1.0f : 0.0f;
    }
}

/* ---------------------------------------------------------------------------------------- */
/* Batched PUCT search: one tree per game, each searched exactly like mcts/mcts.py (one      */
/* simulation at a time, no virtual loss); the batch is across games. Per step, fg_collect   */
/* walks every unfinished search down to a leaf that needs the net and encodes it, Python    */
/* evaluates them all at once, and fg_expand adds the children and backs the values up.      */
/* ---------------------------------------------------------------------------------------- */

typedef struct {
    double prior;
    double value_sum;
    int32_t visits;
    int32_t first_child; /* -1 until expanded */
    int16_t num_children;
    int16_t move;
} fg_node;

typedef struct {
    fg_board root;
    fg_node *nodes;
    int32_t num_nodes, capacity;
    int32_t sims_done, sims_target;
    double c_puct, komi, noise_epsilon;
    double fpu_reduction; /* < 0: an unvisited child counts as 0.0 (mcts.py's fpu_reduction None) */
    double noise[FG_MAX_POINTS + 1];
    int32_t noise_len;
    int32_t active;
    /* The leaf waiting for the net, and its path from the root. */
    int32_t pending;
    int32_t path[FG_MAX_PATH];
    int32_t path_len;
    fg_board leaf;
} fg_search;

typedef struct {
    int32_t num_games, size;
    fg_search *games;
    int32_t *pending_order; /* game index of each leaf in the last fg_collect */
    int32_t num_pending;
} fg_pool;

/** A pool of num_games searches on size x size boards, each tree holding up to node_capacity nodes. */
fg_pool *fg_pool_new(int num_games, int size, int node_capacity) {
    fg_pool *pool = calloc(1, sizeof(fg_pool));
    pool->num_games = num_games;
    pool->size = size;
    pool->games = calloc(num_games, sizeof(fg_search));
    pool->pending_order = calloc(num_games, sizeof(int32_t));
    for (int g = 0; g < num_games; g++) {
        pool->games[g].nodes = malloc(sizeof(fg_node) * (size_t)node_capacity);
        pool->games[g].capacity = node_capacity;
    }
    return pool;
}

void fg_pool_free(fg_pool *pool) {
    for (int g = 0; g < pool->num_games; g++) free(pool->games[g].nodes);
    free(pool->games);
    free(pool->pending_order);
    free(pool);
}

/** Starts game g's search from root: sims simulations, PUCT constant c_puct, komi for finished
 * games. noise (noise_len values, one per legal board move in point order, or NULL) is mixed
 * into the root's priors with weight noise_epsilon when the root is expanded, as mcts.py's
 * _add_root_noise does. */
void fg_search_begin(fg_pool *pool, int g, const fg_board *root, int sims, double c_puct, double komi,
                     const double *noise, int noise_len, double noise_epsilon, double fpu_reduction) {
    fg_search *s = &pool->games[g];
    s->root = *root;
    s->num_nodes = 1;
    s->nodes[0] = (fg_node){.prior = 1.0, .value_sum = 0.0, .visits = 0, .first_child = -1, .num_children = 0, .move = -1};
    s->sims_done = 0;
    s->sims_target = sims;
    s->c_puct = c_puct;
    s->komi = komi;
    s->noise_epsilon = noise_epsilon;
    s->fpu_reduction = fpu_reduction;
    s->noise_len = noise ? noise_len : 0;
    if (noise) memcpy(s->noise, noise, sizeof(double) * noise_len);
    s->active = sims > 0;
    s->pending = 0;
}

static double mean_value(const fg_node *n) { return n->visits == 0 ? 0.0 : n->value_sum / n->visits; }

/* mcts.py's _select_child: the first child with the highest PUCT score. */
static int select_child(const fg_search *s, const fg_node *node) {
    double parent = sqrt((double)node->visits);
    /* mcts.py's _first_play_value: an unvisited child's worth. */
    double fpu = 0.0;
    if (s->fpu_reduction >= 0.0) {
        double visited_prior = 0.0;
        for (int i = 0; i < node->num_children; i++) {
            const fg_node *child = &s->nodes[node->first_child + i];
            if (child->visits > 0) visited_prior += child->prior;
        }
        fpu = mean_value(node) - s->fpu_reduction * sqrt(visited_prior);
    }
    int best = -1;
    double best_score = 0.0;
    for (int i = 0; i < node->num_children; i++) {
        const fg_node *child = &s->nodes[node->first_child + i];
        double exploitation = child->visits > 0 ? -mean_value(child) : fpu;
        double exploration = s->c_puct * child->prior * parent / (1 + child->visits);
        double score = exploitation + exploration;
        if (best < 0 || score > best_score) {
            best = node->first_child + i;
            best_score = score;
        }
    }
    return best;
}

static void backup(fg_search *s, double value) {
    for (int i = s->path_len - 1; i >= 0; i--) {
        fg_node *n = &s->nodes[s->path[i]];
        n->visits += 1;
        n->value_sum += value;
        value = -value;
    }
    s->sims_done += 1;
    if (s->sims_done >= s->sims_target) s->active = 0;
}

static double terminal_value(const fg_search *s, const fg_board *b) {
    int winner = fg_winner(b, s->komi);
    if (winner == b->to_play) return 1.0;
    if (winner == FG_EMPTY) return 0.0;
    return -1.0;
}

/** Walks every active, non-pending search down to a leaf that needs the net, finishing any
 * simulation that ends in a finished game on the way. Writes each such leaf's planes into
 * planes_out ([n][3][size][size]) and returns n. Leaves are in game order; fg_expand takes
 * the net's results in the same order. */
int fg_collect(fg_pool *pool, float *planes_out) {
    int points = pool->size * pool->size, n = 0;
    for (int g = 0; g < pool->num_games; g++) {
        fg_search *s = &pool->games[g];
        while (s->active && !s->pending) {
            fg_board b = s->root;
            int node = 0;
            s->path_len = 0;
            s->path[s->path_len++] = 0;
            for (;;) {
                if (b.pass_count >= 2) {
                    backup(s, terminal_value(s, &b));
                    break;
                }
                fg_node *current = &s->nodes[node];
                if (current->first_child < 0) {
                    s->pending = 1;
                    s->leaf = b;
                    fg_encode(&b, planes_out + (size_t)n * 3 * points);
                    pool->pending_order[n++] = g;
                    break;
                }
                node = select_child(s, current);
                fg_play(&b, s->nodes[node].move);
                s->path[s->path_len++] = node;
            }
        }
    }
    pool->num_pending = n;
    return n;
}

/** Expands the leaves of the last fg_collect with the net's raw policy (n rows of size*size+1,
 * taken for the legal moves as they are, not renormalized, like inference.py's decode_policy)
 * and backs up values (n, from each leaf's player to move). Returns -1 if a tree ran out of
 * nodes, else 0. */
int fg_expand(fg_pool *pool, const float *policy, const float *values) {
    int points = pool->size * pool->size;
    uint8_t legal[FG_MAX_POINTS + 1];
    for (int i = 0; i < pool->num_pending; i++) {
        fg_search *s = &pool->games[pool->pending_order[i]];
        int leaf = s->path[s->path_len - 1];
        int count = fg_legal_mask(&s->leaf, legal);
        if (s->num_nodes + count > s->capacity) return -1;
        int first = s->num_nodes;
        int is_root = leaf == 0;
        int board_index = 0;
        for (int m = 0; m <= points; m++) {
            if (!legal[m]) continue;
            double prior = policy[(size_t)i * (points + 1) + m];
            if (is_root && s->noise_len > 0 && m < points) {
                prior = (1 - s->noise_epsilon) * prior + s->noise_epsilon * s->noise[board_index];
                board_index++;
            }
            s->nodes[s->num_nodes++] = (fg_node){.prior = prior, .value_sum = 0.0, .visits = 0, .first_child = -1, .num_children = 0, .move = (int16_t)m};
        }
        s->nodes[leaf].first_child = first;
        s->nodes[leaf].num_children = (int16_t)count;
        s->pending = 0;
        backup(s, (double)values[i]);
    }
    pool->num_pending = 0;
    return 0;
}

/** Game g's root visit count per move into out (size*size+1, 0 for illegal moves), and its
 * root value (mcts.py's root_value) as the return value. */
double fg_root_visits(const fg_pool *pool, int g, int32_t *out) {
    const fg_search *s = &pool->games[g];
    int points = pool->size * pool->size;
    memset(out, 0, sizeof(int32_t) * (points + 1));
    const fg_node *root = &s->nodes[0];
    for (int i = 0; root->first_child >= 0 && i < root->num_children; i++) {
        const fg_node *child = &s->nodes[root->first_child + i];
        out[child->move] = child->visits;
    }
    return mean_value(root);
}

/** Game g's root priors per move (size*size+1, root noise mixed in; 0 for illegal moves). */
void fg_root_priors(const fg_pool *pool, int g, double *out) {
    const fg_search *s = &pool->games[g];
    int points = pool->size * pool->size;
    memset(out, 0, sizeof(double) * (points + 1));
    const fg_node *root = &s->nodes[0];
    for (int i = 0; root->first_child >= 0 && i < root->num_children; i++) {
        const fg_node *child = &s->nodes[root->first_child + i];
        out[child->move] = child->prior;
    }
}

/** Game g's principal variation into out (at most max_len moves) and its length: from the root,
 * the most visited child at each step, as long as it was visited at least min_visits times and
 * isn't a pass (the app's SearchResult.moveLines rule; best-move puzzles, igo-app PUZZLES.md). */
int fg_principal_variation(const fg_pool *pool, int g, int32_t *out, int max_len, int min_visits) {
    const fg_search *s = &pool->games[g];
    int pass = pool->size * pool->size;
    const fg_node *node = &s->nodes[0];
    int len = 0;
    while (len < max_len && node->first_child >= 0) {
        const fg_node *best = NULL;
        for (int i = 0; i < node->num_children; i++) {
            const fg_node *child = &s->nodes[node->first_child + i];
            if (best == NULL || child->visits > best->visits) best = child;
        }
        if (best == NULL || best->visits < min_visits || best->move == pass) break;
        out[len++] = best->move;
        node = best;
    }
    return len;
}

/** Whether game g's search has run all its simulations. */
int fg_search_done(const fg_pool *pool, int g) { return !pool->games[g].active; }

/** The game index of the i-th leaf of the last fg_collect. */
int fg_pending_game(const fg_pool *pool, int i) { return pool->pending_order[i]; }
