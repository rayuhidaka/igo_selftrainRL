# selfplay/

`generate.py` produces imitation-learning training data by sampling moves
directly from igo-app's real KataGo-derived checkpoint's own policy output
(no tree search) — see its docstring for why that's the right call here
(distillation from an already-strong teacher, not reinforcement-learning
self-play) and `../docs/ROADMAP.md`'s Phase 2.

Fast: ~20ms/move on this machine (one net evaluation per move), so a few
hundred games is a few minutes' work, not hours.

`self_play.py` is that separate, later work (`../docs/ROADMAP.md`'s
Phase 3): genuine self-play using real MCTS search (`../mcts/mcts.py`)
with Ray-zeroGo's own current-best checkpoint, not KataGo's. The
recorded policy target is the search's visit-count distribution, not a
raw net output. Much more expensive per move than `generate.py` (a full
search per move instead of one net evaluation) -- expect real seconds
per move, not milliseconds. Produces the same
`bootstrap.dataset.SelfPlayExamples` format `generate.py` does, so
`bootstrap/train.py` trains on either's output unmodified. The policy/value
update loop itself is just `bootstrap/train.py` again, pointed at
`self_play.py`'s output — no separate training code needed.

## `play_one_game`'s algorithm, move by move

For each move: `mcts.search(position)` runs a full search, the resulting
per-move visit counts become that position's recorded policy target
(`visit_count_policy` — a dense, normalized vector, no separate legal-move
masking needed since `Mcts._expand` already creates a child for every
legal move), then a move is actually *sampled* from that same distribution
via `generate.py`'s `sample_move` — not necessarily the highest-visit move,
this is exploration data generation, not "play the best move" like
`eval/match.py`. The game continues until two passes in a row or
`max_moves`, then every recorded position gets its `z` (win/loss/tie, from
that position's own player-to-move perspective), normalized score margin
(from the final `area_score` + `komi`), and — since 2026-09-11, training
`bootstrap/model.py`'s ownership head — an `ownership_target` filled in.
Both `play_one_game`s (here and `generate.py`'s) compute
`engine.scoring.territory_ownership` **once**, against the game's actual
final position, right alongside `area_score`; each recorded position then
gets that same final read reprojected through `ownership_plane` from *its
own* player-to-move's perspective (`+1` mine / `-1` opponent's / `0`
neutral) — the ownership target is the same underlying ground truth for
every position in a game, only the sign convention rotates with whose turn
it was.

Four config knobs shape *how* that exploration happens, each fixing a
distinct self-play collapse failure mode diagnosed in
`../docs/SELF_PLAY_STABILITY.md`:

- **`temperature`/`temperature_drop_move`** — `sample_move`'s temperature
  controls how sharply move sampling favors high-visit moves; annealing to
  greedy (`temperature=0`) after `temperature_drop_move` (AlphaZero's own
  schedule: explore the opening, play well afterward) keeps randomness out
  of already-decided late-game positions, where it would only degrade
  training-data quality without adding useful diversity.
- **`root_dirichlet_epsilon`/`root_dirichlet_alpha`** — see
  `../mcts/README.md`'s "Root Dirichlet noise" section; this is where
  self-play actually turns that `Mcts`/`MctsConfig` knob on.
- **`no_pass_before_move`** — masks `Pass` out of both the recorded policy
  target and the moves `sample_move` can actually choose, below that move
  count. A hard structural guarantee (every observed collapse happened at
  or below move 20; legitimate games never finish before move 42) layered
  on top of the noise fix above, not a replacement for it.
- **`min_moves_to_keep`/`max_mid_game_pass_weight`** — post-hoc filters,
  applied after a game finishes: discard it outright if it's suspiciously
  short, or if any non-final position assigned Pass more visit weight than
  the threshold (`has_suspicious_mid_game_pass`) — the same underlying
  collapse pattern without it necessarily ending the game outright. Catches
  what the noise/masking fixes above miss, rather than preventing it.

Training on an uncaught collapsed game teaches the exact degenerate
behavior that produced it, compounding worse each generation chained on
top — which is why these are layered (noise addresses the diagnosed root
cause; masking and filtering are structural backstops) rather than any
one being considered sufficient alone.
