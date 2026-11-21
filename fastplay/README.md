# fastplay: batched self-play

Stage 1 of `docs/STRONGER_NET.md`: the same self-play as `selfplay/self_play.py`, many games at
once, so a bigger net can be trained in days instead of weeks.

- `fastgo.c` — the rules (`engine/position.py`, `engine/scoring.py`) and the PUCT search
  (`mcts/mcts.py`) in C. One search tree per game; each game's search runs one simulation at a
  time exactly like `Mcts`, and the batch is across games, so no virtual loss is needed.
- `fastgo.py` — the ctypes wrapper. It compiles `fastgo.c` with the system C compiler into
  `fastplay/build/` on first import (no pip packages needed).
- `self_play.py` — the driver: `python -m fastplay.self_play --config <selfplay config>`. Takes
  `selfplay/self_play.py`'s config keys plus `parallel_games` (default 128) and `device`, and
  writes the same `.npz` fields, so training, export and gating don't change.

`tests/test_fastgo.py` checks the C side decides what the Python side decides: legal moves,
captures, ko and the board after every move of random games on 3x3, 5x5 and 9x9; area scores
and ownership; the net's input planes; and the search's visit counts against `Mcts` with the
same (fake) net, with and without root noise. Change the rules or the search in both places.

`cleanup_labels: true` (step 4 of `docs/STRONGER_NET.md`) plays a game that ends on an unsettled
board on with the cleanup rule (`fg_cleanup_mask`, tested equal to
`mcts/dead_stone_estimator.py`'s `is_cleanup_move`) and takes the ownership targets from where
that ends; the result and the value and score targets stay those of the real end. Smoke test
(gen11, 100 sims, 32 games): 30 ended unsettled, 15.6 points a game changed owner; by eye the
stones removed were dead (in atari inside the other side's area, eyeless groups). Games that
ended by two early passes on an open board can reach the cap (a third of the points in moves)
still unsettled.
