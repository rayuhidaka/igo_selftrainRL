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
