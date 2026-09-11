# eval/

Elo tracking across checkpoints/generations, used to decide which ones
become the app's shippable difficulty tiers — see
`../docs/ARCHITECTURE.md`'s "Difficulty-tier promotion" section for the
design and `../docs/ROADMAP.md`'s Phase 3. Fully implemented end to end:

- `elo.py` — Elo rating math (`update_ratings`, `should_promote`). No
  dependency on anything below.
- `match.py` — plays games between two checkpoints to produce `elo.py`'s
  inputs, using `../mcts/` (search) and `../engine/` (rules) — both Python
  ports of `igo-app`'s Kotlin implementations.
- `promote.py` — orchestrates the two: play a candidate against the
  current tier, update ratings, decide whether to promote.

Tested at two levels: unit tests (`../tests/test_elo.py`,
`../tests/test_match.py`) and a real CLI run
(`../configs/eval_smoke_test.yaml`, see its header for the exact
commands) — `python -m eval.promote` actually playing games and printing
a real Elo verdict, not just a shape check.

What's still ahead isn't in this folder: real training (Phase 2) to
produce checkpoints actually worth running through this.

## The promotion algorithm

`elo.py` is plain, textbook Elo — `expected_score` is the standard
logistic win-probability formula, `update_ratings` applies the usual
`rating += k * (actual - expected)` delta per game, both players starting
from `DEFAULT_INITIAL_RATING` (1500) fresh for **every match** (not a
persistent rating carried across the whole checkpoint history — a match is
always exactly one candidate vs. the one current tier, so a fresh
baseline each time is intentional, not a shortcut). `should_promote` is
one comparison: `candidate_rating >= current_tier_rating + min_elo_gap` —
a candidate has to be *meaningfully* stronger, not just newer, or the
app's difficulty picker would fill up with near-identical tiers.

`match.py`'s `play_match` is what actually generates the `MatchResult`s
`elo.py` scores: `num_games` real games between two `RayZeroPolicyValueNet`
checkpoints, searched with this repo's own `mcts/mcts.py`/`engine/`
(not `igo-app`'s Kotlin versions — see those folders' READMEs for why
using ports at all matters here). Games use a **nonzero temperature**
by default (`promote.py` passes `temperature=1.0`, annealed to greedy
after `temperature_drop_move`, default move 16) rather than `match.py`'s
own more conservative `0.0` default — a real promotion decision needs
statistical power across `num_games` *different* games, not the same
deterministic game replayed `num_games` times over.
