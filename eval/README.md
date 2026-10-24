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
- `round_robin.py` — every checkpoint vs. every other, fit to one shared
  Elo scale (Bradley-Terry).
- `gate.py` — the second half of a promotion decision: after `promote.py`
  confirms a candidate beats its parent, passes it only if it rates at least
  its parent on a round-robin that includes it. Catches a candidate that
  beats its parent by getting weaker against everything older (see its
  module docstring for the gen12 case that motivated it).

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

## Area vs territory scoring (`rule_comparison.py`)

`igo-app` is moving to Japanese (territory) scoring, while every net here was trained with
area scoring. `rule_comparison.py` plays the shipped levels against each other
(`configs/eval_rule_comparison.yaml`), keeps each final board and the prisoners
(`match.py`'s `play_game_to_end`), and counts every game both ways — `engine/scoring.py`'s
`area_score` and `territory_score`. A **flip** is a game whose winner differs. Per-game counts
go to `eval/rule_comparison_games.json`; `--summarize-only` re-reads them for other komi
values.

Useful identity when reading the results: Black's area margin minus its territory margin
equals Black's stone placements minus White's. Under area scoring a placement inside your own
territory is free, so an area-trained net that keeps playing there while its opponent passes
gives away exactly that many points under territory scoring.

**First result (2026-10-23):** 27 of 278 finished games flip (9.7 % at territory komi 6.5),
25 of them decided by 2.5 points or less — the nets play for the smallest winning margin. A
pass rule ("pass after the opponent's pass once nothing is left open") was tried and dropped:
it needs a list of dead stones, and the nets can't supply one when a side is far ahead (it
doesn't bother capturing). See docs/ROADMAP.md's "Margin-aware training".

**Score bonus in the search, no retraining (2026-10-24,** `configs/eval_score_utility.yaml`,
games in `eval/score_utility_games.json`**):** gen11 at `score_utility_weight` 0.1 / 0.25 / 0.5,
50 simulations, 936 finished games.

| | vs gen11 (area win, ~155 games) | vs Beginner: margin area / territory | wins by ≤ 2.5 vs Beginner | self-play flips |
|---|---|---|---|---|
| gen11 | — | 9.1 / 19.0 | 17.5 % | 2 / 76 |
| weight 0.1 | 48.7 % | 8.7 / 18.5 | 15.8 % | 6 / 77 |
| weight 0.25 | 51.0 % | 9.6 / 21.6 | 18.4 % | 3 / 78 |
| weight 0.5 | 52.6 % | 10.9 / 22.0 | 7.9 % | 6 / 80 |

No strength cost at any weight (±8 % at 95 %). Margins against Beginner grow with the weight,
but only 40 games each, so suggestive rather than proven. Flips and close games between equal
players don't improve. The bonus can only reorder moves the policy already proposes, and a
policy trained only to win rarely proposes margin moves — so a search-only change isn't enough
to ship; the policy has to learn it (margin-aware self-play, stage 3).

**Margin-aware generation, stage 3 (2026-10-24,** `run_option3_gen12_margin.sh`, margin check
`configs/eval_score_utility_gen12m.yaml`, games in `eval/score_utility_gen12m_games.json`**):**
gen12m is gen11 fine-tuned on 500 self-play games played with the score bonus at 0.5; in the
margin check it searches win/loss only, as igo-app does.

| | gen11 | gen12m |
|---|---|---|
| field gate (round-robin Elo) | 2231.6 | 2184.2 — FAIL |
| head-to-head vs gen11 (157 games, area / territory) | — | 53.5 % / 51.6 % (±7.8 %) |
| self-play flips | 8 / 118 (6.8 %) | 6 / 116 (5.2 %) |
| self-play games within 2.5 points | 25.4 % | 26.7 % |
| vs Beginner: win rate, territory margin | 98.8 %, 19.1 | 92.5 %, 17.5 |

No margin play learned: flips and close games are within noise of gen11's, margins against
Beginner went down, and the candidate is weaker against the field. Its opening policy spread
out (tengen 0.91 → 0.52). One fine-tune at this recipe's 60-second cap, with win/loss value
targets, doesn't carry the bonus's preference into the net.
