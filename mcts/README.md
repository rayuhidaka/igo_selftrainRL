# mcts/

A Python port of `igo-app/mcts/`'s PUCT search driver (Kotlin), for
`eval/match.py` to actually play moves during checkpoint evaluation
matches with (see `../docs/ARCHITECTURE.md`'s "Difficulty-tier
promotion" section for why: without search, both checkpoints would play
uniformly-random moves and Elo differences between similar-strength
nets wouldn't be measurable). Same rationale as `../engine/`'s port —
avoid a second, independently-written search implementation silently
disagreeing with the first.

| Kotlin (`igo-app/mcts/`) | Python (here)          |
|---------------------------|--------------------------|
| `PolicyValueNet.kt`        | `policy_value_net.py`   |
| `Mcts.kt`                  | `mcts.py`                |

Proven against a full port of `igo-app/mcts/`'s own test suite — see
`../tests/test_mcts.py`.

`bootstrap/inference.py`'s `RayZeroPolicyValueNet` implements this
module's `PolicyValueNet` protocol against a `RayZeroNet` PyTorch
checkpoint, the same way `igo-app/inference/TfLitePolicyValueNet.kt`
implements the Kotlin `PolicyValueNet` interface against a `.tflite`
checkpoint.

**If you change the search, change it on both sides** — same caveat as
`../engine/README.md`.

## The search algorithm

`Mcts.search(root)` runs `MctsConfig.num_simulations` simulations, each a
select → expand → backup walk from a fresh per-call tree of `_Node`s
(prior, visit count, value sum, children). Selection picks the child
maximizing the PUCT score at each step:

```
score(child) = -child.mean_value + c_puct * child.prior * sqrt(parent_visits) / (1 + child.visit_count)
```

(negated exploitation term because a child's value is from *its own*
player-to-move's perspective — the opponent doing well is bad for this
node). The first leaf reached gets expanded: one `net.evaluate(position)`
call creates one child per legal move, seeded with the net's own prior,
and returns the value that gets backed up the path just walked, flipping
sign at every level. `select_move` returns whichever root child ended up
with the most visits — the thing AlphaZero-style search actually trusts,
not the net's raw top pick.

A position with `pass_count >= 2` is terminal and scored by
`engine.scoring.area_score` + `MctsConfig.komi` instead of the net —
ground truth from the rules once a game is actually over, matching
`igo-app/mcts/Mcts.kt` exactly.

## Root Dirichlet noise — the one deliberate divergence from the Kotlin port

`MctsConfig.root_dirichlet_epsilon`/`root_dirichlet_alpha` (both `0` by
default, i.e. off) mix Dirichlet-sampled noise into the *root* node's
priors only, on its one expansion — see `root_dirichlet_epsilon`'s
docstring for the real bug this fixes: with zero noise and a low
simulation budget, a checkpoint's own slightly-elevated Pass prior gets
over-visited by search with nothing to correct it, and training on the
resulting visit counts pushes the *next* checkpoint's prior even higher —
a multiplicative collapse across self-play generations
(`docs/SELF_PLAY_STABILITY.md`). `alpha=0.1` (not AlphaZero's 19x19 value
of `0.03`) is tuned for 9x9's much smaller branching factor via the
standard `alpha ≈ 10 / average legal moves` heuristic, and the noise draw
deliberately **excludes** `Pass` from the moves it can land on (see
`_add_root_noise`'s docstring for why including it made the collapse
*worse*, not better, at first).

This exists only for self-play data generation's exploration needs — left
at `0.0` for `eval/match.py`'s matches and for `igo-app`'s own gameplay
search, where the goal is the engine's actual best move. Deliberately
**not** mirrored to `igo-app/mcts/src/main/kotlin/.../Mcts.kt` for that
reason — if you're porting a search-algorithm change between the two
sides, this config knob is the one exception that should stay one-sided.
