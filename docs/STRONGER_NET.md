# Levels above 10: a bigger net, and the self-play to train it

Plan agreed with the publisher 2026-11-21, after players started beating Level 10 (gen11).

## Decisions (2026-11-21)

- Order: igo-app's variety and dead-stone fixes first, then levels above 10 by more search
  (2-3 s a move on a phone is acceptable), then fast self-play, then the bigger net.
- Spending a week or two on GPU self-play before training a bigger net: agreed.
- Self-play temperature schedule experiment (below): agreed, run once self-play is fast.
- gen11's ownership head is not used in the app (below); revisit with a net trained on
  cleanup-labelled endings.

## Why the current recipe stopped

gen11 (6 blocks, 96 channels) is the ceiling of the current chain. Every gen12 attempt — 1000
games, longer training, a wider data window, 800-simulation self-play, margin-aware training —
beat gen11 head-to-head and failed the round-robin field gate (`eval/gate.py`). One self-play
generation of 500 games already takes ~8 h on 4 CPU workers; a bigger net makes each
evaluation 2-3x slower, so continuing on the CPU would mean days per generation.

## Order of work

1. **More search first (no training).** Level 11/12 = gen11 at more simulations, if a move stays
   within 2-3 s on a phone (`igo-app`'s `SearchSpeedOnDeviceTest`). Rated on the round-robin
   like every level.
2. **Fast self-play (this document's main topic).**
3. **Bigger net, warm-started from the existing games** (gen5-gen12 self-play, ~450k
   positions, plus the 800-sim games): distill until it matches gen11 on the field gate, then
   run the usual chain with the gate. Copying gen11's games alone only reproduces gen11; the gain
   has to come from continued self-play with the extra capacity. Candidate size: 10 blocks /
   128 channels (~3x gen11's compute per evaluation); check on-device speed before committing.
4. **Correct ownership labels in self-play.** Self-play games end with dead stones still on the
   board — far ahead or not (80 close gen12 endings held 407 dead stones by a deep reference,
   their labels only 33) — so the ownership targets call them alive. Use igo-app's cleanup rule
   (2026-11-21, `mcts/dead_stone_estimator.py`'s `is_cleanup_move`): after both players pass
   on an unsettled board, play on, passing only when no move in a contested region is left
   (never in your own area, never inside the other side's territory, never a self-atari), and
   label from that end. Only the labels change; the game result stays the same. Don't use the
   first version of the rule, which also allowed plays in the other side's territory: both
   sides then filled each other's eyes and living groups died.

## gen11's ownership head (checked 2026-11-21)

Measured on gen12's 800-sim self-play (never in gen11's training data), truth = the final
ownership target: near the end of a game the head finds ~60% of the stones that are later
captured and wrongly calls ~1.5 live stones per finished board dead (threshold 0). Retraining
only an ownership head on gen11's frozen trunk (a 1x1 conv, or 3x3 conv to 32 + 1x1) for 10
epochs on gen5-gen12 cut false alarms on finished boards from 750 to 216 per 500 games but left
recall at ~60%. Part of the cause is the labels: self-play endings between a far-ahead side and
a lost one leave dead stones on the board, which the final position then counts as alive. That
is what the cleanup labelling in step 4 above fixes.

igo-app instead fixed its dead-stone proposal with cleanup playouts (the rule in step 4, applied
to the playouts). Against a deep reference: 80 close self-play endings, 396 of 407 dead stones
found vs 45 before, 2 false alarms either way; 14 lopsided gen11-vs-gen5 endings, 74 of 97 vs 2.

## Where self-play time goes (profile, 2026-11-21)

One 200-simulation search with gen11 on one CPU thread takes ~2 s (~10 ms per simulation):

| Part | Share |
|---|---|
| Net (conv layers, batch of 1) | ~40% |
| Rules engine (legal moves, suicide checks, group/liberty flood fills) | ~45% |
| Plane encoding, hashing, tree bookkeeping | ~15% |

So moving the net to the GPU alone gives at most ~2x. Both halves have to go.

## Design

- **Batched evaluation.** Each worker runs many games at once (e.g. 64-128). Every simulation
  step walks one leaf per game, evaluates all leaves as one batch on the RTX 3050, then backs up.
  No virtual loss needed at first: one leaf per game per step already fills a batch.
- **Fast rules engine** (built in C instead, see "Fast self-play: built and measured"). A numpy/numba board with incrementally kept chains and liberties
  (`play` updates only the neighbors' chains; legality and captures become lookups). Keeps
  `engine/`'s behaviour exactly: checked against it on random games move by move (legal move
  sets, captures, ko, area score).
- **Plane encoding on the GPU** from the board arrays, no per-point Python loop.
- **Same outputs as today:** the same `.npz` fields (`board_planes`, `policy_targets`,
  `value_targets`, `score_margin_targets`, `ownership_targets`), so training, export and gating
  don't change. The existing self-play stays as the reference; a short run of both on the same
  checkpoint must give the same statistics (game length, pass rate, opening spread, win rate).
- **Measured, not assumed:** games per hour against today's 4-worker CPU run, at 200 sims.
  Target 10x or more.

## Fast self-play: built and measured (2026-11-22)

`fastplay/` (see its README). The rules and the PUCT search are in C (`fastgo.c`, compiled by
the system compiler via ctypes; numba couldn't be installed, since WSL has no network here),
with one tree per game and 128 games batched on the RTX 3050. `tests/test_fastgo.py` checks it
against `engine/` and `mcts/` move by move on random 3x3/5x5/9x9 games (board, legal moves, ko,
area score, ownership, planes). It also checks that the search's visit counts equal
`mcts.py`'s exactly, with and without root noise.

Benchmark (`configs/fastplay_benchmark_gen11.yaml`, gen12's recipe: gen11, 400 sims, same
noise, temperature and pass guard):

| | fastplay | `selfplay.run_parallel`, 4 CPU workers |
|---|---|---|
| 512 / 500 games | 27 min | ~8 h |
| Games an hour | ~1140 | ~62 |
| Net evaluations a second | ~12,800 | ~720 |

18x faster, with the same games (`fastplay/dataset_stats.py`, vs gen12's self-play from the same
checkpoint): 103.4 vs 103.9 positions a game, Black wins 26.8% vs 28.4%, tengen first 53% vs
54%, pass-favoring targets 2.36% vs 2.20%. The run is one Python process at 100% of a core
with the GPU at ~85%, so there is little left to gain without a faster net call (fp16, CUDA
graphs) or a second process. A 10x128 net (~2.4-3x gen11's compute) should still give roughly
6-8x today's CPU speed with gen11. Not yet done: cleanup-labelled ownership (step 4) and
`score_utility_weight`.

## Self-play diversity (checked 2026-11-21)

Question from the publisher: did gen11 learn little because its self-play games were too alike
(~96% tengen openings)? Measured on the saved games, positions compared up to symmetry:

| Data | Tengen first | Same position after 2 moves | Distinct after 6 | Distinct after 10 |
|---|---|---|---|---|
| gen5 | 38% | 12% | 99% | 100% |
| gen8 | 98% | 69% | 64% | 96% |
| gen9 | 98% | 34% | 72% | 98% |
| gen11 | 96% | 52% | 83% | 100% |
| gen12 | 54% | 49% | 88% | 100% |
| gen12 800 sims | 43% | 38% | 90% | 100% |

Self-play never played argmax in the opening (temperature 1 to move 30, root noise 0.25), and by
move 10 every game is unique; only the first 4-6 moves (~5% of positions) are narrow. The gen12
attempts had far more opening variety and still failed the gate. So "too little variety" isn't
supported. The opposite is worth testing once self-play is fast: temperature 1 for 30 moves is a
third of a 9x9 game (AlphaZero's 30 was out of 361 points), so early value targets come from
games decided partly by sampled weak moves. Experiment: temperature 1 -> ~0.3 over the first
10-15 moves vs today's schedule, judged by the field gate.

## Also worth taking from KataGo once this runs

- Playout cap randomization (most moves with a small search, a few with a full one recorded as
  policy targets): more games for the same compute.
- Policy target pruning, forced playouts.
