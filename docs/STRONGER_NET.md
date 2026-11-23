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
6-8x today's CPU speed with gen11. Cleanup-labelled ownership (step 4) is in as
`cleanup_labels` (fastplay/README.md); not used for training yet. Not supported:
`score_utility_weight`.

## First 10x128 net: distilled (2026-11-22)

`run_big10x128_distill.sh`: 2000 new gen11 games by fastplay (1 h 44 min), then a fresh
10 blocks x 128 channels net trained on them and on the older self-play
(`configs/bootstrap_train_big10x128_distill*.yaml`; fastplay games weighted 0.36, gen11's CPU
games 0.28, gen9-gen11's 0.18, older 0.18) for 60 min at lr 1e-3 and 20 min at 2e-4 on the
RTX 3050.

Held-out losses on fastplay's 512 benchmark games (`bootstrap/holdout_loss.py`, in no
checkpoint's training data):

| Checkpoint | Policy | Value |
|---|---|---|
| gen11 | 2.593 | 0.429 |
| gen12 (trained on gen11's games) | 2.565 | 0.378 |
| 10x128 phase 1, step 16000 | 2.563 | 0.373 |
| 10x128 phase 1 end (step 35699) | 2.546 | 0.407 |
| 10x128 phase 2 end | 2.514 | 0.400 |

The policy kept improving; the value overfit in phase 1's second half (0.373 -> 0.407) on ~7000
games. Next time: a shorter phase 1, or a held-out check that keeps the best checkpoint.

Parent match (`eval.promote`, 80 games at 50 sims, equal sims for both): **1560 vs 1440 for
gen11, +121 Elo, PROMOTE.** At equal simulations a 10x128 move costs ~2.4x gen11's on a phone.

Field gate (`eval/gate_big1_*`, 40 games per pairing at 50 sims, alias `big1`): **PASS, top of
the field at 2284** (gen11 2240, gen10 2219, gen12 2174). It's the first checkpoint to rate above
gen11 there; every gen12 attempt fell below it. It beat every checkpoint: gen11 28-12, gen12
24-16, gen9-gen10 31-9 and 33-7, older ones 30-36 of 40 (gen1 10 wins, the most of any old one).
Against the field the gap is +44, against gen11 head-to-head +121.

So more capacity alone, distilled from gen11-level games, already passes where 6x96 stopped.
Open: the phone check (`igo-app`'s `ParallelSearchOnDeviceTest` with this net: at 800 sims on 4
threads the untrained 10x128 took ~1.5 s on the emulator); whether float16 export keeps it
(12 MB float32 vs gen11's 4.2 MB); then the self-play chain from big1 with fastplay
(`cleanup_labels` on), the field gate deciding each step.

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

## big1 at 800 sims: Level 12, big2, and the first-move flaw (2026-11-22/23)

- **Float16** (`export/to_tflite.py --float16`): 12.2 → 6.1 MB, best move unchanged in 99.95% of
  2000 held-out positions.
- **big2** (first fastplay self-play step from big1, `runs/big2.log`): field gate **FAIL**,
  2241.8 vs big1's 2243.7.
- **As igo-app's Level 12** (big1 at 800 sims vs gen11 at 800, 120 games, first 16 moves sampled,
  no pass before move 20): big1 won 75 by area (+88.7) and 73 by territory (+76.5), ±33. The
  +121 head-to-head at 50 sims shrinks with deeper search. Emulator: 1.59 s a move at 800 sims on 4
  threads.
- **Weak first move.** In the app big1 opened on the second line. Raw policy on the empty board:
  big1 tengen 0.545 with 0.125 on the first line (gen11: 0.914 / 0.009). Its training data is the
  cause: gen11's self-play first-move targets (gen12 CPU data and `fastplay_gen11_2000`, ~64% of
  the distill mix) are ~51% tengen, the rest spread evenly over all 81 points; gen10's (gen11's
  training data) were 96% tengen. Same code and config, different net. **Root cause: first-play
  urgency.** `mcts.py` (and `fastgo.c`) counted an unvisited child as 0.0, an even game. gen11
  thinks Black is slightly behind at komi 7.5 (White +0.12 after Black's tengen; gen10 said
  −0.36), so every untried move looks better than tengen and a 200-sim search visits all 82
  moves; those single visits become the recorded target. big1 is gloomier still (+0.30): 200
  sims give pass 102, tengen 17. Beyond move 1, ~28% of all self-play positions have every legal
  move visited, in every generation's data alike, so the side that's behind gets blurred policy
  targets throughout.
- **Fix (2026-11-23):** `MctsConfig.fpu_reduction` / self-play key `fpu_reduction` (fastplay
  too): an unvisited child is worth the parent's mean value minus `fpu_reduction * sqrt(prior
  mass already visited)` (Leela Zero / KataGo). `None` (default) keeps the old 0.0, so the gate
  and every earlier run are unchanged. Empty board, 200 sims, 0.2: gen11 199 visits on tengen
  (before: all 82 moves visited, tengen 77, pass 42); big1 8 moves, tengen 97, then (3,5) 53,
  (3,3) 12, no edge or pass (before: pass 102, tengen 17). Tests: losing-side widening with and
  without it, and fastgo's visits equal mcts.py's with it on.
- **big2f (2026-11-23): big2 again with `fpu_reduction: 0.2` in self-play, nothing else
  changed — field gate PASS at 2363.6, top of the field** (big1 2254.8, gen11 2209.0); parent
  match vs big1 PROMOTE, 1547.8 vs 1452.2 over 80 games (big2 failed both). Self-play data: every
  legal move visited in 6.2% of positions (big2 27.8%), mean top target 0.43 (0.33), Black won
  46.4% of the games (20.9%). Held-out (gen11 benchmark, old-search targets): policy 2.528 (big1
  2.514; the benchmark's targets carry the old spread), value 0.393 (big1 0.400). First move:
  first line 15% → 3% of the target, tengen 17% → 9%, **second line 50%** — big1's real searched
  belief (Black's value after a 400-sim search: tengen −0.45, star −0.21, second line (1,4)
  −0.21, 3-4 −0.25, first line −0.69; gen11 too rates tengen below star and 3-4 once it searches).
  **Publisher's decision: accept it for training** — no opening steering; let later generations
  find their own openings (an app-side first-move choice stays possible for a shipped level).
- **big3 (2026-11-23): big2f's self-play (fpu 0.2), fine-tune mix big3 0.50 / big2f 0.25 /
  gen11 fastplay 0.10 / older 0.15 — field gate PASS at 2380.4, top** (refit with big3 in the
  field: big2f 2329.6, big1 2223.5, gen11 2175.6); parent match vs big2f PROMOTE, 1587.3 vs
  1412.7 (+175) over 80 games. Gate pairings: 27-13 vs big2f, 33-7 vs big1, 37-3 vs gen11 and
  gen10. Self-play: Black won 43.1%; first moves (1,4) 36%, star (2,2) 18%, (1,3) 16%, 3-4 15% —
  the second-line opening keeps growing. Held-out (gen11 benchmark): policy 2.549, value 0.400
  (big2f 2.528 / 0.393) — slightly worse there while much stronger in play.

## Also worth taking from KataGo once this runs

- Playout cap randomization (most moves with a small search, a few with a full one recorded as
  policy targets): more games for the same compute.
- Policy target pruning, forced playouts.
