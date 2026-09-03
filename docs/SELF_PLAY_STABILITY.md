# Self-play stability: findings, failures, and learning

A running log of Phase 3's self-play fine-tuning investigation (see
`docs/ROADMAP.md`'s Phase 3 for the task-tracking summary — this document
is the detailed narrative behind it). Kept separate because the roadmap
entries compress this into a few bullets; the actual investigation had
several dead ends worth remembering in full, so the next person (or the
next me) doesn't re-derive them.

## Timeline

### 1. First full generation cycle worked (2026-09-03)

100 self-play games with `bootstrap_batch2_residual.pt` (real MCTS
search, not distillation) → fine-tuned a candidate from it (warm start,
unmixed, 10 epochs / 1,230 steps over the 7,879-example self-play batch
alone) → evaluated the candidate against its own parent, both sides
using equal-budget MCTS search → candidate won **40-0**. Looked like a
clean success.

### 2. Generating a second generation exposed a collapse

Self-playing 100 more games with that fine-tuned candidate (to produce
generation 2's training data) produced ~30 games ending in 2-16 moves —
both sides passing on a near-empty board, White "winning" purely from
komi. This pattern did not exist in generation 1's self-play (only
1/100 short games there, using the pre-fine-tune checkpoint).

Diagnosis: querying the checkpoint directly on an empty board showed
Pass probability had risen from ~0.01% (original) to ~1.7% (after one
fine-tune) — small in isolation, but enough that a 100-simulation MCTS
budget could "run away" with it within a single search, since the value
network had barely seen post-pass positions and could misjudge that
branch as good.

### 3. First fix: filter degenerate games — necessary, not sufficient

Added `selfplay/self_play.py`'s `min_moves_to_keep` config: discard any
game shorter than a threshold (20, chosen to cleanly separate the
collapsed cluster — max 16 moves — from every legitimate game observed
— min 42) before it reaches the saved training set.

This didn't fix the underlying problem. Re-generating generation 2's
self-play *with* the filter, then fine-tuning generation 2's candidate
on the cleaned data, produced empty-board Pass probability of **24.9%**
— worse than before, not better. Filtering removed the worst symptom
(training on literally-degenerate games) but not the cause: the network
was overgeneralizing "Pass is good" from *legitimate* near-endgame
positions (where search correctly assigns Pass high confidence) into
completely inappropriate contexts like an empty opening board.

### 4. Mixing in the broad dataset: four attempts, non-monotonic results

Hypothesis: anchor each self-play fine-tune against the original,
known-good imitation-learning dataset (`batch2.npz`, 224,730 examples)
so the network can't drift far on one small new self-play batch alone.
Built `bootstrap/dataset.py`'s `build_training_loader` (weighted
sampling across multiple sources, independent of their relative sizes)
and `bootstrap/train.py`'s `data_source` list-of-`{path, weight}`
support.

All four attempts fine-tuned **from the same pristine
`bootstrap_batch2_residual.pt`** (not chained), varying only the mix
ratio and step count, to isolate that variable:

| Attempt | Mix (anchor/self-play) | Steps | Self-play oversample | Empty-board Pass prob | Eval vs. parent |
|---|---|---|---|---|---|
| v1 | 0/100 (unmixed) | 1,230 | ~10x | 1.7% | 40-0 win |
| v2 | 50/50 | 1,785 | ~7.25x | 0.86% | 1508.2 vs 1491.8 — wash |
| v3 | 50/50 | 3,472 | ~14x | 1.1% | 40-0 **loss** |
| v4 | 25/75 | 1,658 | ~10x | 1.22% | 40-0 win |

v3 was the most informative result: *more* anchor exposure (same 50/50
ratio, just longer) made the candidate decisively *worse*, not better.
That ruled out "just add more anchor data" as a simple fix — heavy
re-exposure to a dataset the base checkpoint was already fully
converged on (495k steps in its own original training) appears to
introduce its own drift, independent of the self-play data. v4 (less
anchor "tax" per step, self-play oversample matched to v1's original
exposure level) was the first result that was both a real win *and*
stable.

### 5. v4's fix didn't survive being chained to a second generation

Regenerating generation 2's self-play using the *properly-fixed*
`bootstrap_gen1_candidate.pt` (v4) still produced 39/100 degenerate
games — similar magnitude to before, despite v4's much healthier
isolated Pass-probability metric (1.22%). Interpretation: self-play
*against itself* lets both sides' identical, small bias reinforce each
other; the same checkpoint played against a *different*, well-calibrated
opponent (the eval matches) showed no such issue. The `min_moves_to_keep`
filter earned its place here — it's doing real, necessary work, not
just a stopgap.

The bigger problem: fine-tuning generation 2's candidate from
`bootstrap_gen1_candidate.pt` (warm start, i.e. chaining), using the
*exact same 25/75 recipe that worked cleanly for generation 1*, produced
empty-board Pass probability of **34.5%** — worse than the original
collapse. Comparing the two hops (0.01%→1.22% for gen1, then
1.22%→34.5% for gen2 using the identical recipe) shows a similar large
*multiplicative* jump both times. The 25/75 ratio was never a stable
equilibrium — it only looked like a fix because gen1 started from an
exceptionally clean baseline. **Chaining warm-starts across generations
compounds whatever bias exists multiplicatively, regardless of the mix
ratio used for any single fine-tune.**

Proposed (not yet implemented) restructuring: stop chaining fine-tunes.
Keep generating each generation's self-play *games* with the current
best candidate (exploration should still come from the strongest
available play), but always fine-tune *from the original pristine
checkpoint*, mixing in all accumulated self-play data so far rather than
warm-starting from the previous fine-tune's weights. This was paused to
investigate a more fundamental question first (next section) — a better
value signal might substantially reduce how much bias accumulates per
fine-tune in the first place, which would change how urgent (or how the)
restructuring is needed.

### 6. A more fundamental question: is win/loss/tie itself part of the cause?

The user's framing: a losing player should keep fighting rather than
pass/give up, and a decisive win should be distinguishable from a lucky
one — neither is true under our current ±1/0/-1 value target, which
treats every win identically regardless of margin or how it was
achieved.

Research (see `docs/ROADMAP.md`'s Phase 3 for the summary, this is the
detail): AlphaZero deliberately uses binary win/loss/tie, not score
margin, specifically to avoid a *different*, well-documented pathology —
training on raw margin makes an already-winning agent take needless
risks purely to inflate the score, rather than playing safely to secure
the win. But pure win/loss has its own documented failure mode: once
value estimates saturate near ±1, the network becomes flat/indifferent
among every move that reaches that same categorical outcome — no signal
distinguishes an efficient win from an inefficient one, or (our exact
bug) a decisive win from a cheap, lucky one.

KataGo's actual fix: keep win/loss/tie as the *primary* value target,
but add **auxiliary** score-margin and per-point board-ownership
prediction heads trained alongside it — richer supervision that KataGo's
own ablation confirmed measurably speeds up learning, without
reintroducing the margin-chasing pathology (since move selection still
weights win probability as primary; score utility is blended in only as
a secondary, lightly-weighted term at search time).

**Decision: implement the cheapest version of this first** — a
score-margin auxiliary head on `RayZeroNet`, trained alongside the
existing win/loss value head, before returning to the warm-start-chaining
restructuring from section 5. A better-calibrated value signal might
reduce how much a single fine-tune drifts in the first place, which
would change how much (if any) of that restructuring is still needed.

## Open items as of this writing

- Score-margin auxiliary head: not yet implemented (in progress).
- Warm-start chaining restructuring (section 5): paused pending the
  above.
- `bootstrap_gen1_candidate.pt` currently holds the v4 checkpoint (25/75
  mix, promoted, stable in isolation). `bootstrap_gen2_candidate.pt`
  from the chained 25/75 attempt is **known-bad** (34.5% Pass
  probability) — don't build on it.
