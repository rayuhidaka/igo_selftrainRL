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

### 7. Score-margin auxiliary head implemented (2026-09-03)

Added `bootstrap/model.py`'s `has_score_head` (default `False`, every
prior checkpoint unaffected): an auxiliary regression head predicting
final score margin (from the mover's own perspective, normalized by
`board_size**2`), trained alongside the existing win/loss value head
with a low weight (`score_loss_weight`, default 0.15) so it regularizes
rather than dominates — same spirit as KataGo's own auxiliary heads.
Training-only: `export/to_tflite.py` wraps the model to strip this
output before ONNX export, so `MODEL_CONTRACT.md`'s policy+value
contract is unaffected regardless of whether a checkpoint has the head.

`selfplay/generate.py`'s new `score_margin()` (reused by
`selfplay/self_play.py`) computes the target from each game's final
`AreaScore` — reused rather than duplicated, unlike the `z` (win/loss)
computation which the two generators do independently. `SelfPlayExamples`
gained a `score_margin_targets` array; loading an older `.npz` without
one fills zeros (a neutral placeholder) rather than forcing every
existing self-play batch to be regenerated.

Adding the head to an already-trained checkpoint goes through
`bootstrap/train.py`'s existing `init_from_checkpoint` warm-start path
with `has_score_head: true` in the new run's config — the new head's
weights are simply absent from the source `state_dict` and load fresh
(`strict=False`), no separate migration step needed.

Not yet done: actually re-running generation 1's fine-tune with this
head enabled to see whether it reduces how much a single fine-tune
drifts (the original motivation, see section 6) — that's the next step,
before deciding whether the warm-start chaining restructuring (section
5) is still needed on top of it.

**Gotcha hit immediately trying this (2026-09-03):** both `batch2.npz`
and `self_play_gen1.npz` predate `score_margin_targets` entirely, so
`SelfPlayExamples.load()`'s backward-compat zero-fill silently kicked
in — the first attempt at this experiment trained the score head to
predict "always 0" (loss went to exactly 0.0000 almost immediately),
which is real but completely uninformative. Any dataset used to
actually test the score head's effect needs regenerating with the
current `selfplay/generate.py`/`self_play.py` first — check
`"score_margin_targets" in np.load(path)` before trusting a run that
uses an old `.npz` for anything score-head-related.

### 8. The score head does not fix the chaining collapse (2026-09-03)

Regenerated `self_play_gen1.npz` with real score margins, re-ran
generation 1's fine-tune with `has_score_head: true` (same 25/75 recipe
as v4): Pass probability 1.46% (v4 without the head: 1.22%, roughly
comparable), real 40-0 win against the pristine baseline — a valid,
working checkpoint. Then ran the actual test this was for: generated
generation 2's self-play with *that* checkpoint, fine-tuned generation
2 with the score head enabled throughout (identical chained-warm-start
recipe that caused the original 1.22%→34.5% collapse) —

**Result: Pass probability 1.46%→27.04%. Still a collapse, almost the
same magnitude as without the score head.**

This is a genuinely useful negative result: it rules out value-target
design (win/loss/tie flatness) as the primary cause of the chaining
collapse. If that were the main driver, richer score-margin supervision
should have visibly dampened the drift — it didn't, within noise of the
no-score-head result. The score head is still worth keeping (sound
practice, real non-zero training signal, no observed downside), but
it's not a fix for *this* failure mode. The cause remains what section
5 identified: chaining warm-starts on self-generated data compounds
whatever bias exists multiplicatively, regardless of what auxiliary
signal accompanies training.

**This makes the warm-start-chaining restructuring (section 5) the
real next step**, not an optional extra on top of the score head.

### 9. The restructuring: partial improvement, new problem (2026-09-03)

Implemented section 5's proposal — turned out to need no new code at
all, `bootstrap/dataset.py`'s existing weighted multi-source mixing
already supports it. `configs/bootstrap_train_gen2_restructured.yaml`:
`init_from_checkpoint` set to the *pristine* `bootstrap_batch2_residual.pt`
(not the previous generation's candidate), `data_source` mixing three
sources — 25% `batch2.npz` (unchanged anchor), 37.5%
`self_play_gen1.npz`, 37.5% `self_play_gen2_score_head.npz` (all
self-play generated by the current-best checkpoint at the time, per the
proposal — no self-play needed regenerating, only the fine-tune's
starting weights changed).

Result: empty-board Pass probability **16.09%** — roughly half the
severity of the chained collapses (27-34%), but still well above the
~1-1.5% single-generation baseline, and Pass is still the single most-
likely move. Evaluated against the pristine baseline: **1491.8 vs
1508.2, "Do not promote"** — a statistical wash, not the decisive 40-0
win every single-generation fine-tune produced.

So the restructuring traded one problem for another: less collapse,
but no longer a clear strength improvement either. Working hypothesis
for both halves of this:
- **Residual instability**: self-play *games* still come from an
  already slightly-biased checkpoint (`bootstrap_gen1_candidate_score_head.pt`,
  1.46% Pass), so `self_play_gen2_score_head.npz` itself may carry some
  of that bias baked into non-degenerate (kept) games too, not just the
  filtered-out short ones — mixing that data in, even without chaining
  weights, could still be a secondary transmission path for the drift.
  `min_moves_to_keep` only filters by game length, not by whether a
  kept game's *non-terminal* positions have suspiciously elevated Pass
  visit-counts.
- **Lost strength**: splitting the self-play weight three ways (anchor
  + 2 generations, 25/37.5/37.5) at the same total step budget as the
  single-source recipes may have diluted the gradient signal below the
  threshold needed for a measurable win — echoes the earlier 50/50
  single-generation result (section 4, v2) that was also a wash for
  similar reasons (signal diluted, not necessarily wrong in kind).

Neither the chained-warm-start recipe nor the from-pristine
restructuring has yet achieved *both* stability and a real second-
generation improvement simultaneously. Only the single first-generation
fine-tune (from pristine, either with or without the score head) has
cleanly achieved both so far.

### 10. Hardened filtering reveals the bias is pervasive, not rare (2026-09-03)

Chose the "harden self-play data quality further" option from the list
above. New `selfplay/self_play.py`: `max_mid_game_pass_weight` +
`has_suspicious_mid_game_pass` — discards a game if any recorded
position *outside the final two* (which legitimately precede the game's
own closing double-pass) assigned Pass more than the threshold (0.5) of
search's visit weight. `min_moves_to_keep` alone only catches a game
short enough to end outright from this dynamic; a longer game can carry
the same bias in one moment without it ending the game.

Regenerated generation 2's self-play with `bootstrap_gen1_candidate_score_head.pt`
(same checkpoint, seed, and settings as the earlier `self_play_gen2_score_head.npz`
run) with both filters active:

- 41/100 discarded as too short (consistent with before)
- **48/100 more discarded for elevated mid-game Pass weight** — a
  category the length-only filter completely missed
- **Only 11/100 games (779 examples) survived both filters**

So **89% of this checkpoint's self-play games show detectable Pass bias
somewhere**, not just the ~40-41% that collapse outright. The bias is
pervasive throughout most games, just not always severe enough to end
the game itself. This is a stronger, more useful diagnostic result than
expected, but it leaves a genuinely awkward practical problem: 779
examples is too little to confidently fine-tune generation 2 on, even
with weighted oversampling — training heavily on such a thin slice
risks a *different* overfitting failure (memorizing 779 specific
positions) rather than answering whether the filtering approach itself
works.

**Stopped here for the day, decision pending:** either (a) generate a
much larger raw self-play batch from this checkpoint so the same
~11% clean-survival rate still leaves a workable amount of data, or (b)
accept that filtering the *symptom* out of an already-biased
checkpoint's self-play has diminishing returns, and address the bias
in the generating checkpoint more directly (e.g. per-position value
calibration, not just output filtering). Not decided; pick up here.

## Open items as of this writing (end of 2026-09-03 session)

- Score-margin auxiliary head: implemented (section 7) and evaluated
  (section 8) — real, working, but does not address the chaining
  collapse by itself. Keep it (no downside, good practice).
- Warm-start chaining restructuring: implemented and evaluated (section
  9) — halves the collapse severity (16% vs. 27-34% Pass probability)
  but loses the clear strength win in the process (statistical wash,
  not a 40-0 promote). Not a complete fix by itself.
- Hardened self-play filtering: implemented (section 10) — works as
  designed, but reveals only ~11% of this checkpoint's self-play
  survives both filters, too little to fine-tune on confidently at that
  scale. **This is where the session stopped — pick up here tomorrow.**
- **Immediate next step, first thing tomorrow:** decide between (a)
  generating a much larger raw self-play batch so ~11% clean survival
  still yields enough data (cheap to try: just re-run
  `configs/selfplay_self_play_gen2_hardened.yaml`-style config with a
  bigger `num_games`, e.g. 500-1000 instead of 100), or (b) treating
  pervasive mid-game Pass bias as a signal that filtering
  self-play *output* has hit diminishing returns and the checkpoint's
  own value calibration needs more direct attention. Leaning toward
  trying (a) first since it's cheap and directly tests whether more
  raw self-play volume resolves the practical data-scarcity problem
  without needing a new approach.
- Other live options not yet tried, still on the table if (a)/(b) above
  don't pan out: retune the restructured mix (more steps, or weight
  recent generations higher than older ones), or step back to a true
  continuous replay-buffer training loop (bigger infrastructure change,
  discussed and deliberately deferred — see the "step 3" discussion
  earlier in this file's git history / the conversation this session).
  Accepting the single first-generation result as Phase 3's current
  deliverable and pausing multi-generation chaining remains a valid
  fallback if none of the above pan out.
- Known-good checkpoints: `bootstrap_gen1_candidate.pt` (v4, no score
  head) and `bootstrap_gen1_candidate_score_head.pt` (with it) — both
  25/75 mix from the pristine baseline, both promoted, both stable in
  isolation. These are the only checkpoints past `bootstrap_batch2_residual.pt`
  worth building on right now.
- Known-bad/known-mediocre checkpoints — don't build on: the two
  chained-warm-start attempts, `bootstrap_gen2_candidate.pt` (34.5%
  Pass) and `bootstrap_gen2_candidate_score_head.pt` (27.04% Pass); and
  the restructured attempt, `bootstrap_gen2_candidate_restructured.pt`
  (16.09% Pass, not promoted — better than the other two but not a
  clean result either).
- Data files: `selfplay_games/self_play_gen2_hardened.npz` (779
  examples, both filters applied) exists but is too small to have been
  used for anything yet — no fine-tune has been run on it.
