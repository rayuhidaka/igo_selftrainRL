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

### 11. Root cause found: no root exploration noise + low simulation budget (2026-09-04)

Stepped back from filtering/value-calibration approaches to look directly
at the search mechanics that *produce* the visit-count policy targets, not
just the checkpoints those targets eventually train. `mcts/mcts.py` had
**no Dirichlet noise at the root and no other exploration-forcing
mechanism**, combined with a low `num_simulations` (100, over ~82 legal
9x9 moves). This is a known, previously-solved AlphaZero pathology —
without root noise, self-play search converges to whatever the current
policy prior already favors, faster than a small simulation budget can
correct it (see AlphaZero's own paper and KataGo's *Accelerating Self-Play
Learning in Go*, whose policy-target-pruning/forced-playouts machinery
exists to guard against exactly this).

This fits every piece of prior evidence better than either paused option:
- It's **multiplicative by construction**: a slightly-elevated Pass prior
  → noise-free/low-sim search over-visits Pass → the visit-count training
  target overshoots the prior further → the next checkpoint imitates that
  harder → repeat. Matches the observed 0.01% → 1.2-1.5% → 27-35% curve
  exactly.
- Explains why the score-margin head (sections 7-8) didn't help — that
  changes value calibration, not search exploration, a different
  subsystem entirely.
- Explains why warm-start restructuring (section 9) only halved the
  problem — that stops weight-chaining, but self-play games are still
  *generated* by the current (already-biased) checkpoint via the same
  noise-free search, so bias still leaks in through the data itself.
- Explains why 89/100 games showed some Pass bias (section 10) — that's
  the search's normal behavior at these settings, not bad luck.

**Fix implemented:** `MctsConfig` gained `root_dirichlet_epsilon`/
`root_dirichlet_alpha` (default `0.0`/`0.03`, i.e. no behavior change
unless opted into — `eval/match.py` and the Android app's live analysis
search are deliberately left at `0.0`, since exploration noise has no
place in an actual best-move search). `selfplay/self_play.py` reads both
from config. `configs/selfplay_self_play_gen1_noise_fix.yaml` re-runs the
original gen1 self-play batch with noise turned on (`epsilon=0.25`,
`alpha=0.03`, AlphaZero's own values) and `num_simulations` raised
100→400. Covered by two new `tests/test_mcts.py` cases. Also found and
fixed an unrelated perf bug while testing this: PyTorch's default
per-core thread pool made single-position MCTS inference ~85x slower
than needed (`torch.set_num_threads(1)` at the self-play/eval CLI entry
points — see code comments for the measured numbers).

### 12. The plain noise recipe made things worse: noise landing on Pass (2026-09-04)

Running `configs/selfplay_self_play_gen1_noise_fix.yaml` (noise over
*all* root actions, including Pass) against the pristine checkpoint
showed short-game rates of 40% (10 games), 35% (20 games), 40% (30
games) — consistently far above that same checkpoint's original 1/100
no-noise baseline (section 1). The fix made the exact problem it was
meant to solve *worse*. Root cause: `alpha=0.03` is a peaked Dirichlet
draw that usually dumps nearly all its mass onto one random action;
roughly 1-in-~82 times that's Pass, and when it is, search gets a real
incentive to explore the post-pass branch — a state the net's value head
is poorly calibrated on precisely because passing was rare in its own
training data. Full writeup: memory `self_play_root_noise_pass_bias`.

**Fix to the fix:** `Mcts._add_root_noise` now excludes `Pass` from the
noise draw entirely — noise is drawn and mixed only over board-play
moves; Pass keeps its own net-derived prior untouched. New
`tests/test_mcts.py::test_root_dirichlet_noise_never_touches_pass`
covers it.

**Confirmed:** a 20-game smoke test (same checkpoint/settings as the
noise-on-Pass run above — `epsilon=0.25`, `alpha=0.03`, 400 sims) came
back **20/20 clean, zero short games**, averaging ~88 moves/game —
matching the healthy 1/100 no-noise baseline's game-length character,
not the 35-40% collapse rate seen with noise-on-Pass. The Pass-exclusion
fix works. `memory self_play_root_noise_pass_bias` updated accordingly.

**Full 100-game batch (2026-09-04):** `configs/selfplay_self_play_gen1_noise_fix.yaml`,
same settings, no filtering (`min_moves_to_keep`/`max_mid_game_pass_weight`
left at their no-op defaults) — **10/100 short games (10%)**, avg length
80.5 moves, 8,045 examples written to
`selfplay_games/self_play_gen1_noise_fix.npz`. Better than the 20-game
smoke test suggested (0%) but not as clean as the original no-noise
baseline (1%).

**Inspected all 10 short games directly** (board_planes/policy_targets
from the saved `.npz`, matched back to per-game boundaries via the run
log): all 10 are genuine collapse artifacts, not legitimate quick
losses — every one won by White, near-empty final boards (0-10 stones),
Pass probability spiking to 30-80% within the first couple of recorded
moves. Notably, some show Pass at ~31% on the *very first* move, before
noise could touch anything but board-play priors. Refined diagnosis:
`alpha=0.03` (AlphaZero's own 19x19 value, ~250+ opening legal moves) is
far more peaked than appropriate for 9x9's ~82 actions. The standard
tuning heuristic is `alpha ≈ 10 / average legal moves` (it's why chess
uses 0.3, shogi 0.15, 19x19 Go 0.03) — for 9x9 that's `alpha ≈ 0.1-0.12`,
not 0.03. Too-peaked noise dumps nearly all its mass on one random
board move and leaves the other ~80 with near-zero effective priors
after mixing; Pass doesn't need to be boosted to win the PUCT comparison
if its ~80 competitors get suppressed instead.

### 13. The fine-tune on noise-fixed data is a wash, not a win (2026-09-04)

Fine-tuned `bootstrap_gen1_noise_fix_candidate.pt` on the 6%-short 100-game
batch (`configs/bootstrap_train_gen1_noise_fix_candidate.yaml`, same 25/75
mix recipe as `bootstrap_train_gen1_candidate.yaml`'s v4 fix, 1,541 steps).
Evaluated against its parent (`configs/eval_gen1_noise_fix_candidate_vs_residual.yaml`,
40 games): **candidate 1491.8 vs. parent 1508.2 — do not promote.** The
*same* 25/75 recipe produced a clean 40-0 win on the original (no-noise)
self-play data (section 1). Fixing the pass-collapse didn't preserve
training-data quality: `epsilon=0.25` root noise perturbs every move's
selection throughout each game (via the visit-count policy target,
sampled at `temperature=1.0`), not just early moves -- plausibly trading
catastrophic early-passing for milder overall move-quality noise in every
recorded example.

**Root-caused further via research (2026-09-04):** confirmed this
project's self-play samples moves at a *flat* `temperature=1.0` for the
entire game, every move, in both `selfplay/generate.py` and
`selfplay/self_play.py` -- no annealing at all. Real AlphaZero anneals
this: `temperature=1` (stochastic sampling) for only the first ~30
plies, then `temperature~=0` (deterministic, most-visited move) for the
rest of the game. This project never makes that switch, which is a
second, independent source of noise throughout the whole game --
separate from the Dirichlet noise fix, and likely the bigger contributor
to section 13's wash, since the pass-collapse problem lives almost
entirely in the *opening* (where high temperature is still needed) while
noise/randomness deep into an already-decided game just degrades
quality for no benefit. Bonus finding: KataGo's own published alpha
heuristic (`10.83 / average legal moves`) gives `~=0.132` for 9x9 --
independently validates this project's `alpha=0.1`, no further tuning
needed there.

**Recommended next fix, not yet implemented:** add temperature annealing
(e.g. a `temperature_drop_move` config, ~30, matching AlphaZero's own
value) to `self_play.py`'s move sampling, try that *alone* before
touching `epsilon` further -- it's cheap, doesn't risk reopening the
collapse fix (sections 11-12), and directly targets the newly-identified
mechanism.

### 14. The actual chaining test: substantially reduced, not eliminated (2026-09-04)

The real question this whole day's investigation was built toward:
generated `configs/selfplay_self_play_gen2_noise_fix.yaml` (100 games,
chained from `bootstrap_gen1_noise_fix_candidate.pt`, same fix settings
throughout -- `epsilon=0.25` excluding Pass, `alpha=0.1`, 400 sims).
Tracked short-game rate every 10 games as it generated: 0%, 10%, 20%,
18%, 20%, 23%, 23%, 20%, 20%, **23% final (23/100)**, avg length 69.3
moves, 6,927 examples written to `selfplay_games/self_play_gen2_noise_fix.npz`.

**Verdict: the fix substantially reduces compounding but does not fully
eliminate it.** Gen1 (from the pristine baseline): 6% short games. Gen2
(chained from gen1's candidate): 23%. That's roughly a 4x increase
across one chain step -- real, and worth taking seriously -- but nowhere
near the original *unfixed* bug's trajectory (~0.01% → 1.2-1.5% →
27-35%, i.e. ~100x-ish per generation, reaching near-total collapse
within two chained fine-tunes). The rate leveled off within this
100-game batch itself (climbed 0%→23% over the first ~30 games, then sat
in the 18-23% band for the remaining 70), which is a better sign than
monotonic runaway growth, but this is a single generation's worth of
evidence -- **whether the 6%→23% pattern continues compounding into a
generation 3, or whether 23% is closer to a new steady state, is not
yet known and would need another chain step to answer.**

Section 13's fine-tune-wash finding and its temperature-annealing fix
(not yet applied) are a plausible lever for *this* result too, not just
training-data quality: annealing wouldn't directly touch these short
games (by construction, a ≤20-move game never reaches the proposed
`temperature_drop_move`≈30 threshold), but a cleaner, less-noisy
generation-1 candidate (a real strength win instead of a wash) might
itself produce a better-calibrated value net, which could reduce how
much the *underlying* pass-miscalibration compounds when chained --
untested, a hypothesis for next time, not a conclusion.

### 15. Temperature annealing fixes the fine-tune wash (2026-09-04)

Implemented `temperature_drop_move` (`play_one_game` in
`selfplay/self_play.py`: samples at `temperature` for moves before it,
greedy/argmax after -- matching AlphaZero's own schedule). Covered by
`tests/test_self_play.py::test_temperature_drop_move_makes_move_selection_deterministic`.

Re-ran the full gen1→fine-tune→eval pipeline with `temperature_drop_move=30`
added on top of the existing collapse fix (`configs/selfplay_self_play_gen1_temp_anneal.yaml`,
`bootstrap_train_gen1_temp_anneal_candidate.yaml`,
`eval_gen1_temp_anneal_candidate_vs_residual.yaml`):
- Self-play: 8/100 short games (8%) -- matches the pre-annealing 6%,
  confirming annealing doesn't disturb the collapse fix (as predicted,
  since collapse lives in the opening, still fully covered by
  `temperature=1` below move 30).
- Fine-tune eval vs. pristine baseline: **candidate 1726.1 vs. parent
  1273.9 — PROMOTE.** Not just fixed, decisively so -- a far larger gap
  than the wash (1491.8 vs 1508.2) and comparable to or exceeding the
  original no-noise recipe's clean win (section 1).

**Confirmed: a flat, un-annealed temperature was indeed the main driver
of section 13's wash**, not the Dirichlet noise itself. `bootstrap_gen1_temp_anneal_candidate.pt`
supersedes `bootstrap_gen1_noise_fix_candidate.pt` as the generation-1
checkpoint to build on. `temperature_drop_move=30` should be included in
all future self-play configs alongside the existing noise settings.

**Re-tested with `alpha=0.1`** (20-game smoke test, otherwise identical
settings): **19/20 clean, 1 short (5%)** — a further improvement over
alpha=0.03's 10%, consistent with the branching-factor theory. Confirmed
at full 100-game scale: **6/100 short (6%)**, avg length 87.1 moves,
8,708 examples written to `selfplay_games/self_play_gen1_noise_fix.npz`
(same path as the alpha=0.03 run — overwritten; that batch's numbers are
preserved above for reference). `alpha=0.1` is now both `MctsConfig`'s
code default and the value in
`configs/selfplay_self_play_gen1_noise_fix.yaml`. Still above the
original 1% no-noise baseline — worth keeping in mind if a future
generation shows renewed instability — but a real, reproducible
improvement (10%→6% held from n=20 to n=100) and a good batch to build
the next fine-tune on.

### 16. Re-tested gen2 chaining from the annealed candidate: improved, not solved (2026-09-05)

Ran `configs/selfplay_self_play_gen2_temp_anneal.yaml` (100 games, chained from
`bootstrap_gen1_temp_anneal_candidate.pt` -- the genuine strength win, not the
wash candidate). Result: **17/100 short games (17%)**, avg 75.6 moves, 7,556
examples written to `selfplay_games/self_play_gen2_temp_anneal.npz`.

Comparison across all three gen2 attempts (same chaining structure, gen1's
own rate for reference):
- Gen1 (from pristine baseline, full fix): 8% short games
- Gen2 chained from `bootstrap_gen1_noise_fix_candidate.pt` (the wash
  candidate, section 14): 23%
- Gen2 chained from `bootstrap_gen1_temp_anneal_candidate.pt` (the real
  strength win): **17%**

**Confirms the section 14 hypothesis partially: a cleaner, real-strength-win
gen1 candidate does reduce gen2's compounding (23%→17%), but doesn't
eliminate it.** Chaining still roughly doubles the collapse rate per
generation (8%→17%) even with every fix applied so far. This is the honest
state of Phase 3 as of this session: single-generation collapse is
well-controlled (~6-8%), chaining compounding is reduced but real
(~2x/generation, down from the original ~100x/generation and section 14's
~4x/generation). Section 17 below (further research) has the ranked next
steps if closing this gap further is worth pursuing.

### 17. Further research: ranked fixes if chaining still isn't fully resolved (2026-09-05)

Deep research turned up additional levers beyond noise/alpha/temperature,
ranked by cost/impact for this project's single-machine scale:

1. **Structurally disallow `Pass` for the first ~6-8 moves during
   self-play generation** (generation only, not eval/inference) --
   highest impact, near-zero cost, not yet tried. Not a named technique
   in the literature (AlphaZero's own opening diversity comes from
   visit-count sampling, not move restriction) -- an engineering
   judgment call, but a well-supported one: every collapse inspected so
   far (section 12) happened in the opening, several as Black's literal
   first move, and there is no legitimate 9x9 scenario where passing
   move 1-8 is correct. Unlike every fix tried today (all probabilistic
   nudges), this is a hard structural guarantee against this specific
   failure class rather than a nudged probability. **Try this first if
   chaining still shows meaningful collapse after sections 11+15's
   fixes.**
2. **KataGo's policy target pruning, now fully specified** (arXiv
   1902.10565): minimum forced-playout floor per child
   `n_forced(c) = (k * P(c) * total_visits)^0.5` with `k=2`; before
   computing the training target, prune each non-best child's visits
   downward (not below its forced-playout floor) as long as doing so
   doesn't push its PUCT score above the best child's
   (`PUCT(c) = V(c) + cPUCT*P(c)*sqrt(total_visits)/(1+N(c))`,
   `cPUCT=1.1`; root FPU=0 when noise is enabled). Decouples "noise
   forced a bad move to get tried" from "that move's inflated visit
   count gets imitated as if it were good" -- addresses a mechanism
   today's noise fix doesn't touch. Moderate implementation effort
   (added to `Mcts.search`'s end); hold in reserve.
3. **KL-divergence regularization against the pristine policy** on the
   same positions (a frozen reference net, extra forward pass per
   batch) -- standard practice in iterated self-training literature for
   exactly this class of compounding drift, and a genuinely different
   mechanism than this project's existing 25/75 data-mixing anchor
   (constrains the output distribution directly rather than hoping data
   composition achieves the same effect indirectly). Worth trying if
   (1) doesn't fully resolve chaining.
4. **Entropy regularization / value-target smoothing** -- low priority:
   this problem is one specific rare action creeping up, not the whole
   policy over-sharpening, so entropy regularization is a blunt
   instrument here; value-target smoothing is nearly free but the score
   head result (section 8) already rules out value-target flatness as
   primary driver, so low expected impact.
5. **Add a move-count/pass-count input feature** (real technique, used
   by KataGo/Leela Zero) -- explicitly deferred: breaks
   `docs/MODEL_CONTRACT.md` (shared with the Android app), requires
   Android inference-wrapper and export-tooling changes, and makes
   every existing checkpoint incompatible (can't warm-start across a
   changed input shape). Longer-term only, not a quick fix.

### 18. Eval methodology bug: "40-game matches" were really 2 games repeated (2026-09-05)

While evaluating `bootstrap_gen1_no_pass_guard_candidate.pt` (section 17's
no-pass guard fix), got the exact same two numbers as section 15's
eval -- 1726.1/1273.9 -- just swapped which side won. That's not a
coincidence: `eval/match.py`'s `_play_one_game` used `Mcts.select_move`,
which is pure deterministic argmax, and `MctsConfig()`'s defaults leave
root noise off (`epsilon=0.0`). With two fixed checkpoints and zero
randomness anywhere, every "A plays Black" game is bit-for-bit identical
to every other "A plays Black" game (same for "B plays Black") -- a
"40-game match" was actually only **2 unique games, each replicated
`num_games/2` times**. `eval/elo.py`'s `update_ratings` applies a fixed
per-game Elo delta sequentially starting from 1500/1500, so any clean
2-for-2 sweep converges to the exact same fixed endpoint regardless of
which checkpoints are being compared -- explaining the repeated numbers
exactly. **This likely affects every "40-0" / "PROMOTE" / "wash"
conclusion in this document and the project's history, including every
prior generation's promotion decisions (section 1 onward)** -- the
qualitative direction (a clean 2-for-2 sweep vs. a mixed one) is still
real signal, but the implied statistical confidence of "40 games" was
never real.

**Fixed:** `eval/match.py`'s `play_match`/`_play_one_game` gained
`temperature`/`temperature_drop_move`/`seed` parameters, using the same
search-then-sample approach as `selfplay/self_play.py` (default `0.0`,
i.e. the original deterministic `select_move` behavior, preserved for
backward compatibility and existing tests that explicitly rely on it).
`eval/promote.py` now defaults to `temperature=1.0`,
`temperature_drop_move=16`, `seed=0` for real promotion decisions, so a
"40-game match" is actually 40 different games. Covered by two new
`tests/test_match.py` cases (variation across same-color games; seeded
reproducibility).

**Re-checked both of today's key comparisons with the fixed eval**
(`temperature=1.0`, `temperature_drop_move=16`, real 40 distinct games):
- `bootstrap_gen1_temp_anneal_candidate.pt`: **1619.0 vs. 1381.0 —
  still PROMOTE.** Smaller margin than the old deterministic result
  (1726.1/1273.9, inflated by only 2 real trials), but the direction
  holds: genuinely stronger.
- `bootstrap_gen1_no_pass_guard_candidate.pt`: **1543.0 vs. 1457.0 —
  PROMOTE.** This *reverses* the earlier deterministic-eval verdict
  ("do not promote", 1273.9/1726.1) -- that result was an artifact of
  the bug, not a real assessment. With genuine statistical testing, the
  no-pass-guard candidate is also a real strength win, just a more
  modest one than temp-anneal's.

**Conclusion: both fixes are validated as genuine strength wins, not
washes.** `bootstrap_gen1_no_pass_guard_candidate.pt` is now the
checkpoint of choice going forward -- it has both the strongest
collapse-rate result of the whole investigation (0/100 short games) and
a confirmed real strength win, correcting the false "wash" reading that
would have wrongly discarded it.

### 19. The chaining test, finally: 0/100 in both generations (2026-09-05)

Ran `configs/selfplay_self_play_gen2_no_pass_guard.yaml` (100 games,
chained from `bootstrap_gen1_no_pass_guard_candidate.pt` -- the
structural-guard candidate, confirmed clean in isolation and a genuine
strength win with the fixed eval). This is the actual test the whole
day's investigation was building toward.

**Result: 0/100 short games, avg 94.8 moves, 9,479 examples written to
`selfplay_games/self_play_gen2_no_pass_guard.npz`.** Tracked every 10
games as it generated: 0%, flat, the entire way through -- not a single
short/collapsed game in the full 100-game generation-2 batch. Compare
to every prior gen2 attempt at this same chaining structure: 23%
(section 14, wash candidate), 17% (section 16, temp-anneal candidate).
**The structural `no_pass_before_move` guard closes the chaining gap
that noise + alpha + temperature annealing alone only narrowed.**

**This is the conclusive result for Phase 3's chaining-collapse
investigation.** Both generation 1 (section 17) and generation 2 (this
section) came back completely clean at full 100-game scale, using the
identical fix stack chained end to end. Unlike every earlier fix
(probabilistic nudges that reduced but never eliminated the compounding
pattern), the hard structural guarantee against the exact observed
failure mode (an illegitimate early Pass) appears to fully solve it.
Phase 3's self-play pass-collapse bug can now be described as
**resolved**, not just "much improved" -- pending only the standard
caveat that this is one 100-game sample, not a formal proof, and a
generation 3 chain step (not yet run) would be the natural further
confirmation if more certainty is wanted before treating this as
permanently settled.

### 20. Generation 2's fine-tune: also a genuine strength win, chained (2026-09-05)

Fine-tuned `bootstrap_gen2_no_pass_guard_candidate.pt` on
`self_play_gen2_no_pass_guard.npz` (same 25/75 mix recipe, warm-started
from `bootstrap_gen1_no_pass_guard_candidate.pt` -- a real chained
fine-tune, not from the pristine baseline). Evaluated with the fixed
eval (section 18) two ways:

- vs. the pristine baseline (`bootstrap_batch2_residual.pt`): **1579.6
  vs. 1420.4 — PROMOTE.**
- vs. its own generation-1 parent (`bootstrap_gen1_no_pass_guard_candidate.pt`):
  **1595.4 vs. 1404.6 — PROMOTE.**

**This completes the picture: chaining is now compounding strength, not
just avoiding collapse.** Generation 2 beats generation 1, which beats
the pristine baseline -- the healthy AlphaZero-style improvement loop
Phase 3 was always supposed to produce, now actually working across a
real chained fine-tune with zero collapse at every step.
`bootstrap_gen2_no_pass_guard_candidate.pt` is the new best checkpoint.

### 21. Generation 3, for extra confidence: the pattern holds a third time (2026-09-05)

Ran the same recipe one more generation (`configs/selfplay_self_play_gen3_no_pass_guard.yaml`,
chained from `bootstrap_gen2_no_pass_guard_candidate.pt`) to check whether
sections 19-20's result was a one-off or a genuinely sustained pattern.

- Self-play: **0/100 short games**, avg 97.3 moves, 9,729 examples
  (`selfplay_games/self_play_gen3_no_pass_guard.npz`) -- third
  consecutive generation completely clean.
- Fine-tuned `bootstrap_gen3_no_pass_guard_candidate.pt` (same 25/75 mix
  recipe, warm-started from the gen2 candidate). Evaluated with the
  fixed eval:
  - vs. the pristine baseline: **1597.9 vs. 1402.1 — PROMOTE.**
  - vs. its own generation-2 parent: **1582.7 vs. 1417.3 — PROMOTE.**

**Three generations in a row, each one a genuine strength win over its
predecessor, with zero collapse at every single step.** This is no
longer just "the fix worked once" -- it's a sustained, repeating,
healthy improvement loop. `bootstrap_gen3_no_pass_guard_candidate.pt` is
the new best checkpoint. Phase 3's investigation can be considered
closed; the natural next work is scaling this proven loop further
(more generations, bigger self-play batches, or moving fully into Phase
4's product integration), not further stability debugging.

### 22. Generation 4: the pattern holds a fourth time, widening igo-app's difficulty spread (2026-09-06)

Ran the same recipe one more generation (`configs/selfplay_self_play_gen4_no_pass_guard.yaml`,
chained from `bootstrap_gen3_no_pass_guard_candidate.pt`) -- this time motivated by a concrete
product need (igo-app's difficulty picker only had two tiers, gen2/gen3) rather than pure
stability confidence-building.

- Self-play: **0/100 short games, 0 discarded for elevated mid-game Pass weight** -- fourth
  consecutive generation completely clean. 9,748 examples from 100 games, ~180 minutes wall
  time (400 sims/move, same budget as gen1-gen3).
- Fine-tuned `bootstrap_gen4_no_pass_guard_candidate.pt` (same 25/75 mix recipe, warm-started
  from the gen3 candidate). Evaluated with the fixed eval:
  - vs. the pristine baseline: **1614.2 vs. 1385.8 -- PROMOTE.**
  - vs. its own generation-3 parent: **1552.3 vs. 1447.7 -- PROMOTE.**

**Four generations in a row now, each a genuine strength win over its predecessor, zero
collapse at every step.** `bootstrap_gen4_no_pass_guard_candidate.pt` is the new best
checkpoint. Exported to `export/ray_zero_gen4_no_pass_guard.tflite` and added to igo-app as a
third difficulty tier (see igo-app/docs/ROADMAP.md) -- the first generation exported
specifically to widen the difficulty spread rather than as a "does the pipeline still work"
check.

One operational note, not a research finding: this run's background process was killed once
by a system-wide low-memory event (idle Gradle/Kotlin build daemons from unrelated igo-app
work were holding ~1.6GB on the Windows host) before completing on retry after those daemons
were stopped (`./gradlew.bat --stop`). Nothing about the self-play run itself was implicated --
worth remembering if a future long-running background job here gets killed unexpectedly:
check for unrelated memory pressure on the host before assuming a bug in this pipeline.

### 23. Generation 5, and parallelizing self-play to make a gen1-gen10 chain practical (2026-09-06)

Motivated by wanting to push the proven chain to (at least) generation 10 for a real
difficulty-tier spread, not just confidence-building -- but generation 4's self-play alone
took ~180 minutes serial, despite this machine having 16 CPU cores. `selfplay/self_play.py`
is deliberately single-threaded per process (single-position MCTS inference doesn't benefit
from multi-threading), so the other 15 cores sat idle the whole time. New
`selfplay/run_parallel.py` splits `num_games` across N real `self_play.py` subprocesses
(unmodified, just a temporary config overriding `num_games`/`seed`/`out_path` each), then
concatenates their outputs -- same recipe and total game count as a serial run, just
wall-clock parallel. Smoke-tested against the existing 3-game config before trusting it for
a real generation.

- Self-play (chained from `bootstrap_gen4_no_pass_guard_candidate.pt`, 4 workers): **0/100
  short games**, 9,810 examples from 100 games, **55 minutes wall time** -- a ~3.3x speedup
  over gen4's serial 180 minutes (not the naive 4x, since the workers weren't perfectly
  balanced: 25/25/25/25 games split evenly, but per-game move-count variance means some
  workers finish their batch later than others). Real memory usage stayed well within the
  WSL VM's 7.7GB limit (~2.5GB used with 4 workers, per `free -h`) -- much of each worker's
  ~1GB RSS is shared library pages (torch/Python runtime) the OS maps once and shares
  read-only across the separate processes, not 4x duplicated.
- Fine-tuned `bootstrap_gen5_no_pass_guard_candidate.pt` (same 25/75 mix recipe, warm-started
  from the gen4 candidate). Evaluated with the fixed eval:
  - vs. the pristine baseline: **1601.8 vs. 1398.2 -- PROMOTE.**
  - vs. its own generation-4 parent: **1557.4 vs. 1442.6 -- PROMOTE.**

**Five generations in a row now, each a genuine strength win over its predecessor, zero
collapse at every step.** `bootstrap_gen5_no_pass_guard_candidate.pt` is the new best
checkpoint. Not yet exported to `.tflite` or added to igo-app -- only gen2/gen3/gen4 are
wired into the app's difficulty picker so far; holding off on exporting every single
generation to avoid churning the app's tiers mid-chain, revisiting once the chain reaches a
natural stopping point near generation 10.

One important caveat on absolute strength, raised directly by the user after playing against
gen4: each generation's *relative* improvement is real and Elo-measured, but the *absolute*
playing strength after even 10 generations will likely still feel weak -- this is a small
net (4 residual blocks/64 channels) trained on only ~10k self-play examples per generation,
nowhere near AlphaZero-scale data volume. Reaching generation 10 proves the loop keeps
compounding and gives igo-app more difficulty tiers; it is not expected to produce a strong
player on its own. A genuinely strong opponent would need a bigger lever than generation
count alone (more games/generation, a bigger net, or substantially more compute) -- worth
revisiting once the gen1-10 chain's trajectory is actually in hand.

### 24. Generation 6's first "do not promote" -- a plateau/noise event, not a collapse (2026-09-06)

Chained from `bootstrap_gen5_no_pass_guard_candidate.pt` with the identical recipe (4-worker
parallel self-play, 25/75 mix fine-tune). Self-play was clean: **0/100 short games**, 9,907
examples, 55m21s wall time. But evaluation gave the **first non-promoted generation in the
entire gen1-gen6 chain**:

- vs. its own generation-5 parent: **1413.1 vs. 1586.9 -- Do not promote.**
- vs. the pristine baseline: **1634.2 vs. 1365.8 -- PROMOTE** (a *wider* margin than gen5's
  own 1601.8 vs. 1398.2 against the same baseline).

Read together, this isn't a collapse: `bootstrap_gen6_no_pass_guard_candidate.pt` is still a
strong net -- it beats the pristine baseline more decisively than gen5 did -- it simply
didn't come out ahead of gen5 specifically in this 40-game sample. Elo from small pairwise
samples is not strictly transitive (different opponents probe different weaknesses), and the
fine-tune step's training budget is small and noisy by design (`max_train_seconds: 60`,
~1600 steps over a ~10k-example batch) -- a run of bad luck in that 60 seconds is a
believable explanation, not something to over-investigate given five prior generations all
worked cleanly with this exact recipe. Treating this as a plateau/noise event: **not**
chaining generation 7 from this weaker `gen6` candidate. Instead, retrying generation 6's
self-play + fine-tune + eval cycle with a new seed (`selfplay_self_play_gen6_no_pass_guard.yaml`'s
seed bumped, chain parent unchanged at `bootstrap_gen5_no_pass_guard_candidate.pt`) to see if
a different self-play sample produces a clean promote, before deciding whether this
generation's difficulty (not the pipeline) needs a bigger fine-tune budget going forward.

**The retry (seed 42) also failed to promote against gen5** -- a second independent
self-play sample, same recipe otherwise:

- vs. its own generation-5 parent: **1510.8 vs. 1489.2 -- Do not promote** (a ~22-Elo gap,
  well under the 50-Elo promotion threshold -- much closer than the first attempt's 174-Elo
  gap, essentially a statistical tie rather than a loss).
- vs. the pristine baseline: **1659.1 vs. 1340.9 -- PROMOTE**, the most decisive baseline win
  of any generation yet (wider than both gen5's 1601.8 vs. 1398.2 and the first gen6
  attempt's 1634.2 vs. 1365.8).

**Conclusion: this is a real plateau at gen5's strength level under the current recipe, not
a one-off unlucky self-play sample.** Two independently-sampled self-play batches, fine-tuned
identically, both failed to clear gen5 head-to-head -- one decisively, one essentially a
coin-flip -- while both kept improving against the fixed pristine baseline. **Paused the
chain here rather than attempting a third retry or chaining generation 7 from a
non-promoted candidate.** Deep-dive into the likely cause(s) (network capacity, self-play
data volume, fine-tune training budget, the 25/75 anchor mix, and eval statistical power)
requested from the user and delegated to a research pass.

**Research findings (2026-09-06):** running the actual Elo math confirmed the two attempts
above are not equal evidence -- attempt 1's -173.8 Elo gap is a real, statistically
distinguishable regression (implied win rate ~27%, 95% CI ~13-41%), while attempt 2's +21.6
Elo gap is statistically indistinguishable from a coin flip at n=40 (implied win rate ~53%,
95% CI ~38-69%) -- likely a true tie, not a loss. Root-cause assessment ranked the missing
accumulated-self-play replay buffer as the most likely real, well-evidenced contributor:
every generation's self-play data was generated once and discarded, the fixed 25% anchor
(`batch2.npz`) is stale pre-self-play imitation data now 5 generations behind the
checkpoint's own quality, and established 9x9 self-play replications (MiniZero: 2,000
games/iteration from a 40,000-game rolling buffer; a population-based-training paper: 5,000
games/iteration) generate 20-50x more new data per iteration than this project's 100 games.
The 60-second fine-tune budget was *not* implicated as under-training -- the math shows each
self-play example already gets seen ~7.8 times within that window, so extending training
time alone would likely deepen overfitting to a narrow batch rather than fix anything.
Network capacity was judged plausible but unproven either way (no cheap diagnostic run yet
under the current chained-fine-tune regime, as opposed to the large-static-dataset regime
where the residual tower was previously confirmed not yet capacity-limited).

**Attempt 3, testing the two free/cheap recommendations together:** reused attempt 2's
self-play data (seed 42, already on disk, no new self-play needed) but replaced the fine-tune's
data mix with a real replay buffer (10% `batch2.npz`, 15% `self_play_gen4_no_pass_guard.npz`,
25% `self_play_gen5_no_pass_guard.npz`, 50% `self_play_gen6_no_pass_guard.npz` -- down from
25% batch2.npz/75% self-play-only), and widened the vs-gen5 eval from 40 to 80 games to
actually resolve a close call statistically.

- vs. its own generation-5 parent: **1515.4 vs. 1484.6 -- Do not promote** (30.8-Elo gap,
  still under the 50-Elo threshold, though nominally higher than attempt 2's 21.6-Elo gap --
  with n=80 vs. n=40 the standard error shrank, but the two results still overlap
  substantially; this shift is not itself strong evidence the mix change helped).
- vs. the pristine baseline: **1645.3 vs. 1354.7 -- PROMOTE**, consistent with every other
  gen6 attempt (1601.8 gen5 / 1634.2 attempt 1 / 1659.1 attempt 2 / 1645.3 attempt 3, all
  comfortably beating baseline in a tight band).

**Conclusion: the free fixes (replay buffer + wider eval) did not resolve the plateau.**
Three independent attempts (two self-play samples, two fine-tune mixes) now land in the same
place -- solidly ahead of the baseline, not reliably ahead of gen5. This shifts the balance
of evidence toward the more expensive, not-yet-tried lever: self-play data volume per
generation (100 games, ~10k examples) may itself be the binding constraint, not the mix or
the eval alone. Paused here, reporting the full picture and remaining options (more
self-play games per generation, a capacity-ceiling diagnostic, or accepting gen5 as the
current plateau) to the user rather than spending more compute unilaterally.

**Attempt 4, tripling self-play games per generation -- resolved the plateau.** Per the
user's choice among the laid-out options, tested the remaining untried lever: `num_games`
raised from 100 to 300 (the top of the research's recommended 250-300 range), keeping the
replay-buffer fine-tune mix from attempt 3. Self-play: **0/300 short games**, 29,467
examples (3x the usual volume), 165m25s wall time with the 4-worker parallel runner (~3x
attempt 3's self-play time, as expected for 3x the games). Fine-tune's mix updated to weight
this much larger batch appropriately (10% `batch2.npz` / 15% `self_play_gen4` / 25%
`self_play_gen5` / 50% `self_play_gen6`, same shape as attempt 3's mix). Evaluated with the
80-game eval:

- vs. its own generation-5 parent: **1556.8 vs. 1443.2 -- PROMOTE** (113.6-Elo gap, decisively
  clear of the 50-Elo threshold -- the first clean promote after three 100-game attempts).
- vs. the pristine baseline: **1655.2 vs. 1344.8 -- PROMOTE**, consistent with every prior
  attempt's baseline win.

**Conclusion confirmed: self-play data volume per generation was the real binding
constraint.** Three independent 100-game attempts (two self-play samples, two fine-tune
mixes) all failed to clear gen5; the first 300-game attempt cleared it decisively on the
first try. `bootstrap_gen6_no_pass_guard_candidate.pt` is the new best checkpoint, and the
chain resumes. **New standing recipe going forward (not reverting to the old
100-games/25-75-mix/40-game-eval recipe):** 300 self-play games/generation, a replay-buffer
fine-tune mix (small legacy `batch2.npz` anchor + a weighted window of recent self-play
generations favoring the newest), and an 80-game eval against the immediate parent (40 games
remains fine for the less-noisy baseline comparison, which has never been a close call).

## Open items as of this writing (end of 2026-09-05 session)

- **Phase 3's self-play pass-collapse bug is resolved (section 19).**
  Both generation 1 and generation 2 came back **0/100 short games** at
  full scale, chained end to end, using: root noise excluding Pass
  (`alpha=0.1`, tuned for 9x9), temperature annealing
  (`temperature_drop_move=30`), and the structural
  `no_pass_before_move=20` guard. This is a real change from the
  ~2x/generation compounding seen with the noise+alpha+temperature-only
  fix (sections 14, 16) -- the structural guard was the piece that
  closed the remaining gap. Standard caveat: one 100-game sample per
  generation, not a formal proof -- a generation 3 chain step (not yet
  run) would be the natural further confirmation if more certainty is
  wanted, but there is no longer a known, reproducible failure mode left
  to chase.
- **Eval methodology fixed (section 18) — re-read before trusting any
  older PROMOTE/wash number in this document without the fix applied.**
  `eval/promote.py` now uses real per-game randomness
  (`temperature=1.0`, `temperature_drop_move=16`); a "40-game match"
  used to be only 2 unique deterministic games repeated. Re-checking
  with the fix reversed one conclusion: `bootstrap_gen1_no_pass_guard_candidate.pt`
  is a genuine PROMOTE (1543.0 vs. 1457.0), not the "do not promote" the
  broken eval implied.
- Not yet done: fine-tune a candidate on `self_play_gen2_no_pass_guard.npz`
  and evaluate it (with the fixed eval) to confirm generation 2 is also
  a real strength win, not just collapse-free -- section 19 only
  confirms the self-play *generation* step, not yet a full second
  fine-tune cycle. Also not yet tried: a generation 3 chain step (see
  above) for extra confidence, and section 17's #2/#3 (KataGo policy
  target pruning, KL anchor) remain available but are no longer
  necessary given section 19's result -- hold in reserve only.
- Score-margin auxiliary head: implemented (section 7) and evaluated
  (section 8) — real, working, keep it (no downside), but was never the
  fix for chaining collapse.
- Warm-start chaining restructuring (section 9) and hardened self-play
  filtering (section 10): both superseded/subsumed by the fixes above,
  but hardened filtering is still worth keeping as a cheap safety net
  regardless.
- A full continuous replay-buffer training loop remains deliberately
  deferred as disproportionate for this project's single-machine scale
  — revisit only if section 17's options don't pan out.
- **Known-good checkpoints, in order of preference:**
  `bootstrap_gen6_no_pass_guard_candidate.pt` (best overall — six chained
  generations, each beating its predecessor: PROMOTE vs. the pristine
  baseline 1655.2 vs. 1344.8, vs. its own gen5 parent 1556.8 vs. 1443.2,
  the first generation trained on 300 self-play games instead of 100 --
  see section 24's attempt 4) > `bootstrap_gen5_no_pass_guard_candidate.pt`
  (its parent — PROMOTE vs. the pristine baseline 1601.8 vs. 1398.2, vs.
  its own gen4 parent 1557.4 vs. 1442.6, with 0/100 short games at every
  generation in the chain; see section 23) > `bootstrap_gen4_no_pass_guard_candidate.pt` (its
  parent — PROMOTE vs. the pristine baseline 1614.2 vs. 1385.8, vs. its
  own gen3 parent 1552.3 vs. 1447.7, 0/100 short games; see section 22)
  > `bootstrap_gen3_no_pass_guard_candidate.pt` (its parent — PROMOTE
  vs. the pristine baseline 1597.9 vs. 1402.1, vs. its own gen2 parent
  1582.7 vs. 1417.3, 0/100 short games)
  > `bootstrap_gen2_no_pass_guard_candidate.pt` (its parent — PROMOTE
  vs. both the pristine baseline, 1579.6 vs. 1420.4, and its own gen1
  parent, 1595.4 vs. 1404.6, 0/100 short games) >
  `bootstrap_gen1_no_pass_guard_candidate.pt` (the first full-fix-stack
  checkpoint — PROMOTE 1543.0 vs. 1457.0, 0/100 short games) >
  `bootstrap_gen1_temp_anneal_candidate.pt` (PROMOTE 1619.0 vs. 1381.0 —
  a bigger single-generation strength margin than the guard candidate,
  but 8% short games vs. 0%, and not chained further) >
  `bootstrap_gen1_candidate.pt` / `bootstrap_gen1_candidate_score_head.pt`
  (original pre-investigation recipe, still valid, promoted, stable in
  isolation) — all build from `bootstrap_batch2_residual.pt`. Note all
  PROMOTE numbers above used the *fixed* eval (section 18); any older
  number elsewhere in this doc used the broken deterministic one.
- **Known-bad/known-mediocre checkpoints — don't build on:** the
  original chained-warm-start attempts, `bootstrap_gen2_candidate.pt`
  (34.5% Pass) and `bootstrap_gen2_candidate_score_head.pt` (27.04%
  Pass); the restructured attempt, `bootstrap_gen2_candidate_restructured.pt`
  (16.09% Pass, not promoted); and `bootstrap_gen1_noise_fix_candidate.pt`
  (real collapse-rate fix, but a strength wash even under re-check —
  this one's "do not promote" was never re-tested with the fixed eval,
  but section 15 already superseded it on other grounds, so low
  priority to re-check).
- Data files: `selfplay_games/self_play_gen2_hardened.npz` (779
  examples, both filters applied) exists but is too small to have been
  used for anything yet — no fine-tune has been run on it.
