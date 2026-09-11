#!/usr/bin/env bash
# Option 1 of the weak-opening follow-up plan (see docs/ROADMAP.md): a longer (900s)
# augmented fine-tune, no new self-play needed -- continues from
# bootstrap_gen11_augment_candidate.pt (already confirmed: restores rotational symmetry,
# all 4 corners ~equal probability). Cheap (~15 min fine-tune + ~15-20 min eval), no
# self-play generation, so this does NOT need to run in a detached tmux session the way
# self-play runs do -- fine and light enough for Claude Code's own background-task
# mechanism, or just run directly.
#
# Stops after eval rather than auto-deciding promotion -- always a human judgment call in
# this project (see docs/ROADMAP.md's gen6/gen8/gen9/gen10/gen11 entries). Read the spot-check
# output AND the eval result before deciding whether to proceed to Option 2 from this
# checkpoint or from plain gen11 -- see selfplay_self_play_gen12_augment.yaml's comment.
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

echo "=== [$(date)] Stage 1/3: extended augmented fine-tune (900s) ==="
python -m bootstrap.train --config configs/bootstrap_train_gen11_augment_extended_candidate.yaml

echo "=== [$(date)] Stage 2/3: opening spot-check ==="
python -m tools.spotcheck_opening \
    checkpoints/bootstrap_gen11_no_pass_guard_candidate.pt \
    checkpoints/bootstrap_gen11_augment_candidate.pt \
    checkpoints/bootstrap_gen11_augment_extended_candidate.pt

echo "=== [$(date)] Stage 3/3: strength check vs plain gen11 ==="
python -m eval.promote --config configs/eval_gen11_augment_extended_candidate_vs_gen11.yaml \
    --candidate checkpoints/bootstrap_gen11_augment_extended_candidate.pt

echo "=== [$(date)] Stage 3/3: strength check vs pristine baseline ==="
python -m eval.promote --config configs/eval_gen11_augment_extended_candidate_vs_residual.yaml \
    --candidate checkpoints/bootstrap_gen11_augment_extended_candidate.pt

echo "=== [$(date)] Option 1 complete -- read the spot-check + both eval results before deciding next step ==="
