#!/usr/bin/env bash
# Option 2 of the weak-opening follow-up plan (see docs/ROADMAP.md): a real generation-12
# cycle with augment: true, same standard recipe/budget as every prior generation. Run this
# after Option 1 (run_option1_augment_extended.sh) and after deciding, from its spot-check +
# eval results, whether generation 12 should be based on plain gen11 or on
# bootstrap_gen11_augment_extended_candidate.pt -- if the latter, edit checkpoint_path in
# configs/selfplay_self_play_gen12_augment.yaml AND init_from_checkpoint in
# configs/bootstrap_train_gen12_augment_candidate.yaml to that checkpoint BEFORE running this
# script. Both default to plain gen11.
#
# Self-play (Stage 1) takes hours (800 games, ~180min/300 games observed elsewhere in this
# chain) -- run this in your own detached WSL tmux session per this project's established
# practice (Claude-Code-launched background jobs risk the host's low-memory guard killing
# them for reasons unrelated to actual usage), not directly through Claude Code.
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

echo "=== [$(date)] Stage 1/4: self-play (800 games, 4 workers) ==="
python -m selfplay.run_parallel --config configs/selfplay_self_play_gen12_augment.yaml --workers 4

echo "=== [$(date)] Stage 2/4: fine-tune (GPU, augment: true) ==="
python -m bootstrap.train --config configs/bootstrap_train_gen12_augment_candidate.yaml

echo "=== [$(date)] Stage 3/4: opening spot-check ==="
python -m tools.spotcheck_opening \
    checkpoints/bootstrap_gen11_no_pass_guard_candidate.pt \
    checkpoints/bootstrap_gen12_augment_candidate.pt

echo "=== [$(date)] Stage 4/4: eval vs gen11 ==="
python -m eval.promote --config configs/eval_gen12_augment_candidate_vs_gen11.yaml \
    --candidate checkpoints/bootstrap_gen12_augment_candidate.pt

echo "=== [$(date)] Stage 4/4: eval vs pristine baseline ==="
python -m eval.promote --config configs/eval_gen12_augment_candidate_vs_residual.yaml \
    --candidate checkpoints/bootstrap_gen12_augment_candidate.pt

echo "=== [$(date)] gen12 (augmented) pipeline complete ==="
