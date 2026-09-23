#!/usr/bin/env bash
# Option 3, generation 8, at the 300-game recipe gen5/gen6/gen7 confirmed works at this
# plateau. Parent is gen7 (promoted +116.6 Elo vs gen6, +452.2 Elo vs baseline). Both of
# gen7's own review red flags (tengen concentration spike, eval-vs-baseline duplicate
# numbers) were investigated 2026-09-22 and confirmed benign -- gen7 is a trustworthy
# parent to chain from.
#
# Self-play (300 games) takes real wall-clock time (~4-5h on this machine) -- run this
# in your own detached WSL tmux session per this project's established practice.
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

echo "=== [$(date)] Stage 1/5: self-play (300 games, 4 workers) ==="
python -m selfplay.run_parallel --config configs/selfplay_self_play_option3_gen8.yaml --workers 4

echo "=== [$(date)] Stage 2/5: fine-tune (augment: true) ==="
python -m bootstrap.train --config configs/bootstrap_train_option3_gen8_candidate.yaml

echo "=== [$(date)] Stage 3/5: opening spot-check ==="
python -m tools.spotcheck_opening \
    checkpoints/bootstrap_batch2_residual_v2.pt \
    checkpoints/bootstrap_option3_gen1_candidate.pt \
    checkpoints/bootstrap_option3_gen2_candidate.pt \
    checkpoints/bootstrap_option3_gen3_candidate.pt \
    checkpoints/bootstrap_option3_gen4_300games_candidate.pt \
    checkpoints/bootstrap_option3_gen5_candidate.pt \
    checkpoints/bootstrap_option3_gen6_candidate.pt \
    checkpoints/bootstrap_option3_gen7_candidate.pt \
    checkpoints/bootstrap_option3_gen8_candidate.pt

echo "=== [$(date)] Stage 4/5: eval vs gen7 (80 games) ==="
python -m eval.promote --config configs/eval_option3_gen8_candidate_vs_gen7.yaml \
    --candidate checkpoints/bootstrap_option3_gen8_candidate.pt

echo "=== [$(date)] Stage 5/5: eval vs new baseline ==="
python -m eval.promote --config configs/eval_option3_gen8_candidate_vs_baseline.yaml \
    --candidate checkpoints/bootstrap_option3_gen8_candidate.pt

echo "=== [$(date)] Option 3 gen8 complete ==="
