#!/usr/bin/env bash
# Option 3, generation 11. Parent is gen10 (promoted +111.6 Elo vs gen9 at the same
# 500-game recipe). Same recipe again, 500 games unchanged.
#
# Self-play (500 games) takes ~8h wall-clock end to end -- run this in your own
# detached WSL tmux session per this project's established practice.
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

echo "=== [$(date)] Stage 1/5: self-play (500 games, 4 workers) ==="
python -m selfplay.run_parallel --config configs/selfplay_self_play_option3_gen11.yaml --workers 4

echo "=== [$(date)] Stage 2/5: fine-tune (augment: true) ==="
python -m bootstrap.train --config configs/bootstrap_train_option3_gen11_candidate.yaml

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
    checkpoints/bootstrap_option3_gen8_candidate.pt \
    checkpoints/bootstrap_option3_gen9_candidate.pt \
    checkpoints/bootstrap_option3_gen10_candidate.pt \
    checkpoints/bootstrap_option3_gen11_candidate.pt

echo "=== [$(date)] Stage 4/5: eval vs gen10 (80 games) ==="
python -m eval.promote --config configs/eval_option3_gen11_candidate_vs_gen10.yaml \
    --candidate checkpoints/bootstrap_option3_gen11_candidate.pt

echo "=== [$(date)] Stage 5/5: eval vs new baseline ==="
python -m eval.promote --config configs/eval_option3_gen11_candidate_vs_baseline.yaml \
    --candidate checkpoints/bootstrap_option3_gen11_candidate.pt

echo "=== [$(date)] Option 3 gen11 complete ==="
