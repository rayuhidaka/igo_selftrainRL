#!/usr/bin/env bash
# Option 3, generation 5, at the 300-game recipe gen4's retry confirmed works at this
# plateau. Parent is gen4_300games (promoted +71 Elo vs gen3, +229 vs baseline).
#
# Self-play (300 games) takes real wall-clock time (~4h on this machine) -- run this
# in your own detached WSL tmux session per this project's established practice.
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

echo "=== [$(date)] Stage 1/5: self-play (300 games, 4 workers) ==="
python -m selfplay.run_parallel --config configs/selfplay_self_play_option3_gen5.yaml --workers 4

echo "=== [$(date)] Stage 2/5: fine-tune (augment: true) ==="
python -m bootstrap.train --config configs/bootstrap_train_option3_gen5_candidate.yaml

echo "=== [$(date)] Stage 3/5: opening spot-check ==="
python -m tools.spotcheck_opening \
    checkpoints/bootstrap_batch2_residual_v2.pt \
    checkpoints/bootstrap_option3_gen1_candidate.pt \
    checkpoints/bootstrap_option3_gen2_candidate.pt \
    checkpoints/bootstrap_option3_gen3_candidate.pt \
    checkpoints/bootstrap_option3_gen4_300games_candidate.pt \
    checkpoints/bootstrap_option3_gen5_candidate.pt

echo "=== [$(date)] Stage 4/5: eval vs gen4 (300 games) ==="
python -m eval.promote --config configs/eval_option3_gen5_candidate_vs_gen4_300games.yaml \
    --candidate checkpoints/bootstrap_option3_gen5_candidate.pt

echo "=== [$(date)] Stage 5/5: eval vs new baseline ==="
python -m eval.promote --config configs/eval_option3_gen5_candidate_vs_baseline.yaml \
    --candidate checkpoints/bootstrap_option3_gen5_candidate.pt

echo "=== [$(date)] Option 3 gen5 complete ==="
