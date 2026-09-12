#!/usr/bin/env bash
# Option 3, generation 4 retry at 300 games -- the 100-game attempt lost to gen3 (-36
# Elo), following a monotonically shrinking vs-parent gap (gen2 +235 -> gen3 +95 -> gen4
# -36). Escalating games, same as the original chain's gen6/gen9 plateau responses.
#
# Self-play (300 games) takes real wall-clock time (~3x a 100-game generation) -- run
# this in your own detached WSL tmux session per this project's established practice.
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

echo "=== [$(date)] Stage 1/5: self-play (300 games, 4 workers) ==="
python -m selfplay.run_parallel --config configs/selfplay_self_play_option3_gen4_300games.yaml --workers 4

echo "=== [$(date)] Stage 2/5: fine-tune (augment: true) ==="
python -m bootstrap.train --config configs/bootstrap_train_option3_gen4_300games_candidate.yaml

echo "=== [$(date)] Stage 3/5: opening spot-check ==="
python -m tools.spotcheck_opening \
    checkpoints/bootstrap_batch2_residual_v2.pt \
    checkpoints/bootstrap_option3_gen1_candidate.pt \
    checkpoints/bootstrap_option3_gen2_candidate.pt \
    checkpoints/bootstrap_option3_gen3_candidate.pt \
    checkpoints/bootstrap_option3_gen4_candidate.pt \
    checkpoints/bootstrap_option3_gen4_300games_candidate.pt

echo "=== [$(date)] Stage 4/5: eval vs gen3 ==="
python -m eval.promote --config configs/eval_option3_gen4_300games_candidate_vs_gen3.yaml \
    --candidate checkpoints/bootstrap_option3_gen4_300games_candidate.pt

echo "=== [$(date)] Stage 5/5: eval vs new baseline ==="
python -m eval.promote --config configs/eval_option3_gen4_300games_candidate_vs_baseline.yaml \
    --candidate checkpoints/bootstrap_option3_gen4_300games_candidate.pt

echo "=== [$(date)] Option 3 gen4 (300 games) complete ==="
