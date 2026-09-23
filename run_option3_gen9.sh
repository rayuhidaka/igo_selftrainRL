#!/usr/bin/env bash
# Option 3, generation 9. Parent is gen8 (promoted +68.8 Elo vs gen7, +452.2 Elo
# vs baseline). Stepping num_games 300->500 this generation -- the vs-parent Elo
# gain has been shrinking each round (gen6 +305.4 -> gen7 +116.6 -> gen8 +68.8),
# the same shrinking-gap shape that preceded this project's two real plateaus
# before -- so this escalates the games count preemptively rather than waiting
# for an outright failed promotion.
#
# Self-play (500 games, up from 300) takes real wall-clock time -- run this in
# your own detached WSL tmux session per this project's established practice.
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

echo "=== [$(date)] Stage 1/5: self-play (500 games, 4 workers) ==="
python -m selfplay.run_parallel --config configs/selfplay_self_play_option3_gen9.yaml --workers 4

echo "=== [$(date)] Stage 2/5: fine-tune (augment: true) ==="
python -m bootstrap.train --config configs/bootstrap_train_option3_gen9_candidate.yaml

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
    checkpoints/bootstrap_option3_gen9_candidate.pt

echo "=== [$(date)] Stage 4/5: eval vs gen8 (80 games) ==="
python -m eval.promote --config configs/eval_option3_gen9_candidate_vs_gen8.yaml \
    --candidate checkpoints/bootstrap_option3_gen9_candidate.pt

echo "=== [$(date)] Stage 5/5: eval vs new baseline ==="
python -m eval.promote --config configs/eval_option3_gen9_candidate_vs_baseline.yaml \
    --candidate checkpoints/bootstrap_option3_gen9_candidate.pt

echo "=== [$(date)] Option 3 gen9 complete ==="
