#!/usr/bin/env bash
# Option 3, generation 4 -- gen3 promoted (~95 Elo vs gen2), no plateau, continuing at
# the same 100-game recipe. See docs/ROADMAP.md's "weak opening moves" item.
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

echo "=== [$(date)] Stage 1/5: self-play (100 games, 4 workers) ==="
python -m selfplay.run_parallel --config configs/selfplay_self_play_option3_gen4.yaml --workers 4

echo "=== [$(date)] Stage 2/5: fine-tune (augment: true) ==="
python -m bootstrap.train --config configs/bootstrap_train_option3_gen4_candidate.yaml

echo "=== [$(date)] Stage 3/5: opening spot-check ==="
python -m tools.spotcheck_opening \
    checkpoints/bootstrap_batch2_residual_v2.pt \
    checkpoints/bootstrap_option3_gen1_candidate.pt \
    checkpoints/bootstrap_option3_gen2_candidate.pt \
    checkpoints/bootstrap_option3_gen3_candidate.pt \
    checkpoints/bootstrap_option3_gen4_candidate.pt

echo "=== [$(date)] Stage 4/5: eval vs gen3 ==="
python -m eval.promote --config configs/eval_option3_gen4_candidate_vs_gen3.yaml \
    --candidate checkpoints/bootstrap_option3_gen4_candidate.pt

echo "=== [$(date)] Stage 5/5: eval vs new baseline ==="
python -m eval.promote --config configs/eval_option3_gen4_candidate_vs_baseline.yaml \
    --candidate checkpoints/bootstrap_option3_gen4_candidate.pt

echo "=== [$(date)] Option 3 gen4 complete ==="
