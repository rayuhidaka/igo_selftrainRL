#!/usr/bin/env bash
# Option 3, generation 2 -- gen1 showed a genuinely different, more diverse opening
# policy (not a plateau), so this continues at the same 100-game recipe rather than
# escalating. See docs/ROADMAP.md's "weak opening moves" item.
#
# Self-play (Stage 1) takes real wall-clock time -- run this in your own detached WSL
# tmux session per this project's established practice, not directly through Claude Code.
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

echo "=== [$(date)] Stage 1/5: self-play (100 games, 4 workers) ==="
python -m selfplay.run_parallel --config configs/selfplay_self_play_option3_gen2.yaml --workers 4

echo "=== [$(date)] Stage 2/5: fine-tune (augment: true) ==="
python -m bootstrap.train --config configs/bootstrap_train_option3_gen2_candidate.yaml

echo "=== [$(date)] Stage 3/5: opening spot-check ==="
python -m tools.spotcheck_opening \
    checkpoints/bootstrap_batch2_residual_v2.pt \
    checkpoints/bootstrap_option3_gen1_candidate.pt \
    checkpoints/bootstrap_option3_gen2_candidate.pt

echo "=== [$(date)] Stage 4/5: eval vs gen1 ==="
python -m eval.promote --config configs/eval_option3_gen2_candidate_vs_gen1.yaml \
    --candidate checkpoints/bootstrap_option3_gen2_candidate.pt

echo "=== [$(date)] Stage 5/5: eval vs new baseline ==="
python -m eval.promote --config configs/eval_option3_gen2_candidate_vs_baseline.yaml \
    --candidate checkpoints/bootstrap_option3_gen2_candidate.pt

echo "=== [$(date)] Option 3 gen2 complete ==="
