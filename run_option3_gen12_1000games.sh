#!/usr/bin/env bash
# Option 3, generation 12 retry at 1000 self-play games (500 new + the original gen12 500 --
# see configs/selfplay_self_play_option3_gen12_extra500.yaml for why). Same 5-stage shape
# as every generation's run script; ~8h, not ~16h, since only 500 games are new.
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

echo "=== [$(date)] Stage 1/5: self-play (500 more games, 4 workers) ==="
python -m selfplay.run_parallel --config configs/selfplay_self_play_option3_gen12_extra500.yaml --workers 4

echo "=== [$(date)] Stage 2/5: fine-tune on 1000 games (augment: true) ==="
python -m bootstrap.train --config configs/bootstrap_train_option3_gen12_1000games_candidate.yaml

echo "=== [$(date)] Stage 3/5: opening spot-check ==="
python -m tools.spotcheck_opening \
    checkpoints/bootstrap_option3_gen10_candidate.pt \
    checkpoints/bootstrap_option3_gen11_candidate.pt \
    checkpoints/bootstrap_option3_gen12_candidate.pt \
    checkpoints/bootstrap_option3_gen12_1000games_candidate.pt

echo "=== [$(date)] Stage 4/5: eval vs gen11 (80 games) ==="
python -m eval.promote --config configs/eval_option3_gen12_candidate_vs_gen11.yaml \
    --candidate checkpoints/bootstrap_option3_gen12_1000games_candidate.pt

echo "=== [$(date)] Stage 5/5: eval vs new baseline ==="
python -m eval.promote --config configs/eval_option3_gen12_candidate_vs_baseline.yaml \
    --candidate checkpoints/bootstrap_option3_gen12_1000games_candidate.pt

echo "=== [$(date)] Option 3 gen12 (1000 games) complete ==="
