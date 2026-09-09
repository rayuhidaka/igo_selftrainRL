#!/usr/bin/env bash
# Runs generation 11's full pipeline unattended: self-play (800 games, 4 workers) ->
# GPU fine-tune -> eval vs gen10 -> eval vs the pristine baseline. Stops after eval
# rather than auto-deciding promotion -- that's always been a human judgment call in this
# project's history (see docs/ROADMAP.md's gen6/gen8/gen9/gen10 entries).
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

echo "=== [$(date)] Stage 1/3: self-play (800 games, 4 workers) ==="
python -m selfplay.run_parallel --config configs/selfplay_self_play_gen11_no_pass_guard.yaml --workers 4

echo "=== [$(date)] Stage 2/3: fine-tune (GPU) ==="
python -m bootstrap.train --config configs/bootstrap_train_gen11_no_pass_guard_candidate.yaml

echo "=== [$(date)] Stage 3/3: eval vs gen10 ==="
python -m eval.promote --config configs/eval_gen11_no_pass_guard_candidate_vs_gen10.yaml \
    --candidate checkpoints/bootstrap_gen11_no_pass_guard_candidate.pt

echo "=== [$(date)] Stage 3/3: eval vs pristine baseline ==="
python -m eval.promote --config configs/eval_gen11_no_pass_guard_candidate_vs_residual.yaml \
    --candidate checkpoints/bootstrap_gen11_no_pass_guard_candidate.pt

echo "=== [$(date)] gen11 pipeline complete ==="
