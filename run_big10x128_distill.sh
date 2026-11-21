#!/usr/bin/env bash
# Bigger net, step 3 of docs/STRONGER_NET.md: distill a fresh 10x128 net from the existing
# self-play games (plus 2000 new gen11 games from fastplay), then judge it like a generation:
# held-out losses, parent match vs gen11, field gate. Expected outcome: about gen11's
# strength; the gains have to come from self-play with the extra capacity afterwards.
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

PHASE1=checkpoints/big10x128_distill_phase1.pt
CANDIDATE=checkpoints/big10x128_distill_phase2.pt
HOLDOUT=selfplay_games/fastplay_benchmark_gen11.npz

echo "=== [$(date)] Stage 1/5: data (fastplay, 2000 gen11 games) ==="
# Run it here unless it's already running (tmux session "fastplay"); then wait for it.
if ! tmux has-session -t fastplay 2>/dev/null && [ ! -f selfplay_games/fastplay_gen11_2000.npz ]; then
    python -u -m fastplay.self_play --config configs/fastplay_gen11_2000.yaml
fi
while [ ! -f selfplay_games/fastplay_gen11_2000.npz ]; do sleep 60; done
sleep 30  # let the save finish

echo "=== [$(date)] Stage 2/5: distill, phase 1 (lr 1e-3, 60 min) ==="
python -m bootstrap.train --config configs/bootstrap_train_big10x128_distill.yaml
cp "$PHASE1" checkpoints/big10x128_distill_phase1_end.pt

echo "=== [$(date)] Stage 3/5: distill, phase 2 (lr 2e-4, 20 min) ==="
python -m bootstrap.train --config configs/bootstrap_train_big10x128_distill_phase2.yaml
python -m bootstrap.holdout_loss --data "$HOLDOUT" \
    checkpoints/bootstrap_option3_gen11_candidate.pt checkpoints/bootstrap_option3_gen12_candidate.pt \
    checkpoints/big10x128_distill_phase1_end.pt "$CANDIDATE"

echo "=== [$(date)] Stage 4/5: parent match vs gen11 (80 games) ==="
python -m eval.promote --config configs/eval_option3_gen12_candidate_vs_gen11.yaml --candidate "$CANDIDATE" || true

echo "=== [$(date)] Stage 5/5: field gate (round-robin incl. candidate, separate output files) ==="
python -m eval.round_robin --config configs/eval_round_robin_option3.yaml --reuse-existing \
    --add-checkpoint "big1=$CANDIDATE" --out-prefix gate_big1_
python -m eval.gate --ratings eval/gate_big1_checkpoints_elo.json --candidate big1 --parent gen11 || true

echo "=== [$(date)] 10x128 distillation complete ==="
