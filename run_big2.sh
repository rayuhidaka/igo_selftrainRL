#!/usr/bin/env bash
# Bigger-net chain, generation 2 (docs/STRONGER_NET.md): big1's self-play by fastplay with
# cleanup-labelled ownership, a fine-tune, held-out losses, the parent match vs big1 and the
# field gate on the big chain's own field (configs/eval_round_robin_big.yaml).
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

CANDIDATE=checkpoints/big2_candidate.pt

echo "=== [$(date)] Stage 1/4: self-play (big1, 2000 games, fastplay, cleanup labels) ==="
python -u -m fastplay.self_play --config configs/fastplay_big2.yaml

echo "=== [$(date)] Stage 2/4: fine-tune (20 min) ==="
python -m bootstrap.train --config configs/bootstrap_train_big2_candidate.yaml
python -m bootstrap.holdout_loss --data selfplay_games/fastplay_benchmark_gen11.npz \
    checkpoints/bootstrap_option3_gen11_candidate.pt checkpoints/big10x128_distill_phase2.pt "$CANDIDATE"

echo "=== [$(date)] Stage 3/4: parent match vs big1 (80 games) ==="
python -m eval.promote --config configs/eval_big2_candidate_vs_big1.yaml --candidate "$CANDIDATE" || true

echo "=== [$(date)] Stage 4/4: field gate (big field incl. candidate, separate output files) ==="
python -m eval.round_robin --config configs/eval_round_robin_big.yaml --reuse-existing \
    --add-checkpoint "big2=$CANDIDATE" --out-prefix gate_big2_
python -m eval.gate --ratings eval/gate_big2_checkpoints_big_elo.json --candidate big2 --parent big1 || true

echo "=== [$(date)] big2 complete ==="
