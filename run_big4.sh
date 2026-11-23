#!/usr/bin/env bash
# big4 (2026-11-23): the chain's next step from big3, same recipe (configs/*big4*).
# Bigger-net chain, generation 2 (docs/STRONGER_NET.md): big1's self-play by fastplay with
# cleanup-labelled ownership, a fine-tune, held-out losses, the parent match vs big1 and the
# field gate on the big chain's own field (configs/eval_round_robin_big.yaml).
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

CANDIDATE=checkpoints/big4_candidate.pt

echo "=== [$(date)] Stage 1/4: self-play (big3, 2000 games, fastplay, cleanup labels, fpu 0.2) ==="
python -u -m fastplay.self_play --config configs/fastplay_big4.yaml

echo "=== [$(date)] Stage 2/4: fine-tune (20 min) ==="
python -m bootstrap.train --config configs/bootstrap_train_big4_candidate.yaml
python -m bootstrap.holdout_loss --data selfplay_games/fastplay_benchmark_gen11.npz \
    checkpoints/bootstrap_option3_gen11_candidate.pt checkpoints/big3_candidate.pt "$CANDIDATE"

echo "=== [$(date)] Stage 3/4: parent match vs big3 (80 games) ==="
python -m eval.promote --config configs/eval_big4_candidate_vs_big3.yaml --candidate "$CANDIDATE" || true

echo "=== [$(date)] Stage 4/4: field gate (big field incl. candidate, separate output files) ==="
python -m eval.round_robin --config configs/eval_round_robin_big.yaml --reuse-existing \
    --add-checkpoint "big4=$CANDIDATE" --out-prefix gate_big4_
python -m eval.gate --ratings eval/gate_big4_checkpoints_big_elo.json --candidate big4 --parent big3 || true

echo "=== [$(date)] big4 complete ==="
