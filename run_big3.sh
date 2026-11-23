#!/usr/bin/env bash
# big3 (2026-11-23): the chain's next step from big2f, same recipe (configs/*big3*).
# Bigger-net chain, generation 2 (docs/STRONGER_NET.md): big1's self-play by fastplay with
# cleanup-labelled ownership, a fine-tune, held-out losses, the parent match vs big1 and the
# field gate on the big chain's own field (configs/eval_round_robin_big.yaml).
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

CANDIDATE=checkpoints/big3_candidate.pt

echo "=== [$(date)] Stage 1/4: self-play (big2f, 2000 games, fastplay, cleanup labels, fpu 0.2) ==="
python -u -m fastplay.self_play --config configs/fastplay_big3.yaml

echo "=== [$(date)] Stage 2/4: fine-tune (20 min) ==="
python -m bootstrap.train --config configs/bootstrap_train_big3_candidate.yaml
python -m bootstrap.holdout_loss --data selfplay_games/fastplay_benchmark_gen11.npz \
    checkpoints/bootstrap_option3_gen11_candidate.pt checkpoints/big2f_candidate.pt "$CANDIDATE"

echo "=== [$(date)] Stage 3/4: parent match vs big2f (80 games) ==="
python -m eval.promote --config configs/eval_big3_candidate_vs_big2f.yaml --candidate "$CANDIDATE" || true

echo "=== [$(date)] Stage 4/4: field gate (big field incl. candidate, separate output files) ==="
python -m eval.round_robin --config configs/eval_round_robin_big.yaml --reuse-existing \
    --add-checkpoint "big3=$CANDIDATE" --out-prefix gate_big3_
python -m eval.gate --ratings eval/gate_big3_checkpoints_big_elo.json --candidate big3 --parent big2f || true

echo "=== [$(date)] big3 complete ==="
