#!/usr/bin/env bash
# Option 3, generation 12 wider-data-window experiment (see
# configs/bootstrap_train_option3_gen12_window_candidate.yaml). No new self-play: retrains from
# gen11 on the existing 1000 gen12 games plus a wider window of older generations, then runs
# the full promotion decision -- the parent match AND eval/gate.py's field gate.
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

CANDIDATE=checkpoints/bootstrap_option3_gen12_window_candidate.pt

echo "=== [$(date)] Stage 1/4: fine-tune (wider data window) ==="
python -m bootstrap.train --config configs/bootstrap_train_option3_gen12_window_candidate.yaml

echo "=== [$(date)] Stage 2/4: opening spot-check ==="
python -m tools.spotcheck_opening \
    checkpoints/bootstrap_option3_gen11_candidate.pt \
    checkpoints/bootstrap_option3_gen12_1000games_candidate.pt \
    "$CANDIDATE"

echo "=== [$(date)] Stage 3/4: parent match vs gen11 (80 games) ==="
python -m eval.promote --config configs/eval_option3_gen12_candidate_vs_gen11.yaml --candidate "$CANDIDATE"

echo "=== [$(date)] Stage 4/4: field gate (round-robin incl. candidate, separate output files) ==="
python -m eval.round_robin --config configs/eval_round_robin_option3.yaml --reuse-existing \
    --add-checkpoint "gen12w=$CANDIDATE" --out-prefix gate_gen12w_
python -m eval.gate --ratings eval/gate_gen12w_checkpoints_elo.json --candidate gen12w --parent gen11 || true

echo "=== [$(date)] gen12 wider-window experiment complete ==="
