#!/usr/bin/env bash
# Option 3, generation 12 stronger-search experiment (see
# configs/selfplay_self_play_option3_gen12_800sims.yaml). Plays 500 games from gen11 at 800
# simulations per move (~16h), trains with the wider data window, then runs
# the full promotion decision -- the parent match AND eval/gate.py's field gate.
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

CANDIDATE=checkpoints/bootstrap_option3_gen12_800sims_candidate.pt

echo "=== [$(date)] Stage 1/5: self-play (500 games at 800 simulations, 4 workers) ==="
python -m selfplay.run_parallel --config configs/selfplay_self_play_option3_gen12_800sims.yaml --workers 4

echo "=== [$(date)] Stage 2/5: fine-tune (wider data window) ==="
python -m bootstrap.train --config configs/bootstrap_train_option3_gen12_800sims_candidate.yaml

echo "=== [$(date)] Stage 3/5: opening spot-check ==="
python -m tools.spotcheck_opening \
    checkpoints/bootstrap_option3_gen11_candidate.pt \
    checkpoints/bootstrap_option3_gen12_1000games_candidate.pt \
    checkpoints/bootstrap_option3_gen12_window_candidate.pt \
    "$CANDIDATE"

echo "=== [$(date)] Stage 4/5: parent match vs gen11 (80 games) ==="
python -m eval.promote --config configs/eval_option3_gen12_candidate_vs_gen11.yaml --candidate "$CANDIDATE"

echo "=== [$(date)] Stage 5/5: field gate (round-robin incl. candidate, separate output files) ==="
python -m eval.round_robin --config configs/eval_round_robin_option3.yaml --reuse-existing \
    --add-checkpoint "gen12s=$CANDIDATE" --out-prefix gate_gen12s_
python -m eval.gate --ratings eval/gate_gen12s_checkpoints_elo.json --candidate gen12s --parent gen11 || true

echo "=== [$(date)] gen12 stronger-search experiment complete ==="
