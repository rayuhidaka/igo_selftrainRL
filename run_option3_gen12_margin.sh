#!/usr/bin/env bash
# Margin-aware training, stage 3 (see configs/selfplay_self_play_option3_gen12_margin.yaml).
# Plays 500 games from gen11 with the score bonus on (~8h), fine-tunes gen11 on them, then
# judges the candidate twice: the usual promotion decision (parent match plus eval/gate.py's
# field gate -- playing for margin mustn't cost strength) and the margin check
# (configs/eval_score_utility_gen12m.yaml, ~1h: flips, close games, margins).
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

CANDIDATE=checkpoints/bootstrap_option3_gen12_margin_candidate.pt

echo "=== [$(date)] Stage 1/6: self-play (500 games, score bonus 0.5, 4 workers) ==="
python -m selfplay.run_parallel --config configs/selfplay_self_play_option3_gen12_margin.yaml --workers 4

echo "=== [$(date)] Stage 2/6: fine-tune ==="
python -m bootstrap.train --config configs/bootstrap_train_option3_gen12_margin_candidate.yaml

echo "=== [$(date)] Stage 3/6: opening spot-check ==="
python -m tools.spotcheck_opening \
    checkpoints/bootstrap_option3_gen11_candidate.pt \
    "$CANDIDATE"

echo "=== [$(date)] Stage 4/6: parent match vs gen11 (80 games) ==="
python -m eval.promote --config configs/eval_option3_gen12_candidate_vs_gen11.yaml --candidate "$CANDIDATE" || true

echo "=== [$(date)] Stage 5/6: field gate (round-robin incl. candidate, separate output files) ==="
python -m eval.round_robin --config configs/eval_round_robin_option3.yaml --reuse-existing \
    --add-checkpoint "gen12m=$CANDIDATE" --out-prefix gate_gen12m_
python -m eval.gate --ratings eval/gate_gen12m_checkpoints_elo.json --candidate gen12m --parent gen11 || true

echo "=== [$(date)] Stage 6/6: margin check (flips, close games, margins) ==="
python -m eval.rule_comparison --config configs/eval_score_utility_gen12m.yaml --workers 4

echo "=== [$(date)] gen12 margin-aware run complete ==="
