#!/usr/bin/env bash
# Temperature experiment (2026-11-24, docs/STRONGER_NET.md "Temperature experiment"): two chain steps
# from big4 that differ only in temperature_drop_move -- big5 (30, every run so far) and big5t (12) --
# then big5t vs big5 head-to-head (120 games) and both in one field gate, parent big4.
# ~13-14 h: self-play ~4.3 h per arm, fine-tune 20 min each, match ~1.5 h, gate ~3 h.
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

for ARM in big5t big5; do
  echo "=== [$(date)] $ARM: self-play (big4, 2000 games, fastplay, fpu 0.2, configs/fastplay_$ARM.yaml) ==="
  python -u -m fastplay.self_play --config configs/fastplay_$ARM.yaml
  echo "=== [$(date)] $ARM: fine-tune (20 min) ==="
  python -m bootstrap.train --config configs/bootstrap_train_${ARM}_candidate.yaml
done

python -m bootstrap.holdout_loss --data selfplay_games/fastplay_benchmark_gen11.npz \
    checkpoints/big4_candidate.pt checkpoints/big5_candidate.pt checkpoints/big5t_candidate.pt

echo "=== [$(date)] big5t vs big5 (120 games) ==="
python -m eval.promote --config configs/eval_big5t_vs_big5.yaml --candidate checkpoints/big5t_candidate.pt || true

echo "=== [$(date)] field gate (both arms added to the big field, separate output files) ==="
python -m eval.round_robin --config configs/eval_round_robin_big.yaml --reuse-existing \
    --add-checkpoint "big5=checkpoints/big5_candidate.pt" --add-checkpoint "big5t=checkpoints/big5t_candidate.pt" \
    --out-prefix gate_big5_temperature_
python -m eval.gate --ratings eval/gate_big5_temperature_checkpoints_big_elo.json --candidate big5 --parent big4 || true
python -m eval.gate --ratings eval/gate_big5_temperature_checkpoints_big_elo.json --candidate big5t --parent big4 || true

echo "=== [$(date)] temperature experiment complete ==="
