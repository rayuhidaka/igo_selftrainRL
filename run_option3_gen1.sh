#!/usr/bin/env bash
# Option 3, generation 1 (see docs/ROADMAP.md's "weak opening moves" item): the actual
# restart of the self-play chain, from the new-architecture baseline
# (bootstrap_batch2_residual_v2.pt), augmentation + every self-play-stability fix on from
# the start. num_games=100 mirrors the original chain's own gen1 -- escalate reactively for
# later generations only if an actual plateau is observed.
#
# Self-play (Stage 1) takes real wall-clock time (100 games at num_simulations=400) --
# run this in your own detached WSL tmux session per this project's established practice
# (Claude-Code-launched background jobs risk the host's low-memory guard killing them for
# reasons unrelated to actual usage), not directly through Claude Code.
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

echo "=== [$(date)] Stage 1/4: self-play (100 games, 4 workers) ==="
python -m selfplay.run_parallel --config configs/selfplay_self_play_option3_gen1.yaml --workers 4

echo "=== [$(date)] Stage 2/4: fine-tune (augment: true) ==="
python -m bootstrap.train --config configs/bootstrap_train_option3_gen1_candidate.yaml

echo "=== [$(date)] Stage 3/4: opening spot-check ==="
python -m tools.spotcheck_opening \
    checkpoints/bootstrap_batch2_residual_v2.pt \
    checkpoints/bootstrap_option3_gen1_candidate.pt

echo "=== [$(date)] Stage 4/4: eval vs new baseline ==="
python -m eval.promote --config configs/eval_option3_gen1_candidate_vs_baseline.yaml \
    --candidate checkpoints/bootstrap_option3_gen1_candidate.pt

echo "=== [$(date)] Option 3 gen1 complete ==="
