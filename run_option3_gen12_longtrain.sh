#!/usr/bin/env bash
# Option 3, generation 12 longer-training experiment. gen12 missed promotion (+42.6 vs gen11,
# gate 50), and every Option 3 fine-tune so far stopped at max_train_seconds: 60 -- only ~53%
# of one epoch. Re-trains gen12 from the same parent (gen11) and the same self-play data for
# 1 and 2 full epochs, then evals both against gen11 in parallel. No new self-play.
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/igo-training/bin/activate

# Don't compete for CPU with the original gen12 run's final eval, if it's still going.
while pgrep -f "eval.promote" >/dev/null; do sleep 30; done

for variant in 1epoch 2epoch; do
    echo "=== [$(date)] Fine-tune: $variant ==="
    python -m bootstrap.train --config configs/bootstrap_train_option3_gen12_${variant}_candidate.yaml
done

echo "=== [$(date)] Opening spot-check ==="
python -m tools.spotcheck_opening \
    checkpoints/bootstrap_option3_gen11_candidate.pt \
    checkpoints/bootstrap_option3_gen12_candidate.pt \
    checkpoints/bootstrap_option3_gen12_1epoch_candidate.pt \
    checkpoints/bootstrap_option3_gen12_2epoch_candidate.pt

echo "=== [$(date)] Eval vs gen11 (80 games each, both variants in parallel) ==="
for variant in 1epoch 2epoch; do
    python -m eval.promote --config configs/eval_option3_gen12_candidate_vs_gen11.yaml \
        --candidate checkpoints/bootstrap_option3_gen12_${variant}_candidate.pt \
        > "/tmp/gen12_${variant}_eval.log" 2>&1 &
done
wait
for variant in 1epoch 2epoch; do
    echo "--- $variant ---"
    grep -E "Candidate rating|PROMOTE|Do not promote|Traceback|Error" "/tmp/gen12_${variant}_eval.log"
done

echo "=== [$(date)] gen12 longer-training experiment complete ==="
