"""Policy and value loss of checkpoints on a held-out self-play dataset.

python -m bootstrap.holdout_loss --data <dataset.npz> <checkpoint.pt> [...]

Same losses as bootstrap/train.py (soft-target cross-entropy, value MSE), no augmentation,
for checking a run against overfitting: a value loss that rises on held-out games while the
training loss falls means the net is memorizing game results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bootstrap.dataset import SelfPlayExamples
from bootstrap.inference import RayZeroPolicyValueNet


@torch.no_grad()
def losses(checkpoint: Path, examples: SelfPlayExamples, device: torch.device, batch: int = 1024) -> tuple[float, float]:
    model = RayZeroPolicyValueNet(checkpoint, examples.board_planes.shape[-1]).model.to(device).eval()
    policy_sum = value_sum = 0.0
    for start in range(0, len(examples), batch):
        planes = torch.from_numpy(examples.board_planes[start:start + batch]).to(device)
        policy_targets = torch.from_numpy(examples.policy_targets[start:start + batch]).to(device)
        value_targets = torch.from_numpy(examples.value_targets[start:start + batch]).to(device)
        policy, value, _, _ = model(planes)
        policy_sum += -(policy_targets * torch.log(policy + 1e-8)).sum().item()
        value_sum += ((value.squeeze(-1) - value_targets) ** 2).sum().item()
    return policy_sum / len(examples), value_sum / len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples = SelfPlayExamples.load(args.data)
    for checkpoint in args.checkpoints:
        policy, value = losses(checkpoint, examples, device)
        print(f"{checkpoint}: policy {policy:.4f}, value {value:.4f} ({len(examples)} positions)")


if __name__ == "__main__":
    main()
