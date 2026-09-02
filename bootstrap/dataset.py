"""Self-play example storage + a torch Dataset over it.

Produced by selfplay/generate.py, consumed by bootstrap/train.py's
training loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SelfPlayExamples:
    """One self-play dataset: `board_planes` (NCHW, matching `RayZeroNet`), target policy
    distributions (dense, length `board_size**2 + 1`), and target values (`z`, `+1`/`-1`/`0`
    from each recorded position's own player-to-move perspective).
    """

    def __init__(self, board_planes: np.ndarray, policy_targets: np.ndarray, value_targets: np.ndarray) -> None:
        assert board_planes.shape[0] == policy_targets.shape[0] == value_targets.shape[0], (
            "board_planes, policy_targets, and value_targets must all have the same example count"
        )
        self.board_planes = board_planes
        self.policy_targets = policy_targets
        self.value_targets = value_targets

    def __len__(self) -> int:
        return self.board_planes.shape[0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            board_planes=self.board_planes,
            policy_targets=self.policy_targets,
            value_targets=self.value_targets,
        )

    @staticmethod
    def load(path: Path) -> "SelfPlayExamples":
        data = np.load(path)
        return SelfPlayExamples(data["board_planes"], data["policy_targets"], data["value_targets"])


class SelfPlayDataset(Dataset):
    """A torch `Dataset` view over `SelfPlayExamples`, for `bootstrap/train.py`'s `DataLoader`."""

    def __init__(self, examples: SelfPlayExamples) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.examples.board_planes[index]),
            torch.from_numpy(self.examples.policy_targets[index]),
            torch.tensor(self.examples.value_targets[index], dtype=torch.float32),
        )
