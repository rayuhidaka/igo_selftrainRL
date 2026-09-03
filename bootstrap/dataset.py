"""Self-play example storage + a torch Dataset over it.

Produced by selfplay/generate.py, consumed by bootstrap/train.py's
training loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler


class SelfPlayExamples:
    """One self-play dataset: `board_planes` (NCHW, matching `RayZeroNet`), target policy
    distributions (dense, length `board_size**2 + 1`), target values (`z`, `+1`/`-1`/`0` from
    each recorded position's own player-to-move perspective), and target score margins
    (final area-score difference, normalized by `board_size**2` and also from each recorded
    position's own player-to-move perspective -- positive means that player ended up ahead).

    `score_margin_targets` trains `RayZeroNet`'s auxiliary score head (see its module
    docstring for why) -- a dataset saved before that head existed has no such column, so
    `load` fills zeros for it (a neutral "tied" placeholder, not a real target) rather than
    forcing every existing self-play batch to be regenerated.
    """

    def __init__(
        self,
        board_planes: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
        score_margin_targets: np.ndarray,
    ) -> None:
        assert (
            board_planes.shape[0]
            == policy_targets.shape[0]
            == value_targets.shape[0]
            == score_margin_targets.shape[0]
        ), "board_planes, policy_targets, value_targets, and score_margin_targets must all have the same example count"
        self.board_planes = board_planes
        self.policy_targets = policy_targets
        self.value_targets = value_targets
        self.score_margin_targets = score_margin_targets

    def __len__(self) -> int:
        return self.board_planes.shape[0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            board_planes=self.board_planes,
            policy_targets=self.policy_targets,
            value_targets=self.value_targets,
            score_margin_targets=self.score_margin_targets,
        )

    @staticmethod
    def load(path: Path) -> "SelfPlayExamples":
        data = np.load(path)
        if "score_margin_targets" in data:
            score_margin_targets = data["score_margin_targets"]
        else:
            score_margin_targets = np.zeros_like(data["value_targets"])
        return SelfPlayExamples(data["board_planes"], data["policy_targets"], data["value_targets"], score_margin_targets)


class SelfPlayDataset(Dataset):
    """A torch `Dataset` view over `SelfPlayExamples`, for `bootstrap/train.py`'s `DataLoader`."""

    def __init__(self, examples: SelfPlayExamples) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.examples.board_planes[index]),
            torch.from_numpy(self.examples.policy_targets[index]),
            torch.tensor(self.examples.value_targets[index], dtype=torch.float32),
            torch.tensor(self.examples.score_margin_targets[index], dtype=torch.float32),
        )


def build_training_loader(sources: list[tuple[Path, float]], batch_size: int) -> tuple[DataLoader, int]:
    """Builds a `DataLoader` mixing multiple `SelfPlayExamples` sources by weight,
    independent of each source's own size -- e.g. `[(broad.npz, 0.5), (new.npz, 0.5)]`
    samples roughly half of each batch from each source regardless of how many examples
    either one actually has.

    Exists for Phase 3's self-play fine-tuning (see docs/ROADMAP.md): fine-tuning on a new,
    small self-play batch in isolation let the network drift into degenerate behavior within
    a couple of generations (a real self-play collapse, not just noisy data) -- anchoring
    each fine-tune against a broader, known-good dataset (Phase 2's imitation-learning data)
    alongside the new batch prevents that drift, the standard fix real self-play pipelines
    use. A single `(path, 1.0)` source behaves like plain unweighted sampling.

    Returns `(loader, total_examples)` -- `total_examples` is every source's combined
    example count, for reporting only (one "epoch" here means that many *weighted-sampled*
    draws, not one full unweighted pass over each source, since a smaller source with a
    large weight will be resampled with repetition, and a larger source with a small weight
    will be undersampled).
    """
    datasets: list[SelfPlayDataset] = []
    per_example_weights: list[float] = []
    for path, weight in sources:
        dataset = SelfPlayDataset(SelfPlayExamples.load(path))
        datasets.append(dataset)
        per_example_weights.extend([weight / len(dataset)] * len(dataset))

    combined = ConcatDataset(datasets)
    sampler = WeightedRandomSampler(per_example_weights, num_samples=len(combined), replacement=True)
    loader = DataLoader(combined, batch_size=batch_size, sampler=sampler, drop_last=True)
    return loader, len(combined)
