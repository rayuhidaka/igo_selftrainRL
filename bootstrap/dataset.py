"""Self-play example storage + a torch Dataset over it.

Produced by selfplay/generate.py, consumed by bootstrap/train.py's
training loop.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

# The board's 8-fold dihedral symmetry group (4 rotations x optional mirror), as
# (rot90 k, mirror) pairs. Applying a random one of these per training example is standard
# AlphaZero/AlphaGo Zero practice -- this pipeline never did it (see
# igo-app/docs/ROADMAP.md's "weak opening moves" item), which plausibly explains why
# self-play's actual opening move differs, arbitrarily, generation to generation (gen1-11
# variously pick (0,0), (2,7), (8,7), (8,8) -- all symmetry-equivalent corner/near-corner
# points, never the same one twice): each generation's small self-play sample happens to
# see one of the 8 equivalent corner opens slightly more than the others by chance, and an
# unaugmented net overfits to that specific one rather than learning "corner opens" as a
# rotation/reflection-invariant class.
_DIHEDRAL_TRANSFORMS = [(k, mirror) for k in range(4) for mirror in (False, True)]


def _rotate_flip(array: np.ndarray, k: int, mirror: bool) -> np.ndarray:
    """Rotates `array` (spatial axes -2/-1) 90*`k` degrees and optionally mirrors it -- the
    single primitive `_apply_dihedral_transform` and `_apply_dihedral_transform_to_ownership`
    both build on, so the exact same `(k, mirror)` pair always produces the exact same
    physical transform regardless of which array (board planes, a policy grid, an ownership
    plane) it's applied to.
    """
    rotated = np.rot90(array, k, axes=(-2, -1))
    if mirror:
        rotated = np.flip(rotated, axis=-1)
    return np.ascontiguousarray(rotated)


def _apply_dihedral_transform(
    board_planes: np.ndarray, policy_target: np.ndarray, board_size: int, transform_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Applies one of the 8 board symmetries (`_DIHEDRAL_TRANSFORMS[transform_index]`) to both
    `board_planes` (`[3, board_size, board_size]`, spatial axes -2/-1 per
    `bootstrap/inference.py.encode_planes`) and `policy_target` (`[board_size**2 + 1]`, row-major
    per `docs/MODEL_CONTRACT.md` plus a trailing Pass entry) *consistently* -- the whole point is
    that a stone's new position after the transform and the policy target's peak at that same
    point must still agree, or this would train the wrong association, actively worse than no
    augmentation at all. `value_targets`/`score_margin_targets` are orientation-invariant scalars,
    not passed here -- nothing to transform. `ownership_targets` *are* spatial and need the same
    treatment -- see `_apply_dihedral_transform_to_ownership`, kept as a separate function (not a
    3rd return value here) so this function's existing signature/tests stay unaffected.
    """
    k, mirror = _DIHEDRAL_TRANSFORMS[transform_index]
    board = _rotate_flip(board_planes, k, mirror)
    board_points, pass_prob = policy_target[:-1], policy_target[-1:]
    grid = _rotate_flip(board_points.reshape(board_size, board_size), k, mirror)
    return board, np.concatenate([grid.reshape(-1), pass_prob])


def _apply_dihedral_transform_to_ownership(
    ownership_target: np.ndarray, board_size: int, transform_index: int
) -> np.ndarray:
    """Applies the same dihedral transform as `_apply_dihedral_transform`
    (`_DIHEDRAL_TRANSFORMS[transform_index]`, via the shared `_rotate_flip` primitive) to a
    flat `[board_size**2]` ownership plane (see `engine/scoring.py`'s `ownership_plane`),
    reshaping to `[board_size, board_size]` and back. Callers must pass the *same*
    `transform_index` used for that example's `board_planes`/`policy_target` -- a mismatched
    index would train the ownership head on the wrong orientation, same failure mode
    `_apply_dihedral_transform`'s own docstring warns about.
    """
    k, mirror = _DIHEDRAL_TRANSFORMS[transform_index]
    grid = _rotate_flip(ownership_target.reshape(board_size, board_size), k, mirror)
    return grid.reshape(-1)


class SelfPlayExamples:
    """One self-play dataset: `board_planes` (NCHW, matching `RayZeroNet`), target policy
    distributions (dense, length `board_size**2 + 1`), target values (`z`, `+1`/`-1`/`0` from
    each recorded position's own player-to-move perspective), target score margins
    (final area-score difference, normalized by `board_size**2` and also from each recorded
    position's own player-to-move perspective -- positive means that player ended up ahead),
    and target ownership planes (flat `[board_size**2]`, `+1`/`-1`/`0` per point from that
    same perspective -- see `engine/scoring.py`'s `ownership_plane`).

    `score_margin_targets`/`ownership_targets` train `RayZeroNet`'s auxiliary score/ownership
    heads (see its module docstring for why) -- a dataset saved before either head existed
    has no such column, so `load` fills zeros for it (a neutral "tied"/no-owner placeholder,
    not a real target) rather than forcing every existing self-play batch to be regenerated.
    """

    def __init__(
        self,
        board_planes: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
        score_margin_targets: np.ndarray,
        ownership_targets: np.ndarray,
    ) -> None:
        assert (
            board_planes.shape[0]
            == policy_targets.shape[0]
            == value_targets.shape[0]
            == score_margin_targets.shape[0]
            == ownership_targets.shape[0]
        ), (
            "board_planes, policy_targets, value_targets, score_margin_targets, and "
            "ownership_targets must all have the same example count"
        )
        self.board_planes = board_planes
        self.policy_targets = policy_targets
        self.value_targets = value_targets
        self.score_margin_targets = score_margin_targets
        self.ownership_targets = ownership_targets

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
            ownership_targets=self.ownership_targets,
        )

    @staticmethod
    def load(path: Path) -> "SelfPlayExamples":
        data = np.load(path)
        if "score_margin_targets" in data:
            score_margin_targets = data["score_margin_targets"]
        else:
            score_margin_targets = np.zeros_like(data["value_targets"])
        if "ownership_targets" in data:
            ownership_targets = data["ownership_targets"]
        else:
            board_size = data["board_planes"].shape[-1]
            ownership_targets = np.zeros((data["board_planes"].shape[0], board_size * board_size), dtype=np.float32)
        return SelfPlayExamples(
            data["board_planes"], data["policy_targets"], data["value_targets"], score_margin_targets, ownership_targets
        )


class SelfPlayDataset(Dataset):
    """A torch `Dataset` view over `SelfPlayExamples`, for `bootstrap/train.py`'s `DataLoader`.

    `augment` (default `False`, preserving every existing config's behavior unchanged): when
    `True`, each `__getitem__` applies a freshly-random one of the board's 8 dihedral symmetries
    (see `_apply_dihedral_transform`) to that example's `board_planes`/`policy_targets` -- so the
    same stored example teaches a different, symmetry-equivalent orientation practically every
    epoch, without needing 8x the stored data or new self-play.
    """

    def __init__(self, examples: SelfPlayExamples, augment: bool = False) -> None:
        self.examples = examples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        board_planes = self.examples.board_planes[index]
        policy_target = self.examples.policy_targets[index]
        ownership_target = self.examples.ownership_targets[index]
        if self.augment:
            board_size = board_planes.shape[-1]
            # One shared transform_index for board_planes/policy_target *and*
            # ownership_target -- a mismatched pair would train the wrong orientation for
            # either target (see _apply_dihedral_transform_to_ownership's docstring).
            transform_index = random.randrange(len(_DIHEDRAL_TRANSFORMS))
            board_planes, policy_target = _apply_dihedral_transform(
                board_planes, policy_target, board_size, transform_index
            )
            ownership_target = _apply_dihedral_transform_to_ownership(ownership_target, board_size, transform_index)
        return (
            torch.from_numpy(board_planes),
            torch.from_numpy(policy_target),
            torch.tensor(self.examples.value_targets[index], dtype=torch.float32),
            torch.tensor(self.examples.score_margin_targets[index], dtype=torch.float32),
            torch.from_numpy(ownership_target),
        )


def build_training_loader(
    sources: list[tuple[Path, float]], batch_size: int, augment: bool = False
) -> tuple[DataLoader, int]:
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

    `augment` (default `False`) is passed straight through to each source's own
    [SelfPlayDataset] -- see its docstring.
    """
    datasets: list[SelfPlayDataset] = []
    per_example_weights: list[float] = []
    for path, weight in sources:
        dataset = SelfPlayDataset(SelfPlayExamples.load(path), augment=augment)
        datasets.append(dataset)
        per_example_weights.extend([weight / len(dataset)] * len(dataset))

    combined = ConcatDataset(datasets)
    sampler = WeightedRandomSampler(per_example_weights, num_samples=len(combined), replacement=True)
    loader = DataLoader(combined, batch_size=batch_size, sampler=sampler, drop_last=True)
    return loader, len(combined)
