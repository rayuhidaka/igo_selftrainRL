"""Checkpoint save/load with architecture + provenance metadata.

Every checkpoint from here on is versioned per CLAUDE.md's convention
("architecture config, training data source, random seed") instead of a
bare `state_dict` -- so a checkpoint self-describes the `RayZeroNet` shape
it needs (`channels`, `num_conv_layers`) rather than requiring the caller
to already know it. That requirement was a real bug: widening the net
(see docs/ROADMAP.md's Phase 2) broke loading every checkpoint saved
before the change, since `RayZeroPolicyValueNet` always built *today's*
architecture before `load_state_dict`.

Checkpoints saved before this module existed (a bare `state_dict`, no
wrapper) still load via `load_checkpoint` -- just with `metadata=None`,
since their architecture isn't recorded anywhere and has to be supplied
by the caller the same way it always did.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

from bootstrap.model import RayZeroNet

_STATE_DICT_KEY = "model_state_dict"
_METADATA_KEY = "metadata"


@dataclass
class CheckpointMetadata:
    """Everything needed to reconstruct the exact `RayZeroNet` a checkpoint was trained
    with, plus where it came from. `elo` is `None` until an `eval/` run actually estimates
    one -- it's not knowable at training time.
    """

    board_size: int
    channels: int
    num_conv_layers: int
    num_residual_blocks: int = 0
    data_source: Optional[str] = None
    seed: Optional[int] = None
    elo: Optional[float] = None


def save_checkpoint(model: RayZeroNet, metadata: CheckpointMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({_STATE_DICT_KEY: model.state_dict(), _METADATA_KEY: asdict(metadata)}, path)


def load_checkpoint(path: Path) -> tuple[dict, Optional[CheckpointMetadata]]:
    """Returns `(state_dict, metadata)`. `metadata` is `None` for a legacy checkpoint (a bare
    `state_dict`, saved before this module existed) -- the caller must already know its
    architecture in that case, same as before this module existed.
    """
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and _STATE_DICT_KEY in raw and _METADATA_KEY in raw:
        return raw[_STATE_DICT_KEY], CheckpointMetadata(**raw[_METADATA_KEY])
    return raw, None


def build_model(
    metadata: Optional[CheckpointMetadata],
    board_size: int,
    channels: Optional[int] = None,
    num_conv_layers: Optional[int] = None,
    num_residual_blocks: Optional[int] = None,
) -> RayZeroNet:
    """Constructs the `RayZeroNet` a checkpoint needs: an explicit `channels`/
    `num_conv_layers`/`num_residual_blocks` always wins (for a legacy checkpoint, or to
    deliberately override), otherwise `metadata`'s own values, otherwise `RayZeroNet`'s
    current defaults.
    """

    def resolve(explicit: Optional[int], metadata_field: str) -> Optional[int]:
        if explicit is not None:
            return explicit
        return getattr(metadata, metadata_field) if metadata else None

    kwargs = {}
    for arg, metadata_field in [
        (channels, "channels"),
        (num_conv_layers, "num_conv_layers"),
        (num_residual_blocks, "num_residual_blocks"),
    ]:
        resolved = resolve(arg, metadata_field)
        if resolved is not None:
            kwargs[metadata_field] = resolved
    return RayZeroNet(board_size=board_size, **kwargs)
