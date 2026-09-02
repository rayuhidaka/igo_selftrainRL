"""Wraps a RayZeroNet checkpoint as an mcts.PolicyValueNet, for eval/match.py
(and, later, bootstrap/'s and selfplay/'s own self-play loops) to search
with. Mirrors igo-app/inference/TfLitePolicyValueNet.kt's encode/decode
logic -- see igo-app/docs/MODEL_CONTRACT.md for the contract both conform
to, and bootstrap/model.py's docstring for why RayZeroNet's own forward()
takes NCHW input while this module still encodes NHWC-shaped planes
conceptually (channel order, not tensor layout, is what matters here).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from bootstrap.model import RayZeroNet
from engine.move import Move, Pass, Play
from engine.point import Point
from engine.position import Position
from mcts.policy_value_net import Evaluation


class RayZeroPolicyValueNet:
    """A `PolicyValueNet` (see `mcts/policy_value_net.py`) backed by a `RayZeroNet` checkpoint."""

    def __init__(self, checkpoint: Optional[Path], board_size: int) -> None:
        self.board_size = board_size
        self.model = RayZeroNet(board_size=board_size)
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, position: Position) -> Evaluation:
        board_planes = encode_planes(position).unsqueeze(0)  # add the batch dim
        policy_out, value_out = self.model(board_planes)
        return Evaluation(
            policy=decode_policy(position, policy_out[0].tolist()),
            value=float(value_out[0, 0]),
        )


def encode_planes(position: Position) -> torch.Tensor:
    """Encodes `position` as the `[3, board_size, board_size]` (own, opponent, empty) tensor
    `RayZeroNet` expects, relative to `position.to_play` -- not a fixed color, so the same
    weights work playing either side.
    """
    size = position.board_size
    planes = torch.zeros(3, size, size, dtype=torch.float32)
    opponent = position.to_play.opponent()
    for row in range(size):
        for col in range(size):
            stone = position.stone_at(Point(row, col))
            if stone == position.to_play:
                planes[0, row, col] = 1.0
            elif stone == opponent:
                planes[1, row, col] = 1.0
            else:
                planes[2, row, col] = 1.0
    return planes


def decode_policy(position: Position, raw_policy: list[float]) -> dict[Move, float]:
    """Decodes a raw `policy` output into a probability per legal move in `position`, per the
    row-major-plus-pass index convention (see igo-app/docs/MODEL_CONTRACT.md).
    """
    size = position.board_size
    result: dict[Move, float] = {}
    for move in position.legal_moves():
        if isinstance(move, Pass):
            result[move] = raw_policy[size * size]
        else:
            assert isinstance(move, Play)
            result[move] = raw_policy[move.point.to_index(size)]
    return result
