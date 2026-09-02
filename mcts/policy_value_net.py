"""Port of igo-app/mcts/PolicyValueNet.kt -- see mcts/README.md."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from engine.move import Move
from engine.position import Position


@dataclass(frozen=True)
class Evaluation:
    """A policy/value net's evaluation of one `Position`: a prior probability for every legal
    move (summing to ~1) plus an expected-outcome estimate.

    `value` is from the perspective of the position's player to move: `+1` is a certain win,
    `-1` a certain loss, `0` even.
    """

    policy: dict[Move, float]
    value: float


class PolicyValueNet(Protocol):
    """Evaluates Go positions with a policy/value net, without committing to any particular
    net architecture or runtime -- see igo-app/docs/MODEL_CONTRACT.md. `mcts.Mcts` drives
    search against this interface alone, so the same search code works for any net that
    implements it (`bootstrap/inference.py`'s `RayZeroPolicyValueNet` today).
    """

    def evaluate(self, position: Position) -> Evaluation: ...
