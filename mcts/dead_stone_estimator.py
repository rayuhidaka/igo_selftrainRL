"""Port of igo-app/mcts/DeadStoneEstimator.kt: proposes a finished game's dead stones by playing
the position out a few times with a quick search and seeing which stones end up captured.

Known blind spot (found 2026-10-23, igo-app's scoring overhaul stage C): every Ray-zeroGo net
was trained only to win, never to win by more, so a side far ahead doesn't bother capturing
dead stones -- in its playouts or in its own games -- and those stones aren't proposed. The
nets' ownership heads, trained on the same endings, share it. Close games, where it matters for
the result, are fine. Margin-aware training is the fix (see docs/ROADMAP.md).
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Optional

from engine.life_and_death import estimate_dead_stones
from engine.move import Move, Pass
from engine.point import Point
from engine.position import Position
from mcts.mcts import Mcts, MctsConfig
from mcts.policy_value_net import PolicyValueNet


@dataclass(frozen=True)
class PlayoutConfig:
    """Settings for `DeadStoneEstimator` -- the same defaults as `DeadStoneEstimator.kt`'s."""

    playouts: int = 8
    simulations_per_move: int = 16
    max_moves: Optional[int] = None
    """`None` means a third of the board's points."""
    move_temperature: float = 0.5
    komi: float = 7.5


class DeadStoneEstimator:
    """Proposes a finished position's dead stones by playing it out with `net` (see the module
    docstring). Port of `igo-app/mcts/DeadStoneEstimator.kt`."""

    def __init__(self, net: PolicyValueNet, config: PlayoutConfig = PlayoutConfig(), rng: Optional[random.Random] = None):
        self._mcts = Mcts(net, MctsConfig(num_simulations=config.simulations_per_move, komi=config.komi))
        self._config = config
        self._rng = rng or random.Random(0)

    def estimate(self, position: Position) -> set[Point]:
        """The stones of `position` the playouts say are dead."""
        return estimate_dead_stones(position, [self.play_out(position) for _ in range(self._config.playouts)])

    def play_out(self, position: Position) -> Position:
        """Plays `position` on (starting with its pass count at zero) until two passes or the
        move cap, and returns where it ends."""
        max_moves = self._config.max_moves or position.board_size * position.board_size // 3
        current = dataclasses.replace(position, pass_count=0)
        moves = 0
        while current.pass_count < 2 and moves < max_moves:
            current = current.play(self._choose(self._mcts.search(current).move_visits))
            moves += 1
        return current

    def _choose(self, move_visits: dict[Move, int]) -> Move:
        """`Mcts.kt`'s `moveTemperature` choice: pass when the most-visited move is a pass,
        otherwise sample a placement with weight `visits ** (1 / temperature)`."""
        if not move_visits:
            return Pass()
        best = max(move_visits.items(), key=lambda item: item[1])[0]
        if isinstance(best, Pass) or self._config.move_temperature <= 0:
            return best
        placements = [(m, v ** (1.0 / self._config.move_temperature)) for m, v in move_visits.items() if not isinstance(m, Pass)]
        total = sum(weight for _, weight in placements)
        if total <= 0:
            return best
        remaining = self._rng.random() * total
        for move, weight in placements:
            remaining -= weight
            if remaining < 0:
                return move
        return placements[-1][0]
