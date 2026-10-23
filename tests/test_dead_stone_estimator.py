"""Tests for mcts/dead_stone_estimator.py, a port of igo-app's DeadStoneEstimatorTest.kt.

Run with: python -m unittest discover
"""

from __future__ import annotations

import dataclasses
import unittest

from engine.move import Move, Pass, Play
from engine.point import Point
from engine.position import Position
from engine.stone import Stone
from mcts.dead_stone_estimator import DeadStoneEstimator, PlayoutConfig
from mcts.policy_value_net import Evaluation


class _ScriptedNet:
    """Puts almost all its prior on `favored` wherever that's legal, and on a pass otherwise."""

    def __init__(self, favored: Move) -> None:
        self.favored = favored

    def evaluate(self, position: Position) -> Evaluation:
        legal = position.legal_moves()
        top = self.favored if self.favored in legal else Pass()
        rest = [m for m in legal if m != top]
        policy = {m: 0.01 / max(len(rest), 1) for m in rest}
        policy[top] = 0.99
        return Evaluation(policy=policy, value=0.0)


def _board(*rows: str) -> Position:
    symbols = {"X": Stone.BLACK, "O": Stone.WHITE, ".": Stone.EMPTY}
    return Position(board_size=len(rows), stones=[symbols[c] for c in "".join(rows)], to_play=Stone.BLACK)


# A lone black stone inside a white group with three eyes: black can only pass, and white
# captures at (2,2). Both players have already passed.
_DEAD_BLACK_STONE = dataclasses.replace(_board("O.OOO", "OOXOO", "OO.OO", "OOOOO", "OOO.O"), pass_count=2)
_CAPTURE = Play(Point(2, 2))
_DETERMINISTIC = PlayoutConfig(playouts=3, simulations_per_move=8, move_temperature=0.0)


class DeadStoneEstimatorTest(unittest.TestCase):
    def test_a_playout_plays_past_the_games_own_passes_and_ends_on_two_new_ones(self) -> None:
        end = DeadStoneEstimator(_ScriptedNet(_CAPTURE), _DETERMINISTIC).play_out(_DEAD_BLACK_STONE)
        self.assertEqual(2, end.pass_count)
        self.assertEqual(Stone.EMPTY, end.stone_at(Point(1, 2)))

    def test_stones_the_playouts_capture_are_proposed_dead(self) -> None:
        self.assertEqual({Point(1, 2)}, DeadStoneEstimator(_ScriptedNet(_CAPTURE), _DETERMINISTIC).estimate(_DEAD_BLACK_STONE))

    def test_when_nobody_captures_nothing_is_proposed_dead(self) -> None:
        self.assertEqual(set(), DeadStoneEstimator(_ScriptedNet(Pass()), _DETERMINISTIC).estimate(_DEAD_BLACK_STONE))

    def test_a_playout_stops_at_the_move_limit(self) -> None:
        # Black passes, white captures: with a limit of two moves the playout stops there.
        config = dataclasses.replace(_DETERMINISTIC, max_moves=2)
        end = DeadStoneEstimator(_ScriptedNet(_CAPTURE), config).play_out(_DEAD_BLACK_STONE)
        self.assertEqual(Stone.EMPTY, end.stone_at(Point(1, 2)))
        self.assertLess(end.pass_count, 2)


if __name__ == "__main__":
    unittest.main()
