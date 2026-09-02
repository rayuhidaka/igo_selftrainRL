"""Port of igo-app/mcts/src/test/kotlin/com/igoapp/mcts/MctsTest.kt.

Run with: python -m unittest discover
"""

from __future__ import annotations

import unittest
from typing import Callable

from engine.move import Move, Pass, Play
from engine.point import Point
from engine.position import Position
from engine.stone import Stone
from mcts.mcts import Mcts, MctsConfig
from mcts.policy_value_net import Evaluation


class FakePolicyValueNet:
    """A configurable PolicyValueNet double; defaults to a value of 0 and a uniform policy over legal moves."""

    def __init__(
        self,
        value: Callable[[Position], float] = lambda position: 0.0,
        policy: Callable[[Position], dict[Move, float]] | None = None,
    ) -> None:
        self._value = value
        self._policy = policy or _uniform_policy

    def evaluate(self, position: Position) -> Evaluation:
        return Evaluation(policy=self._policy(position), value=self._value(position))


def _uniform_policy(position: Position) -> dict[Move, float]:
    legal_moves = position.legal_moves()
    return {move: 1.0 / len(legal_moves) for move in legal_moves}


def _skewed_policy(position: Position, favored_move: Move, favored_prior: float) -> dict[Move, float]:
    """A policy that heavily favors favored_move wherever it's legal, and is uniform otherwise."""
    legal_moves = position.legal_moves()
    if favored_move not in legal_moves:
        return _uniform_policy(position)
    remaining = [move for move in legal_moves if move != favored_move]
    remaining_prior = (1.0 - favored_prior) / len(remaining)
    result = {move: remaining_prior for move in remaining}
    result[favored_move] = favored_prior
    return result


class MctsTest(unittest.TestCase):
    def test_search_distributes_exactly_one_visit_per_simulation_among_the_roots_children(self) -> None:
        mcts = Mcts(FakePolicyValueNet(), MctsConfig(num_simulations=50))
        visits = mcts.search(Position.empty(3)).move_visits

        # The first simulation only expands the root without visiting a child; every
        # simulation after that visits exactly one child, regardless of tree shape.
        self.assertEqual(49, sum(visits.values()))

    def test_search_creates_exactly_one_child_per_legal_move_at_the_root(self) -> None:
        position = Position.empty(3)
        mcts = Mcts(FakePolicyValueNet(), MctsConfig(num_simulations=20))

        visits = mcts.search(position).move_visits

        self.assertEqual(set(position.legal_moves()), set(visits.keys()))

    def test_search_on_an_already_finished_game_returns_no_moves(self) -> None:
        finished = Position(
            board_size=3,
            stones=[Stone.EMPTY] * 9,
            to_play=Stone.BLACK,
            pass_count=2,
        )

        def must_not_be_called(position: Position) -> dict[Move, float]:
            raise AssertionError("should not evaluate a finished game")

        net_that_must_not_be_called = FakePolicyValueNet(policy=must_not_be_called)
        mcts = Mcts(net_that_must_not_be_called, MctsConfig(num_simulations=10))

        self.assertEqual({}, mcts.search(finished).move_visits)

    def test_search_on_an_already_finished_game_still_reports_the_deterministic_outcome_as_root_value(
        self,
    ) -> None:
        # Black outnumbers white 5 stones to 1 on this 3x3 board -- a clear win for black
        # under area scoring once the game is over (pass_count 2).
        finished = Position(
            board_size=3,
            stones=[
                Stone.BLACK, Stone.BLACK, Stone.BLACK,
                Stone.BLACK, Stone.BLACK, Stone.EMPTY,
                Stone.WHITE, Stone.EMPTY, Stone.EMPTY,
            ],
            to_play=Stone.BLACK,
            pass_count=2,
        )

        def must_not_be_called(position: Position) -> dict[Move, float]:
            raise AssertionError("should not evaluate a finished game")

        net_that_must_not_be_called = FakePolicyValueNet(policy=must_not_be_called)
        mcts = Mcts(net_that_must_not_be_called, MctsConfig(num_simulations=10, komi=0.5))

        self.assertEqual(1.0, mcts.search(finished).root_value)

    def test_search_allocates_more_visits_to_the_move_the_net_strongly_favors(self) -> None:
        center = Play(Point(1, 1))
        net = FakePolicyValueNet(policy=lambda position: _skewed_policy(position, center, favored_prior=0.9))
        mcts = Mcts(net, MctsConfig(num_simulations=200))

        visits = mcts.search(Position.empty(3)).move_visits

        other_visits = [v for move, v in visits.items() if move != center]
        self.assertGreater(visits[center], max(other_visits))
        self.assertEqual(center, mcts.select_move(Position.empty(3)))

    def test_a_move_that_immediately_wins_the_game_backs_up_a_favorable_value(self) -> None:
        # A 3x3 board where black outnumbers white 5 stones to 1, with the previous move
        # already a pass: black is winning under area scoring if the game ends now, so
        # passing again should be clearly preferred over an ordinary, non-ending move like
        # (1,2) (whose subtree the net evaluates as neutral, per FakePolicyValueNet's
        # default value of 0 everywhere).
        position = Position(
            board_size=3,
            stones=[
                Stone.BLACK, Stone.BLACK, Stone.BLACK,
                Stone.BLACK, Stone.BLACK, Stone.EMPTY,
                Stone.WHITE, Stone.EMPTY, Stone.EMPTY,
            ],
            to_play=Stone.BLACK,
            pass_count=1,
        )

        mcts = Mcts(FakePolicyValueNet(), MctsConfig(num_simulations=100, komi=0.5))
        visits = mcts.search(position).move_visits

        self.assertGreater(visits[Pass()], visits[Play(Point(1, 2))])
        self.assertEqual(Pass(), mcts.select_move(position))


if __name__ == "__main__":
    unittest.main()
