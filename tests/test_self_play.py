"""Tests for selfplay/self_play.py. Small board, few simulations, few games -- exercises the
real PyTorch model + MCTS search end to end, so it's slower than the rest of the suite even
at this size; not meant to say anything about actual playing strength, just that the data
produced is well-formed (see tests/test_match.py for the same pattern).

Run with: python -m unittest discover
"""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bootstrap.checkpoint import CheckpointMetadata, save_checkpoint
from bootstrap.inference import RayZeroPolicyValueNet
from bootstrap.model import RayZeroNet
from engine.move import Pass, Play
from engine.point import Point
from mcts.mcts import Mcts, MctsConfig
from selfplay.self_play import play_one_game, visit_count_policy

_BOARD_SIZE = 5
_NUM_SIMULATIONS = 8


class VisitCountPolicyTest(unittest.TestCase):
    def test_normalizes_visit_counts_into_a_dense_probability_vector(self) -> None:
        move_visits = {Play(Point(0, 0)): 3, Play(Point(1, 1)): 1, Pass(): 0}
        policy = visit_count_policy(_BOARD_SIZE, move_visits)

        self.assertEqual(policy.shape, (_BOARD_SIZE * _BOARD_SIZE + 1,))
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=5)
        self.assertAlmostEqual(float(policy[Point(0, 0).to_index(_BOARD_SIZE)]), 0.75)
        self.assertAlmostEqual(float(policy[Point(1, 1).to_index(_BOARD_SIZE)]), 0.25)
        self.assertEqual(float(policy[_BOARD_SIZE * _BOARD_SIZE]), 0.0)

    def test_falls_back_to_uniform_when_every_visit_count_is_zero(self) -> None:
        move_visits = {Play(Point(0, 0)): 0, Pass(): 0}
        policy = visit_count_policy(_BOARD_SIZE, move_visits)

        self.assertAlmostEqual(float(policy.sum()), 1.0, places=5)
        self.assertAlmostEqual(float(policy[Point(0, 0).to_index(_BOARD_SIZE)]), 0.5)
        self.assertAlmostEqual(float(policy[_BOARD_SIZE * _BOARD_SIZE]), 0.5)


class PlayOneGameTest(unittest.TestCase):
    def test_produces_one_record_per_move_with_well_formed_policy_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "untrained.pt"
            torch.manual_seed(1)
            model = RayZeroNet(board_size=_BOARD_SIZE, channels=4, num_conv_layers=2)
            save_checkpoint(
                model,
                CheckpointMetadata(board_size=_BOARD_SIZE, channels=4, num_conv_layers=2),
                checkpoint_path,
            )
            net = RayZeroPolicyValueNet(checkpoint_path, _BOARD_SIZE)
            mcts = Mcts(net, MctsConfig(num_simulations=_NUM_SIMULATIONS))

            records, winner = play_one_game(
                mcts, _BOARD_SIZE, komi=7.5, temperature=1.0, max_moves=10, rng=random.Random(0)
            )

        self.assertGreater(len(records), 0)
        for planes, policy_target, to_play in records:
            self.assertEqual(planes.shape, (3, _BOARD_SIZE, _BOARD_SIZE))
            self.assertEqual(policy_target.shape, (_BOARD_SIZE * _BOARD_SIZE + 1,))
            self.assertAlmostEqual(float(policy_target.sum()), 1.0, places=4)
            self.assertIn(to_play.name, ("BLACK", "WHITE"))
        self.assertTrue(winner is None or winner.name in ("BLACK", "WHITE"))


if __name__ == "__main__":
    unittest.main()
