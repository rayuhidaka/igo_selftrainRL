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
from selfplay.self_play import has_suspicious_mid_game_pass, play_one_game, visit_count_policy

_BOARD_SIZE = 5
_NUM_SIMULATIONS = 8
_POLICY_SIZE = _BOARD_SIZE * _BOARD_SIZE + 1


def _record_with_pass_weight(pass_weight: float) -> tuple[None, np.ndarray, None]:
    policy_target = np.zeros(_POLICY_SIZE, dtype=np.float32)
    policy_target[-1] = pass_weight
    policy_target[0] = 1.0 - pass_weight
    return (None, policy_target, None)


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


class HasSuspiciousMidGamePassTest(unittest.TestCase):
    def test_false_when_every_non_terminal_position_has_low_pass_weight(self) -> None:
        records = [_record_with_pass_weight(0.01) for _ in range(10)]
        self.assertFalse(has_suspicious_mid_game_pass(records, threshold=0.5))

    def test_true_when_a_non_terminal_position_exceeds_the_threshold(self) -> None:
        records = [_record_with_pass_weight(0.01) for _ in range(5)]
        records.insert(2, _record_with_pass_weight(0.9))  # well before the final two
        self.assertTrue(has_suspicious_mid_game_pass(records, threshold=0.5))

    def test_ignores_the_final_two_positions_even_with_high_pass_weight(self) -> None:
        # Legitimate: these precede the game's own closing double-pass.
        records = [_record_with_pass_weight(0.01) for _ in range(5)]
        records[-1] = _record_with_pass_weight(0.99)
        records[-2] = _record_with_pass_weight(0.95)
        self.assertFalse(has_suspicious_mid_game_pass(records, threshold=0.5))


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

            records, winner, final_area = play_one_game(
                mcts, _BOARD_SIZE, komi=7.5, temperature=1.0, max_moves=10, rng=random.Random(0)
            )

        self.assertGreater(len(records), 0)
        self.assertGreaterEqual(final_area.black, 0)
        self.assertGreaterEqual(final_area.white, 0)
        for planes, policy_target, to_play in records:
            self.assertEqual(planes.shape, (3, _BOARD_SIZE, _BOARD_SIZE))
            self.assertEqual(policy_target.shape, (_BOARD_SIZE * _BOARD_SIZE + 1,))
            self.assertAlmostEqual(float(policy_target.sum()), 1.0, places=4)
            self.assertIn(to_play.name, ("BLACK", "WHITE"))
        self.assertTrue(winner is None or winner.name in ("BLACK", "WHITE"))

    def test_temperature_drop_move_makes_move_selection_deterministic(self) -> None:
        # temperature_drop_move=0 drops to greedy (argmax) selection from the very first
        # move -- sample_move's temperature<=0 branch ignores rng entirely, so two
        # different seeds should then produce the exact same game. Confirms
        # play_one_game actually anneals rather than ignoring the new parameter (see
        # docs/SELF_PLAY_STABILITY.md section 13).
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

            def play(seed: int) -> list[list[float]]:
                mcts = Mcts(net, MctsConfig(num_simulations=_NUM_SIMULATIONS))
                records, _, _ = play_one_game(
                    mcts,
                    _BOARD_SIZE,
                    komi=7.5,
                    temperature=1.0,
                    max_moves=6,
                    rng=random.Random(seed),
                    temperature_drop_move=0,
                )
                return [policy_target.tolist() for _, policy_target, _ in records]

            self.assertEqual(play(0), play(1))

    def test_no_pass_before_move_masks_pass_out_of_the_recorded_policy_target(self) -> None:
        # A hard structural guarantee against the Pass-collapse pattern (see
        # docs/SELF_PLAY_STABILITY.md section 17): Pass must carry exactly zero weight in
        # every recorded policy target below the threshold, regardless of what MCTS itself
        # searched.
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

            records, _, _ = play_one_game(
                mcts,
                _BOARD_SIZE,
                komi=7.5,
                temperature=1.0,
                max_moves=4,
                rng=random.Random(0),
                no_pass_before_move=4,
            )

        for _, policy_target, _ in records:
            self.assertEqual(0.0, policy_target[-1])


if __name__ == "__main__":
    unittest.main()
