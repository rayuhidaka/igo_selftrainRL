"""Tests for bootstrap/dataset.py's dihedral symmetry augmentation (see
igo-app/docs/ROADMAP.md's "weak opening moves" item for why this exists). The critical
property tested throughout is *consistency*: whatever transform is applied to
`board_planes` must land a stone at exactly the point `policy_targets`' matching transform
says is now the best move -- a mismatch between the two would train the wrong
board-to-move association, actively worse than no augmentation at all.

Run with: python -m unittest discover
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from bootstrap.dataset import (
    _DIHEDRAL_TRANSFORMS,
    SelfPlayDataset,
    SelfPlayExamples,
    _apply_dihedral_transform,
    _apply_dihedral_transform_to_ownership,
)

_BOARD_SIZE = 9


def _one_hot_policy(row: int, col: int, board_size: int, pass_prob: float = 0.0) -> np.ndarray:
    policy = np.zeros(board_size * board_size + 1, dtype=np.float32)
    policy[row * board_size + col] = 1.0 - pass_prob
    policy[-1] = pass_prob
    return policy


def _marker_board(row: int, col: int, board_size: int) -> np.ndarray:
    board = np.zeros((3, board_size, board_size), dtype=np.float32)
    board[0, row, col] = 1.0
    return board


def _marker_ownership(row: int, col: int, board_size: int) -> np.ndarray:
    plane = np.zeros(board_size * board_size, dtype=np.float32)
    plane[row * board_size + col] = 1.0
    return plane


class ApplyDihedralTransformTest(unittest.TestCase):
    def test_identity_transform_leaves_board_and_policy_unchanged(self) -> None:
        board = _marker_board(2, 5, _BOARD_SIZE)
        policy = _one_hot_policy(2, 5, _BOARD_SIZE)
        identity_index = _DIHEDRAL_TRANSFORMS.index((0, False))

        transformed_board, transformed_policy = _apply_dihedral_transform(board, policy, _BOARD_SIZE, identity_index)

        np.testing.assert_array_equal(transformed_board, board)
        np.testing.assert_array_equal(transformed_policy, policy)

    def test_every_transform_keeps_the_stone_and_the_policy_peak_at_the_same_point(self) -> None:
        # The point that actually matters: board_planes and policy_targets must move together.
        board = _marker_board(1, 6, _BOARD_SIZE)
        policy = _one_hot_policy(1, 6, _BOARD_SIZE)

        for transform_index in range(len(_DIHEDRAL_TRANSFORMS)):
            with self.subTest(transform_index=transform_index):
                transformed_board, transformed_policy = _apply_dihedral_transform(board, policy, _BOARD_SIZE, transform_index)

                stone_row, stone_col = np.argwhere(transformed_board[0] == 1.0)[0]
                policy_peak = int(np.argmax(transformed_policy[:-1]))
                self.assertEqual(policy_peak, stone_row * _BOARD_SIZE + stone_col)

    def test_pass_probability_and_total_mass_are_preserved(self) -> None:
        rng = np.random.default_rng(0)
        board = _marker_board(4, 4, _BOARD_SIZE)
        policy = rng.dirichlet(np.ones(_BOARD_SIZE * _BOARD_SIZE + 1)).astype(np.float32)

        for transform_index in range(len(_DIHEDRAL_TRANSFORMS)):
            with self.subTest(transform_index=transform_index):
                _, transformed_policy = _apply_dihedral_transform(board, policy, _BOARD_SIZE, transform_index)

                self.assertAlmostEqual(float(transformed_policy[-1]), float(policy[-1]), places=6)
                self.assertAlmostEqual(float(transformed_policy.sum()), float(policy.sum()), places=5)

    def test_the_8_transforms_are_not_all_the_same(self) -> None:
        # A corner/near-corner marker (the exact failure mode this augmentation targets)
        # should land on 8 distinct points across the 8 transforms, not collapse together.
        board = _marker_board(0, 1, _BOARD_SIZE)
        policy = _one_hot_policy(0, 1, _BOARD_SIZE)

        landing_points = set()
        for transform_index in range(len(_DIHEDRAL_TRANSFORMS)):
            transformed_board, _ = _apply_dihedral_transform(board, policy, _BOARD_SIZE, transform_index)
            landing_points.add(tuple(np.argwhere(transformed_board[0] == 1.0)[0]))

        self.assertEqual(len(landing_points), 8)


class ApplyDihedralTransformToOwnershipTest(unittest.TestCase):
    def test_identity_transform_leaves_the_plane_unchanged(self) -> None:
        ownership = _marker_ownership(2, 5, _BOARD_SIZE)
        identity_index = _DIHEDRAL_TRANSFORMS.index((0, False))

        transformed = _apply_dihedral_transform_to_ownership(ownership, _BOARD_SIZE, identity_index)

        np.testing.assert_array_equal(transformed, ownership)

    def test_the_same_transform_index_moves_an_ownership_marker_to_the_same_point_as_a_board_marker(self) -> None:
        # The property that actually matters: a shared transform_index (as
        # SelfPlayDataset.__getitem__ uses) must move the ownership plane's marker to
        # exactly the same point a board_planes marker at the same original coordinate
        # would land on -- otherwise the ownership head would train on the wrong
        # orientation relative to board_planes/policy_targets.
        board = _marker_board(1, 6, _BOARD_SIZE)
        ownership = _marker_ownership(1, 6, _BOARD_SIZE)
        policy = _one_hot_policy(1, 6, _BOARD_SIZE)

        for transform_index in range(len(_DIHEDRAL_TRANSFORMS)):
            with self.subTest(transform_index=transform_index):
                transformed_board, _ = _apply_dihedral_transform(board, policy, _BOARD_SIZE, transform_index)
                transformed_ownership = _apply_dihedral_transform_to_ownership(
                    ownership, _BOARD_SIZE, transform_index
                )

                board_row, board_col = np.argwhere(transformed_board[0] == 1.0)[0]
                ownership_peak = int(np.argmax(transformed_ownership))
                self.assertEqual(ownership_peak, board_row * _BOARD_SIZE + board_col)

    def test_total_mass_is_preserved(self) -> None:
        rng = np.random.default_rng(0)
        ownership = rng.uniform(-1.0, 1.0, size=_BOARD_SIZE * _BOARD_SIZE).astype(np.float32)

        for transform_index in range(len(_DIHEDRAL_TRANSFORMS)):
            with self.subTest(transform_index=transform_index):
                transformed = _apply_dihedral_transform_to_ownership(ownership, _BOARD_SIZE, transform_index)
                self.assertAlmostEqual(float(transformed.sum()), float(ownership.sum()), places=4)


class SelfPlayDatasetAugmentTest(unittest.TestCase):
    def _examples(self) -> SelfPlayExamples:
        board = _marker_board(0, 0, _BOARD_SIZE)[None, ...]
        policy = _one_hot_policy(0, 0, _BOARD_SIZE)[None, ...]
        value = np.array([1.0], dtype=np.float32)
        score_margin = np.array([0.5], dtype=np.float32)
        ownership = _marker_ownership(0, 0, _BOARD_SIZE)[None, ...]
        return SelfPlayExamples(board, policy, value, score_margin, ownership)

    def test_augment_false_is_a_pure_passthrough(self) -> None:
        dataset = SelfPlayDataset(self._examples(), augment=False)

        board_planes, policy_target, value, score_margin, ownership_target = dataset[0]

        np.testing.assert_array_equal(board_planes.numpy(), self._examples().board_planes[0])
        np.testing.assert_array_equal(policy_target.numpy(), self._examples().policy_targets[0])
        self.assertEqual(float(value), 1.0)
        self.assertEqual(float(score_margin), 0.5)
        np.testing.assert_array_equal(ownership_target.numpy(), self._examples().ownership_targets[0])

    def test_augment_true_applies_the_transform_consistently(self) -> None:
        dataset = SelfPlayDataset(self._examples(), augment=True)
        rot180_index = _DIHEDRAL_TRANSFORMS.index((2, False))

        with patch("bootstrap.dataset.random.randrange", return_value=rot180_index):
            board_planes, policy_target, _, _, ownership_target = dataset[0]

        stone_row, stone_col = np.argwhere(board_planes.numpy()[0] == 1.0)[0]
        policy_peak = int(np.argmax(policy_target.numpy()[:-1]))
        self.assertEqual(policy_peak, stone_row * _BOARD_SIZE + stone_col)
        # A 180-degree rotation of corner (0, 0) on a 9-wide board lands on the opposite corner.
        self.assertEqual((stone_row, stone_col), (_BOARD_SIZE - 1, _BOARD_SIZE - 1))
        # The ownership marker (also originally at (0, 0)) must land at the exact same
        # rotated point as the board marker -- same transform_index, same physical move.
        ownership_peak = int(np.argmax(ownership_target.numpy()))
        self.assertEqual(ownership_peak, stone_row * _BOARD_SIZE + stone_col)

    def test_augment_true_calls_randrange_exactly_once_per_item(self) -> None:
        # SelfPlayDataset.__getitem__ must reuse one transform_index for board_planes,
        # policy_target, AND ownership_target -- calling random.randrange twice (once per
        # target) would let them drift to different orientations.
        dataset = SelfPlayDataset(self._examples(), augment=True)
        identity_index = _DIHEDRAL_TRANSFORMS.index((0, False))

        with patch("bootstrap.dataset.random.randrange", return_value=identity_index) as mock_randrange:
            dataset[0]

        mock_randrange.assert_called_once()


if __name__ == "__main__":
    unittest.main()
