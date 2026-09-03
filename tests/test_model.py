"""Tests for bootstrap/model.py.

Run with: python -m unittest discover
"""

from __future__ import annotations

import unittest

import torch

from bootstrap.model import RayZeroNet

_BOARD_SIZE = 5
_BATCH = 2


class RayZeroNetForwardTest(unittest.TestCase):
    def _assert_correct_output_shapes(self, model: RayZeroNet) -> None:
        board_planes = torch.zeros(_BATCH, 3, _BOARD_SIZE, _BOARD_SIZE)
        policy, value = model(board_planes)
        self.assertEqual(policy.shape, (_BATCH, _BOARD_SIZE * _BOARD_SIZE + 1))
        self.assertEqual(value.shape, (_BATCH, 1))
        self.assertTrue(torch.allclose(policy.sum(dim=1), torch.ones(_BATCH), atol=1e-5))
        self.assertTrue(torch.all(value >= -1.0) and torch.all(value <= 1.0))

    def test_plain_stack_architecture(self) -> None:
        self._assert_correct_output_shapes(RayZeroNet(board_size=_BOARD_SIZE, channels=8, num_conv_layers=3))

    def test_legacy_two_layer_plain_stack_architecture(self) -> None:
        self._assert_correct_output_shapes(RayZeroNet(board_size=_BOARD_SIZE, channels=8, num_conv_layers=2))

    def test_residual_tower_architecture(self) -> None:
        model = RayZeroNet(board_size=_BOARD_SIZE, channels=8, num_residual_blocks=3)
        self._assert_correct_output_shapes(model)
        self.assertFalse(hasattr(model, "conv1"))
        self.assertEqual(len(model.residual_blocks), 3)

    def test_residual_block_preserves_its_input_shape(self) -> None:
        model = RayZeroNet(board_size=_BOARD_SIZE, channels=8, num_residual_blocks=1)
        x = torch.randn(_BATCH, 8, _BOARD_SIZE, _BOARD_SIZE)
        y = model.residual_blocks[0](x)
        self.assertEqual(y.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
