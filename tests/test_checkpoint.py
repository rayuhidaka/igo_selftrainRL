"""Tests for bootstrap/checkpoint.py.

Run with: python -m unittest discover
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from bootstrap.checkpoint import CheckpointMetadata, build_model, load_checkpoint, save_checkpoint
from bootstrap.model import RayZeroNet

_BOARD_SIZE = 5


class SaveAndLoadCheckpointTest(unittest.TestCase):
    def test_round_trips_the_state_dict_and_metadata(self) -> None:
        model = RayZeroNet(board_size=_BOARD_SIZE, channels=8, num_conv_layers=2)
        metadata = CheckpointMetadata(
            board_size=_BOARD_SIZE, channels=8, num_conv_layers=2, data_source="selfplay_games/x.npz", seed=7
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(model, metadata, path)
            state_dict, loaded_metadata = load_checkpoint(path)

        self.assertEqual(loaded_metadata, metadata)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state_dict[key]))

    def test_loads_a_legacy_bare_state_dict_checkpoint_with_no_metadata(self) -> None:
        model = RayZeroNet(board_size=_BOARD_SIZE)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            torch.save(model.state_dict(), path)  # pre-bootstrap/checkpoint.py format
            state_dict, metadata = load_checkpoint(path)

        self.assertIsNone(metadata)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state_dict[key]))


class BuildModelTest(unittest.TestCase):
    def test_uses_metadata_when_no_explicit_override_is_given(self) -> None:
        metadata = CheckpointMetadata(board_size=_BOARD_SIZE, channels=8, num_conv_layers=2)
        model = build_model(metadata, _BOARD_SIZE)
        self.assertEqual(model.conv1.out_channels, 8)
        self.assertFalse(hasattr(model, "conv3"))

    def test_an_explicit_override_wins_over_metadata(self) -> None:
        metadata = CheckpointMetadata(board_size=_BOARD_SIZE, channels=8, num_conv_layers=2)
        model = build_model(metadata, _BOARD_SIZE, channels=16, num_conv_layers=3)
        self.assertEqual(model.conv1.out_channels, 16)
        self.assertTrue(hasattr(model, "conv3"))

    def test_falls_back_to_ray_zero_nets_own_defaults_when_metadata_is_none(self) -> None:
        model = build_model(None, _BOARD_SIZE)
        default_model = RayZeroNet(board_size=_BOARD_SIZE)
        self.assertEqual(model.conv1.out_channels, default_model.conv1.out_channels)
        self.assertEqual(hasattr(model, "conv3"), hasattr(default_model, "conv3"))


if __name__ == "__main__":
    unittest.main()
