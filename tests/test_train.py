"""Tests for bootstrap/train.py's model-construction/metadata helpers -- the warm-start
path (`init_from_checkpoint`) in particular, since silently building a fresh network
instead of continuing an existing one would defeat the point of Phase 3's self-play
fine-tuning (see docs/ROADMAP.md) without erroring.

Run with: python -m unittest discover
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from bootstrap.checkpoint import CheckpointMetadata, save_checkpoint
from bootstrap.model import RayZeroNet
from bootstrap.train import _build_model, _metadata_from_model_and_config

_BOARD_SIZE = 5


class BuildModelTest(unittest.TestCase):
    def test_builds_a_fresh_model_when_init_from_checkpoint_is_absent(self) -> None:
        config = {"board_size": _BOARD_SIZE, "channels": 8, "num_conv_layers": 2}
        model = _build_model(config)
        self.assertEqual(model.channels, 8)
        self.assertEqual(model.num_conv_layers, 2)
        self.assertEqual(model.num_residual_blocks, 0)

    def test_warm_starts_from_an_existing_checkpoints_weights_and_architecture(self) -> None:
        source_model = RayZeroNet(board_size=_BOARD_SIZE, channels=6, num_residual_blocks=2)
        metadata = CheckpointMetadata(board_size=_BOARD_SIZE, channels=6, num_conv_layers=3, num_residual_blocks=2)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "source.pt"
            save_checkpoint(source_model, metadata, checkpoint_path)

            config = {"board_size": _BOARD_SIZE, "init_from_checkpoint": str(checkpoint_path)}
            warm_started_model = _build_model(config)

        self.assertEqual(warm_started_model.channels, 6)
        self.assertEqual(warm_started_model.num_residual_blocks, 2)
        for key, value in source_model.state_dict().items():
            self.assertTrue(torch.equal(value, warm_started_model.state_dict()[key]))


class MetadataFromModelAndConfigTest(unittest.TestCase):
    def test_records_the_models_actual_architecture_and_the_warm_start_source(self) -> None:
        model = RayZeroNet(board_size=_BOARD_SIZE, channels=6, num_residual_blocks=2)
        config = {
            "seed": 7,
            "data_source": "selfplay_games/gen1.npz",
            "init_from_checkpoint": "checkpoints/parent.pt",
        }

        metadata = _metadata_from_model_and_config(model, config)

        self.assertEqual(metadata.channels, 6)
        self.assertEqual(metadata.num_residual_blocks, 2)
        self.assertEqual(metadata.init_from_checkpoint, "checkpoints/parent.pt")

    def test_init_from_checkpoint_is_none_for_a_from_scratch_run(self) -> None:
        model = RayZeroNet(board_size=_BOARD_SIZE)
        config = {"seed": 7, "data_source": "selfplay_games/batch1.npz"}

        metadata = _metadata_from_model_and_config(model, config)

        self.assertIsNone(metadata.init_from_checkpoint)


if __name__ == "__main__":
    unittest.main()
