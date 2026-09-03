"""Tests for bootstrap/dataset.py's build_training_loader (mixed-source, weighted
sampling) -- see its docstring for why this exists: an unmixed self-play fine-tune let
the network drift into degenerate behavior (docs/ROADMAP.md's Phase 3).

Run with: python -m unittest discover
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bootstrap.dataset import SelfPlayExamples, build_training_loader

_BOARD_SIZE = 5
_POLICY_SIZE = _BOARD_SIZE * _BOARD_SIZE + 1


def _write_examples(path: Path, count: int, value_fill: float) -> None:
    examples = SelfPlayExamples(
        board_planes=np.zeros((count, 3, _BOARD_SIZE, _BOARD_SIZE), dtype=np.float32),
        policy_targets=np.full((count, _POLICY_SIZE), 1.0 / _POLICY_SIZE, dtype=np.float32),
        # value_targets distinguishes which source an example came from, for the mixing test.
        value_targets=np.full((count,), value_fill, dtype=np.float32),
        score_margin_targets=np.zeros((count,), dtype=np.float32),
    )
    examples.save(path)


class BuildTrainingLoaderTest(unittest.TestCase):
    def test_a_single_source_behaves_like_plain_unweighted_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.npz"
            _write_examples(path, count=20, value_fill=1.0)

            loader, total = build_training_loader([(path, 1.0)], batch_size=4)

        self.assertEqual(total, 20)
        for _, _, value_targets, _ in loader:
            self.assertTrue((value_targets == 1.0).all())

    def test_mixes_sources_by_weight_independent_of_their_relative_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            large_path = Path(tmp) / "large.npz"
            small_path = Path(tmp) / "small.npz"
            _write_examples(large_path, count=1000, value_fill=0.0)  # the "broad" anchor set
            _write_examples(small_path, count=20, value_fill=1.0)  # the "new" self-play batch

            loader, total = build_training_loader([(large_path, 0.5), (small_path, 0.5)], batch_size=64)

            all_values = []
            for _, _, value_targets, _ in loader:
                all_values.extend(value_targets.tolist())

        self.assertEqual(total, 1020)
        fraction_from_small = sum(1 for v in all_values if v == 1.0) / len(all_values)
        # Weighted 50/50 despite the small source being ~2% of the combined example count --
        # allow generous slack since this is random sampling, not an exact split.
        self.assertGreater(fraction_from_small, 0.35)
        self.assertLess(fraction_from_small, 0.65)


if __name__ == "__main__":
    unittest.main()
