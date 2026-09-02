"""Tests for eval/match.py. Small board, few simulations, few games -- this
exercises the real PyTorch model + MCTS search end to end, so it's slower
than the rest of the suite even at this size; not meant to say anything
about actual playing strength, just that the whole pipeline runs and
produces sane results.

Run with: python -m unittest discover
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from bootstrap.model import RayZeroNet
from eval.match import play_match

_BOARD_SIZE = 5
_NUM_SIMULATIONS = 4


def _save_untrained_checkpoint(path: Path, seed: int) -> None:
    torch.manual_seed(seed)
    model = RayZeroNet(board_size=_BOARD_SIZE)
    torch.save(model.state_dict(), path)


class PlayMatchTest(unittest.TestCase):
    def test_returns_one_result_per_game(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_a = Path(tmp) / "a.pt"
            checkpoint_b = Path(tmp) / "b.pt"
            _save_untrained_checkpoint(checkpoint_a, seed=1)
            _save_untrained_checkpoint(checkpoint_b, seed=2)

            results = play_match(
                checkpoint_a, checkpoint_b, board_size=_BOARD_SIZE, num_games=3, num_simulations=_NUM_SIMULATIONS
            )

            self.assertEqual(3, len(results))
            for result in results:
                self.assertIn(result.score_a, (0.0, 0.5, 1.0))

    def test_a_checkpoint_played_against_itself_averages_to_an_even_match(self) -> None:
        # Same weights on both sides, and neither Mcts nor RayZeroPolicyValueNet introduces
        # any randomness (no Dirichlet noise, no temperature sampling -- matching
        # igo-app/mcts/Mcts.kt), so with an even number of games and play_match alternating
        # which side plays Black, whatever fixed advantage the deterministic game outcome
        # gives one color washes out exactly over the alternation. A meaningful skew away
        # from 0.5 here would point to a real bug (e.g. a color/perspective mixup in
        # play_match or engine/'s rules), not just run-to-run noise.
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "same.pt"
            _save_untrained_checkpoint(checkpoint, seed=3)

            results = play_match(
                checkpoint, checkpoint, board_size=_BOARD_SIZE, num_games=6, num_simulations=_NUM_SIMULATIONS
            )

            average_score_a = sum(r.score_a for r in results) / len(results)
            self.assertAlmostEqual(average_score_a, 0.5, delta=0.05)


if __name__ == "__main__":
    unittest.main()
