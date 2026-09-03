"""Tests for selfplay/generate.py's score_margin -- the new auxiliary training target (see
bootstrap/model.py's module docstring and docs/SELF_PLAY_STABILITY.md for why it exists).

Run with: python -m unittest discover
"""

from __future__ import annotations

import unittest

from engine.scoring import AreaScore
from engine.stone import Stone
from selfplay.generate import score_margin

_BOARD_SIZE = 9


class ScoreMarginTest(unittest.TestCase):
    def test_positive_for_the_perspective_that_ended_up_ahead(self) -> None:
        area = AreaScore(black=50, white=31)  # black wins by 19 before komi
        margin = score_margin(area, komi=7.5, to_play=Stone.BLACK, board_size=_BOARD_SIZE)
        self.assertGreater(margin, 0)

    def test_negated_for_the_opposite_perspective_of_the_same_game(self) -> None:
        area = AreaScore(black=50, white=31)
        black_margin = score_margin(area, komi=7.5, to_play=Stone.BLACK, board_size=_BOARD_SIZE)
        white_margin = score_margin(area, komi=7.5, to_play=Stone.WHITE, board_size=_BOARD_SIZE)
        self.assertAlmostEqual(black_margin, -white_margin, places=5)

    def test_normalized_by_board_size_squared(self) -> None:
        area = AreaScore(black=_BOARD_SIZE * _BOARD_SIZE, white=0)  # black controls the whole board
        margin = score_margin(area, komi=0.0, to_play=Stone.BLACK, board_size=_BOARD_SIZE)
        self.assertAlmostEqual(margin, 1.0, places=5)

    def test_a_narrow_komi_only_win_scores_much_closer_to_zero_than_a_decisive_one(self) -> None:
        # The exact scenario implicated in the self-play collapse: an empty board where
        # White "wins" purely from komi should score far weaker than a real, played-out win.
        empty_board = AreaScore(black=0, white=0)
        decisive_win = AreaScore(black=0, white=_BOARD_SIZE * _BOARD_SIZE)
        narrow_margin = score_margin(empty_board, komi=7.5, to_play=Stone.WHITE, board_size=_BOARD_SIZE)
        decisive_margin = score_margin(decisive_win, komi=7.5, to_play=Stone.WHITE, board_size=_BOARD_SIZE)
        self.assertGreater(narrow_margin, 0)
        self.assertLess(narrow_margin, decisive_margin)


if __name__ == "__main__":
    unittest.main()
