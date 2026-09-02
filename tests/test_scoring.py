"""Port of igo-app/engine/src/test/kotlin/com/igoapp/engine/ScoringTest.kt.

Run with: python -m unittest discover
"""

from __future__ import annotations

import unittest

from engine.point import Point
from engine.position import Position
from engine.scoring import AreaScore, area_score
from engine.stone import Stone


class ScoringTest(unittest.TestCase):
    def test_an_empty_board_scores_zero_for_both_colors(self) -> None:
        score = area_score(Position.empty(9))
        self.assertEqual(AreaScore(0, 0), score)

    def test_stones_and_territory_exclusively_bordered_by_one_color_count_for_that_color(self) -> None:
        # A 3x3 board, entirely black except the empty center point, which is
        # surrounded only by black and so counts as black territory.
        cells = [Stone.BLACK] * 9
        cells[Point(1, 1).to_index(3)] = Stone.EMPTY
        position = Position(board_size=3, stones=cells, to_play=Stone.WHITE)

        score = area_score(position)
        self.assertEqual(AreaScore(black=9, white=0), score)

    def test_an_empty_region_bordered_by_both_colors_counts_for_neither(self) -> None:
        # A 3x3 board split diagonally: black in the top-left corner, white in the
        # bottom-right corner, and a center point touching both.
        position = Position(
            board_size=3,
            stones=[
                Stone.BLACK, Stone.BLACK, Stone.EMPTY,
                Stone.BLACK, Stone.EMPTY, Stone.WHITE,
                Stone.EMPTY, Stone.WHITE, Stone.WHITE,
            ],
            to_play=Stone.BLACK,
        )

        score = area_score(position)
        self.assertEqual(AreaScore(black=3, white=3), score)

    def test_winner_accounts_for_komi(self) -> None:
        self.assertEqual(Stone.BLACK, AreaScore(black=20, white=10).winner(komi=6.5))
        self.assertEqual(Stone.WHITE, AreaScore(black=10, white=20).winner(komi=0.5))
        self.assertIsNone(AreaScore(black=10, white=10).winner(komi=0.0))


if __name__ == "__main__":
    unittest.main()
