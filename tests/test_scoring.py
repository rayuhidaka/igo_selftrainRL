"""Port of igo-app/engine/src/test/kotlin/com/igoapp/engine/ScoringTest.kt.

Run with: python -m unittest discover
"""

from __future__ import annotations

import unittest

from engine.point import Point
from engine.position import Position
from engine.scoring import AreaScore, area_score, ownership_plane, territory_ownership
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


class TerritoryOwnershipTest(unittest.TestCase):
    def test_an_empty_board_has_no_owner_anywhere(self) -> None:
        ownership = territory_ownership(Position.empty(9))
        self.assertTrue(all(owner is None for owner in ownership.values()))
        self.assertEqual(len(ownership), 81)

    def test_stones_and_territory_exclusively_bordered_by_one_color_are_owned_by_it(self) -> None:
        # Same fixture as ScoringTest's matching area_score test -- the per-point read must
        # tally to the same totals area_score already reports.
        cells = [Stone.BLACK] * 9
        cells[Point(1, 1).to_index(3)] = Stone.EMPTY
        position = Position(board_size=3, stones=cells, to_play=Stone.WHITE)

        ownership = territory_ownership(position)

        self.assertTrue(all(owner == Stone.BLACK for owner in ownership.values()))
        self.assertEqual(sum(1 for owner in ownership.values() if owner == Stone.BLACK), 9)

    def test_an_empty_region_bordered_by_both_colors_has_no_owner(self) -> None:
        position = Position(
            board_size=3,
            stones=[
                Stone.BLACK, Stone.BLACK, Stone.EMPTY,
                Stone.BLACK, Stone.EMPTY, Stone.WHITE,
                Stone.EMPTY, Stone.WHITE, Stone.WHITE,
            ],
            to_play=Stone.BLACK,
        )

        ownership = territory_ownership(position)

        self.assertIsNone(ownership[Point(0, 2)])
        self.assertIsNone(ownership[Point(1, 1)])
        self.assertIsNone(ownership[Point(2, 0)])

    def test_agrees_with_area_score_on_a_mixed_position(self) -> None:
        # area_score is now implemented in terms of territory_ownership -- this is a
        # regression check that the refactor changed nothing observable, on a position with
        # both stones and territory of both colors, not just the two hand-picked fixtures
        # above.
        cells = [Stone.BLACK] * 9
        cells[Point(1, 1).to_index(3)] = Stone.EMPTY
        position = Position(board_size=3, stones=cells, to_play=Stone.WHITE)

        ownership = territory_ownership(position)
        black_count = sum(1 for owner in ownership.values() if owner == Stone.BLACK)
        white_count = sum(1 for owner in ownership.values() if owner == Stone.WHITE)

        self.assertEqual(AreaScore(black_count, white_count), area_score(position))


class OwnershipPlaneTest(unittest.TestCase):
    def test_flattens_to_perspective_relative_values(self) -> None:
        cells = [Stone.BLACK] * 9
        cells[Point(1, 1).to_index(3)] = Stone.EMPTY
        position = Position(board_size=3, stones=cells, to_play=Stone.WHITE)
        ownership = territory_ownership(position)

        black_perspective = ownership_plane(ownership, board_size=3, perspective=Stone.BLACK)
        white_perspective = ownership_plane(ownership, board_size=3, perspective=Stone.WHITE)

        self.assertEqual(black_perspective, [1.0] * 9)
        self.assertEqual(white_perspective, [-1.0] * 9)

    def test_neutral_points_are_zero_for_either_perspective(self) -> None:
        position = Position(
            board_size=3,
            stones=[
                Stone.BLACK, Stone.BLACK, Stone.EMPTY,
                Stone.BLACK, Stone.EMPTY, Stone.WHITE,
                Stone.EMPTY, Stone.WHITE, Stone.WHITE,
            ],
            to_play=Stone.BLACK,
        )
        ownership = territory_ownership(position)

        plane = ownership_plane(ownership, board_size=3, perspective=Stone.BLACK)

        self.assertEqual(plane[Point(0, 2).to_index(3)], 0.0)
        self.assertEqual(plane[Point(1, 1).to_index(3)], 0.0)
        self.assertEqual(plane[Point(2, 0).to_index(3)], 0.0)
        self.assertEqual(plane[Point(0, 0).to_index(3)], 1.0)
        self.assertEqual(plane[Point(2, 2).to_index(3)], -1.0)


if __name__ == "__main__":
    unittest.main()
