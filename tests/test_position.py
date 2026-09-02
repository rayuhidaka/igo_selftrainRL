"""Port of igo-app/engine/src/test/kotlin/com/igoapp/engine/PositionTest.kt.

Run with: python -m unittest discover
"""

from __future__ import annotations

import unittest

from engine.illegal_move_error import IllegalMoveError
from engine.move import Pass, Play
from engine.point import Point
from engine.position import Position
from engine.stone import Stone


def board_of(board_size, *stones, to_play=Stone.BLACK, ko_point=None):
    """Builds a Position with specific stones placed, for scenarios that would be tedious to set up move-by-move."""
    cells = [Stone.EMPTY] * (board_size * board_size)
    for point, stone in stones:
        cells[point.to_index(board_size)] = stone
    return Position(board_size, cells, to_play, ko_point)


class PositionTest(unittest.TestCase):
    def test_empty_creates_an_all_empty_board_with_black_to_play(self) -> None:
        position = Position.empty(9)
        self.assertEqual(9, position.board_size)
        self.assertEqual(Stone.BLACK, position.to_play)
        self.assertTrue(all(s == Stone.EMPTY for s in position.stones))
        self.assertEqual(81, len(position.stones))

    def test_stone_at_returns_the_occupant_at_a_point(self) -> None:
        position = board_of(9, (Point(0, 0), Stone.BLACK))
        self.assertEqual(Stone.BLACK, position.stone_at(Point(0, 0)))
        self.assertEqual(Stone.EMPTY, position.stone_at(Point(1, 1)))

    def test_stone_at_raises_for_an_off_board_point(self) -> None:
        position = Position.empty(9)
        with self.assertRaises(ValueError):
            position.stone_at(Point(9, 0))

    def test_is_on_board_is_true_within_bounds_and_false_outside(self) -> None:
        position = Position.empty(9)
        self.assertTrue(position.is_on_board(Point(0, 0)))
        self.assertTrue(position.is_on_board(Point(8, 8)))
        self.assertFalse(position.is_on_board(Point(-1, 0)))
        self.assertFalse(position.is_on_board(Point(0, 9)))

    def test_neighbors_excludes_off_board_points(self) -> None:
        position = Position.empty(9)
        self.assertEqual({Point(0, 1), Point(1, 0)}, set(position.neighbors(Point(0, 0))))
        self.assertEqual(
            {Point(4, 3), Point(4, 5), Point(3, 4), Point(5, 4)},
            set(position.neighbors(Point(4, 4))),
        )

    def test_legal_moves_on_an_empty_board_is_every_point_plus_pass(self) -> None:
        position = Position.empty(9)
        moves = position.legal_moves()
        self.assertEqual(82, len(moves))  # 81 points + pass
        self.assertIn(Pass(), moves)

    def test_play_places_a_stone_and_alternates_the_player_to_move(self) -> None:
        position = Position.empty(9)
        next_position = position.play(Play(Point(2, 2)))
        self.assertEqual(Stone.BLACK, next_position.stone_at(Point(2, 2)))
        self.assertEqual(Stone.WHITE, next_position.to_play)

    def test_play_on_an_occupied_point_is_illegal(self) -> None:
        position = Position.empty(9).play(Play(Point(2, 2)))
        self.assertFalse(position.is_legal(Play(Point(2, 2))))
        with self.assertRaises(IllegalMoveError):
            position.play(Play(Point(2, 2)))

    def test_pass_alternates_the_player_and_clears_ko_without_placing_a_stone(self) -> None:
        position = Position.empty(9)
        next_position = position.play(Pass())
        self.assertEqual(Stone.WHITE, next_position.to_play)
        self.assertEqual(1, next_position.pass_count)
        self.assertTrue(all(s == Stone.EMPTY for s in next_position.stones))

    def test_a_play_that_would_leave_zero_liberties_without_capturing_is_suicide_and_illegal(self) -> None:
        # Black to play at the center of a plus shape fully surrounded by white.
        position = board_of(
            5,
            (Point(1, 2), Stone.WHITE),
            (Point(3, 2), Stone.WHITE),
            (Point(2, 1), Stone.WHITE),
            (Point(2, 3), Stone.WHITE),
            to_play=Stone.BLACK,
        )
        move = Play(Point(2, 2))
        self.assertFalse(position.is_legal(move))
        with self.assertRaises(IllegalMoveError):
            position.play(move)

    def test_a_play_that_captures_is_legal_even_though_the_placed_stone_would_otherwise_have_no_liberties(
        self,
    ) -> None:
        # White stone at (2,2) has a single liberty at (2,1). Black fills it, which
        # simultaneously captures white and gives the black stone a liberty at (2,2).
        position = board_of(
            5,
            (Point(1, 2), Stone.BLACK),
            (Point(3, 2), Stone.BLACK),
            (Point(2, 3), Stone.BLACK),
            (Point(2, 2), Stone.WHITE),
            to_play=Stone.BLACK,
        )
        move = Play(Point(2, 1))
        self.assertTrue(position.is_legal(move))

        next_position = position.play(move)
        self.assertEqual(Stone.EMPTY, next_position.stone_at(Point(2, 2)))
        self.assertEqual(Stone.BLACK, next_position.stone_at(Point(2, 1)))
        self.assertEqual(Stone.WHITE, next_position.to_play)

    def test_capturing_removes_every_stone_in_a_multi_stone_group(self) -> None:
        # A two-stone white group at (2,1)-(2,2) with a single remaining liberty at (2,3).
        position = board_of(
            5,
            (Point(1, 1), Stone.BLACK),
            (Point(3, 1), Stone.BLACK),
            (Point(2, 0), Stone.BLACK),
            (Point(1, 2), Stone.BLACK),
            (Point(3, 2), Stone.BLACK),
            (Point(2, 1), Stone.WHITE),
            (Point(2, 2), Stone.WHITE),
            to_play=Stone.BLACK,
        )
        next_position = position.play(Play(Point(2, 3)))
        self.assertEqual(Stone.EMPTY, next_position.stone_at(Point(2, 1)))
        self.assertEqual(Stone.EMPTY, next_position.stone_at(Point(2, 2)))
        self.assertEqual(Stone.BLACK, next_position.stone_at(Point(2, 3)))

    def test_recapturing_immediately_at_a_single_stone_ko_point_is_illegal(self) -> None:
        # Black captures a lone white stone at (2,2) by playing (2,1); the resulting
        # black stone at (2,1) has a single liberty (2,2), making it a ko.
        position = board_of(
            5,
            (Point(1, 2), Stone.BLACK),
            (Point(3, 2), Stone.BLACK),
            (Point(2, 3), Stone.BLACK),
            (Point(2, 2), Stone.WHITE),
            (Point(1, 1), Stone.WHITE),
            (Point(3, 1), Stone.WHITE),
            (Point(2, 0), Stone.WHITE),
            to_play=Stone.BLACK,
        )
        after_capture = position.play(Play(Point(2, 1)))
        self.assertEqual(Point(2, 2), after_capture.ko_point)
        self.assertFalse(after_capture.is_legal(Play(Point(2, 2))))

    def test_ko_restriction_clears_once_another_move_is_played(self) -> None:
        after_capture = board_of(5, to_play=Stone.WHITE, ko_point=Point(2, 2))
        after_elsewhere = after_capture.play(Play(Point(0, 0)))
        self.assertIsNone(after_elsewhere.ko_point)

    def test_legal_moves_excludes_the_current_ko_point(self) -> None:
        position = board_of(5, to_play=Stone.WHITE, ko_point=Point(2, 2))
        self.assertNotIn(Play(Point(2, 2)), position.legal_moves())

    def test_board_size_is_a_parameter_not_fixed_to_9(self) -> None:
        position = Position.empty(3)
        self.assertEqual(9, len(position.stones))
        self.assertEqual(10, len(position.legal_moves()))  # 9 points + pass


if __name__ == "__main__":
    unittest.main()
