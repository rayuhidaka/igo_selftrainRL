"""Tests for engine/life_and_death.py, a port of igo-app's LifeAndDeathTest.kt.

Run with: python -m unittest discover
"""

from __future__ import annotations

import dataclasses
import unittest

from engine.life_and_death import estimate_dead_stones, unconditionally_alive, without_dead_stones
from engine.point import Point
from engine.position import Position
from engine.scoring import AreaScore, area_score
from engine.stone import Stone


def _board(*rows: str) -> Position:
    symbols = {"X": Stone.BLACK, "O": Stone.WHITE, ".": Stone.EMPTY}
    return Position(board_size=len(rows), stones=[symbols[c] for c in "".join(rows)], to_play=Stone.BLACK)


def _points(position: Position, stone: Stone) -> set[Point]:
    size = position.board_size
    return {Point.from_index(i, size) for i, s in enumerate(position.stones) if s == stone}


_TWO_EYED_CORNER = _board(".X.X.....", "XXXX.....", *["........."] * 7)
_DEAD_INSIDE = _board("OOOOO", "O.X.O", "O...O", "O.X.O", "OOOOO")
_DEAD = {Point(1, 2), Point(3, 2)}


class UnconditionallyAliveTest(unittest.TestCase):
    def test_a_group_with_two_real_eyes_is_alive(self) -> None:
        self.assertEqual(_points(_TWO_EYED_CORNER, Stone.BLACK), unconditionally_alive(_TWO_EYED_CORNER))

    def test_a_group_with_one_eye_is_not(self) -> None:
        one_eye = _board(".XX......", "XXX......", *["........."] * 7)
        self.assertEqual(set(), unconditionally_alive(one_eye))

    def test_an_empty_board_has_nothing_alive(self) -> None:
        self.assertEqual(set(), unconditionally_alive(Position.empty(9)))


class EstimateDeadStonesTest(unittest.TestCase):
    def test_stones_the_playouts_capture_are_dead(self) -> None:
        captured = without_dead_stones(_DEAD_INSIDE, _DEAD)
        self.assertEqual(_DEAD, estimate_dead_stones(_DEAD_INSIDE, [captured, captured]))

    def test_stones_captured_in_only_half_the_playouts_are_not(self) -> None:
        captured = without_dead_stones(_DEAD_INSIDE, _DEAD)
        self.assertEqual(set(), estimate_dead_stones(_DEAD_INSIDE, [captured, _DEAD_INSIDE]))

    def test_an_unconditionally_alive_group_is_never_dead(self) -> None:
        taken_over = dataclasses.replace(
            _TWO_EYED_CORNER, stones=[Stone.WHITE if s == Stone.BLACK else s for s in _TWO_EYED_CORNER.stones]
        )
        self.assertEqual(set(), estimate_dead_stones(_TWO_EYED_CORNER, [taken_over]))

    def test_playouts_on_another_board_size_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_dead_stones(_DEAD_INSIDE, [Position.empty(9)])

    def test_removing_dead_stones_gives_their_area_to_the_surrounding_color(self) -> None:
        self.assertEqual(AreaScore(black=0, white=25), area_score(without_dead_stones(_DEAD_INSIDE, _DEAD)))


if __name__ == "__main__":
    unittest.main()
