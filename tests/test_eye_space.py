"""Tests for engine/eye_space.py, a port of igo-app's EyeSpaceTest.kt.

Run with: python -m unittest discover
"""

from __future__ import annotations

import dataclasses
import unittest

from engine.eye_space import BudgetExceeded, attacker_kills, defender_lives
from engine.life_and_death import unconditionally_alive
from engine.move import Pass, Play
from engine.point import Point
from engine.position import Position
from engine.stone import Stone


def _board(*rows: str, to_play: Stone = Stone.BLACK) -> Position:
    symbols = {"X": Stone.BLACK, "O": Stone.WHITE, ".": Stone.EMPTY}
    return Position(board_size=len(rows), stones=[symbols[c] for c in "".join(rows)], to_play=to_play)


# Black's group on the bottom edge, walled in by White, with a bent three of eye space.
_BENT_THREE = _board(
    ".......",
    ".......",
    ".......",
    "OOOOOO.",
    "XXXXXO.",
    "X.XXXO.",
    "X..XXO.",
    to_play=Stone.WHITE,
)
_REGION = {Point(5, 1), Point(6, 1), Point(6, 2)}
_TARGET = Point(4, 0)
_BEND = Point(6, 1)


class EyeSpaceTest(unittest.TestCase):
    def test_the_attacker_kills_a_bent_three_by_playing_the_bend_first(self) -> None:
        self.assertTrue(attacker_kills(_BENT_THREE, _TARGET, _REGION))
        after_bend = _BENT_THREE.play(Play(_BEND))
        self.assertFalse(defender_lives(after_bend, _TARGET, _REGION))

    def test_the_defender_lives_by_playing_the_bend_first(self) -> None:
        black_to_play = dataclasses.replace(_BENT_THREE, to_play=Stone.BLACK)
        self.assertTrue(defender_lives(black_to_play, _TARGET, _REGION))
        self.assertFalse(attacker_kills(black_to_play, _TARGET, _REGION))
        after_bend = black_to_play.play(Play(_BEND))
        self.assertIn(_TARGET, unconditionally_alive(after_bend))
        self.assertFalse(attacker_kills(after_bend, _TARGET, _REGION))

    def test_a_wrong_first_move_by_the_attacker_lets_the_defender_live(self) -> None:
        wrong = _BENT_THREE.play(Play(Point(6, 2)))
        self.assertTrue(defender_lives(wrong, _TARGET, _REGION))

    def test_a_square_four_dies_even_with_the_defender_to_move(self) -> None:
        square_four = _board(
            ".......",
            ".......",
            "OOOOOO.",
            "XXXXXO.",
            "X..XXO.",
            "X..XXO.",
            "XXXXXO.",
        )
        square = {Point(4, 1), Point(4, 2), Point(5, 1), Point(5, 2)}
        self.assertFalse(defender_lives(square_four, Point(3, 0), square))
        for move in [*(Play(p) for p in square), Pass()]:
            self.assertTrue(attacker_kills(square_four.play(move), Point(3, 0), square), move)

    def test_running_out_of_depth_proves_nothing_either_way(self) -> None:
        self.assertFalse(attacker_kills(_BENT_THREE, _TARGET, _REGION, depth=0))
        self.assertFalse(defender_lives(dataclasses.replace(_BENT_THREE, to_play=Stone.BLACK), _TARGET, _REGION, depth=0))

    def test_a_captured_group_is_killed_and_an_unconditionally_alive_one_lives_before_any_move(self) -> None:
        alive = dataclasses.replace(_BENT_THREE, to_play=Stone.BLACK).play(Play(_BEND))
        self.assertFalse(attacker_kills(alive, _TARGET, _REGION))
        self.assertTrue(defender_lives(dataclasses.replace(alive, to_play=Stone.BLACK), _TARGET, _REGION, depth=0))
        stones = list(_BENT_THREE.stones)
        stones[_TARGET.to_index(7)] = Stone.EMPTY
        empty = dataclasses.replace(_BENT_THREE, stones=tuple(stones))
        self.assertTrue(attacker_kills(empty, _TARGET, _REGION, depth=0))

    def test_running_out_of_node_budget_raises_instead_of_answering(self) -> None:
        with self.assertRaises(BudgetExceeded):
            attacker_kills(_BENT_THREE, _TARGET, _REGION, node_budget=1)


if __name__ == "__main__":
    unittest.main()
