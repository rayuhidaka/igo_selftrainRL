"""Tests for puzzles/generate_capture.py's difficulty from the reading (igo-app/docs/PUZZLES.md,
"Capture puzzles").

Run with: python -m unittest discover
"""

from __future__ import annotations

import unittest

from engine.capture import CaptureReader
from engine.move import Play
from engine.point import Point
from engine.stone import Stone
from puzzles.generate import parse_board
from puzzles.generate_capture import DEPTH, CaptureFight, capture_difficulty, main_line_plies


class MainLinePliesTest(unittest.TestCase):
    def test_counts_answer_and_reply_until_solved(self) -> None:
        tree = {"good": {"3,4": {"reply": [3, 5], "next": {"good": {"2,5": {"reply": "pass", "next": {"solved": True}}}}}}}
        self.assertEqual(main_line_plies(tree), 4)

    def test_an_answer_that_settles_it_at_once_is_one_ply(self) -> None:
        self.assertEqual(main_line_plies({"good": {"3,4": {"reply": [3, 5]}}}), 1)


class CaptureDifficultyTest(unittest.TestCase):
    def test_ataris_and_ladders_stay_easy(self) -> None:
        self.assertEqual(capture_difficulty("atari", 3), 1)
        self.assertEqual(capture_difficulty("ladder", 4), 1)
        self.assertEqual(capture_difficulty("ladder", 12), 2)

    def test_throw_ins_and_nets_grow_with_the_reading(self) -> None:
        self.assertEqual(capture_difficulty("net", 2), 2)
        self.assertEqual(capture_difficulty("throw-in", 5), 3)
        self.assertEqual(capture_difficulty("net", 7), 4)


class RefutationTest(unittest.TestCase):
    """A wrong move's line: the natural escape, ending once the stones are plainly free."""

    def test_atari_from_the_wrong_side_shows_the_escape(self) -> None:
        # White's two stones on the edge have two liberties, (7,5) and (8,6). The atari at (8,6)
        # drives them out: White extends at (7,5), and Black's ataris only push them along until
        # they have three liberties.
        rows = ["........."] * 7 + ["....X....", "...XOO..."]
        position = parse_board(rows, Stone.BLACK).play(Play(Point(8, 6)))
        region = {Point(r, c) for r in range(9) for c in range(9) if position.stone_at(Point(r, c)) == Stone.EMPTY}
        fight = CaptureFight(Point(8, 4), region, Stone.BLACK, 200_000)
        line = fight.refutation(position, DEPTH)
        self.assertEqual(line[0], [7, 5])
        self.assertEqual(len(line) % 2, 1, "ends with the defender's move")
        for i, step in enumerate(line):
            position = position.play(Play(Point(*step)))
            if i % 2 == 1:
                self.assertEqual(fight.target_liberties(position), 1, f"attacker move {step} is an atari")
        self.assertGreaterEqual(fight.target_liberties(position), 3)
        self.assertFalse(CaptureReader(Point(8, 4), node_budget=200_000).captures(position, DEPTH))


if __name__ == "__main__":
    unittest.main()
