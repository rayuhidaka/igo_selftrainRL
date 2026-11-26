"""Tests for puzzles/generate_capture.py's difficulty from the reading (igo-app/docs/PUZZLES.md,
"Capture puzzles").

Run with: python -m unittest discover
"""

from __future__ import annotations

import unittest

from puzzles.generate_capture import capture_difficulty, main_line_plies


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


if __name__ == "__main__":
    unittest.main()
