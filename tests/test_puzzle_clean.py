"""Tests for puzzles/clean.py: a puzzle's board cleared down to its fight, the answer unchanged.

Run with: python -m unittest discover
"""

from __future__ import annotations

import unittest

from engine.eye_space import EYE_SPACE_DEPTH
from engine.point import Point
from engine.stone import Stone
from puzzles.clean import clean_puzzle, fight_stones
from puzzles.generate import Fight, build_tree, parse_board

# Black's corner group around a straight three on the top edge (row 0, columns 0-2), walled
# in by one White chain that runs on down the left side (rows 2-6). Far away: a White stone
# at the bottom right and a Black pair on the right that have nothing to do with the fight.
BOARD = [
    "...XO....",
    "XXXXO....",
    "OOOOO..XX",
    "OOOO.....",
    "OOOO.....",
    "OOOO.....",
    "OOOO.....",
    ".........",
    "........O",
]
TARGET = Point(1, 0)
REGION = [[0, 0], [0, 1], [0, 2]]


def puzzle(kind: str, to_play: str) -> dict:
    position = parse_board(BOARD, Stone.BLACK if to_play == "X" else Stone.WHITE)
    fight = Fight(TARGET, {Point(*rc) for rc in REGION}, Stone.WHITE, 400_000)
    return {
        "id": "ld-test",
        "kind": kind,
        "toPlay": to_play,
        "board": BOARD,
        "target": [TARGET.row, TARGET.col],
        "region": REGION,
        "answer": [0, 1],
        "tree": build_tree(fight, position, EYE_SPACE_DEPTH, [0], 300),
        "history": [
            {"number": 40, "point": [0, 3], "color": "X"},  # in the fight: stays
            {"number": 41, "point": [8, 8], "color": "O"},  # far away: removed with its stone
        ],
    }


class FightStonesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.keep = fight_stones(parse_board(BOARD, Stone.BLACK), TARGET, Stone.WHITE)

    def test_keeps_the_defender_and_the_wall_next_to_the_area(self) -> None:
        for p in [(0, 3), (1, 0), (1, 3), (0, 4), (1, 4), (2, 0), (2, 4)]:
            self.assertIn(Point(*p), self.keep)

    def test_trims_the_wall_tail_but_keeps_it_one_solid_chain(self) -> None:
        self.assertIn(Point(3, 0), self.keep)  # two steps from the area
        self.assertNotIn(Point(4, 0), self.keep)  # three steps: the tail goes
        self.assertNotIn(Point(6, 3), self.keep)

    def test_drops_everything_that_doesnt_touch_the_fight(self) -> None:
        for p in [(2, 7), (2, 8), (8, 8)]:
            self.assertNotIn(Point(*p), self.keep)

    def test_a_wall_in_two_pieces_is_joined_by_the_shortest_run_of_its_own_stones(self) -> None:
        # The same corner, but the wall's two halves only meet by way of a loop round a White eye
        # (rows 2-4, columns 3-5). Near the area they're two pieces; the loop joins them, the
        # rest of the chain (rows 4-5, columns 0-2) is still a tail.
        board = [
            "...XO....",
            "XXXXOO...",
            "OOOO.O...",
            "OOOO.O...",
            "OOOOOO...",
            "OOO......",
            ".........",
            ".........",
            ".........",
        ]
        keep = fight_stones(parse_board(board, Stone.BLACK), TARGET, Stone.WHITE)
        for p in [(4, 3), (4, 4), (4, 5), (3, 5), (2, 5)]:
            self.assertIn(Point(*p), keep)
        for p in [(4, 0), (4, 1), (4, 2), (5, 0), (5, 1), (5, 2)]:
            self.assertNotIn(Point(*p), keep)


class CleanPuzzleTest(unittest.TestCase):
    def test_the_board_is_cleared_and_the_tree_is_the_same(self) -> None:
        for kind, to_play in (("LIVE", "X"), ("KILL", "O")):
            original = puzzle(kind, to_play)
            cleaned, reason = clean_puzzle(original)
            self.assertIsNone(reason)
            self.assertEqual(original["tree"], cleaned["tree"])
            self.assertEqual(BOARD, cleaned["fullBoard"])
            self.assertEqual(
                ["...XO....", "XXXXO....", "OOOOO....", "OOOO.....",
                 ".........", ".........", ".........", ".........", "........."],
                cleaned["board"],
            )

    def test_history_keeps_only_the_stones_still_on_the_board(self) -> None:
        cleaned, _ = clean_puzzle(puzzle("LIVE", "X"))
        self.assertEqual([40], [h["number"] for h in cleaned["history"]])

    def test_a_board_with_nothing_else_on_it_stays_as_it_is(self) -> None:
        cleaned, _ = clean_puzzle(puzzle("LIVE", "X"))
        again, reason = clean_puzzle(cleaned)
        self.assertEqual("nothing to remove", reason)
        self.assertIs(cleaned, again)

    def test_a_puzzle_whose_tree_would_change_keeps_its_full_board(self) -> None:
        wrong = {**puzzle("LIVE", "X"), "tree": {"solved": True}}  # a tree the cleared board can't match
        kept, reason = clean_puzzle(wrong)
        self.assertEqual("tree changed", reason)
        self.assertIs(wrong, kept)


if __name__ == "__main__":
    unittest.main()
