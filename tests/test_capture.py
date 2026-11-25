"""Tests for engine/capture.py on the proverb cards' tactics boards (igo-app ui/ProverbCards.kt)."""

import unittest

from engine.capture import CaptureReader
from engine.point import Point
from engine.position import Position
from engine.stone import Stone

STONES = {"X": Stone.BLACK, "O": Stone.WHITE, ".": Stone.EMPTY}
BUDGET = 200_000


def board(*rows: str, to_play: Stone = Stone.BLACK) -> Position:
    return Position(board_size=len(rows), stones=tuple(STONES[c] for r in rows for c in r), to_play=to_play)


LADDER = board(".XX......", "XOOX.....", ".XOOX....", "..XOO....", "...X.....", ".........", ".........", ".........", ".........")
NET = board(".........", ".........", ".........", "....XX...", "...XO....", "...X.....", "......O..", ".........", ".........")
SNAPBACK = board(".......", ".......", ".......", ".......", "XXXXXX.", "XOOOOX.", "XO..OX.")


class CaptureReaderTest(unittest.TestCase):
    def test_ladder_card(self):
        moves = CaptureReader(Point(3, 4), node_budget=BUDGET).capturing_moves(LADDER)
        self.assertIn(Point(3, 5), moves)

    def test_net_card_only_the_net_works(self):
        # The white stone at (6,6) breaks the ladder: both ataris fail, the net at (5,5) captures.
        self.assertEqual([Point(5, 5)], CaptureReader(Point(4, 4), node_budget=BUDGET).capturing_moves(NET))

    def test_without_the_net_the_stone_escapes(self):
        # White to move (Black passed): it runs out.
        white_to_play = Position(board_size=9, stones=NET.stones, to_play=Stone.WHITE)
        reader = CaptureReader(Point(4, 4), node_budget=BUDGET)
        self.assertFalse(reader.holds(white_to_play))

    def test_snapback_card(self):
        moves = CaptureReader(Point(5, 1), node_budget=BUDGET).capturing_moves(SNAPBACK)
        self.assertTrue({Point(6, 2), Point(6, 3)} & set(moves))

    def test_a_chain_with_four_liberties_has_escaped(self):
        lone = board(".........", ".........", ".........", ".........", "....O....", ".........", ".........", ".........", ".........")
        self.assertFalse(CaptureReader(Point(4, 4), node_budget=BUDGET).captures(lone))


if __name__ == "__main__":
    unittest.main()
