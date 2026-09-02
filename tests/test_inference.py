"""Port of igo-app/inference/src/test/kotlin/com/igoapp/inference/TfLitePolicyValueNetTest.kt.

Covers only the pure tensor encode/decode logic (no torch.nn.Module
forward pass) -- same scope as the Kotlin original, which covers only
the pure logic (no `Interpreter`, no `Context`).

Run with: python -m unittest discover
"""

from __future__ import annotations

import unittest

from bootstrap.inference import decode_policy, encode_planes
from engine.move import Pass, Play
from engine.point import Point
from engine.position import Position
from engine.stone import Stone


class EncodePlanesTest(unittest.TestCase):
    def test_marks_empty_points_on_an_empty_board(self) -> None:
        planes = encode_planes(Position.empty(3))
        for row in range(3):
            for col in range(3):
                self.assertEqual([0.0, 0.0, 1.0], planes[:, row, col].tolist())

    def test_channels_are_relative_to_the_player_to_move_not_a_fixed_color(self) -> None:
        cells = [Stone.EMPTY] * 9
        cells[Point(0, 0).to_index(3)] = Stone.BLACK
        position = Position(board_size=3, stones=cells, to_play=Stone.WHITE)

        planes = encode_planes(position)

        # Black stone at (0,0) belongs to the opponent from white-to-play's perspective.
        self.assertEqual([0.0, 1.0, 0.0], planes[:, 0, 0].tolist())
        self.assertEqual([0.0, 0.0, 1.0], planes[:, 1, 1].tolist())


class DecodePolicyTest(unittest.TestCase):
    def test_maps_each_legal_move_to_its_row_major_index_and_pass_to_the_trailing_index(self) -> None:
        position = Position.empty(3)
        raw_policy = [float(i) for i in range(10)]  # index i has value i, for easy identification

        decoded = decode_policy(position, raw_policy)

        self.assertEqual(10, len(decoded))  # 9 points + pass
        self.assertEqual(0.0, decoded[Play(Point(0, 0))])
        self.assertEqual(4.0, decoded[Play(Point(1, 1))])
        self.assertEqual(8.0, decoded[Play(Point(2, 2))])
        self.assertEqual(9.0, decoded[Pass()])

    def test_excludes_points_that_are_not_currently_legal(self) -> None:
        position = Position.empty(3).play(Play(Point(1, 1)))
        raw_policy = [float(i) for i in range(10)]

        decoded = decode_policy(position, raw_policy)

        self.assertEqual(9, len(decoded))  # 8 empty points + pass, (1,1) now occupied
        self.assertNotIn(Play(Point(1, 1)), decoded)


if __name__ == "__main__":
    unittest.main()
