"""Tests for puzzles/filter.py: one puzzle per fight, the review wins, fewer easy ones.

Run with: python -m unittest discover
"""

from __future__ import annotations

import unittest

from puzzles.filter import eye_shape, filter_puzzles, fights

_EMPTY = ["." * 9] * 9


def puzzle(pid: str, source: str, region: list[list[int]], difficulty: int = 1, kind: str = "LIVE",
           nodes: int = 5) -> dict:
    return {"id": pid, "source": source, "region": region, "difficulty": difficulty, "kind": kind,
            "treeNodes": nodes, "board": _EMPTY}


class FightsTest(unittest.TestCase):
    def test_same_game_and_shared_area_is_one_fight_even_through_a_third_puzzle(self) -> None:
        a = puzzle("a", "g.json#1@10", [[0, 0], [0, 1]])
        b = puzzle("b", "g.json#1@12", [[0, 1], [0, 2]])
        c = puzzle("c", "g.json#1@14", [[0, 2], [0, 3]])  # touches b only
        self.assertEqual(1, len(fights([a, b, c])))

    def test_another_game_or_a_separate_area_is_another_fight(self) -> None:
        a = puzzle("a", "g.json#1@10", [[0, 0]])
        other_game = puzzle("b", "g.json#2@10", [[0, 0]])
        other_area = puzzle("c", "g.json#1@30", [[8, 8]])
        self.assertEqual(3, len(fights([a, other_game, other_area])))


class FilterTest(unittest.TestCase):
    def test_the_harder_puzzle_of_a_fight_survives(self) -> None:
        live = puzzle("ld-0001", "g.json#1@10", [[0, 0]], difficulty=1, kind="LIVE")
        kill = puzzle("ld-0002", "g.json#1@10", [[0, 0]], difficulty=3, kind="KILL")
        survivors, _ = filter_puzzles([live, kill], {}, None)
        self.assertEqual(["ld-0002"], [p["id"] for p in survivors])

    def test_a_kept_puzzle_beats_a_harder_one_and_two_kept_ones_both_stay(self) -> None:
        easy = puzzle("ld-0001", "g.json#1@10", [[0, 0]], difficulty=1)
        hard = puzzle("ld-0002", "g.json#1@10", [[0, 0]], difficulty=4)
        survivors, _ = filter_puzzles([easy, hard], {"ld-0001": "keep"}, None)
        self.assertEqual(["ld-0001"], [p["id"] for p in survivors])

        survivors, report = filter_puzzles([easy, hard], {"ld-0001": "keep", "ld-0002": "keep"}, None)
        self.assertEqual(["ld-0001", "ld-0002"], [p["id"] for p in survivors])
        self.assertEqual([["ld-0001", "ld-0002"]], report["kept_same_fight"])

    def test_a_dropped_puzzle_goes_and_its_fight_partner_takes_over(self) -> None:
        hard = puzzle("ld-0001", "g.json#1@10", [[0, 0]], difficulty=4)
        easy = puzzle("ld-0002", "g.json#1@10", [[0, 0]], difficulty=1)
        survivors, _ = filter_puzzles([hard, easy], {"ld-0001": "drop"}, None)
        self.assertEqual(["ld-0002"], [p["id"] for p in survivors])

    def test_easy_share_drops_easy_puzzles_until_the_share_is_met_but_never_a_kept_one(self) -> None:
        easy = [puzzle(f"e{i}", f"g.json#{i}@1", [[0, 0]], difficulty=1) for i in range(4)]
        hard = [puzzle(f"h{i}", f"h.json#{i}@1", [[0, 0]], difficulty=3) for i in range(2)]
        survivors, report = filter_puzzles(easy + hard, {"e0": "keep", "e1": "keep"}, 0.5)
        ids = {p["id"] for p in survivors}
        self.assertEqual({"e0", "e1", "h0", "h1"}, ids)  # 2 easy of 4 = 50%
        self.assertEqual(2, report["dropped_as_easy"])


class EyeShapeTest(unittest.TestCase):
    def test_a_bent_three_is_the_same_shape_in_every_corner(self) -> None:
        top_left = puzzle("a", "g#1@1", [[0, 0], [0, 1], [1, 0]])
        bottom_right = puzzle("b", "g#1@1", [[8, 8], [8, 7], [7, 8]])
        self.assertEqual(eye_shape(top_left), eye_shape(bottom_right))

    def test_stones_in_the_area_are_not_part_of_the_eye_shape(self) -> None:
        board = ["X" + "." * 8] + ["." * 9] * 8
        with_stone = dict(puzzle("a", "g#1@1", [[0, 0], [0, 1], [0, 2]]), board=board)
        self.assertEqual(((0, 0), (0, 1)), eye_shape(with_stone))


if __name__ == "__main__":
    unittest.main()
