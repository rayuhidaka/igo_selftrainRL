"""Tests for engine/scoring.py's territory_score, eval/match.py's play_game_to_end and
eval/rule_comparison.py.

Run with: python -m unittest discover
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from bootstrap.inference import RayZeroPolicyValueNet
from bootstrap.model import RayZeroNet
from engine.position import Position
from engine.scoring import AreaScore, area_score, territory_score
from engine.stone import Stone
from eval.match import play_game_to_end
from eval.rule_comparison import black_margin, summarize
from mcts.mcts import Mcts, MctsConfig


def _board(*rows: str) -> Position:
    symbols = {"X": Stone.BLACK, "O": Stone.WHITE, ".": Stone.EMPTY}
    return Position(board_size=len(rows), stones=[symbols[c] for c in "".join(rows)], to_play=Stone.BLACK)


def _record(black_area: int, white_area: int, black_territory: int, white_territory: int, finished: bool = True) -> dict:
    return {
        "finished": finished,
        "black_area": black_area,
        "white_area": white_area,
        "black_territory": black_territory,
        "white_territory": white_territory,
    }


class TerritoryScoreTest(unittest.TestCase):
    def test_counts_empty_territory_and_prisoners_but_not_stones(self) -> None:
        position = _board(
            "..X.O",
            "..X.O",
            "..XO.",
            "..XO.",
            "..XO.",
        )
        # Black: the 10 empty points left of its wall + 2 prisoners. (0,3) and (1,3) touch both
        # colors (dame). White: (2,4)-(4,4) + 3 prisoners.
        self.assertEqual(AreaScore(black=12, white=6), territory_score(position, black_prisoners=2, white_prisoners=3))

    def test_area_and_territory_differ_by_stones_and_prisoners(self) -> None:
        position = _board(
            ".X.",
            "XXX",
            "OOO",
        )
        self.assertEqual(AreaScore(black=6, white=3), area_score(position))
        self.assertEqual(AreaScore(black=2, white=1), territory_score(position, black_prisoners=0, white_prisoners=1))


class PlayGameToEndTest(unittest.TestCase):
    def test_prisoners_match_the_stones_missing_from_the_final_board(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "net.pt"
            torch.manual_seed(3)
            torch.save(RayZeroNet(board_size=5).state_dict(), checkpoint)
            mcts = Mcts(RayZeroPolicyValueNet(checkpoint, 5), MctsConfig(num_simulations=4))
            end = play_game_to_end(mcts, mcts, 5)

        # Every move either passes or adds one stone, so stones played = stones left + captured.
        stones_left = sum(1 for s in end.position.stones if s != Stone.EMPTY)
        passes = end.position.pass_count  # at least the final two, if finished
        self.assertGreaterEqual(end.moves, stones_left + end.black_prisoners + end.white_prisoners)
        self.assertGreaterEqual(end.black_prisoners, 0)
        self.assertGreaterEqual(end.white_prisoners, 0)
        if end.finished:
            self.assertGreaterEqual(passes, 2)


class SummarizeTest(unittest.TestCase):
    def test_black_margin_adds_komi_to_white(self) -> None:
        self.assertEqual(0.5, black_margin(41, 40, 0.5))
        self.assertEqual(-6.5, black_margin(40, 40, 6.5))

    def test_counts_a_game_whose_winner_changes_as_a_flip(self) -> None:
        records = [
            # Black wins both ways: area 45 vs 36+7.5, territory 20 vs 12+6.5.
            _record(45, 36, 20, 12),
            # Black wins by area (44 vs 36+7.5) but loses by territory (18 vs 12+6.5).
            _record(44, 36, 18, 12),
        ]
        stats = summarize(records, area_komi=7.5, territory_komi=6.5)
        self.assertEqual(2, stats["games"])
        self.assertEqual(1, stats["flips"])
        self.assertEqual(0.5, stats["flip_rate"])

    def test_leaves_unfinished_games_out(self) -> None:
        stats = summarize([_record(45, 36, 20, 12), _record(0, 0, 0, 0, finished=False)], 7.5, 6.5)
        self.assertEqual(1, stats["games"])
        self.assertEqual(1, stats["unfinished"])

    def test_reports_the_average_margin_shift(self) -> None:
        # Area margin 45-36-7.5 = 1.5; territory margin 20-12-6.5 = 1.5 -> no shift.
        self.assertEqual(0.0, summarize([_record(45, 36, 20, 12)], 7.5, 6.5)["mean_margin_shift"])


if __name__ == "__main__":
    unittest.main()
