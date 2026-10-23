"""Tests for engine/scoring.py's territory_score, eval/match.py's play_game_to_end and
eval/rule_comparison.py.

Run with: python -m unittest discover
"""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

import torch

from bootstrap.inference import RayZeroPolicyValueNet
from bootstrap.model import RayZeroNet
from engine.move import Pass, Play
from engine.point import Point
from engine.position import Position
from engine.scoring import AreaScore, area_score, territory_score
from engine.stone import Stone
from eval.match import play_game_to_end
from eval.rule_comparison import (
    analyze_self_fill,
    black_margin,
    classify_post_pass_move,
    decode_history,
    encode_history,
    saves_chain_in_atari,
    summarize,
    summarize_self_fill,
)
from mcts.mcts import Mcts, MctsConfig


def _board(*rows: str, to_play: Stone = Stone.BLACK) -> Position:
    symbols = {"X": Stone.BLACK, "O": Stone.WHITE, ".": Stone.EMPTY}
    return Position(board_size=len(rows), stones=[symbols[c] for c in "".join(rows)], to_play=to_play)


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

    def test_history_replays_to_the_final_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "net.pt"
            torch.manual_seed(3)
            torch.save(RayZeroNet(board_size=5).state_dict(), checkpoint)
            mcts = Mcts(RayZeroPolicyValueNet(checkpoint, 5), MctsConfig(num_simulations=4))
            end = play_game_to_end(mcts, mcts, 5)

        self.assertEqual(end.moves, len(end.history))
        position = Position.empty(5)
        for move in decode_history(encode_history(end.history, 5), 5):
            position = position.play(move)
        self.assertEqual(end.position, position)


# Black's wall on column 2, White's on column 3; White to play right after Black passed.
_WALLS = ("..XO.",) * 5


def _walls_game_record(white_fills: bool) -> dict:
    """A 5x5 game: the two walls, Black passes, White (optionally) plays (0,4) inside its own
    area, then both pass."""
    history = []
    for row in range(5):
        history += [Play(Point(row, 2)), Play(Point(row, 3))]
    history.append(Pass())
    if white_fills:
        history += [Play(Point(0, 4)), Pass()]
    history.append(Pass())
    position = Position.empty(5)
    for move in history:
        position = position.play(move)
    area = area_score(position)
    territory = territory_score(position, 0, 0)
    return {
        "finished": True,
        "black_area": area.black,
        "white_area": area.white,
        "black_territory": territory.black,
        "white_territory": territory.white,
        "board_size": 5,
        "history": encode_history(history, 5),
    }


class SelfFillTest(unittest.TestCase):
    def test_a_point_only_the_movers_stones_border_is_enclosed(self) -> None:
        before = dataclasses.replace(_board(*_WALLS, to_play=Stone.WHITE), pass_count=1)
        self.assertEqual("enclosed", classify_post_pass_move(before, Point(0, 4), before))

    def test_a_point_in_the_opponents_area_is_elsewhere(self) -> None:
        before = dataclasses.replace(_board(*_WALLS, to_play=Stone.WHITE), pass_count=1)
        self.assertEqual("elsewhere", classify_post_pass_move(before, Point(0, 1), before))

    def test_capturing_stones_the_game_shows_dead_is_dead_stones(self) -> None:
        before = _board("..XO.", "..XOX", "..XO.", "..XO.", "..XO.", to_play=Stone.WHITE)
        # The rest of the game captured the black stone at (1,4).
        final = _board("..XOO", "..XO.", "..XOO", "..XO.", "..XO.")
        self.assertEqual("dead_stones", classify_post_pass_move(before, Point(0, 4), final))
        # Without that hindsight (the stone still stands at the end) it's left to the rule.
        self.assertEqual("elsewhere", classify_post_pass_move(before, Point(0, 4), before))

    def test_saving_an_own_chain_in_atari_is_detected(self) -> None:
        position = _board("XO...", ".....", ".....", ".....", ".....")
        self.assertTrue(saves_chain_in_atari(position, Point(1, 0), Stone.BLACK))
        self.assertFalse(saves_chain_in_atari(position, Point(2, 2), Stone.BLACK))
        # (0,2) is a liberty of White's (0,1) stone, which has two -- not in atari.
        self.assertFalse(saves_chain_in_atari(position, Point(0, 2), Stone.WHITE))

    def test_a_rule_ends_the_game_before_the_fill(self) -> None:
        analysis = analyze_self_fill(_walls_game_record(white_fills=True))
        self.assertEqual(1, analysis["kinds"]["enclosed"])
        # White filled (0,4): actual territory B 10, W 4; ending before it gives W its 5th point.
        self.assertEqual((10, 5), analysis["rule_ends"]["enclosed_only"])
        self.assertEqual((10, 5), analysis["rule_ends"]["with_dead_read"])

    def test_a_game_without_post_pass_fills_keeps_its_count(self) -> None:
        analysis = analyze_self_fill(_walls_game_record(white_fills=False))
        self.assertEqual(0, sum(analysis["kinds"].values()))
        self.assertEqual((10, 5), analysis["rule_ends"]["enclosed_only"])

    def test_summary_counts_flips_the_rule_fixes(self) -> None:
        # Area B 15 vs W 10: White wins at komi 5.5. Territory with the fill: B 10 vs W 4, Black
        # wins at komi 5.5 -- a flip. Ending before the fill: 10 vs 5 -- White wins, fixed.
        stats = summarize_self_fill([_walls_game_record(white_fills=True)], area_komi=5.5, territory_komi=5.5)
        self.assertEqual(1, stats["post_pass_moves"]["enclosed"])
        self.assertEqual({"flips": 0, "fixed": 1, "introduced": 0, "flip_rate": 0.0}, stats["rules"]["enclosed_only"])


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
