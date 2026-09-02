"""Tests for eval/elo.py. Run with: python -m unittest discover"""

from __future__ import annotations

import unittest

from eval.elo import MatchResult, expected_score, should_promote, update_ratings


class ExpectedScoreTest(unittest.TestCase):
    def test_equal_ratings_gives_a_coin_flip(self) -> None:
        self.assertAlmostEqual(expected_score(1500.0, 1500.0), 0.5)

    def test_a_stronger_player_is_favored(self) -> None:
        self.assertGreater(expected_score(1600.0, 1500.0), 0.5)

    def test_symmetric(self) -> None:
        a = expected_score(1600.0, 1400.0)
        b = expected_score(1400.0, 1600.0)
        self.assertAlmostEqual(a + b, 1.0)


class UpdateRatingsTest(unittest.TestCase):
    def test_a_win_raises_the_winners_rating_and_lowers_the_losers(self) -> None:
        rating_a, rating_b = update_ratings(1500.0, 1500.0, [MatchResult(score_a=1.0)])
        self.assertGreater(rating_a, 1500.0)
        self.assertLess(rating_b, 1500.0)

    def test_ratings_move_by_equal_and_opposite_amounts(self) -> None:
        rating_a, rating_b = update_ratings(1500.0, 1500.0, [MatchResult(score_a=1.0)])
        self.assertAlmostEqual((rating_a - 1500.0), -(rating_b - 1500.0))

    def test_an_expected_win_moves_ratings_less_than_an_upset(self) -> None:
        # A is already much stronger, so beating B again is expected -- a small move.
        expected_win_a, _ = update_ratings(1800.0, 1200.0, [MatchResult(score_a=1.0)])
        # B beating a much stronger A is an upset -- a big move for both.
        upset_a, upset_b = update_ratings(1800.0, 1200.0, [MatchResult(score_a=0.0)])
        self.assertLess(abs(expected_win_a - 1800.0), abs(upset_a - 1800.0))
        self.assertLess(upset_a, 1800.0)
        self.assertGreater(upset_b, 1200.0)

    def test_repeated_results_accumulate(self) -> None:
        rating_a, _ = update_ratings(1500.0, 1500.0, [MatchResult(score_a=1.0)] * 10)
        self.assertGreater(rating_a, 1600.0)


class ShouldPromoteTest(unittest.TestCase):
    def test_promotes_when_the_gap_clears_the_threshold(self) -> None:
        self.assertTrue(should_promote(candidate_rating=1600.0, current_tier_rating=1500.0, min_elo_gap=50.0))

    def test_does_not_promote_when_the_gap_is_too_small(self) -> None:
        self.assertFalse(should_promote(candidate_rating=1520.0, current_tier_rating=1500.0, min_elo_gap=50.0))

    def test_promotes_exactly_at_the_threshold(self) -> None:
        self.assertTrue(should_promote(candidate_rating=1550.0, current_tier_rating=1500.0, min_elo_gap=50.0))

    def test_does_not_promote_a_weaker_candidate(self) -> None:
        self.assertFalse(should_promote(candidate_rating=1400.0, current_tier_rating=1500.0, min_elo_gap=50.0))


if __name__ == "__main__":
    unittest.main()
