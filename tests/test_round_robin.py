"""Tests for eval/round_robin.py's Elo-fitting logic (not `main()`'s actual game-playing,
which needs real checkpoints). Run with: python -m unittest discover

`fit_elo_bradley_terry` replaced an earlier `fit_elo_naive_sequential` after a real round-robin
run (igo-app's gen1-9 checkpoints) caught the naive method inverting two direct, unambiguous
head-to-head results -- see that function's docstring. These tests lock in the property that
actually mattered: a simultaneous fit should respect the *whole* comparison graph, including
correctly outranking a player who only narrowly, unreliably beat a stronger opponent once.
"""

from __future__ import annotations

import unittest

from eval.round_robin import fit_elo_bradley_terry


class FitEloBradleyTerryTest(unittest.TestCase):
    def test_anchor_lands_exactly_on_anchor_rating(self) -> None:
        pairing_scores = [("a", "b", [1.0] * 20 + [0.0] * 20)]
        ratings = fit_elo_bradley_terry(["a", "b"], pairing_scores, anchor_name="a", anchor_rating=1500.0)
        self.assertAlmostEqual(ratings["a"], 1500.0, places=6)

    def test_a_dominant_player_rates_above_a_weak_one(self) -> None:
        pairing_scores = [("a", "b", [1.0] * 40)]  # a (score_a=1.0) wins every game
        ratings = fit_elo_bradley_terry(["a", "b"], pairing_scores, anchor_name="b")
        self.assertGreater(ratings["a"], ratings["b"])

    def test_transitive_field_ranks_in_generation_order(self) -> None:
        # a beats b, b beats c, a beats c -- fully transitive, decisive margins.
        pairing_scores = [
            ("a", "b", [1.0] * 35 + [0.0] * 5),
            ("b", "c", [1.0] * 35 + [0.0] * 5),
            ("a", "c", [1.0] * 39 + [0.0] * 1),
        ]
        ratings = fit_elo_bradley_terry(["a", "b", "c"], pairing_scores, anchor_name="c")
        self.assertGreater(ratings["a"], ratings["b"])
        self.assertGreater(ratings["b"], ratings["c"])

    def test_global_evidence_outweighs_a_close_direct_pairing(self) -> None:
        # x narrowly beats y head-to-head (22-18 of 40, matching the real gen4/gen5 case this
        # regression-tests) but loses far worse than y does against a shared, much stronger
        # opponent z -- x's aggregate record against the field is better, so the fit should
        # rank x above y despite y's own direct-match win.
        pairing_scores = [
            ("x", "y", [1.0] * 18 + [0.0] * 22),  # y wins the direct pairing, 22-18
            ("x", "z", [1.0] * 10 + [0.0] * 30),  # x: 25% vs. z
            ("y", "z", [1.0] * 3 + [0.0] * 37),  # y: 7.5% vs. z (much worse than x)
        ]
        ratings = fit_elo_bradley_terry(["x", "y", "z"], pairing_scores, anchor_name="z")
        self.assertGreater(ratings["x"], ratings["y"])


if __name__ == "__main__":
    unittest.main()
