"""Tests for eval/gate.py's field gate. Run with: python -m unittest discover

The numbers in the regression cases are the real round-robin ratings that motivated the gate
(see eval/gate.py's module docstring): gen11 edged past gen10, while the 1000-game gen12 beat
gen11 head-to-head but rated well below it on the shared scale.
"""

from __future__ import annotations

import unittest

from eval.gate import passes_field_gate


class PassesFieldGateTest(unittest.TestCase):
    def test_gen11_over_gen10_passes(self) -> None:
        self.assertTrue(passes_field_gate({"gen10": 2373.0, "gen11": 2384.7}, "gen11", "gen10"))

    def test_gen12_1000games_under_gen11_fails(self) -> None:
        self.assertFalse(passes_field_gate({"gen11": 2303.3, "gen12": 2217.6}, "gen12", "gen11"))

    def test_equal_rating_passes(self) -> None:
        self.assertTrue(passes_field_gate({"a": 2000.0, "b": 2000.0}, "a", "b"))

    def test_max_drop_tolerates_a_small_deficit(self) -> None:
        ratings = {"parent": 2000.0, "candidate": 1990.0}
        self.assertFalse(passes_field_gate(ratings, "candidate", "parent"))
        self.assertTrue(passes_field_gate(ratings, "candidate", "parent", max_drop=15.0))

    def test_missing_candidate_raises_rather_than_deciding(self) -> None:
        with self.assertRaises(KeyError):
            passes_field_gate({"parent": 2000.0}, "candidate", "parent")
