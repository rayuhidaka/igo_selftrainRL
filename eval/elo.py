"""Elo rating updates for eval/'s checkpoint tournaments.

Pure rating math -- no dependency on eval/match.py's (not yet implemented)
actual game-playing. See docs/ARCHITECTURE.md's "Difficulty-tier promotion"
section for how this fits into the bigger picture: checkpoints are saved
often and cheaply during training, but only *promoted* to a shippable
difficulty tier once `should_promote` says so.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_INITIAL_RATING = 1500.0
DEFAULT_K_FACTOR = 32.0


@dataclass(frozen=True)
class MatchResult:
    """The outcome of one game, from player A's perspective: `1.0` a win, `0.0` a loss, `0.5` a draw."""

    score_a: float


def expected_score(rating_a: float, rating_b: float) -> float:
    """Returns A's expected score against B (win probability, treating a draw as 0.5), per the standard Elo formula."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_ratings(
    rating_a: float,
    rating_b: float,
    results: list[MatchResult],
    k_factor: float = DEFAULT_K_FACTOR,
) -> tuple[float, float]:
    """Returns `(new_rating_a, new_rating_b)` after applying every result in `results` sequentially."""
    for result in results:
        expected_a = expected_score(rating_a, rating_b)
        delta = k_factor * (result.score_a - expected_a)
        rating_a += delta
        rating_b -= delta
    return rating_a, rating_b


def should_promote(candidate_rating: float, current_tier_rating: float, min_elo_gap: float) -> bool:
    """Returns True if `candidate_rating` clears `current_tier_rating` by at least `min_elo_gap`.

    A candidate checkpoint only becomes a new shippable difficulty tier once
    it's meaningfully stronger than the last tier shipped, not just because
    it's the newest checkpoint that happened to finish training -- otherwise
    the app's difficulty picker fills up with near-identical-strength
    options. See docs/ARCHITECTURE.md.
    """
    return candidate_rating >= current_tier_rating + min_elo_gap
