"""Plays games between two checkpoints, to produce eval/elo.py's MatchResults.

NOT YET IMPLEMENTED, but half-unblocked: `engine/` (a Python port of
igo-app/engine/'s Go rules, proven against its own test suite -- see
engine/README.md) now provides rules enforcement (legal moves, captures,
ko, scoring). What's still missing is *search* -- something that picks
reasonable moves for each checkpoint rather than uniformly-random ones, or
Elo differences between similar-strength checkpoints won't be measurable.
Doesn't need to be `igo-app/mcts/`'s exact PUCT implementation, just needs
to exist. See docs/ARCHITECTURE.md's "Difficulty-tier promotion" section.
"""

from __future__ import annotations

from pathlib import Path

from eval.elo import MatchResult


def play_match(checkpoint_a: Path, checkpoint_b: Path, board_size: int, num_games: int) -> list[MatchResult]:
    """Plays `num_games` games between `checkpoint_a` and `checkpoint_b` and
    returns one `MatchResult` per game, from `checkpoint_a`'s perspective.
    """
    raise NotImplementedError(
        "eval/match.py needs a search step to actually play games (engine/ provides the rules "
        "already) -- see this module's docstring and docs/ARCHITECTURE.md's Difficulty-tier "
        "promotion section."
    )
