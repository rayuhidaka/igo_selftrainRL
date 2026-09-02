"""Plays games between two checkpoints, to produce eval/elo.py's MatchResults.

NOT YET IMPLEMENTED. Needs a way to actually play a 9x9 Go game between two
loaded RayZeroNet checkpoints -- rules enforcement (legal moves, captures,
ko, scoring) and some search (even a lightweight one; doesn't need to be
`igo-app/mcts/`'s exact PUCT implementation, just something that plays
reasonably rather than uniformly-random moves, or Elo differences between
similar-strength checkpoints won't be measurable).

See docs/ARCHITECTURE.md's "Difficulty-tier promotion" section for the
recommended approach: port igo-app/engine/'s Go rules (Kotlin) to Python,
rather than writing a second implementation from scratch. It's small
(Position, Move, legality, capture/ko, area scoring -- a few hundred lines)
and already unit-tested there; porting it keeps both codebases agreeing on
what a legal game even is, instead of risking silent divergence between two
independently-written rule sets.
"""

from __future__ import annotations

from pathlib import Path

from eval.elo import MatchResult


def play_match(checkpoint_a: Path, checkpoint_b: Path, board_size: int, num_games: int) -> list[MatchResult]:
    """Plays `num_games` games between `checkpoint_a` and `checkpoint_b` and
    returns one `MatchResult` per game, from `checkpoint_a`'s perspective.
    """
    raise NotImplementedError(
        "eval/match.py needs a Python Go rules engine + search to actually play games -- "
        "see this module's docstring and docs/ARCHITECTURE.md's Difficulty-tier promotion section."
    )
