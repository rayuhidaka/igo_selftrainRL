"""Port of igo-app/engine/Stone.kt -- see that file for the canonical version.

This is deliberately a close, file-for-file port (see engine/README.md):
any behavior change should be made on both sides, or the two rules
implementations risk silently disagreeing about what's legal.
"""

from __future__ import annotations

from enum import Enum, auto


class Stone(Enum):
    """The occupant of a single board intersection."""

    EMPTY = auto()
    BLACK = auto()
    WHITE = auto()

    def opponent(self) -> "Stone":
        """Returns the opposing color.

        Raises:
            ValueError: if called on EMPTY, which has no opponent.
        """
        if self is Stone.BLACK:
            return Stone.WHITE
        if self is Stone.WHITE:
            return Stone.BLACK
        raise ValueError("EMPTY has no opponent")
