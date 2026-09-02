"""Port of igo-app/engine/Move.kt -- see engine/README.md.

Kotlin's `sealed class Move { data class Play; object Pass }` becomes a
`Union[Play, Pass]` here -- `isinstance` checks stand in for Kotlin's
`when` exhaustiveness. `Pass` is a frozen dataclass with no fields rather
than a hand-rolled singleton: every `Pass()` instance already compares
and hashes equal, which is all Kotlin's `object Pass` guaranteed anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from engine.point import Point


@dataclass(frozen=True)
class Play:
    """Playing a stone at `point`."""

    point: Point


@dataclass(frozen=True)
class Pass:
    """Passing the turn without playing a stone."""


Move = Union[Play, Pass]
