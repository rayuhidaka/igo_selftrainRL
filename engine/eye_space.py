"""Port of igo-app/engine/EyeSpace.kt -- see engine/README.md.

An exact read of a small, closed life-and-death fight: the defender's group holding `target`
is walled in, and both sides play only inside `region` (its eye space) or pass. Used by
`puzzles/` to find and prove life-and-death puzzles for igo-app's puzzle area
(igo-app/docs/PUZZLES.md); the app re-proves every puzzle it bundles with the Kotlin original.

A proof, not a guess: running out of depth, or both sides passing with `target` still on the
board, counts as "not shown". The group counts as alive -- the attack failed -- once `target`
is unconditionally alive (Benson). Assumes the attacker's wall can't be captured from inside
`region` (it has liberties outside).

One addition the Kotlin side doesn't have: `node_budget`, a cap on positions searched, so a
batch scan can give up on a huge fight instead of stalling. Running out of budget raises
`BudgetExceeded` -- never a wrong answer.
"""

from __future__ import annotations

from engine.life_and_death import _benson_alive
from engine.move import Move, Pass, Play
from engine.point import Point
from engine.position import Position
from engine.stone import Stone

EYE_SPACE_DEPTH = 40
"""How many plies `attacker_kills` / `defender_lives` read by default (the rabbity six needs
more than 24, 32 was enough)."""


class BudgetExceeded(Exception):
    """The read searched more than its `node_budget` positions without an answer."""


def attacker_kills(
    position: Position,
    target: Point,
    region: set[Point],
    depth: int = EYE_SPACE_DEPTH,
    node_budget: int | None = None,
) -> bool:
    """With the side to move as the attacker: True only if every defense leads to `target`
    being captured within `depth` plies."""
    return EyeSpaceReader(target, region, node_budget).kill_wins(position, depth)


def defender_lives(
    position: Position,
    target: Point,
    region: set[Point],
    depth: int = EYE_SPACE_DEPTH,
    node_budget: int | None = None,
) -> bool:
    """The counterpart of `attacker_kills`, with the side to move as the defender: True only
    if some sequence makes `target` unconditionally alive whatever the attacker does inside
    `region`, within `depth` plies."""
    return EyeSpaceReader(target, region, node_budget).life_wins(position, depth)


class EyeSpaceReader:
    """The search behind `attacker_kills` and `defender_lives`, with its results cached per
    position and depth. Reusable across calls for the same `target` and `region`, which
    shares the cache (the puzzle generator asks many questions about one fight)."""

    def __init__(self, target: Point, region: set[Point], node_budget: int | None = None) -> None:
        self.target = target
        self.region = sorted(region, key=lambda p: (p.row, p.col))
        self.node_budget = node_budget
        self.nodes = 0
        self._kill_wins: dict[tuple[Position, int], bool] = {}
        self._kill_holds: dict[tuple[Position, int], bool] = {}
        self._life_wins: dict[tuple[Position, int], bool] = {}
        self._life_holds: dict[tuple[Position, int], bool] = {}
        self._alive: dict[Position, bool] = {}

    def _count(self) -> None:
        self.nodes += 1
        if self.node_budget is not None and self.nodes > self.node_budget:
            raise BudgetExceeded()

    def captured(self, position: Position) -> bool:
        return position.stone_at(self.target) == Stone.EMPTY

    def alive(self, position: Position) -> bool:
        # Membership in Kotlin's unconditionallyAlive(); only the target's own color can hold it.
        cached = self._alive.get(position)
        if cached is None:
            color = position.stone_at(self.target)
            cached = color != Stone.EMPTY and self.target in _benson_alive(position, color)
            self._alive[position] = cached
        return cached

    @staticmethod
    def over(position: Position, depth: int) -> bool:
        return depth == 0 or position.pass_count >= 2

    def moves(self, position: Position) -> list[Move]:
        plays: list[Move] = [
            Play(p) for p in self.region if position.stone_at(p) == Stone.EMPTY and position.is_legal(Play(p))
        ]
        return [*plays, Pass()]

    def kill_wins(self, position: Position, depth: int) -> bool:
        """Attacker to move: some move keeps the kill."""
        key = (position, depth)
        if key not in self._kill_wins:
            self._count()
            if self.captured(position):
                result = True
            elif self.alive(position) or self.over(position, depth):
                result = False
            else:
                result = any(self.kill_holds(position.play(m), depth - 1) for m in self.moves(position))
            self._kill_wins[key] = result
        return self._kill_wins[key]

    def kill_holds(self, position: Position, depth: int) -> bool:
        """Defender to move: every move still loses."""
        key = (position, depth)
        if key not in self._kill_holds:
            self._count()
            if self.captured(position):
                result = True
            elif self.alive(position) or self.over(position, depth):
                result = False
            else:
                result = all(self.kill_wins(position.play(m), depth - 1) for m in self.moves(position))
            self._kill_holds[key] = result
        return self._kill_holds[key]

    def life_wins(self, position: Position, depth: int) -> bool:
        """Defender to move: some move keeps the life."""
        key = (position, depth)
        if key not in self._life_wins:
            self._count()
            if self.alive(position):
                result = True
            elif self.captured(position) or self.over(position, depth):
                result = False
            else:
                result = any(self.life_holds(position.play(m), depth - 1) for m in self.moves(position))
            self._life_wins[key] = result
        return self._life_wins[key]

    def life_holds(self, position: Position, depth: int) -> bool:
        """Attacker to move: every move still lets the group live."""
        key = (position, depth)
        if key not in self._life_holds:
            self._count()
            if self.alive(position):
                result = True
            elif self.captured(position) or self.over(position, depth):
                result = False
            else:
                result = all(self.life_wins(position.play(m), depth - 1) for m in self.moves(position))
            self._life_holds[key] = result
        return self._life_holds[key]
