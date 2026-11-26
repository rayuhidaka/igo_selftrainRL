"""An exact local capture reader, wider than ladder reading: the basis of capture puzzles
(igo-app/docs/PUZZLES.md, type B).

igo-app's `Tactics.kt` reads only ladder-style fights (the attacker plays only on the target's
liberties), so it misses nets, throw-ins and snapbacks. This reader lets both sides play anywhere
near the fight:

- the defender may play anywhere within Manhattan distance 2 of the target chain, plus on the
  liberties of attacking chains touching it with at most two liberties, or pass (broad: leaving
  a defence out would make a capture look forced);
- the attacker tries the target's liberties, with two liberties left also the point next to
  both (the net), and saving its own chains in atari next to the target (narrow: a missing try can
  only hide a capture, never invent one);
- both sets are recomputed at every node, so a ladder can run across the board;
- the target is **captured** when it is off the board, and has **escaped** once it has
  `escape_liberties` liberties or more; an unconditionally alive (Benson) target is never
  capturable, checked once at the start of a read.

Counting a chain with `escape_liberties` liberties as escaped is conservative for the attacker:
every "captures" answer is a real capture within this model (a capture that needs the target to
be chased past that many liberties is not found). Running out of depth counts as escaped too.
Moves far from the fight (a ladder breaker placed elsewhere) are not considered, the same local
assumption the eye-space reader makes about its wall.
"""

from __future__ import annotations

from engine.eye_space import BudgetExceeded
from engine.life_and_death import _benson_alive
from engine.move import Move, Pass, Play
from engine.point import Point
from engine.position import Position
from engine.stone import Stone

CAPTURE_DEPTH = 30
ESCAPE_LIBERTIES = 4


def chain_at(position: Position, point: Point) -> set[Point]:
    color = position.stone_at(point)
    chain, stack = {point}, [point]
    while stack:
        for n in position.neighbors(stack.pop()):
            if n not in chain and position.stone_at(n) == color:
                chain.add(n)
                stack.append(n)
    return chain


def liberties_of(position: Position, chain: set[Point]) -> set[Point]:
    return {n for p in chain for n in position.neighbors(p) if position.stone_at(n) == Stone.EMPTY}


class CaptureReader:
    """Reads one target chain (any stone of it, `target`). Results are cached per position and
    depth, so many questions about one fight share the work."""

    def __init__(self, target: Point, escape_liberties: int = ESCAPE_LIBERTIES, node_budget: int | None = None) -> None:
        self.target = target
        self.escape_liberties = escape_liberties
        self.node_budget = node_budget
        self.nodes = 0
        # Per position: (shallowest depth that proved it, deepest depth that didn't).
        self._wins: dict[Position, list[int]] = {}
        self._holds: dict[Position, list[int]] = {}
        self._status: dict[Position, int] = {}

    def _count(self) -> None:
        self.nodes += 1
        if self.node_budget is not None and self.nodes > self.node_budget:
            raise BudgetExceeded()

    def status(self, position: Position) -> int:
        """1 captured, -1 escaped, 0 still in the fight."""
        cached = self._status.get(position)
        if cached is None:
            color = position.stone_at(self.target)
            if color == Stone.EMPTY:
                cached = 1
            else:
                chain = chain_at(position, self.target)
                if len(liberties_of(position, chain)) >= self.escape_liberties:
                    cached = -1
                else:
                    cached = 0
            self._status[position] = cached
        return cached

    def _weak_neighbours(self, position: Position, chain: set[Point], color: Stone, max_libs: int) -> list[set[Point]]:
        """The `color` chains touching `chain` with at most `max_libs` liberties."""
        out, seen = [], set()
        for p in chain:
            for n in position.neighbors(p):
                if n in seen or position.stone_at(n) != color:
                    continue
                other = chain_at(position, n)
                seen |= other
                if len(liberties_of(position, other)) <= max_libs:
                    out.append(other)
        return out

    def defender_points(self, position: Position) -> list[Point]:
        """The defender may play anywhere within distance 2 of the target chain, plus on the
        liberties of attacking chains touching it that have at most two liberties (to capture
        or atari them). Broad on purpose: leaving a defence out would make a capture look forced."""
        chain = chain_at(position, self.target)
        color = position.stone_at(self.target)
        size = position.board_size
        points: set[Point] = set()
        for p in chain:
            for dr in range(-2, 3):
                for dc in range(-2 + abs(dr), 3 - abs(dr)):
                    q = Point(p.row + dr, p.col + dc)
                    if 0 <= q.row < size and 0 <= q.col < size and position.stone_at(q) == Stone.EMPTY:
                        points.add(q)
        for other in self._weak_neighbours(position, chain, color.opponent(), 2):
            points |= liberties_of(position, other)
        liberties = liberties_of(position, chain)
        return sorted(points, key=lambda q: (q not in liberties, q.row, q.col))

    def attacker_points(self, position: Position) -> list[Point]:
        """The attacker's sensible tries: the target's liberties; with two liberties left also the
        points next to them (nets, loose ladders); and saving its own chains in atari next to the
        target (extending, or capturing what ataris them). Narrow on purpose: a missing try can only
        hide a capture, never invent one."""
        chain = chain_at(position, self.target)
        color = position.stone_at(self.target)
        liberties = liberties_of(position, chain)
        points: set[Point] = set(liberties)
        if len(liberties) == 2:
            # The net (geta): a point next to both liberties.
            a, b = liberties
            points |= {n for n in position.neighbors(a) if position.stone_at(n) == Stone.EMPTY and n in position.neighbors(b)}
        for own in self._weak_neighbours(position, chain, color.opponent(), 1):
            points |= liberties_of(position, own)
            for p in own:
                for n in position.neighbors(p):
                    if position.stone_at(n) == color:
                        enemy = chain_at(position, n)
                        enemy_libs = liberties_of(position, enemy)
                        if len(enemy_libs) == 1:
                            points |= enemy_libs
        return sorted(points, key=lambda q: (q not in liberties, q.row, q.col))

    def moves(self, position: Position, defender: bool) -> list[Move]:
        points = self.defender_points(position) if defender else self.attacker_points(position)
        plays: list[Move] = [Play(p) for p in points if position.is_legal(Play(p))]
        if not defender:
            return plays
        # The defender's real escapes first: the moves that leave the target the most liberties.
        def target_liberties(m: Move) -> int:
            after = position.play(m)
            if after.stone_at(self.target) == Stone.EMPTY:
                return -1
            return len(liberties_of(after, chain_at(after, self.target)))
        return [*sorted(plays, key=target_liberties, reverse=True), Pass()]

    @staticmethod
    def _known(memo: dict[Position, list[int]], position: Position, depth: int) -> bool | None:
        entry = memo.get(position)
        if entry is None:
            return None
        if entry[0] <= depth:
            return True
        if entry[1] >= depth:
            return False
        return None

    @staticmethod
    def _store(memo: dict[Position, list[int]], position: Position, depth: int, result: bool) -> bool:
        entry = memo.setdefault(position, [10**9, -1])
        if result:
            entry[0] = min(entry[0], depth)
        else:
            entry[1] = max(entry[1], depth)
        return result

    def _captures(self, position: Position, depth: int) -> bool:
        known = self._known(self._wins, position, depth)
        if known is not None:
            return known
        self._count()
        s = self.status(position)
        if s != 0 or depth <= 0:
            result = s == 1
        else:
            result = any(self._holds_at(position.play(m), depth - 1) for m in self.moves(position, defender=False))
        return self._store(self._wins, position, depth, result)

    def _holds_at(self, position: Position, depth: int) -> bool:
        known = self._known(self._holds, position, depth)
        if known is not None:
            return known
        self._count()
        s = self.status(position)
        if s != 0 or depth <= 0:
            result = s == 1
        else:
            result = all(self._captures(position.play(m), depth - 1) for m in self.moves(position, defender=True))
        return self._store(self._holds, position, depth, result)

    def unconditionally_alive(self, position: Position) -> bool:
        color = position.stone_at(self.target)
        return color != Stone.EMPTY and self.target in _benson_alive(position, color)

    def captures(self, position: Position, depth: int = CAPTURE_DEPTH) -> bool:
        """Attacker (the side to move, not the target's color) to move: it captures by force within
        `depth` plies. Read with growing depth, so a short capture is found without the long lines."""
        if self.unconditionally_alive(position):
            return False
        return any(self._captures(position, d) for d in range(1, depth + 1, 2))

    def holds(self, position: Position, depth: int = CAPTURE_DEPTH) -> bool:
        """Defender to move: every defence still ends in the capture within `depth` plies (true at
        once if the target is already captured)."""
        if self.status(position) == 1:
            return True
        return any(self._holds_at(position, d) for d in range(2, depth + 1, 2))

    def capture_depth(self, position: Position, depth: int = CAPTURE_DEPTH) -> int | None:
        """The fewest plies in which the attacker (to move) captures by force, or None."""
        return next((d for d in range(1, depth + 1, 2) if self._captures(position, d)), None)

    def capturing_moves(self, position: Position, depth: int = CAPTURE_DEPTH) -> list[Point]:
        """The attacker's first moves that capture by force (attacker to move)."""
        if self.status(position) != 0 or self.unconditionally_alive(position):
            return []
        return [m.point for m in self.moves(position, defender=False) if self.holds(position.play(m), depth - 1)]
