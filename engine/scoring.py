"""Port of igo-app/engine/Scoring.kt -- see engine/README.md.

Kotlin's `Position.areaScore()` extension function becomes a plain
`area_score(position)` function here -- Python has no extension-function
equivalent worth emulating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from engine.point import Point
from engine.position import Position
from engine.stone import Stone


@dataclass(frozen=True)
class AreaScore:
    """A Tromp-Taylor area score for both colors, excluding komi: each color's stones on the
    board plus every empty region bordered exclusively by that color.
    """

    black: int
    white: int

    def winner(self, komi: float) -> Optional[Stone]:
        """Returns the winner once `komi` is added to white's score, or `None` for an exact tie."""
        white_total = self.white + komi
        if self.black > white_total:
            return Stone.BLACK
        if white_total > self.black:
            return Stone.WHITE
        return None


def area_score(position: Position) -> AreaScore:
    """Computes `position`'s Tromp-Taylor area score: each color's stones on the board, plus
    every empty region whose bordering stones are exclusively that color. An empty region
    bordered by both colors (dame) counts for neither.

    Tromp-Taylor scoring assumes dead stones have already been resolved (e.g. by playing
    captures out to completion) -- no dead-stone-removal heuristic is applied here.
    """
    black = 0
    white = 0
    visited: set[Point] = set()

    for index in range(position.board_size * position.board_size):
        point = Point.from_index(index, position.board_size)
        if point in visited:
            continue
        stone = position.stone_at(point)
        if stone == Stone.BLACK:
            black += 1
        elif stone == Stone.WHITE:
            white += 1
        else:
            region = _flood_fill_empty_region(position, point, visited)
            border_colors = {
                position.stone_at(neighbor)
                for member in region
                for neighbor in position.neighbors(member)
                if position.stone_at(neighbor) != Stone.EMPTY
            }
            if border_colors == {Stone.BLACK}:
                black += len(region)
            elif border_colors == {Stone.WHITE}:
                white += len(region)
            # else: neutral (dame) or fully enclosed empty board -- counts for neither

    return AreaScore(black, white)


def _flood_fill_empty_region(position: Position, start: Point, visited: set[Point]) -> set[Point]:
    region = {start}
    visited.add(start)
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in position.neighbors(current):
            if neighbor not in visited and position.stone_at(neighbor) == Stone.EMPTY:
                visited.add(neighbor)
                region.add(neighbor)
                stack.append(neighbor)
    return region
