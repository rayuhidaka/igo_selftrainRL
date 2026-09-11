"""Port of igo-app/engine/Scoring.kt -- see engine/README.md.

Kotlin's `Position.areaScore()`/`Position.territoryOwnership()` extension
functions become plain `area_score(position)`/`territory_ownership(position)`
functions here -- Python has no extension-function equivalent worth
emulating. `ownership_plane` has no Kotlin counterpart -- it's a
training-data-specific helper (see its docstring) added when
`bootstrap/model.py` gained a spatial ownership auxiliary head
(2026-09-11).
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

    Implemented in terms of `territory_ownership`, mirroring `igo-app/engine/Scoring.kt`'s
    own `areaScore()`/`territoryOwnership()` structure exactly (the two had drifted --
    this file used to have its own separate flood-fill pass -- even though they still
    agreed numerically; consolidating avoids two independently-maintained implementations
    of the same walk).
    """
    black = 0
    white = 0
    for owner in territory_ownership(position).values():
        if owner == Stone.BLACK:
            black += 1
        elif owner == Stone.WHITE:
            white += 1
        # else: neutral (dame) or a fully enclosed empty board -- counts for neither
    return AreaScore(black, white)


def territory_ownership(position: Position) -> dict[Point, Optional[Stone]]:
    """Returns, for every point on the board, which color's area it counts toward under
    Tromp-Taylor scoring: the stone's own color if occupied, the bordering color if it's
    part of an empty region surrounded exclusively by one color, or `None` if it's an empty
    point that doesn't count for either side (dame bordered by both colors, or a fully
    enclosed empty board). `area_score` tallies exactly this into totals; this exposes the
    same read per point -- e.g. for a spatial training target (see
    `bootstrap/model.py`'s `has_ownership_head`) or a UI to shade the board.

    Shares `area_score`'s Tromp-Taylor assumption: dead stones must already be resolved (e.g.
    by playing captures out), or they'll show as counting for their own color, not their
    killer's. Port of `igo-app/engine/Scoring.kt`'s `territoryOwnership()` -- see
    `engine/README.md`.
    """
    ownership: dict[Point, Optional[Stone]] = {}
    visited: set[Point] = set()

    for index in range(position.board_size * position.board_size):
        point = Point.from_index(index, position.board_size)
        if point in visited:
            continue
        stone = position.stone_at(point)
        if stone == Stone.EMPTY:
            region = _flood_fill_empty_region(position, point, visited)
            border_colors = {
                position.stone_at(neighbor)
                for member in region
                for neighbor in position.neighbors(member)
                if position.stone_at(neighbor) != Stone.EMPTY
            }
            if border_colors == {Stone.BLACK}:
                owner: Optional[Stone] = Stone.BLACK
            elif border_colors == {Stone.WHITE}:
                owner = Stone.WHITE
            else:
                owner = None
            for region_point in region:
                ownership[region_point] = owner
        else:
            visited.add(point)
            ownership[point] = stone

    return ownership


def ownership_plane(ownership: dict[Point, Optional[Stone]], board_size: int, perspective: Stone) -> list[float]:
    """Converts `territory_ownership`'s per-point dict into a flat, row-major list (matching
    every other flat board array in this codebase, e.g. `selfplay/self_play.py`'s
    `visit_count_policy`) from `perspective`'s point of view: `1.0` where `perspective` owns
    that point, `-1.0` where the opponent does, `0.0` for neutral/dame -- matching
    `RayZeroNet`'s tanh-bounded ownership head convention (`bootstrap/model.py`).

    Plain Python, no numpy: `engine/` stays dependency-light, matching its Kotlin
    counterpart (see `engine/README.md`) -- callers that need an array wrap this themselves.
    """
    opponent = perspective.opponent()
    plane = [0.0] * (board_size * board_size)
    for point, owner in ownership.items():
        if owner == perspective:
            value = 1.0
        elif owner == opponent:
            value = -1.0
        else:
            value = 0.0
        plane[point.to_index(board_size)] = value
    return plane


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
