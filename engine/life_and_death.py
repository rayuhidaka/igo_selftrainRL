"""Port of igo-app/engine/LifeAndDeath.kt (dead-stone handling at the end of a game) -- see
engine/README.md. Used by mcts/dead_stone_estimator.py.

Kotlin's `Position` extension functions become plain functions here, as in engine/scoring.py.
`marking_cluster`/`toggle_dead` aren't ported: they serve igo-app's marking screen only.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from engine.point import Point
from engine.position import Position
from engine.scoring import territory_ownership
from engine.stone import Stone


def unconditionally_alive(position: Position) -> set[Point]:
    """Every stone no sequence of opponent moves can capture, even if its owner passes every
    turn (Benson's algorithm), for both colors. A lower bound on life: it only ever rules
    stones *out* of being dead. See `LifeAndDeath.kt`'s `unconditionallyAlive` for the steps."""
    return _benson_alive(position, Stone.BLACK) | _benson_alive(position, Stone.WHITE)


def _benson_alive(position: Position, color: Stone) -> set[Point]:
    chains = _components(position, lambda p: position.stone_at(p) == color)
    regions = _components(position, lambda p: position.stone_at(p) != color)
    while True:
        alive_chains = []
        for chain in chains:
            liberties = {n for p in chain for n in position.neighbors(p) if position.stone_at(n) == Stone.EMPTY}
            vital = 0
            for region in regions:
                empties = [p for p in region if position.stone_at(p) == Stone.EMPTY]
                if empties and all(p in liberties for p in empties):
                    vital += 1
            if vital >= 2:
                alive_chains.append(chain)
        chain_stones = {p for chain in alive_chains for p in chain}
        healthy_regions = [
            region
            for region in regions
            if all(n in chain_stones for p in region for n in position.neighbors(p) if position.stone_at(n) == color)
        ]
        if len(alive_chains) == len(chains) and len(healthy_regions) == len(regions):
            return chain_stones
        chains, regions = alive_chains, healthy_regions


def _components(position: Position, include: Callable[[Point], bool]) -> list[set[Point]]:
    seen: set[Point] = set()
    components = []
    for index in range(position.board_size * position.board_size):
        start = Point.from_index(index, position.board_size)
        if start in seen or not include(start):
            continue
        found = {start}
        stack = [start]
        while stack:
            for neighbor in position.neighbors(stack.pop()):
                if neighbor not in found and include(neighbor):
                    found.add(neighbor)
                    stack.append(neighbor)
        seen |= found
        components.append(found)
    return components


def without_dead_stones(position: Position, dead: set[Point]) -> Position:
    """`position` with every stone in `dead` taken off the board, as if captured."""
    size = position.board_size
    stones = [
        Stone.EMPTY if Point.from_index(i, size) in dead else stone for i, stone in enumerate(position.stones)
    ]
    return dataclasses.replace(position, stones=stones)


def estimate_dead_stones(position: Position, final_positions: list[Position], threshold: float = 0.5) -> set[Point]:
    """The stones of `position` that `final_positions` (games played out from it) say are dead:
    a chain whose points end up as the other color's area in more than `threshold` of the
    playouts, on average. Unconditionally alive chains never are. See `LifeAndDeath.kt`'s
    `estimateDeadStones`."""
    if not final_positions:
        return set()
    for final in final_positions:
        if final.board_size != position.board_size:
            raise ValueError(f"Playout board size {final.board_size} doesn't match {position.board_size}")
    alive = unconditionally_alive(position)
    ownerships = [territory_ownership(final) for final in final_positions]
    chains = _components(position, lambda p: position.stone_at(p) == Stone.BLACK) + _components(
        position, lambda p: position.stone_at(p) == Stone.WHITE
    )
    dead: set[Point] = set()
    for chain in chains:
        if chain & alive:
            continue
        opponent = position.stone_at(next(iter(chain))).opponent()
        lost = sum(sum(1 for p in chain if o[p] == opponent) / len(chain) for o in ownerships) / len(ownerships)
        if lost > threshold:
            dead |= chain
    return dead
