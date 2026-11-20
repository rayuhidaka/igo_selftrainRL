"""Clear each puzzle's board down to its fight, keeping ids and answers.

The publisher, reviewing (2026-11-20): the game boards are often so crowded that it's hard to
see where the player is meant to play at all. A puzzle's fight only involves

- **the area**: the walled-in points holding the target -- the defender's stones there and the
  empty points both sides play in (the puzzle's ``region``);
- **the wall**: the attacker chains touching the area. Each keeps its stones within
  ``WALL_REACH`` (2) steps of the area -- a solid wall, without the tail that runs across the
  board -- plus the fewest stones that keep it one chain, with at least ``--wall-liberties``
  liberties outside the area (the generator's rule).

Everything else is removed: the rest of the game goes, the corner fight stays, as in a problem
book.

**Why the answer can't change.** Both sides play only inside the region. Every wall stone
next to the area stays, so the area is still closed, and each wall chain is still one chain
with at least ``--wall-liberties`` liberties outside, which the defender can't fill -- the wall
can't be captured. "Alive" for the defender's chains in the area (Benson) only looks at regions
inside the area: any region that reaches outside goes through the wall and holds the wall's
outside liberties, which touch no defender stone in the area, so it never counts as an eye.
Nothing else on the board takes part.

That's the argument; the check is the reader. Each cleaned board gets its reply tree rebuilt
from scratch, and the puzzle is cleaned only if the new tree is exactly the old one (every
answer, reply, refutation and "solved"). Otherwise it keeps its full board and the report lists
it. igo-app's PuzzlesTest then re-proves every puzzle with the Kotlin reader, as always.

The game's last moves (``history``) keep only the stones still on the board; ``fullBoard``
keeps the game position for tracing (the app ignores it).

    python -m puzzles.clean runs/puzzles/puzzles_filtered.json --out runs/puzzles/puzzles_clean.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import deque
from pathlib import Path

from engine.eye_space import EYE_SPACE_DEPTH, BudgetExceeded
from engine.point import Point
from engine.position import Position
from engine.stone import Stone
from puzzles.generate import SYMBOLS, Fight, TreeTooBig, Unclear, build_tree, fight_area, parse_board, weakest_wall

ROW_SYMBOL = {Stone.BLACK: "X", Stone.WHITE: "O", Stone.EMPTY: "."}

# How far from the area wall stones stay: two steps keeps a solid wall, drops the far tails.
WALL_REACH = 2


def chain_of(position: Position, start: Point) -> set[Point]:
    """The chain holding the stone at `start`."""
    color = position.stone_at(start)
    chain = {start}
    stack = [start]
    while stack:
        for n in position.neighbors(stack.pop()):
            if n not in chain and position.stone_at(n) == color:
                chain.add(n)
                stack.append(n)
    return chain


def distances(position: Position, area: set[Point]) -> dict[Point, int]:
    """Each point's distance in steps from the nearest point of `area` (stones don't block)."""
    dist = {p: 0 for p in area}
    frontier = list(area)
    while frontier:
        nxt = []
        for p in frontier:
            for n in position.neighbors(p):
                if n not in dist:
                    dist[n] = dist[p] + 1
                    nxt.append(n)
        frontier = nxt
    return dist


def pieces(position: Position, stones: set[Point]) -> list[set[Point]]:
    """`stones` split into its connected pieces."""
    left = set(stones)
    out = []
    while left:
        start = left.pop()
        piece = {start}
        stack = [start]
        while stack:
            for n in position.neighbors(stack.pop()):
                if n in left:
                    left.discard(n)
                    piece.add(n)
                    stack.append(n)
        out.append(piece)
    return out


def joined(position: Position, part: set[Point], chain: set[Point]) -> set[Point]:
    """`part` of `chain` made one chain again: while it's in pieces, the shortest run of chain
    stones from the biggest piece to another one is put back."""
    part = set(part)
    while len(split := pieces(position, part)) > 1:
        start = max(split, key=lambda piece: (len(piece), min((p.row, p.col) for p in piece)))
        came_from: dict[Point, Point | None] = {p: None for p in start}
        queue = deque(sorted(start, key=lambda p: (p.row, p.col)))
        while queue:
            p = queue.popleft()
            if p in part and p not in start:
                break
            for n in position.neighbors(p):
                if n in chain and n not in came_from:
                    came_from[n] = p
                    queue.append(n)
        while p not in start:
            part.add(p)
            p = came_from[p]
    return part


def outside_liberties(position: Position, stones: set[Point], area: set[Point]) -> int:
    """`stones`' liberties outside `area` once everything else outside the area is cleared."""
    return len({n for p in stones for n in position.neighbors(p) if n not in stones and n not in area})


def trimmed(position: Position, chain: set[Point], area: set[Point], dist: dict[Point, int],
            wall_liberties: int) -> set[Point]:
    """The part of a wall chain within `WALL_REACH` steps of the area, plus the fewest stones
    that keep it one chain (`joined`); reaching further out only if it would have fewer than
    `wall_liberties` liberties outside the area."""
    for reach in range(WALL_REACH, 2 * position.board_size):
        part = joined(position, {p for p in chain if dist[p] <= reach}, chain)
        if outside_liberties(position, part, area) >= wall_liberties:
            return part
    return chain


def fight_stones(position: Position, target: Point, attacker: Stone, wall_liberties: int = 3) -> set[Point]:
    """The stones the fight involves: the defender's stones in the area holding `target`, and
    the attacker's wall around it -- each chain touching the area, trimmed to the stones near it
    (`trimmed`)."""
    area = fight_area(position, target, attacker)
    dist = distances(position, area)
    keep = {p for p in area if position.stone_at(p) != Stone.EMPTY}
    done: set[Point] = set()
    for p in area:
        for n in position.neighbors(p):
            if position.stone_at(n) == attacker and n not in done:
                chain = chain_of(position, n)
                done |= chain
                keep |= trimmed(position, chain, area, dist, wall_liberties)
    return keep


def cleared(position: Position, keep: set[Point]) -> Position:
    """`position` with every stone not in `keep` removed."""
    size = position.board_size
    stones = [s if Point.from_index(i, size) in keep else Stone.EMPTY for i, s in enumerate(position.stones)]
    return dataclasses.replace(position, stones=stones)


def rows(position: Position) -> list[str]:
    size = position.board_size
    return ["".join(ROW_SYMBOL[position.stone_at(Point(r, c))] for c in range(size)) for r in range(size)]


def clean_puzzle(puzzle: dict, budget: int = 400_000, max_nodes: int = 300,
                 wall_liberties: int = 3) -> tuple[dict, str | None]:
    """The puzzle on a cleared board, or the puzzle unchanged plus the reason it couldn't be."""
    player = SYMBOLS[puzzle["toPlay"]]
    full = parse_board(puzzle["board"], player)
    target = Point(*puzzle["target"])
    attacker = full.stone_at(target).opponent()
    region = {Point(*rc) for rc in puzzle["region"]}
    clean = cleared(full, fight_stones(full, target, attacker, wall_liberties))
    if clean.stones == full.stones:
        return puzzle, "nothing to remove"

    area = fight_area(clean, target, attacker)
    if {p for p in area if clean.stone_at(p) == Stone.EMPTY} != region:
        return puzzle, "area changed"  # a guard: the wall is kept whole, so this can't happen
    if weakest_wall(clean, area, attacker) < wall_liberties:
        return puzzle, "wall weaker"  # a guard: `trimmed` keeps the liberties
    try:
        tree = build_tree(Fight(target, region, attacker, budget), clean, EYE_SPACE_DEPTH, [0], max_nodes)
    except (TreeTooBig, Unclear, BudgetExceeded) as e:
        return puzzle, f"reader: {type(e).__name__}"
    if tree != puzzle["tree"]:
        return puzzle, "tree changed"

    history = [h for h in puzzle.get("history", [])
               if clean.stone_at(Point(*h["point"])) == SYMBOLS[h["color"]]]
    return {**puzzle, "board": rows(clean), "history": history, "fullBoard": puzzle["board"]}, None


def stone_count(board: list[str]) -> int:
    return sum(ch != "." for line in board for ch in line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("puzzles", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=400_000)
    parser.add_argument("--max-nodes", type=int, default=300)
    parser.add_argument("--wall-liberties", type=int, default=3)
    args = parser.parse_args()

    started = time.time()
    out, kept_full = [], {}
    before = after = 0
    for puzzle in json.loads(args.puzzles.read_text()):
        cleaned, reason = clean_puzzle(puzzle, args.budget, args.max_nodes, args.wall_liberties)
        if reason is not None:
            kept_full[puzzle["id"]] = reason
        before += stone_count(puzzle["board"])
        after += stone_count(cleaned["board"])
        out.append(cleaned)
        print(f"{puzzle['id']}: {stone_count(puzzle['board'])} -> {stone_count(cleaned['board'])} stones"
              f"{'' if reason is None else '  (kept: ' + reason + ')'}", flush=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({
        "puzzles": len(out),
        "cleaned": len(out) - len(kept_full),
        "kept_full_board": kept_full,
        "stones_before": before,
        "stones_after": after,
        "seconds": round(time.time() - started),
    }, indent=1))


if __name__ == "__main__":
    main()
