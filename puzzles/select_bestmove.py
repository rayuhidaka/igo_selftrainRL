"""Choose the best-move puzzles for the publisher's review (igo-app/docs/PUZZLES.md, C8): about
`--count`, balanced over the kinds (saves stones / capture or atari / quiet) and the levels 1-4,
round robin, the largest gap first within each group.

A puzzle is dropped first if the app would let the player make a move the file has no chance for:
the board in the file has no ko history, so a point that was a ko in the game is legal there.

    python -m puzzles.select_bestmove runs/bestmove_puzzles/puzzles.json --out runs/bestmove_puzzles/review.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from engine.move import Play
from engine.point import Point
from engine.stone import Stone
from puzzles.generate import SYMBOLS, parse_board


def fits_the_app(p: dict) -> bool:
    position = parse_board(p["board"], SYMBOLS[p["toPlay"]])
    size = position.board_size
    for i in range(size * size):
        point = Point.from_index(i, size)
        if position.stone_at(point) == Stone.EMPTY and position.is_legal(Play(point)) and f"{point.row},{point.col}" not in p["values"]:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("puzzles", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=150)
    args = parser.parse_args()

    puzzles = json.loads(args.puzzles.read_text())
    fitting = [p for p in puzzles if fits_the_app(p)]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for p in sorted(fitting, key=lambda p: -p["gap"]):
        groups[(p["technique"], p["difficulty"])].append(p)
    chosen: list[dict] = []
    while len(chosen) < args.count and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(chosen) < args.count:
                chosen.append(groups[key].pop(0))
    chosen.sort(key=lambda p: p["id"])
    args.out.write_text(json.dumps(chosen, indent=1))
    print(f"{len(puzzles)} puzzles, {len(fitting)} fit the app, {len(chosen)} chosen")
    print("  kinds:", dict(Counter(p["technique"] for p in chosen)))
    print("  levels:", dict(sorted(Counter(p["difficulty"] for p in chosen).items())))
    print("  gaps:", dict(sorted(Counter(min(int(p["gap"] * 10) * 10, 50) for p in chosen).items())))


if __name__ == "__main__":
    main()
