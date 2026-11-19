"""Drop puzzles from an existing file whose tree calls a move wrong without proving it loses.

`generate.py` now refuses such puzzles as it builds them (`Unclear`); this applies the same
check to a file made before that, keeping every remaining puzzle exactly as it was (same ids,
same A/B/C order), so a review in progress still matches.

    python -m puzzles.recheck runs/puzzles/puzzles.json --out runs/puzzles/puzzles_rechecked.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.eye_space import BudgetExceeded
from engine.point import Point
from puzzles.generate import SYMBOLS, Fight, Unclear, parse_board, recheck


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("puzzles", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=400_000)
    args = parser.parse_args()

    kept, dropped = [], []
    for p in json.loads(args.puzzles.read_text()):
        player = SYMBOLS[p["toPlay"]]
        position = parse_board(p["board"], player)
        target = Point(*p["target"])
        defender = position.stone_at(target)
        fight = Fight(target, {Point(*rc) for rc in p["region"]}, defender.opponent(), args.budget)
        try:
            recheck(fight, position, p["tree"])
            kept.append(p)
        except (Unclear, BudgetExceeded):
            dropped.append(p["id"])
    args.out.write_text(json.dumps(kept, indent=1))
    print(json.dumps({"kept": len(kept), "dropped": len(dropped), "dropped_ids": dropped}, indent=1))


if __name__ == "__main__":
    main()
