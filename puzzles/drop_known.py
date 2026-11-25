"""Drop candidate shapes an earlier spike already found (igo-app/docs/PUZZLES.md, refill).

`puzzles.spike_mine` counts a shape once per run (its `shape_key`: the fight's area and border,
the defender as Black, under the 8 symmetries). A refill's new games repeat many shapes the first
run already turned into puzzles; this rebuilds each candidate's key from its board and keeps only
the new candidates whose key none of the `--known` candidates has.

    python -m puzzles.drop_known runs/puzzle_refill_spike/candidates.json \\
        --known runs/puzzle_spike/candidates.json --out runs/puzzle_refill_spike/new_candidates.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.point import Point
from engine.position import Position
from engine.stone import Stone
from puzzles.spike_mine import components, shape_key

STONES = {"X": Stone.BLACK, "O": Stone.WHITE, ".": Stone.EMPTY}


def candidate_key(candidate: dict) -> tuple:
    """The candidate's `shape_key`: its area is the non-attacker component holding the target."""
    rows = candidate["board"]
    size = len(rows)
    position = Position(board_size=size, stones=tuple(STONES[ch] for row in rows for ch in row), to_play=Stone.BLACK)
    defender = STONES[candidate["defender"]]
    attacker = defender.opponent()
    target = Point(*candidate["target"])
    area = next(c for c in components(position, lambda p: position.stone_at(p) != attacker) if target in c)
    return shape_key(position, area, defender)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--known", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    known = {candidate_key(c) for path in args.known for c in json.loads(path.read_text())}
    candidates = json.loads(args.candidates.read_text())
    new = [c for c in candidates if candidate_key(c) not in known]
    args.out.write_text(json.dumps(new, indent=1))
    print(f"{len(candidates)} candidates, {len(candidates) - len(new)} already known, {len(new)} new -> {args.out}")


if __name__ == "__main__":
    main()
