"""Rebuild the wrong-move lines of capture puzzles already made (igo-app/docs/PUZZLES.md, "Capture
puzzles"), with `CaptureFight.refutation`: the answers, the reply trees and the areas stay as they
are, only each wrong move's line changes. Puzzles of other kinds pass through untouched.

    python -m puzzles.reline_capture runs/capture_puzzles/review.json --out runs/capture_puzzles/review_relined.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from collections import Counter
from pathlib import Path

from engine.move import Move, Pass, Play
from engine.point import Point
from puzzles.generate import SYMBOLS, parse_board
from puzzles.generate_capture import DEPTH, CaptureFight

BUDGET = 600_000


def parse_key(text: str) -> Move:
    if text == "pass":
        return Pass()
    r, c = (int(x) for x in text.split(","))
    return Play(Point(r, c))


def parse_step(step) -> Move:
    return Pass() if step == "pass" else Play(Point(*step))


def reline(node: dict, position, fight: CaptureFight) -> None:
    """Every wrong move's line below `node` (the player to move at `position`) rebuilt in place."""
    if node.get("solved"):
        return
    for move_key in node.get("bad", {}):
        node["bad"][move_key] = fight.refutation(position.play(parse_key(move_key)), DEPTH)
    for move_key, step in node.get("good", {}).items():
        if "next" in step:
            after = position.play(parse_key(move_key))
            reline(step["next"], after.play(parse_step(step["reply"])), fight)


def lines(node: dict) -> list[list]:
    out = list(node.get("bad", {}).values())
    for step in node.get("good", {}).values():
        if "next" in step:
            out += lines(step["next"])
    return out


def work(puzzle: dict) -> dict:
    if puzzle["kind"] != "CAPTURE":
        return puzzle
    attacker = SYMBOLS[puzzle["toPlay"]]
    fight = CaptureFight(Point(*puzzle["target"]), {Point(*p) for p in puzzle["region"]}, attacker, BUDGET)
    puzzle = json.loads(json.dumps(puzzle))
    reline(puzzle["tree"], parse_board(puzzle["board"], attacker), fight)
    return puzzle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("puzzles", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    before = json.loads(args.puzzles.read_text(encoding="utf-8"))
    with multiprocessing.Pool(args.workers) as pool:
        after = pool.map(work, before)
    args.out.write_text(json.dumps(after, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    old = [len(l) for p in before if p["kind"] == "CAPTURE" for l in lines(p["tree"])]
    new = [len(l) for p in after if p["kind"] == "CAPTURE" for l in lines(p["tree"])]
    print(f"{len(old)} wrong-move lines in {sum(p['kind'] == 'CAPTURE' for p in before)} capture puzzles")
    print("length before", dict(sorted(Counter(old).items())), f"mean {sum(old) / len(old):.1f}")
    print("length after ", dict(sorted(Counter(new).items())), f"mean {sum(new) / len(new):.1f}")


if __name__ == "__main__":
    main()
