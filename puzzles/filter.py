"""Thin out a puzzle file so no two puzzles feel like the same one, keeping ids.

The first generator run asked most fights twice: the live and the kill version of one board
always share the answer point (80 of 80 pairs), and a fight caught a few moves apart in the
same game shows up again with most of its area in common. The publisher noticed while
reviewing ("pretty easy, and I have seen a few of those by now").

Rules, in order:

1. **One puzzle per fight.** Puzzles from the same game whose areas share a point are one
   fight (that covers the live/kill pair, which has the same area). One per fight survives.
2. **The review wins.** A puzzle the publisher kept is never dropped -- if two of them are the
   same fight, both stay and the report lists them. A dropped one always goes.
3. **Fewer easy ones** (``--easy-share``): if difficulty 1 is still more than that share,
   the easy puzzles whose empty eye shape is most common (the bent and straight threes) go
   first, smallest area first.

Within a fight the survivor is: a kept puzzle; else the harder one (the nets' difficulty),
then the bigger reply tree (more to read), then the lower id. Nothing else about a puzzle
changes, so a review in progress still matches.

    python -m puzzles.filter runs/puzzles/puzzles_rechecked.json \\
        --verdicts verdicts.json --easy-share 0.4 --out runs/puzzles/puzzles_filtered.json

``verdicts.json`` maps puzzle id to ``keep``, ``drop`` or ``none`` (the review page's
verdicts collection).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

BOARD_SIZE = 9


def game_of(puzzle: dict) -> str:
    """The recorded game a puzzle came from: its ``source`` without the move number."""
    return puzzle["source"].split("@")[0]


def fights(puzzles: list[dict]) -> list[list[dict]]:
    """Groups puzzles into fights: same game and overlapping areas, joined transitively."""
    parent = list(range(len(puzzles)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_game: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(puzzles):
        by_game[game_of(p)].append(i)
    for members in by_game.values():
        for a_pos, a in enumerate(members):
            area_a = {tuple(rc) for rc in puzzles[a]["region"]}
            for b in members[a_pos + 1:]:
                if area_a & {tuple(rc) for rc in puzzles[b]["region"]}:
                    parent[find(a)] = find(b)
    groups: dict[int, list[dict]] = defaultdict(list)
    for i, p in enumerate(puzzles):
        groups[find(i)].append(p)
    return list(groups.values())


def eye_shape(puzzle: dict, size: int = BOARD_SIZE) -> tuple:
    """The empty points of a puzzle's area, moved to the origin, the same under all 8 symmetries."""
    board = puzzle["board"]
    points = [tuple(rc) for rc in puzzle["region"] if board[rc[0]][rc[1]] == "."]
    best = None
    for k in range(8):
        moved = []
        for r, c in points:
            if k & 4:
                r, c = c, r
            if k & 1:
                r = size - 1 - r
            if k & 2:
                c = size - 1 - c
            moved.append((r, c))
        top = min((r for r, _ in moved), default=0)
        left = min((c for _, c in moved), default=0)
        key = tuple(sorted((r - top, c - left) for r, c in moved))
        if best is None or key < best:
            best = key
    return best


def survivor_rank(puzzle: dict, verdicts: dict[str, str]) -> tuple:
    """Sort key for who survives a fight (smallest wins)."""
    return (
        verdicts.get(puzzle["id"]) != "keep",
        -puzzle["difficulty"],
        -puzzle.get("treeNodes", 0),
        puzzle["id"],
    )


def filter_puzzles(puzzles: list[dict], verdicts: dict[str, str], easy_share: float | None) -> tuple[list[dict], dict]:
    """Applies the module's rules; returns the surviving puzzles (in file order) and a report."""
    candidates = [p for p in puzzles if verdicts.get(p["id"]) != "drop"]
    report: dict = {"input": len(puzzles), "dropped_by_review": len(puzzles) - len(candidates)}

    keep_ids: set[str] = set()
    kept_twins = []
    groups = fights(candidates)
    for group in groups:
        reviewed = [p["id"] for p in group if verdicts.get(p["id"]) == "keep"]
        if len(reviewed) > 1:
            kept_twins.append(sorted(reviewed))
            keep_ids.update(reviewed)
        else:
            keep_ids.add(min(group, key=lambda p: survivor_rank(p, verdicts))["id"])
    report["fights"] = len(groups)
    report["kept_same_fight"] = kept_twins

    survivors = [p for p in candidates if p["id"] in keep_ids]
    if easy_share is not None:
        shapes = Counter(eye_shape(p) for p in survivors)
        easy = [p for p in survivors if p["difficulty"] == 1 and verdicts.get(p["id"]) != "keep"]
        easy.sort(key=lambda p: (-shapes[eye_shape(p)], len(p["region"]), p["id"]))
        dropped_easy = set()
        while easy and sum(p["difficulty"] == 1 for p in survivors if p["id"] not in dropped_easy) > \
                easy_share * (len(survivors) - len(dropped_easy)):
            dropped_easy.add(easy.pop(0)["id"])
        survivors = [p for p in survivors if p["id"] not in dropped_easy]
        report["dropped_as_easy"] = len(dropped_easy)

    report["output"] = len(survivors)
    report["by_difficulty"] = dict(sorted(Counter(p["difficulty"] for p in survivors).items()))
    report["by_kind"] = dict(Counter(p["kind"] for p in survivors))
    report["distinct_eye_shapes"] = len({eye_shape(p) for p in survivors})
    report["dropped_ids"] = sorted({p["id"] for p in puzzles} - {p["id"] for p in survivors})
    return survivors, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("puzzles", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, help="JSON object: puzzle id -> keep | drop | none")
    parser.add_argument("--easy-share", type=float, default=None, help="most difficulty-1 puzzles allowed, e.g. 0.4")
    args = parser.parse_args()

    verdicts = json.loads(args.verdicts.read_text()) if args.verdicts else {}
    survivors, report = filter_puzzles(json.loads(args.puzzles.read_text()), verdicts, args.easy_share)
    args.out.write_text(json.dumps(survivors, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
