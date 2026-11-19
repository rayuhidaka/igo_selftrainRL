"""Spike: how many life-and-death puzzles hide in the games we already recorded?

Step 1 of igo-app's puzzle plan (igo-app/docs/PUZZLES.md, "Phase 1 build plan"). Replays every
game in the given `eval/*games.json` files and, after every move, looks for a small walled-in
fight whose outcome depends on who moves first:

- a component S of points that are empty or the defender's (4-connected), bounded only by the
  attacker's stones and the edge, holding at least one defender stone, at most `--max-empty`
  empty points;
- every attacker chain touching S has at least two liberties outside S (its wall can't simply
  be captured from outside -- the reader assumes that);
- the defender's largest chain in S isn't already unconditionally alive;
- the attacker, moving first, kills it, and the defender, moving first, lives
  (`engine/eye_space.py`, exact).

Each such shape gives two puzzles: "play and live" and "play and kill". Shapes are counted once
under the 8 board symmetries and the color swap. Writes the candidates as JSON and a text sample
to look at by eye.

    python -m puzzles.spike_mine eval/*games.json --out runs/puzzle_spike
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import Counter
from pathlib import Path

from engine.eye_space import BudgetExceeded, EyeSpaceReader, EYE_SPACE_DEPTH
from engine.life_and_death import _benson_alive
from engine.move import Pass, Play
from engine.point import Point
from engine.position import Position
from engine.stone import Stone

SYMBOL = {Stone.BLACK: "X", Stone.WHITE: "O", Stone.EMPTY: "."}


def decode_history(history: list[int], board_size: int):
    return [Pass() if i < 0 else Play(Point.from_index(i, board_size)) for i in history]


def components(position: Position, include) -> list[set[Point]]:
    size = position.board_size
    seen: set[Point] = set()
    out = []
    for i in range(size * size):
        start = Point.from_index(i, size)
        if start in seen or not include(start):
            continue
        comp = {start}
        stack = [start]
        seen.add(start)
        while stack:
            for n in position.neighbors(stack.pop()):
                if n not in seen and include(n):
                    seen.add(n)
                    comp.add(n)
                    stack.append(n)
        out.append(comp)
    return out


def chain(position: Position, start: Point) -> set[Point]:
    color = position.stone_at(start)
    return next(c for c in components(position, lambda p: position.stone_at(p) == color) if start in c)


def wall_is_safe(position: Position, area: set[Point], attacker: Stone) -> bool:
    """Every attacker chain touching `area` keeps two liberties outside it."""
    touching = {n for p in area for n in position.neighbors(p) if position.stone_at(n) == attacker}
    done: set[Point] = set()
    for stone in touching:
        if stone in done:
            continue
        ch = chain(position, stone)
        done |= ch
        outside = {
            n for p in ch for n in position.neighbors(p) if position.stone_at(n) == Stone.EMPTY and n not in area
        }
        if len(outside) < 2:
            return False
    return True


def transforms(size: int):
    m = size - 1
    return [
        lambda r, c: (r, c), lambda r, c: (c, m - r), lambda r, c: (m - r, m - c), lambda r, c: (m - c, r),
        lambda r, c: (r, m - c), lambda r, c: (m - r, c), lambda r, c: (c, r), lambda r, c: (m - c, m - r),
    ]


def shape_key(position: Position, area: set[Point], defender: Stone) -> tuple:
    """The fight's local shape (S and its border), with the defender as Black, under the 8
    symmetries -- the same shape anywhere on the board, either color, counts once."""
    window = area | {n for p in area for n in position.neighbors(p)}
    norm = {defender: 1, defender.opponent(): 2, Stone.EMPTY: 0}
    best = None
    for t in transforms(position.board_size):
        key = tuple(sorted((*t(p.row, p.col), norm[position.stone_at(p)]) for p in window))
        if best is None or key < best:
            best = key
    return best


def rows(position: Position) -> list[str]:
    s = position.board_size
    return ["".join(SYMBOL[position.stone_at(Point(r, c))] for c in range(s)) for r in range(s)]


def fresh(position: Position, to_play: Stone) -> Position:
    return dataclasses.replace(position, to_play=to_play, ko_point=None, pass_count=0)


def analyse(position: Position, area: set[Point], defender: Stone, budget: int) -> dict | None:
    attacker = defender.opponent()
    region = {p for p in area if position.stone_at(p) == Stone.EMPTY}
    defender_chains = components(position, lambda p: p in area and position.stone_at(p) == defender)
    target = min(max(defender_chains, key=len), key=lambda p: (p.row, p.col))
    if target in _benson_alive(position, defender):
        return None
    reader = EyeSpaceReader(target, region, budget)
    kill_first = fresh(position, attacker)
    live_first = fresh(position, defender)
    if not reader.kill_wins(kill_first, EYE_SPACE_DEPTH):
        return None
    if not reader.life_wins(live_first, EYE_SPACE_DEPTH):
        return None
    d = EYE_SPACE_DEPTH - 1
    kill_answers = [p for p in sorted(region, key=lambda p: (p.row, p.col))
                    if kill_first.is_legal(Play(p)) and reader.kill_holds(kill_first.play(Play(p)), d)]
    live_answers = [p for p in sorted(region, key=lambda p: (p.row, p.col))
                    if live_first.is_legal(Play(p)) and reader.life_holds(live_first.play(Play(p)), d)]
    return {
        "defender": SYMBOL[defender],
        "target": [target.row, target.col],
        "region": [[p.row, p.col] for p in sorted(region, key=lambda p: (p.row, p.col))],
        "kill_answers": [[p.row, p.col] for p in kill_answers],
        "live_answers": [[p.row, p.col] for p in live_answers],
        "nodes": reader.nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("games", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-empty", type=int, default=10)
    parser.add_argument("--max-area", type=int, default=24)
    parser.add_argument("--min-move", type=int, default=10)
    parser.add_argument("--budget", type=int, default=200_000)
    parser.add_argument("--limit-games", type=int, default=0)
    args = parser.parse_args()

    seen: dict[tuple, dict | None] = {}
    found: list[dict] = []
    stats = Counter()
    started = time.time()
    games_done = 0
    for path in args.games:
        for game_index, game in enumerate(json.loads(path.read_text())):
            if args.limit_games and games_done >= args.limit_games:
                break
            games_done += 1
            size = game.get("board_size", 9)
            position = Position.empty(size)
            for move_number, move in enumerate(decode_history(game["history"], size), start=1):
                position = position.play(move)
                if move_number < args.min_move:
                    continue
                for defender in (Stone.BLACK, Stone.WHITE):
                    attacker = defender.opponent()
                    for area in components(position, lambda p: position.stone_at(p) != attacker):
                        empties = sum(1 for p in area if position.stone_at(p) == Stone.EMPTY)
                        if not 1 <= empties <= args.max_empty or len(area) > args.max_area or empties == len(area):
                            continue
                        stats["areas"] += 1
                        if not wall_is_safe(position, area, attacker):
                            stats["weak wall"] += 1
                            continue
                        key = shape_key(position, area, defender)
                        if key in seen:
                            stats["repeat"] += 1
                            continue
                        try:
                            result = analyse(position, area, defender, args.budget)
                        except BudgetExceeded:
                            stats["over budget"] += 1
                            result = None
                        seen[key] = result
                        stats["shapes read"] += 1
                        if result is None:
                            continue
                        result.update(
                            source=f"{path.name}#{game_index}@{move_number}",
                            matchup=f"{game['black']} vs {game['white']}",
                            board=rows(position),
                            empties=empties,
                        )
                        found.append(result)
            if games_done % 100 == 0:
                print(f"{games_done} games, {len(found)} puzzles shapes, {dict(stats)}, {time.time() - started:.0f}s",
                      flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "candidates.json").write_text(json.dumps(found, indent=1))
    summary = {
        "games": games_done,
        "seconds": round(time.time() - started),
        "stats": dict(stats),
        "shapes": len(found),
        "unique_kill_answer": sum(len(f["kill_answers"]) == 1 for f in found),
        "unique_live_answer": sum(len(f["live_answers"]) == 1 for f in found),
        "by_empties": dict(sorted(Counter(f["empties"] for f in found).items())),
        "by_matchup": dict(Counter(f["matchup"] for f in found).most_common(12)),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    with (args.out / "sample.txt").open("w") as sample:
        for f in found[:: max(1, len(found) // 30)][:30]:
            region = {tuple(p) for p in f["region"]}
            sample.write(f"{f['source']}  ({f['matchup']})  defender {f['defender']}  "
                         f"kill {f['kill_answers']}  live {f['live_answers']}\n")
            for r, line in enumerate(f["board"]):
                marked = "".join("+" if (r, c) in region and ch == "." else ch for c, ch in enumerate(line))
                sample.write("  " + marked + "\n")
            sample.write("\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
