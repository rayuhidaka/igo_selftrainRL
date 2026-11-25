"""Spike: how many capture puzzles (igo-app/docs/PUZZLES.md, type B) hide in the recorded games?

Replays the given `eval/*games.json` games and, after every move from `--min-move` on, looks at
each chain of the side that just moved with exactly two liberties within distance 2 of that move
(a puzzle comes from the move that left a chain capturable). It is a capture candidate when
(engine/capture.py's reader):

- the side to move captures it by force with exactly one first move, in at least 3 plies (more
  than "atari, then take");
- it is urgent: if the side to move passed instead, the chain would get away;
- it isn't unconditionally alive.

Each candidate gets a technique from its answer: "ladder" (the answer is an atari and every
attacking move of the main line is an atari), "throw-in" (the answer is a self-atari on a
liberty), "net" (the answer isn't on a liberty), or "atari" (anything else on a liberty). Shapes
repeat; the same local window (the chain and the points within distance 2, the defender as Black,
under the 8 symmetries) counts once.

    python -m puzzles.spike_capture eval/puzzle_refill_games.json --out runs/capture_spike
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from collections import Counter
from pathlib import Path

from engine.capture import CaptureReader, chain_at, liberties_of
from engine.eye_space import BudgetExceeded
from engine.move import Pass, Play
from engine.point import Point
from engine.position import Position
from engine.stone import Stone
from puzzles.spike_mine import decode_history, rows, transforms


def window_key(position: Position, chain: set[Point], defender: Stone) -> tuple:
    size = position.board_size
    window = {Point(p.row + dr, p.col + dc) for p in chain for dr in range(-2, 3) for dc in range(-2 + abs(dr), 3 - abs(dr))}
    window = {q for q in window if 0 <= q.row < size and 0 <= q.col < size}
    norm = {defender: 1, defender.opponent(): 2, Stone.EMPTY: 0}
    best = None
    for t in transforms(size):
        key = tuple(sorted((*t(p.row, p.col), norm[position.stone_at(p)]) for p in window))
        if best is None or key < best:
            best = key
    return best


def main_line(reader: CaptureReader, position: Position, answer: Point, depth: int) -> list[Point | None]:
    """The answer, then the defender's longest-lasting reply and the attacker's quickest capture, until
    the capture: the line shown when the puzzle is solved."""
    line: list[Point | None] = [answer]
    pos = position.play(Play(answer))
    while reader.status(pos) == 0 and len(line) < depth:
        # Defender: the move after which the capture takes longest.
        best, best_d = None, -1
        for m in reader.moves(pos, defender=True):
            after = pos.play(m)
            d = reader.capture_depth(after, depth) or 0
            if d > best_d:
                best, best_d = m, d
        line.append(None if isinstance(best, Pass) else best.point)
        pos = pos.play(best)
        if reader.status(pos) != 0:
            break
        # Attacker: a move that keeps the capture quickest.
        quick = None
        for m in reader.moves(pos, defender=False):
            if reader.holds(pos.play(m), best_d - 1):
                quick = m
                break
        if quick is None:
            break
        line.append(quick.point)
        pos = pos.play(quick)
    return line


def technique(position: Position, chain: set[Point], answer: Point, line: list[Point | None]) -> str:
    liberties = liberties_of(position, chain)
    if answer not in liberties:
        return "net"
    after = position.play(Play(answer))
    own = chain_at(after, answer)
    if len(liberties_of(after, own)) == 1:
        return "throw-in"
    attacker_moves = [p for i, p in enumerate(line) if i % 2 == 0 and p is not None]
    if len(attacker_moves) >= 3:
        return "ladder"
    return "atari"


def near(chain: set[Point], point: Point, distance: int = 2) -> bool:
    return any(abs(p.row - point.row) + abs(p.col - point.col) <= distance for p in chain)


def scan_game(job: tuple) -> tuple[list[dict], Counter]:
    """One game's candidates (a worker's job): see the module docstring."""
    path_name, game_index, game, min_move, depth_limit, budget = job
    found: list[dict] = []
    stats = Counter()
    seen: set[tuple] = set()
    size = game.get("board_size", 9)
    position = Position.empty(size)
    for move_number, move in enumerate(decode_history(game["history"], size), start=1):
        position = position.play(move)
        if move_number < min_move or isinstance(move, Pass):
            continue
        defender = position.to_play.opponent()
        checked: set[Point] = set()
        for index, stone in enumerate(position.stones):
            point = Point.from_index(index, size)
            if stone != defender or point in checked:
                continue
            chain = chain_at(position, point)
            checked |= chain
            if len(liberties_of(position, chain)) != 2 or not near(chain, move.point):
                continue
            stats["chains"] += 1
            key = window_key(position, chain, defender)
            if key in seen:
                stats["repeat"] += 1
                continue
            seen.add(key)
            target = min(chain, key=lambda p: (p.row, p.col))
            reader = CaptureReader(target, node_budget=budget)
            try:
                answers = reader.capturing_moves(position, depth_limit)
                if len(answers) != 1:
                    stats["no capture" if not answers else "several answers"] += 1
                    continue
                depth = reader.capture_depth(position, depth_limit)
                if depth is None or depth < 3:
                    stats["too short"] += 1
                    continue
                if reader.holds(position.play(Pass()), depth_limit):
                    stats["not urgent"] += 1
                    continue
                line = main_line(reader, position, answers[0], depth_limit)
            except BudgetExceeded:
                stats["over budget"] += 1
                continue
            kind = technique(position, chain, answers[0], line)
            stats[kind] += 1
            found.append({
                "key": repr(key),
                "attacker": "X" if position.to_play == Stone.BLACK else "O",
                "target": [target.row, target.col],
                "chain": len(chain),
                "answer": [answers[0].row, answers[0].col],
                "depth": depth,
                "line": [None if p is None else [p.row, p.col] for p in line],
                "technique": kind,
                "source": f"{path_name}#{game_index}@{move_number}",
                "matchup": f"{game['black']} vs {game['white']}",
                "board": rows(position),
            })
    return found, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("games", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-move", type=int, default=8)
    parser.add_argument("--depth", type=int, default=17)
    parser.add_argument("--budget", type=int, default=40_000)
    parser.add_argument("--limit-games", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    jobs = []
    for path in args.games:
        for game_index, game in enumerate(json.loads(path.read_text())):
            jobs.append((path.name, game_index, game, args.min_move, args.depth, args.budget))
    if args.limit_games:
        jobs = jobs[: args.limit_games]
    found: list[dict] = []
    keys: set[str] = set()
    stats = Counter()
    started = time.time()
    with multiprocessing.Pool(args.workers) as pool:
        for done, (game_found, game_stats) in enumerate(pool.imap_unordered(scan_game, jobs, chunksize=4), start=1):
            stats.update(game_stats)
            for candidate in game_found:
                if candidate["key"] in keys:
                    stats["repeat across games"] += 1
                    continue
                keys.add(candidate["key"])
                found.append(candidate)
            if done % 50 == 0:
                print(f"{done}/{len(jobs)} games, {len(found)} candidates, {dict(stats)}, {time.time() - started:.0f}s", flush=True)

    for candidate in found:
        del candidate["key"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "candidates.json").write_text(json.dumps(found, indent=1))
    summary = {"games": len(jobs), "seconds": round(time.time() - started), "stats": dict(stats), "candidates": len(found),
               "by_technique": dict(Counter(c["technique"] for c in found)), "by_depth": dict(sorted(Counter(c["depth"] for c in found).items()))}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
