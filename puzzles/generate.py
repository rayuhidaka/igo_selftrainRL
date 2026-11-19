"""Turn the spike's candidate shapes into igo-app's puzzle file (igo-app/docs/PUZZLES.md, step 2).

Input: `candidates.json` from `puzzles.spike_mine` (walled-in fights whose result depends on who
moves first, proven by `engine/eye_space.py`). Each shape gives up to two puzzles:

- **live**: the defender to play and live; **kill**: the attacker to play and kill.

A puzzle is kept only if:

- it has exactly one correct first move (phase 1 shows single-answer puzzles only);
- every attacker chain touching the fight has at least `--wall-liberties` liberties outside it,
  so a move outside the marked area can't threaten the wall in one go -- the reader only plays
  inside, and the app counts an outside tap as a wrong answer;
- its reply tree (below) stays within `--max-nodes` nodes.

**The reply tree** (what the app plays back, so the phone needs no search). At each node the
player is to move:

- `"solved": true` -- the goal is reached, or the player could even pass and every reply of the
  opponent inside the area still loses for it ("nothing better than to pass");
- otherwise `good`: each correct move (by the exact reader) with the opponent's most stubborn
  reply (the one that keeps the fight going longest) and the next node; and `bad`: each wrong
  move inside the area, plus `pass` (also used for any tap outside the area), with a refutation
  line -- the opponent's fastest win, then the player's most stubborn defense, until the result
  is settled (at most `REFUTATION_PLIES` plies).

Points are written `[row, col]`, zero-indexed, row-major, as in the app.

**Ranking with the nets** (policy only, limited to the legal moves inside the area -- the nets
play to win the game, not to live locally, so their whole-board top move says little here):

- `difficulty`: 1 if gen1's preferred point is the answer, 2 if gen5's is, 3 if gen11's is,
  otherwise 4.
- `choices`: the answer plus the two wrong moves gen11 rates highest -- the most tempting
  mistakes -- in shuffled order, for the A/B/C format.

    python -m puzzles.generate runs/puzzle_spike/candidates.json --out runs/puzzles
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import time
from collections import Counter
from pathlib import Path

from engine.eye_space import EYE_SPACE_DEPTH, BudgetExceeded, EyeSpaceReader
from engine.move import Move, Pass, Play
from engine.point import Point
from engine.position import Position
from engine.stone import Stone

REFUTATION_PLIES = 10
LEVELS = {"gen1": 1, "gen5": 2, "gen11": 3}
DEFAULT_CHECKPOINTS = {
    "gen1": "checkpoints/bootstrap_option3_gen1_candidate.pt",
    "gen5": "checkpoints/bootstrap_option3_gen5_candidate.pt",
    "gen11": "checkpoints/bootstrap_option3_gen11_candidate.pt",
}
SYMBOLS = {"X": Stone.BLACK, "O": Stone.WHITE, ".": Stone.EMPTY}


class TreeTooBig(Exception):
    """The reply tree went over `--max-nodes`."""


class Unclear(Exception):
    """A move the player can make is neither a proven win nor a proven loss (seki, ko, or a
    fight longer than the reader's depth). Calling it wrong would be unfair, so the puzzle is
    dropped. Found by igo-app's PuzzlesTest on the first run (2026-11-19)."""


def parse_board(rows: list[str], to_play: Stone) -> Position:
    return Position(board_size=len(rows), stones=[SYMBOLS[c] for c in "".join(rows)], to_play=to_play)


def point_json(move: Move) -> list[int] | str:
    return "pass" if isinstance(move, Pass) else [move.point.row, move.point.col]


def key(move: Move) -> str:
    return "pass" if isinstance(move, Pass) else f"{move.point.row},{move.point.col}"


def fight_area(position: Position, target: Point, attacker: Stone) -> set[Point]:
    """The component of non-attacker points holding `target` (the spike's S)."""
    area = {target}
    stack = [target]
    while stack:
        for n in position.neighbors(stack.pop()):
            if n not in area and position.stone_at(n) != attacker:
                area.add(n)
                stack.append(n)
    return area


def weakest_wall(position: Position, area: set[Point], attacker: Stone) -> int:
    """The fewest liberties outside `area` of any attacker chain touching it."""
    touching = {n for p in area for n in position.neighbors(p) if position.stone_at(n) == attacker}
    done: set[Point] = set()
    weakest = 99
    for stone in touching:
        if stone in done:
            continue
        chain = {stone}
        stack = [stone]
        while stack:
            for n in position.neighbors(stack.pop()):
                if n not in chain and position.stone_at(n) == attacker:
                    chain.add(n)
                    stack.append(n)
        done |= chain
        outside = {n for p in chain for n in position.neighbors(p) if position.stone_at(n) == Stone.EMPTY and n not in area}
        weakest = min(weakest, len(outside))
    return weakest


class Fight:
    """The exact reader, phrased for one puzzle: `player` tries to reach `goal` ("live" or
    "kill") for the group holding `target`, both sides playing inside `region`."""

    def __init__(self, target: Point, region: set[Point], attacker: Stone, budget: int) -> None:
        self.reader = EyeSpaceReader(target, region, budget)
        self.target = target
        self.region = sorted(region, key=lambda p: (p.row, p.col))
        self.attacker = attacker

    def moves(self, position: Position) -> list[Move]:
        return [Play(p) for p in self.region if position.stone_at(p) == Stone.EMPTY and position.is_legal(Play(p))]

    def mover_wins(self, position: Position, depth: int) -> bool:
        """The side to move reaches its own goal (kill for the attacker, life for the defender)."""
        if position.to_play == self.attacker:
            return self.reader.kill_wins(position, depth)
        return self.reader.life_wins(position, depth)

    def other_wins(self, position: Position, depth: int) -> bool:
        """The side *not* to move reaches its goal, whatever the side to move does."""
        if position.to_play == self.attacker:
            return self.reader.life_holds(position, depth)
        return self.reader.kill_holds(position, depth)

    def settled(self, position: Position) -> bool:
        return self.reader.captured(position) or self.reader.alive(position)

    def fastest(self, position: Position, test, depth: int) -> int | None:
        """The smallest depth at which `test(position, d)` holds, if any up to `depth`."""
        for d in range(1, depth + 1):
            if test(position, d):
                return d
        return None


def build_tree(fight: Fight, position: Position, depth: int, counter: list[int], max_nodes: int) -> dict:
    """The node for `position`, with the player to move and the player winning."""
    counter[0] += 1
    if counter[0] > max_nodes:
        raise TreeTooBig()
    if fight.settled(position) or nothing_better_than_pass(fight, position, depth):
        return {"solved": True}
    good: dict[str, dict] = {}
    bad: dict[str, list] = {}
    for move in [*fight.moves(position), Pass()]:
        after = position.play(move)
        if not isinstance(move, Pass) and fight.other_wins(after, depth - 1):
            reply = stubborn_reply(fight, after, depth - 1)
            if reply is None:
                good[key(move)] = {"solved": True}
            else:
                good[key(move)] = {
                    "reply": point_json(reply),
                    "next": build_tree(fight, after.play(reply), depth - 2, counter, max_nodes),
                }
        else:
            if not fight.mover_wins(after, depth - 1):
                raise Unclear()  # not a proven win, and not a proven loss either
            bad[key(move)] = refutation(fight, after, depth - 1)
    return {"good": good, "bad": bad}


def recheck(fight: Fight, position: Position, node: dict, depth: int = EYE_SPACE_DEPTH) -> None:
    """Re-walks an existing tree and raises `Unclear` where a wrong move isn't a proven loss
    (the check `build_tree` now makes as it goes) -- for files made before it existed."""
    if node.get("solved"):
        return
    for move in [*fight.moves(position), Pass()]:
        after = position.play(move)
        step = node["good"].get(key(move))
        if step is not None:
            if "next" in step:
                reply = Pass() if step["reply"] == "pass" else Play(Point(*step["reply"]))
                recheck(fight, after.play(reply), step["next"], depth - 2)
        elif not fight.mover_wins(after, depth - 1):
            raise Unclear()


def nothing_better_than_pass(fight: Fight, position: Position, depth: int) -> bool:
    """The player (to move) could pass and every opponent move inside the area still loses."""
    after_pass = position.play(Pass())
    replies = fight.moves(after_pass)
    return bool(replies) and all(fight.mover_wins(after_pass.play(r), depth - 2) for r in replies) or (
        not replies
    )


def stubborn_reply(fight: Fight, position: Position, depth: int) -> Move | None:
    """The opponent (to move, losing) picks the area move after which the player needs the
    most plies to win; None if the fight is settled or the opponent has no move inside."""
    if fight.settled(position):
        return None
    best, best_d = None, -1
    for move in fight.moves(position):
        after = position.play(move)
        d = fight.fastest(after, fight.mover_wins, depth - 1)
        if d is None:  # shouldn't happen -- the player wins against everything
            continue
        if d > best_d:
            best, best_d = move, d
    return best


def refutation(fight: Fight, position: Position, depth: int) -> list:
    """After a wrong move: the opponent (to move, winning) plays its fastest win, the player its
    most stubborn defense, until the fight is settled or `REFUTATION_PLIES` plies."""
    line: list = []
    winner_to_move = True
    for _ in range(REFUTATION_PLIES):
        if fight.settled(position) or depth <= 1:
            break
        moves = fight.moves(position)
        if not moves:
            break
        if winner_to_move:
            scored = [(fight.fastest(position.play(m), fight.other_wins, depth - 1), m) for m in moves]
            scored = [(d, m) for d, m in scored if d is not None]
            if not scored:
                break
            move = min(scored, key=lambda dm: dm[0])[1]
        else:
            scored = [(fight.fastest(position.play(m), fight.mover_wins, depth - 1), m) for m in moves]
            scored = [(d, m) for d, m in scored if d is not None]
            if not scored:
                break
            move = max(scored, key=lambda dm: dm[0])[1]
        line.append(point_json(move))
        position = position.play(move)
        depth -= 1
        winner_to_move = not winner_to_move
    return line


def tree_nodes(node: dict) -> int:
    return 1 + sum(tree_nodes(child["next"]) for child in node.get("good", {}).values() if "next" in child)


def region_policy(net, position: Position, region: list[Point]) -> dict[Point, float]:
    policy = net.evaluate(position).policy
    return {p: policy.get(Play(p), 0.0) for p in region if position.stone_at(p) == Stone.EMPTY and position.is_legal(Play(p))}


def history_moves(game: dict, move_number: int, position: Position, count: int = 5) -> list[dict]:
    """The last `count` moves up to `move_number` whose stones are still on the board."""
    moves = game["history"][:move_number]
    out = []
    for n in range(max(0, move_number - count), move_number):
        index = moves[n]
        if index < 0:
            continue
        p = Point.from_index(index, position.board_size)
        color = Stone.BLACK if n % 2 == 0 else Stone.WHITE
        if position.stone_at(p) == color:
            out.append({"number": n + 1, "point": [p.row, p.col], "color": "X" if color == Stone.BLACK else "O"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games-dir", type=Path, default=Path("eval"))
    parser.add_argument("--wall-liberties", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=300)
    parser.add_argument("--budget", type=int, default=400_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--no-nets", action="store_true", help="skip the net ranking (for a quick tree-only run)")
    args = parser.parse_args()

    nets = {}
    if not args.no_nets:
        from bootstrap.inference import RayZeroPolicyValueNet  # torch only when needed

        nets = {name: RayZeroPolicyValueNet(Path(path), 9) for name, path in DEFAULT_CHECKPOINTS.items()}

    rng = random.Random(args.seed)
    candidates = json.loads(args.candidates.read_text())
    if args.limit:
        candidates = candidates[: args.limit]
    games_cache: dict[str, list] = {}
    puzzles = []
    stats = Counter()
    started = time.time()

    for shape_index, cand in enumerate(candidates):
        defender = SYMBOLS[cand["defender"]]
        attacker = defender.opponent()
        target = Point(*cand["target"])
        region = {Point(*p) for p in cand["region"]}
        base = parse_board(cand["board"], defender)
        area = fight_area(base, target, attacker)
        if weakest_wall(base, area, attacker) < args.wall_liberties:
            stats["weak wall"] += 2
            continue
        file_name, rest = cand["source"].split("#")
        game_index, move_number = (int(x) for x in rest.split("@"))
        if file_name not in games_cache:
            games_cache[file_name] = json.loads((args.games_dir / file_name).read_text())
        game = games_cache[file_name][game_index]

        for goal, player, answers in (("live", defender, cand["live_answers"]), ("kill", attacker, cand["kill_answers"])):
            if len(answers) != 1:
                stats["several answers"] += 1
                continue
            position = dataclasses.replace(base, to_play=player)
            fight = Fight(target, region, attacker, args.budget)
            try:
                tree = build_tree(fight, position, EYE_SPACE_DEPTH, [0], args.max_nodes)
            except TreeTooBig:
                stats["tree too big"] += 1
                continue
            except Unclear:
                stats["unclear move"] += 1
                continue
            except BudgetExceeded:
                stats["over budget"] += 1
                continue
            good = list(tree.get("good", {}))
            if good != [f"{answers[0][0]},{answers[0][1]}"]:
                stats["tree disagrees"] += 1  # a guard: the tree must find the spike's answer, alone
                continue
            answer = Point(*answers[0])
            wrong = [p for p in sorted(region, key=lambda p: (p.row, p.col))
                     if p != answer and position.stone_at(p) == Stone.EMPTY and position.is_legal(Play(p))]

            difficulty, choices = 4, None
            if nets:
                difficulty = 4
                for name, level in LEVELS.items():
                    prefs = region_policy(nets[name], position, sorted(region, key=lambda p: (p.row, p.col)))
                    if prefs and max(prefs, key=prefs.get) == answer:
                        difficulty = level
                        break
                gen11 = region_policy(nets["gen11"], position, sorted(region, key=lambda p: (p.row, p.col)))
                tempting = sorted(wrong, key=lambda p: -gen11.get(p, 0.0))[:2]
                if len(tempting) == 2:
                    choices = [[p.row, p.col] for p in [answer, *tempting]]
                    rng.shuffle(choices)

            puzzles.append({
                "id": f"ld-{len(puzzles) + 1:04d}",
                "kind": goal.upper(),
                "toPlay": "X" if player == Stone.BLACK else "O",
                "difficulty": difficulty,
                "board": cand["board"],
                "target": [target.row, target.col],
                "region": [[p.row, p.col] for p in sorted(region, key=lambda p: (p.row, p.col))],
                "answer": [answer.row, answer.col],
                "choices": choices,
                "tree": tree,
                "treeNodes": tree_nodes(tree),
                "history": history_moves(game, move_number, base),
                "source": cand["source"],
                "matchup": cand["matchup"],
            })
            stats["kept"] += 1
        if (shape_index + 1) % 25 == 0:
            print(f"{shape_index + 1}/{len(candidates)} shapes, {dict(stats)}, {time.time() - started:.0f}s", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "puzzles.json").write_text(json.dumps(puzzles, indent=1))
    summary = {
        "shapes": len(candidates),
        "seconds": round(time.time() - started),
        "stats": dict(stats),
        "puzzles": len(puzzles),
        "by_kind": dict(Counter(p["kind"] for p in puzzles)),
        "by_difficulty": dict(sorted(Counter(p["difficulty"] for p in puzzles).items())),
        "with_choices": sum(p["choices"] is not None for p in puzzles),
        "tree_nodes": dict(sorted(Counter(min(p["treeNodes"], 50) // 10 * 10 for p in puzzles).items())),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
