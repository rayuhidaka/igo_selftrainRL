"""Turn capture candidates (`puzzles.spike_capture`) into igo-app puzzles (igo-app/docs/PUZZLES.md,
type B: "capture the marked stones").

The same answer trees as the life-and-death puzzles (`puzzles.generate.build_tree`: every move the
player could make at every node, the right ones with the opponent's most stubborn reply, the wrong
ones with the line that refutes them), judged by `engine.capture.CaptureReader` through
`CaptureFight`, which speaks `generate.Fight`'s interface.

The puzzle's area (`region`): every point either side's reader would consider at the start and
along the candidate's main line, grown where the tree's own lines need more (up to three
rounds), so a ladder's later moves are inside it. A tap outside the area
counts as a pass in the app, as for life and death. Kept only if every node of the tree that
isn't solved has a right move (the area is big enough for the whole solution) and the only right
first move is the spike's answer.

    python -m puzzles.generate_capture runs/capture_spike/candidates.json --out runs/capture_puzzles
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import time
from collections import Counter
from pathlib import Path

from engine.capture import CAPTURE_DEPTH, CaptureReader, chain_at, liberties_of
from engine.eye_space import BudgetExceeded
from engine.move import Move, Pass, Play
from engine.point import Point
from engine.position import Position
from engine.stone import Stone
from puzzles.generate import (
    DEFAULT_CHECKPOINTS, SYMBOLS, TreeTooBig, Unclear, build_tree, history_moves, parse_board,
    region_policy, tree_nodes,
)

DEPTH = 17


class CaptureFight:
    """`generate.Fight` for a capture: `attacker` tries to capture the chain holding `target`, both
    sides playing inside `region` (or passing)."""

    def __init__(self, target: Point, region: set[Point], attacker: Stone, budget: int) -> None:
        self.reader = CaptureReader(target, node_budget=budget)
        self.target = target
        self.region = sorted(region, key=lambda p: (p.row, p.col))
        self.attacker = attacker

    def moves(self, position: Position) -> list[Move]:
        return [Play(p) for p in self.region if position.stone_at(p) == Stone.EMPTY and position.is_legal(Play(p))]

    # Every node is judged with the full DEPTH, whatever depth `build_tree` has left: the app's
    # PuzzlesTest does the same (engine/Capture.kt, CAPTURE_DEPTH), so a move the tree calls wrong
    # deep down can't be one that still captures given more plies.
    def mover_wins(self, position: Position, depth: int) -> bool:
        """The attacker (to move) captures by force; or the defender (to move) gets away."""
        if position.to_play == self.attacker:
            return self.reader.captures(position, DEPTH)
        return not self.reader.holds(position, DEPTH)

    def other_wins(self, position: Position, depth: int) -> bool:
        """The side *not* to move reaches its goal, whatever the side to move does."""
        if position.to_play == self.attacker:
            return not self.reader.captures(position, DEPTH)
        return self.reader.holds(position, DEPTH)

    # A line ends once the stones can no longer escape (build_tree's nothing_better_than_pass), as for
    # life and death: insisting on the stones coming off makes a pass there neither right nor wrong.

    def settled(self, position: Position) -> bool:
        return self.reader.status(position) != 0

    def reply(self, position: Position, depth: int) -> Move | None:
        """The defender's answer (to move, losing): an escape attempt a person would make -- extend
        on a liberty, or capture an attacking stone next to the target -- the one leaving the most
        liberties, then the one keeping the stones longest; another move only if none is legal.
        None once the stones are off the board."""
        if self.settled(position):
            return None
        chain = chain_at(position, self.target)
        liberties = liberties_of(position, chain)
        rescue: set[Point] = set()
        for p in chain:
            for n in position.neighbors(p):
                if position.stone_at(n) == self.attacker:
                    libs = liberties_of(position, chain_at(position, n))
                    if len(libs) == 1:
                        rescue |= libs
        attempts = [Play(p) for p in sorted(liberties | rescue, key=lambda q: (q.row, q.col)) if position.is_legal(Play(p))]
        if not attempts:
            attempts = self.moves(position)
        if not attempts:
            return None

        def score(move: Move) -> tuple[int, int]:
            after = position.play(move)
            if after.stone_at(self.target) == Stone.EMPTY:
                return (-1, 0)
            libs = len(liberties_of(after, chain_at(after, self.target)))
            plies = self.reader.capture_depth(after, DEPTH) or 0
            return (libs, plies)

        return max(attempts, key=score)

    def fastest(self, position: Position, test, depth: int) -> int | None:
        for d in range(1, depth + 1):
            if test(position, d):
                return d
        return None


def area(reader: CaptureReader, position: Position, line: list) -> set[Point]:
    """Every point either side would consider at the start and along `line` (the spike's main line)."""
    points: set[Point] = set()
    pos = position
    for index in range(len(line) + 1):
        if index > 0:
            step = line[index - 1]
            move = Pass() if step is None else Play(Point(*step))
            if not pos.is_legal(move):
                break
            pos = pos.play(move)
        if reader.status(pos) != 0:
            break
        points |= set(reader.defender_points(pos)) | set(reader.attacker_points(pos))
    return points


def stuck_positions(node: dict, position: Position) -> list[Position]:
    """The positions (player to move) of the tree's unsolved nodes that have no right move."""
    if node.get("solved"):
        return []
    good = node.get("good", {})
    if not good:
        return [position]
    out: list[Position] = []
    for move_key, step in good.items():
        if "next" not in step:
            continue
        r, c = (int(x) for x in move_key.split(","))
        after = position.play(Play(Point(r, c)))
        reply = Pass() if step["reply"] == "pass" else Play(Point(*step["reply"]))
        out += stuck_positions(step["next"], after.play(reply))
    return out


def main_line_plies(tree: dict) -> int:
    """Moves along the answer's line (answer, reply, next answer ...) until the stones can't escape."""
    plies, node = 0, tree
    while not node.get("solved"):
        good = node.get("good", {})
        if not good:
            break
        step = next(iter(good.values()))
        plies += 1
        if "next" not in step:
            break
        plies += 1
        node = step["next"]
    return plies


def capture_difficulty(technique: str, plies: int) -> int:
    """1 (easy) to 4, from the reading rather than the nets.

    The life-and-death rank (which net's first choice in the area is the answer) doesn't work here:
    in a game position the nets weigh how big a move is, so a two-move capture often came out 4
    (publisher, 2026-11-26: "level 4 is not too difficult for a beginner"). Ataris and ladders go
    on with ataris a beginner can follow: 1, or 2 when long. A throw-in (into atari on purpose) or
    a net (not an atari) is a new idea: 2, then 3 from 4 moves and 4 from 7.
    """
    if technique in ("atari", "ladder"):
        return 1 if plies <= 4 else 2
    return 2 if plies <= 3 else 3 if plies <= 6 else 4


def complete(node: dict) -> bool:
    """Every node that isn't solved has a right move, all the way down."""
    if node.get("solved"):
        return True
    good = node.get("good", {})
    return bool(good) and all("next" not in step or complete(step["next"]) for step in good.values())


_NETS = None
_ARGS = None


def _init(args) -> None:
    """A worker's setup: gen11, which picks the A/B/C choices' tempting wrong moves, loaded once."""
    global _NETS, _ARGS
    _ARGS = args
    if not args.no_nets:
        from bootstrap.inference import RayZeroPolicyValueNet  # torch only when needed

        _NETS = {"gen11": RayZeroPolicyValueNet(Path(DEFAULT_CHECKPOINTS["gen11"]), 9)}


def make_puzzle(job: tuple[int, dict]) -> tuple[int, dict | None, str]:
    """One candidate -> (its index, the puzzle or None, what happened)."""
    index, cand = job
    args, nets = _ARGS, _NETS
    attacker = SYMBOLS[cand["attacker"]]
    target = Point(*cand["target"])
    position = parse_board(cand["board"], attacker)
    answer = Point(*cand["answer"])
    try:
        scout = CaptureReader(target, node_budget=args.budget)
        empty = {p for p in area(scout, position, cand["line"]) if position.stone_at(p) == Stone.EMPTY}
        region = empty
        for _ in range(4):
            fight = CaptureFight(target, region, attacker, args.budget)
            tree = build_tree(fight, position, DEPTH, [0], args.max_nodes)
            stuck = stuck_positions(tree, position)
            if not stuck:
                break
            # Grow the area where the tree got stuck (points empty at the start only), and build again.
            grown = set(region)
            for pos in stuck:
                grown |= {p for p in set(scout.attacker_points(pos)) | set(scout.defender_points(pos)) if position.stone_at(p) == Stone.EMPTY}
            if grown == region:
                break
            region = grown
    except TreeTooBig:
        return index, None, "tree too big"
    except Unclear:
        return index, None, "unclear move"
    except BudgetExceeded:
        return index, None, "over budget"
    if list(tree.get("good", {})) != [f"{answer.row},{answer.col}"]:
        return index, None, "several answers" if len(tree.get("good", {})) > 1 else "tree disagrees"
    if not complete(tree):
        return index, None, "area too small"

    file_name, rest = cand["source"].split("#")
    game_index, move_number = (int(x) for x in rest.split("@"))
    game = json.loads((args.games_dir / file_name).read_text())[game_index]

    ordered = sorted(region, key=lambda p: (p.row, p.col))
    wrong = [p for p in ordered if p != answer and position.is_legal(Play(p))]
    difficulty = capture_difficulty(cand["technique"], main_line_plies(tree))
    choices = None
    if nets:
        gen11 = region_policy(nets["gen11"], position, ordered)
        tempting = sorted(wrong, key=lambda p: -gen11.get(p, 0.0))[:2]
        if len(tempting) == 2:
            choices = [[p.row, p.col] for p in [answer, *tempting]]
            random.Random(args.seed + index).shuffle(choices)
    return index, {
        "kind": "CAPTURE",
        "toPlay": cand["attacker"],
        "difficulty": difficulty,
        "board": cand["board"],
        "target": [target.row, target.col],
        "region": [[p.row, p.col] for p in ordered],
        "answer": [answer.row, answer.col],
        "choices": choices,
        "tree": tree,
        "treeNodes": tree_nodes(tree),
        "history": history_moves(game, move_number, position),
        "source": cand["source"],
        "matchup": cand["matchup"],
        "technique": cand["technique"],
    }, "kept"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games-dir", type=Path, default=Path("eval"))
    parser.add_argument("--max-nodes", type=int, default=300)
    parser.add_argument("--budget", type=int, default=200_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--first-id", type=int, default=1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--no-nets", action="store_true")
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text())
    if args.limit:
        candidates = candidates[: args.limit]
    results: dict[int, dict] = {}
    stats = Counter()
    started = time.time()
    with multiprocessing.Pool(args.workers, initializer=_init, initargs=(args,)) as pool:
        for done, (index, puzzle, outcome) in enumerate(pool.imap_unordered(make_puzzle, list(enumerate(candidates))), start=1):
            stats[outcome] += 1
            if puzzle is not None:
                results[index] = puzzle
            if done % 25 == 0:
                print(f"{done}/{len(candidates)} candidates, {dict(stats)}, {time.time() - started:.0f}s", flush=True)
    puzzles = []
    for index in sorted(results):
        puzzles.append({"id": f"cap-{len(puzzles) + args.first_id:04d}", **results[index]})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "puzzles.json").write_text(json.dumps(puzzles, indent=1))
    summary = {
        "candidates": len(candidates),
        "seconds": round(time.time() - started),
        "stats": dict(stats),
        "puzzles": len(puzzles),
        "by_technique": dict(Counter(p["technique"] for p in puzzles)),
        "by_difficulty": dict(sorted(Counter(p["difficulty"] for p in puzzles).items())),
        "with_choices": sum(p["choices"] is not None for p in puzzles),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
