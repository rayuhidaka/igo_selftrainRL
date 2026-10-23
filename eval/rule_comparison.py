"""Counts the same games under area (Tromp-Taylor) and Japanese-style territory scoring, to
decide whether igo-app's switch to Japanese scoring needs a retrained net (igo-app's
docs/ROADMAP.md, scoring overhaul stage C).

Every Ray-zeroGo generation was trained with area scoring, under which playing inside your
own territory costs nothing -- so the nets happily spend moves there, e.g. capturing dead
stones before passing. Under territory scoring each such move costs a point. Counting the
nets' own games both ways measures exactly that: a game whose winner differs between the
two counts ("a flip") is one where switching rules would change the result. Few flips means
the nets can play under Japanese scoring as they are.

The games come from pairings of the shipped levels (config `pairings`), colors alternating,
with the same opening randomness as eval/round_robin.py so the games differ. Raw counts per
game (area and territory points, prisoners) go to a JSON file, so other komi values can be
summarized later with `--summarize-only` without replaying anything.

Each game's moves are saved too, for the self-fill analysis (`summarize_self_fill`, added for
igo-app's "AI stops filling its own territory once the opponent passes" item): which stones
placed right after an opponent's pass a rule would turn into a pass, and what that would do to
the flip rate.

Usage:
    python -m eval.rule_comparison --config configs/eval_rule_comparison.yaml
    python -m eval.rule_comparison --config configs/eval_rule_comparison.yaml --summarize-only
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import yaml

from engine.life_and_death import estimate_dead_stones, without_dead_stones
from engine.move import Move, Pass, Play
from engine.point import Point
from engine.position import Position
from engine.scoring import area_score, territory_ownership, territory_score
from engine.stone import Stone


def game_record(end: "GameEnd", black: str, white: str) -> dict:
    """One finished game as stored: who played which color, its length, and both counts
    (excluding komi) -- `black_area`/`white_area` from `area_score`, `black_territory`/
    `white_territory` from `territory_score` -- and its moves as `history` (`encode_history`)."""
    area = area_score(end.position)
    territory = territory_score(end.position, end.black_prisoners, end.white_prisoners)
    return {
        "black": black,
        "white": white,
        "moves": end.moves,
        "finished": end.finished,
        "black_area": area.black,
        "white_area": area.white,
        "black_territory": territory.black,
        "white_territory": territory.white,
        "black_prisoners": end.black_prisoners,
        "white_prisoners": end.white_prisoners,
        "board_size": end.position.board_size,
        "history": encode_history(end.history, end.position.board_size),
    }


def encode_history(history: "tuple[Move, ...] | list[Move]", board_size: int) -> list[int]:
    """Moves as JSON-friendly ints: a play's row-major point index, -1 for a pass."""
    return [move.point.to_index(board_size) if isinstance(move, Play) else -1 for move in history]


def decode_history(history: list[int], board_size: int) -> list[Move]:
    """The inverse of `encode_history`."""
    return [Pass() if index < 0 else Play(Point.from_index(index, board_size)) for index in history]


def black_margin(black: int, white: int, komi: float) -> float:
    """Black's margin once `komi` is added to White: positive means Black wins."""
    return black - (white + komi)


def summarize(records: list[dict], area_komi: float, territory_komi: float) -> dict:
    """Flip statistics over `records` (games that hit the move cap are left out): how many
    games each count gives the other winner, and how much the territory count shifts Black's
    margin on average -- a systematic shift would mean the two komi values aren't
    equivalent for these nets' play."""
    finished = [r for r in records if r["finished"]]
    flips = 0
    shifts = []
    for r in finished:
        area = black_margin(r["black_area"], r["white_area"], area_komi)
        territory = black_margin(r["black_territory"], r["white_territory"], territory_komi)
        if (area > 0) != (territory > 0):
            flips += 1
        shifts.append(territory - area)
    return {
        "games": len(finished),
        "unfinished": len(records) - len(finished),
        "flips": flips,
        "flip_rate": flips / len(finished) if finished else 0.0,
        "mean_margin_shift": sum(shifts) / len(shifts) if shifts else 0.0,
        "max_abs_margin_shift": max((abs(s) for s in shifts), default=0.0),
    }


# --- Self-fill analysis (igo-app's "AI stops filling its own territory once the opponent passes")
#
# Replays each game and classifies every stone placed right after the opponent passed -- the only
# moves the proposed rule would turn into a pass, which then ends the game:
#   "enclosed"    -- on a point only the mover's own stones border (territory_ownership): the
#                    cheap check, no dead-stone read needed
#   "dead_stones" -- on a point that is the mover's territory once the opponent's dead stones are
#                    taken off (capturing stones that were already dead): needs a dead-stone read
#   "defense"     -- either of those, but giving a liberty to one of the mover's own chains in
#                    atari: a move the rule must still allow
#   "elsewhere"   -- anything else (dame, the opponent's area, a live fight): the rule leaves it
# The dead-stone read uses the game's own continuation as its one playout
# (`estimate_dead_stones(before, [final])`), i.e. hindsight; the app would use its playouts.

POST_PASS_KINDS = ("enclosed", "dead_stones", "defense", "elsewhere")

PASS_RULES = {
    "enclosed_only": {"enclosed"},
    "with_dead_read": {"enclosed", "dead_stones"},
}
"""The rule variants measured: the post-pass kinds each one turns into a pass."""


def saves_chain_in_atari(position: Position, point: Point, color: Stone) -> bool:
    """Whether playing `point` gives a liberty to a `color` chain in atari -- one whose only
    liberty is `point`."""
    for neighbor in position.neighbors(point):
        if position.stone_at(neighbor) != color:
            continue
        chain, liberties = {neighbor}, set()
        stack = [neighbor]
        while stack:
            for p in position.neighbors(stack.pop()):
                stone = position.stone_at(p)
                if stone == Stone.EMPTY:
                    liberties.add(p)
                elif stone == color and p not in chain:
                    chain.add(p)
                    stack.append(p)
        if liberties == {point}:
            return True
    return False


def classify_post_pass_move(before: Position, point: Point, final: Position) -> str:
    """Which of `POST_PASS_KINDS` a stone at `point` is, played from `before` (the opponent just
    passed) in a game that ended at `final`."""
    mover = before.to_play
    if territory_ownership(before)[point] == mover:
        kind = "enclosed"
    else:
        cleaned = without_dead_stones(before, estimate_dead_stones(before, [final]))
        if territory_ownership(cleaned)[point] != mover:
            return "elsewhere"
        kind = "dead_stones"
    return "defense" if saves_chain_in_atari(before, point, mover) else kind


def _territory_if_ended(before: Position, final: Position, black_prisoners: int, white_prisoners: int) -> tuple[int, int]:
    """The territory count had the game ended at `before` (the mover passing after the
    opponent's pass): stones the rest of the game shows were dead come off as prisoners, as
    igo-app's marking phase would take them."""
    dead = estimate_dead_stones(before, [final])
    dead_black = sum(1 for p in dead if before.stone_at(p) == Stone.BLACK)
    score = territory_score(without_dead_stones(before, dead), black_prisoners + len(dead) - dead_black, white_prisoners + dead_black)
    return score.black, score.white


def analyze_self_fill(record: dict) -> dict:
    """Replays `record` (one game with a `history`) and returns `kinds` -- a Counter of its
    post-pass placements by kind --, `enclosed_not_after_pass` (placements on an enclosed point
    when the opponent hadn't passed, which no post-pass rule reaches), and per `PASS_RULES`
    variant the `(black, white)` territory count the game would have ended with under it."""
    size = record["board_size"]
    moves = decode_history(record["history"], size)
    positions = [Position.empty(size)]
    for move in moves:
        positions.append(positions[-1].play(move))
    final = positions[-1]

    kinds: Counter = Counter()
    enclosed_not_after_pass = 0
    rule_ends: dict[str, Optional[tuple[int, int]]] = {name: None for name in PASS_RULES}
    prisoners = {Stone.BLACK: 0, Stone.WHITE: 0}
    for before, move, after in zip(positions, moves, positions[1:]):
        if isinstance(move, Play):
            mover = before.to_play
            if before.pass_count == 1:
                kind = classify_post_pass_move(before, move.point, final)
                kinds[kind] += 1
                for name, triggers in PASS_RULES.items():
                    if rule_ends[name] is None and kind in triggers:
                        rule_ends[name] = _territory_if_ended(before, final, prisoners[Stone.BLACK], prisoners[Stone.WHITE])
            elif territory_ownership(before)[move.point] == mover:
                enclosed_not_after_pass += 1
            prisoners[mover] += before.stones.count(mover.opponent()) - after.stones.count(mover.opponent())
    actual = (record["black_territory"], record["white_territory"])
    return {
        "kinds": kinds,
        "enclosed_not_after_pass": enclosed_not_after_pass,
        "rule_ends": {name: end or actual for name, end in rule_ends.items()},
    }


def summarize_self_fill(records: list[dict], area_komi: float, territory_komi: float) -> dict:
    """`analyze_self_fill` over the finished games: post-pass placements by kind (moves, and
    games with at least one), and per rule variant how many games would flip winner against
    the area count -- with how many of today's flips it fixes and how many new ones it makes."""
    finished = [r for r in records if r["finished"]]
    moves: Counter = Counter()
    games: Counter = Counter()
    enclosed_not_after_pass = 0
    rules = {name: {"flips": 0, "fixed": 0, "introduced": 0} for name in PASS_RULES}
    for r in finished:
        analysis = analyze_self_fill(r)
        moves.update(analysis["kinds"])
        games.update(analysis["kinds"].keys())
        enclosed_not_after_pass += analysis["enclosed_not_after_pass"]
        area_black_wins = black_margin(r["black_area"], r["white_area"], area_komi) > 0
        flipped_now = (black_margin(r["black_territory"], r["white_territory"], territory_komi) > 0) != area_black_wins
        for name, (black, white) in analysis["rule_ends"].items():
            flipped = (black_margin(black, white, territory_komi) > 0) != area_black_wins
            rules[name]["flips"] += flipped
            rules[name]["fixed"] += flipped_now and not flipped
            rules[name]["introduced"] += flipped and not flipped_now
    for stats in rules.values():
        stats["flip_rate"] = stats["flips"] / len(finished) if finished else 0.0
    return {
        "games": len(finished),
        "post_pass_moves": {kind: moves[kind] for kind in POST_PASS_KINDS},
        "post_pass_games": {kind: games[kind] for kind in POST_PASS_KINDS},
        "enclosed_not_after_pass": enclosed_not_after_pass,
        "rules": rules,
    }


def _play_pairing(
    name_a: str,
    name_b: str,
    spec_a: "str | dict",
    spec_b: "str | dict",
    board_size: int,
    num_games: int,
    num_simulations: int,
    komi: float,
    temperature: float,
    temperature_drop_move: int,
    seed: int,
) -> list[dict]:
    # Heavy imports inside the worker, as in eval/round_robin.py's own _play_pairing.
    import torch

    from bootstrap.inference import RayZeroPolicyValueNet
    from eval.match import play_game_to_end
    from eval.round_robin import checkpoint_spec
    from mcts.mcts import Mcts, MctsConfig

    torch.set_num_threads(1)
    (path_a, style_a), (path_b, style_b) = checkpoint_spec(spec_a), checkpoint_spec(spec_b)
    mcts_a = Mcts(
        RayZeroPolicyValueNet(Path(path_a), board_size),
        MctsConfig(num_simulations=style_a.num_simulations or num_simulations, komi=komi),
    )
    mcts_b = Mcts(
        RayZeroPolicyValueNet(Path(path_b), board_size),
        MctsConfig(num_simulations=style_b.num_simulations or num_simulations, komi=komi),
    )
    rng = random.Random(seed)
    records = []
    for game_index in range(num_games):
        a_black = game_index % 2 == 0
        end = play_game_to_end(
            mcts_a if a_black else mcts_b,
            mcts_b if a_black else mcts_a,
            board_size,
            temperature,
            temperature_drop_move,
            rng,
            style_a if a_black else style_b,
            style_b if a_black else style_a,
        )
        records.append(game_record(end, name_a if a_black else name_b, name_b if a_black else name_a))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--summarize-only", action="store_true", help="re-summarize the saved games, play none")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    out = Path(config.get("out_dir", "eval")) / config.get("out_name", "rule_comparison_games.json")
    area_komi = config.get("area_komi", 7.5)
    territory_komi = config.get("territory_komi", 6.5)

    if args.summarize_only:
        records = json.loads(out.read_text())
    else:
        checkpoints = config["checkpoints"]
        pairings = [tuple(p) for p in config["pairings"]]
        print(f"Playing {len(pairings)} pairings x {config['num_games']} games across {args.workers} workers...")
        records = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _play_pairing,
                    a,
                    b,
                    checkpoints[a],
                    checkpoints[b],
                    config["board_size"],
                    config["num_games"],
                    config.get("num_simulations", 50),
                    area_komi,
                    config.get("temperature", 1.0),
                    config.get("temperature_drop_move", 16),
                    config.get("seed", 0) + i,
                ): (a, b)
                for i, (a, b) in enumerate(pairings)
            }
            for done, future in enumerate(as_completed(futures), start=1):
                a, b = futures[future]
                pairing_records = future.result()
                records.extend(pairing_records)
                stats = summarize(pairing_records, area_komi, territory_komi)
                print(f"[{done}/{len(pairings)}] {a} vs {b}: {stats}", flush=True)
                # Saved as it goes, so a stopped run keeps what it finished.
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(records, indent=1))

    print(f"Overall (area komi {area_komi}, territory komi {territory_komi}): {summarize(records, area_komi, territory_komi)}")
    for komi in (5.5, 6.5, 7.5):
        print(f"  territory komi {komi}: {summarize(records, area_komi, komi)}")
    if all("history" in r for r in records):
        print(f"Self-fill after the opponent's pass: {json.dumps(summarize_self_fill(records, area_komi, territory_komi), indent=1)}")
    else:
        print("Self-fill analysis skipped: these games were saved without move histories (replay them).")


if __name__ == "__main__":
    main()
