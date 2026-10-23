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

Usage:
    python -m eval.rule_comparison --config configs/eval_rule_comparison.yaml
    python -m eval.rule_comparison --config configs/eval_rule_comparison.yaml --summarize-only
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml


def game_record(end: "GameEnd", black: str, white: str) -> dict:
    """One finished game as stored: who played which color, its length, and both counts
    (excluding komi) -- `black_area`/`white_area` from `area_score`, `black_territory`/
    `white_territory` from `territory_score`."""
    from engine.scoring import area_score, territory_score

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
    }


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


if __name__ == "__main__":
    main()
