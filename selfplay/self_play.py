"""Generates self-play training examples via real MCTS search (`mcts/mcts.py`'s `Mcts`)
using Ray-zeroGo's own current-best checkpoint -- unlike `selfplay/generate.py`'s
distillation approach (sampling directly from KataGo's raw policy output, no search), this
is genuine AlphaZero-style self-play: the recorded policy target is the search's own
visit-count distribution (a refinement of the raw policy prior, not the prior itself), and
value targets come from actually played-out game outcomes. See docs/ROADMAP.md's Phase 3.

Much more expensive per move than `generate.py` (one full `num_simulations`-simulation
search per move, instead of one net evaluation) -- expect real seconds per move, not
milliseconds, so size `num_games`/`num_simulations` down for a first run rather than
assuming Phase 2's game counts translate directly.

Produces the same `bootstrap.dataset.SelfPlayExamples` format Phase 2's `generate.py` did,
so `bootstrap/train.py` trains on it completely unmodified.

`config["min_moves_to_keep"]` (default 0, no filtering) discards any game shorter than
this before it's folded into the saved dataset -- guards against a self-play collapse
artifact seen in practice (2026-09-03, see docs/ROADMAP.md's Phase 3): a fine-tuned
candidate whose value net hadn't seen an early-pass position developing a small but
nonzero prior on Pass, which a small `num_simulations` budget can then run away with
within a single search, ending the game in a handful of moves with whoever benefits from
komi "winning" a board neither side actually played on. Training on those teaches the
exact behavior that produced them, compounding worse each generation.

Usage:
    python -m selfplay.self_play --config configs/selfplay_self_play_smoke_test.yaml
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import yaml

from bootstrap.dataset import SelfPlayExamples
from bootstrap.inference import RayZeroPolicyValueNet, encode_planes
from engine.move import Move, Pass
from engine.position import Position
from engine.scoring import area_score
from engine.stone import Stone
from mcts.mcts import Mcts, MctsConfig
from selfplay.generate import sample_move


def visit_count_policy(board_size: int, move_visits: dict[Move, int]) -> np.ndarray:
    """Converts `Mcts.search`'s per-move visit counts into a dense, normalized policy
    target -- the "improved policy" self-play trains toward, per the row-major-plus-pass
    index convention (see igo-app/docs/MODEL_CONTRACT.md). `move_visits` already covers
    every legal move (each gets a child node on expansion, even if never selected again --
    see `mcts/mcts.py`'s `_expand`), so no separate legal-move masking is needed here,
    unlike `generate.py`'s `legal_policy_vector`. Falls back to uniform over the moves
    considered if every visit count is zero (only possible with a very small
    `num_simulations`, since the root's first simulation only expands children without
    visiting any of them).
    """
    vector = np.zeros(board_size * board_size + 1, dtype=np.float32)
    for move, visits in move_visits.items():
        index = board_size * board_size if isinstance(move, Pass) else move.point.to_index(board_size)
        vector[index] = visits
    total = vector.sum()
    if total <= 0:
        for move in move_visits:
            index = board_size * board_size if isinstance(move, Pass) else move.point.to_index(board_size)
            vector[index] = 1.0 / len(move_visits)
    else:
        vector /= total
    return vector


def play_one_game(
    mcts: Mcts, board_size: int, komi: float, temperature: float, max_moves: int, rng: random.Random
) -> tuple[list[tuple[np.ndarray, np.ndarray, Stone]], "Stone | None"]:
    """Plays one self-play game and returns `(records, winner)`, where `records` has one
    `(board_planes, policy_target, to_play)` tuple per move played.
    """
    position = Position.empty(board_size)
    records: list[tuple[np.ndarray, np.ndarray, Stone]] = []
    moves_played = 0
    while position.pass_count < 2 and moves_played < max_moves:
        result = mcts.search(position)
        policy_target = visit_count_policy(board_size, result.move_visits)
        records.append((encode_planes(position).numpy(), policy_target, position.to_play))

        move = sample_move(board_size, policy_target, temperature, rng)
        position = position.play(move)
        moves_played += 1

    winner = area_score(position).winner(komi)
    return records, winner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    board_size = config["board_size"]
    komi = config.get("komi", 7.5)

    # channels/num_conv_layers/num_residual_blocks only need setting for a legacy
    # checkpoint (no recorded architecture) -- see RayZeroPolicyValueNet's docstring.
    net = RayZeroPolicyValueNet(
        Path(config["checkpoint_path"]),
        board_size,
        channels=config.get("channels"),
        num_conv_layers=config.get("num_conv_layers"),
        num_residual_blocks=config.get("num_residual_blocks"),
    )
    mcts_config = MctsConfig(
        num_simulations=config.get("num_simulations", MctsConfig().num_simulations),
        exploration_constant=config.get("exploration_constant", MctsConfig().exploration_constant),
        komi=komi,
    )
    mcts = Mcts(net, mcts_config)
    rng = random.Random(config["seed"])

    all_planes: list[np.ndarray] = []
    all_policies: list[np.ndarray] = []
    all_values: list[float] = []

    min_moves_to_keep = config.get("min_moves_to_keep", 0)
    games_discarded = 0

    start = time.time()
    games_played = 0
    for game_index in range(config["num_games"]):
        records, winner = play_one_game(mcts, board_size, komi, config["temperature"], config["max_moves"], rng)

        # A real MCTS-searched game ending in only a handful of moves (both players passing
        # on a near-empty board) is a self-play collapse artifact, not a meaningful outcome
        # -- almost always both players passing immediately, with whoever benefits from komi
        # "winning" a board neither of them actually played on. Training on these teaches
        # the exact behavior that produced them, compounding worse across generations (see
        # docs/ROADMAP.md's Phase 3) -- discard rather than fold into the saved dataset.
        if len(records) < min_moves_to_keep:
            games_discarded += 1
            print(f"Game {game_index + 1}/{config['num_games']} discarded ({len(records)} moves, too short)")
            continue

        for planes, policy_target, to_play in records:
            if winner is None:
                z = 0.0
            elif winner == to_play:
                z = 1.0
            else:
                z = -1.0
            all_planes.append(planes)
            all_policies.append(policy_target)
            all_values.append(z)

        games_played += 1
        elapsed = time.time() - start
        print(
            f"Game {games_played}/{config['num_games']} done "
            f"({len(records)} moves, winner={winner}), {elapsed:.1f}s elapsed, "
            f"{len(all_planes)} examples so far"
        )
        if config.get("max_seconds") is not None and elapsed > config["max_seconds"]:
            print(f"Hit max_seconds ({config['max_seconds']}), stopping early.")
            break

    examples = SelfPlayExamples(
        board_planes=np.stack(all_planes).astype(np.float32),
        policy_targets=np.stack(all_policies).astype(np.float32),
        value_targets=np.array(all_values, dtype=np.float32),
    )
    out_path = Path(config["out_path"])
    examples.save(out_path)
    print(
        f"Wrote {len(examples)} training examples from {games_played} games to {out_path} "
        f"({games_discarded} game(s) discarded as too short)"
    )


if __name__ == "__main__":
    main()
