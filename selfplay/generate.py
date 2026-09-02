"""Generates self-play training examples using the real KataGo-derived
checkpoint already shipped in igo-app (see igo-app/tools/README.md for how
it was produced), by sampling moves directly from its own policy output --
no tree search in the loop.

This is imitation-learning data generation (see docs/ROADMAP.md's Phase 2
and docs/ARCHITECTURE.md), not reinforcement-learning self-play: the
KataGo net is already strong, so sampling moves straight from its policy
is standard practice for distillation, and is far cheaper per move than
running MCTS (one net evaluation per move instead of dozens to hundreds) --
that speed is what makes generating a meaningful number of games in a
short wall-clock budget practical at all.

Usage:
    python -m selfplay.generate --config configs/selfplay_generate_base.yaml
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from bootstrap.dataset import SelfPlayExamples
from bootstrap.inference import encode_planes
from engine.move import Move, Pass, Play
from engine.point import Point
from engine.position import Position
from engine.scoring import area_score
from engine.stone import Stone


class KataGoNet:
    """Wraps igo-app's real KataGo-derived `.tflite` checkpoint for direct policy/value
    inference (no search) -- conforms to igo-app/docs/MODEL_CONTRACT.md, same as
    `bootstrap/inference.py`'s `RayZeroPolicyValueNet`, but NHWC at the tensor boundary
    (TFLite's convention) rather than NCHW (`RayZeroNet`'s).
    """

    def __init__(self, tflite_path: Path) -> None:
        self._interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        self._interpreter.allocate_tensors()
        inputs = {detail["name"]: detail for detail in self._interpreter.get_input_details()}
        outputs = {detail["name"]: detail for detail in self._interpreter.get_output_details()}
        self._input_index = inputs["board_planes"]["index"]
        self._policy_index = outputs["policy"]["index"]
        self._value_index = outputs["value"]["index"]

    def evaluate_raw(self, position: Position) -> tuple[np.ndarray, float]:
        """Returns `(policy[board_size**2 + 1], value)` for `position`, straight from the net
        -- no illegal-move filtering (see `legal_policy_vector` for that).
        """
        board_planes_nchw = encode_planes(position).numpy()  # [3, H, W], canonical layout
        board_planes_nhwc = np.transpose(board_planes_nchw, (1, 2, 0))[None, ...]  # -> [1, H, W, 3]
        self._interpreter.set_tensor(self._input_index, board_planes_nhwc)
        self._interpreter.invoke()
        policy = self._interpreter.get_tensor(self._policy_index)[0].copy()
        value = float(self._interpreter.get_tensor(self._value_index)[0, 0])
        return policy, value


def legal_policy_vector(position: Position, raw_policy: np.ndarray) -> np.ndarray:
    """Zeroes out illegal moves in `raw_policy` and renormalizes over the remaining ones."""
    size = position.board_size
    legal_indices = {
        (size * size if isinstance(move, Pass) else move.point.to_index(size)) for move in position.legal_moves()
    }
    masked = np.zeros_like(raw_policy)
    for index in legal_indices:
        masked[index] = raw_policy[index]
    total = masked.sum()
    if total <= 0:
        # Degenerate (net assigned ~zero probability to every legal move) -- shouldn't
        # normally happen with a real net; fall back to uniform over legal moves.
        for index in legal_indices:
            masked[index] = 1.0 / len(legal_indices)
    else:
        masked /= total
    return masked


def sample_move(size: int, policy: np.ndarray, temperature: float, rng: random.Random) -> Move:
    """Samples a move from `policy` (a dense, already-legal-masked distribution). Greedy
    (argmax) if `temperature <= 0`, otherwise softmax-with-temperature sampling for
    move-to-move variety across generated games.
    """
    if temperature <= 0:
        chosen = int(np.argmax(policy))
    else:
        weights = np.power(policy, 1.0 / temperature)
        total = weights.sum()
        weights = weights / total if total > 0 else np.full_like(weights, 1.0 / len(weights))
        chosen = rng.choices(range(len(weights)), weights=weights.tolist(), k=1)[0]
    if chosen == size * size:
        return Pass()
    return Play(Point.from_index(chosen, size))


def play_one_game(
    net: KataGoNet, board_size: int, komi: float, temperature: float, max_moves: int, rng: random.Random
) -> tuple[list[tuple[np.ndarray, np.ndarray, Stone]], "Stone | None"]:
    """Plays one self-play game and returns `(records, winner)`, where `records` has one
    `(board_planes, policy_target, to_play)` tuple per move played.
    """
    position = Position.empty(board_size)
    records: list[tuple[np.ndarray, np.ndarray, Stone]] = []
    moves_played = 0
    while position.pass_count < 2 and moves_played < max_moves:
        raw_policy, _ = net.evaluate_raw(position)
        policy_target = legal_policy_vector(position, raw_policy)
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

    net = KataGoNet(Path(config["katago_tflite_path"]))
    rng = random.Random(config["seed"])

    all_planes: list[np.ndarray] = []
    all_policies: list[np.ndarray] = []
    all_values: list[float] = []

    start = time.time()
    games_played = 0
    for game_index in range(config["num_games"]):
        records, winner = play_one_game(
            net, config["board_size"], config["komi"], config["temperature"], config["max_moves"], rng
        )
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
    print(f"Wrote {len(examples)} training examples from {games_played} games to {out_path}")


if __name__ == "__main__":
    main()
