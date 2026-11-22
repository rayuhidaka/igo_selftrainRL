"""Batched self-play: selfplay/self_play.py's games, many at a time, with the net on the GPU.

Plays `parallel_games` games at once. Every search step walks one leaf per game (fastgo.c,
in C), evaluates all of them as one batch, then backs the values up; each game's search is the
same as mcts/mcts.py's, one simulation at a time (tests/test_fastgo.py checks the visit
counts match exactly). Moves, policy targets, filters and the saved dataset follow
selfplay/self_play.py and take the same config keys, plus:

- `parallel_games` (default 128): games in flight at once, the net's batch size.
- `device` (default "auto"): "cuda" when available, else "cpu".
- `cleanup_labels` (default false): docs/STRONGER_NET.md's step 4. When a game ends on an
  unsettled board, play it on with the cleanup rule (mcts/dead_stone_estimator.py's
  `is_cleanup_move`: only moves in contested regions, never filling your own area or playing
  inside the other side's, never a self-atari; pass when none is left) until two passes, a
  settled board or a third of the board's points in moves, and take the ownership targets from
  there. Each cleanup move is the most visited allowed move (prior breaks ties) of a search
  of `cleanup_simulations` (default `num_simulations`) without root noise. The game's result,
  and so the value and score targets, stay those of the real end, and cleanup positions
  aren't recorded.

`score_utility_weight` isn't supported (the margin-aware experiment is paused).

Usage:
    python -m fastplay.self_play --config <selfplay config>
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from bootstrap.dataset import SelfPlayExamples
from bootstrap.inference import RayZeroPolicyValueNet
from fastplay.fastgo import BLACK, WHITE, Board, SearchPool


def sample_index(policy: np.ndarray, temperature: float, rng: random.Random) -> int:
    """selfplay/generate.py's sample_move, as a move index: argmax at temperature <= 0, else
    sampled from policy ** (1 / temperature) with the same `rng.choices` call."""
    if temperature <= 0:
        return int(np.argmax(policy))
    weights = np.power(policy, 1.0 / temperature)
    total = weights.sum()
    weights = weights / total if total > 0 else np.full_like(weights, 1.0 / len(weights))
    return rng.choices(range(len(weights)), weights=weights.tolist(), k=1)[0]


def visit_policy(visits: np.ndarray, considered: np.ndarray) -> np.ndarray:
    """self_play.py's visit_count_policy: visits normalized, or uniform over the `considered`
    moves (a 0/1 mask) if none has a visit."""
    vector = visits.astype(np.float32)
    total = vector.sum()
    if total <= 0:
        return (considered / considered.sum()).astype(np.float32)
    return vector / total


@dataclass
class Game:
    board: Board
    moves: int = 0
    records: list[tuple[np.ndarray, np.ndarray, int]] = field(default_factory=list)
    end: Board | None = None
    """The board where the game really ended, once it's being cleaned up (`cleanup_labels`)."""
    cleanup_moves: int = 0


class BatchedNet:
    """The checkpoint's net on `device`, evaluating a batch of planes at once."""

    def __init__(self, checkpoint: Path, board_size: int, device: str) -> None:
        self.device = torch.device(device)
        self.model = RayZeroPolicyValueNet(checkpoint, board_size).model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        policy, value, _, _ = self.model(torch.from_numpy(planes).to(self.device))
        return policy.float().cpu().numpy(), value[:, 0].float().cpu().numpy()


def cleanup_move(visits: np.ndarray, priors: np.ndarray, allowed: np.ndarray) -> int:
    """The most visited `allowed` move, the higher prior breaking ties; the pass (last index)
    when none is allowed."""
    if not allowed.any():
        return len(allowed) - 1
    return int(np.argmax(np.where(allowed > 0, visits + priors, -1.0)))


def finish(game: Game, komi: float, size: int) -> dict:
    """The training rows of a finished game (self_play.py's main loop, per game). The result
    comes from where the game ended, the ownership from where its cleanup ended, if any."""
    end = game.end if game.end is not None else game.board
    black, white = end.area_score()
    black_margin = black - (white + komi)
    winner = BLACK if black_margin > 0 else WHITE if black_margin < 0 else None
    owner = game.board.ownership()
    rows = {"planes": [], "policy": [], "value": [], "score": [], "ownership": []}
    for planes, policy, to_play in game.records:
        rows["planes"].append(planes)
        rows["policy"].append(policy)
        rows["value"].append(0.0 if winner is None else 1.0 if winner == to_play else -1.0)
        margin = black_margin if to_play == BLACK else -black_margin
        rows["score"].append(margin / (size * size))
        opponent = WHITE if to_play == BLACK else BLACK
        rows["ownership"].append(np.where(owner == to_play, 1.0, np.where(owner == opponent, -1.0, 0.0)).astype(np.float32))
    relabelled = 0 if game.end is None else int((end.ownership() != owner).sum())
    return {"rows": rows, "winner": winner, "moves": game.moves, "relabelled": relabelled}


def run(config: dict, log=print) -> SelfPlayExamples:
    if config.get("score_utility_weight", 0.0):
        raise ValueError("fastplay doesn't support score_utility_weight")
    size = config["board_size"]
    komi = config.get("komi", 7.5)
    sims = config["num_simulations"]
    if sims > 4096:
        raise ValueError("fastgo.c searches at most 4096 simulations")
    c_puct = config.get("exploration_constant", 1.5)
    fpu_reduction = config.get("fpu_reduction")  # MctsConfig.fpu_reduction; None keeps the old 0.0
    alpha = config.get("root_dirichlet_alpha", 0.1)
    epsilon = config.get("root_dirichlet_epsilon", 0.0)
    temperature = config["temperature"]
    temperature_drop = config.get("temperature_drop_move")
    no_pass_before = config.get("no_pass_before_move")
    max_moves = config["max_moves"]
    min_moves = config.get("min_moves_to_keep", 0)
    max_pass_weight = config.get("max_mid_game_pass_weight")
    cleanup_labels = config.get("cleanup_labels", False)
    cleanup_sims = config.get("cleanup_simulations", sims)
    cleanup_cap = size * size // 3
    num_games = config["num_games"]
    parallel = min(config.get("parallel_games", 128), num_games)
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    net = BatchedNet(Path(config["checkpoint_path"]), size, device)
    rng = random.Random(config["seed"])
    noise_rng = np.random.default_rng(config["seed"])
    pool = SearchPool(parallel, size, sims)
    points = size * size

    slots: list[Game | None] = [Game(Board.empty(size)) for _ in range(parallel)]
    started = parallel
    out = {"planes": [], "policy": [], "value": [], "score": [], "ownership": []}
    played = discarded_short = discarded_pass = cleaned = relabelled = 0
    start = time.time()
    evaluations = 0

    while any(slots):
        for i, game in enumerate(slots):
            if game is None:
                continue
            if game.end is not None:
                pool.begin(i, game.board, cleanup_sims, c_puct, komi, fpu_reduction=fpu_reduction)
                continue
            noise = None
            if epsilon > 0:
                board_moves = int(game.board.legal_mask()[:points].sum())
                if board_moves > 0:
                    noise = noise_rng.dirichlet([alpha] * board_moves)
            pool.begin(i, game.board, sims, c_puct, komi, noise, epsilon, fpu_reduction)
        while True:
            planes = pool.collect()
            if len(planes) == 0:
                break
            policy, values = net(planes)
            pool.expand(policy, values)
            evaluations += len(planes)

        for i, game in enumerate(slots):
            if game is None:
                continue
            visits, _ = pool.root_visits(i)
            if game.end is not None:
                game.board.play(cleanup_move(visits, pool.root_priors(i), game.board.cleanup_mask()))
                game.cleanup_moves += 1
                if game.board.pass_count < 2 and game.cleanup_moves < cleanup_cap and not game.board.is_settled():
                    continue
            else:
                considered = game.board.legal_mask().astype(np.float32)
                if no_pass_before is not None and game.moves < no_pass_before:
                    visits = visits.copy()
                    visits[points] = 0
                    considered[points] = 0
                target = visit_policy(visits, considered)
                game.records.append((game.board.encode(), target, game.board.to_play))
                effective = temperature if temperature_drop is None or game.moves < temperature_drop else 0.0
                game.board.play(sample_index(target, effective, rng))
                game.moves += 1
                if game.board.pass_count < 2 and game.moves < max_moves:
                    continue
                if cleanup_labels and not game.board.is_settled():
                    game.end = game.board.copy()
                    game.board.pass_count = 0
                    continue

            result = finish(game, komi, size)
            slots[i] = Game(Board.empty(size)) if started < num_games else None
            started += slots[i] is not None
            if len(game.records) < min_moves:
                discarded_short += 1
                continue
            if max_pass_weight is not None and any(p[-1] > max_pass_weight for _, p, _ in game.records[:-2]):
                discarded_pass += 1
                continue
            for key in out:
                out[key].extend(result["rows"][key])
            played += 1
            if game.end is not None:
                cleaned += 1
                relabelled += result["relabelled"]
            elapsed = time.time() - start
            log(f"Game {played}/{num_games} done ({result['moves']} moves, winner={result['winner']}), "
                f"{elapsed:.1f}s elapsed, {len(out['planes'])} examples, {evaluations / elapsed:.0f} evals/s")
            if config.get("max_seconds") is not None and elapsed > config["max_seconds"]:
                slots = [None] * parallel
                break

    pool.close()
    log(f"{played} games kept ({discarded_short} too short, {discarded_pass} elevated mid-game Pass weight), "
        f"{time.time() - start:.1f}s")
    if cleanup_labels:
        log(f"Cleanup: {cleaned} kept games ended unsettled and were played on; "
            f"{relabelled / max(1, cleaned):.1f} points a game changed owner")
    return SelfPlayExamples(
        board_planes=np.stack(out["planes"]).astype(np.float32),
        policy_targets=np.stack(out["policy"]).astype(np.float32),
        value_targets=np.array(out["value"], dtype=np.float32),
        score_margin_targets=np.array(out["score"], dtype=np.float32),
        ownership_targets=np.stack(out["ownership"]).astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(config.get("torch_num_threads", 1))
    examples = run(config)
    out_path = Path(config["out_path"])
    examples.save(out_path)
    print(f"Wrote {len(examples)} training examples to {out_path}")


if __name__ == "__main__":
    main()
