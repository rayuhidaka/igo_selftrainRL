"""Runs `selfplay.self_play` as several parallel worker processes instead of one serial
loop, then merges their output into a single dataset at the original config's `out_path`.

`self_play.py` is deliberately single-threaded per worker (`torch.set_num_threads(1)`,
see its own comment) because single-position MCTS inference doesn't benefit from
multi-threading -- but that leaves most of a multi-core machine idle during a run that's
otherwise strictly serial across games. This splits `num_games` evenly across `--workers`
subprocesses (each a real `python -m selfplay.self_play` invocation against a temporary
config that only overrides `num_games`/`seed`/`out_path`), waits for all of them, then
concatenates their `bootstrap.dataset.SelfPlayExamples` outputs -- same total game count
and recipe as a serial run, just wall-clock parallel.

Worker count is a memory/wall-time tradeoff, not a correctness one: each worker's RSS is
roughly the tiny RayZeroNet plus torch/Python overhead (~1GB observed on this machine),
so keep `--workers` low enough to leave real headroom against the WSL VM's memory limit
(`free -h`) -- a killed worker fails the whole run, same as any other crash.

Usage:
    python -m selfplay.run_parallel --config configs/selfplay_self_play_gen5_no_pass_guard.yaml --workers 4
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

from bootstrap.dataset import SelfPlayExamples

_SUMMARY_RE = re.compile(
    r"Wrote (\d+) training examples from (\d+) games to .* "
    r"\((\d+) discarded as too short, (\d+) discarded for elevated mid-game Pass weight\)"
)


def split_evenly(total: int, parts: int) -> list[int]:
    """Splits `total` into `parts` near-equal positive chunks (earlier chunks get the remainder)."""
    base, remainder = divmod(total, parts)
    return [base + 1 if i < remainder else base for i in range(parts)]


def run_worker(base_config: dict, num_games: int, seed: int, out_path: Path, config_path: Path) -> None:
    worker_config = dict(base_config)
    worker_config["num_games"] = num_games
    worker_config["seed"] = seed
    worker_config["out_path"] = str(out_path)
    config_path.write_text(yaml.safe_dump(worker_config))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    base_config = yaml.safe_load(args.config.read_text())
    final_out_path = Path(base_config["out_path"])
    base_seed = base_config["seed"]
    game_counts = split_evenly(base_config["num_games"], args.workers)

    with tempfile.TemporaryDirectory(prefix="selfplay_parallel_") as tmp_str:
        tmp_dir = Path(tmp_str)
        shard_out_paths = [tmp_dir / f"shard_{i}.npz" for i in range(args.workers)]
        shard_config_paths = [tmp_dir / f"shard_{i}.yaml" for i in range(args.workers)]

        for i, (games, out_path, config_path) in enumerate(zip(game_counts, shard_out_paths, shard_config_paths)):
            run_worker(base_config, games, base_seed + i, out_path, config_path)

        print(f"Launching {args.workers} workers for {sum(game_counts)} total games ({game_counts} split)...")
        processes = [
            subprocess.Popen(
                [sys.executable, "-m", "selfplay.self_play", "--config", str(config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for config_path in shard_config_paths
        ]

        outputs = [proc.communicate()[0] for proc in processes]
        for i, (proc, output) in enumerate(zip(processes, outputs)):
            if proc.returncode != 0:
                print(f"--- worker {i} FAILED (exit {proc.returncode}) ---\n{output}")
                raise RuntimeError(f"Worker {i} failed; see output above.")

        total_written = total_played = total_short = total_pass_bias = 0
        for i, output in enumerate(outputs):
            match = _SUMMARY_RE.search(output)
            if match is None:
                raise RuntimeError(f"Worker {i} produced no recognizable summary line:\n{output}")
            written, played, short, pass_bias = (int(g) for g in match.groups())
            total_written += written
            total_played += played
            total_short += short
            total_pass_bias += pass_bias
            print(f"Worker {i}: {written} examples from {played} games ({short} short, {pass_bias} pass-biased)")

        shards = [SelfPlayExamples.load(p) for p in shard_out_paths]
        merged = SelfPlayExamples(
            board_planes=np.concatenate([s.board_planes for s in shards]),
            policy_targets=np.concatenate([s.policy_targets for s in shards]),
            value_targets=np.concatenate([s.value_targets for s in shards]),
            score_margin_targets=np.concatenate([s.score_margin_targets for s in shards]),
        )
        merged.save(final_out_path)

    print(
        f"Merged {total_written} training examples from {total_played} games into {final_out_path} "
        f"(totals: {total_short} discarded as too short, {total_pass_bias} discarded for elevated mid-game Pass weight)"
    )


if __name__ == "__main__":
    main()
