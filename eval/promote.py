"""Decides whether a candidate checkpoint becomes a new shippable difficulty
tier, by playing it against the current tier and applying eval/elo.py's
promotion gate. See docs/ARCHITECTURE.md's "Difficulty-tier promotion".

Usage:
    python -m eval.promote --config configs/eval_base.yaml --candidate <path>.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from eval.elo import DEFAULT_INITIAL_RATING, should_promote, update_ratings
from eval.match import DEFAULT_KOMI, DEFAULT_NUM_SIMULATIONS, play_match


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True, help="Checkpoint to evaluate for promotion")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    # See selfplay/self_play.py's main() for why: MCTS calls net.evaluate() on one board
    # position at a time, and PyTorch's default per-core thread pool is overhead, not
    # speedup, at that batch size.
    torch.set_num_threads(config.get("torch_num_threads", 1))

    current_tier = Path(config["current_tier_checkpoint"])

    results = play_match(
        args.candidate,
        current_tier,
        config["board_size"],
        config["num_games"],
        num_simulations=config.get("num_simulations", DEFAULT_NUM_SIMULATIONS),
        komi=config.get("komi", DEFAULT_KOMI),
        # Only need setting for a legacy checkpoint (no recorded architecture)
        # or to deliberately override -- see RayZeroPolicyValueNet's docstring.
        channels_a=config.get("candidate_channels"),
        channels_b=config.get("current_tier_channels"),
        num_conv_layers_a=config.get("candidate_num_conv_layers"),
        num_conv_layers_b=config.get("current_tier_num_conv_layers"),
        num_residual_blocks_a=config.get("candidate_num_residual_blocks"),
        num_residual_blocks_b=config.get("current_tier_num_residual_blocks"),
        # Real promotion decisions need real statistical power across num_games -- default
        # to nonzero temperature (unlike play_match's own conservative 0.0 default) so a
        # match isn't just 2 unique deterministic games each replicated num_games/2 times.
        # See eval/match.py's docstring and docs/SELF_PLAY_STABILITY.md.
        temperature=config.get("temperature", 1.0),
        temperature_drop_move=config.get("temperature_drop_move", 16),
        seed=config.get("seed", 0),
    )
    candidate_rating, tier_rating = update_ratings(DEFAULT_INITIAL_RATING, DEFAULT_INITIAL_RATING, results)

    promoted = should_promote(candidate_rating, tier_rating, config["min_elo_gap"])
    print(f"Candidate rating: {candidate_rating:.1f} vs current tier ({current_tier}): {tier_rating:.1f}")
    print("PROMOTE" if promoted else "Do not promote")


if __name__ == "__main__":
    main()
