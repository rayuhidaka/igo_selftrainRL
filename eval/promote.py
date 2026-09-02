"""Decides whether a candidate checkpoint becomes a new shippable difficulty
tier, by playing it against the current tier and applying eval/elo.py's
promotion gate. See docs/ARCHITECTURE.md's "Difficulty-tier promotion".

Usage:
    python -m eval.promote --config configs/eval_base.yaml --candidate <path>.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from eval.elo import DEFAULT_INITIAL_RATING, should_promote, update_ratings
from eval.match import DEFAULT_KOMI, DEFAULT_NUM_SIMULATIONS, play_match


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True, help="Checkpoint to evaluate for promotion")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    current_tier = Path(config["current_tier_checkpoint"])

    results = play_match(
        args.candidate,
        current_tier,
        config["board_size"],
        config["num_games"],
        num_simulations=config.get("num_simulations", DEFAULT_NUM_SIMULATIONS),
        komi=config.get("komi", DEFAULT_KOMI),
    )
    candidate_rating, tier_rating = update_ratings(DEFAULT_INITIAL_RATING, DEFAULT_INITIAL_RATING, results)

    promoted = should_promote(candidate_rating, tier_rating, config["min_elo_gap"])
    print(f"Candidate rating: {candidate_rating:.1f} vs current tier ({current_tier}): {tier_rating:.1f}")
    print("PROMOTE" if promoted else "Do not promote")


if __name__ == "__main__":
    main()
