"""Prints a checkpoint's raw policy argmax + top-5 moves on an empty board -- the cheap
diagnostic this project uses to compare opening-move quality across checkpoints without a
full export-to-tflite + igo-app on-device cycle. See igo-app/docs/ROADMAP.md's "weak opening
moves" item for why this exists and docs/ROADMAP.md's mirrored entry here for how it's used
in the augmentation follow-up plan.

Usage:
    python -m tools.spotcheck_opening checkpoints/bootstrap_gen11_no_pass_guard_candidate.pt [more checkpoints...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bootstrap.inference import RayZeroPolicyValueNet
from engine.position import Position

_BOARD_SIZE = 9


def spot_check(checkpoint_path: Path, board_size: int = _BOARD_SIZE, top_n: int = 5) -> None:
    net = RayZeroPolicyValueNet(checkpoint_path, board_size)
    evaluation = net.evaluate(Position.empty(board_size))
    ranked = sorted(evaluation.policy.items(), key=lambda item: item[1], reverse=True)[:top_n]

    print(f"\n{checkpoint_path}")
    for move, prob in ranked:
        print(f"  {move}: {prob:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--board-size", type=int, default=_BOARD_SIZE)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    for checkpoint_path in args.checkpoints:
        spot_check(checkpoint_path, args.board_size, args.top_n)


if __name__ == "__main__":
    main()
