"""Game statistics of a self-play dataset, to check fastplay's games look like selfplay's.

python -m fastplay.dataset_stats <dataset.npz> [...]

Per dataset: games (positions on an empty board), positions per game, Black's win rate (the
value target of each empty-board position), the first move's spread (the most common first
moves, up to symmetry), and how often the recorded policy puts weight on a pass.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

from bootstrap.dataset import SelfPlayExamples


def canonical(point: int, size: int) -> int:
    """The smallest index among `point`'s 8 symmetric images."""
    row, col = divmod(point, size)
    images = []
    for r, c in ((row, col), (col, row)):
        for rr, cc in ((r, c), (size - 1 - r, c), (r, size - 1 - c), (size - 1 - r, size - 1 - c)):
            images.append(rr * size + cc)
    return min(images)


def stats(path: Path) -> str:
    data = SelfPlayExamples.load(path)
    planes = data.board_planes
    size = planes.shape[-1]
    stones = planes[:, 0].sum(axis=(1, 2)) + planes[:, 1].sum(axis=(1, 2))
    empty = np.where(stones == 0)[0]
    one = np.where(stones == 1)[0]
    games = len(empty)
    black_wins = float((data.value_targets[empty] > 0).mean())
    first = Counter()
    for index in one:
        point = int(np.argmax(planes[index, 1].reshape(-1)))  # the opponent's (Black's) stone
        first[canonical(point, size)] += 1
    total_first = sum(first.values())
    spread = ", ".join(f"{divmod(p, size)} {100 * n / total_first:.0f}%" for p, n in first.most_common(4))
    pass_weight = float((data.policy_targets[:, -1] > 0.5).mean())
    return (f"{path.name}: {games} games, {len(planes) / games:.1f} positions a game, Black wins "
            f"{100 * black_wins:.1f}%, first moves {spread}, pass-favoring targets {100 * pass_weight:.2f}%")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(stats(Path(arg)))
