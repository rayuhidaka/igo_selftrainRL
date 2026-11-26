"""Pick a balanced set of capture candidates to generate (igo-app/docs/PUZZLES.md, type B).

The spike finds thousands (most of them ladders and plain ataris); a reviewable set needs a mix.
Keeps at most one candidate per game within `--gap` moves (the same fight shows up again a few
moves later), then up to `--per-technique` of each technique, spread evenly over the capture
lengths (3, 5, 7 ... plies), in a fixed random order.

    python -m puzzles.select_capture runs/capture_spike/candidates.json --out runs/capture_spike/selected.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-technique", type=int, default=75)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    candidates = json.loads(args.candidates.read_text())
    rng.shuffle(candidates)

    # One per game within `gap` moves.
    taken: dict[str, list[int]] = defaultdict(list)
    spread: list[dict] = []
    for c in candidates:
        game, move = c["source"].split("@")
        move = int(move)
        if any(abs(move - m) <= args.gap for m in taken[game]):
            continue
        taken[game].append(move)
        spread.append(c)

    # Per technique, round-robin over the capture lengths.
    chosen: list[dict] = []
    for technique in sorted({c["technique"] for c in spread}):
        by_depth: dict[int, list[dict]] = defaultdict(list)
        for c in spread:
            if c["technique"] == technique:
                by_depth[c["depth"]].append(c)
        picked: list[dict] = []
        while len(picked) < args.per_technique and any(by_depth.values()):
            for depth in sorted(by_depth):
                if by_depth[depth] and len(picked) < args.per_technique:
                    picked.append(by_depth[depth].pop())
        chosen += picked

    args.out.write_text(json.dumps(chosen, indent=1))
    print(f"{len(candidates)} candidates, {len(spread)} after one per fight, {len(chosen)} chosen:",
          dict(Counter(c["technique"] for c in chosen)), dict(sorted(Counter(c["depth"] for c in chosen).items())))


if __name__ == "__main__":
    main()
