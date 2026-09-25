"""The second half of a promotion decision: after `eval/promote.py` confirms a candidate beats
its parent head-to-head, check it didn't get there by losing ground against the rest of the
field.

Why this exists: a self-play chain can produce a candidate that beats its own recent parents
while getting weaker against everything older -- it learns to exploit the specific opponents
it trained against rather than to play better generally. Option 3's gen12 did exactly this
twice (docs/ROADMAP.md): the 1000-game retry beat gen11 by +61.6 in its 80-game promotion match,
yet `eval/round_robin.py` rated it 2217.6 against gen11's 2303.3. `eval/promote.py` only ever
plays the parent, so it cannot see this by construction.

So the gate compares the candidate's and the parent's shared-scale ratings from a round-robin
that includes the candidate (run with `--add-checkpoint` and `--out-prefix`, so a rejected
candidate never lands in the shared ratings igo-app ships), and passes only if the candidate
rates at least `parent - max_drop`.

Usage:
    python -m eval.gate --ratings eval/gate_gen13_checkpoints_elo.json \
        --candidate gen13 --parent gen12 [--max-drop 0]
Exit status 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def passes_field_gate(ratings: dict[str, float], candidate: str, parent: str, max_drop: float = 0.0) -> bool:
    """True iff `candidate`'s shared-scale rating is at least `parent`'s minus `max_drop`.

    `ratings` is a fitted-ratings mapping as `eval/round_robin.py` writes it (name -> Elo).
    Raises `KeyError` naming the missing checkpoint if either isn't in it, rather than
    silently passing or failing a gate on incomplete data.
    """
    for name in (candidate, parent):
        if name not in ratings:
            raise KeyError(f"{name!r} not in fitted ratings -- was it included in the round-robin?")
    return ratings[candidate] >= ratings[parent] - max_drop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ratings", type=Path, required=True, help="Fitted ratings JSON from eval/round_robin.py")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--max-drop", type=float, default=0.0, help="Elo the candidate may trail the parent by")
    args = parser.parse_args()

    ratings = json.loads(args.ratings.read_text())
    passed = passes_field_gate(ratings, args.candidate, args.parent, args.max_drop)
    print(
        f"Field gate: {args.candidate} {ratings[args.candidate]:.1f} vs parent {args.parent} "
        f"{ratings[args.parent]:.1f} (max drop {args.max_drop:.0f}) -> {'PASS' if passed else 'FAIL'}"
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
