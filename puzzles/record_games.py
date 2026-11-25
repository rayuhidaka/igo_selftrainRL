"""Source games for a puzzle refill (igo-app/docs/PUZZLES.md, "refill to 100+"): games between two
igo-app levels of different strength, so fights get decided by real mistakes, recorded in the
`eval/*games.json` format `puzzles.spike_mine` reads (`history`: point indices, -1 a pass).

Each move is a search of the mover's net with fastplay's batched C search on the GPU, `--sims`
simulations (200, as in the app's lower levels); the first `--sampled` moves are sampled from the
visits for variety, later ones are the most visited; no pass before move 20 (as in the app); at
most 150 moves. Every pairing plays `--games` games, half with each side as Black.

    python -m puzzles.record_games --games 256 --out eval/puzzle_refill_games.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from fastplay.fastgo import Board, SearchPool
from fastplay.self_play import BatchedNet, sample_index, visit_policy

SIZE, KOMI, CPUCT = 9, 7.5, 1.5
POINTS = SIZE * SIZE
NO_PASS_BEFORE, MAX_MOVES = 20, 150
CK = "checkpoints/"
# (name, weaker, stronger): from Beginner-ish nets up to Level 11's, so both easy and hard fights appear.
PAIRS = [
    ("gen1_vs_gen5", CK + "bootstrap_option3_gen1_candidate.pt", CK + "bootstrap_option3_gen5_candidate.pt"),
    ("gen2_vs_gen6", CK + "bootstrap_option3_gen2_candidate.pt", CK + "bootstrap_option3_gen6_candidate.pt"),
    ("gen5_vs_gen10", CK + "bootstrap_option3_gen5_candidate.pt", CK + "bootstrap_option3_gen10_candidate.pt"),
    ("gen8_vs_gen11", CK + "bootstrap_option3_gen8_candidate.pt", CK + "bootstrap_option3_gen11_candidate.pt"),
    ("gen10_vs_big1", CK + "bootstrap_option3_gen10_candidate.pt", CK + "big10x128_distill_phase2.pt"),
    ("gen11_vs_big4", CK + "bootstrap_option3_gen11_candidate.pt", CK + "big4_candidate.pt"),
]


def search(pool: SearchPool, net: BatchedNet, items: list[tuple[int, Board]], sims: int) -> dict[int, np.ndarray]:
    """Searches every (slot, board) in `items` on `pool` with `net`; returns each slot's visits."""
    for slot, board in items:
        pool.begin(slot, board, sims, CPUCT, KOMI)
    if items:
        while True:
            planes = pool.collect()
            if len(planes) == 0:
                break
            policy, values = net(planes)
            pool.expand(policy, values)
    return {slot: pool.root_visits(slot)[0] for slot, _ in items}


def play_pair(name: str, weak_path: str, strong_path: str, games: int, sims: int, sampled: int, seed: int, device: str) -> list[dict]:
    weak, strong = BatchedNet(Path(weak_path), SIZE, device), BatchedNet(Path(strong_path), SIZE, device)
    rng = random.Random(seed)
    pool_w, pool_s = SearchPool(games, SIZE, sims), SearchPool(games, SIZE, sims)
    boards = [Board.empty(SIZE) for _ in range(games)]
    history: list[list[int]] = [[] for _ in range(games)]
    weak_black = [g % 2 == 0 for g in range(games)]
    live = set(range(games))
    move = 0
    while live:
        black_to_play = move % 2 == 0  # every game alternates in step: same color to move everywhere
        visits = search(pool_w, weak, [(g, boards[g]) for g in live if weak_black[g] == black_to_play], sims)
        visits.update(search(pool_s, strong, [(g, boards[g]) for g in live if weak_black[g] != black_to_play], sims))
        for g in list(live):
            v = visits[g].copy()
            legal = boards[g].legal_mask().astype(np.float32)
            if move < NO_PASS_BEFORE:
                v[POINTS] = 0
                legal[POINTS] = 0
            index = sample_index(visit_policy(v, legal), 1.0 if move < sampled else 0.0, rng)
            boards[g].play(index)
            history[g].append(-1 if index == POINTS else index)
            if boards[g].pass_count >= 2 or len(history[g]) >= MAX_MOVES:
                live.discard(g)
        move += 1
    pool_w.close()
    pool_s.close()
    weak_name, strong_name = name.split("_vs_")
    out = []
    for g in range(games):
        black, white = (weak_name, strong_name) if weak_black[g] else (strong_name, weak_name)
        out.append({"black": black, "white": white, "board_size": SIZE, "moves": len(history[g]), "history": history[g]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=256, help="games per pairing")
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--sampled", type=int, default=8, help="moves sampled from the visits at the start")
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    games: list[dict] = []
    for i, (name, weak, strong) in enumerate(PAIRS):
        started = time.time()
        games += play_pair(name, weak, strong, args.games, args.sims, args.sampled, args.seed + i, device)
        print(f"{time.strftime('%H:%M:%S')} {name}: {args.games} games in {time.time() - started:.0f}s", flush=True)
    args.out.write_text(json.dumps(games))
    print(f"{len(games)} games, {sum(g['moves'] for g in games)} moves -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
