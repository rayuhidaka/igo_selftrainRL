"""Spike for best-move puzzles (type C, igo-app/docs/PUZZLES.md): how often does one move stand
clearly above every other, in positions where a player in a real game missed it?

Source: the analysis bench (`~/igo_training_runs/analysis_bench`): 72 games between app levels of
different strength and big4's 1600-simulation reading of every position (Black's win chance).
Candidates: the position before every move that lost at least `--min-loss` of the mover's chances
by that reading, from move 6 on, before the game's last move.

For each candidate:
- a `--sims` root search with big4 (the AI Hint net): visits per move;
- the top `--top` moves by visits, plus the move played, each read by its own `--sims` search of
  the position after it: the mover's chance after that move ("the move's value");
- the gap between the best and the second-best move;
- a second opinion: big5t's own root search, does it choose the same move?
- difficulty signals: the raw policy's first choice of gen1, gen5, gen11 and big4.

Writes `runs/bestmove_spike/candidates.json` and prints a summary.

    python -m puzzles.spike_bestmove
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np

from fastplay.fastgo import BLACK, Board, SearchPool
from fastplay.self_play import BatchedNet

SIZE, KOMI, CPUCT = 9, 7.5, 1.5
PASS = SIZE * SIZE
BENCH = Path.home() / "igo_training_runs/analysis_bench"
CK = Path("checkpoints")
JUDGE = CK / "big4_candidate.pt"
SECOND = CK / "big5t_candidate.pt"
POLICY_NETS = {"gen1": CK / "bootstrap_option3_gen1_candidate.pt", "gen5": CK / "bootstrap_option3_gen5_candidate.pt",
               "gen11": CK / "bootstrap_option3_gen11_candidate.pt", "big4": JUDGE}
BATCH = 128


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def search(net, boards: list[Board], sims: int) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """(visits, root value for the mover, raw priors) for every board, `BATCH` at a time."""
    out = []
    pool = SearchPool(BATCH, SIZE, max(sims, 2))
    for lo in range(0, len(boards), BATCH):
        chunk = boards[lo:lo + BATCH]
        for slot, board in enumerate(chunk):
            pool.begin(slot, board, sims, CPUCT, KOMI)
        while True:
            planes = pool.collect()
            if len(planes) == 0:
                break
            policy, values = net(planes)
            pool.expand(policy, values)
        for slot in range(len(chunk)):
            visits, value = pool.root_visits(slot)
            out.append((visits, value, pool.root_priors(slot)))
    pool.close()
    return out


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sims", type=int, default=1600)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--min-loss", type=float, default=0.10)
    parser.add_argument("--min-move", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("runs/bestmove_spike"))
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    games = pickle.loads((BENCH / "games.pkl").read_bytes())
    bench = np.load(BENCH / "analysis.npz")
    index, black = bench["index"], bench["p1600"]
    at = {(int(g), int(t)): i for i, (g, t) in enumerate(index)}

    # Candidates: the position before a move that lost >= min_loss for its mover.
    cands = []
    for gi, game in enumerate(games):
        moves = game["moves"]
        board = Board.empty(SIZE)
        for t, move in enumerate(moves):
            if t >= args.min_move and t + 1 < len(moves) and move != PASS and (gi, t + 1) in at:
                before, after = black[at[(gi, t)]], black[at[(gi, t + 1)]]
                mover_black = board.to_play == BLACK
                loss = (before - after) if mover_black else (after - before)
                if loss >= args.min_loss:
                    cands.append({"game": gi, "move_number": t, "played": int(move), "loss": float(loss),
                                  "pair": game["pair"], "board": board.copy()})
            board.play(move)
    log(f"{len(games)} games, {len(cands)} candidate positions (a move losing >= {args.min_loss:.0%})")

    judge = BatchedNet(JUDGE, SIZE, device)
    start = time.time()
    roots = search(judge, [c["board"] for c in cands], args.sims)
    log(f"root searches: {time.time() - start:.0f}s")

    # Each candidate move's value: its own search of the position after it.
    children, owners = [], []
    for ci, (c, (visits, _, _)) in enumerate(zip(cands, roots)):
        moves = [int(m) for m in np.argsort(-visits)[: args.top] if visits[m] > 0 and m != PASS]
        if c["played"] not in moves:
            moves.append(c["played"])
        c["moves"] = moves
        for m in moves:
            b = c["board"].copy()
            b.play(m)
            children.append(b)
            owners.append((ci, m))
    start = time.time()
    child_results = search(judge, children, args.sims)
    log(f"{len(children)} move searches: {time.time() - start:.0f}s")
    for (ci, m), (_, value, _) in zip(owners, child_results):
        cands[ci].setdefault("values", {})[m] = 1.0 - (value + 1.0) / 2.0  # the mover's chance after m

    second = BatchedNet(SECOND, SIZE, device)
    start = time.time()
    second_roots = search(second, [c["board"] for c in cands], args.sims)
    log(f"second opinion (big5t): {time.time() - start:.0f}s")

    firsts = {}
    for name, path in POLICY_NETS.items():
        net = BatchedNet(path, SIZE, device)
        firsts[name] = [int(np.argmax(np.where(np.arange(PASS + 1) == PASS, -1, priors)))
                        for _, _, priors in search(net, [c["board"] for c in cands], 1)]

    rows = []
    for ci, c in enumerate(cands):
        ranked = sorted(c["values"].items(), key=lambda kv: -kv[1])
        best, best_v = ranked[0]
        second_v = ranked[1][1] if len(ranked) > 1 else 0.0
        rows.append({
            "game": c["game"], "pair": c["pair"], "move_number": c["move_number"], "played": c["played"],
            "loss": round(c["loss"], 3), "best": best, "best_value": round(best_v, 3),
            "gap": round(best_v - second_v, 3), "played_value": round(c["values"][c["played"]], 3),
            "values": {str(m): round(v, 3) for m, v in ranked},
            "second_opinion": int(np.argmax(second_roots[ci][0])),
            "policy_first": {name: firsts[name][ci] for name in POLICY_NETS},
        })
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "candidates.json").write_text(json.dumps(rows, indent=1))

    n = len(rows)
    print(f"\n{n} candidates; best move = the move played: {sum(r['best'] == r['played'] for r in rows)}")
    for gap in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        sel = [r for r in rows if r["gap"] >= gap and r["best"] != r["played"]]
        agree = sum(r["second_opinion"] == r["best"] for r in sel)
        ahead = sum(r["best_value"] >= 0.5 for r in sel)
        print(f"gap >= {gap:.0%}: {len(sel):4d}  big5t agrees {agree:4d}  best keeps the mover ahead (>= 50%) {ahead:4d}")
    sel = [r for r in rows if r["gap"] >= 0.15 and r["best"] != r["played"] and r["second_opinion"] == r["best"]]
    print(f"\ngap >= 15% and big5t agrees: {len(sel)}")
    for name in POLICY_NETS:
        print(f"  {name}'s raw first choice is the answer: {sum(r['policy_first'][name] == r['best'] for r in sel)}")
    print("  by pair:", dict(Counter(r["pair"] for r in sel)))
    print("  by move number:", dict(sorted(Counter(r["move_number"] // 10 * 10 for r in sel).items())))
    print("  best value:", dict(sorted(Counter(int(r["best_value"] * 10) * 10 for r in sel).items())))


if __name__ == "__main__":
    main()
