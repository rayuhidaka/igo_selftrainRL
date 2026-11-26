"""Best-move puzzles (type C): igo-app/docs/PUZZLES.md, "Best-move puzzles (type C): design",
decisions C1-C9 (publisher, 2026-11-26).

From games between app levels (`eval/puzzle_refill_games.json`), the positions where a player
missed one clearly best move:

1. **Sweep:** every position read by big4 (the AI Hint net) at `SWEEP_SIMS`; a move whose mover
   lost at least `SUSPECT_DROP` there is a suspect.
2. **Deep read** of each suspect: a `DEEP_SIMS` root search, then its top `TOP` moves and the move
   played each searched on their own (`DEEP_SIMS` of the position after the move: the mover's
   chance after it, and the AI's line). Kept if the played move lost at least `MIN_LOSS` against
   the best one, and the best is ahead of the second by at least `MIN_GAP` (C1).
3. **Confirm (C1):** big5t's own `DEEP_SIMS` search chooses the same move, and so does big4 on the
   board turned by a symmetry, by the same gap.
4. **One per fight:** per game, the largest gap first, none within `GAP_MOVES` moves of another.
5. **Every move (C4):** each other legal move searched on its own at `REST_SIMS`, so the app can
   answer any tap with its chance and line without searching.
6. **The puzzle:** difficulty by which net's raw first choice is the answer (C5: gen1 1, gen5 2,
   gen11 or big4 3, else 4), A/B/C = the answer and the two next-best moves, the hint area = the
   points within two of the answer, the kind (saves stones in atari / capture or atari / quiet).

Each stage is saved under `--out` and reused when the run is restarted.

    python -m puzzles.generate_bestmove --games 400 --out runs/bestmove_puzzles
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from engine.capture import chain_at, liberties_of
from engine.point import Point
from engine.stone import Stone
from fastplay.fastgo import BLACK, NO_KO, Board, SearchPool
from fastplay.self_play import BatchedNet
from puzzles.generate import history_moves

SIZE, KOMI, CPUCT = 9, 7.5, 1.5
POINTS = SIZE * SIZE
PASS = POINTS
CK = Path("checkpoints")
JUDGE = CK / "big4_candidate.pt"
SECOND = CK / "big5t_candidate.pt"
POLICY_NETS = {"gen1": CK / "bootstrap_option3_gen1_candidate.pt", "gen5": CK / "bootstrap_option3_gen5_candidate.pt",
               "gen11": CK / "bootstrap_option3_gen11_candidate.pt", "big4": JUDGE}
SWEEP_SIMS, DEEP_SIMS, REST_SIMS = 200, 1600, 400
TOP, MIN_GAP, MIN_LOSS, SUSPECT_DROP = 5, 0.15, 0.10, 0.07
MIN_MOVE, GAP_MOVES, LINE_LEN, HINT_RADIUS = 6, 8, 8, 2
BATCH = 128


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def stage(out: Path, name: str, compute):
    """`compute()`'s result, saved as `out/name.pkl` and reused on a restart."""
    path = out / f"{name}.pkl"
    if path.exists():
        log(f"{name}: reusing {path}")
        return pickle.loads(path.read_bytes())
    result = compute()
    path.write_bytes(pickle.dumps(result))
    return result


def search(net, boards: list[Board], sims: int) -> list[tuple[np.ndarray, float, list[int]]]:
    """(root visits, root value for the mover, principal variation) for every board."""
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
            out.append((visits, value, pool.principal_variation(slot, max_len=LINE_LEN)))
    pool.close()
    return out


def after(board: Board, move: int) -> Board:
    b = board.copy()
    b.play(move)
    return b


def chance_after(value_for_next_mover: float) -> float:
    """The mover's chance after its move, from the search of the position after it."""
    return 1.0 - (value_for_next_mover + 1.0) / 2.0


def sym_index(index: int, k: int) -> int:
    """Point `index` under symmetry `k` (0-7: rotations by 90 degrees, then a mirror)."""
    r, c = divmod(index, SIZE)
    for _ in range(k % 4):
        r, c = c, SIZE - 1 - r
    if k >= 4:
        c = SIZE - 1 - c
    return r * SIZE + c


def sym_board(board: Board, k: int) -> Board:
    b = Board.empty(SIZE)
    for i in range(POINTS):
        b.stones[sym_index(i, k)] = board.stones[i]
    b.to_play, b.pass_count = board.to_play, board.pass_count
    b.ko = NO_KO if board.ko == NO_KO else sym_index(board.ko, k)
    return b


def game_moves(game: dict) -> list[int]:
    return [PASS if m < 0 else m for m in game["history"]]


def boards_of(moves: list[int]) -> list[Board]:
    board = Board.empty(SIZE)
    out = [board.copy()]
    for m in moves:
        board.play(m)
        out.append(board.copy())
    return out


def evaluate_moves(net, items: list[tuple[int, Board, list[int]]], sims: int) -> dict[int, dict[int, tuple[float, list[int]]]]:
    """For each (key, board, moves): every move's chance for the mover and its line."""
    flat = [(key, m, after(board, m)) for key, board, moves in items for m in moves]
    results = search(net, [b for _, _, b in flat], sims)
    out: dict[int, dict[int, tuple[float, list[int]]]] = {}
    for (key, m, _), (_, value, line) in zip(flat, results):
        out.setdefault(key, {})[m] = (chance_after(value), line)
    return out


def kind_of(position, answer: int) -> str:
    p = Point.from_index(answer, SIZE)
    me = position.to_play
    save = capture = False
    for n in position.neighbors(p):
        s = position.stone_at(n)
        if s == Stone.EMPTY:
            continue
        libs = len(liberties_of(position, chain_at(position, n)))
        if s == me and libs == 1:
            save = True
        if s != me and libs <= 2:
            capture = True
    return "saves stones" if save else "capture or atari" if capture else "quiet"


def rows_of(position) -> list[str]:
    sym = {Stone.BLACK: "X", Stone.WHITE: "O", Stone.EMPTY: "."}
    return ["".join(sym[position.stone_at(Point(r, c))] for c in range(SIZE)) for r in range(SIZE)]


def rc(index: int) -> list[int]:
    return [index // SIZE, index % SIZE]


def key(index: int) -> str:
    return f"{index // SIZE},{index % SIZE}"


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games-file", type=Path, default=Path("eval/puzzle_refill_games.json"))
    parser.add_argument("--games", type=int, default=400, help="spread evenly over the file")
    parser.add_argument("--out", type=Path, default=Path("runs/bestmove_puzzles"))
    parser.add_argument("--first-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261126)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_games = json.loads(args.games_file.read_text())
    stride = max(1, len(all_games) // args.games)
    picked = list(range(0, len(all_games), stride))[: args.games]
    games = {gi: game_moves(all_games[gi]) for gi in picked}
    judge = BatchedNet(JUDGE, SIZE, device)

    # 1. Sweep.
    def sweep():
        flat = [(gi, t, b) for gi, moves in games.items() for t, b in enumerate(boards_of(moves))]
        start = time.time()
        res = search(judge, [b for _, _, b in flat], SWEEP_SIMS)
        black = {}
        for (gi, t, b), (_, value, _) in zip(flat, res):
            mover = (value + 1) / 2
            black[(gi, t)] = mover if b.to_play == BLACK else 1 - mover
        log(f"sweep: {len(flat)} positions of {len(games)} games in {time.time() - start:.0f}s")
        return black

    black = stage(args.out, "1_sweep", sweep)
    suspects = []
    for gi, moves in games.items():
        boards = boards_of(moves)
        for t in range(MIN_MOVE, len(moves) - 1):
            if moves[t] == PASS:
                continue
            drop = black[(gi, t)] - black[(gi, t + 1)]
            if boards[t].to_play != BLACK:
                drop = -drop
            if drop >= SUSPECT_DROP:
                suspects.append((gi, t))
    log(f"{len(suspects)} suspects (a {SUSPECT_DROP:.0%} drop at {SWEEP_SIMS} simulations)")

    def board_at(gi: int, t: int) -> Board:
        return boards_of(games[gi][:t])[-1]

    # 2. Deep read.
    def deep():
        boards = [board_at(gi, t) for gi, t in suspects]
        start = time.time()
        roots = search(judge, boards, DEEP_SIMS)
        # First only the move played: a suspect whose deep reading shows no real loss stops here
        # (most of them), before its top moves are searched.
        played = evaluate_moves(judge, [(n, b, [games[gi][t]]) for n, ((gi, t), b) in enumerate(zip(suspects, boards))], DEEP_SIMS)
        items = []
        for n, ((gi, t), board, (visits, root_value, _)) in enumerate(zip(suspects, boards, roots)):
            if (root_value + 1) / 2 - played[n][games[gi][t]][0] < MIN_LOSS * 0.8:
                continue
            moves = [int(m) for m in np.argsort(-visits)[:TOP] if visits[m] > 0 and m != PASS and m != games[gi][t]]
            items.append((n, board, moves))
        values = evaluate_moves(judge, items, DEEP_SIMS)
        for n in values:
            values[n].update(played[n])
        log(f"deep read: {len(items)} of {len(suspects)} suspects lose at least {MIN_LOSS * 0.8:.0%} at {DEEP_SIMS}")
        kept = []
        for n, (gi, t) in enumerate(suspects):
            if n not in values:
                continue
            ranked = sorted(values[n].items(), key=lambda kv: -kv[1][0])
            best, (best_v, _) = ranked[0]
            second_v = ranked[1][1][0] if len(ranked) > 1 else 0.0
            played_v = values[n][games[gi][t]][0]
            if best != games[gi][t] and best_v - played_v >= MIN_LOSS and best_v - second_v >= MIN_GAP:
                kept.append({"game": gi, "t": t, "best": best, "gap": best_v - second_v, "values": values[n]})
        log(f"deep read: {len(suspects)} suspects in {time.time() - start:.0f}s, {len(kept)} with a clear best move")
        return kept

    found = stage(args.out, "2_deep", deep)

    # 3. Confirm: big5t, and big4 on a turned board.
    def confirm():
        start = time.time()
        boards = [board_at(c["game"], c["t"]) for c in found]
        second = BatchedNet(SECOND, SIZE, device)
        second_choice = [int(np.argmax(v[:POINTS])) for v, _, _ in search(second, boards, DEEP_SIMS)]
        rng = random.Random(args.seed)
        ks = [rng.randrange(1, 8) for _ in found]
        turned = [sym_board(b, k) for b, k in zip(boards, ks)]
        roots = search(judge, turned, DEEP_SIMS)
        items = []
        for n, (c, k, (visits, _, _)) in enumerate(zip(found, ks, roots)):
            moves = [int(m) for m in np.argsort(-visits)[:TOP] if visits[m] > 0 and m != PASS]
            answer = sym_index(c["best"], k)
            if answer not in moves:
                moves.append(answer)
            items.append((n, turned[n], moves))
        values = evaluate_moves(judge, items, DEEP_SIMS)
        kept = []
        for n, (c, k) in enumerate(zip(found, ks)):
            ranked = sorted(values[n].items(), key=lambda kv: -kv[1][0])
            turned_ok = ranked[0][0] == sym_index(c["best"], k) and (
                len(ranked) < 2 or ranked[0][1][0] - ranked[1][1][0] >= MIN_GAP)
            c["second_agrees"] = second_choice[n] == c["best"]
            c["turned_ok"] = turned_ok
            if c["second_agrees"] and turned_ok:
                kept.append(c)
        log(f"confirm: {len(found)} in {time.time() - start:.0f}s; big5t agrees {sum(c['second_agrees'] for c in found)}, "
            f"turned board agrees {sum(c['turned_ok'] for c in found)}, both {len(kept)}")
        return kept

    confirmed = stage(args.out, "3_confirm", confirm)

    # 4. One per fight.
    chosen = []
    for gi in games:
        mine = sorted((c for c in confirmed if c["game"] == gi), key=lambda c: -c["gap"])
        taken: list[int] = []
        for c in mine:
            if all(abs(c["t"] - t) > GAP_MOVES for t in taken):
                taken.append(c["t"])
                chosen.append(c)
    log(f"one per fight: {len(chosen)}")

    # 5. Every legal move.
    def every_move():
        start = time.time()
        items = []
        for n, c in enumerate(chosen):
            board = board_at(c["game"], c["t"])
            legal = [m for m in range(POINTS) if board.legal_mask()[m] and m not in c["values"]]
            items.append((n, board, legal))
        values = evaluate_moves(judge, items, REST_SIMS)
        for n, c in enumerate(chosen):
            c["rest"] = values.get(n, {})
        log(f"every move: {sum(len(i[2]) for i in items)} moves of {len(chosen)} positions in {time.time() - start:.0f}s")
        # C1 is about every other move, not only the top ones: a move the light search puts within
        # MIN_GAP of the answer is read deeply too, and if it stays that close the puzzle goes.
        recheck = []
        for n, c in enumerate(chosen):
            best_v = c["values"][c["best"]][0]
            close = [m for m, (v, _) in c["rest"].items() if v > best_v - MIN_GAP]
            if close:
                recheck.append((n, board_at(c["game"], c["t"]), close))
        deep_values = evaluate_moves(judge, recheck, DEEP_SIMS)
        kept = []
        for n, c in enumerate(chosen):
            c["rest"].update(deep_values.get(n, {}))
            best_v = c["values"][c["best"]][0]
            others = [v for m, (v, _) in {**c["rest"], **c["values"]}.items() if m != c["best"]]
            c["gap"] = best_v - max(others, default=0.0)
            if c["gap"] >= MIN_GAP:
                kept.append(c)
        log(f"every move: {sum(len(r[2]) for r in recheck)} close moves read deeply; {len(kept)} of {len(chosen)} keep the gap")
        return kept

    chosen = stage(args.out, "5_every_move", every_move)

    # 6. Puzzles.
    boards = [board_at(c["game"], c["t"]) for c in chosen]
    firsts = {}
    for name, path in POLICY_NETS.items():
        net = SearchPoolPriors(path, device)
        firsts[name] = net.first_choices(boards)
    puzzles = []
    rng = random.Random(args.seed)
    for n, (c, board) in enumerate(zip(chosen, boards)):
        gi, t = c["game"], c["t"]
        position = board.to_position()
        everything = {**{m: v for m, v in c["rest"].items()}, **c["values"]}
        ranked = sorted(everything.items(), key=lambda kv: -kv[1][0])
        answer = c["best"]
        assert ranked[0][0] == answer
        level = 4
        if firsts["gen1"][n] == answer:
            level = 1
        elif firsts["gen5"][n] == answer:
            level = 2
        elif firsts["gen11"][n] == answer or firsts["big4"][n] == answer:
            level = 3
        choices = [rc(answer), rc(ranked[1][0]), rc(ranked[2][0])]
        rng.shuffle(choices)
        ar, ac = divmod(answer, SIZE)
        region = [[r, cc] for r in range(SIZE) for cc in range(SIZE) if max(abs(r - ar), abs(cc - ac)) <= HINT_RADIUS]
        game = all_games[gi]
        puzzles.append({
            "id": f"bm-{len(puzzles) + args.first_id:04d}",
            "kind": "BEST",
            "toPlay": "X" if board.to_play == BLACK else "O",
            "difficulty": level,
            "board": rows_of(position),
            "target": rc(answer),
            "region": region,
            "answer": rc(answer),
            "choices": choices,
            "values": {key(m): round(v, 3) for m, (v, _) in ranked},
            "lines": {key(m): [rc(x) for x in line] for m, (_, line) in ranked},
            "history": history_moves(game, t, position),
            "source": f"{args.games_file.name}#{gi}@{t}",
            "matchup": f"{game['black']} vs {game['white']}",
            "technique": kind_of(position, answer),
            "gap": round(c["gap"], 3),
            "played": rc(games[gi][t]),
        })
    (args.out / "puzzles.json").write_text(json.dumps(puzzles, indent=1))
    summary = {
        "games": len(games), "suspects": len(suspects), "clear": len(found), "confirmed": len(confirmed),
        "puzzles": len(puzzles),
        "by_level": dict(sorted(Counter(p["difficulty"] for p in puzzles).items())),
        "by_kind": dict(Counter(p["technique"] for p in puzzles)),
        "by_matchup": dict(Counter(p["matchup"] for p in puzzles)),
        "best_below_half": sum(p["values"][key(rc_to_index(p["answer"]))] < 0.5 for p in puzzles),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def rc_to_index(point: list[int]) -> int:
    return point[0] * SIZE + point[1]


class SearchPoolPriors:
    """A net's raw first choice (no search, no pass) for each board."""

    def __init__(self, path: Path, device: str) -> None:
        self.net = BatchedNet(path, SIZE, device)

    def first_choices(self, boards: list[Board]) -> list[int]:
        out = []
        for lo in range(0, len(boards), BATCH):
            planes = np.stack([b.encode() for b in boards[lo:lo + BATCH]]).astype(np.float32)
            policy, _ = self.net(planes)
            for b, p in zip(boards[lo:lo + BATCH], policy):
                legal = b.legal_mask()[:POINTS].astype(bool)
                scores = np.where(legal, p[:POINTS], -np.inf)
                out.append(int(np.argmax(scores)))
        return out


if __name__ == "__main__":
    main()
