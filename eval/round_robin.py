"""Plays every checkpoint in a named set against every other one, to put them all on a
single shared Elo scale -- unlike `eval/promote.py`'s per-match ratings, which each reset
both players to `DEFAULT_INITIAL_RATING` (1500) and are therefore not comparable across
different matches (see docs/ROADMAP.md's "round-robin tournament" backlog item this
implements).

Parallelized across pairings (not within a single pairing's games, which are inherently
sequential -- one Mcts driven loop per game) via `ProcessPoolExecutor`, since `play_match`
is already a plain importable function -- no need for `selfplay/run_parallel.py`'s
subprocess+temp-config plumbing, which exists there only because `self_play.py` is a CLI
entry point, not a function this process could just call directly. `bootstrap/inference.py`
never moves a checkpoint onto a GPU (see its own lack of any `.cuda()`/`device=` handling),
so parallel worker processes never contend over one GPU the way parallel *training* would.

Usage:
    python -m eval.round_robin --config configs/eval_round_robin_option3.yaml
    python -m eval.round_robin --config ... --reuse-existing   # only play pairings not already saved
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from eval.elo import DEFAULT_INITIAL_RATING, DEFAULT_K_FACTOR, MatchResult, expected_score

DEFAULT_NUM_FIT_PASSES = 5


def _play_pairing(
    name_a: str,
    name_b: str,
    checkpoint_a: str,
    checkpoint_b: str,
    board_size: int,
    num_games: int,
    num_simulations: int,
    komi: float,
    temperature: float,
    temperature_drop_move: int,
    seed: int,
) -> tuple[str, str, list[float]]:
    # Imported inside the worker function, not at module scope -- ProcessPoolExecutor pickles
    # this function's closure for each worker process, and torch/mcts import machinery is
    # heavy enough that deferring it to the worker (which needs it anyway) avoids paying the
    # cost twice (once in the parent process that never uses it, again in each worker).
    import torch

    from eval.match import play_match

    # Same reasoning as eval/promote.py's own torch_num_threads: MCTS evaluates one board
    # position at a time, so PyTorch's default per-core thread pool is pure oversubscription
    # overhead once multiple worker processes run concurrently, not speedup.
    torch.set_num_threads(1)

    results = play_match(
        Path(checkpoint_a),
        Path(checkpoint_b),
        board_size,
        num_games,
        num_simulations=num_simulations,
        komi=komi,
        temperature=temperature,
        temperature_drop_move=temperature_drop_move,
        seed=seed,
    )
    return name_a, name_b, [r.score_a for r in results]


def missing_pairings(
    pairings: list[tuple[str, str]], saved_scores: list[tuple[str, str, list[float]]]
) -> list[tuple[int, str, str]]:
    """Returns each of `pairings` (with its index in that list, which seeds its games) that has
    no result in `saved_scores` yet, in either orientation -- for `--reuse-existing`, so adding
    one new checkpoint to a finished round-robin only plays that checkpoint's own pairings
    rather than replaying the whole field. Every existing pairing is already a fixed 40-game
    sample either way, so replaying it would only swap one sample for another.
    """
    played = {frozenset((a, b)) for a, b, _ in saved_scores}
    return [(i, a, b) for i, (a, b) in enumerate(pairings) if frozenset((a, b)) not in played]


def fit_elo_naive_sequential(
    checkpoint_names: list[str],
    pairing_scores: list[tuple[str, str, list[float]]],
    anchor_name: str,
    anchor_rating: float = DEFAULT_INITIAL_RATING,
    k_factor: float = DEFAULT_K_FACTOR,
    num_passes: int = DEFAULT_NUM_FIT_PASSES,
    seed: int = 0,
) -> dict[str, float]:
    """Fits one Elo rating per checkpoint via naive sequential Elo -- kept for comparison, but
    **not used by `main()`** (see `fit_elo_bradley_terry`, which is) after this method was
    caught producing a real inversion: gen5 beats gen4 22-18 and gen8 beats gen7 22-18 in
    their own direct pairings (unambiguous, matching `igo-training`'s own chain-promotion
    history), but this method's fitted ratings ranked gen4 above gen5 and gen7 above gen8
    anyway. Every individual game, across every pairing, is flattened into one big list and
    applied as a standard sequential Elo update against one shared rating pool; re-shuffled and
    re-applied `num_passes` times from the *previous* pass's ratings (not resetting to 1500
    each time) to reduce order-dependence -- evidently not enough of a fix. Left in place as a
    concrete illustration of why a simultaneous fit matters here, not as a live code path.
    """
    ratings = {name: DEFAULT_INITIAL_RATING for name in checkpoint_names}
    games: list[tuple[str, str, float]] = [
        (name_a, name_b, score_a) for name_a, name_b, scores in pairing_scores for score_a in scores
    ]

    rng = random.Random(seed)
    for _ in range(num_passes):
        shuffled = games[:]
        rng.shuffle(shuffled)
        for name_a, name_b, score_a in shuffled:
            rating_a, rating_b = ratings[name_a], ratings[name_b]
            expected_a = expected_score(rating_a, rating_b)
            delta = k_factor * (score_a - expected_a)
            ratings[name_a] = rating_a + delta
            ratings[name_b] = rating_b - delta

    shift = anchor_rating - ratings[anchor_name]
    return {name: rating + shift for name, rating in ratings.items()}


def fit_elo_bradley_terry(
    checkpoint_names: list[str],
    pairing_scores: list[tuple[str, str, list[float]]],
    anchor_name: str,
    anchor_rating: float = DEFAULT_INITIAL_RATING,
    num_iterations: int = 500,
) -> dict[str, float]:
    """Fits one Elo rating per checkpoint via a proper Bradley-Terry maximum-likelihood fit --
    all pairings considered simultaneously, unlike `fit_elo_naive_sequential`'s order-dependent
    running updates (which this replaced after being caught inverting two direct, unambiguous
    head-to-head results -- see that function's docstring).

    Uses the classic Zermelo/Hunter (2004) MM (minorization-maximization) iteration: with
    `p_i = 10**(rating_i / 400)` (chosen so the fitted model's implied win probability,
    `p_i / (p_i + p_j)`, is exactly `eval/elo.py`'s own `expected_score` formula -- the same
    Elo scale, not a different one that happens to share a name), each iteration sets

        p_i <- (total wins by i) / sum_over_opponents_j[ (games_ij) / (p_i + p_j) ]

    which provably converges to the unique (up to an overall scale) maximum-likelihood fit for
    a fully-connected comparison graph -- guaranteed here, since every checkpoint played every
    other one. A draw counts as half a win to each side, standard Bradley-Terry practice
    (irrelevant in practice for this project -- Tromp-Taylor scoring under a non-integer komi
    essentially never ties, and every pairing collected here has zero draws).

    Finally shifts every rating by a constant so `anchor_name` lands exactly on `anchor_rating`,
    same as the naive method -- Elo is only meaningful up to an additive constant.
    """
    total_wins = {name: 0.0 for name in checkpoint_names}
    games_between: dict[tuple[str, str], int] = {}
    for name_a, name_b, scores in pairing_scores:
        total_wins[name_a] += sum(scores)
        total_wins[name_b] += sum(1.0 - s for s in scores)
        games_between[(name_a, name_b)] = games_between.get((name_a, name_b), 0) + len(scores)
        games_between[(name_b, name_a)] = games_between.get((name_b, name_a), 0) + len(scores)

    # A checkpoint that swept every game (or lost every game) against the whole field has an
    # unregularized Bradley-Terry MLE at 0 or +infinity -- a known degenerate case, not a bug in
    # the iteration itself. Floored well below any realistic fitted strength so it saturates at
    # a very high/low (but finite, loggable) rating instead of crashing `math.log10` outright.
    strength_floor = 1e-9

    strengths = {name: 1.0 for name in checkpoint_names}
    for _ in range(num_iterations):
        next_strengths = {}
        for name_i in checkpoint_names:
            denominator = sum(
                games_between.get((name_i, name_j), 0) / (strengths[name_i] + strengths[name_j])
                for name_j in checkpoint_names
                if name_j != name_i
            )
            next_strengths[name_i] = (
                max(total_wins[name_i] / denominator, strength_floor) if denominator > 0 else strengths[name_i]
            )
        # Rescale each iteration to prevent the whole strength vector drifting to 0/infinity --
        # harmless, since only *ratios* of strengths matter to the fit (the final anchor shift
        # fixes the overall scale/offset anyway).
        mean_strength = sum(next_strengths.values()) / len(next_strengths)
        strengths = {name: value / mean_strength for name, value in next_strengths.items()}

    import math

    ratings = {name: 400.0 * math.log10(strength) for name, strength in strengths.items()}
    shift = anchor_rating - ratings[anchor_name]
    return {name: rating + shift for name, rating in ratings.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--refit-only",
        action="store_true",
        help="Skip playing any games; re-fit ratings from an existing raw_out_name file (see "
        "fit_elo_bradley_terry's docstring for why you might want to re-fit without new games).",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Keep every pairing already saved in raw_out_name and only play the ones missing from "
        "it (e.g. a newly added checkpoint's pairings), then re-fit ratings over the combined set.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    checkpoints: dict[str, str] = config["checkpoints"]
    names = list(checkpoints.keys())
    pairings = list(itertools.combinations(names, 2))

    out_dir = Path(config.get("out_dir", "eval"))
    raw_out = out_dir / config.get("raw_out_name", "round_robin_results.json")

    if args.refit_only:
        raw_data = json.loads(raw_out.read_text())
        pairing_scores = [(entry["a"], entry["b"], entry["scores_a"]) for entry in raw_data]
        print(f"Re-fitting from {len(pairing_scores)} pairings already saved at {raw_out} (no games played).")
    else:
        base_seed = config.get("seed", 0)
        pairing_scores = []
        if args.reuse_existing and raw_out.exists():
            saved = [(entry["a"], entry["b"], entry["scores_a"]) for entry in json.loads(raw_out.read_text())]
            # Drop any saved pairing involving a checkpoint no longer in the config, so the fit
            # below only ever sees names it was asked to rate.
            pairing_scores = [(a, b, scores) for a, b, scores in saved if a in checkpoints and b in checkpoints]
        to_play = missing_pairings(pairings, pairing_scores)
        print(
            f"Launching {len(to_play)} pairings ({config['num_games']} games each) across {args.workers} workers"
            f" ({len(pairing_scores)} reused from {raw_out})..."
        )

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _play_pairing,
                    name_a,
                    name_b,
                    checkpoints[name_a],
                    checkpoints[name_b],
                    config["board_size"],
                    config["num_games"],
                    config.get("num_simulations", 50),
                    config.get("komi", 7.5),
                    config.get("temperature", 1.0),
                    config.get("temperature_drop_move", 16),
                    base_seed + i,
                ): (name_a, name_b)
                for i, name_a, name_b in to_play
            }
            completed = 0
            for future in as_completed(futures):
                name_a, name_b, scores = future.result()
                pairing_scores.append((name_a, name_b, scores))
                completed += 1
                wins_a = sum(1 for s in scores if s == 1.0)
                wins_b = sum(1 for s in scores if s == 0.0)
                draws = len(scores) - wins_a - wins_b
                print(
                    f"[{completed}/{len(to_play)}] {name_a} vs {name_b}: {wins_a}-{wins_b}-{draws} (A-B-draw)",
                    flush=True,
                )

        raw_out.parent.mkdir(parents=True, exist_ok=True)
        raw_out.write_text(
            json.dumps([{"a": a, "b": b, "scores_a": scores} for a, b, scores in pairing_scores], indent=2)
        )

    ratings = fit_elo_bradley_terry(names, pairing_scores, anchor_name=config["anchor"])

    for name in sorted(ratings, key=lambda n: ratings[n]):
        print(f"{name}: {ratings[name]:.1f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    elo_out = out_dir / config.get("elo_out_name", "checkpoints_elo.json")
    elo_out.write_text(json.dumps(ratings, indent=2))

    print(f"Fitted ratings written to {elo_out} (raw results at {raw_out}).")


if __name__ == "__main__":
    main()
