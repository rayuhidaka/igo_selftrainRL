"""Plays games between two checkpoints, to produce eval/elo.py's MatchResults.

Uses mcts/ (a Python port of igo-app/mcts/) for search and engine/ (a
Python port of igo-app/engine/) for rules -- see their READMEs and
docs/ARCHITECTURE.md's "Difficulty-tier promotion" section. Both
checkpoints get equal simulations per move; which checkpoint plays Black
alternates each game so neither gets a first-move-advantage bias across
a match.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from bootstrap.inference import RayZeroPolicyValueNet
from engine.position import Position
from engine.scoring import area_score
from engine.stone import Stone
from eval.elo import MatchResult
from mcts.mcts import Mcts, MctsConfig
from selfplay.generate import sample_move
from selfplay.self_play import visit_count_policy

DEFAULT_NUM_SIMULATIONS = 200
DEFAULT_KOMI = 7.5


def play_match(
    checkpoint_a: Path,
    checkpoint_b: Path,
    board_size: int,
    num_games: int,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    komi: float = DEFAULT_KOMI,
    channels_a: Optional[int] = None,
    channels_b: Optional[int] = None,
    num_conv_layers_a: Optional[int] = None,
    num_conv_layers_b: Optional[int] = None,
    num_residual_blocks_a: Optional[int] = None,
    num_residual_blocks_b: Optional[int] = None,
    temperature: float = 0.0,
    temperature_drop_move: Optional[int] = None,
    seed: Optional[int] = None,
) -> list[MatchResult]:
    """Plays `num_games` games between `checkpoint_a` and `checkpoint_b` and returns one
    `MatchResult` per game, from `checkpoint_a`'s perspective. Alternates which checkpoint
    plays Black each game. `channels_*`/`num_conv_layers_*`/`num_residual_blocks_*` are only
    needed for a *legacy* checkpoint (saved before bootstrap/checkpoint.py existed, so its
    architecture isn't recorded) or to deliberately override -- see `RayZeroPolicyValueNet`.
    They need not match each other, e.g. comparing checkpoints from before/after an
    architecture change.

    `temperature` (default `0.0`, meaning fully deterministic `Mcts.select_move` -- the
    original behavior, matching `igo-app/mcts/Mcts.kt`'s lack of randomness) and
    `temperature_drop_move` control move-selection randomness exactly like
    `selfplay.self_play.play_one_game`'s own parameters, when `temperature > 0`. Added
    2026-09-05 (see docs/SELF_PLAY_STABILITY.md's eval-determinism finding): with
    `temperature=0.0`, two fixed checkpoints and zero noise anywhere means every "A plays
    Black" game is bit-for-bit identical to every other one (same for "B plays Black") --
    a "40-game match" is then really only 2 unique games, each replicated `num_games/2`
    times, giving `eval/elo.py`'s rating update the exact same fixed endpoint for any clean
    2-for-2 sweep regardless of which checkpoints are being compared. Pass a nonzero
    `temperature` (and a `seed`) for a match whose result should carry real statistical
    weight -- e.g. `eval/promote.py`'s actual promotion decisions.
    """
    net_a = RayZeroPolicyValueNet(
        checkpoint_a,
        board_size,
        channels=channels_a,
        num_conv_layers=num_conv_layers_a,
        num_residual_blocks=num_residual_blocks_a,
    )
    net_b = RayZeroPolicyValueNet(
        checkpoint_b,
        board_size,
        channels=channels_b,
        num_conv_layers=num_conv_layers_b,
        num_residual_blocks=num_residual_blocks_b,
    )
    config = MctsConfig(num_simulations=num_simulations, komi=komi)
    mcts_a = Mcts(net_a, config)
    mcts_b = Mcts(net_b, config)
    rng = random.Random(seed)

    results: list[MatchResult] = []
    for game_index in range(num_games):
        a_plays_black = game_index % 2 == 0
        black_mcts, white_mcts = (mcts_a, mcts_b) if a_plays_black else (mcts_b, mcts_a)
        winner = _play_one_game(
            black_mcts, white_mcts, board_size, komi, temperature, temperature_drop_move, rng
        )

        if winner is None:
            score_a = 0.5
        else:
            a_won = (winner == Stone.BLACK) == a_plays_black
            score_a = 1.0 if a_won else 0.0
        results.append(MatchResult(score_a=score_a))

    return results


def _play_one_game(
    black_mcts: Mcts,
    white_mcts: Mcts,
    board_size: int,
    komi: float,
    temperature: float = 0.0,
    temperature_drop_move: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Optional[Stone]:
    position = Position.empty(board_size)
    # A generous cap against a pathological non-terminating game; any
    # reasonably-trained net should finish well before this via two passes.
    max_moves = board_size * board_size * 4
    moves_played = 0
    while position.pass_count < 2 and moves_played < max_moves:
        mover = black_mcts if position.to_play == Stone.BLACK else white_mcts
        if temperature > 0:
            effective_temperature = (
                temperature if temperature_drop_move is None or moves_played < temperature_drop_move else 0.0
            )
            result = mover.search(position)
            policy = visit_count_policy(board_size, result.move_visits)
            move = sample_move(board_size, policy, effective_temperature, rng)
        else:
            move = mover.select_move(position)
        position = position.play(move)
        moves_played += 1
    return area_score(position).winner(komi)
