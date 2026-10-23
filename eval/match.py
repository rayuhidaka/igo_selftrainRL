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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bootstrap.inference import RayZeroPolicyValueNet
from engine.move import Move, Pass, Play
from engine.position import Position
from engine.scoring import area_score, is_own_territory_fill, territory_score
from engine.stone import Stone
from eval.elo import MatchResult
from mcts.mcts import Mcts, MctsConfig
from selfplay.generate import sample_move
from selfplay.self_play import visit_count_policy

DEFAULT_NUM_SIMULATIONS = 200
DEFAULT_KOMI = 7.5


@dataclass(frozen=True)
class PlayerStyle:
    """How one side of a match searches and picks its moves, when that differs from the match's
    shared settings -- e.g. igo-app's "Beginner" level, which runs the gen1 checkpoint with far
    fewer simulations and samples its moves instead of always playing the best one. Mirrors
    `igo-app/mcts/Mcts.kt`'s `MctsConfig.numSimulations` / `MctsConfig.moveTemperature`, so the
    level the app ships is the level this measures.
    """

    num_simulations: Optional[int] = None
    """This side's simulations per move; `None` uses the match's `num_simulations`."""

    temperature: Optional[float] = None
    """When set, this side samples every move of the game from its visit counts at this
    temperature (no `temperature_drop_move` cutoff) instead of following the match's own
    temperature schedule. Like `Mcts.kt`'s `moveTemperature`, it only samples among stone
    placements: it passes when the most-visited move is a pass, and never samples a pass
    otherwise -- a weak level should misplace stones, not end the game by accident."""

    pass_instead_of_self_fill: bool = False
    """When true, right after the opponent passes this side passes (ending the game) instead of
    placing a stone that fills its own territory (`engine.scoring.is_own_territory_fill`) --
    igo-app's proposed rule for territory scoring, measured head-to-head by
    `configs/eval_self_fill_head_to_head.yaml`."""

    score_utility_weight: float = 0.0
    """This side's `MctsConfig.score_utility_weight` -- `0.0` plays for the win only, as every
    shipped level does. Measured by `configs/eval_score_utility.yaml` (margin-aware training,
    stage 2)."""

    def mcts_config(self, num_simulations: int, komi: float) -> MctsConfig:
        """This side's search settings, falling back to the match's shared `num_simulations`."""
        return MctsConfig(
            num_simulations=self.num_simulations or num_simulations,
            komi=komi,
            score_utility_weight=self.score_utility_weight,
        )


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
    style_a: PlayerStyle = PlayerStyle(),
    style_b: PlayerStyle = PlayerStyle(),
    territory_komi: Optional[float] = None,
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

    `style_a`/`style_b` override one side's search budget and move selection (see
    `PlayerStyle`); the defaults leave both sides on the shared settings above.

    `territory_komi`, when set, decides each game by Japanese-style territory scoring
    (`engine.scoring.territory_score`, prisoners included) at that komi instead of area scoring
    at `komi` -- the games themselves are played the same way, the nets searching with the area
    `komi` they were trained on. Added 2026-10-23 so igo-app can re-fit its level ratings under
    the territory scoring it now uses by default.
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
    mcts_a = Mcts(net_a, style_a.mcts_config(num_simulations, komi))
    mcts_b = Mcts(net_b, style_b.mcts_config(num_simulations, komi))
    rng = random.Random(seed)

    results: list[MatchResult] = []
    for game_index in range(num_games):
        a_plays_black = game_index % 2 == 0
        black_mcts, white_mcts = (mcts_a, mcts_b) if a_plays_black else (mcts_b, mcts_a)
        black_style, white_style = (style_a, style_b) if a_plays_black else (style_b, style_a)
        if territory_komi is None:
            winner = _play_one_game(
                black_mcts,
                white_mcts,
                board_size,
                komi,
                temperature,
                temperature_drop_move,
                rng,
                black_style,
                white_style,
            )
        else:
            end = play_game_to_end(
                black_mcts, white_mcts, board_size, temperature, temperature_drop_move, rng, black_style, white_style
            )
            winner = territory_score(end.position, end.black_prisoners, end.white_prisoners).winner(territory_komi)

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
    black_style: PlayerStyle = PlayerStyle(),
    white_style: PlayerStyle = PlayerStyle(),
) -> Optional[Stone]:
    end = play_game_to_end(
        black_mcts, white_mcts, board_size, temperature, temperature_drop_move, rng, black_style, white_style
    )
    return area_score(end.position).winner(komi)


@dataclass(frozen=True)
class GameEnd:
    """Where one game finished: its final `position` (two passes, unless it hit the move cap),
    how many `moves` it took, and the prisoners each color took -- `black_prisoners` is how many
    white stones Black captured. Prisoners are what `engine.scoring.territory_score` adds on top
    of territory; area scoring doesn't need them. `history` is every move in order, so the game
    can be replayed (`eval/rule_comparison.py`'s self-fill analysis does). `self_fill_pass` is
    the color whose `PlayerStyle.pass_instead_of_self_fill` ended the game, if one did."""

    position: Position
    moves: int
    black_prisoners: int
    white_prisoners: int
    history: tuple[Move, ...] = ()
    self_fill_pass: Optional[Stone] = None

    @property
    def finished(self) -> bool:
        """Whether the game ended in two passes rather than at the move cap."""
        return self.position.pass_count >= 2


def play_game_to_end(
    black_mcts: Mcts,
    white_mcts: Mcts,
    board_size: int,
    temperature: float = 0.0,
    temperature_drop_move: Optional[int] = None,
    rng: Optional[random.Random] = None,
    black_style: PlayerStyle = PlayerStyle(),
    white_style: PlayerStyle = PlayerStyle(),
) -> GameEnd:
    """Plays one game between `black_mcts` and `white_mcts` (move selection as in `play_match`)
    and returns where it ended, prisoners included -- `eval/rule_comparison.py` counts the same
    game under both area and territory scoring."""
    position = Position.empty(board_size)
    # A generous cap against a pathological non-terminating game; any
    # reasonably-trained net should finish well before this via two passes.
    max_moves = board_size * board_size * 4
    moves_played = 0
    prisoners = {Stone.BLACK: 0, Stone.WHITE: 0}
    history: list[Move] = []
    self_fill_pass: Optional[Stone] = None
    while position.pass_count < 2 and moves_played < max_moves:
        mover = black_mcts if position.to_play == Stone.BLACK else white_mcts
        style = black_style if position.to_play == Stone.BLACK else white_style
        if style.temperature is not None and style.temperature > 0:
            move = sample_placement(board_size, mover.search(position).move_visits, style.temperature, rng)
        elif temperature > 0:
            effective_temperature = (
                temperature if temperature_drop_move is None or moves_played < temperature_drop_move else 0.0
            )
            result = mover.search(position)
            policy = visit_count_policy(board_size, result.move_visits)
            move = sample_move(board_size, policy, effective_temperature, rng)
        else:
            move = mover.select_move(position)
        if (
            style.pass_instead_of_self_fill
            and position.pass_count == 1
            and isinstance(move, Play)
            and is_own_territory_fill(position, move.point)
        ):
            move = Pass()
            self_fill_pass = position.to_play
        mover_color = position.to_play
        opponent_stones = position.stones.count(mover_color.opponent())
        position = position.play(move)
        history.append(move)
        # Suicide is illegal, so a move only ever removes the opponent's stones.
        prisoners[mover_color] += opponent_stones - position.stones.count(mover_color.opponent())
        moves_played += 1
    return GameEnd(position, moves_played, prisoners[Stone.BLACK], prisoners[Stone.WHITE], tuple(history), self_fill_pass)


def sample_placement(
    board_size: int, move_visits: dict[Move, int], temperature: float, rng: Optional[random.Random]
) -> Move:
    """`PlayerStyle.temperature`'s move choice: passes when the most-visited move is a pass (or the
    search considered nothing), otherwise samples a stone placement from `move_visits` at
    `temperature`, with the pass's own visits left out."""
    if not move_visits:
        return Pass()
    best = max(move_visits.items(), key=lambda item: item[1])[0]
    if isinstance(best, Pass):
        return best
    placements = {move: visits for move, visits in move_visits.items() if not isinstance(move, Pass)}
    return sample_move(board_size, visit_count_policy(board_size, placements), temperature, rng or random.Random())
