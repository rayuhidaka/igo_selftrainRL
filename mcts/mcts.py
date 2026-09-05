"""Port of igo-app/mcts/Mcts.kt -- see mcts/README.md.

Kept as close a line-for-line translation as idiomatic Python allows, same
spirit as engine/'s port. If you change the search here, change it on the
Kotlin side too (or vice versa).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from engine.move import Move, Pass
from engine.position import Position
from engine.scoring import area_score
from mcts.policy_value_net import PolicyValueNet


@dataclass(frozen=True)
class MctsConfig:
    """Tuning knobs for `Mcts.search`."""

    num_simulations: int = 200
    """How many simulations `Mcts.search` runs per call."""

    exploration_constant: float = 1.5
    """The PUCT exploration constant (often called `cPuct`); higher favors exploring low-visit moves."""

    komi: float = 7.5
    """Komi added to white's `AreaScore` when a search reaches a finished game."""

    root_dirichlet_alpha: float = 0.1
    """Concentration parameter for the Dirichlet noise mixed into the root's priors -- only has
    an effect when `root_dirichlet_epsilon` is nonzero. AlphaZero's own value for 19x19 Go is
    0.03, but that's tuned for 19x19's ~250+ opening legal moves, not 9x9's ~82 -- the standard
    heuristic is `alpha ~= 10 / average legal moves` (why chess uses 0.3, shogi 0.15, 19x19 Go
    0.03), which lands around 0.1-0.12 for 9x9. Naively reusing 0.03 here (2026-09-04, see
    docs/SELF_PLAY_STABILITY.md) produced noise so peaked it dumped nearly all its mass onto
    one random board move and left the rest -- Pass included, even though Pass itself is
    excluded from the noise draw (see `root_dirichlet_epsilon`'s docstring) -- with near-zero
    *effective* priors after mixing, letting Pass win root selection by its competitors being
    suppressed rather than by being boosted itself. `0.1` measured meaningfully better on this
    board (5% vs. 10% short/collapsed self-play games at the same `root_dirichlet_epsilon`,
    smoke-tested at n=20 -- see the doc above before assuming this is the final word).
    """

    root_dirichlet_epsilon: float = 0.0
    """Weight given to root Dirichlet noise vs. the net's own prior, `0.0` (the default) meaning
    no noise -- i.e. `Mcts.search`'s existing, unchanged behavior. Self-play data generation
    should pass a nonzero value (AlphaZero's own default is `0.25`); this exists to fix a real
    diagnosed bug (see docs/SELF_PLAY_STABILITY.md): with zero root noise and a low simulation
    budget, a checkpoint's own slightly-elevated Pass prior gets over-visited by search with no
    mechanism to correct it, and training on the resulting visit counts pushes the *next*
    checkpoint's prior even higher -- a multiplicative feedback loop across self-play
    generations. Left at `0.0` for `eval/match.py` and the Android app's live analysis search,
    where the goal is the engine's actual best move, not exploration -- so this is
    deliberately NOT mirrored to `igo-app/mcts/src/main/kotlin/.../Mcts.kt`, unlike this file's
    other config knobs.
    """


@dataclass(frozen=True)
class SearchResult:
    """The result of `Mcts.search`: a visit count for every move considered at the root, plus
    the root's own visit-weighted value.
    """

    move_visits: dict[Move, int]
    """Visit count per move considered at the root -- the search's move-strength signal, not raw policy priors."""

    root_value: float
    """The visit-weighted average value of the searched root position, from the perspective of
    its player to move: `+1` a certain win, `-1` a certain loss, `0` even. This is the
    search-refined counterpart to a raw `PolicyValueNet.evaluate` value, suitable for a
    win-rate readout.
    """


class _Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children")

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[Move, "_Node"] = {}

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def record_visit(self, value: float) -> None:
        self.visit_count += 1
        self.value_sum += value


class Mcts:
    """A PUCT-style ("Predictor + UCT") Monte Carlo tree search driver over `engine/`'s rules
    and a `PolicyValueNet`. Only depends on `PolicyValueNet`, never on a specific net
    architecture -- the same driver runs any checkpoint that implements it.

    A game that reaches two consecutive passes (see `Position.pass_count`) is scored with
    `engine.scoring.area_score` and `MctsConfig.komi` instead of being evaluated by the net.
    """

    def __init__(
        self, net: PolicyValueNet, config: MctsConfig = MctsConfig(), rng: np.random.Generator | None = None
    ) -> None:
        self._net = net
        self._config = config
        self._rng = rng if rng is not None else np.random.default_rng()

    def search(self, root: Position) -> SearchResult:
        """Runs `config.num_simulations` simulations from `root` and returns the resulting
        `SearchResult`. `move_visits` is empty if `root` is already a finished game (two
        consecutive passes) -- `root_value` is still meaningful in that case (the
        deterministic outcome of `area_score`).
        """
        root_node = _Node(prior=1.0)
        for _ in range(self._config.num_simulations):
            self._simulate(root_node, root, is_root=True)
        return SearchResult(
            move_visits={move: child.visit_count for move, child in root_node.children.items()},
            root_value=root_node.mean_value,
        )

    def select_move(self, root: Position) -> Move:
        """Returns the most-visited move after `search`ing from `root`, or `Pass()` if `root` is already finished."""
        result = self.search(root)
        if not result.move_visits:
            return Pass()
        return max(result.move_visits.items(), key=lambda item: item[1])[0]

    def _simulate(self, node: _Node, position: Position, is_root: bool = False) -> float:
        """Runs one simulation from `node`/`position` and returns its value from the
        perspective of `position.to_play` -- i.e. how good this position is for whoever is
        about to move here. Also records that value into `node`'s own visit statistics.
        """
        if self._is_terminal(position):
            value = self._terminal_value(position)
        elif not node.children:
            value = self._expand(node, position, is_root=is_root)
        else:
            move, child = self._select_child(node)
            value = -self._simulate(child, position.play(move))
        node.record_visit(value)
        return value

    def _expand(self, node: _Node, position: Position, is_root: bool = False) -> float:
        """Evaluates `position` with `net`, creates one child per legal move, and returns the
        net's value. `is_root` mixes `MctsConfig.root_dirichlet_epsilon` noise into the priors
        when set (see that field's docstring) -- this only ever fires on a node's first
        expansion, so it naturally applies exactly once per `search` call, to the root only.
        """
        evaluation = self._net.evaluate(position)
        priors = evaluation.policy
        if is_root and self._config.root_dirichlet_epsilon > 0:
            priors = self._add_root_noise(priors)
        for move, prior in priors.items():
            node.children[move] = _Node(prior)
        return evaluation.value

    def _add_root_noise(self, priors: dict[Move, float]) -> dict[Move, float]:
        """Mixes Dirichlet noise into `priors`, weighted by `MctsConfig.root_dirichlet_epsilon`
        -- over the board-play moves only, deliberately excluding `Pass` (2026-09-04, see
        docs/SELF_PLAY_STABILITY.md and memory `self_play_root_noise_pass_bias`). A batch
        test showed noise-including-Pass made short/collapsed games *more* common (35-40%
        vs. a 1% no-noise baseline on the same checkpoint), not less: with a peaked
        `alpha`, a noise draw usually dumps nearly all its mass on one random action, and
        roughly 1-in-`len(priors)` times that's Pass -- sending real search budget down the
        post-pass branch, a state this net's value head is poorly calibrated on precisely
        because passing was rare in its own training data. That's the exact collapse this
        noise was meant to fix, just relocated. Leaving Pass's own net-derived prior alone
        keeps noise's intended benefit (varied board-play exploration) without reopening
        that hole.
        """
        board_moves = [move for move in priors if not isinstance(move, Pass)]
        noise = self._rng.dirichlet([self._config.root_dirichlet_alpha] * len(board_moves))
        epsilon = self._config.root_dirichlet_epsilon
        result = dict(priors)
        for move, sample in zip(board_moves, noise):
            result[move] = (1 - epsilon) * priors[move] + epsilon * sample
        return result

    def _select_child(self, node: _Node) -> tuple[Move, _Node]:
        """Picks `node`'s child maximizing the PUCT score, from `node`'s own player-to-move's perspective."""
        parent_visits = node.visit_count
        best_move, best_child = max(
            node.children.items(), key=lambda item: self._puct_score(item[1], parent_visits)
        )
        return best_move, best_child

    def _puct_score(self, child: _Node, parent_visits: int) -> float:
        exploitation = -child.mean_value
        exploration = (
            self._config.exploration_constant * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
        )
        return exploitation + exploration

    def _is_terminal(self, position: Position) -> bool:
        return position.pass_count >= 2

    def _terminal_value(self, position: Position) -> float:
        winner = area_score(position).winner(self._config.komi)
        if winner == position.to_play:
            return 1.0
        if winner is None:
            return 0.0
        return -1.0
