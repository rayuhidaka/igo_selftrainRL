"""Port of igo-app/mcts/Mcts.kt -- see mcts/README.md.

Kept as close a line-for-line translation as idiomatic Python allows, same
spirit as engine/'s port. If you change the search here, change it on the
Kotlin side too (or vice versa).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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

    def __init__(self, net: PolicyValueNet, config: MctsConfig = MctsConfig()) -> None:
        self._net = net
        self._config = config

    def search(self, root: Position) -> SearchResult:
        """Runs `config.num_simulations` simulations from `root` and returns the resulting
        `SearchResult`. `move_visits` is empty if `root` is already a finished game (two
        consecutive passes) -- `root_value` is still meaningful in that case (the
        deterministic outcome of `area_score`).
        """
        root_node = _Node(prior=1.0)
        for _ in range(self._config.num_simulations):
            self._simulate(root_node, root)
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

    def _simulate(self, node: _Node, position: Position) -> float:
        """Runs one simulation from `node`/`position` and returns its value from the
        perspective of `position.to_play` -- i.e. how good this position is for whoever is
        about to move here. Also records that value into `node`'s own visit statistics.
        """
        if self._is_terminal(position):
            value = self._terminal_value(position)
        elif not node.children:
            value = self._expand(node, position)
        else:
            move, child = self._select_child(node)
            value = -self._simulate(child, position.play(move))
        node.record_visit(value)
        return value

    def _expand(self, node: _Node, position: Position) -> float:
        """Evaluates `position` with `net`, creates one child per legal move, and returns the net's value."""
        evaluation = self._net.evaluate(position)
        for move, prior in evaluation.policy.items():
            node.children[move] = _Node(prior)
        return evaluation.value

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
