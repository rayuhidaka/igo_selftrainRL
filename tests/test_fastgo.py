"""fastplay/fastgo.c must decide exactly what engine/ and mcts/ decide.

Run with: python -m unittest tests.test_fastgo
"""

from __future__ import annotations

import random
import unittest

import numpy as np

from engine.move import Pass
from engine.point import Point
from engine.position import Position
from engine.scoring import area_score, territory_ownership
from engine.stone import Stone
from fastplay.fastgo import BLACK, EMPTY, WHITE, Board, SearchPool, index_move, move_index
from mcts.mcts import Mcts, MctsConfig
from mcts.policy_value_net import Evaluation

_CODE = {Stone.EMPTY: EMPTY, Stone.BLACK: BLACK, Stone.WHITE: WHITE}


def _random_games(size: int, games: int, seed: int, pass_rate: float = 0.03):
    """Yields (position, board) pairs after every move of random games, played on both engines."""
    rng = random.Random(seed)
    for _ in range(games):
        position, board = Position.empty(size), Board.empty(size)
        moves = 0
        while position.pass_count < 2 and moves < 3 * size * size:
            yield position, board
            legal = position.legal_moves()
            plays = [move for move in legal if not isinstance(move, Pass)]
            move = Pass() if not plays or rng.random() < pass_rate else rng.choice(plays)
            position = position.play(move)
            board.play(move_index(move, size))
            moves += 1
        yield position, board


class BoardTest(unittest.TestCase):
    def assert_same(self, position: Position, board: Board) -> None:
        size = position.board_size
        self.assertEqual(board.to_position(), position)
        expected = np.zeros(size * size + 1, dtype=np.uint8)
        for move in position.legal_moves():
            expected[move_index(move, size)] = 1
        np.testing.assert_array_equal(board.legal_mask(), expected)

    def test_random_games_match_the_python_engine_move_by_move(self) -> None:
        checked = 0
        for size, games in ((9, 40), (5, 120), (3, 200)):
            for position, board in _random_games(size, games, seed=size):
                self.assert_same(position, board)
                checked += 1
        self.assertGreater(checked, 5000)

    def test_scores_and_ownership_match_at_game_ends(self) -> None:
        for size in (9, 5):
            ends = [p for p, _ in _random_games(size, 30, seed=100 + size)][::7]
            for position in ends:
                board = Board.from_position(position)
                area = area_score(position)
                self.assertEqual(board.area_score(), (area.black, area.white))
                owners = territory_ownership(position)
                expected = [_CODE[owners[Point.from_index(i, size)]] if owners[Point.from_index(i, size)] else EMPTY
                            for i in range(size * size)]
                self.assertEqual(board.ownership().tolist(), expected)

    def test_planes_match_encode_planes(self) -> None:
        from bootstrap.inference import encode_planes

        for i, (position, board) in enumerate(_random_games(9, 3, seed=7)):
            if i % 10 == 0:  # the generator's board changes in place, so check it as it's yielded
                np.testing.assert_array_equal(board.encode(), encode_planes(position).numpy())

    def test_ko_is_found_like_the_python_engine(self) -> None:
        # Black captures the white stone at (2,2) by playing (2,3): a ko, so white may not retake at once.
        position = Position.empty(5)
        board = Board.empty(5)
        for row, col in [(1, 2), (1, 3), (2, 1), (2, 4), (3, 2), (3, 3), (0, 0), (2, 2), (2, 3)]:
            move = index_move(row * 5 + col, 5)
            position = position.play(move)
            board.play(move_index(move, 5))
            self.assertEqual(board.to_position(), position)
        self.assertIsNotNone(position.ko_point)
        self.assertEqual(board.ko, position.ko_point.to_index(5))
        with self.assertRaises(ValueError):
            board.copy().play(board.ko)


class _PlanesNet:
    """A deterministic fake net that only looks at the encoded planes, so the batched search and
    mcts.Mcts see exactly the same numbers."""

    def __init__(self, size: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.size = size
        self.policy_weights = rng.normal(size=(3 * size * size, size * size + 1)).astype(np.float32)
        self.value_weights = rng.normal(size=3 * size * size).astype(np.float32) * np.float32(0.1)

    def batch(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat = planes.reshape(len(planes), -1).astype(np.float32)
        logits = flat @ self.policy_weights
        logits = logits - logits.max(axis=1, keepdims=True)
        policy = np.exp(logits)
        policy = (policy / policy.sum(axis=1, keepdims=True)).astype(np.float32)
        values = np.tanh(flat @ self.value_weights).astype(np.float32)
        return policy, values

    def evaluate(self, position: Position) -> Evaluation:
        policy, values = self.batch(Board.from_position(position).encode()[None])
        size = self.size
        return Evaluation(
            policy={move: float(policy[0, move_index(move, size)]) for move in position.legal_moves()},
            value=float(values[0]),
        )


def _run_pool(pool: SearchPool, net: _PlanesNet) -> None:
    while True:
        planes = pool.collect()
        if len(planes) == 0:
            return
        policy, values = net.batch(planes)
        pool.expand(policy, values)


class SearchPoolTest(unittest.TestCase):
    def test_visits_match_mcts_py_exactly(self) -> None:
        size, sims = 9, 120
        net = _PlanesNet(size)
        positions = [p for p, _ in _random_games(size, 2, seed=3)][::9][:12]
        pool = SearchPool(len(positions), size, sims)
        for game, position in enumerate(positions):
            pool.begin(game, Board.from_position(position), sims, c_puct=1.5, komi=7.5)
        _run_pool(pool, net)
        for game, position in enumerate(positions):
            expected = Mcts(net, MctsConfig(num_simulations=sims, exploration_constant=1.5, komi=7.5)).search(position)
            visits, root_value = pool.root_visits(game)
            for move, count in expected.move_visits.items():
                self.assertEqual(visits[move_index(move, size)], count, f"game {game} move {move}")
            self.assertEqual(int(visits.sum()), sum(expected.move_visits.values()))
            # The fake net's float32 matmul rounds differently in a batch of 12 than of 1.
            self.assertAlmostEqual(root_value, expected.root_value, places=6)

    def test_finished_games_are_scored_not_evaluated(self) -> None:
        # Black far ahead on 3x3 after white passed: passing out wins, and the search sees it.
        size = 3
        position = Position(
            size,
            (Stone.BLACK, Stone.BLACK, Stone.BLACK, Stone.BLACK, Stone.BLACK, Stone.EMPTY,
             Stone.WHITE, Stone.EMPTY, Stone.EMPTY),
            Stone.BLACK,
            pass_count=1,
        )
        net = _PlanesNet(size, seed=5)
        pool = SearchPool(1, size, 100)
        pool.begin(0, Board.from_position(position), 100, c_puct=1.5, komi=0.5)
        _run_pool(pool, net)
        expected = Mcts(net, MctsConfig(num_simulations=100, komi=0.5)).search(position)
        visits, _ = pool.root_visits(0)
        self.assertEqual(visits[size * size], expected.move_visits[Pass()])

    def test_root_noise_is_mixed_like_mcts_py(self) -> None:
        size, sims = 9, 80
        net = _PlanesNet(size, seed=9)
        position = Position.empty(size)
        alpha, epsilon = 0.1, 0.25
        board_moves = size * size
        noise = np.random.default_rng(42).dirichlet([alpha] * board_moves)
        pool = SearchPool(1, size, sims)
        pool.begin(0, Board.from_position(position), sims, c_puct=1.5, komi=7.5, noise=noise, noise_epsilon=epsilon)
        _run_pool(pool, net)
        mcts = Mcts(net, MctsConfig(num_simulations=sims, root_dirichlet_alpha=alpha, root_dirichlet_epsilon=epsilon),
                    rng=np.random.default_rng(42))
        expected = mcts.search(position)
        visits, _ = pool.root_visits(0)
        for move, count in expected.move_visits.items():
            self.assertEqual(visits[move_index(move, size)], count)


if __name__ == "__main__":
    unittest.main()
