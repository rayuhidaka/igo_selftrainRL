"""ctypes wrapper around fastgo.c (see fastplay/README.md): the board and the batched search.

The shared library is compiled with the system C compiler the first time this module is
imported (and again whenever fastgo.c is newer), into fastplay/build/ (gitignored).
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

from engine.move import Move, Pass, Play
from engine.point import Point
from engine.position import Position
from engine.stone import Stone

_HERE = Path(__file__).resolve().parent
_SOURCE = _HERE / "fastgo.c"
_LIBRARY = _HERE / "build" / "libfastgo.so"

MAX_POINTS = 361
EMPTY, BLACK, WHITE = 0, 1, 2
NO_KO = -1


def _build() -> None:
    _LIBRARY.parent.mkdir(exist_ok=True)
    if _LIBRARY.exists() and _LIBRARY.stat().st_mtime >= _SOURCE.stat().st_mtime:
        return
    # Built under another name and renamed into place, so a process that has the old library
    # loaded keeps running on it.
    temporary = _LIBRARY.with_name(f"{_LIBRARY.name}.{os.getpid()}.tmp")
    subprocess.run(
        ["cc", "-O3", "-march=native", "-shared", "-fPIC", "-o", str(temporary), str(_SOURCE), "-lm"],
        check=True,
    )
    os.replace(temporary, _LIBRARY)


class Board(ctypes.Structure):
    """fastgo.c's fg_board. Use `from_position` / `to_position` to convert from engine/."""

    _fields_ = [
        ("size", ctypes.c_int32),
        ("to_play", ctypes.c_int32),
        ("ko", ctypes.c_int32),
        ("pass_count", ctypes.c_int32),
        ("stones", ctypes.c_int8 * MAX_POINTS),
    ]

    @staticmethod
    def empty(size: int) -> "Board":
        board = Board()
        board.size, board.to_play, board.ko, board.pass_count = size, BLACK, NO_KO, 0
        return board

    @staticmethod
    def from_position(position: Position) -> "Board":
        board = Board.empty(position.board_size)
        for index, stone in enumerate(position.stones):
            board.stones[index] = _STONE_CODE[stone]
        board.to_play = _STONE_CODE[position.to_play]
        board.ko = NO_KO if position.ko_point is None else position.ko_point.to_index(position.board_size)
        board.pass_count = position.pass_count
        return board

    def to_position(self) -> Position:
        points = self.size * self.size
        return Position(
            board_size=self.size,
            stones=tuple(_CODE_STONE[self.stones[i]] for i in range(points)),
            to_play=_CODE_STONE[self.to_play],
            ko_point=None if self.ko == NO_KO else Point.from_index(self.ko, self.size),
            pass_count=self.pass_count,
        )

    def copy(self) -> "Board":
        board = Board()
        ctypes.pointer(board)[0] = self
        return board

    def legal_mask(self) -> np.ndarray:
        """1 per legal move: `size * size` points, then the pass."""
        out = np.zeros(self.size * self.size + 1, dtype=np.uint8)
        _lib.fg_legal_mask(ctypes.byref(self), out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        return out

    def play(self, move: int) -> int:
        """Plays `move` (a point index or `size * size` for a pass) in place; returns the stones
        captured. Raises ValueError if it's illegal."""
        captured = _lib.fg_play(ctypes.byref(self), move)
        if captured < 0:
            raise ValueError(f"illegal move {move}")
        return captured

    def ownership(self) -> np.ndarray:
        """scoring.py's territory_ownership per point: 1 black, 2 white, 0 neither."""
        out = np.zeros(self.size * self.size, dtype=np.int8)
        _lib.fg_ownership(ctypes.byref(self), out.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)))
        return out

    def area_score(self) -> tuple[int, int]:
        black, white = ctypes.c_int(), ctypes.c_int()
        _lib.fg_area_score(ctypes.byref(self), ctypes.byref(black), ctypes.byref(white))
        return black.value, white.value

    def cleanup_mask(self) -> np.ndarray:
        """1 per legal move dead_stone_estimator.py's is_cleanup_move allows (never the pass)."""
        out = np.zeros(self.size * self.size + 1, dtype=np.uint8)
        _lib.fg_cleanup_mask(ctypes.byref(self), out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        return out

    def is_settled(self) -> bool:
        """dead_stone_estimator.py's is_settled: every empty point counts for exactly one color."""
        return bool(_lib.fg_is_settled(ctypes.byref(self)))

    def encode(self) -> np.ndarray:
        """encode_planes as a [3, size, size] float32 array."""
        out = np.zeros((3, self.size, self.size), dtype=np.float32)
        _lib.fg_encode(ctypes.byref(self), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return out


def move_index(move: Move, size: int) -> int:
    return size * size if isinstance(move, Pass) else move.point.to_index(size)


def index_move(index: int, size: int) -> Move:
    return Pass() if index == size * size else Play(Point.from_index(index, size))


class SearchPool:
    """`num_games` PUCT searches advanced together (fastgo.c's batched search). Each searches
    like mcts/mcts.py's Mcts, one simulation at a time; the net sees one leaf per game per step.
    """

    def __init__(self, num_games: int, size: int, max_simulations: int) -> None:
        self.num_games = num_games
        self.size = size
        points = size * size
        capacity = max_simulations * (points + 1) + 1
        self._pool = _lib.fg_pool_new(num_games, size, capacity)
        self._planes = np.zeros((num_games, 3, size, size), dtype=np.float32)
        self._visits = np.zeros(points + 1, dtype=np.int32)

    def close(self) -> None:
        if self._pool:
            _lib.fg_pool_free(self._pool)
            self._pool = None

    def __del__(self) -> None:
        self.close()

    def begin(
        self,
        game: int,
        root: Board,
        simulations: int,
        c_puct: float,
        komi: float,
        noise: np.ndarray | None = None,
        noise_epsilon: float = 0.0,
        fpu_reduction: float | None = None,
    ) -> None:
        """Starts `game`'s search from `root`. `noise` (one value per legal board move, in point
        order) is mixed into the root's priors with weight `noise_epsilon`. `fpu_reduction` is
        `MctsConfig.fpu_reduction` (`None`: an unvisited child counts as 0.0)."""
        fpu = -1.0 if fpu_reduction is None else fpu_reduction
        if noise is None:
            _lib.fg_search_begin(self._pool, game, ctypes.byref(root), simulations, c_puct, komi, None, 0, 0.0, fpu)
        else:
            noise = np.ascontiguousarray(noise, dtype=np.float64)
            _lib.fg_search_begin(
                self._pool, game, ctypes.byref(root), simulations, c_puct, komi,
                noise.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(noise), noise_epsilon, fpu,
            )

    def collect(self) -> np.ndarray:
        """The planes of every leaf waiting for the net, `[n, 3, size, size]` (a view, valid until
        the next call)."""
        n = _lib.fg_collect(self._pool, self._planes.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return self._planes[:n]

    def expand(self, policy: np.ndarray, values: np.ndarray) -> None:
        """The net's raw policy `[n, size*size+1]` and values `[n]` for the last `collect`."""
        policy = np.ascontiguousarray(policy, dtype=np.float32)
        values = np.ascontiguousarray(values, dtype=np.float32)
        if _lib.fg_expand(
            self._pool,
            policy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ) != 0:
            raise RuntimeError("search tree ran out of nodes")

    def root_priors(self, game: int) -> np.ndarray:
        """`game`'s root priors per move (`size*size+1`, root noise mixed in, 0 if illegal)."""
        out = np.zeros(self.size * self.size + 1, dtype=np.float64)
        _lib.fg_root_priors(self._pool, game, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return out

    def done(self, game: int) -> bool:
        return bool(_lib.fg_search_done(self._pool, game))

    def root_visits(self, game: int) -> tuple[np.ndarray, float]:
        """`game`'s root visit counts per move (`size*size+1`) and its root value."""
        value = _lib.fg_root_visits(self._pool, game, self._visits.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return self._visits.copy(), value


_STONE_CODE = {Stone.EMPTY: EMPTY, Stone.BLACK: BLACK, Stone.WHITE: WHITE}
_CODE_STONE = {code: stone for stone, code in _STONE_CODE.items()}

_build()
_lib = ctypes.CDLL(str(_LIBRARY))
_board_p = ctypes.POINTER(Board)
_lib.fg_is_legal.argtypes = [_board_p, ctypes.c_int]
_lib.fg_legal_mask.argtypes = [_board_p, ctypes.POINTER(ctypes.c_uint8)]
_lib.fg_play.argtypes = [_board_p, ctypes.c_int]
_lib.fg_ownership.argtypes = [_board_p, ctypes.POINTER(ctypes.c_int8)]
_lib.fg_area_score.argtypes = [_board_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
_lib.fg_winner.argtypes = [_board_p, ctypes.c_double]
_lib.fg_encode.argtypes = [_board_p, ctypes.POINTER(ctypes.c_float)]
_lib.fg_cleanup_mask.argtypes = [_board_p, ctypes.POINTER(ctypes.c_uint8)]
_lib.fg_is_settled.argtypes = [_board_p]
_lib.fg_pool_new.restype = ctypes.c_void_p
_lib.fg_pool_new.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.fg_pool_free.argtypes = [ctypes.c_void_p]
_lib.fg_search_begin.argtypes = [
    ctypes.c_void_p, ctypes.c_int, _board_p, ctypes.c_int, ctypes.c_double, ctypes.c_double,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double, ctypes.c_double,
]
_lib.fg_collect.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
_lib.fg_expand.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
_lib.fg_root_visits.restype = ctypes.c_double
_lib.fg_root_visits.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32)]
_lib.fg_root_priors.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
_lib.fg_search_done.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.fg_pending_game.argtypes = [ctypes.c_void_p, ctypes.c_int]
