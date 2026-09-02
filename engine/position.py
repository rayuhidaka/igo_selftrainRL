"""Port of igo-app/engine/Position.kt -- see engine/README.md.

Kept as close a line-for-line translation as idiomatic Python allows,
specifically so the two implementations are easy to diff against each
other by eye. If you change the rules here, change them on the Kotlin
side too (or vice versa) -- see igo-app/CLAUDE.md's own warning about
this for the Kotlin side.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

from engine.illegal_move_error import IllegalMoveError
from engine.move import Move, Pass, Play
from engine.point import Point
from engine.stone import Stone


@dataclass(frozen=True)
class Position:
    """An immutable snapshot of a Go position on a `board_size` x `board_size` board.

    `stones` is row-major and zero-indexed (see `Point`): the stone at
    `(row, col)` sits at index `row * board_size + col`. A `Position` is
    never mutated in place -- `play` returns a new `Position` reflecting
    the result of a move, leaving the receiver untouched.

    Ko is enforced with the simple ("positional") ko rule: immediately
    recapturing at the point of a just-completed single-stone capture is
    forbidden until another move is played. This is not full positional
    superko.
    """

    board_size: int
    stones: tuple[Stone, ...]
    to_play: Stone
    ko_point: Optional[Point] = None
    pass_count: int = 0

    def __post_init__(self) -> None:
        # Normalize any iterable (e.g. a list, as Kotlin's MutableList-based
        # test helpers use) to a tuple, so Position stays hashable.
        object.__setattr__(self, "stones", tuple(self.stones))

        if self.board_size <= 0:
            raise ValueError(f"board_size must be positive, was {self.board_size}")
        if len(self.stones) != self.board_size * self.board_size:
            raise ValueError(
                f"stones must have {self.board_size * self.board_size} entries for a "
                f"{self.board_size}x{self.board_size} board, had {len(self.stones)}"
            )
        if self.to_play not in (Stone.BLACK, Stone.WHITE):
            raise ValueError(f"to_play must be BLACK or WHITE, was {self.to_play}")

    def stone_at(self, point: Point) -> Stone:
        """Returns the stone at `point`.

        Raises:
            ValueError: if `point` is outside the board.
        """
        if not self.is_on_board(point):
            raise ValueError(f"{point} is outside the {self.board_size}x{self.board_size} board")
        return self.stones[point.to_index(self.board_size)]

    def is_on_board(self, point: Point) -> bool:
        """Returns True if `point` lies within the bounds of this board."""
        return 0 <= point.row < self.board_size and 0 <= point.col < self.board_size

    def neighbors(self, point: Point) -> list[Point]:
        """Returns the orthogonal neighbors of `point` that lie on the board."""
        result: list[Point] = []
        if point.row > 0:
            result.append(Point(point.row - 1, point.col))
        if point.row < self.board_size - 1:
            result.append(Point(point.row + 1, point.col))
        if point.col > 0:
            result.append(Point(point.row, point.col - 1))
        if point.col < self.board_size - 1:
            result.append(Point(point.row, point.col + 1))
        return result

    def is_legal(self, move: Move) -> bool:
        """Returns True if `move` can legally be played by `to_play` in this position.

        `Pass` is always legal. A `Play` is illegal if its point is
        off-board, occupied, the current `ko_point`, or a suicide (the
        played stone's group would end up with zero liberties without
        capturing any opponent stones).
        """
        if isinstance(move, Pass):
            return True
        return self._is_legal_play(move.point)

    def legal_moves(self) -> list[Move]:
        """Returns every legal move for `to_play` in this position, including `Pass()`."""
        plays = [
            Play(point)
            for point in (Point.from_index(i, self.board_size) for i in range(self.board_size * self.board_size))
            if self._is_legal_play(point)
        ]
        return [*plays, Pass()]

    def play(self, move: Move) -> "Position":
        """Returns the position resulting from `to_play` playing `move`.

        Raises:
            IllegalMoveError: if `move` is not legal in this position.
        """
        if not self.is_legal(move):
            raise IllegalMoveError(move, self)
        if isinstance(move, Pass):
            return dataclasses.replace(
                self, to_play=self.to_play.opponent(), ko_point=None, pass_count=self.pass_count + 1
            )
        return self._apply_play(move.point)

    def _is_legal_play(self, point: Point) -> bool:
        if not self.is_on_board(point):
            return False
        if self.stone_at(point) != Stone.EMPTY:
            return False
        if point == self.ko_point:
            return False
        return not self._would_be_suicide(point)

    def _would_be_suicide(self, point: Point) -> bool:
        trial = list(self.stones)
        trial[point.to_index(self.board_size)] = self.to_play

        opponent = self.to_play.opponent()
        for neighbor in self.neighbors(point):
            if trial[neighbor.to_index(self.board_size)] == opponent:
                group = self._group_at(neighbor, trial)
                if not self._liberties(group, trial):
                    return False  # this play captures at least one opponent group
        own_group = self._group_at(point, trial)
        return not self._liberties(own_group, trial)

    def _apply_play(self, point: Point) -> "Position":
        mutable = list(self.stones)
        mutable[point.to_index(self.board_size)] = self.to_play

        opponent = self.to_play.opponent()
        captured: set[Point] = set()
        for neighbor in self.neighbors(point):
            if mutable[neighbor.to_index(self.board_size)] == opponent:
                group = self._group_at(neighbor, mutable)
                if not self._liberties(group, mutable):
                    captured |= group

        for captured_point in captured:
            mutable[captured_point.to_index(self.board_size)] = Stone.EMPTY

        new_ko_point: Optional[Point] = None
        if len(captured) == 1:
            own_group = self._group_at(point, mutable)
            (single_captured,) = captured
            if len(own_group) == 1 and self._liberties(own_group, mutable) == {single_captured}:
                new_ko_point = single_captured

        return Position(
            board_size=self.board_size,
            stones=tuple(mutable),
            to_play=opponent,
            ko_point=new_ko_point,
            pass_count=0,
        )

    def _group_at(self, start: Point, board: list[Stone]) -> set[Point]:
        color = board[start.to_index(self.board_size)]
        visited = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in self.neighbors(current):
                if neighbor not in visited and board[neighbor.to_index(self.board_size)] == color:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return visited

    def _liberties(self, group: set[Point], board: list[Stone]) -> set[Point]:
        libs: set[Point] = set()
        for stone in group:
            for neighbor in self.neighbors(stone):
                if board[neighbor.to_index(self.board_size)] == Stone.EMPTY:
                    libs.add(neighbor)
        return libs

    @staticmethod
    def empty(board_size: int) -> "Position":
        """Returns an empty `board_size` x `board_size` position with `Stone.BLACK` to play."""
        return Position(
            board_size=board_size,
            stones=tuple(Stone.EMPTY for _ in range(board_size * board_size)),
            to_play=Stone.BLACK,
        )
