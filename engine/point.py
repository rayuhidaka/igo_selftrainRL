"""Port of igo-app/engine/Point.kt -- see engine/README.md."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    """A zero-indexed board coordinate. `row` and `col` each range over
    `0` to `board_size - 1` for whatever board they're used with.

    Row-major linear indexing (`row * board_size + col`) matches the
    tensor layout the policy/value net expects, per the project's
    coordinate convention (see igo-app/docs/MODEL_CONTRACT.md).
    """

    row: int
    col: int

    def to_index(self, board_size: int) -> int:
        """Returns the row-major linear index of this point on a `board_size` x `board_size` board."""
        return self.row * board_size + self.col

    @staticmethod
    def from_index(index: int, board_size: int) -> "Point":
        """Returns the `Point` at row-major linear `index` on a `board_size` x `board_size` board."""
        return Point(index // board_size, index % board_size)
