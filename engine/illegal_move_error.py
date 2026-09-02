"""Port of igo-app/engine/IllegalMoveException.kt -- see engine/README.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.move import Move
    from engine.position import Position


class IllegalMoveError(ValueError):
    """Raised by `Position.play` when `move` is not legal for `position`'s player to move."""

    def __init__(self, move: "Move", position: "Position") -> None:
        self.move = move
        self.position = position
        super().__init__(
            f"Illegal move {move!r} for {position.to_play} on a "
            f"{position.board_size}x{position.board_size} board"
        )
