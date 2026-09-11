# engine/

A Python port of `igo-app/engine/`'s Go rules (Kotlin), for `eval/match.py`
to play games between checkpoints with — see
`../docs/ARCHITECTURE.md`'s "Difficulty-tier promotion" section for why
this exists (avoid a second, independently-written rules implementation
silently disagreeing with the first) and `../tests/test_position.py` /
`../tests/test_scoring.py` for a full port of `igo-app/engine`'s own test
suite, proving the two agree.

| Kotlin (`igo-app/engine/`)   | Python (here)              |
|-------------------------------|-----------------------------|
| `Stone.kt`                    | `stone.py`                  |
| `Point.kt`                    | `point.py`                  |
| `Move.kt`                     | `move.py`                   |
| `IllegalMoveException.kt`     | `illegal_move_error.py`     |
| `Position.kt`                 | `position.py`               |
| `Scoring.kt`                  | `scoring.py`                |

**If you change the rules, change them on both sides.** There's no
automated cross-check between the two codebases (different languages,
different test runners) — the port is only as good as the last time
someone kept them in sync by hand.

Board size is a parameter here too, not hardcoded to 9, matching
`igo-app/engine/`'s own design (see its CLAUDE.md) — but nothing in
`igo-training` needs anything other than 9x9 yet.

## The rules algorithm itself

`position.py`'s `Position` is immutable, same as the Kotlin side — `play()`
returns a new `Position`, never mutates. Captures and suicide both reduce
to the same two primitives: a flood-fill `_group_at` (connected
same-colored stones from one point) and `_liberties` (every empty point
orthogonally adjacent to a group). Playing a stone: any *opponent* group
adjacent to the new stone that ends up with zero liberties is captured
(removed); if nothing was captured and the newly-played stone's *own*
group has zero liberties, the move is suicide (illegal). Ko is the simple/
positional rule — a single-stone capture marks that point unplayable for
one turn, no move-history-based superko.

`scoring.py`'s `area_score` is Tromp-Taylor area scoring, computed by one
flood-fill pass: every stone counts for its own color, every empty region
counts for whichever color exclusively borders it (dame — bordered by
both — counts for neither). **It assumes dead stones have already been
captured** — there's no life/death resolution here, matching
`igo-app/engine/`'s own documented limitation exactly. `mcts.py`'s `Mcts`
only calls this once a position actually reaches two passes, so this
matters for self-play/eval game quality (a game that ends with an
uncaptured dead group on the board scores it for the wrong side), not for
correctness of the function itself.
