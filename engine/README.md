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
