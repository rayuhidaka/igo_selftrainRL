# eval/

Elo tracking across checkpoints/generations, used to decide which ones
become the app's shippable difficulty tiers — see
`../docs/ARCHITECTURE.md`'s "Difficulty-tier promotion" section for the
design and `../docs/ROADMAP.md`'s Phase 3. Fully implemented end to end:

- `elo.py` — Elo rating math (`update_ratings`, `should_promote`). No
  dependency on anything below.
- `match.py` — plays games between two checkpoints to produce `elo.py`'s
  inputs, using `../mcts/` (search) and `../engine/` (rules) — both Python
  ports of `igo-app`'s Kotlin implementations.
- `promote.py` — orchestrates the two: play a candidate against the
  current tier, update ratings, decide whether to promote.

Tested at two levels: unit tests (`../tests/test_elo.py`,
`../tests/test_match.py`) and a real CLI run
(`../configs/eval_smoke_test.yaml`, see its header for the exact
commands) — `python -m eval.promote` actually playing games and printing
a real Elo verdict, not just a shape check.

What's still ahead isn't in this folder: real training (Phase 2) to
produce checkpoints actually worth running through this.
