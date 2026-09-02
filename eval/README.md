# eval/

Elo tracking across checkpoints/generations, used to decide which ones
become the app's shippable difficulty tiers — see
`../docs/ARCHITECTURE.md`'s "Difficulty-tier promotion" section for the
design and `../docs/ROADMAP.md`'s Phase 3.

- `elo.py` — Elo rating math (`update_ratings`, `should_promote`). Fully
  implemented and tested (`../tests/test_elo.py`) — no dependency on
  anything below.
- `match.py` — plays games between two checkpoints to produce `elo.py`'s
  inputs. **Not implemented yet**: needs a Python Go rules engine (see its
  docstring for the recommended approach) that doesn't exist yet.
- `promote.py` — orchestrates the two: play a candidate against the
  current tier, update ratings, decide whether to promote. Blocked on
  `match.py`.
