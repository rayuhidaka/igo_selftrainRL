# Roadmap — igo-training

This repo's own phase numbering — distinct from `igo-app`'s, which only
tracks the handoff points that matter to the app (see its Phase 2-4).

**Training runs locally, on the user's own machine — not a rented/cloud
GPU.** How to actually work within that (phased training runs, batch
scheduling, session length) is still to be decided — revisit when Phase
2 starts in earnest.

## Phase 1 (current) — scaffolding
- [x] Repo/folder structure in place (`bootstrap/`, `selfplay/`,
      `export/`, `eval/`, `configs/`)
- [x] Export contract confirmed against `igo-app/docs/MODEL_CONTRACT.md`
      (see `docs/ARCHITECTURE.md`)
- [x] Placeholder/tiny checkpoint exported end-to-end through the
      pipeline just to validate the `igo-app` loading path works — an
      untrained net (`bootstrap/train.py` → `export/to_tflite.py`),
      same idea as `igo-app/tools/build_placeholder_net.py` but produced
      via this repo's actual PyTorch → ONNX → TFLite path. Verified: the
      resulting `.tflite`'s tensors are exactly `board_planes [1,9,9,3]`
      in, `policy [1,82]` / `value [1,1]` out, correctly named — see
      `docs/ARCHITECTURE.md`'s `export/` section for what it took to get
      there. Not yet verified loading inside `igo-app`/on-device
      specifically (only Python-side, same as this pipeline's own
      checks) — should behave identically to the expert-tier checkpoint
      there, since both conform to the same contract, but hasn't been
      separately confirmed.

## Phase 2 — Ray-zeroGo bootstrap
- [ ] Collect KataGo self-play games or public 9x9 game records
- [ ] Imitation-learning training loop (`bootstrap/`) — monitoring
      (TensorBoard, `bootstrap/monitoring.py`) and the config fields for
      checkpoint cadence are already wired up ahead of this, see
      `docs/ARCHITECTURE.md`'s "Monitoring" section; the loop itself
      (actually training on data) is still a placeholder
- [ ] Produce Ray-zeroGo's first bootstrapped checkpoint

## Phase 3 — Self-play fine-tuning
- [ ] Self-play generation loop (`selfplay/`)
- [ ] Policy/value update loop, run locally (see the local-training note
      at the top of this file)
- [ ] Save checkpoints at intervals — these become candidate difficulty
      tiers, gated on Elo (`eval/`), not shipped automatically
- [x] Elo rating math (`eval/elo.py`) and the promotion gate
      (`should_promote`) — fully implemented and tested
      (`tests/test_elo.py`). See `docs/ARCHITECTURE.md`'s "Difficulty-tier
      promotion" section for the design.
- [x] `engine/` — Python port of `igo-app/engine/`'s Go rules (rules
      enforcement: legality, capture/ko, area scoring). Proven against a
      full port of `igo-app/engine/`'s own test suite
      (`tests/test_position.py`, `tests/test_scoring.py`, 20 tests, all
      passing) — not just eyeballed.
- [ ] Actually playing games between checkpoints (`eval/match.py`) —
      `engine/` now covers rules, but there's no *search* yet (something
      that plays better than uniformly-random moves). See
      `eval/match.py`'s docstring. Without this, `eval/promote.py`
      can't run.

## Phase 4 — Handoff to igo-app
- [ ] Export selected checkpoints to `.tflite` (`export/`)
- [ ] Hand off checkpoints + Elo metadata to the app for the difficulty
      picker and optional Elo-over-generations chart
