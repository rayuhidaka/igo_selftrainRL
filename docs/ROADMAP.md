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
- [x] Collect KataGo self-play games (`selfplay/generate.py`) — samples
      moves directly from igo-app's real KataGo checkpoint's own policy
      output (no tree search needed: it's already strong, this is
      distillation, not reinforcement-learning self-play), ~20ms/move on
      this machine. Public 9x9 game records as an alternative/additional
      source not pursued yet.
- [x] Imitation-learning training loop (`bootstrap/train.py`) — real
      gradient descent now (standard AlphaZero-style loss: policy
      cross-entropy + value MSE against `selfplay/generate.py`'s data),
      not the Phase 1 placeholder. TensorBoard logging
      (`bootstrap/monitoring.py`) and checkpoint-interval saving wired
      in. Smoke-tested end to end (3 games → 214 examples → a real
      training run, loss decreasing) before the first real batch.
- [ ] Produce Ray-zeroGo's first bootstrapped checkpoint — infrastructure
      is proven, first real batch not run yet

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
- [x] `mcts/` — Python port of `igo-app/mcts/`'s PUCT search, and
      `bootstrap/inference.py`'s `RayZeroPolicyValueNet` (mirrors
      `igo-app/inference/TfLitePolicyValueNet.kt`'s encode/decode).
      Proven against ports of both their test suites (`tests/test_mcts.py`,
      `tests/test_inference.py`).
- [x] Actually playing games between checkpoints (`eval/match.py`) — done,
      using `mcts/` + `engine/`. Verified two ways: a symmetry test
      (`tests/test_match.py`, a checkpoint played against itself averages
      to ~50/50) and a real CLI run (`configs/eval_smoke_test.yaml`, two
      untrained 9x9 checkpoints, 4 games via `python -m eval.promote`,
      produced a real Elo verdict in ~23s). `eval/`'s whole pipeline
      (`elo.py` → `match.py` → `promote.py`) is now implemented end to
      end — what's missing is real checkpoints worth evaluating, i.e.
      Phase 2 actually happening.

## Phase 4 — Handoff to igo-app
- [ ] Export selected checkpoints to `.tflite` (`export/`)
- [ ] Hand off checkpoints + Elo metadata to the app for the difficulty
      picker and optional Elo-over-generations chart
