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
- [x] Produce Ray-zeroGo's first bootstrapped checkpoint — first real run:
      300 self-play games → 22,621 examples (~7.6 min), then training
      capped at 20 minutes (`configs/bootstrap_train_batch1.yaml`).
      Policy loss dropped from ~4.4 (the exact value a uniform-random
      82-way guess scores, `ln(82)`) to ~2.3 -- genuine learning, not
      noise. Two things to know about this specific checkpoint before
      reading too much into it:
      - It saw the same fixed 22,621-example batch roughly 780 times
        over (a lot of steps, not much data) -- almost certainly
        overfit/memorized rather than generalized. Proves the loop
        learns; isn't a meaningfully strong net. More self-play data
        (not more epochs on the same data) is the fix for a real
        attempt.
      - A `torch.save` call failed once near the very end of the run
        (`RuntimeError: File ... cannot be opened`) -- looked like a
        transient issue writing to `/mnt/c` (the Windows-mounted
        project path) under heavy repeated I/O (~1300 checkpoint saves
        over the run), not a logic bug; a checkpoint from shortly before
        the failure survived and loads fine.
      - **Hardened (2026-09-03):** `checkpoints/`, `runs/`, and
        `selfplay_games/` are now symlinks (from the repo, still on
        `/mnt/c`) to real directories on native WSL storage
        (`~/igo_training_runs/...`), same fix as the venv itself (see
        `docs/BUILD_NOTES.md`-equivalent notes in igo-app) — repo-relative
        config paths (`checkpoints/foo.pt`, `log_dir: runs`, etc.) work
        unchanged. `.gitignore` entries for these three switched from
        `dir/` to `dir` (no trailing slash), since a trailing-slash
        pattern only matches a real directory, not a symlink to one.
        `bootstrap/train.py`'s checkpoint saves also now retry (3
        attempts, 2s apart) on `RuntimeError` before giving up. Verified:
        full unit test suite (43 tests) and a real smoke-test training
        run both pass with the new symlinked paths.
- [x] **First real (non-overfit) bootstrap attempt (2026-09-03):** 10x
      batch1's scale. `configs/selfplay_generate_batch2.yaml` generated
      3,000 self-play games → 224,730 examples (~78 min) to
      `selfplay_games/batch2.npz`; `configs/bootstrap_train_batch2.yaml`
      trained 50 epochs over the full set (175,550 steps, ~12.6 min, no
      save failures — the hardening above held up) to
      `checkpoints/bootstrap_batch2.pt`. Total loss dropped from ~5.3
      (near the uniform-random baseline) to ~3.3 by epoch 25, then
      plateaued/noisy around 3.1-3.6 through epoch 50 rather than
      continuing to fall — unlike batch1's near-800-epoch run on a fixed
      small batch, this looks like the tiny 2-conv-layer net (see
      `bootstrap/model.py`) hitting its capacity ceiling on real data,
      not memorization. This is now a genuine (if weak, architecture-
      limited) bootstrapped checkpoint, not just a proof the loop learns.
      Not yet evaluated for actual playing strength (`eval/match.py`
      against `bootstrap_batch1.pt` or a random baseline) or exported to
      `.tflite` — both open for next time.

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
