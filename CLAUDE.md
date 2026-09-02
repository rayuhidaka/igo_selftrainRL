# CLAUDE.md — igo-training

## Project overview
Python training pipeline that produces the "Ray-zeroGo" opponent-tier
neural net checkpoints consumed by the `igo-app` Android app: bootstrapped
via imitation learning, then fine-tuned via self-play, saved at intervals
to produce progressively stronger difficulty checkpoints (see
`igo-app/docs/ROADMAP.md`'s difficulty-picker plan).

There is no expert-tier work here. `igo-app`'s live analysis overlay and
"Deep Analyze" already run on a real, working expert-tier checkpoint —
KataGo's own pretrained net, converted directly (`igo-app/tools/`), no
training involved. An earlier version of this doc planned to also
independently distill an expert-tier net here; that's dropped as
redundant now that direct conversion solved the same need more simply.

This is its own git repository, currently living as a (gitignored)
subfolder of `igo-app` on disk purely for convenience while both are
under active development side by side — it has no dependency on `igo-app`
and no remote yet. Written to be fully self-contained: avoid adding
dependencies on anything outside this folder.

## Tech stack
- Python, PyTorch (training)
- ONNX for the intermediate export format, then converted to TensorFlow
  Lite for the app
- All training — bootstrap, self-play, everything — runs locally on this
  machine. No rented/cloud GPU. Exact workflow (phased training, batch
  scheduling, how long runs are allowed to take) isn't decided yet — see
  docs/ROADMAP.md's Phase 2.

## Architecture
- `bootstrap/` — imitation-learning training against KataGo self-play
  games or public 9x9 game records; produces Ray-zeroGo's initial
  checkpoint, before any self-play fine-tuning
- `selfplay/` — self-play game generation + policy/value update loop,
  used to fine-tune Ray-zeroGo past its bootstrapped starting point
- `export/` — PyTorch → ONNX → TensorFlow Lite conversion; this is the
  contract boundary with `igo-app` (see docs/ARCHITECTURE.md for the
  exact tensor shapes `igo-app/inference/` expects — it's the same
  `docs/MODEL_CONTRACT.md` contract the expert-tier checkpoint already
  conforms to, so Ray-zeroGo checkpoints are drop-in interchangeable
  with it from `igo-app`'s side)
- `eval/` — Elo tracking across checkpoints/generations, used to pick
  which checkpoints become difficulty tiers

## Conventions
- Every checkpoint is versioned and logged with: architecture config,
  training data source, random seed, and Elo estimate at save time
- Hyperparameters live in config files, not hardcoded in scripts
- Keep this folder importable/runnable independent of the Android app —
  no assumptions about Kotlin, Gradle, or Android paths

## Development commands
- `python -m bootstrap.train --config configs/bootstrap_base.yaml`
- `python -m selfplay.run --config configs/selfplay_base.yaml`
- `python -m export.to_tflite --checkpoint <path> --out <path>.tflite`
- `python -m eval.elo --checkpoints <dir>`

## Things to avoid
- Don't change the exported tensor shapes without updating
  `docs/ARCHITECTURE.md` and flagging it — `igo-app/inference/` depends
  on that contract staying stable.
- Don't check large checkpoint files or training game logs into git —
  keep those out via `.gitignore` and store them separately.

## Where things stand
Check ROADMAP.md for the current phase and open tasks (this repo's own
phase numbering — not `igo-app`'s).
