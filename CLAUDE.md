# CLAUDE.md — igo-training

## Project overview
Python training pipeline that produces the "Ray-zeroGo" opponent-tier
neural net checkpoints consumed by the `igo-app` Android app: bootstrapped
via imitation learning, then fine-tuned via self-play, saved at intervals
to produce progressively stronger difficulty checkpoints (see
`igo-app/docs/ROADMAP.md`'s difficulty-picker plan).

These checkpoints are the only nets `igo-app` ships: its "AI Hint"
overlay and every "Play vs AI" difficulty tier run on them. (`igo-app`
once also bundled a converted KataGo net for an expert-tier "Deep
Analyze" mode; that was removed 2026-09-12 so the app relies solely on
what this repo trains. There is no expert-tier work here.)

This is its own git repository (remote: `rayuhidaka/igo_selftrainRL`),
living as a (gitignored) subfolder of `igo-app` on disk purely for
convenience while both are under active development side by side — it
has no dependency on `igo-app`. Written to be fully self-contained: avoid
adding dependencies on anything outside this folder.

## Tech stack
- Python, PyTorch (training)
- ONNX for the intermediate export format, then converted to TensorFlow
  Lite for the app
- All training runs locally on this machine, inside WSL (venv at
  `~/venvs/igo-training`), never on a rented/cloud GPU. Self-play and
  evaluation are CPU-bound (4 worker processes); fine-tuning uses the
  local GPU via `device: auto` when CUDA is available.
- Each generation is one `run_option3_genN.sh` script (self-play →
  fine-tune → opening spot-check → promotion match → round-robin field
  gate), launched in a detached WSL tmux session so a multi-hour run
  survives independently of any interactive session.

## Architecture
- `bootstrap/` — imitation-learning training against KataGo self-play
  games or public 9x9 game records; produces Ray-zeroGo's initial
  checkpoint, before any self-play fine-tuning
- `selfplay/` — self-play game generation + policy/value update loop,
  used to fine-tune Ray-zeroGo past its bootstrapped starting point
- `export/` — PyTorch → ONNX → TensorFlow Lite conversion; this is the
  contract boundary with `igo-app` (see docs/ARCHITECTURE.md for the
  exact tensor shapes `igo-app/inference/` expects — it's the same
  `docs/MODEL_CONTRACT.md` contract `igo-app/inference/` loads every
  checkpoint through, so any generation is a drop-in replacement for
  any other from `igo-app`'s side)
- `eval/` — deciding which checkpoints become difficulty tiers:
  `promote.py` (candidate vs. its parent), `round_robin.py` (every
  checkpoint vs. every other on one shared Elo scale — the ratings
  `igo-app` ships), and `gate.py` (a candidate must also rate at least
  its parent on that round-robin, not just beat it head-to-head — see
  `eval/README.md`)

## Conventions
- Every checkpoint is versioned and logged with: architecture config,
  training data source, random seed, and Elo estimate at save time
- Hyperparameters live in config files, not hardcoded in scripts
- Keep this folder importable/runnable independent of the Android app —
  no assumptions about Kotlin, Gradle, or Android paths

## Development commands
Run from inside WSL with `~/venvs/igo-training` activated:
- `python -m unittest discover` — the test suite
- `./run_option3_genN.sh` — one full generation (see the Tech stack note
  on tmux)
- `python -m selfplay.run_parallel --config <selfplay config> --workers 4`
- `python -m bootstrap.train --config <training config>`
- `python -m eval.promote --config <eval config> --candidate <path>.pt`
- `python -m eval.round_robin --config configs/eval_round_robin_option3.yaml --reuse-existing`
  (add `--add-checkpoint NAME=<path>.pt --out-prefix gate_NAME_` for a
  candidate's field check, then `python -m eval.gate`)
- `python -m export.to_tflite --checkpoint <path>.pt --board-size 9 --out <path>.tflite`

## Things to avoid
- Don't change the exported tensor shapes without updating
  `docs/ARCHITECTURE.md` and flagging it — `igo-app/inference/` depends
  on that contract staying stable.
- Don't check large checkpoint files or training game logs into git —
  keep those out via `.gitignore` and store them separately.

## Where things stand
Check ROADMAP.md for the current phase and open tasks (this repo's own
phase numbering — not `igo-app`'s).
