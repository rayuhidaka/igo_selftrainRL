# igo-training

Training pipeline for "Ray-zeroGo", the opponent-tier neural net used by
`igo-app` — bootstrapped via imitation learning, then self-play fine-tuned,
saved at intervals to become the app's selectable difficulty levels.

There's no expert-tier work here: `igo-app`'s live analysis overlay and
"Deep Analyze" already run on a real checkpoint (KataGo's own pretrained
net, converted directly, no training involved) — see `igo-app/tools/`.

## Status
Phase 1 (scaffolding). See `docs/ROADMAP.md`.

## Setup (Windows)
1. Install Python 3.11+ and a CUDA-capable environment if training
   locally, or plan to run on a rented cloud GPU (spot instance) for
   anything beyond small experiments — 9x9 self-play is far cheaper than
   19x19 but still meaningfully GPU-bound.
2. If working under WSL2 (this repo's actual dev environment so far —
   native Windows hit TensorFlow-blocked-by-Smart-App-Control the same
   way igo-app's tools/ did, see igo-app/docs/BUILD_NOTES.md): put the
   venv on WSL's **native filesystem**, not `/mnt/c/...` — e.g.
   `python -m venv ~/venvs/igo-training`. A venv under `/mnt/c` made
   `import torch` take 30+ seconds (drvfs overhead reading torch's many
   large shared libraries) vs ~5 seconds natively. The project files
   themselves are fine to keep on the Windows-mounted path; it's only the
   venv's many-large-files layout that's affected.
3. `pip install -r requirements.txt` -- on a CPU-only machine, prefer
   `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`
   instead: torch's default PyPI wheel bundles ~10GB of CUDA/cuDNN
   dependencies that a CPU-only box will never use.

## Output contract
This pipeline's job is to produce `.tflite` checkpoints `igo-app` can load
directly, conforming to `igo-app/docs/MODEL_CONTRACT.md` — the same
contract the expert-tier checkpoint already conforms to, so a Ray-zeroGo
checkpoint is a drop-in replacement from `igo-app`'s side. See
`docs/ARCHITECTURE.md` here for how this pipeline meets that contract.

## Relationship to igo-app
Its own git repository (no remote yet), living as a subfolder of `igo-app`
on disk purely for convenience while both are under active development —
no code dependency in either direction.
