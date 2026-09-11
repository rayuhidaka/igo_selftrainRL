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
1. Training runs locally on this machine — no rented/cloud GPU. Install
   Python 3.11+ and a CUDA-capable environment if a local GPU is
   available; 9x9 self-play is far cheaper than 19x19 but still
   meaningfully GPU-bound, so expect training to take real wall-clock
   time on consumer hardware. How to actually structure that (phased
   training, batch sizes/scheduling, etc.) is still to be worked out —
   see docs/ROADMAP.md's Phase 2.
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

## How the algorithm works

An AlphaZero-shaped pipeline, split across six modules:

- **`engine/`** and **`mcts/`** — Python ports of `igo-app/engine/` and
  `igo-app/mcts/` (rules and PUCT search), used so this repo's self-play
  and evaluation matches run against the *exact same* rules and search
  algorithm the Android app itself plays with — see each folder's own
  README for the port and for one deliberate divergence (self-play-only
  root Dirichlet noise, not present on the Kotlin side).
- **`bootstrap/`** — the net (`RayZeroNet`: a residual-tower policy/value
  net, AlphaZero/KataGo-shaped) and the training loop that fits it to
  data, whichever stage produced that data. See `bootstrap/README.md` for
  the architecture and the loss.
- **`selfplay/`** — produces the training data `bootstrap/` consumes, in
  two stages: `generate.py` (Phase 2) distills moves directly from
  `igo-app`'s existing strong KataGo checkpoint, no search — cheap, and
  good enough for an initial imitation-learning warm start. `self_play.py`
  (Phase 3) is genuine self-play: real MCTS search with Ray-zeroGo's own
  current-best checkpoint, recording the search's visit-count distribution
  (not the raw policy) as the training target — much more expensive per
  move, but the actual reinforcement-learning loop that improves the net
  past what it was distilled from.
- **`eval/`** — Elo-rates a freshly-trained checkpoint against the current
  shippable difficulty tier by playing real games between them
  (`match.py`, using this repo's own `engine/`/`mcts/` ports), and decides
  whether it's improved *enough* to promote (`elo.py`'s `should_promote`)
  — checkpoints save often and cheaply during training, but only promoted
  checkpoints become new difficulty levels in the app.
- **`export/`** — converts a promoted `RayZeroNet` checkpoint to the
  `.tflite` file `igo-app` actually loads.

The generation loop is: `self_play.py` (current-best checkpoint) →
`bootstrap/train.py` (fine-tune, optionally with dihedral augmentation) →
`eval/promote.py` (does it beat the current tier?) → `export/to_tflite.py`
(if promoted) → repeat with the new checkpoint as the next generation's
self-play source. See `docs/ARCHITECTURE.md` for the full design and
`docs/ROADMAP.md` for exactly how far around this loop the project has
actually gotten.

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
