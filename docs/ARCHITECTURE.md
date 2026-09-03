# Architecture — igo-training

## The contract with igo-app

`igo-app/inference/` loads a `.tflite` checkpoint by these exact tensor
names and shapes, per `igo-app/docs/MODEL_CONTRACT.md` (the authoritative
copy — this is a summary, not a fork of it; if the two ever disagree,
that file wins):

- **Input** `board_planes`: `[1, 9, 9, 3]` float32, NHWC. Channel 0 = the
  point is occupied by the player to move, channel 1 = occupied by the
  opponent, channel 2 = empty — relative to whoever's turn it is, not a
  fixed color, so the same weights work playing either side.
- **Output** `policy`: `[1, 82]` float32, softmax. Index `row * 9 + col` is
  that point's move probability; index 81 is the pass probability.
- **Output** `value`: `[1, 1]` float32, tanh-bounded expected outcome for
  the player to move (`+1` certain win, `-1` certain loss, `0` even).
- Outputs are looked up **by name** (`policy`, `value`) at runtime, not by
  index — `export/` must produce a `.tflite` file where those are the
  actual graph output tensor names, not just correctly-shaped anonymous
  outputs. (`igo-app/tools/convert_katago_to_tflite.py` had to work around
  TFLite converters silently renaming outputs to `Identity`/`Identity_N`
  — see that file if `export/` hits the same thing.)

No history, komi, or ruleset context is part of this input — a `Position`
snapshot is all the net gets. `igo-app`'s expert-tier checkpoint (see its
`docs/MODEL_CONTRACT.md`) currently approximates komi as 0 and zero-fills
history/ladder/territory features for the same reason: this contract has
nowhere to put them. Ray-zeroGo should be **trained** against this same
constraint (no history/komi as input) rather than trained with richer
features and then having them dropped at export time — otherwise its
training-time and inference-time inputs would silently diverge.

## PyTorch checkpoint format (internal, not the igo-app contract above)

Distinct from the `.tflite` export contract above: `bootstrap/checkpoint.py`
wraps a `RayZeroNet` `state_dict` with its own architecture (`channels`,
`num_conv_layers`), training data source, seed, and (once `eval/` has
estimated one) Elo — `{"model_state_dict": ..., "metadata": {...}}`, not a
bare `state_dict`. Every loader (`bootstrap/inference.py`'s
`RayZeroPolicyValueNet`, `export/to_tflite.py`) goes through
`load_checkpoint`/`build_model` so a checkpoint self-describes the net it
needs, rather than the caller having to already know it.

This exists because of a real bug (2026-09-03): widening `RayZeroNet`
(see docs/ROADMAP.md's Phase 2) changed its constructor's defaults, which
broke loading every checkpoint saved before the change — callers always
built *today's* architecture before `load_state_dict`, so an old
checkpoint's missing/extra keys crashed it. Checkpoints saved before this
module existed (a bare `state_dict`) still load fine (`metadata=None`),
they just need `channels`/`num_conv_layers` supplied explicitly by the
caller the same way every checkpoint used to require — see
`eval/promote.py`'s `current_tier_channels`/`current_tier_num_conv_layers`
config keys for that path.

## Pipeline stages

1. `bootstrap/` — trains Ray-zeroGo's initial network via imitation
   learning against KataGo self-play games or public 9x9 game records.
2. `selfplay/` — takes the bootstrapped checkpoint and continues training
   via self-play generation + policy/value updates. Saves checkpoints at
   intervals; each saved checkpoint is a candidate difficulty tier.
3. `eval/` — plays checkpoints against each other (and/or fixed
   benchmarks) to estimate Elo, so `igo-app` can label difficulty tiers
   sensibly ("early Ray-zeroGo" through "latest Ray-zeroGo").
4. `export/` — converts a chosen PyTorch checkpoint to ONNX, then to
   TensorFlow Lite (via `onnx2tf`, same tail end as
   `igo-app/tools/convert_katago_to_tflite.py`), validating against the
   contract above before handoff. Working end to end as of this writing
   (`export/to_tflite.py`, proven against an untrained placeholder
   checkpoint) — non-obvious things it took to get there, in case the next
   change to this file breaks one of them again:
   - **`bootstrap/model.py`'s `RayZeroNet.forward` takes NCHW input, not
     NHWC**, despite the exported `.tflite`'s `board_planes` being NHWC.
     `onnx2tf` converts an NCHW ONNX graph to a NHWC TFLite one
     automatically (same mechanism the KataGo conversion relies on) — but
     only if the ONNX graph is actually NCHW-native. An earlier version of
     this model pre-permuted NHWC→NCHW *inside* `forward()`, which
     confused `onnx2tf`'s own layout inference into producing a
     mis-transposed `[1, 9, 3, 9]` input instead of `[1, 9, 9, 3]`.
   - `torch.onnx.export(..., dynamo=False)` is required — torch's default
     "dynamo" exporter (torch ≥2.5) needs the `onnxscript` package, which
     isn't in `requirements.txt`. The legacy exporter doesn't need it and
     is equivalent for a model this simple. Its `input_names`/
     `output_names` do carry through to the ONNX graph correctly, but...
   - ...`onnx2tf` still renames the **final** `.tflite`'s output tensors to
     `Identity`/`Identity_N` regardless of what the ONNX graph itself calls
     them — same issue `convert_katago_to_tflite.py` hit, same fix
     (`export/to_tflite.py`'s `rename_io_tensors`, a flatbuffer-level
     rename after conversion).
   - `onnx2tf` unconditionally tries to download a calibration image
     sample whenever an input happens to look like a batch of 3-channel
     images (`board_planes` qualifies, since docs/MODEL_CONTRACT.md's 3
     channels happen to match that shape) — even though it's only used for
     an optional numerical check this project doesn't enable. Needs
     network access to even attempt, and even with network on, the
     downloaded file didn't survive a current numpy's stricter
     `np.load(allow_pickle=False)` default and then failed to unpickle
     cleanly at all. Not worth chasing further — `export/to_tflite.py`
     stubs out the download function entirely
     (`skip_onnx2tf_calibration_data_download`) rather than fighting either
     problem.
   - A `tf_keras` dependency pulls in full `tensorflow` (not just
     `tensorflow-cpu`) and crashes trying to init CUDA even when none is
     requested — fix: `CUDA_VISIBLE_DEVICES=-1` in the environment running
     `export/to_tflite.py`.

## Compute notes
Training runs locally on the user's own machine — no rented/cloud GPU,
by deliberate choice (see docs/ROADMAP.md's Phase 2). 9x9 self-play is
far cheaper than 19x19, but still real GPU work, so expect it to take
real wall-clock time on consumer hardware. Bootstrapping via imitation
learning from KataGo first (rather than training from a random
initialization) cuts the compute needed substantially — this is the
intended default path, not an optional shortcut. How to actually
structure the work given a local-only budget (phased training runs,
batch scheduling, how long a run is allowed to take before checking in)
is not decided yet — to be worked out when Phase 2 starts in earnest.

## Monitoring
`bootstrap/monitoring.py`'s `TrainingMonitor` wraps TensorBoard
(`torch.utils.tensorboard`), run entirely locally — no account or cloud
service, consistent with the local-training decision above. View a run
with `tensorboard --logdir runs/` while training is in progress or after.
`bootstrap/train.py` already logs the run's config at start; the real
training loop (Phase 2) should call `monitor.log_scalar(...)` for
policy/value loss at `config["log_every_n_steps"]` — see the TODO block in
`bootstrap/train.py`. `selfplay/`'s loop, once it exists, should use the
same `TrainingMonitor` rather than inventing a second logging path.

## Difficulty-tier promotion (eval/)

Checkpointing and *promoting* a checkpoint to a shippable difficulty tier
are different decisions on different cadences:

- **Checkpointing** (`bootstrap/train.py`'s `config["checkpoint_interval_steps"]`,
  and later `selfplay/`'s equivalent) is cheap and frequent — purely for
  resumability, not a quality judgment.
- **Promotion** (`eval/promote.py`) happens far less often, and is gated on
  actually being stronger: a candidate checkpoint is played against the
  currently-shipped tier (`eval/match.py`, `configs/eval_base.yaml`'s
  `num_games`), the results feed `eval/elo.py`'s `update_ratings`, and the
  candidate is only promoted (`should_promote`) if it clears the current
  tier by at least `min_elo_gap`. This is deliberate: checkpointing every
  N steps and shipping *all* of them as difficulty tiers would fill the
  app's difficulty picker with many near-identical-strength options
  instead of a meaningful ladder from weak to strong.

`eval/elo.py` (the rating math) is fully implemented and tested
(`tests/test_elo.py`) — it has no dependency on anything else. `engine/`
(rules enforcement: legality, capture/ko, area scoring) is now a Python
port of `igo-app/engine/`'s Go rules (Kotlin) — see `engine/README.md` —
rather than a second, independently-written implementation, specifically
to avoid the two codebases silently disagreeing about what a legal game
even is. It's proven against a full port of `igo-app/engine/`'s own test
suite (`tests/test_position.py`, `tests/test_scoring.py`), not just
eyeballed — all passing.

`eval/match.py` (actually playing a game between two checkpoints) is done
too, closing out the whole `eval/` pipeline: it uses `mcts/` (a Python
port of `igo-app/mcts/`'s PUCT search — see `mcts/README.md`, proven
against a full port of `igo-app/mcts/`'s own test suite,
`tests/test_mcts.py`) driven by `bootstrap/inference.py`'s
`RayZeroPolicyValueNet` (a `PolicyValueNet` implementation over a
`RayZeroNet` checkpoint, mirroring `igo-app/inference/TfLitePolicyValueNet.kt`'s
encode/decode logic — `tests/test_inference.py` ports its test suite too).
`play_match` alternates which checkpoint plays Black each game so a color
advantage doesn't bias the result. Verified two ways:
- `tests/test_match.py`: a checkpoint played against itself averages to a
  ~50/50 result (expected, since neither `Mcts` nor `RayZeroPolicyValueNet`
  introduces randomness — the alternation exactly cancels whatever a fixed
  color advantage would otherwise cause).
- A real CLI run, not just unit tests: two untrained 9x9 checkpoints,
  `configs/eval_smoke_test.yaml` (4 games, 16 simulations/move) via
  `python -m eval.promote`, produced a real Elo verdict in ~23 seconds
  (`Candidate rating: 1555.8 vs current tier: 1444.2` → `PROMOTE`). See
  that config's header for the exact commands to reproduce.

`eval/`'s design is now fully implemented end to end — what's left is
real training producing checkpoints actually worth evaluating (Phase 2).

## Explicitly deferred
- Training on board sizes other than 9x9
- Any online/continuous training after initial checkpoints ship
- Feeding real move history / komi / ruleset into the net (see above) —
  would need a coordinated contract change on the `igo-app` side too
