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

## Explicitly deferred
- Training on board sizes other than 9x9
- Any online/continuous training after initial checkpoints ship
- Feeding real move history / komi / ruleset into the net (see above) —
  would need a coordinated contract change on the `igo-app` side too
