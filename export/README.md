# export/

`to_tflite.py` converts a `RayZeroNet` checkpoint (`bootstrap/`) into the
`.tflite` file `igo-app/inference/TfLitePolicyValueNet.kt` actually loads
— the one place this repo's output crosses into `igo-app`'s contract
(`igo-app/docs/MODEL_CONTRACT.md`).

## The pipeline: PyTorch → ONNX → TFLite

1. **Strip the auxiliary heads.** `_PolicyValueOnlyExportWrapper` calls the
   real model and discards `score_margin`/`ownership` before export — the
   exported net always conforms to the contract's fixed `(policy, value)`
   two-tensor output regardless of whether the source checkpoint has a
   score head, an ownership head, both, or neither, exactly like KataGo
   drops its own auxiliary heads at export time (they're training-only,
   never used for actual play).
2. **`torch.onnx.export`** with a dummy `[1, 3, board_size, board_size]`
   NCHW input (`RayZeroNet.forward`'s native layout — see `bootstrap/model.py`'s
   docstring for why nothing pre-permutes to NHWC before this step), named
   `input_names=["board_planes"]`/`output_names=["policy", "value"]`,
   opset 17, using the legacy TorchScript-based exporter rather than
   torch's newer "dynamo" one (avoids an extra `onnxscript` dependency for
   a model this simple).
3. **`onnx2tf.convert`** turns the ONNX graph into a TF SavedModel + raw
   float32 `.tflite`, converting NCHW to TFLite's native NHWC layout
   automatically as part of the graph translation — the same mechanism
   `igo-app/tools/convert_katago_to_tflite.py` relies on for KataGo's own
   ONNX export.
4. **Rename I/O tensors.** `onnx2tf`'s conversion renames every output to
   generic `Identity`/`Identity_N` names in the final `.tflite`, regardless
   of the ONNX graph's own correct names — `inference/` and
   `igo-app/inference/` both look outputs up **by name**
   (`"policy"`/`"value"`), so `rename_io_tensors` post-processes the
   `.tflite` flatbuffer directly (via `schema_py_generated`) to restore the
   intended names after conversion. Same workaround `igo-app/tools/` needed
   for KataGo's conversion, independently discovered here.

## A non-obvious operational detail

`skip_onnx2tf_calibration_data_download` stubs out an `onnx2tf` side quirk:
whenever an input looks like a batch of 3-channel images —
`board_planes`'s shape happens to qualify — `onnx2tf` unconditionally
tries to download a calibration sample, even though nothing here enables
the optional numerical comparison it's for. The download itself doesn't
load cleanly on this project's numpy version regardless, so this stubs it
out entirely with a same-shaped random array and moves on; see the
function's own docstring for the two concrete failures hit trying to let
the real download through.

## Usage

```
python -m export.to_tflite --checkpoint <path>.pt --board-size 9 --out <path>.tflite
python -m export.to_tflite --board-size 9 --out <path>.tflite   # fresh untrained net, Phase 1 style
```

Prints the resulting tensor names/shapes/dtypes after conversion, so a
mismatch against `igo-app/docs/MODEL_CONTRACT.md` is visible immediately
rather than discovered later inside the Android app.
