"""Exports a RayZeroNet checkpoint to a .tflite conforming to igo-app's
docs/MODEL_CONTRACT.md (see docs/ARCHITECTURE.md here for the summary).

Usage:
    python -m export.to_tflite --checkpoint <path>.pt --board-size 9 --out <path>.tflite
    python -m export.to_tflite --board-size 9 --out <path>.tflite   # fresh untrained net
"""

from __future__ import annotations

import argparse
import contextlib
import tempfile
from pathlib import Path

import flatbuffers
import numpy as np
import onnx2tf
import tensorflow as tf
import torch
from tensorflow.lite.python import schema_py_generated as schema_fb

from bootstrap.checkpoint import build_model, load_checkpoint


class _PolicyValueOnlyExportWrapper(torch.nn.Module):
    """Strips RayZeroNet.forward()'s auxiliary score-margin/ownership outputs (see its
    module docstring) before ONNX export -- the exported .tflite conforms to
    igo-app/docs/MODEL_CONTRACT.md's fixed policy+value contract regardless of whether the
    source checkpoint has a score or ownership head at all, exactly like KataGo drops its
    own auxiliary heads at export time (they're training-only, never used for actual play).
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, board_planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy, value, _, _ = self.model(board_planes)
        return policy, value


@contextlib.contextmanager
def skip_onnx2tf_calibration_data_download():
    """Stubs out an onnx2tf side quirk, scoped to one call.

    Whenever an input looks like a batch of 3-channel images (board_planes
    qualifies, since docs/MODEL_CONTRACT.md's 3 input channels happen to
    match that shape), onnx2tf unconditionally downloads a calibration
    sample -- even though it's only used for an optional numerical
    ONNX-vs-TF comparison this project doesn't enable
    (`check_onnx_tf_outputs_elementwise_close` defaults to, and stays,
    False). Hit two real problems trying to let the download go through
    (see docs/ARCHITECTURE.md): the file needs `allow_pickle=True` for a
    current numpy, and even then its content didn't unpickle cleanly. Not
    our bug to chase further -- stub out the whole download with a
    same-shape random array (20x128x128x3, per the file's own name) and
    move on; nothing downstream depends on the values.
    """
    original = onnx2tf.onnx2tf.download_test_image_data
    onnx2tf.onnx2tf.download_test_image_data = lambda: np.random.rand(20, 128, 128, 3).astype(np.float32)
    try:
        yield
    finally:
        onnx2tf.onnx2tf.download_test_image_data = original


def export_onnx(model: torch.nn.Module, board_size: int, onnx_path: Path) -> None:
    # NCHW, matching RayZeroNet.forward's native conv layout (see its
    # docstring) -- onnx2tf below converts this to the NHWC TFLite input
    # docs/MODEL_CONTRACT.md requires automatically.
    dummy_input = torch.zeros(1, 3, board_size, board_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["board_planes"],
        output_names=["policy", "value"],
        opset_version=17,
        # torch>=2.5's default "dynamo" exporter needs the onnxscript package
        # (not in requirements.txt); the legacy TorchScript-based exporter
        # doesn't and produces the same result for a model this simple.
        dynamo=False,
    )


def rename_io_tensors(tflite_bytes: bytes, input_names: dict[int, str], output_names: dict[int, str]) -> bytes:
    """Renames a .tflite model's graph input/output tensors in place.

    onnx2tf's conversion renames outputs to "Identity"/"Identity_N" in the
    final .tflite regardless of the ONNX graph's own (correct) output
    names -- same issue igo-app/tools/convert_katago_to_tflite.py hit and
    worked around the same way. `input_names`/`output_names` map graph
    input/output *position* (0, 1, ...) to the desired tensor name.
    """
    model = schema_fb.ModelT.InitFromPackedBuf(bytearray(tflite_bytes), 0)
    subgraph = model.subgraphs[0]
    for position, tensor_index in enumerate(subgraph.inputs):
        if position in input_names:
            subgraph.tensors[tensor_index].name = input_names[position].encode("utf-8")
    for position, tensor_index in enumerate(subgraph.outputs):
        if position in output_names:
            subgraph.tensors[tensor_index].name = output_names[position].encode("utf-8")

    builder = flatbuffers.Builder(len(tflite_bytes))
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def convert(
    checkpoint: Path | None,
    board_size: int,
    out_path: Path,
    channels: int | None = None,
    num_conv_layers: int | None = None,
    num_residual_blocks: int | None = None,
    has_score_head: bool | None = None,
    use_global_pooling: bool | None = None,
    has_ownership_head: bool | None = None,
) -> bytes:
    # channels/num_conv_layers/num_residual_blocks/has_score_head/use_global_pooling/
    # has_ownership_head only need supplying for a legacy checkpoint (saved before
    # bootstrap/checkpoint.py existed) or to deliberately override -- see
    # bootstrap/checkpoint.py's build_model.
    state_dict, metadata = (None, None) if checkpoint is None else load_checkpoint(checkpoint)
    model = build_model(
        metadata,
        board_size,
        channels=channels,
        num_conv_layers=num_conv_layers,
        num_residual_blocks=num_residual_blocks,
        has_score_head=has_score_head,
        use_global_pooling=use_global_pooling,
        has_ownership_head=has_ownership_head,
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()
    export_model = _PolicyValueOnlyExportWrapper(model)

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "model.onnx"
        export_onnx(export_model, board_size, onnx_path)

        saved_model_dir = Path(tmp) / "saved_model"
        with skip_onnx2tf_calibration_data_download():
            onnx2tf.convert(
                input_onnx_file_path=str(onnx_path),
                output_folder_path=str(saved_model_dir),
            )

        candidates = list(saved_model_dir.glob("*_float32.tflite"))
        if not candidates:
            raise RuntimeError(f"onnx2tf did not produce a float32 .tflite in {saved_model_dir}")
        tflite_bytes = candidates[0].read_bytes()

    tflite_bytes = rename_io_tensors(
        tflite_bytes, input_names={0: "board_planes"}, output_names={0: "policy", 1: "value"}
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_bytes)
    return tflite_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path; omit for a fresh, untrained net")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--channels", type=int, default=None, help="Only needed for a legacy checkpoint or to override"
    )
    parser.add_argument(
        "--num-conv-layers", type=int, default=None, help="Only needed for a legacy checkpoint or to override"
    )
    parser.add_argument(
        "--num-residual-blocks", type=int, default=None, help="Only needed for a legacy checkpoint or to override"
    )
    parser.add_argument(
        "--has-score-head", type=bool, default=None, help="Only needed for a legacy checkpoint or to override"
    )
    parser.add_argument(
        "--use-global-pooling", type=bool, default=None, help="Only needed for a legacy checkpoint or to override"
    )
    parser.add_argument(
        "--has-ownership-head", type=bool, default=None, help="Only needed for a legacy checkpoint or to override"
    )
    args = parser.parse_args()

    tflite_bytes = convert(
        args.checkpoint,
        args.board_size,
        args.out,
        args.channels,
        args.num_conv_layers,
        args.num_residual_blocks,
        args.has_score_head,
        args.use_global_pooling,
        args.has_ownership_head,
    )
    print(f"Wrote {args.out} ({len(tflite_bytes)} bytes)")

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    print("Inputs:")
    for detail in interpreter.get_input_details():
        print(f"  {detail['name']}: shape={detail['shape'].tolist()} dtype={detail['dtype']}")
    print("Outputs:")
    for detail in interpreter.get_output_details():
        print(f"  {detail['name']}: shape={detail['shape'].tolist()} dtype={detail['dtype']}")


if __name__ == "__main__":
    main()
