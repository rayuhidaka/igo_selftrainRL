"""Ray-zeroGo's policy/value net architecture.

Conforms to igo-app's docs/MODEL_CONTRACT.md so a checkpoint trained here is
a drop-in replacement for igo-app's expert-tier checkpoint from the app's
side -- see docs/ARCHITECTURE.md for the full contract and why history/komi
aren't inputs here.

Widened from the original 2-conv-layer/32-channel placeholder (2026-09-03):
`bootstrap_batch2.pt` (3000 games, 224,730 examples) plateaued in training
loss around epoch 25 rather than continuing to fall, suggesting the
original size had hit a capacity ceiling on real data rather than a data
ceiling -- see docs/ROADMAP.md's Phase 2. This version (3 conv layers,
64 channels) is the first attempt at testing that theory; still not
seriously tuned.
"""

from __future__ import annotations

import torch
from torch import nn


class RayZeroNet(nn.Module):
    """Policy/value net for a `board_size` x `board_size` Go board.

    Input: `board_planes`, float32 `[N, 3, board_size, board_size]`, NCHW
    -- PyTorch/ONNX's native conv layout, NOT the NHWC layout
    docs/MODEL_CONTRACT.md specifies. `export/to_tflite.py`'s onnx2tf step
    converts NCHW-in/NCHW-out ONNX graphs to NHWC TFLite ones automatically
    (the same mechanism igo-app/tools/convert_katago_to_tflite.py relies on
    for KataGo's ONNX export) -- do NOT pre-permute to NHWC inside this
    module's forward(); an earlier version did, and it broke onnx2tf's
    layout inference (produced a mis-transposed [1,9,3,9] final input
    instead of [1,9,9,3]). Channels are (own stone, opponent stone, empty),
    relative to the player to move.

    Outputs:
      - `policy`: float32 `[N, board_size*board_size + 1]`, softmax over
        every board point (row-major) plus one trailing pass probability.
      - `value`: float32 `[N, 1]`, tanh-bounded expected outcome for the
        player to move (`+1` certain win, `-1` certain loss, `0` even).
    """

    def __init__(self, board_size: int = 9, channels: int = 64, num_conv_layers: int = 3) -> None:
        # num_conv_layers exists so a checkpoint saved before the 2->3-layer
        # widening (see this file's module docstring) can still be
        # reconstructed for loading (num_conv_layers=2) -- it's not a knob
        # meant for new checkpoints going forward. conv1/conv2/conv3 stay
        # named attributes rather than an nn.ModuleList specifically so
        # existing state_dict keys ("conv1.weight", ...) keep working.
        super().__init__()
        self.board_size = board_size
        self.num_conv_layers = num_conv_layers

        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        if num_conv_layers >= 3:
            self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        self.pass_fc = nn.Linear(channels, 1)

        self.value_fc1 = nn.Linear(channels, channels)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, board_planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.conv1(board_planes))
        x = torch.relu(self.conv2(x))
        if self.num_conv_layers >= 3:
            x = torch.relu(self.conv3(x))

        policy_map = torch.relu(self.policy_conv(x))
        policy_points = self.policy_fc(torch.flatten(policy_map, start_dim=1))

        pooled = x.mean(dim=[2, 3])
        policy_pass = self.pass_fc(pooled)
        policy = torch.softmax(torch.cat([policy_points, policy_pass], dim=1), dim=1)

        value_hidden = torch.relu(self.value_fc1(pooled))
        value = torch.tanh(self.value_fc2(value_hidden))

        return policy, value
