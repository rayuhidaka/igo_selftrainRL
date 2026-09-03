"""Ray-zeroGo's policy/value net architecture.

Conforms to igo-app's docs/MODEL_CONTRACT.md so a checkpoint trained here is
a drop-in replacement for igo-app's expert-tier checkpoint from the app's
side -- see docs/ARCHITECTURE.md for the full contract and why history/komi
aren't inputs here.

Architecture history (see docs/ROADMAP.md's Phase 2 for the full story):
2 conv layers/32 channels (original placeholder) -> 3 layers/64 channels
(2026-09-03, confirmed the original had hit a capacity ceiling on real
data) -> confirmed the 3-layer net has its own ceiling too (141 epochs,
no further improvement past ~epoch 20) -> a small residual tower
(`num_residual_blocks > 0`), the standard next step for small-board Go
nets, tested here for the first time. `num_conv_layers` only matters when
`num_residual_blocks == 0` (the plain-stack path); both exist so a
checkpoint from any point in that history stays loadable -- see
bootstrap/checkpoint.py.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """One AlphaZero/KataGo-style residual block: two 3x3 convs (each followed by
    BatchNorm), a skip connection back to the block's input, then a final ReLU. Channel
    count is preserved end to end, which is what makes the skip-add valid without a
    projection layer.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(y + residual)


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

    def __init__(
        self, board_size: int = 9, channels: int = 64, num_conv_layers: int = 3, num_residual_blocks: int = 0
    ) -> None:
        # conv1/conv2/conv3 and residual_blocks stay named/ModuleList attributes matching
        # exactly what each architecture generation was actually saved with -- see this
        # file's module docstring and bootstrap/checkpoint.py -- so existing state_dict
        # keys keep working for whichever generation a checkpoint came from.
        super().__init__()
        self.board_size = board_size
        self.channels = channels
        self.num_conv_layers = num_conv_layers
        self.num_residual_blocks = num_residual_blocks

        if num_residual_blocks > 0:
            self.stem_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1)
            self.stem_bn = nn.BatchNorm2d(channels)
            self.residual_blocks = nn.ModuleList(ResidualBlock(channels) for _ in range(num_residual_blocks))
        else:
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
        if self.num_residual_blocks > 0:
            x = torch.relu(self.stem_bn(self.stem_conv(board_planes)))
            for block in self.residual_blocks:
                x = block(x)
        else:
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
