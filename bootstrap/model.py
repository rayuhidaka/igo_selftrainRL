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

`has_score_head` (2026-09-03, see docs/SELF_PLAY_STABILITY.md) adds an
auxiliary score-margin regression head, KataGo-style: pure win/loss/tie
value targets are the right *primary* training signal (training on raw
score margin instead causes a well-documented opposite pathology -- an
already-winning net takes needless risks to inflate the margin), but on
their own they saturate flat near +-1 with no signal distinguishing an
efficient win from an inefficient one, or a decisive win from a lucky
one -- directly implicated in a real self-play collapse this project
hit (empty-board Pass climbing to 25%+ within two fine-tune
generations). The auxiliary head's own output is never used for actual
play (see `export/to_tflite.py`'s export wrapper, which strips it) --
purely a training-time regularizer, exactly as KataGo uses it.

`use_global_pooling`/`has_ownership_head` (2026-09-11): even after
dihedral augmentation (`bootstrap/dataset.py`) fixed rotational symmetry,
self-play never preferred a star point over a corner opening -- and this
net's own architecture history was never actually driven to a proven
ceiling the way the two prior upgrades above were (the one real residual-
tower run hit its time cap while loss was still falling). Research into
KataGo's own fix for this class of problem (a whole-board strategic
judgment, not a local pattern) pointed at global pooling and a spatial
ownership auxiliary target rather than blind capacity increases -- see
`GlobalPoolingBias`'s docstring and igo-app/docs/ROADMAP.md's "weak
opening moves" item for the full reasoning.
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


class GlobalPoolingBias(nn.Module):
    """KataGo-style global pooling bias (Wu et al. 2020, "Accelerating Self-Play Learning
    in Go," section 3.3), simplified for this net's much smaller scale: KataGo pools a
    separate "gating" channel subset G and biases a different channel subset X; this module
    pools and biases the same trunk features into themselves (no artificial channel split),
    since this net's trunk is a single homogeneous channel bank rather than KataGo's much
    wider one. Lets otherwise-local 3x3 convolutions condition on whole-board context --
    e.g. "is a symmetric corner or a star point the better opening move right now," a
    whole-board strategic judgment plain local convolution can't directly make. Applied once,
    after the full residual tower (see RayZeroNet.forward), not per-block like KataGo's -- a
    simpler starting point for this net's scale; revisit if it doesn't move the needle. See
    igo-app/docs/ROADMAP.md's "weak opening moves" item for the diagnosed problem this
    targets.

    The three pooled statistics (mean, a board-width-scaled mean, and max, each per channel)
    follow KataGo's paper description; the exact scaling constant KataGo uses for the
    width-scaled mean isn't published, so this uses the board width itself as the scale -- a
    reasonable interpretation, not a verified byte-exact port.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.fc = nn.Linear(channels * 3, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.relu(self.bn(x))
        board_width = g.shape[-1]
        mean = g.mean(dim=[2, 3])
        scaled_mean = mean * board_width
        max_ = g.amax(dim=[2, 3])
        pooled = torch.cat([mean, scaled_mean, max_], dim=1)
        bias = self.fc(pooled).unsqueeze(-1).unsqueeze(-1)
        return x + bias


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

    `forward()` always returns a 4-tuple, `(policy, value, score_margin,
    ownership)` -- `score_margin` is `None` when `has_score_head` is
    `False` (every checkpoint before 2026-09-03), `ownership` is `None`
    when `has_ownership_head` is `False` (every checkpoint before this
    field existed). See this file's module docstring for why the
    auxiliary heads exist and why both are training-only.

    `use_global_pooling` (2026-09-11, see igo-app/docs/ROADMAP.md's "weak
    opening moves" item) inserts one `GlobalPoolingBias` after the full
    residual tower, before either head reads from it -- only meaningful on
    the residual path (`num_residual_blocks > 0`; raises `ValueError`
    otherwise). Targets a specific diagnosed problem: dihedral-augmented
    checkpoints still never preferred a star point over a corner opening,
    which looks like a whole-board strategic judgment plain local
    convolution can't make, not a raw-capacity problem (this net's
    receptive field already spans the entire 9x9 board with far fewer
    blocks than it now has).

    `has_ownership_head` (2026-09-11, same motivation) adds a per-point
    ownership regression head (`ownership_conv`, tanh-bounded, flattened to
    `[N, board_size**2]` row-major -- matching `policy`'s own flat
    convention, not left spatial -- `+1`/`-1` per point from the
    player-to-move's perspective, `0` neutral/dame) --
    KataGo's own biggest sample-efficiency win (Wu et al. 2020 section
    4.1): a wrong per-point prediction gives localized gradient feedback,
    needing far fewer self-play games to learn from than the scalar score
    head alone. Simpler than KataGo's categorical/pdf-cdf scheme -- plain
    MSE regression, consistent with this file's existing (simpler) choice
    of MSE over classification for the score head. Training-only, like the
    score head -- stripped at export (see `export/to_tflite.py`).
    """

    def __init__(
        self,
        board_size: int = 9,
        channels: int = 64,
        num_conv_layers: int = 3,
        num_residual_blocks: int = 0,
        has_score_head: bool = False,
        use_global_pooling: bool = False,
        has_ownership_head: bool = False,
    ) -> None:
        # conv1/conv2/conv3 and residual_blocks stay named/ModuleList attributes matching
        # exactly what each architecture generation was actually saved with -- see this
        # file's module docstring and bootstrap/checkpoint.py -- so existing state_dict
        # keys keep working for whichever generation a checkpoint came from.
        super().__init__()
        if use_global_pooling and num_residual_blocks == 0:
            raise ValueError("use_global_pooling requires num_residual_blocks > 0")

        self.board_size = board_size
        self.channels = channels
        self.num_conv_layers = num_conv_layers
        self.num_residual_blocks = num_residual_blocks
        self.has_score_head = has_score_head
        self.use_global_pooling = use_global_pooling
        self.has_ownership_head = has_ownership_head

        if num_residual_blocks > 0:
            self.stem_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1)
            self.stem_bn = nn.BatchNorm2d(channels)
            self.residual_blocks = nn.ModuleList(ResidualBlock(channels) for _ in range(num_residual_blocks))
        else:
            self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            if num_conv_layers >= 3:
                self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        if use_global_pooling:
            self.global_pooling = GlobalPoolingBias(channels)

        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        self.pass_fc = nn.Linear(channels, 1)

        self.value_fc1 = nn.Linear(channels, channels)
        self.value_fc2 = nn.Linear(channels, 1)

        if has_score_head:
            self.score_fc1 = nn.Linear(channels, channels)
            self.score_fc2 = nn.Linear(channels, 1)

        if has_ownership_head:
            self.ownership_conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(
        self, board_planes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, "torch.Tensor | None", "torch.Tensor | None"]:
        if self.num_residual_blocks > 0:
            x = torch.relu(self.stem_bn(self.stem_conv(board_planes)))
            for block in self.residual_blocks:
                x = block(x)
        else:
            x = torch.relu(self.conv1(board_planes))
            x = torch.relu(self.conv2(x))
            if self.num_conv_layers >= 3:
                x = torch.relu(self.conv3(x))

        if self.use_global_pooling:
            x = self.global_pooling(x)

        policy_map = torch.relu(self.policy_conv(x))
        policy_points = self.policy_fc(torch.flatten(policy_map, start_dim=1))

        pooled = x.mean(dim=[2, 3])
        policy_pass = self.pass_fc(pooled)
        policy = torch.softmax(torch.cat([policy_points, policy_pass], dim=1), dim=1)

        value_hidden = torch.relu(self.value_fc1(pooled))
        value = torch.tanh(self.value_fc2(value_hidden))

        score_margin = None
        if self.has_score_head:
            score_hidden = torch.relu(self.score_fc1(pooled))
            score_margin = self.score_fc2(score_hidden)

        ownership = None
        if self.has_ownership_head:
            # Flattened to [N, board_size**2] (row-major), not left spatial [N, H, W] --
            # matches policy's own flat convention and bootstrap/dataset.py's
            # ownership_targets shape (engine/scoring.py's ownership_plane is flat too).
            ownership = torch.tanh(self.ownership_conv(x)).flatten(start_dim=1)

        return policy, value, score_margin, ownership
