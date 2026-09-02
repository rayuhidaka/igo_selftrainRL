"""Shared training-monitoring helpers, for bootstrap/ and, later, selfplay/.

TensorBoard, run entirely locally -- no account or cloud dependency,
consistent with this project's local-only training decision (see
docs/ROADMAP.md). View a run with:
    tensorboard --logdir runs/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class TrainingMonitor:
    """Wraps a `SummaryWriter` and also prints to stdout, so a run is
    inspectable both live (TensorBoard) and from a plain terminal/log file.
    """

    def __init__(self, log_dir: Path, run_name: str) -> None:
        self.writer = SummaryWriter(log_dir=str(Path(log_dir) / run_name))

    def log_config(self, config: dict[str, Any]) -> None:
        """Records the run's config as TensorBoard text, so a run is self-describing later."""
        lines = "\n".join(f"{key}: {value}" for key, value in config.items())
        self.writer.add_text("config", f"```\n{lines}\n```")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)
        print(f"[step {step}] {tag} = {value:.4f}")

    def close(self) -> None:
        self.writer.close()
