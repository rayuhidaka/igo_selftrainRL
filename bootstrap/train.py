"""Builds and saves a RayZeroNet checkpoint.

Phase 1 scaffolding: this does NOT do real imitation-learning training yet
(see docs/ROADMAP.md's Phase 2) -- it builds a deterministically-seeded,
untrained net and saves it, purely so export/to_tflite.py and igo-app's
loading path have a real checkpoint to prove the pipeline against.

Usage:
    python -m bootstrap.train --config configs/bootstrap_base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from bootstrap.model import RayZeroNet
from bootstrap.monitoring import TrainingMonitor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    monitor = TrainingMonitor(log_dir=Path(config["log_dir"]), run_name=config["run_name"])
    monitor.log_config(config)

    torch.manual_seed(config["seed"])
    model = RayZeroNet(board_size=config["board_size"])

    # ------------------------------------------------------------------
    # Phase 2 TODO: replace this with the real imitation-learning loop
    # (load config["data_source"], train for config["epochs"] over
    # config["batch_size"]-sized batches at config["learning_rate"]).
    # Two things the real loop must do that this placeholder doesn't:
    #   - call monitor.log_scalar("loss/policy", ..., step) and
    #     ("loss/value", ..., step) every config["log_every_n_steps"]
    #   - save a checkpoint every config["checkpoint_interval_steps"]
    #     (cheap, for resumability -- NOT the same as eval/promote.py
    #     deciding a checkpoint becomes a shippable difficulty tier, see
    #     docs/ARCHITECTURE.md's "Difficulty-tier promotion" section)
    # ------------------------------------------------------------------
    out_path = Path(config["checkpoint_out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Wrote untrained checkpoint to {out_path} (board_size={config['board_size']}, seed={config['seed']})")
    print("This is NOT a trained net -- see docs/ROADMAP.md's Phase 2 for the real imitation-learning loop.")

    monitor.close()


if __name__ == "__main__":
    main()
