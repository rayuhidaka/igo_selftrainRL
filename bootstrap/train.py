"""Trains a RayZeroNet checkpoint via imitation learning against self-play
examples (see selfplay/generate.py) -- standard AlphaZero-style loss:
cross-entropy between predicted and target policy distributions, plus MSE
between predicted and target (game-outcome) value.

If `config["data_source"]` is null, falls back to Phase 1's behavior: save
a fresh, untrained, deterministically-seeded checkpoint, purely to prove
the export/loading path (see docs/ROADMAP.md's Phase 1).

Usage:
    python -m bootstrap.train --config configs/bootstrap_base.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bootstrap.dataset import SelfPlayDataset, SelfPlayExamples
from bootstrap.model import RayZeroNet
from bootstrap.monitoring import TrainingMonitor

# checkpoints/runs/selfplay_games are symlinked to native WSL storage (not
# /mnt/c) specifically to avoid this, but a transient drvfs-style write
# failure is cheap to retry and expensive to lose a run over.
_SAVE_RETRIES = 3
_SAVE_RETRY_DELAY_SECONDS = 2.0


def _save_checkpoint(model: RayZeroNet, checkpoint_out: Path) -> None:
    for attempt in range(1, _SAVE_RETRIES + 1):
        try:
            torch.save(model.state_dict(), checkpoint_out)
            return
        except RuntimeError as e:
            if attempt == _SAVE_RETRIES:
                raise
            print(f"torch.save failed (attempt {attempt}/{_SAVE_RETRIES}): {e} -- retrying")
            time.sleep(_SAVE_RETRY_DELAY_SECONDS)


def train(model: RayZeroNet, config: dict, monitor: TrainingMonitor) -> None:
    examples = SelfPlayExamples.load(Path(config["data_source"]))
    dataset = SelfPlayDataset(examples)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    print(f"Training on {len(dataset)} examples ({len(loader)} batches/epoch, batch_size={config['batch_size']})")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    max_train_seconds = config.get("max_train_seconds")
    checkpoint_out = Path(config["checkpoint_out"])

    start = time.time()
    step = 0
    for epoch in range(config["epochs"]):
        for board_planes, policy_targets, value_targets in loader:
            predicted_policy, predicted_value = model(board_planes)

            # Cross-entropy against a soft (distribution, not single-label) policy target.
            policy_loss = -(policy_targets * torch.log(predicted_policy + 1e-8)).sum(dim=1).mean()
            value_loss = torch.nn.functional.mse_loss(predicted_value.squeeze(-1), value_targets)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % config["log_every_n_steps"] == 0:
                monitor.log_scalar("loss/policy", policy_loss.item(), step)
                monitor.log_scalar("loss/value", value_loss.item(), step)
                monitor.log_scalar("loss/total", loss.item(), step)

            if step % config["checkpoint_interval_steps"] == 0:
                _save_checkpoint(model, checkpoint_out)
                print(f"[step {step}] checkpointed to {checkpoint_out}")

            elapsed = time.time() - start
            if max_train_seconds is not None and elapsed > max_train_seconds:
                print(f"Hit max_train_seconds ({max_train_seconds}) at step {step}, stopping.")
                _save_checkpoint(model, checkpoint_out)
                print(f"Wrote final checkpoint to {checkpoint_out} after {step} steps, {elapsed:.1f}s")
                return

        print(f"Epoch {epoch + 1}/{config['epochs']} complete ({step} steps, {time.time() - start:.1f}s elapsed)")

    _save_checkpoint(model, checkpoint_out)
    print(f"Wrote final checkpoint to {checkpoint_out} after {step} steps, {time.time() - start:.1f}s")


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    monitor = TrainingMonitor(log_dir=Path(config["log_dir"]), run_name=config["run_name"])
    monitor.log_config(config)

    torch.manual_seed(config["seed"])
    model = RayZeroNet(board_size=config["board_size"])

    if config.get("data_source"):
        train(model, config, monitor)
    else:
        # Phase 1 fallback: no data to train on yet, just prove the export/loading path
        # with a fresh, untrained checkpoint -- see docs/ROADMAP.md's Phase 1.
        out_path = Path(config["checkpoint_out"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
        print(f"Wrote untrained checkpoint to {out_path} (board_size={config['board_size']}, seed={config['seed']})")
        print("No data_source configured -- see docs/ROADMAP.md's Phase 2 for real training.")

    monitor.close()


if __name__ == "__main__":
    main()
