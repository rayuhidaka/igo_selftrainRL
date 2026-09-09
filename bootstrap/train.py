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

from bootstrap.checkpoint import CheckpointMetadata, build_model, load_checkpoint, save_checkpoint
from bootstrap.dataset import SelfPlayDataset, SelfPlayExamples, build_training_loader
from bootstrap.model import RayZeroNet
from bootstrap.monitoring import TrainingMonitor

# checkpoints/runs/selfplay_games are symlinked to native WSL storage (not
# /mnt/c) specifically to avoid this, but a transient drvfs-style write
# failure is cheap to retry and expensive to lose a run over.
_SAVE_RETRIES = 3
_SAVE_RETRY_DELAY_SECONDS = 2.0


def _save_checkpoint(model: RayZeroNet, metadata: CheckpointMetadata, checkpoint_out: Path) -> None:
    for attempt in range(1, _SAVE_RETRIES + 1):
        try:
            save_checkpoint(model, metadata, checkpoint_out)
            return
        except RuntimeError as e:
            if attempt == _SAVE_RETRIES:
                raise
            print(f"torch.save failed (attempt {attempt}/{_SAVE_RETRIES}): {e} -- retrying")
            time.sleep(_SAVE_RETRY_DELAY_SECONDS)


def _data_source_summary(data_source) -> str | None:
    """`config["data_source"]` is either a single path (plain, unweighted training) or a
    list of `{path, weight}` dicts (mixed-source training, see `build_loader`) --
    normalizes either into one descriptive string for `CheckpointMetadata`.
    """
    if data_source is None:
        return None
    if isinstance(data_source, list):
        return "+".join(f"{item['path']}(w={item['weight']})" for item in data_source)
    return data_source


def _metadata_from_model_and_config(model: RayZeroNet, config: dict) -> CheckpointMetadata:
    # Reads the architecture back off the actual model rather than re-deriving it from
    # config, so a warm-started run's metadata is correct even when it inherited its
    # architecture from init_from_checkpoint's own metadata rather than this config.
    return CheckpointMetadata(
        board_size=model.board_size,
        channels=model.channels,
        num_conv_layers=model.num_conv_layers,
        num_residual_blocks=model.num_residual_blocks,
        has_score_head=model.has_score_head,
        data_source=_data_source_summary(config.get("data_source")),
        seed=config["seed"],
        init_from_checkpoint=config.get("init_from_checkpoint"),
    )


def _build_loader(config: dict) -> tuple[DataLoader, int]:
    """`config["data_source"]` is either a single path (plain, unweighted training -- every
    config before Phase 3) or a list of `{path, weight}` dicts (mixed-source training, see
    `bootstrap.dataset.build_training_loader` for why: anchoring a self-play fine-tune
    against a broader, known-good dataset alongside the new small batch prevents the
    self-play collapse documented in docs/ROADMAP.md's Phase 3).
    """
    data_source = config["data_source"]
    if isinstance(data_source, list):
        sources = [(Path(item["path"]), item["weight"]) for item in data_source]
        return build_training_loader(sources, config["batch_size"])
    examples = SelfPlayExamples.load(Path(data_source))
    dataset = SelfPlayDataset(examples)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    return loader, len(dataset)


def _build_model(config: dict) -> RayZeroNet:
    init_from_checkpoint = config.get("init_from_checkpoint")
    if init_from_checkpoint is None:
        return RayZeroNet(
            board_size=config["board_size"],
            channels=config.get("channels", 64),
            num_conv_layers=config.get("num_conv_layers", 3),
            num_residual_blocks=config.get("num_residual_blocks", 0),
            has_score_head=config.get("has_score_head", False),
        )
    # Warm start (Phase 3's self-play fine-tuning, see docs/ROADMAP.md): continue training
    # an existing checkpoint instead of a fresh, randomly-initialized network. channels/
    # num_conv_layers/num_residual_blocks/has_score_head in config only need setting if
    # init_from_checkpoint is a legacy checkpoint with no recorded architecture -- see
    # build_model's docstring. Loading with strict=False since adding has_score_head to an
    # existing (non-score-head) checkpoint means the new score_fc1/score_fc2 weights are
    # intentionally absent from the source state_dict -- they start freshly initialized.
    state_dict, metadata = load_checkpoint(Path(init_from_checkpoint))
    model = build_model(
        metadata,
        config["board_size"],
        channels=config.get("channels"),
        num_conv_layers=config.get("num_conv_layers"),
        num_residual_blocks=config.get("num_residual_blocks"),
        has_score_head=config.get("has_score_head"),
    )
    model.load_state_dict(state_dict, strict=False)
    return model


def resolve_device(config: dict) -> torch.device:
    """`config["device"]` is `"auto"` (default, prefer CUDA when available -- this pipeline's
    self-play/eval stay CPU-only by design, see selfplay/self_play.py's module docstring on
    why single-position MCTS inference doesn't benefit from a GPU; only this batched training
    step does), or an explicit `"cpu"`/`"cuda"` override.
    """
    requested = config.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train(model: RayZeroNet, config: dict, monitor: TrainingMonitor, device: torch.device) -> None:
    loader, dataset_size = _build_loader(config)
    print(f"Training on {dataset_size} examples ({len(loader)} batches/epoch, batch_size={config['batch_size']}) on {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    max_train_seconds = config.get("max_train_seconds")
    checkpoint_out = Path(config["checkpoint_out"])
    metadata = _metadata_from_model_and_config(model, config)
    # Kept low relative to policy/value -- this is meant to regularize the value head
    # against saturating flat near +-1 (see bootstrap/model.py's module docstring and
    # docs/SELF_PLAY_STABILITY.md), not to make the network chase score margin the way
    # AlphaGo found actively harmful when done directly.
    score_loss_weight = config.get("score_loss_weight", 0.15)

    start = time.time()
    step = 0
    for epoch in range(config["epochs"]):
        for board_planes, policy_targets, value_targets, score_margin_targets in loader:
            board_planes = board_planes.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            score_margin_targets = score_margin_targets.to(device)

            predicted_policy, predicted_value, predicted_score = model(board_planes)

            # Cross-entropy against a soft (distribution, not single-label) policy target.
            policy_loss = -(policy_targets * torch.log(predicted_policy + 1e-8)).sum(dim=1).mean()
            value_loss = torch.nn.functional.mse_loss(predicted_value.squeeze(-1), value_targets)
            loss = policy_loss + value_loss

            score_loss = None
            if predicted_score is not None:
                score_loss = torch.nn.functional.mse_loss(predicted_score.squeeze(-1), score_margin_targets)
                loss = loss + score_loss_weight * score_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % config["log_every_n_steps"] == 0:
                monitor.log_scalar("loss/policy", policy_loss.item(), step)
                monitor.log_scalar("loss/value", value_loss.item(), step)
                if score_loss is not None:
                    monitor.log_scalar("loss/score_margin", score_loss.item(), step)
                monitor.log_scalar("loss/total", loss.item(), step)

            if step % config["checkpoint_interval_steps"] == 0:
                _save_checkpoint(model, metadata, checkpoint_out)
                print(f"[step {step}] checkpointed to {checkpoint_out}")

            elapsed = time.time() - start
            if max_train_seconds is not None and elapsed > max_train_seconds:
                print(f"Hit max_train_seconds ({max_train_seconds}) at step {step}, stopping.")
                _save_checkpoint(model, metadata, checkpoint_out)
                print(f"Wrote final checkpoint to {checkpoint_out} after {step} steps, {elapsed:.1f}s")
                return

        print(f"Epoch {epoch + 1}/{config['epochs']} complete ({step} steps, {time.time() - start:.1f}s elapsed)")

    _save_checkpoint(model, metadata, checkpoint_out)
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
    model = _build_model(config)

    if config.get("data_source"):
        device = resolve_device(config)
        model = model.to(device)
        train(model, config, monitor, device)
    else:
        # Phase 1 fallback: no data to train on yet, just prove the export/loading path
        # with a fresh, untrained checkpoint -- see docs/ROADMAP.md's Phase 1.
        out_path = Path(config["checkpoint_out"])
        save_checkpoint(model, _metadata_from_model_and_config(model, config), out_path)
        print(f"Wrote untrained checkpoint to {out_path} (board_size={config['board_size']}, seed={config['seed']})")
        print("No data_source configured -- see docs/ROADMAP.md's Phase 2 for real training.")

    monitor.close()


if __name__ == "__main__":
    main()
