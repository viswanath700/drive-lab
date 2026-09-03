"""Run a small supervised-learning smoke test on captured driving data.

This validates the data loader, CNN forward pass, loss calculation, gradient
descent, and checkpoint writing. It is not an obstacle-avoidance benchmark.

Usage:
    python -m training.train --epochs 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from sim.config import REPO_ROOT
from training.dataset import DrivingDataset
from training.model import DrivingPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint", default="training/checkpoints/smoke_test.pt")
    return parser.parse_args()


def build_datasets(manifest_path: Path) -> tuple[DrivingDataset, DrivingDataset]:
    manifest = pd.read_csv(manifest_path)
    scenario_ids = sorted(manifest["scenario_id"].unique())
    if len(scenario_ids) < 2:
        raise ValueError("Need at least two scenario_ids for a scenario-held-out validation set.")

    validation_scenarios = [scenario_ids[-1]]
    training_scenarios = scenario_ids[:-1]
    return (
        DrivingDataset(manifest_path, "train", scenario_ids=training_scenarios),
        DrivingDataset(manifest_path, "train", scenario_ids=validation_scenarios),
    )


def run_epoch(
    model: DrivingPolicy,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            predictions = model(images)
            loss = loss_function(predictions, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(images)
            total_samples += len(images)

    return total_loss / total_samples


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path

    train_dataset, validation_dataset = build_datasets(manifest_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DrivingPolicy().to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, loss_function, optimizer, device)
        validation_loss = run_epoch(model, validation_loader, loss_function, None, device)
        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.6f} validation_loss={validation_loss:.6f}"
        )

    checkpoint_path = REPO_ROOT / args.checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()