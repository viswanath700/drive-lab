"""Run a small supervised-learning smoke test on captured driving data.

This validates the data loader, CNN forward pass, loss calculation, gradient
descent, and checkpoint writing. It is not an obstacle-avoidance benchmark.

Usage:
python -m training.train --epochs 1
python -m training.train --epochs 5 --validation-scenarios obstacle_cone_session1
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

SEED = 42
CONTROL_NAMES = ("steer", "throttle", "brake")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint", default="training/checkpoints/smoke_test.pt")
    parser.add_argument(
        "--validation-scenarios",
        nargs="+",
        default=None,
        help=(
            "Scenario IDs to hold out for validation instead of using the manifest's "
            "split column. Ignores the split column entirely (pools train+val rows "
            "for the listed scenarios vs. the rest) — use for ad hoc holdouts, e.g. "
            "validating generalization to one held-out obstacle scenario."
        ),
    )
    return parser.parse_args()


def build_datasets(
    manifest_path: Path, validation_scenarios: list[str] | None
) -> tuple[DrivingDataset, DrivingDataset]:
    if validation_scenarios is None:
        # Default: trust the manifest's own split column (set per capture session).
        return (
            DrivingDataset(manifest_path, "train"),
            DrivingDataset(manifest_path, "val"),
        )

    manifest = pd.read_csv(manifest_path)
    scenario_ids = sorted(manifest["scenario_id"].unique())
    unknown = set(validation_scenarios) - set(scenario_ids)
    if unknown:
        raise ValueError(f"Unknown validation scenario_id(s): {sorted(unknown)}")

    training_scenarios = [s for s in scenario_ids if s not in validation_scenarios]
    if not training_scenarios:
        raise ValueError("No scenario_ids left for training after removing validation scenarios.")

    return (
        DrivingDataset(manifest_path, split=None, scenario_ids=training_scenarios),
        DrivingDataset(manifest_path, split=None, scenario_ids=validation_scenarios),
    )


def run_epoch(
    model: DrivingPolicy,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, tuple[float, float, float]]:
    """Run one pass over `loader`. Returns (combined_loss, (steer_loss, throttle_loss, brake_loss)).

    Per-control losses are reported separately from the combined MSE because
    a single blended number can look fine while the model is specifically
    bad at steering — the output that matters most for obstacle avoidance.
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    per_control_loss = torch.zeros(3, device=device)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in loader:
            images = images.to(device) # move this batch's images to GPU
            targets = targets.to(device) # move this batch's labels to GPU
            predictions = model(images) # forward pass: (32,3,192,256) -> (32,3)
            loss = loss_function(predictions, targets) # compare predicted vs. recorded controls
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = len(images)
            total_loss += loss.item() * batch_size
            per_control_loss += ((predictions - targets) ** 2).mean(dim=0).detach() * batch_size
            total_samples += batch_size

    combined = total_loss / total_samples
    per_control = tuple((per_control_loss / total_samples).tolist())
    return combined, per_control  # type: ignore[return-value]


def format_per_control(per_control: tuple[float, float, float]) -> str:
    return " ".join(f"{name}_loss={value:.6f}" for name, value in zip(CONTROL_NAMES, per_control))


def main() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path

    train_dataset, validation_dataset = build_datasets(manifest_path, args.validation_scenarios)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
    )

    model = DrivingPolicy().to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

    checkpoint_path = REPO_ROOT / args.checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss, train_per_control = run_epoch(model, train_loader, loss_function, optimizer, device)
        validation_loss, val_per_control = run_epoch(model, validation_loader, loss_function, None, device)

        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.6f} ({format_per_control(train_per_control)}) "
            f"validation_loss={validation_loss:.6f} ({format_per_control(val_per_control)})"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "train_per_control_loss": dict(zip(CONTROL_NAMES, train_per_control)),
                    "validation_per_control_loss": dict(zip(CONTROL_NAMES, val_per_control)),
                },
                checkpoint_path,
            )
            print(f"  New best validation_loss — saved checkpoint: {checkpoint_path}")

    print(f"Training complete. Best validation_loss={best_validation_loss:.6f} -> {checkpoint_path}")


if __name__ == "__main__":
    main()