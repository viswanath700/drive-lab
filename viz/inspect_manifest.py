"""Sanity-check summary of the captured manifest: row/scenario counts,
control-value statistics, and a steer-angle histogram. The histogram matters
because autopilot-driven data is usually dominated by near-zero steer (driving
straight), which can bias a model trained on it if left unaddressed later.

Usage:
    python -m viz.inspect_manifest
    python -m viz.inspect_manifest --manifest data/manifest.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sim.config import REPO_ROOT, load_config

STEER_BINS = [-1.0, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV.")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML.")
    return parser.parse_args()


def print_steer_histogram(steer: pd.Series) -> None:
    counts = pd.cut(steer, bins=STEER_BINS).value_counts().sort_index()
    total = len(steer)
    print("\nSteer distribution:")
    for interval, count in counts.items():
        bar = "#" * int(50 * count / total) if total else ""
        print(f"  {str(interval):>18}: {count:5d} {bar}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest_path = Path(args.manifest) if args.manifest else REPO_ROOT / config["capture"]["manifest_path"]

    df = pd.read_csv(manifest_path)

    print(f"Manifest: {manifest_path}")
    print(f"Total frames: {len(df)}")

    print("\nFrames per scenario_id / split:")
    print(df.groupby(["scenario_id", "split"]).size().to_string())

    print("\nControl/speed statistics:")
    print(df[["steer", "throttle", "brake", "speed_kmh"]].describe().to_string())

    print_steer_histogram(df["steer"])


if __name__ == "__main__":
    main()
