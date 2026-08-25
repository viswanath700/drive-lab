"""Stitch captured frames into a short preview video, with logged
steer/throttle/brake/speed overlaid on each frame -- a fast visual sanity
check of the data pipeline before training.

Usage:
    python -m viz.stitch_video
    python -m viz.stitch_video --output viz/output/preview.mp4 --fps 20 --limit 300
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from sim.config import REPO_ROOT, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV.")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML.")
    parser.add_argument("--output", type=str, default="viz/output/preview.mp4", help="Output video path.")
    parser.add_argument("--fps", type=int, default=20, help="Playback frame rate of the output video.")
    parser.add_argument("--limit", type=int, default=None, help="Only stitch the first N frames.")
    return parser.parse_args()


def overlay_telemetry(frame, row: pd.Series):
    lines = [
        f"steer:    {row.steer:+.3f}",
        f"throttle: {row.throttle:.3f}",
        f"brake:    {row.brake:.3f}",
        f"speed:    {row.speed_kmh:.1f} km/h",
    ]
    for i, line in enumerate(lines):
        y = 25 + i * 22
        # Dark outline pass then bright fill pass keeps text legible over any frame background.
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest_path = Path(args.manifest) if args.manifest else REPO_ROOT / config["capture"]["manifest_path"]
    output_path = REPO_ROOT / args.output

    df = pd.read_csv(manifest_path).sort_values("timestamp").reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    if df.empty:
        raise SystemExit(f"No rows found in manifest: {manifest_path}")

    first_frame = cv2.imread(str(REPO_ROOT / df.iloc[0]["image_path"]))
    if first_frame is None:
        raise SystemExit(f"Could not read first frame: {df.iloc[0]['image_path']}")
    height, width = first_frame.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width, height))

    written = 0
    for _, row in df.iterrows():
        frame_path = REPO_ROOT / row["image_path"]
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Skipping missing frame: {frame_path}")
            continue
        frame = overlay_telemetry(frame, row)
        writer.write(frame)
        written += 1

    writer.release()
    print(f"Wrote {written} frames -> {output_path}")


if __name__ == "__main__":
    main()
