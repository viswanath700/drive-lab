"""PyTorch dataset for camera images and recorded driving controls."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, PILToTensor, Resize

from sim.config import REPO_ROOT

logger = logging.getLogger("drivelab.training.dataset")

CONTROL_COLUMNS = ["steer", "throttle", "brake"]

# Fractional tolerance used when checking that the requested image_size
# preserves the source images' aspect ratio (see _check_aspect_ratio).
_ASPECT_RATIO_TOLERANCE = 0.02


class DrivingDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load image/control pairs from the manifest for one data split.

    Images are returned as float tensors shaped ``(channels, height, width)``
    with values normalized around zero. Controls are returned as a float tensor
    shaped ``(3,)`` in steer/throttle/brake order.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        image_size: tuple[int, int] = (192, 256),
        scenario_ids: list[str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_absolute():
            self.manifest_path = REPO_ROOT / self.manifest_path

        manifest = pd.read_csv(self.manifest_path)
        required_columns = {"image_path", "split", *CONTROL_COLUMNS}
        missing_columns = required_columns.difference(manifest.columns)
        if missing_columns:
            raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

        self.records = manifest[manifest["split"] == split]
        if scenario_ids is not None:
            self.records = self.records[self.records["scenario_id"].isin(scenario_ids)]
        self.records = self.records.reset_index(drop=True)
        if self.records.empty:
            raise ValueError(f"No records found for split={split!r} in {self.manifest_path}")

        self.image_size = image_size
        self._check_aspect_ratio()

        # NOTE: Resize() stretches to the exact (height, width) given below —
        # it does NOT preserve aspect ratio on its own. This currently looks
        # correct only because the source camera is configured at 800x600
        # (4:3), matching image_size's 256x192 (also 4:3). If the camera's
        # width/height/FOV changes in config.yaml (e.g. for a new obstacle
        # scenario), update image_size to match the new aspect ratio, or
        # images will be silently stretched/squashed with no error.
        self.transform = Compose(
            [
                Resize(image_size),
                PILToTensor(),
                ConvertImageDtype(torch.float32),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def _check_aspect_ratio(self) -> None:
        """Warn (don't fail) if image_size's aspect ratio doesn't match the
        first sample's source image aspect ratio, since Resize() would
        silently distort mismatched images rather than raising an error."""
        sample_path = REPO_ROOT / str(self.records.iloc[0]["image_path"])
        try:
            with Image.open(sample_path) as sample_image:
                source_w, source_h = sample_image.size
        except FileNotFoundError:
            logger.warning("Could not open sample image %s for aspect-ratio check.", sample_path)
            return

        target_h, target_w = self.image_size
        source_ratio = source_w / source_h
        target_ratio = target_w / target_h
        if abs(source_ratio - target_ratio) > _ASPECT_RATIO_TOLERANCE:
            logger.warning(
                "image_size aspect ratio (%.3f) does not match source images "
                "(%.3f, from %s). Resize() will distort images — update "
                "image_size or the camera config to match.",
                target_ratio,
                source_ratio,
                sample_path,
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        record = self.records.iloc[index]
        image_path = REPO_ROOT / str(record["image_path"])
        try:
            with Image.open(image_path) as image:
                image_tensor = cast(Tensor, self.transform(image.convert("RGB")))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Manifest row {index} references missing image: {image_path}. "
                "Check for a crashed/incomplete capture session or a manually "
                "deleted frame."
            ) from exc

        controls = torch.tensor(record[CONTROL_COLUMNS].to_numpy(dtype="float32"))
        return image_tensor, controls

    # TODO: if adding horizontal-flip augmentation later (a natural way to
    # double obstacle-avoidance examples), remember to also negate `steer`
    # on the label for flipped samples. Flipping only the image inverts the
    # left/right semantics without updating the label, which silently trains
    # the model on contradictory steer targets for visually mirrored scenes.