"""PyTorch dataset for camera images and recorded driving controls."""
from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, PILToTensor, Resize

from sim.config import REPO_ROOT

CONTROL_COLUMNS = ["steer", "throttle", "brake"]


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

        self.transform = Compose(
            [
                Resize(image_size),
                PILToTensor(),
                ConvertImageDtype(torch.float32),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        record = self.records.iloc[index]
        image_path = REPO_ROOT / str(record["image_path"])
        with Image.open(image_path) as image:
            image_tensor = cast(Tensor, self.transform(image.convert("RGB")))

        controls = torch.tensor(record[CONTROL_COLUMNS].to_numpy(dtype="float32"))
        return image_tensor, controls