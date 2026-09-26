"""Small convolutional driving policy for the first training experiment."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class DrivingPolicy(nn.Module):
    """Predict continuous steering, throttle, and brake controls from an image."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Pool to a small spatial grid, not a single point. Pooling all
            # the way to (1, 1) averages away WHERE in the image something
            # happened, keeping only THAT it happened. For obstacle
            # avoidance, position is the entire signal — an obstacle on the
            # left vs. right of frame can produce nearly identical per-
            # channel averages after (1, 1) pooling, even though the correct
            # steering response is opposite. (4, 4) keeps coarse
            # left/center/right and near/far position information intact.
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, images: Tensor) -> Tensor:
        raw = self.head(self.features(images))

        # Ranges are constrained to match CARLA's valid control ranges.
        # To avoid physically invalid predictions
        steer = torch.tanh(raw[:, 0:1])        # [-1, 1]
        throttle = torch.sigmoid(raw[:, 1:2])  # [0, 1]
        brake = torch.sigmoid(raw[:, 2:3])     # [0, 1]

        return torch.cat([steer, throttle, brake], dim=1)