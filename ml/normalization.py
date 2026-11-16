"""Lightweight preprocessing shared by model training and inference."""

from __future__ import annotations

import numpy as np

from app.config import PATCH_CHANNELS


def normalize_patch(image: np.ndarray) -> np.ndarray:
    """Map reflectance and NDVI ranges into stable 0..1 model inputs."""
    if image.shape[0] != len(PATCH_CHANNELS):
        raise ValueError(f"Expected {len(PATCH_CHANNELS)} channels, received {image.shape[0]}.")
    normalized = image.astype(np.float32, copy=True)
    normalized[:6] = np.clip((normalized[:6] + 0.2) / 1.2, 0.0, 1.0)
    normalized[6] = np.clip((normalized[6] + 1.0) / 2.0, 0.0, 1.0)
    return normalized
