"""Segmentation pipeline interface."""

from typing import Protocol

import numpy as np


class SegmentationPipeline(Protocol):
    """Produces a semantic label map from an image and per-id text prompts, one image at a time."""

    name: str

    def predict(self, image: np.ndarray, prompts: dict[int, str]) -> np.ndarray: ...
