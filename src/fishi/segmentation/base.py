"""Segmentation pipeline interface."""

from typing import Protocol

import numpy as np


class SegmentationPipeline(Protocol):
    """Produces a semantic label map from an image and per-id text prompts."""

    name: str

    def predict(self, image: np.ndarray, prompts: dict[int, str]) -> np.ndarray: ...

    def predict_batch(
        self, images: list[np.ndarray], prompts: dict[int, str]
    ) -> list[np.ndarray]: ...
