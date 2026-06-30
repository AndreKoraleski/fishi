"""Segmentation pipeline interface."""

from typing import Protocol

import numpy as np


class SegmentationPipeline(Protocol):
    """A segmentation model: a semantic label map from an image and per-id text prompts.

    The extension contract for a new model: set name and implement predict, then evaluate it on the
    canonical split with fishi.run.

    Batching is being actively considered, but for now the interface is one image at a time.
    """

    name: str

    def predict(self, image: np.ndarray, prompts: dict[int, str]) -> np.ndarray: ...
