"""Paired image processors: preprocess (input) + postprocess (output)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from fishi.woodscape.calibration import Calibration
from fishi.woodscape.dataset import Sample


class SampleSource(Protocol):
    """Any dataset that yields Samples by index."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Sample: ...

    def stem(self, index: int) -> str: ...

    def label(self, index: int) -> np.ndarray: ...


@dataclass
class ProcessedSample:
    """A dataset sample after preprocessing.

    Attributes
    ----------
    views : list of np.ndarray
        Model-input images: one for None/Rectify, several for Patches.
    label : np.ndarray
        Class-id mask in the original fisheye space.
    calibration : Calibration
        Calibration of the source image.
    stem : str
        Filename stem.
    camera : str
        Camera id.
    """

    views: list[np.ndarray]
    label: np.ndarray
    calibration: Calibration
    stem: str
    camera: str


class Processor(ABC):
    """A paired image processor: preprocess and postprocess."""

    name: str

    @abstractmethod
    def preprocess(self, image: np.ndarray, calibration: Calibration) -> list[np.ndarray]:
        """Map a fisheye image to one or more model-input views."""

    @abstractmethod
    def postprocess(self, predictions: list[np.ndarray], calibration: Calibration) -> np.ndarray:
        """Merge per-view predictions back into one fisheye-space mask."""

    def wrap(self, dataset: SampleSource) -> "ProcessedDataset":
        """Return a dataset that applies this processor to every sample."""
        return ProcessedDataset(dataset, self)


class ProcessedDataset:
    """A dataset view that preprocesses each sample on access."""

    def __init__(self, dataset: SampleSource, processor: Processor) -> None:
        self.dataset = dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> ProcessedSample:
        sample = self.dataset[index]
        views = self.processor.preprocess(sample.image, sample.calibration)
        return ProcessedSample(
            views=views,
            label=sample.label,
            calibration=sample.calibration,
            stem=sample.stem,
            camera=sample.camera,
        )
