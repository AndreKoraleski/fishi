"""Paired image processors (preprocess + postprocess) and dataset wrapping."""

from fishi.preprocess.base import ProcessedDataset, ProcessedSample, Processor
from fishi.preprocess.identity import Identity
from fishi.preprocess.patches import Patches
from fishi.preprocess.rectify import Rectify

__all__ = ["Identity", "Patches", "ProcessedDataset", "ProcessedSample", "Processor", "Rectify"]
