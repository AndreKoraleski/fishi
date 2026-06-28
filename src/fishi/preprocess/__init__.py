"""Paired image processors (preprocess + postprocess) and dataset wrapping."""

from fishi.preprocess.base import ProcessedDataset, ProcessedSample, Processor
from fishi.preprocess.identity import Identity

__all__ = ["Identity", "ProcessedDataset", "ProcessedSample", "Processor"]
