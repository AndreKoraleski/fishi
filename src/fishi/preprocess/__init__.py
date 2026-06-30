"""Paired image processors (preprocess + postprocess) and dataset wrapping.

The geometric processors (Rectify, Patches, TangentImages) reproject through each image's
calibration, so they are camera-specific. Identity is the camera-agnostic baseline.
"""

from fishi.preprocess.base import ProcessedDataset, ProcessedSample, Processor
from fishi.preprocess.gnomonic import GnomonicMultiView
from fishi.preprocess.identity import Identity
from fishi.preprocess.patches import Patches
from fishi.preprocess.rectify import Rectify
from fishi.preprocess.tangent import TangentImages

__all__ = [
    "GnomonicMultiView",
    "Identity",
    "Patches",
    "ProcessedDataset",
    "ProcessedSample",
    "Processor",
    "Rectify",
    "TangentImages",
]
