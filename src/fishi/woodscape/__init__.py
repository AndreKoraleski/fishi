"""WoodScape: dataset access, calibration, splits, taxonomy, and download."""

from fishi.woodscape import classes
from fishi.woodscape.calibration import Calibration
from fishi.woodscape.config import Settings, get_settings
from fishi.woodscape.dataset import Sample, Subset, WoodScapeDataset
from fishi.woodscape.download import download_woodscape
from fishi.woodscape.splits import (
    canonical_split,
    load_split,
    make_split,
    split_datasets,
)

__all__ = [
    "Calibration",
    "Sample",
    "Settings",
    "Subset",
    "WoodScapeDataset",
    "canonical_split",
    "classes",
    "download_woodscape",
    "get_settings",
    "load_split",
    "make_split",
    "split_datasets",
]
