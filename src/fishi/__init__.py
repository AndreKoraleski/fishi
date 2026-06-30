"""fishi: a zero-shot segmentation benchmark on the fisheye WoodScape dataset.

Public API. The two extension points are the Processor and SegmentationPipeline protocols: bring
your own preprocessing or model, evaluate it on the canonical split with run() (or score() for
predictions made elsewhere), and get numbers comparable to ours. The heavy model dependencies
(torch, transformers) load lazily, so importing fishi needs only the core libraries.
"""

from fishi.evaluation import evaluate, run, score
from fishi.metrics import SegmentationMetrics
from fishi.preprocess import Identity, Patches, Processor, Rectify, TangentImages
from fishi.segmentation import (
    GroundedSam1,
    GroundedSam2,
    OpenWorldSam,
    SamThree,
    SegmentationPipeline,
)
from fishi.sweep import sweep
from fishi.woodscape import WoodScapeDataset, get_settings, load_split

__version__ = "0.1.0"

__all__ = [
    "GroundedSam1",
    "GroundedSam2",
    "Identity",
    "OpenWorldSam",
    "Patches",
    "Processor",
    "Rectify",
    "SamThree",
    "SegmentationMetrics",
    "SegmentationPipeline",
    "TangentImages",
    "WoodScapeDataset",
    "evaluate",
    "get_settings",
    "load_split",
    "run",
    "score",
    "sweep",
]
