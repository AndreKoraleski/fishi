"""Open-vocabulary segmentation pipelines (text prompts -> semantic masks)."""

from fishi.segmentation.base import SegmentationPipeline
from fishi.segmentation.grounded_sam import GroundedSam1, GroundedSam2
from fishi.segmentation.openworldsam import OpenWorldSam
from fishi.segmentation.sam3 import SamThree
from fishi.segmentation.semantic import match_label, semantic_from_instances

__all__ = [
    "GroundedSam1",
    "GroundedSam2",
    "OpenWorldSam",
    "SamThree",
    "SegmentationPipeline",
    "match_label",
    "semantic_from_instances",
]
