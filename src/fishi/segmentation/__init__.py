"""Open-vocabulary segmentation pipelines (text prompts -> semantic masks)."""

from fishi.segmentation.base import SegmentationPipeline, match_label, semantic_from_instances
from fishi.segmentation.grounded_sam import GroundedSam1, GroundedSam2
from fishi.segmentation.openworldsam import OpenWorldSam

__all__ = [
    "GroundedSam1",
    "GroundedSam2",
    "OpenWorldSam",
    "SegmentationPipeline",
    "match_label",
    "semantic_from_instances",
]
