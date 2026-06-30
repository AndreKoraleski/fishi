"""Segmentation metrics: the confusion-matrix core and its diagnostics."""

from fishi.metrics.core import SegmentationMetrics
from fishi.metrics.diagnostics import (
    error_breakdown,
    frequency_weighted_iou,
    grouped_miou,
    mean_accuracy,
    per_class_accuracy,
    pixel_accuracy,
)

__all__ = [
    "SegmentationMetrics",
    "error_breakdown",
    "frequency_weighted_iou",
    "grouped_miou",
    "mean_accuracy",
    "per_class_accuracy",
    "pixel_accuracy",
]
