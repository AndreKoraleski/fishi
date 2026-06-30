"""Diagnostic metrics derived from a SegmentationMetrics confusion matrix.

Pixel and per-class accuracy, frequency-weighted IoU, class-group means, and the error breakdown
(correct vs confused vs missed) that separates classification error from recall. Each takes a
SegmentationMetrics and reads its confusion matrix.
"""

import numpy as np

from fishi.metrics.core import SegmentationMetrics


def _scored(metrics: SegmentationMetrics) -> np.ndarray:
    """Boolean mask of class ids included in scoring (all but ignore_index)."""
    mask = np.ones(metrics.class_count, dtype=bool)
    if metrics.ignore_index is not None and 0 <= metrics.ignore_index < metrics.class_count:
        mask[metrics.ignore_index] = False
    return mask


def pixel_accuracy(metrics: SegmentationMetrics) -> float:
    """Fraction of scored pixels predicted correctly."""
    total = float(metrics.confusion.sum())
    return float(np.diag(metrics.confusion).sum() / total) if total else float("nan")


def per_class_accuracy(metrics: SegmentationMetrics) -> np.ndarray:
    """Per-class recall (correct over target pixels). NaN for classes absent from the target."""
    support = metrics.confusion.sum(axis=1).astype(np.float64)
    correct = np.diag(metrics.confusion).astype(np.float64)
    return np.divide(correct, support, out=np.full_like(correct, np.nan), where=support > 0)


def mean_accuracy(metrics: SegmentationMetrics) -> float:
    """Mean per-class recall over present scored classes."""
    accuracy = per_class_accuracy(metrics)
    accuracy[~_scored(metrics)] = np.nan
    return float(np.nanmean(accuracy))


def frequency_weighted_iou(metrics: SegmentationMetrics) -> float:
    """Per-class IoU weighted by each class's target-pixel frequency (scored classes only)."""
    iou = metrics.per_class_iou()
    support = metrics.confusion.sum(axis=1).astype(np.float64)
    present = _scored(metrics) & ~np.isnan(iou) & (support > 0)
    if not present.any():
        return float("nan")
    weight = support[present] / support[present].sum()
    return float(np.sum(weight * iou[present]))


def grouped_miou(metrics: SegmentationMetrics, groups: dict[str, list[int]]) -> dict[str, float]:
    """Mean IoU within each named group of class ids (e.g. things vs stuff)."""
    iou = metrics.per_class_iou()
    return {
        name: float(np.nanmean([iou[class_id] for class_id in ids])) if ids else float("nan")
        for name, ids in groups.items()
    }


def error_breakdown(metrics: SegmentationMetrics) -> dict[str, float]:
    """Split scored target pixels into correct, confused, and missed fractions.

    correct predicts the right class. confused predicts another scored class (a semantic error).
    missed predicts the ignore/background class (a recall failure). The three sum to one, so
    confused versus missed separates classification error from recall.
    """
    scored = _scored(metrics)
    scored_rows = metrics.confusion[scored]
    total = float(scored_rows.sum())
    if not total:
        return {"correct": float("nan"), "confused": float("nan"), "missed": float("nan")}
    correct = float(np.diag(metrics.confusion)[scored].sum())
    missed = 0.0
    if metrics.ignore_index is not None and 0 <= metrics.ignore_index < metrics.class_count:
        missed = float(scored_rows[:, metrics.ignore_index].sum())
    confused = total - correct - missed
    return {"correct": correct / total, "confused": confused / total, "missed": missed / total}
