import numpy as np

from fishi.metrics import (
    SegmentationMetrics,
    error_breakdown,
    frequency_weighted_iou,
    grouped_miou,
)


def test_frequency_weighted_iou_weights_by_support():
    metric = SegmentationMetrics(class_count=2)
    metric.update(np.array([0, 1, 1, 1]), np.array([0, 0, 1, 1]))
    np.testing.assert_allclose(frequency_weighted_iou(metric), (0.5 + 2 / 3) / 2)  # equal support


def test_error_breakdown_splits_confusion_and_miss():
    metric = SegmentationMetrics(class_count=3, ignore_index=0)  # 0 is background
    metric.update(np.array([1, 0, 2, 1]), np.array([1, 1, 2, 2]))
    errors = error_breakdown(metric)
    np.testing.assert_allclose(errors["correct"], 0.5)  # 1->1 and 2->2
    np.testing.assert_allclose(errors["missed"], 0.25)  # 1->0, predicted background
    np.testing.assert_allclose(errors["confused"], 0.25)  # 2->1, wrong real class


def test_grouped_miou_averages_within_groups():
    metric = SegmentationMetrics(class_count=3)
    metric.update(np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert grouped_miou(metric, {"stuff": [0, 1], "things": [2]}) == {"stuff": 1.0, "things": 1.0}


def test_frequency_weighted_iou_empty_is_nan():
    assert np.isnan(frequency_weighted_iou(SegmentationMetrics(class_count=3, ignore_index=0)))


def test_grouped_miou_empty_group_is_nan():
    metric = SegmentationMetrics(class_count=3)
    metric.update(np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert np.isnan(grouped_miou(metric, {"empty": []})["empty"])
