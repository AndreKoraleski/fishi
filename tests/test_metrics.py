import numpy as np

from fishi.metrics import SegmentationMetrics


def test_perfect_prediction_scores_one():
    metric = SegmentationMetrics(class_count=3)
    target = np.array([[0, 1], [2, 1]])
    metric.update(target, target)
    result = metric.compute()
    assert result["miou"] == 1.0
    assert result["mdice"] == 1.0


def test_known_iou_and_dice_values():
    metric = SegmentationMetrics(class_count=2)
    metric.update(np.array([0, 1, 1, 1]), np.array([0, 0, 1, 1]))
    result = metric.compute()
    np.testing.assert_allclose(result["iou"], [0.5, 2 / 3])
    np.testing.assert_allclose(result["dice"], [2 / 3, 0.8])


def test_ignore_index_excluded():
    metric = SegmentationMetrics(class_count=2, ignore_index=255)
    predictions = np.array([0, 1, 0])
    target = np.array([0, 1, 255])
    metric.update(predictions, target)
    result = metric.compute()
    assert result["miou"] == 1.0


def test_absent_class_is_nan_not_zero():
    metric = SegmentationMetrics(class_count=3)
    metric.update(np.array([0, 1]), np.array([0, 1]))
    iou = metric.per_class_iou()
    assert np.isnan(iou[2])
    assert metric.compute()["miou"] == 1.0


def test_reset_clears_accumulation():
    metric = SegmentationMetrics(class_count=2)
    metric.update(np.array([0, 1]), np.array([0, 1]))
    metric.reset()
    assert metric.confusion.sum() == 0
