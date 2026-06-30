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


def test_ignore_index_class_dropped_from_mean():
    # void (0) predicted as background must not earn a spurious 0 IoU that drags mIoU down.
    metric = SegmentationMetrics(class_count=3, ignore_index=0)
    metric.update(np.array([1, 0, 2]), np.array([1, 1, 2]))  # one class-1 pixel predicted as 0
    result = metric.compute()
    iou = np.asarray(result["iou"])
    assert np.isnan(iou[0])  # void excluded from the per-class report
    np.testing.assert_allclose(result["miou"], 0.75)  # mean over classes 1 (0.5) and 2 (1.0)


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
