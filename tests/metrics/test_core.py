import numpy as np

from fishi.metrics import SegmentationMetrics


def test_perfect_prediction_scores_one():
    metric = SegmentationMetrics(class_count=3)
    target = np.array([[0, 1], [2, 1]])
    metric.update(target, target)
    result = metric.compute()
    assert result["miou"] == 1.0
    assert result["macc"] == 1.0


def test_known_iou_and_accuracy_values():
    metric = SegmentationMetrics(class_count=2)
    metric.update(np.array([0, 1, 1, 1]), np.array([0, 0, 1, 1]))
    result = metric.compute()
    np.testing.assert_allclose(result["iou"], [0.5, 2 / 3])
    np.testing.assert_allclose(result["accuracy"], [0.5, 1.0])  # recalls: class 0 half, class 1 all
    assert result["macc"] == 0.75


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


def test_accuracy_absent_class_is_nan():
    metric = SegmentationMetrics(class_count=3)
    metric.update(np.array([0, 1]), np.array([0, 1]))  # class 2 never appears
    assert np.isnan(metric.per_class_accuracy()[2])


def test_ignore_index_dropped_from_mean_accuracy():
    metric = SegmentationMetrics(class_count=3, ignore_index=0)
    metric.update(np.array([1, 0, 2]), np.array([1, 1, 2]))  # one class-1 pixel predicted as void
    result = metric.compute()
    assert np.isnan(np.asarray(result["accuracy"])[0])  # void excluded
    np.testing.assert_allclose(result["macc"], 0.75)  # recalls 0.5 (class 1) and 1.0 (class 2)


def test_empty_metric_is_nan_without_warning():
    import warnings

    metric = SegmentationMetrics(class_count=3, ignore_index=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the guard must prevent a mean-of-empty-slice warning
        result = metric.compute()
    assert np.isnan(result["miou"])
    assert np.isnan(result["macc"])
