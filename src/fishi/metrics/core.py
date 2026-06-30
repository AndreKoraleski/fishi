"""Segmentation metrics core: per-class and mean IoU and accuracy."""

import numpy as np


class SegmentationMetrics:
    """Accumulate predictions and report IoU and accuracy, per class and mean.

    A single confusion matrix over the classes backs both metrics, so IoU and accuracy stay
    consistent. Call update() per sample, then compute(). The confusion matrix is public, so the
    diagnostics module derives its richer views (frequency-weighted IoU, error breakdown) from it.

    Parameters
    ----------
    class_count : int
        Number of valid class ids (labels 0..class_count-1).
    ignore_index : int, optional
        Target label to exclude from scoring (e.g. void/unlabeled).
    """

    def __init__(self, class_count: int, ignore_index: int | None = None):
        self.class_count = class_count
        self.ignore_index = ignore_index
        self.confusion = np.zeros((class_count, class_count), dtype=np.int64)

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        """Accumulate one prediction/target pair of integer class-id arrays."""
        prediction = np.asarray(prediction).ravel()
        target = np.asarray(target).ravel()
        if self.ignore_index is not None:
            keep = target != self.ignore_index
            prediction, target = prediction[keep], target[keep]
        valid = (
            (target >= 0)
            & (target < self.class_count)
            & (prediction >= 0)
            & (prediction < self.class_count)
        )
        index = self.class_count * target[valid] + prediction[valid]
        counts = np.bincount(index, minlength=self.class_count**2)
        self.confusion += counts.reshape(self.class_count, self.class_count)

    def _tp_fp_fn(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        true_positive = np.diag(self.confusion).astype(np.float64)
        false_positive = self.confusion.sum(axis=0) - true_positive
        false_negative = self.confusion.sum(axis=1) - true_positive
        return true_positive, false_positive, false_negative

    def per_class_iou(self) -> np.ndarray:
        """Per-class IoU. NaN for classes absent from both prediction and target."""
        tp, fp, fn = self._tp_fp_fn()
        denominator = tp + fp + fn
        return np.divide(tp, denominator, out=np.full_like(tp, np.nan), where=denominator > 0)

    def per_class_accuracy(self) -> np.ndarray:
        """Per-class accuracy (recall). NaN for classes absent from the target."""
        tp, _, fn = self._tp_fp_fn()
        support = tp + fn
        return np.divide(tp, support, out=np.full_like(tp, np.nan), where=support > 0)

    def compute(self) -> dict[str, np.ndarray | float]:
        """Return per-class and mean IoU and accuracy.

        Returns
        -------
        dict
            Keys: iou and accuracy (per-class arrays), miou and macc (means over classes present in
                the data).
        """
        iou = self.per_class_iou()
        accuracy = self.per_class_accuracy()
        if self.ignore_index is not None and 0 <= self.ignore_index < self.class_count:
            iou[self.ignore_index] = np.nan
            accuracy[self.ignore_index] = np.nan
        return {
            "iou": iou,
            "accuracy": accuracy,
            "miou": float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else float("nan"),
            "macc": float(np.nanmean(accuracy)) if np.any(~np.isnan(accuracy)) else float("nan"),
        }

    def reset(self) -> None:
        """Clear all accumulated counts."""
        self.confusion[:] = 0
