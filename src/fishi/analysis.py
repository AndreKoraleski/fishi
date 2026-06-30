"""Analysis over the benchmark cache and dataset.

resampling_ceiling rounds-trips the ground-truth label through each processor (nearest-neighbour
preprocess, then postprocess back to fisheye) and scores it against the original. The gap from 1.0
is the loss the projection imposes on its own, a ceiling no segmentation model can beat.
"""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from fishi.metrics import (
    SegmentationMetrics,
    error_breakdown,
    frequency_weighted_iou,
    grouped_miou,
)
from fishi.preprocess.base import Processor, SampleSource
from fishi.woodscape import classes


def resampling_ceiling(
    processors: Sequence[Processor],
    dataset: SampleSource,
    count: int | None = None,
    class_count: int | None = None,
    ignore_index: int = classes.VOID_ID,
) -> dict[str, float]:
    """Per processor, the max mIoU a perfect model could reach (ground-truth label round-trip)."""
    class_count = classes.CLASS_COUNT if class_count is None else class_count
    total = len(dataset) if count is None else min(count, len(dataset))
    ceilings: dict[str, float] = {}
    for processor in processors:
        metric = SegmentationMetrics(class_count, ignore_index=ignore_index)
        for index in range(total):
            sample = dataset[index]
            views = processor.preprocess(
                sample.label, sample.calibration, interpolation=cv2.INTER_NEAREST
            )
            recovered = processor.postprocess(views, sample.calibration)
            metric.update(recovered, sample.label)
        ceilings[processor.name] = float(metric.compute()["miou"])
    return ceilings


def error_decomposition(
    cache_directory: str | Path,
    dataset: SampleSource,
    class_count: int | None = None,
    ignore_index: int = classes.VOID_ID,
) -> dict[str, dict[str, float]]:
    """Recompute per-cell diagnostics from the cached predictions in cache_directory.

    Each cache file is named pipeline__preprocessing.npz and maps stems to fisheye label maps. For
    each, returns mIoU, frequency-weighted IoU, the things and stuff means, and the error split of
    the scored pixels into confused (predicted another class) and missed (predicted background).
    """
    class_count = classes.CLASS_COUNT if class_count is None else class_count
    labels = {dataset.stem(index): dataset.label(index) for index in range(len(dataset))}
    groups = {"things": classes.THING_IDS, "stuff": classes.STUFF_IDS}
    results: dict[str, dict[str, float]] = {}
    for path in sorted(Path(cache_directory).glob("*.npz")):
        metric = SegmentationMetrics(class_count, ignore_index=ignore_index)
        with np.load(path) as predictions:
            for stem in predictions.files:
                if stem in labels:
                    metric.update(predictions[stem], labels[stem])
        grouped = grouped_miou(metric, groups)
        errors = error_breakdown(metric)
        results[path.stem] = {
            "miou": float(metric.compute()["miou"]),
            "fwiou": frequency_weighted_iou(metric),
            "things": grouped["things"],
            "stuff": grouped["stuff"],
            "confused": errors["confused"],
            "missed": errors["missed"],
        }
    return results
