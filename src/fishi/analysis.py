"""Analysis over the benchmark cache and dataset.

resampling_ceiling rounds-trips the ground-truth label through each processor (nearest-neighbour
preprocess, then postprocess back to fisheye) and scores it against the original. The gap from 1.0
is the loss the projection imposes on its own, a ceiling no segmentation model can beat.
"""

from collections.abc import Sequence

import cv2

from fishi.metrics import SegmentationMetrics
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
