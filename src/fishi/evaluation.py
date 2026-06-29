"""Evaluation harness: run a pipeline over a preprocessed dataset and score it."""

import numpy as np
import structlog

from fishi.metrics import SegmentationMetrics
from fishi.preprocess.base import ProcessedDataset
from fishi.segmentation.base import SegmentationPipeline

logger = structlog.get_logger(__name__)


def evaluate(
    processed_dataset: ProcessedDataset,
    pipeline: SegmentationPipeline,
    prompts: dict[int, str],
    class_count: int,
    ignore_index: int = 0,
) -> dict[str, np.ndarray | float]:
    """Score one (preprocessing x pipeline) cell of the experiment.

    Parameters
    ----------
    processed_dataset : ProcessedDataset
        A dataset wrapped by a Processor (carries the matching postprocess).
    pipeline : SegmentationPipeline
        The segmentation model to evaluate.
    prompts : dict of int to str
        Class id to text prompt.
    class_count : int
        Number of classes (for the metrics).
    ignore_index : int
        Label id excluded from scoring (void).

    Returns
    -------
    dict
        Per-class and mean IoU and Dice.
    """
    processor = processed_dataset.processor
    metric = SegmentationMetrics(class_count, ignore_index=ignore_index)
    total = len(processed_dataset)
    for index in range(total):
        sample = processed_dataset[index]
        predictions = [pipeline.predict(view, prompts) for view in sample.views]
        fisheye_prediction = processor.postprocess(predictions, sample.calibration)
        metric.update(fisheye_prediction, sample.label)
        if not (index + 1) % 50:
            logger.info("evaluated", done=index + 1, total=total)
    return metric.compute()
