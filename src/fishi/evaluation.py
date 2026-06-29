"""Evaluation harness: run one preprocessing x pipeline cell and score it."""

from pathlib import Path

import numpy as np
import structlog
from PIL import Image

from fishi.metrics import SegmentationMetrics
from fishi.preprocess.base import ProcessedDataset, ProcessedSample, Processor, SampleSource
from fishi.segmentation.base import SegmentationPipeline

logger = structlog.get_logger(__name__)


def _prediction(
    sample: ProcessedSample,
    pipeline: SegmentationPipeline,
    prompts: dict[int, str],
    processor: Processor,
    cache_directory: Path | None,
) -> np.ndarray:
    """Load a sample's cached fisheye prediction, or infer it and cache it."""
    path = cache_directory / f"{sample.stem}.png" if cache_directory is not None else None
    if path is not None and path.exists():
        return np.asarray(Image.open(path))
    per_view = [pipeline.predict(view, prompts) for view in sample.views]
    prediction = processor.postprocess(per_view, sample.calibration).astype(np.uint8)
    if path is not None:
        Image.fromarray(prediction).save(path)
    return prediction


def evaluate(
    processed_dataset: ProcessedDataset,
    pipeline: SegmentationPipeline,
    prompts: dict[int, str],
    class_count: int,
    ignore_index: int = 0,
    cache_directory: str | Path | None = None,
) -> dict[str, np.ndarray | float]:
    """Score one (preprocessing x pipeline) cell over a preprocessed dataset.

    For each sample: run the pipeline on every view, merge the per-view predictions back to fisheye
    space via the processor's postprocess, and accumulate IoU/Dice against the fisheye ground truth.

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
    cache_directory : str or Path, optional
        If given, fisheye predictions are cached as PNGs under
        cache_directory/<pipeline>/<processor>/<stem>.png and reused on re-runs.

    Returns
    -------
    dict
        Per-class and mean IoU and Dice.
    """
    processor = processed_dataset.processor
    metric = SegmentationMetrics(class_count, ignore_index=ignore_index)
    cell_cache: Path | None = None
    if cache_directory is not None:
        cell_cache = Path(cache_directory) / pipeline.name / processor.name
        cell_cache.mkdir(parents=True, exist_ok=True)
    total = len(processed_dataset)
    for index in range(total):
        sample = processed_dataset[index]
        prediction = _prediction(sample, pipeline, prompts, processor, cell_cache)
        metric.update(prediction, sample.label)
        if not (index + 1) % 50:
            logger.info("evaluated", done=index + 1, total=total)
    return metric.compute()


def run(
    processor: Processor,
    pipeline: SegmentationPipeline,
    dataset: SampleSource,
    prompts: dict[int, str],
    class_count: int,
    ignore_index: int = 0,
    cache_directory: str | Path | None = None,
) -> dict[str, np.ndarray | float]:
    """Wrap a dataset with a processor and evaluate one cell, returning its metrics."""
    return evaluate(
        processor.wrap(dataset), pipeline, prompts, class_count, ignore_index, cache_directory
    )
