"""Evaluation harness: run one preprocessing x pipeline cell and score it."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import structlog

from fishi.metrics import SegmentationMetrics
from fishi.preprocess.base import ProcessedDataset, Processor, SampleSource
from fishi.segmentation.base import SegmentationPipeline
from fishi.woodscape.dataset import Sample

logger = structlog.get_logger(__name__)


def _cache_path(cache_directory: str | Path | None, pipeline: str, processor: str) -> Path | None:
    """Path of a cell's prediction archive, or None when caching is off."""
    if cache_directory is None:
        return None
    return Path(cache_directory) / f"{pipeline}__{processor}.npz"


def _load_cell_cache(path: Path | None) -> dict[str, np.ndarray]:
    """Load a cell's cached fisheye predictions ({stem: mask}) from its .npz, or {}."""
    if path is None or not path.exists():
        return {}
    with np.load(path) as data:
        return {stem: data[stem] for stem in data.files}


def _score_window(
    dataset: SampleSource,
    indices: Iterable[int],
    pipeline: SegmentationPipeline,
    prompts: dict[int, str],
    processor: Processor,
    cached: dict[str, np.ndarray],
    fresh: dict[str, np.ndarray],
    metric: SegmentationMetrics,
) -> None:
    """Score one window: reuse cached masks (reading only the label), buffer fresh ones."""
    flat_views: list[np.ndarray] = []
    spans: list[tuple[Sample, int, int]] = []  # (sample, offset into flat_views, view count)
    for index in indices:
        stem = dataset.stem(index)
        if stem in cached:
            metric.update(cached[stem], dataset.label(index))  # skip the RGB decode
            continue
        sample = dataset[index]
        views = processor.preprocess(sample.image, sample.calibration)
        spans.append((sample, len(flat_views), len(views)))
        flat_views.extend(views)

    if not flat_views:
        return

    predictions = pipeline.predict_batch(flat_views, prompts)
    for sample, offset, count in spans:
        fisheye = processor.postprocess(
            predictions[offset : offset + count], sample.calibration
        ).astype(np.uint8)
        fresh[sample.stem] = fisheye
        metric.update(fisheye, sample.label)


def evaluate(
    processed_dataset: ProcessedDataset,
    pipeline: SegmentationPipeline,
    prompts: dict[int, str],
    class_count: int,
    ignore_index: int = 0,
    cache_directory: str | Path | None = None,
    batch_size: int = 8,
) -> dict[str, np.ndarray | float]:
    """Score one (preprocessing x pipeline) cell over a preprocessed dataset.

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
        If given, the cell's fisheye predictions are cached as one .npz and reused on re-runs.
    batch_size : int
        Samples grouped per inference window; raise it to keep larger GPUs fed.

    Returns
    -------
    dict
        Per-class and mean IoU and Dice.
    """
    dataset = processed_dataset.dataset
    processor = processed_dataset.processor
    metric = SegmentationMetrics(class_count, ignore_index=ignore_index)

    path = _cache_path(cache_directory, pipeline.name, processor.name)
    cached = _load_cell_cache(path)
    fresh: dict[str, np.ndarray] = {}

    total = len(dataset)
    for start in range(0, total, batch_size):
        indices = range(start, min(start + batch_size, total))
        _score_window(dataset, indices, pipeline, prompts, processor, cached, fresh, metric)
        logger.info("evaluated", done=min(start + batch_size, total), total=total)

    if path is not None and fresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{**cached, **fresh})  # type: ignore[arg-type]
    return metric.compute()


def run(
    processor: Processor,
    pipeline: SegmentationPipeline,
    dataset: SampleSource,
    prompts: dict[int, str],
    class_count: int,
    ignore_index: int = 0,
    cache_directory: str | Path | None = None,
    batch_size: int = 8,
) -> dict[str, np.ndarray | float]:
    """Wrap a dataset with a processor and evaluate one cell, returning its metrics."""
    return evaluate(
        processor.wrap(dataset),
        pipeline,
        prompts,
        class_count,
        ignore_index,
        cache_directory,
        batch_size,
    )
