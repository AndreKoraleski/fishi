"""Evaluation harness: run one preprocessing x pipeline cell and score it."""

from pathlib import Path

import numpy as np
import structlog
from PIL import Image

from fishi.metrics import SegmentationMetrics
from fishi.preprocess.base import ProcessedDataset, Processor, SampleSource
from fishi.segmentation.base import SegmentationPipeline
from fishi.woodscape.dataset import Sample

logger = structlog.get_logger(__name__)


def _load_cached(directory: Path | None, stem: str) -> np.ndarray | None:
    """Return the cached fisheye prediction for a stem, or None if not cached."""
    if directory is None:
        return None
    path = directory / f"{stem}.png"
    return np.asarray(Image.open(path)) if path.exists() else None


def _save_cached(directory: Path | None, stem: str, prediction: np.ndarray) -> None:
    """Cache a fisheye prediction, writing atomically so a crash can't truncate it."""
    if directory is None:
        return
    path = directory / f"{stem}.png"
    temporary = path.with_suffix(".png.tmp")
    Image.fromarray(prediction).save(temporary, format="PNG")
    temporary.replace(path)


def _score_window(
    window: list[Sample],
    pipeline: SegmentationPipeline,
    prompts: dict[int, str],
    processor: Processor,
    cache: Path | None,
    metric: SegmentationMetrics,
) -> None:
    """Score one window of samples, batching all uncached views into one inference."""
    flat_views: list[np.ndarray] = []
    spans: list[tuple[Sample, int, int]] = []  # (sample, offset into flat_views, view count)
    for sample in window:
        cached = _load_cached(cache, sample.stem)
        if cached is not None:
            metric.update(cached, sample.label)
            continue
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
        _save_cached(cache, sample.stem, fisheye)
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

    Samples are processed in windows of batch_size: every uncached view in a window (a window of
    patches yields several views per sample) is fed to the pipeline in a single predict_batch call,
    which auto-shrinks on GPU OOM. Cached samples skip preprocessing entirely.

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
    cache: Path | None = None
    if cache_directory is not None:
        cache = Path(cache_directory) / pipeline.name / processor.name
        cache.mkdir(parents=True, exist_ok=True)

    total = len(dataset)
    for start in range(0, total, batch_size):
        window = [dataset[index] for index in range(start, min(start + batch_size, total))]
        _score_window(window, pipeline, prompts, processor, cache, metric)
        logger.info("evaluated", done=min(start + batch_size, total), total=total)
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
