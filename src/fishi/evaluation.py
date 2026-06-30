"""Evaluation harness: score one preprocessing x pipeline cell at a time.

evaluate scores a cell. run wraps a processor, caches predictions, saves the cell report, and skips
cells already done. A driver (e.g. a notebook) loads one model at a time and loops these over the
preprocessings, so the heavy pipelines never coexist in memory. Aggregate the saved cell reports
with report.to_matrix.
"""

import json
from pathlib import Path

import numpy as np
import structlog

from fishi.metrics import SegmentationMetrics
from fishi.preprocess.base import ProcessedDataset, Processor, SampleSource
from fishi.report import cell_report, save_cell
from fishi.segmentation.base import SegmentationPipeline
from fishi.woodscape import classes

logger = structlog.get_logger(__name__)


def _cache_path(cache_directory: str | Path | None, pipeline: str, processor: str) -> Path | None:
    """Path of a cell's prediction archive, or None when caching is off."""
    if cache_directory is None:
        return None
    return Path(cache_directory) / f"{pipeline}__{processor}.npz"


def _load_cache(path: Path | None) -> dict[str, np.ndarray]:
    """Load a cell's cached fisheye predictions ({stem: mask}) from its .npz, or {}."""
    if path is None or not path.exists():
        return {}
    with np.load(path) as data:
        return {stem: data[stem] for stem in data.files}


def _save_cache(path: Path, predictions: dict[str, np.ndarray]) -> None:
    """Write the cell's predictions ({stem: mask}) to its .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **predictions)  # type: ignore[arg-type]


def evaluate(
    processed_dataset: ProcessedDataset,
    pipeline: SegmentationPipeline,
    prompts: dict[int, str] | None = None,
    class_count: int | None = None,
    ignore_index: int = classes.VOID_ID,
    cache_directory: str | Path | None = None,
    report_every: int = 50,
    checkpoint_every: int = 200,
) -> dict[str, np.ndarray | float]:
    """Score one (preprocessing x pipeline) cell, one image at a time.

    Parameters
    ----------
    processed_dataset : ProcessedDataset
        A dataset wrapped by a Processor (carries the matching postprocess).
    pipeline : SegmentationPipeline
        The segmentation model to evaluate.
    prompts : dict of int to str, optional
        Class id to text prompt. Defaults to the WoodScape taxonomy.
    class_count : int, optional
        Number of classes. Defaults to the WoodScape taxonomy.
    ignore_index : int
        Label id excluded from scoring (void).
    cache_directory : str or Path, optional
        If given, the cell's fisheye predictions are cached as one .npz and reused on re-runs.
    report_every : int
        Log progress every this many samples.
    checkpoint_every : int
        Flush the prediction cache every this many samples, so a crash resumes mid-cell.

    Returns
    -------
    dict
        Per-class and mean IoU and accuracy.
    """
    prompts = classes.PROMPTS if prompts is None else prompts
    class_count = classes.CLASS_COUNT if class_count is None else class_count
    dataset = processed_dataset.dataset
    processor = processed_dataset.processor
    metric = SegmentationMetrics(class_count, ignore_index=ignore_index)

    path = _cache_path(cache_directory, pipeline.name, processor.name)
    cached = _load_cache(path)
    fresh: dict[str, np.ndarray] = {}

    total = len(dataset)
    for index in range(total):
        stem = dataset.stem(index)
        if stem in cached:
            metric.update(cached[stem], dataset.label(index))  # skip the RGB decode
        else:
            sample = dataset[index]
            views = processor.preprocess(sample.image, sample.calibration)
            predictions = [pipeline.predict(view, prompts) for view in views]
            fisheye = processor.postprocess(predictions, sample.calibration).astype(np.uint8)
            fresh[stem] = fisheye
            metric.update(fisheye, sample.label)
        done = index + 1
        if done % report_every == 0 or done == total:
            logger.info("evaluated", done=done, total=total)
        if path is not None and fresh and done % checkpoint_every == 0:
            _save_cache(path, {**cached, **fresh})

    if path is not None and fresh:
        _save_cache(path, {**cached, **fresh})
    return metric.compute()


def run(
    processor: Processor,
    pipeline: SegmentationPipeline,
    dataset: SampleSource,
    prompts: dict[int, str] | None = None,
    class_count: int | None = None,
    ignore_index: int = classes.VOID_ID,
    cache_directory: str | Path | None = None,
    metrics_directory: str | Path | None = None,
    report_every: int = 50,
    checkpoint_every: int = 200,
    force: bool = False,
) -> dict:
    """Evaluate one cell and return its cell report, saving it and skipping cells already finished.

    The returned dict is the cell report (pipeline, preprocessing, miou, macc, per_class), not the
    raw metric arrays that evaluate and score return. With metrics_directory set, the cell report is
    written there, and a cell whose report already exists is skipped (its saved report is returned)
    unless force is True.
    """
    if metrics_directory is not None and not force:
        report_path = Path(metrics_directory) / f"{pipeline.name}__{processor.name}.json"
        if report_path.exists():
            logger.info("cell_cached", pipeline=pipeline.name, preprocessing=processor.name)
            return json.loads(report_path.read_text())
    metrics = evaluate(
        processor.wrap(dataset),
        pipeline,
        prompts,
        class_count,
        ignore_index,
        cache_directory,
        report_every,
        checkpoint_every,
    )
    if metrics_directory is not None:
        save_cell(metrics, classes.CLASS_NAMES, pipeline.name, processor.name, metrics_directory)
    return cell_report(metrics, classes.CLASS_NAMES, pipeline.name, processor.name)


def score(
    predictions: dict[str, np.ndarray],
    dataset: SampleSource,
    class_count: int | None = None,
    ignore_index: int = classes.VOID_ID,
) -> dict[str, np.ndarray | float]:
    """Score externally-produced predictions against a dataset, on the project's protocol.

    predictions maps each sample stem to its fisheye-space label map (the project's class ids).
    Pair it with a split, e.g. score(my_predictions, load_split("test")), to compare a model
    trained elsewhere against our numbers. Stems missing from predictions are skipped.
    """
    class_count = classes.CLASS_COUNT if class_count is None else class_count
    metric = SegmentationMetrics(class_count, ignore_index=ignore_index)
    for index in range(len(dataset)):
        stem = dataset.stem(index)
        if stem in predictions:
            metric.update(predictions[stem], dataset.label(index))
    return metric.compute()
