"""Sweep one loaded pipeline over the preprocessings, scoring each cell.

A driver loads a single model and calls sweep, so the heavy pipelines never coexist in memory.
"""

from pathlib import Path

from fishi.evaluation import run
from fishi.preprocess import Identity, Patches, Rectify, TangentImages
from fishi.preprocess.base import Processor, SampleSource
from fishi.segmentation.base import SegmentationPipeline

PREPROCESSORS: tuple[Processor, ...] = (Identity(), Rectify(), Patches(), TangentImages())


def sweep(
    pipeline: SegmentationPipeline,
    dataset: SampleSource,
    metrics_directory: str | Path,
    cache_directory: str | Path | None = None,
    processors: tuple[Processor, ...] = PREPROCESSORS,
) -> None:
    """Run one loaded pipeline over each preprocessing, saving a report per cell (resumable)."""
    for processor in processors:
        run(
            processor,
            pipeline,
            dataset,
            metrics_directory=metrics_directory,
            cache_directory=cache_directory,
        )
