"""The fishi command line: download, run, report, diagnose, ceiling, and demos."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from fishi.analysis import error_decomposition, resampling_ceiling
from fishi.evaluation import run
from fishi.preprocess import patches, rectify, tangent
from fishi.report import to_csv, to_matrix
from fishi.sweep import PREPROCESSORS
from fishi.woodscape import Calibration, download_woodscape, get_settings, load_split

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
run_app = typer.Typer(no_args_is_help=True, help="Run a model over the preprocessing sweep.")
app.add_typer(run_app, name="run")
console = Console()

DataDirectory = Annotated[str, typer.Option(help="WoodScape root")]
MetricsDirectory = Annotated[str, typer.Option(help="where per-cell reports are saved")]
CacheDirectory = Annotated[str, typer.Option(help="where predictions are cached")]
Split = Annotated[str, typer.Option(help="split to evaluate")]


def _sweep(
    pipeline, data_directory: str, metrics_directory: str, cache_directory: str, split: str
) -> None:
    dataset = load_split(split, get_settings(data_directory=data_directory))
    for processor in PREPROCESSORS:
        console.rule(f"[bold cyan]{pipeline.name} x {processor.name}")
        run(
            processor,
            pipeline,
            dataset,
            metrics_directory=metrics_directory,
            cache_directory=cache_directory,
        )
    console.print(f"[green]done: reports in {metrics_directory}/")


@app.command()
def download(data_directory: DataDirectory = "data") -> None:
    """Download the WoodScape dataset."""
    console.print("[bold]Downloading WoodScape...")
    download_woodscape(get_settings(data_directory=data_directory))


@run_app.command("sam3")
def run_sam3(
    data_directory: DataDirectory = "data",
    metrics_directory: MetricsDirectory = "metrics",
    cache_directory: CacheDirectory = "cache",
    split: Split = "test",
) -> None:
    """SAM 3 (needs HF_TOKEN, facebook/sam3 is gated)."""
    from fishi.segmentation import SamThree

    _sweep(SamThree(), data_directory, metrics_directory, cache_directory, split)


@run_app.command("gdino-sam1")
def run_gdino_sam1(
    data_directory: DataDirectory = "data",
    metrics_directory: MetricsDirectory = "metrics",
    cache_directory: CacheDirectory = "cache",
    split: Split = "test",
) -> None:
    """Grounding DINO + SAM 1."""
    from fishi.segmentation import GroundedSam1

    _sweep(GroundedSam1(), data_directory, metrics_directory, cache_directory, split)


@run_app.command("gdino-sam2")
def run_gdino_sam2(
    data_directory: DataDirectory = "data",
    metrics_directory: MetricsDirectory = "metrics",
    cache_directory: CacheDirectory = "cache",
    split: Split = "test",
) -> None:
    """Grounding DINO + SAM 2."""
    from fishi.segmentation import GroundedSam2

    _sweep(GroundedSam2(), data_directory, metrics_directory, cache_directory, split)


@run_app.command("openworldsam")
def run_openworldsam(
    config_file: Annotated[str, typer.Option(help="OpenWorldSAM detectron2 YAML config")],
    weights: Annotated[str, typer.Option(help="OpenWorldSAM checkpoint path")],
    repo_path: Annotated[str | None, typer.Option(help="OpenWorldSAM repo root")] = None,
    data_directory: DataDirectory = "data",
    metrics_directory: MetricsDirectory = "metrics",
    cache_directory: CacheDirectory = "cache",
    split: Split = "test",
) -> None:
    """OpenWorldSAM (detectron2, needs its repo via --repo-path)."""
    from fishi.segmentation import OpenWorldSam

    pipeline = OpenWorldSam(config_file, weights, repo_path=repo_path)
    _sweep(pipeline, data_directory, metrics_directory, cache_directory, split)


@app.command()
def report(
    metrics_directory: MetricsDirectory = "metrics",
    csv: Annotated[str | None, typer.Option(help="also write a CSV here")] = None,
) -> None:
    """Aggregate per-cell metrics into tables (pipeline x preprocessing, mIoU and mean accuracy)."""
    for metric, title in (("miou", "mIoU"), ("macc", "mean accuracy")):
        matrix = to_matrix(metrics_directory, metric)
        preprocessings = sorted({name for row in matrix.values() for name in row})
        table = Table(title=title)
        table.add_column("pipeline", style="bold")
        for name in preprocessings:
            table.add_column(name, justify="right")
        for pipeline in sorted(matrix):
            row = matrix[pipeline]
            cells = [f"{row[p]:.3f}" if row.get(p) is not None else "-" for p in preprocessings]
            table.add_row(pipeline, *cells)
        console.print(table)
    if csv:
        to_csv(metrics_directory, csv)
        console.print(f"[green]wrote {csv}")


@app.command()
def ceiling(
    data_directory: DataDirectory = "data",
    split: Split = "test",
    count: Annotated[int | None, typer.Option(help="samples to average over")] = None,
) -> None:
    """Resampling ceiling per preprocessing: the max mIoU a perfect model could reach (no model)."""
    dataset = load_split(split, get_settings(data_directory=data_directory))
    with console.status("computing resampling ceilings..."):
        ceilings = resampling_ceiling(PREPROCESSORS, dataset, count)
    table = Table(title="resampling ceiling")
    table.add_column("preprocessing", style="bold")
    table.add_column("ceiling mIoU", justify="right")
    for name, value in ceilings.items():
        table.add_row(name, f"{value:.4f}")
    console.print(table)


@app.command()
def demos(
    data_directory: DataDirectory = "data",
    output_directory: Annotated[str, typer.Option(help="where the PNGs go")] = "demos",
    count: Annotated[int, typer.Option(help="number of samples to render")] = 3,
) -> None:
    """Render preprocessing demonstration images."""
    dataset = load_split("test", get_settings(data_directory=data_directory))
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    renderers: dict[str, Callable[[np.ndarray, Calibration], np.ndarray]] = {
        "rectify": rectify.demonstration,
        "patches": patches.demonstration,
        "tangent": tangent.demonstration,
    }
    for index in track(range(min(count, len(dataset))), description="rendering demos"):
        sample = dataset[index]
        for name, render in renderers.items():
            rgba = render(sample.image, sample.calibration)
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(str(output / f"{sample.stem}_{name}.png"), bgra)
    console.print(f"[green]wrote demos to {output}/")


@app.command()
def diagnose(
    cache_directory: CacheDirectory = "cache",
    data_directory: DataDirectory = "data",
    split: Split = "test",
) -> None:
    """Per-cell diagnostics recomputed from the cached predictions (error split, FWIoU, groups)."""
    dataset = load_split(split, get_settings(data_directory=data_directory))
    rows = error_decomposition(cache_directory, dataset)
    table = Table(title="diagnostics")
    table.add_column("cell", style="bold")
    for name in ("mIoU", "FWIoU", "things", "stuff", "confused", "missed"):
        table.add_column(name, justify="right")
    for cell, values in rows.items():
        table.add_row(
            cell,
            f"{values['miou']:.3f}",
            f"{values['fwiou']:.3f}",
            f"{values['things']:.3f}",
            f"{values['stuff']:.3f}",
            f"{values['confused']:.3f}",
            f"{values['missed']:.3f}",
        )
    console.print(table)


if __name__ == "__main__":
    app()
