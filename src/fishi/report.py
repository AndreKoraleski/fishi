"""Per-cell experiment metrics: one JSON per (pipeline, preprocessing), plus CSV export.

Each cell writes its own file, so runs are resumable and both notebooks can write into the same
metrics directory with no merge step.
"""

import csv
import json
from pathlib import Path

import numpy as np


def _finite(value: float) -> float | None:
    """JSON/CSV have no NaN, so map absent-class metrics to None."""
    return None if np.isnan(value) else float(value)


def cell_report(metrics: dict, class_names: list[str], pipeline: str, preprocessing: str) -> dict:
    """Build the report dict for one (pipeline, preprocessing) cell."""
    iou = np.asarray(metrics["iou"], dtype=float)
    dice = np.asarray(metrics["dice"], dtype=float)
    report = {
        "pipeline": pipeline,
        "preprocessing": preprocessing,
        "miou": float(metrics["miou"]),
        "mdice": float(metrics["mdice"]),
        "per_class": {
            name: {"iou": _finite(iou[index]), "dice": _finite(dice[index])}
            for index, name in enumerate(class_names)
        },
    }
    for key in ("pixel_accuracy", "mean_accuracy", "fwiou"):
        if key in metrics:
            report[key] = _finite(float(metrics[key]))
    if "errors" in metrics:
        report["errors"] = {
            name: _finite(float(value)) for name, value in metrics["errors"].items()
        }
    return report


def save_cell(
    metrics: dict,
    class_names: list[str],
    pipeline: str,
    preprocessing: str,
    directory: str | Path,
) -> Path:
    """Write one cell's metrics to <directory>/<pipeline>__<preprocessing>.json."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{pipeline}__{preprocessing}.json"
    path.write_text(
        json.dumps(cell_report(metrics, class_names, pipeline, preprocessing), indent=2)
    )
    return path


def load_cells(directory: str | Path) -> list[dict]:
    """Load every cell report JSON in the metrics directory (sorted by filename)."""
    return [json.loads(path.read_text()) for path in sorted(Path(directory).glob("*.json"))]


def to_csv(directory: str | Path, path: str | Path) -> Path:
    """Flatten the metrics directory into one CSV (a row per cell, per-class columns)."""
    cells = load_cells(directory)
    class_names = list(cells[0]["per_class"]) if cells else []
    fieldnames = ["pipeline", "preprocessing", "miou", "mdice"]
    for name in class_names:
        fieldnames += [f"iou_{name}", f"dice_{name}"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            row = {key: cell[key] for key in ("pipeline", "preprocessing", "miou", "mdice")}
            for name, scores in cell["per_class"].items():
                row[f"iou_{name}"] = scores["iou"]
                row[f"dice_{name}"] = scores["dice"]
            writer.writerow(row)
    return path


def to_matrix(directory: str | Path, metric: str = "miou") -> dict[str, dict[str, float | None]]:
    """Pivot the cells into {pipeline: {preprocessing: value}} for one top-level metric."""
    matrix: dict[str, dict[str, float | None]] = {}
    for cell in load_cells(directory):
        matrix.setdefault(cell["pipeline"], {})[cell["preprocessing"]] = cell.get(metric)
    return matrix
