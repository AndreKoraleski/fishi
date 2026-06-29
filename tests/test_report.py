import numpy as np

from fishi.report import cell_report, load_cells, save_cell, to_csv


def _metrics():
    return {
        "iou": np.array([np.nan, 0.5]),
        "dice": np.array([np.nan, 2 / 3]),
        "miou": 0.5,
        "mdice": 2 / 3,
    }


def test_cell_report_structure_and_nan_handling():
    report = cell_report(_metrics(), ["void", "road"], "gdino+sam1", "none")
    assert report["pipeline"] == "gdino+sam1"
    assert report["preprocessing"] == "none"
    assert report["per_class"]["void"]["iou"] is None  # absent class -> null
    assert report["per_class"]["road"]["iou"] == 0.5


def test_save_cell_and_load_cells(tmp_path):
    save_cell(_metrics(), ["void", "road"], "gdino+sam1", "none", tmp_path)
    save_cell(_metrics(), ["void", "road"], "gdino+sam2", "rectify", tmp_path)
    cells = load_cells(tmp_path)
    assert len(cells) == 2
    assert {cell["pipeline"] for cell in cells} == {"gdino+sam1", "gdino+sam2"}


def test_to_csv_flattens_cells(tmp_path):
    save_cell(_metrics(), ["void", "road"], "gdino+sam1", "none", tmp_path)
    text = to_csv(tmp_path, tmp_path / "metrics.csv").read_text()
    assert "pipeline,preprocessing,miou,mdice,iou_void,dice_void,iou_road,dice_road" in text
    assert "gdino+sam1,none,0.5," in text
