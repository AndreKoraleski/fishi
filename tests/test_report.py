import numpy as np

from fishi.report import cell_report, load_cells, save_cell, to_csv, to_markdown, to_matrix


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
    assert report["per_class"]["void"]["iou"] is None  # absent class gives null
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


def test_cell_report_carries_extended_metrics():
    metrics = {
        **_metrics(),
        "pixel_accuracy": 0.9,
        "mean_accuracy": 0.8,
        "fwiou": 0.7,
        "errors": {"correct": 0.6, "confused": 0.3, "missed": 0.1},
    }
    report = cell_report(metrics, ["void", "road"], "sam3", "none")
    assert report["pixel_accuracy"] == 0.9
    assert report["fwiou"] == 0.7
    assert report["errors"]["confused"] == 0.3


def test_to_matrix_and_markdown_show_delta_vs_baseline(tmp_path):
    save_cell(_metrics(), ["void", "road"], "sam3", "none", tmp_path)  # miou 0.5
    save_cell({**_metrics(), "miou": 0.6}, ["void", "road"], "sam3", "rectify", tmp_path)
    matrix = to_matrix(tmp_path)
    assert matrix["sam3"]["none"] == 0.5
    assert matrix["sam3"]["rectify"] == 0.6
    markdown = to_markdown(tmp_path)
    assert "sam3" in markdown
    assert "(+0.100)" in markdown  # rectify 0.6 against the none baseline 0.5
