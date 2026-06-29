"""WoodScape tests: calibration, classes, dataset, download, and splits."""

import json

import numpy as np
import pytest

from fishi.woodscape import classes, download
from fishi.woodscape.calibration import Calibration
from fishi.woodscape.config import get_settings
from fishi.woodscape.dataset import Subset, WoodScapeDataset
from fishi.woodscape.splits import canonical_split, load_split, make_split, split_datasets

# Calibration


FV = {
    "intrinsic": {
        "aspect_ratio": 1.0,
        "cx_offset": 3.942,
        "cy_offset": -3.093,
        "height": 966.0,
        "k1": 339.749,
        "k2": -31.988,
        "k3": 48.275,
        "k4": -7.201,
        "model": "radial_poly",
        "poly_order": 4,
        "width": 1280.0,
    },
    "extrinsic": {
        "quaternion": [0.5946, -0.5837, 0.3906, -0.3910],
        "translation": [3.7484, 0.0, 0.6578],
    },
    "name": "FV",
}


def test_principal_point():
    calibration = Calibration.from_dict(FV)
    center_x, center_y = calibration.principal_point
    assert np.isclose(center_x, 0.5 * 1280 + 3.942 - 0.5)
    assert np.isclose(center_y, 0.5 * 966 - 3.093 - 0.5)


def test_optical_axis_projects_to_center():
    calibration = Calibration.from_dict(FV)
    uv = calibration.project(np.array([0.0, 0.0, 1.0]))
    assert np.allclose(uv, calibration.principal_point)


def test_project_unproject_roundtrip():
    calibration = Calibration.from_dict(FV)
    rays = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.2, 0.1, 1.0],
            [-0.3, 0.25, 1.0],
            [0.5, -0.4, 1.0],
        ]
    )
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    recovered = calibration.unproject(calibration.project(rays))
    assert np.allclose(recovered, rays, atol=1e-3)


def test_from_json(tmp_path):
    path = tmp_path / "00000_FV.json"
    path.write_text(json.dumps(FV))
    calibration = Calibration.from_json(path)
    assert calibration.name == "FV"
    assert calibration.width == 1280
    assert calibration.k1 == 339.749


def test_unproject_handles_non_monotonic_polynomial(make_calibration):
    calibration = make_calibration(k1=1.0, k2=-1.0, width=8.0, height=8.0)
    rays = calibration.unproject(np.array([[4.0, 4.0], [0.0, 0.0]]))
    assert rays.shape == (2, 3)


# Classes


def test_class_taxonomy():
    assert len(classes.CLASS_NAMES) == 10
    assert classes.CLASS_NAMES[classes.VOID_ID] == "void"
    assert classes.VOID_ID not in classes.PROMPTS
    assert len(classes.PROMPTS) == 9
    assert classes.PROMPTS[9] == "traffic sign"


def test_class_names_match_official_taxonomy():
    assert classes.CLASS_NAMES == [
        "void",
        "road",
        "lanemarks",
        "curb",
        "person",
        "rider",
        "vehicles",
        "bicycle",
        "motorcycle",
        "traffic_sign",
    ]
    assert classes.CLASS_COUNT == 10
    assert classes.ID_TO_NAME[6] == "vehicles"


# Dataset


def test_len_and_pairing(woodscape_root):
    dataset = WoodScapeDataset(get_settings(data_directory=woodscape_root))
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample.image.shape == (4, 4, 3)
    assert sample.label.shape == (4, 4)
    assert sample.stem == "00000_FV"
    assert sample.camera == "FV"
    assert sample.calibration.name == "FV"


def test_camera_filter(woodscape_root):
    dataset = WoodScapeDataset(get_settings(data_directory=woodscape_root), cameras=["FV"])
    assert len(dataset) == 1
    assert dataset[0].camera == "FV"


def test_subset_selects_given_indices(woodscape_root):
    dataset = WoodScapeDataset(get_settings(data_directory=woodscape_root))
    subset = Subset(dataset, [1])
    assert len(subset) == 1
    assert subset[0].stem == dataset[1].stem


def test_incomplete_samples_are_skipped(woodscape_root):
    (woodscape_root / "semantic_annotations" / "gtLabels" / "00001_RV.png").unlink()
    dataset = WoodScapeDataset(get_settings(data_directory=woodscape_root))
    assert dataset.stems == ["00000_FV"]


def test_missing_images_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        WoodScapeDataset(get_settings(data_directory=tmp_path))


# Download


@pytest.fixture
def record_downloads(monkeypatch) -> list[str]:
    calls: list[str] = []

    def fake_download(file_id, output, quiet) -> None:
        output.write_bytes(b"stub")
        calls.append(output.name)

    monkeypatch.setattr(download, "fetch", fake_download)
    monkeypatch.setattr(download, "extract", lambda archive, destination: None)
    return calls


def test_default_run_fetches_all_artifacts(record_downloads, tmp_path):
    download.download_woodscape(get_settings(data_directory=tmp_path))
    assert {"rgb_images.zip", "semantic_annotations.zip", "calibration.zip"} <= set(
        record_downloads
    )


def test_markers_make_reruns_idempotent(record_downloads, tmp_path):
    download.download_woodscape(get_settings(data_directory=tmp_path))
    count = len(record_downloads)
    download.download_woodscape(get_settings(data_directory=tmp_path))
    assert len(record_downloads) == count


def test_failed_download_removes_partial_and_is_not_recorded(monkeypatch, tmp_path):
    def boom(file_id, output, quiet) -> None:
        output.write_bytes(b"partial")
        raise RuntimeError("network died")

    monkeypatch.setattr(download, "fetch", boom)
    monkeypatch.setattr(download, "extract", lambda archive, dest: None)

    with pytest.raises(RuntimeError):
        download.download_woodscape(get_settings(data_directory=tmp_path))

    assert not (tmp_path / "rgb_images.zip").exists()
    assert not (tmp_path / ".fishi_completed.txt").exists()


# Splits


def test_make_split_is_deterministic():
    stems = [f"{i:05d}_FV" for i in range(100)]
    assert make_split(stems) == make_split(stems)


def test_make_split_partitions_with_proportions():
    stems = [f"{i:05d}_FV" for i in range(100)]
    split = make_split(stems)
    assert (len(split["train"]), len(split["validation"]), len(split["test"])) == (70, 15, 15)
    combined = split["train"] + split["validation"] + split["test"]
    assert sorted(combined) == stems


def test_canonical_split_is_a_clean_partition():
    split = canonical_split()
    assert set(split) == {"train", "validation", "test"}
    combined = [stem for stems in split.values() for stem in stems]
    assert len(combined) == len(set(combined))
    assert round(len(split["train"]) / len(combined), 2) == 0.70


def test_split_datasets_maps_stems_to_subsets(woodscape_root):
    dataset = WoodScapeDataset(get_settings(data_directory=woodscape_root))
    subsets = split_datasets(
        dataset, {"train": ["00000_FV"], "validation": ["00001_RV"], "test": []}
    )
    assert {name: len(subset) for name, subset in subsets.items()} == {
        "train": 1,
        "validation": 1,
        "test": 0,
    }
    assert subsets["train"][0].stem == "00000_FV"


def test_split_datasets_drops_missing_stems(woodscape_root):
    dataset = WoodScapeDataset(get_settings(data_directory=woodscape_root))
    subsets = split_datasets(
        dataset, {"train": ["00000_FV", "99999_XX"], "validation": [], "test": []}
    )
    assert len(subsets["train"]) == 1


def test_load_split_returns_named_subset(woodscape_root):
    subset = load_split(
        "train",
        get_settings(data_directory=woodscape_root),
        split={"train": ["00000_FV"], "validation": [], "test": []},
    )
    assert len(subset) == 1
    assert subset[0].stem == "00000_FV"
