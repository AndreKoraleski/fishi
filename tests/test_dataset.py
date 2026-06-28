import json

import numpy as np
import pytest
from PIL import Image

from fishi.woodscape.config import get_settings
from fishi.woodscape.dataset import WoodScapeDataset

FV_CALIB = {
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
    "extrinsic": {"quaternion": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
    "name": "FV",
}


def _write_sample(root, stem, height=4, width=4):
    rgb = root / "rgb_images"
    label = root / "semantic_annotations" / "gtLabels"
    calib = root / "calibration"
    for directory in (rgb, label, calib):
        directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(rgb / f"{stem}.png")
    Image.fromarray(np.ones((height, width), dtype=np.uint8)).save(label / f"{stem}.png")
    (calib / f"{stem}.json").write_text(json.dumps({**FV_CALIB, "name": stem.rsplit("_", 1)[-1]}))


@pytest.fixture
def dataset_root(tmp_path):
    _write_sample(tmp_path, "00000_FV")
    _write_sample(tmp_path, "00001_RV")
    return tmp_path


def test_len_and_pairing(dataset_root):
    dataset = WoodScapeDataset(get_settings(data_directory=dataset_root))
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample.image.shape == (4, 4, 3)
    assert sample.label.shape == (4, 4)
    assert sample.stem == "00000_FV"
    assert sample.camera == "FV"
    assert sample.calibration.name == "FV"


def test_camera_filter(dataset_root):
    dataset = WoodScapeDataset(get_settings(data_directory=dataset_root), cameras=["FV"])
    assert len(dataset) == 1
    assert dataset[0].camera == "FV"


def test_preprocess_applied(dataset_root):
    def to_white(image, calibration):
        return np.full_like(image, 255)

    dataset = WoodScapeDataset(get_settings(data_directory=dataset_root), preprocess=to_white)
    assert (dataset[0].image == 255).all()
