import json

import numpy as np
import pytest
from PIL import Image

from fishi.woodscape.calibration import Calibration
from fishi.woodscape.dataset import Sample


class _FakeDataset:
    """Minimal in-memory SampleSource for tests."""

    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def stem(self, index):
        return self.samples[index].stem

    def label(self, index):
        return self.samples[index].label


def _calibration_dict(
    width=4.0,
    height=4.0,
    k1=1.0,
    k2=0.0,
    k3=0.0,
    k4=0.0,
    cx_offset=0.0,
    cy_offset=0.0,
    aspect_ratio=1.0,
    name="FV",
):
    return {
        "intrinsic": {
            "aspect_ratio": aspect_ratio,
            "cx_offset": cx_offset,
            "cy_offset": cy_offset,
            "height": height,
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "k4": k4,
            "model": "radial_poly",
            "poly_order": 4,
            "width": width,
        },
        "extrinsic": {"quaternion": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
        "name": name,
    }


@pytest.fixture
def make_calibration():
    """Factory for a small synthetic Calibration. Override any intrinsic by kwarg."""

    def _make(**overrides):
        return Calibration.from_dict(_calibration_dict(**overrides))

    return _make


@pytest.fixture
def make_sample(make_calibration):
    """Factory for a Sample with a zero image and a constant label."""

    def _make(stem="00000_FV", label_value=1, size=4):
        return Sample(
            image=np.zeros((size, size, 3), dtype=np.uint8),
            label=np.full((size, size), label_value, dtype=np.uint8),
            calibration=make_calibration(),
            stem=stem,
            camera=stem.rsplit("_", 1)[-1],
        )

    return _make


@pytest.fixture
def fake_dataset():
    """Factory building an in-memory dataset from a list of samples."""
    return _FakeDataset


def _write_woodscape_sample(root, stem, size=4):
    rgb = root / "rgb_images"
    label = root / "semantic_annotations" / "gtLabels"
    calibration = root / "calibration"
    for directory in (rgb, label, calibration):
        directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8)).save(rgb / f"{stem}.png")
    Image.fromarray(np.ones((size, size), dtype=np.uint8)).save(label / f"{stem}.png")
    (calibration / f"{stem}.json").write_text(
        json.dumps(_calibration_dict(name=stem.rsplit("_", 1)[-1]))
    )


@pytest.fixture
def woodscape_root(tmp_path):
    """A tiny on-disk WoodScape layout with two samples (FV, RV)."""
    for stem in ("00000_FV", "00001_RV"):
        _write_woodscape_sample(tmp_path, stem)
    return tmp_path
