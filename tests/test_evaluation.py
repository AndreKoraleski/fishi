import numpy as np

from fishi.evaluation import evaluate
from fishi.preprocess import Identity
from fishi.woodscape.calibration import Calibration
from fishi.woodscape.dataset import Sample

CALIB = Calibration.from_dict(
    {
        "intrinsic": {
            "aspect_ratio": 1.0,
            "cx_offset": 0.0,
            "cy_offset": 0.0,
            "height": 4.0,
            "k1": 1.0,
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "model": "radial_poly",
            "poly_order": 4,
            "width": 4.0,
        },
        "extrinsic": {"quaternion": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
        "name": "FV",
    }
)


class FakeDataset:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class DummyPipeline:
    name = "dummy"

    def predict(self, image, prompts):
        return np.zeros(image.shape[:2], dtype=np.uint8)


def _sample():
    return Sample(
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        label=np.ones((4, 4), dtype=np.uint8),  # all class 1 (road)
        calibration=CALIB,
        stem="00000_FV",
        camera="FV",
    )


def test_evaluate_runs_and_returns_metrics():
    dataset = Identity().wrap(FakeDataset([_sample(), _sample()]))
    result = evaluate(dataset, DummyPipeline(), prompts={1: "road"}, class_count=10)
    assert {"iou", "dice", "miou", "mdice"} <= set(result)
    assert result["miou"] == 0.0  # all-void predictions vs road labels
