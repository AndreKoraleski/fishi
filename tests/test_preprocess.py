import numpy as np

from fishi.preprocess import Identity, ProcessedSample
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


def _sample(stem="00000_FV"):
    return Sample(
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        label=np.ones((4, 4), dtype=np.uint8),
        calibration=CALIB,
        stem=stem,
        camera=stem.rsplit("_", 1)[-1],
    )


def test_identity_wrap_yields_single_view():
    dataset = Identity().wrap(FakeDataset([_sample(), _sample("00001_RV")]))
    assert len(dataset) == 2
    processed = dataset[0]
    assert isinstance(processed, ProcessedSample)
    assert len(processed.views) == 1
    assert np.array_equal(processed.views[0], np.zeros((4, 4, 3), dtype=np.uint8))
    assert processed.stem == "00000_FV"
    assert processed.camera == "FV"


def test_identity_postprocess_is_identity():
    prediction = np.array([[0, 1], [2, 0]])
    assert np.array_equal(Identity().postprocess([prediction], CALIB), prediction)
