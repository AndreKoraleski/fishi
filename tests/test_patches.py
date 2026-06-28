import numpy as np

from fishi.preprocess.patches import _FACES, Patches, _assignment, demonstration
from fishi.woodscape.calibration import Calibration

CALIB = Calibration.from_dict(
    {
        "intrinsic": {
            "aspect_ratio": 1.0,
            "cx_offset": 0.0,
            "cy_offset": 0.0,
            "height": 64.0,
            "k1": 20.0,
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "model": "radial_poly",
            "poly_order": 4,
            "width": 64.0,
        },
        "extrinsic": {"quaternion": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
        "name": "FV",
    }
)


def test_preprocess_returns_cubemap_faces():
    views = Patches().preprocess(np.zeros((64, 64, 3), dtype=np.uint8), CALIB)
    assert len(views) == len(_FACES) == 5
    assert all(view.shape == (64, 64, 3) for view in views)


def test_front_face_covers_optical_axis():
    patch_index, _, _ = _assignment(CALIB, 64, 64, 64)
    assert patch_index[32, 32] == 0  # the (0, 0) front face is index 0


def test_postprocess_returns_fisheye_size():
    predictions = [np.zeros((64, 64), dtype=np.uint8) for _ in _FACES]
    assert Patches().postprocess(predictions, CALIB).shape == (64, 64)


def test_roundtrip_recovers_patch_assignment():
    patch_index, _, _ = _assignment(CALIB, 64, 64, 64)
    predictions = [np.full((64, 64), k, dtype=np.uint8) for k in range(len(_FACES))]
    result = Patches().postprocess(predictions, CALIB)
    covered = patch_index >= 0
    assert np.array_equal(result[covered], patch_index[covered].astype(np.uint8))


def test_demonstration_is_transparent_rgba():
    demo = demonstration(np.zeros((64, 64, 3), dtype=np.uint8), CALIB)
    assert demo.shape[2] == 4
    assert demo[:, :, 3].min() == 0
