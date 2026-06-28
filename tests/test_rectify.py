import numpy as np

from fishi.preprocess.rectify import Rectify, _forward_maps, demonstration
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


def test_preprocess_returns_one_view_at_input_size():
    views = Rectify().preprocess(np.zeros((64, 64, 3), dtype=np.uint8), CALIB)
    assert len(views) == 1
    assert views[0].shape == (64, 64, 3)


def test_center_samples_principal_point():
    map_x, map_y = _forward_maps(CALIB, 64, 64)
    center_x, center_y = CALIB.principal_point
    assert abs(map_x[32, 32] - center_x) < 1.0
    assert abs(map_y[32, 32] - center_y) < 1.0


def test_postprocess_returns_fisheye_size():
    assert Rectify().postprocess([np.zeros((64, 64), dtype=np.uint8)], CALIB).shape == (64, 64)


def test_uniform_image_rectifies_to_same_color():
    image = np.full((64, 64, 3), 200, dtype=np.uint8)
    assert Rectify().preprocess(image, CALIB)[0][32, 32].tolist() == [200, 200, 200]


def test_demonstration_is_transparent_rgba():
    demo = demonstration(np.zeros((64, 64, 3), dtype=np.uint8), CALIB)
    assert demo.shape[2] == 4
    assert demo[:, :, 3].min() == 0
