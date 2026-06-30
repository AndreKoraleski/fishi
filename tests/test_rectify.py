import numpy as np
import pytest

from fishi.preprocess.rectify import Rectify, demonstration, forward_maps


@pytest.fixture
def calibration(make_calibration):
    return make_calibration(width=64.0, height=64.0, k1=20.0)


def test_preprocess_returns_one_view_at_input_size(calibration):
    views = Rectify().preprocess(np.zeros((64, 64, 3), dtype=np.uint8), calibration)
    assert len(views) == 1
    assert views[0].shape == (64, 64, 3)


def test_center_samples_principal_point(calibration):
    map_x, map_y = forward_maps(calibration, 64, 64)
    center_x, center_y = calibration.principal_point
    assert abs(map_x[32, 32] - center_x) < 1.0
    assert abs(map_y[32, 32] - center_y) < 1.0


def test_postprocess_returns_fisheye_size(calibration):
    assert Rectify().postprocess([np.zeros((64, 64), dtype=np.uint8)], calibration).shape == (
        64,
        64,
    )


def test_uniform_image_rectifies_to_same_color(calibration):
    image = np.full((64, 64, 3), 200, dtype=np.uint8)
    assert Rectify().preprocess(image, calibration)[0][32, 32].tolist() == [200, 200, 200]


def test_demonstration_is_transparent_rgba(calibration):
    demo = demonstration(np.zeros((64, 64, 3), dtype=np.uint8), calibration)
    assert demo.shape[2] == 4
    assert demo[:, :, 3].min() == 0
