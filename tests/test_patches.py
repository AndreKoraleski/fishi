import numpy as np
import pytest

from fishi.preprocess.patches import _FACES, Patches, _assignment, demonstration


@pytest.fixture
def calibration(make_calibration):
    return make_calibration(width=64.0, height=64.0, k1=20.0)


def test_preprocess_returns_cubemap_faces(calibration):
    views = Patches().preprocess(np.zeros((64, 64, 3), dtype=np.uint8), calibration)
    assert len(views) == len(_FACES) == 5
    assert all(view.shape == (64, 64, 3) for view in views)


def test_front_face_covers_optical_axis(calibration):
    patch_index, _, _ = _assignment(calibration, 64, 64, 64)
    assert patch_index[32, 32] == 0  # the (0, 0) front face is index 0


def test_postprocess_returns_fisheye_size(calibration):
    predictions = [np.zeros((64, 64), dtype=np.uint8) for _ in _FACES]
    assert Patches().postprocess(predictions, calibration).shape == (64, 64)


def test_roundtrip_recovers_patch_assignment(calibration):
    patch_index, _, _ = _assignment(calibration, 64, 64, 64)
    predictions = [np.full((64, 64), k, dtype=np.uint8) for k in range(len(_FACES))]
    result = Patches().postprocess(predictions, calibration)
    covered = patch_index >= 0
    assert np.array_equal(result[covered], patch_index[covered].astype(np.uint8))


def test_demonstration_is_transparent_rgba(calibration):
    demo = demonstration(np.zeros((64, 64, 3), dtype=np.uint8), calibration)
    assert demo.shape[2] == 4
    assert demo[:, :, 3].min() == 0
