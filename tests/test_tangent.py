import numpy as np
import pytest

from fishi.preprocess.tangent import TangentImages, demonstration, icosphere, rotation_to


@pytest.fixture
def calibration(make_calibration):
    return make_calibration(width=64.0, height=64.0, k1=20.0)


def test_icosphere_level0_has_20_faces():
    centers, fov = icosphere(0)
    assert centers.shape == (20, 3)
    np.testing.assert_allclose(np.linalg.norm(centers, axis=1), 1.0)
    assert 70 < fov < 90  # ~75 degrees of face coverage plus overlap


def test_subdivision_multiplies_faces():
    assert icosphere(1)[0].shape[0] == 80  # 20 * 4


def test_rotation_to_maps_z_to_direction():
    direction = np.array([0.3, -0.5, 0.8])
    direction /= np.linalg.norm(direction)
    matrix = rotation_to(direction)
    np.testing.assert_allclose(matrix @ np.array([0.0, 0.0, 1.0]), direction, atol=1e-9)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-9)


def test_rotation_to_handles_near_pole_up():
    matrix = rotation_to(np.array([0.0, 1.0, 0.0]))  # forces the alternate up vector
    np.testing.assert_allclose(matrix @ np.array([0.0, 0.0, 1.0]), [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-9)


def test_preprocess_returns_one_view_per_visible_tile(calibration):
    processor = TangentImages()
    views = processor.preprocess(np.zeros((64, 64, 3), dtype=np.uint8), calibration)
    assert len(views) == len(processor.directions)
    assert 5 < len(views) <= 20  # more tiles than the cubemap, at most the full icosphere
    assert all(view.shape == (64, 64, 3) for view in views)


def test_view_size_overrides_resolution(calibration):
    views = TangentImages(view_size=32).preprocess(np.zeros((64, 64, 3), np.uint8), calibration)
    assert all(view.shape == (32, 32, 3) for view in views)


def test_postprocess_returns_fisheye_size(calibration):
    processor = TangentImages()
    predictions = [np.zeros((64, 64), dtype=np.uint8) for _ in processor.directions]
    assert processor.postprocess(predictions, calibration).shape == (64, 64)


def test_roundtrip_recovers_tile_assignment(calibration):
    processor = TangentImages()
    index, _, _ = processor.assign(calibration, 64, 64, 64)
    predictions = [np.full((64, 64), k, dtype=np.uint8) for k in range(len(processor.directions))]
    result = processor.postprocess(predictions, calibration)
    covered = index >= 0
    assert np.array_equal(result[covered], index[covered].astype(np.uint8))


def test_demonstration_is_transparent_rgba(calibration):
    demo = demonstration(np.zeros((64, 64, 3), dtype=np.uint8), calibration)
    assert demo.shape[2] == 4
    assert demo[:, :, 3].min() == 0
