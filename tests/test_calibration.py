import json

import numpy as np

from fishi.woodscape.calibration import Calibration

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
    # rho(theta) = theta - theta**2 is non-monotonic, so the inverse LUT must
    # be trimmed to its increasing prefix.
    calibration = make_calibration(k1=1.0, k2=-1.0, width=8.0, height=8.0)
    rays = calibration.unproject(np.array([[4.0, 4.0], [0.0, 0.0]]))
    assert rays.shape == (2, 3)
