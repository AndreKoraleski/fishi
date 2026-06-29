"""Rectify processor: fisheye image <-> a central virtual pinhole view.

Fixed to a 90-degree pinhole (the central cubemap face), output at the input size.
"""

from math import radians, tan

import cv2
import numpy as np

from fishi.preprocess.base import Processor
from fishi.preprocess.visualization import row, to_rgba
from fishi.woodscape.calibration import Calibration

_FOV_DEGREES = 90.0


def _focal(width: float) -> float:
    return (width / 2) / tan(radians(_FOV_DEGREES) / 2)


def _forward_maps(
    calibration: Calibration, out_height: int, out_width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Map each pinhole pixel to the fisheye pixel it samples from."""
    focal = _focal(out_width)
    center_x, center_y = out_width / 2, out_height / 2
    columns, rows = np.meshgrid(np.arange(out_width), np.arange(out_height))
    rays = np.stack(
        [
            (columns - center_x) / focal,
            (rows - center_y) / focal,
            np.ones_like(columns, dtype=float),
        ],
        axis=-1,
    )
    fisheye = calibration.project(rays)
    return fisheye[..., 0].astype(np.float32), fisheye[..., 1].astype(np.float32)


def _inverse_maps(
    calibration: Calibration,
    fisheye_height: int,
    fisheye_width: int,
    out_height: int,
    out_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each fisheye pixel to the pinhole pixel whose prediction it takes."""
    focal = _focal(out_width)
    center_x, center_y = out_width / 2, out_height / 2
    columns, rows = np.meshgrid(np.arange(fisheye_width), np.arange(fisheye_height))
    rays = calibration.unproject(np.stack([columns, rows], axis=-1))
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        map_x = focal * x / z + center_x
        map_y = focal * y / z + center_y
    behind = z <= 0  # rays past 180 degrees have no pinhole pixel
    map_x[behind] = -1.0
    map_y[behind] = -1.0
    return map_x.astype(np.float32), map_y.astype(np.float32)


class Rectify(Processor):
    """Rectifies the fisheye to a central 90-degree pinhole view."""

    name = "rectify"

    def preprocess(self, image: np.ndarray, calibration: Calibration) -> list[np.ndarray]:
        height, width = image.shape[:2]
        map_x, map_y = _forward_maps(calibration, height, width)
        return [cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)]

    def postprocess(self, predictions: list[np.ndarray], calibration: Calibration) -> np.ndarray:
        prediction = predictions[0]
        output_height, output_width = prediction.shape[:2]
        fisheye_height, fisheye_width = int(calibration.height), int(calibration.width)
        map_x, map_y = _inverse_maps(
            calibration, fisheye_height, fisheye_width, output_height, output_width
        )
        remapped = cv2.remap(prediction.astype(np.int16), map_x, map_y, cv2.INTER_NEAREST)
        return remapped.astype(prediction.dtype)


def demonstration(image: np.ndarray, calibration: Calibration) -> np.ndarray:
    """Three panels: input | the fisheye zone the view keeps (rest black) | the rectified view."""
    height, width = image.shape[:2]
    inverse_x, inverse_y = _inverse_maps(calibration, height, width, height, width)
    covered = (inverse_x >= 0) & (inverse_x < width) & (inverse_y >= 0) & (inverse_y < height)
    zone = np.zeros_like(image)
    zone[covered] = image[covered]
    forward_x, forward_y = _forward_maps(calibration, height, width)
    rectified = cv2.remap(image, forward_x, forward_y, cv2.INTER_LINEAR)
    return row([to_rgba(image), to_rgba(zone), to_rgba(rectified)])
