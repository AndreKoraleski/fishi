"""Patches processor: fisheye to cubemap perspective faces, and back.

The standard cubemap layout minus the rear face: front, left, right, up, and down at 90 degrees.
The rear face is dropped because a forward-facing fisheye sees at most a hemisphere, so nothing
projects onto it.
"""

from math import cos, radians, sin

import numpy as np

from fishi.preprocess.gnomonic import GnomonicMultiView, coverage_demonstration
from fishi.woodscape.calibration import Calibration

FACES = [(0.0, 0.0), (-90.0, 0.0), (90.0, 0.0), (0.0, -90.0), (0.0, 90.0)]


def rotation(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    """Rotation taking a face-frame ray to the camera frame (yaw then pitch)."""
    yaw, pitch = radians(yaw_degrees), radians(pitch_degrees)
    rotation_yaw = np.array([[cos(yaw), 0, sin(yaw)], [0, 1, 0], [-sin(yaw), 0, cos(yaw)]])
    rotation_pitch = np.array(
        [[1, 0, 0], [0, cos(pitch), -sin(pitch)], [0, sin(pitch), cos(pitch)]]
    )
    return rotation_yaw @ rotation_pitch


class Patches(GnomonicMultiView):
    """Reprojects the fisheye into the standard cubemap faces (90-degree views)."""

    name = "patches"
    fov_degrees = 90.0

    def __init__(self) -> None:
        self.rotations = [rotation(yaw, pitch) for yaw, pitch in FACES]


def demonstration(image: np.ndarray, calibration: Calibration) -> np.ndarray:
    """Three panels: input, per-face coverage (coloured), and the generated cubemap faces."""
    return coverage_demonstration(image, calibration, Patches(), columns=3)
