"""Patches processor: fisheye -> cubemap perspective faces, and back.

Fixed to the standard cubemap layout (front, left, right, up, down faces at 90 degrees).
"""

from math import cos, radians, sin, tan

import cv2
import numpy as np

from fishi.preprocess.base import Processor
from fishi.preprocess.visualization import side_by_side
from fishi.woodscape.calibration import Calibration

_FOV_DEGREES = 90.0
_FACES = [(0.0, 0.0), (-90.0, 0.0), (90.0, 0.0), (0.0, -90.0), (0.0, 90.0)]


def _focal(face_size: float) -> float:
    return (face_size / 2) / tan(radians(_FOV_DEGREES) / 2)


def _rotation(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    """Rotation taking a face-frame ray to the camera frame (yaw then pitch)."""
    yaw, pitch = radians(yaw_degrees), radians(pitch_degrees)
    rotation_yaw = np.array([[cos(yaw), 0, sin(yaw)], [0, 1, 0], [-sin(yaw), 0, cos(yaw)]])
    rotation_pitch = np.array(
        [[1, 0, 0], [0, cos(pitch), -sin(pitch)], [0, sin(pitch), cos(pitch)]]
    )
    return rotation_yaw @ rotation_pitch


def _assignment(
    calibration: Calibration, height: int, width: int, face_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each fisheye pixel pick the most central covering cubemap face."""
    focal = _focal(face_size)
    center = face_size / 2
    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    rays_camera = calibration.unproject(np.stack([columns, rows], axis=-1))
    patch_index = np.full((height, width), -1, dtype=int)
    best = np.full((height, width), -1.0)
    u_map = np.zeros((height, width))
    v_map = np.zeros((height, width))
    for k, (yaw, pitch) in enumerate(_FACES):
        rays_face = rays_camera @ _rotation(yaw, pitch)
        x, y, z = rays_face[..., 0], rays_face[..., 1], rays_face[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = focal * x / z + center
            v = focal * y / z + center
        inside = (z > 0) & (u >= 0) & (u < face_size) & (v >= 0) & (v < face_size)
        take = inside & (z > best)
        patch_index[take] = k
        best[take] = z[take]
        u_map[take] = u[take]
        v_map[take] = v[take]
    return patch_index, u_map, v_map


class Patches(Processor):
    """Reprojects the fisheye into the standard cubemap faces (90-degree views)."""

    name = "patches"

    def preprocess(self, image: np.ndarray, calibration: Calibration) -> list[np.ndarray]:
        face_size = image.shape[0]  # square faces sized to the input height
        focal = _focal(face_size)
        center = face_size / 2
        columns, rows = np.meshgrid(np.arange(face_size), np.arange(face_size))
        rays_face = np.stack(
            [
                (columns - center) / focal,
                (rows - center) / focal,
                np.ones_like(columns, dtype=float),
            ],
            axis=-1,
        )
        views = []
        for yaw, pitch in _FACES:
            rays_camera = rays_face @ _rotation(yaw, pitch).T
            fisheye = calibration.project(rays_camera)
            views.append(
                cv2.remap(
                    image,
                    fisheye[..., 0].astype(np.float32),
                    fisheye[..., 1].astype(np.float32),
                    cv2.INTER_LINEAR,
                )
            )
        return views

    def postprocess(self, predictions: list[np.ndarray], calibration: Calibration) -> np.ndarray:
        height, width = int(calibration.height), int(calibration.width)
        face_size = predictions[0].shape[0]
        patch_index, u_map, v_map = _assignment(calibration, height, width, face_size)
        result = np.zeros((height, width), dtype=predictions[0].dtype)
        u_index = np.clip(np.rint(u_map), 0, face_size - 1).astype(int)
        v_index = np.clip(np.rint(v_map), 0, face_size - 1).astype(int)
        for k, prediction in enumerate(predictions):
            mask = patch_index == k
            result[mask] = prediction[v_index[mask], u_index[mask]]
        return result


def _palette(count: int) -> np.ndarray:
    """`count` visually distinct RGB colors."""
    hues = np.linspace(0, 179, count, endpoint=False).astype(np.uint8)
    saturation = np.full(count, 255, dtype=np.uint8)
    hsv = np.stack([hues, saturation, saturation], axis=-1).reshape(count, 1, 3)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(count, 3)


def demonstration(image: np.ndarray, calibration: Calibration) -> np.ndarray:
    """Side-by-side [input | per-face colored coverage] as transparent RGBA."""
    patch_index, _, _ = _assignment(calibration, image.shape[0], image.shape[1], image.shape[0])
    palette = _palette(len(_FACES))
    overlay = image.copy()
    for k in range(len(_FACES)):
        overlay[patch_index == k] = (0.5 * overlay[patch_index == k] + 0.5 * palette[k]).astype(
            np.uint8
        )
    alpha = (patch_index >= 0).astype(np.uint8) * 255
    return side_by_side(image, overlay, alpha)
