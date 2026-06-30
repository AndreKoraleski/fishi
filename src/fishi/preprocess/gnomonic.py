"""Shared machinery for multi-view gnomonic processors (cubemap faces and Tangent Images)."""

from math import radians, tan

import cv2
import numpy as np

from fishi.preprocess.base import Processor
from fishi.preprocess.visualization import grid, palette, row, to_rgba
from fishi.woodscape.calibration import Calibration


def focal(size: float, fov_degrees: float) -> float:
    """Pinhole focal length (pixels) for a square view of the given size and field of view."""
    return (size / 2) / tan(radians(fov_degrees) / 2)


class GnomonicMultiView(Processor):
    """Reproject the fisheye onto a set of gnomonic tangent views, and merge predictions back.

    Attributes
    ----------
    rotations : list of np.ndarray
        Per-view rotation matrices (view frame to camera frame), shape (3, 3) each.
    fov_degrees : float
        Field of view of each square view.
    view_size : int or None
        Output view side in pixels. None matches the input image height.
    """

    rotations: list[np.ndarray]
    fov_degrees: float
    view_size: int | None = None

    def preprocess(
        self, image: np.ndarray, calibration: Calibration, interpolation: int = cv2.INTER_LINEAR
    ) -> list[np.ndarray]:
        size = self.view_size or image.shape[0]
        focal_px = focal(size, self.fov_degrees)
        center = size / 2
        columns, rows = np.meshgrid(np.arange(size), np.arange(size))
        rays_view = np.stack(
            [
                (columns - center) / focal_px,
                (rows - center) / focal_px,
                np.ones_like(columns, dtype=float),
            ],
            axis=-1,
        )
        views = []
        for rotation in self.rotations:
            rays_camera = rays_view @ rotation.T
            fisheye = calibration.project(rays_camera)
            views.append(
                cv2.remap(
                    image,
                    fisheye[..., 0].astype(np.float32),
                    fisheye[..., 1].astype(np.float32),
                    interpolation,
                )
            )
        return views

    def postprocess(self, predictions: list[np.ndarray], calibration: Calibration) -> np.ndarray:
        height, width = int(calibration.height), int(calibration.width)
        size = predictions[0].shape[0]
        view_index, u_map, v_map = self.assign(calibration, height, width, size)
        # Pixels no view covers stay 0. For WoodScape that is void (the ignore class), so they
        # score as misses rather than as a real class.
        result = np.zeros((height, width), dtype=predictions[0].dtype)
        u_index = np.clip(np.rint(u_map), 0, size - 1).astype(int)
        v_index = np.clip(np.rint(v_map), 0, size - 1).astype(int)
        for k, prediction in enumerate(predictions):
            mask = view_index == k
            result[mask] = prediction[v_index[mask], u_index[mask]]
        return result

    def assign(
        self, calibration: Calibration, height: int, width: int, size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each fisheye pixel pick the most central covering view, with its (u, v).

        Returns
        -------
        view_index : np.ndarray
            Index of the chosen view per pixel, shape (height, width). Minus one where uncovered.
        u_map, v_map : np.ndarray
            Sampling coordinates in the chosen view, shape (height, width) each.
        """
        focal_pixels = focal(size, self.fov_degrees)
        center = size / 2
        columns, rows = np.meshgrid(np.arange(width), np.arange(height))
        rays_camera = calibration.unproject(np.stack([columns, rows], axis=-1))
        view_index = np.full((height, width), -1, dtype=int)
        best = np.full((height, width), -1.0)
        u_map = np.zeros((height, width))
        v_map = np.zeros((height, width))
        for k, rotation in enumerate(self.rotations):
            rays_view = rays_camera @ rotation
            x, y, z = rays_view[..., 0], rays_view[..., 1], rays_view[..., 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = focal_pixels * x / z + center
                v = focal_pixels * y / z + center
            inside = (z > 0) & (u >= 0) & (u < size) & (v >= 0) & (v < size)
            take = inside & (z > best)
            view_index[take] = k
            best[take] = z[take]
            u_map[take] = u[take]
            v_map[take] = v[take]
        return view_index, u_map, v_map


def coverage_demonstration(
    image: np.ndarray, calibration: Calibration, processor: GnomonicMultiView, columns: int
) -> np.ndarray:
    """Three panels: input, per-view coverage (coloured), and the generated views in a grid."""
    size = image.shape[0]
    view_index, _, _ = processor.assign(calibration, size, image.shape[1], size)
    colors = palette(len(processor.rotations))
    overlay = image.copy()
    for k in range(len(processor.rotations)):
        overlay[view_index == k] = (0.5 * overlay[view_index == k] + 0.5 * colors[k]).astype(
            np.uint8
        )
    views = processor.preprocess(image, calibration)
    return row([to_rgba(image), to_rgba(overlay), grid(views, columns=columns)])
