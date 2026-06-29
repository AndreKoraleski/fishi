"""Tangent Images processor: fisheye -> icosahedral gnomonic tangent views, and back.

Tangent Images (Eder et al., CVPR 2020): reproject onto the perspective tangent planes of a
subdivided icosahedron.
"""

from math import degrees, radians, tan

import cv2
import numpy as np

from fishi.preprocess.base import Processor
from fishi.preprocess.visualization import grid, palette, row, to_rgba
from fishi.woodscape.calibration import Calibration

_OVERLAP = 1.1  # widen each tile's FOV past its face so neighbouring tiles overlap (no gaps)


def _subdivide(
    vertices: list[np.ndarray], faces: list[tuple[int, int, int]]
) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    """Split every triangle into four at its edge midpoints, reprojected onto the sphere.

    Parameters
    ----------
    vertices : list of np.ndarray
        Unit vertices of the current mesh; new midpoints are appended in place.
    faces : list of tuple of int
        Vertex-index triples of the current triangles.

    Returns
    -------
    vertices : list of np.ndarray
        Vertices including the new edge midpoints.
    faces : list of tuple of int
        The four-way subdivided triangles.
    """
    cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key not in cache:
            point = vertices[a] + vertices[b]
            vertices.append(point / np.linalg.norm(point))
            cache[key] = len(vertices) - 1
        return cache[key]

    new_faces: list[tuple[int, int, int]] = []
    for a, b, c in faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return vertices, new_faces


def _icosphere(base_level: int) -> tuple[np.ndarray, float]:
    """Face-centre directions of a subdivided icosahedron, with a field of view covering a face.

    Parameters
    ----------
    base_level : int
        Number of subdivisions; level b has 20 * 4**b faces (20 at level 0).

    Returns
    -------
    centers : np.ndarray
        Unit face-centre directions of shape (faces, 3).
    fov_degrees : float
        A per-tile field of view (degrees) that spans a face plus a small overlap.
    """
    phi = (1 + 5**0.5) / 2
    raw = [
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ]  # fmt: skip
    vertices = [np.array(vertex, dtype=float) / np.linalg.norm(vertex) for vertex in raw]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    for _ in range(base_level):
        vertices, faces = _subdivide(vertices, faces)
    centers = []
    radius = 0.0
    for face in faces:
        triangle = np.stack([vertices[index] for index in face])
        center = triangle.mean(axis=0)
        center /= np.linalg.norm(center)
        centers.append(center)
        radius = max(radius, float(np.arccos(np.clip(triangle @ center, -1.0, 1.0)).max()))
    return np.stack(centers), degrees(2 * radius) * _OVERLAP


def _rotation_to(direction: np.ndarray) -> np.ndarray:
    """Orthonormal basis whose +z column points along the given direction.

    The matrix maps a tangent-plane ray to the camera frame (its columns are the tile axes).

    Parameters
    ----------
    direction : np.ndarray
        Target direction (x, y, z) in the camera frame.

    Returns
    -------
    np.ndarray
        Rotation matrix of shape (3, 3).
    """
    z = direction / np.linalg.norm(direction)
    up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def _focal(size: float, fov_degrees: float) -> float:
    """Pinhole focal length (pixels) for a square tile of the given size and field of view."""
    return (size / 2) / tan(radians(fov_degrees) / 2)


class TangentImages(Processor):
    """Reprojects the fisheye onto the gnomonic tangent planes of an icosphere (Tangent Images)."""

    name = "tangent"

    def __init__(
        self,
        base_level: int = 0,
        fov_degrees: float | None = None,
        tile_size: int | None = None,
        max_angle_degrees: float = 100.0,
    ) -> None:
        directions, default_fov = _icosphere(base_level)
        # keep only tiles whose centre is within the fisheye field of view
        keep = directions[:, 2] >= np.cos(radians(max_angle_degrees))
        self.directions = directions[keep]
        self.fov_degrees = fov_degrees if fov_degrees is not None else default_fov
        self.tile_size = tile_size

    def _assign(
        self, calibration: Calibration, height: int, width: int, size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each fisheye pixel pick the most central covering tile, with its (u, v)."""
        focal = _focal(size, self.fov_degrees)
        center = size / 2
        columns, rows = np.meshgrid(np.arange(width), np.arange(height))
        rays_camera = calibration.unproject(np.stack([columns, rows], axis=-1))
        tile_index = np.full((height, width), -1, dtype=int)
        best = np.full((height, width), -1.0)
        u_map = np.zeros((height, width))
        v_map = np.zeros((height, width))
        for k, direction in enumerate(self.directions):
            rays_tile = rays_camera @ _rotation_to(direction)
            x, y, z = rays_tile[..., 0], rays_tile[..., 1], rays_tile[..., 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = focal * x / z + center
                v = focal * y / z + center
            inside = (z > 0) & (u >= 0) & (u < size) & (v >= 0) & (v < size)
            take = inside & (z > best)
            tile_index[take] = k
            best[take] = z[take]
            u_map[take] = u[take]
            v_map[take] = v[take]
        return tile_index, u_map, v_map

    def preprocess(self, image: np.ndarray, calibration: Calibration) -> list[np.ndarray]:
        size = self.tile_size or image.shape[0]
        focal = _focal(size, self.fov_degrees)
        center = size / 2
        columns, rows = np.meshgrid(np.arange(size), np.arange(size))
        rays_tile = np.stack(
            [
                (columns - center) / focal,
                (rows - center) / focal,
                np.ones_like(columns, dtype=float),
            ],
            axis=-1,
        )
        views = []
        for direction in self.directions:
            rays_camera = rays_tile @ _rotation_to(direction).T
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
        size = predictions[0].shape[0]
        tile_index, u_map, v_map = self._assign(calibration, height, width, size)
        result = np.zeros((height, width), dtype=predictions[0].dtype)
        u_index = np.clip(np.rint(u_map), 0, size - 1).astype(int)
        v_index = np.clip(np.rint(v_map), 0, size - 1).astype(int)
        for k, prediction in enumerate(predictions):
            mask = tile_index == k
            result[mask] = prediction[v_index[mask], u_index[mask]]
        return result


def demonstration(
    image: np.ndarray, calibration: Calibration, processor: TangentImages | None = None
) -> np.ndarray:
    """Three panels: input | per-tile coverage (coloured) | the generated tangent tiles."""
    processor = processor or TangentImages()
    size = image.shape[0]
    tile_index, _, _ = processor._assign(calibration, size, image.shape[1], size)
    colors = palette(len(processor.directions))
    overlay = image.copy()
    for k in range(len(processor.directions)):
        overlay[tile_index == k] = (0.5 * overlay[tile_index == k] + 0.5 * colors[k]).astype(
            np.uint8
        )
    tiles = processor.preprocess(image, calibration)
    return row([to_rgba(image), to_rgba(overlay), grid(tiles, columns=4)])
