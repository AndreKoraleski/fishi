"""Tangent Images processor: fisheye to icosahedral gnomonic tangent views, and back.

Tangent Images (Eder et al., CVPR 2020): reproject onto the perspective tangent planes of a
subdivided icosahedron. Level 0 is the bare icosahedron (20 faces). Each level splits every
triangle into four. Only the tiles whose centre falls within the fisheye field of view are kept.
"""

from math import degrees, radians

import numpy as np

from fishi.preprocess.gnomonic import GnomonicMultiView, coverage_demonstration
from fishi.woodscape.calibration import Calibration

OVERLAP = 1.1  # widen each tile's FOV past its face so neighbouring tiles overlap (no gaps)


def subdivide(
    vertices: list[np.ndarray], faces: list[tuple[int, int, int]]
) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    """Split every triangle into four at its edge midpoints, reprojected onto the sphere.

    Parameters
    ----------
    vertices : list of np.ndarray
        Unit vertices of the current mesh. New midpoints are appended in place.
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


def icosphere(base_level: int) -> tuple[np.ndarray, float]:
    """Face-centre directions of a subdivided icosahedron, with a field of view covering a face.

    Parameters
    ----------
    base_level : int
        Number of subdivisions. Level b has 20 * 4**b faces (20 at level 0).

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
        vertices, faces = subdivide(vertices, faces)
    centers = []
    radius = 0.0
    for face in faces:
        triangle = np.stack([vertices[index] for index in face])
        center = triangle.mean(axis=0)
        center /= np.linalg.norm(center)
        centers.append(center)
        radius = max(radius, float(np.arccos(np.clip(triangle @ center, -1.0, 1.0)).max()))
    return np.stack(centers), degrees(2 * radius) * OVERLAP


def rotation_to(direction: np.ndarray) -> np.ndarray:
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


class TangentImages(GnomonicMultiView):
    """Reprojects the fisheye onto the gnomonic tangent planes of an icosphere (Tangent Images)."""

    name = "tangent"

    def __init__(
        self,
        base_level: int = 0,
        fov_degrees: float | None = None,
        view_size: int | None = None,
        max_angle_degrees: float = 100.0,
    ) -> None:
        """Place the tangent tiles.

        Parameters
        ----------
        base_level : int
            Icosahedron subdivisions. Level 0 gives 20 faces before filtering.
        fov_degrees : float, optional
            Per-tile field of view. Defaults to a value that spans a face plus a small overlap.
        view_size : int, optional
            Output view side in pixels. Defaults to the input image height.
        max_angle_degrees : float
            Keep only tiles whose centre lies within this angle of the forward axis. The default
            of 100 covers a WoodScape fisheye (about 190 degrees, so a 95-degree half-angle) with
            a small margin, and drops the rear-facing tiles that capture nothing.
        """
        directions, default_fov = icosphere(base_level)
        keep = directions[:, 2] >= np.cos(radians(max_angle_degrees))
        self.directions = directions[keep]
        self.fov_degrees = fov_degrees if fov_degrees is not None else default_fov
        self.view_size = view_size
        self.rotations = [rotation_to(direction) for direction in self.directions]


def demonstration(
    image: np.ndarray, calibration: Calibration, processor: TangentImages | None = None
) -> np.ndarray:
    """Three panels: input, per-tile coverage (coloured), and the generated tangent tiles."""
    return coverage_demonstration(image, calibration, processor or TangentImages(), columns=4)
