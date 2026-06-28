"""WoodScape fisheye camera calibration: parsing and projection.

The projection maps the incidence angle theta (radians) to image radius rho (pixels) with a
4th-order polynomial:

    rho(theta) = k1*theta + k2*theta**2 + k3*theta**3 + k4*theta**4
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Calibration:
    """A WoodScape per-image fisheye calibration."""

    k1: float
    k2: float
    k3: float
    k4: float
    center_x_offset: float
    center_y_offset: float
    aspect_ratio: float
    width: int
    height: int
    quaternion: tuple[float, float, float, float]
    translation: tuple[float, float, float]
    name: str
    _inverse_lut: tuple[np.ndarray, np.ndarray] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        """Build a Calibration from a parsed WoodScape calibration dict."""
        intrinsic = data["intrinsic"]
        extrinsic = data["extrinsic"]
        return cls(
            k1=intrinsic["k1"],
            k2=intrinsic["k2"],
            k3=intrinsic["k3"],
            k4=intrinsic["k4"],
            center_x_offset=intrinsic["cx_offset"],
            center_y_offset=intrinsic["cy_offset"],
            aspect_ratio=intrinsic["aspect_ratio"],
            width=int(intrinsic["width"]),
            height=int(intrinsic["height"]),
            quaternion=tuple(extrinsic["quaternion"]),
            translation=tuple(extrinsic["translation"]),
            name=data["name"],
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Calibration":
        """Load a Calibration from a WoodScape JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    @property
    def principal_point(self) -> tuple[float, float]:
        """Optical center (cx, cy) in pixels."""
        center_x = 0.5 * self.width + self.center_x_offset - 0.5
        center_y = 0.5 * self.height + self.center_y_offset - 0.5
        return center_x, center_y

    def _rho(self, theta: np.ndarray) -> np.ndarray:
        """Radial polynomial: image radius (pixels) for incidence angle theta."""
        return self.k1 * theta + self.k2 * theta**2 + self.k3 * theta**3 + self.k4 * theta**4

    def project(self, points: np.ndarray) -> np.ndarray:
        """Project 3D camera-frame points onto the fisheye image.

        Parameters
        ----------
        points : np.ndarray
            Points of shape (..., 3) as (x, y, z) in the camera frame.

        Returns
        -------
        np.ndarray
            Pixel coordinates of shape (..., 2) as (u, v).
        """
        points = np.asarray(points, dtype=np.float64)
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        chi = np.sqrt(x**2 + y**2)
        theta = np.arctan2(chi, z)
        rho = self._rho(theta)
        scale = np.divide(rho, chi, out=np.zeros_like(rho), where=chi > 0)
        center_x, center_y = self.principal_point
        u = x * scale + center_x
        v = y * scale * self.aspect_ratio + center_y
        return np.stack([u, v], axis=-1)

    def unproject(self, pixels: np.ndarray) -> np.ndarray:
        """Unproject fisheye pixels to unit ray directions.

        Parameters
        ----------
        pixels : np.ndarray
            Pixel coordinates of shape (..., 2) as (u, v).

        Returns
        -------
        np.ndarray
            Unit rays of shape (..., 3) as (x, y, z) in the camera frame.
        """
        pixels = np.asarray(pixels, dtype=np.float64)
        center_x, center_y = self.principal_point
        u = pixels[..., 0] - center_x
        v = (pixels[..., 1] - center_y) / self.aspect_ratio
        rho = np.sqrt(u**2 + v**2)
        theta = self._theta(rho)
        sin_theta = np.sin(theta)
        scale = np.divide(sin_theta, rho, out=np.zeros_like(rho), where=rho > 0)
        return np.stack([u * scale, v * scale, np.cos(theta)], axis=-1)

    def _theta(self, rho: np.ndarray) -> np.ndarray:
        """Invert the radial polynomial (image radius -> incidence angle) via a LUT."""
        lut = self._inverse_lut
        if lut is None:
            thetas = np.linspace(0.0, np.pi, 4000)
            rhos = self._rho(thetas)
            increasing = np.concatenate(([True], np.diff(rhos) > 0))
            if not increasing.all():
                cut = int(np.argmin(increasing))
                thetas, rhos = thetas[:cut], rhos[:cut]
            lut = (rhos, thetas)
            self._inverse_lut = lut
        rhos, thetas = lut
        return np.interp(rho, rhos, thetas)
