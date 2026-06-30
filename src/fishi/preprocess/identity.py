"""Identity processor: the None baseline (no geometric change)."""

import numpy as np

from fishi.preprocess.base import Processor
from fishi.woodscape.calibration import Calibration


class Identity(Processor):
    """No-op processor: passes the fisheye image through unchanged."""

    name = "none"

    def preprocess(
        self, image: np.ndarray, calibration: Calibration, interpolation: int = 1
    ) -> list[np.ndarray]:
        return [image]

    def postprocess(self, predictions: list[np.ndarray], calibration: Calibration) -> np.ndarray:
        return predictions[0]
