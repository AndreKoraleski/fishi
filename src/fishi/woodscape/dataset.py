"""WoodScape dataset loader: pairs images, labels, and calibration by stem."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog
from PIL import Image

from fishi.woodscape.calibration import Calibration
from fishi.woodscape.config import Settings, get_settings

logger = structlog.get_logger(__name__)

RGB_DIRECTORY = "rgb_images"
LABEL_DIRECTORY = "semantic_annotations/gtLabels"
CALIBRATION_DIRECTORY = "calibration"


@dataclass
class Sample:
    """One WoodScape sample.

    Attributes
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3), uint8.
    label : np.ndarray
        Class-id segmentation mask of shape (H, W).
    calibration : Calibration
        Fisheye calibration for this image.
    stem : str
        Filename stem, e.g. "00015_FV".
    camera : str
        Camera id parsed from the stem (FV, RV, MVL, MVR).
    """

    image: np.ndarray
    label: np.ndarray
    calibration: Calibration
    stem: str
    camera: str


class WoodScapeDataset:
    """Pairs WoodScape images, segmentation labels, and calibration by stem.

    Only samples with all three files are kept. An image missing its label or calibration is
    skipped (with a warning), so iteration never hits a missing file mid-run.

    Parameters
    ----------
    settings : Settings, optional
        Project settings. The dataset root is settings.data_directory. Loaded from the environment
        when omitted.
    cameras : sequence of str, optional
        Keep only these cameras (FV, RV, MVL, MVR). All cameras when omitted.

    Raises
    ------
    FileNotFoundError
        If the images directory is absent (run download_woodscape(settings) first).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        cameras: Sequence[str] | None = None,
    ) -> None:
        root: Path = (settings or get_settings()).data_directory
        self._rgb_directory = root / RGB_DIRECTORY
        self._label_directory = root / LABEL_DIRECTORY
        self._calibration_directory = root / CALIBRATION_DIRECTORY
        if not self._rgb_directory.is_dir():
            raise FileNotFoundError(
                f"{self._rgb_directory} not found. Run download_woodscape(settings) first"
            )

        rgb = {path.stem for path in self._rgb_directory.glob("*.png")}
        labelled = {path.stem for path in self._label_directory.glob("*.png")}
        calibrated = {path.stem for path in self._calibration_directory.glob("*.json")}
        complete = rgb & labelled & calibrated
        incomplete = rgb - complete
        if incomplete:
            logger.warning(
                "incomplete_samples_skipped",
                skipped=len(incomplete),
                total=len(rgb),
            )

        stems = sorted(complete)
        if cameras is not None:
            keep = set(cameras)
            stems = [stem for stem in stems if stem.rsplit("_", 1)[-1] in keep]
        self.stems = stems

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> Sample:
        stem = self.stems[index]
        image = np.array(Image.open(self._rgb_directory / f"{stem}.png").convert("RGB"))
        label = np.array(Image.open(self._label_directory / f"{stem}.png"))
        calibration = Calibration.from_json(self._calibration_directory / f"{stem}.json")
        return Sample(
            image=image,
            label=label,
            calibration=calibration,
            stem=stem,
            camera=stem.rsplit("_", 1)[-1],
        )

    def stem(self, index: int) -> str:
        """Filename stem for a sample, without reading any file."""
        return self.stems[index]

    def label(self, index: int) -> np.ndarray:
        """Load only the class-id label for a sample, skipping the RGB image."""
        return np.array(Image.open(self._label_directory / f"{self.stems[index]}.png"))


class Subset:
    """A view over selected indices of a dataset (e.g. one split)."""

    def __init__(self, dataset: WoodScapeDataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Sample:
        return self.dataset[self.indices[index]]

    def stem(self, index: int) -> str:
        return self.dataset.stem(self.indices[index])

    def label(self, index: int) -> np.ndarray:
        return self.dataset.label(self.indices[index])
