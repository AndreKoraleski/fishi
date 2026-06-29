"""Generate preprocessing demonstration images locally.

Usage:
    uv run python scripts/demonstrations.py --data-dir data --output-dir demos --count 3
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from fishi.preprocess import patches, rectify, tangent
from fishi.woodscape.calibration import Calibration
from fishi.woodscape.config import get_settings
from fishi.woodscape.dataset import WoodScapeDataset
from fishi.woodscape.splits import split_datasets

_DEMOS: dict[str, Callable[[np.ndarray, Calibration], np.ndarray]] = {
    "rectify": rectify.demonstration,
    "patches": patches.demonstration,
    "tangent": tangent.demonstration,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="WoodScape root (rgb_images/, ...)")
    parser.add_argument("--output-dir", default="demos")
    parser.add_argument("--count", type=int, default=3, help="number of test samples to render")
    args = parser.parse_args()

    settings = get_settings(data_directory=args.data_dir)
    test = split_datasets(WoodScapeDataset(settings))["test"]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    count = min(args.count, len(test))
    for index in range(count):
        sample = test[index]
        for name, demonstrate in _DEMOS.items():
            rgba = demonstrate(sample.image, sample.calibration)
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(str(output / f"{sample.stem}_{name}.png"), bgra)
    print(f"wrote {count} x {len(_DEMOS)} demos to {output}/")


if __name__ == "__main__":
    main()
