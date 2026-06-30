"""Resampling ceiling per preprocessing: the max mIoU achievable with a perfect model.

Round-trips the ground-truth label through each processor (preprocess with nearest-neighbour, then
postprocess back to fisheye) and scores it against the original GT. The gap from 1.0 is the loss
the projection imposes on its own (interpolation artefacts plus any field-of-view dropped),
a ceiling the segmentation model can never beat. CPU only, no model.

Usage:
    uv run python scripts/resampling_ceiling.py --data-dir data --count 200
"""

import argparse

import cv2

from fishi.metrics import SegmentationMetrics
from fishi.preprocess import Identity, Patches, Rectify, TangentImages
from fishi.woodscape import classes
from fishi.woodscape.config import get_settings
from fishi.woodscape.dataset import WoodScapeDataset
from fishi.woodscape.splits import split_datasets

_PROCESSORS = [Identity(), Rectify(), Patches(), TangentImages()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--count", type=int, default=200, help="test samples to average over")
    args = parser.parse_args()

    test = split_datasets(WoodScapeDataset(get_settings(data_directory=args.data_dir)))["test"]
    count = min(args.count, len(test))
    class_count = len(classes.CLASS_NAMES)

    print(f"resampling ceiling over {count} test samples (perfect model):")
    for processor in _PROCESSORS:
        metric = SegmentationMetrics(class_count, ignore_index=classes.VOID_ID)
        for index in range(count):
            sample = test[index]
            tiles = processor.preprocess(
                sample.label, sample.calibration, interpolation=cv2.INTER_NEAREST
            )
            recovered = processor.postprocess(tiles, sample.calibration)
            metric.update(recovered, sample.label)
        result = metric.compute()
        print(f"  {processor.name:8s} ceiling mIoU = {result['miou']:.4f}")


if __name__ == "__main__":
    main()
