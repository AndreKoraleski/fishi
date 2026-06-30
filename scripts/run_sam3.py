"""Run SAM 3 over every preprocessing on a WoodScape split, saving one report per cell.

    uv run python scripts/run_sam3.py --data-dir data --metrics-dir metrics --cache-dir cache

Loads a single model, so it fits in memory. Needs HF_TOKEN set (facebook/sam3 is gated).
Re-runnable: finished cells are skipped and predictions are cached.
"""

import argparse

from fishi.evaluation import run
from fishi.preprocess import Identity, Patches, Rectify, TangentImages
from fishi.segmentation import SamThree
from fishi.woodscape import get_settings, load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="WoodScape root")
    parser.add_argument("--metrics-dir", default="metrics", help="where per-cell reports are saved")
    parser.add_argument("--cache-dir", default="cache", help="where predictions are cached")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    dataset = load_split(args.split, get_settings(data_directory=args.data_dir))
    pipeline = SamThree()
    for processor in (Identity(), Rectify(), Patches(), TangentImages()):
        run(
            processor,
            pipeline,
            dataset,
            metrics_directory=args.metrics_dir,
            cache_directory=args.cache_dir,
        )


if __name__ == "__main__":
    main()
