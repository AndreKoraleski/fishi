"""Run OpenWorldSAM over every preprocessing on a WoodScape split, saving one report per cell.

    uv run python scripts/run_openworldsam.py --config CONFIG.yaml --weights MODEL.pth \
        --repo-path /path/to/OpenWorldSAM --data-dir data --metrics-dir metrics --cache-dir cache

Loads a single model, so it fits in memory. The config, weights, and repo path come from the
OpenWorldSAM repo. Re-runnable: finished cells are skipped and predictions are cached.
"""

import argparse

from fishi.evaluation import run
from fishi.preprocess import Identity, Patches, Rectify, TangentImages
from fishi.segmentation import OpenWorldSam
from fishi.woodscape import get_settings, load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="OpenWorldSAM detectron2 YAML config")
    parser.add_argument("--weights", required=True, help="OpenWorldSAM checkpoint path")
    parser.add_argument("--repo-path", default=None, help="OpenWorldSAM repo root for imports")
    parser.add_argument("--data-dir", default="data", help="WoodScape root")
    parser.add_argument("--metrics-dir", default="metrics", help="where per-cell reports are saved")
    parser.add_argument("--cache-dir", default="cache", help="where predictions are cached")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    dataset = load_split(args.split, get_settings(data_directory=args.data_dir))
    pipeline = OpenWorldSam(args.config, args.weights, repo_path=args.repo_path)
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
