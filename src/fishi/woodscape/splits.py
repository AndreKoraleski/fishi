"""Canonical train/val/test split of WoodScape (70/15/15)."""

import json
from collections.abc import Iterable
from importlib.resources import files

import numpy as np
import structlog

from fishi.woodscape.dataset import Subset, WoodScapeDataset

logger = structlog.get_logger(__name__)

SPLIT_SEED = 0
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
_SPLIT_FILE = "canonical_split.json"


def make_split(
    stems: Iterable[str],
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
    seed: int = SPLIT_SEED,
) -> dict[str, list[str]]:
    """Deterministically split stems into train/val/test by the given fractions."""
    ordered = sorted(stems)
    permutation = np.random.RandomState(seed).permutation(len(ordered))
    shuffled = [ordered[index] for index in permutation]
    count = len(shuffled)
    train_end = round(fractions[0] * count)
    val_end = train_end + round(fractions[1] * count)
    return {
        "train": sorted(shuffled[:train_end]),
        "val": sorted(shuffled[train_end:val_end]),
        "test": sorted(shuffled[val_end:]),
    }


def load_canonical_split() -> dict[str, list[str]]:
    """Load the committed canonical 70/15/15 split."""
    return json.loads(files("fishi.woodscape").joinpath(_SPLIT_FILE).read_text())


def split_datasets(
    dataset: WoodScapeDataset, split: dict[str, list[str]] | None = None
) -> dict[str, Subset]:
    """Build one Subset per split key from a dataset, matching samples by stem.

    Stems missing from the dataset are dropped (with a warning) rather than failing.
    """
    split = split or load_canonical_split()
    index_of = {stem: index for index, stem in enumerate(dataset.stems)}
    subsets: dict[str, Subset] = {}
    for name, stems in split.items():
        present = [index_of[stem] for stem in stems if stem in index_of]
        dropped = len(stems) - len(present)
        if dropped:
            logger.warning(
                "split stems missing from dataset",
                split=name,
                dropped=dropped,
                requested=len(stems),
            )
        subsets[name] = Subset(dataset, present)
    return subsets
