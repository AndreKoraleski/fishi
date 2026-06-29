"""Train/validation/test splitting: the canonical 70/15/15 partition and helpers to apply it."""

import json
from collections.abc import Iterable, Sequence
from importlib.resources import files

import numpy as np
import structlog

from fishi.woodscape.config import Settings
from fishi.woodscape.dataset import Subset, WoodScapeDataset

logger = structlog.get_logger(__name__)

SPLIT_SEED = 0
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
SPLITS_FILE = "splits.json"


def make_split(
    stems: Iterable[str],
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
    seed: int = SPLIT_SEED,
) -> dict[str, list[str]]:
    """Deterministically partition stems into train/validation/test.

    Parameters
    ----------
    stems : iterable of str
        Sample stems to partition.
    fractions : tuple of float
        (train, validation, test) fractions that should sum to 1.
    seed : int
        Seed for the shuffle, for reproducibility.

    Returns
    -------
    dict of str to list of str
        Sorted stems under "train", "validation", and "test".
    """
    ordered = sorted(stems)
    permutation = np.random.RandomState(seed).permutation(len(ordered))
    shuffled = [ordered[index] for index in permutation]
    count = len(shuffled)
    train_end = round(fractions[0] * count)
    validation_end = train_end + round(fractions[1] * count)
    return {
        "train": sorted(shuffled[:train_end]),
        "validation": sorted(shuffled[train_end:validation_end]),
        "test": sorted(shuffled[validation_end:]),
    }


def canonical_split() -> dict[str, list[str]]:
    """Load the committed canonical 70/15/15 split (stem lists per split name)."""
    return json.loads(files("fishi.woodscape").joinpath(SPLITS_FILE).read_text())


def split_datasets(
    dataset: WoodScapeDataset, split: dict[str, list[str]] | None = None
) -> dict[str, Subset]:
    """Build one Subset per split key, matching samples to the dataset by stem.

    Stems missing from the dataset are dropped (with a warning) rather than failing.

    Parameters
    ----------
    dataset : WoodScapeDataset
        The dataset to slice.
    split : dict of str to list of str, optional
        Stem lists per split name. Defaults to the canonical split.

    Returns
    -------
    dict of str to Subset
        One Subset per split name.
    """
    split = split or canonical_split()
    index_of = {stem: index for index, stem in enumerate(dataset.stems)}
    subsets: dict[str, Subset] = {}
    for name, stems in split.items():
        present = [index_of[stem] for stem in stems if stem in index_of]
        dropped = len(stems) - len(present)
        if dropped:
            logger.warning(
                "split_stems_missing",
                split=name,
                dropped=dropped,
                requested=len(stems),
            )
        subsets[name] = Subset(dataset, present)
    return subsets


def load_split(
    name: str = "test",
    settings: Settings | None = None,
    cameras: Sequence[str] | None = None,
    split: dict[str, list[str]] | None = None,
) -> Subset:
    """Load one named split as a Subset. The common dataset and split entry point.

    Parameters
    ----------
    name : str
        Split to return ("train", "validation", or "test").
    settings : Settings, optional
        Project settings. Loaded from the environment when omitted.
    cameras : sequence of str, optional
        Restrict to these cameras.
    split : dict of str to list of str, optional
        Stem lists per split name. Defaults to the canonical split.

    Returns
    -------
    Subset
        The requested split as a dataset view.
    """
    return split_datasets(WoodScapeDataset(settings, cameras), split)[name]
