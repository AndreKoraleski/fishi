from fishi.woodscape.config import get_settings
from fishi.woodscape.dataset import WoodScapeDataset
from fishi.woodscape.splits import load_canonical_split, make_split, split_datasets


def test_make_split_is_deterministic():
    stems = [f"{i:05d}_FV" for i in range(100)]
    assert make_split(stems) == make_split(stems)


def test_make_split_partitions_with_proportions():
    stems = [f"{i:05d}_FV" for i in range(100)]
    split = make_split(stems)
    assert (len(split["train"]), len(split["val"]), len(split["test"])) == (70, 15, 15)
    combined = split["train"] + split["val"] + split["test"]
    assert sorted(combined) == stems  # complete partition, no overlap


def test_canonical_split_is_a_clean_partition():
    split = load_canonical_split()
    assert set(split) == {"train", "val", "test"}
    combined = [stem for stems in split.values() for stem in stems]
    assert len(combined) == len(set(combined))  # no overlap
    assert round(len(split["train"]) / len(combined), 2) == 0.70


def test_split_datasets_maps_stems_to_subsets(woodscape_root):
    dataset = WoodScapeDataset(get_settings(data_directory=woodscape_root))
    subsets = split_datasets(dataset, {"train": ["00000_FV"], "val": ["00001_RV"], "test": []})
    assert {name: len(subset) for name, subset in subsets.items()} == {
        "train": 1,
        "val": 1,
        "test": 0,
    }
    assert subsets["train"][0].stem == "00000_FV"


def test_split_datasets_drops_missing_stems(woodscape_root):
    dataset = WoodScapeDataset(get_settings(data_directory=woodscape_root))
    subsets = split_datasets(dataset, {"train": ["00000_FV", "99999_XX"], "val": [], "test": []})
    assert len(subsets["train"]) == 1  # 99999_XX is absent -> dropped (with a warning)
