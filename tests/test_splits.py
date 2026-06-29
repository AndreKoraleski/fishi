from fishi.woodscape.splits import load_canonical_split, make_split


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
