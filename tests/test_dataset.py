from fishi.woodscape.config import get_settings
from fishi.woodscape.dataset import Subset, WoodScapeDataset


def test_len_and_pairing(woodscape_root):
    dataset = WoodScapeDataset(get_settings(data_directory=woodscape_root))
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample.image.shape == (4, 4, 3)
    assert sample.label.shape == (4, 4)
    assert sample.stem == "00000_FV"
    assert sample.camera == "FV"
    assert sample.calibration.name == "FV"


def test_camera_filter(woodscape_root):
    dataset = WoodScapeDataset(get_settings(data_directory=woodscape_root), cameras=["FV"])
    assert len(dataset) == 1
    assert dataset[0].camera == "FV"


def test_subset_selects_given_indices(woodscape_root):
    dataset = WoodScapeDataset(get_settings(data_directory=woodscape_root))
    subset = Subset(dataset, [1])
    assert len(subset) == 1
    assert subset[0].stem == dataset[1].stem
