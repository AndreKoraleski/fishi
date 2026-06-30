import numpy as np

from fishi.analysis import error_decomposition, resampling_ceiling
from fishi.preprocess import Identity


def test_resampling_ceiling_is_one_for_identity(make_sample, fake_dataset):
    dataset = fake_dataset([make_sample("00000_FV"), make_sample("00001_RV")])
    ceilings = resampling_ceiling([Identity()], dataset)
    assert ceilings["none"] == 1.0  # the no-op processor round-trips the label exactly


def test_error_decomposition_reads_the_cache(tmp_path, make_sample, fake_dataset):
    dataset = fake_dataset([make_sample("frame_FV")])  # label is all class 1
    np.savez(tmp_path / "dummy__none.npz", frame_FV=np.ones((4, 4), dtype=np.uint8))
    rows = error_decomposition(tmp_path, dataset)
    assert rows["dummy__none"]["miou"] == 1.0  # the prediction matches the label
    assert rows["dummy__none"]["confused"] == 0.0
