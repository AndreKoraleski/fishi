from fishi.analysis import resampling_ceiling
from fishi.preprocess import Identity


def test_resampling_ceiling_is_one_for_identity(make_sample, fake_dataset):
    dataset = fake_dataset([make_sample("00000_FV"), make_sample("00001_RV")])
    ceilings = resampling_ceiling([Identity()], dataset)
    assert ceilings["none"] == 1.0  # the no-op processor round-trips the label exactly
