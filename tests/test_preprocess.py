import numpy as np

from fishi.preprocess import Identity, ProcessedSample


def test_identity_wrap_yields_single_view(make_sample, fake_dataset):
    dataset = Identity().wrap(fake_dataset([make_sample(), make_sample("00001_RV")]))
    assert len(dataset) == 2
    processed = dataset[0]
    assert isinstance(processed, ProcessedSample)
    assert len(processed.views) == 1
    assert np.array_equal(processed.views[0], np.zeros((4, 4, 3), dtype=np.uint8))
    assert processed.stem == "00000_FV"
    assert processed.camera == "FV"


def test_identity_postprocess_is_identity(make_calibration):
    prediction = np.array([[0, 1], [2, 0]])
    assert np.array_equal(Identity().postprocess([prediction], make_calibration()), prediction)
