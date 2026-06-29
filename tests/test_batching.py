import pytest

from fishi.segmentation.batching import predict_with_oom_backoff


class FakeOOMError(Exception):
    """Stand-in for torch.cuda.OutOfMemoryError so the logic is testable without a GPU."""


def test_runs_in_one_chunk_when_nothing_fails():
    results, limit = predict_with_oom_backoff(
        [1, 2, 3], lambda chunk: [x * 10 for x in chunk], FakeOOMError
    )
    assert results == [10, 20, 30]
    assert limit == 3


def test_halves_chunk_on_oom_until_it_fits():
    tried: list[int] = []
    shrinks: list[int] = []

    def predict_chunk(chunk: list) -> list:
        tried.append(len(chunk))
        if len(chunk) > 2:
            raise FakeOOMError
        return [x * 10 for x in chunk]

    results, limit = predict_with_oom_backoff(
        [1, 2, 3, 4], predict_chunk, FakeOOMError, on_shrink=lambda: shrinks.append(1)
    )
    assert results == [10, 20, 30, 40]  # every item still processed, in order
    assert limit == 2  # 4 -> OOM -> 2 fits
    assert tried[0] == 4 and shrinks  # tried the full batch, shrank, retried


def test_reraises_when_a_single_item_will_not_fit():
    def always_oom(chunk: list) -> list:
        raise FakeOOMError

    with pytest.raises(FakeOOMError):
        predict_with_oom_backoff([1, 2], always_oom, FakeOOMError)


def test_empty_items_returns_empty():
    results, limit = predict_with_oom_backoff([], lambda chunk: [], FakeOOMError)
    assert results == []
    assert limit == 1
