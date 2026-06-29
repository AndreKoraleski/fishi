import numpy as np

from fishi.evaluation import evaluate, run
from fishi.preprocess import Identity


class CountingPipeline:
    name = "dummy"

    def __init__(self):
        self.calls = 0
        self.batches = 0

    def predict(self, image, prompts):
        return self.predict_batch([image], prompts)[0]

    def predict_batch(self, images, prompts):
        self.calls += len(images)
        self.batches += 1
        return [np.zeros(image.shape[:2], dtype=np.uint8) for image in images]


def test_evaluate_runs_and_returns_metrics(make_sample, fake_dataset):
    dataset = Identity().wrap(fake_dataset([make_sample(), make_sample()]))
    result = evaluate(dataset, CountingPipeline(), prompts={1: "road"}, class_count=10)
    assert {"iou", "dice", "miou", "mdice"} <= set(result)
    assert result["miou"] == 0.0  # all-void predictions vs road labels


def test_run_wraps_processor_and_scores(make_sample, fake_dataset):
    dataset = fake_dataset([make_sample()])
    result = run(Identity(), CountingPipeline(), dataset, {1: "road"}, class_count=10)
    assert "miou" in result


def test_evaluate_batches_window_into_one_call(make_sample, fake_dataset):
    samples = [make_sample(f"{index}_FV") for index in range(5)]
    pipeline = CountingPipeline()
    evaluate(Identity().wrap(fake_dataset(samples)), pipeline, {1: "road"}, 10, batch_size=8)
    assert pipeline.calls == 5  # one Identity view per sample
    assert pipeline.batches == 1  # all five fed to a single predict_batch call


def test_evaluate_respects_batch_size_windows(make_sample, fake_dataset):
    samples = [make_sample(f"{index}_FV") for index in range(5)]
    pipeline = CountingPipeline()
    evaluate(Identity().wrap(fake_dataset(samples)), pipeline, {1: "road"}, 10, batch_size=2)
    assert pipeline.batches == 3  # windows of 2, 2, 1
    assert pipeline.calls == 5


def test_cache_writes_then_skips_reinference(tmp_path, make_sample, fake_dataset):
    samples = [make_sample("a_FV"), make_sample("b_FV")]

    first = CountingPipeline()
    run(Identity(), first, fake_dataset(samples), {1: "road"}, 10, cache_directory=tmp_path)
    assert first.calls == 2  # one view per sample
    assert len(list(tmp_path.rglob("*.png"))) == 2

    second = CountingPipeline()
    result = run(
        Identity(), second, fake_dataset(samples), {1: "road"}, 10, cache_directory=tmp_path
    )
    assert second.calls == 0  # everything served from cache
    assert "miou" in result
