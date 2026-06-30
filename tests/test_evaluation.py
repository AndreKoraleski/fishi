import numpy as np

from fishi.evaluation import evaluate, run, score
from fishi.preprocess import Identity


class CountingPipeline:
    name = "dummy"

    def __init__(self):
        self.calls = 0

    def predict(self, image, prompts):
        self.calls += 1
        return np.zeros(image.shape[:2], dtype=np.uint8)


def test_evaluate_runs_and_returns_metrics(make_sample, fake_dataset):
    dataset = Identity().wrap(fake_dataset([make_sample(), make_sample("00001_RV")]))
    pipeline = CountingPipeline()
    result = evaluate(dataset, pipeline, prompts={1: "road"}, class_count=10)
    assert {"iou", "accuracy", "miou", "macc"} <= set(result)
    assert result["miou"] == 0.0  # all-void predictions vs road labels
    assert pipeline.calls == 2  # one Identity view per sample, one image at a time


def test_evaluate_uses_taxonomy_defaults(make_sample, fake_dataset):
    dataset = Identity().wrap(fake_dataset([make_sample()]))
    result = evaluate(dataset, CountingPipeline())  # no prompts or class_count
    assert "miou" in result  # ran with the WoodScape taxonomy defaults


def test_run_returns_the_cell_report(make_sample, fake_dataset):
    report = run(Identity(), CountingPipeline(), fake_dataset([make_sample()]), {1: "road"}, 10)
    assert report["pipeline"] == "dummy"
    assert report["preprocessing"] == "none"
    assert "miou" in report
    assert "macc" in report  # the cell report carries both challenge metrics
    assert "accuracy" in report["per_class"]["road"]


def test_run_saves_report_and_skips_finished_cell(tmp_path, make_sample, fake_dataset):
    samples = [make_sample("00000_FV")]
    first = CountingPipeline()
    run(Identity(), first, fake_dataset(samples), {1: "road"}, 10, metrics_directory=tmp_path)
    assert first.calls == 1
    assert (tmp_path / "dummy__none.json").exists()

    second = CountingPipeline()
    run(Identity(), second, fake_dataset(samples), {1: "road"}, 10, metrics_directory=tmp_path)
    assert second.calls == 0  # finished cell skipped via its saved report


def test_cache_writes_then_skips_reinference(tmp_path, make_sample, fake_dataset):
    samples = [make_sample("00000_FV"), make_sample("00001_RV")]

    first = CountingPipeline()
    run(Identity(), first, fake_dataset(samples), {1: "road"}, 10, cache_directory=tmp_path)
    assert first.calls == 2
    assert len(list(tmp_path.glob("*.npz"))) == 1

    second = CountingPipeline()
    run(Identity(), second, fake_dataset(samples), {1: "road"}, 10, cache_directory=tmp_path)
    assert second.calls == 0  # whole cell served from the .npz


def test_checkpoint_flushes_cache_midway(tmp_path, make_sample, fake_dataset):
    samples = [make_sample(f"0000{index}_FV") for index in range(3)]
    run(
        Identity(),
        CountingPipeline(),
        fake_dataset(samples),
        {1: "road"},
        10,
        cache_directory=tmp_path,
        checkpoint_every=1,
    )
    assert len(list(tmp_path.glob("*.npz"))) == 1


def test_score_external_predictions(make_sample, fake_dataset):
    dataset = fake_dataset([make_sample("00000_FV"), make_sample("00001_RV")])
    predictions = {
        "00000_FV": np.ones((4, 4), dtype=np.uint8),  # matches the all-class-1 label
        "00001_RV": np.zeros((4, 4), dtype=np.uint8),  # predicts void (wrong)
    }
    result = score(predictions, dataset, class_count=10)
    assert result["miou"] == 0.5  # class 1: one image right, one wrong


def test_run_skips_each_preprocessing_independently(tmp_path, make_sample, fake_dataset):
    samples = [make_sample("00000_FV")]
    pipeline = CountingPipeline()
    # a driver loops run() over preprocessings for one loaded pipeline
    run(Identity(), pipeline, fake_dataset(samples), {1: "road"}, 10, metrics_directory=tmp_path)
    assert (tmp_path / "dummy__none.json").exists()
    run(Identity(), pipeline, fake_dataset(samples), {1: "road"}, 10, metrics_directory=tmp_path)
    assert pipeline.calls == 1  # the finished cell is skipped on the second pass


def test_sweep_covers_every_preprocessing(tmp_path, make_sample, fake_dataset):
    from fishi.sweep import PREPROCESSORS, sweep

    sweep(CountingPipeline(), fake_dataset([make_sample("00000_FV")]), tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == len(PREPROCESSORS)
