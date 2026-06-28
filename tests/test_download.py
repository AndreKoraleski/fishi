import pytest

from fishi.woodscape import download
from fishi.woodscape.config import get_settings


@pytest.fixture
def record_downloads(monkeypatch) -> list[str]:
    calls: list[str] = []

    def fake_download(file_id, output, quiet) -> None:
        output.write_bytes(b"stub")
        calls.append(output.name)

    monkeypatch.setattr(download, "_download", fake_download)
    monkeypatch.setattr(download, "_extract", lambda archive, destination: None)
    return calls


def test_default_run_fetches_all_artifacts(record_downloads, tmp_path):
    download.download_woodscape(get_settings(data_directory=tmp_path))
    assert {"rgb_images.zip", "semantic_annotations.zip", "calibration.zip"} <= set(
        record_downloads
    )


def test_markers_make_reruns_idempotent(record_downloads, tmp_path):
    download.download_woodscape(get_settings(data_directory=tmp_path))
    count = len(record_downloads)
    download.download_woodscape(get_settings(data_directory=tmp_path))
    assert len(record_downloads) == count  # second run skipped everything


def test_failed_download_removes_partial_and_is_not_recorded(monkeypatch, tmp_path):
    def boom(file_id, output, quiet) -> None:
        output.write_bytes(b"partial")  # a half-written file is left behind
        raise RuntimeError("network died")

    monkeypatch.setattr(download, "_download", boom)
    monkeypatch.setattr(download, "_extract", lambda archive, dest: None)

    with pytest.raises(RuntimeError):
        download.download_woodscape(get_settings(data_directory=tmp_path))

    assert not (tmp_path / "rgb_images.zip").exists()
    assert not (tmp_path / ".fishi_completed.txt").exists()
