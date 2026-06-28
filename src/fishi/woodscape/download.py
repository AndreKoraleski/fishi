"""Download and extract the WoodScape dataset from Google Drive."""

from pathlib import Path
from zipfile import ZipFile

import structlog
from gdown.download import download

from fishi.woodscape.config import Settings, get_settings

logger = structlog.get_logger(__name__)

_MANIFEST_NAME = ".fishi_completed.txt"


def _download(file_id: str, destination: Path, quiet: bool) -> None:
    """Download one Google Drive file by id."""
    download(id=file_id, output=str(destination), quiet=quiet)


def _extract(file: Path, destination: Path) -> None:
    """Extract a zip archive into a directory."""
    with ZipFile(file) as zf:
        zf.extractall(destination)


def _read_manifest(manifest: Path) -> set[str]:
    """Return the artifact names already recorded as downloaded."""
    if not manifest.exists():
        return set()
    return {line.strip() for line in manifest.read_text().splitlines() if line.strip()}


def _append_manifest(manifest: Path, name: str) -> None:
    """Record one completed artifact name in the manifest file."""
    with manifest.open("a") as handle:
        handle.write(f"{name}\n")


def download_woodscape(settings: Settings | None = None) -> Path:
    """Download and extract the configured WoodScape files.

    Skips any artifact already downloaded and extracted, unless settings.download.force is True.

    Parameters
    ----------
    settings : Settings, optional
        Configuration to use. Loaded from the environment when omitted.

    Returns
    -------
    Path
        The dataset root directory.
    """
    settings = settings or get_settings()
    config = settings.download
    destination = settings.data_directory
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / _MANIFEST_NAME
    completed = set() if config.force else _read_manifest(manifest)

    for artifact in config.artifacts:
        if artifact.name in completed:
            if not config.quiet:
                logger.info("skipping, already downloaded", artifact=artifact.name)
            continue

        target = destination / artifact.name
        try:
            if not target.exists() or config.force:
                if not config.quiet:
                    logger.info("downloading", artifact=artifact.name)
                _download(artifact.file_id, target, config.quiet)

            if target.suffix == ".zip":
                if not config.quiet:
                    logger.info("extracting", artifact=artifact.name)
                _extract(target, destination)
                if config.remove_archives:
                    target.unlink(missing_ok=True)
        except Exception:
            target.unlink(missing_ok=True)
            logger.error("download failed, removed partial file", artifact=artifact.name)
            raise

        _append_manifest(manifest, artifact.name)
        completed.add(artifact.name)

    return destination


if __name__ == "__main__":
    root = download_woodscape()
    logger.info("woodscape ready", path=str(root.resolve()))
