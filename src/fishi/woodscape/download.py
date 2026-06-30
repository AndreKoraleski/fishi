"""Download and extract the WoodScape dataset from Google Drive."""

from pathlib import Path
from zipfile import ZipFile

import structlog

from fishi.woodscape.config import Settings, get_settings

logger = structlog.get_logger(__name__)

MANIFEST_NAME = ".fishi_completed.txt"


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
    manifest = destination / MANIFEST_NAME
    completed = set() if config.force else read_manifest(manifest)

    for artifact in config.artifacts:
        if artifact.name in completed:
            if not config.quiet:
                logger.info("artifact_cached", artifact=artifact.name)
            continue

        target = destination / artifact.name
        try:
            if not target.exists() or config.force:
                if not config.quiet:
                    logger.info("artifact_downloading", artifact=artifact.name)
                fetch(artifact.file_id, target, config.quiet)

            if target.suffix == ".zip":
                if not config.quiet:
                    logger.info("artifact_extracting", artifact=artifact.name)
                extract(target, destination)
                if config.remove_archives:
                    target.unlink(missing_ok=True)
        except Exception:
            target.unlink(missing_ok=True)
            logger.error("artifact_download_failed", artifact=artifact.name)
            raise

        append_manifest(manifest, artifact.name)
        completed.add(artifact.name)

    return destination


def fetch(file_id: str, destination: Path, quiet: bool) -> None:
    """Download one Google Drive file by id. Needs the optional download extra (gdown)."""
    try:
        from gdown.download import download
    except ImportError as error:
        raise ImportError(
            "Fetching WoodScape needs gdown. Install it with: pip install 'fishi[download]'"
        ) from error
    download(id=file_id, output=str(destination), quiet=quiet)


def extract(file: Path, destination: Path) -> None:
    """Extract a zip archive into a directory."""
    with ZipFile(file) as archive:
        archive.extractall(destination)


def read_manifest(manifest: Path) -> set[str]:
    """Return the artifact names already recorded as downloaded."""
    if not manifest.exists():
        return set()
    return {line.strip() for line in manifest.read_text().splitlines() if line.strip()}


def append_manifest(manifest: Path, name: str) -> None:
    """Record one completed artifact name in the manifest file."""
    with manifest.open("a") as handle:
        handle.write(f"{name}\n")


if __name__ == "__main__":
    root = download_woodscape()
    logger.info("dataset_ready", path=str(root.resolve()))
