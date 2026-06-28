"""WoodScape config: where the data lives and how to download it."""

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DriveArtifact(BaseModel):
    """One WoodScape file hosted on Google Drive."""

    name: str  # local filename under the data directory
    file_id: str


_DEFAULT_ARTIFACTS: list[dict] = [
    {"name": "rgb_images.zip", "file_id": "1xQ5J4huNmyK9WPoipHTnuZ7lw_J0xhvL"},
    {"name": "semantic_annotations.zip", "file_id": "1CBwi0fpDE2G99hHiINTI-AXlOMQSUnb-"},
    {"name": "calibration.zip", "file_id": "1o7KBl1QzTkugMDOadvJFSbN87njuajYc"},
    {"name": "seg_annotation_info.json", "file_id": "1NUVfLV1U44nkR9PbTW56I_KvppGcOFSt"},
]


class DownloadConfig(BaseModel):
    """How WoodScape is downloaded from Google Drive."""

    remove_archives: bool = True  # delete zips after extracting
    force: bool = False  # redo work even if already done
    quiet: bool = False  # silence progress output
    artifacts: list[DriveArtifact] = Field(
        default_factory=lambda: [DriveArtifact(**artifact) for artifact in _DEFAULT_ARTIFACTS]
    )


class Settings(BaseSettings):
    """WoodScape settings: data location and download options."""

    model_config = SettingsConfigDict(
        env_prefix="FISHI_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    data_directory: Path = Path("data")
    download: DownloadConfig = Field(default_factory=DownloadConfig)


def get_settings(**overrides) -> Settings:
    """Build settings, applying any keyword overrides."""
    return Settings(**overrides)
