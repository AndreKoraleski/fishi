# fishi

Preprocessing techniques for zero-shot segmentation on the fisheye [WoodScape](https://woodscape.valeo.com) dataset.

## Setup

```bash
uv sync
```

## Download the dataset

```bash
uv run python -m fishi.woodscape.download
```

Fetches WoodScape (RGB images, segmentation labels, calibration) into `data/`.

## Development

```bash
uv run ruff check .
uv run mypy
uv run pytest
```
