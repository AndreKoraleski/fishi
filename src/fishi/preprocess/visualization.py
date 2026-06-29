"""Visualization helpers for preprocessing demos."""

from math import ceil

import cv2
import numpy as np


def palette(count: int) -> np.ndarray:
    """Generate visually distinct RGB colors by evenly spacing the hue wheel.

    Parameters
    ----------
    count : int
        Number of colors to generate.

    Returns
    -------
    np.ndarray
        Colors of shape (count, 3), uint8 RGB.
    """
    hues = np.linspace(0, 179, count, endpoint=False).astype(np.uint8)
    saturation = np.full(count, 255, dtype=np.uint8)
    hsv = np.stack([hues, saturation, saturation], axis=-1).reshape(count, 1, 3)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(count, 3)


def to_rgba(image: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    """Convert an RGB image to RGBA.

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3).
    alpha : np.ndarray, optional
        Alpha mask of shape (H, W); fully opaque when omitted.

    Returns
    -------
    np.ndarray
        RGBA image of shape (H, W, 4).
    """
    height, width = image.shape[:2]
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = image[..., :3]
    rgba[..., 3] = 255 if alpha is None else alpha
    return rgba


def grid(views: list[np.ndarray], columns: int, gap: int = 4) -> np.ndarray:
    """Tile square views into a grid roughly one view tall.

    Parameters
    ----------
    views : list of np.ndarray
        Square RGB views to tile, in row-major order.
    columns : int
        Number of columns; rows follow from the view count.
    gap : int
        Transparent pixels between cells.

    Returns
    -------
    np.ndarray
        RGBA image; cells are opaque, gaps and padding transparent.
    """
    rows = ceil(len(views) / columns)
    cell = max(1, views[0].shape[0] // rows)
    canvas = np.zeros(
        (rows * cell + (rows - 1) * gap, columns * cell + (columns - 1) * gap, 4), dtype=np.uint8
    )
    for index, view in enumerate(views):
        resized = cv2.resize(view, (cell, cell))
        row_index, column_index = divmod(index, columns)
        y, x = row_index * (cell + gap), column_index * (cell + gap)
        canvas[y : y + cell, x : x + cell, :3] = resized[..., :3]
        canvas[y : y + cell, x : x + cell, 3] = 255
    return canvas


def row(panels: list[np.ndarray], gap: int = 10) -> np.ndarray:
    """Place RGBA panels left to right on a transparent canvas, top-aligned.

    Parameters
    ----------
    panels : list of np.ndarray
        RGBA panels of shape (H, W, 4); panel heights may differ.
    gap : int
        Transparent pixels between panels.

    Returns
    -------
    np.ndarray
        RGBA image sized (max panel height, total width); gaps and padding transparent.
    """
    height = max(panel.shape[0] for panel in panels)
    width = sum(panel.shape[1] for panel in panels) + gap * (len(panels) - 1)
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    x = 0
    for panel in panels:
        panel_height, panel_width = panel.shape[:2]
        canvas[:panel_height, x : x + panel_width] = panel
        x += panel_width + gap
    return canvas
