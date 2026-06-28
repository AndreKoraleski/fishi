"""Visualization helpers for preprocessing demos."""

import numpy as np


def side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    right_alpha: np.ndarray | None = None,
    gap: int = 10,
) -> np.ndarray:
    """Place two RGB images side by side on a transparent RGBA canvas.

    Parameters
    ----------
    left, right : np.ndarray
        RGB images of shape (H, W, 3).
    right_alpha : np.ndarray, optional
        Alpha mask (H, W) for the right image; fully opaque when omitted.
    gap : int
        Transparent pixels separating the two images.

    Returns
    -------
    np.ndarray
        RGBA image (H, W, 4); padding and the gap are transparent.
    """
    left_height, left_width = left.shape[:2]
    right_height, right_width = right.shape[:2]
    height = max(left_height, right_height)
    canvas = np.zeros((height, left_width + gap + right_width, 4), dtype=np.uint8)

    canvas[:left_height, :left_width, :3] = left[..., :3]
    canvas[:left_height, :left_width, 3] = 255

    start = left_width + gap
    canvas[:right_height, start : start + right_width, :3] = right[..., :3]
    canvas[:right_height, start : start + right_width, 3] = (
        255 if right_alpha is None else right_alpha
    )
    return canvas
