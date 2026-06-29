"""Turn open-set detector outputs into a single semantic label map."""

import numpy as np


def match_label(text_label: str, prompt_to_id: dict[str, int]) -> int | None:
    """Map a detector's returned text label back to a class id.

    Tries an exact (case-insensitive) match first, then a substring match, since open-set detectors
    may return partial or merged label text.
    """
    text = text_label.strip().lower()
    for prompt, class_id in prompt_to_id.items():
        if text == prompt.lower():
            return class_id
    for prompt, class_id in prompt_to_id.items():
        if prompt.lower() in text or text in prompt.lower():
            return class_id
    return None


def semantic_from_instances(
    masks: list[np.ndarray],
    class_ids: list[int],
    scores: list[float],
    shape: tuple[int, int],
    background: int = 0,
) -> np.ndarray:
    """Flatten instance masks into one semantic map.

    On overlap the higher-scoring instance wins (painted last).

    Parameters
    ----------
    masks : list of np.ndarray
        Boolean masks of shape (H, W), one per instance.
    class_ids : list of int
        Class id for each mask.
    scores : list of float
        Confidence for each mask; controls overlap priority.
    shape : tuple of int
        (H, W) of the output map.
    background : int
        Value for pixels covered by no instance.

    Returns
    -------
    np.ndarray
        Semantic map of shape (H, W).
    """
    result = np.full(shape, background, dtype=np.uint8)
    for index in np.argsort(scores):  # ascending: low score first, high score overwrites
        result[masks[index]] = class_ids[index]
    return result
