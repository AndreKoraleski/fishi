import numpy as np

from fishi.segmentation.semantic import match_label, semantic_from_instances


def test_match_label_exact_and_substring():
    prompt_to_id = {"road": 1, "traffic sign": 9}
    assert match_label("road", prompt_to_id) == 1
    assert match_label("Road", prompt_to_id) == 1
    assert match_label("a traffic sign", prompt_to_id) == 9  # substring
    assert match_label("sky", prompt_to_id) is None


def test_semantic_from_instances_higher_score_wins_on_overlap():
    full = np.ones((4, 4), dtype=bool)
    corner = np.zeros((4, 4), dtype=bool)
    corner[:2, :2] = True
    # low-score full mask of class 1, high-score corner of class 2
    result = semantic_from_instances([full, corner], [1, 2], [0.3, 0.9], (4, 4))
    assert result[3, 3] == 1  # only the full mask covers here
    assert result[0, 0] == 2  # on overlap the higher score (corner) wins


def test_semantic_from_instances_background_when_empty():
    result = semantic_from_instances([], [], [], (3, 3))
    assert result.shape == (3, 3)
    assert (result == 0).all()
