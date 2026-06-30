import numpy as np

from fishi.segmentation.semantic import match_label, semantic_from_instances


def test_match_label_exact_is_case_insensitive():
    prompts = {"road": 1, "bicycle": 7}
    assert match_label("road", prompts) == 1
    assert match_label("Bicycle", prompts) == 7


def test_match_label_prefers_the_most_specific_overlap():
    prompts = {"road": 1, "bicycle": 7}  # road is first in iteration order
    # both prompts overlap the phrase, so the longer (more specific) one must win, not the first
    assert match_label("a bicycle on the road", prompts) == 7


def test_match_label_no_overlap_is_none():
    assert match_label("sky", {"road": 1}) is None


def test_semantic_from_instances_higher_score_wins_overlap():
    a = np.array([[True, True], [False, False]])
    b = np.array([[True, False], [False, False]])  # overlaps a at (0, 0)
    result = semantic_from_instances([a, b], [1, 2], [0.4, 0.9], (2, 2))
    assert result[0, 0] == 2  # higher-scoring instance paints over the lower
    assert result[0, 1] == 1
