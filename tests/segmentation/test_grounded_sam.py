import numpy as np

from fishi.segmentation.grounded_sam import GroundedSam1, GroundedSam2, pad_boxes


def test_module_imports_without_the_sam_stack():
    assert GroundedSam1.name == "gdino+sam1"
    assert GroundedSam2.name == "gdino+sam2"


def testpad_boxes_pads_each_image_to_the_max_count():
    boxes = [
        np.array([[5.0, 5.0, 6.0, 6.0]]),  # one box
        np.array([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]),  # two boxes
    ]
    padded = pad_boxes(boxes)
    assert [len(image_boxes) for image_boxes in padded] == [2, 2]  # both padded to the max
    assert padded[0][0] == [5.0, 5.0, 6.0, 6.0]  # real box preserved
    assert padded[0][1] == [0.0, 0.0, 1.0, 1.0]  # padded with the dummy box
    assert padded[1] == [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]  # already the max, untouched


def testpad_boxes_single_image_is_unchanged():
    assert pad_boxes([np.array([[0.0, 0.0, 1.0, 1.0]])]) == [[[0.0, 0.0, 1.0, 1.0]]]
