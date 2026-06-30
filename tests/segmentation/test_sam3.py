from fishi.segmentation.sam3 import SamThree


def test_module_imports_without_the_sam3_stack():
    assert SamThree.name == "sam3"
    assert hasattr(SamThree, "predict")
