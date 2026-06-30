from fishi.segmentation.grounded_sam import GroundedSam1, GroundedSam2


def test_module_imports_without_the_sam_stack():
    assert GroundedSam1.name == "gdino+sam1"
    assert GroundedSam2.name == "gdino+sam2"
    assert hasattr(GroundedSam1, "predict")
    assert hasattr(GroundedSam2, "predict")
