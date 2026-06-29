from fishi.segmentation.openworldsam import OpenWorldSam


def test_module_imports_without_the_openworldsam_stack():
    assert OpenWorldSam.name == "openworldsam"
    assert hasattr(OpenWorldSam, "predict")
