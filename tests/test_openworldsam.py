from fishi.segmentation.openworldsam import OpenWorldSam


def test_module_imports_without_the_openworldsam_stack():
    # heavy deps (torch, detectron2, the OWS repo) are imported lazily, so the
    # module/class load fine without them installed.
    assert OpenWorldSam.name == "openworldsam"
    assert hasattr(OpenWorldSam, "predict")
