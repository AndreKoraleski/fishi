import subprocess
import sys

import fishi


def test_version():
    assert fishi.__version__ == "0.1.0"


def test_front_door_exposes_the_public_api():
    for name in ("run", "evaluate", "score", "load_split", "Processor", "SegmentationPipeline"):
        assert hasattr(fishi, name)


def test_importing_fishi_does_not_pull_heavy_deps():
    code = (
        "import fishi, sys\n"
        "heavy = {'torch', 'transformers', 'gdown'}\n"
        "assert not heavy & set(sys.modules), heavy & set(sys.modules)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
