# pyright: reportMissingImports=false, reportArgumentType=false, reportCallIssue=false
"""OpenWorldSAM pipeline: text-prompted semantic segmentation (SAM2 + a VLM).

OpenWorldSAM (NeurIPS 2025, https://github.com/GinnyXiao/OpenWorldSAM).
"""

import sys
import tempfile
from typing import cast

import cv2
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class OpenWorldSam:
    """Text-prompted dense semantic segmentation via OpenWorldSAM.

    Parameters
    ----------
    config_file : str
        Detectron2 YAML config from the OpenWorldSAM repo.
    weights : str
        Trained checkpoint path (from the repo's Google Drive links).
    repo_path : str, optional
        OpenWorldSAM repo root, prepended to sys.path so demo.inference_utils
        is importable.
    device : str, optional
        Torch device; defaults to cuda when available.
    """

    name = "openworldsam"

    def __init__(
        self,
        config_file: str,
        weights: str,
        repo_path: str | None = None,
        device: str | None = None,
    ) -> None:
        if repo_path:
            sys.path.insert(0, repo_path)
        import torch
        from demo.inference_utils import get_metadata, load_model, setup_cfg

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._max_batch: int | None = None  # largest model batch that fit (set on OOM)
        config = setup_cfg(config_file, weights=weights, device=self.device)
        config.MODEL.OpenWorldSAM2.TEST.SEMANTIC_ON = True
        config.MODEL.OpenWorldSAM2.TEST.INSTANCE_ON = False
        self._config = config
        self.model = load_model(config)
        self.metadata = get_metadata(config)

    def predict(self, image: np.ndarray, prompts: dict[int, str]) -> np.ndarray:
        return self.predict_batch([image], prompts)[0]

    def predict_batch(self, images: list[np.ndarray], prompts: dict[int, str]) -> list[np.ndarray]:
        """Segment a batch of images in one model call, halving the batch on CUDA OOM."""
        from demo.inference_utils import build_inference_inputs, prepare_image_inputs

        torch = self._torch
        class_ids = list(prompts.keys())
        texts = [prompts[class_id] for class_id in class_ids]
        category_ids = list(range(len(texts)))

        inputs = []
        for image in images:
            with tempfile.NamedTemporaryFile(suffix=".png") as handle:
                cv2.imwrite(handle.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                _, sam_tensor, beit_tensor, height, width = prepare_image_inputs(
                    handle.name, self._config.INPUT.FORMAT
                )
            inputs.append(
                build_inference_inputs(sam_tensor, beit_tensor, height, width, texts, category_ids)[
                    0
                ]
            )

        labels = np.asarray(class_ids, dtype=np.uint8)
        results: list[np.ndarray | None] = [None] * len(inputs)
        index = 0
        limit = self._max_batch or len(inputs)
        while index < len(inputs):
            size = min(limit, len(inputs) - index)
            try:
                with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16):
                    outputs = self.model(inputs[index : index + size])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size == 1:
                    raise
                limit = max(1, size // 2)
                self._max_batch = limit
                logger.warning("oom_backoff", batch_size=limit)
                continue
            for offset, output in enumerate(outputs):
                channel = output["sem_seg"].argmax(dim=0).cpu().numpy()
                results[index + offset] = labels[channel]
            index += size
        return cast(list[np.ndarray], results)
