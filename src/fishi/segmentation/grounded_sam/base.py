# pyright: reportMissingImports=false, reportArgumentType=false, reportCallIssue=false
"""Grounded SAM base: Grounding DINO detection feeding a SAM segmenter.

Grounding DINO turns the text prompts into boxes (checkpoint IDEA-Research/grounding-dino-base).
A SAM variant turns those boxes into masks.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image

from fishi.segmentation.semantic import match_label, semantic_from_instances


class GroundedSAM(ABC):
    """Grounding DINO detection plus a SAM segmenter, producing a semantic map per image."""

    name = "gdino+sam"
    detector_checkpoint = "IDEA-Research/grounding-dino-base"
    segmenter_checkpoint = ""  # set by each subclass

    def __init__(
        self,
        detector_checkpoint: str | None = None,
        segmenter_checkpoint: str | None = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._segmenter_dtype = torch.bfloat16

        detector = detector_checkpoint or self.detector_checkpoint
        self.detector_processor = AutoProcessor.from_pretrained(detector)
        self.detector = (
            AutoModelForZeroShotObjectDetection.from_pretrained(detector).to(self.device).eval()
        )
        self.load_segmenter(segmenter_checkpoint or self.segmenter_checkpoint)

    @abstractmethod
    def load_segmenter(self, checkpoint: str) -> None:
        """Load the SAM processor and model onto the device."""

    @abstractmethod
    def segment(self, pil_image, boxes) -> tuple[Any, Any]:
        """Return (masks, best_index) for this image's boxes."""

    def predict(self, image: np.ndarray, prompts: dict[int, str]) -> np.ndarray:
        pil_image = Image.fromarray(image)
        detection = self._detect(pil_image, prompts)
        if len(detection["boxes"]) == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        prompt_to_id = {text: class_id for class_id, text in prompts.items()}
        masks, best = self.segment(pil_image, detection["boxes"])
        return self._assemble(detection, masks, best, prompt_to_id, image.shape[:2])

    def _assemble(
        self,
        detection: dict[str, Any],
        masks: Any,
        best: Any,
        prompt_to_id: dict[str, int],
        shape: tuple[int, int],
    ) -> np.ndarray:
        chosen_masks: list[np.ndarray] = []
        class_ids: list[int] = []
        scores: list[float] = []
        for index, (label, score) in enumerate(
            zip(detection["text_labels"], detection["scores"].tolist(), strict=False)
        ):
            class_id = match_label(label, prompt_to_id)
            if class_id is None:
                continue
            chosen_masks.append(masks[index, best[index]].numpy().astype(bool))
            class_ids.append(class_id)
            scores.append(score)
        return semantic_from_instances(chosen_masks, class_ids, scores, shape)

    def _detect(self, pil_image, prompts: dict[int, str]):
        torch = self._torch
        inputs = self.detector_processor(
            images=pil_image, text=[list(prompts.values())], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.detector(**inputs)
        return self.detector_processor.post_process_grounded_object_detection(
            outputs,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(pil_image.height, pil_image.width)],
        )[0]
