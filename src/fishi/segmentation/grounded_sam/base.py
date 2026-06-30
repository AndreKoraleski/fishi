# pyright: reportMissingImports=false, reportArgumentType=false, reportCallIssue=false
"""Grounded SAM base: Grounding DINO detection feeding a SAM segmenter.

Grounding DINO turns the text prompts into boxes (checkpoint IDEA-Research/grounding-dino-base).
A SAM variant turns those boxes into masks. Subclasses supply the segmenter via load_segmenter
and segment. Detection, out-of-memory-adaptive batching, and assembly into a semantic map live
here.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image

from fishi.segmentation.batching import predict_with_oom_backoff
from fishi.segmentation.semantic import match_label, semantic_from_instances


def pad_boxes(boxes: list) -> list:
    """Pad each image's box list to the batch's max count so SAM gets a rectangular batch."""
    box_lists = [box.tolist() for box in boxes]
    width = max(len(boxes) for boxes in box_lists)
    return [boxes + [[0.0, 0.0, 1.0, 1.0]] * (width - len(boxes)) for boxes in box_lists]


class GroundedSAM(ABC):
    """Grounding DINO detection + a SAM segmenter, producing a semantic map.

    predict_batch runs the whole list in one forward and halves the batch on CUDA out-of-memory.
    """

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
        self._max_batch: int | None = None  # largest chunk that fit so far

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
    def segment(self, pil_images: list, boxes: list) -> list[tuple[Any, Any]]:
        """Per image, return (masks, best_index) for that image's boxes."""

    def predict(self, image: np.ndarray, prompts: dict[int, str]) -> np.ndarray:
        return self.predict_batch([image], prompts)[0]

    def predict_batch(self, images: list[np.ndarray], prompts: dict[int, str]) -> list[np.ndarray]:
        """Segment a list of images, auto-shrinking the batch on CUDA OOM."""
        torch = self._torch
        results, self._max_batch = predict_with_oom_backoff(
            images,
            lambda chunk: self._predict_chunk(chunk, prompts),
            torch.cuda.OutOfMemoryError,
            on_shrink=torch.cuda.empty_cache,
            start_batch=self._max_batch,
        )
        return results

    def _predict_chunk(self, images: list[np.ndarray], prompts: dict[int, str]) -> list[np.ndarray]:
        pil_images = [Image.fromarray(image) for image in images]
        detections = self._detect(pil_images, prompts)
        prompt_to_id = {text: class_id for class_id, text in prompts.items()}

        detected = [index for index, det in enumerate(detections) if len(det["boxes"]) > 0]
        segmented: dict[int, tuple[Any, Any]] = {}
        if detected:
            pairs = self.segment(
                [pil_images[index] for index in detected],
                [detections[index]["boxes"] for index in detected],
            )
            segmented = dict(zip(detected, pairs, strict=True))

        results = []
        for index, image in enumerate(images):
            shape = image.shape[:2]
            if index not in segmented:
                results.append(np.zeros(shape, dtype=np.uint8))
                continue
            masks, best = segmented[index]
            results.append(self._assemble(detections[index], masks, best, prompt_to_id, shape))
        return results

    def _assemble(
        self,
        detections: dict[str, Any],
        masks: Any,
        best: Any,
        prompt_to_id: dict[str, int],
        shape: tuple[int, int],
    ) -> np.ndarray:
        chosen_masks: list[np.ndarray] = []
        class_ids: list[int] = []
        scores: list[float] = []
        for index, (label, score) in enumerate(
            zip(detections["text_labels"], detections["scores"].tolist(), strict=False)
        ):
            class_id = match_label(label, prompt_to_id)
            if class_id is None:
                continue
            chosen_masks.append(masks[index, best[index]].numpy().astype(bool))
            class_ids.append(class_id)
            scores.append(score)
        return semantic_from_instances(chosen_masks, class_ids, scores, shape)

    def _detect(self, pil_images, prompts: dict[int, str]):
        torch = self._torch
        texts = [list(prompts.values())] * len(pil_images)
        inputs = self.detector_processor(images=pil_images, text=texts, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.detector(**inputs)
        return self.detector_processor.post_process_grounded_object_detection(
            outputs,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width) for image in pil_images],
        )
