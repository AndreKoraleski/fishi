# pyright: reportMissingImports=false, reportArgumentType=false, reportCallIssue=false
"""Grounded SAM pipelines: Grounding DINO (text -> boxes) + SAM (boxes -> masks)."""

from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import structlog
from PIL import Image

from fishi.segmentation.base import match_label, semantic_from_instances

logger = structlog.get_logger(__name__)


def _pad_boxes(boxes: list) -> list:
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
        self._load_segmenter(segmenter_checkpoint or self.segmenter_checkpoint)

    @abstractmethod
    def _load_segmenter(self, checkpoint: str) -> None:
        """Load the SAM processor and model onto the device."""

    @abstractmethod
    def _segment(self, pil_images: list, boxes: list) -> list[tuple[Any, Any]]:
        """Per image, return (masks, best_index) for that image's boxes."""

    def predict(self, image: np.ndarray, prompts: dict[int, str]) -> np.ndarray:
        return self.predict_batch([image], prompts)[0]

    def predict_batch(self, images: list[np.ndarray], prompts: dict[int, str]) -> list[np.ndarray]:
        """Segment a list of images, auto-shrinking the batch on CUDA OOM."""
        torch = self._torch
        results: list[np.ndarray | None] = [None] * len(images)
        index = 0
        limit = self._max_batch or len(images)
        while index < len(images):
            size = min(limit, len(images) - index)
            try:
                chunk = self._predict_chunk(images[index : index + size], prompts)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size == 1:
                    raise
                limit = max(1, size // 2)
                self._max_batch = limit
                logger.warning("oom_backoff", batch_size=limit)
                continue
            for offset, prediction in enumerate(chunk):
                results[index + offset] = prediction
            index += size
        return cast(list[np.ndarray], results)

    def _predict_chunk(self, images: list[np.ndarray], prompts: dict[int, str]) -> list[np.ndarray]:
        pil_images = [Image.fromarray(image) for image in images]
        detections = self._detect(pil_images, prompts)
        prompt_to_id = {text: class_id for class_id, text in prompts.items()}

        detected = [index for index, det in enumerate(detections) if len(det["boxes"]) > 0]
        segmented: dict[int, tuple[Any, Any]] = {}
        if detected:
            pairs = self._segment(
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


class GroundedSam1(GroundedSAM):
    """Grounding DINO + SAM 1."""

    name = "gdino+sam1"
    segmenter_checkpoint = "facebook/sam-vit-huge"

    def _load_segmenter(self, checkpoint: str) -> None:
        from transformers import SamModel, SamProcessor

        self.segmenter_processor = SamProcessor.from_pretrained(checkpoint)
        self.segmenter = (
            SamModel.from_pretrained(checkpoint, torch_dtype=self._segmenter_dtype)
            .to(self.device)
            .eval()
        )

    def _segment(self, pil_images, boxes):
        torch = self._torch
        inputs = self.segmenter_processor(
            pil_images, input_boxes=_pad_boxes(boxes), return_tensors="pt"
        ).to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self._segmenter_dtype)
        inputs["input_boxes"] = inputs["input_boxes"].to(self._segmenter_dtype)
        with torch.no_grad():
            outputs = self.segmenter(**inputs)
        masks = self.segmenter_processor.image_processor.post_process_masks(
            outputs.pred_masks.float().cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        best = outputs.iou_scores.float().cpu().argmax(dim=-1)
        return [(masks[index], best[index]) for index in range(len(pil_images))]


class GroundedSam2(GroundedSAM):
    """Grounding DINO + SAM 2."""

    name = "gdino+sam2"
    segmenter_checkpoint = "facebook/sam2-hiera-large"

    def _load_segmenter(self, checkpoint: str) -> None:
        from transformers import Sam2Model, Sam2Processor

        self.segmenter_processor = Sam2Processor.from_pretrained(checkpoint)
        self.segmenter = (
            Sam2Model.from_pretrained(checkpoint, torch_dtype=self._segmenter_dtype)
            .to(self.device)
            .eval()
        )

    def _segment(self, pil_images, boxes):
        torch = self._torch
        inputs = self.segmenter_processor(
            images=pil_images, input_boxes=_pad_boxes(boxes), return_tensors="pt"
        ).to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self._segmenter_dtype)
        with torch.no_grad():
            outputs = self.segmenter(**inputs)
        masks = self.segmenter_processor.post_process_masks(
            outputs.pred_masks.float().cpu(), inputs["original_sizes"].cpu()
        )
        best = outputs.iou_scores.float().cpu().argmax(dim=-1)
        return [(masks[index], best[index]) for index in range(len(pil_images))]
