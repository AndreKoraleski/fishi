# mypy: ignore-errors
# pyright: reportArgumentType=false, reportCallIssue=false
"""Grounded SAM pipelines: Grounding DINO (text -> boxes) + SAM (boxes -> masks)."""

import numpy as np
import structlog
from PIL import Image

from fishi.segmentation.base import match_label, semantic_from_instances

logger = structlog.get_logger(__name__)


class GroundedSAM:
    """Grounding DINO detection + a SAM segmenter, producing a semantic map."""

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
        self._segmenter_dtype = torch.float16 if self.device == "cuda" else torch.float32

        detector = detector_checkpoint or self.detector_checkpoint
        self.detector_processor = AutoProcessor.from_pretrained(detector)
        self.detector = (
            AutoModelForZeroShotObjectDetection.from_pretrained(detector).to(self.device).eval()
        )
        self._load_segmenter(segmenter_checkpoint or self.segmenter_checkpoint)

    def _load_segmenter(self, checkpoint: str) -> None:
        raise NotImplementedError

    def _segment(self, pil_image, boxes):
        """Return (masks, best_index) for the given boxes."""
        raise NotImplementedError

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

    def predict(self, image: np.ndarray, prompts: dict[int, str]) -> np.ndarray:
        height, width = image.shape[:2]
        pil_image = Image.fromarray(image)
        detections = self._detect(pil_image, prompts)
        boxes = detections["boxes"]
        if len(boxes) == 0:
            return np.zeros((height, width), dtype=np.uint8)

        masks, best = self._segment(pil_image, boxes)
        prompt_to_id = {text: class_id for class_id, text in prompts.items()}
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
        return semantic_from_instances(chosen_masks, class_ids, scores, (height, width))


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

    def _segment(self, pil_image, boxes):
        torch = self._torch
        inputs = self.segmenter_processor(
            pil_image, input_boxes=[boxes.tolist()], return_tensors="pt"
        ).to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self._segmenter_dtype)
        inputs["input_boxes"] = inputs["input_boxes"].to(self._segmenter_dtype)
        with torch.no_grad():
            outputs = self.segmenter(**inputs)
        masks = self.segmenter_processor.image_processor.post_process_masks(
            outputs.pred_masks.float().cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        best = outputs.iou_scores.float().cpu()[0].argmax(dim=-1)
        return masks, best


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

    def _segment(self, pil_image, boxes):
        torch = self._torch
        inputs = self.segmenter_processor(
            images=pil_image, input_boxes=[boxes.tolist()], return_tensors="pt"
        ).to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self._segmenter_dtype)
        with torch.no_grad():
            outputs = self.segmenter(**inputs)
        masks = self.segmenter_processor.post_process_masks(
            outputs.pred_masks.float().cpu(), inputs["original_sizes"].cpu()
        )[0]
        best = outputs.iou_scores.float().cpu()[0].argmax(dim=-1)
        return masks, best
