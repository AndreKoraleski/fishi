# pyright: reportMissingImports=false, reportArgumentType=false, reportCallIssue=false
"""SAM 3 pipeline: native text-promptable segmentation, no detector."""

import numpy as np
from PIL import Image

from fishi.segmentation.semantic import semantic_from_instances


class SamThree:
    """Text-prompted segmentation via SAM 3, producing a semantic map.

    SAM 3 takes one image and one text concept per forward pass, so each prompt is run in turn and
    the resulting instance masks are flattened into a semantic map with the highest-scoring instance
    winning on overlap.

    Parameters
    ----------
    checkpoint : str, optional
        HuggingFace model id (gated; needs HF_TOKEN).
    score_threshold : float
        Minimum instance confidence to keep.
    mask_threshold : float
        Threshold for binarizing each mask.
    device : str, optional
        Torch device; defaults to cuda when available.
    """

    name = "sam3"
    checkpoint = "facebook/sam3"

    def __init__(
        self,
        checkpoint: str | None = None,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import Sam3Model, Sam3Processor

        self._torch = torch
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model_id = checkpoint or self.checkpoint
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()

    def predict(self, image: np.ndarray, prompts: dict[int, str]) -> np.ndarray:
        torch = self._torch
        pil_image = Image.fromarray(image)
        masks: list[np.ndarray] = []
        class_ids: list[int] = []
        scores: list[float] = []

        for class_id, concept in prompts.items():
            inputs = self.processor(images=pil_image, text=concept, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16):
                outputs = self.model(**inputs)
            result = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.score_threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=inputs["original_sizes"].tolist(),
            )[0]
            for mask, score in zip(result["masks"], result["scores"], strict=True):
                array = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
                masks.append(array.astype(bool))
                class_ids.append(class_id)
                scores.append(float(score))
        return semantic_from_instances(masks, class_ids, scores, image.shape[:2])

    def predict_batch(self, images: list[np.ndarray], prompts: dict[int, str]) -> list[np.ndarray]:
        """Segment each image in turn (SAM 3 takes one image + one concept per call)."""
        return [self.predict(image, prompts) for image in images]
