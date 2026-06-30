# pyright: reportMissingImports=false, reportArgumentType=false, reportCallIssue=false
"""SAM 3 pipeline: native text-promptable segmentation, no detector (checkpoint facebook/sam3)."""

import numpy as np
from PIL import Image

from fishi.segmentation.semantic import semantic_from_instances


def repeat_batch(value, n, torch):
    """Repeat a vision-feature field (a tensor, or a nested tuple/list) along the batch dim."""
    if torch.is_tensor(value):
        return value.repeat(n, *([1] * (value.dim() - 1)))
    if isinstance(value, (tuple, list)):
        return type(value)(repeat_batch(item, n, torch) for item in value)
    return value


class SamThree:
    """Text-prompted segmentation via SAM 3, producing a semantic map.

    Parameters
    ----------
    checkpoint : str, optional
        HuggingFace model id (gated, needs HF_TOKEN).
    score_threshold : float
        Minimum instance confidence to keep.
    mask_threshold : float
        Threshold for binarizing each mask.
    device : str, optional
        Torch device. Defaults to cuda when available.
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
        class_ids = list(prompts.keys())
        concepts = [prompts[class_id] for class_id in class_ids]

        image_inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16):
            vision_embeds = self.model.get_vision_features(pixel_values=image_inputs.pixel_values)
        count = len(class_ids)
        batched_embeds = type(vision_embeds)(
            **{key: repeat_batch(value, count, torch) for key, value in vision_embeds.items()}
        )
        text_inputs = self.processor(text=concepts, return_tensors="pt", padding=True).to(
            self.device
        )
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16):
            outputs = self.model(vision_embeds=batched_embeds, **text_inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=image_inputs["original_sizes"].tolist() * count,
        )

        masks: list[np.ndarray] = []
        out_class_ids: list[int] = []
        scores: list[float] = []
        for class_id, result in zip(class_ids, results, strict=True):
            for mask, score in zip(result["masks"], result["scores"], strict=True):
                array = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
                masks.append(array.astype(bool))
                out_class_ids.append(class_id)
                scores.append(float(score))
        return semantic_from_instances(masks, out_class_ids, scores, image.shape[:2])

    def predict_batch(self, images: list[np.ndarray], prompts: dict[int, str]) -> list[np.ndarray]:
        """Segment each image in turn (all concepts are batched within a single image)."""
        return [self.predict(image, prompts) for image in images]
