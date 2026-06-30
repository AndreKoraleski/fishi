# pyright: reportMissingImports=false, reportArgumentType=false, reportCallIssue=false
"""Grounded SAM 2: Grounding DINO boxes segmented by SAM 2 (facebook/sam2-hiera-large)."""

from fishi.segmentation.grounded_sam.base import GroundedSam


class GroundedSam2(GroundedSam):
    """Grounding DINO + SAM 2."""

    name = "gdino+sam2"
    segmenter_checkpoint = "facebook/sam2-hiera-large"

    def load_segmenter(self, checkpoint: str) -> None:
        from transformers import Sam2Model, Sam2Processor

        self.segmenter_processor = Sam2Processor.from_pretrained(checkpoint)
        self.segmenter = (
            Sam2Model.from_pretrained(checkpoint, torch_dtype=self._segmenter_dtype)
            .to(self.device)
            .eval()
        )

    def segment(self, pil_image, boxes):
        torch = self._torch
        inputs = self.segmenter_processor(
            images=pil_image, input_boxes=[boxes.tolist()], return_tensors="pt"
        ).to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self._segmenter_dtype)
        with torch.no_grad():
            outputs = self.segmenter(**inputs)
        masks = self.segmenter_processor.post_process_masks(
            outputs.pred_masks.float().cpu(), inputs["original_sizes"].cpu()
        )
        best = outputs.iou_scores.float().cpu().argmax(dim=-1)
        return masks[0], best[0]
