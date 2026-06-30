# pyright: reportMissingImports=false, reportArgumentType=false, reportCallIssue=false
"""Grounded SAM 1: Grounding DINO boxes segmented by SAM 1 (facebook/sam-vit-huge)."""

from fishi.segmentation.grounded_sam.base import GroundedSam


class GroundedSam1(GroundedSam):
    """Grounding DINO + SAM 1."""

    name = "gdino+sam1"
    segmenter_checkpoint = "facebook/sam-vit-huge"

    def load_segmenter(self, checkpoint: str) -> None:
        from transformers import SamModel, SamProcessor

        self.segmenter_processor = SamProcessor.from_pretrained(checkpoint)
        self.segmenter = (
            SamModel.from_pretrained(checkpoint, torch_dtype=self._segmenter_dtype)
            .to(self.device)
            .eval()
        )

    def segment(self, pil_image, boxes):
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
        )
        best = outputs.iou_scores.float().cpu().argmax(dim=-1)
        return masks[0], best[0]
