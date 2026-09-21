"""SAM2 客户端：本地(torch) + 远程(HTTP)。bbox -> 精分割 mask。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from .base import SAM2Base, RemoteMixin, image_to_b64, mask_from_b64

logger = logging.getLogger("model_clients.sam2")


class SAM2Local(SAM2Base):
    def __init__(self, path: str | Path, model_id: Optional[str] = None):
        from transformers import AutoModelForMaskGeneration as Sam2Model
        from transformers import AutoProcessor as Sam2Processor
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        local = Path(path)
        if (local / "config.json").exists():
            logger.info("Loading SAM2 (local offline): %s", local)
            self.model = Sam2Model.from_pretrained(str(local), local_files_only=True).to(self.device)
            self.processor = Sam2Processor.from_pretrained(str(local), local_files_only=True)
        else:
            mid = str(model_id or "facebook/sam2.1-hiera-small")
            logger.info("Loading SAM2 by model_id: %s", mid)
            self.model = Sam2Model.from_pretrained(mid).to(self.device)
            self.processor = Sam2Processor.from_pretrained(mid)
        self.model.eval()

    def segment(self, image: Image.Image, bbox_pixel: List[int]) -> np.ndarray:
        inputs = self.processor(images=image, input_boxes=[[bbox_pixel]], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        # 兼容 transformers 4.x/5.x：reshaped_input_sizes 仅 4.x 提供
        kwargs = {"masks": outputs.pred_masks.cpu(), "original_sizes": inputs.original_sizes.cpu()}
        reshaped = inputs.get("reshaped_input_sizes")
        if reshaped is not None:
            kwargs["reshaped_input_sizes"] = reshaped.cpu()
        masks = self.processor.post_process_masks(**kwargs)[0]
        if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
            best_idx = int(outputs.iou_scores[0, 0].argmax().item())
        else:
            best_idx = 0
        return (masks[0][best_idx].numpy() > 0).astype(np.uint8)


class SAM2Remote(RemoteMixin, SAM2Base):
    def segment(self, image: Image.Image, bbox_pixel: List[int]) -> np.ndarray:
        data = self._post("/segment", {"image_b64": image_to_b64(image), "bbox": [int(v) for v in bbox_pixel]})
        mask_b64 = data.get("mask_b64")
        if not mask_b64:
            return np.zeros((image.height, image.width), dtype=np.uint8)
        return mask_from_b64(mask_b64)
