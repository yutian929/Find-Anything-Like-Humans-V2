"""SAM3 客户端：本地(torch) + 远程(HTTP)。

仅作 semantic_query 的仲裁检测器：对候选图按英文 prompt 检测，返回实例 {bbox, score}，
供 engine 按 bbox 重合度给 RynnBrain 候选打分。不含全类别检索。
"""
from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

import config
import perception as P
from .base import SAM3Base, RemoteMixin, image_to_b64

logger = logging.getLogger("model_clients.sam3")


class SAM3Local(SAM3Base):
    def __init__(self, path: str | Path, model_id: Optional[str] = None):
        from transformers import Sam3Model, Sam3Processor
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = Path(path)
        if not (self.model_dir / "config.json").exists():
            # 本地权重缺失时尝试用 modelscope 下载（内网/离线可提前放好权重跳过）
            from modelscope import snapshot_download
            mid = str(model_id or "facebook/sam3")
            logger.info("SAM3 本地权重不存在，下载到: %s", self.model_dir)
            snapshot_download(model_id=mid, local_dir=str(self.model_dir),
                              ignore_file_pattern=["*.pt", "*.pth", "*.bin"])
        self.processor = Sam3Processor.from_pretrained(str(self.model_dir), local_files_only=True)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = Sam3Model.from_pretrained(
            str(self.model_dir), local_files_only=True, torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    def _autocast(self):
        if self.device == "cuda":
            return self.torch.amp.autocast("cuda", dtype=self.torch.bfloat16)
        return nullcontext()

    def detect(
        self, images: List[Image.Image], prompt: str,
        threshold: Optional[float] = None, mask_threshold: Optional[float] = None,
    ) -> List[List[Dict[str, Any]]]:
        if not images:
            return []
        thr = float(config.SAM3_DEFAULT_THRESHOLD if threshold is None else threshold)
        mthr = float(config.SAM3_DEFAULT_MASK_THRESHOLD if mask_threshold is None else mask_threshold)
        othr = float(config.SAM3_OVERLAP_THRESHOLD)
        inputs = self.processor(images=images, text=[prompt] * len(images), return_tensors="pt").to(self.device)
        with self.torch.inference_mode(), self._autocast():
            outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs, threshold=thr, mask_threshold=mthr,
            target_sizes=inputs.get("original_sizes").tolist(),
        )
        out: List[List[Dict[str, Any]]] = []
        for r in results:
            boxes = r["boxes"].detach().cpu().numpy()
            scores = r["scores"].detach().float().cpu().numpy()
            masks = r["masks"].detach().cpu().numpy()
            instances = P.filter_nested_instances(masks, boxes, scores, othr)
            # 仲裁只需 bbox + score
            out.append([{"bbox": [float(v) for v in inst["bbox"]], "score": float(inst["score"])}
                        for inst in instances])
        return out


class SAM3Remote(RemoteMixin, SAM3Base):
    def detect(
        self, images: List[Image.Image], prompt: str,
        threshold: Optional[float] = None, mask_threshold: Optional[float] = None,
    ) -> List[List[Dict[str, Any]]]:
        if not images:
            return []
        payload = {
            "images_b64": [image_to_b64(i) for i in images],
            "prompt": prompt,
            "threshold": config.SAM3_DEFAULT_THRESHOLD if threshold is None else threshold,
            "mask_threshold": config.SAM3_DEFAULT_MASK_THRESHOLD if mask_threshold is None else mask_threshold,
        }
        data = self._post("/detect", payload)
        out: List[List[Dict[str, Any]]] = []
        for item in data.get("results", []):
            instances = item.get("instances", []) if isinstance(item, dict) else []
            out.append([{"bbox": [float(v) for v in inst["bbox"]], "score": float(inst.get("score", 0.0))}
                        for inst in instances])
        return out
