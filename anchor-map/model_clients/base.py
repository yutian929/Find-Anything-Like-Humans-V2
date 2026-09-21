"""模型客户端抽象基类 + 远程 HTTP 协议辅助。

每个模型都有 Local（本地 torch 加载）与 Remote（连服务器端口）两实现，
统一实现下面的抽象方法；上层（engine）只面向抽象接口，不关心本地/远程。

远程 HTTP 协议 v1（JSON over POST）：
- 图像 image_b64 = base64(JPEG)；mask 回传 mask_b64 = base64(PNG 单通道 0/255)。
- bbox 统一 [x1, y1, x2, y2]（像素整数）；RynnBrain 归一化 bbox 为 [0,1000]。
- 每服务 GET /healthz -> {"ok": true, ...}；出错 4xx/5xx + {"error": "..."}。
"""
from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# 编解码辅助
# --------------------------------------------------------------------------- #
def image_to_b64(image: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def mask_from_b64(mask_b64: str) -> np.ndarray:
    raw = base64.b64decode(mask_b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    return (np.asarray(img) > 0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 远程调用 mixin
# --------------------------------------------------------------------------- #
class RemoteMixin:
    """基于 requests 的 JSON-over-HTTP 客户端。"""

    def __init__(self, url: str, timeout: float = 30.0):
        import requests  # 延迟导入：纯本地部署无需 requests
        self._requests = requests
        self.url = str(url).rstrip("/")
        self.timeout = float(timeout)

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._requests.post(f"{self.url}{path}", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def healthz(self) -> Dict[str, Any]:
        try:
            resp = self._requests.get(f"{self.url}/healthz", timeout=5)
            return {"ok": resp.ok, **(resp.json() if resp.ok else {})}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


# --------------------------------------------------------------------------- #
# 抽象接口
# --------------------------------------------------------------------------- #
class RynnBrainBase(ABC):
    """开放词表存在性检测 + bbox 定位 + 中译英。"""

    batch_size: int = 4

    @abstractmethod
    def exists_batch(self, images: List[Image.Image], obj: str) -> List[bool]: ...

    @abstractmethod
    def bbox_batch(self, images: List[Image.Image], obj: str) -> List[Optional[List[int]]]:
        """返回与 images 等长的归一化 bbox [x1,y1,x2,y2]（0-1000）或 None。"""

    @abstractmethod
    def translate(self, zh: str) -> str: ...

    def recall_batch(self, images: List[Image.Image], obj: str) -> List[Optional[List[int]]]:
        """两阶段召回（不早退）：先存在性过滤，再对通过者出 bbox。返回归一化 bbox 或 None。"""
        n = len(images)
        if n == 0:
            return []
        exists = self.exists_batch(images, obj)
        keep_idx = [i for i, ok in enumerate(exists) if ok]
        results: List[Optional[List[int]]] = [None] * n
        if not keep_idx:
            return results
        bboxes = self.bbox_batch([images[i] for i in keep_idx], obj)
        for idx, bbox in zip(keep_idx, bboxes):
            results[idx] = bbox
        return results

    def detect_one(self, image: Image.Image, obj: str) -> Optional[List[int]]:
        """单图召回：存在性通过则返回归一化 bbox，否则 None。"""
        return self.recall_batch([image], obj)[0]


class SAM2Base(ABC):
    @abstractmethod
    def segment(self, image: Image.Image, bbox_pixel: List[int]) -> np.ndarray:
        """返回 HxW uint8 二值 mask。"""


class SAM3Base(ABC):
    @abstractmethod
    def detect(
        self, images: List[Image.Image], prompt: str,
        threshold: Optional[float] = None, mask_threshold: Optional[float] = None,
    ) -> List[List[Dict[str, Any]]]:
        """对每张图返回该 prompt 的实例列表 [{bbox:[x1,y1,x2,y2], score:float}, ...]。"""
