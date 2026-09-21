"""模型客户端工厂：按 config 的 mode 建本地或远程实现。

用法：
    from model_clients import build_rynnbrain, build_sam2, build_sam3
    rynnbrain = build_rynnbrain(config.RYNNBRAIN)
本地实现按需 import torch/transformers；远程实现只需 requests。
"""
from __future__ import annotations

from typing import Any, Dict

from .base import RynnBrainBase, SAM2Base, SAM3Base


def _mode(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("mode", "local")).lower()


def build_rynnbrain(cfg: Dict[str, Any]) -> RynnBrainBase:
    if _mode(cfg) == "remote":
        from .rynnbrain import RynnBrainRemote
        return RynnBrainRemote(cfg["url"], timeout=cfg.get("timeout", 30.0), batch_size=cfg.get("batch_size", 4))
    from .rynnbrain import RynnBrainLocal
    return RynnBrainLocal(cfg["path"], device=cfg.get("device", "auto"), batch_size=cfg.get("batch_size", 4))


def build_sam2(cfg: Dict[str, Any]) -> SAM2Base:
    if _mode(cfg) == "remote":
        from .sam2 import SAM2Remote
        return SAM2Remote(cfg["url"], timeout=cfg.get("timeout", 30.0))
    from .sam2 import SAM2Local
    return SAM2Local(cfg["path"], model_id=cfg.get("model_id"))


def build_sam3(cfg: Dict[str, Any]) -> SAM3Base:
    if _mode(cfg) == "remote":
        from .sam3 import SAM3Remote
        return SAM3Remote(cfg["url"], timeout=cfg.get("timeout", 30.0))
    from .sam3 import SAM3Local
    return SAM3Local(cfg["path"], model_id=cfg.get("model_id"))


__all__ = ["build_rynnbrain", "build_sam2", "build_sam3",
           "RynnBrainBase", "SAM2Base", "SAM3Base"]
