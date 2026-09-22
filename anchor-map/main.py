"""anchor-map 单进程入口：装配 FrameSource + AnchorStore + Engine + HTTP Server。

    python main.py

浏览器打开 http://127.0.0.1:9200/ 进入控制台。
"""
from __future__ import annotations

import logging
import os

import config

# CUDA / HF 离线设置须在 import torch/transformers 前完成
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
if getattr(config, "HF_OFFLINE", False):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import uvicorn

from anchor_store import AnchorStore
from engine import Engine
from frame_source import build_frame_source
from server import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("main")


def main() -> None:
    from pathlib import Path
    anchors_dir = Path(config.MAP_ROOT) / "anchors"
    source = build_frame_source()
    store = AnchorStore(anchors_dir, xy_thresh=config.XY_THRESH, yaw_thresh_deg=config.YAW_THRESH_DEG)
    engine = Engine(source, store)
    engine.start()
    app = create_app(engine)
    logger.info("anchor-map 就绪：http://%s:%d/", config.HOST, config.PORT)
    try:
        uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
    finally:
        engine.stop()
        close = getattr(source, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
