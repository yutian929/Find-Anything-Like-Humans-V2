"""FastAPI 服务：控制台 UI + 3D 地图页 + /api/*（单进程单源，无跨节点代理）。

所有 /api/* 请求最终落到 Engine（检索/检测/控制走队列，状态/图片只读）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse, Response

import config
import perception as P
from engine import Engine

logger = logging.getLogger("server")


def _command(path: str, params: dict, action):
    logger.info("HTTP POST %s params=%s", path, params)
    return action()


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="Find-Anything-Like-Humans V2 · anchor-map")
    static_dir = Path(config.STATIC_DIR)
    results_dir = engine.results_dir

    # ---------------- 页面 ---------------- #
    @app.get("/")
    def index():
        return FileResponse(str(static_dir / "index.html"))

    @app.get("/map")
    def map_page():
        return FileResponse(str(static_dir / "map.html"))

    # ---------------- 状态 / 只读 ---------------- #
    @app.get("/api/status")
    def api_status():
        return engine.status()

    @app.get("/api/anchors")
    def api_anchors():
        anchors = engine.store.anchors()
        return {"success": True, "anchors": anchors, "num_anchors": len(anchors)}

    @app.get("/api/objects")
    def api_objects():
        return {"success": True, "history": engine.history()}

    @app.get("/api/predefined_objects")
    def api_predefined_objects(scope: str = "full"):
        objs = (config.PREDEFINED_QUERY_OBJECTS_PARTIAL if str(scope).lower() == "partial"
                else config.PREDEFINED_QUERY_OBJECTS_FULL)
        return {"success": True, "scope": scope, "objects": objs}

    @app.get("/api/rgb")
    def api_rgb():
        frame = engine.source.latest()
        if frame is None:
            return JSONResponse(status_code=503, content={"success": False, "error": "无当前帧"})
        ok, buf = cv2.imencode(".jpg", frame.rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return JSONResponse(status_code=500, content={"success": False, "error": "encode failed"})
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    @app.get("/api/map/meta")
    def api_map_meta():
        occ = engine.source.occupancy_map()
        if occ is None:
            return {"success": False, "has_map": False}
        return {"success": True, "has_map": True, "resolution": occ.resolution,
                "origin": occ.origin, "width": occ.width, "height": occ.height, "version": occ.version}

    @app.get("/api/map/image")
    def api_map_image():
        occ = engine.source.occupancy_map()
        if occ is None:
            return JSONResponse(status_code=404, content={"success": False, "error": "无地图"})
        return Response(content=occ.png_bytes, media_type="image/png")

    @app.get("/api/results/{object_name}/{asset}")
    def api_results_asset(object_name: str, asset: str):
        path = results_dir / P.safe_object_name(object_name) / asset
        if not path.exists():
            return JSONResponse(status_code=404, content={"success": False, "error": "not found"})
        return FileResponse(str(path))

    # ---------------- 检索 / 检测 ---------------- #
    @app.post("/api/query")
    def api_query(
        object: str = Form(...),
        x_min: str = Form(""), x_max: str = Form(""),
        y_min: str = Form(""), y_max: str = Form(""),
    ):
        xy = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
        return _command("/api/query", {"object": object, **xy},
                        lambda: engine.semantic_query(object, xy))

    @app.post("/api/detect")
    def api_detect(object: str = Form(...)):
        return _command("/api/detect", {"object": object},
                        lambda: engine.instant_detect(object))

    # ---------------- 锚点控制 ---------------- #
    @app.post("/api/anchor/build")
    def api_anchor_build():
        return _command("/api/anchor/build", {}, engine.build_anchor)

    @app.post("/api/anchor/{anchor_id}/delete")
    def api_anchor_delete(anchor_id: str):
        return _command(f"/api/anchor/{anchor_id}/delete", {"anchor_id": anchor_id},
                        lambda: engine.delete_anchor(anchor_id))

    @app.post("/api/anchors/clear")
    def api_anchors_clear():
        return _command("/api/anchors/clear", {}, engine.clear_anchors)

    @app.post("/api/history/clear")
    def api_history_clear():
        return _command("/api/history/clear", {}, engine.clear_history)

    return app
