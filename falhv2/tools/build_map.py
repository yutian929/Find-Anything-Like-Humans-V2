"""Read-only robot map acquisition; replace visible old boxes using depth."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import json
import numpy as np

from ..map_maintainers.depth_replace import DepthReplacementMap
from ..model_clients import build_cubify_anything
from ..robot_adapters.anchor_map import AnchorMapRobot


def save_coverage_svg(path, frame, builder, previous, mask):
    """Top-down display of the exact XY replacement mask; occupancy is unchanged."""
    from html import escape
    grid = builder.grid
    h, w = mask.shape
    scale = min(6., 1000/max(w, h))
    left, top = 20., 100.
    body = []
    def rect(x, y, width, height, color, stroke=None):
        body.append(f'<rect x="{left+x*scale:.3f}" y="{top+y*scale:.3f}" '
                    f'width="{width*scale:.3f}" height="{height*scale:.3f}" '
                    f'fill="{color}"'+(f' stroke="{stroke}" stroke-width="2"' if stroke else '')+'/>')
    def runs(cells, color):
        for y, row in enumerate(cells):
            indices = np.flatnonzero(np.diff(np.r_[False, row, False]))
            for a, b in indices.reshape(-1, 2):
                rect(a, y, b-a, 1, color)
    cells = np.array([list(row) for row in grid.rows])
    runs(cells == "?", "#eef0f3")
    runs(cells == "#", "#555e6b")
    runs(mask, "#8cdae0")
    def box(obj, color):
        x = (obj.lower[0]-grid.origin[0])/grid.resolution
        y = h-(obj.upper[1]-grid.origin[1])/grid.resolution
        rect(x, y, (obj.upper[0]-obj.lower[0])/grid.resolution,
             (obj.upper[1]-obj.lower[1])/grid.resolution, "none", color)
    removed = set(builder.last_update["removed_ids"])
    added = set(builder.last_update["added_ids"])
    for obj in previous:
        if obj.object_id in removed:
            box(obj, "#ec8a23")
    for obj in builder.objects:
        box(obj, "#138a45" if obj.object_id in added else "#8b4db8")
    x, y = frame.T_map_camera[:2, 3]
    x = left+(x-grid.origin[0])/grid.resolution*scale
    y = top+(h-(y-grid.origin[1])/grid.resolution)*scale
    body.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="5" fill="#152b44"/>')
    title = escape(f"Frame {frame.frame_id}: {len(removed)} removed, {len(added)} added")
    text = (
        f'<text x="20" y="25" font-size="19">{title}</text>'
        '<text x="20" y="50" font-size="14">Cyan: sampled depth coverage | gray: SLAM occupancy</text>'
        '<text x="20" y="73" font-size="14">Orange: deleted | purple: retained | green: new</text>'
        '<text x="20" y="91" font-size="12">Shared XY cell: remove old box; add only predictions intersecting cyan.</text>'
    )
    Path(path).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{max(700,w*scale+40)}" '
                         f'height="{h*scale+130}" style="background:white;font-family:sans-serif">'
                         +text+"".join(body)+"</svg>", encoding="utf-8")


def main(argv=None, *, model_config, robot_config, mapping_config, output_dir):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-id", default=robot_config.get("map_id"),
                        required=not robot_config.get("map_id"))
    parser.add_argument("--map-revision", default=robot_config.get("map_revision"),
                        required=not robot_config.get("map_revision"),
                        help="SLAM coordinate revision; change after reset/relocalization/correction")
    parser.add_argument("--endpoint", default=robot_config["endpoint"])
    parser.add_argument("--chassis-url", default=robot_config["chassis_url"])
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=5.)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume a map saved by this task")
    parser.add_argument("--coverage-stride", type=int, default=None)
    args = parser.parse_args(argv)
    if args.frames < 1 or not np.isfinite(args.timeout_s) or args.timeout_s <= 0 or (args.coverage_stride is not None and args.coverage_stride < 1):
        parser.error("frames, timeout and coverage stride must be positive")
    output = args.output or args.resume or (Path(output_dir)/(
        "robot_map_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".json"))
    if output.suffix.lower() != ".json":
        parser.error("--output must end in .json")
    if output.exists() and (args.resume is None or output.resolve() != args.resume.resolve()):
        parser.error("Output already exists; explicitly use --resume or another --output")
    mapping_config = dict(mapping_config)
    if args.coverage_stride is not None:
        mapping_config["pixel_stride"] = args.coverage_stride
    client = None
    try:
        builder = (DepthReplacementMap.load(args.resume, **mapping_config) if args.resume
                   else DepthReplacementMap(**mapping_config))
        # Reject mismatched saved maps before model loading or hardware access.
        if builder.grid is not None and (builder.grid.map_id, builder.grid.map_revision) != (
                args.map_id, args.map_revision):
            raise ValueError("Saved map coordinates differ; start a new object map")
        client = build_cubify_anything(model_config)
        with AnchorMapRobot(map_id=args.map_id, map_revision=args.map_revision,
                            endpoint=args.endpoint, chassis_url=args.chassis_url) as robot:
            for _ in range(args.frames):
                builder.set_grid(robot.grid_snapshot(timeout_s=args.timeout_s))
                frame = robot.capture(timeout_s=args.timeout_s)
                result = client.detect(frame)
                previous = builder.objects
                update = builder.integrate(frame, result)
                mask = builder.coverage
                builder.save(output)
                save_coverage_svg(output.with_suffix(".svg"), frame, builder, previous, mask)
                print(json.dumps(dict(output=str(output), frame_id=frame.frame_id,
                                      removed=len(update["removed_ids"]), added=len(update["added_ids"]),
                                      objects=len(builder.objects)), ensure_ascii=False), flush=True)
    finally:
        if client is not None:
            client.close()
    return 0
