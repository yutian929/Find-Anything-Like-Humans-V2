"""Try the recommended EDT + interval sweep pipeline without changing A1's default."""
import argparse
import json
import math
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/falhv2-matplotlib")
import numpy as np
from scipy.ndimage import label

from benchmark_preprocessing import distance_field
from coverage_candidates import greedy, tree_walk, scanline_sweep, load_estc, open_route
from compare_coverage import metrics
from path_planner import GridMap, Robot, PlanningError

ROOT = Path(__file__).resolve().parents[1]


def plan_candidate(scene, config, method="scanline_sweep", checkout="/tmp/falhv2-ls-mcpp"):
    grid, robot = GridMap.from_scene(scene), Robot.from_config(config)
    started = perf_counter()
    blocked, dense = distance_field(grid, full=True)
    safe = (~blocked) & (dense[1::2,1::2]+1e-10 >= robot.clearance_m)
    labels, count = label(safe)
    if not count:
        raise PlanningError("no_traversable_cells", "机器人尺寸与安全距离下没有可通行格中心")
    cells = [set(map(tuple, np.argwhere(labels == i))) for i in range(1,count+1)]
    preprocessing = perf_counter()-started
    methods = {"greedy": greedy, "tree_walk": tree_walk, "scanline_sweep": scanline_sweep}
    fn = load_estc(checkout) if method == "estc_upstream" else methods[method]
    started = perf_counter()
    paths = [open_route(fn(region)) for region in cells]
    routing = perf_counter()-started
    started = perf_counter()
    checked = metrics(paths, cells, grid.resolution, dense, robot.clearance_m)
    validation = perf_counter()-started
    regions = []
    for i, (region, route) in enumerate(zip(cells, paths), 1):
        positions = [grid.cell_to_world(cell) for cell in route]
        waypoints = []
        yaw = 0.0
        for j, (x, y) in enumerate(positions):
            if j+1 < len(positions):
                nx, ny = positions[j+1]
                yaw = math.atan2(ny-y,nx-x)
            waypoints.append({"x_m": x, "y_m": y, "yaw_rad": yaw})
        regions.append({"id": i, "cells": [list(map(int, cell)) for cell in sorted(region)],
                        "route_cells": [list(map(int, cell)) for cell in route],
                        "waypoints": waypoints, "path_length_m": (len(route)-1)*grid.resolution})
    return {"status": "complete", "experimental": True,
            "scope": "all_traversable_cell_centers_by_connected_region",
            "scene_name": scene.get("name", "unnamed"), "robot": config["robot"],
            "grid": {"width": grid.width, "height": grid.height, "resolution": grid.resolution, "origin": list(grid.origin)},
            "internal_config": {"path_method": method, "clearance_method": "half_grid_edt",
                                "connectivity": 4, "unknown_cells_passable": False,
                                "assumes_in_place_rotation": True, "no_external_endpoints": True},
            "statistics": {"free_cells": int((~blocked).sum()), "traversable_cells": int(safe.sum()),
                           "region_count": count, "total_path_length_m": checked["path_length_m"],
                           "visited_traversable_cells": int(safe.sum()), "total_waypoint_count": checked["waypoints"],
                           "quarter_turns": checked["quarter_turns"], "preprocessing_seconds": preprocessing,
                           "routing_seconds": routing, "validation_seconds": validation},
            "regions": regions, "visual_object_coverage": "not_evaluated"}


def render(scene, result, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    grid = GridMap.from_scene(scene)
    background = np.array([[0 if v != "." else 1 for v in row] for row in grid.rows])
    for region in result["regions"]:
        for r,c in region["cells"]:
            background[r,c] = 2
    fig, ax = plt.subplots(figsize=(8,8), layout="constrained")
    x0,y0 = grid.origin
    ax.imshow(background, extent=(x0,x0+grid.width*grid.resolution,y0,y0+grid.height*grid.resolution),
              cmap=ListedColormap(["#253348","#d7dce3","#f0f8f4"]), vmin=0,vmax=2)
    for region in result["regions"]:
        p=region["waypoints"]
        ax.plot([w["x_m"] for w in p],[w["y_m"] for w in p],color="#087f8c",lw=0.5)
    ax.set_title(f"{result['internal_config']['path_method']} | {result['statistics']['total_path_length_m']:.2f} m")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.savefig(destination,dpi=150)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene",type=Path,default=ROOT/"outputs/map.json")
    parser.add_argument("--robot",type=Path,default=ROOT/"config/robot.example.json")
    parser.add_argument("--method",choices=["greedy","tree_walk","scanline_sweep","estc_upstream"],default="scanline_sweep")
    parser.add_argument("--checkout",type=Path,default=Path("/tmp/falhv2-ls-mcpp"))
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/coverage_experiment/selected_plan.json")
    args=parser.parse_args()
    scene=json.loads(args.scene.read_text()); config=json.loads(args.robot.read_text())
    result=plan_candidate(scene,config,args.method,args.checkout)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False)+"\n")
    render(scene,result,args.output.with_suffix(".png"))
    print(json.dumps(result["statistics"],ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
