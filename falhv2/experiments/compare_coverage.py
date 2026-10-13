"""Reproducible full-cell coverage benchmark with isolated, time-limited workers."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/falhv2-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import scipy
from scipy.ndimage import label

from benchmark_preprocessing import distance_field, synthetic_scene, verify_small_maps
from coverage_candidates import greedy, tree_walk, scanline_sweep, load_estc, open_route, turns
from path_planner import GridMap, Robot

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("greedy", "tree_walk", "scanline_sweep", "estc_upstream")


def metrics(paths, regions, resolution, dense, radius):
    if len(paths) != len(regions):
        raise ValueError("Expected one route per connected region")
    steps = events = quarter = count = 0
    for path, cells in zip(paths, regions):
        if set(path) != cells:
            raise ValueError(f"Coverage mismatch: missed={len(cells-set(path))}, outside={len(set(path)-cells)}")
        arr = np.asarray(path, dtype=int)
        if len(arr) > 1:
            if not np.all(np.abs(np.diff(arr, axis=0)).sum(axis=1) == 1):
                raise ValueError("Route contains jumps or duplicate adjacent waypoints")
            midpoint = arr[:-1] + arr[1:] + 1
            if not np.all(dense[midpoint[:, 0], midpoint[:, 1]] + 1e-10 >= radius):
                raise ValueError("An edge violates footprint clearance")
        if not np.all(dense[2*arr[:, 0]+1, 2*arr[:, 1]+1] + 1e-10 >= radius):
            raise ValueError("A vertex violates footprint clearance")
        e, q = turns(path)
        events += e
        quarter += q
        steps += len(path)-1
        count += len(cells)
    return {"coverage_fraction": 1.0, "collision_free": True,
            "path_length_m": steps*resolution, "turn_events": events,
            "quarter_turns": quarter, "waypoints": steps+len(paths),
            "revisit_fraction": (steps+len(paths)-count)/count if count else 0,
            "length_over_vertex_lower_bound": steps/(count-len(paths)) if count > len(paths) else 1}


def worker(args):
    data = np.load(args.input)
    labels = data["labels"]
    regions = [set(map(tuple, np.argwhere(labels == i))) for i in range(1, int(labels.max())+1)]
    fn = load_estc(args.checkout) if args.method == "estc_upstream" else globals()[args.method]
    times = []
    paths = None
    fingerprints = []
    for _ in range(args.repeats):
        started = perf_counter()
        current = [open_route(fn(cells)) for cells in regions]
        times.append(perf_counter()-started)
        fingerprints.append(hashlib.sha256(repr(current).encode()).hexdigest())
        current_metrics = metrics(current, regions, float(data["resolution"]), data["dense"], float(data["radius"]))
        paths = current
    result = {"status": "complete", "seconds": times, "median_seconds": statistics.median(times),
              "deterministic_within_process": len(set(fingerprints)) == 1, **current_metrics}
    Path(args.result).write_text(json.dumps(result, indent=2)+"\n")
    if args.routes:
        np.savez_compressed(args.routes, **{f"region_{i}": np.asarray(p, dtype=np.int32) for i,p in enumerate(paths)})


def cases():
    real = json.loads((ROOT / "outputs/map.json").read_text())
    config = json.loads((ROOT / "config/robot.example.json").read_text())
    yield "user_map", real, config
    zero_margin = json.loads(json.dumps(config))
    zero_margin["robot"]["safety_margin_m"] = 0.0
    yield "user_map_zero_margin", real, zero_margin
    for size in (300, 512):
        yield f"rectangles_{size}", synthetic_scene(size), config
    # Thin passages, unknown walls, odd dimensions, single-cell components.
    cells = np.full((65, 81), "#", dtype="<U1")
    for i, row in enumerate(range(2, 61, 4)):
        cells[row, 2:65] = "."
        if row+4 <= 60:
            cells[row:row+5, 64 if i % 2 == 0 else 2] = "."
    cells[4:18, 70:78] = "."
    cells[30, 75] = "."
    cells[0, :] = "?"
    small_robot = {"robot": {"footprint_width_m": 0.28, "footprint_length_m": 0.28, "safety_margin_m": 0.0}}
    yield "thin_disconnected", {"grid": {"rows": ["".join(row) for row in cells], "resolution": 0.5, "origin": [2.3,-1.7]}}, small_robot
    # Several obstacle islands create split/merge events in scanline decomposition.
    cells = np.full((120, 160), ".", dtype="<U1")
    cells[[0,-1], :] = "#"
    cells[:, [0,-1]] = "#"
    rng = np.random.default_rng(41)
    for _ in range(65):
        row, col = int(rng.integers(2,108)), int(rng.integers(2,148))
        cells[row:row+int(rng.integers(2,10)), col:col+int(rng.integers(2,10))] = "#"
    yield "cluttered_120x160", {"grid": {"rows": ["".join(row) for row in cells], "resolution": 0.1, "origin": [0,0]}}, small_robot


def main(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    sha = subprocess.check_output(["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True).strip()
    result = {"baseline_commit": "7eadb72", "environment": {"python": platform.python_version(),
              "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
              "estc_source": {"url": "https://github.com/reso1/LS-MCPP", "commit": sha, "unmodified": True},
              "repeats": args.repeats, "worker_timeout_seconds": args.timeout,
              "timing_scope": "routing includes candidate/graph building and open-route normalization; excludes imports, shared preprocessing, validation, plotting and IO",
              "coverage_contract": "all safe cell centers, four-connected paths, one path per connected region; no prescribed endpoints",
              "small_map_clearance_check": verify_small_maps(), "cases": []}
    specifications = json.loads(args.fixtures.read_text()) if args.fixtures else list(cases())
    (out / "fixtures.json").write_text(json.dumps(specifications, ensure_ascii=False)+"\n")
    with tempfile.TemporaryDirectory(prefix="falhv2-benchmark-") as temp:
        for name, scene, config in specifications:
            grid = GridMap.from_scene(scene)
            robot = Robot.from_config(config)
            durations = []
            for _ in range(args.repeats):
                started = perf_counter()
                blocked, dense = distance_field(grid, full=True)
                safe = (~blocked) & (dense[1::2,1::2] + 1e-10 >= robot.clearance_m)
                labels, n = label(safe)
                durations.append(perf_counter()-started)
            case = {"name": name, "shape": [grid.height,grid.width], "resolution_m": grid.resolution,
                    "robot": config["robot"], "grid_sha256": hashlib.sha256(json.dumps(scene["grid"],sort_keys=True).encode()).hexdigest(),
                    "safe_cells": int(safe.sum()), "region_count": n,
                    "preprocessing_seconds": durations, "median_preprocessing_seconds": statistics.median(durations), "algorithms": {}}
            result["cases"].append(case)
            npz = Path(temp)/f"{name}.npz"
            np.savez_compressed(npz, labels=labels, dense=dense, resolution=grid.resolution, radius=robot.clearance_m)
            if name == "user_map":
                np.savez_compressed(out / "user_map_geometry.npz", labels=labels, blocked=blocked, resolution=grid.resolution)
            for method in METHODS:
                target = Path(temp)/f"{name}_{method}.json"
                command = [sys.executable, "-B", __file__, "--worker", "--input", str(npz), "--method", method,
                           "--checkout", str(args.checkout), "--result", str(target), "--repeats", str(args.repeats)]
                if name == "user_map":
                    command += ["--routes", str(out / f"user_map_{method}.npz")]
                started = perf_counter()
                try:
                    proc = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout,
                                          env={**os.environ, "PYTHONHASHSEED": "0"})
                    measured = json.loads(target.read_text()) if proc.returncode == 0 else {"status": "error", "returncode": proc.returncode, "stderr": proc.stderr[-2500:]}
                except subprocess.TimeoutExpired:
                    measured = {"status": "timeout", "worker_wall_seconds": perf_counter()-started}
                case["algorithms"][method] = measured
                (out / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
                print(name, method, measured.get("status"), measured.get("median_seconds"), measured.get("path_length_m"), measured.get("quarter_turns"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path("/tmp/falhv2-ls-mcpp"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/coverage_experiment")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--fixtures", type=Path, help="Replay the saved scene/robot snapshots")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--routes", type=Path)
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    worker(args) if args.worker else main(args)
