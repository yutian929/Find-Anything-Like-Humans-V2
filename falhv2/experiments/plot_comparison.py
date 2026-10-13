"""Render the measured user-map routes with a bounded image size."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/falhv2-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def main():
    folder = Path(__file__).resolve().parents[1] / "outputs/coverage_experiment"
    result = json.loads((folder / "comparison.json").read_text())
    case = next(c for c in result["cases"] if c["name"] == "user_map")
    geometry = np.load(folder / "user_map_geometry.npz")
    labels, blocked = geometry["labels"], geometry["blocked"]
    resolution = float(geometry["resolution"])
    height, width = labels.shape
    background = np.where(blocked, 0, np.where(labels > 0, 2, 1))
    titles = {"greedy": "Nearest-unvisited BFS", "tree_walk": "Iterative tree walk",
              "scanline_sweep": "Row-interval sweep prototype", "estc_upstream": "Upstream ESTC"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), layout="constrained")
    for axis, (method, title) in zip(axes.flat, titles.items()):
        metrics = case["algorithms"][method]
        axis.imshow(background, origin="upper", extent=(0,width*resolution,0,height*resolution),
                    cmap=ListedColormap(["#253348", "#d7dce3", "#f0f8f4"]), vmin=0, vmax=2)
        if metrics["status"] == "complete":
            with np.load(folder / f"user_map_{method}.npz") as paths:
                for region in paths.files:
                    p = paths[region]
                    axis.plot((p[:,1]+0.5)*resolution, (height-p[:,0]-0.5)*resolution,
                              color="#087f8c", linewidth=0.45, alpha=0.8)
            subtitle = f"{metrics['path_length_m']:.2f} m | {metrics['quarter_turns']} quarter-turns | {metrics['median_seconds']:.3f} s"
        else:
            subtitle = metrics["status"]
        axis.set_title(f"{title}\n{subtitle}", fontsize=11)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal")
    fig.suptitle("Same 8,842 safe cell centers; 100% coverage; disconnected regions planned separately", fontsize=13)
    fig.savefig(folder / "user_map_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
