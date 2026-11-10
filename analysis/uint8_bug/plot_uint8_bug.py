"""Explainer figure for the uint8 distance-map bug fixed in 59a7235.

Usage: python analysis/uint8_bug/plot_uint8_bug.py MAP_PGM OUTPUT_PNG

Runs the real dismapConstruction_start_target twice on one fully known map:
the current version (float64) and the pre-fix version read from git
(59a7235^), both given the uint8 belief encoding research sessions use.
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "grid_simulator"))

FIXED, BUGGY = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
BLUES.set_bad("#e1e0d9")

plt.rcParams.update({
    "font.family": ["system-ui", "-apple-system", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def distance_functions():
    fixed = load_module("GridEnv_fixed", ROOT / "grid_simulator" / "GridEnv.py")
    source = subprocess.run(["git", "-C", str(ROOT), "show", "59a7235^:grid_simulator/GridEnv.py"],
                            check=True, capture_output=True, text=True).stdout
    handle = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    handle.write(source)
    handle.close()
    buggy = load_module("GridEnv_prefix", handle.name)
    # The method never touches self, so it can be called unbound.
    return (lambda start, grid: fixed.GridEnv.dismapConstruction_start_target(None, start, grid),
            lambda start, grid: buggy.GridEnv.dismapConstruction_start_target(None, start, grid))


def main(map_path, output):
    grid = np.array(Image.open(map_path), dtype=np.uint8)
    free = grid == 254
    rows, columns = np.nonzero(free)
    first = np.argmin(rows + columns)
    start = [int(rows[first]), int(columns[first])]
    fixed_distance, buggy_distance = distance_functions()
    true = np.asarray(fixed_distance(start, grid.copy()), dtype=float)
    seen = np.asarray(buggy_distance(start, grid.copy()), dtype=float)
    reached = free & (true > 0)
    print(f"start {start}, max true distance {true[reached].max():.0f} cells, "
          f"cells beyond 255: {(true[reached] > 255).mean():.0%}, "
          f"buggy == true mod 256 on {(seen[reached] == true[reached] % 256).mean():.1%} of cells")

    figure = plt.figure(figsize=(16, 5.6))
    layout = figure.add_gridspec(1, 3, width_ratios=[1, 1, 1.3], wspace=0.15)
    axes = [figure.add_subplot(layout[0, column]) for column in range(3)]
    top = true[reached].max()
    for ax, values, title in ((axes[0], true, "What the distance really is (fixed)"),
                              (axes[1], seen, "What Cost saw before the fix (uint8)")):
        shown = np.where(reached, values, np.nan)
        image = ax.imshow(shown, cmap=BLUES, vmin=0, vmax=top, interpolation="nearest")
        ax.plot(start[1], start[0], marker="*", markersize=16, color=INK, markeredgecolor=SURFACE)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xticks([]), ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    bar = figure.colorbar(image, ax=axes[:2], orientation="horizontal", fraction=0.05, pad=0.04, aspect=40)
    bar.set_label("Path distance from the robot (cells, 0.1 m each). Light = looks near: before the fix, "
                  "the far rooms turn light again once the path passes 255 cells.")
    bar.outline.set_visible(False)
    axes[0].text(start[1] + 6, start[0] - 4, "robot", color=INK, fontsize=10, va="bottom")

    ax = axes[2]
    order = np.argsort(true[reached])
    x, y = true[reached][order], seen[reached][order]
    ax.plot([0, top], [0, top], color=FIXED, linewidth=2, label="Fixed (float64)")
    ax.scatter(x, y, s=3, color=BUGGY, linewidths=0, label="Before fix (uint8)", rasterized=True, zorder=3)
    for wrap in range(256, int(top) + 1, 256):
        ax.axvline(wrap, color=AXIS, linewidth=1, linestyle="--", zorder=0)
    ax.text(150, 112, "identical up to 255", color=INK2, fontsize=10, rotation=0, ha="left")
    ax.annotate("260 cells away reads as 4", xy=(261, 4), xytext=(300, 4), color=INK2, fontsize=10,
                va="center", arrowprops={"arrowstyle": "->", "color": MUTED})
    ax.set_ylim(-12, top + 15)
    ax.set_xlabel("True distance to a cell (cells)")
    ax.set_ylabel("Distance Cost used to rank frontiers (cells)")
    ax.set_title("Every 256 cells the count wraps to zero", loc="left", fontsize=12)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", markerscale=6, labelcolor=INK2)
    figure.suptitle(f"uint8 distance overflow on {Path(map_path).stem} (Cost goes to the nearest frontier)",
                    x=0.01, ha="left", color=INK, fontsize=14)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main(*sys.argv[1:])
