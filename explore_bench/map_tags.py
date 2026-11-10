"""Shape tags for a Level-0 occupancy map, so results can be split by map type.

Usage: python -m explore_bench.map_tags MAP.pgm [MAP.pgm ...]   (prints CSV)

A map is a 2D grid of 254 = free, 0 = wall, 205 = outside, at 0.1 m per cell.
Tags describe the map's shape only; how hard a map is for a method comes
from runs, not from here. Source facts such as room or door counts are not
computed: map-set manifests copy them from the generator or HouseExpo.
"""

import csv
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

FREE, WALL, OUTSIDE = 254, 0, 205
FIELDS = ["map_name", "free_area_m2", "free_components", "loops", "largest_island_m2", "narrow_share",
          "geodesic_diameter_m", "free_height_m", "free_width_m", "extent"]


def load(path):
    grid = np.asarray(Image.open(path))
    if grid.ndim != 2 or not set(np.unique(grid)) <= {FREE, WALL, OUTSIDE}:
        raise ValueError("map must be a 2D occupancy grid containing 0/205/254")
    return grid


def bfs_farthest(free, start):
    """4-connected walk from start: (farthest cell, its distance in cells).

    4-connected because Cost's distance map (dismapConstruction_start_target
    in grid_simulator/GridEnv.py) floods the same neighbours.
    """
    distance = np.full(free.shape, -1, dtype=np.int64)
    distance[start] = 0
    queue, farthest = deque([start]), start
    while queue:
        row, column = queue.popleft()
        if distance[row, column] > distance[farthest]:
            farthest = (row, column)
        for nearby in ((row + 1, column), (row - 1, column), (row, column + 1), (row, column - 1)):
            if (0 <= nearby[0] < free.shape[0] and 0 <= nearby[1] < free.shape[1]
                    and free[nearby] and distance[nearby] < 0):
                distance[nearby] = distance[row, column] + 1
                queue.append(nearby)
    return farthest, int(distance[farthest])


def tags(grid, resolution=0.1, narrow_m=0.5, small_fraction=0.5):
    """Return the shape tags of one occupancy grid.

    free_area_m2         free cells x cell area.
    free_components      4-connected free regions; benchmark maps should have 1.
    loops                wall islands with free space all round them (a pillar
                         or a wall between two doors). Walls use 8-connectivity,
                         the dual of 4-connected free space, so two wall cells
                         touching only at a corner still block a 4-connected
                         walk and count as one island.
    narrow_share         share of free cells closer than narrow_m to a wall or
                         outside, so a corridor up to about 2 x narrow_m wide
                         is entirely narrow.
    geodesic_diameter_m  longest 4-connected walk inside the largest free region,
                         by two BFS sweeps: exact when that region has no loops,
                         otherwise a lower bound.
    free_height_m, free_width_m, extent
                         size of the free-space bounding box; "small" when its
                         longer side is at most small_fraction of the grid's
                         shorter side.
    """
    free = grid == FREE
    if not free.any():
        raise ValueError("map has no free cells")
    count, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    sizes = np.bincount(labels[free])
    largest = labels == int(np.argmax(sizes[1:]) + 1)

    blocked_count, blocked = cv2.connectedComponents((~free).astype(np.uint8), connectivity=8)
    edge = np.concatenate([blocked[0], blocked[-1], blocked[:, 0], blocked[:, -1]])
    touching_edge = set(edge[edge > 0].tolist())
    islands = [label for label in range(1, blocked_count) if label not in touching_edge]
    island_cells = np.bincount(blocked.ravel(), minlength=blocked_count)
    largest_island = max((int(island_cells[label]) for label in islands), default=0)

    # distanceTransform measures from each free cell's centre to the nearest
    # blocked cell's centre, so a free cell beside a wall is 1 cell away.
    clearance_m = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5) * resolution
    narrow_share = float(np.mean(clearance_m[free] < narrow_m))

    start = tuple(np.argwhere(largest)[0])
    end, _ = bfs_farthest(largest, start)
    _, diameter = bfs_farthest(largest, end)

    rows, columns = np.nonzero(free)
    height, width = rows.max() - rows.min() + 1, columns.max() - columns.min() + 1
    small = max(height, width) <= small_fraction * min(grid.shape)
    return {"free_area_m2": round(float(free.sum()) * resolution ** 2, 2),
            "free_components": count - 1, "loops": len(islands),
            "largest_island_m2": round(largest_island * resolution ** 2, 2),
            "narrow_share": round(narrow_share, 4),
            "geodesic_diameter_m": round(diameter * resolution, 2),
            "free_height_m": round(height * resolution, 2), "free_width_m": round(width * resolution, 2),
            "extent": "small" if small else "full"}


def main(paths):
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for path in paths:
        writer.writerow(dict(map_name=Path(path).stem, **tags(load(path))))


if __name__ == "__main__":
    main(sys.argv[1:])
