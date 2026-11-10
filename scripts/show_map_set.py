"""Show every map in a frozen map set with its split and shape tags.

Usage: .venv/bin/python scripts/show_map_set.py maps/rooms_v1 [--save PNG] [--loops]

Opens a window (or saves a PNG with --save). Each panel's title gives the
split, the largest island and loop count (explore_bench/map_tags.py), free
area and the source's room count. --loops draws each island in red: a wall
piece with free floor all round it, which robots can drive around.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

WALL, FREE = 0, 254


def islands(grid):
    count, labels = cv2.connectedComponents((grid != FREE).astype(np.uint8), connectivity=8)
    edge = set(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]).tolist())
    return np.isin(labels, [label for label in range(1, count) if label not in edge])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--save", type=Path, help="write a PNG instead of opening a window")
    parser.add_argument("--loops", action="store_true", help="draw loop islands in red")
    args = parser.parse_args(argv)
    manifest = json.loads((args.directory / "manifest.json").read_text())
    missing = [item["file"] for item in manifest["maps"] if not (args.directory / item["file"]).exists()]
    if missing:
        sys.exit("{} maps missing; for HouseExpo run scripts/freeze_map_set.py --rebuild {}"
                 .format(len(missing), args.directory))

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    columns = min(6, len(manifest["maps"]))
    rows = -(-len(manifest["maps"]) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.5 * rows), squeeze=False)
    colors = ListedColormap(["#26313d", "#e1e0d9", "#ffffff", "#d03b2f"])
    for ax in axes.flat:
        ax.axis("off")
    for ax, item in zip(axes.flat, manifest["maps"]):
        grid = np.asarray(Image.open(args.directory / item["file"]))
        shown = np.select([grid == WALL, grid == FREE], [0, 2], 1)
        if args.loops:
            shown[islands(grid)] = 3
        ax.imshow(shown, cmap=colors, vmin=0, vmax=3, interpolation="nearest")
        tags = item["tags"]
        label = item.get("seed", item.get("source_id", ""))
        ax.set_title("{} | {}\nisland {:.0f} m², {} loops\n{:.0f} m² free, {} rooms".format(
            item["split"], str(label)[:12], tags["largest_island_m2"], tags["loops"],
            tags["free_area_m2"], item.get("rooms", "?")), loc="left", fontsize=8)
    figure.suptitle("{}: {} maps from {}, {} candidates rejected".format(
        manifest["name"], len(manifest["maps"]), manifest["source"], len(manifest["rejected"])),
        x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    if args.save:
        figure.savefig(args.save, dpi=80)
        print(args.save)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
