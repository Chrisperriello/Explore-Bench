"""Highest team coverage any robot route could reach on a map, with a given sensor.

Usage: python analysis/newmaps/coverage_ceiling.py OUT.csv MAP.pgm [MAP.pgm ...] [--stride 3]

Why: coverage (explore_bench/session.py measurement()) divides known cells by
every non-outside cell, walls included. The legacy 360° scanner never sees
deep inside a thick wall block, so on maps with a solid courtyard core the
99% success target can be out of reach for every method. This senses from
every stride-th free cell and reports the union of everything revealed: an
upper bound, since a robot's 0.9 m footprint cannot stand on every free cell.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "onpolicy"))
from onpolicy.envs.GridEnv.sensors import OmnidirectionalSensor, SensorConfig, UNKNOWN  # noqa: E402


def ceiling(path, sensor_range=3.5, resolution=0.1, stride=3):
    grid = np.asarray(Image.open(path)).copy()
    sensor = OmnidirectionalSensor(SensorConfig("omnidirectional", sensor_range))
    known = np.zeros(grid.shape, bool)
    for x, y in np.argwhere(grid == 254)[::stride]:
        known |= sensor.sense((x, y), 0.0, grid, resolution).full_map != UNKNOWN
    valid = grid != UNKNOWN
    return float((known & valid).sum() / valid.sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("maps", nargs="+", type=Path)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args(argv)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["map", "coverage_ceiling"])
        for path in args.maps:
            value = ceiling(path, stride=args.stride)
            writer.writerow([path.stem, "{:.4f}".format(value)])
            print(path.stem, "{:.4f}".format(value), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
