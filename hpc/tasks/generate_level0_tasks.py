#!/usr/bin/env python3
"""Expand one Level-0 suite description into a Slurm task table."""

import argparse
import csv
import json
import os
from pathlib import Path


FIELDNAMES = (
    "task_id",
    "task_key",
    "method",
    "map_name",
    "map_path",
    "team_size",
    "seed",
    "resolution",
    "sensor_range",
    "communication_mode",
    "communication_protocol",
    "max_steps",
    "no_progress_patience",
    "coverage_thresholds",
    "target_coverage",
    "record",
    "map_snapshots",
)
ALLOWED_METHODS = {"cost", "mmpf"}


def load_suite(path, repository_root):
    """Load and validate the small JSON file that defines all paired tasks."""
    with path.open(encoding="utf-8") as handle:
        suite = json.load(handle)

    maps = suite.get("maps")
    methods = suite.get("methods")
    seeds = suite.get("seeds")
    if not isinstance(maps, list) or not maps:
        raise ValueError("maps must be a non-empty list")
    if not isinstance(methods, list) or not methods:
        raise ValueError("methods must be a non-empty list")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list")
    if len(set(methods)) != len(methods) or not set(methods) <= ALLOWED_METHODS:
        raise ValueError("methods must be unique and contain only cost or mmpf")
    if len(set(seeds)) != len(seeds) or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must be unique integers")

    names = set()
    for map_entry in maps:
        if not isinstance(map_entry, dict) or not map_entry.get("name") or not map_entry.get("path"):
            raise ValueError("each map needs a name and path")
        if map_entry["name"] in names:
            raise ValueError("map names must be unique")
        names.add(map_entry["name"])
        map_path = repository_root / map_entry["path"]
        if not map_path.is_file():
            raise ValueError("map does not exist: {}".format(map_entry["path"]))

    required = (
        "team_size",
        "resolution",
        "sensor_range",
        "communication_mode",
        "communication_protocol",
        "max_steps",
        "no_progress_patience",
        "coverage_thresholds",
        "target_coverage",
        "record",
        "map_snapshots",
    )
    missing = [name for name in required if name not in suite]
    if missing:
        raise ValueError("missing suite settings: {}".format(", ".join(missing)))
    if suite["team_size"] != 2:
        raise ValueError("this collection is fixed to two robots")
    if not suite["coverage_thresholds"]:
        raise ValueError("coverage_thresholds cannot be empty")
    return suite


def build_rows(suite):
    """Create one sensor-independent row for every method, map, and seed."""
    rows = []
    task_id = 0
    for method in suite["methods"]:
        for map_entry in suite["maps"]:
            for seed in suite["seeds"]:
                task_key = "{}__{}__seed_{:04d}".format(
                    method, map_entry["name"], seed
                )
                rows.append(
                    {
                        "task_id": task_id,
                        "task_key": task_key,
                        "method": method,
                        "map_name": map_entry["name"],
                        "map_path": map_entry["path"],
                        "team_size": suite["team_size"],
                        "seed": seed,
                        "resolution": suite["resolution"],
                        "sensor_range": suite["sensor_range"],
                        "communication_mode": suite["communication_mode"],
                        "communication_protocol": suite["communication_protocol"],
                        "max_steps": suite["max_steps"],
                        "no_progress_patience": suite["no_progress_patience"],
                        "coverage_thresholds": ";".join(
                            str(value) for value in suite["coverage_thresholds"]
                        ),
                        "target_coverage": suite["target_coverage"],
                        "record": suite["record"],
                        "map_snapshots": suite["map_snapshots"],
                    }
                )
                task_id += 1
    return rows


def write_rows(path, rows):
    """Write the complete task table atomically without replacing old work."""
    if path.exists():
        raise ValueError("refusing to replace existing task table: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    suite = load_suite(args.config.resolve(), args.repository_root.resolve())
    rows = build_rows(suite)
    write_rows(args.output.resolve(), rows)
    print(len(rows))


if __name__ == "__main__":
    main()
