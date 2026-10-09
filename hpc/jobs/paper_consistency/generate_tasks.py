#!/usr/bin/env python3
"""Create one matched ten-row task table for each diagnostic treatment."""

import argparse
import csv
import json
import os
from pathlib import Path


FIELDNAMES = (
    "task_id",
    "task_key",
    "treatment",
    "method",
    "map_name",
    "map_path",
    "team_size",
    "seed",
    "resolution",
    "sensor_type",
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

EXPECTED_TREATMENTS = {
    "legacy_3p5": ("legacy", 3.5),
    "legacy_7p0": ("legacy", 7.0),
    "perfect_3p5": ("perfect", 3.5),
}


def load_suite(path, repository_root):
    with path.open(encoding="utf-8") as handle:
        suite = json.load(handle)

    if suite.get("methods") != ["cost", "mmpf"]:
        raise ValueError("methods must be exactly cost and mmpf")
    if suite.get("seeds") != [1, 2, 3, 4, 5]:
        raise ValueError("seeds must be exactly 1 through 5")
    if suite.get("sensor_type") != "omnidirectional":
        raise ValueError("this diagnostic uses only the omnidirectional sensor")
    if suite.get("team_size") != 2:
        raise ValueError("this diagnostic is fixed to two robots")

    map_entry = suite.get("map", {})
    if map_entry.get("name") != "room" or not map_entry.get("path"):
        raise ValueError("this diagnostic is fixed to the five-room map")
    if not (repository_root / map_entry["path"]).is_file():
        raise ValueError("map does not exist: {}".format(map_entry["path"]))

    treatments = suite.get("treatments", [])
    actual = {
        item.get("name"): (item.get("communication_mode"), item.get("sensor_range"))
        for item in treatments
    }
    if actual != EXPECTED_TREATMENTS or len(treatments) != 3:
        raise ValueError("the three diagnostic treatments were changed")

    required = (
        "resolution",
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
    return suite


def build_rows(suite):
    """Return tables whose equal task IDs are the same method and seed."""
    tables = {}
    map_entry = suite["map"]
    for treatment in suite["treatments"]:
        rows = []
        task_id = 0
        for method in suite["methods"]:
            for seed in suite["seeds"]:
                task_key = "{}__{}__{}__seed_{:04d}".format(
                    treatment["name"], method, map_entry["name"], seed
                )
                rows.append(
                    {
                        "task_id": task_id,
                        "task_key": task_key,
                        "treatment": treatment["name"],
                        "method": method,
                        "map_name": map_entry["name"],
                        "map_path": map_entry["path"],
                        "team_size": suite["team_size"],
                        "seed": seed,
                        "resolution": suite["resolution"],
                        "sensor_type": suite["sensor_type"],
                        "sensor_range": treatment["sensor_range"],
                        "communication_mode": treatment["communication_mode"],
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
        tables[treatment["name"]] = rows
    return tables


def write_table(path, rows):
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    args = parser.parse_args(argv)

    suite = load_suite(args.config.resolve(), args.repository_root.resolve())
    tables = build_rows(suite)
    for treatment, rows in tables.items():
        write_table(args.output_directory.resolve() / (treatment + ".csv"), rows)
    print(sum(len(rows) for rows in tables.values()))


if __name__ == "__main__":
    main()
