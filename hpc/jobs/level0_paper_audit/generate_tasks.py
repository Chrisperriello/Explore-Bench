#!/usr/bin/env python3
"""Create four paired task tables for the Level-0 paper audit."""

import argparse
import csv
import json
import os
from pathlib import Path


FIELDNAMES = (
    "task_id",
    "task_key",
    "treatment",
    "treatment_role",
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
    "coverage_thresholds",
    "target_coverage",
)

EXPECTED_TREATMENTS = {
    "omnidirectional_3p5": ("omnidirectional", 3.5, "paper_code_control"),
    "four_beam_3p5": ("four_beam", 3.5, "primary_sensor_extension"),
    "omnidirectional_7p0": (
        "omnidirectional",
        7.0,
        "gazebo_range_sensitivity",
    ),
    "four_beam_7p0": ("four_beam", 7.0, "paired_range_sensitivity"),
}


def load_suite(path, repository_root):
    with path.open(encoding="utf-8") as handle:
        suite = json.load(handle)

    if suite.get("methods") != ["cost", "mmpf"]:
        raise ValueError("methods must be exactly cost and mmpf")
    if suite.get("seeds") != list(range(1, 11)):
        raise ValueError("seeds must be exactly 1 through 10")
    if suite.get("team_size") != 2:
        raise ValueError("the audit is fixed to two robots")
    if suite.get("communication_mode") != "legacy":
        raise ValueError("the audit must preserve legacy/oracle navigation")
    if suite.get("coverage_thresholds") != [0.9, 0.98, 0.99]:
        raise ValueError("coverage thresholds must be 0.90, 0.98, and 0.99")

    map_entry = suite.get("map", {})
    if map_entry.get("name") != "room" or not map_entry.get("path"):
        raise ValueError("the audit is fixed to the five-room map")
    if not (repository_root / map_entry["path"]).is_file():
        raise ValueError("map does not exist: {}".format(map_entry["path"]))

    treatments = suite.get("treatments", [])
    actual = {
        item.get("name"): (
            item.get("sensor_type"),
            item.get("sensor_range"),
            item.get("role"),
        )
        for item in treatments
    }
    if actual != EXPECTED_TREATMENTS or len(treatments) != 4:
        raise ValueError("the four frozen audit treatments were changed")

    expected_settings = {
        "resolution": 0.1,
        "communication_protocol": "round_robin",
        "max_steps": 1000,
        "target_coverage": 0.99,
    }
    changed = [
        name
        for name, expected in expected_settings.items()
        if suite.get(name) != expected
    ]
    if changed:
        raise ValueError(
            "frozen suite settings were changed: {}".format(", ".join(changed))
        )
    return suite


def build_rows(suite):
    """Return tables whose equal task IDs identify equal method/seed pairs."""
    tables = {}
    map_entry = suite["map"]
    for treatment in suite["treatments"]:
        rows = []
        task_id = 0
        for method in suite["methods"]:
            for seed in suite["seeds"]:
                rows.append(
                    {
                        "task_id": task_id,
                        "task_key": "{}__{}__room__seed_{:04d}".format(
                            treatment["name"], method, seed
                        ),
                        "treatment": treatment["name"],
                        "treatment_role": treatment["role"],
                        "method": method,
                        "map_name": map_entry["name"],
                        "map_path": map_entry["path"],
                        "team_size": suite["team_size"],
                        "seed": seed,
                        "resolution": suite["resolution"],
                        "sensor_type": treatment["sensor_type"],
                        "sensor_range": treatment["sensor_range"],
                        "communication_mode": suite["communication_mode"],
                        "communication_protocol": suite["communication_protocol"],
                        "max_steps": suite["max_steps"],
                        "coverage_thresholds": ";".join(
                            str(value) for value in suite["coverage_thresholds"]
                        ),
                        "target_coverage": suite["target_coverage"],
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
