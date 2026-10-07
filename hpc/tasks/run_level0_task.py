#!/usr/bin/env python3
"""Run one row of a Level-0 task table with one selected sensor."""

import argparse
import csv
import datetime as datetime_module
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


SENSORS = ("omnidirectional", "four_beam")


def utc_now():
    return datetime_module.datetime.now(datetime_module.timezone.utc).isoformat()


def write_json(path, value):
    """Atomically update task status so the monitor never reads partial JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def read_task(path, task_id):
    """Return the row whose explicit ID matches the Slurm array index."""
    with path.open(encoding="utf-8", newline="") as handle:
        matches = [
            row for row in csv.DictReader(handle) if int(row["task_id"]) == task_id
        ]
    if len(matches) != 1:
        raise ValueError(
            "task ID {} matched {} rows in {}".format(task_id, len(matches), path)
        )
    return matches[0]


def complete_runs(task_directory):
    """Find collector runs already marked complete beneath one logical task."""
    complete = []
    for manifest_path in task_directory.glob("*/run*/manifest.json"):
        try:
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, ValueError):
            continue
        if (
            manifest.get("status") == "complete"
            and manifest.get("episodes_planned") == 1
            and manifest.get("episodes_completed") == 1
            and (manifest_path.parent / "episodes.csv").is_file()
        ):
            complete.append(manifest_path.parent)
    return sorted(complete)


def collector_command(repository_root, task, sensor, task_directory):
    """Translate a task row into the existing collector's command line."""
    thresholds = task["coverage_thresholds"].split(";")
    map_path = (repository_root / task["map_path"]).resolve()
    if not map_path.is_file():
        raise ValueError("map does not exist: {}".format(map_path))
    return [
        sys.executable,
        str(repository_root / "grid_simulator" / "collect_experiments.py"),
        "--experiment-name",
        task["task_key"],
        "--method",
        task["method"],
        "--maps",
        str(map_path),
        "--team-sizes",
        task["team_size"],
        "--seeds",
        task["seed"],
        "--resolution",
        task["resolution"],
        "--sensor-types",
        sensor,
        "--sensor-ranges",
        task["sensor_range"],
        "--communication-mode",
        task["communication_mode"],
        "--communication-protocol",
        task["communication_protocol"],
        "--max-steps",
        task["max_steps"],
        "--no-progress-patience",
        task["no_progress_patience"],
        "--coverage-thresholds",
    ] + thresholds + [
        "--target-coverage",
        task["target_coverage"],
        "--record",
        task["record"],
        "--map-snapshots",
        task["map_snapshots"],
        "--output",
        str(task_directory),
    ]


def run_task(args):
    task = read_task(args.task_file, args.task_id)
    task_directory = (
        args.output_root
        / args.variant
        / "task_{:06d}".format(args.task_id)
    ).resolve()
    task_directory.mkdir(parents=True, exist_ok=True)
    status_path = task_directory / "task_status.json"

    existing = complete_runs(task_directory)
    if existing:
        status = {
            "status": "complete",
            "skipped_existing": True,
            "checked_at": utc_now(),
            "task": task,
            "sensor": args.sensor,
            "variant": args.variant,
            "run_directory": str(existing[-1]),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        write_json(status_path, status)
        print("Task {} already has complete data: {}".format(args.task_id, existing[-1]))
        return

    command = collector_command(
        args.repository_root.resolve(), task, args.sensor, task_directory
    )
    status = {
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "task": task,
        "sensor": args.sensor,
        "variant": args.variant,
        "command": command,
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json(status_path, status)

    try:
        subprocess.run(command, cwd=str(args.repository_root), check=True)
        completed = complete_runs(task_directory)
        if not completed:
            raise RuntimeError("collector exited successfully without a complete run")
        status.update(
            {
                "status": "complete",
                "completed_at": utc_now(),
                "run_directory": str(completed[-1]),
            }
        )
        write_json(status_path, status)
        print("Task {} complete: {}".format(args.task_id, completed[-1]))
    except Exception as error:
        status.update(
            {
                "status": "failed",
                "completed_at": utc_now(),
                "error": "{}: {}".format(type(error).__name__, error),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(status_path, status)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--sensor", required=True, choices=SENSORS)
    parser.add_argument("--variant", required=True, choices=("original", "four_beam"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    expected = "omnidirectional" if args.variant == "original" else "four_beam"
    if args.sensor != expected:
        raise ValueError(
            "variant {} requires sensor {}, not {}".format(
                args.variant, expected, args.sensor
            )
        )
    run_task(args)


if __name__ == "__main__":
    main()
