#!/usr/bin/env python3
"""Run one row from a paper-consistency diagnostic task table."""

import argparse
import csv
import datetime as datetime_module
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


def utc_now():
    return datetime_module.datetime.now(datetime_module.timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def read_task(path, task_id):
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


def collector_command(repository_root, task, task_directory):
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
        str(task["team_size"]),
        "--seeds",
        str(task["seed"]),
        "--resolution",
        str(task["resolution"]),
        "--sensor-types",
        task["sensor_type"],
        "--sensor-ranges",
        str(task["sensor_range"]),
        "--communication-mode",
        task["communication_mode"],
        "--communication-protocol",
        task["communication_protocol"],
        "--max-steps",
        str(task["max_steps"]),
        "--no-progress-patience",
        str(task["no_progress_patience"]),
        "--coverage-thresholds",
    ] + str(task["coverage_thresholds"]).split(";") + [
        "--target-coverage",
        str(task["target_coverage"]),
        "--record",
        task["record"],
        "--map-snapshots",
        task["map_snapshots"],
        "--output",
        str(task_directory),
    ]


def run_task(args):
    task = read_task(args.task_file, args.task_id)
    if task["treatment"] != args.treatment:
        raise ValueError(
            "task treatment {} does not match {}".format(
                task["treatment"], args.treatment
            )
        )

    task_directory = (
        args.output_root / args.treatment / "task_{:06d}".format(args.task_id)
    ).resolve()
    task_directory.mkdir(parents=True, exist_ok=True)
    status_path = task_directory / "task_status.json"

    existing = complete_runs(task_directory)
    if existing:
        write_json(
            status_path,
            {
                "status": "complete",
                "skipped_existing": True,
                "checked_at": utc_now(),
                "task": task,
                "treatment": args.treatment,
                "run_directory": str(existing[-1]),
                "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
        )
        print("Task {} already has complete data: {}".format(args.task_id, existing[-1]))
        return

    command = collector_command(args.repository_root.resolve(), task, task_directory)
    status = {
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "task": task,
        "treatment": args.treatment,
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    run_task(parser.parse_args(argv))


if __name__ == "__main__":
    main()
