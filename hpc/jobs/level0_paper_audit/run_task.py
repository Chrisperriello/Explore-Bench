#!/usr/bin/env python3
"""Run one immutable task from a Level-0 paper-audit table."""

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
        rows = [
            row for row in csv.DictReader(handle) if int(row["task_id"]) == task_id
        ]
    if len(rows) != 1:
        raise ValueError("task ID {} matched {} rows".format(task_id, len(rows)))
    return rows[0]


def complete_runs(task_directory):
    complete = []
    for manifest_path in task_directory.glob("run*/manifest.json"):
        try:
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, ValueError):
            continue
        if (
            manifest.get("status") == "complete"
            and manifest.get("episodes_completed") == 1
            and (manifest_path.parent / "episodes.csv").is_file()
            and (manifest_path.parent / "steps.csv").is_file()
            and (manifest_path.parent / "agents.csv").is_file()
        ):
            complete.append(manifest_path.parent)
    return sorted(
        complete,
        key=lambda path: int(path.name[3:]) if path.name[3:].isdigit() else -1,
    )


def next_run_directory(task_directory):
    numbers = []
    for child in task_directory.iterdir():
        if child.is_dir() and child.name.startswith("run"):
            suffix = child.name[3:]
            if suffix.isdigit():
                numbers.append(int(suffix))
    return task_directory / "run{}".format(max(numbers, default=0) + 1)


def run_task(args):
    task = read_task(args.task_file.resolve(), args.task_id)
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
                "run_directory": str(existing[-1]),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
        )
        print("Task {} already complete: {}".format(args.task_id, existing[-1]))
        return

    run_directory = next_run_directory(task_directory)
    command = [
        sys.executable,
        str(args.repository_root / "hpc/jobs/level0_paper_audit/collect_episode.py"),
        "--task-file",
        str(args.task_file.resolve()),
        "--task-id",
        str(args.task_id),
        "--repository-root",
        str(args.repository_root.resolve()),
        "--run-directory",
        str(run_directory),
    ]
    status = {
        "status": "running",
        "started_at": utc_now(),
        "task": task,
        "command": command,
        "run_directory": str(run_directory),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json(status_path, status)
    try:
        subprocess.run(command, cwd=str(args.repository_root), check=True)
        completed = complete_runs(task_directory)
        if run_directory not in completed:
            raise RuntimeError("collector exited without a complete audit run")
        status.update({"status": "complete", "completed_at": utc_now()})
        write_json(status_path, status)
        print("Task {} complete: {}".format(args.task_id, run_directory))
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
