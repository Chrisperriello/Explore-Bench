#!/usr/bin/env python3
"""Combine one completed Level-0 collection without aggregating its metrics."""

import argparse
import csv
import datetime as datetime_module
import hashlib
import json
import os
import shutil
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_SLURM_ROOT = REPOSITORY_ROOT.parent / "slurm"
VARIANTS = ("original", "four_beam")
TABLES = ("episodes.csv", "steps.csv", "agents.csv")
PREFIX_FIELDS = (
    "collection_id",
    "variant",
    "sensor_variant",
    "task_id",
    "task_key",
    "scenario",
    "source_attempt",
    "source_run_directory",
)


def utc_now():
    return datetime_module.datetime.now(datetime_module.timezone.utc).isoformat()


def read_json(path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tasks(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return {int(row["task_id"]): row for row in csv.DictReader(handle)}


def attempt_number(run_directory):
    name = run_directory.name
    if name.startswith("run") and name[3:].isdigit():
        return int(name[3:])
    return -1


def inspect_attempts(task_directory):
    """Return all attempts and the newest complete, structurally valid attempt."""
    attempts = []
    if task_directory.is_dir():
        for manifest_path in task_directory.glob("*/run*/manifest.json"):
            run_directory = manifest_path.parent
            manifest = read_json(manifest_path)
            files_present = all((run_directory / name).is_file() for name in TABLES)
            complete = (
                manifest.get("status") == "complete"
                and manifest.get("episodes_planned") == 1
                and manifest.get("episodes_completed") == 1
                and files_present
            )
            attempts.append(
                {
                    "number": attempt_number(run_directory),
                    "run_directory": run_directory,
                    "manifest": manifest,
                    "complete": complete,
                    "files_present": files_present,
                }
            )
    attempts.sort(key=lambda value: value["number"])
    complete_attempts = [attempt for attempt in attempts if attempt["complete"]]
    selected = complete_attempts[-1] if complete_attempts else None
    return attempts, selected


def inspect_collection(slurm_root, collection):
    """Match every expected sensor/task pair with its newest complete attempt."""
    task_file = slurm_root / "tasks" / "level0" / collection / "tasks.csv"
    if not task_file.is_file():
        raise ValueError("task table not found: {}".format(task_file))
    tasks = load_tasks(task_file)
    data_root = slurm_root / "data" / "level0" / collection
    entries = []
    for variant in VARIANTS:
        for task_id in sorted(tasks):
            task = tasks[task_id]
            task_directory = data_root / variant / "task_{:06d}".format(task_id)
            attempts, selected = inspect_attempts(task_directory)
            entries.append(
                {
                    "variant": variant,
                    "sensor_variant": (
                        "omnidirectional" if variant == "original" else "four_beam"
                    ),
                    "task_id": task_id,
                    "task": task,
                    "attempts": attempts,
                    "selected": selected,
                }
            )
    return task_file, tasks, entries


def read_single_episode(run_directory):
    with (run_directory / "episodes.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(
            "expected one episode row in {}, found {}".format(run_directory, len(rows))
        )
    return rows[0]


def source_prefix(collection, entry):
    selected = entry["selected"]
    task = entry["task"]
    return {
        "collection_id": collection,
        "variant": entry["variant"],
        "sensor_variant": entry["sensor_variant"],
        "task_id": entry["task_id"],
        "task_key": task.get("task_key", ""),
        "scenario": task.get("map_name", ""),
        "source_attempt": selected["number"],
        "source_run_directory": str(selected["run_directory"]),
    }


def merge_table(collection, selected_entries, table_name, output_path):
    """Stream compatible source CSVs into one table with source identifiers."""
    writer = None
    source_fields = None
    row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as output_handle:
        for entry in selected_entries:
            source_path = entry["selected"]["run_directory"] / table_name
            with source_path.open(encoding="utf-8", newline="") as source_handle:
                reader = csv.DictReader(source_handle)
                if reader.fieldnames is None:
                    raise ValueError("missing CSV header: {}".format(source_path))
                if source_fields is None:
                    source_fields = list(reader.fieldnames)
                    duplicates = set(PREFIX_FIELDS) & set(source_fields)
                    if duplicates:
                        raise ValueError(
                            "source columns conflict with merge columns: {}".format(
                                ", ".join(sorted(duplicates))
                            )
                        )
                    writer = csv.DictWriter(
                        output_handle, fieldnames=list(PREFIX_FIELDS) + source_fields
                    )
                    writer.writeheader()
                elif list(reader.fieldnames) != source_fields:
                    raise ValueError("CSV schema differs: {}".format(source_path))

                prefix = source_prefix(collection, entry)
                for row in reader:
                    combined = dict(prefix)
                    combined.update(row)
                    writer.writerow(combined)
                    row_count += 1
    return row_count


def write_inventory(collection, entries, output_path):
    fields = (
        "collection_id",
        "variant",
        "sensor_variant",
        "task_id",
        "task_key",
        "method",
        "scenario",
        "seed",
        "merge_status",
        "attempts_found",
        "selected_attempt",
        "source_run_directory",
        "success",
        "termination_reason",
        "steps_executed",
        "final_coverage_ratio",
        "coverage_90_reached",
        "coverage_90_step",
        "coverage_99_reached",
        "coverage_99_step",
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            task = entry["task"]
            selected = entry["selected"]
            episode = (
                read_single_episode(selected["run_directory"]) if selected else {}
            )
            writer.writerow(
                {
                    "collection_id": collection,
                    "variant": entry["variant"],
                    "sensor_variant": entry["sensor_variant"],
                    "task_id": entry["task_id"],
                    "task_key": task.get("task_key", ""),
                    "method": task.get("method", ""),
                    "scenario": task.get("map_name", ""),
                    "seed": task.get("seed", ""),
                    "merge_status": "selected" if selected else "missing",
                    "attempts_found": len(entry["attempts"]),
                    "selected_attempt": selected["number"] if selected else "",
                    "source_run_directory": (
                        str(selected["run_directory"]) if selected else ""
                    ),
                    "success": episode.get("success", ""),
                    "termination_reason": episode.get("termination_reason", ""),
                    "steps_executed": episode.get("steps_executed", ""),
                    "final_coverage_ratio": episode.get(
                        "final_coverage_ratio", ""
                    ),
                    "coverage_90_reached": episode.get(
                        "coverage_90_reached", ""
                    ),
                    "coverage_90_step": episode.get("coverage_90_step", ""),
                    "coverage_99_reached": episode.get(
                        "coverage_99_reached", ""
                    ),
                    "coverage_99_step": episode.get("coverage_99_step", ""),
                }
            )


def merge_collection(slurm_root, collection, output, allow_incomplete=False):
    task_file, tasks, entries = inspect_collection(slurm_root, collection)
    missing = [
        "{}:{}".format(entry["variant"], entry["task_id"])
        for entry in entries
        if entry["selected"] is None
    ]
    if missing and not allow_incomplete:
        raise ValueError(
            "{} of {} expected runs have no complete attempt: {}".format(
                len(missing), len(entries), ", ".join(missing)
            )
        )
    if output.exists():
        raise ValueError("refusing to replace existing output: {}".format(output))

    selected_entries = [entry for entry in entries if entry["selected"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / (output.name + ".tmp.{}".format(os.getpid()))
    if temporary.exists():
        raise ValueError("temporary output already exists: {}".format(temporary))
    temporary.mkdir()
    try:
        row_counts = {}
        for table_name in TABLES:
            row_counts[table_name] = merge_table(
                collection,
                selected_entries,
                table_name,
                temporary / table_name,
            )
        write_inventory(collection, entries, temporary / "run_inventory.csv")
        manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "collection_id": collection,
            "task_table": str(task_file),
            "task_table_sha256": file_sha256(task_file),
            "tasks_per_sensor": len(tasks),
            "expected_runs": len(entries),
            "selected_complete_runs": len(selected_entries),
            "missing_runs": missing,
            "allow_incomplete": allow_incomplete,
            "row_counts": row_counts,
            "output_directory": str(output),
        }
        with (temporary / "merge_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(str(temporary), str(output))
    except Exception:
        shutil.rmtree(str(temporary), ignore_errors=True)
        raise
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_SLURM_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="merge available complete runs and list missing runs in the inventory",
    )
    return parser


def main():
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else root / "combined" / "level0" / args.collection
    )
    try:
        manifest = merge_collection(
            root, args.collection, output, allow_incomplete=args.allow_incomplete
        )
    except ValueError as error:
        raise SystemExit(str(error))
    print("Combined {} complete runs".format(manifest["selected_complete_runs"]))
    print("Episodes: {} rows".format(manifest["row_counts"]["episodes.csv"]))
    print("Steps: {} rows".format(manifest["row_counts"]["steps.csv"]))
    print("Agents: {} rows".format(manifest["row_counts"]["agents.csv"]))
    print("Output: {}".format(manifest["output_directory"]))


if __name__ == "__main__":
    main()
