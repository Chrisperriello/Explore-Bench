#!/usr/bin/env python3
"""Collect one finite Level-0 paper-audit episode with independent metrics."""

import argparse
import contextlib
import csv
import datetime as datetime_module
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np


UNKNOWN = 205
THRESHOLD_LABELS = ((0.90, "90"), (0.98, "98"), (0.99, "99"))


def utc_now():
    return datetime_module.datetime.now(datetime_module.timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata(repository_root):
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repository_root), text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(repository_root),
        text=True,
    ).strip()
    return {"commit": commit, "dirty": bool(status), "status_porcelain": status}


def read_task(path, task_id):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle) if int(row["task_id"]) == task_id
        ]
    if len(rows) != 1:
        raise ValueError("task ID {} matched {} rows".format(task_id, len(rows)))
    return rows[0]


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def measurement(env, path_lengths_m):
    """Independently calculate paper, released-code, and collector conventions."""
    valid = np.asarray(env.gt_map) != UNKNOWN
    known = np.stack(
        [((np.asarray(grid) != UNKNOWN) & valid) for grid in env.built_map], axis=0
    )
    agent_cells = known.reshape(env.num_agents, -1).sum(axis=1).astype(np.int64)
    team_cells = int(np.any(known, axis=0).sum())
    total_cells = int(valid.sum())
    agent_ratios = agent_cells.astype(np.float64) / float(total_cells)
    overlap_cells = int(agent_cells.sum() - team_cells)
    cell_area = float(env.resolution) ** 2
    # Equation (3) uses the total terrain area in both the subtraction and
    # denominator. Keep it separate from the collector's geometric overlap,
    # whose numerator subtracts the currently sensed union.
    paper_overlap = float(agent_cells.sum() - total_cells) / float(total_cells)

    return {
        "team_coverage_cells": team_cells,
        "team_coverage_ratio": team_cells / float(total_cells),
        "total_explorable_cells": total_cells,
        "agent_coverage_cells": [int(value) for value in agent_cells],
        "agent_coverage_ratios": [float(value) for value in agent_ratios],
        "overlap_cells": overlap_cells,
        "paper_sigma_area_m2": float(np.std(agent_cells * cell_area)),
        "paper_overlap_ratio_total_terrain": paper_overlap,
        "released_sigma_ratio": float(np.std(agent_ratios)),
        "released_overlap_ratio": float(agent_ratios.sum() - 1.0),
        "collector_sigma_area_m2": float(np.std(agent_cells * cell_area)),
        "collector_sigma_ratio": float(np.std(agent_ratios)),
        "collector_overlap_ratio_total": overlap_cells / float(total_cells),
        "team_path_length_m": float(np.sum(path_lengths_m)),
        "maximum_agent_path_length_m": float(np.max(path_lengths_m)),
        "agent_path_lengths_m": [float(value) for value in path_lengths_m],
        "historical_path_log_counts": [len(path) for path in env.path_log],
        "positions": np.asarray(env.agent_pos, dtype=np.int64).tolist(),
        "headings_rad": np.asarray(env.agent_yaw, dtype=np.float64).tolist(),
    }


def flattened_metrics(values):
    names = (
        "team_coverage_cells",
        "team_coverage_ratio",
        "total_explorable_cells",
        "overlap_cells",
        "paper_sigma_area_m2",
        "paper_overlap_ratio_total_terrain",
        "released_sigma_ratio",
        "released_overlap_ratio",
        "collector_sigma_area_m2",
        "collector_sigma_ratio",
        "collector_overlap_ratio_total",
        "team_path_length_m",
        "maximum_agent_path_length_m",
    )
    result = {name: values[name] for name in names}
    result["agent_coverage_cells"] = json.dumps(values["agent_coverage_cells"])
    result["agent_coverage_ratios"] = json.dumps(values["agent_coverage_ratios"])
    result["agent_path_lengths_m"] = json.dumps(values["agent_path_lengths_m"])
    result["historical_path_log_counts"] = json.dumps(
        values["historical_path_log_counts"]
    )
    result["positions"] = json.dumps(values["positions"])
    result["headings_rad"] = json.dumps(values["headings_rad"])
    return result


def all_agents_stuck(env):
    if env.last_goals is None or env.positions_before_goal is None:
        return False
    return bool(np.all(env.last_goals == env.positions_before_goal))


def common_row(task):
    return {
        "schema_version": 1,
        "task_id": int(task["task_id"]),
        "task_key": task["task_key"],
        "treatment": task["treatment"],
        "treatment_role": task["treatment_role"],
        "method": task["method"],
        "map_name": task["map_name"],
        "team_size": int(task["team_size"]),
        "seed": int(task["seed"]),
        "resolution": float(task["resolution"]),
        "sensor_type": task["sensor_type"],
        "sensor_range": float(task["sensor_range"]),
        "communication_mode": task["communication_mode"],
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def collect(task, repository_root, run_directory):
    sys.path.insert(0, str(repository_root / "grid_simulator"))
    from collect_experiments import HeadlessGridEnv  # noqa: E402
    from sensors import sensor_configs_from_values  # noqa: E402

    map_path = (repository_root / task["map_path"]).resolve()
    seed = int(task["seed"])
    team_size = int(task["team_size"])
    sensor_range = float(task["sensor_range"])
    resolution = float(task["resolution"])
    max_steps = int(task["max_steps"])
    target_coverage = float(task["target_coverage"])

    # The historical planner uses both an environment-owned RNG and Python's
    # module RNG. Seed both so four-beam headings are exactly repeatable.
    random.seed(seed)
    np.random.seed(seed)
    sensor_configs = sensor_configs_from_values(
        [task["sensor_type"]], [sensor_range], team_size, sensor_range
    )
    env = HeadlessGridEnv(
        resolution,
        sensor_range,
        team_size,
        max_steps,
        str(map_path),
        visualization=False,
        sensor_configs=sensor_configs,
        communication_config=None,
        communication_protocol=task["communication_protocol"],
        seed=seed,
    )
    env.seed(seed)
    random.seed(seed)
    env.reset_for_traditional()

    start_positions = np.asarray(env.agent_pos, dtype=np.int64).tolist()
    start_headings = np.asarray(env.agent_yaw, dtype=np.float64).tolist()
    previous_positions = np.asarray(env.agent_pos, dtype=np.float64)
    path_lengths_m = np.zeros(team_size, dtype=np.float64)
    crossings = {}
    released_crossings = {}
    step_rows = []
    agent_rows = []
    started_wall = time.perf_counter()
    started_cpu = time.process_time()

    def capture(step, termination_reason=""):
        values = measurement(env, path_lengths_m)
        wall_seconds = time.perf_counter() - started_wall if step else 0.0
        cpu_seconds = time.process_time() - started_cpu if step else 0.0
        row = common_row(task)
        row.update(
            {
                "step": step,
                "wall_seconds": wall_seconds,
                "cpu_seconds": cpu_seconds,
                "termination_reason": termination_reason,
            }
        )
        row.update(flattened_metrics(values))
        step_rows.append(row)
        goals = env.last_goals
        for agent_id in range(team_size):
            goal = None if goals is None else goals[agent_id]
            agent_rows.append(
                dict(
                    common_row(task),
                    step=step,
                    agent_id=agent_id,
                    row=values["positions"][agent_id][0],
                    column=values["positions"][agent_id][1],
                    heading_rad=values["headings_rad"][agent_id],
                    goal_row="" if goal is None else int(goal[0]),
                    goal_column="" if goal is None else int(goal[1]),
                    coverage_cells=values["agent_coverage_cells"][agent_id],
                    coverage_ratio=values["agent_coverage_ratios"][agent_id],
                    path_length_m=values["agent_path_lengths_m"][agent_id],
                    historical_path_log_count=values[
                        "historical_path_log_counts"
                    ][agent_id],
                )
            )
        for threshold, label in THRESHOLD_LABELS:
            if label not in crossings and values["team_coverage_ratio"] >= threshold:
                crossings[label] = {
                    "step": step,
                    "wall_seconds": wall_seconds,
                    "cpu_seconds": cpu_seconds,
                    "metrics": dict(values),
                }
        for threshold, label in THRESHOLD_LABELS[:2]:
            if (
                label not in released_crossings
                and values["team_coverage_ratio"] > threshold
            ):
                released_crossings[label] = {
                    "step": step,
                    "wall_seconds": wall_seconds,
                    "cpu_seconds": cpu_seconds,
                    "metrics": dict(values),
                }
        return values

    final = capture(0)
    termination_reason = ""
    steps_executed = 0
    if final["team_coverage_ratio"] >= target_coverage:
        termination_reason = "target_coverage"

    while not termination_reason and steps_executed < max_steps:
        with contextlib.redirect_stdout(io.StringIO()):
            if task["method"] == "cost":
                env.step_for_cost()
            else:
                env.step_for_mmpf()
        current_positions = np.asarray(env.agent_pos, dtype=np.float64)
        path_lengths_m += (
            np.linalg.norm(current_positions - previous_positions, axis=1) * resolution
        )
        previous_positions = current_positions.copy()
        steps_executed += 1
        preview = measurement(env, path_lengths_m)
        if preview["team_coverage_ratio"] >= target_coverage:
            termination_reason = "target_coverage"
        elif all_agents_stuck(env):
            termination_reason = "all_agents_stuck"
        elif steps_executed >= max_steps:
            termination_reason = "max_steps"
        final = capture(steps_executed, termination_reason)

    episode = common_row(task)
    episode.update(
        {
            "steps_executed": steps_executed,
            "success": final["team_coverage_ratio"] >= target_coverage,
            "termination_reason": termination_reason,
            "target_coverage": target_coverage,
            "start_positions": json.dumps(start_positions, separators=(",", ":")),
            "start_headings_rad": json.dumps(
                start_headings, separators=(",", ":")
            ),
            "final_positions": json.dumps(
                final["positions"], separators=(",", ":")
            ),
            "final_headings_rad": json.dumps(
                final["headings_rad"], separators=(",", ":")
            ),
        }
    )
    episode.update({"final_" + key: value for key, value in flattened_metrics(final).items()})
    for _, label in THRESHOLD_LABELS:
        crossing = crossings.get(label)
        episode["coverage_{}_reached".format(label)] = crossing is not None
        for name in ("step", "wall_seconds", "cpu_seconds"):
            episode["coverage_{}_{}".format(label, name)] = (
                "" if crossing is None else crossing[name]
            )
        metric_names = (
            "team_path_length_m",
            "maximum_agent_path_length_m",
            "paper_sigma_area_m2",
            "paper_overlap_ratio_total_terrain",
            "released_sigma_ratio",
            "released_overlap_ratio",
            "collector_sigma_area_m2",
            "collector_sigma_ratio",
            "collector_overlap_ratio_total",
            "historical_path_log_counts",
        )
        for name in metric_names:
            value = "" if crossing is None else crossing["metrics"][name]
            if isinstance(value, (list, tuple, np.ndarray)):
                value = json.dumps(json_value(value))
            episode["coverage_{}_{}".format(label, name)] = value

    for _, label in THRESHOLD_LABELS[:2]:
        crossing = released_crossings.get(label)
        episode["released_coverage_{}_reached".format(label)] = crossing is not None
        for name in ("step", "wall_seconds", "cpu_seconds"):
            episode["released_coverage_{}_{}".format(label, name)] = (
                "" if crossing is None else crossing[name]
            )
        for name in (
            "released_sigma_ratio",
            "released_overlap_ratio",
            "historical_path_log_counts",
        ):
            value = "" if crossing is None else crossing["metrics"][name]
            if isinstance(value, (list, tuple, np.ndarray)):
                value = json.dumps(json_value(value))
            episode["released_coverage_{}_{}".format(label, name)] = value

    write_csv(run_directory / "episodes.csv", [episode], list(episode.keys()))
    write_csv(run_directory / "steps.csv", step_rows, list(step_rows[0].keys()))
    write_csv(run_directory / "agents.csv", agent_rows, list(agent_rows[0].keys()))
    return episode, map_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--run-directory", required=True, type=Path)
    args = parser.parse_args(argv)

    repository_root = args.repository_root.resolve()
    run_directory = args.run_directory.resolve()
    if run_directory.exists():
        raise ValueError("refusing to replace run directory: {}".format(run_directory))
    run_directory.mkdir(parents=True)
    task = read_task(args.task_file.resolve(), args.task_id)
    source_path = Path(__file__).resolve()
    repository = git_metadata(repository_root)
    manifest_path = run_directory / "manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "episodes_planned": 1,
        "episodes_completed": 0,
        "task": task,
        "task_table_sha256": file_sha256(args.task_file.resolve()),
        "repository_commit": repository["commit"],
        "repository_dirty": repository["dirty"],
        "repository_status_porcelain": repository["status_porcelain"],
        "collector_sha256": file_sha256(source_path),
        "python_version": sys.version,
        "numpy_version": np.__version__,
    }
    write_json(manifest_path, manifest)
    try:
        episode, map_path = collect(task, repository_root, run_directory)
        manifest.update(
            {
                "status": "complete",
                "completed_at": utc_now(),
                "episodes_completed": 1,
                "map_sha256": file_sha256(map_path),
                "success": bool(episode["success"]),
                "termination_reason": episode["termination_reason"],
            }
        )
        write_json(manifest_path, manifest)
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "completed_at": utc_now(),
                "error": "{}: {}".format(type(error).__name__, error),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
