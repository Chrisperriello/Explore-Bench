#!/usr/bin/env python3
"""Headless, reproducible data collection for standalone Level-0 planners.

This module deliberately sits outside ``GridEnv``.  It observes environment
state after each decision, but it never writes observations or ground truth
back into the planner.  The separation keeps evaluation privileged information
from becoming controller input.

The existing interactive entry point remains unchanged.  This script imports
the same environment, sensors, communication broker, cost planner, MMPF
planner, and A* implementation, then adds finite episodes and structured files.
"""

import argparse
import contextlib
import csv
import datetime as datetime_module
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from GridEnv import COMMUNICATION_PROTOCOLS, GridEnv  # noqa: E402
from communication import (  # noqa: E402
    COMMUNICATION_MODES,
    MAP_PATCH,
    SILENCE,
    UNKNOWN,
    CommunicationConfig,
)
from sensors import sensor_configs_from_values  # noqa: E402


SCHEMA_VERSION = 1
LEGACY = "legacy"
PLANNERS = ("cost", "mmpf")
RECORD_LEVELS = ("episode", "steps", "full")
SNAPSHOT_LEVELS = ("none", "milestones", "all")
COMMUNICATION_METRICS = (
    "attempted_messages",
    "transmitted_messages",
    "delivered_messages",
    "collision_messages",
    "lost_messages",
    "expired_messages",
    "in_range_recipients",
    "attempted_bits",
    "delivered_bits",
    "pose_messages",
    "map_patch_messages",
    "latency_total",
)
BEAM_NAMES = ("front", "back", "left", "right")


class HeadlessGridEnv(GridEnv):
    """Observe the standalone planner without changing its decision logic.

    The adapter suppresses optional plotting for batch runs and remembers the
    selected goals and requested fixed-protocol communication actions for CSV
    output.  Sensing, planning, movement, belief updates, and broker behavior
    remain inherited from :class:`GridEnv`.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the existing environment plus evaluator-only trace state."""
        super(HeadlessGridEnv, self).__init__(*args, **kwargs)
        self.last_goals = None
        self.positions_before_goal = None
        self.last_requested_communication_actions = np.full(
            self.num_agents, SILENCE, dtype=np.int64
        )

    def plot_map_with_path(self):
        """Skip the legacy plotting hook unless visualization was requested."""
        # The original traditional step calls this unconditionally.  Preserve
        # its behavior only when the user explicitly requests visualization.
        if self.visualization:
            return super(HeadlessGridEnv, self).plot_map_with_path()
        return None

    def get_goal_for_cost(self):
        """Run cost goal selection while recording its input positions/output."""
        self.positions_before_goal = np.asarray(self.agent_pos, dtype=np.int64).copy()
        goals = super(HeadlessGridEnv, self).get_goal_for_cost()
        self.last_goals = np.asarray(goals, dtype=np.int64).copy()
        return goals

    def get_goal_for_mmpf(self):
        """Run MMPF goal selection while recording positions and chosen goals."""
        self.positions_before_goal = np.asarray(self.agent_pos, dtype=np.int64).copy()
        goals = super(HeadlessGridEnv, self).get_goal_for_mmpf()
        self.last_goals = np.asarray(goals, dtype=np.int64).copy()
        return goals

    def _transmit_traditional_communication(self):
        """Record the requested fixed-protocol action, then call the real broker.

        This reconstructs the transparent round-robin/always request only for
        reporting.  The superclass still owns resource validation, cursor
        movement, collision handling, and actual transmission.
        """
        # Observe the deterministic protocol request without replacing the
        # broker call that actually controls the experiment.
        actions = np.full(self.num_agents, SILENCE, dtype=np.int64)
        if self.communication_broker is not None:
            if self.communication_protocol == "always":
                actions[:] = MAP_PATCH
            elif self.communication_protocol == "round_robin":
                for offset in range(self.num_agents):
                    sender = (self._round_robin_cursor + offset) % self.num_agents
                    available = self.communication_broker.available_actions(sender)
                    if np.any(available[MAP_PATCH:] > 0):
                        actions[sender] = MAP_PATCH
                        break
        self.last_requested_communication_actions = actions
        return super(HeadlessGridEnv, self)._transmit_traditional_communication()


def _utc_now():
    """Return an ISO-8601 UTC timestamp for the run manifest."""
    return datetime_module.datetime.now(datetime_module.timezone.utc).isoformat()


def _json_safe(value):
    """Recursively convert paths/NumPy values/non-finite floats for strict JSON."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    return value


def _write_json(path, payload):
    """Atomically replace a JSON file so interrupted updates remain readable."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _file_sha256(path):
    """Return a streaming SHA-256 digest used to identify an input map exactly."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(repository_root):
    """Capture commit, branch, and dirty status without requiring Git to exist."""

    def run_git(*arguments):
        """Run one read-only Git query, returning ``None`` when unavailable."""
        try:
            completed = subprocess.run(
                ["git"] + list(arguments),
                cwd=str(repository_root),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return completed.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run_git("status", "--porcelain")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
        "status_porcelain": status,
    }


def _threshold_label(threshold):
    """Convert a ratio such as ``0.985`` to a stable CSV suffix such as ``98p5``."""
    percent = threshold * 100.0
    if math.isclose(percent, round(percent), rel_tol=0.0, abs_tol=1e-9):
        return str(int(round(percent)))
    return ("{:.3f}".format(percent)).rstrip("0").rstrip(".").replace(".", "p")


def _action_name(action):
    """Return a human-readable semantic label for a communication action."""
    action = int(action)
    if action == SILENCE:
        return "silence"
    if action == 1:
        return "pose"
    return "map_patch"


def _zero_communication_metrics():
    """Create a complete zero-valued metric schema for legacy/no-broker runs."""
    result = {name: 0 for name in COMMUNICATION_METRICS}
    result["mean_latency"] = 0.0
    return result


def _communication_metrics(env):
    """Return cumulative broker metrics with missing fields filled by zero."""
    metrics = _zero_communication_metrics()
    metrics.update(env.communication_metrics())
    return metrics


def _communication_delta(before, after):
    """Convert two cumulative snapshots into metrics for one decision.

    ``delivered_messages`` counts receiver-copies, so mean latency divides the
    step's latency sum by receiver deliveries rather than sender broadcasts.
    """
    result = {
        name: after.get(name, 0) - before.get(name, 0)
        for name in COMMUNICATION_METRICS
    }
    delivered = result["delivered_messages"]
    result["mean_latency"] = (
        result["latency_total"] / float(delivered) if delivered else 0.0
    )
    return result


def _measurement(env, path_lengths_m):
    """Measure exploration without feeding privileged data to the controller.

    Coverage is the union of cells physically known in agents' private
    ``built_map`` arrays.  Delivered belief cells do not directly increase the
    numerator; communication can improve coverage only by changing later
    motion and sensing.  Ground truth is read solely to exclude invalid map
    cells from the denominator.

    Args:
        env: Completed/reset standalone environment state to inspect.
        path_lengths_m: Accumulated executed distance for each agent.

    Returns:
        Dictionary of team, agent, overlap, path, pose, and communication
        measurements used by step and episode outputs.
    """
    valid = np.asarray(env.gt_map) != UNKNOWN
    private_known = np.stack(
        [((np.asarray(grid) != UNKNOWN) & valid) for grid in env.built_map], axis=0
    )
    agent_cells = private_known.reshape(env.num_agents, -1).sum(axis=1)
    team_known = np.any(private_known, axis=0)
    team_cells = int(team_known.sum())
    total_cells = int(valid.sum())
    agent_ratios = agent_cells.astype(np.float64) / float(total_cells)
    overlap_cells = int(agent_cells.sum() - team_cells)
    cell_area = float(env.resolution) ** 2
    communication = _communication_metrics(env)
    in_range = communication["in_range_recipients"]

    return {
        "team_coverage_cells": team_cells,
        "team_coverage_ratio": team_cells / float(total_cells),
        "total_explorable_cells": total_cells,
        "agent_coverage_cells": [int(value) for value in agent_cells],
        "agent_coverage_ratios": [float(value) for value in agent_ratios],
        "coverage_std_ratio": float(np.std(agent_ratios)),
        "coverage_std_area_m2": float(np.std(agent_cells * cell_area)),
        "overlap_cells": overlap_cells,
        "overlap_ratio_total": overlap_cells / float(total_cells),
        "overlap_area_m2": overlap_cells * cell_area,
        "positions": np.asarray(env.agent_pos, dtype=np.int64).copy(),
        "headings_rad": np.asarray(env.agent_yaw, dtype=np.float64).copy(),
        "path_lengths_m": np.asarray(path_lengths_m, dtype=np.float64).copy(),
        "team_path_length_m": float(np.sum(path_lengths_m)),
        "maximum_agent_path_length_m": float(np.max(path_lengths_m)),
        "communication": communication,
        "delivery_ratio": (
            communication["delivered_messages"] / float(in_range)
            if in_range
            else 0.0
        ),
    }


def _step_row(
    common,
    step,
    measurement,
    previous_team_cells,
    communication_delta,
    communication_cost,
    termination_reason,
):
    """Flatten one team measurement into the stable step-level CSV schema."""
    new_cells = measurement["team_coverage_cells"] - previous_team_cells
    row = dict(common)
    row.update(
        {
            "step": step,
            "team_coverage_cells": measurement["team_coverage_cells"],
            "team_coverage_ratio": measurement["team_coverage_ratio"],
            "total_explorable_cells": measurement["total_explorable_cells"],
            "new_team_cells": new_cells,
            "team_path_length_m": measurement["team_path_length_m"],
            "maximum_agent_path_length_m": measurement[
                "maximum_agent_path_length_m"
            ],
            "coverage_std_ratio": measurement["coverage_std_ratio"],
            "coverage_std_area_m2": measurement["coverage_std_area_m2"],
            "overlap_cells": measurement["overlap_cells"],
            "overlap_ratio_total": measurement["overlap_ratio_total"],
            "overlap_area_m2": measurement["overlap_area_m2"],
            "delivery_ratio": measurement["delivery_ratio"],
            "communication_cost": communication_cost,
            "termination_reason": termination_reason or "",
        }
    )
    for name in COMMUNICATION_METRICS:
        row["comm_" + name] = measurement["communication"][name]
        row["comm_step_" + name] = communication_delta[name]
    row["comm_mean_latency"] = measurement["communication"]["mean_latency"]
    row["comm_step_mean_latency"] = communication_delta["mean_latency"]
    return row


def _agent_rows(common, step, env, measurement):
    """Build long-form per-agent rows for pose, goal, sensing, and path data."""
    rows = []
    goals = env.last_goals
    actions = env.last_requested_communication_actions
    readings = env.latest_sensor_readings
    for agent_id in range(env.num_agents):
        reading = readings[agent_id] or {}
        beams = reading.get("beams", {})
        goal = None if goals is None else goals[agent_id]
        row = dict(common)
        row.update(
            {
                "step": step,
                "agent_id": agent_id,
                "row": int(measurement["positions"][agent_id][0]),
                "column": int(measurement["positions"][agent_id][1]),
                "heading_rad": float(measurement["headings_rad"][agent_id]),
                "goal_row": "" if goal is None else int(goal[0]),
                "goal_column": "" if goal is None else int(goal[1]),
                "coverage_cells": measurement["agent_coverage_cells"][agent_id],
                "coverage_ratio": measurement["agent_coverage_ratios"][agent_id],
                "path_length_m": float(measurement["path_lengths_m"][agent_id]),
                "sensor_type": reading.get("sensor_type", ""),
                "sensor_max_range": reading.get("max_range", ""),
                "requested_communication_action": int(actions[agent_id]),
                "requested_communication_action_name": _action_name(
                    actions[agent_id]
                ),
            }
        )
        for beam_name in BEAM_NAMES:
            beam = beams.get(beam_name, {})
            row[beam_name + "_distance"] = beam.get("distance", "")
            row[beam_name + "_hit"] = beam.get("hit", "")
        rows.append(row)
    return rows


class CsvOutputs:
    """Own CSV writers selected by the requested recording level.

    ``episode`` creates only ``episodes.csv``; ``steps`` also creates
    ``steps.csv``; ``full`` additionally creates ``agents.csv``.  Long-form
    agent rows avoid changing schemas when team size changes.
    """

    def __init__(self, run_directory, record_level, thresholds):
        """Open output files, write headers, and retain handles for flushing."""
        self.record_level = record_level
        self.handles = []
        self.episode_fields = self._episode_fields(thresholds)
        self.step_fields = self._step_fields()
        self.agent_fields = self._agent_fields()
        self.episodes = self._open_writer(
            run_directory / "episodes.csv", self.episode_fields
        )
        self.steps = None
        self.agents = None
        if record_level in ("steps", "full"):
            self.steps = self._open_writer(
                run_directory / "steps.csv", self.step_fields
            )
        if record_level == "full":
            self.agents = self._open_writer(
                run_directory / "agents.csv", self.agent_fields
            )

    def _open_writer(self, path, fields):
        """Open one new CSV and return a strict dictionary writer."""
        handle = path.open("w", encoding="utf-8", newline="")
        self.handles.append(handle)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        return writer

    @staticmethod
    def _common_fields():
        """Return identifiers shared by episode, step, and agent tables."""
        return [
            "schema_version",
            "episode_id",
            "method",
            "map_name",
            "map_path",
            "team_size",
            "seed",
            "communication_mode",
            "communication_protocol",
        ]

    @classmethod
    def _episode_fields(cls, thresholds):
        """Return episode columns, including dynamic coverage milestones."""
        fields = cls._common_fields() + [
            "steps_executed",
            "success",
            "termination_reason",
            "target_coverage",
            "initial_coverage_cells",
            "initial_coverage_ratio",
            "total_explorable_cells",
            "final_coverage_cells",
            "final_coverage_ratio",
            "newly_explored_cells",
            "team_path_length_m",
            "maximum_agent_path_length_m",
            "final_agent_path_lengths_m",
            "final_agent_coverage_ratios",
            "start_positions",
            "final_positions",
            "coverage_std_ratio",
            "coverage_std_area_m2",
            "overlap_cells",
            "overlap_ratio_total",
            "overlap_area_m2",
            "delivery_ratio",
            "communication_cost",
            "attempted_bits_per_new_cell",
            "delivered_bits_per_new_cell",
        ]
        for threshold in thresholds:
            label = _threshold_label(threshold)
            fields.extend(
                [
                    "coverage_{}_reached".format(label),
                    "coverage_{}_step".format(label),
                    "coverage_{}_team_path_m".format(label),
                ]
            )
        fields.extend(["comm_" + name for name in COMMUNICATION_METRICS])
        fields.append("comm_mean_latency")
        return fields

    @classmethod
    def _step_fields(cls):
        """Return cumulative and per-decision team metric columns."""
        fields = cls._common_fields() + [
            "step",
            "team_coverage_cells",
            "team_coverage_ratio",
            "total_explorable_cells",
            "new_team_cells",
            "team_path_length_m",
            "maximum_agent_path_length_m",
            "coverage_std_ratio",
            "coverage_std_area_m2",
            "overlap_cells",
            "overlap_ratio_total",
            "overlap_area_m2",
            "delivery_ratio",
            "communication_cost",
            "termination_reason",
        ]
        fields.extend(["comm_" + name for name in COMMUNICATION_METRICS])
        fields.extend(["comm_step_" + name for name in COMMUNICATION_METRICS])
        fields.extend(["comm_mean_latency", "comm_step_mean_latency"])
        return fields

    @classmethod
    def _agent_fields(cls):
        """Return long-form agent columns, including named four-beam readings."""
        fields = cls._common_fields() + [
            "step",
            "agent_id",
            "row",
            "column",
            "heading_rad",
            "goal_row",
            "goal_column",
            "coverage_cells",
            "coverage_ratio",
            "path_length_m",
            "sensor_type",
            "sensor_max_range",
            "requested_communication_action",
            "requested_communication_action_name",
        ]
        for beam_name in BEAM_NAMES:
            fields.extend([beam_name + "_distance", beam_name + "_hit"])
        return fields

    def write_step(self, row):
        """Write one step row when step recording is enabled."""
        if self.steps is not None:
            self.steps.writerow(_json_safe(row))

    def write_agents(self, rows):
        """Write per-agent rows when full recording is enabled."""
        if self.agents is not None:
            for row in rows:
                self.agents.writerow(_json_safe(row))

    def write_episode(self, row):
        """Write and immediately flush one completed episode summary."""
        self.episodes.writerow(_json_safe(row))
        self.flush()

    def flush(self):
        """Flush every open CSV so completed episodes survive interruption."""
        for handle in self.handles:
            handle.flush()

    def close(self):
        """Close all CSV handles owned by this output group."""
        for handle in self.handles:
            handle.close()


def _create_run_directory(output_root, experiment_name):
    """Create the next non-destructive ``experiment_name/runN`` directory."""
    experiment_directory = output_root / experiment_name
    experiment_directory.mkdir(parents=True, exist_ok=True)
    existing = []
    for child in experiment_directory.iterdir():
        if child.is_dir() and child.name.startswith("run"):
            suffix = child.name[3:]
            if suffix.isdigit():
                existing.append(int(suffix))
    run_number = max(existing, default=0) + 1
    run_directory = experiment_directory / "run{}".format(run_number)
    run_directory.mkdir()
    return run_directory


def _communication_config(args):
    """Build the broker treatment, or ``None`` for explicit legacy behavior.

    ``legacy`` and broker mode ``none`` are intentionally different.  Legacy
    preserves ground-truth standalone navigation; none constructs private
    beliefs and is the fair no-message communication control.
    """
    if args.communication_mode == LEGACY:
        return None
    return CommunicationConfig(
        mode=args.communication_mode,
        tile_size=args.comm_tile_size,
        candidate_count=args.comm_candidate_count,
        radio_range_cells=args.comm_range_cells,
        latency_min_steps=args.comm_latency_min_steps,
        latency_max_steps=args.comm_latency_max_steps,
        packet_loss=args.comm_packet_loss,
        cooldown_steps=args.comm_cooldown_steps,
        bucket_capacity_bits=args.comm_bucket_capacity_bits,
        bucket_refill_bits=args.comm_bucket_refill_bits,
        episode_budget_bits=args.comm_episode_budget_bits,
        ttl_steps=args.comm_ttl_steps,
        timestamp_bits=args.comm_timestamp_bits,
        cost_per_patch=args.comm_cost_per_patch,
    )


def _save_snapshot(run_directory, episode_id, label, env):
    """Save compressed private, team, pose, and optional belief map arrays."""
    episode_directory = run_directory / "maps" / episode_id
    episode_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "private_maps": np.stack(env.built_map, axis=0),
        "team_sensed_map": np.asarray(env.complete_map),
        "positions": np.asarray(env.agent_pos, dtype=np.int64),
        "headings_rad": np.asarray(env.agent_yaw, dtype=np.float64),
        "step": np.asarray([env.num_step], dtype=np.int64),
    }
    if env.communication_broker is not None:
        payload["belief_maps"] = np.stack(
            env.communication_broker.belief_maps, axis=0
        )
    np.savez_compressed(str(episode_directory / (label + ".npz")), **payload)


def _common_row(episode_id, args, map_path, team_size, seed):
    """Build identifiers copied into every row belonging to one episode."""
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "method": args.method,
        "map_name": map_path.name,
        "map_path": str(map_path),
        "team_size": team_size,
        "seed": seed,
        "communication_mode": args.communication_mode,
        "communication_protocol": args.communication_protocol,
    }


def _crossed_thresholds(measurement, thresholds, crossings, step):
    """Record first-passage step/path for newly reached coverage thresholds."""
    newly_crossed = []
    for threshold in thresholds:
        if threshold not in crossings and measurement["team_coverage_ratio"] >= threshold:
            crossings[threshold] = {
                "step": step,
                "team_path_m": measurement["team_path_length_m"],
            }
            newly_crossed.append(threshold)
    return newly_crossed


def _all_agents_stuck(env):
    """Return true only when every planner goal equals its pre-move position."""
    if env.last_goals is None or env.positions_before_goal is None:
        return False
    return bool(np.all(env.last_goals == env.positions_before_goal))


def _run_episode(
    args,
    run_directory,
    outputs,
    episode_index,
    map_path,
    team_size,
    seed,
    env_class=HeadlessGridEnv,
):
    """Run one finite planner episode and write all requested records.

    A fresh environment is created for every map/team-size/seed tuple so no
    map, queue, resource, metric, or RNG state crosses episodes.  Step zero is
    recorded immediately after reset.  Later termination uses the documented
    precedence: target coverage, all agents stuck, no progress, then max steps.

    Returns:
        The episode summary dictionary written to ``episodes.csv``.
    """
    episode_id = "episode_{:06d}".format(episode_index)
    sensor_configs = sensor_configs_from_values(
        args.sensor_types,
        args.sensor_ranges,
        team_size,
        args.sensor_ranges[0],
    )
    communication_config = _communication_config(args)
    env = env_class(
        args.resolution,
        args.sensor_ranges[0],
        team_size,
        args.max_steps,
        str(map_path),
        visualization=args.visualize,
        sensor_configs=sensor_configs,
        communication_config=communication_config,
        communication_protocol=args.communication_protocol,
        seed=seed,
    )
    env.seed(seed)
    env.reset_for_traditional()

    common = _common_row(episode_id, args, map_path, team_size, seed)
    path_lengths_m = np.zeros(team_size, dtype=np.float64)
    previous_positions = np.asarray(env.agent_pos, dtype=np.float64).copy()
    previous_communication = _communication_metrics(env)
    communication_cost = 0.0
    crossings = {}
    no_progress_steps = 0
    termination_reason = None

    initial = _measurement(env, path_lengths_m)
    start_positions = initial["positions"].copy()
    previous_team_cells = initial["team_coverage_cells"]
    initially_crossed = _crossed_thresholds(
        initial, args.coverage_thresholds, crossings, 0
    )
    zero_delta = _communication_delta(previous_communication, previous_communication)
    outputs.write_step(
        _step_row(common, 0, initial, previous_team_cells, zero_delta, 0.0, None)
    )
    outputs.write_agents(_agent_rows(common, 0, env, initial))
    if args.map_snapshots == "all":
        _save_snapshot(run_directory, episode_id, "step_000000", env)
    elif args.map_snapshots == "milestones":
        for threshold in initially_crossed:
            _save_snapshot(
                run_directory,
                episode_id,
                "coverage_{}".format(_threshold_label(threshold)),
                env,
            )

    final = initial
    steps_executed = 0
    if initial["team_coverage_ratio"] >= args.target_coverage:
        termination_reason = "target_coverage"

    while termination_reason is None and steps_executed < args.max_steps:
        step = steps_executed + 1
        output_context = contextlib.nullcontext()
        if not args.verbose:
            output_context = contextlib.redirect_stdout(io.StringIO())
        with output_context:
            if args.method == "cost":
                env.step_for_cost()
            else:
                env.step_for_mmpf()

        current_positions = np.asarray(env.agent_pos, dtype=np.float64)
        path_lengths_m += (
            np.linalg.norm(current_positions - previous_positions, axis=1)
            * args.resolution
        )
        previous_positions = current_positions.copy()

        final = _measurement(env, path_lengths_m)
        communication_delta = _communication_delta(
            previous_communication, final["communication"]
        )
        previous_communication = final["communication"]
        if communication_config is not None:
            patch_bits = env.communication_broker.message_bits(MAP_PATCH)
            communication_cost += (
                communication_config.cost_per_patch
                * communication_delta["attempted_bits"]
                / float(patch_bits)
            )

        new_cells = final["team_coverage_cells"] - previous_team_cells
        no_progress_steps = no_progress_steps + 1 if new_cells == 0 else 0
        newly_crossed = _crossed_thresholds(
            final, args.coverage_thresholds, crossings, step
        )

        if final["team_coverage_ratio"] >= args.target_coverage:
            termination_reason = "target_coverage"
        elif _all_agents_stuck(env):
            termination_reason = "all_agents_stuck"
        elif (
            args.no_progress_patience > 0
            and no_progress_steps >= args.no_progress_patience
        ):
            termination_reason = "no_progress"
        elif step >= args.max_steps:
            termination_reason = "max_steps"

        outputs.write_step(
            _step_row(
                common,
                step,
                final,
                previous_team_cells,
                communication_delta,
                communication_cost,
                termination_reason,
            )
        )
        outputs.write_agents(_agent_rows(common, step, env, final))

        if args.map_snapshots == "all":
            _save_snapshot(
                run_directory, episode_id, "step_{:06d}".format(step), env
            )
        elif args.map_snapshots == "milestones":
            for threshold in newly_crossed:
                _save_snapshot(
                    run_directory,
                    episode_id,
                    "coverage_{}".format(_threshold_label(threshold)),
                    env,
                )

        previous_team_cells = final["team_coverage_cells"]
        steps_executed = step

    if args.map_snapshots == "milestones":
        _save_snapshot(run_directory, episode_id, "final", env)

    newly_explored = final["team_coverage_cells"] - initial["team_coverage_cells"]
    episode_row = dict(common)
    episode_row.update(
        {
            "steps_executed": steps_executed,
            "success": final["team_coverage_ratio"] >= args.target_coverage,
            "termination_reason": termination_reason,
            "target_coverage": args.target_coverage,
            "initial_coverage_cells": initial["team_coverage_cells"],
            "initial_coverage_ratio": initial["team_coverage_ratio"],
            "total_explorable_cells": final["total_explorable_cells"],
            "final_coverage_cells": final["team_coverage_cells"],
            "final_coverage_ratio": final["team_coverage_ratio"],
            "newly_explored_cells": newly_explored,
            "team_path_length_m": final["team_path_length_m"],
            "maximum_agent_path_length_m": final["maximum_agent_path_length_m"],
            "final_agent_path_lengths_m": json.dumps(
                _json_safe(final["path_lengths_m"]), separators=(",", ":")
            ),
            "final_agent_coverage_ratios": json.dumps(
                _json_safe(final["agent_coverage_ratios"]), separators=(",", ":")
            ),
            "start_positions": json.dumps(
                _json_safe(start_positions), separators=(",", ":")
            ),
            "final_positions": json.dumps(
                _json_safe(final["positions"]), separators=(",", ":")
            ),
            "coverage_std_ratio": final["coverage_std_ratio"],
            "coverage_std_area_m2": final["coverage_std_area_m2"],
            "overlap_cells": final["overlap_cells"],
            "overlap_ratio_total": final["overlap_ratio_total"],
            "overlap_area_m2": final["overlap_area_m2"],
            "delivery_ratio": final["delivery_ratio"],
            "communication_cost": communication_cost,
            "attempted_bits_per_new_cell": (
                final["communication"]["attempted_bits"] / float(newly_explored)
                if newly_explored
                else 0.0
            ),
            "delivered_bits_per_new_cell": (
                final["communication"]["delivered_bits"] / float(newly_explored)
                if newly_explored
                else 0.0
            ),
        }
    )
    for threshold in args.coverage_thresholds:
        label = _threshold_label(threshold)
        crossing = crossings.get(threshold)
        episode_row["coverage_{}_reached".format(label)] = crossing is not None
        episode_row["coverage_{}_step".format(label)] = (
            "" if crossing is None else crossing["step"]
        )
        episode_row["coverage_{}_team_path_m".format(label)] = (
            "" if crossing is None else crossing["team_path_m"]
        )
    for name in COMMUNICATION_METRICS:
        episode_row["comm_" + name] = final["communication"][name]
    episode_row["comm_mean_latency"] = final["communication"]["mean_latency"]
    outputs.write_episode(episode_row)
    return episode_row


def _build_manifest(args, repository_root, run_directory, maps):
    """Build the reproducibility manifest before the first episode begins."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_at": _utc_now(),
        "completed_at": None,
        "run_directory": str(run_directory),
        "command": sys.argv,
        "configuration": vars(args),
        "maps": [
            {
                "path": str(path),
                "sha256": _file_sha256(path),
            }
            for path in maps
        ],
        "git": _git_metadata(repository_root),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "episodes_planned": len(maps) * len(args.team_sizes) * len(args.seeds),
        "episodes_completed": 0,
    }


def run_collection(args):
    """Execute the map × team-size × seed sweep and maintain manifest status.

    The manifest begins as ``running``, updates after each completed episode,
    and ends as ``complete`` or ``failed``.  Existing run directories are never
    overwritten.

    Returns:
        Path to the newly created run directory.
    """
    repository_root = SCRIPT_DIR.parent
    maps = [Path(value).expanduser().resolve() for value in args.maps]
    run_directory = _create_run_directory(
        Path(args.output).expanduser().resolve(), args.experiment_name
    )
    manifest_path = run_directory / "manifest.json"
    manifest = _build_manifest(args, repository_root, run_directory, maps)
    _write_json(manifest_path, manifest)
    outputs = CsvOutputs(run_directory, args.record, args.coverage_thresholds)
    episode_index = 0
    try:
        for map_path in maps:
            for team_size in args.team_sizes:
                for seed in args.seeds:
                    episode_index += 1
                    row = _run_episode(
                        args,
                        run_directory,
                        outputs,
                        episode_index,
                        map_path,
                        team_size,
                        seed,
                    )
                    manifest["episodes_completed"] = episode_index
                    _write_json(manifest_path, manifest)
                    print(
                        "{}: coverage={:.4f}, steps={}, reason={}".format(
                            row["episode_id"],
                            row["final_coverage_ratio"],
                            row["steps_executed"],
                            row["termination_reason"],
                        )
                    )
    except Exception as error:
        manifest["status"] = "failed"
        manifest["completed_at"] = _utc_now()
        manifest["error"] = "{}: {}".format(type(error).__name__, error)
        _write_json(manifest_path, manifest)
        raise
    finally:
        outputs.close()

    manifest["status"] = "complete"
    manifest["completed_at"] = _utc_now()
    _write_json(manifest_path, manifest)
    return run_directory


def build_parser():
    """Create the command-line interface for collection and radio treatments."""
    parser = argparse.ArgumentParser(
        description="Collect headless Level-0 cost/MMPF exploration data."
    )
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--method", choices=PLANNERS, default="cost")
    parser.add_argument("--maps", nargs="+", required=True)
    parser.add_argument("--team-sizes", nargs="+", type=int, default=[2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--no-progress-patience", type=int, default=50)
    parser.add_argument(
        "--coverage-thresholds", nargs="+", type=float, default=[0.90, 0.98, 0.99]
    )
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=None,
        help="termination target; defaults to the largest recorded threshold",
    )
    parser.add_argument(
        "--sensor-types",
        nargs="+",
        choices=["omnidirectional", "four_beam"],
        default=["four_beam"],
    )
    parser.add_argument(
        "--sensor-ranges",
        nargs="+",
        type=float,
        default=[3.5],
        help="one range for every agent or one range per agent; accepts inf",
    )
    parser.add_argument(
        "--communication-mode",
        choices=(LEGACY,) + COMMUNICATION_MODES,
        default="none",
        help="legacy is the original omniscient navigation compatibility path",
    )
    parser.add_argument(
        "--communication-protocol",
        choices=COMMUNICATION_PROTOCOLS,
        default="round_robin",
    )
    parser.add_argument("--comm-tile-size", type=int, default=8)
    parser.add_argument("--comm-candidate-count", type=int, default=8)
    parser.add_argument("--comm-range-cells", type=float, default=40.0)
    parser.add_argument("--comm-latency-min-steps", type=int, default=1)
    parser.add_argument("--comm-latency-max-steps", type=int)
    parser.add_argument("--comm-packet-loss", type=float, default=0.0)
    parser.add_argument("--comm-cooldown-steps", type=int, default=3)
    parser.add_argument("--comm-bucket-capacity-bits", type=int, default=314)
    parser.add_argument("--comm-bucket-refill-bits", type=int, default=53)
    parser.add_argument("--comm-episode-budget-bits", type=int)
    parser.add_argument("--comm-ttl-steps", type=int, default=8)
    parser.add_argument("--comm-timestamp-bits", type=int, default=16)
    parser.add_argument("--comm-cost-per-patch", type=float, default=0.01)
    parser.add_argument("--record", choices=RECORD_LEVELS, default="steps")
    parser.add_argument(
        "--map-snapshots", choices=SNAPSHOT_LEVELS, default="none"
    )
    parser.add_argument("--output", default="results")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="open the existing UI; allowed only for a single episode",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show the existing planner's per-step terminal output",
    )
    return parser


def validate_args(args, parser):
    """Validate scientific configuration before creating any output files."""
    if (
        not args.experiment_name
        or Path(args.experiment_name).name != args.experiment_name
        or args.experiment_name in (".", "..")
    ):
        parser.error("experiment name must be one non-empty path component")
    args.coverage_thresholds = sorted(set(args.coverage_thresholds))
    if not args.coverage_thresholds:
        parser.error("at least one coverage threshold is required")
    if any(not 0.0 < value <= 1.0 for value in args.coverage_thresholds):
        parser.error("coverage thresholds must be in (0, 1]")
    if args.target_coverage is None:
        args.target_coverage = max(args.coverage_thresholds)
    if not 0.0 < args.target_coverage <= 1.0:
        parser.error("target coverage must be in (0, 1]")
    if args.max_steps <= 0:
        parser.error("max steps must be positive")
    if args.no_progress_patience < 0:
        parser.error("no-progress patience cannot be negative")
    if args.resolution <= 0 or not math.isfinite(args.resolution):
        parser.error("resolution must be positive and finite")
    if any(team_size <= 0 for team_size in args.team_sizes):
        parser.error("team sizes must be positive")
    if len(set(args.team_sizes)) != len(args.team_sizes):
        parser.error("team sizes must not contain duplicates")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must not contain duplicates")
    maps = [Path(value).expanduser() for value in args.maps]
    missing = [str(path) for path in maps if not path.is_file()]
    if missing:
        parser.error("map files do not exist: {}".format(", ".join(missing)))
    episode_count = len(args.maps) * len(args.team_sizes) * len(args.seeds)
    if args.visualize and episode_count != 1:
        parser.error("--visualize is limited to a single map/team-size/seed episode")
    for team_size in args.team_sizes:
        try:
            sensor_configs_from_values(
                args.sensor_types,
                args.sensor_ranges,
                team_size,
                args.sensor_ranges[0],
            )
        except (TypeError, ValueError) as error:
            parser.error("invalid sensor configuration for team size {}: {}".format(team_size, error))
    try:
        _communication_config(args)
    except (TypeError, ValueError) as error:
        parser.error("invalid communication configuration: {}".format(error))
    return args


def main(argv=None):
    """Parse, validate, and run a collection command."""
    parser = build_parser()
    args = validate_args(parser.parse_args(argv), parser)
    run_directory = run_collection(args)
    print("results written to {}".format(run_directory))


if __name__ == "__main__":
    main()
