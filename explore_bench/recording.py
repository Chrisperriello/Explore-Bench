"""Algorithm-independent episode accounting and schema-v2 CSV recording."""

from pathlib import Path
import numpy as np

from .artifacts import Table
from .config import SCHEMA, SEMANTICS


COMMON = ["schema_version", "environment_semantics", "episode_id", "suite", "method",
          "checkpoint_sha256", "training_seed", "map_name", "map_path", "team_size", "seed",
          "communication_mode", "communication_protocol"]
EPISODE = ["steps_executed", "success", "termination_reason", "terminated", "truncated",
           "target_coverage", "initial_coverage_ratio", "final_coverage_ratio", "coverage_auc",
           "team_path_length_m", "maximum_agent_path_length_m", "coverage_std_ratio",
           "coverage_std_area_m2", "overlap_cells", "overlap_ratio_total", "start_positions",
           "start_headings_rad", "final_positions", "task_reward", "communication_cost",
           "learning_rewards", "attempted_bits", "delivered_bits", "communication_metrics"]
STEP = ["step", "team_coverage_cells", "team_coverage_ratio", "total_explorable_cells",
        "new_team_cells", "team_path_length_m", "maximum_agent_path_length_m", "coverage_std_ratio",
        "coverage_std_area_m2", "overlap_cells", "overlap_ratio_total", "task_reward",
        "communication_cost", "learning_rewards", "termination_reason", "communication_metrics"]
AGENT = ["step", "agent_id", "row", "column", "heading_rad", "goal_row", "goal_column",
         "executed_path", "coverage_cells", "coverage_ratio", "path_length_m", "sensor_type",
         "sensor_max_range", "sensor_reading", "requested_communication_action", "learning_reward"]


def label(threshold):
    return str(int(round(100 * threshold)))


class Recorder:
    def __init__(self, directory, thresholds, full=True):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        milestone_fields = ["coverage_{}_{}".format(label(t), key)
                            for t in thresholds for key in ("reached", "step", "team_path_m")]
        self.episodes = Table(directory / "episodes.csv", COMMON + EPISODE + milestone_fields)
        self.steps = Table(directory / "steps.csv", COMMON + STEP) if full else None
        self.agents = Table(directory / "agents.csv", COMMON + AGENT) if full else None

    def close(self):
        for table in (self.episodes, self.steps, self.agents):
            if table:
                table.close()


class EpisodeLedger:
    """Observe transitions; never supply derived evaluator values to a policy."""

    def __init__(self, session, method, training_seed="", checkpoint_hash=""):
        task = session.task
        self.common = dict(zip(COMMON, [SCHEMA, SEMANTICS, task.task_id, task.suite, method,
            checkpoint_hash, training_seed, Path(task.map_path).stem, task.map_path,
            session.num_agents, task.seed, session.spec["communication"]["mode"],
            "none" if method in ("cost", "mmpf", "random", "wait") else "policy"]))
        self.crossings = {}
        self.initial = session.measurement()["team_coverage_ratio"]
        self.previous_ratio = self.initial
        self.previous_cells = session.initial_cells
        self.area = 0.0

    def observe(self, session, recorder):
        measurement = session.measurement()
        ratio = measurement["team_coverage_ratio"]
        if session.num_step:
            self.area += (self.previous_ratio + ratio) / 2.0
        for threshold in session.spec["thresholds"]:
            if ratio >= threshold and threshold not in self.crossings:
                self.crossings[threshold] = (session.num_step, measurement["team_path_length_m"])
        termination = ("target_coverage" if session.terminated else "max_steps" if session.truncated else "")
        if recorder.steps:
            row = {key: measurement[key] for key in STEP if key in measurement}
            row.update(self.common, step=session.num_step,
                       new_team_cells=measurement["team_coverage_cells"] - self.previous_cells,
                       **session.last_reward, termination_reason=termination,
                       communication_metrics=measurement["communication"])
            recorder.steps.append(row)
        if recorder.agents:
            for agent, position in enumerate(session.agent_pos):
                recorder.agents.append(dict(self.common, step=session.num_step, agent_id=agent,
                    row=position[0], column=position[1], heading_rad=session.agent_yaw[agent],
                    goal_row=int(session.last_goals[agent, 0]), goal_column=int(session.last_goals[agent, 1]),
                    executed_path=session.last_paths[agent],
                    coverage_cells=int(measurement["agent_coverage_cells"][agent]),
                    coverage_ratio=float(measurement["agent_coverage_ratios"][agent]),
                    path_length_m=float(session.path_lengths_m[agent]), sensor_type=session.spec["sensor_type"],
                    sensor_max_range=session.spec["sensor_range"], sensor_reading=session.latest_sensor_readings[agent],
                    requested_communication_action=int(session.last_communication_actions[agent]),
                    learning_reward=float(session.last_reward["learning_rewards"][agent])))
        self.previous_ratio, self.previous_cells = ratio, measurement["team_coverage_cells"]

    def finish(self, session, recorder):
        measurement = session.measurement()
        metrics = measurement["communication"]
        row = dict(self.common, steps_executed=session.num_step, success=session.terminated,
                   termination_reason="target_coverage" if session.terminated else "max_steps",
                   terminated=session.terminated, truncated=session.truncated,
                   target_coverage=session.spec["target_coverage"], initial_coverage_ratio=self.initial,
                   final_coverage_ratio=measurement["team_coverage_ratio"],
                   coverage_auc=(self.area + (session.max_steps - session.num_step) * self.previous_ratio) / session.max_steps,
                   start_positions=session.task.start_positions, start_headings_rad=session.task.start_headings_rad,
                   final_positions=session.agent_pos, task_reward=session.task_reward_total,
                   communication_cost=session.communication_cost_total, learning_rewards=session.learning_reward_total,
                   attempted_bits=metrics["attempted_bits"], delivered_bits=metrics["delivered_bits"],
                   communication_metrics=metrics)
        for key in ("team_path_length_m", "maximum_agent_path_length_m", "coverage_std_ratio",
                    "coverage_std_area_m2", "overlap_cells", "overlap_ratio_total"):
            row[key] = measurement[key]
        for threshold in session.spec["thresholds"]:
            crossing = self.crossings.get(threshold)
            prefix = "coverage_" + label(threshold)
            row[prefix + "_reached"] = crossing is not None
            row[prefix + "_step"] = "" if crossing is None else crossing[0]
            row[prefix + "_team_path_m"] = "" if crossing is None else crossing[1]
        recorder.episodes.append(row)
        return row
