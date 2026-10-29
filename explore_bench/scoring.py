"""Audit evaluation artifacts, select on validation, and score locked tests."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from .artifacts import Table, write_json
from .config import SCHEMA, SEMANTICS, digest, file_hash


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def is_true(value):
    return str(value).lower() in ("true", "1")


def checked_rows(directory):
    directory = Path(directory)
    with (directory / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (manifest["kind"] != "evaluation" or manifest["status"] != "complete"
            or manifest["schema_version"] != SCHEMA or manifest["environment_semantics"] != SEMANTICS):
        raise ValueError("only complete schema-v2 evaluations may be scored")
    for name in ("episodes.csv", "steps.csv", "agents.csv"):
        if manifest.get("output_sha256", {}).get(name) != file_hash(directory / name):
            raise ValueError("recorded output hash mismatch: " + name)
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    if digest(manifest["tasks"]) != manifest["tasks_sha256"]:
        raise ValueError("evaluation task hash mismatch")
    episodes = read_csv(directory / "episodes.csv")
    if len(episodes) != len(tasks) or {row["episode_id"] for row in episodes} != set(tasks):
        raise ValueError("missing or duplicated evaluation episodes")
    steps = defaultdict(list)
    for row in read_csv(directory / "steps.csv"):
        steps[row["episode_id"]].append(row)
    agents = defaultdict(list)
    for row in read_csv(directory / "agents.csv"):
        agents[(row["episode_id"], int(row["step"]))].append(row)
    if set(steps) != set(tasks) or {key[0] for key in agents} != set(tasks):
        raise ValueError("unexpected or missing step/agent episode")
    spec = manifest["configuration"]
    for episode in episodes:
        for field, expected in (("schema_version", SCHEMA), ("environment_semantics", SEMANTICS),
                                ("method", manifest["controller"]), ("training_seed", manifest["training_seed"]),
                                ("checkpoint_sha256", manifest["checkpoint_sha256"])):
            if episode[field] != str(expected):
                raise ValueError("episode/manifest identity mismatch: " + field)
        key = episode["episode_id"]
        task = tasks[key]
        trajectory = steps[key]
        if [int(row["step"]) for row in trajectory] != list(range(int(episode["steps_executed"]) + 1)):
            raise ValueError("step trace is incomplete")
        if json.loads(episode["start_positions"]) != task["start_positions"]:
            raise ValueError("episode starts changed")
        if json.loads(episode["start_headings_rad"]) != task["start_headings_rad"]:
            raise ValueError("episode headings changed")
        previous_coverage = previous_distance = area = task_reward = communication_cost = 0.0
        previous_agents = None
        for row in trajectory:
            coverage, distance = float(row["team_coverage_ratio"]), float(row["team_path_length_m"])
            if not 0 <= previous_coverage <= coverage <= 1 or not np.isfinite(distance) or distance < previous_distance:
                raise ValueError("invalid coverage or path trace")
            if int(row["step"]) and distance - previous_distance > spec["num_agents"] * spec["resolution"] * np.sqrt(2) + 1e-9:
                raise ValueError("movement exceeded the fixed-edge allowance")
            group = agents[(key, int(row["step"]))]
            if len(group) != spec["num_agents"] or {int(item["agent_id"]) for item in group} != set(range(spec["num_agents"])):
                raise ValueError("agent trace is incomplete")
            if not np.isclose(sum(float(item["path_length_m"]) for item in group), distance):
                raise ValueError("agent/team distance mismatch")
            if not np.isclose(float(row["team_coverage_cells"]), coverage * float(row["total_explorable_cells"])):
                raise ValueError("coverage numerator/denominator mismatch")
            if not np.isclose(sum(float(item["coverage_cells"]) for item in group) - float(row["team_coverage_cells"]), float(row["overlap_cells"])):
                raise ValueError("overlap accounting mismatch")
            current_agents = {int(item["agent_id"]): item for item in group}
            if previous_agents is not None:
                for agent, item in current_agents.items():
                    old = previous_agents[agent]
                    delta = np.asarray([int(item["row"]) - int(old["row"]), int(item["column"]) - int(old["column"])])
                    if np.any(np.abs(delta) > 1) or not np.isclose(
                        float(item["path_length_m"]) - float(old["path_length_m"]), np.linalg.norm(delta) * spec["resolution"]):
                        raise ValueError("executed distance disagrees with agent motion")
                area += (previous_coverage + coverage) / 2
            previous_agents = current_agents
            if not np.isclose(sum(float(item["learning_reward"]) for item in group),
                              spec["num_agents"] * float(row["task_reward"]) - float(row["communication_cost"])):
                raise ValueError("learning/task/communication reward mismatch")
            task_reward += float(row["task_reward"])
            communication_cost += float(row["communication_cost"])
            previous_coverage, previous_distance = coverage, distance
        for threshold in spec["thresholds"]:
            prefix = "coverage_{}".format(int(round(100 * threshold)))
            observed = next((row for row in trajectory if float(row["team_coverage_ratio"]) >= threshold), None)
            if is_true(episode[prefix + "_reached"]) != (observed is not None):
                raise ValueError("milestone success mismatch")
            if observed is not None and int(episode[prefix + "_step"]) != int(observed["step"]):
                raise ValueError("milestone extraction mismatch")
        if not np.isclose(float(episode["final_coverage_ratio"]), previous_coverage):
            raise ValueError("final coverage does not match steps")
        if is_true(episode["success"]) != (previous_coverage >= spec["target_coverage"]):
            raise ValueError("episode success disagrees with coverage")
        expected_area = (area + (spec["max_steps"] - int(episode["steps_executed"])) * previous_coverage) / spec["max_steps"]
        if not np.isclose(float(episode["coverage_auc"]), expected_area):
            raise ValueError("coverage AUC mismatch")
        if not np.isclose(float(episode["task_reward"]), task_reward) or not np.isclose(float(episode["communication_cost"]), communication_cost):
            raise ValueError("episode/step reward totals mismatch")
        expected_reward = spec["reward_new_cell"] * (float(trajectory[-1]["team_coverage_cells"]) - float(trajectory[0]["team_coverage_cells"]))
        if is_true(episode["success"]) and int(episode["steps_executed"]) > 0:
            expected_reward += spec["reward_success"]
        if not np.isclose(task_reward, expected_reward):
            raise ValueError("task reward disagrees with independent coverage accounting")
    return manifest, episodes, steps


def summary(rows, cap):
    result = {"episodes": len(rows), "success_rate": statistics.mean(is_true(row["success"]) for row in rows),
              "mean_final_coverage": statistics.mean(float(row["final_coverage_ratio"]) for row in rows),
              "mean_coverage_auc": statistics.mean(float(row["coverage_auc"]) for row in rows),
              "mean_capped_T99": statistics.mean(float(row["coverage_99_step"]) if row.get("coverage_99_step", "") != "" else cap + 1 for row in rows)}
    for field in ("team_path_length_m", "overlap_ratio_total", "coverage_std_area_m2", "attempted_bits", "delivered_bits", "task_reward", "communication_cost"):
        result["mean_" + field] = statistics.mean(float(row[field]) for row in rows)
    for threshold in (90, 98, 99):
        values = [float(row["coverage_{}_step".format(threshold)]) for row in rows if row.get("coverage_{}_step".format(threshold), "") != ""]
        result["T{}_successes".format(threshold)] = len(values)
        result["median_T{}_conditional".format(threshold)] = statistics.median(values) if values else ""
    return result


def select(validation_directory, output):
    validation_directory, output = Path(validation_directory), Path(output)
    if output.exists():
        raise FileExistsError("selection record is immutable")
    with (validation_directory / "manifest.json").open(encoding="utf-8") as handle:
        validation = json.load(handle)
    if validation["kind"] != "validation" or validation["status"] != "complete":
        raise ValueError("validation is incomplete")
    with (Path(validation["training_run"]) / "manifest.json").open(encoding="utf-8") as handle:
        training = json.load(handle)
    if training["status"] != "complete":
        raise ValueError("final checkpoint selection requires a complete training attempt")
    from .config import evaluation_tasks
    expected_tasks = [task.to_dict() for task in evaluation_tasks(validation["configuration"], "validation")]
    candidates = []
    for entry in validation["evaluations"]:
        record, rows, _ = checked_rows(entry["directory"])
        if any(task["suite"] != "validation" for task in record["tasks"]):
            raise ValueError("test tasks must never select checkpoints")
        if (record["tasks"] != expected_tasks or record["configuration"] != validation["configuration"]
                or record["training_seed"] != validation["training_seed"]):
            raise ValueError("validation changed tasks, configuration or training seed")
        if record["checkpoint_sha256"] != entry["actor_sha256"]:
            raise ValueError("validation checkpoint mismatch")
        statistics_row = summary(rows, validation["configuration"]["max_steps"])
        candidates.append(dict(entry, score=statistics_row))
    if not candidates or len(candidates) != validation["checkpoint_count"]:
        raise ValueError("checkpoint validation inventory is incomplete")
    best = min(candidates, key=lambda item: (-item["score"]["success_rate"],
               -item["score"]["mean_coverage_auc"], item["score"]["mean_capped_T99"], item["environment_steps"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"kind": "selection", "status": "complete", "configuration": validation["configuration"],
                        "configuration_sha256": digest(validation["configuration"]),
                        "training_seed": validation["training_seed"], "selected": best,
                        "candidates": candidates, "validation_manifest_sha256": file_hash(validation_directory / "manifest.json")})
    return best


def score(inputs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    groups, inventories, curves = defaultdict(list), [], []
    # Per-episode coverage traces, keyed by (controller, suite, training seed)
    # and pooled across ALL input directories before curves are computed.
    #
    # Why: slow scripted planners are now run one directory per map
    # (submit_baselines.sh -> evaluate --maps), e.g. ten MMPF directories.
    # The old code built curves inside the directory loop, so MMPF appeared in
    # coverage_curves.csv as ten separate per-map curves instead of one.
    # Pooling here makes a sharded run score exactly like an unsharded one;
    # aggregate.csv/per_seed.csv were already pooled via `groups` and did not
    # change. Verified by test_sharded_evaluations_score_as_one_policy.
    trajectories = defaultdict(list)
    physical_contract = None
    seen = set()
    for directory in inputs:
        record, rows, traces = checked_rows(directory)
        spec = record["configuration"]
        contract = {key: spec[key] for key in ("num_agents", "resolution", "sensor_type", "sensor_range",
                    "communication", "target_coverage", "max_steps", "thresholds", "maps")}
        # The navigation rule changes results as much as the sensor does
        # (classical planners stall under "belief", finish under
        # "ground_truth"), so it is part of the physical contract: scoring a
        # legacy directory together with a belief directory raises below.
        # .get() keeps pre-2026-10-28 manifests valid; they lack the key and
        # were all belief routing.
        contract["navigation"] = spec.get("navigation", "belief")
        if physical_contract is not None and contract != physical_contract:
            raise ValueError("score each sensor/channel/environment contract separately")
        physical_contract = contract
        inventories.append({"directory": str(Path(directory).resolve()), "manifest_sha256": file_hash(Path(directory) / "manifest.json")})
        for row in rows:
            identity = (row["method"], row["training_seed"], row["episode_id"])
            if identity in seen:
                raise ValueError("duplicate policy/task result")
            seen.add(identity)
            groups[(row["method"], row["suite"], row["training_seed"])].append(row)
            trajectories[(record["controller"], row["suite"], record["training_seed"])].append(
                [float(item["team_coverage_ratio"]) for item in traces[row["episode_id"]]])
    # One curve per (controller, suite, seed): the mean coverage over every
    # pooled episode at each decision. Episodes that hit 99% early stop
    # recording, so their final coverage is carried forward to the cap,
    # the same convention as coverage AUC. Keys sort as strings because
    # scripted controllers use "" as training seed while MAPPO uses ints.
    for (method, suite, seed), series in sorted(trajectories.items(), key=lambda item: str(item[0])):
        for step in range(physical_contract["max_steps"] + 1):
            curves.append({"method": method, "suite": suite, "training_seed": seed, "step": step,
                           "mean_coverage": statistics.mean(trace[min(step, len(trace) - 1)] for trace in series)})
    per_seed, aggregate = [], []
    for (method, suite, seed), rows in sorted(groups.items()):
        split, start = {"validation": ("validation", 10001), "familiar": ("train", 20001),
                        "unseen": ("test", 30001)}[suite]
        expected_tasks = {"{}__{}__seed_{}".format(suite, Path(name).stem, episode_seed)
                          for name in physical_contract["maps"][split]
                          for episode_seed in range(start, start + 10)}
        if {row["episode_id"] for row in rows} != expected_tasks:
            raise ValueError("every policy must complete the same full suite of paired tasks")
        per_seed.append(dict(method=method, suite=suite, training_seed=seed,
                             **summary(rows, physical_contract["max_steps"])))
    grouped = defaultdict(list)
    for row in per_seed:
        grouped[(row["method"], row["suite"])].append(row)
    rng = np.random.RandomState(20261021)
    for (method, suite), rows in grouped.items():
        for metric in ("success_rate", "mean_final_coverage", "mean_coverage_auc", "mean_capped_T99"):
            values = np.asarray([row[metric] for row in rows])
            independent = all(row["training_seed"] != "" for row in rows)
            ci = ["", ""]
            if independent and len(values) >= 2:
                samples = values[rng.randint(0, len(values), size=(10000, len(values)))].mean(axis=1)
                ci = np.percentile(samples, [2.5, 97.5]).tolist()
            aggregate.append({"method": method, "suite": suite, "metric": metric,
                              "training_seeds": len(values) if independent else 0,
                              "mean": float(values.mean()), "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else "",
                              "ci95_low": ci[0], "ci95_high": ci[1]})
    if not per_seed:
        raise ValueError("no evaluation inputs")
    for name, rows in (("per_seed.csv", per_seed), ("aggregate.csv", aggregate), ("coverage_curves.csv", curves)):
        table = Table(output / name, list(rows[0]))
        try:
            for row in rows:
                table.append(row)
        finally:
            table.close()
    write_json(output / "inventory.json", {"inputs": inventories, "physical_contract": physical_contract,
                                          "bootstrap_seed": 20261021, "bootstrap_replicates": 10000})
    lines = ["# Level-0 policy scores", "", "Each learned-policy sample is one independent training seed.",
             "Familiar and unseen suites are reported separately. Missing milestones are failures, not zero times.", "",
             "| Method | Suite | Metric | Training seeds | Mean | 95% interval |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in aggregate:
        lines.append("| {method} | {suite} | {metric} | {training_seeds} | {mean:.6g} | {ci95_low}, {ci95_high} |".format(**row))
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate
