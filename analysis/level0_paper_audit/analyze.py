#!/usr/bin/env python3
"""Validate and report a completed Level-0 paper-audit collection."""

import argparse
import csv
import html
import json
import math
import statistics
from pathlib import Path


TREATMENTS = (
    "omnidirectional_3p5",
    "four_beam_3p5",
    "omnidirectional_7p0",
    "four_beam_7p0",
)
METHODS = ("cost", "mmpf")
TREATMENT_SETTINGS = {
    "omnidirectional_3p5": ("omnidirectional", 3.5, "paper_code_control"),
    "four_beam_3p5": ("four_beam", 3.5, "primary_sensor_extension"),
    "omnidirectional_7p0": (
        "omnidirectional",
        7.0,
        "gazebo_range_sensitivity",
    ),
    "four_beam_7p0": ("four_beam", 7.0, "paired_range_sensitivity"),
}
PAPER = {
    "Ttopo": {"cost": 134.0, "mmpf": 100.0},
    "Ttotal": {"cost": 172.0, "mmpf": 125.0},
    "sigma": {"cost": 0.09, "mmpf": 0.01},
    "r_o": {"cost": 0.10, "mmpf": 0.04},
}
CLAIM_TREATMENT = "omnidirectional_3p5"
CLAIM_METRICS = (
    ("Ttopo", "coverage_90_step", 90),
    ("Ttotal", "coverage_99_step", 99),
)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    if value in (None, ""):
        return None
    return float(value)


def integer(value):
    if value in (None, ""):
        return None
    return int(float(value))


def truth(value):
    return str(value).lower() in ("true", "1", "yes")


def scaled_number(value, factor):
    value = number(value)
    return None if value is None else value * factor


def validate_manifest_task(episode, treatment, manifest, run_directory):
    task = manifest.get("task") or {}
    episode_matches = {
        "task_id": str(episode.get("task_id", "")),
        "treatment": episode.get("treatment", ""),
        "method": episode.get("method", ""),
        "seed": str(episode.get("seed", "")),
    }
    for name, episode_value in episode_matches.items():
        if str(task.get(name, "")) != episode_value:
            raise ValueError("manifest/episode {} mismatch in {}".format(name, run_directory))
    expected = {
        "treatment": treatment,
        "map_name": "room",
        "map_path": "onpolicy/onpolicy/envs/GridEnv/datasets/room1_modified.pgm",
        "team_size": "2",
        "resolution": "0.1",
        "communication_mode": "legacy",
        "communication_protocol": "round_robin",
        "max_steps": "1000",
        "coverage_thresholds": "0.9;0.98;0.99",
        "target_coverage": "0.99",
    }
    for name, expected_value in expected.items():
        if str(task.get(name, "")) != expected_value:
            raise ValueError("frozen {} changed in {}".format(name, run_directory))


def newest_complete_run(task_directory):
    candidates = []
    for manifest_path in task_directory.glob("run*/manifest.json"):
        try:
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, ValueError):
            continue
        run_directory = manifest_path.parent
        if (
            manifest.get("status") == "complete"
            and manifest.get("episodes_completed") == 1
            and (run_directory / "episodes.csv").is_file()
            and (run_directory / "steps.csv").is_file()
            and (run_directory / "agents.csv").is_file()
        ):
            suffix = run_directory.name[3:]
            candidates.append((int(suffix) if suffix.isdigit() else -1, run_directory, manifest))
    return max(candidates, default=None, key=lambda item: item[0])


def load_collection(input_directory):
    episodes = []
    steps = []
    inventory = []
    for treatment in TREATMENTS:
        treatment_directory = input_directory / treatment
        task_directories = sorted(treatment_directory.glob("task_*"))
        if len(task_directories) != 20:
            raise ValueError(
                "{} has {} task directories; expected 20".format(
                    treatment, len(task_directories)
                )
            )
        for task_directory in task_directories:
            selected = newest_complete_run(task_directory)
            if selected is None:
                raise ValueError("no complete run in {}".format(task_directory))
            attempt, run_directory, manifest = selected
            episode_rows = read_csv(run_directory / "episodes.csv")
            if len(episode_rows) != 1:
                raise ValueError("{} must contain one episode".format(run_directory))
            episode = episode_rows[0]
            if episode["treatment"] != treatment:
                raise ValueError("treatment mismatch in {}".format(run_directory))
            expected_task_id = int(task_directory.name.split("_", 1)[1])
            if integer(episode["task_id"]) != expected_task_id:
                raise ValueError("task directory mismatch in {}".format(run_directory))
            validate_manifest_task(episode, treatment, manifest, run_directory)
            episodes.append(episode)
            steps.extend(read_csv(run_directory / "steps.csv"))
            inventory.append(
                {
                    "treatment": treatment,
                    "task_id": episode["task_id"],
                    "method": episode["method"],
                    "seed": episode["seed"],
                    "attempt": attempt,
                    "run_directory": str(run_directory),
                    "repository_commit": manifest.get("repository_commit", ""),
                    "map_sha256": manifest.get("map_sha256", ""),
                    "task_table_sha256": manifest.get("task_table_sha256", ""),
                    "repository_dirty": manifest.get("repository_dirty", ""),
                    "collector_sha256": manifest.get("collector_sha256", ""),
                }
            )
    if len(episodes) != 80:
        raise ValueError("expected 80 episodes, found {}".format(len(episodes)))
    return episodes, steps, inventory


def validate_pairs(episodes, inventory):
    checks = []
    keys = {}
    for row in episodes:
        key = (row["method"], integer(row["seed"]))
        keys.setdefault(key, []).append(row)
    expected = {(method, seed) for method in METHODS for seed in range(1, 11)}
    if set(keys) != expected:
        raise ValueError("method/seed pairs do not match the frozen design")
    for key, rows in keys.items():
        if {row["treatment"] for row in rows} != set(TREATMENTS):
            raise ValueError("{} does not contain all treatments".format(key))
        starts = {row["start_positions"] for row in rows}
        headings = {row["start_headings_rad"] for row in rows}
        task_ids = {integer(row["task_id"]) for row in rows}
        if len(starts) != 1 or len(headings) != 1 or len(task_ids) != 1:
            raise ValueError("{} is not start/headings paired".format(key))
        for row in rows:
            expected_sensor, expected_range, expected_role = TREATMENT_SETTINGS[
                row["treatment"]
            ]
            if (
                row["sensor_type"] != expected_sensor
                or not math.isclose(number(row["sensor_range"]), expected_range)
                or row["treatment_role"] != expected_role
                or row["communication_mode"] != "legacy"
            ):
                raise ValueError("frozen treatment settings changed in {}".format(key))
    checks.append(
        {
            "check": "paired_starts_and_headings",
            "status": "pass",
            "detail": "20 method/seed blocks contain all four treatments with exact starts",
        }
    )
    commits = {row["repository_commit"] for row in inventory}
    maps = {row["map_sha256"] for row in inventory}
    collectors = {row["collector_sha256"] for row in inventory}
    dirty = {row["repository_dirty"] for row in inventory}
    task_hashes_by_treatment = {
        treatment: {
            row["task_table_sha256"]
            for row in inventory
            if row["treatment"] == treatment
        }
        for treatment in TREATMENTS
    }
    if len(commits) != 1 or "" in commits:
        raise ValueError("audit tasks do not share one repository commit")
    if len(maps) != 1 or "" in maps:
        raise ValueError("audit tasks do not share one map hash")
    if len(collectors) != 1 or "" in collectors:
        raise ValueError("audit tasks do not share one collector hash")
    if dirty != {False}:
        raise ValueError("audit tasks must be collected from a clean worktree")
    if any(len(hashes) != 1 or "" in hashes for hashes in task_hashes_by_treatment.values()):
        raise ValueError("a treatment used more than one task table")
    checks.extend(
        [
            {
                "check": "repository_commit",
                "status": "pass",
                "detail": next(iter(commits)),
            },
            {
                "check": "map_hash",
                "status": "pass",
                "detail": next(iter(maps)),
            },
            {
                "check": "collector_hash",
                "status": "pass",
                "detail": next(iter(collectors)),
            },
        ]
    )
    return checks


def describe(values):
    values = [float(value) for value in values if value not in (None, "")]
    if not values:
        return {"n": 0, "mean": "", "median": "", "minimum": "", "maximum": ""}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def summaries(episodes):
    fields = (
        "final_team_coverage_ratio",
        "coverage_90_step",
        "coverage_98_step",
        "coverage_99_step",
        "coverage_90_wall_seconds",
        "coverage_99_wall_seconds",
        "coverage_99_paper_sigma_area_m2",
        "coverage_99_paper_overlap_ratio_total_terrain",
        "released_coverage_98_released_sigma_ratio",
        "released_coverage_98_released_overlap_ratio",
    )
    rows = []
    for treatment in TREATMENTS:
        for method in METHODS:
            group = [
                row
                for row in episodes
                if row["treatment"] == treatment and row["method"] == method
            ]
            summary = {
                "treatment": treatment,
                "method": method,
                "runs": len(group),
                "coverage_99_successes": sum(
                    truth(row["coverage_99_reached"]) for row in group
                ),
            }
            for field in fields:
                values = describe(number(row.get(field)) for row in group)
                for statistic_name, value in values.items():
                    summary[field + "_" + statistic_name] = value
            rows.append(summary)
    return rows


def paired_rows(episodes):
    lookup = {
        (row["treatment"], row["method"], integer(row["seed"])): row
        for row in episodes
    }
    rows = []
    for sensor_range, suffix in ((3.5, "3p5"), (7.0, "7p0")):
        omni = "omnidirectional_" + suffix
        beam = "four_beam_" + suffix
        for method in METHODS:
            for seed in range(1, 11):
                original = lookup[(omni, method, seed)]
                extension = lookup[(beam, method, seed)]
                original_99 = number(original["coverage_99_step"])
                extension_99 = number(extension["coverage_99_step"])
                rows.append(
                    {
                        "sensor_range": sensor_range,
                        "method": method,
                        "seed": seed,
                        "start_positions": original["start_positions"],
                        "omnidirectional_final_coverage": original[
                            "final_team_coverage_ratio"
                        ],
                        "four_beam_final_coverage": extension[
                            "final_team_coverage_ratio"
                        ],
                        "delta_four_beam_minus_omnidirectional_final_coverage": (
                            number(extension["final_team_coverage_ratio"])
                            - number(original["final_team_coverage_ratio"])
                        ),
                        "omnidirectional_99_step": original["coverage_99_step"],
                        "four_beam_99_step": extension["coverage_99_step"],
                        "delta_four_beam_minus_omnidirectional_99_step": (
                            ""
                            if original_99 is None or extension_99 is None
                            else extension_99 - original_99
                        ),
                    }
                )
    return rows


def paired_summaries(pairs):
    rows = []
    for sensor_range in (3.5, 7.0):
        for method in METHODS:
            group = [
                row
                for row in pairs
                if row["sensor_range"] == sensor_range and row["method"] == method
            ]
            coverage = describe(
                row["delta_four_beam_minus_omnidirectional_final_coverage"]
                for row in group
            )
            steps = describe(
                row["delta_four_beam_minus_omnidirectional_99_step"]
                for row in group
            )
            rows.append(
                {
                    "sensor_range": sensor_range,
                    "method": method,
                    "pairs": len(group),
                    "median_delta_final_coverage": coverage["median"],
                    "minimum_delta_final_coverage": coverage["minimum"],
                    "maximum_delta_final_coverage": coverage["maximum"],
                    "complete_99_step_pairs": steps["n"],
                    "median_delta_99_step": steps["median"],
                }
            )
    return rows


def one_sided_sign_test(favoring, opposing):
    """Exact P(X >= favoring) for non-tied pairs under p=0.5."""
    non_tied = favoring + opposing
    if not non_tied:
        return None
    return sum(
        math.comb(non_tied, successes)
        for successes in range(favoring, non_tied + 1)
    ) / float(2 ** non_tied)


def median_order_statistic_interval(values, confidence=0.95):
    """Return the narrowest distribution-free interval meeting confidence."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None, None, None
    n = len(ordered)
    usable = []
    for lower_index in range(n // 2):
        tail = sum(math.comb(n, j) for j in range(lower_index + 1)) / float(2 ** n)
        coverage = 1.0 - 2.0 * tail
        if coverage >= confidence:
            usable.append((lower_index, coverage))
    lower_index, coverage = max(usable, default=(0, 1.0 - 2.0 / float(2 ** n)))
    return ordered[lower_index], ordered[-lower_index - 1], coverage


def method_ranking_pairs(episodes):
    """Pair the paper-control Cost and MMPF observations by frozen seed."""
    lookup = {
        (row["method"], integer(row["seed"])): row
        for row in episodes
        if row["treatment"] == CLAIM_TREATMENT
    }
    rows = []
    for seed in range(1, 11):
        cost = lookup[("cost", seed)]
        mmpf = lookup[("mmpf", seed)]
        row = {
            "seed": seed,
            "cost_start_positions": cost["start_positions"],
            "mmpf_start_positions": mmpf["start_positions"],
            "cost_start_headings_rad": cost["start_headings_rad"],
            "mmpf_start_headings_rad": mmpf["start_headings_rad"],
            "starts_and_headings_match": (
                cost["start_positions"] == mmpf["start_positions"]
                and cost["start_headings_rad"] == mmpf["start_headings_rad"]
            ),
        }
        for metric, field, _ in CLAIM_METRICS:
            cost_value = number(cost.get(field))
            mmpf_value = number(mmpf.get(field))
            row["cost_{}".format(metric)] = "" if cost_value is None else cost_value
            row["mmpf_{}".format(metric)] = "" if mmpf_value is None else mmpf_value
            row["mmpf_minus_cost_{}".format(metric)] = (
                "" if cost_value is None or mmpf_value is None else mmpf_value - cost_value
            )
        rows.append(row)
    return rows


def claim_reproduction_rows(pairs):
    """Evaluate the paper's Level-0 claim that field/MMPF beats Cost."""
    rows = []
    for metric, _, threshold in CLAIM_METRICS:
        matched_starts = sum(truth(row["starts_and_headings_match"]) for row in pairs)
        cost_values = [number(row["cost_{}".format(metric)]) for row in pairs]
        mmpf_values = [number(row["mmpf_{}".format(metric)]) for row in pairs]
        deltas = [
            number(row["mmpf_minus_cost_{}".format(metric)]) for row in pairs
        ]
        complete_deltas = [value for value in deltas if value is not None]
        favoring = sum(value < 0 for value in complete_deltas)
        ties = sum(value == 0 for value in complete_deltas)
        opposing = sum(value > 0 for value in complete_deltas)
        p_value = one_sided_sign_test(favoring, opposing)
        ci_low, ci_high, ci_coverage = median_order_statistic_interval(
            complete_deltas
        )
        cost_summary = describe(cost_values)
        mmpf_summary = describe(mmpf_values)
        delta_summary = describe(complete_deltas)
        passed = (
            len(pairs) == 10
            and matched_starts == 10
            and cost_summary["n"] == 10
            and mmpf_summary["n"] == 10
            and len(complete_deltas) == 10
            and favoring >= 9
            and delta_summary["median"] < 0
            and ci_high is not None
            and ci_high < 0
            and p_value is not None
            and p_value < 0.05
        )
        detail = (
            "MMPF is faster in {}/10 pairs; median delta {:.6g}; "
            "interval [{:.6g}, {:.6g}]; one-sided sign-test p={:.6g}"
        ).format(
            favoring,
            delta_summary["median"] if complete_deltas else math.nan,
            ci_low if ci_low is not None else math.nan,
            ci_high if ci_high is not None else math.nan,
            p_value if p_value is not None else math.nan,
        )
        rows.append(
            {
                "claim": "MMPF requires fewer decisions than Cost",
                "metric": metric,
                "coverage_threshold_percent": threshold,
                "total_pairs": len(pairs),
                "matched_start_pairs": matched_starts,
                "cost_successes": cost_summary["n"],
                "mmpf_successes": mmpf_summary["n"],
                "complete_pairs": len(complete_deltas),
                "cost_median": cost_summary["median"],
                "mmpf_median": mmpf_summary["median"],
                "median_mmpf_minus_cost": delta_summary["median"],
                "median_delta_interval_low": ci_low if ci_low is not None else "",
                "median_delta_interval_high": ci_high if ci_high is not None else "",
                "interval_coverage": ci_coverage if ci_coverage is not None else "",
                "pairs_favoring_mmpf": favoring,
                "ties": ties,
                "pairs_favoring_cost": opposing,
                "one_sided_sign_test_p_value": p_value if p_value is not None else "",
                "gate_status": "pass" if passed else "fail",
                "detail": detail,
            }
        )
    return rows


def parsed_agent_zero(value):
    if value in (None, ""):
        return None
    values = json.loads(value)
    return float(values[0])


def metric_candidates(episodes, method, metric, treatment="omnidirectional_3p5"):
    rows = [
        row
        for row in episodes
        if row["treatment"] == treatment and row["method"] == method
    ]
    if metric == "Ttopo":
        return {
            "decision_step_at_90": [number(row["coverage_90_step"]) for row in rows],
            "released_strict_path_count_at_90": [
                parsed_agent_zero(
                    row["released_coverage_90_historical_path_log_counts"]
                )
                for row in rows
            ],
            "wall_seconds_at_90": [
                number(row["coverage_90_wall_seconds"]) for row in rows
            ],
            "cpu_seconds_at_90": [
                number(row["coverage_90_cpu_seconds"]) for row in rows
            ],
            "inclusive_path_count_at_90": [
                parsed_agent_zero(row["coverage_90_historical_path_log_counts"])
                for row in rows
            ],
        }
    if metric == "Ttotal":
        return {
            "decision_step_at_99": [number(row["coverage_99_step"]) for row in rows],
            "wall_seconds_at_99": [
                number(row["coverage_99_wall_seconds"]) for row in rows
            ],
            "cpu_seconds_at_99": [
                number(row["coverage_99_cpu_seconds"]) for row in rows
            ],
            "historical_agent0_path_count_at_99": [
                parsed_agent_zero(row["coverage_99_historical_path_log_counts"])
                for row in rows
            ],
            "released_boundary_decision_step_at_98": [
                number(row["released_coverage_98_step"]) for row in rows
            ],
            "released_boundary_path_count_at_98": [
                parsed_agent_zero(
                    row["released_coverage_98_historical_path_log_counts"]
                )
                for row in rows
            ],
        }
    if metric == "sigma":
        return {
            "paper_equation_m2_at_99": [
                number(row["coverage_99_paper_sigma_area_m2"]) for row in rows
            ],
            "released_normalized_ratio_at_98": [
                number(row["released_coverage_98_released_sigma_ratio"])
                for row in rows
            ],
            "collector_normalized_ratio_at_99": [
                number(row["coverage_99_collector_sigma_ratio"]) for row in rows
            ],
        }
    return {
        "paper_equation_total_terrain_at_99": [
            number(row["coverage_99_paper_overlap_ratio_total_terrain"])
            for row in rows
        ],
        "paper_equation_percent_at_99": [
            scaled_number(
                row["coverage_99_paper_overlap_ratio_total_terrain"], 100.0
            )
            for row in rows
        ],
        "released_sum_agent_ratios_minus_one_at_98": [
            number(row["released_coverage_98_released_overlap_ratio"])
            for row in rows
        ],
        "released_percent_at_98": [
            scaled_number(row["released_coverage_98_released_overlap_ratio"], 100.0)
            for row in rows
        ],
        "collector_overlap_over_total_at_99": [
            number(row["coverage_99_collector_overlap_ratio_total"]) for row in rows
        ],
        "collector_percent_at_99": [
            scaled_number(row["coverage_99_collector_overlap_ratio_total"], 100.0)
            for row in rows
        ],
    }


def compatible(metric, paper_value, stats):
    if not stats["n"]:
        return False
    if metric in ("Ttopo", "Ttotal"):
        return abs(stats["median"] - paper_value) / paper_value <= 0.20
    return abs(stats["median"] - paper_value) <= 0.05


def comparable_candidate(name):
    """Exclude machine-dependent timing from numerical reproduction claims."""
    return "wall_seconds" not in name and "cpu_seconds" not in name


def candidate_rows(episodes):
    rows = []
    for treatment in ("omnidirectional_3p5", "omnidirectional_7p0"):
        for method in METHODS:
            for metric in ("Ttopo", "Ttotal", "sigma", "r_o"):
                paper_value = PAPER[metric][method]
                for candidate, values in metric_candidates(
                    episodes, method, metric, treatment=treatment
                ).items():
                    stats = describe(values)
                    rows.append(
                        {
                            "treatment": treatment,
                            "method": method,
                            "metric": metric,
                            "paper_value": paper_value,
                            "candidate": candidate,
                            "machine_comparable": comparable_candidate(candidate),
                            "n": stats["n"],
                            "mean": stats["mean"],
                            "median": stats["median"],
                            "minimum": stats["minimum"],
                            "maximum": stats["maximum"],
                            "within_tolerance": comparable_candidate(candidate)
                            and compatible(metric, paper_value, stats),
                        }
                    )
    return rows


def verdict_rows(episodes):
    results = []
    for method in METHODS:
        for metric in ("Ttopo", "Ttotal", "sigma", "r_o"):
            paper_value = PAPER[metric][method]
            primary_candidates = metric_candidates(episodes, method, metric)
            candidate_stats = {
                name: describe(values) for name, values in primary_candidates.items()
            }
            primary_name = next(iter(candidate_stats))
            primary = candidate_stats[primary_name]
            primary_compatible = compatible(metric, paper_value, primary)
            primary_error = (
                abs(primary["median"] - paper_value) if primary["n"] else math.inf
            )

            range_candidates = metric_candidates(
                episodes, method, metric, treatment="omnidirectional_7p0"
            )
            alternatives = {
                name: stats
                for name, stats in candidate_stats.items()
                if name != primary_name
            }
            alternatives.update(
                {
                    "range_7m__" + name: describe(values)
                    for name, values in range_candidates.items()
                }
            )
            usable = [
                (name, stats)
                for name, stats in alternatives.items()
                if stats["n"] and comparable_candidate(name)
            ]
            if usable:
                best_name, best = min(
                    usable, key=lambda item: abs(item[1]["median"] - paper_value)
                )
                best_error = abs(best["median"] - paper_value)
                reduction = (
                    (primary_error - best_error) / primary_error
                    if primary_error not in (0.0, math.inf)
                    else 0.0
                )
            else:
                best_name = "no_observation"
                best = describe([])
                best_error = math.inf
                reduction = 0.0

            if primary_compatible:
                verdict = "supported"
                explanation = "primary paper interpretation is within tolerance"
            elif not usable:
                verdict = "unresolved_mismatch"
                explanation = "no run reached the required coverage boundary"
            elif compatible(metric, paper_value, best) and reduction >= 0.80:
                verdict = "explained_mismatch"
                explanation = (
                    "{} is within tolerance and reduces the primary numerical "
                    "gap by {:.1%}"
                ).format(best_name, reduction)
            else:
                verdict = "unresolved_mismatch"
                explanation = "best candidate {} leaves a gap of {:.6g}".format(
                    best_name, best_error
                )
            results.append(
                {
                    "method": method,
                    "metric": metric,
                    "paper_value": paper_value,
                    "primary_interpretation": primary_name,
                    "primary_n": primary["n"],
                    "primary_median": primary["median"],
                    "primary_minimum": primary["minimum"],
                    "primary_maximum": primary["maximum"],
                    "best_interpretation": best_name,
                    "best_median": best["median"],
                    "gap_reduction": reduction,
                    "verdict": verdict,
                    "explanation": explanation,
                }
            )
    return results


def previous_rows(input_directory):
    rows = []
    treatment_directory = input_directory / "legacy_3p5"
    for task_directory in sorted(treatment_directory.glob("task_*")):
        candidates = []
        paths = list(task_directory.glob("run*/episodes.csv"))
        paths.extend(task_directory.glob("*/run*/episodes.csv"))
        for path in paths:
            suffix = path.parent.name[3:]
            candidates.append((int(suffix) if suffix.isdigit() else -1, path))
        if candidates:
            path = max(candidates)[1]
            rows.extend(read_csv(path))
    return rows


def regression_check(episodes, regression_input):
    if regression_input is None:
        return {
            "check": "legacy_3p5_seed_1_to_5_regression",
            "status": "not_run",
            "detail": "pass --regression-input pointing to paper_consistency_01",
        }
    old = previous_rows(regression_input)
    old_lookup = {(row["method"], integer(row["seed"])): row for row in old}
    new_lookup = {
        (row["method"], integer(row["seed"])): row
        for row in episodes
        if row["treatment"] == "omnidirectional_3p5" and integer(row["seed"]) <= 5
    }
    expected = {(method, seed) for method in METHODS for seed in range(1, 6)}
    if set(old_lookup) != expected or set(new_lookup) != expected:
        return {
            "check": "legacy_3p5_seed_1_to_5_regression",
            "status": "fail",
            "detail": "the old or new control does not contain the expected ten rows",
        }
    mappings = (
        ("steps_executed", "steps_executed"),
        ("termination_reason", "termination_reason"),
        ("start_positions", "start_positions"),
        ("final_coverage_ratio", "final_team_coverage_ratio"),
        ("coverage_90_step", "coverage_90_step"),
        ("coverage_98_step", "coverage_98_step"),
        ("coverage_99_step", "coverage_99_step"),
        ("coverage_std_area_m2", "final_collector_sigma_area_m2"),
        ("overlap_ratio_total", "final_collector_overlap_ratio_total"),
    )
    failures = []
    for key in sorted(expected):
        for old_name, new_name in mappings:
            old_value = old_lookup[key].get(old_name, "")
            new_value = new_lookup[key].get(new_name, "")
            try:
                equal = math.isclose(
                    float(old_value), float(new_value), rel_tol=1e-12, abs_tol=1e-12
                )
            except (TypeError, ValueError):
                equal = str(old_value) == str(new_value)
            if not equal:
                failures.append("{} {} != {}".format(key, old_name, new_name))
    return {
        "check": "legacy_3p5_seed_1_to_5_regression",
        "status": "pass" if not failures else "fail",
        "detail": "exactly matches prior ten controls"
        if not failures
        else "; ".join(failures[:10]),
    }


def markdown_table(rows, columns):
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row.get(name, "")) for name, _ in columns)
            + " |"
        )
    return "\n".join(lines)


def write_report(
    output_directory,
    episodes,
    summary,
    pair_summary,
    claim_results,
    verdicts,
    checks,
):
    defect = any(row["status"] == "fail" for row in checks)
    incomplete = any(row["status"] == "not_run" for row in checks)
    claim_failed = len(claim_results) != len(CLAIM_METRICS) or any(
        row["gate_status"] != "pass" for row in claim_results
    )
    if defect:
        overall = "PIPELINE DEFECT"
    elif incomplete:
        overall = "INCOMPLETE — regression comparison not run"
    elif claim_failed:
        overall = "FAIL — Level-0 method-ranking claim was not reproduced"
    else:
        overall = "PASS — Level-0 method-ranking claim reproduced"

    control = [
        row for row in summary if row["treatment"] == "omnidirectional_3p5"
    ]
    text = "\n".join(
        [
            "# Level-0 paper-validation audit",
            "",
            "## Overall verdict",
            "",
            "**{}**".format(overall),
            "",
            "The paper-control lane is the released Level-0 3.5 m omnidirectional sensor. Four-beam and 7 m results are controlled extensions, not paper reproductions.",
            "",
            "The continuation gate tests the paper's qualitative Level-0 claim that potential-field/MMPF requires fewer decisions than Cost. Exact Table-II values remain a separate diagnostic and are not claimed as reproduced.",
            "",
            "## Integrity checks",
            "",
            markdown_table(checks, (("check", "Check"), ("status", "Status"), ("detail", "Detail"))),
            "",
            "## Claim-reproduction gate",
            "",
            "Each threshold passes only when all ten paired controls have identical starts and headings, both methods reach the threshold in all ten runs, at least nine pairs favor MMPF, the at-least-95% distribution-free paired-median interval is below zero, and the one-sided exact sign-test p-value is below 0.05.",
            "",
            markdown_table(
                claim_results,
                (
                    ("metric", "Metric"),
                    ("matched_start_pairs", "Matched starts"),
                    ("cost_successes", "Cost successes"),
                    ("mmpf_successes", "MMPF successes"),
                    ("cost_median", "Cost median"),
                    ("mmpf_median", "MMPF median"),
                    ("median_mmpf_minus_cost", "Median paired delta"),
                    ("median_delta_interval_low", "Interval low"),
                    ("median_delta_interval_high", "Interval high"),
                    ("pairs_favoring_mmpf", "Pairs favoring MMPF"),
                    ("one_sided_sign_test_p_value", "Sign-test p"),
                    ("gate_status", "Gate"),
                ),
            ),
            "",
            "## Paper-control summary",
            "",
            markdown_table(
                control,
                (
                    ("method", "Method"),
                    ("runs", "Runs"),
                    ("coverage_99_successes", "99% successes"),
                    ("final_team_coverage_ratio_median", "Median final coverage"),
                    ("coverage_90_step_median", "Median T90 decisions"),
                    ("coverage_99_step_median", "Median T99 decisions"),
                ),
            ),
            "",
            "## Paired four-beam extension",
            "",
            "Positive coverage deltas favor four-beam; positive T99 deltas mean four-beam required more decisions. Missing T99 pairs are excluded from the decision delta.",
            "",
            markdown_table(
                pair_summary,
                (
                    ("sensor_range", "Range (m)"),
                    ("method", "Method"),
                    ("pairs", "Pairs"),
                    ("median_delta_final_coverage", "Median coverage delta"),
                    ("complete_99_step_pairs", "Complete T99 pairs"),
                    ("median_delta_99_step", "Median T99 delta"),
                ),
            ),
            "",
            "## Exact Table-II diagnostics",
            "",
            markdown_table(
                verdicts,
                (
                    ("method", "Method"),
                    ("metric", "Metric"),
                    ("paper_value", "Paper"),
                    ("primary_median", "Primary median"),
                    ("best_interpretation", "Best interpretation"),
                    ("best_median", "Best median"),
                    ("verdict", "Verdict"),
                    ("explanation", "Evidence"),
                ),
            ),
            "",
            "## Interpretation boundary",
            "",
            "A gate pass supports only the paper's directional Level-0 method-ranking claim under the released control configuration. It does not reproduce the published Table-II cells, explain unresolved metric mismatches, validate the four-beam extension, or establish cross-level reproduction.",
            "",
        ]
    )
    (output_directory / "audit_report.md").write_text(text, encoding="utf-8")
    escaped = html.escape(text)
    html_text = "<!doctype html><html><head><meta charset='utf-8'><title>Level-0 paper audit</title><style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto;white-space:pre-wrap}}</style></head><body>{}</body></html>".format(escaped)
    (output_directory / "audit_report.html").write_text(html_text, encoding="utf-8")
    return overall


def analyze(input_directory, output_directory, regression_input=None):
    output_directory.mkdir(parents=True, exist_ok=True)
    episodes, steps, inventory = load_collection(input_directory)
    checks = validate_pairs(episodes, inventory)
    checks.append(regression_check(episodes, regression_input))
    summary = summaries(episodes)
    pairs = paired_rows(episodes)
    pair_summary = paired_summaries(pairs)
    ranking_pairs = method_ranking_pairs(episodes)
    claim_results = claim_reproduction_rows(ranking_pairs)
    verdicts = verdict_rows(episodes)
    candidates = candidate_rows(episodes)
    write_csv(output_directory / "episodes_combined.csv", episodes)
    write_csv(output_directory / "steps_combined.csv", steps)
    write_csv(output_directory / "run_inventory.csv", inventory)
    write_csv(output_directory / "summary.csv", summary)
    write_csv(output_directory / "paired_sensor_results.csv", pairs)
    write_csv(output_directory / "paired_sensor_summary.csv", pair_summary)
    write_csv(output_directory / "method_ranking_pairs.csv", ranking_pairs)
    write_csv(output_directory / "claim_reproduction.csv", claim_results)
    write_csv(output_directory / "paper_verdicts.csv", verdicts)
    write_csv(output_directory / "metric_candidates.csv", candidates)
    write_csv(output_directory / "pipeline_checks.csv", checks)
    return write_report(
        output_directory,
        episodes,
        summary,
        pair_summary,
        claim_results,
        verdicts,
        checks,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--regression-input", type=Path)
    args = parser.parse_args(argv)
    overall = analyze(
        args.input.resolve(),
        args.output.resolve(),
        args.regression_input.resolve() if args.regression_input else None,
    )
    print(overall)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
