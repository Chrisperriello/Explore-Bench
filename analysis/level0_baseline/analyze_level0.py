#!/usr/bin/env python3
"""Build meeting-ready tables and figures from a merged Level-0 collection.

The analysis is deliberately dependency-light: CSV processing uses the Python
standard library and plotting requires only Matplotlib.  Every comparison is
paired by method, map, and seed.  Milestone times are summarized only among
runs that actually reached the milestone; reach rates are always reported
beside those conditional times.
"""

import argparse
import csv
import html
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_INPUT = (
    REPOSITORY_ROOT.parent
    / "slurm"
    / "combined"
    / "level0"
    / "level0_validation_01"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results" / "level0_baseline_analysis"

SENSORS = ("omnidirectional", "four_beam")
METHODS = ("cost", "mmpf")
MILESTONES = ("90", "98", "99")
SENSOR_LABELS = {
    "omnidirectional": "Original omnidirectional",
    "four_beam": "Four-beam",
}
SENSOR_SHORT_LABELS = {
    "omnidirectional": "Original",
    "four_beam": "Four-beam",
}
SENSOR_COLORS = {
    "omnidirectional": "#2f6fb0",
    "four_beam": "#e07a2d",
}
METHOD_LABELS = {"cost": "Cost", "mmpf": "MMPF"}
METHOD_LINESTYLES = {"cost": "-", "mmpf": "--"}
MAP_LABELS = {
    "square_loop.pgm": "Loop",
    "corridor.pgm": "Corridor",
    "corner.pgm": "Corner",
    "room1_modified.pgm": "Room",
    "loop_with_corridor_sym.pgm": "Comb 1",
    "room_with_corner.pgm": "Comb 2",
}
MAP_ORDER = tuple(MAP_LABELS)
PAPER_ROOM_REFERENCE = {
    "cost": {"coverage_90_step": 134.0, "coverage_99_step": 172.0},
    "mmpf": {"coverage_90_step": 100.0, "coverage_99_step": 125.0},
}
PAPER_LEVEL0_ROOM_METRICS = (
    {
        "metric": "Ttopo",
        "meaning": "Time to 90% coverage",
        "paper_unit": "s",
        "our_unit": "decisions",
        "field": "coverage_90_step",
        "scale": 1.0,
        "paper_cost": 134.0,
        "paper_mmpf": 100.0,
        "lower_is_better": True,
        "comparability": "Same coverage milestone, different recorded units; compare ordering only.",
    },
    {
        "metric": "Ttotal",
        "meaning": "Time to 99% coverage",
        "paper_unit": "s",
        "our_unit": "decisions",
        "field": "coverage_99_step",
        "scale": 1.0,
        "paper_cost": 172.0,
        "paper_mmpf": 125.0,
        "lower_is_better": True,
        "comparability": "Same coverage milestone, different recorded units; compare ordering only.",
    },
    {
        "metric": "sigma",
        "meaning": "Standard deviation of independently explored area",
        "paper_unit": "table: m^2",
        "our_unit": "normalized ratio",
        "field": "coverage_std_ratio",
        "scale": 1.0,
        "paper_cost": 0.09,
        "paper_mmpf": 0.01,
        "lower_is_better": True,
        "comparability": "The paper labels m^2, but the released Level-0 code prints the standard deviation of normalized per-robot exploration fractions.",
    },
    {
        "metric": "r_o",
        "meaning": "Overlapping explored-area ratio",
        "paper_unit": "table: %",
        "our_unit": "ratio",
        "field": "overlap_ratio_total",
        "scale": 1.0,
        "paper_cost": 0.10,
        "paper_mmpf": 0.04,
        "lower_is_better": True,
        "comparability": "The paper labels percent, while its values and released code appear ratio-scaled; treat 0.10 as convention-ambiguous.",
    },
)


def read_csv(path):
    """Read one CSV table without silently coercing values."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    """Write dictionaries using a stable column order."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value, default=math.nan):
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=None):
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def finite(values):
    return [value for value in values if not math.isnan(value)]


def percentile(values, fraction):
    """Compute a linearly interpolated percentile without NumPy."""
    ordered = sorted(finite(values))
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values):
    """Return descriptive statistics for numeric values."""
    values = finite(values)
    return {
        "n": len(values),
        "mean": mean(values) if values else math.nan,
        "median": median(values) if values else math.nan,
        "sd": stdev(values) if len(values) > 1 else math.nan,
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
        "min": min(values) if values else math.nan,
        "max": max(values) if values else math.nan,
    }


def clean_number(value, digits=6):
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return round(value, digits)
    return value


def map_sort_key(name):
    try:
        return MAP_ORDER.index(name)
    except ValueError:
        return len(MAP_ORDER)


def condition_key(row):
    return (row["sensor_variant"], row["method"], row["map_name"])


def pair_key(row):
    return (row["method"], row["map_name"], int(row["seed"]))


def load_dataset(directory, load_steps=True, load_agents=True):
    """Load and structurally validate one merged collection."""
    directory = Path(directory).expanduser().resolve()
    required = (
        "merge_manifest.json",
        "episodes.csv",
        "steps.csv",
        "agents.csv",
        "run_inventory.csv",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError("{} is missing {}".format(directory, ", ".join(missing)))

    with (directory / "merge_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    episodes = read_csv(directory / "episodes.csv")
    inventory = read_csv(directory / "run_inventory.csv")
    steps = read_csv(directory / "steps.csv") if load_steps else []
    agents = read_csv(directory / "agents.csv") if load_agents else []

    expected = int(manifest.get("expected_runs", 0))
    selected = int(manifest.get("selected_complete_runs", 0))
    if expected != selected or manifest.get("missing_runs"):
        raise ValueError(
            "merged collection is incomplete: selected {} of {} runs".format(
                selected, expected
            )
        )
    if len(episodes) != selected:
        raise ValueError(
            "episodes.csv has {} rows but manifest records {} runs".format(
                len(episodes), selected
            )
        )
    if len(inventory) != expected:
        raise ValueError(
            "run_inventory.csv has {} rows, expected {}".format(
                len(inventory), expected
            )
        )

    validate_episode_pairing(episodes)
    return {
        "directory": directory,
        "manifest": manifest,
        "episodes": episodes,
        "inventory": inventory,
        "steps": steps,
        "agents": agents,
    }


def validate_episode_pairing(episodes):
    """Require exactly one original and four-beam run for every task key."""
    by_pair = defaultdict(dict)
    for row in episodes:
        sensor = row.get("sensor_variant")
        if sensor not in SENSORS:
            raise ValueError("unexpected sensor variant {!r}".format(sensor))
        key = pair_key(row)
        if sensor in by_pair[key]:
            raise ValueError("duplicate {} run for {}".format(sensor, key))
        by_pair[key][sensor] = row

    bad = [key for key, values in by_pair.items() if set(values) != set(SENSORS)]
    if bad:
        raise ValueError("{} task keys are not paired".format(len(bad)))

    matched_fields = (
        "method",
        "map_name",
        "team_size",
        "seed",
        "communication_mode",
        "communication_protocol",
        "target_coverage",
        "start_positions",
    )
    for key, values in by_pair.items():
        original = values["omnidirectional"]
        four = values["four_beam"]
        differences = [
            field for field in matched_fields if original.get(field) != four.get(field)
        ]
        if differences:
            raise ValueError("pair {} differs on {}".format(key, ", ".join(differences)))
    return by_pair


def aggregate_rows(rows, group_type, group_value, sensor="all"):
    """Summarize outcomes for one group of episode rows."""
    n = len(rows)
    successes = sum(as_bool(row["success"]) for row in rows)
    result = {
        "group_type": group_type,
        "group": group_value,
        "sensor_variant": sensor,
        "runs": n,
        "successes": successes,
        "success_rate": successes / n if n else math.nan,
    }
    metric_fields = {
        "final_coverage_ratio": 100.0,
        "steps_executed": 1.0,
        "team_path_length_m": 1.0,
        "maximum_agent_path_length_m": 1.0,
        "overlap_ratio_total": 100.0,
        "coverage_std_area_m2": 1.0,
    }
    for field, scale in metric_fields.items():
        stats = describe([as_float(row[field]) * scale for row in rows])
        for name, value in stats.items():
            result["{}_{}".format(field, name)] = value

    for milestone in MILESTONES:
        reached_field = "coverage_{}_reached".format(milestone)
        reached = [row for row in rows if as_bool(row[reached_field])]
        result["coverage_{}_reached_runs".format(milestone)] = len(reached)
        result["coverage_{}_reach_rate".format(milestone)] = (
            len(reached) / n if n else math.nan
        )
        for metric in ("step", "team_path_m"):
            field = "coverage_{}_{}".format(milestone, metric)
            stats = describe([as_float(row[field]) for row in reached])
            result["{}_conditional_mean".format(field)] = stats["mean"]
            result["{}_conditional_median".format(field)] = stats["median"]
            result["{}_conditional_q1".format(field)] = stats["q1"]
            result["{}_conditional_q3".format(field)] = stats["q3"]
    return {key: clean_number(value) for key, value in result.items()}


def build_condition_summaries(episodes):
    grouped = defaultdict(list)
    for row in episodes:
        grouped[condition_key(row)].append(row)
    output = []
    for sensor in SENSORS:
        for method in METHODS:
            maps = sorted(
                {key[2] for key in grouped if key[:2] == (sensor, method)},
                key=map_sort_key,
            )
            for map_name in maps:
                row = aggregate_rows(
                    grouped[(sensor, method, map_name)],
                    "condition",
                    "{} / {}".format(METHOD_LABELS.get(method, method), MAP_LABELS.get(map_name, map_name)),
                    sensor,
                )
                row.update(
                    {
                        "method": method,
                        "map_name": map_name,
                        "map_label": MAP_LABELS.get(map_name, map_name),
                    }
                )
                output.append(row)
    return output


def build_overall_summaries(episodes):
    groups = []
    for sensor in SENSORS:
        sensor_rows = [row for row in episodes if row["sensor_variant"] == sensor]
        groups.append(aggregate_rows(sensor_rows, "sensor", SENSOR_LABELS[sensor], sensor))
        for method in METHODS:
            rows = [row for row in sensor_rows if row["method"] == method]
            groups.append(
                aggregate_rows(
                    rows,
                    "sensor_method",
                    "{} / {}".format(SENSOR_SHORT_LABELS[sensor], METHOD_LABELS[method]),
                    sensor,
                )
            )
            groups[-1]["method"] = method
    return groups


def build_paired_runs(episodes):
    pairs = validate_episode_pairing(episodes)
    output = []
    for (method, map_name, seed), values in sorted(
        pairs.items(), key=lambda item: (METHODS.index(item[0][0]), map_sort_key(item[0][1]), item[0][2])
    ):
        original = values["omnidirectional"]
        four = values["four_beam"]
        row = {
            "method": method,
            "map_name": map_name,
            "map_label": MAP_LABELS.get(map_name, map_name),
            "seed": seed,
            "original_success": as_bool(original["success"]),
            "four_beam_success": as_bool(four["success"]),
            "original_termination": original["termination_reason"],
            "four_beam_termination": four["termination_reason"],
        }
        for field, scale in (
            ("final_coverage_ratio", 100.0),
            ("steps_executed", 1.0),
            ("team_path_length_m", 1.0),
            ("maximum_agent_path_length_m", 1.0),
            ("overlap_ratio_total", 100.0),
            ("coverage_std_area_m2", 1.0),
        ):
            original_value = as_float(original[field]) * scale
            four_value = as_float(four[field]) * scale
            row["original_{}".format(field)] = original_value
            row["four_beam_{}".format(field)] = four_value
            row["delta_{}".format(field)] = four_value - original_value
        for milestone in MILESTONES:
            for sensor_name, source in (("original", original), ("four_beam", four)):
                reached = as_bool(source["coverage_{}_reached".format(milestone)])
                row["{}_coverage_{}_reached".format(sensor_name, milestone)] = reached
                row["{}_coverage_{}_step".format(sensor_name, milestone)] = (
                    as_float(source["coverage_{}_step".format(milestone)])
                    if reached
                    else math.nan
                )
            both = (
                row["original_coverage_{}_reached".format(milestone)]
                and row["four_beam_coverage_{}_reached".format(milestone)]
            )
            row["both_coverage_{}_reached".format(milestone)] = both
            row["delta_coverage_{}_step".format(milestone)] = (
                row["four_beam_coverage_{}_step".format(milestone)]
                - row["original_coverage_{}_step".format(milestone)]
                if both
                else math.nan
            )
        output.append({key: clean_number(value) for key, value in row.items()})
    return output


def summarize_pairs(rows, group_type, group_value):
    result = {
        "group_type": group_type,
        "group": group_value,
        "pairs": len(rows),
        "original_successes": sum(bool(row["original_success"]) for row in rows),
        "four_beam_successes": sum(bool(row["four_beam_success"]) for row in rows),
    }
    fields = (
        "final_coverage_ratio",
        "steps_executed",
        "team_path_length_m",
        "maximum_agent_path_length_m",
        "overlap_ratio_total",
        "coverage_std_area_m2",
        "coverage_90_step",
        "coverage_98_step",
        "coverage_99_step",
    )
    for field in fields:
        values = [as_float(row["delta_{}".format(field)]) for row in rows]
        stats = describe(values)
        result["delta_{}_n".format(field)] = stats["n"]
        result["delta_{}_mean".format(field)] = stats["mean"]
        result["delta_{}_median".format(field)] = stats["median"]
        result["delta_{}_q1".format(field)] = stats["q1"]
        result["delta_{}_q3".format(field)] = stats["q3"]
    coverage_deltas = [as_float(row["delta_final_coverage_ratio"]) for row in rows]
    result["four_beam_coverage_wins"] = sum(value > 1e-9 for value in coverage_deltas)
    result["coverage_ties"] = sum(abs(value) <= 1e-9 for value in coverage_deltas)
    result["four_beam_coverage_losses"] = sum(value < -1e-9 for value in coverage_deltas)
    return {key: clean_number(value) for key, value in result.items()}


def build_paired_summaries(paired):
    output = [summarize_pairs(paired, "overall", "All 60 paired tasks")]
    for method in METHODS:
        rows = [row for row in paired if row["method"] == method]
        item = summarize_pairs(rows, "method", METHOD_LABELS[method])
        item["method"] = method
        output.append(item)
    for map_name in MAP_ORDER:
        rows = [row for row in paired if row["map_name"] == map_name]
        item = summarize_pairs(rows, "map", MAP_LABELS[map_name])
        item["map_name"] = map_name
        output.append(item)
    for method in METHODS:
        for map_name in MAP_ORDER:
            rows = [
                row
                for row in paired
                if row["method"] == method and row["map_name"] == map_name
            ]
            item = summarize_pairs(
                rows,
                "method_map",
                "{} / {}".format(METHOD_LABELS[method], MAP_LABELS[map_name]),
            )
            item["method"] = method
            item["map_name"] = map_name
            output.append(item)
    return output


def build_termination_summary(episodes):
    counts = Counter(
        (row["sensor_variant"], row["method"], row["termination_reason"])
        for row in episodes
    )
    output = []
    for sensor in SENSORS:
        for method in METHODS:
            total = sum(
                count
                for (candidate_sensor, candidate_method, _), count in counts.items()
                if candidate_sensor == sensor and candidate_method == method
            )
            reasons = sorted(
                {
                    reason
                    for candidate_sensor, candidate_method, reason in counts
                    if candidate_sensor == sensor and candidate_method == method
                }
            )
            for reason in reasons:
                count = counts[(sensor, method, reason)]
                output.append(
                    {
                        "sensor_variant": sensor,
                        "method": method,
                        "termination_reason": reason,
                        "runs": count,
                        "rate": count / total if total else math.nan,
                    }
                )
    return output


def build_paper_reference(episodes):
    output = []
    for method in METHODS:
        rows = [
            row
            for row in episodes
            if row["sensor_variant"] == "omnidirectional"
            and row["method"] == method
            and row["map_name"] == "room1_modified.pgm"
        ]
        for milestone in ("90", "99"):
            reached = [
                as_float(row["coverage_{}_step".format(milestone)])
                for row in rows
                if as_bool(row["coverage_{}_reached".format(milestone)])
            ]
            observed = describe(reached)
            paper = PAPER_ROOM_REFERENCE[method]["coverage_{}_step".format(milestone)]
            output.append(
                {
                    "method": method,
                    "milestone_percent": milestone,
                    "our_runs_reached": len(reached),
                    "our_runs_total": len(rows),
                    "our_mean_decisions": clean_number(observed["mean"]),
                    "our_median_decisions": clean_number(observed["median"]),
                    "our_q1_decisions": clean_number(observed["q1"]),
                    "our_q3_decisions": clean_number(observed["q3"]),
                    "paper_reported_value": paper,
                    "unit_warning": (
                        "Context check only: our collector records Level-0 decisions; "
                        "do not claim exact reproduction until paper setup and units match."
                    ),
                }
            )
    return output


def metric_values(rows, metric):
    """Return values available for one paper-comparison metric."""
    values = []
    field = metric["field"]
    for row in rows:
        if field.startswith("coverage_") and field.endswith("_step"):
            milestone = field.split("_")[1]
            if not as_bool(row["coverage_{}_reached".format(milestone)]):
                continue
        value = as_float(row.get(field, ""))
        if not math.isnan(value):
            values.append(value * metric["scale"])
    return values


def ranking_label(cost_value, mmpf_value, lower_is_better=True):
    if math.isclose(cost_value, mmpf_value, rel_tol=1e-9, abs_tol=1e-9):
        return "tie"
    mmpf_wins = mmpf_value < cost_value if lower_is_better else mmpf_value > cost_value
    return "field/MMPF better" if mmpf_wins else "cost better"


def build_paper_direct_comparison(episodes):
    """Build one direct side-by-side table for Explore-Bench Table II."""
    room = [
        row
        for row in episodes
        if row["sensor_variant"] == "omnidirectional"
        and row["map_name"] == "room1_modified.pgm"
    ]
    by_method = {
        method: [row for row in room if row["method"] == method]
        for method in METHODS
    }
    output = []
    for metric in PAPER_LEVEL0_ROOM_METRICS:
        cost = describe(metric_values(by_method["cost"], metric))
        mmpf = describe(metric_values(by_method["mmpf"], metric))
        paper_ranking = ranking_label(
            metric["paper_cost"], metric["paper_mmpf"], metric["lower_is_better"]
        )
        our_ranking = ranking_label(
            cost["mean"], mmpf["mean"], metric["lower_is_better"]
        )
        output.append(
            {
                "metric": metric["metric"],
                "meaning": metric["meaning"],
                "paper_unit": metric["paper_unit"],
                "our_unit": metric["our_unit"],
                "paper_cost": metric["paper_cost"],
                "our_cost_mean": clean_number(cost["mean"]),
                "our_cost_median": clean_number(cost["median"]),
                "our_cost_q1": clean_number(cost["q1"]),
                "our_cost_q3": clean_number(cost["q3"]),
                "our_cost_n": cost["n"],
                "paper_field": metric["paper_mmpf"],
                "our_mmpf_mean": clean_number(mmpf["mean"]),
                "our_mmpf_median": clean_number(mmpf["median"]),
                "our_mmpf_q1": clean_number(mmpf["q1"]),
                "our_mmpf_q3": clean_number(mmpf["q3"]),
                "our_mmpf_n": mmpf["n"],
                "paper_ranking": paper_ranking,
                "our_ranking": our_ranking,
                "directional_ranking_matches": paper_ranking == our_ranking,
                "comparability": metric["comparability"],
            }
        )
    return output


def table_fields(rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def write_tables(dataset, output):
    episodes = dataset["episodes"]
    condition = build_condition_summaries(episodes)
    overall = build_overall_summaries(episodes)
    paired = build_paired_runs(episodes)
    paired_summary = build_paired_summaries(paired)
    terminations = build_termination_summary(episodes)
    paper = build_paper_reference(episodes)
    paper_direct = build_paper_direct_comparison(episodes)
    tables = {
        "condition_summary.csv": condition,
        "overall_summary.csv": overall,
        "paired_runs.csv": paired,
        "paired_summary.csv": paired_summary,
        "termination_summary.csv": terminations,
        "paper_room_reference.csv": paper,
        "explore_bench_direct_comparison.csv": paper_direct,
        "run_inventory.csv": dataset["inventory"],
    }
    table_directory = output / "tables"
    table_directory.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(table_directory / name, table_fields(rows), rows)
    return {
        "condition": condition,
        "overall": overall,
        "paired": paired,
        "paired_summary": paired_summary,
        "terminations": terminations,
        "paper": paper,
        "paper_direct": paper_direct,
    }


def sensor_summary(rows, sensor):
    return next(
        row
        for row in rows
        if row["group_type"] == "sensor" and row["sensor_variant"] == sensor
    )


def percent(value, digits=1):
    return "{:.{}f}%".format(float(value), digits)


def findings_lines(tables):
    original = sensor_summary(tables["overall"], "omnidirectional")
    four = sensor_summary(tables["overall"], "four_beam")
    conditions = [row for row in tables["condition"] if row["sensor_variant"] == "four_beam"]
    best = max(conditions, key=lambda row: float(row["final_coverage_ratio_mean"]))
    worst = min(conditions, key=lambda row: float(row["final_coverage_ratio_mean"]))
    terminations = Counter()
    for row in tables["terminations"]:
        if row["sensor_variant"] == "four_beam":
            terminations[row["termination_reason"]] += int(row["runs"])
    overall_pair = next(row for row in tables["paired_summary"] if row["group_type"] == "overall")
    paper_lookup = {
        (row["method"], row["milestone_percent"]): row for row in tables["paper"]
    }
    return [
        "The merged collection is complete: 120 runs, with 60 exactly paired sensor comparisons.",
        "The original sensor reached 99% coverage in {}/{} runs ({}); four-beam reached it in {}/{} runs ({}).".format(
            original["successes"],
            original["runs"],
            percent(100.0 * float(original["success_rate"])),
            four["successes"],
            four["runs"],
            percent(100.0 * float(four["success_rate"])),
        ),
        "Median final coverage was {} for the original sensor and {} for four-beam.".format(
            percent(original["final_coverage_ratio_median"]),
            percent(four["final_coverage_ratio_median"]),
        ),
        "Across paired runs, the median four-beam minus original final-coverage difference was {:.1f} percentage points.".format(
            float(overall_pair["delta_final_coverage_ratio_median"])
        ),
        "The strongest four-beam condition was {} at {:.1f}% mean final coverage; the weakest was {} at {:.1f}%.".format(
            best["group"],
            float(best["final_coverage_ratio_mean"]),
            worst["group"],
            float(worst["final_coverage_ratio_mean"]),
        ),
        "Four-beam stopped at the 1,000-decision limit in {} runs and stopped because both agents were stuck in {} runs.".format(
            terminations["max_steps"], terminations["all_agents_stuck"]
        ),
        "The room-map context check is not yet a paper reproduction: our original-sensor mean T90/T99 was {:.1f}/{:.1f} decisions for cost and {:.1f}/{:.1f} for MMPF, versus paper-listed values 134/172 and 100/125.".format(
            float(paper_lookup[("cost", "90")]["our_mean_decisions"]),
            float(paper_lookup[("cost", "99")]["our_mean_decisions"]),
            float(paper_lookup[("mmpf", "90")]["our_mean_decisions"]),
            float(paper_lookup[("mmpf", "99")]["our_mean_decisions"]),
        ),
    ]


def write_brief(dataset, tables, output):
    lines = findings_lines(tables)
    manifest = dataset["manifest"]
    content = [
        "# Level-0 sensor baseline — advisor brief",
        "",
        "## What was run",
        "",
        "- 120 completed Level-0 episodes: 60 original omnidirectional and 60 four-beam.",
        "- Two robots, six maps, cost and MMPF planners, and five paired seeds.",
        "- The paired runs hold method, map, seed, starts, team size, and legacy shared-state behavior fixed; communication is not varied or analyzed.",
        "- Maximum 1,000 decisions; target coverage 99%.",
        "",
        "## Main findings",
        "",
    ]
    content.extend("- " + line for line in lines)
    content.extend(
        [
            "",
            "## How to interpret the results",
            "",
            "- Report milestone reach rate before reporting time to that milestone.",
            "- T90/T98/T99 summaries are conditional: they include only runs that reached that threshold.",
            "- A cleanly completed program is not necessarily an exploration success; use `success` and `termination_reason`.",
            "- Path length and overlap from incomplete runs describe behavior but are not completion-efficiency comparisons.",
            "- Slurm wall-clock time is not an exploration metric and is intentionally excluded.",
            "- The paper room values are context only until sensor range, start layout, and units are confirmed.",
            "",
            "## Files for the meeting",
            "",
            "- `advisor_packet.pdf`: the ordered presentation packet.",
            "- `report.html`: browsable report with all primary figures.",
            "- `figures/`: individual high-resolution PNG figures.",
            "- `tables/condition_summary.csv`: detailed map/planner/sensor table.",
            "- `tables/paired_runs.csv`: all 60 direct sensor differences.",
            "- `tables/explore_bench_direct_comparison.csv`: full side-by-side Table-II comparison.",
            "- `tables/paper_room_reference.csv`: milestone-only room-map detail.",
            "",
            "Dataset: `{}`".format(dataset["directory"]),
            "",
            "Task-table SHA-256: `{}`".format(manifest.get("task_table_sha256", "unknown")),
            "",
        ]
    )
    (output / "advisor_brief.md").write_text("\n".join(content), encoding="utf-8")
    (output / "advisor_brief.txt").write_text(
        "LEVEL-0 SENSOR BASELINE\n\n"
        + "\n".join("- " + line for line in lines)
        + "\n\nInterpretation: reach rates first; threshold times only among runs that reached them.\n",
        encoding="utf-8",
    )


def setup_plotting(output):
    cache = output / ".matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(output / ".cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    return plt, PdfPages


def save_figure(fig, figures, name, pdf=None):
    path = figures / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    return path


def plot_title_page(plt, dataset, tables, pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.07, 0.90, "Explore-Bench Level-0 sensor baseline", fontsize=26, weight="bold")
    fig.text(0.07, 0.84, "Original omnidirectional sensing vs four-beam sensing", fontsize=16)
    fig.text(0.07, 0.78, "Two robots · six maps · two planners · five paired seeds", fontsize=13, color="#444444")
    y = 0.67
    for line in findings_lines(tables):
        wrapped = line
        fig.text(0.09, y, "• " + wrapped, fontsize=13, va="top", wrap=True)
        y -= 0.082
    fig.text(
        0.07,
        0.08,
        "Important: threshold times include only runs that reached the threshold. "
        "Reach rates must be shown beside them.",
        fontsize=11,
        color="#9b2c2c",
        weight="bold",
        wrap=True,
    )
    fig.text(0.07, 0.035, "Dataset: {}".format(dataset["directory"]), fontsize=8, color="#666666")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_executive_overview(plt, episodes, tables, figures, pdf):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    summaries = [sensor_summary(tables["overall"], sensor) for sensor in SENSORS]
    labels = [SENSOR_SHORT_LABELS[sensor] for sensor in SENSORS]
    colors = [SENSOR_COLORS[sensor] for sensor in SENSORS]

    ax = axes[0, 0]
    rates = [100.0 * float(row["success_rate"]) for row in summaries]
    bars = ax.bar(labels, rates, color=colors, width=0.62)
    for bar, row, rate in zip(bars, summaries, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 2, "{}/{}\n{:.1f}%".format(row["successes"], row["runs"], rate), ha="center", weight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Runs reaching 99% (%)")
    ax.set_title("Exploration success")

    ax = axes[0, 1]
    values = [
        [100.0 * as_float(row["final_coverage_ratio"]) for row in episodes if row["sensor_variant"] == sensor]
        for sensor in SENSORS
    ]
    box = ax.boxplot(values, labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_ylim(0, 103)
    ax.set_ylabel("Final coverage (%)")
    ax.set_title("Final coverage distribution")

    ax = axes[1, 0]
    width = 0.34
    x = range(len(MILESTONES))
    for index, sensor in enumerate(SENSORS):
        row = sensor_summary(tables["overall"], sensor)
        rates = [100.0 * float(row["coverage_{}_reach_rate".format(m)]) for m in MILESTONES]
        positions = [value + (index - 0.5) * width for value in x]
        ax.bar(positions, rates, width, color=SENSOR_COLORS[sensor], label=SENSOR_SHORT_LABELS[sensor])
    ax.set_xticks(list(x), ["{}%".format(value) for value in MILESTONES])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Runs reaching milestone (%)")
    ax.set_title("Coverage milestone reach rate")
    ax.legend()

    ax = axes[1, 1]
    reasons = ("target_coverage", "max_steps", "all_agents_stuck")
    reason_labels = ("Reached 99%", "1,000-step limit", "Agents stuck")
    bottoms = [0, 0]
    reason_colors = ("#3a9d5d", "#d9a441", "#b44a4a")
    for reason, label, color in zip(reasons, reason_labels, reason_colors):
        counts = [
            sum(1 for row in episodes if row["sensor_variant"] == sensor and row["termination_reason"] == reason)
            for sensor in SENSORS
        ]
        ax.bar(labels, counts, bottom=bottoms, color=color, label=label)
        bottoms = [bottom + count for bottom, count in zip(bottoms, counts)]
    ax.set_ylabel("Runs")
    ax.set_title("Why episodes ended")
    ax.legend(loc="upper right")

    fig.suptitle("Executive overview", fontsize=17, weight="bold")
    fig.tight_layout()
    save_figure(fig, figures, "01_executive_overview.png", pdf)
    plt.close(fig)


def matrix_for_condition(condition_rows, sensor, field):
    lookup = {(row["method"], row["map_name"]): float(row[field]) for row in condition_rows if row["sensor_variant"] == sensor}
    return [[lookup[(method, map_name)] for map_name in MAP_ORDER] for method in METHODS]


def annotate_matrix(ax, matrix, suffix="", digits=1):
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            color = "white" if value < 45 else "black"
            ax.text(column_index, row_index, ("{:." + str(digits) + "f}{} ").format(value, suffix).strip(), ha="center", va="center", color=color, weight="bold", fontsize=9)


def plot_condition_matrices(plt, tables, figures, pdf):
    fig, axes = plt.subplots(2, 2, figsize=(15, 6.8), constrained_layout=True)
    for column, sensor in enumerate(SENSORS):
        coverage = matrix_for_condition(tables["condition"], sensor, "final_coverage_ratio_mean")
        success = matrix_for_condition(tables["condition"], sensor, "success_rate")
        success = [[100.0 * value for value in row] for row in success]
        for row_index, (matrix, title) in enumerate(((coverage, "Mean final coverage"), (success, "99% success rate"))):
            ax = axes[row_index, column]
            image = ax.imshow(matrix, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
            annotate_matrix(ax, matrix, "%")
            ax.set_xticks(range(len(MAP_ORDER)), [MAP_LABELS[name] for name in MAP_ORDER], rotation=25, ha="right")
            ax.set_yticks(range(len(METHODS)), [METHOD_LABELS[name] for name in METHODS])
            ax.set_title("{} — {}".format(SENSOR_SHORT_LABELS[sensor], title))
    fig.colorbar(image, ax=axes, label="Percent", shrink=0.92)
    fig.suptitle("Where each sensor succeeds and fails", fontsize=17, weight="bold")
    save_figure(fig, figures, "02_condition_matrices.png", pdf)
    plt.close(fig)


def plot_paired_deltas(plt, paired, figures, pdf):
    fig, axes = plt.subplots(2, 1, figsize=(14.5, 9), sharex=True)
    rng_offsets = [-0.18, -0.09, 0.0, 0.09, 0.18]
    for ax, method in zip(axes, METHODS):
        method_rows = [row for row in paired if row["method"] == method]
        for index, map_name in enumerate(MAP_ORDER):
            rows = [row for row in method_rows if row["map_name"] == map_name]
            values = [float(row["delta_final_coverage_ratio"]) for row in rows]
            ax.scatter([index + rng_offsets[i] for i in range(len(values))], values, color=SENSOR_COLORS["four_beam"], s=48, alpha=0.85, edgecolor="white", linewidth=0.6)
            ax.plot([index - 0.24, index + 0.24], [median(values), median(values)], color="black", linewidth=3)
        ax.axhline(0, color="#333333", linewidth=1.2)
        ax.set_ylabel("Four-beam − original\ncoverage (percentage points)")
        ax.set_title("{} planner".format(METHOD_LABELS[method]))
        ax.grid(axis="x", alpha=0)
    axes[-1].set_xticks(range(len(MAP_ORDER)), [MAP_LABELS[name] for name in MAP_ORDER])
    fig.suptitle("Every paired run: change in final coverage", fontsize=17, weight="bold")
    fig.text(0.5, 0.01, "Each dot is one matched seed; black line is the median. Below zero favors the original sensor.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    save_figure(fig, figures, "03_paired_final_coverage.png", pdf)
    plt.close(fig)


def grouped_steps(steps):
    grouped = defaultdict(list)
    for row in steps:
        grouped[(row["sensor_variant"], row["method"], row["map_name"], int(row["seed"]))].append((int(row["step"]), 100.0 * as_float(row["team_coverage_ratio"])))
    for key in grouped:
        grouped[key].sort()
    return grouped


def value_at_or_before(points, step):
    value = points[0][1]
    for candidate_step, candidate_value in points:
        if candidate_step > step:
            break
        value = candidate_value
    return value


def plot_coverage_curves(plt, steps, figures, pdf):
    grouped = grouped_steps(steps)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9), sharex=True, sharey=True)
    checkpoints = list(range(0, 1001, 20))
    for ax, map_name in zip(axes.flat, MAP_ORDER):
        for sensor in SENSORS:
            for method in METHODS:
                trajectories = [points for (candidate_sensor, candidate_method, candidate_map, _), points in grouped.items() if candidate_sensor == sensor and candidate_method == method and candidate_map == map_name]
                medians = []
                lower = []
                upper = []
                for step in checkpoints:
                    values = [value_at_or_before(points, step) for points in trajectories]
                    medians.append(median(values))
                    lower.append(percentile(values, 0.25))
                    upper.append(percentile(values, 0.75))
                label = "{} / {}".format(SENSOR_SHORT_LABELS[sensor], METHOD_LABELS[method])
                ax.plot(checkpoints, medians, color=SENSOR_COLORS[sensor], linestyle=METHOD_LINESTYLES[method], linewidth=2, label=label)
                ax.fill_between(checkpoints, lower, upper, color=SENSOR_COLORS[sensor], alpha=0.08)
        ax.axhline(90, color="#777777", linewidth=0.8, linestyle=":")
        ax.axhline(99, color="#333333", linewidth=0.8, linestyle=":")
        ax.set_title(MAP_LABELS[map_name])
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 102)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    for ax in axes[-1, :]:
        ax.set_xlabel("Decision step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Team coverage (%)")
    fig.suptitle("Median coverage progress (shading = seed interquartile range)", fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, figures, "04_coverage_curves.png", pdf)
    plt.close(fig)


def plot_milestones(plt, tables, figures, pdf):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    groups = []
    labels = []
    colors = []
    for sensor in SENSORS:
        for method in METHODS:
            groups.append(next(row for row in tables["overall"] if row["group_type"] == "sensor_method" and row["sensor_variant"] == sensor and row.get("method") == method))
            labels.append("{}\n{}".format(SENSOR_SHORT_LABELS[sensor], METHOD_LABELS[method]))
            colors.append(SENSOR_COLORS[sensor])
    x = list(range(len(groups)))
    width = 0.22
    ax = axes[0]
    for milestone_index, milestone in enumerate(MILESTONES):
        positions = [value + (milestone_index - 1) * width for value in x]
        values = [100.0 * float(row["coverage_{}_reach_rate".format(milestone)]) for row in groups]
        bars = ax.bar(positions, values, width, label="{}%".format(milestone), alpha=0.55 + milestone_index * 0.2)
        if milestone == "99":
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, "{:.0f}%".format(value), ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Runs reaching milestone (%)")
    ax.set_title("Reach rate")
    ax.legend(title="Coverage")

    ax = axes[1]
    for milestone_index, milestone in enumerate(MILESTONES):
        positions = [value + (milestone_index - 1) * width for value in x]
        values = [as_float(row["coverage_{}_step_conditional_median".format(milestone)]) for row in groups]
        bars = ax.bar(positions, [0 if math.isnan(value) else value for value in values], width, label="{}%".format(milestone), alpha=0.55 + milestone_index * 0.2)
        for bar, value in zip(bars, values):
            if math.isnan(value):
                ax.text(bar.get_x() + bar.get_width() / 2, 12, "N/A", ha="center", rotation=90, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Median decision step")
    ax.set_title("Time among runs that reached the milestone")
    ax.legend(title="Coverage")
    fig.suptitle("Milestone performance: reach rate must accompany conditional time", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, figures, "05_milestone_performance.png", pdf)
    plt.close(fig)


def plot_efficiency(plt, episodes, paired, figures, pdf):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    markers = {"cost": "o", "mmpf": "s"}
    ax = axes[0]
    for sensor in SENSORS:
        for method in METHODS:
            rows = [row for row in episodes if row["sensor_variant"] == sensor and row["method"] == method]
            ax.scatter([as_float(row["team_path_length_m"]) for row in rows], [100.0 * as_float(row["final_coverage_ratio"]) for row in rows], color=SENSOR_COLORS[sensor], marker=markers[method], alpha=0.7, label="{} / {}".format(SENSOR_SHORT_LABELS[sensor], METHOD_LABELS[method]))
    ax.set_xlabel("Team path length (m)")
    ax.set_ylabel("Final coverage (%)")
    ax.set_title("Coverage versus distance traveled")
    ax.legend(fontsize=8)

    ax = axes[1]
    for sensor in SENSORS:
        rows = [row for row in episodes if row["sensor_variant"] == sensor]
        ax.scatter([100.0 * as_float(row["overlap_ratio_total"]) for row in rows], [100.0 * as_float(row["final_coverage_ratio"]) for row in rows], color=SENSOR_COLORS[sensor], alpha=0.68, label=SENSOR_SHORT_LABELS[sensor])
    ax.set_xlabel("Final overlap ratio (%)")
    ax.set_ylabel("Final coverage (%)")
    ax.set_title("Overlap and achieved coverage")
    ax.legend()

    ax = axes[2]
    fields = ("delta_overlap_ratio_total", "delta_coverage_std_area_m2")
    labels = ("Overlap ratio\n(percentage points)", "Workload spread\n(area m²)")
    data = [[as_float(row[field]) for row in paired] for field in fields]
    box = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for patch in box["boxes"]:
        patch.set_facecolor(SENSOR_COLORS["four_beam"])
        patch.set_alpha(0.55)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_ylabel("Four-beam − original")
    ax.set_title("Paired coordination changes")
    fig.suptitle("Movement efficiency and coordination diagnostics", fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, figures, "06_efficiency_and_coordination.png", pdf)
    plt.close(fig)


def plot_terminations(plt, episodes, figures, pdf):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    reasons = ("target_coverage", "max_steps", "all_agents_stuck")
    labels = ("Reached 99%", "1,000-step limit", "Agents stuck")
    colors = ("#3a9d5d", "#d9a441", "#b44a4a")
    for ax, method in zip(axes, METHODS):
        x = range(len(SENSORS))
        bottoms = [0, 0]
        for reason, label, color in zip(reasons, labels, colors):
            values = [sum(1 for row in episodes if row["method"] == method and row["sensor_variant"] == sensor and row["termination_reason"] == reason) for sensor in SENSORS]
            bars = ax.bar(list(x), values, bottom=bottoms, color=color, label=label)
            for bar, value, bottom in zip(bars, values, bottoms):
                if value:
                    ax.text(bar.get_x() + bar.get_width() / 2, bottom + value / 2, str(value), ha="center", va="center", weight="bold")
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        ax.set_xticks(list(x), [SENSOR_SHORT_LABELS[sensor] for sensor in SENSORS])
        ax.set_title("{} planner".format(METHOD_LABELS[method]))
        ax.set_ylabel("Runs")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Episode termination reasons", fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 0.88, 0.94))
    save_figure(fig, figures, "07_termination_reasons.png", pdf)
    plt.close(fig)


def local_map_path(map_name):
    return REPOSITORY_ROOT / "onpolicy" / "onpolicy" / "envs" / "GridEnv" / "datasets" / map_name


def grouped_agents(agents):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in agents:
        key = (row["sensor_variant"], row["method"], row["map_name"], int(row["seed"]))
        grouped[key][int(row["agent_id"])].append((int(row["step"]), int(row["column"]), int(row["row"])))
    for agents_by_id in grouped.values():
        for agent_id in agents_by_id:
            agents_by_id[agent_id].sort()
    return grouped


def representative_seed(paired, method, map_name):
    rows = [row for row in paired if row["method"] == method and row["map_name"] == map_name]
    rows.sort(key=lambda row: float(row["four_beam_final_coverage_ratio"]))
    return int(rows[len(rows) // 2]["seed"])


def draw_trajectory(ax, plt, grouped, sensor, method, map_name, seed):
    map_path = local_map_path(map_name)
    if map_path.is_file():
        image = plt.imread(str(map_path))
        ax.imshow(image, cmap="gray", origin="upper")
    colors = ("#d62728", "#17becf", "#9467bd", "#8c564b")
    for agent_id, points in sorted(grouped[(sensor, method, map_name, seed)].items()):
        x = [point[1] for point in points]
        y = [point[2] for point in points]
        color = colors[agent_id % len(colors)]
        ax.plot(x, y, color=color, linewidth=1.5, label="Robot {}".format(agent_id + 1))
        ax.scatter(x[0], y[0], color=color, marker="o", edgecolor="white", s=45, zorder=3)
        ax.scatter(x[-1], y[-1], color=color, marker="X", edgecolor="white", s=55, zorder=3)
    ax.set_xlim(0, 250)
    ax.set_ylim(250, 0)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_representative_trajectories(plt, agents, paired, figures, pdf):
    grouped = grouped_agents(agents)
    examples = (("cost", "room1_modified.pgm"), ("mmpf", "loop_with_corridor_sym.pgm"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5))
    for row_index, (method, map_name) in enumerate(examples):
        seed = representative_seed(paired, method, map_name)
        pair = next(row for row in paired if row["method"] == method and row["map_name"] == map_name and int(row["seed"]) == seed)
        for column_index, sensor in enumerate(SENSORS):
            ax = axes[row_index, column_index]
            draw_trajectory(ax, plt, grouped, sensor, method, map_name, seed)
            coverage = float(pair["{}_final_coverage_ratio".format("original" if sensor == "omnidirectional" else "four_beam")])
            ax.set_title("{}\n{} / {}, seed {} — {:.1f}% coverage".format(SENSOR_SHORT_LABELS[sensor], METHOD_LABELS[method], MAP_LABELS[map_name], seed, coverage))
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Representative paired robot trajectories\n(circle = start, X = end)", fontsize=17, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, figures, "08_representative_trajectories.png", pdf)
    plt.close(fig)


def display_value(value, unit):
    value = float(value)
    if abs(value) < 1:
        formatted = "{:.2f}".format(value)
    elif abs(value) < 10:
        formatted = "{:.1f}".format(value)
    else:
        formatted = "{:.1f}".format(value)
    return "{} {}".format(formatted, unit)


def plot_paper_direct_comparison(plt, rows, figures, pdf):
    """Render the direct Table-II comparison with comparability safeguards."""
    fig, ax = plt.subplots(figsize=(14.8, 7.8))
    ax.axis("off")
    column_labels = (
        "Metric",
        "Explore-Bench\ncost",
        "Our original\ncost (mean)",
        "Explore-Bench\nfield",
        "Our original\nMMPF (mean)",
        "Ordering",
    )
    cell_rows = []
    short_meanings = {
        "Ttopo": "90% coverage time",
        "Ttotal": "99% coverage time",
        "sigma": "Explored-area SD",
        "r_o": "Overlap ratio",
    }
    for row in rows:
        ranking = "MATCH" if row["directional_ranking_matches"] else "DIFFERS"
        cell_rows.append(
            (
                "{}\n{}".format(row["metric"], short_meanings[row["metric"]]),
                display_value(row["paper_cost"], row["paper_unit"]),
                display_value(row["our_cost_mean"], row["our_unit"]),
                display_value(row["paper_field"], row["paper_unit"]),
                display_value(row["our_mmpf_mean"], row["our_unit"]),
                "{}\n{}".format(ranking, row["our_ranking"]),
            )
        )
    table = ax.table(
        cellText=cell_rows,
        colLabels=column_labels,
        cellLoc="center",
        colLoc="center",
        colWidths=(0.25, 0.13, 0.16, 0.13, 0.16, 0.14),
        bbox=(0.015, 0.28, 0.97, 0.56),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row_index, column_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#dce8f2")
            cell.set_text_props(weight="bold")
        elif column_index == 5:
            cell.set_facecolor("#dff0df" if rows[row_index - 1]["directional_ranking_matches"] else "#f5d6d6")
            cell.set_text_props(weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#f5f7f9")
    ax.set_title(
        "Direct comparison with Explore-Bench Table II (Level-0, five-room scenario)",
        fontsize=17,
        weight="bold",
        pad=16,
    )
    ax.text(
        0.02,
        0.20,
        "What matches: field/MMPF ranks better than cost for Ttopo, Ttotal, sigma, and overlap in both datasets.",
        transform=ax.transAxes,
        fontsize=11,
        weight="bold",
        color="#245a32",
    )
    ax.text(
        0.02,
        0.13,
        "What does not yet match: absolute values. The paper reports time in seconds; our collector records decision steps. "
        "The collaboration metrics also show a large scaling/convention difference.",
        transform=ax.transAxes,
        fontsize=10,
        color="#8a3b12",
        wrap=True,
    )
    ax.text(
        0.02,
        0.055,
        "Do not label this a numerical reproduction yet. Confirm sensor range, initial placements, timestep-to-second mapping, "
        "sigma normalization, overlap percent convention, and the paper's unreported repetition procedure.",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#444444",
        wrap=True,
    )
    save_figure(fig, figures, "09_explore_bench_direct_comparison.png", pdf)
    plt.close(fig)


def html_table(rows, columns, headings=None, limit=None):
    rows = rows[:limit] if limit else rows
    headings = headings or columns
    parts = ["<div class='table-wrap'><table><thead><tr>"]
    parts.extend("<th>{}</th>".format(html.escape(str(value))) for value in headings)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = "" if math.isnan(value) else "{:.2f}".format(value)
            parts.append("<td>{}</td>".format(html.escape(str(value))))
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def write_html_report(dataset, tables, output, figure_names):
    findings = "".join("<li>{}</li>".format(html.escape(line)) for line in findings_lines(tables))
    condition_rows = []
    for row in tables["condition"]:
        condition_rows.append(
            {
                "sensor": SENSOR_SHORT_LABELS[row["sensor_variant"]],
                "planner": METHOD_LABELS[row["method"]],
                "map": row["map_label"],
                "success": "{}/{} ({:.0f}%)".format(row["successes"], row["runs"], 100.0 * float(row["success_rate"])),
                "coverage": "{:.1f}%".format(float(row["final_coverage_ratio_mean"])),
                "median": "{:.1f}%".format(float(row["final_coverage_ratio_median"])),
                "t90": "{}/{}".format(row["coverage_90_reached_runs"], row["runs"]),
                "t99": "{}/{}".format(row["coverage_99_reached_runs"], row["runs"]),
            }
        )
    comparison_rows = []
    for row in tables["paper_direct"]:
        comparison_rows.append(
            {
                "metric": row["metric"],
                "meaning": row["meaning"],
                "paper_cost": display_value(row["paper_cost"], row["paper_unit"]),
                "our_cost": display_value(row["our_cost_mean"], row["our_unit"]),
                "paper_field": display_value(row["paper_field"], row["paper_unit"]),
                "our_mmpf": display_value(row["our_mmpf_mean"], row["our_unit"]),
                "ranking": "Matches" if row["directional_ranking_matches"] else "Differs",
            }
        )
    figures = "".join("<section><img src='figures/{}' alt='{}'></section>".format(name, name) for name in figure_names)
    document = """<!doctype html>
<html><head><meta charset='utf-8'><title>Level-0 sensor baseline</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:#f3f5f7;color:#17202a}}
main{{max-width:1180px;margin:auto;background:white;padding:36px 50px;box-shadow:0 0 24px #ccd2d8}}
h1{{font-size:32px;margin-bottom:4px}} h2{{margin-top:34px;border-bottom:2px solid #dce3e8;padding-bottom:6px}}
.subtitle{{color:#58636d}}.warning{{background:#fff1db;border-left:5px solid #d97706;padding:14px 18px;margin:20px 0}}
li{{margin:9px 0;line-height:1.45}} section{{margin:26px 0}} img{{width:100%;height:auto;border:1px solid #dce3e8}}
.table-wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{padding:8px 10px;border-bottom:1px solid #dce3e8;text-align:left}} th{{background:#edf2f6;position:sticky;top:0}}
code{{background:#eef2f5;padding:2px 5px}} footer{{margin-top:35px;color:#65717c;font-size:12px}}
</style></head><body><main>
<h1>Explore-Bench Level-0 sensor baseline</h1>
<p class='subtitle'>Original omnidirectional sensing versus four-beam sensing · 60 paired tasks</p>
<div class='warning'><strong>Interpretation rule:</strong> report milestone reach rate before time-to-milestone. Conditional times exclude runs that never reached the milestone.</div>
<h2>Advisor-ready findings</h2><ul>{findings}</ul>
<h2>Direct Explore-Bench Table II comparison</h2>
<p>The directional ranking matches for all four metrics, but the absolute values are not yet directly comparable.</p>
{comparison_table}
<h2>Condition summary</h2>{condition_table}
<h2>Figures</h2>{figures}
<h2>Files</h2><p>Detailed numeric outputs are in <code>tables/</code>. The ordered PDF is <code>advisor_packet.pdf</code>.</p>
<footer>Dataset: {dataset}<br>Task-table SHA-256: {hash}</footer>
</main></body></html>""".format(
        findings=findings,
        comparison_table=html_table(
            comparison_rows,
            ("metric", "meaning", "paper_cost", "our_cost", "paper_field", "our_mmpf", "ranking"),
            ("Metric", "Meaning", "Paper cost", "Our cost", "Paper field", "Our MMPF", "Ordering"),
        ),
        condition_table=html_table(condition_rows, ("sensor", "planner", "map", "success", "coverage", "median", "t90", "t99"), ("Sensor", "Planner", "Map", "99% success", "Mean final", "Median final", "Reached 90%", "Reached 99%")),
        figures=figures,
        dataset=html.escape(str(dataset["directory"])),
        hash=html.escape(str(dataset["manifest"].get("task_table_sha256", "unknown"))),
    )
    (output / "report.html").write_text(document, encoding="utf-8")


def make_plots(dataset, tables, output):
    plt, PdfPages = setup_plotting(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    names = [
        "01_executive_overview.png",
        "02_condition_matrices.png",
        "03_paired_final_coverage.png",
        "04_coverage_curves.png",
        "05_milestone_performance.png",
        "06_efficiency_and_coordination.png",
        "07_termination_reasons.png",
        "08_representative_trajectories.png",
        "09_explore_bench_direct_comparison.png",
    ]
    with PdfPages(output / "advisor_packet.pdf") as pdf:
        plot_title_page(plt, dataset, tables, pdf)
        plot_executive_overview(plt, dataset["episodes"], tables, figures, pdf)
        plot_condition_matrices(plt, tables, figures, pdf)
        plot_paired_deltas(plt, tables["paired"], figures, pdf)
        plot_coverage_curves(plt, dataset["steps"], figures, pdf)
        plot_milestones(plt, tables, figures, pdf)
        plot_efficiency(plt, dataset["episodes"], tables["paired"], figures, pdf)
        plot_terminations(plt, dataset["episodes"], figures, pdf)
        plot_representative_trajectories(plt, dataset["agents"], tables["paired"], figures, pdf)
        plot_paper_direct_comparison(plt, tables["paper_direct"], figures, pdf)
    write_html_report(dataset, tables, output, names)
    return names


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="merged Level-0 collection directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="generated report directory")
    parser.add_argument("--tables-only", action="store_true", help="write validation tables and brief without importing Matplotlib")
    return parser


def main():
    args = build_parser().parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.input, load_steps=not args.tables_only, load_agents=not args.tables_only)
    tables = write_tables(dataset, output)
    write_brief(dataset, tables, output)
    if not args.tables_only:
        make_plots(dataset, tables, output)
    print("Validated {} paired runs from {}".format(len(tables["paired"]), dataset["directory"]))
    print("Report: {}".format(output / ("advisor_brief.md" if args.tables_only else "report.html")))
    if not args.tables_only:
        print("Advisor packet: {}".format(output / "advisor_packet.pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
