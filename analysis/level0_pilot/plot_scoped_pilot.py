#!/usr/bin/env python3
"""Validate and plot paired fixed-horizon Level-0 pilot conditions.

The script refuses mismatched treatments before plotting.  It compares every
paired seed directly and does not draw a confidence interval from the three-run
calibration pilot.  The output is descriptive evidence for pipeline behavior,
not a thesis-scale statistical result.
"""

import argparse
import csv
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


PLOT_CONDITIONS = ("none", "limited", "perfect")
LABELS = {
    "none": "No communication",
    "limited": "Finite radio (150 cells)",
    "perfect": "Perfect sharing",
}
COLORS = {
    "none": "tab:blue",
    "limited": "tab:orange",
    "perfect": "tab:green",
}


def read_csv(path):
    """Read a complete CSV table as dictionaries without type coercion."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_run(directory):
    """Load one complete full-recording run after checking required artifacts."""
    directory = Path(directory).expanduser().resolve()
    required = ["manifest.json", "episodes.csv", "steps.csv", "agents.csv"]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError(
            "{} is missing {}".format(directory, ", ".join(missing))
        )
    with (directory / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "complete":
        raise ValueError("{} is not complete".format(directory))
    return {
        "directory": directory,
        "manifest": manifest,
        "episodes": read_csv(directory / "episodes.csv"),
        "steps": read_csv(directory / "steps.csv"),
        "agents": read_csv(directory / "agents.csv"),
    }


def config_value(run, name):
    """Read one recorded configuration field from a loaded run manifest."""
    return run["manifest"]["configuration"].get(name)


def validate_pairing(runs):
    """Reject comparisons that differ outside the intended radio treatment.

    The check also requires identical episode seed sets and warns when the
    finite channel delivered no bits, because that would be an out-of-contact
    control rather than an effective information-sharing condition.
    """
    expected_modes = {
        "none": "none",
        "limited": "shared_collision",
        "perfect": "perfect",
    }
    matched_fields = (
        "method",
        "maps",
        "team_sizes",
        "seeds",
        "resolution",
        "max_steps",
        "no_progress_patience",
        "coverage_thresholds",
        "target_coverage",
        "sensor_types",
        "sensor_ranges",
    )
    reference = runs["none"]
    problems = []
    for condition, run in runs.items():
        mode = config_value(run, "communication_mode")
        if mode != expected_modes[condition]:
            problems.append(
                "{} has communication_mode={!r}, expected {!r}".format(
                    condition, mode, expected_modes[condition]
                )
            )
        for field in matched_fields:
            if config_value(run, field) != config_value(reference, field):
                problems.append("{} differs on {}".format(condition, field))
    if problems:
        raise ValueError("Runs are not paired:\n- " + "\n- ".join(problems))

    seeds = {
        condition: {int(row["seed"]) for row in run["episodes"]}
        for condition, run in runs.items()
    }
    if any(value != seeds["none"] for value in seeds.values()):
        raise ValueError("episode CSV seed sets do not match: {}".format(seeds))
    if len(seeds["none"]) < 2:
        warnings.warn("Only one paired seed is present; do not summarize variability.")

    delivered = sum(
        int(row["comm_delivered_bits"]) for row in runs["limited"]["episodes"]
    )
    if delivered == 0:
        warnings.warn(
            "The finite-radio treatment delivered zero bits; it is another "
            "out-of-contact control, not an effective communication treatment."
        )


def grouped_steps(rows):
    """Group step rows by seed and order each trajectory chronologically."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["seed"])].append(row)
    for seed in grouped:
        grouped[seed].sort(key=lambda row: int(row["step"]))
    return grouped


def episode_by_seed(run):
    """Index a run's episode summaries by paired seed."""
    return {int(row["seed"]): row for row in run["episodes"]}


def robot_separation_by_seed(run):
    """Compute two-robot Euclidean separation in grid cells at every step."""
    positions = defaultdict(dict)
    for row in run["agents"]:
        key = (int(row["seed"]), int(row["step"]))
        positions[key][int(row["agent_id"])] = (
            int(row["row"]),
            int(row["column"]),
        )
    result = defaultdict(list)
    for (seed, step), agents in sorted(positions.items()):
        if len(agents) != 2:
            continue
        first, second = agents[sorted(agents)[0]], agents[sorted(agents)[1]]
        result[seed].append(
            (step, math.hypot(first[0] - second[0], first[1] - second[1]))
        )
    return result


def mean_or_zero(rows, field):
    """Return the arithmetic mean of one CSV field, or zero for no rows."""
    values = [float(row[field]) for row in rows]
    return mean(values) if values else 0.0


def write_summary(runs, output_path):
    """Write one descriptive aggregate row per communication condition."""
    fields = [
        "condition",
        "episodes",
        "mean_final_coverage_ratio",
        "sample_sd_final_coverage_ratio",
        "mean_new_cells_per_decision",
        "mean_team_path_length_m",
        "mean_overlap_ratio_total",
        "mean_coverage_std_area_m2",
        "mean_attempted_bits",
        "mean_delivered_bits",
        "mean_delivery_ratio",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in PLOT_CONDITIONS:
            rows = runs[condition]["episodes"]
            coverages = [float(row["final_coverage_ratio"]) for row in rows]
            rates = [
                float(row["newly_explored_cells"])
                / max(1, int(row["steps_executed"]))
                for row in rows
            ]
            writer.writerow(
                {
                    "condition": condition,
                    "episodes": len(rows),
                    "mean_final_coverage_ratio": mean(coverages),
                    "sample_sd_final_coverage_ratio": (
                        stdev(coverages) if len(coverages) > 1 else math.nan
                    ),
                    "mean_new_cells_per_decision": mean(rates),
                    "mean_team_path_length_m": mean_or_zero(
                        rows, "team_path_length_m"
                    ),
                    "mean_overlap_ratio_total": mean_or_zero(
                        rows, "overlap_ratio_total"
                    ),
                    "mean_coverage_std_area_m2": mean_or_zero(
                        rows, "coverage_std_area_m2"
                    ),
                    "mean_attempted_bits": mean_or_zero(
                        rows, "comm_attempted_bits"
                    ),
                    "mean_delivered_bits": mean_or_zero(
                        rows, "comm_delivered_bits"
                    ),
                    "mean_delivery_ratio": mean_or_zero(rows, "delivery_ratio"),
                }
            )


def plot_summary(runs, output_path):
    """Plot coverage, paired outcomes, contact opportunity, and logical bits."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    for condition in PLOT_CONDITIONS:
        grouped = grouped_steps(runs[condition]["steps"])
        values_at_step = defaultdict(list)
        for seed, rows in sorted(grouped.items()):
            x = [int(row["step"]) for row in rows]
            y = [100.0 * float(row["team_coverage_ratio"]) for row in rows]
            ax.plot(x, y, color=COLORS[condition], alpha=0.18, linewidth=1)
            for step, value in zip(x, y):
                values_at_step[step].append(value)
        steps = sorted(values_at_step)
        ax.plot(
            steps,
            [mean(values_at_step[step]) for step in steps],
            color=COLORS[condition],
            linewidth=2.5,
            label=LABELS[condition],
        )
    ax.set_title("Team coverage at a fixed decision budget")
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Team coverage (%)")
    ax.grid(alpha=0.25)
    ax.legend()

    episode_maps = {
        condition: episode_by_seed(runs[condition])
        for condition in PLOT_CONDITIONS
    }
    seeds = sorted(episode_maps["none"])
    x_positions = list(range(len(PLOT_CONDITIONS)))

    ax = axes[0, 1]
    for seed in seeds:
        values = [
            100.0 * float(episode_maps[condition][seed]["final_coverage_ratio"])
            for condition in PLOT_CONDITIONS
        ]
        ax.plot(x_positions, values, marker="o", alpha=0.8, label="Seed {}".format(seed))
    ax.set_xticks(x_positions, [LABELS[value] for value in PLOT_CONDITIONS])
    ax.set_ylabel("Final team coverage (%)")
    ax.set_title("Paired final coverage by seed")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    separations = robot_separation_by_seed(runs["limited"])
    for seed in seeds:
        points = separations[seed]
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            linewidth=1.8,
            label="Seed {}".format(seed),
        )
    radio_range = float(config_value(runs["limited"], "comm_range_cells"))
    ax.axhline(
        radio_range,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Radio range ({:g} cells)".format(radio_range),
    )
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Robot separation (cells)")
    ax.set_title("Contact opportunities explain delivered traffic")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    limited = episode_maps["limited"]
    attempted = [int(limited[seed]["comm_attempted_bits"]) for seed in seeds]
    delivered = [int(limited[seed]["comm_delivered_bits"]) for seed in seeds]
    centers = list(range(len(seeds)))
    width = 0.36
    ax.bar(
        [value - width / 2 for value in centers],
        attempted,
        width,
        label="Attempted",
        color="tab:orange",
    )
    ax.bar(
        [value + width / 2 for value in centers],
        delivered,
        width,
        label="Delivered",
        color="tab:green",
    )
    ax.set_xticks(centers, ["Seed {}".format(seed) for seed in seeds])
    ax.set_ylabel("Logical bits per episode")
    ax.set_title("Finite-radio traffic")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.suptitle(
        "Scoped four-beam communication pilot (three paired seeds, 200 decisions)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    """Build arguments for the three completed run directories and output."""
    parser = argparse.ArgumentParser(
        description="Validate and plot none/finite/perfect scoped pilot runs."
    )
    parser.add_argument("--none", required=True, help="completed no-radio run directory")
    parser.add_argument(
        "--limited", required=True, help="completed finite-radio run directory"
    )
    parser.add_argument(
        "--perfect", required=True, help="completed perfect-sharing run directory"
    )
    parser.add_argument("--output", default="results/scoped_pilot_analysis")
    return parser


def main():
    """Load, validate, summarize, and plot the scoped pilot."""
    args = build_parser().parse_args()
    runs = {
        "none": load_run(args.none),
        "limited": load_run(args.limited),
        "perfect": load_run(args.perfect),
    }
    validate_pairing(runs)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plot_path = output / "scoped_pilot_summary.png"
    summary_path = output / "scoped_pilot_summary.csv"
    plot_summary(runs, plot_path)
    write_summary(runs, summary_path)
    print("Saved {}".format(plot_path))
    print("Saved {}".format(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
