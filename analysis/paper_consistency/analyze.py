#!/usr/bin/env python3
"""Create meeting figures from the three-treatment paper-consistency run."""

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPOSITORY_ROOT.parent
    / "slurm"
    / "data"
    / "paper_consistency"
    / "paper_consistency_01"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT.parent
    / "slurm"
    / "reports"
    / "paper_consistency"
    / "paper_consistency_01"
)
TREATMENTS = ("legacy_3p5", "legacy_7p0", "perfect_3p5")
TREATMENT_LABELS = {
    "legacy_3p5": "Legacy, 3.5 m",
    "legacy_7p0": "Legacy, 7.0 m",
    "perfect_3p5": "Perfect sharing, 3.5 m",
}
METHODS = ("cost", "mmpf")
METHOD_LABELS = {"cost": "Cost", "mmpf": "MMPF"}
COLORS = {
    "legacy_3p5": "#4472C4",
    "legacy_7p0": "#70AD47",
    "perfect_3p5": "#ED7D31",
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def as_float(value, default=math.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def attempt_number(run_directory):
    suffix = run_directory.name[3:]
    return int(suffix) if run_directory.name.startswith("run") and suffix.isdigit() else -1


def newest_complete_run(task_directory):
    candidates = []
    for manifest_path in task_directory.glob("*/run*/manifest.json"):
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
        ):
            candidates.append(run_directory)
    return max(candidates, key=attempt_number) if candidates else None


def load_runs(input_directory):
    episodes = []
    steps = []
    missing = []
    for treatment in TREATMENTS:
        for task_id in range(10):
            task_directory = input_directory / treatment / "task_{:06d}".format(task_id)
            run_directory = newest_complete_run(task_directory)
            if run_directory is None:
                missing.append("{}:{}".format(treatment, task_id))
                continue
            episode_rows = read_csv(run_directory / "episodes.csv")
            if len(episode_rows) != 1:
                raise ValueError("expected one episode in {}".format(run_directory))
            prefix = {
                "treatment": treatment,
                "treatment_label": TREATMENT_LABELS[treatment],
                "task_id": task_id,
                "source_run_directory": str(run_directory),
            }
            episode = dict(prefix)
            episode.update(episode_rows[0])
            episodes.append(episode)
            for source in read_csv(run_directory / "steps.csv"):
                row = dict(prefix)
                row.update(source)
                steps.append(row)
    if missing:
        raise ValueError("missing complete runs: {}".format(", ".join(missing)))
    if len(episodes) != 30:
        raise ValueError("expected 30 episodes, found {}".format(len(episodes)))
    return episodes, steps


def summarize_group(rows, treatment, method):
    result = {
        "treatment": treatment,
        "treatment_label": TREATMENT_LABELS[treatment],
        "method": method,
        "method_label": "All" if method == "all" else METHOD_LABELS[method],
        "runs": len(rows),
        "successes": sum(as_bool(row["success"]) for row in rows),
    }
    result["success_rate"] = result["successes"] / result["runs"]
    final_coverage = [as_float(row["final_coverage_ratio"]) for row in rows]
    result["final_coverage_mean"] = mean(final_coverage)
    result["final_coverage_median"] = median(final_coverage)
    result["steps_mean"] = mean(as_float(row["steps_executed"]) for row in rows)
    result["team_path_length_m_mean"] = mean(
        as_float(row["team_path_length_m"]) for row in rows
    )
    for milestone in ("90", "98", "99"):
        reached = [row for row in rows if as_bool(row["coverage_{}_reached".format(milestone)])]
        result["coverage_{}_reach_rate".format(milestone)] = len(reached) / len(rows)
        result["coverage_{}_step_conditional_mean".format(milestone)] = (
            mean(as_float(row["coverage_{}_step".format(milestone)]) for row in reached)
            if reached
            else ""
        )
    return result


def build_summaries(episodes):
    summaries = []
    for treatment in TREATMENTS:
        treatment_rows = [row for row in episodes if row["treatment"] == treatment]
        summaries.append(summarize_group(treatment_rows, treatment, "all"))
        for method in METHODS:
            rows = [row for row in treatment_rows if row["method"] == method]
            summaries.append(summarize_group(rows, treatment, method))
    return summaries


def build_pairs(episodes):
    indexed = {
        (row["treatment"], row["method"], as_int(row["seed"])): row
        for row in episodes
    }
    pairs = []
    for method in METHODS:
        for seed in range(1, 6):
            row = {"method": method, "seed": seed}
            for treatment in TREATMENTS:
                episode = indexed[(treatment, method, seed)]
                row["{}_success".format(treatment)] = as_bool(episode["success"])
                row["{}_final_coverage".format(treatment)] = as_float(
                    episode["final_coverage_ratio"]
                )
                row["{}_steps".format(treatment)] = as_int(episode["steps_executed"])
            row["range_effect_coverage"] = (
                row["legacy_7p0_final_coverage"] - row["legacy_3p5_final_coverage"]
            )
            row["information_effect_coverage"] = (
                row["perfect_3p5_final_coverage"] - row["legacy_3p5_final_coverage"]
            )
            pairs.append(row)
    return pairs


def find_summary(summaries, treatment, method="all"):
    return next(
        row
        for row in summaries
        if row["treatment"] == treatment and row["method"] == method
    )


def make_figures(episodes, steps, summaries, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    width = 0.24
    positions = list(range(len(METHODS)))
    for offset, treatment in enumerate(TREATMENTS):
        rows = [find_summary(summaries, treatment, method) for method in METHODS]
        x = [position + (offset - 1) * width for position in positions]
        axes[0].bar(
            x,
            [100 * row["success_rate"] for row in rows],
            width,
            label=TREATMENT_LABELS[treatment],
            color=COLORS[treatment],
        )
        axes[1].bar(
            x,
            [100 * row["final_coverage_mean"] for row in rows],
            width,
            color=COLORS[treatment],
        )
    for axis, title, ylabel in (
        (axes[0], "Reached 99% coverage", "Successful runs (%)"),
        (axes[1], "Final coverage", "Mean coverage (%)"),
    ):
        axis.set_xticks(positions, [METHOD_LABELS[value] for value in METHODS])
        axis.set_ylim(0, 105)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Five-room diagnostic: success and final coverage")
    fig.tight_layout()
    fig.savefig(figures / "01_success_and_coverage.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    x = list(range(len(TREATMENTS)))
    for axis, method in zip(axes, METHODS):
        method_rows = [row for row in episodes if row["method"] == method]
        by_seed = defaultdict(dict)
        for row in method_rows:
            by_seed[as_int(row["seed"])][row["treatment"]] = 100 * as_float(
                row["final_coverage_ratio"]
            )
        for seed, values in sorted(by_seed.items()):
            axis.plot(
                x,
                [values[treatment] for treatment in TREATMENTS],
                marker="o",
                alpha=0.75,
                label="Seed {}".format(seed),
            )
        axis.axhline(99, color="black", linestyle="--", linewidth=1, label="99% target")
        axis.set_xticks(x, ["Legacy\n3.5 m", "Legacy\n7.0 m", "Perfect\n3.5 m"])
        axis.set_ylim(0, 103)
        axis.set_title(METHOD_LABELS[method])
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Final coverage (%)")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Matched-seed final coverage")
    fig.tight_layout()
    fig.savefig(figures / "02_paired_final_coverage.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    indexed_steps = defaultdict(dict)
    final_steps = {}
    for row in steps:
        key = (row["treatment"], row["method"], as_int(row["seed"]))
        step = as_int(row["step"])
        indexed_steps[key][step] = as_float(row["team_coverage_ratio"])
        final_steps[key] = max(final_steps.get(key, 0), step)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    sample_steps = list(range(0, 1001, 10))
    for axis, method in zip(axes, METHODS):
        for treatment in TREATMENTS:
            curves = []
            for seed in range(1, 6):
                observed = indexed_steps[(treatment, method, seed)]
                ordered = sorted(observed)
                curve = []
                position = 0
                last = observed[ordered[0]]
                for target in sample_steps:
                    while position + 1 < len(ordered) and ordered[position + 1] <= target:
                        position += 1
                        last = observed[ordered[position]]
                    curve.append(last)
                curves.append(curve)
            average = [100 * mean(values) for values in zip(*curves)]
            axis.plot(
                sample_steps,
                average,
                label=TREATMENT_LABELS[treatment],
                color=COLORS[treatment],
                linewidth=2,
            )
        axis.axhline(99, color="black", linestyle="--", linewidth=1)
        axis.set_xlim(0, 1000)
        axis.set_ylim(0, 103)
        axis.set_xlabel("Exploration decisions")
        axis.set_title(METHOD_LABELS[method])
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean coverage (%)")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Coverage progress (final value held after a run stops)")
    fig.tight_layout()
    fig.savefig(figures / "03_coverage_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_assessment(path, summaries, pairs):
    reference = find_summary(summaries, "legacy_3p5")
    long_range = find_summary(summaries, "legacy_7p0")
    perfect = find_summary(summaries, "perfect_3p5")
    range_deltas = [row["range_effect_coverage"] for row in pairs]
    information_deltas = [row["information_effect_coverage"] for row in pairs]
    text = """# Paper-consistency diagnostic

## Main result

| Treatment | 99% successes | Success rate | Mean final coverage |
| --- | ---: | ---: | ---: |
| Legacy, 3.5 m | {ref_success}/{ref_runs} | {ref_rate:.0%} | {ref_coverage:.1%} |
| Legacy, 7.0 m | {range_success}/{range_runs} | {range_rate:.0%} | {range_coverage:.1%} |
| Perfect sharing, 3.5 m | {perfect_success}/{perfect_runs} | {perfect_rate:.0%} | {perfect_coverage:.1%} |

The 7.0 m legacy treatment changed matched-seed final coverage by an average
of {range_delta:+.1%} relative to legacy 3.5 m. The perfect-sharing treatment
changed it by {information_delta:+.1%}.

## Interpretation

Increasing the sensor distance did not produce a consistent improvement over
the current legacy reference. The major change came from replacing legacy
navigation with sensed-map-only navigation. Under perfect sharing, both robots
immediately share everything they sense, but unknown space is no longer usable
as privileged route knowledge. Most runs then fail to reach 99% within 1,000
decisions.

This means the main difficulty is not communication loss: perfect sharing has
no packet loss or delay. The difficulty is planning under incomplete spatial
information. Cost and MMPF are static frontier planners and do not adapt their
behavior to recover from sparse knowledge as effectively as they do under the
legacy compatibility path.

This result motivates a learned policy, but it does not yet show that learning
solves the problem. MAPPO must be trained and evaluated with the same restricted
observation and navigation rules, then compared with these static baselines on
the same maps, starts, and seeds.

## Important caution

Slurm `COMPLETED` means the program finished correctly. It does not mean the
robot team reached the 99% exploration target. Use the success rate and final
coverage above as the exploration outcome.
""".format(
        ref_success=reference["successes"],
        ref_runs=reference["runs"],
        ref_rate=reference["success_rate"],
        ref_coverage=reference["final_coverage_mean"],
        range_success=long_range["successes"],
        range_runs=long_range["runs"],
        range_rate=long_range["success_rate"],
        range_coverage=long_range["final_coverage_mean"],
        perfect_success=perfect["successes"],
        perfect_runs=perfect["runs"],
        perfect_rate=perfect["success_rate"],
        perfect_coverage=perfect["final_coverage_mean"],
        range_delta=mean(range_deltas),
        information_delta=mean(information_deltas),
    )
    path.write_text(text, encoding="utf-8")
    return text


def write_html(path, assessment):
    body = "<br>".join(html.escape(line) for line in assessment.splitlines())
    images = "".join(
        "<img src='figures/{}' alt='{}'>".format(name, name)
        for name in (
            "01_success_and_coverage.png",
            "02_paired_final_coverage.png",
            "03_coverage_curves.png",
        )
    )
    path.write_text(
        """<!doctype html><meta charset='utf-8'><title>Paper consistency</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;line-height:1.5}}img{{width:100%;margin:24px 0}}table{{border-collapse:collapse}}</style>
<h1>Paper-consistency diagnostic</h1><div>{}</div>{}""".format(body, images),
        encoding="utf-8",
    )


def analyze(input_directory, output, tables_only=False):
    input_directory = input_directory.expanduser().resolve()
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    episodes, steps = load_runs(input_directory)
    summaries = build_summaries(episodes)
    pairs = build_pairs(episodes)
    write_csv(output / "episodes_combined.csv", episodes)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paired_runs.csv", pairs)
    assessment = write_assessment(output / "meeting_assessment.md", summaries, pairs)
    if not tables_only:
        make_figures(episodes, steps, summaries, output)
        write_html(output / "report.html", assessment)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        summaries = analyze(args.input, args.output, args.tables_only)
    except ValueError as error:
        raise SystemExit(str(error))
    for treatment in TREATMENTS:
        row = find_summary(summaries, treatment)
        print(
            "{}: {}/{} successes, mean final coverage {:.1%}".format(
                TREATMENT_LABELS[treatment],
                row["successes"],
                row["runs"],
                row["final_coverage_mean"],
            )
        )
    print("Report: {}".format(args.output.expanduser().resolve()))


if __name__ == "__main__":
    main()
