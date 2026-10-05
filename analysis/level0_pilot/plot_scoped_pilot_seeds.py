#!/usr/bin/env python3
"""Create one detailed diagnostic figure per paired scoped-pilot seed.

Each figure keeps individual seeds visible and separates absolute coverage,
the paired treatment difference, cumulative logical traffic, and spatial robot
paths.  Identical trajectories are drawn once rather than overplotted.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_scoped_pilot import (
    COLORS,
    LABELS,
    PLOT_CONDITIONS,
    episode_by_seed,
    load_run,
    validate_pairing,
)


LINESTYLES = {
    "none": "-",
    "limited": "--",
    "perfect": ":",
}
AGENT_COLORS = ("tab:blue", "tab:red", "tab:purple", "tab:brown")


def rows_for_seed(rows, seed):
    """Filter CSV rows to one seed and return them in decision-step order."""
    selected = [row for row in rows if int(row["seed"]) == seed]
    selected.sort(key=lambda row: int(row["step"]))
    return selected


def step_values(rows, field, scale=1.0):
    """Index a numeric step field by decision number with optional scaling."""
    return {
        int(row["step"]): scale * float(row[field])
        for row in rows
    }


def agent_trajectories(rows, seed):
    """Group one seed's recorded ``(step, row, column)`` points by agent."""
    trajectories = {}
    for row in rows_for_seed(rows, seed):
        agent_id = int(row["agent_id"])
        trajectories.setdefault(agent_id, []).append(
            (int(row["step"]), int(row["row"]), int(row["column"]))
        )
    return trajectories


def trajectories_match(runs, seed):
    """Return whether all three information treatments produced identical paths."""
    reference = agent_trajectories(runs["none"]["agents"], seed)
    for condition in ("limited", "perfect"):
        if agent_trajectories(runs[condition]["agents"], seed) != reference:
            return False
    return True


def map_path(run):
    """Resolve the exact input map from manifest metadata or configuration."""
    maps = run["manifest"].get("maps", [])
    if maps and isinstance(maps[0], dict) and maps[0].get("path"):
        candidate = Path(maps[0]["path"])
        if candidate.is_file():
            return candidate
    configured = run["manifest"]["configuration"].get("maps", [])
    if configured:
        candidate = Path(configured[0]).expanduser()
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parents[2] / candidate
        if candidate.is_file():
            return candidate
    return None


def set_zero_friendly_limits(ax, values):
    """Keep an all-zero treatment-difference plot visually interpretable."""
    largest = max((abs(value) for value in values), default=0.0)
    if largest < 1e-9:
        ax.set_ylim(-0.05, 0.05)


def plot_seed(runs, seed, output_path, show=False):
    """Write a four-panel diagnostic for one seed shared by all treatments."""
    episode_maps = {
        condition: episode_by_seed(runs[condition])
        for condition in PLOT_CONDITIONS
    }
    if any(seed not in values for values in episode_maps.values()):
        raise ValueError("seed {} is not present in every condition".format(seed))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel 1: absolute coverage curves for this seed.
    ax = axes[0, 0]
    coverage = {}
    for condition in PLOT_CONDITIONS:
        rows = rows_for_seed(runs[condition]["steps"], seed)
        coverage[condition] = step_values(
            rows, "team_coverage_ratio", scale=100.0
        )
        steps = sorted(coverage[condition])
        ax.plot(
            steps,
            [coverage[condition][step] for step in steps],
            color=COLORS[condition],
            linestyle=LINESTYLES[condition],
            linewidth=2.2,
            label=LABELS[condition],
        )
    ax.set_title("Team coverage")
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Coverage (%)")
    ax.grid(alpha=0.25)
    ax.legend()

    # Panel 2: treatment effects relative to the paired private control.
    ax = axes[0, 1]
    difference_values = []
    for condition in ("limited", "perfect"):
        shared_steps = sorted(set(coverage["none"]) & set(coverage[condition]))
        differences = [
            coverage[condition][step] - coverage["none"][step]
            for step in shared_steps
        ]
        difference_values.extend(differences)
        ax.plot(
            shared_steps,
            differences,
            color=COLORS[condition],
            linestyle=LINESTYLES[condition],
            linewidth=2.0,
            label="{} minus private".format(LABELS[condition]),
        )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    set_zero_friendly_limits(ax, difference_values)
    ax.set_title("Paired coverage difference")
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Difference (percentage points)")
    ax.grid(alpha=0.25)
    ax.legend()

    # Panel 3: cumulative logical traffic in the finite-radio treatment.
    ax = axes[1, 0]
    limited_rows = rows_for_seed(runs["limited"]["steps"], seed)
    steps = [int(row["step"]) for row in limited_rows]
    attempted = [int(row["comm_attempted_bits"]) for row in limited_rows]
    delivered = [int(row["comm_delivered_bits"]) for row in limited_rows]
    ax.plot(steps, attempted, color="tab:orange", label="Attempted bits")
    ax.plot(steps, delivered, color="tab:green", label="Delivered bits")
    ax.fill_between(
        steps,
        delivered,
        attempted,
        color="tab:red",
        alpha=0.12,
        label="Attempted without delivery",
    )
    ax.set_title("Finite-radio traffic")
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Cumulative logical bits")
    ax.grid(alpha=0.25)
    ax.legend()

    # Panel 4: spatial trajectories. Avoid drawing identical paths three times.
    ax = axes[1, 1]
    environment_map = map_path(runs["none"])
    if environment_map is not None:
        ax.imshow(plt.imread(environment_map), cmap="gray", origin="upper")
    identical = trajectories_match(runs, seed)
    conditions = ("none",) if identical else PLOT_CONDITIONS
    for condition in conditions:
        trajectories = agent_trajectories(runs[condition]["agents"], seed)
        for agent_id, points in sorted(trajectories.items()):
            color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]
            columns = [point[2] for point in points]
            rows = [point[1] for point in points]
            label = "Robot {}".format(agent_id)
            if not identical:
                label += ", {}".format(LABELS[condition])
            ax.plot(
                columns,
                rows,
                color=color,
                linestyle=LINESTYLES[condition],
                linewidth=1.8,
                label=label,
            )
            ax.scatter(columns[0], rows[0], color=color, marker="o", s=35)
            ax.scatter(columns[-1], rows[-1], color=color, marker="x", s=45)
    title = "Robot trajectories"
    if identical:
        title += " (identical across conditions)"
    ax.set_title(title)
    ax.set_xlabel("Map column")
    ax.set_ylabel("Map row")
    ax.legend(fontsize=8)

    final_coverage = {
        condition: 100.0
        * float(episode_maps[condition][seed]["final_coverage_ratio"])
        for condition in PLOT_CONDITIONS
    }
    delivered_bits = int(episode_maps["limited"][seed]["comm_delivered_bits"])
    fig.suptitle(
        (
            "Seed {seed}: final coverage private={none:.2f}%, "
            "finite={limited:.2f}%, perfect={perfect:.2f}%; "
            "finite delivered={bits:,} bits"
        ).format(seed=seed, bits=delivered_bits, **final_coverage),
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def build_parser():
    """Build arguments for paired runs, optional seed filtering, and output."""
    parser = argparse.ArgumentParser(
        description="Create one four-panel diagnostic image for every paired seed."
    )
    parser.add_argument("--none", required=True, help="completed private run directory")
    parser.add_argument(
        "--limited", required=True, help="completed finite-radio run directory"
    )
    parser.add_argument(
        "--perfect", required=True, help="completed perfect-sharing run directory"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="specific seeds to plot; defaults to every paired seed",
    )
    parser.add_argument("--output", default="results/scoped_pilot_analysis/seeds")
    parser.add_argument(
        "--show",
        action="store_true",
        help="also open each figure interactively after saving it",
    )
    return parser


def main():
    """Validate paired runs and create one diagnostic image per selected seed."""
    args = build_parser().parse_args()
    runs = {
        "none": load_run(args.none),
        "limited": load_run(args.limited),
        "perfect": load_run(args.perfect),
    }
    validate_pairing(runs)
    available = sorted(episode_by_seed(runs["none"]))
    seeds = args.seeds if args.seeds is not None else available
    missing = sorted(set(seeds) - set(available))
    if missing:
        raise ValueError("requested seeds are unavailable: {}".format(missing))

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        path = output / "seed_{:03d}.png".format(seed)
        plot_seed(runs, seed, path, show=args.show)
        print("Saved {}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
