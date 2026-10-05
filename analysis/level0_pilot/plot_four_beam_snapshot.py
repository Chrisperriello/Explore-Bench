#!/usr/bin/env python3
"""Plot one logged four-beam sensor snapshot on the experiment map.

This is a read-only diagnostic of measurements already stored in
``agents.csv``.  It reconstructs beam endpoints geometrically and does not
invoke the sensor model or alter an experiment.
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

from plot_scoped_pilot import load_run
from plot_scoped_pilot_seeds import AGENT_COLORS, map_path


BEAMS = (
    ("front", 0.0, "tab:green"),
    ("back", math.pi, "tab:orange"),
    ("left", math.pi / 2.0, "tab:purple"),
    ("right", -math.pi / 2.0, "tab:cyan"),
)


def as_bool(value):
    """Interpret common CSV truth spellings as a Python boolean."""
    return str(value).strip().lower() in ("true", "1", "yes")


def plot_snapshot(run, seed, step, output_path, show=False):
    """Overlay every robot's four named beams for one recorded seed and step."""
    environment_map = map_path(run)
    if environment_map is None:
        raise ValueError("could not locate the map recorded in the manifest")

    rows = [
        row
        for row in run["agents"]
        if int(row["seed"]) == seed and int(row["step"]) == step
    ]
    rows.sort(key=lambda row: int(row["agent_id"]))
    if not rows:
        raise ValueError("seed {} has no recorded step {}".format(seed, step))

    resolution = float(run["manifest"]["configuration"]["resolution"])
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(plt.imread(environment_map), cmap="gray", origin="upper")

    used_beam_labels = set()
    for row in rows:
        agent_id = int(row["agent_id"])
        start_row = float(row["row"])
        start_column = float(row["column"])
        yaw = float(row["heading_rad"])
        agent_color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]
        ax.scatter(
            start_column,
            start_row,
            color=agent_color,
            marker="o",
            s=100,
            edgecolors="white",
            linewidths=1.2,
            label="Robot {}".format(agent_id),
            zorder=4,
        )

        for name, offset, color in BEAMS:
            distance = float(row[name + "_distance"])
            angle = (yaw + offset) % (2.0 * math.pi)
            cells = distance / resolution
            end_row = start_row - math.sin(angle) * cells
            end_column = start_column + math.cos(angle) * cells
            label = name.capitalize() if name not in used_beam_labels else None
            used_beam_labels.add(name)
            ax.plot(
                [start_column, end_column],
                [start_row, end_row],
                color=color,
                linewidth=2.4,
                label=label,
                zorder=3,
            )
            ax.scatter(
                end_column,
                end_row,
                color=color,
                marker="x" if as_bool(row[name + "_hit"]) else ".",
                s=60,
                zorder=4,
            )

    ax.set_title(
        "{} — seed {} — step {} — four-beam observations".format(
            Path(environment_map).name, seed, step
        )
    )
    ax.set_xlabel("Map column")
    ax.set_ylabel("Map row")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def build_parser():
    """Build arguments identifying the run, seed, step, and PNG destination."""
    parser = argparse.ArgumentParser(
        description="Display one logged four-beam snapshot on its sandbox map."
    )
    parser.add_argument("--run", required=True, help="completed full-recording run")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument(
        "--output",
        help="PNG path; defaults to results/scoped_pilot_analysis",
    )
    parser.add_argument("--show", action="store_true")
    return parser


def main():
    """Load a completed run and save the requested sensor snapshot."""
    args = build_parser().parse_args()
    run = load_run(args.run)
    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        output = (
            Path("results/scoped_pilot_analysis")
            / "seed_{:03d}_step_{:03d}_four_beam.png".format(
                args.seed, args.step
            )
        ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_snapshot(run, args.seed, args.step, output, show=args.show)
    print("Saved {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
