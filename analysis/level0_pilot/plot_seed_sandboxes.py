#!/usr/bin/env python3
"""Display the ground-truth sandbox and robot paths for each recorded seed.

Ground truth is used only as a visualization background after collection; it
is never sent to the controller.  Paths come from executed positions in the
collector's long-form agent table.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_scoped_pilot import episode_by_seed, load_run
from plot_scoped_pilot_seeds import AGENT_COLORS, agent_trajectories, map_path


def plot_seed(run, seed, output_path, show=False):
    """Plot one seed's map, robot starts/ends, paths, and final coverage."""
    environment_map = map_path(run)
    if environment_map is None:
        raise ValueError("could not locate the map recorded in the manifest")

    episodes = episode_by_seed(run)
    if seed not in episodes:
        raise ValueError("seed {} is not present in the run".format(seed))

    trajectories = agent_trajectories(run["agents"], seed)
    episode = episodes[seed]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(plt.imread(environment_map), cmap="gray", origin="upper")

    for agent_id, points in sorted(trajectories.items()):
        color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]
        columns = [point[2] for point in points]
        rows = [point[1] for point in points]
        ax.plot(
            columns,
            rows,
            color=color,
            linewidth=2.2,
            label="Robot {} path".format(agent_id),
        )
        ax.scatter(
            columns[0],
            rows[0],
            color=color,
            marker="o",
            s=75,
            label="Robot {} start".format(agent_id),
        )
        ax.scatter(
            columns[-1],
            rows[-1],
            color=color,
            marker="x",
            s=90,
            linewidths=2.5,
            label="Robot {} end".format(agent_id),
        )

    coverage = 100.0 * float(episode["final_coverage_ratio"])
    ax.set_title(
        "{} — seed {} — {} robots — final coverage {:.2f}%".format(
            Path(environment_map).name,
            seed,
            episode["team_size"],
            coverage,
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
    """Build arguments for a completed run, selected seeds, and output folder."""
    parser = argparse.ArgumentParser(
        description="Plot the recorded map and robot paths for each seed."
    )
    parser.add_argument("--run", required=True, help="completed run directory")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="specific seeds to display; defaults to every seed",
    )
    parser.add_argument("--output", default="results/scoped_pilot_analysis/sandboxes")
    parser.add_argument(
        "--show",
        action="store_true",
        help="also open each image interactively after saving it",
    )
    return parser


def main():
    """Load one run and save a sandbox trajectory image for each selected seed."""
    args = build_parser().parse_args()
    run = load_run(args.run)
    available = sorted(episode_by_seed(run))
    seeds = args.seeds if args.seeds is not None else available
    missing = sorted(set(seeds) - set(available))
    if missing:
        raise ValueError("requested seeds are unavailable: {}".format(missing))

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        path = output / "seed_{:03d}_sandbox.png".format(seed)
        plot_seed(run, seed, path, show=args.show)
        print("Saved {}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
