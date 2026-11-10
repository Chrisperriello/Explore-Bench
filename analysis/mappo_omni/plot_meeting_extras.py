"""Validation learning curves and the routing-contract comparison for meetings.

Usage: python analysis/mappo_omni/plot_meeting_extras.py COLLECTION_DIR LEGACY_EPISODES_CSV OUTPUT_DIR

COLLECTION_DIR is a MAPPO collection (validation/, selections/, scores_prelim_seeds1-4/).
LEGACY_EPISODES_CSV is the Oct 7 legacy-routing baseline
(combined/level0/level0_validation_01/episodes.csv). Its task set differs from the
MAPPO study, so the two halves of the routing figure are separated and labelled.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_prelim_scores import INK, INK2, MUTED, SERIES, SURFACE, read, spread

SEEDS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def validation_points(collection):
    """Per seed: sorted (environment_steps, success_rate, mean_coverage_auc) from raw validation episodes."""
    points = defaultdict(list)
    for directory in sorted((collection / "validation").glob("seed_*/step_*")):
        episodes = directory / "episodes.csv"
        if not episodes.exists():
            continue
        rows = read(episodes)
        if not rows:
            continue
        seed = int(directory.parent.name.split("_")[1])
        success = sum(r["success"] == "True" for r in rows) / len(rows)
        auc = sum(float(r["coverage_auc"]) for r in rows) / len(rows)
        points[seed].append((int(directory.name.split("_")[1]), success, auc, len(rows)))
    return points


def figure_learning(collection, output):
    points = validation_points(collection)
    selected = {}
    for path in (collection / "selections").glob("seed_*.json"):
        record = json.loads(path.read_text())
        selected[record["training_seed"]] = record["selected"]["environment_steps"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    panels = ((1, "Validation episodes reaching 99% coverage", True), (2, "Validation coverage AUC", False))
    for ax, (column, title, percent) in zip(axes, panels):
        ends = []
        for seed in sorted(points):
            series = [p for p in points[seed] if p[3] == 20]
            if not series:
                continue
            xs = [p[0] / 1e6 for p in series]
            ys = [p[column] for p in series]
            label = "seed {}".format(seed) + ("" if seed in selected else " (validating)")
            ax.plot(xs, ys, color=SEEDS[seed - 1], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=2)
            if seed in selected:
                step = selected[seed]
                y = next(p[column] for p in series if p[0] == step)
                ax.scatter([step / 1e6], [y], s=170, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=3)
            ends.append((xs[-1], ys[-1], seed))
        for (x, _, seed), y in zip(ends, spread([e[1] for e in ends], 0.045 if percent else 0.018)):
            ax.text(x + 0.02, y, "s{}".format(seed), va="center", fontsize=9, color=INK2)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel("Training environment steps (millions)")
        ax.set_xlim(-0.03, 1.1)
        if percent:
            ax.set_ylim(-0.03, 1.05)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=11, markerfacecolor="none",
                              markeredgecolor=INK, markeredgewidth=1.6))
    labels.append("selected checkpoint")
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, labelcolor=INK2,
               bbox_to_anchor=(0.99, 0.99), fontsize=9)
    fig.suptitle("MAPPO learning: checkpoints scored on held-out validation maps", x=0.01, ha="left",
                 fontsize=13, color=INK)
    fig.text(0.01, 0.005, "360° sensor, 3.5 m, 2 robots, belief-only routing. 20 validation episodes per checkpoint "
             "(room_with_corner, room_with_corridor); never used for training or testing.",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(output / "04_learning_curves.png", dpi=200)
    plt.close(fig)


def figure_routing(collection, legacy_csv, output):
    legacy = defaultdict(list)
    with Path(legacy_csv).open(newline="") as handle:
        for row in csv.DictReader(handle):
            legacy[(row["method"], row["sensor_variant"])].append(row["success"] == "True")
    aggregate = {(r["method"], r["suite"]): float(r["mean"])
                 for r in read(collection / "scores_prelim_seeds1-4" / "aggregate.csv")
                 if r["metric"] == "success_rate"}
    groups = [
        ("Legacy routing\n360°", lambda m: legacy.get((m, "omnidirectional"))),
        ("Legacy routing\nfour-beam", lambda m: legacy.get((m, "four_beam"))),
        ("Belief-only routing\n360°, familiar maps", lambda m: aggregate.get((m, "familiar"))),
        ("Belief-only routing\n360°, unseen maps", lambda m: aggregate.get((m, "unseen"))),
    ]
    methods = ["cost", "mmpf", "mappo"]
    labels = {"cost": "Cost", "mmpf": "MMPF", "mappo": "MAPPO"}
    width = 0.26
    fig, ax = plt.subplots(figsize=(13, 5.2))
    for index, (_, lookup) in enumerate(groups):
        for offset, method in enumerate(methods):
            x = index + (offset - 1) * width
            value = lookup(method)
            if isinstance(value, list):
                value = sum(value) / len(value)
            if value is None:
                ax.text(x, 0.02, "running", rotation=90, ha="center", va="bottom", fontsize=9, color=MUTED)
                continue
            ax.bar(x, value, width, color=SERIES[method], edgecolor=SURFACE, linewidth=2, zorder=2,
                   label=labels[method] if index == 2 else None)
            ax.text(x, value + 0.015, "{:.0%}".format(value), ha="center", va="bottom", fontsize=9, color=INK2)
    ax.axvline(1.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.5, 1.06, "Oct 7 run: 6 maps, merged-map sharing, 30 episodes each", ha="center", fontsize=9, color=MUTED)
    ax.text(2.5, 1.06, "Current study: frozen test tasks, perfect sharing", ha="center", fontsize=9, color=MUTED)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Episodes reaching 99% coverage")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", visible=False)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper right", frameon=False, labelcolor=INK2, ncol=3,
               bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("Routing on the true map props up the scripted planners; MAPPO never uses it",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.005, "Legacy = A* may cross unseen free space (oracle). Belief-only = A* uses only the shared "
             "explored map. Left and right use different task sets; compare within each side. MAPPO: 4 seeds, preliminary.",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(output / "05_routing_comparison.png", dpi=200)
    plt.close(fig)


def main():
    collection, legacy_csv, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    output.mkdir(parents=True, exist_ok=True)
    figure_learning(collection, output)
    figure_routing(collection, legacy_csv, output)
    print(output)


if __name__ == "__main__":
    main()
