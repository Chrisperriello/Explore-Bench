"""Meeting figures from an `explore_bench score` output directory.

Usage: python analysis/mappo_omni/plot_prelim_scores.py SCORES_DIR OUTPUT_DIR

Reads aggregate.csv, per_seed.csv and coverage_curves.csv only. Sharded
inputs (one directory per map, equal episode counts) appear as several curve
rows per method/suite; they are averaged with equal weight, which equals the
episode mean because every shard holds the same number of episodes.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["mappo", "cost", "mmpf", "random", "wait"]
LABELS = {"mappo": "MAPPO", "cost": "Cost", "mmpf": "MMPF", "random": "Random", "wait": "Wait"}
SERIES = {"mappo": "#2a78d6", "cost": "#eb6834", "mmpf": "#1baf7a", "random": "#eda100", "wait": "#e87ba4"}
SUITES = {"familiar": ("Familiar maps (training layouts)", "#2a78d6"),
          "unseen": ("Unseen maps (held-out loops)", "#eb6834")}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": ["system-ui", "-apple-system", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def read(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    return float(value) if value not in ("", None) else None


def spread(values, gap):
    """Nudge label positions apart so neighbouring direct labels never overlap."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for previous, current in zip(order, order[1:]):
        placed[current] = max(placed[current], placed[previous] + gap)
    return placed


def bar_panel(ax, aggregate, metric, title, percent):
    width, gap = 0.36, 0.04
    for offset, suite in ((-1, "familiar"), (1, "unseen")):
        label, color = SUITES[suite]
        for index, method in enumerate(METHODS):
            row = aggregate[(method, suite, metric)]
            value = float(row["mean"])
            x = index + offset * (width + gap) / 2
            ax.bar(x, value, width, color=color, label=label if index == 0 else None,
                   edgecolor=SURFACE, linewidth=2, zorder=2)
            low, high = number(row["ci95_low"]), number(row["ci95_high"])
            if low is not None:
                ax.errorbar(x, value, yerr=[[value - low], [high - value]], color=INK, linewidth=1.2,
                            capsize=3, zorder=3)
            text = "{:.0%}".format(value) if percent else "{:.2f}".format(value)
            top = high if low is not None else value
            ax.text(x, top + 0.02, text, ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([LABELS[m] for m in METHODS], color=INK)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    if percent:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


def figure_scores(aggregate, output, seeds):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    bar_panel(axes[0], aggregate, "success_rate", "Episodes reaching 99% coverage", True)
    bar_panel(axes[1], aggregate, "mean_coverage_auc", "Coverage AUC (higher = explores faster)", False)
    axes[0].legend(loc="upper right", frameon=False, labelcolor=INK2)
    fig.suptitle("360° sensor, 3.5 m, 2 robots: MAPPO vs scripted planners on the frozen test tasks",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.005, "Preliminary: MAPPO = {} training seeds, whiskers = 95% bootstrap interval over seeds. "
             "Scripted planners are deterministic single runs (80 familiar + 20 unseen episodes).".format(seeds),
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(output / "01_success_and_auc.png", dpi=200)
    plt.close(fig)


def figure_curves(curves, output, seeds):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in curves:
        grouped[(row["method"], row["suite"])][int(row["step"])].append(float(row["mean_coverage"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, suite in zip(axes, ("familiar", "unseen")):
        ends = []
        for method in METHODS:
            series = grouped[(method, suite)]
            steps = sorted(series)
            mean = [sum(series[s]) / len(series[s]) for s in steps]
            if method == "mappo":
                ax.fill_between(steps, [min(series[s]) for s in steps], [max(series[s]) for s in steps],
                                color=SERIES[method], alpha=0.15, linewidth=0)
            ax.plot(steps, mean, color=SERIES[method], linewidth=2, label=LABELS[method])
            ends.append((steps[-1], mean[-1]))
        for method, (x, _), y in zip(METHODS, ends, spread([e[1] for e in ends], 0.045)):
            ax.text(x + 12, y, LABELS[method], va="center", fontsize=9, color=INK2)
        ax.axhline(0.99, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(5, 0.995, "99% target", fontsize=8, color=MUTED, va="bottom")
        ax.set_xlim(0, 1100)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Joint decisions (one grid edge per robot)")
        ax.set_title(SUITES[suite][0], loc="left", fontsize=12, fontweight="bold")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("Mean team coverage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=5, frameon=False, labelcolor=INK2,
               bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("Coverage over time on the test tasks", x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.005, "MAPPO line = mean of {} training seeds; shaded band = range across seeds.".format(seeds),
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(output / "02_coverage_curves.png", dpi=200)
    plt.close(fig)


def figure_seeds(per_seed, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, suite in zip(axes, ("familiar", "unseen")):
        learned = sorted((r for r in per_seed if r["method"] == "mappo" and r["suite"] == suite),
                         key=lambda r: int(r["training_seed"]))
        xs = [int(r["training_seed"]) for r in learned]
        ax.scatter(xs, [float(r["success_rate"]) for r in learned], s=90, color=SERIES["mappo"],
                   edgecolor=SURFACE, linewidth=2, zorder=3, label="MAPPO (each training seed)")
        references = []
        for method in ("cost", "mmpf"):
            row = next(r for r in per_seed if r["method"] == method and r["suite"] == suite)
            value = float(row["success_rate"])
            ax.plot([min(xs) - 0.5, max(xs) + 0.4], [value, value], color=SERIES[method], linewidth=2, zorder=2,
                    label=LABELS[method] + " (scripted)" if suite == "familiar" else None)
            references.append((method, value))
        for (method, value), y in zip(references, spread([v for _, v in references], 0.05)):
            ax.text(max(xs) + 0.5, y, "{} {:.0%}".format(LABELS[method], value), va="center",
                    fontsize=9, color=INK2)
        ax.set_xticks(xs)
        ax.set_xticklabels(["seed {}".format(x) for x in xs], color=INK)
        ax.set_xlim(min(xs) - 0.5, max(xs) + 1.3)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(axis="x", visible=False)
        ax.set_title(SUITES[suite][0], loc="left", fontsize=12, fontweight="bold")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("Episodes reaching 99% coverage")
    axes[0].legend(loc="lower left", frameon=False, labelcolor=INK2, bbox_to_anchor=(0, 0.06))
    fig.suptitle("Every MAPPO training seed vs the scripted planners", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "03_mappo_seeds.png", dpi=200)
    plt.close(fig)


def main():
    scores, output = Path(sys.argv[1]), Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    aggregate = {(r["method"], r["suite"], r["metric"]): r for r in read(scores / "aggregate.csv")}
    per_seed = read(scores / "per_seed.csv")
    seeds = len({r["training_seed"] for r in per_seed if r["method"] == "mappo"})
    figure_scores(aggregate, output, seeds)
    figure_curves(read(scores / "coverage_curves.csv"), output, seeds)
    figure_seeds(per_seed, output)
    print(output)


if __name__ == "__main__":
    main()
