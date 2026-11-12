"""Meeting figures for re-tests on the frozen map sets (rooms_v1 + houseexpo_v1).

Usage:
  python analysis/newmaps/plot_newmaps.py OUTPUT_DIR \
      --collection results/newmaps_2026-11-12/mappo_omni_3p5_newmaps_01 \
      --old-scores results/scores_v2_2026-11-06/mappo_omni_3p5_full_01/scores_v2 \
      [--collection FOUR_BEAM_DIR --old-scores FOUR_BEAM_OLD_SCORES]

Each --collection is one sensor (score() refuses to mix sensors). Reads the
episodes.csv of every run directory, plus scores/ for coverage curves and
inventory.json for the sensor name. Writes one set of PNGs per sensor; the
paired sensor-gap figure is written only when two collections are given.

Uncertainty: "map interval" = 95% percentile bootstrap over the 60 maps
(each map's mean success is one sample), so it shows how much a result
depends on which maps were drawn. MAPPO is first averaged over its 5
training seeds; seed spread is drawn separately as dots.

Map tags (loops, narrow_share, rooms) are NOT used: they are labels only
until fixed (docs/map-sets-plan.md, "Known problem"). Splits are by source.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHODS = ["mappo", "cost", "mmpf", "random", "wait"]
PLANNERS = ["mappo", "cost", "mmpf"]
LABELS = {"mappo": "MAPPO", "cost": "Cost", "mmpf": "MMPF", "random": "Random", "wait": "Wait"}
SERIES = {"mappo": "#2a78d6", "cost": "#eb6834", "mmpf": "#1baf7a", "random": "#eda100", "wait": "#e87ba4"}
SOURCES = {"rooms": "rooms_v1, no courtyard (16)", "courtyard": "rooms_v1 courtyard (14)",
           "houseexpo": "HouseExpo (30 real outlines)"}
SENSORS = {"omnidirectional": "360° sensor", "four_beam": "four-beam sensor"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BOOTSTRAP = 10000

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


def map_facts():
    """Free area (a measured tag, reliable) and the generator's courtyard flag
    (a construction fact, not a shape tag) for every frozen map."""
    areas, courtyards = {}, set()
    for manifest in ROOT.glob("maps/*/manifest.json"):
        for item in json.loads(manifest.read_text())["maps"]:
            areas[Path(item["file"]).stem] = item["tags"]["free_area_m2"]
            if item.get("courtyard"):
                courtyards.add(Path(item["file"]).stem)
    return areas, courtyards


def source_of(name, courtyards):
    if name.startswith("rooms"):
        return "courtyard" if name in courtyards else "rooms"
    return "houseexpo"


def load(collection):
    """One dict per episode with the fields the figures use."""
    (areas, courtyards), episodes = map_facts(), []
    for path in sorted(Path(collection).glob("*/*/*/episodes.csv")):
        for row in read(path):
            name = Path(row["map_name"]).stem
            coverage = float(row["final_coverage_ratio"])
            episodes.append({
                "method": row["method"], "seed": row["training_seed"], "map": name,
                "source": source_of(name, courtyards),
                "episode": row["episode_id"], "success": row["success"] == "True",
                "coverage": coverage, "path_m": float(row["team_path_length_m"]),
                "explored_m2": coverage * areas[name], "overlap": float(row["overlap_ratio_total"]),
                "t99": int(row["coverage_99_step"]) if row["coverage_99_step"] else None,
                "reason": row["termination_reason"]})
    if not episodes:
        raise SystemExit("no episodes.csv under " + str(collection))
    inventory = json.loads((Path(collection) / "scores/inventory.json").read_text())
    return episodes, SENSORS.get(inventory["physical_contract"]["sensor_type"], "?")


def per_map(episodes, method, key="success", source=None):
    """Map -> mean of key over that map's episodes (MAPPO pools all seeds)."""
    values = defaultdict(list)
    for e in episodes:
        if e["method"] == method and (source is None or e["source"] == source):
            values[e["map"]].append(float(e[key]))
    return {name: float(np.mean(v)) for name, v in values.items()}


def interval(values, seed=20261113):
    values = np.asarray(list(values))
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def seed_means(episodes, key="success", source=None):
    values = defaultdict(list)
    for e in episodes:
        if e["method"] == "mappo" and (source is None or e["source"] == source):
            values[e["seed"]].append(float(e[key]))
    return {seed: float(np.mean(v)) for seed, v in sorted(values.items())}


def present(episodes):
    found = {e["method"] for e in episodes}
    return [m for m in METHODS if m in found]


def finish(figure, title, note, path, bottom=0.06):
    figure.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    figure.text(0.01, 0.01, note, fontsize=8.5, color=MUTED, ha="left", va="bottom", wrap=True)
    figure.tight_layout(rect=(0, bottom, 1, 0.94))
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(path)


def bars_with_intervals(ax, episodes, methods, key, percent=True):
    for x, method in enumerate(methods):
        maps = per_map(episodes, method, key)
        mean = float(np.mean(list(maps.values())))
        low, high = interval(maps.values())
        ax.bar(x, mean, 0.62, color=SERIES[method], edgecolor=SURFACE, linewidth=2, zorder=2)
        ax.errorbar(x, mean, yerr=[[mean - low], [high - mean]], color=INK, linewidth=1.2, capsize=4, zorder=3)
        if method == "mappo":
            seeds = list(seed_means(episodes, key).values())
            ax.scatter(np.full(len(seeds), x) + np.linspace(-0.18, 0.18, len(seeds)), seeds, s=22,
                       color=SURFACE, edgecolor=INK, linewidth=1, zorder=4)
        text = "{:.1f}%".format(100 * mean) if percent else "{:.2f}".format(mean)
        ax.text(x, high + 0.02, text, ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([LABELS[m] for m in methods], color=INK)


def headline(episodes, sensor, out, excluded):
    maps = len({e["map"] for e in episodes})
    methods = present(episodes)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, key, label in ((axes[0], "success", "Success rate (reached 99% within 1000 steps)"),
                           (axes[1], "coverage", "Mean final coverage")):
        bars_with_intervals(ax, episodes, methods, key)
        ax.set_ylim(0, 1.12)
        ax.set_title(label, loc="left", fontsize=11)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    finish(figure, "{} new test maps where 99% is reachable, {}: all methods, same episodes".format(maps, sensor),
           "Bars: mean over maps (10 episodes each). Black whisker: 95% map interval (bootstrap over maps). "
           "White dots: the 5 MAPPO training seeds, trained on the original 8 maps only. "
           "Left out: {} courtyard maps whose coverage ceiling is below the 99% target.".format(excluded),
           out)


def old_vs_new(episodes, old_scores, sensor, out, unreachable):
    old = {(r["method"], r["suite"], r["training_seed"]): r for r in read(Path(old_scores) / "per_seed.csv")}
    groups = [("familiar", "Familiar\n(8 training maps)"), ("unseen", "Old test\n(2 loop maps)"),
              ("rooms", "rooms_v1\n(16, no courtyard)"), ("courtyard", "rooms_v1 courtyard\n(14; {} cannot reach 99%)".format(
                  sum(m.startswith("rooms") for m in unreachable))),
              ("houseexpo", "HouseExpo\n(30 new)")]
    figure, ax = plt.subplots(figsize=(12.5, 4.8))
    width = 0.26
    for index, method in enumerate(PLANNERS):
        for g, (group, _) in enumerate(groups):
            if group in ("familiar", "unseen"):
                seeds = [float(r["success_rate"]) for (m, s, _), r in old.items() if m == method and s == group]
            else:
                seeds = list(seed_means(episodes, source=group).values()) if method == "mappo" else \
                    [float(np.mean(list(per_map(episodes, method, source=group).values())))]
            if not seeds:
                continue
            x = g + (index - 1) * width
            mean = float(np.mean(seeds))
            ax.bar(x, mean, width, color=SERIES[method], edgecolor=SURFACE, linewidth=2, zorder=2,
                   label=LABELS[method] if g == 0 else None)
            if method == "mappo":
                ax.scatter(np.full(len(seeds), x), seeds, s=18, color=SURFACE, edgecolor=INK, linewidth=1, zorder=4)
            ax.text(x, max(seeds) + 0.02, "{:.0f}".format(100 * mean), ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([label for _, label in groups], color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Success rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="upper left", frameon=False, ncol=3, labelcolor=INK2)
    finish(figure, "Does the old ranking survive a bigger test set? ({})".format(sensor),
           "Old bars: scores_v2 (familiar 80 episodes, old test 20). New bars: 10 episodes per map. "
           "Dots: MAPPO training seeds. Same checkpoints throughout; only the test maps changed.",
           out)


def coverage_over_time(collection, sensor, out):
    rows = read(Path(collection) / "scores/coverage_curves.csv")
    curves = defaultdict(lambda: defaultdict(list))
    for r in rows:
        curves[r["method"]][r["training_seed"]].append(float(r["mean_coverage"]))
    finals = [np.mean([v[-1] for v in curves[m].values()]) for m in PLANNERS if m in curves]
    floor = min(0.85, np.floor((min(finals) - 0.05) * 20) / 20)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharex=True)
    for ax, zoom in zip(axes, (False, True)):
        for method in METHODS:
            if method not in curves:
                continue
            series = np.array([v for _, v in sorted(curves[method].items())])
            steps = np.arange(series.shape[1])
            if len(series) > 1:
                ax.fill_between(steps, series.min(axis=0), series.max(axis=0), color=SERIES[method],
                                alpha=0.18, linewidth=0)
            ax.plot(steps, series.mean(axis=0), color=SERIES[method], linewidth=2, label=LABELS[method])
        ax.axhline(0.99, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xlabel("Decision step")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        if zoom:
            ax.set_ylim(floor, 1.0)
            ax.set_title("Zoom: {:.0f}–100%".format(100 * floor), loc="left", fontsize=11)
        else:
            ax.set_ylim(0, 1.02)
            ax.set_title("Mean team coverage", loc="left", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=len(labels),
                  frameon=False, labelcolor=INK2)
    finish(figure, "How fast each method explores, 60 new maps ({})".format(sensor),
           "Mean coverage over all 600 episodes at each step; finished episodes carry their final coverage forward. "
           "MAPPO band: lowest to highest of the 5 seed curves. Dashed line: 99% target.",
           out, bottom=0.12)


def paired_gap(episodes, sensor, out):
    mappo = per_map(episodes, "mappo")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, rival in zip(axes, ("cost", "mmpf")):
        other = per_map(episodes, rival)
        gaps = sorted(((mappo[m] - other[m], m) for m in mappo), key=lambda item: item[0])
        values = np.array([g for g, _ in gaps])
        colors = [INK2 if m.startswith("rooms") else AXIS for _, m in gaps]
        count = len(gaps)
        ax.bar(range(len(gaps)), values, 0.8, color=colors, zorder=2)
        ax.axhline(0, color=INK, linewidth=0.8)
        low, high = interval(values)
        ax.set_title("MAPPO − {}: mean {:+.1f} pts, map interval {:+.1f} to {:+.1f}\n"
                     "MAPPO better on {}, worse on {}, tied on {} maps".format(
                         LABELS[rival], 100 * values.mean(), 100 * low, 100 * high,
                         (values > 0).sum(), (values < 0).sum(), (values == 0).sum()), loc="left", fontsize=10)
        ax.set_xticks([])
        ax.set_xlabel("{} maps where 99% is reachable, sorted by gap".format(count))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("Success difference")
    handles = [matplotlib.patches.Patch(color=INK2, label="rooms_v1"), matplotlib.patches.Patch(color=AXIS, label="HouseExpo")]
    axes[1].legend(handles=handles, loc="upper left", frameon=False, labelcolor=INK2)
    finish(figure, "Paired comparison per map ({})".format(sensor),
           "Each bar: one map, MAPPO success (mean of 5 seeds × 10 episodes) minus the planner's success on the "
           "same 10 start positions. Below zero: the planner did better on that map.",
           out)


def difficulty_grid(episodes, sensor, out):
    rows = [("cost", None), ("mmpf", None)] + [("mappo", s) for s in sorted(seed_means(episodes))] + [("random", None)]
    rows = [r for r in rows if r[0] in present(episodes)]
    sums = defaultdict(list)
    for e in episodes:
        if e["method"] in PLANNERS:
            sums[e["map"]].append(e["success"])
    rank = {"rooms": 0, "courtyard": 1, "houseexpo": 2}
    source = {e["map"]: e["source"] for e in episodes}
    order = sorted(sums, key=lambda m: (rank[source[m]], -np.mean(sums[m])))
    grid = np.zeros((len(rows), len(order)))
    for i, (method, seed) in enumerate(rows):
        values = defaultdict(list)
        for e in episodes:
            if e["method"] == method and (seed is None or e["seed"] == seed):
                values[e["map"]].append(e["success"])
        grid[i] = [np.mean(values[m]) for m in order]
    figure, ax = plt.subplots(figsize=(13, 4.4))
    image = ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    edges = np.cumsum([sum(source[m] == g for m in order) for g in rank])
    for edge in edges[:-1]:
        ax.axvline(edge - 0.5, color=INK, linewidth=1.5)
    starts = np.concatenate([[0], edges[:-1]])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[m] + ("" if s is None else " seed " + s) for m, s in rows], color=INK)
    ax.set_xticks((starts + edges) / 2 - 0.5)
    ax.set_xticklabels([SOURCES[g] for g in rank], color=INK)
    ax.grid(False)
    bar = figure.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    bar.set_label("Success rate on the map (10 episodes)")
    finish(figure, "Which maps break which method ({})".format(sensor),
           "Columns: maps, grouped by source, then sorted easiest to hardest by the pooled success of Cost, MMPF and "
           "MAPPO. Courtyard maps: 0 successes for every method, because 99% coverage is out of reach there.",
           out)


def efficiency(episodes, sensor, out):
    methods = [m for m in PLANNERS if m in present(episodes)]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    panels = [("path per area", "Team path per m² explored (m/m²)", lambda e: e["path_m"] / e["explored_m2"], False),
              ("overlap", "Overlap: share of cells both robots saw", lambda e: e["overlap"], True),
              ("t99", "Median steps to 99% (successes only)", lambda e: e["t99"], False)]
    for ax, (_, title, value, percent) in zip(axes, panels):
        for x, method in enumerate(methods):
            chosen = [e for e in episodes if e["method"] == method and (value(e) is not None)]
            reduce = np.median if title.startswith("Median") else np.mean
            mean = float(reduce([value(e) for e in chosen]))
            ax.bar(x, mean, 0.62, color=SERIES[method], edgecolor=SURFACE, linewidth=2, zorder=2)
            if method == "mappo":
                by_seed = defaultdict(list)
                for e in chosen:
                    by_seed[e["seed"]].append(value(e))
                seeds = [float(reduce(v)) for _, v in sorted(by_seed.items())]
                ax.scatter(np.full(len(seeds), x) + np.linspace(-0.18, 0.18, len(seeds)), seeds, s=22,
                           color=SURFACE, edgecolor=INK, linewidth=1, zorder=4)
                top = max(seeds + [mean])
            else:
                top = mean
            ax.text(x, top * 1.03, "{:.0%}".format(mean) if percent else "{:.2f}".format(mean) if mean < 10
                    else "{:.0f}".format(mean), ha="center", va="bottom", fontsize=9, color=INK2)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([LABELS[m] for m in methods], color=INK)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        if percent:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    finish(figure, "Efficiency and teamwork, 60 new maps ({})".format(sensor),
           "Left, middle: mean over all 600 episodes per method. Right: median over successful episodes only, so it "
           "compares speed, not reliability. Dots: MAPPO seeds. Lower is better in all three panels.",
           out)


def seed_consistency(episodes, sensor, out):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, source in zip(axes, ("rooms", "houseexpo")):
        seeds = seed_means(episodes, source=source)
        ax.bar(range(len(seeds)), list(seeds.values()), 0.62, color=SERIES["mappo"], edgecolor=SURFACE,
               linewidth=2, zorder=2)
        for x, v in enumerate(seeds.values()):
            ax.text(x, v + 0.02, "{:.0f}%".format(100 * v), ha="center", fontsize=9, color=INK2)
        for rival, dash in (("cost", (0, (5, 3))), ("mmpf", (0, (1.5, 2)))):
            level = float(np.mean(list(per_map(episodes, rival, source=source).values())))
            ax.axhline(level, color=SERIES[rival], linewidth=2, linestyle=dash, zorder=3,
                       label="{} {:.0f}%".format(LABELS[rival], 100 * level))
        ax.legend(loc="upper left", frameon=False, labelcolor=INK2, fontsize=9, ncol=2)
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels(["seed " + s for s in seeds], color=INK)
        ax.set_title(SOURCES[source], loc="left", fontsize=11)
        ax.set_ylim(0, 1.2)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlim(-0.6, len(seeds) - 0.4)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("Success rate")
    finish(figure, "Is MAPPO's result stable across training seeds? ({})".format(sensor),
           "Bars: each of the 5 independently trained MAPPO policies (10 episodes per map). Courtyard maps omitted: 0% for all. "
           "Lines: Cost and MMPF on the same episodes.",
           out)


def failures(episodes, sensor, out):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, source in zip(axes, ("rooms", "courtyard", "houseexpo")):
        notes = []
        for method in PLANNERS:
            failed = np.sort([e["coverage"] for e in episodes
                              if e["method"] == method and e["source"] == source and not e["success"]])
            total = sum(e["method"] == method and e["source"] == source for e in episodes)
            if len(failed):
                ax.plot(failed, np.arange(1, len(failed) + 1) / len(failed), color=SERIES[method], linewidth=2,
                        label="{} ({} of {} failed)".format(LABELS[method], len(failed), total), drawstyle="steps-post")
            reasons = {e["reason"] for e in episodes if e["method"] == method and not e["success"]}
            notes.append("/".join(sorted(reasons)))
        ax.set_title(SOURCES[source], loc="left", fontsize=11)
        ax.set_xlabel("Final coverage of failed episodes")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.legend(loc="upper left", frameon=False, labelcolor=INK2, fontsize=9)
    axes[0].set_ylabel("Share of failed episodes at or below")
    finish(figure, "How failures fail: near-misses or stuck? ({})".format(sensor),
           "Cumulative distribution over failed episodes only. A curve hugging the right edge means failures "
           "ran out of the 1000 steps just short of 99%. Termination reason of every failure: "
           + ", ".join(sorted(set(notes))) + ".",
           out)


def sensor_gap(first, second, out):
    (a, a_name), (b, b_name) = first, second
    figure, ax = plt.subplots(figsize=(9, 4.4))
    methods = [m for m in METHODS if m in present(a) and m in present(b)]
    for x, method in enumerate(methods):
        ca, cb = per_map(a, method, "coverage"), per_map(b, method, "coverage")
        gaps = np.array([ca[m] - cb[m] for m in ca if m in cb])
        low, high = interval(gaps)
        ax.bar(x, gaps.mean(), 0.62, color=SERIES[method], edgecolor=SURFACE, linewidth=2, zorder=2)
        ax.errorbar(x, gaps.mean(), yerr=[[gaps.mean() - low], [high - gaps.mean()]], color=INK, capsize=4, zorder=3)
        ax.text(x, high + 0.01, "{:+.1f} pts".format(100 * gaps.mean()), ha="center", fontsize=9, color=INK2)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([LABELS[m] for m in methods], color=INK)
    ax.set_ylabel("Final coverage: {} − {}".format(a_name, b_name))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    finish(figure, "How much each method depends on its sensor (paired, 60 new maps)",
           "Per map: mean final coverage with the {} minus with the {}, same start positions. "
           "Whisker: 95% map interval.".format(a_name, b_name), out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--collection", type=Path, action="append", required=True)
    parser.add_argument("--old-scores", type=Path, action="append", required=True)
    parser.add_argument("--ceilings", type=Path, action="append", default=[],
                        help="coverage_ceiling.py CSV, one per --collection (same sensor); maps below the "
                        "target are left out of the headline and paired figures")
    args = parser.parse_args(argv)
    if len(args.collection) != len(args.old_scores):
        parser.error("give one --old-scores per --collection")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.ceilings and len(args.ceilings) != len(args.collection):
        parser.error("give one --ceilings per --collection")
    loaded = []
    for index, (collection, old) in enumerate(zip(args.collection, args.old_scores)):
        unreachable = set()
        if args.ceilings:
            unreachable = {r["map"] for r in read(args.ceilings[index]) if float(r["coverage_ceiling"]) < 0.99}
        episodes, sensor = load(collection)
        tag = "360" if sensor.startswith("360") else "four_beam"
        out = lambda name: args.output / "{}_{}.png".format(name, tag)  # noqa: E731
        reachable = [e for e in episodes if e["map"] not in unreachable]
        headline(reachable, sensor, out("01_headline"), len(unreachable))
        old_vs_new(episodes, old, sensor, out("02_old_vs_new_test_sets"), unreachable)
        coverage_over_time(collection, sensor, out("03_coverage_over_time"))
        paired_gap(reachable, sensor, out("04_paired_gap_per_map"))
        difficulty_grid(episodes, sensor, out("05_difficulty_grid"))
        efficiency(episodes, sensor, out("06_efficiency_teamwork"))
        seed_consistency(episodes, sensor, out("07_mappo_seed_consistency"))
        failures(episodes, sensor, out("08_failures"))
        loaded.append((episodes, sensor))
    if len(loaded) == 2:
        sensor_gap(loaded[0], loaded[1], args.output / "09_sensor_gap_paired.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
