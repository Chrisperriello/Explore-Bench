"""Create the original two-condition four-beam pipeline-proof figure.

This small analysis script predates the stricter three-condition scoped-pilot
plotter.  It reads already collected CSV artifacts and visualizes coverage,
paired final outcomes, logical communication bits, and one robot's four-beam
measurements.  It does not run the simulator or alter experiment data.
"""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


RESULTS = {
    "Four-beam, no communication": Path(
        "results/pipeline_proof_cost_four_beam/run1"
    ),
    "Four-beam + round-robin radio": Path(
        "results/pipeline_proof_cost_four_beam_round_robin/run1"
    ),
}


def read_csv(path):
    """Load one collector CSV as a list of string-valued dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


episode_data = {
    label: read_csv(directory / "episodes.csv")
    for label, directory in RESULTS.items()
}

step_data = {
    label: read_csv(directory / "steps.csv")
    for label, directory in RESULTS.items()
}

agent_data = {
    label: read_csv(directory / "agents.csv")
    for label, directory in RESULTS.items()
}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# 1. Coverage curves
ax = axes[0, 0]

for label, rows in step_data.items():
    grouped = defaultdict(list)

    for row in rows:
        grouped[int(row["seed"])].append(
            (int(row["step"]), 100 * float(row["team_coverage_ratio"]))
        )

    for seed, points in sorted(grouped.items()):
        points.sort()
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            alpha=0.45,
            label=f"{label}, seed {seed}",
        )

ax.set_title("Team coverage over time")
ax.set_xlabel("Decision step")
ax.set_ylabel("Team coverage (%)")
ax.grid(alpha=0.25)
ax.legend(fontsize=7)

# 2. Paired final coverage
ax = axes[0, 1]
labels = list(RESULTS)

coverage_by_condition = {
    label: {
        int(row["seed"]): 100 * float(row["final_coverage_ratio"])
        for row in episode_data[label]
    }
    for label in labels
}

shared_seeds = sorted(
    set(coverage_by_condition[labels[0]])
    & set(coverage_by_condition[labels[1]])
)

for seed in shared_seeds:
    values = [
        coverage_by_condition[labels[0]][seed],
        coverage_by_condition[labels[1]][seed],
    ]
    ax.plot([0, 1], values, marker="o", label=f"Seed {seed}")

ax.set_xticks([0, 1], ["No communication", "Round-robin radio"])
ax.set_ylabel("Final coverage (%)")
ax.set_title("Paired coverage after 200 decisions")
ax.grid(axis="y", alpha=0.25)
ax.legend()

# 3. Communication bits
ax = axes[1, 0]
radio_rows = episode_data["Four-beam + round-robin radio"]

attempted_bits = mean(float(row["comm_attempted_bits"]) for row in radio_rows)
delivered_bits = mean(float(row["comm_delivered_bits"]) for row in radio_rows)

bars = ax.bar(
    ["Attempted", "Delivered"],
    [attempted_bits, delivered_bits],
    color=["tab:orange", "tab:green"],
)

ax.bar_label(bars, fmt="%.0f")
ax.set_ylabel("Mean bits per episode")
ax.set_title("Radio use: broadcasts had no in-range receiver")
ax.grid(axis="y", alpha=0.25)

# 4. Four-beam readings
ax = axes[1, 1]
sensor_rows = [
    row
    for row in agent_data["Four-beam, no communication"]
    if int(row["seed"]) == 1 and int(row["agent_id"]) == 0
]

sensor_rows.sort(key=lambda row: int(row["step"]))

for beam in ["front", "back", "left", "right"]:
    ax.plot(
        [int(row["step"]) for row in sensor_rows],
        [float(row[f"{beam}_distance"]) for row in sensor_rows],
        label=beam,
    )

ax.set_title("Four-beam measurements: seed 1, robot 0")
ax.set_xlabel("Decision step")
ax.set_ylabel("Measured distance (m)")
ax.set_ylim(0, 3.7)
ax.grid(alpha=0.25)
ax.legend()

fig.suptitle("Explore-Bench four-beam pipeline proof", fontsize=15)
fig.tight_layout()

output = Path("results/pipeline_proof_plots.png")
fig.savefig(output, dpi=200, bbox_inches="tight")
print(f"Saved {output}")
