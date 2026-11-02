"""Find evaluation episodes whose coverage stops rising before the 99% target.

Usage: python analysis/mappo_omni/find_stalls.py OUTPUT_CSV EVALUATION_DIR [EVALUATION_DIR ...]

For each episode this records the last decision that added coverage, how many
decisions followed without any gain, and how far the team still travelled in
that stretch. Zero travel means the robots stood still (no routable goal);
positive travel means they moved without sensing anything new.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

FIELDS = ["method", "suite", "map_name", "episode_id", "success", "final_coverage", "steps_executed",
          "last_gain_step", "stalled_steps", "path_m_during_stall"]


def episodes(directory):
    traces = defaultdict(list)
    with (Path(directory) / "steps.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            traces[row["episode_id"]].append(row)
    for episode_id, rows in traces.items():
        rows.sort(key=lambda row: int(row["step"]))
        coverage = [float(row["team_coverage_ratio"]) for row in rows]
        path = [float(row["team_path_length_m"]) for row in rows]
        last_gain = max((i for i in range(1, len(rows)) if coverage[i] > coverage[i - 1]), default=0)
        yield {"method": rows[0]["method"], "suite": rows[0]["suite"], "map_name": rows[0]["map_name"],
               "episode_id": episode_id, "success": coverage[-1] >= 0.99, "final_coverage": round(coverage[-1], 4),
               "steps_executed": int(rows[-1]["step"]), "last_gain_step": last_gain,
               "stalled_steps": int(rows[-1]["step"]) - last_gain,
               "path_m_during_stall": round(path[-1] - path[last_gain], 2)}


def main():
    output, directories = Path(sys.argv[1]), sys.argv[2:]
    rows = [row for directory in directories for row in episodes(directory)]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["method"], row["map_name"], row["episode_id"])))
    print(output)


if __name__ == "__main__":
    main()
