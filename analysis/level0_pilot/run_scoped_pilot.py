#!/usr/bin/env python3
"""Print or explicitly run one condition from the scoped Level-0 pilot.

This wrapper intentionally does not execute by default. Pass ``--execute``
when the person operating the machine is ready to start the CPU-heavy job.
The underlying collector remains the source of truth for simulation behavior
and output schemas.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
COLLECTOR = REPOSITORY_ROOT / "grid_simulator" / "collect_experiments.py"
MAP = Path("onpolicy/onpolicy/envs/GridEnv/datasets/corner.pgm")

CONDITIONS = {
    "none": {
        "experiment": "scoped_pilot_cost_four_beam_none",
        "mode": "none",
        "range_cells": 150.0,
    },
    "range40": {
        "experiment": "scoped_pilot_cost_four_beam_shared_r40",
        "mode": "shared_collision",
        "range_cells": 40.0,
    },
    "limited150": {
        "experiment": "scoped_pilot_cost_four_beam_shared_r150",
        "mode": "shared_collision",
        "range_cells": 150.0,
    },
    "perfect": {
        "experiment": "scoped_pilot_cost_four_beam_perfect",
        "mode": "perfect",
        "range_cells": 150.0,
    },
}


def build_command(condition, output):
    """Return the fully specified collector command for one pilot treatment.

    The only intended information-treatment changes are communication mode and
    range.  Map, planner, sensors, seeds, horizon, protocol, and other broker
    parameters remain fixed for paired comparison.
    """
    treatment = CONDITIONS[condition]
    return [
        sys.executable,
        str(COLLECTOR),
        "--experiment-name",
        treatment["experiment"],
        "--method",
        "cost",
        "--maps",
        str(MAP),
        "--team-sizes",
        "2",
        "--seeds",
        "1",
        "2",
        "3",
        "--resolution",
        "0.1",
        "--sensor-types",
        "four_beam",
        "--sensor-ranges",
        "3.5",
        "--communication-mode",
        treatment["mode"],
        "--communication-protocol",
        "round_robin",
        "--comm-tile-size",
        "8",
        "--comm-candidate-count",
        "8",
        "--comm-range-cells",
        str(treatment["range_cells"]),
        "--comm-latency-min-steps",
        "1",
        "--comm-latency-max-steps",
        "1",
        "--comm-packet-loss",
        "0",
        "--comm-cooldown-steps",
        "3",
        "--comm-bucket-capacity-bits",
        "314",
        "--comm-bucket-refill-bits",
        "53",
        "--comm-ttl-steps",
        "8",
        "--comm-cost-per-patch",
        "0.01",
        "--max-steps",
        "200",
        "--no-progress-patience",
        "50",
        "--coverage-thresholds",
        "0.90",
        "0.98",
        "0.99",
        "--target-coverage",
        "0.99",
        "--record",
        "full",
        "--map-snapshots",
        "none",
        "--output",
        str(output),
    ]


def build_parser():
    """Build a safe preview-first CLI; execution requires ``--execute``."""
    parser = argparse.ArgumentParser(
        description=(
            "Print one reproducible scoped-pilot command; add --execute to run it."
        )
    )
    parser.add_argument("condition", choices=sorted(CONDITIONS))
    parser.add_argument("--output", default="results")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually start the CPU-heavy collector (default: print only)",
    )
    return parser


def main():
    """Print the reproducible command and optionally execute it from repo root."""
    args = build_parser().parse_args()
    command = build_command(args.condition, args.output)
    print(shlex.join(command), flush=True)
    if not args.execute:
        print("Dry run only. Add --execute when you are ready to run it.")
        return 0
    completed = subprocess.run(command, cwd=str(REPOSITORY_ROOT), check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
