# Level-0 baseline analysis

This package turns a merged Level-0 collection into an advisor-ready report.
It compares the original omnidirectional sensor with four-beam sensing using
the matched method, map, seed, starts, team size, and legacy behavior.

Run from the repository root:

```bash
bash analysis/level0_baseline/run_advisor_report.sh
```

To use another merged collection or output directory:

```bash
bash analysis/level0_baseline/run_advisor_report.sh \
  /path/to/merged/collection \
  /path/to/report
```

The default output is ignored by Git:

```text
results/level0_baseline_analysis/
├── advisor_brief.md
├── advisor_brief.txt
├── advisor_packet.pdf
├── report.html
├── figures/
│   ├── 01_executive_overview.png
│   ├── 02_condition_matrices.png
│   ├── 03_paired_final_coverage.png
│   ├── 04_coverage_curves.png
│   ├── 05_milestone_performance.png
│   ├── 06_efficiency_and_coordination.png
│   ├── 07_termination_reasons.png
│   ├── 08_representative_trajectories.png
│   └── 09_explore_bench_direct_comparison.png
└── tables/
    ├── condition_summary.csv
    ├── overall_summary.csv
    ├── paired_runs.csv
    ├── paired_summary.csv
    ├── termination_summary.csv
    ├── explore_bench_direct_comparison.csv
    ├── paper_room_reference.csv
    └── run_inventory.csv
```

## Interpretation rules

- A completed process is not automatically an exploration success.
- Success means that the episode reached the configured 99% coverage target.
- Time-to-90%, time-to-98%, and time-to-99% include only runs that reached the
  corresponding milestone. Their reach rates are reported beside them.
- Incomplete-run path length and overlap are diagnostics, not direct
  completion-efficiency comparisons.
- Slurm wall-clock time is intentionally excluded.
- The paper room values are shown only as context. They are not labeled an
  exact reproduction until the paper's sensor range, starts, and units are
  confirmed against this collection.

For validation without Matplotlib:

```bash
python3 analysis/level0_baseline/analyze_level0.py --tables-only
```
