# Level-0 paper-validation audit

This is a new, isolated audit lane for Explore-Bench Table II. It does not
modify or replace the existing Level-0 collector, the 120-run baseline, or the
30-run `paper_consistency` diagnostic.

## Frozen experiment

The audit runs Cost and MMPF on the five-room map with two robots, seeds 1–10,
and a 1,000-decision cap. It records the first 90%, 98%, and 99% coverage
crossings.

| Treatment | Sensor | Range | Role |
| --- | --- | ---: | --- |
| `omnidirectional_3p5` | 360-degree | 3.5 m | Released Level-0 code control |
| `four_beam_3p5` | Four beams | 3.5 m | Primary sensor extension |
| `omnidirectional_7p0` | 360-degree | 7.0 m | Range-sensitivity check |
| `four_beam_7p0` | Four beams | 7.0 m | Paired range/sensor check |

Equal task IDs have the same method, seed, start cells, and initial headings.
There are 20 tasks per treatment and 80 runs total. All treatments retain the
released legacy planning behavior, including its ground-truth navigation map.

The 3.5 m omnidirectional lane is the primary control because that is the
released Level-0 default. The paper's 7 m statement describes the Gazebo robot,
so 7 m is reported as a sensitivity analysis. Neither four-beam lane is called
a paper reproduction.

## Submit on Ada

Commit the audit code and use a checkout with no tracked modifications before
collecting. The analyzer rejects tracked-code changes and mixed commits, map
hashes, collector hashes, or task tables. Untracked notes do not invalidate a
run because they cannot change imported tracked code.

```bash
bash hpc/jobs/level0_paper_audit/submit.sh 4 level0_paper_audit_01
```

The first argument limits simultaneous tasks in each of four arrays. With `4`,
Slurm may run at most 16 audit jobs at once. A collection ID is immutable: use
a new ID for another collection. A failed task retry writes `run2`, `run3`, and
so on; it never overwrites a previous attempt. A completed task is skipped.

Generated files live outside the repository:

```text
../slurm/
├── tasks/level0_paper_audit/level0_paper_audit_01/
├── data/level0_paper_audit/level0_paper_audit_01/
└── logs/level0_paper_audit/level0_paper_audit_01/
```

Each successful run contains a provenance manifest plus `episodes.csv`,
`steps.csv`, and `agents.csv`.

## Analyze

Pass the existing diagnostic collection so seeds 1–5 of the new 3.5 m control
are checked exactly against the preserved pipeline:

```bash
python3 analysis/level0_paper_audit/analyze.py \
  --input ../slurm/data/level0_paper_audit/level0_paper_audit_01 \
  --regression-input ../slurm/data/paper_consistency/paper_consistency_01 \
  --output ../slurm/reports/level0_paper_audit/level0_paper_audit_01
```

Without `--regression-input`, the report is deliberately marked incomplete.
See [`analysis/level0_paper_audit`](../../../analysis/level0_paper_audit/README.md)
for the metric conventions and verdict rules.

The continuation gate is claim-level rather than exact-cell reproduction. It
requires the paired 3.5 m omnidirectional controls to show MMPF outperforming
Cost at both 90% and 99% coverage under the rules documented in the analysis
README. Exact Table-II comparisons remain in the generated report as
diagnostics and are not relabeled as reproduced results.
