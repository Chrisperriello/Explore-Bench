# Paper-consistency analysis

This report reads the 30 raw runs produced by
`hpc/jobs/paper_consistency/submit.sh`. It creates combined tables, three PNG
figures, a short written assessment, and an HTML report.

Run it on Ada from the repository root:

```bash
MPLBACKEND=Agg .conda-env/bin/python \
  analysis/paper_consistency/analyze.py \
  --input ../slurm/data/paper_consistency/paper_consistency_01 \
  --output ../slurm/reports/paper_consistency/paper_consistency_01
```

Outputs:

```text
../slurm/reports/paper_consistency/paper_consistency_01/
├── meeting_assessment.md
├── report.html
├── episodes_combined.csv
├── summary.csv
├── paired_runs.csv
└── figures/
    ├── 01_success_and_coverage.png
    ├── 02_paired_final_coverage.png
    └── 03_coverage_curves.png
```

`Slurm COMPLETED` means a task exited normally. Exploration success is a
separate measurement: a run succeeds only if it reaches 99% coverage.
