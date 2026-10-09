# Result merging (separate from collection)

`merge_level0.py` combines checked per-task CSV files into larger tables. It
adds the collection, sensor variant, task number, scenario, attempt number, and
source directory to every row so the source remains identifiable.

Run it after all original and retry jobs finish:

```bash
python3 hpc/jobs/merge/merge_level0.py \
  --collection level0_validation_01
```

The default output is outside the repository:

```text
../slurm/combined/level0/level0_validation_01/
├── episodes.csv
├── steps.csv
├── agents.csv
├── run_inventory.csv
└── merge_manifest.json
```

- `episodes.csv` has one row per completed run and is the main input for tables.
- `steps.csv` contains coverage and other team metrics over time.
- `agents.csv` contains per-robot paths, sensor readings, and coverage.
- `run_inventory.csv` shows which attempt was selected for every expected run.
- `merge_manifest.json` records source counts and the frozen task-table hash.

For a retried task, the newest complete `runN` is selected. Timed-out partial
attempts are never copied. By default the command refuses to merge unless all
120 expected runs have a complete attempt. `--allow-incomplete` can create a
clearly marked preliminary dataset when needed.
