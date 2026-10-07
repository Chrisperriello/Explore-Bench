# Smoke job

A smoke job is one small run used to catch environment, path, dependency, and
Slurm problems before submitting many jobs.

`run_one_pass.sbatch` runs one headless Explore-Bench episode with:

- two robots;
- the cost planner;
- the `corner.pgm` map;
- seed `1`;
- the original omnidirectional sensor at 3.5 m;
- legacy map sharing and navigation behavior;
- detailed episode, step, and per-robot CSV output.

Submit it from the repository root with the wrapper that creates the external
log and data directories:

```bash
bash hpc/jobs/smoke/submit_one_pass.sh
```

Nothing generated is written inside Explore-Bench:

```text
../slurm/
├── data/one_pass/job_<job-id>/
└── logs/
    ├── explore-one-pass-<job-id>.out
    └── explore-one-pass-<job-id>.err
```

Submitting `run_one_pass.sbatch` directly discards its terminal log. Use the
wrapper so the log is retained outside the repository.
