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

Submit it from the repository root with:

```bash
sbatch hpc/jobs/smoke/run_one_pass.sbatch
```

The data is written under `results/slurm_one_pass/job_<job-id>/`. Slurm writes
the terminal output to its normal `slurm-<job-id>.out` file.
