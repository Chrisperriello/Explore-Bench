# HPC Level-0 collection

This folder contains the files used to run Explore-Bench on a Slurm computing
cluster. Slurm is the system that starts and manages jobs across cluster nodes.

The collection path is:

1. `jobs/smoke/`: run one small job to prove the environment and collector work.
2. `tasks/`: define the maps, methods, seeds, and fixed collection settings.
3. `jobs/collect/`: create one task table and submit two matched Slurm arrays.
4. Each array task writes one independent raw run outside the repository.

`jobs/check/` and `jobs/merge/` are separate later tools. They are not part of
collecting the raw data.

`monitor/` displays Slurm state and recorded-data health in the terminal.

`jobs/paper_consistency/` is a separate 30-run diagnostic. It checks whether
sensor distance or map-sharing behavior explains disagreement with the paper;
it does not replace the original-versus-four-beam collection.

Only scripts, task definitions, and documentation belong here. Generated data
and Slurm logs are written outside the repository:

```text
../slurm/
├── tasks/level0/<collection-id>/
├── data/level0/<collection-id>/
└── logs/level0/<collection-id>/
```

## Current status

The one-pass smoke job, terminal monitor, and paired Level-0 Slurm arrays are
implemented. The collection uses one CPU per array task because each collector
process is single-threaded. Parallelism comes from Slurm running different
array tasks at the same time.

## Submit the Level-0 collection

From the repository root on Ada:

```bash
bash hpc/jobs/collect/submit_level0.sh
```

The default limit is 16 simultaneous tasks in each array. To use a different
limit or a memorable collection ID:

```bash
bash hpc/jobs/collect/submit_level0.sh 8 level0_validation_01
```

The command submits one original 360-degree array and one four-beam array. Both
read the same generated task table, so matching array indices use the same map,
method, team size, and seed.

## Environment

Jobs expect the repository-local Conda environment at `.conda-env`. On Ada,
the job loads `python/anaconda3`, clears the module's `PYTHONHOME`, and activates
that environment before running Python.
