# HPC data pipeline

This folder contains the files used to run Explore-Bench on a Slurm computing
cluster. Slurm is the system that starts and manages jobs across cluster nodes.

The folders are organized by the order in which the data pipeline will run:

1. `jobs/smoke/`: run one small job to prove the environment and collector work.
2. `tasks/`: describe the runs that the full data collection must execute.
3. `jobs/collect/`: submit many independent collection jobs in parallel.
4. `jobs/check/`: confirm that every expected job finished and wrote valid data.
5. `jobs/merge/`: combine the checked job outputs into analysis-ready tables.

Only scripts, task definitions, and documentation belong here. Generated data
belongs under `results/`, which is ignored by Git.

## Current status

Only the one-pass smoke job is implemented. The other folders document the
next pipeline pieces and prevent their responsibilities from being mixed
together.

## Environment

Jobs expect the repository-local Conda environment at `.conda-env`. On Ada,
the job loads `python/anaconda3`, clears the module's `PYTHONHOME`, and activates
that environment before running Python.
