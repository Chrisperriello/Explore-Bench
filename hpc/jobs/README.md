# Slurm jobs

Each subfolder contains one stage of the cluster workflow:

- `smoke`: a single end-to-end test run.
- `collect`: future parallel data-collection jobs.
- `check`: future checks for missing, failed, or mismatched results.
- `merge`: future jobs that combine checked outputs into final tables.

Keeping the stages separate makes it clear whether a script creates raw data,
checks data, or combines data.
