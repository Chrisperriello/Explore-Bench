# Slurm jobs

Each subfolder contains one stage of the cluster workflow:

- `smoke`: a single end-to-end test run.
- `collect`: paired original and four-beam Level-0 data collection.
- `check`: future checks for missing, failed, or mismatched results.
- `merge`: future scripts that combine raw outputs into final tables.

Keeping the stages separate makes it clear whether a script creates raw data,
checks data, or combines data.

Submission wrappers create `../slurm/logs` before calling `sbatch`. This is
necessary because Slurm opens its output file before the batch script begins.

Checking and merging are separate follow-up work; the collection finishes when
each array task has written its raw collector files.
