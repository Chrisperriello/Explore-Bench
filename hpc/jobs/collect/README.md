# Collection jobs

This folder will contain the Slurm array job for full data collection. A Slurm
array starts many copies of one job script, with each copy reading a different
line from a task file.

The collection job will run the existing headless collector. Each job will
write to its own directory so parallel jobs cannot overwrite one another.

No full collection job is implemented yet. It will be added only after the
one-pass smoke job succeeds on Ada and the task-file format is agreed upon.
