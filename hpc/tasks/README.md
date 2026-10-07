# Task files

A task file is a plain table in which each row describes work for one Slurm
array job, such as the map, planner, sensor, and seeds to run.

The original omnidirectional runs and four-beam runs will use separate task
files with matching maps and seeds. No task files are created yet because the
one-pass smoke job must succeed first.
